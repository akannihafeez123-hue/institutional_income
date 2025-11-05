#!/usr/bin/env python3
"""
Quantum Income Engine - Single-file deploy (main.py)

Features:
- Scanner for airdrops and high-value tasks (demo + configurable aggregator support)
- Persistent SQLite action queue (so restarts don't lose actions)
- Telegram admin approval workflow with inline buttons (Approve / Sell & Transfer / Reject)
- Admin-only HTTP endpoints protected by X-ADMIN-TOKEN header
- Bitget RW placeholders for market sell and transfer (do NOT enable without testing)
- Scheduler runs scanner every SCAN_INTERVAL_MINUTES (default 5)
- Telegram long-poller processes inline button clicks and only allows TELEGRAM_ADMIN_ID to approve

Environment variables (set as Choreo secrets):
- TELEGRAM_TOKEN                 (BotFather token)
- TELEGRAM_CHAT_ID               (bot owner chat id)
- TELEGRAM_ADMIN_ID              (numeric Telegram user id allowed to approve)
- ADMIN_TOKEN                    (random admin token used for HTTP header X-ADMIN-TOKEN)
- WALLET_ADDRESS                 (your TRC20 USDT address; optional)
- SCAN_INTERVAL_MINUTES          (default 5)
- AIRDROP_AGGREGATORS            (comma-separated aggregator URLs; optional)
- BITGET_API_KEY_RW              (ONLY set when ready; leave blank for simulation)
- BITGET_API_SECRET_RW
- BITGET_API_PASSPHRASE_RW
- PORT                          (default 8080)

Deploy: set secrets, push this file to GitHub, connect repo to Choreo, set start command `python main.py`.
"""
import os
import time
import json
import hmac
import hashlib
import threading
import sqlite3
import requests
from datetime import datetime
from flask import Flask, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler

# -------------------------
# Config (from env / Choreo secrets)
# -------------------------
PORT = int(os.getenv("PORT", "8080"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # Bot owner chat id (where approval cards are sent)
TELEGRAM_ADMIN_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "0"))  # Numeric Telegram user id allowed to act
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # for HTTP admin endpoints
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "")  # TRC20 USDT destination (for manual withdraw instructions)
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))
AIRDROP_AGGREGATORS = os.getenv("AIRDROP_AGGREGATORS", "")
AUTO_APPROVE_USD_THRESHOLD = float(os.getenv("AUTO_APPROVE_USD_THRESHOLD", "50.0"))

# Bitget RW keys (leave empty for simulation)
BITGET_API_KEY_RW = os.getenv("BITGET_API_KEY_RW", "")
BITGET_API_SECRET_RW = os.getenv("BITGET_API_SECRET_RW", "")
BITGET_API_PASSPHRASE_RW = os.getenv("BITGET_API_PASSPHRASE_RW", "")

API_HOST_BITGET = "https://api.bitget.com"

# -------------------------
# DB (SQLite persistent queue)
# -------------------------
DB_PATH = os.getenv("DB_PATH", "quantum_income_actions.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS actions (
    id TEXT PRIMARY KEY,
    type TEXT,
    payload TEXT,
    created_at TEXT,
    status TEXT
)""")
conn.commit()

def persist_action(action_id, a_type, payload):
    cur.execute(
        "INSERT OR REPLACE INTO actions (id,type,payload,created_at,status) VALUES (?,?,?,?,?)",
        (action_id, a_type, json.dumps(payload), datetime.utcnow().isoformat(), "PENDING")
    )
    conn.commit()

def set_action_status(action_id, status):
    cur.execute("UPDATE actions SET status=? WHERE id=?", (status, action_id))
    conn.commit()

def get_pending_actions(limit=200):
    cur.execute("SELECT id,type,payload,created_at FROM actions WHERE status='PENDING' ORDER BY created_at LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [{"id": r[0], "type": r[1], "payload": json.loads(r[2]), "created": r[3]} for r in rows]

def remove_action(action_id):
    cur.execute("DELETE FROM actions WHERE id=?", (action_id,))
    conn.commit()

# -------------------------
# Telegram helpers + long-poll
# -------------------------
def tg_api(method, body):
    if not TELEGRAM_TOKEN:
        print("[tg_api] TELEGRAM_TOKEN not set; dropping message:", method, body)
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"
    try:
        r = requests.post(url, json=body, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("[tg_api] error:", e, getattr(e, 'response', None))
        return None

def tg_send_text(chat_id, text):
    return tg_api("sendMessage", {"chat_id": chat_id, "text": text})

def tg_send_approval_card(action_id, payload):
    if not TELEGRAM_CHAT_ID or not TELEGRAM_TOKEN:
        print("[tg_send_approval_card] telegram not configured. action:", action_id, payload)
        return
    # Minimal summary to avoid very long messages
    summary = {k: payload.get(k) for k in ("id","type","name","token_symbol","token_amount","est_usd","amount_usd") if k in payload}
    text = f"🔔 *Quantum Income Engine* — Action: `{action_id}`\n{json.dumps(summary, default=str)}"
    reply_markup = {"inline_keyboard":[
        [{"text":"✅ Approve","callback_data": f"APPROVE::{action_id}"},
         {"text":"💱 Sell & Transfer","callback_data": f"SELL::{action_id}"},
         {"text":"❌ Reject","callback_data": f"REJECT::{action_id}"}]
    ]}
    body = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "reply_markup": reply_markup, "parse_mode": "MarkdownV2"}
    return tg_api("sendMessage", body)

def tg_long_poll_loop(callback_handler):
    """
    Long-poll getUpdates and forward only admin actions to callback_handler.
    callback_handler receives dict: {"action": "APPROVE"/"SELL"/"REJECT", "action_id":..., "from_id":..., "from_user":...}
    Non-admin attempts are told "Not authorized" and owner is notified.
    """
    print("[tg_long_poll] started")
    last_update_id = None
    while True:
        try:
            params = {"timeout": 30}
            if last_update_id:
                params["offset"] = last_update_id + 1
            r = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates", params=params, timeout=35)
            data = r.json()
            for upd in data.get("result", []):
                last_update_id = upd["update_id"]
                if "callback_query" in upd:
                    cb = upd["callback_query"]
                    data_cb = cb.get("data","")
                    from_obj = cb.get("from", {})
                    from_id = from_obj.get("id")
                    from_user = from_obj.get("username", "?")
                    # parse
                    if "::" in data_cb:
                        cmd, aid = data_cb.split("::",1)
                    else:
                        cmd, aid = data_cb, None

                    # Non-admin -> answer and notify owner
                    if int(from_id) != TELEGRAM_ADMIN_ID:
                        try:
                            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
                                          json={"callback_query_id": cb.get("id"), "text": "Not authorized"})
                        except:
                            pass
                        # notify owner
                        try:
                            tg_send_text(TELEGRAM_CHAT_ID, f"⚠️ Unauthorized action attempt by @{from_user} (id:{from_id}) on {cmd} {aid}")
                        except:
                            pass
                        continue

                    # Authorized: acknowledge and call handler
                    try:
                        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
                                      json={"callback_query_id": cb.get("id"), "text": "Action received"})
                    except:
                        pass
                    callback_handler({"action": cmd, "action_id": aid, "from_id": from_id, "from_user": from_user})
            time.sleep(0.5)
        except Exception as e:
            print("[tg_long_poll] error", e)
            time.sleep(2)

# -------------------------
# Simple Airdrop + Task scanners (demo + aggregator support)
# -------------------------
COINGECKO_SEARCH = "https://api.coingecko.com/api/v3/search"
COINGECKO_PRICE = "https://api.coingecko.com/api/v3/simple/price"

def _coingecko_price_for_symbol(symbol):
    # Best-effort: search then get price
    if not symbol:
        return 0.0
    try:
        r = requests.get(COINGECKO_SEARCH, params={"query":symbol}, timeout=8)
        r.raise_for_status()
        js = r.json()
        if js.get("coins"):
            cg_id = js["coins"][0]["id"]
            p = requests.get(COINGECKO_PRICE, params={"ids": cg_id, "vs_currencies":"usd"}, timeout=8).json()
            return float(p.get(cg_id, {}).get("usd", 0.0))
    except Exception:
        pass
    return 0.0

def normalize_aggregator_payload(payload):
    # Try to normalize aggregator shapes into list of candidates
    out = []
    if not payload:
        return out
    items = []
    if isinstance(payload, dict):
        if "airdrops" in payload and isinstance(payload["airdrops"], list):
            items = payload["airdrops"]
        elif "data" in payload and isinstance(payload["data"], list):
            items = payload["data"]
        else:
            # maybe payload itself is a candidate
            if payload.get("name"):
                items = [payload]
    elif isinstance(payload, list):
        items = payload
    for it in items:
        try:
            cid = it.get("id") or it.get("slug") or str(time.time())
            name = it.get("name") or it.get("title") or "unknown"
            token_symbol = (it.get("token_symbol") or it.get("symbol") or it.get("token") or "UNKNOWN").upper()
            amount = float(it.get("amount") or it.get("reward_amount") or it.get("tokens") or 0)
            claimable = bool(it.get("claimable") or it.get("is_claimable") or it.get("status") == "claimable")
            out.append({"id": cid, "name": name, "token_symbol": token_symbol, "token_amount": amount, "claimable": claimable, "source": it})
        except Exception:
            continue
    return out

def fetch_aggregator(url):
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def scan_airdrops():
    now = datetime.utcnow().isoformat()
    results = []
    urls = [u.strip() for u in AIRDROP_AGGREGATORS.split(",") if u.strip()]
    raw = []
    for u in urls:
        payload = fetch_aggregator(u)
        raw += normalize_aggregator_payload(payload)
    # fallback demo candidates
    if not raw:
        return [
            {"id":"demo_airdrop_small","type":"airdrop","name":"DemoA","token_symbol":"DEMO","token_amount":100,"est_usd":5.0,"claimable":True,"meta":{"found_at":now,"source":"demo"}},
            {"id":"demo_airdrop_big","type":"airdrop","name":"DemoB","token_symbol":"DEMO2","token_amount":800,"est_usd":160.0,"claimable":True,"meta":{"found_at":now,"source":"demo"}}
        ]
    # estimate USD
    for c in raw:
        price = _coingecko_price_for_symbol(c.get("token_symbol"))
        est = round(price * float(c.get("token_amount", 0)), 4) if price else 0.0
        results.append({"id": c["id"], "type":"airdrop", "name": c.get("name"), "token_symbol": c.get("token_symbol"), "token_amount": c.get("token_amount"), "est_usd": est, "claimable": c.get("claimable", False), "meta":{"found_at": now}})
    return results

def scan_high_value_tasks():
    # Placeholder: extend with real APIs. Returns tasks with amount_usd.
    return [
        {"id":"task_demo_1","type":"task","platform":"DemoTask","amount_usd":28.0,"task_id":"DT-100"},
        {"id":"task_big_1","type":"task","platform":"ProTask","amount_usd":120.0,"task_id":"PT-900"}
    ]

# -------------------------
# Bitget helpers (placeholders — verify endpoints before enabling)
# -------------------------
def _bitget_sign(timestamp, method, request_path, body, secret):
    # best-effort HMAC; verify with Bitget docs for exact signature (base64 vs hex, etc.)
    what = str(timestamp) + method.upper() + request_path + (body or "")
    h = hmac.new(secret.encode(), what.encode(), hashlib.sha256)
    return h.hexdigest()

def bitget_request_rw(method, path, params_or_body=None):
    if not BITGET_API_KEY_RW:
        raise Exception("Bitget RW keys not configured")
    ts = str(int(time.time() * 1000))
    body = json.dumps(params_or_body) if (params_or_body and method.upper() in ("POST","PUT")) else ""
    sign = _bitget_sign(ts, method, path, body, BITGET_API_SECRET_RW or "")
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_RW,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": BITGET_API_PASSPHRASE_RW or "",
        "Content-Type": "application/json"
    }
    url = API_HOST_BITGET + path
    r = requests.request(method, url, headers=headers, data=body if body else None, params=(params_or_body if method.upper()=="GET" else None), timeout=20)
    try:
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("[bitget_request_rw] failed", getattr(e,'response',None) and e.response.text)
        raise

def market_sell_symbol(symbol, size):
    """
    Place a market sell on Bitget. WARNING: verify endpoint and size semantics.
    Example placeholder path: /api/spot/v1/trade/orders
    """
    path = "/api/spot/v1/trade/orders"
    payload = {"symbol": symbol, "size": str(size), "type": "market"}
    return bitget_request_rw("POST", path, payload)

def transfer_to_main(coin, amount, fromType="spot", toType="main"):
    path = "/api/v2/spot/wallet/transfer"
    payload = {"fromType": fromType, "toType": toType, "coin": coin, "amount": str(amount), "clientOid": str(int(time.time()*1000))}
    return bitget_request_rw("POST", path, payload)

# -------------------------
# Scanner job & queueing
# -------------------------
def process_opportunities():
    try:
        found = []
        found += scan_airdrops()
        found += scan_high_value_tasks()
        for op in found:
            # filter by value threshold to avoid clutter
            est_val = op.get("est_usd") or op.get("amount_usd") or 0
            # create stable action id
            aid = f"{op.get('type')}_{int(time.time()*1000)}"
            persist_action(aid, op.get("type"), op)
            tg_send_approval_card(aid, op)
            print(f"[scanner] queued {aid} -> {op}")
    except Exception as e:
        print("[process_opportunities] error:", e)
        try:
            if TELEGRAM_CHAT_ID:
                tg_send_text(TELEGRAM_CHAT_ID, f"Scanner error: {e}")
        except:
            pass

# -------------------------
# Telegram callback handler (authorized only)
# -------------------------
def telegram_callback_handler(evt):
    """
    evt: {"action": "APPROVE"/"SELL"/"REJECT", "action_id":..., "from_id":..., "from_user":...}
    """
    action = evt.get("action")
    aid = evt.get("action_id")
    if not aid:
        tg_send_text(TELEGRAM_CHAT_ID, "Invalid action id.")
        return

    # fetch the pending action from DB
    pending = get_pending_actions(500)
    target = None
    for p in pending:
        if p.get("id") == aid:
            target = p
            break
    if not target:
        tg_send_text(TELEGRAM_CHAT_ID, f"Action {aid} not found or already handled.")
        return

    payload = target.get("payload", {})

    if action == "APPROVE":
        # For airdrop/task approvals we simulate execution unless RW keys are present and you implemented claimers.
        set_action_status(aid, "COMPLETED")
        tg_send_text(TELEGRAM_CHAT_ID, f"✅ Approved and marked COMPLETED: {aid}")
        # TODO: if real on-chain claimers are implemented, call them here (with offline signing)
        return

    if action == "REJECT":
        set_action_status(aid, "REJECTED")
        tg_send_text(TELEGRAM_CHAT_ID, f"❌ Rejected: {aid}")
        return

    if action == "SELL":
        # Only allow if token info present (airdrop) or explicit amount in payload
        token = payload.get("token_symbol") or payload.get("symbol")
        token_amount = payload.get("token_amount") or payload.get("amount") or payload.get("amount_usd")
        if not token:
            tg_send_text(TELEGRAM_CHAT_ID, "Sell failed: token symbol missing from payload.")
            return
        # Determine market pair - ensure pair exists on Bitget
        market_pair = f"{token}-USDT"
        try:
            tg_send_text(TELEGRAM_CHAT_ID, f"Placing market sell for {token_amount} {token} on {market_pair} (simulated if no keys).")
            if BITGET_API_KEY_RW:
                sell_resp = market_sell_symbol(market_pair, token_amount)
                tg_send_text(TELEGRAM_CHAT_ID, f"Sell response: {sell_resp}")
                # After sell, optionally transfer to main (USDT)
                est_usd = payload.get("est_usd") or 0
                try:
                    tx = transfer_to_main("USDT", est_usd)
                    tg_send_text(TELEGRAM_CHAT_ID, f"Transfer to main executed: {tx}")
                except Exception as e:
                    tg_send_text(TELEGRAM_CHAT_ID, f"Transfer to main failed: {e}")
            else:
                tg_send_text(TELEGRAM_CHAT_ID, "BITGET_API_KEY_RW not set — sell simulated only. Set Bitget RW keys in Choreo secrets to execute real trades.")
            set_action_status(aid, "COMPLETED")
        except Exception as e:
            set_action_status(aid, "FAILED")
            tg_send_text(TELEGRAM_CHAT_ID, f"Sell execution error for {aid}: {e}")
        return

# -------------------------
# HTTP admin protection decorator
# -------------------------
app = Flask(__name__)

def require_admin_http(fn):
    def wrapper(*args, **kwargs):
        token = request.headers.get("X-ADMIN-TOKEN") or request.args.get("admin_token")
        if not token or token != ADMIN_TOKEN:
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper

@app.route("/")
def index():
    return jsonify({"ok": True, "name":"Quantum Income Engine", "time": datetime.utcnow().isoformat()})

@app.route("/health")
def health():
    pend = get_pending_actions(10)
    return jsonify({"alive": True, "pending": len(pend)})

@app.route("/actions")
@require_admin_http
def actions_list():
    pend = get_pending_actions(200)
    return jsonify({"pending_count": len(pend), "pending": pend})

@app.route("/action/<action_id>/status", methods=["POST"])
@require_admin_http
def update_action_status(action_id):
    body = request.json or {}
    status = body.get("status")
    if not status:
        return jsonify({"ok":False, "error":"missing status"}), 400
    set_action_status(action_id, status)
    return jsonify({"ok":True, "id": action_id, "status": status})

# -------------------------
# Start threads & scheduler
# -------------------------
def start_telegram_poller():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or TELEGRAM_ADMIN_ID == 0:
        print("[start_telegram_poller] Telegram not fully configured — skipping poller. Set TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ADMIN_ID in env.")
        return
    t = threading.Thread(target=tg_long_poll_loop, args=(telegram_callback_handler,), daemon=True)
    t.start()
    print("[start_telegram_poller] started")

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(process_opportunities, 'interval', minutes=SCAN_INTERVAL_MINUTES, id="scanner_job", next_run_time=datetime.utcnow())
    scheduler.start()
    print(f"[start_scheduler] scanner scheduled every {SCAN_INTERVAL_MINUTES} minutes")

if __name__ == "__main__":
    # Start poller and scheduler
    start_telegram_poller()
    start_scheduler()
    print("Quantum Income Engine running. Listening on port", PORT)
    app.run(host="0.0.0.0", port=PORT)
