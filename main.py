#!/usr/bin/env python3
"""
Quantum Income Engine - single-file full app (main.py)

Features:
- Live scanner (airdrops + tasks) running on a scheduler
- Telegram bot integration with admin-only command handlers:
    /help    - show commands
    /status  - show pending actions + uptime
    /run     - trigger a manual scan now
    /restart - restart the service (admin only)
- Inline approval buttons for each queued action (Approve / Sell & Transfer / Reject)
  (Only TELEGRAM_ADMIN_ID is allowed to click inline buttons)
- Startup / error / progress notifications forwarded to Telegram admin
- Flask keep-alive endpoints: / , /ping , /status  (HTTP admin endpoints protected by X-ADMIN-TOKEN)
- Persistent SQLite action queue to survive restarts

IMPORTANT:
- Put secrets in Choreo secrets / env vars, not in code.
- Do NOT enable BITGET_API_KEY_RW unless you fully tested sandbox and understand risks.
- This app queues actions and sends approvals; money-moving operations are gated behind admin approvals.
"""

import os
import time
import json
import hmac
import hashlib
import threading
import sqlite3
import requests
import traceback
from datetime import datetime
from flask import Flask, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler

# -------------------------
# Configuration (set these in Choreo secrets)
# -------------------------
PORT = int(os.getenv("PORT", "8080"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")         # owner bot chat id (string)
TELEGRAM_ADMIN_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "0")) # numeric Telegram user id allowed to act
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")                   # for HTTP admin endpoints
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))
AIRDROP_AGGREGATORS = os.getenv("AIRDROP_AGGREGATORS", "")   # csv of aggregator URLs (optional)
AUTO_APPROVE_USD_THRESHOLD = float(os.getenv("AUTO_APPROVE_USD_THRESHOLD", "50.0"))

# Bitget RW keys (leave empty for simulation; store in Choreo secrets if used)
BITGET_API_KEY_RW = os.getenv("BITGET_API_KEY_RW", "")
BITGET_API_SECRET_RW = os.getenv("BITGET_API_SECRET_RW", "")
BITGET_API_PASSPHRASE_RW = os.getenv("BITGET_API_PASSPHRASE_RW", "")
API_HOST_BITGET = "https://api.bitget.com"

# -------------------------
# DB - SQLite persistent queue
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
# Telegram helpers
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
        # capture response text if available
        resp_text = getattr(e, 'response', None) and getattr(e.response, 'text', None)
        print("[tg_api] error:", e, resp_text)
        return None

def tg_send_text(chat_id, text):
    # simple sendMessage using HTML mode to avoid Markdown pitfalls
    if not TELEGRAM_TOKEN or not chat_id:
        print("[tg_send_text] missing token or chat_id", chat_id)
        return
    return tg_api("sendMessage", {"chat_id": chat_id, "text": text, "parse_mode":"HTML"})

def tg_send_startup():
    try:
        tg_send_text(TELEGRAM_CHAT_ID, f"✅ Quantum Income Engine started at {datetime.utcnow().isoformat()} UTC")
    except Exception as e:
        print("[tg_send_startup] failed:", e)

def tg_send_error(e, context=""):
    msg = f"⚠️ Error in Quantum Income Engine\nContext: {context}\n{str(e)}\n{traceback.format_exc()[:1500]}"
    try:
        tg_send_text(TELEGRAM_CHAT_ID, msg)
    except Exception:
        print("[tg_send_error] failed to send to telegram:", msg)

def tg_send_approval_card(action_id, payload):
    if not TELEGRAM_CHAT_ID:
        print("[tg_send_approval_card] TELEGRAM_CHAT_ID not set; action:", action_id)
        return
    # create a compact summary
    summary = {}
    for k in ("type","name","token_symbol","token_amount","est_usd","amount_usd","platform","task_id"):
        if payload.get(k) is not None:
            summary[k] = payload.get(k)
    text = f"🔔 <b>Quantum Income Engine</b>\nAction: <code>{action_id}</code>\n{json.dumps(summary, default=str)}"
    reply_markup = {"inline_keyboard":[
        [{"text":"✅ Approve","callback_data": f"APPROVE::{action_id}"},
         {"text":"💱 Sell & Transfer","callback_data": f"SELL::{action_id}"},
         {"text":"❌ Reject","callback_data": f"REJECT::{action_id}"}]
    ]}
    body = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "reply_markup": reply_markup, "parse_mode": "HTML"}
    return tg_api("sendMessage", body)

# -------------------------
# Telegram poller - getUpdates long poll to handle messages and callbacks
# -------------------------
def tg_long_poll_loop():
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
                # callback_query (inline buttons)
                if "callback_query" in upd:
                    cb = upd["callback_query"]
                    data_cb = cb.get("data","")
                    from_obj = cb.get("from", {})
                    from_id = from_obj.get("id")
                    from_user = from_obj.get("username", "?")
                    if "::" in data_cb:
                        cmd, aid = data_cb.split("::",1)
                    else:
                        cmd, aid = data_cb, None

                    # Non-admin clicking inline buttons
                    if int(from_id) != TELEGRAM_ADMIN_ID:
                        try:
                            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
                                          json={"callback_query_id": cb.get("id"), "text": "Not authorized"})
                        except:
                            pass
                        # notify owner
                        tg_send_text(TELEGRAM_CHAT_ID, f"⚠️ Unauthorized button attempt by @{from_user} (id:{from_id}) on {cmd} {aid}")
                        continue

                    # Admin clicked: answer and handle
                    try:
                        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
                                      json={"callback_query_id": cb.get("id"), "text": "Action received"})
                    except:
                        pass
                    handle_telegram_action({"action": cmd, "action_id": aid, "from_id": from_id, "from_user": from_user})

                # plain messages (commands)
                elif "message" in upd:
                    msg = upd["message"]
                    chat = msg.get("chat",{})
                    from_obj = msg.get("from",{})
                    text = msg.get("text","")
                    chat_id = chat.get("id")
                    user_id = from_obj.get("id")
                    username = from_obj.get("username","?")
                    # Only accept commands from admin user (or allow owner chat ID if needed)
                    if text and text.startswith("/"):
                        handle_command(text.strip(), user_id, chat_id, username)
            time.sleep(0.5)
        except Exception as e:
            print("[tg_long_poll] error:", e)
            time.sleep(2)

# -------------------------
# Command & callback handlers
# -------------------------
def handle_command(text, user_id, chat_id, username):
    # Only allow admin to use bot commands
    if int(user_id) != TELEGRAM_ADMIN_ID:
        tg_send_text(chat_id, "Not authorized to use this bot.")
        return

    cmd_parts = text.split()
    cmd = cmd_parts[0].lower()

    if cmd == "/help":
        tg_send_text(chat_id, "Commands:\n/help - show this\n/status - show pending actions & uptime\n/run - trigger manual scan now\n/restart - restart service (Choreo will restart it)")
        return
    if cmd == "/status":
        pending = get_pending_actions(200)
        tg_send_text(chat_id, f"Status:\nPending actions: {len(pending)}\nUptime (UTC): {datetime.utcnow().isoformat()}")
        return
    if cmd == "/run":
        tg_send_text(chat_id, "Manual scan triggered.")
        # run scan in new thread to avoid blocking poller
        threading.Thread(target=process_opportunities, daemon=True).start()
        return
    if cmd == "/restart":
        tg_send_text(chat_id, "Restarting service now (approved by admin).")
        # flush DB commit, then exit process. Choreo should restart the app automatically.
        try:
            conn.commit()
        except:
            pass
        # small delay so admin sees message
        time.sleep(1)
        os._exit(0)
        return
    tg_send_text(chat_id, "Unknown command. Send /help for commands.")

def handle_telegram_action(evt):
    """
    evt: {"action":"APPROVE"/"SELL"/"REJECT", "action_id":..., "from_id":..., "from_user":...}
    """
    action = evt.get("action")
    aid = evt.get("action_id")
    if not aid:
        tg_send_text(TELEGRAM_CHAT_ID, "Invalid action id.")
        return

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

    try:
        if action == "APPROVE":
            # Mark as completed; in future integrate actual claimers / executors (admin-only operations)
            set_action_status(aid, "COMPLETED")
            tg_send_text(TELEGRAM_CHAT_ID, f"✅ Approved and marked COMPLETED: {aid}")
            return

        if action == "REJECT":
            set_action_status(aid, "REJECTED")
            tg_send_text(TELEGRAM_CHAT_ID, f"❌ Rejected: {aid}")
            return

        if action == "SELL":
            # SELL action will attempt to call Bitget sell if RW keys set; otherwise simulated
            token = payload.get("token_symbol") or payload.get("symbol")
            token_amount = payload.get("token_amount") or payload.get("amount") or payload.get("amount_usd")
            if not token:
                tg_send_text(TELEGRAM_CHAT_ID, "Sell failed: missing token info")
                return
            market_pair = f"{token}-USDT"
            tg_send_text(TELEGRAM_CHAT_ID, f"Attempting sell for {token_amount} {token} on {market_pair} (simulated if no keys).")
            if BITGET_API_KEY_RW:
                try:
                    sell_resp = market_sell_symbol(market_pair, token_amount)
                    tg_send_text(TELEGRAM_CHAT_ID, f"Sell response: {sell_resp}")
                    # optionally transfer to main to prepare for withdraw
                    est_usd = payload.get("est_usd") or 0
                    try:
                        tx = transfer_to_main("USDT", est_usd)
                        tg_send_text(TELEGRAM_CHAT_ID, f"Transfer to main executed: {tx}")
                    except Exception as e:
                        tg_send_text(TELEGRAM_CHAT_ID, f"Transfer to main failed: {e}")
                except Exception as e:
                    tg_send_text(TELEGRAM_CHAT_ID, f"Sell failed: {e}")
            else:
                tg_send_text(TELEGRAM_CHAT_ID, "BITGET API RW keys not set — sell simulated only.")
            set_action_status(aid, "COMPLETED")
            return

    except Exception as e:
        tg_send_error(e, context=f"handle_telegram_action {action} {aid}")
        set_action_status(aid, "FAILED")
        return

# -------------------------
# Scanners (airdrops + tasks) - live scans that queue actions (no auto-claim)
# -------------------------
COINGECKO_SEARCH = "https://api.coingecko.com/api/v3/search"
COINGECKO_PRICE = "https://api.coingecko.com/api/v3/simple/price"

def _coingecko_price_for_symbol(symbol):
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
    urls = [u.strip() for u in (AIRDROP_AGGREGATORS or "").split(",") if u.strip()]
    raw = []
    for u in urls:
        try:
            payload = fetch_aggregator(u)
            raw += normalize_aggregator_payload(payload)
        except Exception:
            continue
    # If no external aggregators configured, we still run quiet (no demo spam)
    if not raw:
        return []
    for c in raw:
        price = _coingecko_price_for_symbol(c.get("token_symbol"))
        est = round(price * float(c.get("token_amount", 0)), 4) if price else 0.0
        results.append({"id": c["id"], "type":"airdrop", "name": c.get("name"), "token_symbol": c.get("token_symbol"), "token_amount": c.get("token_amount"), "est_usd": est, "claimable": c.get("claimable", False), "meta":{"found_at": now}})
    return results

def scan_high_value_tasks():
    # Replace with real task-API integrations when you want (Remotasks, Clickworker, Toloka, etc.)
    # For now, perform no fake/demo returns — keep it live-only if configured
    return []

# -------------------------
# Bitget minimal helpers (placeholders) - DO NOT ENABLE WITHOUT TESTING
# -------------------------
def _bitget_sign(timestamp, method, request_path, body, secret):
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
        print("[bitget_request_rw] failed", getattr(e,'response',None) and getattr(e.response,'text',None))
        raise

def market_sell_symbol(symbol, size):
    # placeholder -> verify with Bitget docs before enabling
    path = "/api/spot/v1/trade/orders"
    payload = {"symbol": symbol, "size": str(size), "type": "market"}
    return bitget_request_rw("POST", path, payload)

def transfer_to_main(coin, amount, fromType="spot", toType="main"):
    path = "/api/v2/spot/wallet/transfer"
    payload = {"fromType": fromType, "toType": toType, "coin": coin, "amount": str(amount), "clientOid": str(int(time.time()*1000))}
    return bitget_request_rw("POST", path, payload)

# -------------------------
# Core scanner loop
# -------------------------
def process_opportunities():
    try:
        found = []
        # gather live airdrops (only if you configured aggregators)
        try:
            found += scan_airdrops()
        except Exception as e:
            print("[process_opportunities] airdrop scan error:", e)
        # gather tasks if you add any real integrations
        try:
            found += scan_high_value_tasks()
        except Exception as e:
            print("[process_opportunities] tasks scan error:", e)

        # queue found opportunities for admin approval
        for op in found:
            # optionally filter small ones if you want
            est_val = op.get("est_usd") or op.get("amount_usd") or 0
            # create action id
            aid = f"{op.get('type')}_{int(time.time()*1000)}"
            persist_action(aid, op.get("type"), op)
            tg_send_approval_card(aid, op)
            print(f"[scanner] queued {aid} -> {op.get('name')}")
        # send a small heartbeat if any found
        if found:
            tg_send_text(TELEGRAM_CHAT_ID, f"Scanner discovered {len(found)} opportunities. Use /status to inspect.")
    except Exception as e:
        tg_send_error(e, context="process_opportunities")
        print("[process_opportunities] error:", e)

# -------------------------
# Flask app & admin-protected endpoints
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

@app.route("/ping")
def ping():
    return "pong", 200

@app.route("/status")
@require_admin_http
def status():
    pending = get_pending_actions(200)
    return jsonify({"pending_count": len(pending), "pending": pending})

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
# Scheduler & threads start
# -------------------------
def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(process_opportunities, 'interval', minutes=SCAN_INTERVAL_MINUTES, id="scanner_job", next_run_time=datetime.utcnow())
    scheduler.start()
    print(f"[scheduler] scanner scheduled every {SCAN_INTERVAL_MINUTES} minutes")

def start_telegram_poller():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or TELEGRAM_ADMIN_ID == 0:
        print("[start_telegram_poller] Telegram not fully configured; set TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ADMIN_ID in env.")
        return
    t = threading.Thread(target=tg_long_poll_loop, daemon=True)
    t.start()
    print("[tg_poller] started")

if __name__ == "__main__":
    # start background services
    try:
        start_telegram_poller()
        start_scheduler()
        # send startup message
        try:
            tg_send_startup()
        except Exception as e:
            print("[startup] failed to send telegram startup:", e)
    except Exception as e:
        print("[init] startup error:", e)
        tg_send_error(e, context="startup")

    print("Quantum Income Engine running. Listening on port", PORT)
    # run Flask app (main thread)
    app.run(host="0.0.0.0", port=PORT)
