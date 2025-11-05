#!/usr/bin/env python3
"""
Quantum Income Engine - single-file live aggregator (main.py)

Set env vars in Choreo (do NOT commit secrets to git):
TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ADMIN_ID, ADMIN_TOKEN,
AIRDROP_AGGREGATORS, ZEALY_COMMUNITIES, GALXE_API_URLS, TASKON_API_URLS,
BITGET_EARN_URLS, SCAN_INTERVAL_MINUTES (default 15), MIN_TASK_USD (default 20), PORT (default 8080)
"""
import os, time, json, hmac, hashlib, sqlite3, traceback, threading, asyncio
from datetime import datetime
from flask import Flask, jsonify, request
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler, MessageHandler, filters
)

# -------------------------
# Config
# -------------------------
PORT = int(os.getenv("PORT", "8080"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ADMIN_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "0"))  # numeric admin id
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "15"))
AIRDROP_AGGREGATORS = os.getenv("AIRDROP_AGGREGATORS", "")
ZEALY_COMMUNITIES = os.getenv("ZEALY_COMMUNITIES", "")
GALXE_API_URLS = os.getenv("GALXE_API_URLS", "")
TASKON_API_URLS = os.getenv("TASKON_API_URLS", "")
BITGET_EARN_URLS = os.getenv("BITGET_EARN_URLS", "")
MIN_TASK_USD = float(os.getenv("MIN_TASK_USD", "20"))

# Bitget keys (leave blank for simulation)
BITGET_API_KEY_RW = os.getenv("BITGET_API_KEY_RW", "")
BITGET_API_SECRET_RW = os.getenv("BITGET_API_SECRET_RW", "")
BITGET_API_PASSPHRASE_RW = os.getenv("BITGET_API_PASSPHRASE_RW", "")
API_HOST_BITGET = "https://api.bitget.com"

# -------------------------
# Persistent DB (SQLite)
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

def persist_action(aid, a_type, payload):
    cur.execute("INSERT OR REPLACE INTO actions (id,type,payload,created_at,status) VALUES (?,?,?,?,?)",
                (aid, a_type, json.dumps(payload), datetime.utcnow().isoformat(), "PENDING"))
    conn.commit()

def set_action_status(aid, status):
    cur.execute("UPDATE actions SET status=? WHERE id=?", (status, aid))
    conn.commit()

def list_pending(limit=200):
    cur.execute("SELECT id,type,payload,created_at FROM actions WHERE status='PENDING' ORDER BY created_at LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [{"id":r[0],"type":r[1],"payload":json.loads(r[2]),"created":r[3]} for r in rows]

def remove_action(aid):
    cur.execute("DELETE FROM actions WHERE id=?", (aid,))
    conn.commit()

# -------------------------
# Flask keep-alive app
# -------------------------
flask_app = Flask(__name__)

@flask_app.route("/")
def index():
    return jsonify({"ok": True, "name": "Quantum Income Engine", "time": datetime.utcnow().isoformat()})

@flask_app.route("/ping")
def ping():
    return "pong", 200

def require_admin_http(fn):
    def wrapper(*args, **kwargs):
        token = request.headers.get("X-ADMIN-TOKEN") or request.args.get("admin_token")
        if not token or token != ADMIN_TOKEN:
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper

@flask_app.route("/status")
@require_admin_http
def http_status():
    pend = list_pending(200)
    return jsonify({"pending_count": len(pend), "pending": pend})

# -------------------------
# Telegram bot setup (async)
# -------------------------
APPLICATION = None  # will be set to Application instance

async def send_startup_message():
    try:
        if TELEGRAM_CHAT_ID and APPLICATION:
            await APPLICATION.bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                                               text=f"✅ Quantum Income Engine started at {datetime.utcnow().isoformat()} UTC")
    except Exception as e:
        print("[startup msg failed]", e)

async def send_text_async(chat_id, text):
    if not APPLICATION:
        print("[send_text_async] application not ready")
        return
    try:
        await APPLICATION.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
    except Exception as e:
        print("[send_text_async] error:", e)

def send_text(chat_id, text):
    # helper usable from sync threads
    if not APPLICATION:
        print("[send_text] app not ready:", text)
        return
    coro = APPLICATION.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
    asyncio.run_coroutine_threadsafe(coro, APPLICATION.updater.loop)

def send_approval_card_sync(aid, payload):
    # builds message and inline keyboard, sends from sync code
    summary = {}
    for k in ("type","name","token_symbol","token_amount","est_usd","amount_usd","platform","task_id","title"):
        if payload.get(k) is not None:
            summary[k] = payload.get(k)
    text = f"🔔 <b>Quantum Income Engine</b>\nAction: <code>{aid}</code>\n{json.dumps(summary, default=str)}"
    kb = [
        [InlineKeyboardButton("✅ Approve", callback_data=f"APPROVE::{aid}"),
         InlineKeyboardButton("💱 Sell & Transfer", callback_data=f"SELL::{aid}"),
         InlineKeyboardButton("❌ Reject", callback_data=f"REJECT::{aid}")]
    ]
    markup = InlineKeyboardMarkup(kb)
    if APPLICATION:
        coro = APPLICATION.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, reply_markup=markup, parse_mode="HTML")
        asyncio.run_coroutine_threadsafe(coro, APPLICATION.updater.loop)
    else:
        print("[approval_card] application not ready. action:", aid)

# -------------------------
# Live aggregator scanners (Zealy + generic)
# -------------------------
COINGECKO_SEARCH = "https://api.coingecko.com/api/v3/search"
COINGECKO_PRICE = "https://api.coingecko.com/api/v3/simple/price"

def _safe_get_json(url, headers=None, params=None, timeout=10):
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[safe_get] failed {url}: {e}")
        return None

# Zealy public quests
ZEALY_BASE = "https://api-v2.zealy.io/public/communities/{subdomain}/quests"
def scan_zealy():
    out = []
    subs = [s.strip() for s in ZEALY_COMMUNITIES.split(",") if s.strip()]
    for sub in subs:
        url = ZEALY_BASE.format(subdomain=sub)
        js = _safe_get_json(url)
        if not js:
            continue
        items = js.get("quests") or js.get("data") or js.get("result") or js
        if isinstance(items, dict):
            items = list(items.values())
        if not isinstance(items, list):
            continue
        for q in items:
            try:
                qid = q.get("id") or q.get("questId") or str(int(time.time()*1000))
                title = q.get("name") or q.get("title") or (q.get("description") or "")[:120]
                amount_usd = float(q.get("rewardAmount") or q.get("amountUsd") or 0)
                out.append({"id": f"zealy_{sub}_{qid}", "type":"task", "platform":"Zealy", "task_id": qid, "title": title, "amount_usd": amount_usd, "meta":{"community":sub}})
            except Exception:
                continue
    return out

# generic aggregator URLs for Galxe/TaskOn/Bitget Earn
def _urls_from_env(name):
    return [u.strip() for u in (os.getenv(name,"") or "").split(",") if u.strip()]

def scan_generic_aggregators():
    out = []
    # Galxe-like endpoints (airdrops/campaigns)
    for url in _urls_from_env("GALXE_API_URLS"):
        js = _safe_get_json(url)
        if not js:
            continue
        cand = js.get("airdrops") or js.get("campaigns") or js.get("data") or js.get("quests") or js
        if isinstance(cand, dict):
            cand = list(cand.values())
        if isinstance(cand, list):
            for c in cand:
                try:
                    cid = c.get("id") or c.get("slug") or str(int(time.time()*1000))
                    name = c.get("name") or c.get("title") or ""
                    token_symbol = (c.get("token_symbol") or c.get("symbol") or "").upper()
                    token_amount = float(c.get("amount") or c.get("reward_amount") or 0)
                    est_usd = 0.0
                    out.append({"id": f"galxe_{cid}", "type":"airdrop", "name":name, "token_symbol":token_symbol, "token_amount":token_amount, "est_usd":est_usd, "meta":{"source":url}})
                except Exception:
                    continue
    # TaskOn
    for url in _urls_from_env("TASKON_API_URLS"):
        js = _safe_get_json(url)
        if not js:
            continue
        cand = js.get("tasks") or js.get("data") or js
        if isinstance(cand, dict):
            cand = list(cand.values())
        if isinstance(cand, list):
            for t in cand:
                try:
                    tid = t.get("id") or t.get("taskId") or str(int(time.time()*1000))
                    title = t.get("title") or t.get("name") or ""
                    amount_usd = float(t.get("amount_usd") or t.get("reward_usd") or 0)
                    out.append({"id": f"taskon_{tid}", "type":"task", "platform":"TaskOn", "task_id":tid, "title":title, "amount_usd":amount_usd, "meta":{"source":url}})
                except Exception:
                    continue
    # Bitget Earn endpoints
    for url in _urls_from_env("BITGET_EARN_URLS"):
        js = _safe_get_json(url)
        if not js:
            continue
        cand = js.get("data") or js.get("products") or js
        if isinstance(cand, dict):
            cand = list(cand.values())
        if isinstance(cand, list):
            for b in cand:
                try:
                    bid = b.get("id") or b.get("productId") or str(int(time.time()*1000))
                    name = b.get("name") or b.get("title") or ""
                    est_usd = float(b.get("estimatedRewardUsd") or 0)
                    out.append({"id": f"bitget_earn_{bid}", "type":"airdrop", "name":name, "est_usd":est_usd, "meta":{"source":url}})
                except Exception:
                    continue
    return out

def scan_airdrops():
    results = []
    results += scan_generic_aggregators()
    return results

def scan_high_value_tasks():
    results = []
    results += scan_zealy()
    # include task-type items from generic aggregators
    for r in scan_generic_aggregators():
        if r.get("type") == "task" or r.get("platform") == "TaskOn":
            results.append(r)
    # filter by MIN_TASK_USD (keep unknown amount items)
    filtered = []
    for r in results:
        amt = r.get("amount_usd") or r.get("est_usd") or 0
        if amt == 0 or amt >= MIN_TASK_USD:
            filtered.append(r)
    return filtered

# -------------------------
# Bitget placeholders (do not enable without sandbox/testing)
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
    headers = {"ACCESS-KEY": BITGET_API_KEY_RW, "ACCESS-SIGN": sign, "ACCESS-TIMESTAMP": ts, "ACCESS-PASSPHRASE": BITGET_API_PASSPHRASE_RW or "", "Content-Type":"application/json"}
    url = API_HOST_BITGET + path
    r = requests.request(method, url, headers=headers, data=body if body else None, params=(params_or_body if method.upper()=="GET" else None), timeout=20)
    r.raise_for_status()
    return r.json()

# -------------------------
# Core processing & scheduling
# -------------------------
def process_opportunities():
    try:
        found = []
        found += scan_airdrops()
        found += scan_high_value_tasks()
        for op in found:
            aid = f"{op.get('type')}_{int(time.time()*1000)}"
            persist_action(aid, op.get("type"), op)
            print(f"[scanner] queued {aid} -> {op}")
            send_approval_card_sync(aid, op)
        if found:
            send_text(TELEGRAM_CHAT_ID, f"Scanner discovered {len(found)} opportunities. Use /tasks or /airdrops to inspect.")
    except Exception as e:
        print("[process_opportunities] error:", e)
        send_text(TELEGRAM_CHAT_ID, f"Scanner error: {e}")

# -------------------------
# Telegram command handlers (async)
# -------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = ("👋 Welcome to <b>Quantum Income Engine</b>\n"
           "Admin-only control. Use /help for commands.")
    await update.message.reply_text(txt, parse_mode="HTML")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = ("/help - show this\n/status - system status (admin)\n/scan - run live scan (admin)\n"
           "/tasks - list recent tasks (admin)\n/airdrops - list recent airdrops (admin)\n/stop - pause scanning (admin)\n/resume - resume scanning (admin)")
    await update.message.reply_text(txt)

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_ADMIN_ID:
        await update.message.reply_text("🚫 Not authorized.")
        return
    pending = list_pending(200)
    next_run = None
    try:
        sched = scheduler.get_jobs()
        if sched:
            next_run = sched[0].next_run_time
    except Exception:
        pass
    txt = f"🩵 System Status\nPending: {len(pending)}\nNext scan: {next_run}\nUptime: {datetime.utcnow().isoformat()} UTC"
    await update.message.reply_text(txt)

async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_ADMIN_ID:
        await update.message.reply_text("🚫 Not authorized.")
        return
    await update.message.reply_text("🔍 Running live scan now...")
    threading.Thread(target=process_opportunities, daemon=True).start()

async def tasks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_ADMIN_ID:
        await update.message.reply_text("🚫 Not authorized.")
        return
    pending = list_pending(100)
    tasks = [p for p in pending if p.get("type") == "task"]
    if not tasks:
        await update.message.reply_text("⚠️ No tasks queued.")
        return
    msg = "🧩 Pending Tasks:\n"
    for t in tasks[:10]:
        pl = t["payload"]
        amt = pl.get("amount_usd") or pl.get("est_usd") or 0
        title = pl.get("title") or pl.get("name") or pl.get("platform") or ""
        msg += f"• {title} — ${amt} (id: {t['id']})\n"
    await update.message.reply_text(msg)

async def airdrops_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_ADMIN_ID:
        await update.message.reply_text("🚫 Not authorized.")
        return
    pending = list_pending(100)
    ads = [p for p in pending if p.get("type") == "airdrop"]
    if not ads:
        await update.message.reply_text("⚠️ No airdrops queued.")
        return
    msg = "🎁 Pending Airdrops:\n"
    for a in ads[:10]:
        pl = a["payload"]
        token = pl.get("token_symbol") or "?"
        amt = pl.get("token_amount") or pl.get("est_usd") or 0
        msg += f"• {pl.get('name','(no name)')} — {amt} {token} (id: {a['id']})\n"
    await update.message.reply_text(msg)

async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_ADMIN_ID:
        await update.message.reply_text("🚫 Not authorized.")
        return
    scheduler.pause()
    await update.message.reply_text("🛑 Auto-scan paused.")

async def resume_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_ADMIN_ID:
        await update.message.reply_text("🚫 Not authorized.")
        return
    scheduler.resume()
    await update.message.reply_text("▶️ Auto-scan resumed.")

# -------------------------
# CallbackQuery handler (Approve / Sell / Reject)
# -------------------------
async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # remove loading
    from_user = update.effective_user
    user_id = from_user.id
    data = query.data or ""
    if "::" in data:
        cmd, aid = data.split("::",1)
    else:
        await query.edit_message_text("Invalid callback.")
        return

    if user_id != TELEGRAM_ADMIN_ID:
        # not admin: notify owner and decline
        await query.edit_message_text("Not authorized.")
        await APPLICATION.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"⚠️ Unauthorized attempt by @{from_user.username} (id:{user_id}) on {cmd} {aid}")
        return

    # admin is allowed — handle actions
    pending = list_pending(500)
    target = next((p for p in pending if p['id'] == aid), None)
    if not target:
        await APPLICATION.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"Action {aid} not found or already handled.")
        return

    payload = target['payload']
    try:
        if cmd == "APPROVE":
            set_action_status(aid, "COMPLETED")
            await APPLICATION.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"✅ Approved: {aid}")
            return
        if cmd == "REJECT":
            set_action_status(aid, "REJECTED")
            await APPLICATION.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"❌ Rejected: {aid}")
            return
        if cmd == "SELL":
            token = payload.get("token_symbol") or payload.get("symbol")
            token_amount = payload.get("token_amount") or payload.get("amount") or payload.get("amount_usd")
            if not token:
                await APPLICATION.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="Sell failed: token info missing.")
                return
            market_pair = f"{token}-USDT"
            await APPLICATION.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"Attempting sell for {token_amount} {token} on {market_pair} (simulated if no keys).")
            if BITGET_API_KEY_RW:
                try:
                    resp = bitget_request_rw("POST", "/api/spot/v1/trade/orders", {"symbol": market_pair, "size": str(token_amount), "type":"market"})
                    await APPLICATION.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"Sell resp: {resp}")
                except Exception as e:
                    await APPLICATION.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"Sell error: {e}")
            else:
                await APPLICATION.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="BITGET RW keys not set — sell simulated only.")
            set_action_status(aid, "COMPLETED")
            return
    except Exception as e:
        await APPLICATION.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"Execution error: {e}")
        set_action_status(aid, "FAILED")
        return

# -------------------------
# Start everything
# -------------------------
# scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(process_opportunities, 'interval', minutes=SCAN_INTERVAL_MINUTES, id="scanner_job", next_run_time=datetime.utcnow())

def run_flask():
    flask_app.run(host="0.0.0.0", port=PORT)

def start_worker_and_bot():
    global APPLICATION
    APPLICATION = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # add command handlers
    APPLICATION.add_handler(CommandHandler("start", start_cmd))
    APPLICATION.add_handler(CommandHandler("help", help_cmd))
    APPLICATION.add_handler(CommandHandler("status", status_cmd))
    APPLICATION.add_handler(CommandHandler("scan", scan_cmd))
    APPLICATION.add_handler(CommandHandler("tasks", tasks_cmd))
    APPLICATION.add_handler(CommandHandler("airdrops", airdrops_cmd))
    APPLICATION.add_handler(CommandHandler("stop", stop_cmd))
    APPLICATION.add_handler(CommandHandler("resume", resume_cmd))

    # callback queries
    APPLICATION.add_handler(CallbackQueryHandler(callback_query_handler))

    # start scheduler
    scheduler.start()
    # start bot polling (blocking)
    APPLICATION.run_polling()

if __name__ == "__main__":
    # Run Flask in background thread (for web health)
    threading.Thread(target=run_flask, daemon=True).start()

    # Send startup message once bot is ready (we'll start bot in start_worker_and_bot)
    try:
        # start bot and scheduler together
        start_worker_and_bot()
    except Exception as e:
        print("Fatal error starting bot:", e)
        traceback.print_exc()
        # attempt notifying admin via direct HTTP call if token set
        try:
            if TELEGRAM_CHAT_ID and TELEGRAM_TOKEN:
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": f"Startup failed: {e}"})
        except Exception:
            pass
