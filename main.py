#!/usr/bin/env python3
"""
Quantum Income Engine - Single-file bot (signal-safe)

Environment variables:
TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ADMIN_ID, ADMIN_TOKEN,
AIRDROP_AGGREGATORS, ZEALY_COMMUNITIES, GALXE_API_URLS, TASKON_API_URLS,
BITGET_EARN_URLS, SCAN_INTERVAL_MINUTES (default 15), MIN_TASK_USD (default 20), PORT (default 8080)
"""
import os
import time
import json
import hmac
import hashlib
import sqlite3
import traceback
import threading
from datetime import datetime
from flask import Flask, jsonify, request
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler
)

# -------------------------
# Config
# -------------------------
PORT = int(os.getenv("PORT", "8080"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ADMIN_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "0"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "defaultadmintoken123")
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "15"))
AIRDROP_AGGREGATORS = os.getenv("AIRDROP_AGGREGATORS", "")
ZEALY_COMMUNITIES = os.getenv("ZEALY_COMMUNITIES", "")
GALXE_API_URLS = os.getenv("GALXE_API_URLS", "")
TASKON_API_URLS = os.getenv("TASKON_API_URLS", "")
BITGET_EARN_URLS = os.getenv("BITGET_EARN_URLS", "")
MIN_TASK_USD = float(os.getenv("MIN_TASK_USD", "20"))

BITGET_API_KEY_RW = os.getenv("BITGET_API_KEY_RW", "")
BITGET_API_SECRET_RW = os.getenv("BITGET_API_SECRET_RW", "")
BITGET_API_PASSPHRASE_RW = os.getenv("BITGET_API_PASSPHRASE_RW", "")
API_HOST_BITGET = "https://api.bitget.com"

# -------------------------
# SQLite persistence
# -------------------------
DB_PATH = os.getenv("DB_PATH", "quantum_income_actions.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute(
    """CREATE TABLE IF NOT EXISTS actions (
        id TEXT PRIMARY KEY,
        type TEXT,
        payload TEXT,
        created_at TEXT,
        status TEXT
    )"""
)
conn.commit()

def persist_action(aid, a_type, payload):
    cur.execute(
        "INSERT OR REPLACE INTO actions (id,type,payload,created_at,status) VALUES (?,?,?,?,?)",
        (aid, a_type, json.dumps(payload, default=str), datetime.utcnow().isoformat(), "PENDING"),
    )
    conn.commit()

def set_action_status(aid, status):
    cur.execute("UPDATE actions SET status=? WHERE id=?", (status, aid))
    conn.commit()

def list_pending(limit=200):
    cur.execute(
        "SELECT id,type,payload,created_at FROM actions WHERE status='PENDING' ORDER BY created_at LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    return [{"id": r[0], "type": r[1], "payload": json.loads(r[2]), "created": r[3]} for r in rows]

def remove_action(aid):
    cur.execute("DELETE FROM actions WHERE id=?", (aid,))
    conn.commit()

# -------------------------
# Flask keep-alive / health
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
# Telegram HTTP helpers
# -------------------------
APPLICATION = None  # set at runtime

def send_text(chat_id, text):
    if not TELEGRAM_TOKEN or not chat_id:
        print("[send_text] missing token or chat_id")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": str(chat_id), "text": text, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"[send_text] error: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                print(f"[send_text] response: {e.response.text}")
            except Exception:
                pass

def send_approval_card_sync(aid, payload):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[approval_card] missing config")
        return
    try:
        summary = {}
        for k in ("type", "name", "token_symbol", "token_amount", "est_usd", "amount_usd", "platform", "task_id", "title"):
            if payload.get(k) is not None:
                summary[k] = payload.get(k)
        text = f"🔔 <b>Quantum Income Engine</b>\nAction: <code>{aid}</code>\n<pre>{json.dumps(summary, indent=2, default=str)}</pre>"
        kb = {
            "inline_keyboard": [
                [
                    {"text": "✅ Approve", "callback_data": f"APPROVE::{aid}"},
                    {"text": "💱 Sell & Transfer", "callback_data": f"SELL::{aid}"},
                    {"text": "❌ Reject", "callback_data": f"REJECT::{aid}"}
                ]
            ]
        }
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload_data = {"chat_id": str(TELEGRAM_CHAT_ID), "text": text, "parse_mode": "HTML", "reply_markup": kb}
        r = requests.post(url, json=payload_data, timeout=10)
        r.raise_for_status()
        print(f"[approval_card] sent for {aid}")
        return r.json()
    except Exception as e:
        print(f"[approval_card] error: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                print(f"[approval_card] response: {e.response.text}")
            except Exception:
                pass

# -------------------------
# HTTP fetch helper and scanners
# -------------------------
def _safe_get_json(url, headers=None, params=None, timeout=10):
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[safe_get] failed {url}: {e}")
        return None

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
                qid = q.get("id") or q.get("questId") or str(int(time.time() * 1000))
                title = q.get("name") or q.get("title") or (q.get("description") or "")[:120]
                amount_usd = float(q.get("rewardAmount") or q.get("amountUsd") or 0)
                out.append({
                    "id": f"zealy_{sub}_{qid}",
                    "type": "task",
                    "platform": "Zealy",
                    "task_id": qid,
                    "title": title,
                    "amount_usd": amount_usd,
                    "meta": {"community": sub}
                })
            except Exception:
                continue
    return out

def _urls_from_env(name):
    return [u.strip() for u in (os.getenv(name, "") or "").split(",") if u.strip()]

def scan_generic_aggregators():
    out = []
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
                    cid = c.get("id") or c.get("slug") or str(int(time.time() * 1000))
                    name = c.get("name") or c.get("title") or ""
                    token_symbol = (c.get("token_symbol") or c.get("symbol") or "").upper()
                    token_amount = float(c.get("amount") or c.get("reward_amount") or 0)
                    est_usd = float(c.get("est_usd") or c.get("estimatedUsd") or 0)
                    out.append({
                        "id": f"galxe_{cid}",
                        "type": "airdrop",
                        "name": name,
                        "token_symbol": token_symbol,
                        "token_amount": token_amount,
                        "est_usd": est_usd,
                        "claimable": True,
                        "meta": {"source": url}
                    })
                except Exception:
                    continue

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
                    tid = t.get("id") or t.get("taskId") or str(int(time.time() * 1000))
                    title = t.get("title") or t.get("name") or ""
                    amount_usd = float(t.get("amount_usd") or t.get("reward_usd") or 0)
                    out.append({
                        "id": f"taskon_{tid}",
                        "type": "task",
                        "platform": "TaskOn",
                        "task_id": tid,
                        "title": title,
                        "amount_usd": amount_usd,
                        "meta": {"source": url}
                    })
                except Exception:
                    continue

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
                    bid = b.get("id") or b.get("productId") or str(int(time.time() * 1000))
                    name = b.get("name") or b.get("title") or ""
                    est_usd = float(b.get("estimatedRewardUsd") or b.get("est_usd") or 0)
                    out.append({
                        "id": f"bitget_earn_{bid}",
                        "type": "airdrop",
                        "name": name,
                        "est_usd": est_usd,
                        "claimable": True,
                        "meta": {"source": url}
                    })
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
    for r in scan_generic_aggregators():
        if r.get("type") == "task" or r.get("platform") == "TaskOn":
            results.append(r)
    filtered = []
    for r in results:
        amt = r.get("amount_usd") or r.get("est_usd") or 0
        if amt == 0 or amt >= MIN_TASK_USD:
            filtered.append(r)
    return filtered

# -------------------------
# Bitget API integration
# -------------------------
def _bitget_sign(timestamp, method, request_path, body, secret):
    what = str(timestamp) + method.upper() + request_path + (body or "")
    h = hmac.new(secret.encode(), what.encode(), hashlib.sha256)
    return h.hexdigest()

def bitget_request_rw(method, path, params_or_body=None):
    if not BITGET_API_KEY_RW:
        raise Exception("Bitget RW keys not configured")
    ts = str(int(time.time() * 1000))
    body = json.dumps(params_or_body) if (params_or_body and method.upper() in ("POST", "PUT")) else ""
    sign = _bitget_sign(ts, method, path, body, BITGET_API_SECRET_RW or "")
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_RW,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": BITGET_API_PASSPHRASE_RW or "",
        "Content-Type": "application/json"
    }
    url = API_HOST_BITGET + path
    r = requests.request(method, url, headers=headers, data=body if body else None,
                         params=(params_or_body if method.upper() == "GET" else None), timeout=20)
    r.raise_for_status()
    return r.json()

# -------------------------
# Core processing
# -------------------------
def process_opportunities():
    try:
        print("[scanner] Starting scan...")
        found = []
        found += scan_airdrops()
        found += scan_high_value_tasks()

        if not found:
            print("[scanner] No opportunities found")
            return

        for op in found:
            aid = f"{op.get('type')}_{int(time.time() * 1000)}"
            persist_action(aid, op.get("type"), op)
            print(f"[scanner] queued {aid} -> {op}")
            send_approval_card_sync(aid, op)

        send_text(TELEGRAM_CHAT_ID, f"✅ Scanner discovered {len(found)} opportunities. Use /tasks or /airdrops to inspect.")
    except Exception as e:
        print("[process_opportunities] error:", e)
        traceback.print_exc()
        send_text(TELEGRAM_CHAT_ID, f"❌ Scanner error: {e}")

# -------------------------
# Telegram command handlers
# -------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = "👋 Welcome to <b>Quantum Income Engine</b>\nAdmin-only control. Use /help for commands."
    await update.message.reply_text(txt, parse_mode="HTML")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "/help - show this\n/status - system status (admin)\n/scan - run live scan (admin)\n"
        "/tasks - list recent tasks (admin)\n/airdrops - list recent airdrops (admin)\n"
        "/stop - pause scanning (admin)\n/resume - resume scanning (admin)"
    )
    await update.message.reply_text(txt)

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_ADMIN_ID:
        await update.message.reply_text("⛔ Not authorized.")
        return
    pending = list_pending(200)
    next_run = None
    try:
        jobs = scheduler.get_jobs()
        if jobs:
            next_run = jobs[0].next_run_time
    except Exception:
        pass
    txt = f"🧾 System Status\nPending: {len(pending)}\nNext scan: {next_run}\nUptime: {datetime.utcnow().isoformat()} UTC"
    await update.message.reply_text(txt)

async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_ADMIN_ID:
        await update.message.reply_text("⛔ Not authorized.")
        return
    await update.message.reply_text("🔍 Running live scan now...")
    threading.Thread(target=process_opportunities, daemon=True).start()

async def tasks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_ADMIN_ID:
        await update.message.reply_text("⛔ Not authorized.")
        return
    pending = list_pending(100)
    tasks = [p for p in pending if p.get("type") == "task"]
    if not tasks:
        await update.message.reply_text("❌ No tasks queued.")
        return
    msg = "🧾 Pending Tasks:\n"
    for t in tasks[:10]:
        pl = t["payload"]
        amt = pl.get("amount_usd") or pl.get("est_usd") or 0
        title = pl.get("title") or pl.get("name") or pl.get("platform") or ""
        msg += f"• {title[:60]} — ${amt} (id: {t['id']})\n"
    await update.message.reply_text(msg)

async def airdrops_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_ADMIN_ID:
        await update.message.reply_text("⛔ Not authorized.")
        return
    pending = list_pending(100)
    ads = [p for p in pending if p.get("type") == "airdrop"]
    if not ads:
        await update.message.reply_text("❌ No airdrops queued.")
        return
    msg = "🎁 Pending Airdrops:\n"
    for a in ads[:10]:
        pl = a["payload"]
        token = pl.get("token_symbol") or "?"
        amt = pl.get("token_amount") or pl.get("est_usd") or 0
        msg += f"• {pl.get('name','(no name)')[:50]} — {amt} {token} (id: {a['id']})\n"
    await update.message.reply_text(msg)

async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_ADMIN_ID:
        await update.message.reply_text("⛔ Not authorized.")
        return
    scheduler.pause()
    await update.message.reply_text("⏹️ Auto-scan paused.")

async def resume_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_ADMIN_ID:
        await update.message.reply_text("⛔ Not authorized.")
        return
    scheduler.resume()
    await update.message.reply_text("▶️ Auto-scan resumed.")

# -------------------------
# CallbackQuery handler
# -------------------------
async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    from_user = update.effective_user
    user_id = from_user.id
    data = query.data or ""

    if "::" in data:
        cmd, aid = data.split("::", 1)
    else:
        await query.edit_message_text("Invalid callback.")
        return

    if user_id != TELEGRAM_ADMIN_ID:
        await query.edit_message_text("⛔ Not authorized.")
        send_text(TELEGRAM_CHAT_ID, f"❌ Unauthorized attempt by @{getattr(from_user, 'username', 'unknown')} (id:{user_id}) on {cmd} {aid}")
        return

    pending = list_pending(500)
    target = next((p for p in pending if p["id"] == aid), None)
    if not target:
        send_text(TELEGRAM_CHAT_ID, f"✖ Action {aid} not found or already handled.")
        return

    payload = target["payload"]
    try:
        if cmd == "APPROVE":
            set_action_status(aid, "COMPLETED")
            await query.edit_message_text(f"✅ Approved: {aid}")
            send_text(TELEGRAM_CHAT_ID, f"✅ Action {aid} approved and marked complete.")
            return

        if cmd == "REJECT":
            set_action_status(aid, "REJECTED")
            await query.edit_message_text(f"❌ Rejected: {aid}")
            send_text(TELEGRAM_CHAT_ID, f"❌ Action {aid} rejected.")
            return

        if cmd == "SELL":
            token = payload.get("token_symbol") or payload.get("symbol")
            token_amount = payload.get("token_amount") or payload.get("amount") or payload.get("amount_usd")

            if not token:
                await query.edit_message_text("❌ Sell failed: token info missing.")
                send_text(TELEGRAM_CHAT_ID, "❌ Sell failed: no token symbol found.")
                return

            market_pair = f"{token}USDT"
            await query.edit_message_text(f"💱 Processing sell for {token_amount} {token}...")

            if BITGET_API_KEY_RW:
                try:
                    resp = bitget_request_rw("POST", "/api/spot/v1/trade/orders", {
                        "symbol": market_pair,
                        "size": str(token_amount),
                        "side": "sell",
                        "orderType": "market"
                    })
                    send_text(TELEGRAM_CHAT_ID, f"✅ Sell order placed:\n<pre>{json.dumps(resp, indent=2)}</pre>")
                except Exception as e:
                    send_text(TELEGRAM_CHAT_ID, f"❌ Sell error: {e}")
            else:
                send_text(TELEGRAM_CHAT_ID, f"⚠️ BITGET RW keys not set — sell simulated for {token_amount} {token}")

            set_action_status(aid, "COMPLETED")
            return

    except Exception as e:
        send_text(TELEGRAM_CHAT_ID, f"❌ Execution error for {aid}: {e}")
        traceback.print_exc()
        set_action_status(aid, "FAILED")
        return

# -------------------------
# Scheduler and startup
# -------------------------
scheduler = BackgroundScheduler()
scheduler.add_job(
    process_opportunities,
    "interval",
    minutes=SCAN_INTERVAL_MINUTES,
    id="scanner_job",
    next_run_time=datetime.utcnow()
)

def run_flask():
    flask_app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

def start_worker_and_bot():
    global APPLICATION
    print("[bot] Initializing Telegram bot...")
    APPLICATION = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # handlers
    APPLICATION.add_handler(CommandHandler("start", start_cmd))
    APPLICATION.add_handler(CommandHandler("help", help_cmd))
    APPLICATION.add_handler(CommandHandler("status", status_cmd))
    APPLICATION.add_handler(CommandHandler("scan", scan_cmd))
    APPLICATION.add_handler(CommandHandler("tasks", tasks_cmd))
    APPLICATION.add_handler(CommandHandler("airdrops", airdrops_cmd))
    APPLICATION.add_handler(CommandHandler("stop", stop_cmd))
    APPLICATION.add_handler(CommandHandler("resume", resume_cmd))
    APPLICATION.add_handler(CallbackQueryHandler(callback_query_handler))

    # scheduler
    print("[scheduler] Starting background scheduler...")
    scheduler.start()

    # startup notification
    send_text(TELEGRAM_CHAT_ID, f"✅ Quantum Income Engine started at {datetime.utcnow().isoformat()} UTC")

    # start polling - disable internal signal registration to avoid set_wakeup_fd errors
    print("[bot] Starting polling (signal handling disabled)...")
    APPLICATION.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=None)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 QUANTUM INCOME ENGINE - Starting...")
    print("=" * 60)
    print(f"Port: {PORT}")
    print(f"Telegram Chat ID: {TELEGRAM_CHAT_ID}")
    print(f"Admin ID: {TELEGRAM_ADMIN_ID}")
    print(f"Scan Interval: {SCAN_INTERVAL_MINUTES} minutes")
    print(f"Min Task USD: ${MIN_TASK_USD}")
    print("=" * 60)

    print("[flask] Starting web server thread...")
    threading.Thread(target=run_flask, daemon=True).start()
    time.sleep(2)

    try:
        start_worker_and_bot()
    except KeyboardInterrupt:
        print("\n[shutdown] Received interrupt signal")
        try:
            scheduler.shutdown()
        except Exception:
            pass
        print("[shutdown] Graceful shutdown complete")
    except Exception as e:
        print(f"[fatal] Fatal error starting bot: {e}")
        traceback.print_exc()
        try:
            if TELEGRAM_CHAT_ID and TELEGRAM_TOKEN:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    json={"chat_id": TELEGRAM_CHAT_ID, "text": f"❌ Startup failed: {e}"}
                )
        except Exception:
            pass
