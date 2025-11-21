#!/usr/bin/env python3
"""
Legendary Empire — Single-file application
- Features: Bot (AI error engine + claim scaffold), Dashboard, Config Tester, Debug Imports
- Import-safe: no network calls at module import time
- Dry-run by default; set ENABLE_LIVE_TRANSACTIONS=true to allow live actions (use cautiously)
"""

import argparse
import asyncio
import base58
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

load_dotenv()


def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y")


def require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def run_debug_imports(modules: List[str]) -> int:
    failed = False
    for m in modules:
        try:
            __import__(m)
            print(f"IMPORT OK: {m}")
        except Exception:
            print(f"IMPORT FAILED: {m}")
            traceback.print_exc()
            failed = True
    return 2 if failed else 0


class AIErrorEngine:
    def __init__(self):
        self.error_patterns: Dict[str, Dict] = {}
        self.success_patterns: Dict[str, Dict] = {}
        self.token_blacklist: Set[str] = set()
        self.confidence: Dict[str, float] = {}

    def should_attempt_claim(self, airdrop: Dict) -> Dict:
        token_key = airdrop.get("mint") or airdrop.get("token") or "unknown"
        if token_key in self.token_blacklist:
            return {"should_attempt": False, "reason": "Token blacklisted by AI"}
        key = f"{airdrop.get('protocol','unknown')}_{token_key}"
        pat = self.error_patterns.get(key)
        if pat and pat.get("count", 0) > 3 and time.time() - pat.get("last_attempt", 0) < 3600:
            return {"should_attempt": False, "reason": "Recent repeated failures"}
        if self.confidence.get(key, 100) < 20:
            return {"should_attempt": False, "reason": "Low AI confidence (<20%)"}
        return {"should_attempt": True}

    def analyze_error(self, error: Exception, airdrop: Dict) -> Dict:
        msg = str(error).lower()
        key = f"{airdrop.get('protocol','unknown')}_{airdrop.get('token','unknown')}"
        pat = self.error_patterns.setdefault(key, {"count": 0, "types": [], "last_attempt": time.time()})
        pat["count"] += 1
        pat["last_attempt"] = time.time()

        if "insufficient" in msg or "balance" in msg:
            pat["types"].append("insufficient_balance")
            return {"should_retry": False, "should_skip": True, "reason": "Insufficient balance"}
        if "timeout" in msg or "network" in msg:
            pat["types"].append("network_issue")
            return {"should_retry": True, "should_skip": False, "strategy": "switch_rpc", "delay": 2.0}
        if "signature" in msg:
            pat["types"].append("signature_error")
            return {"should_retry": True, "should_skip": False, "strategy": "rebuild_tx", "delay": 3.0}

        pat["types"].append("unknown")
        if pat["count"] > 7:
            return {"should_retry": False, "should_skip": True, "reason": "Unknown repeated failures"}
        delay = min(1.0 * (2 ** pat["count"]), 60.0)
        return {"should_retry": True, "should_skip": False, "strategy": "backoff", "delay": delay}

    def learn_success(self, airdrop: Dict, claim_time: float):
        key = f"{airdrop.get('protocol','unknown')}_{airdrop.get('token','unknown')}"
        p = self.success_patterns.setdefault(key, {"count": 0, "avg_time": 0.0, "last_success": time.time()})
        p["count"] += 1
        p["avg_time"] = (p["avg_time"] * (p["count"] - 1) + claim_time) / p["count"]
        p["last_success"] = time.time()
        self.confidence[key] = min(100, 50 + p["count"] * 10)
        self.error_patterns.pop(key, None)


class LegendaryAirdropEmpire:
    def __init__(self):
        self.rpc_endpoints = self._load_rpc_endpoints()
        self.current_rpc_index = 0
        self._clients = None
        self.wallets = []
        self.phantom_address_str: Optional[str] = None
        self.telegram_token: Optional[str] = None
        self.telegram_chat_id: Optional[str] = None
        self.admin_ids: Set[int] = set()
        self.ai_engine = AIErrorEngine()
        self.stats = defaultdict(int)
        self.config = {
            "scan_interval": int(os.getenv("SCAN_INTERVAL", "2000")) / 1000.0,
            "max_concurrent_claims": int(os.getenv("MAX_CONCURRENT_CLAIMS", "10")),
            "ai_learning_enabled": env_bool("AI_LEARNING_ENABLED", True),
            "auto_withdraw": env_bool("AUTO_WITHDRAW", False),
            "withdraw_threshold": float(os.getenv("WITHDRAW_THRESHOLD", "0.1")),
        }
        self.is_paused = False
        self.claim_semaphore: Optional[asyncio.Semaphore] = None

    def _load_rpc_endpoints(self) -> List[str]:
        endpoints = []
        primary = os.getenv("SOLANA_RPC_URL")
        if primary:
            endpoints.append(primary)
        for i in range(1, 6):
            e = os.getenv(f"SOLANA_RPC_URL_BACKUP_{i}")
            if e:
                endpoints.append(e)
        if not endpoints:
            endpoints = ["https://api.mainnet-beta.solana.com"]
        return endpoints

    def _load_admin_ids(self) -> Set[int]:
        raw = os.getenv("ADMIN_IDS", "")
        ids = set()
        for part in [p.strip() for p in raw.split(",") if p.strip()]:
            try:
                ids.add(int(part))
            except Exception:
                continue
        return ids

    def _validate_startup_env(self):
        self.phantom_address_str = require_env("PHANTOM_WALLET_ADDRESS")
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.admin_ids = self._load_admin_ids()

    async def _create_clients(self):
        if self._clients:
            return
        from solana.rpc.async_api import AsyncClient
        self._clients = [AsyncClient(url, timeout=30) for url in self.rpc_endpoints]

    async def get_client(self):
        await self._create_clients()
        client = self._clients[self.current_rpc_index]
        self.current_rpc_index = (self.current_rpc_index + 1) % len(self._clients)
        return client

    def _load_wallets(self):
        wallets = []
        primary_env = os.getenv("WALLET_PRIVATE_KEY")
        if not primary_env:
            raise RuntimeError("WALLET_PRIVATE_KEY is required")
        try:
            raw = base58.b58decode(primary_env)
            from solders.keypair import Keypair
            wallets.append(Keypair.from_bytes(raw))
        except Exception as e:
            raise RuntimeError(f"Failed to decode primary wallet: {e}")
        for i in range(2, 6):
            k = os.getenv(f"WALLET_PRIVATE_KEY_{i}")
            if k and k.lower() != "optional":
                try:
                    raw = base58.b58decode(k)
                    from solders.keypair import Keypair
                    wallets.append(Keypair.from_bytes(raw))
                except Exception:
                    continue
        return wallets

    def _live_transactions_enabled(self) -> bool:
        return env_bool("ENABLE_LIVE_TRANSACTIONS", False)

    async def send_telegram_message(self, text: str):
        if not self.telegram_token or not self.telegram_chat_id:
            return
        try:
            from telegram import Bot
            bot = Bot(token=self.telegram_token)
            coro = bot.send_message(chat_id=self.telegram_chat_id, text=text, parse_mode="HTML")
            if asyncio.iscoroutine(coro):
                await coro
        except Exception:
            return

    async def claim_airdrop(self, airdrop: Dict) -> Dict:
        if self.claim_semaphore is None:
            self.claim_semaphore = asyncio.Semaphore(self.config["max_concurrent_claims"])
        async with self.claim_semaphore:
            pre = self.ai_engine.should_attempt_claim(airdrop)
            if not pre.get("should_attempt", True):
                return {"success": False, "skipped": True, "reason": pre.get("reason")}
            start = time.time()
            try:
                if not self._live_transactions_enabled():
                    await self.send_telegram_message(
                        f"🧪 Dry-run: would claim {airdrop.get('token')} on {airdrop.get('protocol')}"
                    )
                    return {"success": False, "dry_run": True}
                return {"success": True, "signature": "SIMULATED_SIGNATURE"}
            except Exception as e:
                resolution = self.ai_engine.analyze_error(e, airdrop)
                if resolution.get("should_retry"):
                    await asyncio.sleep(resolution.get("delay", 1.0))
                    try:
                        if not self._live_transactions_enabled():
                            return {"success": False, "dry_run": True}
                        return {"success": True, "signature": "SIMULATED_RECOVERED"}
                    except Exception as e2:
                        self.ai_engine.token_blacklist.add(airdrop.get("mint") or airdrop.get("token") or "unknown")
                        return {"success": False, "error": str(e2)}
                if resolution.get("should_skip"):
                    self.ai_engine.token_blacklist.add(airdrop.get("mint") or airdrop.get("token") or "unknown")
                    await self.send_telegram_message(
                        f"🤖 AI skipped token {airdrop.get('token')}: {resolution.get('reason')}"
                    )
                return {"success": False, "error": str(e)}
            finally:
                duration = time.time() - start
                if duration > 0 and self.config["ai_learning_enabled"]:
                    self.ai_engine.learn_success(airdrop, duration)

    async def auto_withdraw(self):
        if not self.config["auto_withdraw"] or not self._live_transactions_enabled():
            return

    async def start(self):
        self._validate_startup_env()
        self.wallets = self._load_wallets()
        from solders.pubkey import Pubkey
        _ = Pubkey.from_string(self.phantom_address_str)

        print("Legendary Empire starting (dry-run by default)")
        try:
            cycle = 0
            while True:
                if self.is_paused:
                    await asyncio.sleep(1)
                    continue
                cycle += 1
                print(f"[{datetime.now().isoformat()}] Mega scan #{cycle}")
                airdrops: List[Dict] = []
                if airdrops:
                    tasks = [self.claim_airdrop(a) for a in airdrops[: self.config["max_concurrent_claims"]]]
                    await asyncio.gather(*tasks, return_exceptions=True)
                await self.auto_withdraw()
                await asyncio.sleep(self.config["scan_interval"])
        except asyncio.CancelledError:
            pass
        finally:
            if self._clients:
                for c in self._clients:
                    try:
                        await c.close()
                    except Exception:
                        pass


class LegendaryDashboard:
    def __init__(self):
        self.rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com").strip()
        self.phantom_address_str = os.getenv("PHANTOM_WALLET_ADDRESS", "").strip()
        self.client = None
        self.wallets = []
        self.stats = {"bot_balance": 0.0, "phantom_balance": 0.0, "uptime": time.time()}
        self.refresh_interval = float(os.getenv("DASHBOARD_REFRESH_INTERVAL", "3"))

    def _load_wallets(self):
        wallets = []
        primary_env = os.getenv("WALLET_PRIVATE_KEY", "").strip()
        if primary_env:
            try:
                raw = base58.b58decode(primary_env)
                from solders.keypair import Keypair
                wallets.append({"keypair": Keypair.from_bytes(raw), "name": "Primary", "index": 1})
            except Exception:
                pass
        for i in range(2, 6):
            k = os.getenv(f"WALLET_PRIVATE_KEY_{i}", "").strip()
            if k and k.lower() != "optional":
                try:
                    raw = base58.b58decode(k)
                    from solders.keypair import Keypair
                    wallets.append({"keypair": Keypair.from_bytes(raw), "name": f"Wallet {i}", "index": i})
                except Exception:
                    continue
        return wallets

    async def _ensure_client(self):
        if not self.client:
            from solana.rpc.async_api import AsyncClient
            self.client = AsyncClient(self.rpc_url, timeout=30)

    async def update_balances(self):
        await self._ensure_client()
        total = 0.0
        self.wallets = self._load_wallets()
        for wallet in self.wallets:
            try:
                resp = await self.client.get_balance(wallet["keypair"].pubkey())
                wallet["balance"] = float(resp.value) / 1e9
                total += wallet["balance"]
            except Exception:
                wallet["balance"] = 0.0
        self.stats["bot_balance"] = total
        try:
            if self.phantom_address_str:
                from solders.pubkey import Pubkey
                phantom = Pubkey.from_string(self.phantom_address_str)
                resp = await self.client.get_balance(phantom)
                self.stats["phantom_balance"] = float(resp.value) / 1e9
            else:
                self.stats["phantom_balance"] = 0.0
        except Exception:
            self.stats["phantom_balance"] = 0.0

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def format_uptime(self):
        uptime = time.time() - self.stats["uptime"]
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        return f"{minutes}m {seconds}s"

    def center_text(self, text, width):
        text = str(text)
        if len(text) >= width:
            return text
        padding = (width - len(text)) // 2
        return " " * padding + text + " " * (width - padding - len(text))

    async def display(self):
        await self.update_balances()
        self.clear_screen()
        width = 80
        line = "═" * width
        fire = "🔥" * max(1, width // 8)
        print("\n" + fire)
        print(line)
        print("║" + self.center_text("🚀 LEGENDARY EMPIRE DASHBOARD 🚀", width - 2) + "║")
        print(line)
        print(fire + "\n")
        print("╔" + "═" * (width - 2) + "╗")
        print("║" + self.center_text("💎 EMPIRE OVERVIEW 💎", width - 2) + "║")
        print("╠" + "═" * (width - 2) + "╣")
        print(f"║  👛 Active Wallets: {len(self.wallets):<{width - 24}}║")
        print(f"║  💰 Total Bot Balance: {self.stats['bot_balance']:.6f} SOL{' ' * (width - 47)}║")
        print(f"║  👻 Phantom Balance: {self.stats['phantom_balance']:.6f} SOL{' ' * (width - 46)}║")
        combined = self.stats["bot_balance"] + self.stats["phantom_balance"]
        print(f"║  💵 Combined Holdings: {combined:.6f} SOL{' ' * (width - 47)}║")
        print(f"║  ⏱  Uptime: {self.format_uptime():<{width - 16}}║")
        print("╚" + "═" * (width - 2) + "╝\n")
        print("╔" + "═" * (width - 2) + "╗")
        print("║" + self.center_text("👛 WALLET BREAKDOWN 👛", width - 2) + "║")
        print("╠" + "═" * (width - 2) + "╣")
        for wallet in self.wallets:
            addr = str(wallet["keypair"].pubkey())
            address = (addr[:12] + "...") if len(addr) > 15 else addr
            balance = f"{wallet.get('balance', 0.0):.6f} SOL"
            status = "🟢" if wallet.get("balance", 0) > 0.01 else "🟡"
            name = wallet["name"].ljust(12)[:12]
            address_str = address.ljust(18)[:18]
            balance_str = balance.ljust(20)[:20]
            print(f"║  {status} {name} {address_str} {balance_str}║")
        print("╚" + "═" * (width - 2) + "╝\n")
        print("╔" + "═" * (width - 2) + "╗")
        print("║" + self.center_text("⚡ PERFORMANCE METRICS ⚡", width - 2) + "║")
        print("╠" + "═" * (width - 2) + "╣")
        print(f"║  📊 Status: {'🟢 LEGENDARY MODE ACTIVE':<{width - 17}}║")
        print(f"║  ⏱  Uptime: {self.format_uptime():<{width - 17}}║")
        print(f"║  💰 Total Value: {combined:.6f} SOL{' ' * (width - 39)}║")
        print("╚" + "═" * (width - 2) + "╝\n")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"  Last Update: {now}")
        print("  Press Ctrl+C to exit | Empire is DOMINATING! 🔥\n")

    async def start(self):
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def _signal_handler():
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                pass

        try:
            while not stop_event.is_set():
                try:
                    await self.display()
                    await asyncio.wait_for(stop_event.wait(), timeout=self.refresh_interval)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    await asyncio.sleep(1.0)
        finally:
            if self.client:
                try:
                    await self.client.close()
                except Exception:
                    pass


class ConfigTester:
    def __init__(self):
        self.results = {
            "rpc": {"status": "⏳", "message": "Testing..."},
            "wallet": {"status": "⏳", "message": "Testing..."},
            "phantom": {"status": "⏳", "message": "Testing..."},
            "telegram": {"status": "⏳", "message": "Testing..."},
            "balance": {"status": "⏳", "message": "Testing..."},
        }

    async def test_rpc_connection(self) -> bool:
        client = None
        try:
            from solana.rpc.async_api import AsyncClient
            client = AsyncClient(os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"), timeout=10)
            version = await client.get_version()
            core = None
            try:
                core = version.value.get("solana-core") if isinstance(version.value, dict) else None
            except Exception:
                core = None
            self.results["rpc"] = {"status": "✅", "message": f"Connected: {core or str(version.value)}"}
            return True
        except Exception as e:
            self.results["rpc"] = {"status": "❌", "message": f"Failed: {e}"}
            return False
        finally:
            if client:
                try:
                    await client.close()
                except Exception:
                    pass

    async def test_wallet_key(self):
        key = os.getenv("WALLET_PRIVATE_KEY")
        if not key or key == "your_base58_private_key_here":
            self.results["wallet"] = {"status": "❌", "message": "WALLET_PRIVATE_KEY not configured"}
            return None
        try:
            raw = base58.b58decode(key)
            from solders.keypair import Keypair
            kp = Keypair.from_bytes(raw)
            self.results["wallet"] = {"status": "✅", "message": f"Address: {str(kp.pubkey())[:12]}..."}
            return kp
        except Exception as e:
            self.results["wallet"] = {"status": "❌", "message": f"Invalid key: {e}"}
            return None

    async def test_phantom_address(self) -> bool:
        phantom = os.getenv("PHANTOM_WALLET_ADDRESS")
        if not phantom:
            self.results["phantom"] = {"status": "❌", "message": "PHANTOM_WALLET_ADDRESS not configured"}
            return False
        try:
            from solders.pubkey import Pubkey
            Pubkey.from_string(phantom)
            self.results["phantom"] = {"status": "✅", "message": "Valid Phantom address"}
            return True
        except Exception as e:
            self.results["phantom"] = {"status": "❌", "message": f"Invalid Phantom address: {e}"}
            return False

    async def test_telegram_bot(self) -> bool:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            self.results["telegram"] = {"status": "❌", "message": "Telegram token/chat not configured"}
            return False
        try:
            from telegram import Bot
            bot = Bot(token=token)
            coro = bot.send_message(chat_id=chat_id, text="✅ Configuration test message", parse_mode="HTML")
            if asyncio.iscoroutine(coro):
                await coro
            self.results["telegram"] = {"status": "✅", "message": "Telegram test message sent"}
            return True
        except Exception as e:
            self.results["telegram"] = {"status": "❌", "message": f"Telegram error: {e}"}
            return False

    async def test_wallet_balance(self, wallet) -> bool:
        if not wallet:
            self.results["balance"] = {"status": "❌", "message": "No wallet for balance check"}
            return False
        client = None
        try:
            from solana.rpc.async_api import AsyncClient
            client = AsyncClient(os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"), timeout=10)
            resp = await client.get_balance(wallet.pubkey())
            bal = float(resp.value) / 1e9
            if bal < 0.01:
                self.results["balance"] = {"status": "⚠️", "message": f"Low balance: {bal:.6f} SOL"}
            else:
                self.results["balance"] = {"status": "✅", "message": f"Balance: {bal:.6f} SOL"}
            return True
        except Exception as e:
            self.results["balance"] = {"status": "❌", "message": f"Balance check failed: {e}"}
            return False
        finally:
            if client:
                try:
                    await client.close()
                except Exception:
                    pass

    def display_results(self) -> int:
        for k, v in self.results.items():
            print(f"{k.ljust(12)} {v['status']}  {v['message']}")
        statuses = [r["status"] for r in self.results.values()]
        if all(s == "✅" for s in statuses):
            return 0
        if any(s == "❌" for s in statuses):
            return 1
        return 0

    async def run_all_tests(self) -> int:
        await self.test_rpc_connection()
        wallet = await self.test_wallet_key()
        await self.test_phantom_address()
        await self.test_telegram_bot()
        await self.test_wallet_balance(wallet)
        return self.display_results()


async def run_bot():
    empire = LegendaryAirdropEmpire()
    await empire.start()


async def run_dashboard():
    dash = LegendaryDashboard()
    await dash.start()


async def run_tests():
    tester = ConfigTester()
    return await tester.run_all_tests()


def main():
    parser = argparse.ArgumentParser(prog="legendary_empire_app")
    parser.add_argument(
        "command",
        choices=["run-bot", "run-dashboard", "test-config", "run-both", "debug-imports"],
        help="command to run",
    )
    args = parser.parse_args()

    if args.command == "debug-imports":
        modname = os.path.splitext(os.path.basename(__file__))[0]
        sys.exit(run_debug_imports([modname]))

    loop = asyncio.get_event_loop()
    if args.command == "run-bot":
        try:
            loop.run_until_complete(run_bot())
        except KeyboardInterrupt:
            pass
    elif args.command == "run-dashboard":
        try:
            loop.run_until_complete(run_dashboard())
        except KeyboardInterrupt:
            pass
    elif args.command == "test-config":
        code = loop.run_until_complete(run_tests())
        sys.exit(int(code))
    elif args.command == "run-both":
        try:
            loop.run_until_complete(asyncio.gather(run_bot(), run_dashboard()))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
