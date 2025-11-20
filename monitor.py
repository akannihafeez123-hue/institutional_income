#!/usr/bin/env python3
"""
🔥 LEGENDARY EMPIRE DASHBOARD 🔥
Real-time monitoring for your airdrop empire
"""

import asyncio
import os
import time
from datetime import datetime, timedelta
import base58
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from dotenv import load_dotenv

load_dotenv()


class LegendaryDashboard:
    """Real-time dashboard for monitoring"""
    
    def __init__(self):
        self.client = AsyncClient(os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com'))
        self.wallets = self._load_wallets()
        self.phan tom_address = Pubkey.from_string(os.getenv('PHAN TOM_WALLET_ADDRESS'))
        
        self.stats = {
            'bot_balance': 0.0,
            'phant om_balance': 0.0,
            'total_transactions': 0,
            'uptime': time.time()
        }
        
    def _load_wallets(self):
        """Load all wallets"""
        wallets = []
        
        try:
            primary_key = base58.b58decode(os.getenv('WALLET_PRIVATE_KEY'))
            wallets.append({'keypair': Keypair.from_bytes(primary_key), 'name': 'Primary', 'index': 1})
        except Exception as e:
            print(f"Error loading primary wallet: {e}")
            return wallets
        
        for i in range(2, 11):
            key_env = os.getenv(f'WALLET_PRIVATE_KEY_{i}')
            if key_env and key_env != 'optional':
                try:
                    key = base58.b58decode(key_env)
                    keypair = Keypair.from_bytes(key)
                    wallets.append({'keypair': keypair, 'name': f'Wallet {i}', 'index': i})
                except:
                    pass
        
        return wallets
    
    async def update_balances(self):
        """Update all wallet balances"""
        try:
            total = 0.0
            
            for wallet in self.wallets:
                try:
                    balance_resp = await self.client.get_balance(wallet['keypair'].pubkey())
                    wallet['balance'] = balance_resp.value / 1e9
                    total += wallet['balance']
                except:
                    wallet['balance'] = 0.0
            
            self.stats['bot_balance'] = total
            
            try:
                phant om_resp = await self.client.get_balance(self.phant om_address)
                self.stats['phant om_balance'] = phant om_resp.value / 1e9
            except:
                self.stats['phant om_balance'] = 0.0
                
        except Exception as e:
            print(f"Balance update error: {e}")
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def format_uptime(self):
        """Format uptime duration"""
        uptime = time.time() - self.stats['uptime']
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        return f"{minutes}m {seconds}s"
    
    def center_text(self, text, width):
        """Center text in given width"""
        padding = (width - len(text)) // 2
        return ' ' * padding + text + ' ' * (width - padding - len(text))
    
    async def display(self):
        """Display dashboard"""
        await self.update_balances()
        
        self.clear_screen()
        
        width = 80
        line = '═' * width
        fire = '🔥' * (width // 2)
        
        print('\n' + fire)
        print(line)
        print('║' + self.center_text('🚀 LEGENDARY EMPIRE DASHBOARD 🚀', width - 2) + '║')
        print(line)
        print(fire + '\n')
        
        # Empire Overview
        print('╔' + '═' * (width - 2) + '╗')
        print('║' + self.center_text('💎 EMPIRE OVERVIEW 💎', width - 2) + '║')
        print('╠' + '═' * (width - 2) + '╣')
        print(f'║  👛 Active Wallets: {len(self.wallets):<{width - 24}}║')
        print(f'║  💰 Total Bot Balance: {self.stats["bot_balance"]:.6f} SOL{" " * (width - 47)}║')
        print(f'║  👻 Phant om Balance: {self.stats["phant om_balance"]:.6f} SOL{" " * (width - 45)}║')
        combined = self.stats["bot_balance"] + self.stats["phant om_balance"]
        print(f'║  💵 Combined Holdings: {combined:.6f} SOL{" " * (width - 47)}║')
        print(f'║  ⏱️  Uptime: {self.format_uptime():<{width - 16}}║')
        print('╚' + '═' * (width - 2) + '╝\n')
        
        # Individual Wallets
        print('╔' + '═' * (width - 2) + '╗')
        print('║' + self.center_text('👛 WALLET BREAKDOWN 👛', width - 2) + '║')
        print('╠' + '═' * (width - 2) + '╣')
        
        for wallet in self.wallets:
            address = str(wallet['keypair'].pubkey())[:12] + '...'
            balance = f"{wallet.get('balance', 0.0):.6f} SOL"
            status = '🟢' if wallet.get('balance', 0) > 0.01 else '🟡'
            
            name = wallet['name'].ljust(12)
            address_str = address.ljust(18)
            balance_str = balance.ljust(20)
            
            print(f'║  {status} {name} {address_str} {balance_str}║')
        
        print('╚' + '═' * (width - 2) + '╝\n')
        
        # Performance Metrics
        print('╔' + '═' * (width - 2) + '╗')
        print('║' + self.center_text('⚡ PERFORMANCE METRICS ⚡', width - 2) + '║')
        print('╠' + '═' * (width - 2) + '╣')
        print(f'║  📊 Status: {"🟢 LEGENDARY MODE ACTIVE":<{width - 17}}║')
        print(f'║  ⏱️  Uptime: {self.format_uptime():<{width - 17}}║')
        print(f'║  💰 Total Value: {combined:.6f} SOL{" " * (width - 39)}║')
        print('╚' + '═' * (width - 2) + '╝\n')
        
        # Footer
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'  Last Update: {now}')
        print('  Press Ctrl+C to exit | Empire is DOMINATING! 🔥\n')
    
    async def start(self):
        """Start the dashboard"""
        print('🔥 Starting Legendary Dashboard...\n')
        print('Loading empire data...\n')
        
        while True:
            try:
                await self.display()
                await asyncio.sleep(3)
            except KeyboardInterrupt:
                print('\n\n✨ Dashboard stopped. Empire continues! ✨\n')
                break
            except Exception as e:
                print(f'Dashboard error: {e}')
                await asyncio.sleep(5)


async def main():
    """Main entry point"""
    dashboard = LegendaryDashboard()
    await dashboard.start()


if __name__ == "__main__":
    asyncio.run(main())
