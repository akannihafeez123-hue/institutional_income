#!/usr/bin/env python3
"""
Configuration Test Script
Validates all settings before starting the bot
"""

import asyncio
import os
import sys
import base58
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from telegram import Bot
from dotenv import load_dotenv

load_dotenv()


class ConfigTester:
    """Test all configurations"""
    
    def __init__(self):
        self.results = {
            'rpc': {'status': '⏳', 'message': 'Testing...'},
            'wallet': {'status': '⏳', 'message': 'Testing...'},
            'phantom': {'status': '⏳', 'message': 'Testing...'},
            'telegram': {'status': '⏳', 'message': 'Testing...'},
            'balance': {'status': '⏳', 'message': 'Testing...'}
        }
    
    async def test_rpc_connection(self):
        """Test RPC connection"""
        try:
            print('\n🔗 Testing RPC connection...')
            rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
            
            client = AsyncClient(rpc_url)
            version = await client.get_version()
            
            self.results['rpc'] = {
                'status': '✅',
                'message': f"Connected! Version: {version.value.get('solana-core', 'unknown')}"
            }
            return True
            
        except Exception as e:
            self.results['rpc'] = {
                'status': '❌',
                'message': f"Failed: {str(e)}"
            }
            return False
    
    async def test_wallet_key(self):
        """Test wallet private key"""
        try:
            print('🔑 Testing wallet private key...')
            
            wallet_key = os.getenv('WALLET_PRIVATE_KEY')
            if not wallet_key or wallet_key == 'your_base58_private_key_here':
                raise Exception('WALLET_PRIVATE_KEY not configured in .env')
            
            secret_key = base58.b58decode(wallet_key)
            wallet = Keypair.from_bytes(secret_key)
            
            self.results['wallet'] = {
                'status': '✅',
                'message': f"Valid! Address: {str(wallet.pubkey())[:12]}..."
            }
            return wallet
            
        except Exception as e:
            self.results['wallet'] = {
                'status': '❌',
                'message': f"Failed: {str(e)}"
            }
            return None
    
    async def test_phantom_address(self):
        """Test Phantom wallet address"""
        try:
            print('👻 Testing Phantom wallet address...')
            
            phantom_addr = os.getenv('PHANTOM_WALLET_ADDRESS')
            if not phantom_addr or phantom_addr == 'your_phantom_wallet_address_here':
                raise Exception('PHANTOM_WALLET_ADDRESS not configured in .env')
            
            phantom_pubkey = Pubkey.from_string(phantom_addr)
            
            self.results['phantom'] = {
                'status': '✅',
                'message': f"Valid! Address: {str(phantom_pubkey)[:12]}..."
            }
            return True
            
        except Exception as e:
            self.results['phantom'] = {
                'status': '❌',
                'message': f"Failed: {str(e)}"
            }
            return False
    
    async def test_telegram_bot(self):
        """Test Telegram bot"""
        try:
            print('📱 Testing Telegram bot...')
            
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if not bot_token or bot_token == 'your_telegram_bot_token_here':
                raise Exception('TELEGRAM_BOT_TOKEN not configured in .env')
            
            if not chat_id or chat_id == 'your_telegram_chat_id_here':
                raise Exception('TELEGRAM_CHAT_ID not configured in .env')
            
            bot = Bot(token=bot_token)
            
            await bot.send_message(
                chat_id=chat_id,
                text='✅ <b>Configuration Test</b>\n\nYour bot is configured correctly!\nYou should see this message in Telegram.',
                parse_mode='HTML'
            )
            
            self.results['telegram'] = {
                'status': '✅',
                'message': 'Connected! Test message sent to Telegram'
            }
            return True
            
        except Exception as e:
            self.results['telegram'] = {
                'status': '❌',
                'message': f"Failed: {str(e)}"
            }
            return False
    
    async def test_wallet_balance(self, wallet):
        """Test wallet balance"""
        try:
            print('💰 Checking wallet balance...')
            
            if not wallet:
                raise Exception('Wallet not available')
            
            rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
            client = AsyncClient(rpc_url)
            
            balance_resp = await client.get_balance(wallet.pubkey())
            sol_balance = balance_resp.value / 1e9
            
            if sol_balance < 0.01:
                self.results['balance'] = {
                    'status': '⚠️',
                    'message': f"Low balance: {sol_balance:.6f} SOL (need 0.01+ for fees)"
                }
            else:
                self.results['balance'] = {
                    'status': '✅',
                    'message': f"Good! Balance: {sol_balance:.6f} SOL"
                }
            
            return True
            
        except Exception as e:
            self.results['balance'] = {
                'status': '❌',
                'message': f"Failed: {str(e)}"
            }
            return False
    
    def display_results(self):
        """Display test results"""
        width = 80
        
        print('\n' + '╔' + '═' * (width - 2) + '╗')
        print('║' + ' ' * 18 + '📊 CONFIGURATION TEST RESULTS' + ' ' * 18 + '║')
        print('╚' + '═' * (width - 2) + '╝\n')
        
        print('┌' + '─' * (width - 2) + '┐')
        print('│' + '  Component'.ljust(22) + 'Status  Details'.ljust(width - 23) + '│')
        print('├' + '─' * (width - 2) + '┤')
        
        for component, result in self.results.items():
            name = component.ljust(20)
            status = result['status'].ljust(6)
            message = result['message'][:40].ljust(40)
            print(f'│  {name} {status} {message}│')
        
        print('└' + '─' * (width - 2) + '┘\n')
        
        # Check overall status
        all_passed = all(r['status'] == '✅' for r in self.results.values())
        has_warnings = any(r['status'] == '⚠️' for r in self.results.values())
        
        if all_passed:
            print('✅ ALL TESTS PASSED! Your bot is ready to start.\n')
            print('To start the bot, run:')
            print('  python main.py              (bot only)')
            print('  python monitor.py           (dashboard only)')
            print('  python run_both.py          (bot + dashboard)\n')
            return 0
        elif has_warnings and not any(r['status'] == '❌' for r in self.results.values()):
            print('⚠️  TESTS PASSED WITH WARNINGS\n')
            print('Please review the warnings above.')
            print('You can start the bot, but consider addressing warnings.\n')
            return 0
        else:
            print('❌ TESTS FAILED!\n')
            print('Please fix the errors above before starting the bot.')
            print('Review the README.md for configuration help.\n')
            return 1
    
    async def run_all_tests(self):
        """Run all tests"""
        print('╔' + '═' * 78 + '╗')
        print('║' + ' ' * 20 + '🧪 RUNNING CONFIGURATION TESTS...' + ' ' * 20 + '║')
        print('╚' + '═' * 78 + '╝')
        
        # Run tests sequentially
        await self.test_rpc_connection()
        wallet = await self.test_wallet_key()
        await self.test_phantom_address()
        await self.test_telegram_bot()
        await self.test_wallet_balance(wallet)
        
        # Display results
        return self.display_results()


async def main():
    """Main entry point"""
    tester = ConfigTester()
    exit_code = await tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
