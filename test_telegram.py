import os
import asyncio
from telegram import Bot
from telegram.ext import Application, CommandHandler

async def test_bot():
    """Test if Telegram bot can connect and respond"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN environment variable not set")
        return
    
    if not chat_id:
        print("❌ TELEGRAM_CHAT_ID environment variable not set")
        return
    
    print(f"Testing with token: {token[:10]}...")
    print(f"Chat ID: {chat_id}")
    
    try:
        # Create bot instance
        bot = Bot(token=token)
        
        # Test connection
        me = await bot.get_me()
        print(f"✅ Connected to bot: @{me.username} ({me.first_name})")
        
        # Send test message
        await bot.send_message(
            chat_id=chat_id,
            text="🤖 APEX Test Message\n\nBot is working correctly!",
            parse_mode='HTML'
        )
        print("✅ Test message sent successfully!")
        
        # Create a simple command handler
        async def start(update, context):
            await update.message.reply_text("Test bot is working!")
        
        # Start polling
        app = Application.builder().token(token).build()
        app.add_handler(CommandHandler("start", start))
        
        print("🤖 Bot is ready. Send /start to your bot in Telegram...")
        
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        
        # Keep running
        await asyncio.Event().wait()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure your bot token is correct")
        print("2. Make sure you've started the bot with @BotFather")
        print("3. Make sure your chat_id is correct (send a message to @userinfobot to get your ID)")
        print("4. Make sure your bot is not in privacy mode (disable in @BotFather if needed)")

if __name__ == "__main__":
    asyncio.run(test_bot())
