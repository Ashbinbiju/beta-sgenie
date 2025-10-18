# 📱 Telegram Alerts Setup Guide

Get instant notifications when StockGenie Pro finds high-scoring trading opportunities!

## 🚀 Quick Setup (5 minutes)

### Step 1: Create Your Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Start a chat and send `/newbot`
3. Choose a name for your bot (e.g., "My StockGenie Bot")
4. Choose a username (must end with 'bot', e.g., "mystockgenie_bot")
5. **Copy the Bot Token** (looks like: `7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps`)

### Step 2: Get Your Chat ID

**Option A: Using @userinfobot (Recommended)**
1. Search for `@userinfobot` in Telegram
2. Start a chat with it
3. It will show your User ID
4. **Copy your Chat ID** (looks like: `123456789`)

**Option B: Using Your Bot**
1. Start a chat with your newly created bot
2. Send any message to it
3. Open this URL in your browser (replace `YOUR_BOT_TOKEN`):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
4. Look for `"chat":{"id":123456789}` in the response
5. **Copy the Chat ID**

**Option C: For Group Chats**
1. Add your bot to a group
2. Send a message in the group
3. Use the URL method above
4. The Chat ID will be negative (e.g., `-1002411670969`)

### Step 3: Configure StockGenie Pro

#### For Local Deployment (.env file):

1. Open or create `.env` file in the project root
2. Add these lines:
   ```env
   TELEGRAM_ENABLED=true
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```

#### For Streamlit Cloud (Secrets):

1. Go to your app settings on Streamlit Cloud
2. Click on "Secrets" in the left menu
3. Add this configuration:
   ```toml
   TELEGRAM_ENABLED = "true"
   TELEGRAM_BOT_TOKEN = "7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps"
   TELEGRAM_CHAT_ID = "-1002411670969"
   ```

### Step 4: Test Your Setup

1. Restart StockGenie Pro
2. Check the sidebar - you should see "✅ Telegram: Connected"
3. Expand "⚙️ Alert Settings"
4. Click "📤 Send Test Alert"
5. You should receive a message in Telegram! 🎉

## 📊 Alert Types

### 1. Scan Summary Alert
Sent when a scan completes:
```
🔍 Stock Scanner Alert

⏰ Time: 14:30:00
📊 Style: Swing Trading
📈 Found: 8 opportunities

🏆 Top Picks:
🟢 SBIN-EQ - Score: 82/100
   Signal: Strong Buy | Price: ₹750.50
   Target: ₹820.00 | SL: ₹720.00
...
```

### 2. High Score Individual Alerts
Sent for stocks with score ≥ 75 (configurable):
```
🚨 HIGH SCORE ALERT 🚨

📊 TATAMOTORS-EQ
💯 Score: 85/100

📈 Signal: Strong Buy
🎯 Regime: Strong Uptrend

💰 Current Price: ₹950.25
🎯 Target: ₹1050.00
🛑 Stop Loss: ₹920.00

📝 Reason: Strong momentum with bullish market
```

## ⚙️ Alert Settings

Customize your alerts in the sidebar:

- **Scan Summary**: Get notified when any scan completes
- **High Score Alerts**: Receive alerts for individual high-scoring stocks
- **Alert Threshold**: Set minimum score (60-90) for individual alerts
- **Max Alerts per Scan**: Limit to top 5 stocks (prevents spam)

## 🔒 Security Tips

1. **Never share your bot token publicly**
2. **Use environment variables** - don't hardcode credentials
3. **For private groups**: Make sure your bot has permission to send messages
4. **Revoke compromised tokens**: Use @BotFather → `/mybots` → Select bot → API Token → Revoke

## 🛠️ Troubleshooting

### "⚠️ Telegram: Not configured"
- Check if `TELEGRAM_ENABLED=true` is set
- Verify bot token and chat ID are correct
- Restart the app after adding credentials

### "Failed to send test alert"
- Verify bot token is valid
- Check if chat ID is correct (negative for groups)
- Ensure you've started a chat with the bot
- Check your internet connection

### "403 Forbidden" Error
- You haven't started a chat with the bot yet
- For groups: make sure the bot is added to the group

### Not receiving alerts
- Check "⚙️ Alert Settings" in sidebar
- Verify alerts are enabled
- Check Telegram notification settings
- Ensure minimum score threshold is appropriate

## 📱 Using with Multiple Devices

Want alerts on multiple devices or groups?

1. **Personal + Group**: Use your personal chat ID for priority alerts
2. **Multiple bots**: Create separate bots for different purposes
3. **Channel broadcasting**: Create a Telegram channel, add bot as admin

## 🎯 Best Practices

1. **Test first**: Always send a test alert before running scans
2. **Adjust threshold**: Set alert threshold based on your strategy
   - Swing Trading: 75+ for quality picks
   - Intraday: 70+ for more opportunities
3. **Monitor spam**: If too many alerts, increase threshold
4. **Group usage**: Great for sharing with trading groups
5. **Combine with auto-scan**: Set up automatic scanning + alerts

## 📞 Need Help?

- Telegram Bot API Docs: https://core.telegram.org/bots/api
- BotFather Commands: https://core.telegram.org/bots#6-botfather
- Check logs for detailed error messages

## 🌟 Pro Tips

- **Mute during market hours** if alerts are too frequent
- **Create dedicated trading channel** for organized alerts
- **Use bot commands** to query stock info (coming soon!)
- **Set up with Live Intraday Scanner** for real-time opportunities

---

Made with ❤️ by StockGenie Pro Team
