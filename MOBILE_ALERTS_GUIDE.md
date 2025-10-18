# 📱 Mobile & Alerts Features Guide

## 🔔 WhatsApp & Telegram Alerts

### Telegram Setup

1. **Create a Telegram Bot:**
   - Open Telegram and search for `@BotFather`
   - Send `/newbot` command
   - Follow instructions to create your bot
   - Copy the **Bot Token** provided (format: `1234567890:ABCdefGHI...`)

2. **Get Your Chat ID:**
   - Start a chat with your new bot
   - Send any message to the bot
   - Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Look for `"chat":{"id": YOUR_CHAT_ID}`
   - Copy the Chat ID number

3. **Configure in `.env`:**
   ```env
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```

4. **Enable in App:**
   - Go to sidebar → Alert Settings
   - Enable "📱 Enable Telegram Alerts"
   - Click "🧪 Test Alert" to verify

### WhatsApp Setup

1. **Enable in App:**
   - Go to sidebar → Alert Settings
   - Enable "💬 Enable WhatsApp Alerts"

2. **How It Works:**
   - WhatsApp alerts use the share URL scheme
   - Clicking alert opens WhatsApp with pre-filled message
   - Works on mobile and desktop
   - No API key needed!

## 📱 Mobile-Responsive Dashboard

### Enable Mobile Mode

1. **Toggle in Sidebar:**
   - Check "📱 Mobile Mode" in sidebar
   - UI automatically optimizes for touch

2. **Features:**
   - ✅ Larger touch targets (48px minimum)
   - ✅ Compact metrics and tables
   - ✅ Quick Action buttons
   - ✅ Swipeable charts
   - ✅ Full-width cards
   - ✅ Stacked columns on small screens

### Quick Actions

Three quick action buttons for mobile users:
- **📊 Scan Now** - Jump directly to scanner
- **📈 Analyze** - Quick analysis view
- **🔔 Alerts** - View alert settings

## 🎯 Using Alerts

### From Analysis Tab

After analyzing a stock, you'll see three alert options:

1. **📱 Telegram Alert**
   - Sends formatted trade setup to Telegram
   - Includes entry, target, stop loss
   - Risk:Reward ratio included

2. **💬 WhatsApp Alert**
   - Opens WhatsApp with pre-filled message
   - Easy to forward to groups
   - Works on all devices

3. **📋 Copy Alert**
   - Displays formatted text
   - Copy and share anywhere
   - Good for other messaging apps

### Alert Format

```
🟢 BUY SIGNAL - SBIN-EQ

📊 Score: 75/100
📈 Signal: Strong Buy
💹 Regime: Strong Uptrend

💰 Entry: ₹725.50
🎯 Target: ₹780.25 (+7.55%)
🛑 Stop Loss: ₹705.30 (-2.78%)

⚖️ Risk:Reward = 1:2.71

📝 Reason: Above 200 EMA, MACD bullish, Market: Bullish

⏰ 18 Oct 2025, 02:30 PM
```

## ⚙️ Configuration

### Alert Cooldown

- Default: 5 minutes between same stock alerts
- Prevents alert spam
- Configurable in `ALERT_CONFIG`

### Mobile Settings

Configure in `MOBILE_CONFIG`:
```python
MOBILE_CONFIG = {
    "compact_mode": False,  # Auto-detect
    "show_quick_actions": True,
    "swipeable_charts": True,
    "touch_friendly_buttons": True,
}
```

## 🎨 Mobile UI Features

### Responsive Breakpoints

- **Desktop**: > 768px - Full layout
- **Mobile**: ≤ 768px - Optimized layout

### Touch-Friendly Elements

- Buttons: Minimum 48px height
- Tabs: Larger padding (12px)
- Charts: Pan and zoom enabled
- Tables: Scrollable horizontally

### Dark Theme Optimized

All alerts and mobile UI work perfectly with dark theme!

## 🚀 Tips & Best Practices

1. **Test Your Alerts:**
   - Always use "🧪 Test Alert" button first
   - Verify both Telegram and WhatsApp work

2. **Mobile Browsing:**
   - Enable Mobile Mode for better experience
   - Use Quick Actions for faster navigation
   - Swipe charts for different views

3. **Alert Management:**
   - Check alert history in session state
   - Cooldown prevents spam (5 min default)
   - Format is consistent across platforms

4. **Security:**
   - Keep `.env` file secure
   - Never share bot tokens publicly
   - Use `.env.example` as template only

## 🐛 Troubleshooting

### Telegram Alerts Not Working

1. Check bot token is correct
2. Verify chat ID is a number
3. Ensure bot is not blocked
4. Test with `/start` command to bot

### WhatsApp Not Opening

1. Ensure WhatsApp is installed
2. Check browser allows `wa.me` links
3. Try from mobile device

### Mobile Mode Issues

1. Clear browser cache
2. Toggle mobile mode off/on
3. Refresh page
4. Check screen width < 768px

## 📞 Support

For issues or questions:
- Check logs in console
- Verify credentials in `.env`
- Test with simple alerts first
- Report bugs on GitHub

---

**Enjoy your mobile-first trading experience! 📱📊**
