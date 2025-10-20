# 🚀 AI Insights Tab - Quick Start Guide

## 🎉 What's New?

The **🤖 AI Insights** tab is now live in StockGenie Pro! It brings machine learning predictions directly into your workflow.

---

## 📍 Location

Find it in the main navigation:
```
📈 Analysis | 🔍 Scanner | 🎯 Technical Screener | 🔄 Live Intraday | 
💰 Paper Trading | 🤖 AI Insights | 🌍 Market Dashboard
                    ↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                    NEW TAB HERE!
```

---

## 🎯 3 Sub-Tabs

### 1. 📊 AI Predictions
**View latest ML predictions from database**

Features:
- Filter by signal (BUY/HOLD/SELL)
- Set minimum confidence threshold
- See prediction accuracy
- View detailed reasoning for any stock

Best for:
- Finding AI-recommended stocks quickly
- Checking model performance
- Understanding why AI made predictions

### 2. 🎯 Single Stock Analysis  
**Run real-time AI analysis on any stock**

Features:
- Select any stock from your list
- Get instant AI prediction
- See fusion score (AI + Technical + Sentiment)
- View probability distribution (UP/DOWN/FLAT)
- Understand top contributing features
- Save prediction to database

Best for:
- Deep-dive analysis on specific stocks
- Comparing AI vs Technical analysis
- Understanding what drives predictions

### 3. ⚙️ Model Info
**Monitor AI model performance**

Features:
- View active models
- Check training metrics
- See backtest performance
- Track live accuracy
- View feature importance

Best for:
- Model management
- Performance monitoring
- Understanding model behavior

---

## 📋 Prerequisites

### ✅ Required (Already Done!)
- [x] Streamlit app updated
- [x] `ai_features.py` module
- [x] `ai_model.py` module
- [x] Database schema

### ⏳ To Enable Full Functionality

#### 1. Install Dependencies
```bash
pip install lightgbm scikit-learn
```

#### 2. Setup Supabase Tables
Run this in Supabase SQL Editor:
```sql
-- Copy & paste AI_SUPABASE_SCHEMA.sql
-- Creates: ai_features, ai_predictions, model_metadata, backtest_results
```

#### 3. Train Your First Model

**Option A: Quick Test (Mock Data)**
```python
# Create models/ directory
mkdir -p models

# Will show instructions in UI
# Can still use tab, just shows setup guide
```

**Option B: Real Training**
```python
from ai_features import FeatureEngineer
from ai_model import StockDirectionModel
import pandas as pd

# 1. Extract features for training
engineer = FeatureEngineer()
all_features = []

for symbol in ['SBIN', 'RELIANCE', 'TCS', ...]:  # 100+ stocks
    df = fetch_stock_data(symbol, period='1y')
    features = engineer.extract_features(df, symbol)
    if features:
        all_features.append(features)

features_df = pd.DataFrame(all_features)

# 2. Prepare training data
# (Need price data to calculate future returns)
# See AI_IMPLEMENTATION_GUIDE.md for full code

# 3. Train model
model = StockDirectionModel(model_version='v1.0.0')
X, y = model.prepare_training_data(features_df, price_df)
metrics = model.train(X, y)

# 4. Save model
model.save_model('models/stock_direction_v1.0.0.pkl')
```

---

## 🎮 How to Use

### Scenario 1: Find Today's AI Picks

1. Go to **🤖 AI Insights** tab
2. Select **📊 AI Predictions** sub-tab
3. Set filters:
   - Signal: **BUY**
   - Min Confidence: **0.65** (65%)
   - Show Top: **10**
4. View ranked predictions
5. Click on stock to see reasoning

### Scenario 2: Analyze a Specific Stock

1. Go to **🤖 AI Insights** tab
2. Select **🎯 Single Stock Analysis** sub-tab
3. Choose stock from dropdown (e.g., SBIN)
4. Click **🤖 Run AI Analysis**
5. View results:
   - AI Prediction (ML direction, confidence)
   - Technical Analysis (existing system)
   - Fusion Score (combined recommendation)
   - Probability distribution
   - Top features driving prediction
6. Optional: Click **💾 Save** to database

### Scenario 3: Monitor Model Performance

1. Go to **🤖 AI Insights** tab
2. Select **⚙️ Model Info** sub-tab
3. View active models
4. Expand model to see:
   - Training accuracy
   - Backtest metrics (Sharpe, Win Rate)
   - Live performance
   - Feature importance

---

## 🎯 Fusion Scoring

The AI tab combines 3 signals:

```python
Fusion Score = (50% × Technical) + (30% × AI ML) + (20% × Sentiment)
```

**Example:**
- Technical Score: 75/100 (BUY signal)
- AI ML Score: 0.80 (80% UP probability)
- Sentiment: 0.50 (neutral, placeholder)

```
Fusion = (0.5 × 0.75) + (0.3 × 0.80) + (0.2 × 0.50)
       = 0.375 + 0.24 + 0.10
       = 0.715 × 100
       = 71.5/100
```

**Result: 🟢 BUY** (score > 70)

---

## 🎨 Signal Colors

| Score | Signal | Color | Action |
|-------|--------|-------|--------|
| 70-100 | BUY | 🟢 | Strong buy opportunity |
| 40-69 | HOLD | 🟡 | Wait for better entry |
| 0-39 | SELL | 🔴 | Avoid or exit |

---

## 📊 Understanding AI Output

### ML Direction
- **UP**: Model predicts price will rise >2%
- **FLAT**: Model predicts price will move -2% to +2%
- **DOWN**: Model predicts price will fall >2%

### ML Confidence
- **0.8-1.0**: Very confident (80-100%)
- **0.65-0.8**: Confident (65-80%)
- **0.5-0.65**: Moderate (50-65%)
- **<0.5**: Low confidence (< 50%)

### ML Score
- Probability that stock will go UP
- Range: 0.0 to 1.0
- Higher = More bullish

---

## 🚨 What If...?

### ❓ Tab shows "AI Module Not Configured"

**Solution:**
```bash
pip install lightgbm scikit-learn
```

Restart Streamlit app.

### ❓ "No trained model found"

**Solution:**
Either:
1. Train a model (see AI_IMPLEMENTATION_GUIDE.md)
2. Or wait - tab still works for viewing predictions from database

### ❓ "Supabase not configured"

**Solution:**
1. Setup Supabase project
2. Add credentials to `.env` or Streamlit secrets:
   ```
   SUPABASE_URL=your_url
   SUPABASE_KEY=your_key
   ```
3. Run `AI_SUPABASE_SCHEMA.sql` to create tables

### ❓ "No AI predictions found"

**Solution:**
AI predictions need to be generated first:
1. Use **Single Stock Analysis** to manually analyze stocks
2. Click **💾 Save** to database
3. Or set up daily scheduler (Phase 2)

---

## 🎓 Tips & Best Practices

### 1. **Use Fusion Score, Not Just AI**
- AI is experimental, technical analysis is proven
- Fusion combines best of both worlds
- Never ignore technical signals completely

### 2. **Check Confidence**
- Only trade high confidence (>0.65) predictions
- Low confidence = Model is uncertain

### 3. **Understand Top Features**
- See what drives AI decision
- If features don't make sense, be cautious

### 4. **Track Accuracy**
- Monitor live accuracy in Model Info tab
- If accuracy drops below 52%, investigate

### 5. **Compare Multiple Signals**
- Check AI Prediction tab for ranked list
- Cross-reference with Scanner results
- Look for stocks appearing in both

---

## 📈 Expected Results

### Training (First Time)
- **Time**: 2-5 minutes with 100+ stocks
- **Accuracy Target**: 55-60% (baseline)
- **Good Performance**: 60-65%
- **Excellent**: 65%+

### Inference (Live Usage)
- **Speed**: <100ms per stock
- **Resources**: CPU only, no GPU needed
- **Updates**: Real-time on-demand

---

## 🔜 Coming Soon (Phase 2)

- [ ] **Daily Scheduler**: Auto-run predictions before market open
- [ ] **Sentiment Integration**: Add news/social sentiment
- [ ] **Backtesting Tab**: See historical performance
- [ ] **Position Sizing**: AI-suggested allocation
- [ ] **Alerts**: Notify when high-confidence signals appear

---

## 📚 Learn More

- **Full Guide**: `AI_IMPLEMENTATION_GUIDE.md`
- **Architecture**: `AI_BASELINE_SUMMARY.md`
- **Database Schema**: `AI_SUPABASE_SCHEMA.sql`
- **Code**: `ai_features.py`, `ai_model.py`

---

## 🆘 Need Help?

### Common Issues

**Issue**: ImportError: No module named 'lightgbm'
```bash
pip install lightgbm scikit-learn
```

**Issue**: Prediction takes too long
- Check data size (should be < 1000 candles)
- Ensure model file is not corrupted
- Try with smaller stock first

**Issue**: Low accuracy (<50%)
- Need more training data (100+ stocks minimum)
- Retrain with longer history (1+ years)
- Check feature quality (no NaN values)

---

## 🎉 You're Ready!

The AI Insights tab is now part of your trading workflow.

**Start exploring:**
1. Check out AI Predictions for today
2. Analyze a few stocks manually
3. Compare AI vs Technical signals
4. Monitor model performance

**Remember:** AI is a tool to enhance decision-making, not replace it. Always combine with your own analysis!

---

**Happy Trading! 🚀**

*StockGenie Pro V2.9 - AI Enhanced*
