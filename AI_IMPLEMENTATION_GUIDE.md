# 🤖 AI Stock Genie - Implementation Guide

## 📋 Overview

This implementation adds **AI-powered stock predictions** to StockGenie Pro using a lightweight, production-ready approach:

- **LightGBM** model for direction prediction (UP/DOWN/FLAT)
- **Feature engineering** from existing technical indicators
- **Supabase** integration for data storage and model versioning
- **Explainable AI** with feature importance and reasoning

---

## 🏗️ Architecture

```
┌─────────────────┐
│  SmartAPI Data  │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Engineer   │ ← Extract 28+ ML features
│  (ai_features.py)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Supabase Storage   │ ← Cache features, predictions
│  (ai_features table)│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LightGBM Model     │ ← Train & predict
│  (ai_model.py)      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  AI Predictions     │ ← Store results
│  (ai_predictions)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Streamlit UI       │ ← Display insights
│  (AI Insights tab)  │
└─────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install lightgbm scikit-learn pandas-ta
```

### 2. Setup Supabase Tables

Run the SQL schema in your Supabase project:

```bash
# Open Supabase SQL Editor
# Copy & paste contents of AI_SUPABASE_SCHEMA.sql
# Execute to create tables
```

Tables created:
- `ai_features` - Cached ML features
- `ai_predictions` - Model predictions
- `model_metadata` - Model versioning
- `backtest_results` - Performance tracking
- `user_feedback` - Continuous learning

### 3. Extract Features

```python
from ai_features import FeatureEngineer
import streamlit as st

# Initialize engineer
engineer = FeatureEngineer()

# Extract features for a stock
df = fetch_stock_data('SBIN', interval='1d', period='1y')
features = engineer.extract_features(df, symbol='SBIN', sector='Banking')

# Save to Supabase
if st.session_state.get('supabase'):
    save_features_to_supabase(
        pd.DataFrame([features]), 
        st.session_state.supabase
    )
```

### 4. Train Model

```python
from ai_model import StockDirectionModel

# Initialize model
model = StockDirectionModel(model_version='v1.0.0')

# Prepare training data
X, y = model.prepare_training_data(features_df, price_df, forecast_horizon=5)

# Train with cross-validation
metrics = model.train(X, y, n_splits=5)

# Save model
model.save_model('models/stock_direction_v1.0.0.pkl')

# Save metadata to Supabase
save_model_metadata_to_supabase(model, supabase_client)
```

### 5. Make Predictions

```python
# Load trained model
model = StockDirectionModel()
model.load_model('models/stock_direction_v1.0.0.pkl')

# Get features for a stock
features = engineer.extract_features(df, 'SBIN', 'Banking')

# Predict
prediction = model.predict(features)

print(f"Direction: {prediction['ml_direction']}")
print(f"Confidence: {prediction['ml_confidence']:.2%}")
print(f"ML Score: {prediction['ml_score']:.2f}")
```

---

## 📊 Features Extracted

### Price-Based (5 features)
- `returns_1d`, `returns_5d`, `returns_20d`
- `volatility_10d`, `volatility_30d`

### Technical Indicators (12 features)
- RSI(14), MACD, MACD Signal, MACD Histogram
- ADX(14), ATR(14)
- EMA(9, 21, 50)
- SMA(20, 50, 200)

### Volume (4 features)
- Current volume, Volume ratio (20-day)
- OBV, VWAP

### Momentum (3 features)
- Momentum (5-day, 20-day)
- ROC(10)

### Trend (4 features)
- Trend strength
- Support level, Resistance level
- Distance from 52-week high/low

---

## 🎯 Model Output

```python
{
    "ml_direction": "UP",           # UP, DOWN, or FLAT
    "ml_confidence": 0.72,           # Model confidence (0-1)
    "ml_score": 0.72,                # Probability of UP (0-1)
    "probabilities": {
        "DOWN": 0.15,
        "FLAT": 0.13,
        "UP": 0.72
    },
    "top_features": [
        {"feature": "rsi_14", "value": 65.2, "importance": 850.5},
        {"feature": "macd", "value": 2.3, "importance": 720.3},
        ...
    ],
    "model_version": "v1.0.0"
}
```

---

## 🔧 Integration with Existing System

### Fusion Scoring

Combine AI predictions with existing technical analysis:

```python
def get_ai_enhanced_recommendation(symbol):
    # 1. Get existing technical score
    tech_rec = generate_recommendation(data, symbol, ...)
    tech_score = tech_rec['score'] / 100  # Normalize to 0-1
    
    # 2. Get AI prediction
    features = engineer.extract_features(data, symbol)
    ai_pred = model.predict(features)
    ai_score = ai_pred['ml_score']
    
    # 3. Fusion (weighted average)
    final_score = (
        0.5 * tech_score +      # Technical analysis (proven)
        0.3 * ai_score +         # ML prediction (experimental)
        0.2 * sentiment_score    # Sentiment (contextual)
    ) * 100  # Scale back to 0-100
    
    # 4. Generate explainability
    reasons = {
        "technical_score": tech_score * 100,
        "ai_score": ai_score * 100,
        "ai_direction": ai_pred['ml_direction'],
        "ai_confidence": ai_pred['ml_confidence'],
        "top_features": ai_pred['top_features'],
        "fusion_score": final_score
    }
    
    return {
        "score": final_score,
        "signal": map_score_to_signal(final_score),
        "reasons": reasons
    }
```

---

## 📈 Performance Tracking

### Backtest Metrics

Store in `backtest_results` table:
- Sharpe Ratio
- Win Rate
- Max Drawdown
- Total Return
- CAGR

### Live Performance

Track actual predictions:

```python
# After prediction is made
supabase.table('ai_predictions').insert({
    'symbol': 'SBIN',
    'ml_score': 0.72,
    'ml_direction': 'UP',
    'final_score': 75.5,
    'final_signal': 'BUY',
    'model_version': 'v1.0.0'
})

# Update after realization (next day)
actual_return = (price_tomorrow - price_today) / price_today
prediction_correct = (actual_return > 0.02 and ml_direction == 'UP')

supabase.table('ai_predictions').update({
    'actual_return_1d': actual_return,
    'prediction_correct': prediction_correct
}).eq('id', prediction_id).execute()
```

---

## 🔄 Training Pipeline

### Step-by-Step

1. **Data Collection** (Daily)
   ```python
   # Fetch historical data for all stocks
   # Extract features using FeatureEngineer
   # Save to ai_features table
   ```

2. **Model Training** (Weekly/Monthly)
   ```python
   # Load features from last 6-12 months
   # Prepare training data (X, y)
   # Train model with cross-validation
   # Save model and metadata
   ```

3. **Daily Inference** (Before market open)
   ```python
   # For each stock:
   #   1. Extract latest features
   #   2. Run prediction
   #   3. Combine with technical score
   #   4. Store in ai_predictions table
   ```

4. **Performance Monitoring** (Daily)
   ```python
   # Update prediction accuracy
   # Calculate rolling metrics
   # Alert if performance degrades
   ```

---

## 🎨 Streamlit Integration

Add AI Insights tab to `streamlit_app.py`:

```python
with tab_ai_insights:
    st.markdown("### 🤖 AI-Powered Stock Insights")
    
    # Get latest AI predictions
    predictions = supabase.table('ai_predictions')\
        .select('*')\
        .order('created_at', desc=True)\
        .limit(50)\
        .execute()
    
    if predictions.data:
        df = pd.DataFrame(predictions.data)
        
        # Filter by signal
        signal_filter = st.multiselect(
            "Filter by Signal", 
            ['BUY', 'HOLD', 'SELL'],
            default=['BUY']
        )
        
        df_filtered = df[df['final_signal'].isin(signal_filter)]
        
        # Display top picks
        st.dataframe(
            df_filtered[['symbol', 'final_score', 'ml_confidence', 
                         'predicted_return', 'risk_reward_ratio']],
            use_container_width=True
        )
        
        # Explainability section
        if st.button("Show AI Reasoning"):
            selected = df_filtered.iloc[0]
            reasons = selected['reasons']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Technical Score", f"{reasons['technical_score']:.1f}")
            col2.metric("AI Score", f"{reasons['ai_score']:.1f}")
            col3.metric("Fusion Score", f"{reasons['fusion_score']:.1f}")
            
            # Top features
            st.markdown("#### 📊 Top Contributing Features")
            for feat in reasons['top_features']:
                st.markdown(f"- **{feat['feature']}**: {feat['value']:.2f} (importance: {feat['importance']:.1f})")
```

---

## ⚠️ Important Notes

### Before Going Live

1. **Backtest thoroughly** (6+ months of data)
2. **Paper trade** (1-3 months validation)
3. **Monitor performance** (daily accuracy checks)
4. **Set thresholds** (min confidence, max drawdown limits)
5. **A/B test** (compare vs baseline technical system)

### Resource Requirements

- **Training**: 2-5 minutes per model (CPU)
- **Inference**: <100ms per stock (CPU)
- **Storage**: ~100MB for 1 year of features
- **API calls**: Same as current (uses cached data)

### Limitations

- Requires 200+ days of history per stock
- Performance varies by market conditions
- Not suitable for HFT or scalping
- Best for swing trading (3-10 day holds)

---

## 📚 Next Steps

### Phase 1 (Current) ✅
- [x] Database schema
- [x] Feature engineering
- [x] LightGBM baseline model

### Phase 2 (Next 2-3 weeks)
- [ ] Backtesting framework
- [ ] Sentiment integration
- [ ] Streamlit UI integration
- [ ] Daily inference scheduler

### Phase 3 (1-2 months)
- [ ] LSTM for sequential patterns
- [ ] Position sizing optimization
- [ ] Multi-model ensemble
- [ ] Live performance tracking

### Phase 4 (Future)
- [ ] Reinforcement learning for timing
- [ ] Alternative data sources
- [ ] Multi-timeframe analysis
- [ ] Portfolio optimization

---

## 🆘 Support

### Common Issues

**Q: LightGBM import error**
```bash
pip install lightgbm scikit-learn
```

**Q: Features not saving to Supabase**
- Check Supabase connection
- Verify table schema matches
- Check service role key permissions

**Q: Low model accuracy**
- Increase training data (more stocks, longer history)
- Tune hyperparameters
- Add more features (sentiment, sector strength)
- Check data quality (missing values, outliers)

---

## 📝 License

Part of StockGenie Pro V2.9
For internal use only.

---

**Built with ❤️ for smarter trading decisions**
