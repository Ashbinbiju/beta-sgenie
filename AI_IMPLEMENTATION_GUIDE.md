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

## 🧪 Backtesting Framework

The `ai_backtest.py` module provides comprehensive performance validation using walk-forward testing methodology.

### Quick Start

```python
from ai_backtest import AIBacktester
import pandas as pd

# Load predictions and price data
predictions_df = pd.read_sql("SELECT * FROM ai_predictions WHERE ts >= '2024-01-01'", supabase_conn)
price_data = load_historical_prices(['SBIN', 'RELIANCE', 'TCS'])

# Initialize backtester
backtester = AIBacktester(
    initial_capital=100000,
    commission_rate=0.0003,   # 0.03% brokerage
    slippage_rate=0.0005,     # 0.05% market impact
    position_sizing='confidence'  # Scale by ML confidence
)

# Run backtest
results = backtester.backtest_predictions(
    predictions_df=predictions_df,
    price_data=price_data,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Display results
backtester.print_report(results)

# Save to database
backtester.save_results_to_supabase(results, model_id='model-123')
```

### Position Sizing Strategies

**1. Fixed Size** (Conservative)
```python
AIBacktester(position_sizing='fixed', default_position_size=10000)
# Every trade uses ₹10,000
```

**2. Equal Weight** (Balanced)
```python
AIBacktester(position_sizing='equal_weight', max_positions=10)
# Splits capital equally: ₹100k / 10 = ₹10k per position
```

**3. Confidence-Based** (Aggressive)
```python
AIBacktester(position_sizing='confidence')
# High confidence (90%) = ₹15k
# Medium confidence (70%) = ₹10k
# Low confidence (60%) = ₹7.5k
```

### Performance Metrics Explained

#### Returns Metrics
- **Total Return**: Overall profit/loss percentage
- **Annualized Return**: CAGR equivalent
- **Monthly Return**: Average per month

#### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted returns (target: >1.5)
  - `< 1.0`: Poor risk-adjusted performance
  - `1.0-2.0`: Acceptable
  - `> 2.0`: Excellent
  
- **Sortino Ratio**: Downside risk focus (target: >2.0)
- **Max Drawdown**: Largest peak-to-trough decline (target: <20%)
- **Volatility**: Annualized return std dev

#### Trading Metrics
- **Win Rate**: % of profitable trades (target: >55%)
- **Profit Factor**: Gross profit / Gross loss (target: >1.5)
- **Expectancy**: Average ₹ per trade (target: >₹200)
- **Avg Win/Loss**: Size of typical winner vs loser

#### Additional Metrics
- **Recovery Factor**: Total return / Max drawdown
- **Calmar Ratio**: Return / Max drawdown
- **Consecutive Wins/Losses**: Longest streaks
- **Total Trades**: Sample size for statistical validity

### Interpreting Results

#### 🟢 Good Model Performance
```
✅ Sharpe Ratio > 1.5
✅ Win Rate > 55%
✅ Profit Factor > 1.5
✅ Max Drawdown < 20%
✅ Total Trades > 100
```
**Action**: Ready for paper trading

#### 🟡 Marginal Performance
```
⚠️ Sharpe Ratio 0.8-1.5
⚠️ Win Rate 50-55%
⚠️ Profit Factor 1.2-1.5
⚠️ Max Drawdown 20-30%
```
**Action**: Tune hyperparameters or add features

#### 🔴 Poor Performance
```
❌ Sharpe Ratio < 0.8
❌ Win Rate < 50%
❌ Profit Factor < 1.2
❌ Max Drawdown > 30%
```
**Action**: Major model revision needed

### Trading Simulation Details

#### Entry Logic
- **BUY** signal + available capital → Enter long position
- Max 10 concurrent positions
- Skip if capital insufficient

#### Exit Logic
- **SELL** signal → Close position immediately
- **10-day holding period** → Force exit (prevent indefinite holds)
- Track P&L per trade

#### Transaction Costs
```python
# Realistic Indian market costs
commission = 0.03%        # Discount broker
slippage = 0.05%          # Market impact
stt = 0.10%              # Securities Transaction Tax
total_cost = ~0.18% per round trip
```

### Example Outputs

#### Sample Report
```
================================================================
                     BACKTEST REPORT                           
================================================================

CONFIGURATION
  Period: 2024-01-01 to 2024-12-31
  Initial Capital: ₹100,000
  Position Sizing: confidence
  Commission: 0.03% | Slippage: 0.05% | STT: 0.10%

RETURNS
  Total Return: 24.8% (₹24,800)
  Annualized Return: 26.3%
  Monthly Return: 2.1%
  CAGR: 25.1%

RISK METRICS
  Sharpe Ratio: 2.15
  Sortino Ratio: 3.42
  Max Drawdown: -12.5%
  Volatility: 18.2%
  Recovery Factor: 1.98

TRADING METRICS
  Total Trades: 127
  Win Rate: 58.3%
  Profit Factor: 1.85
  Avg Win: ₹1,450 (4.2%)
  Avg Loss: -₹875 (-2.8%)
  Expectancy: ₹195 per trade
```

#### Top Trades
```
   Symbol  Entry Date   Exit Date    P&L    Return  Confidence
0  RELIANCE  2024-03-15  2024-03-22  ₹2,340   6.8%     92%
1  TCS       2024-06-10  2024-06-18  ₹1,890   5.4%     88%
2  INFY      2024-08-05  2024-08-12  ₹1,650   5.1%     85%
```

### Backtesting Best Practices

#### 1. Data Requirements
```python
# Minimum requirements
MIN_STOCKS = 20           # Diversification
MIN_HISTORY_DAYS = 252    # 1 year
MIN_PREDICTIONS = 100     # Statistical validity
```

#### 2. Walk-Forward Validation
```python
# Train on past → Test on future
# Prevents look-ahead bias
train_period = '2023-01-01 to 2023-12-31'
test_period = '2024-01-01 to 2024-06-30'
```

#### 3. Out-of-Sample Testing
```python
# Never backtest on training data
# Use completely unseen time periods
# Retrain quarterly with rolling window
```

#### 4. Reality Checks
```python
# If results seem too good:
if sharpe_ratio > 3.0 or win_rate > 70%:
    # Check for data leakage
    # Verify no future information in features
    # Test on different time periods
```

### Advanced Usage

#### Compare Strategies
```python
strategies = {
    'Fixed': AIBacktester(position_sizing='fixed'),
    'Confidence': AIBacktester(position_sizing='confidence'),
    'Equal': AIBacktester(position_sizing='equal_weight')
}

for name, bt in strategies.items():
    results = bt.backtest_predictions(predictions_df, price_data)
    print(f"{name}: Sharpe={results['metrics']['sharpe_ratio']:.2f}")
```

#### Confidence Analysis
```python
# Group trades by confidence level
trades_df = pd.DataFrame(results['trade_log'])
trades_df['confidence_bucket'] = pd.cut(
    trades_df['confidence'], 
    bins=[0, 0.7, 0.8, 0.9, 1.0]
)
performance_by_confidence = trades_df.groupby('confidence_bucket')['pnl'].mean()
```

#### Parameter Optimization
```python
# Grid search for best parameters
for max_pos in [5, 10, 15]:
    for pos_size in [5000, 10000, 15000]:
        bt = AIBacktester(
            max_positions=max_pos,
            default_position_size=pos_size
        )
        results = bt.backtest_predictions(predictions_df, price_data)
        # Log metrics for comparison
```

### Troubleshooting

**Q: Zero trades executed**
```python
# Check:
1. Do predictions have 'BUY'/'SELL' signals?
2. Is price_data available for all symbols?
3. Is date range aligned between predictions and prices?

# Debug:
print(predictions_df['final_signal'].value_counts())
print(price_data.keys())
```

**Q: Unrealistic results**
```python
# Common issues:
1. Look-ahead bias: Future data in features
2. Survivorship bias: Only successful stocks
3. Missing transaction costs
4. Overfitting on in-sample data

# Solution: Use walk-forward + out-of-sample testing
```

**Q: High drawdown**
```python
# Reduce risk:
backtester = AIBacktester(
    max_positions=5,              # Less exposure
    default_position_size=5000,   # Smaller positions
    position_sizing='fixed'       # Conservative sizing
)
```

---

## 📚 Next Steps

### Phase 1 ✅
- [x] Database schema
- [x] Feature engineering
- [x] LightGBM baseline model

### Phase 2 (Current) 🔄
- [x] Backtesting framework
- [x] Streamlit UI integration
- [ ] Sentiment integration
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
