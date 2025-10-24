# Unified Analysis Engine Architecture

## Overview
This document outlines the comprehensive unified analysis system that combines all available data sources into a single powerful recommendation engine for both intraday and swing trading.

## Data Sources

### 1. Zerodha APIs
**a) Shareholding Pattern**
- Endpoint: `https://zerodha.com/markets/stocks/NSE/{symbol}/shareholdings/`
- Data: Promoter %, FII %, DII %, Public %, Pledge %
- Weight: Part of Fundamental Score (10-15%)

**b) Financial Summary** (5 sections)
- Endpoint: `https://zerodha.com/markets/stocks/NSE/{symbol}/financials/`
- Sections:
  - Summary: Revenue, PPOP, OPM
  - P&L Statement: Income, Expenses, Profit
  - Balance Sheet: Assets, Liabilities, Equity
  - Cash Flow: Operating, Investing, Financing
  - Financial Ratios: OPM, NPM, ROE, ROCE, D/E
- Weight: Part of Fundamental Score (20-30%)

### 2. Streak APIs
**a) Technical Analysis**
- Endpoint: `https://technicalwidget.streak.tech/api/streak_tech_analysis/`
- Data: RSI, MACD, ADX, State (1/-1), Win %
- Weight: Part of Technical Score (30-40%)

**b) Support & Resistance**
- Endpoint: `https://mo.streak.tech/api/sr_analysis_multi/`
- Data: Pivot, R1-R3, S1-S3, Current Price
- Weight: Critical for entry/exit timing (penalties up to -45 pts)

**c) Candlestick Data**
- Endpoint: `https://technicalwidget.streak.tech/api/candles/`
- Data: Historical price action, patterns
- Weight: Visual confirmation, not scored

### 3. News Sentiment
**a) StockEdge News API**
- Endpoint: `https://api.stockedge.com/Api/SecurityDashboardApi/GetNewsitemsForSecurity/{id}`
- Data: Headlines, descriptions, details, dates
- Sentiment: Keyword-based (positive/negative/neutral)
- Weight: News Score (5-10%)

### 4. Market Context
**a) Market Breadth**
- Endpoint: `https://brkpoint.in/api/market-stats`
- Data: Advancing/Declining ratio, industry performance
- Weight: Market Health multiplier (0.7x to 1.1x)

**b) Sector Performance**
- Endpoint: `https://brkpoint.in/api/sector-indices-performance`
- Data: Nifty50, Nifty500, sector indices, momentum, volatility
- Weight: Sector alignment bonus (±5 pts)

**c) Index Trend**
- Endpoint: `https://brkpoint.in/api/indextrend`
- Data: Nifty & Bank Nifty trend analysis
- Weight: Index alignment (contextual)

## Unified Scoring Framework

### For SWING Trading
```
Total Score (0-100) = Base Score × Market Health Multiplier

Base Score Components:
- Fundamental Score (40%):
  • Revenue Growth: 0-20 pts
  • Profitability (OPM, NPM): 0-25 pts
  • Shareholding Pattern: 0-15 pts
  • Financial Health: 0-20 pts
  • Subtotal: 0-80 pts normalized to 0-40

- Technical Score (35%):
  • Trend (EMA, MACD): 0-20 pts
  • Momentum (RSI, ADX): 0-20 pts
  • Volume: 0-10 pts
  • Win Rate: 0-15 pts
  • Subtotal: 0-65 pts normalized to 0-35

- Support/Resistance (15%):
  • Near Support (S1): +15 pts
  • Above Pivot: +10 pts
  • Near Resistance (R1): -25 pts
  • Near R2: -35 pts
  • Near R3: -45 pts
  • Range: -45 to +15 pts normalized to 0-15

- News Sentiment (5%):
  • Very Positive: 5 pts
  • Positive: 3 pts
  • Neutral: 0 pts
  • Negative: -3 pts
  • Very Negative: -5 pts
  • Range: -5 to +5 pts normalized to 0-5

- Sector/Market Alignment (5%):
  • Industry alignment: ±3 pts
  • Market breadth: ±2 pts
  • Range: -5 to +5 pts normalized to 0-5

Market Health Multiplier:
- Very Bearish (<20): 0.7x
- Bearish (20-40): 0.85x
- Neutral (40-60): 1.0x
- Bullish (60-80): 1.05x
- Very Bullish (80+): 1.1x
```

### For INTRADAY Trading
```
Total Score (0-100) = Base Score × Market Health Multiplier

Base Score Components:
- Fundamental Score (15%):
  • Quick health check (promoter, margins)
  • Subtotal: 0-100 normalized to 0-15

- Technical Score (50%):
  • Price vs VWAP: 0-25 pts
  • EMA alignment: 0-25 pts
  • OR Breakout: 0-25 pts
  • RSI (intraday): 0-15 pts
  • Volume: 0-10 pts
  • Subtotal: 0-100 normalized to 0-50

- Support/Resistance (20%):
  • Near Support (S1): +20 pts
  • Above Pivot: +15 pts
  • Near Resistance (R1): -30 pts
  • Near R2: -40 pts
  • Near R3: -50 pts
  • Range: -50 to +20 pts normalized to 0-20

- News Sentiment (10%):
  • Higher weight for breaking news impact
  • Recent news (last 2 hours) gets 2x weight
  • Range: -10 to +10 pts normalized to 0-10

- Market/Index Momentum (5%):
  • Nifty/BankNifty real-time trend
  • Range: -5 to +5 pts normalized to 0-5

Market Health Multiplier:
- Very Bearish (<20): 0.65x (more aggressive for intraday)
- Bearish (20-40): 0.8x
- Neutral (40-60): 1.0x
- Bullish (60-80): 1.1x
- Very Bullish (80+): 1.15x
```

## Signal Generation Logic

### Score to Signal Mapping
```python
if score >= 75: signal = "STRONG BUY"
elif score >= 65: signal = "BUY"
elif score >= 55: signal = "MODERATE BUY"
elif score >= 45: signal = "HOLD"
elif score >= 35: signal = "MODERATE SELL"
elif score >= 25: signal = "SELL"
else: signal = "STRONG SELL"
```

### Override Rules (Safety Mechanisms)
1. **Resistance Zone Override**
   - If near R1 and signal = STRONG BUY → Downgrade to BUY
   - If near R2 and signal in [STRONG BUY, BUY] → Downgrade to MODERATE BUY
   - If near R3 → Block all BUY signals, max HOLD

2. **News Sentiment Override**
   - If Very Negative news + STRONG BUY → Downgrade to BUY with warning
   - If Very Positive news + STRONG SELL → Downgrade to SELL with note

3. **Market Crash Override**
   - If market_health < 15 → Block all BUY signals
   - Allow only SELL/STRONG SELL

4. **Promoter Pledge Override**
   - If pledge > 75% → Block STRONG BUY, add critical warning
   - If pledge > 90% → Allow max HOLD signal

## Confidence Score Calculation
```python
confidence = 0

# Data availability (max 30 points)
if financials: confidence += 10
if shareholdings: confidence += 5
if technical_data: confidence += 10
if sr_data: confidence += 5

# Signal alignment (max 40 points)
if fundamental_signal == technical_signal: confidence += 20
if news_sentiment aligns with signal: confidence += 10
if market_trend aligns with signal: confidence += 10

# Data freshness (max 15 points)
if news_age < 24h: confidence += 10
if technical_data_fresh: confidence += 5

# Trend strength (max 15 points)
if ADX > 25: confidence += 10
if volume > avg_volume: confidence += 5

Total Confidence: 0-100
```

## Risk Metrics
```python
# Position Sizing
max_risk_per_trade = account_size * 0.02  # 2% rule
stop_loss_distance = abs(entry - stop_loss)
position_size = max_risk_per_trade / stop_loss_distance

# Risk-Reward Ratio
rr_ratio = (target - entry) / (entry - stop_loss)
min_acceptable_rr = 1.5 for swing, 1.2 for intraday

# Max Drawdown Alert
if consecutive_losses > 3: reduce_position_size by 50%
```

## Output Structure
```python
{
    "symbol": str,
    "trading_style": "SWING" | "INTRADAY",
    "timestamp": datetime,
    
    # Core Recommendation
    "overall_score": 0-100,
    "signal": str,
    "confidence": 0-100,
    
    # Component Scores
    "scores": {
        "fundamental": 0-100,
        "technical": 0-100,
        "support_resistance": 0-100,
        "news_sentiment": 0-100,
        "market_alignment": 0-100
    },
    
    # Entry/Exit Levels
    "entry_price": float,
    "stop_loss": float,
    "target": float,
    "risk_reward_ratio": float,
    
    # Position Sizing
    "position_size": int,
    "max_loss": float,
    "potential_profit": float,
    
    # Key Insights (AI-generated)
    "strengths": [str],
    "weaknesses": [str],
    "opportunities": [str],
    "threats": [str],
    
    # Supporting Data
    "market_context": {
        "health": 0-100,
        "signal": str,
        "factors": dict
    },
    "news_summary": {
        "sentiment": str,
        "recent_count": int,
        "key_headlines": [str]
    },
    "technical_levels": {
        "pivot": float,
        "support": [float],
        "resistance": [float]
    }
}
```

## Implementation Plan

### Phase 1: Core Engine
1. Create `UnifiedAnalysisEngine` class
2. Implement data fetching layer with caching
3. Build scoring system for both swing and intraday
4. Add override rules and safety mechanisms

### Phase 2: Integration
1. Replace `get_comprehensive_analysis()`
2. Replace `generate_recommendation()`
3. Merge into single function call

### Phase 3: UI Refinement
1. Remove duplicate analysis tabs
2. Create unified recommendation view
3. Build comprehensive insights dashboard
4. Add interactive charts with levels

### Phase 4: Testing & Validation
1. Test with 50+ stocks (swing + intraday)
2. Validate score calculations
3. Ensure logical consistency
4. Performance optimization

## Success Metrics
- Single coherent analysis instead of fragmented views
- Clear, actionable recommendations
- Proper weight to fundamentals for swing, technicals for intraday
- News sentiment integration
- Market context awareness
- No logical contradictions (e.g., STRONG BUY near resistance)
