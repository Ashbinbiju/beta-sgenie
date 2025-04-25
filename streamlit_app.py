import yfinance as yf
import pandas as pd
import ta
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from tqdm import tqdm
import plotly.express as px
import time
import requests
import io
import random
import spacy
from pytrends.request import TrendReq
import numpy as np
import itertools
from arch import arch_model
from tenacity import retry,stop_after_attempt, wait_fixed


TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Measures market volatility",
    "MACD": "Moving Average Convergence Divergence - Trend following",
    "ADX": "Average Directional Index (25+ = Strong Trend)",
    "Bollinger": "Price volatility bands around moving average",
    "Stop Loss": "Risk management price level based on ATR",
    "VWAP": "Volume Weighted Average Price - Intraday trend indicator",
    "Parabolic_SAR": "Parabolic Stop and Reverse - Trend reversal indicator",
    "Fib_Retracements": "Fibonacci Retracements - Support and resistance levels",
    "Ichimoku": "Ichimoku Cloud - Comprehensive trend indicator",
    "CMF": "Chaikin Money Flow - Buying/selling pressure",
    "Donchian": "Donchian Channels - Breakout detection",
    "Keltner": "Keltner Channels - Volatility bands based on EMA and ATR",
    "TRIX": "Triple Exponential Average - Momentum oscillator with triple smoothing",
    "Ultimate_Osc": "Ultimate Oscillator - Combines short, medium, and long-term momentum",
    "CMO": "Chande Momentum Oscillator - Measures raw momentum (-100 to 100)",
    "VPT": "Volume Price Trend - Tracks trend strength with price and volume",
    "Supertrend": "Trend-following indicator that changes color based on direction",
    "Elder_Ray": "Elder's Impulse System - Combines trend and momentum",
    "Z_Score": "Statistical measure of how far from mean",
}

SECTORS = {
    "Agri": ["BHAGCHEM.NS", "CHAMBLFERT.NS", "FACT.NS", "GSFC.NS", "INSECTICID.NS"],
    "Alcohol": ["ABDL.NS", "GLOBUSSPR.NS", "PICCADIL.NS", "RADICO.NS", "SDBL.NS"],
    "Automobile": ["APOLLOTYRE.NS", "ASHOKLEY.NS", "TATAMOTORS.NS", "UNOMINDA.NS"],
    "Bank": ["AUBANK.NS", "BANDHANBNK.NS", "SBIN.NS", "INDUSINDBK.NS"],
    "IT": ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "LUPIN.NS"],
    "FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "Metal": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS"],
    "Oil & Gas": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS"],
    "Power": ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIPOWER.NS"]
}

def tooltip(label, explanation):
    return f"{label} 📌 ({explanation})"

@retry(max_retries=3, delay=2)
def fetch_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        stock_list = [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
        return stock_list
    except Exception:
        return list(set([stock for sector in SECTORS.values() for stock in sector]))

@lru_cache(maxsize=100)
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    try:
        if ".NS" not in symbol:
            symbol += ".NS"
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        return data
    except Exception:
        return pd.DataFrame()

def calculate_advance_decline_ratio(stock_list):
    advances = 0
    declines = 0
    for symbol in stock_list:
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                advances += 1
            else:
                declines += 1
    return advances / declines if declines != 0 else 0

def monte_carlo_simulation(data, simulations=1000, days=30):
    returns = data['Close'].pct_change().dropna()
    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='Normal')
    garch_fit = model.fit(disp='off')
    forecasts = garch_fit.forecast(horizon=days)
    volatility = np.sqrt(forecasts.variance.iloc[-1].values)
    mean_return = returns.mean()
    simulation_results = []
    for _ in range(simulations):
        price_series = [data['Close'].iloc[-1]]
        for i in range(days):
            price = price_series[-1] * (1 + np.random.normal(mean_return, volatility[i]))
            price_series.append(price)
        simulation_results.append(price_series)
    return simulation_results

def calculate_confidence_score(data):
    score = 0
    if 'RSI' in data.columns and data['RSI'].iloc[-1] < 30:
        score += 30 * (1 - (data['RSI'].iloc[-1]/30))
    elif 'RSI' in data.columns and data['RSI'].iloc[-1] > 70:
        score -= 30 * ((data['RSI'].iloc[-1]-70)/30)
    
    if 'MACD' in data.columns and 'MACD_signal' in data.columns:
        macd_diff = data['MACD'].iloc[-1] - data['MACD_signal'].iloc[-1]
        score += 15 * (macd_diff / (data['MACD'].std() or 1))
    
    if 'ADX' in data.columns and data['ADX'].iloc[-1] > 25:
        if 'EMA_20' in data.columns and 'EMA_50' in data.columns:
            if data['EMA_20'].iloc[-1] > data['EMA_50'].iloc[-1]:
                score += 20
            else:
                score -= 20
    
    if 'Volume_Spike' in data.columns and data['Volume_Spike'].iloc[-1]:
        if 'VWAP' in data.columns and data['Close'].iloc[-1] > data['VWAP'].iloc[-1]:
            score += 15
        else:
            score -= 15
    
    if 'ATR' in data.columns:
        atr_ratio = data['ATR'].iloc[-1] / data['ATR'].mean()
        score += 10 * (1 - min(atr_ratio, 2)/2)
    
    return min(100, max(0, score))

def assess_risk(data):
    if 'ATR' not in data.columns:
        return "Medium Risk"
    
    atr_ratio = data['ATR'].iloc[-1] / data['ATR'].mean()
    if atr_ratio > 1.5:
        return "High Volatility"
    elif atr_ratio < 0.7:
        return "Low Volatility"
    return "Medium Risk"

def optimize_rsi_window(data, windows=range(5, 50)):
    best_window, best_sharpe = 14, -float('inf')
    returns = data['Close'].pct_change().dropna()
    
    for window in windows:
        try:
            rsi = ta.momentum.RSIIndicator(data['Close'], window=window).rsi()
            signals = (rsi < 30).astype(int) - (rsi > 70).astype(int)
            strategy_returns = signals.shift(1) * returns
            sharpe = strategy_returns.mean() / (strategy_returns.std() or 1)
            if sharpe > best_sharpe:
                best_sharpe, best_window = sharpe, window
        except:
            continue
    return best_window

def detect_divergence(data):
    if len(data) < 10:
        return "No Divergence"
    
    price = data['Close']
    rsi = data['RSI']
    
    price_highs = price.rolling(5).max()
    price_lows = price.rolling(5).min()
    rsi_highs = rsi.rolling(5).max()
    rsi_lows = rsi.rolling(5).min()
    
    bullish = (price_lows.iloc[-1] < price_lows.iloc[-2]) and (rsi_lows.iloc[-1] > rsi_lows.iloc[-2])
    bearish = (price_highs.iloc[-1] > price_highs.iloc[-2]) and (rsi_highs.iloc[-1] < rsi_highs.iloc[-2])
    
    return "Bullish Divergence" if bullish else "Bearish Divergence" if bearish else "No Divergence"

def add_supertrend(data, period=7, multiplier=3):
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr = ta.volatility.true_range(high, low, close)
    atr = ta.volatility.average_true_range(high, low, close, window=period)
    
    hl2 = (high + low) / 2
    data['Supertrend_Upper'] = hl2 + (multiplier * atr)
    data['Supertrend_Lower'] = hl2 - (multiplier * atr)
    
    data['Supertrend'] = 0
    data['Supertrend_Direction'] = ""
    
    for i in range(1, len(data)):
        if close.iloc[i] > data['Supertrend_Upper'].iloc[i-1]:
            data['Supertrend'].iloc[i] = data['Supertrend_Lower'].iloc[i]
            data['Supertrend_Direction'].iloc[i] = "Up"
        elif close.iloc[i] < data['Supertrend_Lower'].iloc[i-1]:
            data['Supertrend'].iloc[i] = data['Supertrend_Upper'].iloc[i]
            data['Supertrend_Direction'].iloc[i] = "Down"
        else:
            data['Supertrend'].iloc[i] = data['Supertrend'].iloc[i-1]
            data['Supertrend_Direction'].iloc[i] = data['Supertrend_Direction'].iloc[i-1]
    
    return data

def add_elder_ray(data, period=13):
    ema = ta.trend.ema_indicator(data['Close'], window=period)
    data['Elder_Bull'] = data['High'] - ema
    data['Elder_Bear'] = data['Low'] - ema
    data['Elder_Impulse'] = np.where(ema > ema.shift(1), 
                                    np.where(data['Close'] > data['Close'].shift(1), 
                                    "Strong Buy", "Buy"),
                                    np.where(data['Close'] < data['Close'].shift(1), 
                                    "Strong Sell", "Sell"))
    return data

def analyze_stock(data):
    if data.empty or len(data) < 50:
        return data
    
    # Core Indicators
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=optimize_rsi_window(data)).rsi()
    
    macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    
    data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
    data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
    data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
    data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    
    bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data['Upper_Band'] = bollinger.bollinger_hband()
    data['Lower_Band'] = bollinger.bollinger_lband()
    
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
    
    # Volume Indicators
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
    data['Volume_Spike'] = data['Volume'] > (data['Volume'].rolling(20).mean() * 2)
    
    # Advanced Indicators
    data = add_supertrend(data)
    data = add_elder_ray(data)
    
    ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'])
    data['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
    data['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
    data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
    data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
    
    # Divergence Detection
    data['Divergence'] = detect_divergence(data)
    
    # Statistical Indicators
    data['Z_Score'] = (data['Close'] - data['Close'].rolling(20).mean()) / data['Close'].rolling(20).std()
    
    return data

def calculate_position_size(data, risk_per_trade=0.01, account_size=100000):
    if 'ATR' not in data.columns or pd.isna(data['ATR'].iloc[-1]):
        return 0
    
    risk_amount = account_size * risk_per_trade
    atr = data['ATR'].iloc[-1]
    position_size = risk_amount / atr
    return int(position_size)

def generate_recommendations(data, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold", "Position": "Hold",
        "Current Price": data['Close'].iloc[-1] if not data.empty else None,
        "Buy At": None, "Stop Loss": None, "Target": None,
        "Confidence": 0, "Risk": "Medium", "Position Size": 0
    }
    
    if data.empty:
        return recommendations
    
    # Calculate confidence score
    confidence = calculate_confidence_score(data)
    recommendations["Confidence"] = confidence
    recommendations["Risk"] = assess_risk(data)
    
    # Price levels
    recommendations["Buy At"] = data['Close'].iloc[-1] * 0.995 if confidence > 60 else data['Close'].iloc[-1]
    recommendations["Stop Loss"] = data['Close'].iloc[-1] - (2 * data['ATR'].iloc[-1])
    recommendations["Target"] = data['Close'].iloc[-1] + (3 * data['ATR'].iloc[-1])
    
    # Position sizing
    recommendations["Position Size"] = calculate_position_size(data)
    
    # Strategy recommendations
    if confidence > 70:
        recommendations["Intraday"] = "Strong Buy"
        recommendations["Swing"] = "Buy"
    elif confidence > 60:
        recommendations["Intraday"] = "Buy"
        recommendations["Swing"] = "Buy"
    elif confidence < 30:
        recommendations["Intraday"] = "Strong Sell"
        recommendations["Swing"] = "Sell"
    elif confidence < 40:
        recommendations["Intraday"] = "Sell"
        recommendations["Swing"] = "Sell"
    
    # Position trading recommendation
    if data['SMA_50'].iloc[-1] > data['SMA_200'].iloc[-1]:
        recommendations["Position"] = "Buy on Dips" if confidence > 50 else "Hold"
    else:
        recommendations["Position"] = "Sell on Rallies" if confidence < 50 else "Hold"
    
    # Supertrend confirmation
    if 'Supertrend_Direction' in data.columns:
        if data['Supertrend_Direction'].iloc[-1] == "Up":
            recommendations["Intraday"] = max(recommendations["Intraday"], "Buy")
        else:
            recommendations["Intraday"] = min(recommendations["Intraday"], "Sell")
    
    return recommendations

def display_dashboard(symbol=None, data=None, recommendations=None, selected_stocks=None):
    st.title("📊 Enhanced StockGenie Pro")
    
    if st.button("🚀 Generate Top Picks"):
        with st.spinner("Analyzing market..."):
            results = analyze_all_stocks(selected_stocks)
            if not results.empty:
                st.subheader("🏆 Top 10 Stocks")
                for _, row in results.iterrows():
                    with st.expander(f"{row['Symbol']} - Confidence: {row['Confidence']}%"):
                        st.markdown(f"""
                        **Price:** ₹{row['Current Price']:.2f}  
                        **Buy At:** ₹{row['Buy At']:.2f} | **Stop Loss:** ₹{row['Stop Loss']:.2f}  
                        **Target:** ₹{row['Target']:.2f}  
                        **Intraday:** {colored_recommendation(row['Intraday'])}  
                        **Swing:** {colored_recommendation(row['Swing'])}  
                        **Position:** {colored_recommendation(row['Position'])}  
                        **Risk:** {row['Risk']} | **Position Size:** {row['Position Size']}
                        """)
    
    if symbol and not data.empty:
        st.header(f"{symbol.split('.')[0]} Analysis")
        
        # Key Metrics
        cols = st.columns(4)
        cols[0].metric("Current Price", f"₹{recommendations['Current Price']:.2f}")
        cols[1].metric("Buy At", f"₹{recommendations['Buy At']:.2f}")
        cols[2].metric("Stop Loss", f"₹{recommendations['Stop Loss']:.2f}")
        cols[3].metric("Target", f"₹{recommendations['Target']:.2f}")
        
        # Recommendations
        st.subheader("Trading Recommendations")
        rec_cols = st.columns(3)
        rec_cols[0].markdown(f"**Intraday**\n\n{colored_recommendation(recommendations['Intraday'])}")
        rec_cols[1].markdown(f"**Swing**\n\n{colored_recommendation(recommendations['Swing'])}")
        rec_cols[2].markdown(f"**Position**\n\n{colored_recommendation(recommendations['Position'])}")
        
        # Charts
        tab1, tab2, tab3 = st.tabs(["Price Action", "Indicators", "Advanced"])
        
        with tab1:
            fig = px.line(data, y=['Close', 'SMA_50', 'SMA_200', 'Supertrend'], 
                          title="Price with Moving Averages & Supertrend")
            st.plotly_chart(fig)
        
        with tab2:
            fig = px.line(data, y=['RSI', 'MACD', 'MACD_signal'], 
                          title="Momentum Indicators")
            st.plotly_chart(fig)
        
        with tab3:
            fig = px.line(data, y=['Ichimoku_Conversion', 'Ichimoku_Base', 'Ichimoku_Span_A', 'Ichimoku_Span_B'], 
                          title="Ichimoku Cloud")
            st.plotly_chart(fig)

def main():
    st.sidebar.title("Navigation")
    NSE_STOCKS = fetch_nse_stock_list()
    
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        options=list(SECTORS.keys()),
        default=["IT", "Bank"]
    )
    
    selected_stocks = list(set([stock for sector in selected_sectors for stock in SECTORS[sector] if stock in NSE_STOCKS]))
    
    symbol = st.sidebar.selectbox(
        "Select Stock",
        options=[""] + selected_stocks,
        format_func=lambda x: x.split('.')[0] if x else ""
    )
    
    if symbol:
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data, symbol)
            display_dashboard(symbol, data, recommendations, selected_stocks)
        else:
            st.error("Failed to load data for selected stock")
    else:
        display_dashboard(selected_stocks=selected_stocks)

if __name__ == "__main__":
    main()