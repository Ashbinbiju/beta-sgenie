# ============================================================================
# STOCKGENIE PRO - COMPLETE REFACTORED VERSION
# Swing + Intraday Trading System with Streamlined Indicators
# ============================================================================

import pandas as pd
import numpy as np
import ta
import logging
import streamlit as st
from datetime import datetime, timedelta, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import time as time_module
import requests
import io
import random
import warnings
import sqlite3
from diskcache import Cache
from SmartApi import SmartConnect
import pyotp
import os
from dotenv import load_dotenv
from itertools import cycle

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

load_dotenv()

# Environment variables
CLIENT_ID = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
API_KEYS = {
    "Historical": "c3C0tMGn",
    "Trading": os.getenv("TRADING_API_KEY"),
    "Market": os.getenv("MARKET_API_KEY")
}

# Global session cache
_global_smart_api = None
_session_timestamp = None
SESSION_EXPIRY = 900  # 15 mins

cache = Cache("stock_data_cache")

# User agents for API calls
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
]

# Sector definitions
SECTORS = {
    "Bank": [
        "HDFCBANK-EQ", "ICICIBANK-EQ", "SBIN-EQ", "KOTAKBANK-EQ", "AXISBANK-EQ",
        "INDUSINDBK-EQ", "PNB-EQ", "BANKBARODA-EQ", "CANBK-EQ", "UNIONBANK-EQ"
    ],
    "IT": [
        "TCS-EQ", "INFY-EQ", "HCLTECH-EQ", "WIPRO-EQ", "TECHM-EQ", "LTIM-EQ",
        "MPHASIS-EQ", "COFORGE-EQ", "PERSISTENT-EQ"
    ],
    "Auto": [
        "MARUTI-EQ", "TATAMOTORS-EQ", "M&M-EQ", "BAJAJ-AUTO-EQ", "HEROMOTOCO-EQ",
        "EICHERMOT-EQ", "TVSMOTOR-EQ", "ASHOKLEY-EQ"
    ],
    "Pharma": [
        "SUNPHARMA-EQ", "CIPLA-EQ", "DRREDDY-EQ", "DIVISLAB-EQ", "AUROPHARMA-EQ",
        "LUPIN-EQ", "TORNTPHARM-EQ", "ALKEM-EQ"
    ],
    "FMCG": [
        "HINDUNILVR-EQ", "ITC-EQ", "NESTLEIND-EQ", "BRITANNIA-EQ", "DABUR-EQ",
        "MARICO-EQ", "GODREJCP-EQ", "TATACONSUM-EQ"
    ],
    "Energy": [
        "RELIANCE-EQ", "ONGC-EQ", "IOC-EQ", "BPCL-EQ", "HPCL-EQ", "GAIL-EQ",
        "COALINDIA-EQ", "NTPC-EQ", "POWERGRID-EQ"
    ],
    "Metals": [
        "TATASTEEL-EQ", "JSWSTEEL-EQ", "HINDALCO-EQ", "VEDL-EQ", "SAIL-EQ",
        "NMDC-EQ", "HINDZINC-EQ", "JINDALSTEL-EQ"
    ]
}

# Tooltips
TOOLTIPS = {
    "Score": "Combined signal strength (0-100). 50=neutral, >65=buy, <35=sell",
    "RSI": "Relative Strength Index - Momentum indicator (30=oversold, 70=overbought)",
    "MACD": "Trend following indicator - Crossovers indicate trend changes",
    "ATR": "Average True Range - Measures volatility for stop-loss placement",
    "ADX": "Trend strength (>25 = strong trend, <20 = weak/choppy)",
    "VWAP": "Volume Weighted Average Price - Intraday benchmark",
    "EMA": "Exponential Moving Average - Trend direction filter"
}

# ============================================================================
# API & DATA FETCHING
# ============================================================================

def get_global_smart_api():
    """Manage global SmartAPI session with auto-refresh"""
    global _global_smart_api, _session_timestamp
    now = time_module.time()
    
    if _global_smart_api is None or (now - _session_timestamp) > SESSION_EXPIRY:
        st.info("🔄 Refreshing SmartAPI session...")
        _global_smart_api = init_smartapi_client()
        _session_timestamp = now
        if not _global_smart_api:
            st.error("❌ Failed to initialize session.")
            return None
    
    return _global_smart_api

def init_smartapi_client():
    """Initialize SmartAPI client with authentication"""
    try:
        smart_api = SmartConnect(api_key=API_KEYS["Historical"])
        totp = pyotp.TOTP(TOTP_SECRET)
        data = smart_api.generateSession(CLIENT_ID, PASSWORD, totp.now())
        
        if data['status']:
            return smart_api
        else:
            st.error(f"⚠️ SmartAPI auth failed: {data['message']}")
            return None
    except Exception as e:
        st.error(f"⚠️ Error initializing SmartAPI: {str(e)}")
        return None

@st.cache_data(ttl=86400)
def load_symbol_token_map():
    """Load instrument token mapping"""
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {entry["symbol"]: entry["token"] for entry in data if "symbol" in entry and "token" in entry}
    except Exception as e:
        st.warning(f"⚠️ Failed to load instrument list: {str(e)}")
        return {}

def retry(max_retries=3, delay=5, backoff_factor=2):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        if attempt == max_retries:
                            raise
                        sleep_time = delay * (backoff_factor ** (attempt - 1))
                        st.warning(f"Rate limit. Retry {attempt}/{max_retries} in {sleep_time}s...")
                        time_module.sleep(sleep_time)
                    else:
                        raise
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    time_module.sleep(delay)
            raise RuntimeError("Max retries exhausted")
        return wrapper
    return decorator

@retry(max_retries=3, delay=5)
def fetch_stock_data_with_auth(symbol, period="1y", interval="1d"):
    """Fetch stock data from SmartAPI with caching"""
    cache_key = f"{symbol}_{period}_{interval}"
    cached_data = cache.get(cache_key)
    
    if cached_data is not None:
        return pd.read_pickle(io.BytesIO(cached_data))
    
    try:
        if "-EQ" not in symbol:
            symbol = f"{symbol.split('.')[0]}-EQ"
        
        smart_api = get_global_smart_api()
        if not smart_api:
            raise ValueError("SmartAPI session unavailable")
        
        # Calculate date range
        end_date = datetime.now()
        if period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "1d":
            start_date = end_date - timedelta(days=1)
        else:
            start_date = end_date - timedelta(days=365)
        
        # Map interval
        interval_map = {
            "1d": "ONE_DAY",
            "1h": "ONE_HOUR",
            "30m": "THIRTY_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "5m": "FIVE_MINUTE"
        }
        api_interval = interval_map.get(interval, "ONE_DAY")
        
        # Get token
        symbol_token_map = load_symbol_token_map()
        symboltoken = symbol_token_map.get(symbol)
        
        if not symboltoken:
            st.warning(f"⚠️ Token not found for {symbol}")
            return pd.DataFrame()
        
        # Fetch data
        historical_data = smart_api.getCandleData({
            "exchange": "NSE",
            "symboltoken": symboltoken,
            "interval": api_interval,
            "fromdate": start_date.strftime("%Y-%m-%d %H:%M"),
            "todate": end_date.strftime("%Y-%m-%d %H:%M")
        })
        
        if historical_data['status'] and historical_data['data']:
            data = pd.DataFrame(
                historical_data['data'],
                columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            # Cache for 1 hour
            buffer = io.BytesIO()
            data.to_pickle(buffer)
            cache.set(cache_key, buffer.getvalue(), expire=3600)
            
            return data
        else:
            raise ValueError(f"No data for {symbol}")
    
    except Exception as e:
        logging.error(f"Error fetching {symbol}: {str(e)}")
        return pd.DataFrame()

@lru_cache(maxsize=500)
def fetch_stock_data_cached(symbol, period="1y", interval="1d"):
    """Cached wrapper for stock data fetching"""
    return fetch_stock_data_with_auth(symbol, period, interval)

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_data(data, required_columns=None, min_length=50):
    """Comprehensive OHLCV validation"""
    if required_columns is None:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    if data is None or data.empty:
        logging.warning("Empty data")
        return False
    
    if len(data) < min_length:
        logging.warning(f"Insufficient data: {len(data)} < {min_length}")
        return False
    
    missing = [c for c in required_columns if c not in data.columns]
    if missing:
        logging.warning(f"Missing columns: {missing}")
        return False
    
    if data[required_columns].isnull().any().any():
        logging.warning("Null values detected")
        return False
    
    price_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in data.columns]
    if (data[price_cols] <= 0).any().any():
        logging.warning("Invalid prices (<=0)")
        return False
    
    return True

# ============================================================================
# SWING TRADING INDICATORS (DAILY TIMEFRAME)
# ============================================================================

def calculate_swing_indicators(data):
    """
    Calculate swing trading indicators:
    - MACD (12/26/9)
    - 200 EMA
    - RSI (14) with dynamic thresholds
    - ATR (14)
    - Bollinger Bands (20,2)
    - ADX (14)
    - Volume analysis
    """
    if not validate_data(data, min_length=200):
        return data
    
    df = data.copy()
    
    # MACD (Trend)
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # 200 EMA (Primary trend filter)
    df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
    df['Above_EMA200'] = df['Close'] > df['EMA_200']
    
    # RSI with dynamic thresholds
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['RSI_Oversold'] = np.where(df['Above_EMA200'], 40, 30)
    df['RSI_Overbought'] = np.where(df['Above_EMA200'], 70, 60)
    
    # ATR (Volatility)
    df['ATR'] = ta.volatility.AverageTrueRange(
        df['High'], df['Low'], df['Close'], window=14
    ).average_true_range()
    df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
    
    # Bollinger Bands (Support/Resistance)
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # ADX (Trend strength filter)
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    df['Trending'] = df['ADX'] > 25
    
    # Volume analysis
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Spike'] = df['Volume'] > (df['Volume_SMA'] * 1.5)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    return df

def detect_swing_regime(df):
    """Classify swing trading market regime"""
    if len(df) < 200:
        return "Unknown"
    
    close = df['Close'].iloc[-1]
    ema200 = df['EMA_200'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    
    if pd.isna(ema200) or pd.isna(adx):
        return "Unknown"
    
    above_ema = close > ema200
    trending = adx > 25
    macd_bullish = macd > macd_signal
    
    if above_ema and trending and macd_bullish:
        return "Strong Uptrend"
    elif above_ema and trending:
        return "Weak Uptrend"
    elif above_ema:
        return "Consolidation (Above EMA)"
    elif not above_ema and trending and not macd_bullish:
        return "Strong Downtrend"
    elif not above_ema and trending:
        return "Weak Downtrend"
    elif not above_ema:
        return "Consolidation (Below EMA)"
    else:
        return "Neutral"

def calculate_swing_score(df):
    """Calculate swing trading score (0-100)"""
    score = 0
    
    close = df['Close'].iloc[-1]
    ema200 = df['EMA_200'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    macd_prev = df['MACD'].iloc[-2]
    macd_signal_prev = df['MACD_Signal'].iloc[-2]
    rsi = df['RSI'].iloc[-1]
    rsi_oversold = df['RSI_Oversold'].iloc[-1]
    rsi_overbought = df['RSI_Overbought'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    bb_position = df['BB_Position'].iloc[-1]
    volume_ratio = df['Volume_Ratio'].iloc[-1]
    
    # Above/below 200 EMA (±3 points)
    if close > ema200:
        score += 3
    else:
        score -= 3
    
    # MACD crossover (±4 points for fresh, ±1 for sustained)
    if macd > macd_signal and macd_prev <= macd_signal_prev:
        score += 4
    elif macd > macd_signal:
        score += 1
    elif macd < macd_signal and macd_prev >= macd_signal_prev:
        score -= 4
    elif macd < macd_signal:
        score -= 1
    
    # RSI (±3 points scaled by strength)
    if rsi < rsi_oversold:
        strength = (rsi_oversold - rsi) / rsi_oversold
        score += 3 * strength
    elif rsi > rsi_overbought:
        strength = (rsi - rsi_overbought) / (100 - rsi_overbought)
        score -= 3 * strength
    
    # ADX trend strength (±2 points)
    if adx > 25:
        if close > ema200:
            score += 2
        else:
            score -= 2
    
    # Bollinger Band position (±2 points)
    if bb_position < 0.2:
        score += 2
    elif bb_position > 0.8:
        score -= 2
    
    # Volume confirmation (±1 point)
    if volume_ratio > 1.5:
        if score > 0:
            score += 1
        else:
            score -= 1
    
    # Normalize to 0-100
    normalized = np.clip(50 + (score * 5), 0, 100)
    return round(normalized, 1)

# ============================================================================
# INTRADAY INDICATORS (5MIN/15MIN TIMEFRAME)
# ============================================================================

def calculate_intraday_indicators(data, timeframe='15m'):
    """
    Calculate intraday indicators:
    - EMA crossover (9/21 or 20/50 based on timeframe)
    - VWAP with bands
    - Fast RSI (7-9)
    - Tight ATR (10)
    - Volume spike
    - Opening Range
    - Fast MACD (5/13/5 or 12/26/9)
    - ADX (>20 filter)
    """
    if len(data) < 200:
        return data
    
    df = data.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # EMA crossover
    if timeframe == '5m':
        fast, slow = 9, 21
        rsi_period = 7
        macd_fast, macd_slow, macd_sign = 5, 13, 5
    else:  # 15m, 30m
        fast, slow = 20, 50
        rsi_period = 9
        macd_fast, macd_slow, macd_sign = 12, 26, 9
    
    df['EMA_Fast'] = ta.trend.EMAIndicator(df['Close'], window=fast).ema_indicator()
    df['EMA_Slow'] = ta.trend.EMAIndicator(df['Close'], window=slow).ema_indicator()
    df['EMA_Bullish'] = df['EMA_Fast'] > df['EMA_Slow']
    df['EMA_Crossover'] = (df['EMA_Bullish'] != df['EMA_Bullish'].shift(1)) & df['EMA_Bullish']
    df['EMA_Crossunder'] = (df['EMA_Bullish'] != df['EMA_Bullish'].shift(1)) & ~df['EMA_Bullish']
    
    # VWAP
    df['Date'] = df.index.date
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    df['VWAP'] = df.groupby('Date').apply(
        lambda x: (x['Typical_Price'] * x['Volume']).cumsum() / x['Volume'].cumsum()
    ).reset_index(level=0, drop=True)
    
    df['VWAP_Std'] = df.groupby('Date')['Typical_Price'].transform(lambda x: x.expanding().std())
    df['VWAP_Upper1'] = df['VWAP'] + df['VWAP_Std']
    df['VWAP_Upper2'] = df['VWAP'] + (df['VWAP_Std'] * 2)
    df['VWAP_Lower1'] = df['VWAP'] - df['VWAP_Std']
    df['VWAP_Lower2'] = df['VWAP'] - (df['VWAP_Std'] * 2)
    df['Above_VWAP'] = df['Close'] > df['VWAP']
    
    # Fast RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=rsi_period).rsi()
    
    # Tight ATR
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=10).average_true_range()
    df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
    
    # Volume
    df['Avg_Volume'] = df['Volume'].rolling(20).mean()
    df['RVOL'] = df['Volume'] / df['Avg_Volume']
    df['Volume_Spike'] = df['RVOL'] > 1.5
    
    # Opening Range (first 30 min)
    df['Time'] = df.index.time
    df['Is_OR'] = df['Time'] <= time(9, 45)
    df['OR_High'] = df[df['Is_OR']].groupby('Date')['High'].transform('max')
    df['OR_Low'] = df[df['Is_OR']].groupby('Date')['Low'].transform('min')
    df['OR_High'] = df.groupby('Date')['OR_High'].ffill()
    df['OR_Low'] = df.groupby('Date')['OR_Low'].ffill()
    df['OR_Breakout_Long'] = (df['Close'] > df['OR_High']) & (df['Close'].shift(1) <= df['OR_High'])
    df['OR_Breakout_Short'] = (df['Close'] < df['OR_Low']) & (df['Close'].shift(1) >= df['OR_Low'])
    
    # Fast MACD
    macd = ta.trend.MACD(df['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_sign)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    df['MACD_Bullish'] = df['MACD'] > df['MACD_Signal']
    
    # ADX
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    df['Trending_Intraday'] = df['ADX'] > 20
    
    # Time filters
    df['Safe_Hours'] = (df['Time'] >= time(9, 30)) & (df['Time'] <= time(15, 15))
    df['Prime_Hours'] = (
        ((df['Time'] >= time(9, 30)) & (df['Time'] <= time(11, 30))) |
        ((df['Time'] >= time(14, 0)) & (df['Time'] <= time(15, 15)))
    )
    
    df.drop(columns=['Date', 'Typical_Price', 'Time', 'Is_OR'], inplace=True, errors='ignore')
    
    return df

def detect_intraday_regime(df):
    """Classify intraday regime"""
    if len(df) < 50:
        return "Unknown"
    
    current_time = df.index[-1].time()
    
    if current_time < time(9, 15):
        return "Pre-Market"
    elif current_time <= time(9, 45):
        return "Opening Range"
    elif current_time > time(15, 15):
        return "Closing Session"
    
    close = df['Close'].iloc[-1]
    vwap = df['VWAP'].iloc[-1]
    ema_bullish = df['EMA_Bullish'].iloc[-1]
    macd_bullish = df['MACD_Bullish'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    
    if pd.isna(vwap) or pd.isna(adx):
        return "Unknown"
    
    above_vwap = close > vwap
    trending = adx > 20
    
    if above_vwap and ema_bullish and macd_bullish and trending:
        return "Strong Uptrend"
    elif above_vwap and ema_bullish:
        return "Weak Uptrend"
    elif not above_vwap and not ema_bullish and not macd_bullish and trending:
        return "Strong Downtrend"
    elif not above_vwap and not ema_bullish:
        return "Weak Downtrend"
    elif not trending:
        return "Choppy"
    else:
        return "Neutral"

def calculate_intraday_score(df):
    """Calculate intraday score (0-100)"""
    regime = detect_intraday_regime(df)
    
    if regime in ["Pre-Market", "Closing Session", "Unknown"]:
        return 50  # Neutral
    
    score = 0
    
    close = df['Close'].iloc[-1]
    vwap = df['VWAP'].iloc[-1]
    ema_bullish = df['EMA_Bullish'].iloc[-1]
    ema_crossover = df['EMA_Crossover'].iloc[-1]
    ema_crossunder = df['EMA_Crossunder'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd_bullish = df['MACD_Bullish'].iloc[-1]
    rvol = df['RVOL'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    or_breakout_long = df['OR_Breakout_Long'].iloc[-1]
    or_breakout_short = df['OR_Breakout_Short'].iloc[-1]
    
    # VWAP position (±3)
    if close > vwap:
        score += 3
    else:
        score -= 3
    
    # EMA crossover (±4 fresh, ±1 sustained)
    if ema_crossover:
        score += 4
    elif ema_bullish:
        score += 1
    elif ema_crossunder:
        score -= 4
    elif not ema_bullish:
        score -= 1
    
    # RSI (±3)
    if rsi < 30:
        score += 3 * (30 - rsi) / 30
    elif rsi > 70:
        score -= 3 * (rsi - 70) / 30
    
    # MACD (±1)
    if macd_bullish:
        score += 1
    else:
        score -= 1
    
    # Volume (±2)
    if rvol > 2.0:
        score += 2 if score > 0 else -2
    elif rvol > 1.5:
        score += 1 if score > 0 else -1
    
    # ADX (±2)
    if adx > 25:
        score += 2 if score > 0 else -2
    elif adx > 20:
        score += 1 if score > 0 else -1
    
    # OR breakout (±4)
    if or_breakout_long:
        score += 4
    elif or_breakout_short:
        score -= 4
    
    normalized = np.clip(50 + (score * 3.33), 0, 100)
    return round(normalized, 1)

# ============================================================================
# POSITION SIZING & RISK MANAGEMENT
# ============================================================================

def calculate_swing_position(df, account_size=30000, risk_pct=0.02):
    """Calculate swing position parameters"""
    close = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    ema200 = df['EMA_200'].iloc[-1]
    bb_lower = df['BB_Lower'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    regime = detect_swing_regime(df)
    
    buy_at = round(close * 1.001, 2)
    
    # Stop loss
    if regime in ["Strong Uptrend", "Weak Uptrend"]:
        atr_stop = close - (2 * atr)
        ema_stop = ema200 * 0.98
        stop_loss = max(atr_stop, ema_stop)
    elif regime in ["Consolidation (Above EMA)", "Consolidation (Below EMA)"]:
        stop_loss = bb_lower * 0.98
    else:
        stop_loss = close - (1.5 * atr)
    
    stop_loss = max(stop_loss, close * 0.92)  # Max 8% loss
    stop_loss = round(stop_loss, 2)
    
    # Target
    risk = buy_at - stop_loss
    if regime == "Strong Uptrend":
        rr_ratio = 3
    elif regime in ["Weak Uptrend", "Consolidation (Above EMA)"]:
        rr_ratio = 2
    else:
        rr_ratio = 1.5
    
    target = buy_at + (risk * rr_ratio)
    target = round(target, 2)
    
    # Position size
    risk_amount = account_size * risk_pct
    position_size = int(risk_amount / risk) if risk > 0 else 0
    max_position = int((account_size * 0.1) / buy_at)
    position_size = min(position_size, max_position)
    
    # Trailing stop
    trailing_stop = close - (2.5 * atr) if adx > 30 else close - (1.5 * atr)
    trailing_stop = round(trailing_stop, 2)
    
    return {
        "current_price": round(close, 2),
        "buy_at": buy_at,
        "stop_loss": stop_loss,
        "target": target,
        "position_size": position_size,
        "trailing_stop": trailing_stop,
        "rr_ratio": round((target - buy_at) / risk, 2) if risk > 0 else 0,
        "risk_amount": round(position_size * risk, 2),
        "potential_profit": round(position_size * (target - buy_at), 2)
    }

def calculate_intraday_position(df, account_size=30000, risk_pct=0.01):
    """Calculate intraday position parameters"""
    close = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    vwap = df['VWAP'].iloc[-1]
    or_high = df['OR_High'].iloc[-1]
    or_low = df['OR_Low'].iloc[-1]
    regime = detect_intraday_regime(df)
    
    buy_at = round(close, 2)
    
    # Tighter stops for intraday
    if regime in ["Strong Uptrend", "Weak Uptrend"]:
        stop_loss = max(close - (1.5 * atr), vwap - (0.5 * atr))
    elif regime == "Opening Range":
        stop_loss = or_low - (0.5 * atr)
    else:
        stop_loss = close - (1.0 * atr)
    
    stop_loss = max(stop_loss, close * 0.97)  # Max 3% loss
    stop_loss = round(stop_loss, 2)
    
    # Target
    risk = buy_at - stop_loss
    if regime == "Strong Uptrend":
        rr_ratio = 2
    elif regime == "Opening Range":
        rr_ratio = 1.5
    else:
        rr_ratio = 1
    
    target = buy_at + (risk * rr_ratio)
    target = round(target, 2)
    
    # Smaller position for intraday
    risk_amount = account_size * risk_pct
    position_size = int(risk_amount / risk) if risk > 0 else 0
    max_position = int((account_size * 0.05) / buy_at)
    position_size = min(position_size, max_position)
    
    trailing_stop = close - (1.0 * atr)
    trailing_stop = round(trailing_stop, 2)
    
    current_time = df.index[-1].time()
    hours_to_close = (time(15, 15).hour * 60 + time(15, 15).minute - 
                     current_time.hour * 60 - current_time.minute) / 60
    
    return {
        "current_price": round(close, 2),
        "buy_at": buy_at,
        "stop_loss": stop_loss,
        "target": target,
        "position_size": position_size,
        "trailing_stop": trailing_stop,
        "rr_ratio": round((target - buy_at) / risk, 2) if risk > 0 else 0,
        "risk_amount": round(position_size * risk, 2),
        "potential_profit": round(position_size * (target - buy_at), 2),
        "vwap": round(vwap, 2),
        "or_high": round(or_high, 2) if pd.notna(or_high) else None,
        "or_low": round(or_low, 2) if pd.notna(or_low) else None,
        "hours_to_close": round(hours_to_close, 2)
    }

# ============================================================================
# UNIFIED RECOMMENDATION GENERATION
# ============================================================================

def generate_recommendation(data, symbol, trading_style='swing', timeframe='1d', account_size=30000):
    """Generate unified recommendations for swing or intraday"""
    
    if trading_style == 'swing':
        df = calculate_swing_indicators(data)
        regime = detect_swing_regime(df)
        score = calculate_swing_score(df)
        position = calculate_swing_position(df, account_size)
        
    else:  # intraday
        df = calculate_intraday_indicators(data, timeframe)
        regime = detect_intraday_regime(df)
        score = calculate_intraday_score(df)
        position = calculate_intraday_position(df, account_size)
    
    # Signal generation
    if score >= 75:
        signal = "Strong Buy"
    elif score >= 60:
        signal = "Buy"
    elif score <= 25:
        signal = "Strong Sell"
    elif score <= 40:
        signal = "Sell"
    else:
        signal = "Hold"
    
    # Build reason
    reasons = []
    close = df['Close'].iloc[-1]
    
    if trading_style == 'swing':
        ema200 = df['EMA_200'].iloc[-1]
        macd_bullish = df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        adx = df['ADX'].iloc[-1]
        
        reasons.append("Above 200 EMA" if close > ema200 else "Below 200 EMA")
        reasons.append("MACD bullish" if macd_bullish else "MACD bearish")
        if rsi < df['RSI_Oversold'].iloc[-1]:
            reasons.append("RSI oversold")
        elif rsi > df['RSI_Overbought'].iloc[-1]:
            reasons.append("RSI overbought")
        reasons.append(f"ADX {adx:.1f}")
        if df['Volume_Spike'].iloc[-1]:
            reasons.append("Volume spike")
    
    else:  # intraday
        vwap = df['VWAP'].iloc[-1]
        ema_bullish = df['EMA_Bullish'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        
        reasons.append("Above VWAP" if close > vwap else "Below VWAP")
        reasons.append("EMA bullish" if ema_bullish else "EMA bearish")
        if df['EMA_Crossover'].iloc[-1]:
            reasons.append("Fresh EMA cross")
        if rsi < 30:
            reasons.append("RSI oversold")
        elif rsi > 70:
            reasons.append("RSI overbought")
        if df['OR_Breakout_Long'].iloc[-1]:
            reasons.append("OR breakout")
    
    return {
        "symbol": symbol,
        "trading_style": trading_style.capitalize(),
        "timeframe": timeframe,
        "score": score,
        "signal": signal,
        "regime": regime,
        "reason": ", ".join(reasons),
        **position
    }

# ============================================================================
# BACKTESTING
# ============================================================================

def backtest_strategy(data, symbol, trading_style='swing', timeframe='1d', initial_capital=30000):
    """Backtest strategy with realistic execution"""
    
    results = {
        "total_return": 0,
        "annual_return": 0,
        "sharpe_ratio": 0,
        "max_drawdown": 0,
        "trades": 0,
        "win_rate": 0,
        "trades_list": [],
        "equity_curve": []
    }
    
    if len(data) < 200:
        return results
    
    cash = initial_capital
    position = None
    entry_price = 0
    entry_date = None
    qty = 0
    trades = []
    returns = []
    
    for i in range(200, len(data)):
        sliced = data.iloc[:i+1]
        
        try:
            rec = generate_recommendation(sliced, symbol, trading_style, timeframe, cash)
            current_price = data['Close'].iloc[i]
            current_date = data.index[i]
            
            # Sell logic
            if position and rec['signal'] in ['Sell', 'Strong Sell']:
                pnl = (current_price - entry_price) * qty
                cash += pnl
                returns.append(pnl / (entry_price * qty))
                
                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": current_date,
                    "exit_price": current_price,
                    "pnl": pnl
                })
                
                position = None
                qty = 0
            
            # Buy logic
            if not position and rec['signal'] in ['Buy', 'Strong Buy']:
                entry_price = current_price
                entry_date = current_date
                qty = rec['position_size']
                cash -= qty * entry_price
                position = "Long"
            
            equity = cash + (qty * current_price if position else 0)
            results['equity_curve'].append((current_date, equity))
            
        except Exception as e:
            continue
    
    # Close final position
    if position:
        current_price = data['Close'].iloc[-1]
        pnl = (current_price - entry_price) * qty
        cash += pnl
        returns.append(pnl / (entry_price * qty))
        trades.append({
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": data.index[-1],
            "exit_price": current_price,
            "pnl": pnl
        })
    
    # Calculate metrics
    if trades:
        results['trades'] = len(trades)
        results['trades_list'] = trades
        results['total_return'] = ((cash - initial_capital) / initial_capital) * 100
        results['win_rate'] = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100
        
        if returns:
            results['sharpe_ratio'] = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252)
            results['annual_return'] = np.mean(returns) * 252 * 100
    
    # Max drawdown
    if results['equity_curve']:
        equity_df = pd.DataFrame(results['equity_curve'], columns=['Date', 'Equity'])
        equity_df['Peak'] = equity_df['Equity'].cummax()
        equity_df['DD'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
        results['max_drawdown'] = equity_df['DD'].min() * 100
    
    return results

# ============================================================================
# BATCH ANALYSIS
# ============================================================================

def analyze_stock_batch(symbol, trading_style='swing', timeframe='1d'):
    """Analyze single stock for batch processing"""
    try:
        data = fetch_stock_data_cached(symbol, interval=timeframe)
        
        if data.empty or len(data) < 200:
            return None
        
        rec = generate_recommendation(data, symbol, trading_style, timeframe)
        
        return {
            "Symbol": rec['symbol'],
            "Score": rec['score'],
            "Signal": rec['signal'],
            "Regime": rec['regime'],
            "Current Price": rec['current_price'],
            "Buy At": rec['buy_at'],
            "Stop Loss": rec['stop_loss'],
            "Target": rec['target'],
            "R:R": rec['rr_ratio'],
            "Position Size": rec['position_size'],
            "Reason": rec['reason']
        }
    
    except Exception as e:
        logging.error(f"Error analyzing {symbol}: {str(e)}")
        return None

def analyze_multiple_stocks(stock_list, trading_style='swing', timeframe='1d', progress_callback=None):
    """Analyze multiple stocks sequentially"""
    results = []
    
    for i, symbol in enumerate(stock_list):
        result = analyze_stock_batch(symbol, trading_style, timeframe)
        if result:
            results.append(result)
        
        if progress_callback:
            progress_callback((i + 1) / len(stock_list))
        
        time_module.sleep(3)  # Rate limiting
    
    df = pd.DataFrame(results)
    if df.empty:
        return df
    
    # Filter by signal
    if trading_style == 'intraday':
        df = df[df['Signal'].str.contains('Buy', na=False)]
    
    return df.sort_values('Score', ascending=False).head(10)

# ============================================================================
# DATABASE
# ============================================================================

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('stock_picks.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS picks (
            date TEXT,
            symbol TEXT,
            trading_style TEXT,
            score REAL,
            signal TEXT,
            regime TEXT,
            current_price REAL,
            buy_at REAL,
            stop_loss REAL,
            target REAL,
            reason TEXT,
            PRIMARY KEY (date, symbol, trading_style)
        )
    ''')
    conn.close()

def save_picks(results_df, trading_style):
    """Save top picks to database"""
    conn = sqlite3.connect('stock_picks.db')
    today = datetime.now().strftime('%Y-%m-%d')
    
    for _, row in results_df.head(5).iterrows():
        conn.execute('''
            INSERT OR REPLACE INTO picks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            today,
            row.get('Symbol'),
            trading_style,
            row.get('Score'),
            row.get('Signal'),
            row.get('Regime'),
            row.get('Current Price'),
            row.get('Buy At'),
            row.get('Stop Loss'),
            row.get('Target'),
            row.get('Reason')
        ))
    
    conn.commit()
    conn.close()

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main Streamlit application"""
    
    init_database()
    
    st.set_page_config(page_title="StockGenie Pro", layout="wide")
    st.title("📊 StockGenie Pro - NSE Analysis")
    st.subheader(f"📅 {datetime.now().strftime('%d %b %Y, %A')}")
    
    # Sidebar
    st.sidebar.title("🔍 Configuration")
    
    # Trading style
    trading_style = st.sidebar.radio(
        "Trading Style",
        ["Swing Trading", "Intraday Trading"],
        help="Swing: Hold days-weeks. Intraday: Close same day"
    )
    
    # Timeframe
    if trading_style == "Intraday Trading":
        timeframe_display = st.sidebar.selectbox("Timeframe", ["5 min", "15 min", "30 min"], index=1)
        timeframe = timeframe_display.replace(" ", "").lower()[:-2] + "m"
    else:
        timeframe_display = "Daily"
        timeframe = "1d"
    
    # Sector selection
    sector_options = ["All"] + list(SECTORS.keys())
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        sector_options,
        default=["Bank", "IT"],
        help="Choose sectors to analyze"
    )
    
    # Stock selection
    if "All" in selected_sectors:
        stock_list = [s for sector in SECTORS.values() for s in sector]
    else:
        stock_list = [s for sector in selected_sectors for s in SECTORS.get(sector, [])]
    
    symbol = st.sidebar.selectbox("Select Stock", stock_list, index=0)
    
    # Account size
    account_size = st.sidebar.number_input("Account Size (₹)", min_value=10000, max_value=10000000, value=30000, step=5000)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analysis", "🔍 Scanner", "📊 Backtest", "📜 History"])
    
    # TAB 1: Single Stock Analysis
    with tab1:
        if st.button("🔍 Analyze Selected Stock"):
            with st.spinner(f"Analyzing {symbol}..."):
                data = fetch_stock_data_with_auth(symbol, interval=timeframe)
                
                if not data.empty:
                    rec = generate_recommendation(
                        data, symbol,
                        'swing' if trading_style == "Swing Trading" else 'intraday',
                        timeframe, account_size
                    )
                    
                    # Display metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Score", f"{rec['score']}/100")
                    col2.metric("Signal", rec['signal'])
                    col3.metric("Regime", rec['regime'])
                    col4.metric("Current Price", f"₹{rec['current_price']}")
                    
                    if trading_style == "Intraday Trading":
                        col5.metric("Hours to Close", f"{rec['hours_to_close']}h")
                    else:
                        col5.metric("Timeframe", timeframe_display)
                    
                    # Trade setup
                    st.subheader("📋 Trade Setup")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Buy At**: ₹{rec['buy_at']}")
                        st.write(f"**Position Size**: {rec['position_size']} shares")
                    with col2:
                        st.write(f"**Stop Loss**: ₹{rec['stop_loss']}")
                        st.write(f"**Risk Amount**: ₹{rec['risk_amount']}")
                    with col3:
                        st.write(f"**Target**: ₹{rec['target']}")
                        st.write(f"**Potential Profit**: ₹{rec['potential_profit']}")
                    
                    st.write(f"**R:R Ratio**: {rec['rr_ratio']}:1")
                    st.write(f"**Trailing Stop**: ₹{rec['trailing_stop']}")
                    
                    # Reason
                    st.info(f"**Reason**: {rec['reason']}")
                    
                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='Price'
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} - {timeframe_display}",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available")
    
    # TAB 2: Scanner
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Scan Top Swing Picks"):
                progress = st.progress(0)
                status = st.empty()
                
                results = analyze_multiple_stocks(
                    stock_list[:20],  # Limit to 20 for speed
                    'swing',
                    '1d',
                    lambda p: progress.progress(p)
                )
                
                progress.empty()
                status.empty()
                
                if not results.empty:
                    save_picks(results, 'swing')
                    st.subheader("🏆 Top Swing Trading Picks")
                    st.dataframe(results, use_container_width=True)
                else:
                    st.warning("No picks found")
        
        with col2:
            if st.button("⚡ Scan Top Intraday Picks"):
                progress = st.progress(0)
                status = st.empty()
                
                results = analyze_multiple_stocks(
                    stock_list[:20],
                    'intraday',
                    timeframe,
                    lambda p: progress.progress(p)
                )
                
                progress.empty()
                status.empty()
                
                if not results.empty:
                    save_picks(results, 'intraday')
                    st.subheader("🏆 Top Intraday Picks")
                    st.dataframe(results, use_container_width=True)
                else:
                    st.warning("No picks found")
    
    # TAB 3: Backtest
    with tab3:
        if st.button("📊 Run Backtest"):
            with st.spinner("Running backtest..."):
                data = fetch_stock_data_with_auth(symbol, period="2y", interval=timeframe)
                
                if not data.empty:
                    results = backtest_strategy(
                        data, symbol,
                        'swing' if trading_style == "Swing Trading" else 'intraday',
                        timeframe, account_size
                    )
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Return", f"{results['total_return']:.2f}%")
                    col2.metric("Annual Return", f"{results['annual_return']:.2f}%")
                    col3.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                    col4.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Total Trades", results['trades'])
                    col2.metric("Win Rate", f"{results['win_rate']:.2f}%")
                    
                    # Trades table
                    if results['trades_list']:
                        st.subheader("Trade History")
                        trades_df = pd.DataFrame(results['trades_list'])
                        st.dataframe(trades_df, use_container_width=True)
                    
                    # Equity curve
                    if results['equity_curve']:
                        st.subheader("Equity Curve")
                        equity_df = pd.DataFrame(results['equity_curve'], columns=['Date', 'Equity'])
                        fig = px.line(equity_df, x='Date', y='Equity', title='Portfolio Value Over Time')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Insufficient data for backtest")
    
    # TAB 4: History
    with tab4:
        conn = sqlite3.connect('stock_picks.db')
        history = pd.read_sql_query("SELECT * FROM picks ORDER BY date DESC LIMIT 100", conn)
        conn.close()
        
        if not history.empty:
            st.subheader("📜 Historical Picks")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                date_filter = st.selectbox("Date", ["All"] + list(history['date'].unique()))
            with col2:
                style_filter = st.selectbox("Style", ["All", "Swing Trading", "Intraday Trading"])
            
            filtered = history.copy()
            if date_filter != "All":
                filtered = filtered[filtered['date'] == date_filter]
            if style_filter != "All":
                filtered = filtered[filtered['trading_style'] == style_filter]
            
            st.dataframe(filtered, use_container_width=True)
        else:
            st.info("No historical data available")

if __name__ == "__main__":
    main()
