# ============================================================================
# STOCKGENIE PRO - PRODUCTION VERSION V2.3 (BALANCED SCORING)
# Enhanced Swing + Intraday Trading System
# FIXED: Proper score distribution across sectors
# ============================================================================

import pandas as pd
import numpy as np
import ta
import logging
import streamlit as st
from datetime import datetime, timedelta, time
from functools import wraps
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
SESSION_EXPIRY = 3600

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
        "INDUSINDBK-EQ", "PNB-EQ", "BANKBARODA-EQ", "CANBK-EQ", "UNIONBANK-EQ",
        "FEDERALBNK-EQ", "IDFCFIRSTB-EQ", "BANDHANBNK-EQ", "AUBANK-EQ", "RBLBANK-EQ"
    ],
    "IT": [
        "TCS-EQ", "INFY-EQ", "HCLTECH-EQ", "WIPRO-EQ", "TECHM-EQ", "LTIM-EQ",
        "MPHASIS-EQ", "COFORGE-EQ", "PERSISTENT-EQ", "TATAELXSI-EQ"
    ],
    "Auto": [
        "MARUTI-EQ", "TATAMOTORS-EQ", "M&M-EQ", "BAJAJ-AUTO-EQ", "HEROMOTOCO-EQ",
        "EICHERMOT-EQ", "TVSMOTOR-EQ", "ASHOKLEY-EQ", "ESCORTS-EQ", "BALKRISIND-EQ"
    ],
    "Pharma": [
        "SUNPHARMA-EQ", "CIPLA-EQ", "DRREDDY-EQ", "DIVISLAB-EQ", "AUROPHARMA-EQ",
        "LUPIN-EQ", "TORNTPHARM-EQ", "ALKEM-EQ", "BIOCON-EQ", "APOLLOHOSP-EQ"
    ],
    "FMCG": [
        "HINDUNILVR-EQ", "ITC-EQ", "NESTLEIND-EQ", "BRITANNIA-EQ", "DABUR-EQ",
        "MARICO-EQ", "GODREJCP-EQ", "TATACONSUM-EQ", "COLPAL-EQ", "PGHH-EQ"
    ],
    "Energy": [
        "RELIANCE-EQ", "ONGC-EQ", "IOC-EQ", "BPCL-EQ", "HPCL-EQ", "GAIL-EQ",
        "COALINDIA-EQ", "NTPC-EQ", "POWERGRID-EQ", "ADANIPOWER-EQ"
    ],
    "Metals": [
        "TATASTEEL-EQ", "JSWSTEEL-EQ", "HINDALCO-EQ", "VEDL-EQ", "SAIL-EQ",
        "NMDC-EQ", "HINDZINC-EQ", "JINDALSTEL-EQ", "NATIONALUM-EQ", "MOIL-EQ"
    ],
    "Cement": [
        "ULTRACEMCO-EQ", "SHREECEM-EQ", "AMBUJACEM-EQ", "ACC-EQ", "JKCEMENT-EQ",
        "DALBHARAT-EQ", "RAMCOCEM-EQ", "NUVOCO-EQ"
    ]
}

# Industry mapping for stocks
INDUSTRY_MAP = {
    "Financial Services": ["HDFCBANK-EQ", "ICICIBANK-EQ", "SBIN-EQ", "KOTAKBANK-EQ", "AXISBANK-EQ", 
                          "INDUSINDBK-EQ", "PNB-EQ", "BANKBARODA-EQ", "CANBK-EQ", "UNIONBANK-EQ"],
    "Information Technology": ["TCS-EQ", "INFY-EQ", "HCLTECH-EQ", "WIPRO-EQ", "TECHM-EQ", "LTIM-EQ"],
    "Automobile and Auto Components": ["MARUTI-EQ", "TATAMOTORS-EQ", "M&M-EQ", "BAJAJ-AUTO-EQ", "HEROMOTOCO-EQ"],
    "Healthcare": ["SUNPHARMA-EQ", "CIPLA-EQ", "DRREDDY-EQ", "DIVISLAB-EQ", "AUROPHARMA-EQ", "APOLLOHOSP-EQ"],
    "Fast Moving Consumer Goods": ["HINDUNILVR-EQ", "ITC-EQ", "NESTLEIND-EQ", "BRITANNIA-EQ", "DABUR-EQ"],
    "Oil Gas & Consumable Fuels": ["RELIANCE-EQ", "ONGC-EQ", "IOC-EQ", "BPCL-EQ", "HPCL-EQ"],
    "Metals & Mining": ["TATASTEEL-EQ", "JSWSTEEL-EQ", "HINDALCO-EQ", "VEDL-EQ", "SAIL-EQ"],
    "Construction Materials": ["ULTRACEMCO-EQ", "SHREECEM-EQ", "AMBUJACEM-EQ", "ACC-EQ"]
}

# Tooltips
TOOLTIPS = {
    "Score": "Signal strength (0-100). 50=neutral, 65+=buy zone, 35-=sell zone",
    "RSI": "Momentum indicator (30=oversold, 70=overbought)",
    "MACD": "Trend indicator - crossovers signal trend changes",
    "ATR": "Volatility measure for stop-loss placement",
    "ADX": "Trend strength (>25 = strong, <20 = weak/choppy)",
    "VWAP": "Volume-weighted price - intraday benchmark",
    "EMA": "Exponential Moving Average - trend filter",
    "OR": "Opening Range - first 15-30min high/low levels",
    "Breadth": "Market internals - advancing vs declining stocks"
}

# ============================================================================
# MARKET BREADTH & SECTOR PERFORMANCE INTEGRATION
# ============================================================================

@st.cache_data(ttl=900)
def fetch_market_breadth():
    """Fetch overall market breadth data"""
    try:
        response = requests.get(
            "https://brkpoint.in/api/market-stats",
            timeout=10,
            headers={"User-Agent": random.choice(USER_AGENTS)}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.warning(f"Market breadth API failed: {str(e)}")
        return None

@st.cache_data(ttl=900)
def fetch_sector_performance():
    """Fetch sector indices performance"""
    try:
        response = requests.get(
            "https://brkpoint.in/api/sector-indices-performance",
            timeout=10,
            headers={"User-Agent": random.choice(USER_AGENTS)}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.warning(f"Sector performance API failed: {str(e)}")
        return None

def calculate_market_health_score():
    """Calculate overall market health score (0-100)"""
    breadth_data = fetch_market_breadth()
    sector_data = fetch_sector_performance()
    
    if not breadth_data or not sector_data:
        return 50, "Unknown", {}
    
    score = 0
    factors = {}
    
    # Market Breadth (40 points max)
    breadth = breadth_data.get('breadth', {})
    total = breadth.get('total', 1)
    advancing = breadth.get('advancing', 0)
    declining = breadth.get('declining', 0)
    
    adv_ratio = (advancing / total) * 100 if total > 0 else 50
    factors['advance_ratio'] = adv_ratio
    
    if adv_ratio > 70:
        score += 40
        breadth_signal = "Very Strong"
    elif adv_ratio > 60:
        score += 30
        breadth_signal = "Strong"
    elif adv_ratio > 50:
        score += 20
        breadth_signal = "Moderate"
    elif adv_ratio > 40:
        score += 10
        breadth_signal = "Weak"
    else:
        score += 0
        breadth_signal = "Very Weak"
    
    factors['breadth_signal'] = breadth_signal
    
    # Sector Indices Momentum (30 points max)
    sectors = sector_data.get('data', [])
    
    nifty50 = next((s for s in sectors if s['sector_index'] == 'Nifty50'), None)
    nifty500 = next((s for s in sectors if s['sector_index'] == 'Nifty500'), None)
    
    avg_momentum = 0
    count = 0
    
    if nifty50:
        avg_momentum += nifty50.get('momentum', 0)
        count += 1
    
    if nifty500:
        avg_momentum += nifty500.get('momentum', 0)
        count += 1
    
    avg_momentum = avg_momentum / count if count > 0 else 0
    factors['avg_momentum'] = avg_momentum
    
    if avg_momentum > 25:
        score += 30
        momentum_signal = "Very Strong"
    elif avg_momentum > 15:
        score += 20
        momentum_signal = "Strong"
    elif avg_momentum > 5:
        score += 10
        momentum_signal = "Moderate"
    elif avg_momentum > -5:
        score += 5
        momentum_signal = "Weak"
    else:
        score += 0
        momentum_signal = "Very Weak"
    
    factors['momentum_signal'] = momentum_signal
    
    # Volatility & Risk (30 points max)
    if nifty50:
        volatility = nifty50.get('volatility_score', 0)
        factors['volatility'] = volatility
        
        if volatility < 5:
            score += 30
            vol_signal = "Very Low (Ideal)"
        elif volatility < 10:
            score += 20
            vol_signal = "Low (Good)"
        elif volatility < 15:
            score += 10
            vol_signal = "Moderate"
        elif volatility < 20:
            score += 5
            vol_signal = "High"
        else:
            score += 0
            vol_signal = "Very High (Risky)"
        
        factors['volatility_signal'] = vol_signal
    
    # Overall signal
    if score >= 80:
        overall_signal = "Very Bullish"
    elif score >= 60:
        overall_signal = "Bullish"
    elif score >= 40:
        overall_signal = "Neutral"
    elif score >= 20:
        overall_signal = "Bearish"
    else:
        overall_signal = "Very Bearish"
    
    return score, overall_signal, factors

def get_industry_performance(symbol):
    """Get industry performance for a specific stock"""
    breadth_data = fetch_market_breadth()
    
    if not breadth_data:
        return None
    
    stock_industry = None
    for industry, stocks in INDUSTRY_MAP.items():
        if symbol in stocks:
            stock_industry = industry
            break
    
    if not stock_industry:
        return None
    
    industries = breadth_data.get('industry', [])
    industry_data = next((ind for ind in industries if ind['Industry'] == stock_industry), None)
    
    return industry_data

def calculate_industry_alignment_score(industry_data, signal_direction):
    """Calculate bonus/penalty based on industry performance (±3 points)"""
    if not industry_data:
        return 0
    
    avg_change = industry_data.get('avgChange', 0)
    advance_ratio = (industry_data.get('advancing', 0) / industry_data.get('total', 1)) * 100
    
    score_adjustment = 0
    
    if signal_direction == 'bullish':
        if avg_change > 1.5 and advance_ratio > 70:
            score_adjustment += 3
        elif avg_change > 1.0 and advance_ratio > 60:
            score_adjustment += 2
        elif avg_change > 0.5 and advance_ratio > 50:
            score_adjustment += 1
        elif avg_change < 0 or advance_ratio < 40:
            score_adjustment -= 2
    
    elif signal_direction == 'bearish':
        if avg_change < -1.0 and advance_ratio < 30:
            score_adjustment += 3
        elif avg_change < -0.5 and advance_ratio < 40:
            score_adjustment += 2
        elif avg_change > 1.0 or advance_ratio > 60:
            score_adjustment -= 2
    
    return score_adjustment

def calculate_market_breadth_alignment(signal_direction):
    """Calculate alignment with overall market breadth (±5 points - REDUCED)"""
    market_health, market_signal, factors = calculate_market_health_score()
    
    score_adjustment = 0
    
    if signal_direction == 'bullish':
        if market_health >= 80:
            score_adjustment += 5
        elif market_health >= 60:
            score_adjustment += 3
        elif market_health <= 30:
            score_adjustment -= 5
        elif market_health <= 40:
            score_adjustment -= 2
    
    elif signal_direction == 'bearish':
        if market_health <= 20:
            score_adjustment += 5
        elif market_health <= 40:
            score_adjustment += 3
        elif market_health >= 70:
            score_adjustment -= 5
        elif market_health >= 60:
            score_adjustment -= 2
    
    return score_adjustment

# ============================================================================
# INDEX TREND INTEGRATION
# ============================================================================

@st.cache_data(ttl=900)
def fetch_index_trend():
    """Fetch Nifty & Bank Nifty trend from external API"""
    try:
        response = requests.get(
            "https://brkpoint.in/api/indextrend",
            timeout=10,
            headers={"User-Agent": random.choice(USER_AGENTS)}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.warning(f"Index trend API failed: {str(e)}")
        return None

def get_index_trend_for_timeframe(timeframe='15m'):
    """Get relevant index trend based on trading timeframe"""
    data = fetch_index_trend()
    if not data:
        return None
    
    if timeframe in ['5m', '15m', '30m']:
        nifty_key = 'nif_min15trend'
        bnf_key = 'bnf_min15trend'
    elif timeframe in ['1h']:
        nifty_key = 'nif_hr1trend'
        bnf_key = 'bnf_hr1trend'
    else:
        nifty_key = 'nif_day1trend'
        bnf_key = 'bnf_day1trend'
    
    return {
        'nifty': data.get(nifty_key, {}),
        'banknifty': data.get(bnf_key, {})
    }

def get_relevant_index(symbol):
    """Determine which index is relevant for the stock"""
    bank_stocks = [s for s in SECTORS.get('Bank', [])]
    
    if symbol in bank_stocks:
        return 'banknifty'
    else:
        return 'nifty'

def calculate_index_alignment_score(trend_data, signal_direction):
    """Calculate bonus/penalty based on index alignment (±5 points - REDUCED)"""
    if not trend_data or 'analysis' not in trend_data:
        return 0
    
    analysis = trend_data['analysis']
    trend = analysis.get('15m_trend') or analysis.get('1h_trend') or analysis.get('1d_trend', 'Unknown')
    adx = analysis.get('ADX_analysis', {}).get('value', 0)
    supertrend = trend_data.get('indicators', {}).get('Supertrend', 0)
    
    score_adjustment = 0
    
    if signal_direction == 'bullish':
        if 'Strong Uptrend' in trend:
            score_adjustment += 5
        elif 'Uptrend' in trend or 'Weak Uptrend' in trend:
            score_adjustment += 3
        elif 'Downtrend' in trend:
            score_adjustment -= 5
        elif 'Consolidation' in trend and adx < 20:
            score_adjustment += 1
    
    elif signal_direction == 'bearish':
        if 'Strong Downtrend' in trend:
            score_adjustment += 5
        elif 'Downtrend' in trend or 'Weak Downtrend' in trend:
            score_adjustment += 3
        elif 'Uptrend' in trend:
            score_adjustment -= 5
    
    # Supertrend confirmation
    if signal_direction == 'bullish' and supertrend == 1:
        score_adjustment += 2
    elif signal_direction == 'bearish' and supertrend == -1:
        score_adjustment += 2
    elif signal_direction == 'bullish' and supertrend == -1:
        score_adjustment -= 2
    elif signal_direction == 'bearish' and supertrend == 1:
        score_adjustment -= 2
    
    return score_adjustment

# ============================================================================
# API & DATA FETCHING
# ============================================================================

def get_global_smart_api():
    """Manage global SmartAPI session with auto-refresh"""
    global _global_smart_api, _session_timestamp
    now = time_module.time()
    
    if _global_smart_api is None or (now - _session_timestamp) > SESSION_EXPIRY:
        _global_smart_api = init_smartapi_client()
        _session_timestamp = now
        if not _global_smart_api:
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
            st.error(f"⚠️ SmartAPI auth failed: {data.get('message', 'Unknown error')}")
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
        logging.warning(f"Failed to load instrument list: {str(e)}")
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
    
def calculate_rma(series, period):
    """Wilder's smoothing (RMA) used in ATR calculation"""
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()
    
@retry(max_retries=3, delay=5)
def fetch_stock_data_with_auth(symbol, period="1y", interval="1d"):
    """Fetch stock data from SmartAPI with intelligent caching"""
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
        
        end_date = datetime.now()
        period_map = {
            "2y": 730, "1y": 365, "6mo": 180, 
            "1mo": 30, "1d": 1
        }
        days = period_map.get(period, 365)
        start_date = end_date - timedelta(days=days)
        
        interval_map = {
            "1d": "ONE_DAY",
            "1h": "ONE_HOUR",
            "30m": "THIRTY_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "5m": "FIVE_MINUTE"
        }
        api_interval = interval_map.get(interval, "ONE_DAY")
        
        symbol_token_map = load_symbol_token_map()
        symboltoken = symbol_token_map.get(symbol)
        
        if not symboltoken:
            logging.warning(f"Token not found for {symbol}")
            return pd.DataFrame()
        
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
            
            if interval == "1d":
                expire = 86400
            else:
                expire = 300
            
            buffer = io.BytesIO()
            data.to_pickle(buffer)
            cache.set(cache_key, buffer.getvalue(), expire=expire)
            
            return data
        else:
            raise ValueError(f"No data for {symbol}")
    
    except Exception as e:
        logging.error(f"Error fetching {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_stock_data_cached(symbol, period="1y", interval="1d"):
    """Wrapper for stock data fetching"""
    return fetch_stock_data_with_auth(symbol, period, interval)

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_data(data, required_columns=None, min_length=50):
    """Comprehensive OHLCV validation"""
    if required_columns is None:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    if data is None or data.empty:
        return False
    
    if len(data) < min_length:
        return False
    
    missing = [c for c in required_columns if c not in data.columns]
    if missing:
        return False
    
    if data[required_columns].isnull().any().any():
        return False
    
    price_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in data.columns]
    if (data[price_cols] <= 0).any().any():
        return False
    
    return True

# ============================================================================
# SWING TRADING INDICATORS
# ============================================================================

def calculate_swing_indicators(data):
    """Calculate swing trading indicators matching TradingView defaults"""
    if not validate_data(data, min_length=200):
        return data
    
    df = data.copy()
    
    # MACD
    df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = ta.trend.EMAIndicator(df['MACD'], window=9).ema_indicator()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df.drop(['EMA_12', 'EMA_26'], axis=1, inplace=True)
    
    # 200 EMA
    df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
    df['Above_EMA200'] = df['Close'] > df['EMA_200']
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['RSI_Oversold'] = np.where(df['Above_EMA200'], 40, 30)
    df['RSI_Overbought'] = np.where(df['Above_EMA200'], 70, 60)
    
    # ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = calculate_rma(df['TR'], 14)
    df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
    df.drop('TR', axis=1, inplace=True)
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # ADX
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    df['Trending'] = df['ADX'] > 25
    
    # Volume
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Spike'] = df['Volume'] > (df['Volume_SMA'] * 1.5)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA'].clip(lower=0.001)
    
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

def calculate_swing_score(df, symbol=None, timeframe='1d'):
    """Calculate swing trading score with BALANCED market context"""
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
    
    # EMA position (±3)
    if close > ema200:
        score += 3
    else:
        score -= 3
    
    # MACD (±4 fresh, ±1 sustained)
    if macd > macd_signal and macd_prev <= macd_signal_prev:
        score += 4
    elif macd > macd_signal:
        score += 1
    elif macd < macd_signal and macd_prev >= macd_signal_prev:
        score -= 4
    elif macd < macd_signal:
        score -= 1
    
    # RSI (±3 scaled)
    if rsi < rsi_oversold:
        strength = (rsi_oversold - rsi) / rsi_oversold
        score += 3 * strength
    elif rsi > rsi_overbought:
        strength = (rsi - rsi_overbought) / (100 - rsi_overbought)
        score -= 3 * strength
    
    # ADX (±2)
    if adx > 25:
        if close > ema200:
            score += 2
        else:
            score -= 2
    
    # BB position (±2)
    if bb_position < 0.2:
        score += 2
    elif bb_position > 0.8:
        score -= 2
    
    # Volume (±1)
    if volume_ratio > 1.5:
        if score > 0:
            score += 1
        else:
            score -= 1
    
    # ===== MARKET CONTEXT ADJUSTMENTS (BALANCED) =====
    if symbol:
        signal_direction = 'bullish' if score > 0 else 'bearish'
        
        # 1. Index Alignment (±5)
        index_trends = get_index_trend_for_timeframe(timeframe)
        if index_trends:
            relevant_index = get_relevant_index(symbol)
            index_data = index_trends.get(relevant_index)
            index_adjustment = calculate_index_alignment_score(index_data, signal_direction)
            score += index_adjustment
        
        # 2. Market Breadth Alignment (±5)
        breadth_adjustment = calculate_market_breadth_alignment(signal_direction)
        score += breadth_adjustment
        
        # 3. Industry Performance (±3)
        industry_data = get_industry_performance(symbol)
        if industry_data:
            industry_adjustment = calculate_industry_alignment_score(industry_data, signal_direction)
            score += industry_adjustment
    
    # IMPROVED NORMALIZATION (prevents ceiling effect)
    # Technical base: -15 to +15
    # Context bonus: -13 to +13
    # Total range: -28 to +28
    
    if score >= 0:
        normalized = 50 + (score * 1.8)
    else:
        normalized = 50 + (score * 1.8)
    
    normalized = np.clip(normalized, 0, 100)
    return round(normalized, 1)

# ============================================================================
# INTRADAY INDICATORS
# ============================================================================

def calculate_intraday_indicators(data, timeframe='15m'):
    """Enhanced intraday indicators with CORRECTED VWAP bands"""
    if len(data) < 200:
        return data
    
    df = data.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # EMA CROSSOVER
    if timeframe == '5m':
        fast, slow = 9, 21
        rsi_period = 7
        macd_fast, macd_slow, macd_sign = 5, 13, 5
    else:
        fast, slow = 20, 50
        rsi_period = 9
        macd_fast, macd_slow, macd_sign = 12, 26, 9
    
    df['EMA_Fast'] = ta.trend.EMAIndicator(df['Close'], window=fast).ema_indicator()
    df['EMA_Slow'] = ta.trend.EMAIndicator(df['Close'], window=slow).ema_indicator()
    df['EMA_Bullish'] = df['EMA_Fast'] > df['EMA_Slow']
    df['EMA_Crossover'] = (df['EMA_Bullish'] != df['EMA_Bullish'].shift(1)) & df['EMA_Bullish']
    df['EMA_Crossunder'] = (df['EMA_Bullish'] != df['EMA_Bullish'].shift(1)) & ~df['EMA_Bullish']
    
    # VWAP WITH BANDS
    df['Date'] = df.index.date
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    df['TPV'] = df['Typical_Price'] * df['Volume']
    df['Cumul_TPV'] = df.groupby('Date')['TPV'].cumsum()
    df['Cumul_Vol'] = df.groupby('Date')['Volume'].cumsum()
    df['VWAP'] = df['Cumul_TPV'] / df['Cumul_Vol'].replace(0, np.nan)
    
    df['Deviation_Squared'] = (df['Typical_Price'] - df['VWAP']) ** 2
    df['Cumul_Dev_Sq'] = df.groupby('Date')['Deviation_Squared'].cumsum()
    df['Bar_Count'] = df.groupby('Date').cumcount() + 1
    df['VWAP_Std'] = np.sqrt(df['Cumul_Dev_Sq'] / df['Bar_Count'])
    
    df['VWAP_Upper1'] = df['VWAP'] + (df['VWAP_Std'] * 1)
    df['VWAP_Upper2'] = df['VWAP'] + (df['VWAP_Std'] * 2)
    df['VWAP_Lower1'] = df['VWAP'] - (df['VWAP_Std'] * 1)
    df['VWAP_Lower2'] = df['VWAP'] - (df['VWAP_Std'] * 2)
    
    df['Above_VWAP'] = df['Close'] > df['VWAP']
    df['At_VWAP_Upper_Extreme'] = df['Close'] >= df['VWAP_Upper2']
    df['At_VWAP_Lower_Extreme'] = df['Close'] <= df['VWAP_Lower2']
    df['In_VWAP_Channel'] = (df['Close'] >= df['VWAP_Lower1']) & (df['Close'] <= df['VWAP_Upper1'])
    
    df['VWAP_Upper_Breakout'] = (df['Close'] > df['VWAP_Upper1']) & (df['Close'].shift(1) <= df['VWAP_Upper1'])
    df['VWAP_Lower_Breakdown'] = (df['Close'] < df['VWAP_Lower1']) & (df['Close'].shift(1) >= df['VWAP_Lower1'])
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=rsi_period).rsi()
    
    # ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = calculate_rma(df['TR'], 14)
    df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
    df.drop('TR', axis=1, inplace=True)

    # VOLUME
    df['Avg_Volume'] = df['Volume'].rolling(20).mean()
    df['RVOL'] = df['Volume'] / df['Avg_Volume'].clip(lower=0.001)
    df['Volume_Spike'] = df['RVOL'] > 1.5
    df['High_Volume'] = df['RVOL'] > 2.0
    
    # OPENING RANGE
    df['Time'] = df.index.time
    
    or_window = time(9, 30) if timeframe == '5m' else time(9, 45)
    df['Is_OR'] = (df['Time'] >= time(9, 15)) & (df['Time'] <= or_window)
    
    df['OR_High'] = df[df['Is_OR']].groupby('Date')['High'].transform('max')
    df['OR_Low'] = df[df['Is_OR']].groupby('Date')['Low'].transform('min')
    
    df['OR_High'] = df.groupby('Date')['OR_High'].transform(lambda x: x.ffill())
    df['OR_Low'] = df.groupby('Date')['OR_Low'].transform(lambda x: x.ffill())
    
    df['OR_Mid'] = (df['OR_High'] + df['OR_Low']) / 2
    df['OR_Range'] = df['OR_High'] - df['OR_Low']
    
    df['After_OR'] = df['Time'] > or_window
    df['Inside_OR'] = (df['Close'] >= df['OR_Low']) & (df['Close'] <= df['OR_High'])
    
    df['OR_Breakout_Long'] = (
        df['After_OR'] & 
        (df['Close'] > df['OR_High']) & 
        (df['Close'].shift(1) <= df['OR_High'])
    )
    
    df['OR_Breakout_Short'] = (
        df['After_OR'] & 
        (df['Close'] < df['OR_Low']) & 
        (df['Close'].shift(1) >= df['OR_Low'])
    )
    
    df['Failed_OR_Breakout'] = (
        ((df['High'].shift(1) > df['OR_High']) | (df['Low'].shift(1) < df['OR_Low'])) &
        df['Inside_OR']
    )
    
    # MACD
    df['EMA_Fast_MACD'] = ta.trend.EMAIndicator(df['Close'], window=macd_fast).ema_indicator()
    df['EMA_Slow_MACD'] = ta.trend.EMAIndicator(df['Close'], window=macd_slow).ema_indicator()
    df['MACD'] = df['EMA_Fast_MACD'] - df['EMA_Slow_MACD']
    df['MACD_Signal'] = ta.trend.EMAIndicator(df['MACD'], window=macd_sign).ema_indicator()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Bullish'] = df['MACD'] > df['MACD_Signal']
    df['MACD_Hist_Rising'] = df['MACD_Hist'] > df['MACD_Hist'].shift(1)
    df.drop(['EMA_Fast_MACD', 'EMA_Slow_MACD'], axis=1, inplace=True)
    
    # ADX
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    df['Trending_Intraday'] = df['ADX'] > 20
    
    # TIME FILTERS
    df['Pre_Market'] = df['Time'] < time(9, 15)
    df['Opening_Range_Period'] = df['Is_OR']
    df['Safe_Hours'] = (df['Time'] >= time(9, 30)) & (df['Time'] <= time(15, 0))
    
    df['Prime_Hours'] = (
        ((df['Time'] >= time(9, 45)) & (df['Time'] <= time(11, 30))) |
        ((df['Time'] >= time(14, 0)) & (df['Time'] <= time(15, 0)))
    )
    
    df['Lunch_Hours'] = (df['Time'] >= time(12, 0)) & (df['Time'] <= time(13, 30))
    df['Last_30_Min'] = df['Time'] >= time(15, 0)
    df['Closing_Session'] = df['Time'] > time(15, 15)

    df.drop(columns=['Date', 'Typical_Price', 'Time', 'Is_OR', 'TPV', 'Cumul_TPV', 
                     'Cumul_Vol', 'Deviation_Squared', 'Cumul_Dev_Sq', 'Bar_Count'], 
            inplace=True, errors='ignore')
    return df

def detect_intraday_regime(df):
    """Enhanced regime detection with time awareness"""
    if len(df) < 50:
        return "Unknown"
    
    if df['Pre_Market'].iloc[-1]:
        return "Pre-Market"
    elif df['Opening_Range_Period'].iloc[-1]:
        return "Opening Range Formation"
    elif df['Closing_Session'].iloc[-1]:
        return "Closing Session"
    elif df['Last_30_Min'].iloc[-1]:
        return "Last 30 Min (Exit Only)"
    
    close = df['Close'].iloc[-1]
    vwap = df['VWAP'].iloc[-1]
    ema_bullish = df['EMA_Bullish'].iloc[-1]
    macd_bullish = df['MACD_Bullish'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    in_vwap_channel = df['In_VWAP_Channel'].iloc[-1]
    
    if pd.isna(vwap) or pd.isna(adx):
        return "Unknown"
    
    above_vwap = close > vwap
    trending = adx > 20
    
    if above_vwap and ema_bullish and macd_bullish and trending:
        return "Strong Uptrend"
    elif not above_vwap and not ema_bullish and not macd_bullish and trending:
        return "Strong Downtrend"
    elif above_vwap and ema_bullish:
        return "Weak Uptrend"
    elif not above_vwap and not ema_bullish:
        return "Weak Downtrend"
    elif in_vwap_channel and not trending:
        return "Choppy (VWAP Range)"
    else:
        return "Neutral"

# ============================================================================
# INTRADAY SCORING STRATEGIES
# ============================================================================

def calculate_opening_range_score(df):
    """Opening Range Breakout Strategy"""
    score = 0
    
    if not df['After_OR'].iloc[-1]:
        return 0
    
    or_breakout_long = df['OR_Breakout_Long'].iloc[-1]
    or_breakout_short = df['OR_Breakout_Short'].iloc[-1]
    failed_breakout = df['Failed_OR_Breakout'].iloc[-1]
    rvol = df['RVOL'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    macd_bullish = df['MACD_Bullish'].iloc[-1]
    or_range = df['OR_Range'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    
    if pd.isna(or_range) or pd.isna(atr) or or_range < (atr * 0.5):
        return 0
    
    if or_breakout_long:
        score += 5
        
        if rvol > 2.5:
            score += 3
        elif rvol > 1.8:
            score += 2
        elif rvol > 1.5:
            score += 1
        else:
            score -= 2
        
        if macd_bullish:
            score += 1
        
        if adx > 20:
            score += 2
    
    elif or_breakout_short:
        score -= 5
        
        if rvol > 2.5:
            score -= 3
        elif rvol > 1.8:
            score -= 2
        elif rvol > 1.5:
            score -= 1
        else:
            score += 2
        
        if not macd_bullish:
            score -= 1
        
        if adx > 20:
            score -= 2
    
    elif failed_breakout:
        if df['High'].shift(1).iloc[-1] > df['OR_High'].iloc[-1]:
            score -= 3
        elif df['Low'].shift(1).iloc[-1] < df['OR_Low'].iloc[-1]:
            score += 3
    
    return score

def calculate_vwap_mean_reversion_score(df):
    """VWAP Mean Reversion Strategy"""
    score = 0
    
    close = df['Close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    at_lower_extreme = df['At_VWAP_Lower_Extreme'].iloc[-1]
    at_upper_extreme = df['At_VWAP_Upper_Extreme'].iloc[-1]
    vwap_upper_breakout = df['VWAP_Upper_Breakout'].iloc[-1]
    vwap_lower_breakdown = df['VWAP_Lower_Breakdown'].iloc[-1]
    rvol = df['RVOL'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    vwap_lower1 = df['VWAP_Lower1'].iloc[-1]
    vwap_upper1 = df['VWAP_Upper1'].iloc[-1]
    
    if at_lower_extreme and rsi < 30:
        rsi_strength = (30 - rsi) / 30
        score += 4 * rsi_strength
        
        if adx < 20:
            score += 2
        
        if rvol > 1.5:
            score += 1
    
    elif at_upper_extreme and rsi > 70:
        rsi_strength = (rsi - 70) / 30
        score -= 4 * rsi_strength
        
        if adx < 20:
            score -= 2
        
        if rvol > 1.5:
            score -= 1
    
    elif close <= vwap_lower1 and rsi < 40:
        score += 2
    elif close >= vwap_upper1 and rsi > 60:
        score -= 2
    
    if vwap_upper_breakout and rvol > 2.0 and adx > 20:
        score = max(score, 0)
        score += 2
    
    elif vwap_lower_breakdown and rvol > 2.0 and adx > 20:
        score = min(score, 0)
        score -= 2
    
    return score

def calculate_intraday_trend_score(df):
    """Intraday Trend Following"""
    score = 0
    
    close = df['Close'].iloc[-1]
    vwap = df['VWAP'].iloc[-1]
    ema_bullish = df['EMA_Bullish'].iloc[-1]
    ema_crossover = df['EMA_Crossover'].iloc[-1]
    ema_crossunder = df['EMA_Crossunder'].iloc[-1]
    macd_bullish = df['MACD_Bullish'].iloc[-1]
    macd_hist_rising = df['MACD_Hist_Rising'].iloc[-1]
    rvol = df['RVOL'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    
    if close > vwap:
        score += 2
        
        if ema_crossover:
            score += 3
        elif ema_bullish:
            score += 1
        
        if macd_bullish:
            score += 1
        
        if macd_hist_rising:
            score += 1
        
        if rvol > 2.0:
            score += 2
        elif rvol > 1.5:
            score += 1
        
        if adx > 25:
            score += 2
        elif adx > 20:
            score += 1
    
    elif close < vwap:
        score -= 2
        
        if ema_crossunder:
            score -= 3
        elif not ema_bullish:
            score -= 1
        
        if not macd_bullish:
            score -= 1
        
        if not macd_hist_rising:
            score -= 1
        
        if rvol > 2.0:
            score -= 2
        elif rvol > 1.5:
            score -= 1
        
        if adx > 25:
            score -= 2
        elif adx > 20:
            score -= 1
    
    return score

def calculate_intraday_score(df, symbol=None, timeframe='15m'):
    """Unified intraday scoring with BALANCED market context"""
    regime = detect_intraday_regime(df)
    
    if regime in ["Pre-Market", "Closing Session", "Unknown", "Last 30 Min (Exit Only)", "Opening Range Formation"]:
        return 50
    
    safe_hours = df['Safe_Hours'].iloc[-1]
    prime_hours = df['Prime_Hours'].iloc[-1]
    lunch_hours = df['Lunch_Hours'].iloc[-1]
    
    if not safe_hours:
        return 50
    
    or_score = calculate_opening_range_score(df)
    mean_reversion_score = calculate_vwap_mean_reversion_score(df)
    trend_score = calculate_intraday_trend_score(df)
    
    current_time = df.index[-1].time()
    
    if time(9, 45) <= current_time <= time(11, 0):
        if or_score != 0:
            raw_score = or_score
        else:
            raw_score = trend_score * 0.5
    
    elif regime in ["Strong Uptrend", "Strong Downtrend"] and not lunch_hours:
        raw_score = trend_score
    
    elif prime_hours:
        if regime in ["Choppy (VWAP Range)", "Weak Uptrend", "Weak Downtrend"]:
            raw_score = mean_reversion_score
        else:
            raw_score = trend_score
    
    else:
        raw_score = trend_score
    
    # ===== MARKET CONTEXT ADJUSTMENTS (BALANCED) =====
    if symbol:
        signal_direction = 'bullish' if raw_score > 0 else 'bearish'
        
        # 1. Index Alignment (±5)
        index_trends = get_index_trend_for_timeframe(timeframe)
        if index_trends:
            relevant_index = get_relevant_index(symbol)
            index_data = index_trends.get(relevant_index)
            index_adjustment = calculate_index_alignment_score(index_data, signal_direction)
            raw_score += index_adjustment
        
        # 2. Market Breadth Alignment (±5)
        breadth_adjustment = calculate_market_breadth_alignment(signal_direction)
        raw_score += breadth_adjustment
        
        # 3. Industry Performance (±3)
        industry_data = get_industry_performance(symbol)
        if industry_data:
            industry_adjustment = calculate_industry_alignment_score(industry_data, signal_direction)
            raw_score += industry_adjustment
    
    # Time modifiers
    if prime_hours:
        raw_score *= 1.2
    elif lunch_hours:
        raw_score *= 0.7
    
    # IMPROVED NORMALIZATION
    if raw_score >= 0:
        normalized = 50 + (raw_score * 2.5)
    else:
        normalized = 50 + (raw_score * 2.5)
    
    normalized = np.clip(normalized, 0, 100)
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
    
    max_loss_pct = 0.08
    max_acceptable_stop = close * (1 - max_loss_pct)
    
    if regime in ["Strong Uptrend", "Weak Uptrend"]:
        atr_stop = close - (2 * atr)
        ema_stop = ema200 * 0.98
        stop_loss = max(atr_stop, ema_stop)
    elif regime in ["Consolidation (Above EMA)", "Consolidation (Below EMA)"]:
        stop_loss = bb_lower * 0.98
    else:
        stop_loss = close - (1.5 * atr)
    
    stop_loss = max(stop_loss, max_acceptable_stop)
    stop_loss = round(stop_loss, 2)
    
    risk = buy_at - stop_loss
    if regime == "Strong Uptrend":
        rr_ratio = 3
    elif regime in ["Weak Uptrend", "Consolidation (Above EMA)"]:
        rr_ratio = 2
    else:
        rr_ratio = 1.5
    
    target = buy_at + (risk * rr_ratio)
    target = round(target, 2)
    
    risk_amount = account_size * risk_pct
    position_size = int(risk_amount / risk) if risk > 0 else 0
    max_position = int((account_size * 0.1) / buy_at)
    position_size = min(position_size, max_position)
    
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
    """Enhanced intraday position with OR-aware stops"""
    close = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    vwap = df['VWAP'].iloc[-1]
    vwap_lower1 = df['VWAP_Lower1'].iloc[-1]
    or_high = df['OR_High'].iloc[-1]
    or_low = df['OR_Low'].iloc[-1]
    or_mid = df['OR_Mid'].iloc[-1]
    regime = detect_intraday_regime(df)
    
    buy_at = round(close, 2)
    
    max_loss_pct = 0.03
    max_acceptable_stop = close * (1 - max_loss_pct)
    
    if regime == "Strong Uptrend":
        vwap_stop = vwap - (0.3 * atr)
        or_stop = or_low - (0.2 * atr) if pd.notna(or_low) else vwap_stop
        atr_stop = close - (1.5 * atr)
        stop_loss = max(vwap_stop, or_stop, atr_stop)
    
    elif regime in ["Weak Uptrend", "Choppy (VWAP Range)"]:
        stop_loss = max(close - (1.0 * atr), vwap_lower1 - (0.2 * atr))
    
    elif pd.notna(or_low) and df['After_OR'].iloc[-1]:
        stop_loss = or_low - (0.5 * atr)
    
    else:
        stop_loss = close - (1.5 * atr)
    
    stop_loss = max(stop_loss, max_acceptable_stop)
    stop_loss = round(stop_loss, 2)
    
    risk = buy_at - stop_loss
    
    if regime == "Strong Uptrend":
        rr_ratio = 2.0
        target = buy_at + (risk * rr_ratio)
    elif pd.notna(or_high) and df['OR_Breakout_Long'].iloc[-1]:
        or_range = or_high - or_low
        target = or_high + or_range
    else:
        rr_ratio = 1.0
        target = buy_at + risk
    
    target = round(target, 2)
    
    risk_amount = account_size * risk_pct
    position_size = int(risk_amount / risk) if risk > 0 else 0
    max_position = int((account_size * 0.05) / buy_at)
    position_size = min(position_size, max_position)
    
    trailing_stop = close - (1.0 * atr)
    trailing_stop = round(trailing_stop, 2)
    
    current_time = df.index[-1].time()
    minutes_to_close = (15 * 60 + 15) - (current_time.hour * 60 + current_time.minute)
    hours_to_close = round(minutes_to_close / 60, 2)
    
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
        "vwap_lower1": round(vwap_lower1, 2) if pd.notna(vwap_lower1) else None,
        "or_high": round(or_high, 2) if pd.notna(or_high) else None,
        "or_low": round(or_low, 2) if pd.notna(or_low) else None,
        "or_mid": round(or_mid, 2) if pd.notna(or_mid) else None,
        "hours_to_close": hours_to_close
    }

# ============================================================================
# UNIFIED RECOMMENDATION
# ============================================================================

def generate_recommendation(data, symbol, trading_style='swing', timeframe='1d', account_size=30000):
    """Generate unified recommendations with COMPLETE MARKET CONTEXT"""
    
    if trading_style == 'swing':
        df = calculate_swing_indicators(data)
        regime = detect_swing_regime(df)
        score = calculate_swing_score(df, symbol, timeframe)
        position = calculate_swing_position(df, account_size)
        
    else:
        df = calculate_intraday_indicators(data, timeframe)
        regime = detect_intraday_regime(df)
        score = calculate_intraday_score(df, symbol, timeframe)
        position = calculate_intraday_position(df, account_size)
    
    # Get market context
    market_health, market_signal, market_factors = calculate_market_health_score()
    
    # Get index context
    index_trends = get_index_trend_for_timeframe(timeframe)
    index_context = None
    if index_trends:
        relevant_index = get_relevant_index(symbol)
        index_data = index_trends.get(relevant_index)
        if index_data and 'analysis' in index_data:
            index_context = {
                'index_name': 'Nifty' if relevant_index == 'nifty' else 'Bank Nifty',
                'trend': index_data['analysis'].get(f'{timeframe}_trend') or 
                        index_data['analysis'].get('15m_trend') or 
                        index_data['analysis'].get('1h_trend') or 
                        index_data['analysis'].get('1d_trend', 'Unknown'),
                'adx': index_data['analysis'].get('ADX_analysis', {}).get('value', 0),
                'trend_strength': index_data['analysis'].get('trend_strength', 'Unknown'),
                'supertrend': index_data.get('indicators', {}).get('Supertrend', 0)
            }
    
    # Get industry context
    industry_data = get_industry_performance(symbol)
    industry_context = None
    if industry_data:
        industry_context = {
            'industry_name': industry_data.get('Industry'),
            'avg_change': industry_data.get('avgChange', 0),
            'advancing': industry_data.get('advancing', 0),
            'declining': industry_data.get('declining', 0),
            'total': industry_data.get('total', 1)
        }
    
    # Signal
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
    
    else:
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
            reasons.append("OR breakout (bullish)")
        elif df['OR_Breakout_Short'].iloc[-1]:
            reasons.append("OR breakdown (bearish)")
    
    # Add market context
    if index_context:
        reasons.append(f"{index_context['index_name']}: {index_context['trend']}")
    
    if industry_context:
        reasons.append(f"Industry: {industry_context['avg_change']:.2f}%")
    
    reasons.append(f"Market: {market_signal}")
    
    return {
        "symbol": symbol,
        "trading_style": trading_style.capitalize(),
        "timeframe": timeframe,
        "score": score,
        "signal": signal,
        "regime": regime,
        "reason": ", ".join(reasons),
        "index_context": index_context,
        "industry_context": industry_context,
        "market_health": market_health,
        "market_signal": market_signal,
        "market_factors": market_factors,
        "processed_data": df, 
        **position
    }

# ============================================================================
# BACKTESTING
# ============================================================================

def backtest_strategy(data, symbol, trading_style='swing', timeframe='1d', initial_capital=30000):
    """Backtest strategy with realistic costs"""
    
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
    
    BROKERAGE = 0.0003
    STT = 0.001
    SLIPPAGE = 0.0005
    
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
            
            if position and rec['signal'] in ['Sell', 'Strong Sell']:
                exit_price_effective = current_price * (1 - BROKERAGE - STT - SLIPPAGE)
                pnl = (exit_price_effective - entry_price) * qty
                cash += (current_price * qty * (1 - BROKERAGE - STT))
                returns.append(pnl / (entry_price * qty))
                
                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": current_date,
                    "exit_price": current_price,
                    "pnl": pnl,
                    "return_pct": (pnl / (entry_price * qty)) * 100
                })
                
                position = None
                qty = 0
            
            if not position and rec['signal'] in ['Buy', 'Strong Buy']:
                entry_price = current_price * (1 + BROKERAGE + SLIPPAGE)
                entry_date = current_date
                qty = rec['position_size']
                cash -= qty * current_price * (1 + BROKERAGE)
                position = "Long"
            
            equity = cash + (qty * current_price if position else 0)
            results['equity_curve'].append((current_date, equity))
            
        except Exception as e:
            logging.warning(f"Backtest error at {current_date}: {str(e)}")
            continue
    
    if position:
        current_price = data['Close'].iloc[-1]
        exit_price_effective = current_price * (1 - BROKERAGE - STT - SLIPPAGE)
        pnl = (exit_price_effective - entry_price) * qty
        cash += (current_price * qty * (1 - BROKERAGE - STT))
        returns.append(pnl / (entry_price * qty))
        trades.append({
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": data.index[-1],
            "exit_price": current_price,
            "pnl": pnl,
            "return_pct": (pnl / (entry_price * qty)) * 100
        })
    
    if trades:
        results['trades'] = len(trades)
        results['trades_list'] = trades
        results['total_return'] = ((cash - initial_capital) / initial_capital) * 100
        results['win_rate'] = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100
        
        if returns:
            periods_per_year = {
                '5m': 252 * 75,
                '15m': 252 * 25,
                '30m': 252 * 12.5,
                '1h': 252 * 6,
                '1d': 252
            }
            annualization_factor = np.sqrt(periods_per_year.get(timeframe, 252))
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            results['sharpe_ratio'] = (mean_return / (std_return + 1e-9)) * annualization_factor
            results['annual_return'] = mean_return * periods_per_year.get(timeframe, 252) * 100
    
    if results['equity_curve']:
        equity_df = pd.DataFrame(results['equity_curve'], columns=['Date', 'Equity'])
        equity_df['Peak'] = equity_df['Equity'].cummax()
        equity_df['DD'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
        results['max_drawdown'] = equity_df['DD'].min() * 100
    
    return results

# ============================================================================
# BATCH ANALYSIS WITH SECTOR DIVERSITY
# ============================================================================

def analyze_stock_batch(symbol, trading_style='swing', timeframe='1d'):
    """Analyze single stock with error handling"""
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
    """Analyze multiple stocks with SECTOR DIVERSITY"""
    all_results = []
    
    for i, symbol in enumerate(stock_list):
        try:
            result = analyze_stock_batch(symbol, trading_style, timeframe)
            if result:
                # Add sector information
                for sector_name, sector_stocks in SECTORS.items():
                    if symbol in sector_stocks:
                        result['Sector'] = sector_name
                        break
                all_results.append(result)
        except Exception as e:
            logging.error(f"Failed to analyze {symbol}: {str(e)}")
            
        if progress_callback:
            progress_callback((i + 1) / len(stock_list))
        
        time_module.sleep(5)
    
    if not all_results:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    # Filter based on trading style
    if trading_style == 'intraday':
        df = df[df['Signal'].str.contains('Buy', na=False)]
    else:
        # For swing, show Buy signals AND scores > 60
        df = df[(df['Signal'].str.contains('Buy', na=False)) | (df['Score'] >= 60)]
    
    # ENSURE SECTOR DIVERSITY - Take top 2 from each sector
    diverse_results = []
    
    if 'Sector' in df.columns and not df.empty:
        for sector in df['Sector'].unique():
            sector_df = df[df['Sector'] == sector].nlargest(2, 'Score')
            diverse_results.append(sector_df)
        
        if diverse_results:
            diverse_df = pd.concat(diverse_results, ignore_index=True)
            result_df = diverse_df.sort_values('Score', ascending=False).head(10)
        else:
            result_df = df.sort_values('Score', ascending=False).head(10)
    else:
        result_df = df.sort_values('Score', ascending=False).head(10)
    
    return result_df

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
    """Save picks to database with bulk insert"""
    conn = sqlite3.connect('stock_picks.db')
    today = datetime.now().strftime('%Y-%m-%d')
    
    records = []
    for _, row in results_df.head(5).iterrows():
        records.append((
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
    
    conn.executemany('''
        INSERT OR REPLACE INTO picks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', records)
    
    conn.commit()
    conn.close()

# ============================================================================
# STREAMLIT UI
# ============================================================================

def display_intraday_chart(rec, data):
    """Enhanced intraday chart with OR and VWAP"""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    if 'VWAP' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['VWAP'],
            mode='lines', name='VWAP',
            line=dict(color='blue', width=2)
        ))
    
    if 'VWAP_Upper1' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['VWAP_Upper1'],
            mode='lines', name='VWAP Upper',
            line=dict(color='orange', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['VWAP_Lower1'],
            mode='lines', name='VWAP Lower',
            line=dict(color='orange', width=1, dash='dash')
        ))
    
    if rec.get('or_high'):
        fig.add_hline(y=rec['or_high'], line_dash="dot", 
                     annotation_text="OR High", line_color="green")
        fig.add_hline(y=rec['or_low'], line_dash="dot", 
                     annotation_text="OR Low", line_color="red")
    
    fig.add_hline(y=rec['buy_at'], line_dash="solid", 
                 annotation_text="Entry", line_color="white")
    fig.add_hline(y=rec['stop_loss'], line_dash="dash", 
                 annotation_text="Stop", line_color="red")
    fig.add_hline(y=rec['target'], line_dash="dash", 
                 annotation_text="Target", line_color="green")
    
    fig.update_layout(
        title=f"{rec['symbol']} - {rec['timeframe']} Intraday",
        xaxis_title="Time",
        yaxis_title="Price (₹)",
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    """Main Streamlit app"""
    
    init_database()
    
    st.set_page_config(page_title="StockGenie Pro", layout="wide")
    st.title("📊 StockGenie Pro V2.3 - Professional NSE Analysis")
    st.caption("✨ BALANCED SCORING: Proper sector diversity + Realistic score distribution")
    st.subheader(f"📅 {datetime.now().strftime('%d %b %Y, %A')}")
    
    # Sidebar
    st.sidebar.title("🔍 Configuration")
    
    trading_style = st.sidebar.radio(
        "Trading Style",
        ["Swing Trading", "Intraday Trading"],
        help="Swing: Hold days-weeks. Intraday: Close same day"
    )
    
    if trading_style == "Intraday Trading":
        timeframe_display = st.sidebar.selectbox("Timeframe", ["5 min", "15 min", "30 min"], index=1)
        timeframe = timeframe_display.replace(" ", "").lower()[:-2] + "m"
    else:
        timeframe_display = "Daily"
        timeframe = "1d"
    
    sector_options = ["All"] + list(SECTORS.keys())
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        sector_options,
        default=["Bank", "IT"],
        help="Choose sectors"
    )
    
    if "All" in selected_sectors:
        stock_list = [s for sector in SECTORS.values() for s in sector]
    else:
        stock_list = [s for sector in selected_sectors for s in SECTORS.get(sector, [])]
    
    symbol = st.sidebar.selectbox("Select Stock", stock_list, index=0)
    account_size = st.sidebar.number_input("Account Size (₹)", min_value=10000, max_value=10000000, value=30000, step=5000)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Analysis", "🔍 Scanner", "📊 Backtest", "📜 History", "🌍 Market Dashboard"])
    
    # TAB 1: Analysis
    with tab1:
        # Market Health Dashboard
        st.subheader("🌍 Market Health")
        
        market_health, market_signal, market_factors = calculate_market_health_score()
        
        col1, col2, col3, col4 = st.columns(4)
        
        health_color = "🟢" if market_health >= 60 else "🟡" if market_health >= 40 else "🔴"
        col1.metric("Market Health", f"{health_color} {market_health}/100", market_signal)
        
        if 'advance_ratio' in market_factors:
            col2.metric("Breadth", f"{market_factors['advance_ratio']:.1f}%", market_factors.get('breadth_signal', ''))
        
        if 'avg_momentum' in market_factors:
            col3.metric("Momentum", f"{market_factors['avg_momentum']:.1f}", market_factors.get('momentum_signal', ''))
        
        if 'volatility' in market_factors:
            col4.metric("Volatility", f"{market_factors['volatility']:.1f}", market_factors.get('volatility_signal', ''))
        
        st.divider()
        
        # Index Dashboard
        st.subheader("📊 Index Trends")
        index_trends = get_index_trend_for_timeframe(timeframe)
        
        if index_trends:
            col1, col2 = st.columns(2)
            
            with col1:
                nifty = index_trends.get('nifty', {})
                if nifty and 'analysis' in nifty:
                    trend = nifty['analysis'].get('15m_trend') or nifty['analysis'].get('1h_trend') or nifty['analysis'].get('1d_trend', 'Unknown')
                    adx = nifty['analysis'].get('ADX_analysis', {}).get('value', 0)
                    supertrend = nifty.get('indicators', {}).get('Supertrend', 0)
                    
                    trend_emoji = "🟢" if "Uptrend" in trend else "🔴" if "Downtrend" in trend else "⚪"
                    st.metric("Nifty 50", f"{trend_emoji} {trend}", f"ADX: {adx:.1f}")
                    st.caption(f"Supertrend: {'Bullish ✅' if supertrend == 1 else 'Bearish ⚠️'}")
            
            with col2:
                bnf = index_trends.get('banknifty', {})
                if bnf and 'analysis' in bnf:
                    trend = bnf['analysis'].get('15m_trend') or bnf['analysis'].get('1h_trend') or bnf['analysis'].get('1d_trend', 'Unknown')
                    adx = bnf['analysis'].get('ADX_analysis', {}).get('value', 0)
                    supertrend = bnf.get('indicators', {}).get('Supertrend', 0)
                    
                    trend_emoji = "🟢" if "Uptrend" in trend else "🔴" if "Downtrend" in trend else "⚪"
                    st.metric("Bank Nifty", f"{trend_emoji} {trend}", f"ADX: {adx:.1f}")
                    st.caption(f"Supertrend: {'Bullish ✅' if supertrend == 1 else 'Bearish ⚠️'}")
        else:
            st.info("⚠️ Index trend data unavailable")
        
        st.divider()
        
        if st.button("🔍 Analyze Selected Stock"):
            with st.spinner(f"Analyzing {symbol}..."):
                try:
                    data = fetch_stock_data_with_auth(symbol, interval=timeframe)
                    
                    if not data.empty:
                        rec = generate_recommendation(
                            data, symbol,
                            'swing' if trading_style == "Swing Trading" else 'intraday',
                            timeframe, account_size
                        )
                        processed_data = rec.get('processed_data', data)
                        
                        # Metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Score", f"{rec['score']}/100")
                        col2.metric("Signal", rec['signal'])
                        col3.metric("Regime", rec['regime'])
                        col4.metric("Current Price", f"₹{rec['current_price']}")
                        
                        if trading_style == "Intraday Trading":
                            col5.metric("Hours Left", f"{rec['hours_to_close']}h")
                            if rec['hours_to_close'] < 0.5:
                                st.warning("⚠️ Less than 30 min to close - EXIT ONLY!")
                        else:
                            col5.metric("Timeframe", timeframe_display)
                        
                        # Alignment Warnings
                        signal_bullish = rec['signal'] in ['Buy', 'Strong Buy']
                        
                        if rec.get('index_context'):
                            idx = rec['index_context']
                            index_bullish = 'Uptrend' in idx['trend']
                            
                            if signal_bullish and not index_bullish:
                                st.warning(f"⚠️ **Counter-Index Trade**: Stock bullish but {idx['index_name']} is in {idx['trend']}. Higher risk!")
                            elif not signal_bullish and index_bullish:
                                st.warning(f"⚠️ **Counter-Index Trade**: Stock bearish but {idx['index_name']} is in {idx['trend']}. Higher risk!")
                            elif signal_bullish and index_bullish:
                                st.success(f"✅ **Index Aligned**: {idx['index_name']} also in {idx['trend']}")
                        
                        market_bullish = rec['market_signal'] in ['Very Bullish', 'Bullish']
                        if signal_bullish and not market_bullish:
                            st.warning(f"⚠️ **Weak Market Breadth**: Stock bullish but overall market is {rec['market_signal']}. Reduce position size!")
                        elif signal_bullish and market_bullish:
                            st.success(f"✅ **Strong Market Support**: Breadth is {rec['market_signal']}")
                        
                        if rec.get('industry_context'):
                            ind = rec['industry_context']
                            industry_bullish = ind['avg_change'] > 0.5
                            
                            if signal_bullish and not industry_bullish:
                                st.warning(f"⚠️ **Weak Industry**: {ind['industry_name']} avg {ind['avg_change']:.2f}%")
                            elif signal_bullish and industry_bullish:
                                st.success(f"✅ **Strong Industry**: {ind['industry_name']} avg +{ind['avg_change']:.2f}%")
                        
                        # Trade Setup
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
                        
                        # Intraday Levels
                        if trading_style == "Intraday Trading":
                            st.subheader("🎯 Key Intraday Levels")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Opening Range:**")
                                if rec.get('or_high'):
                                    st.write(f"  - OR High: ₹{rec['or_high']}")
                                    st.write(f"  - OR Mid: ₹{rec['or_mid']}")
                                    st.write(f"  - OR Low: ₹{rec['or_low']}")
                                else:
                                    st.write("  - Not yet formed")
                            
                            with col2:
                                st.markdown("**VWAP Bands:**")
                                st.write(f"  - VWAP: ₹{rec['vwap']}")
                                if rec.get('vwap_lower1'):
                                    st.write(f"  - Lower Band: ₹{rec['vwap_lower1']}")
                        
                        st.info(f"**Reason**: {rec['reason']}")
                        
                        # Chart
                        if trading_style == "Intraday Trading":
                            fig = display_intraday_chart(rec, data)
                        else:
                            fig = go.Figure()
                            fig.add_trace(go.Candlestick(
                                x=processed_data.index,
                                open=processed_data['Open'],
                                high=processed_data['High'],
                                low=processed_data['Low'],
                                close=processed_data['Close']
                            ))
                            
                            if 'EMA_200' in processed_data.columns:
                                fig.add_trace(go.Scatter(
                                    x=processed_data.index, 
                                    y=processed_data['EMA_200'],
                                    mode='lines',
                                    name='200 EMA',
                                    line=dict(color='purple', width=2)
                                ))
                            
                            fig.update_layout(title=f"{symbol} - Daily", height=500, xaxis_rangeslider_visible=False)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No data available")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # TAB 2: Scanner
    with tab2:
        if st.button("🚀 Scan Top Picks (Max 2 per sector)"):
            progress = st.progress(0)
            status_text = st.empty()
            
            try:
                results = analyze_multiple_stocks(
                    stock_list[:40],  # Scan more stocks for diversity
                    'swing' if trading_style == "Swing Trading" else 'intraday',
                    timeframe,
                    lambda p: (progress.progress(p), status_text.text(f"Scanning... {int(p*100)}%"))
                )
                
                progress.empty()
                status_text.empty()
                
                if not results.empty:
                    save_picks(results, trading_style)
                    st.subheader(f"🏆 Top {trading_style} Picks (Sector Diversified)")
                    st.dataframe(results, use_container_width=True)
                    
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"stock_picks_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No picks found")
            except Exception as e:
                st.error(f"❌ Scanner error: {str(e)}")
    
    # TAB 3: Backtest
    with tab3:
        if st.button("📊 Run Backtest"):
            with st.spinner("Backtesting..."):
                try:
                    data = fetch_stock_data_with_auth(symbol, period="2y", interval=timeframe)
                    
                    if not data.empty:
                        results = backtest_strategy(
                            data, symbol,
                            'swing' if trading_style == "Swing Trading" else 'intraday',
                            timeframe, account_size
                        )
                        
                        st.success("✅ Backtest complete (includes transaction costs)")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Return", f"{results['total_return']:.2f}%")
                        col2.metric("Annual Return", f"{results['annual_return']:.2f}%")
                        col3.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                        col4.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Total Trades", results['trades'])
                        col2.metric("Win Rate", f"{results['win_rate']:.2f}%")
                        
                        if results['trades_list']:
                            st.subheader("Trade History")
                            trades_df = pd.DataFrame(results['trades_list'])
                            st.dataframe(trades_df, use_container_width=True)
                        
                        if results['equity_curve']:
                            st.subheader("Equity Curve")
                            equity_df = pd.DataFrame(results['equity_curve'], columns=['Date', 'Equity'])
                            fig = px.line(equity_df, x='Date', y='Equity', title="Portfolio Value Over Time")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Insufficient data for backtest")
                except Exception as e:
                    st.error(f"❌ Backtest error: {str(e)}")
    
    # TAB 4: History
    with tab4:
        try:
            conn = sqlite3.connect('stock_picks.db')
            history = pd.read_sql_query("SELECT * FROM picks ORDER BY date DESC LIMIT 100", conn)
            conn.close()
            
            if not history.empty:
                st.subheader("📜 Historical Picks")
                st.dataframe(history, use_container_width=True)
            else:
                st.info("No historical data available")
        except Exception as e:
            st.error(f"❌ Database error: {str(e)}")
    
    # TAB 5: Market Dashboard
    with tab5:
        st.subheader("🌍 Complete Market Overview")
        
        breadth_data = fetch_market_breadth()
        
        if breadth_data:
            st.markdown("### 📊 Market Breadth")
            breadth = breadth_data.get('breadth', {})
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Stocks", breadth.get('total', 0))
            col2.metric("Advancing", breadth.get('advancing', 0), delta_color="normal")
            col3.metric("Declining", breadth.get('declining', 0), delta_color="inverse")
            col4.metric("Unchanged", breadth.get('unchanged', 0))
            
            st.markdown("### 🏆 Top Performing Industries")
            industries = breadth_data.get('industry', [])[:10]
            
            if industries:
                industry_df = pd.DataFrame(industries)
                industry_df = industry_df[['Industry', 'avgChange', 'advancing', 'declining', 'total']]
                industry_df.columns = ['Industry', 'Avg Change %', 'Advancing', 'Declining', 'Total']
                st.dataframe(industry_df, use_container_width=True)
        
        sector_data = fetch_sector_performance()
        
        if sector_data:
            st.markdown("### 📈 Sector Indices Performance")
            
            sectors = sector_data.get('data', [])
            
            if sectors:
                sector_df = pd.DataFrame(sectors)
                sector_df = sector_df[['sector_index', 'avg_change', 'advance_ratio', 'momentum', 'signal', 'volatility_score']]
                sector_df.columns = ['Index', 'Avg Change %', 'Advance Ratio %', 'Momentum', 'Signal', 'Volatility']
                st.dataframe(sector_df, use_container_width=True)

if __name__ == "__main__":
    main()
