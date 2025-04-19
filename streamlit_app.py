import yfinance as yf
import pandas as pd
import ta
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
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
import warnings
import logging
import logging.handlers
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import joblib
import psutil
import os
from scipy.stats import t
from sklearn.feature_selection import SelectKBest, f_classif

# Configure logging with rotation
log_handler = logging.handlers.RotatingFileHandler(
 'stockgenie.log', maxBytes=10*1024*1024, backupCount=5
)
logging.basicConfig(
 level=logging.DEBUG,
 format='%(asctime)s - %(levelname)s - %(message)s',
 handlers=[log_handler, logging.StreamHandler()]
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load SpaCy model globally
try:
 NLP = spacy.load("en_core_web_sm")
except OSError:
 logging.error("SpaCy model 'en_core_web_sm' not installed. Install with: python -m spacy download en_core_web_sm")
 NLP = None

USER_AGENTS = [
 "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
 "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
 "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:130.0) Gecko/20100101 Firefox/130.0",
 "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1; rv:130.0) Gecko/20100101 Firefox/130.0",
 "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/128.0.2739.79 Safari/537.36",
 "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15",
 "Mozilla/5.0 (iPhone; CPU iPhone OS 17_6_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Mobile/15E148 Safari/604.1",
 "Mozilla/5.0 (Linux; Android 14; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36",
 "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
 "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
 "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
 "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
 "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
 "Mozilla/5.0 (iPad; CPU OS 17_6_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Mobile/15E148 Safari/604.1",
 "Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36"
]

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
 "TRIX": "Triple Exponential Average - Momentum oscillator",
 "Ultimate_Osc": "Ultimate Oscillator - Combines short, medium, long-term momentum",
 "CMO": "Chande Momentum Oscillator - Measures raw momentum (-100 to 100)",
 "VPT": "Volume Price Trend - Tracks trend strength with price and volume",
 "Pivot Points": "Support and resistance levels based on previous day's prices",
 "Heikin-Ashi": "Smoothed candlestick chart to identify trends",
 "P/E": "Price-to-Earnings Ratio - Valuation metric",
 "EPS": "Earnings Per Share - Profitability metric",
 "ROE": "Return on Equity - Efficiency of equity use",
 "Debt/Equity": "Debt-to-Equity Ratio - Financial leverage",
 "FII/DII Activity": "Foreign/Domestic Institutional Investor buying/selling trends",
 "Promoter Holding": "Percentage of shares held by promoters"
}

SECTORS = {
 "Bank": [
 "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
 "INDUSINDBK.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "UNIONBANK.NS",
 "IDFCFIRSTB.NS", "FEDERALBNK.NS", "RBLBANK.NS", "BANDHANBNK.NS", "INDIANB.NS",
 "BANKINDIA.NS", "KARURVYSYA.NS", "CUB.NS", "J&KBANK.NS", "DCBBANK.NS",
 "AUBANK.NS", "YESBANK.NS", "IDBI.NS", "SOUTHBANK.NS", "CSBBANK.NS",
 "TMB.NS", "KTKBANK.NS", "EQUITASBNK.NS", "UJJIVANSFB.NS"
 ],
 "Aviation": [
 "INDIGO.NS", "SPICEJET.NS", "AAI.NS", "GMRINFRA.NS"
 ]
}

# Dynamic indicator weights (updated by ML model)
INDICATOR_WEIGHTS = {
 'RSI': 1.5, 'MACD': 1.2, 'Bollinger': 1.0, 'VWAP': 1.0, 'Volume_Spike': 0.8,
 'Divergence': 1.0, 'Ichimoku': 1.5, 'CMF': 0.8, 'Donchian': 1.0, 'Keltner': 1.0,
 'TRIX': 0.7, 'Ultimate_Osc': 0.7, 'CMO': 0.7, 'VPT': 0.8, 'Fibonacci': 0.9,
 'Parabolic_SAR': 0.9, 'OBV': 0.8, 'Pivot': 1.0, 'Heikin-Ashi': 1.0
}

# Sector-specific weight adjustments
SECTOR_WEIGHT_ADJUSTMENTS = {
 "Bank": {'Volume_Spike': 1.2, 'VWAP': 1.3, 'RSI': 1.7},
 "Aviation": {'ATR': 1.5, 'MACD': 1.4, 'Ichimoku': 1.8}
}

def tooltip(label, explanation):
 return f"{label} 📌 ({explanation})"

def retry(max_retries=3, delay=1, backoff_factor=2, jitter=0.5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, ConnectionError) as e:
                    retries += 1
                    if retries == max_retries:
                        logging.error(f"Failed after {max_retries} retries: {str(e)}")
                        return None
                    sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                    time.sleep(sleep_time)
        return wrapper
    return decorator

@retry(max_retries=3, delay=2)
def fetch_nse_stock_list():
 url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
 try:
 session = requests.Session()
 session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
 response = session.get(url, timeout=10)
 response.raise_for_status()
 nse_data = pd.read_csv(io.StringIO(response.text))
 stock_list = [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
 logging.info("Successfully fetched NSE stock list")
 return stock_list
 except Exception as e:
 logging.error(f"Failed to fetch NSE stock list: {str(e)}")
 if st._is_running_with_streamlit:
 st.warning("Failed to fetch stock list. Using fallback sectors.")
 fallback_list = list(set([stock for sector in SECTORS.values() for stock in sector]))
 return fallback_list

@retry(max_retries=3, delay=2)
def fetch_stock_data_with_auth(symbol, period="5y", interval="1d", exchange="NS"):
 logging.debug(f"Fetching data for {symbol}, period={period}, interval={interval}")
 if not isinstance(symbol, str):
 logging.error(f"Invalid symbol type: {type(symbol)} for symbol: {symbol}")
 raise TypeError("Please provide a valid stock symbol.")
 try:
 if not symbol.endswith(f".{exchange}"):
 symbol += f".{exchange}"
 session = requests.Session()
 if session is None:
 raise ValueError("Failed to create requests session.")
 session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
 stock = yf.Ticker(symbol, session=session)
 data = stock.history(period=period, interval=interval)
 if data.empty:
 raise ValueError(f"No data available for {symbol}.")
 required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
 if not all(col in data.columns for col in required_cols):
 raise ValueError(f"Missing required data columns for {symbol}.")
 if data['Volume'].mean() < 5000:
 raise ValueError(f"Insufficient trading volume for {symbol}.")
 for col in ['Open', 'High', 'Low', 'Close']:
 data[col] = data[col].astype(np.float32)
 data['Volume'] = data['Volume'].astype(np.int32)
 expected_dates = pd.date_range(start=data.index.min(), end=data.index.max(), freq='B')
 if len(data) < 0.8 * len(expected_dates):
 logging.warning(f"Significant data gaps detected for {symbol}")
 logging.info(f"Successfully fetched data for {symbol}")
 return data.copy() # Return a copy to avoid mutable caching issues
 except Exception as e:
 logging.error(f"Error fetching data for {symbol}: {str(e)}")
 if st._is_running_with_streamlit:
 st.warning(f"Unable to fetch data for {symbol}: {str(e)}")
 return pd.DataFrame()

@lru_cache(maxsize=500)
def fetch_stock_data_cached(symbol, period="5y", interval="1d", exchange="NS"):
 return fetch_stock_data_with_auth(symbol, period, interval, exchange)

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
 if declines == 0 or advances == 0:
 return 0
 return advances / declines

def monte_carlo_simulation(data, simulations=1000, days=30):
 returns = data['Close'].pct_change().dropna()
 if len(returns) >= 30:
 # Use GARCH for sufficient data
 model = arch_model(returns, vol='GARCH', p=1, q=1, dist='t', rescale=False)
 garch_fit = model.fit(disp='off')
 forecasts = garch_fit.forecast(horizon=days)
 volatility = np.sqrt(forecasts.variance.iloc[-1].values)
 mean_return = returns.mean()
 simulation_results = []
 for _ in range(simulations):
 price_series = [data['Close'].iloc[-1]]
 for i in range(days):
 # Use Student's t-distribution for fat tails
 price = price_series[-1] * (1 + t.rvs(df=5, loc=mean_return, scale=volatility[i]))
 price_series.append(price)
 simulation_results.append(price_series)
 return simulation_results
 else:
 # Simple simulation for limited data
 mean_return = returns.mean()
 std_return = returns.std()
 simulation_results = []
 for _ in range(simulations):
 price_series = [data['Close'].iloc[-1]]
 for _ in range(days):
 price = price_series[-1] * (1 + t.rvs(df=5, loc=mean_return, scale=std_return))
 price_series.append(price)
 simulation_results.append(price_series)
 return simulation_results

def extract_entities(text):
 if NLP is None:
 logging.warning("SpaCy model not available")
 if st._is_running_with_streamlit:
 st.warning("Spacy model not installed. Install it with: python -m spacy download en_core_web_sm")
 return []
 try:
 doc = NLP(text)
 entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
 logging.debug(f"Extracted entities: {entities}")
 return entities
 except Exception as e:
 logging.error(f"Error in extract_entities: {str(e)}")
 return []

def get_trending_stocks():
 try:
 pytrends = TrendReq(hl='en-US', tz=360, retries=3, backoff_factor=2)
 trending = pytrends.trending_searches(pn='india')
 return trending[0][:5].tolist()
 except Exception as e:
 logging.error(f"Error fetching trending stocks: {str(e)}")
 if st._is_running_with_streamlit:
 st.warning("Unable to fetch trending stocks. Using fallback list.")
 return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]

def calculate_confidence_score(data, weights):
 score = 0
 total_weight = 0
 signals = []

 # RSI
 if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]):
 rsi = data['RSI'].iloc[-1]
 weight = weights.get('RSI', 1.0)
 if rsi < 30:
 score += weight * 1.0
 signals.append(('RSI', 'Oversold'))
 elif rsi > 70:
 score -= weight * 0.5
 signals.append(('RSI', 'Overbought'))
 total_weight += weight

 # MACD
 if 'MACD' in data.columns and 'MACD_signal' in data.columns and not pd.isna(data['MACD'].iloc[-1]):
 macd = data['MACD'].iloc[-1]
 macd_signal = data['MACD_signal'].iloc[-1]
 weight = weights.get('MACD', 1.0)
 if macd > macd_signal:
 score += weight * 1.0
 signals.append(('MACD', 'Bullish Crossover'))
 elif macd < macd_signal:
 score -= weight * 0.5
 signals.append(('MACD', 'Bearish Crossover'))
 total_weight += weight

 # Ichimoku
 if 'Ichimoku_Span_A' in data.columns and not pd.isna(data['Ichimoku_Span_A'].iloc[-1]):
 close = data['Close'].iloc[-1]
 span_a = data['Ichimoku_Span_A'].iloc[-1]
 span_b = data['Ichimoku_Span_B'].iloc[-1]
 weight = weights.get('Ichimoku', 1.0)
 if close > max(span_a, span_b):
 score += weight * 1.0
 signals.append(('Ichimoku', 'Above Cloud'))
 elif close < min(span_a, span_b):
 score -= weight * 0.5
 signals.append(('Ichimoku', 'Below Cloud'))
 total_weight += weight

 # ATR
 if 'ATR' in data.columns and not pd.isna(data['ATR'].iloc[-1]):
 atr_volatility = data['ATR'].iloc[-1] / data['Close'].iloc[-1]
 weight = weights.get('ATR', 1.0)
 if atr_volatility < 0.02:
 score += weight * 0.5
 signals.append(('ATR', 'Low Volatility'))
 elif atr_volatility > 0.05:
 score -= weight * 0.5
 signals.append(('ATR', 'High Volatility'))
 total_weight += weight

 # Bollinger Bands
 if 'Lower_Band' in data.columns and not pd.isna(data['Lower_Band'].iloc[-1]):
 close = data['Close'].iloc[-1]
 lower = data['Lower_Band'].iloc[-1]
 upper = data['Upper_Band'].iloc[-1]
 weight = weights.get('Bollinger', 1.0)
 if close < lower:
 score += weight * 0.8
 signals.append(('Bollinger', 'Below Lower Band'))
 elif close > upper:
 score -= weight * 0.8
 signals.append(('Bollinger', 'Above Upper Band'))
 total_weight += weight

 # Volume Spike
 if 'Volume_Spike' in data.columns and not pd.isna(data['Volume_Spike'].iloc[-1]):
 weight = weights.get('Volume_Spike', 1.0)
 if data['Volume_Spike'].iloc[-1]:
 score += weight * 0.6
 signals.append(('Volume_Spike', 'Significant Volume'))
 total_weight += weight

 confidence = score / max(total_weight, 1e-10) if total_weight > 0 else 0
 return min(max(confidence, 0), 1), signals

def assess_risk(data):
 if 'ATR' in data.columns and not pd.isna(data['ATR'].iloc[-1]):
 if data['ATR'].iloc[-1] > data['ATR'].mean() + data['ATR'].std():
 return "High Volatility Warning"
 return "Low Volatility"

def optimize_rsi_window(data, windows=range(5, 20)):
 best_window, best_sharpe = 14, -float('inf')
 returns = data['Close'].pct_change().dropna()
 if len(returns) < 50:
 logging.warning("Insufficient data for RSI optimization")
 return best_window
 for window in windows:
 rsi = ta.momentum.RSIIndicator(data['Close'], window=window).rsi()
 signals = (rsi < 30).astype(int) - (rsi > 70).astype(int)
 strategy_returns = signals.shift(1) * returns
 sharpe = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
 if sharpe > best_sharpe:
 best_sharpe, best_window = sharpe, window
 logging.debug(f"Best RSI window: {best_window}, Sharpe: {best_sharpe}")
 return best_window

def optimize_macd_params(data, fast_range=range(8, 20), slow_range=range(20, 40), signal_range=range(5, 15)):
 best_params = (12, 26, 9)
 best_sharpe = -float('inf')
 returns = data['Close'].pct_change().dropna()
 if len(returns) < 100:
 logging.warning("Insufficient data for MACD optimization")
 return best_params
 # Limit combinations to prevent overfitting
 combinations = list(itertools.product(fast_range, slow_range, signal_range))[:100]
 for fast, slow, signal in combinations:
 if fast >= slow:
 continue
 macd = ta.trend.MACD(data['Close'], window_fast=fast, window_slow=slow, window_sign=signal)
 signals = (macd.macd() > macd.macd_signal()).astype(int) - (macd.macd() < macd.macd_signal()).astype(int)
 strategy_returns = signals.shift(1) * returns
 sharpe = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
 if sharpe > best_sharpe:
 best_sharpe, best_params = sharpe, (fast, slow, signal)
 logging.debug(f"Best MACD params: {best_params}, Sharpe: {best_sharpe}")
 return best_params

def detect_divergence(data):
 if 'RSI' not in data.columns or 'Close' not in data.columns:
 return "No Divergence"
 rsi = data['RSI']
 price = data['Close']
 lookback = 10 # Extended lookback
 recent_highs = price[-lookback:].idxmax()
 recent_lows = price[-lookback:].idxmin()
 rsi_highs = rsi[-lookback:].idxmax()
 rsi_lows = rsi[-lookback:].idxmin()
 # Confirm with MACD
 macd_confirm = False
 if 'MACD' in data.columns and not pd.isna(data['MACD'].iloc[-1]):
 macd = data['MACD'].iloc[-1]
 macd_signal = data['MACD_signal'].iloc[-1]
 macd_confirm = macd > macd_signal if price[-1] > price[-2] else macd < macd_signal
 bullish_div = (recent_lows > rsi_lows) and (price[recent_lows] < price[-1]) and (rsi[rsi_lows] < rsi[-1]) and macd_confirm
 bearish_div = (recent_highs < rsi_highs) and (price[recent_highs] > price[-1]) and (rsi[rsi_highs] > rsi[-1]) and macd_confirm
 return "Bullish Divergence" if bullish_div else "Bearish Divergence" if bearish_div else "No Divergence"

def calculate_cmo(close, window=14):
 try:
 diff = close.diff()
 up_sum = diff.where(diff > 0, 0).rolling(window=window).sum()
 down_sum = abs(diff.where(diff < 0, 0)).rolling(window=window).sum()
 cmo = 100 * (up_sum - down_sum) / (up_sum + down_sum + 1e-10)
 return cmo.astype(np.float32)
 except Exception as e:
 logging.error(f"Failed to compute CMO: {str(e)}")
 return pd.Series([np.nan] * len(close), index=close.index, dtype=np.float32)

def calculate_pivot_points(data):
 try:
 high = data['High'].iloc[-1]
 low = data['Low'].iloc[-1]
 close = data['Close'].iloc[-1]
 pivot = (high + low + close) / 3
 support1 = (2 * pivot) - high
 resistance1 = (2 * pivot) - low
 support2 = pivot - (high - low)
 resistance2 = pivot + (high - low)
 return {
 'Pivot': np.float32(pivot),
 'Support1': np.float32(support1),
 'Resistance1': np.float32(resistance1),
 'Support2': np.float32(support2),
 'Resistance2': np.float32(resistance2)
 }
 except Exception as e:
 logging.error(f"Failed to compute Pivot Points: {str(e)}")
 return None

def calculate_heikin_ashi(data):
 try:
 ha_data = pd.DataFrame(index=data.index)
 ha_data['HA_Close'] = ((data['Open'] + data['High'] + data['Low'] + data['Close']) / 4).astype(np.float32)
 ha_data['HA_Open'] = ((data['Open'].shift(1) + data['Close'].shift(1)) / 2).fillna(data['Open'].iloc[0]).astype(np.float32)
 ha_data['HA_High'] = np.maximum.reduce([data['High'], ha_data['HA_Open'], ha_data['HA_Close']]).astype(np.float32)
 ha_data['HA_Low'] = np.minimum.reduce([data['Low'], ha_data['HA_Open'], ha_data['HA_Close']]).astype(np.float32)
 return ha_data[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]
 except Exception as e:
 logging.error(f"Failed to compute Heikin-Ashi: {str(e)}")
 return None

def calculate_adaptive_volume_spike(data, window=20):
 try:
 avg_volume = data['Volume'].rolling(window=window).mean()
 std_volume = data['Volume'].rolling(window=window).std()
 # Dynamic threshold based on recent volatility
 atr = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
 volatility_factor = atr.iloc[-1] / data['Close'].iloc[-1] if not pd.isna(atr.iloc[-1]) else 1
 threshold = avg_volume + (2 + volatility_factor) * std_volume
 return (data['Volume'] > threshold).astype(bool)
 except Exception as e:
 logging.error(f"Failed to compute adaptive volume spike: {str(e)}")
 return pd.Series([False] * len(data), index=data.index)

def train_ml_model(data, indicators):
 try:
 features = data[[ind for ind in indicators if ind in data.columns]].dropna()
 if features.empty or len(features) < 20:
 logging.warning("Insufficient data for ML model training")
 return None
 returns = data['Close'].pct_change().shift(-1).dropna()
 labels = (returns > 0).astype(int)
 features = features.loc[labels.index]
 if len(features) < 20:
 return None
 # Feature selection
 selector = SelectKBest(score_func=f_classif, k=min(10, len(indicators)))
 X_selected = selector.fit_transform(features, labels)
 selected_indices = selector.get_support(indices=True)
 selected_features = features.columns[selected_indices].tolist()
 logging.debug(f"Selected features: {selected_features}")
 X_train, X_test, y_train, y_test = train_test_split(X_selected, labels, test_size=0.2, random_state=42)
 model = RandomForestClassifier(n_estimators=100, random_state=42)
 model.fit(X_train, y_train)
 # Cross-validation
 cv_scores = cross_val_score(model, X_selected, labels, cv=5)
 accuracy = accuracy_score(y_test, model.predict(X_test))
 logging.info(f"ML model trained with accuracy: {accuracy}, CV mean: {cv_scores.mean()}")
 if cv_scores.mean() < 0.6:
 logging.warning("ML model has low cross-validation score")
 return None
 return model, selected_features
 except Exception as e:
 logging.error(f"Failed to train ML model: {str(e)}")
 return None

def update_indicator_weights(data, indicators):
 global INDICATOR_WEIGHTS
 try:
 model, selected_features = train_ml_model(data, indicators)
 if model is None:
 return INDICATOR_WEIGHTS
 importances = model.feature_importances_
 new_weights = {ind: INDICATOR_WEIGHTS.get(ind, 1.0) for ind in indicators}
 for i, feature in enumerate(selected_features):
 if feature in new_weights:
 new_weights[feature] = min(2.0, max(0.5, importances[i] * 2))
 INDICATOR_WEIGHTS.update(new_weights)
 logging.info(f"Updated indicator weights: {INDICATOR_WEIGHTS}")
 return INDICATOR_WEIGHTS
 except Exception as e:
 logging.error(f"Failed to update indicator weights: {str(e)}")
 return INDICATOR_WEIGHTS

def get_sector(symbol):
 for sector, stocks in SECTORS.items():
 if symbol in stocks:
 return sector
 return None

def analyze_stock(data, indicators=None, symbol=None):
 min_data_requirements = {
 'RSI': 14, 'MACD': 26, 'SMA_EMA': 200, 'Bollinger': 20, 'Stochastic': 14,
 'ATR': 14, 'ADX': 14, 'OBV': 2, 'VWAP': 2, 'Volume_Spike': 20,
 'Parabolic_SAR': 2, 'Fibonacci': 2, 'Divergence': 14, 'Ichimoku': 52,
 'CMF': 20, 'Donchian': 20, 'Keltner': 20, 'TRIX': 15, 'Ultimate_Osc': 28,
 'CMO': 14, 'VPT': 2, 'Pivot': 2, 'Heikin-Ashi': 2
 }

 if data.empty or len(data) < max(min_data_requirements.values(), default=2):
 logging.warning("Not enough data to analyze stock")
 if st._is_running_with_streamlit:
 st.warning("Not enough data to analyze this stock.")
 return None
 required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
 missing_cols = [col for col in required_columns if col not in data.columns]
 if missing_cols:
 logging.error(f"Missing required data: {', '.join(missing_cols)}")
 if st._is_running_with_streamlit:
 st.warning(f"Missing required data: {', '.join(missing_cols)}")
 return None

 data = data.copy()
 data = data.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

 all_indicators = [
 'RSI', 'MACD', 'SMA_EMA', 'Bollinger', 'Stochastic', 'ATR', 'ADX', 'OBV',
 'VWAP', 'Volume_Spike', 'Parabolic_SAR', 'Fibonacci', 'Divergence', 'Ichimoku',
 'CMF', 'Donchian', 'Keltner', 'TRIX', 'Ultimate_Osc', 'CMO', 'VPT', 'Pivot',
 'Heikin-Ashi'
 ]
 indicators = all_indicators if indicators is None else indicators
 computed_indicators = []

 indicator_order = ['ATR', 'RSI', 'MACD', 'SMA_EMA', 'Bollinger', 'Stochastic', 'ADX', 'OBV',
 'VWAP', 'Volume_Spike', 'Parabolic_SAR', 'Fibonacci', 'Divergence',
 'Ichimoku', 'CMF', 'Donchian', 'Keltner', 'TRIX', 'Ultimate_Osc',
 'CMO', 'VPT', 'Pivot', 'Heikin-Ashi']

 indicators = [ind for ind in indicator_order if ind in indicators]

 for indicator in indicators:
 if len(data) < min_data_requirements.get(indicator, 2):
 logging.warning(f"Not enough data for {indicator} (requires {min_data_requirements[indicator]} periods)")
 if st._is_running_with_streamlit:
 st.warning(f"Not enough data for {indicator} (requires {min_data_requirements[indicator]} periods).")
 continue
 try:
 if indicator == 'RSI':
 rsi_window = optimize_rsi_window(data)
 data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi().astype(np.float32)
 elif indicator == 'MACD':
 fast, slow, signal = optimize_macd_params(data)
 macd = ta.trend.MACD(data['Close'], window_slow=slow, window_fast=fast, window_sign=signal)
 data['MACD'] = macd.macd().astype(np.float32)
 data['MACD_signal'] = macd.macd_signal().astype(np.float32)
 data['MACD_hist'] = macd.macd_diff().astype(np.float32)
 elif indicator == 'SMA_EMA':
 data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator().astype(np.float32)
 data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator().astype(np.float32)
 data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator().astype(np.float32)
 data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator().astype(np.float32)
 elif indicator == 'Bollinger':
 bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
 data['Upper_Band'] = bollinger.bollinger_hband().astype(np.float32)
 data['Middle_Band'] = bollinger.bollinger_mavg().astype(np.float32)
 data['Lower_Band'] = bollinger.bollinger_lband().astype(np.float32)
 elif indicator == 'Stochastic':
 stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
 data['SlowK'] = stoch.stoch().astype(np.float32)
 data['SlowD'] = stoch.stoch_signal().astype(np.float32)
 elif indicator == 'ATR':
 data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range().astype(np.float32)
 elif indicator == 'ADX':
 data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx().astype(np.float32)
 elif indicator == 'OBV':
 data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume().astype(np.float32)
 elif indicator == 'VWAP':
 data['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3).astype(np.float32) * data['Volume']
 data['Cumulative_Volume'] = data['Volume'].cumsum().astype(np.float32)
 data['VWAP'] = (data['Cumulative_TP'].cumsum() / data['Cumulative_Volume']).astype(np.float32)
 elif indicator == 'Volume_Spike':
 data['Volume_Spike'] = calculate_adaptive_volume_spike(data)
 elif indicator == 'Parabolic_SAR':
 data['Parabolic_SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar().astype(np.float32)
 elif indicator == 'Fibonacci':
 lookback = min(len(data), 100)
 high = data['High'][-lookback:].max()
 low = data['Low'][-lookback:].min()
 diff = high - low
 data['Fib_23.6'] = np.float32(high - diff * 0.236)
 data['Fib_38.2'] = np.float32(high - diff * 0.382)
 data['Fib_50.0'] = np.float32(high - diff * 0.5)
 data['Fib_61.8'] = np.float32(high - diff * 0.618)
 elif indicator == 'Divergence':
 data['Divergence'] = detect_divergence(data)
 elif indicator == 'Ichimoku':
 ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
 data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line().astype(np.float32)
 data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line().astype(np.float32)
 data['Ichimoku_Span_A'] = ichimoku.ichimoku_a().astype(np.float32)
 data['Ichimoku_Span_B'] = ichimoku.ichimoku_b().astype(np.float32)
 data['Ichimoku_Chikou'] = data['Close'].shift(-26).astype(np.float32)
 elif indicator == 'CMF':
 data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow().astype(np.float32)
 elif indicator == 'Donchian':
 donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
 data['Donchian_Upper'] = donchian.donchian_channel_hband().astype(np.float32)
 data['Donchian_Lower'] = donchian.donchian_channel_lband().astype(np.float32)
 data['Donchian_Middle'] = donchian.donchian_channel_mband().astype(np.float32)
 elif indicator == 'Keltner':
 keltner = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'], window=20, window_atr=10)
 data['Keltner_Upper'] = keltner.keltner_channel_hband().astype(np.float32)
 data['Keltner_Middle'] = keltner.keltner_channel_mband().astype(np.float32)
 data['Keltner_Lower'] = keltner.keltner_channel_lband().astype(np.float32)
 elif indicator == 'TRIX':
 data['TRIX'] = ta.trend.TRIXIndicator(data['Close'], window=15).trix().astype(np.float32)
 elif indicator == 'Ultimate_Osc':
 data['Ultimate_Osc'] = ta.momentum.UltimateOscillator(
 data['High'], data['Low'], data['Close'], window1=7, window2=14, window3=28
 ).ultimate_oscillator().astype(np.float32)
 elif indicator == 'CMO':
 data['CMO'] = calculate_cmo(data['Close'], window=14)
 elif indicator == 'VPT':
 data['VPT'] = ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume']).volume_price_trend().astype(np.float32)
 elif indicator == 'Pivot':
 pivot_points = calculate_pivot_points(data)
 if pivot_points:
 data['Pivot'] = pivot_points['Pivot']
 data['Support1'] = pivot_points['Support1']
 data['Resistance1'] = pivot_points['Resistance1']
 data['Support2'] = pivot_points['Support2']
 data['Resistance2'] = pivot_points['Resistance2']
 elif indicator == 'Heikin-Ashi':
 ha_data = calculate_heikin_ashi(data)
 if ha_data is not None:
 data['HA_Open'] = ha_data['HA_Open']
 data['HA_High'] = ha_data['HA_High']
 data['HA_Low'] = ha_data['HA_Low']
 data['HA_Close'] = ha_data['HA_Close']
 computed_indicators.append(indicator)
 except Exception as e:
 logging.error(f"Failed to compute {indicator}: {str(e)}")
 if st._is_running_with_streamlit:
 st.warning(f"Unable to compute {indicator}: {str(e)}")

 data.drop(columns=[col for col in ['Cumulative_TP', 'Cumulative_Volume', 'MACD_hist'] if col in data.columns], inplace=True)

 if symbol and computed_indicators:
 # Update weights dynamically
 weights = update_indicator_weights(data, computed_indicators)
 model, selected_features = train_ml_model(data, computed_indicators)
 if model:
 joblib.dump(model, f"{symbol}_rf_model.pkl")
 data['ML_Score'] = model.predict_proba(data[[ind for ind in selected_features if ind in data.columns]])[:, 1]

 return data

def calculate_buy_at(data):
 if data is None or 'RSI' not in data.columns or pd.isna(data['RSI'].iloc[-1]):
 logging.warning("Cannot calculate Buy At price due to missing RSI data")
 return None
 last_close = data['Close'].iloc[-1]
 last_rsi = data['RSI'].iloc[-1]
 support = data['Support1'].iloc[-1] if 'Support1' in data.columns and not pd.isna(data['Support1'].iloc[-1]) else last_close * 0.99
 buy_at = support if last_rsi < 30 else last_close
 return round(float(buy_at), 2)

def calculate_stop_loss(data, atr_multiplier=2.5):
 if data is None or 'ATR' not in data.columns or pd.isna(data['ATR'].iloc[-1]):
 logging.warning("Cannot calculate Stop Loss due to missing ATR data")
 return None
 last_close = data['Close'].iloc[-1]
 last_atr = data['ATR'].iloc[-1]
 atr_std = data['ATR'].std() if data['ATR'].std() != 0 else 1
 atr_multiplier = 3.0 if 'ADX' in data.columns and not pd.isna(data['ADX'].iloc[-1]) and data['ADX'].iloc[-1] > 25 else 1.5
 atr_multiplier += (last_atr / atr_std) * 0.5
 stop_loss = last_close - (atr_multiplier * last_atr)
 if stop_loss < last_close * 0.80: # Relaxed cap for volatile markets
 stop_loss = last_close * 0.80
 return round(float(stop_loss), 2)

def calculate_target(data, risk_reward_ratio=3):
 if data is None:
 logging.warning("Cannot calculate Target price due to missing data")
 return None
 stop_loss = calculate_stop_loss(data)
 if stop_loss is None:
 logging.warning("Cannot calculate Target price due to missing Stop Loss")
 return None
 last_close = data['Close'].iloc[-1]
 risk = last_close - stop_loss
 adjusted_ratio = min(risk_reward_ratio, 5) if 'ADX' in data.columns and not pd.isna(data['ADX'].iloc[-1]) and data['ADX'].iloc[-1] > 30 else min(risk_reward_ratio, 3)
 target = last_close + (risk * adjusted_ratio)
 max_target = last_close * 1.5 if 'ADX' in data.columns and not pd.isna(data['ADX'].iloc[-1]) and data['ADX'].iloc[-1] > 30 else last_close * 1.3
 if target > max_target:
 target = max_target
 return round(float(target), 2)

def fetch_fundamentals(symbol):
 try:
 stock = yf.Ticker(symbol)
 info = stock.info
 if not info:
 logging.warning(f"No fundamentals data for {symbol}")
 return {
 'P/E': np.float32(50), 'EPS': np.float32(0), 'RevenueGrowth': np.float32(0),
 'Debt/Equity': np.float32(1), 'ROE': np.float32(0), 'FII_Activity': np.float32(0),
 'Promoter_Holding': np.float32(50)
 }
 pe = info.get('trailingPE', 50)
 pe = min(pe, 50) if pe != float('inf') else 50
 debt_equity = info.get('debtToEquity', 100) / 100 if info.get('debtToEquity') else 1
 roe = info.get('returnOnEquity', 0)
 # Placeholder for FII/DII activity and promoter holding (requires external API)
 fii_activity = info.get('netMoneyFlow', 0) if 'netMoneyFlow' in info else 0
 promoter_holding = info.get('promoterHolding', 50) if 'promoterHolding' in info else 50
 return {
 'P/E': np.float32(pe),
 'EPS': np.float32(info.get('trailingEps', 0)),
 'RevenueGrowth': np.float32(info.get('revenueGrowth', 0)),
 'Debt/Equity': np.float32(debt_equity),
 'ROE': np.float32(roe),
 'FII_Activity': np.float32(fii_activity),
 'Promoter_Holding': np.float32(promoter_holding)
 }
 except Exception as e:
 logging.error(f"Failed to fetch fundamentals for {symbol}: {str(e)}")
 if st._is_running_with_streamlit: 
 st.warning(f"Unable to fetch fundamental data for {symbol}.")
 return {
 'P/E': np.float32(50), 'EPS': np.float32(0), 'RevenueGrowth': np.float32(0),
 'Debt/Equity': np.float32(1), 'ROE': np.float32(0), 'FII_Activity': np.float32(0),
 'Promoter_Holding': np.float32(50)
 }

def generate_recommendations(data, symbol=None):
 default_recommendations = {
 "Intraday": "N/A", "Swing": "N/A", "Short-Term": "N/A", "Long-Term": "N/A",
 "Mean_Reversion": "N/A", "Breakout": "N/A", "Ichimoku_Trend": "N/A",
 "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None,
 "Score": 0, "Net_Score": 0, "Confidence": 0, "Signals": []
 }

 if data is None or data.empty or 'Close' not in data.columns or pd.isna(data['Close'].iloc[-1]):
 return default_recommendations

 buy_score = 0
 sell_score = 0
 momentum_score = 0
 max_momentum_score = sum([INDICATOR_WEIGHTS.get(ind, 1.0) for ind in ['RSI', 'CMO', 'Ultimate_Osc', 'TRIX']])
 
 # Apply sector-specific weights
 sector = get_sector(symbol) if symbol else None
 weights = INDICATOR_WEIGHTS.copy()
 if sector and sector in SECTOR_WEIGHT_ADJUSTMENTS:
 for ind, adj in SECTOR_WEIGHT_ADJUSTMENTS[sector].items():
 weights[ind] = weights.get(ind, 1.0) * adj

 if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]) and 0 <= data['RSI'].iloc[-1] <= 100:
 rsi = data['RSI'].iloc[-1]
 if rsi <= 30:
 momentum_score += weights['RSI'] * 2
 elif rsi >= 70:
 momentum_score -= weights['RSI'] * 2

 if 'MACD' in data.columns and 'MACD_signal' in data.columns and not pd.isna(data['MACD'].iloc[-1]):
 macd = data['MACD'].iloc[-1]
 macd_signal = data['MACD_signal'].iloc[-1]
 if macd > macd_signal:
 buy_score += weights['MACD']
 elif macd < macd_signal:
 sell_score += weights['MACD']

 if 'Lower_Band' in data.columns and not pd.isna(data['Lower_Band'].iloc[-1]):
 close = data['Close'].iloc[-1]
 lower = data['Lower_Band'].iloc[-1]
 upper = data['Upper_Band'].iloc[-1]
 if close < lower:
 buy_score += weights['Bollinger']
 elif close > upper:
 sell_score += weights['Bollinger']

 if 'VWAP' in data.columns and not pd.isna(data['VWAP'].iloc[-1]):
 vwap = data['VWAP'].iloc[-1]
 close = data['Close'].iloc[-1]
 if close > vwap:
 buy_score += weights['VWAP']
 elif close < vwap:
 sell_score += weights['VWAP']

 if 'Volume_Spike' in data.columns and not pd.isna(data['Volume_Spike'].iloc[-1]):
 spike = data['Volume_Spike'].iloc[-1]
 close = data['Close'].iloc[-1]
 prev_close = data['Close'].iloc[-2]
 if spike and close > prev_close:
 buy_score += weights['Volume_Spike']
 elif spike and close < prev_close:
 sell_score += weights['Volume_Spike']

 if 'Divergence' in data.columns:
 divergence = data['Divergence'].iloc[-1]
 if divergence == "Bullish Divergence":
 buy_score += weights['Divergence']
 elif divergence == "Bearish Divergence":
 sell_score += weights['Divergence']

 if 'Ichimoku_Tenkan' in data.columns and not pd.isna(data['Ichimoku_Tenkan'].iloc[-1]):
 tenkan = data['Ichimoku_Tenkan'].iloc[-1]
 kijun = data['Ichimoku_Kijun'].iloc[-1]
 span_a = data['Ichimoku_Span_A'].iloc[-1]
 span_b = data['Ichimoku_Span_B'].iloc[-1]
 chikou = data['Ichimoku_Chikou'].iloc[-1]
 close = data['Close'].iloc[-1]
 cloud_top = max(span_a, span_b)
 cloud_bottom = min(span_a, span_b)
 if tenkan > kijun and close > cloud_top and chikou > close:
 buy_score += weights['Ichimoku'] * 2
 default_recommendations["Ichimoku_Trend"] = "Strong Buy"
 elif tenkan < kijun and close < cloud_bottom and chikou < close:
 sell_score += weights['Ichimoku'] * 2
 default_recommendations["Ichimoku_Trend"] = "Strong Sell"

 if 'CMF' in data.columns and not pd.isna(data['CMF'].iloc[-1]):
 cmf = data['CMF'].iloc[-1]
 if cmf > 0:
 buy_score += weights['CMF']
 elif cmf < 0:
 sell_score += weights['CMF']

 if 'Donchian_Upper' in data.columns and not pd.isna(data['Donchian_Upper'].iloc[-1]):
 close = data['Close'].iloc[-1]
 upper = data['Donchian_Upper'].iloc[-1]
 lower = data['Donchian_Lower'].iloc[-1]
 if close > upper:
 buy_score += weights['Donchian']
 default_recommendations["Breakout"] = "Buy"
 elif close < lower:
 sell_score += weights['Donchian']
 default_recommendations["Breakout"] = "Sell"

 if 'RSI' in data.columns and 'Lower_Band' in data.columns and not pd.isna(data['RSI'].iloc[-1]):
 rsi = data['RSI'].iloc[-1]
 close = data['Close'].iloc[-1]
 lower = data['Lower_Band'].iloc[-1]
 upper = data['Upper_Band'].iloc[-1]
 if rsi < 30 and close >= lower:
 momentum_score += weights['RSI'] * 2
 default_recommendations["Mean_Reversion"] = "Buy"
 elif rsi > 70 and close >= upper:
 momentum_score -= weights['RSI'] * 2
 default_recommendations["Mean_Reversion"] = "Sell"

 if 'Keltner_Upper' in data.columns and not pd.isna(data['Keltner_Upper'].iloc[-1]):
 close = data['Close'].iloc[-1]
 upper = data['Keltner_Upper'].iloc[-1]
 lower = data['Keltner_Lower'].iloc[-1]
 if close < lower:
 buy_score += weights['Keltner']
 elif close > upper:
 sell_score += weights['Keltner']

 if 'TRIX' in data.columns and not pd.isna(data['TRIX'].iloc[-1]):
 trix = data['TRIX'].iloc[-1]
 prev_trix = data['TRIX'].iloc[-2]
 if trix > 0 and trix > prev_trix:
 momentum_score += weights['TRIX']
 elif trix < 0 and trix < prev_trix:
 momentum_score -= weights['TRIX']

 if 'Ultimate_Osc' in data.columns and not pd.isna(data['Ultimate_Osc'].iloc[-1]):
 uo = data['Ultimate_Osc'].iloc[-1]
 if uo < 30:
 momentum_score += weights['Ultimate_Osc']
 elif uo > 70:
 momentum_score -= weights['Ultimate_Osc']

 if 'CMO' in data.columns and not pd.isna(data['CMO'].iloc[-1]):
 cmo = data['CMO'].iloc[-1]
 if cmo < -50:
 momentum_score += weights['CMO']
 elif cmo > 50:
 momentum_score -= weights['CMO']

 if 'VPT' in data.columns and not pd.isna(data['VPT'].iloc[-1]):
 vpt = data['VPT'].iloc[-1]
 prev_vpt = data['VPT'].iloc[-2]
 if vpt > prev_vpt:
 buy_score += weights['VPT']
 elif vpt < prev_vpt:
 sell_score += weights['VPT']

 if 'Fib_23.6' in data.columns and not pd.isna(data['Fib_23.6'].iloc[-1]):
 current_price = data['Close'].iloc[-1]
 fib_levels = [data['Fib_23.6'].iloc[-1], data['Fib_38.2'].iloc[-1],
 data['Fib_50.0'].iloc[-1], data['Fib_61.8'].iloc[-1]]
 for level in fib_levels:
 if not pd.isna(level) and abs(current_price - level) / current_price < 0.01:
 if current_price > level:
 buy_score += weights['Fibonacci']
 else:
 sell_score += weights['Fibonacci']

 if 'Parabolic_SAR' in data.columns and not pd.isna(data['Parabolic_SAR'].iloc[-1]):
 sar = data['Parabolic_SAR'].iloc[-1]
 close = data['Close'].iloc[-1]
 if close > sar:
 buy_score += weights['Parabolic_SAR']
 elif close < sar:
 sell_score += weights['Parabolic_SAR']

 if 'OBV' in data.columns and not pd.isna(data['OBV'].iloc[-1]):
 obv = data['OBV'].iloc[-1]
 prev_obv = data['OBV'].iloc[-2]
 if obv > prev_obv:
 buy_score += weights['OBV']
 elif obv < prev_obv:
 sell_score += weights['OBV']

 if 'Pivot' in data.columns and not pd.isna(data['Pivot'].iloc[-1]):
 close = data['Close'].iloc[-1]
 pivot = data['Pivot'].iloc[-1]
 support1 = data['Support1'].iloc[-1]
 resistance1 = data['Resistance1'].iloc[-1]
 if abs(close - support1) / close < 0.01:
 buy_score += weights['Pivot']
 elif abs(close - resistance1) / close < 0.01:
 sell_score += weights['Pivot']

 if 'HA_Close' in data.columns and not pd.isna(data['HA_Close'].iloc[-1]):
 ha_close = data['HA_Close'].iloc[-1]
 ha_open = data['HA_Open'].iloc[-1]
 prev_ha_close = data['HA_Close'].iloc[-2]
 prev_ha_open = data['HA_Open'].iloc[-2]
 if ha_close > ha_open and prev_ha_close > prev_ha_open:
 buy_score += weights['Heikin-Ashi']
 elif ha_close < ha_open and prev_ha_close < prev_ha_open:
 sell_score += weights['Heikin-Ashi']

 if symbol:
 fundamentals = fetch_fundamentals(symbol)
 if fundamentals['P/E'] < 20 and fundamentals['EPS'] > 0:
 buy_score += 1.0
 elif fundamentals['P/E'] > 30 or fundamentals['EPS'] < 0:
 sell_score += 1.0
 if fundamentals['RevenueGrowth'] > 0.1:
 buy_score += 0.5
 elif fundamentals['RevenueGrowth'] < 0:
 sell_score += 0.5
 if fundamentals['Debt/Equity'] < 0.5:
 buy_score += 0.5
 elif fundamentals['Debt/Equity'] > 1.5:
 sell_score += 0.5
 if fundamentals['ROE'] > 0.15:
 buy_score += 0.5
 elif fundamentals['ROE'] < 0:
 sell_score += 0.5
 if fundamentals['FII_Activity'] > 0:
 buy_score += 0.3
 elif fundamentals['FII_Activity'] < 0:
 sell_score += 0.3
 if fundamentals['Promoter_Holding'] > 60:
 buy_score += 0.3
 elif fundamentals['Promoter_Holding'] < 30:
 sell_score += 0.3

 momentum_score = min(momentum_score, max_momentum_score) if momentum_score > 0 else max(momentum_score, -max_momentum_score)
 buy_score += momentum_score if momentum_score > 0 else 0
 sell_score += abs(momentum_score) if momentum_score < 0 else 0

 ml_score = data['ML_Score'].iloc[-1] if 'ML_Score' in data.columns and not pd.isna(data['ML_Score'].iloc[-1]) else 0.5
 buy_score += ml_score * 1.5 # Reduced ML influence
 sell_score += (1 - ml_score) * 1.5

 total_signals = max(buy_score + sell_score, 10) # Increased minimum signals
 net_score = (buy_score - sell_score) / total_signals * 5
 confidence, signals = calculate_confidence_score(data, weights)
 default_recommendations["Signals"] = signals

 # Dynamic thresholds based on volatility
 atr_volatility = data['ATR'].iloc[-1] / data['Close'].iloc[-1] if 'ATR' in data.columns and not pd.isna(data['ATR'].iloc[-1]) else 0.02
 strong_threshold = 2.5 + atr_volatility * 2
 weak_threshold = 1.0 + atr_volatility

 if net_score >= strong_threshold and confidence > 0.7:
 default_recommendations["Intraday"] = "Strong Buy"
 default_recommendations["Swing"] = "Buy" if net_score >= strong_threshold - 0.5 else "Hold"
 default_recommendations["Short-Term"] = "Buy" if net_score >= strong_threshold - 1 else "Hold"
 default_recommendations["Long-Term"] = "Buy" if net_score >= strong_threshold - 1.5 else "Hold"
 elif net_score >= weak_threshold:
 default_recommendations["Intraday"] = "Buy"
 default_recommendations["Swing"] = "Hold"
 default_recommendations["Short-Term"] = "Hold"
 default_recommendations["Long-Term"] = "Hold"
 elif net_score <= -strong_threshold and confidence > 0.7:
 default_recommendations["Intraday"] = "Strong Sell"
 default_recommendations["Swing"] = "Sell" if net_score <= -strong_threshold + 0.5 else "Hold"
 default_recommendations["Short-Term"] = "Sell" if net_score <= -strong_threshold + 1 else "Hold"
 default_recommendations["Long-Term"] = "Sell" if net_score <= -strong_threshold + 1.5 else "Hold"
 elif net_score <= -weak_threshold:
 default_recommendations["Intraday"] = "Sell"
 default_recommendations["Swing"] = "Hold"
 default_recommendations["Short-Term"] = "Hold"
 default_recommendations["Long-Term"] = "Hold"
 else:
 default_recommendations["Intraday"] = "Hold"

 default_recommendations["Current Price"] = float(data['Close'].iloc[-1]) if not pd.isna(data['Close'].iloc[-1]) else None
 default_recommendations["Buy At"] = calculate_buy_at(data)
 default_recommendations["Stop Loss"] = calculate_stop_loss(data)
 default_recommendations["Target"] = calculate_target(data)
 default_recommendations["Score"] = buy_score - sell_score
 default_recommendations["Net_Score"] = round(net_score, 2)
 default_recommendations["Confidence"] = round(confidence, 2)

 return default_recommendations

def analyze_batch(stock_batch, batch_size, timeout=60):
 results = []
 errors = []
 max_workers = min(batch_size, os.cpu_count() or 4)
 with ProcessPoolExecutor(max_workers=max_workers) as executor:
 futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
 for future in as_completed(futures, timeout=timeout):
 symbol = futures[future]
 try:
 result = future.result()
 if result:
 results.append(result)
 except Exception as e:
 errors.append(f"Error analyzing {symbol}: {str(e)}")
 logging.error(f"Error analyzing {symbol}: {str(e)}")
 gc.collect()
 for error in errors:
 if st._is_running_with_streamlit:
 st.warning(error)
 return results

def analyze_stock_parallel(symbol):
 data = fetch_stock_data_cached(symbol)
 if not data.empty:
 data = analyze_stock(data, symbol=symbol)
 if data is None:
 return None
 recommendations = generate_recommendations(data, symbol)
 return {
 "Symbol": symbol,
 "Current Price": recommendations["Current Price"],
 "Buy At": recommendations["Buy At"],
 "Stop Loss": recommendations["Stop Loss"],
 "Target": recommendations["Target"],
 "Intraday": recommendations["Intraday"],
 "Swing": recommendations["Swing"],
 "Short-Term": recommendations ["Short-Term"],
 "Long-Term": recommendations["Long-Term"],
 "Mean_Reversion": recommendations["Mean_Reversion"],
 "Breakout": recommendations["Breakout"],
 "Ichimoku_Trend": recommendations["Ichimoku_Trend"],
 "Score": recommendations.get("Score", 0),
 "Net_Score": recommendations.get("Net_Score", 0),
 "Confidence": recommendations.get("Confidence", 0),
 "Signals": recommendations.get("Signals", [])
 }
 return None

def analyze_all_stocks(stock_list, batch_size=10, progress_callback=None):
 filtered_stocks = []
 for symbol in stock_list:
 data = fetch_stock_data_cached(symbol)
 if not data.empty and data['Close'].iloc[-1] > 10:
 filtered_stocks.append(symbol)

 results = []
 total_batches = (len(filtered_stocks) // batch_size) + (1 if len(filtered_stocks) % batch_size != 0 else 0)
 for i in range(0, len(filtered_stocks), batch_size):
 batch = filtered_stocks[i:i + batch_size]
 batch_results = analyze_batch(batch, batch_size)
 results.extend(batch_results)
 if progress_callback:
 progress_callback((i + len(batch)) / len(filtered_stocks))
 memory_usage = psutil.Process().memory_info().rss / 1024**2
 if memory_usage > 2000:
 gc.collect()
 logging.warning(f"High memory usage detected: {memory_usage} MB")

 results_df = pd.DataFrame([r for r in results if r is not None])
 if results_df.empty:
 if st._is_running_with_streamlit:
 st.warning("No valid stock data retrieved.")
 return pd.DataFrame()
 for col in ["Score", "Net_Score", "Confidence", "Current Price"]:
 if col not in results_df.columns:
 results_df[col] = 0 if col != "Current Price" else None
 return results_df.sort_values(by=["Confidence", "Net_Score"], ascending=False).head(10)

def analyze_intraday_stocks(stock_list, batch_size=10, progress_callback=None):
 filtered_stocks = []
 for symbol in stock_list:
 data = fetch_stock_data_cached(symbol, period="1mo", interval="15m")
 if not data.empty and data['Close'].iloc[-1] > 10:
 filtered_stocks.append(symbol)

 results = []
 total_batches = (len(filtered_stocks) // batch_size) + (1 if len(filtered_stocks) % batch_size != 0 else 0)
 for i in range(0, len(filtered_stocks), batch_size):
 batch = filtered_stocks[i:i + batch_size]
 batch_results = analyze_batch(batch, batch_size)
 results.extend(batch_results)
 if progress_callback:
 progress_callback((i + len(batch)) / len(filtered_stocks))
 memory_usage = psutil.Process().memory_info().rss / 1024**2
 if memory_usage > 2000:
 gc.collect()
 logging.warning(f"High memory usage detected: {memory_usage} MB")

 results_df = pd.DataFrame([r for r in results if r is not None])
 if results_df.empty:
 if st._is_running_with_streamlit:
 st.warning("No valid intraday stock data retrieved.")
 return pd.DataFrame()
 if "Net_Score" not in results_df.columns:
 results_df["Net_Score"] = 0
 if "Confidence" not in results_df.columns:
 results_df["Confidence"] = 0
 return results_df[results_df["Intraday"].isin(["Buy", "Strong Buy"])].sort_values(by=["Confidence", "Net_Score"], ascending=False).head(5)

def colored_recommendation(recommendation):
 if "Buy" in recommendation:
 return f"🟢 {recommendation}"
 elif "Sell" in recommendation:
 return f"🔴 {recommendation}"
 elif "Hold" in recommendation:
 return f"🟡 {recommendation}"
 else:
 return recommendation

def update_progress(progress_bar, loading_text, progress, messages):
 progress_bar.progress(min(progress, 1.0))
 loading_text.text(next(messages))

def display_dashboard(symbol=None, data=None, recommendations=None, selected_stocks=None, stock_list=None):
 st.title("📊 StockGenie Pro - NSE Analysis")
 st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")

 ad_ratio = calculate_advance_decline_ratio(stock_list[:50])
 st.metric("Market Sentiment (A/D Ratio)", f"{ad_ratio:.2f}", delta="Bullish" if ad_ratio > 1 else "Bearish")

 if st.button("🚀 Generate Daily Top Picks"):
 progress_bar = st.progress(0)
 loading_text = st.empty()
 loading_messages = itertools.cycle([
 "Analyzing trends...", "Fetching data...", "Crunching numbers...",
 "Evaluating indicators...", "Finalizing results..."
 ])
 results_df = analyze_all_stocks(
 selected_stocks,
 batch_size=st.session_state.get('batch_size', 10),
 progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages)
 )
 progress_bar.empty()
 loading_text.empty()
 if not results_df.empty:
 st.subheader("🏆 Today's Top 10 Stocks")
 for _, row in results_df.iterrows():
 with st.expander(f"{row['Symbol']} - Confidence: {row['Confidence']}, Net Score: {row['Net_Score']}"):
 current_price = row['Current Price'] if pd.notnull(row['Current Price']) else "N/A"
 buy_at = row['Buy At'] if pd.notnull(row['Buy At']) else "N/A"
 stop_loss = row['Stop Loss'] if pd.notnull(row['Stop Loss']) else "N/A"
 target = row['Target'] if pd.notnull(row['Target']) else "N/A"
 signals = "; ".join([f"{s[0]}: {s[1]}" for s in row['Signals']]) if row['Signals'] else "N/A"
 st.markdown(f"""
 {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price} 
 Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss} 
 Target: ₹{target} 
 Intraday: {colored_recommendation(row['Intraday'])} 
 Swing: {colored_recommendation(row['Swing'])} 
 Short-Term: {colored_recommendation(row['Short-Term'])} 
 Long-Term: {colored_recommendation(row['Long-Term'])} 
 Mean Reversion: {colored_recommendation(row['Mean_Reversion'])} 
 Breakout: {colored_recommendation(row['Breakout'])} 
 Ichimoku Trend: {colored_recommendation(row['Ichimoku_Trend'])} 
 Signals: {signals}
 """)
 else:
 st.warning("No top picks available due to data issues.")

 if st.button("⚡ Generate Intraday Top 5 Picks"):
 progress_bar = st.progress(0)
 loading_text = st.empty()
 loading_messages = itertools.cycle([
 "Scanning intraday trends...", "Detecting buy signals...", "Calculating stop-loss levels...",
 "Optimizing targets...", "Finalizing top picks..."
 ])
 intraday_results = analyze_intraday_stocks(
 selected_stocks,
 batch_size=st.session_state.get('batch_size', 10),
 progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages)
 )
 progress_bar.empty()
 loading_text.empty()
 if not intraday_results.empty:
 st.subheader("🏆 Top 5 Intraday Stocks")
 for _, row in intraday_results.iterrows():
 with st.expander(f"{row['Symbol']} - Confidence: {row['Confidence']}, Net Score: {row['Net_Score']}"):
 current_price = row['Current Price'] if pd.notnull(row['Current Price']) else "N/A"
 buy_at = row['Buy At'] if pd.notnull(row['Buy At']) else "N/A"
 stop_loss = row['Stop Loss'] if pd.notnull(row['Stop Loss']) else "N/A"
 target = row['Target'] if pd.notnull(row['Target']) else "N/A"
 signals = "; ".join([f"{s[0]}: {s[1]}" for s in row['Signals']]) if row['Signals'] else "N/A"
 st.markdown(f"""
 {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price} 
 Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss} 
 Target: ₹{target} 
 Intraday: {colored_recommendation(row['Intraday'])} 
 Signals: {signals}
 """)
 else:
 st.warning("No intraday picks available due to data issues.")

 if st.button("🔍 Show Trending Stocks"):
 trending = get_trending_stocks()
 if trending:
 st.subheader("🔥 Trending Stocks in India")
 st.write(", ".join(trending))
 else:
 st.warning("No trending stocks data available.")

 if symbol and data is not None and recommendations is not None:
 st.header(f"📋 {symbol.split('.')[0]} Analysis")
 col1, col2, col3, col4 = st.columns(4)
 with col1:
 current_price = recommendations['Current Price'] if recommendations['Current Price'] is not None else "N/A"
 st.metric(tooltip("Current Price", TOOLTIPS['Stop Loss']), f"₹{current_price}")
 with col2:
 buy_at = recommendations['Buy At'] if recommendations['Buy At'] is not None else "N/A"
 st.metric(tooltip("Buy At", "Recommended entry price"), f"₹{buy_at}")
         with col3:
            stop_loss = recommendations['Stop Loss'] if recommendations['Stop Loss'] is not None else "N/A"
            st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), f"₹{stop_loss}")
        with col4:
            target = recommendations['Target'] if recommendations['Target'] is not None else "N/A"
            st.metric(tooltip("Target", "Projected price target"), f"₹{target}")

        st.subheader("📈 Trading Recommendations")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric(tooltip("Intraday", "Recommendation for day trading"), colored_recommendation(recommendations['Intraday']))
        with col6:
            st.metric(tooltip("Swing", "Recommendation for 1-2 week trades"), colored_recommendation(recommendations['Swing']))
        with col7:
            st.metric(tooltip("Short-Term", "Recommendation for 1-3 month trades"), colored_recommendation(recommendations['Short-Term']))
        with col8:
            st.metric(tooltip("Long-Term", "Recommendation for 6+ month trades"), colored_recommendation(recommendations['Long-Term']))

        st.subheader("🔍 Additional Strategies")
        col9, col10, col11 = st.columns(3)
        with col9:
            st.metric(tooltip("Mean Reversion", "Strategy based on price reverting to mean"), colored_recommendation(recommendations['Mean_Reversion']))
        with col10:
            st.metric(tooltip("Breakout", "Strategy based on price breaking key levels"), colored_recommendation(recommendations['Breakout']))
        with col11:
            st.metric(tooltip("Ichimoku Trend", "Trend-following based on Ichimoku Cloud"), colored_recommendation(recommendations['Ichimoku_Trend']))

        st.subheader("📊 Technical Signals")
        signals = recommendations.get('Signals', [])
        if signals:
            st.write("Key Signals:")
            for signal in signals:
                st.write(f"- {signal[0]}: {signal[1]}")
        else:
            st.write("No significant signals detected.")

        st.subheader("📉 Price Chart with Indicators")
        if 'Close' in data.columns and not data['Close'].isna().all():
            fig = px.line(data, x=data.index, y='Close', title=f"{symbol} Price Trend")
            if 'SMA_50' in data.columns:
                fig.add_scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='orange'))
            if 'SMA_200' in data.columns:
                fig.add_scatter(x=data.index, y=data['SMA_200'], name='SMA 200', line=dict(color='purple'))
            if 'Upper_Band' in data.columns and 'Lower_Band' in data.columns:
                fig.add_scatter(x=data.index, y=data['Upper_Band'], name='Bollinger Upper', line=dict(color='red', dash='dash'))
                fig.add_scatter(x=data.index, y=data['Lower_Band'], name='Bollinger Lower', line=dict(color='red', dash='dash'))
            fig.update_layout(hovermode='x unified', xaxis_title="Date", yaxis_title="Price (₹)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No price data available for charting.")

        st.subheader("📊 Monte Carlo Simulation")
        simulations = monte_carlo_simulation(data)
        if simulations:
            sim_df = pd.DataFrame(simulations).T
            sim_df.index = pd.date_range(start=datetime.now(), periods=len(sim_df), freq='D')
            fig_sim = px.line(sim_df, title="Monte Carlo Price Projections")
            fig_sim.update_layout(xaxis_title="Date", yaxis_title="Projected Price (₹)", showlegend=False)
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.warning("Unable to perform Monte Carlo simulation due to insufficient data.")

        st.subheader("📊 Fundamental Analysis")
        fundamentals = fetch_fundamentals(symbol)
        col12, col13, col14, col15 = st.columns(4)
        with col12:
            st.metric(tooltip("P/E Ratio", TOOLTIPS['P/E']), f"{fundamentals['P/E']:.2f}")
        with col13:
            st.metric(tooltip("EPS", TOOLTIPS['EPS']), f"₹{fundamentals['EPS']:.2f}")
        with col14:
            st.metric(tooltip("Debt/Equity", TOOLTIPS['Debt/Equity']), f"{fundamentals['Debt/Equity']:.2f}")
        with col15:
            st.metric(tooltip("ROE", TOOLTIPS['ROE']), f"{fundamentals['ROE']*100:.2f}%")

        col16, col17, col18 = st.columns(3)
        with col16:
            st.metric(tooltip("Revenue Growth", "Annual revenue growth rate"), f"{fundamentals['RevenueGrowth']*100:.2f}%")
        with col17:
            st.metric(tooltip("FII/DII Activity", TOOLTIPS['FII/DII Activity']), f"{fundamentals['FII_Activity']:.2f}")
        with col18:
            st.metric(tooltip("Promoter Holding", TOOLTIPS['Promoter Holding']), f"{fundamentals['Promoter_Holding']:.2f}%")

        st.subheader("⚠️ Risk Assessment")
        risk = assess_risk(data)
        st.write(f"Risk Level: {risk}")

def main():
    st.set_page_config(page_title="StockGenie Pro", page_icon="📊", layout="wide")
    
    # Initialize session state
    if 'batch_size' not in st.session_state:
        st.session_state['batch_size'] = 10
    if 'selected_stocks' not in st.session_state:
        st.session_state['selected_stocks'] = []

    st.sidebar.header("⚙️ Settings")
    sector = st.sidebar.selectbox("Select Sector", ["All"] + list(SECTORS.keys()))
    batch_size = st.sidebar.slider("Batch Size", min_value=5, max_value=20, value=10)
    st.session_state['batch_size'] = batch_size

    stock_list = fetch_nse_stock_list()
    if sector == "All":
        selected_stocks = stock_list
    else:
        selected_stocks = SECTORS.get(sector, [])
    st.session_state['selected_stocks'] = selected_stocks

    st.sidebar.header("🔍 Stock Search")
    symbol_input = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS)", "").upper()
    if symbol_input and symbol_input in stock_list:
        data = fetch_stock_data_cached(symbol_input)
        if not data.empty:
            analyzed_data = analyze_stock(data, symbol=symbol_input)
            if analyzed_data is not None:
                recommendations = generate_recommendations(analyzed_data, symbol_input)
                display_dashboard(
                    symbol=symbol_input,
                    data=analyzed_data,
                    recommendations=recommendations,
                    selected_stocks=selected_stocks,
                    stock_list=stock_list
                )
            else:
                st.error(f"Unable to analyze {symbol_input}. Please check the symbol or try again later.")
        else:
            st.error(f"No data available for {symbol_input}. Please check the symbol.")
    else:
        display_dashboard(
            selected_stocks=selected_stocks,
            stock_list=stock_list
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        if st._is_running_with_streamlit:
            st.error(f"An unexpected error occurred: {str(e)}")
