import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import sqlite3
import requests
import pyotp
import time
import random
import os
from smartapi import SmartConnect

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/124.0.2478.80 Safari/537.36"
]

# Angel One Smart API Credentials
CLIENT_ID = "AAAG399109"
PASSWORD = "1503"
TOTP_SECRET = "OLRQ3CYBLPN2XWQPHLKMB7WEKI"
HISTORICAL_API_KEY = "c3C0tMGn"
TRADING_API_KEY = "ruseeaBq"
MARKET_API_KEY = "PflRFXyd"

# Database setup
def init_database():
    conn = sqlite3.connect('backtest_results.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS backtest_results (
            symbol TEXT,
            strategy TEXT,
            start_date TEXT,
            end_date TEXT,
            total_return REAL,
            annualized_return REAL,
            max_drawdown REAL,
            sharpe_ratio REAL,
            win_rate REAL,
            profit_factor REAL,
            avg_trade_duration INTEGER,
            num_trades INTEGER,
            PRIMARY KEY (symbol, strategy, start_date, end_date)
        )
    ''')
    conn.close()

# Smart API Session Manager
class SmartAPISession:
    def __init__(self):
        self.client = SmartConnect(api_key=HISTORICAL_API_KEY)
        self.totp = pyotp.TOTP(TOTP_SECRET)
        self.session = None

    def login(self):
        try:
            totp_code = self.totp.now()
            self.session = self.client.generateSession(CLIENT_ID, PASSWORD, totp_code)
            if self.session['status']:
                print("Smart API login successful")
                return True
            else:
                print("Smart API login failed:", self.session['message'])
                return False
        except Exception as e:
            print(f"Error during Smart API login: {str(e)}")
            return False

    def get_client(self):
        if not self.session:
            self.login()
        return self.client

# Data fetching functions
def retry(max_retries=5, delay=5, backoff_factor=2, jitter=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                        retries += 1
                        if retries == max_retries:
                            raise e
                        sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                        print(f"Rate limit hit. Retrying after {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    else:
                        retries += 1
                        if retries == max_retries:
                            raise e
                        sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                        time.sleep(sleep_time)
        return wrapper
    return decorator

@retry(max_retries=5, delay=5)
def fetch_stock_data(symbol, period="5y", interval="1d"):
    try:
        # Initialize Smart API session
        session = SmartAPISession()
        client = session.get_client()
        if not client:
            return pd.DataFrame()

        # Convert Yahoo Finance symbol to Smart API format (e.g., RELIANCE.NS -> RELIANCE-EQ)
        symbol_name = symbol.replace(".NS", "-EQ")
        
        # Map symbol to token (simplified; ideally, fetch from instrument list)
        symbol_map = {
            "RELIANCE-EQ": "2885",
            "TATASTEEL-EQ": "3499",
            "HDFCBANK-EQ": "1333",
            "INFY-EQ": "1594",
            "ICICIBANK-EQ": "1394",
            "BHARTIARTL-EQ": "106",
            "ITC-EQ": "1660",
            "KOTAKBANK-EQ": "1922",
            "HINDUNILVR-EQ": "1348",
            "SBIN-EQ": "3045",
            "BAJFINANCE-EQ": "317",
            "TCS-EQ": "11536"
        }
        
        if symbol_name not in symbol_map:
            print(f"Symbol {symbol_name} not found in token map")
            return pd.DataFrame()

        symbol_token = symbol_map[symbol_name]
        
        # Calculate date range
        end_date = datetime.now()
        if period == "5y":
            start_date = end_date - timedelta(days=5*365)
        else:
            print(f"Unsupported period: {period}")
            return pd.DataFrame()

        # Fetch historical data
        historical_data = client.getCandleData({
            "exchange": "NSE",
            "symboltoken": symbol_token,
            "interval": "ONE_DAY" if interval == "1d" else interval,
            "fromdate": start_date.strftime('%Y-%m-%d %H:%M'),
            "todate": end_date.strftime('%Y-%m-%d %H:%M')
        })

        if historical_data['status'] and historical_data['data']:
            data = historical_data['data']
            df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df.dropna()
        else:
            print(f"No data returned for {symbol}: {historical_data.get('message', 'Unknown error')}")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

# Technical analysis functions
def analyze_stock(data):
    if data.empty or len(data) < 27:
        return data

    # RSI with optimized window
    def optimize_rsi_window(data, windows=range(5, 15)):
        best_window, best_sharpe = 9, -float('inf')
        returns = data['Close'].pct_change().dropna()
        if len(returns) < 50:
            return best_window
        for window in windows:
            rsi = ta.momentum.RSIIndicator(data['Close'], window=window).rsi()
            signals = (rsi < 30).astype(int) - (rsi > 70).astype(int)
            strategy_returns = signals.shift(1) * returns
            sharpe = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
            if sharpe > best_sharpe:
                best_sharpe, best_window = sharpe, window
        return best_window

    rsi_window = optimize_rsi_window(data)
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
    
    # MACD
    macd = ta.trend.MACD(data['Close'], window_slow=17, window_fast=8, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    data['MACD_hist'] = macd.macd_diff()
    
    # Moving Averages
    data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
    data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
    data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
    data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data['Upper_Band'] = bollinger.bollinger_hband()
    data['Middle_Band'] = bollinger.bollinger_mavg()
    data['Lower_Band'] = bollinger.bollinger_lband()
    
    # ATR
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    
    # ADX
    if len(data) >= 27:
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
    else:
        data['ADX'] = None
    
    # Volume indicators
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    data['Avg_Volume'] = data['Volume'].rolling(window=10).mean()
    data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 1.5)
    
    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
    data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
    data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
    data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
    data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
    data['Ichimoku_Chikou'] = data['Close'].shift(-26)
    
    return data

# Strategy evaluation functions
def generate_recommendations(data):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold",
        "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold"
    }
    
    if data.empty or len(data) < 27:
        return recommendations
    
    buy_score = 0
    sell_score = 0
    
    # RSI scoring
    if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None:
        if data['RSI'].iloc[-1] <= 20:
            buy_score += 4
        elif data['RSI'].iloc[-1] < 30:
            buy_score += 2
        elif data['RSI'].iloc[-1] > 70:
            sell_score += 2
    
    # MACD scoring
    if 'MACD' in data.columns and 'MACD_signal' in data.columns:
        if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
            buy_score += 1
        elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
            sell_score += 1
    
    # Bollinger Bands scoring
    if 'Close' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns:
        if data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]:
            buy_score += 1
        elif data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]:
            sell_score += 1
    
    # Ichimoku scoring
    if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns:
        if data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
            buy_score += 1
            recommendations["Ichimoku_Trend"] = "Buy"
        elif data['Close'].iloc[-1] < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
            sell_score += 1
            recommendations["Ichimoku_Trend"] = "Sell"
    
    # Volume scoring
    if 'Volume_Spike' in data.columns and data['Volume_Spike'].iloc[-1]:
        if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
            buy_score += 1
        else:
            sell_score += 1
    
    # Determine final recommendations
    net_score = buy_score - sell_score
    if buy_score > sell_score and buy_score >= 4:
        recommendations["Intraday"] = "Strong Buy"
        recommendations["Swing"] = "Buy" if buy_score >= 3 else "Hold"
        recommendations["Short-Term"] = "Buy" if buy_score >= 2 else "Hold"
        recommendations["Long-Term"] = "Buy" if buy_score >= 1 else "Hold"
    elif sell_score > buy_score and sell_score >= 4:
        recommendations["Intraday"] = "Strong Sell"
        recommendations["Swing"] = "Sell" if sell_score >= 3 else "Hold"
        recommendations["Short-Term"] = "Sell" if sell_score >= 2 else "Hold"
        recommendations["Long-Term"] = "Sell" if sell_score >= 1 else "Hold"
    elif net_score > 0:
        recommendations["Intraday"] = "Buy" if net_score >= 3 else "Hold"
        recommendations["Swing"] = "Buy" if net_score >= 2 else "Hold"
        recommendations["Short-Term"] = "Buy" if net_score >= 1 else "Hold"
        recommendations["Long-Term"] = "Hold"
    elif net_score < 0:
        recommendations["Intraday"] = "Sell" if net_score <= -3 else "Hold"
        recommendations["Swing"] = "Sell" if net_score <= -2 else "Hold"
        recommendations["Short-Term"] = "Sell" if net_score <= -1 else "Hold"
        recommendations["Long-Term"] = "Hold"
    
    return recommendations

# Backtesting engine
def backtest_strategy(data, strategy="Swing", initial_capital=25000):
    if data.empty or len(data) < 200:
        return None
    
    # Initialize variables
    capital = initial_capital
    position = 0
    trades = []
    equity_curve = []
    max_capital = initial_capital
    max_drawdown = 0
    
    # Main backtesting loop
    for i in range(200, len(data) - 1):
        current_data = data.iloc[:i+1]
        current_data = analyze_stock(current_data)
        recommendations = generate_recommendations(current_data)
        
        row = data.iloc[i]
        next_row = data.iloc[i + 1]
        
        # Entry signal
        if position == 0 and recommendations[strategy] in ["Buy", "Strong Buy"]:
            entry_price = next_row['Open']
            shares = capital // entry_price
            if shares > 0:
                position = shares
                capital -= shares * entry_price
                trades.append({
                    "entry_date": next_row.name,
                    "entry_price": entry_price,
                    "shares": shares
                })
        
        # Exit signal
        elif position > 0 and recommendations[strategy] in ["Sell", "Strong Sell"]:
            exit_price = next_row['Open']
            capital += position * exit_price
            trades[-1]["exit_date"] = next_row.name
            trades[-1]["exit_price"] = exit_price
            trades[-1]["return"] = (exit_price - trades[-1]["entry_price"]) / trades[-1]["entry_price"]
            position = 0
        
        # Update equity curve and drawdown
        current_value = capital + (position * next_row['Open'])
        equity_curve.append(current_value)
        max_capital = max(max_capital, current_value)
        drawdown = (max_capital - current_value) / max_capital
        max_drawdown = max(max_drawdown, drawdown)
    
    # Close any open position at the end
    if position > 0:
        exit_price = data['Close'].iloc[-1]
        capital += position * exit_price
        trades[-1]["exit_date"] = data.index[-1]
        trades[-1]["exit_price"] = exit_price
        trades[-1]["return"] = (exit_price - trades[-1]["entry_price"]) / trades[-1]["entry_price"]
    
    # Calculate performance metrics
    final_value = capital
    total_return = (final_value - initial_capital) / initial_capital
    annualized_return = (1 + total_return) ** (365 / ((data.index[-1] - data.index[200]).days)) - 1
    
    winning_trades = [t for t in trades if "return" in t and t["return"] > 0]
    losing_trades = [t for t in trades if "return" in t and t["return"] <= 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0
    
    avg_win = np.mean([t["return"] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t["return"] for t in losing_trades]) if losing_trades else 0
    profit_factor = (avg_win * len(winning_trades)) / (abs(avg_loss) * len(losing_trades)) if losing_trades else float('inf')
    
    trade_durations = [(t["exit_date"] - t["entry_date"]).days for t in trades if "exit_date" in t]
    avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
    
    returns = pd.Series([t["return"] for t in trades if "return" in t])
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0
    
    return {
        "total_return": total_return * 100,
        "annualized_return": annualized_return * 100,
        "max_drawdown": max_drawdown * 100,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate * 100,
        "profit_factor": profit_factor,
        "avg_trade_duration": avg_trade_duration,
        "num_trades": len(trades),
        "trades": trades,
        "equity_curve": equity_curve
    }

# Visualization functions
def plot_equity_curve(results, symbol, strategy):
    plt.figure(figsize=(12, 6))
    plt.plot(results['equity_curve'], label='Equity Curve')
    plt.title(f"{symbol} - {strategy} Strategy Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value (₹)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_trade_distribution(results):
    returns = [t["return"]*100 for t in results["trades"] if "return" in t]
    if not returns:
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=20, edgecolor='black')
    plt.title("Trade Return Distribution")
    plt.xlabel("Return (%)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# Main backtesting function
def backtest_stock(symbol, period="5y", strategies=["Swing", "Intraday"]):
    print(f"\n🔍 Backtesting {symbol}...")
    data = fetch_stock_data(symbol, period=period)
    if data.empty:
        print(f"⚠️ No data available for {symbol}")
        return None
    
    data = analyze_stock(data)
    all_results = {}
    
    for strategy in strategies:
        print(f"\n📊 Running {strategy} strategy...")
        results = backtest_strategy(data, strategy=strategy)
        
        if results:
            all_results[strategy] = results
            print(f"\n📈 {strategy} Strategy Results:")
            print(f"Total Return: {results['total_return']:.2f}%")
            print(f"Annualized Return: {results['annualized_return']:.2f}%")
            print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Win Rate: {results['win_rate']:.2f}%")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
            print(f"Avg Trade Duration: {results['avg_trade_duration']:.1f} days")
            print(f"Number of Trades: {results['num_trades']}")
            
            # Plot results
            plot_equity_curve(results, symbol, strategy)
            plot_trade_distribution(results)
            
            # Save to database
            conn = sqlite3.connect('backtest_results.db')
            conn.execute('''
                INSERT OR REPLACE INTO backtest_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, strategy, data.index[0].strftime('%Y-%m-%d'), data.index[-1].strftime('%Y-%m-%d'),
                results['total_return'], results['annualized_return'], results['max_drawdown'],
                results['sharpe_ratio'], results['win_rate'], results['profit_factor'],
                results['avg_trade_duration'], results['num_trades']
            ))
            conn.commit()
            conn.close()
    
    return all_results

# Batch backtesting
def batch_backtest(stock_list, strategies=["Swing", "Intraday"]):
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(backtest_stock, symbol, "5y", strategies): symbol for symbol in stock_list}
        for future in tqdm(as_completed(futures), total=len(stock_list)):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    results.append((symbol, result))
            except Exception as e:
                print(f"⚠️ Error backtesting {symbol}: {str(e)}")
    
    # Generate summary report
    print("\n📊 Batch Backtesting Summary:")
    summary = []
    for symbol, result in results:
        for strategy in result:
            summary.append({
                "Symbol": symbol,
                "Strategy": strategy,
                "Total Return (%)": result[strategy]['total_return'],
                "Annualized Return (%)": result[strategy]['annualized_return'],
                "Max Drawdown (%)": result[strategy]['max_drawdown'],
                "Sharpe Ratio": result[strategy]['sharpe_ratio'],
                "Win Rate (%)": result[strategy]['win_rate'],
                "Num Trades": result[strategy]['num_trades']
            })
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    # Plot strategy comparison
    if not summary_df.empty:
        plt.figure(figsize=(12, 6))
        for strategy in summary_df['Strategy'].unique():
            strat_data = summary_df[summary_df['Strategy'] == strategy]
            plt.scatter(strat_data['Max Drawdown (%)'], strat_data['Annualized Return (%)'], 
                        label=strategy, s=100, alpha=0.6)
        
        plt.title("Strategy Comparison: Return vs Drawdown")
        plt.xlabel("Max Drawdown (%)")
        plt.ylabel("Annualized Return (%)")
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return summary_df

# Main function
def main():
    init_database()
    
    print("""
    #######################################################
    #               StockGenie Pro Backtester             #
    #               Comprehensive Strategy Testing        #
    #######################################################
    """)
    
    # Example stocks to backtest (NSE symbols in Yahoo Finance format)
    stock_list = [
        "RELIANCE.NS", "TATASTEEL.NS", "HDFCBANK.NS", 
        "INFY.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
        "ITC.NS", "KOTAKBANK.NS", "HINDUNILVR.NS",
        "SBIN.NS", "BAJFINANCE.NS", "TCS.NS"
    ]
    
    while True:
        print("\nMenu:")
        print("1. Backtest single stock")
        print("2. Batch backtest multiple stocks")
        print("3. View historical backtest results")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            symbol = input("Enter stock symbol (e.g., RELIANCE.NS): ").strip()
            strategies = input("Enter strategies to test (comma separated, e.g., Swing,Intraday): ").strip().split(',')
            backtest_stock(symbol, strategies=strategies)
        
        elif choice == "2":
            print(f"\nAvailable stocks: {', '.join(stock_list)}")
            custom_list = input("Enter stocks to backtest (comma separated, or press Enter for default): ").strip()
            if custom_list:
                stock_list = [s.strip() for s in custom_list.split(',')]
            
            strategies = input("Enter strategies to test (comma separated, e.g., Swing,Intraday): ").strip().split(',')
            batch_backtest(stock_list, strategies=strategies)
        
        elif choice == "3":
            conn = sqlite3.connect('backtest_results.db')
            history_df = pd.read_sql_query("SELECT * FROM backtest_results ORDER BY symbol, strategy", conn)
            conn.close()
            
            if not history_df.empty:
                print("\nHistorical Backtest Results:")
                print(history_df.to_string(index=False))
                
                # Plot historical performance
                plt.figure(figsize=(14, 7))
                for strategy in history_df['strategy'].unique():
                    strat_data = history_df[history_df['strategy'] == strategy]
                    plt.plot(strat_data['symbol'], strat_data['total_return'], 'o-', label=strategy)
                
                plt.title("Historical Strategy Performance")
                plt.xlabel("Stock")
                plt.ylabel("Total Return (%)")
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print("⚠️ No historical backtest data available.")
        
        elif choice == "4":
            print("Exiting StockGenie Pro Backtester...")
            break
        
        else:
            print("⚠️ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
