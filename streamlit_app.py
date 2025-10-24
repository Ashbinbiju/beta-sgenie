import pandas as pd
import numpy as np
import ta
import logging
import streamlit as st
from datetime import datetime, timedelta, time, timezone
from functools import wraps
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import time as time_module
import requests
import re
import io
import random
import warnings
import sqlite3
import gc
import json
import pickle
from pathlib import Path
from diskcache import Cache
from SmartApi import SmartConnect
import pyotp
import os
import subprocess
from dotenv import load_dotenv
from dhanhq import dhanhq # New import for Dhan

# Try to import supabase (optional)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logging.warning("Supabase not installed. Paper trading will be disabled. Install with: pip install supabase")

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

load_dotenv()

# --- Environment variables for both APIs ---
# SmartAPI
CLIENT_ID = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
# Dhan
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

API_KEYS = {
    "Historical": "c3C0tMGn",
    "Trading": os.getenv("TRADING_API_KEY"),
    "Market": os.getenv("MARKET_API_KEY")
}

# Initialize Supabase client
supabase: Client = None
if SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("✅ Supabase connected successfully")
    except Exception as e:
        logging.error(f"Failed to connect to Supabase: {e}")
        supabase = None
else:
    if not SUPABASE_AVAILABLE:
        logging.warning("Supabase package not installed")
    elif not SUPABASE_URL or not SUPABASE_KEY:
        logging.warning("Supabase credentials not configured")

# --- Global session caches ---
_global_smart_api = None
_global_dhan_client = None # New global for Dhan client
_session_timestamp = None
SESSION_EXPIRY = 1800  # 30 minutes for safety

cache = Cache("stock_data_cache")

# Enhanced scan configuration for large stock lists
SCAN_CONFIG = {
    "batch_size": 8,  # Increased for better efficiency
    "delay_within_batch": 1.5,  # Reduced delay for faster scanning
    "delay_between_batches": 4,  # Slightly reduced
    "session_refresh_interval": 25,  # More frequent refreshes
    "max_stocks_per_scan": 1000,
    "checkpoint_interval_small": 5,  # For <100 stocks
    "checkpoint_interval_large": 3,  # For >100 stocks  
    "memory_cleanup_interval": 20,  # Clean memory every 20 stocks
    "max_consecutive_failures": 10,  # Stop if too many failures
    "api_health_check_interval": 50,  # Check API health every 50 stocks
}

# Live scanner configuration
LIVE_SCAN_CONFIG = {
    "scan_interval": 120,  # Scan every 2 minutes
    "stocks_per_batch": 3,  # Reduced to 3 stocks per batch to respect rate limits
    "batch_delay": 2,  # 2 seconds between batches (allows 3 req/sec limit)
    "min_sector_change": 0.5,  # Minimum sector change % to consider bullish
    "min_sector_advance_ratio": 60,  # Minimum advance ratio (60%)
    "cooldown_period": 300,  # 5 minutes cooldown for same stock alert
    "alert_score_threshold": 65,  # Minimum score to trigger alert
}

# SmartAPI Rate Limiting (from official documentation)
# getCandleData: 3 req/sec, 180 req/min, 5000 req/hour
SMARTAPI_RATE_LIMITS = {
    "requests_per_second": 3,
    "requests_per_minute": 180,
    "requests_per_hour": 5000,
    "min_delay_between_requests": 0.35  # ~333ms to stay under 3 req/sec
}

# Auto-update configuration
AUTO_UPDATE_CONFIG = {
    "enabled": True,  # Set to False to disable auto-updates
    "check_interval": 300,  # Check for updates every 5 minutes
    "github_repo": "Ashbinbiju/beta-sgenie",
    "branch": "main"
}

# Telegram configuration
TELEGRAM_CONFIG = {
    "enabled": os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    "alert_on_scan": True,  # Send alert when scan finds stocks
    "alert_on_high_score": True,  # Send alert for scores >= 75
    "alert_threshold": 75,  # Minimum score for individual alerts
    "max_alerts_per_scan": 5,  # Maximum number of individual stock alerts per scan
}

# Simple Rate Limiter for SmartAPI
class SimpleRateLimiter:
    """Simple rate limiter to respect SmartAPI limits"""
    def __init__(self, min_delay=0.35):
        self.min_delay = min_delay
        self.last_call_time = 0
    
    def wait_if_needed(self):
        """Wait if needed to respect rate limits"""
        current_time = time_module.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            time_module.sleep(sleep_time)
        
        self.last_call_time = time_module.time()

# Global rate limiter instance
_smartapi_rate_limiter = SimpleRateLimiter(SMARTAPI_RATE_LIMITS['min_delay_between_requests'])

def send_telegram_message(message, parse_mode="HTML"):
    """Send message to Telegram"""
    if not TELEGRAM_CONFIG["enabled"]:
        return False
    
    if not TELEGRAM_CONFIG["bot_token"] or not TELEGRAM_CONFIG["chat_id"]:
        logging.warning("Telegram bot token or chat ID not configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_CONFIG['bot_token']}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CONFIG["chat_id"],
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logging.info("Telegram message sent successfully")
            return True
        else:
            logging.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")
        return False

def send_scan_results_alert(results_df, trading_style, scan_time=None):
    """Send scan results summary to Telegram"""
    if not TELEGRAM_CONFIG["enabled"] or not TELEGRAM_CONFIG["alert_on_scan"]:
        return False
    
    if results_df.empty:
        return False
    
    try:
        scan_time = scan_time or datetime.now().strftime("%H:%M:%S")
        
        # Create summary message
        message = f"🔍 <b>Stock Scanner Alert</b>\n\n"
        message += f"⏰ Time: {scan_time}\n"
        message += f"📊 Style: {trading_style}\n"
        message += f"📈 Found: {len(results_df)} opportunities\n\n"
        
        # Add top 5 stocks
        message += "<b>🏆 Top Picks:</b>\n"
        for idx, row in results_df.head(5).iterrows():
            score_emoji = "🟢" if row['Score'] >= 75 else "🟡" if row['Score'] >= 60 else "⚪"
            message += f"{score_emoji} <b>{row['Symbol']}</b> - Score: {row['Score']}/100\n"
            message += f"   Signal: {row['Signal']} | Price: ₹{row['Current Price']:.2f}\n"
            message += f"   Target: ₹{row['Target']:.2f} | SL: ₹{row['Stop Loss']:.2f}\n\n"
        
        return send_telegram_message(message)
        
    except Exception as e:
        logging.error(f"Error sending scan results alert: {e}")
        return False

def send_high_score_alert(stock_data):
    """Send alert for individual high-scoring stock"""
    if not TELEGRAM_CONFIG["enabled"] or not TELEGRAM_CONFIG["alert_on_high_score"]:
        return False
    
    try:
        message = f"🚨 <b>HIGH SCORE ALERT</b> 🚨\n\n"
        message += f"📊 <b>{stock_data['Symbol']}</b>\n"
        message += f"💯 Score: <b>{stock_data['Score']}/100</b>\n\n"
        message += f"📈 Signal: {stock_data['Signal']}\n"
        message += f"🎯 Regime: {stock_data['Regime']}\n\n"
        message += f"💰 Current Price: ₹{stock_data['Current Price']:.2f}\n"
        message += f"🎯 Target: ₹{stock_data['Target']:.2f}\n"
        message += f"🛑 Stop Loss: ₹{stock_data['Stop Loss']:.2f}\n\n"
        message += f"📝 Reason: {stock_data['Reason']}\n"
        
        return send_telegram_message(message)
        
    except Exception as e:
        logging.error(f"Error sending high score alert: {e}")
        return False

def check_for_github_updates():
    """Check if there are new commits on GitHub"""
    try:
        # Get the directory of the current script dynamically
        repo_path = os.path.dirname(os.path.abspath(__file__))
        
        # First, fetch the latest changes from remote (without merging)
        fetch_result = subprocess.run(
            ["git", "fetch", "origin", "main"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if fetch_result.returncode != 0:
            logging.warning(f"Git fetch warning: {fetch_result.stderr}")
        
        # Get local commit hash
        local_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            cwd=repo_path,
            timeout=5
        ).decode('utf-8').strip()
        
        # Get remote commit hash (after fetch)
        remote_commit = subprocess.check_output(
            ["git", "rev-parse", "origin/main"], 
            cwd=repo_path,
            timeout=5
        ).decode('utf-8').strip()
        
        has_update = local_commit != remote_commit
        
        if has_update:
            logging.info(f"Update available: {local_commit[:7]} -> {remote_commit[:7]}")
        else:
            logging.debug(f"Already up to date: {local_commit[:7]}")
        
        return has_update, local_commit, remote_commit
    except subprocess.TimeoutExpired:
        logging.error("Git command timed out")
        return False, None, None
    except Exception as e:
        logging.error(f"Error checking for updates: {e}")
        return False, None, None

def get_changelog(local_commit, remote_commit):
    """Get changelog between two commits"""
    try:
        repo_path = os.path.dirname(os.path.abspath(__file__))
        
        # Get commit messages between local and remote
        result = subprocess.run(
            ["git", "log", f"{local_commit}..{remote_commit}", "--pretty=format:%s", "--no-merges"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout:
            # Split into individual commit messages
            commits = [msg.strip() for msg in result.stdout.strip().split('\n') if msg.strip()]
            return commits
        else:
            return []
            
    except Exception as e:
        logging.error(f"Error getting changelog: {e}")
        return []

def pull_github_updates():
    """Pull latest changes from GitHub with automatic stash handling"""
    try:
        # Get the directory of the current script dynamically
        repo_path = os.path.dirname(os.path.abspath(__file__))
        
        # Check if there are local changes
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        has_changes = bool(status_result.stdout.strip())
        
        if has_changes:
            # Stash local changes before pulling
            logging.info("Local changes detected, stashing before pull...")
            stash_result = subprocess.run(
                ["git", "stash", "push", "-u", "-m", "Auto-stash before update"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if stash_result.returncode != 0:
                logging.warning(f"Stash warning: {stash_result.stderr}")
        
        # Pull latest changes
        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        success = result.returncode == 0
        message = result.stdout if success else result.stderr
        
        if success:
            logging.info(f"Successfully pulled updates: {message}")
            if has_changes:
                # Try to reapply stashed changes
                pop_result = subprocess.run(
                    ["git", "stash", "pop"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if pop_result.returncode == 0:
                    logging.info("Reapplied stashed changes")
                else:
                    logging.warning(f"Could not reapply stash: {pop_result.stderr}")
        else:
            logging.error(f"Failed to pull updates: {message}")
        
        return success, message
    except subprocess.TimeoutExpired:
        error_msg = "Git pull timed out (30s)"
        logging.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Error pulling updates: {str(e)}"
        logging.error(error_msg)
        return False, error_msg

def auto_update_check():
    """Check for updates and notify user"""
    if not AUTO_UPDATE_CONFIG["enabled"]:
        return
    
    # Check if enough time has passed since last check
    if 'last_update_check' not in st.session_state:
        st.session_state.last_update_check = datetime.now()
    
    time_since_check = (datetime.now() - st.session_state.last_update_check).total_seconds()
    
    if time_since_check < AUTO_UPDATE_CONFIG["check_interval"]:
        return
    
    # Update last check time
    st.session_state.last_update_check = datetime.now()
    
    # Check for updates
    has_update, local, remote = check_for_github_updates()
    
    if has_update:
        st.session_state.update_available = True
        st.session_state.local_commit = local[:7] if local else "unknown"
        st.session_state.remote_commit = remote[:7] if remote else "unknown"
        st.session_state.local_commit_full = local  # Store full hash for changelog
        st.session_state.remote_commit_full = remote  # Store full hash for changelog

def get_bullish_sectors():
    """Get list of currently bullish sectors from market breadth"""
    try:
        breadth_data = fetch_market_breadth()
        if not breadth_data or 'industry' not in breadth_data:
            logging.warning("No industry data available")
            return []
        
        industries = breadth_data['industry']
        bullish_sectors = []
        
        for industry in industries:
            avg_change = industry.get('avgChange', 0)
            total = industry.get('total', 1)
            advancing = industry.get('advancing', 0)
            advance_ratio = (advancing / total) * 100 if total > 0 else 0
            
            # Filter bullish sectors
            if (avg_change >= LIVE_SCAN_CONFIG['min_sector_change'] and 
                advance_ratio >= LIVE_SCAN_CONFIG['min_sector_advance_ratio']):
                
                sector_name = industry.get('Industry', '')
                bullish_sectors.append({
                    'sector': sector_name,
                    'change': avg_change,
                    'advance_ratio': advance_ratio,
                    'advancing': advancing,
                    'total': total
                })
        
        # Sort by strength (combination of change and advance ratio)
        bullish_sectors.sort(key=lambda x: (x['change'] * x['advance_ratio']), reverse=True)
        
        logging.info(f"Found {len(bullish_sectors)} bullish sectors")
        return bullish_sectors
        
    except Exception as e:
        logging.error(f"Error getting bullish sectors: {e}")
        return []

def get_neutral_sectors():
    """Get list of currently neutral sectors from market breadth"""
    try:
        breadth_data = fetch_market_breadth()
        if not breadth_data or 'industry' not in breadth_data:
            logging.warning("No industry data available")
            return []
        
        industries = breadth_data['industry']
        neutral_sectors = []
        
        for industry in industries:
            avg_change = industry.get('avgChange', 0)
            total = industry.get('total', 1)
            advancing = industry.get('advancing', 0)
            advance_ratio = (advancing / total) * 100 if total > 0 else 0
            
            # Filter neutral sectors (not bullish but not too bearish)
            # Neutral: -0.5% to +0.5% change OR 40-60% advance ratio
            if ((abs(avg_change) <= 0.5) or 
                (40 <= advance_ratio <= 60 and avg_change >= -1.0)):
                
                sector_name = industry.get('Industry', '')
                neutral_sectors.append({
                    'sector': sector_name,
                    'change': avg_change,
                    'advance_ratio': advance_ratio,
                    'advancing': advancing,
                    'total': total
                })
        
        # Sort by advance ratio (higher is better for neutral)
        neutral_sectors.sort(key=lambda x: x['advance_ratio'], reverse=True)
        
        logging.info(f"Found {len(neutral_sectors)} neutral sectors")
        return neutral_sectors
        
    except Exception as e:
        logging.error(f"Error getting neutral sectors: {e}")
        return []

def get_stocks_from_bullish_sectors(bullish_sectors):
    """Get stock list from bullish sectors only"""
    if not bullish_sectors:
        return []
    
    sector_names = [s['sector'] for s in bullish_sectors]
    stock_list = []
    matched_sectors = []
    unmatched_sectors = []
    
    # Enhanced sector mapping with multiple variations
    sector_mapping = {
        'Bank': 'Bank',
        'Banking': 'Bank',
        'IT': 'IT',
        'Information Technology': 'IT',
        'Software': 'IT',
        'Finance': 'Finance',
        'Financial Services': 'Finance',
        'Auto': 'Auto',
        'Automobile': 'Auto',
        'Automotive': 'Auto',
        'Automobile and Auto Components': 'Auto',
        'Pharma': 'Pharma',
        'Pharmaceuticals': 'Pharma',
        'Healthcare': 'Pharma',
        'Metals': 'Metals',
        'Metal': 'Metals',
        'Steel': 'Metals',
        'FMCG': 'FMCG',
        'Fast Moving Consumer Goods': 'FMCG',
        'Consumer Goods': 'FMCG',
        'Power': 'Power',
        'Energy': 'Power',
        'Utilities': 'Power',
        'Capital Goods': 'Capital Goods',
        'Industrials': 'Capital Goods',
        'Oil & Gas': 'Oil & Gas',
        'Oil and Gas': 'Oil & Gas',
        'Chemicals': 'Chemicals',
        'Chemical': 'Chemicals',
        'Telecom': 'Telecom',
        'Telecommunications': 'Telecom',
        'Telecommunication': 'Telecom',
        'Infrastructure': 'Infrastructure',
        'Insurance': 'Insurance',
        'Cement': 'Cement',
        'Construction Materials': 'Cement',
        'Realty': 'Realty',
        'Real Estate': 'Realty',
        'Media': 'Media',
        'Entertainment': 'Media',
        'Aviation': 'Aviation',
        'Airlines': 'Aviation',
        'Retail': 'Retail',
        'Consumer Discretionary': 'Retail',
        'Consumer Durables': 'Consumer Durables',
        'Consumer Services': 'Consumer Services',
        'Services': 'Consumer Services',
        'Hospitality': 'Consumer Services',
        'Travel': 'Consumer Services',
        'Food Services': 'Consumer Services'
    }
    
    for sector_name in sector_names:
        # Direct mapping only - no fuzzy matching
        sector_key = sector_mapping.get(sector_name)
        if sector_key and sector_key in SECTORS:
            stock_list.extend(SECTORS[sector_key])
            matched_sectors.append(sector_name)
        else:
            unmatched_sectors.append(sector_name)
    
    # Remove duplicates while preserving order
    stock_list = list(dict.fromkeys(stock_list))
    
    logging.info(f"Got {len(stock_list)} stocks from {len(matched_sectors)} matched sectors: {matched_sectors}")
    if unmatched_sectors:
        logging.warning(f"Could not match {len(unmatched_sectors)} sectors: {unmatched_sectors}")
        logging.info(f"Available sectors in SECTORS dict: {list(SECTORS.keys())}")
    
    return stock_list

def get_stocks_from_bullish_and_neutral_sectors():
    """Get stock list from both bullish and neutral sectors"""
    bullish_sectors = get_bullish_sectors()
    neutral_sectors = get_neutral_sectors()
    
    # Combine both lists
    all_sectors = bullish_sectors + neutral_sectors
    
    if not all_sectors:
        logging.warning("No bullish or neutral sectors found, using all sectors")
        return get_unique_stock_list(SECTORS), [], []
    
    # Get stocks from combined sectors
    stock_list = get_stocks_from_bullish_sectors(all_sectors)  # Reuse the mapping logic
    
    return stock_list, bullish_sectors, neutral_sectors

def live_scan_iteration(stock_list, timeframe, api_provider, alert_history, status_callback=None):
    """Single iteration of live scanner with status updates"""
    results = []
    new_alerts = []
    current_time = time_module.time()
    total_stocks = len(stock_list)
    processed = 0
    
    if status_callback:
        status_callback(f"🔍 Starting scan of {total_stocks} stocks from bullish sectors...")
    
    # Analyze stocks in batches
    for batch_idx in range(0, len(stock_list), LIVE_SCAN_CONFIG['stocks_per_batch']):
        batch = stock_list[batch_idx:batch_idx + LIVE_SCAN_CONFIG['stocks_per_batch']]
        batch_num = (batch_idx // LIVE_SCAN_CONFIG['stocks_per_batch']) + 1
        total_batches = (total_stocks + LIVE_SCAN_CONFIG['stocks_per_batch'] - 1) // LIVE_SCAN_CONFIG['stocks_per_batch']
        
        if status_callback:
            status_callback(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} stocks)...")
        
        for symbol in batch:
            try:
                processed += 1
                # Show current stock being analyzed
                if status_callback:
                    if processed % 5 == 0 or processed <= 3:
                        status_callback(f"⏳ Analyzing {symbol}... ({processed}/{total_stocks} | {len(results)} opportunities)")
                
                # Check if stock is in cooldown
                last_alert_time = alert_history.get(symbol, 0)
                if (current_time - last_alert_time) < LIVE_SCAN_CONFIG['cooldown_period']:
                    continue
                
                # Analyze stock
                result = analyze_stock_batch(
                    symbol, 
                    trading_style='intraday', 
                    timeframe=timeframe, 
                    contrarian_mode=False, 
                    max_retries=2,
                    api_provider=api_provider
                )
                
                if result:
                    result['Sector'] = assign_primary_sector(symbol, SECTORS)
                    result['Timestamp'] = datetime.now().strftime('%H:%M:%S')
                    results.append(result)
                    
                    # Check if this is a new alert
                    if result['Score'] >= LIVE_SCAN_CONFIG['alert_score_threshold']:
                        if result['Signal'] in ['Buy', 'Strong Buy']:
                            new_alerts.append(result)
                            alert_history[symbol] = current_time
                            logging.info(f"🚨 NEW ALERT: {symbol} - Score: {result['Score']}")
                            if status_callback:
                                status_callback(f"🚨 ALERT: {symbol} - Score: {result['Score']}")
                
                # Delay between stocks to respect rate limits (already handled by rate limiter)
                # But add small buffer for safety
                time_module.sleep(0.5)
                
            except Exception as e:
                logging.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Delay between batches to ensure we don't exceed limits
        if batch_idx + LIVE_SCAN_CONFIG['stocks_per_batch'] < len(stock_list):
            if status_callback:
                status_callback(f"⏸️ Waiting {LIVE_SCAN_CONFIG['batch_delay']}s before next batch...")
            time_module.sleep(LIVE_SCAN_CONFIG['batch_delay'])
    
    if status_callback:
        status_callback(f"✅ Scan complete! Found {len(results)} opportunities, {len(new_alerts)} new alerts")
    
    return results, new_alerts

def get_unique_stock_list(sectors_dict):
    """Get unique stock list from SECTORS dictionary"""
    all_stocks = []
    for sector_stocks in sectors_dict.values():
        all_stocks.extend(sector_stocks)
    
    unique_stocks = list(dict.fromkeys(all_stocks))
    
    if len(all_stocks) != len(unique_stocks):
        duplicates = len(all_stocks) - len(unique_stocks)
        logging.info(f"Consolidated {duplicates} duplicate entries")
    
    return unique_stocks

def get_stock_list_from_sectors(sectors_dict, selected_sectors):
    """Get unique stock list with validation"""
    if not selected_sectors or "All" in selected_sectors:
        stock_list = get_unique_stock_list(sectors_dict)
    else:
        temp_list = []
        for sector in selected_sectors:
            temp_list.extend(sectors_dict.get(sector, []))
        stock_list = list(dict.fromkeys(temp_list))
    
    if not stock_list:
        logging.error("No stocks found for selected sectors")
        return ["SBIN-EQ"]  # Fallback default
    
    return stock_list
# User agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/124.0.2478.80 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 OPR/110.0.0.0",
    "Mozilla/5.0 (Linux; Android 14; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/23.0 Chrome/115.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Brave/124.0.0.0"
]

# Sector definitions
SECTORS = {
    "Bank": [
        "HDFCBANK-EQ", "ICICIBANK-EQ", "SBIN-EQ", "KOTAKBANK-EQ", "AXISBANK-EQ",
        "INDUSINDBK-EQ", "PNB-EQ", "BANKBARODA-EQ", "CANBK-EQ", "UNIONBANK-EQ",
        "IDFCFIRSTB-EQ", "FEDERALBNK-EQ", "RBLBANK-EQ", "BANDHANBNK-EQ", "INDIANB-EQ",
        "BANKINDIA-EQ", "KARURVYSYA-EQ", "CUB-EQ", "J&KBANK-EQ", "DCBBANK-EQ",
        "AUBANK-EQ", "YESBANK-EQ", "IDBI-EQ", "SOUTHBANK-EQ", "CSBBANK-EQ",
        "TMB-EQ", "KTKBANK-EQ", "EQUITASBNK-EQ", "UJJIVANSFB-EQ","CENTURYPLY-EQ"
    ],
    "IT": [
        "TCS-EQ", "INFY-EQ", "HCLTECH-EQ", "WIPRO-EQ", "TECHM-EQ", "LTIM-EQ",
        "MPHASIS-EQ", "FSL-EQ", "BSOFT-EQ", "NEWGEN-EQ", "ZENSARTECH-EQ",
        "RATEGAIN-EQ", "TANLA-EQ", "COFORGE-EQ", "PERSISTENT-EQ", "CYIENT-EQ",
        "SONATSOFTW-EQ", "KPITTECH-EQ", "TATAELXSI-EQ",
        "INTELLECT-EQ", "HAPPSTMNDS-EQ", "MASTEK-EQ", "ECLERX-EQ", "NIITLTD-EQ",
        "RSYSTEMS-EQ", "OFSS-EQ", "AURIONPRO-EQ", "DATAMATICS-EQ",
        "QUICKHEAL-EQ", "CIGNITITEC-EQ", "SAGILITY-EQ", "ALLDIGI-EQ","BLS-EQ"
    ],
    "Finance": [
        "HDFCBANK-EQ", "ICICIBANK-EQ", "SBIN-EQ", "KOTAKBANK-EQ", "BAJFINANCE-EQ",
        "AXISBANK-EQ", "BAJAJFINSV-EQ", "INDUSINDBK-EQ", "SHRIRAMFIN-EQ", "CHOLAFIN-EQ",
        "SBICARD-EQ", "M&MFIN-EQ", "MUTHOOTFIN-EQ", "LICHSGFIN-EQ", "IDFCFIRSTB-EQ",
        "AUBANK-EQ", "POONAWALLA-EQ", "SUNDARMFIN-EQ", "IIFL-EQ", "ABCAPITAL-EQ",
        "LTF-EQ", "CREDITACC-EQ", "MANAPPURAM-EQ", "DHANI-EQ", "JMFINANCIL-EQ",
        "EDELWEISS-EQ", "INDIASHLTR-EQ", "MOTILALOFS-EQ", "CDSL-EQ", "BSE-EQ",
        "MCX-EQ", "ANGELONE-EQ", "KARURVYSYA-EQ", "RBLBANK-EQ", "PNB-EQ",
        "CANBK-EQ", "UNIONBANK-EQ", "IOB-EQ", "YESBANK-EQ", "UCOBANK-EQ",
        "BANKINDIA-EQ", "CENTRALBK-EQ", "IDBI-EQ", "J&KBANK-EQ", "DCBBANK-EQ",
        "FEDERALBNK-EQ", "SOUTHBANK-EQ", "CSBBANK-EQ", "TMB-EQ", "KTKBANK-EQ",
        "EQUITASBNK-EQ", "UJJIVANSFB-EQ", "BANDHANBNK-EQ", "SURYODAY-EQ", "PSB-EQ",
        "PFS-EQ", "HDFCAMC-EQ", "UTIAMC-EQ", "ABSLAMC-EQ",
        "360ONE-EQ", "ANANDRATHI-EQ", "PNBHOUSING-EQ", "HOMEFIRST-EQ", "AAVAS-EQ",
        "APTUS-EQ", "RECLTD-EQ", "PFC-EQ", "IREDA-EQ", "SMCGLOBAL-EQ", "CHOICEIN-EQ",
        "KFINTECH-EQ", "MASFIN-EQ", "TRIDENT-EQ", "SBFC-EQ",
        "UGROCAP-EQ", "FUSION-EQ", "PAISALO-EQ", "CAPITALSFB-EQ", "NSIL-EQ",
        "SATIN-EQ"
    ],
    "Auto": [
        "MARUTI-EQ","BELRISE-EQ", "TATAMOTORS-EQ", "M&M-EQ", "BAJAJ-AUTO-EQ", "HEROMOTOCO-EQ",
        "EICHERMOT-EQ", "TVSMOTOR-EQ", "ASHOKLEY-EQ", "MRF-EQ", "BALKRISIND-EQ",
        "APOLLOTYRE-EQ", "CEATLTD-EQ", "JKTYRE-EQ", "MOTHERSON-EQ", "BHARATFORG-EQ",
        "SUNDRMFAST-EQ", "EXIDEIND-EQ", "BOSCHLTD-EQ", "ENDURANCE-EQ",
        "UNOMINDA-EQ", "ZFCVINDIA-EQ", "GABRIEL-EQ", "SUPRAJIT-EQ", "LUMAXTECH-EQ",
        "FIEMIND-EQ", "SUBROS-EQ", "JAMNAAUTO-EQ", "SHRIRAMFIN-EQ", "ESCORTS-EQ",
        "ATULAUTO-EQ", "OLECTRA-EQ", "GREAVESCOT-EQ", "SMLISUZU-EQ", "VSTTILLERS-EQ",
        "HINDMOTORS-EQ", "MAHSCOOTER-EQ"
    ],
    "Pharma": [
        "SUNPHARMA-EQ", "CIPLA-EQ", "DRREDDY-EQ", "APOLLOHOSP-EQ", "LUPIN-EQ",
        "DIVISLAB-EQ", "AUROPHARMA-EQ", "ALKEM-EQ", "TORNTPHARM-EQ", "ZYDUSLIFE-EQ",
        "IPCALAB-EQ", "GLENMARK-EQ", "BIOCON-EQ", "ABBOTINDIA-EQ", "SANOFI-EQ",
        "PFIZER-EQ", "GLAXO-EQ", "NATCOPHARM-EQ", "AJANTPHARM-EQ", "GRANULES-EQ",
        "LAURUSLABS-EQ", "STAR-EQ", "JUBLPHARMA-EQ", "ASTRAZEN-EQ", "WOCKPHARMA-EQ",
        "FORTIS-EQ", "MAXHEALTH-EQ", "METROPOLIS-EQ", "THYROCARE-EQ", "POLYMED-EQ",
        "KIMS-EQ", "LALPATHLAB-EQ", "MEDPLUS-EQ", "ERIS-EQ", "INDOCO-EQ",
        "CAPLIPOINT-EQ", "NEULANDLAB-EQ", "SHILPAMED-EQ", "SUVEN-EQ", "AARTIDRUGS-EQ",
        "PGHL-EQ", "SYNGENE-EQ", "VINATIORGA-EQ", "GLAND-EQ", "JBCHEPHARM-EQ",
        "HCG-EQ", "RAINBOW-EQ", "ASTERDM-EQ", "KRSNAA-EQ", "VIJAYA-EQ", "MEDANTA-EQ",
        "BLISSGVS-EQ", "MOREPENLAB-EQ", "RPGLIFE-EQ"
    ],
    "Metals": [
        "TATASTEEL-EQ", "JSWSTEEL-EQ", "HINDALCO-EQ", "VEDL-EQ", "SAIL-EQ",
        "NMDC-EQ", "HINDZINC-EQ", "NATIONALUM-EQ", "JINDALSTEL-EQ", "MOIL-EQ",
        "APLAPOLLO-EQ", "RATNAMANI-EQ", "JSL-EQ", "WELCORP-EQ",
        "SHYAMMETL-EQ", "MIDHANI-EQ", "GRAVITA-EQ", "SARDAEN-EQ", "ASHAPURMIN-EQ",
        "JTLIND-EQ", "RAMASTEEL-EQ", "MAITHANALL-EQ", "KIOCL-EQ", "IMFA-EQ",
        "GMDCLTD-EQ", "VISHNU-EQ", "SANDUMA-EQ", "VRAJ-EQ", "COALINDIA-EQ"
    ],
    "FMCG": [
        "HINDUNILVR-EQ", "ITC-EQ", "NESTLEIND-EQ", "BRITANNIA-EQ",
        "GODREJCP-EQ", "DABUR-EQ", "COLPAL-EQ", "MARICO-EQ", "PGHH-EQ",
        "EMAMILTD-EQ", "GILLETTE-EQ", "HATSUN-EQ", "JYOTHYLAB-EQ", "BAJAJCON-EQ",
        "RADICO-EQ", "TATACONSUM-EQ", "UNITDSPR-EQ", "CCL-EQ", "AVANTIFEED-EQ",
        "BIKAJI-EQ", "VBL-EQ", "ETERNAL-EQ", "DOMS-EQ",
        "GODREJAGRO-EQ", "SAPPHIRE-EQ", "VENKEYS-EQ", "BECTORFOOD-EQ", "KRBL-EQ"
    ],
    "Power": [
        "NTPC-EQ", "POWERGRID-EQ", "ADANIPOWER-EQ", "TATAPOWER-EQ", "JSWENERGY-EQ",
        "NHPC-EQ", "SJVN-EQ", "TORNTPOWER-EQ", "CESC-EQ", "ADANIENSOL-EQ",
        "INDIGRID-EQ", "POWERMECH-EQ", "KEC-EQ", "INOXWIND-EQ", "KPIL-EQ",
        "SUZLON-EQ", "BHEL-EQ", "THERMAX-EQ", "GVPIL-EQ", "VOLTAMP-EQ",
        "TARIL-EQ", "TDPOWERSYS-EQ", "JYOTISTRUC-EQ", "IWEL-EQ", "ACMESOLAR-EQ"
    ],
    "Capital Goods": [
        "LT-EQ", "SIEMENS-EQ", "ABB-EQ", "BEL-EQ", "BHEL-EQ", "HAL-EQ",
        "CUMMINSIND-EQ", "THERMAX-EQ", "AIAENG-EQ", "SKFINDIA-EQ", "GRINDWELL-EQ",
        "TIMKEN-EQ", "KSB-EQ", "ELGIEQUIP-EQ", "LMW-EQ", "KIRLOSENG-EQ",
        "GREAVESCOT-EQ", "TRITURBINE-EQ", "VOLTAS-EQ", "BLUESTARCO-EQ", "HAVELLS-EQ",
        "DIXON-EQ", "KAYNES-EQ", "SYRMA-EQ", "AMBER-EQ", "SUZLON-EQ", "CGPOWER-EQ",
        "APARINDS-EQ", "HBLENGINE-EQ", "KEI-EQ", "POLYCAB-EQ", "RRKABEL-EQ",
        "SCHNEIDER-EQ", "TDPOWERSYS-EQ", "KIRLOSBROS-EQ", "JYOTICNC-EQ", "DATAPATTNS-EQ",
        "INOXWIND-EQ", "KALPATPOWR-EQ", "MAZDOCK-EQ", "COCHINSHIP-EQ", "GRSE-EQ",
        "POWERMECH-EQ", "ISGEC-EQ", "HPL-EQ", "VTL-EQ", "DYNAMATECH-EQ", "JASH-EQ",
        "GMMPFAUDLR-EQ", "ESABINDIA-EQ", "CENTEXT-EQ", "SALASAR-EQ", "TITAGARH-EQ",
        "VGUARD-EQ", "WABAG-EQ", "AZAD-EQ"
    ],
    "Oil & Gas": [
        "RELIANCE-EQ", "ONGC-EQ", "IOC-EQ", "BPCL-EQ", "HINDPETRO-EQ", "GAIL-EQ",
        "PETRONET-EQ", "OIL-EQ", "IGL-EQ", "MGL-EQ", "GUJGASLTD-EQ", "GSPL-EQ",
        "AEGISLOG-EQ", "CHENNPETRO-EQ", "MRPL-EQ", "FLUOROCHEM-EQ", "CASTROLIND-EQ",
        "SOTL-EQ", "PANAMAPET-EQ", "GOCLCORP-EQ"
    ],
    "Chemicals": [
        "PIDILITIND-EQ", "SRF-EQ", "DEEPAKNTR-EQ", "ATUL-EQ", "AARTIIND-EQ",
        "NAVINFLUOR-EQ", "VINATIORGA-EQ", "FINEORG-EQ", "ALKYLAMINE-EQ", "BALAMINES-EQ",
        "GUJFLUORO-EQ", "CLEAN-EQ", "JUBLINGREA-EQ", "GALAXYSURF-EQ", "PCBL-EQ",
        "NOCIL-EQ", "BASF-EQ", "SUDARSCHEM-EQ", "NEOGEN-EQ", "PRIVISCL-EQ",
        "ROSSARI-EQ", "LXCHEM-EQ", "ANURAS-EQ", "CHEMCON-EQ",
        "DMCC-EQ", "TATACHEM-EQ", "COROMANDEL-EQ", "UPL-EQ", "BAYERCROP-EQ",
        "SUMICHEM-EQ", "PIIND-EQ", "EIDPARRY-EQ", "CHEMPLASTS-EQ",
        "IGPL-EQ", "TIRUMALCHM-EQ", "RALLIS-EQ"
    ],
    "Telecom": [
        "BHARTIARTL-EQ", "IDEA-EQ", "INDUSTOWER-EQ", "TATACOMM-EQ",
        "HFCL-EQ", "TEJASNET-EQ", "STLTECH-EQ", "ITI-EQ", "ASTEC-EQ"
    ],
    "Infrastructure": [
        "LT-EQ", "GMRAIRPORT-EQ", "IRB-EQ", "NBCC-EQ", "RVNL-EQ", "KEC-EQ",
        "PNCINFRA-EQ", "KNRCON-EQ", "GRINFRA-EQ", "NCC-EQ", "HGINFRA-EQ",
        "ASHOKA-EQ", "SADBHAV-EQ", "JWL-EQ", "PATELENG-EQ", "KALPATPOWR-EQ",
        "IRCON-EQ", "ENGINERSIN-EQ", "AHLUWALIA-EQ", "PSPPROJECTS-EQ", "CAPACITE-EQ",
        "WELSPUNIND-EQ", "HCC-EQ", "MANINFRA-EQ", "RIIL-EQ",
        "JAYBARMARU-EQ"
    ],
    "Insurance": [
        "SBILIFE-EQ", "HDFCLIFE-EQ", "ICICIGI-EQ", "ICICIPRULI-EQ", "LICI-EQ",
        "GICRE-EQ", "NIACL-EQ", "STARHEALTH-EQ", "MAXFIN-EQ"
    ],
    "Diversified": [
        "ADANIENT-EQ", "GRASIM-EQ",
        "DCMSHRIRAM-EQ", "3MINDIA-EQ", "CENTURYPLY-EQ", "KFINTECH-EQ", "BALMERLAWRI-EQ",
        "GODREJIND-EQ", "BIRLACORPN-EQ"
    ],
    "Cement": [
        "ULTRACEMCO-EQ", "SHREECEM-EQ", "AMBUJACEM-EQ", "ACC-EQ", "JKCEMENT-EQ",
        "DALBHARAT-EQ", "RAMCOCEM-EQ", "NUVOCO-EQ", "JKLAKSHMI-EQ",
        "HEIDELBERG-EQ", "INDIACEM-EQ", "PRISMJOHNS-EQ", "STARCEMENT-EQ", "SAGCEM-EQ",
        "DECCANCE-EQ", "KCP-EQ", "ORIENTCEM-EQ", "HIL-EQ", "EVERESTIND-EQ",
        "VISAKAIND-EQ", "BIGBLOC-EQ"
    ],
    "Realty": [
        "DLF-EQ", "GODREJPROP-EQ", "OBEROIRLTY-EQ", "PHOENIXLTD-EQ", "PRESTIGE-EQ",
        "BRIGADE-EQ", "SOBHA-EQ", "SUNTECK-EQ", "MAHLIFE-EQ", "ANANTRAJ-EQ",
        "KOLTEPATIL-EQ", "PURVA-EQ", "ARVSMART-EQ", "RUSTOMJEE-EQ", "DBREALTY-EQ",
        "IBREALEST-EQ", "OMAXE-EQ", "ASHIANA-EQ", "ELDEHSG-EQ", "TARC-EQ"
    ],
    "Aviation": [
        "INDIGO-EQ", "SPICEJET-EQ", "GMRINFRA-EQ"
    ],
    "Retail": [
        "DMART-EQ", "TRENT-EQ", "ABFRL-EQ", "VMART-EQ", "SHOPERSTOP-EQ",
        "BATAINDIA-EQ", "METROBRAND-EQ", "ARVINDFASN-EQ", "CANTABIL-EQ", "ZOMATO-EQ",
        "NYKAA-EQ", "MANYAVAR-EQ", "LANDMARK-EQ", "V2RETAIL-EQ",
        "THANGAMAYL-EQ", "KALYANKJIL-EQ", "TITAN-EQ"
    ],
    "Media": [
        "ZEEL-EQ", "SUNTV-EQ", "TVTODAY-EQ", "DISHTV-EQ", "HATHWAY-EQ",
        "PVR-EQ", "INOXLEISUR-EQ", "SAREGAMA-EQ", "TIPS-EQ"
    ],
    "Consumer Durables": [
        "WHIRLPOOL-EQ", "DIXON-EQ", "AMBER-EQ", "VOLTAS-EQ", "BLUESTARCO-EQ",
        "HAVELLS-EQ", "CROMPTON-EQ", "VGUARD-EQ", "ORIENTELEC-EQ", "KIRIINDUS-EQ",
        "RAJESHEXPO-EQ", "SYMPHONY-EQ", "TITAN-EQ", "KALPATPOWR-EQ", "RELAXO-EQ",
        "TTKHLTCARE-EQ", "VAIBHAVGBL-EQ", "BAJAJELEC-EQ", "FINEORG-EQ", "CHOLAHLDNG-EQ",
        "BSLIMITED-EQ", "SUPRAJIT-EQ", "NIITLTD-EQ", "APARINDS-EQ"
    ],
    "Consumer Services": [
        "ZOMATO-EQ", "NYKAA-EQ", "ADANIPORTS-EQ", "IRCTC-EQ", "PAYTM-EQ",
        "JUBLFOOD-EQ", "DEVYANI-EQ", "WESTLIFE-EQ", "SAPPHIRE-EQ", "BIKAJI-EQ",
        "EASEMYTRIP-EQ", "IXIGO-EQ", "TEAMLEASE-EQ", "QUESS-EQ", "FIRSTSOURCE-EQ",
        "MINDSPACE-EQ", "MAHINDCIE-EQ", "TATAMTRDVR-EQ", "VMART-EQ", "SHOPERSTOP-EQ",
        "TRENT-EQ", "DMART-EQ", "ABFRL-EQ", "MANYAVAR-EQ", "V2RETAIL-EQ"
    ]
}
# ============================================================================
# SECTOR VALIDATION & CLEANUP
# ============================================================================

def validate_and_clean_sectors(sectors_dict):
    """Remove duplicates and validate sector definitions"""
    seen_stocks = {}
    cleaned_sectors = {}
    duplicates = []
    
    for sector, stocks in sectors_dict.items():
        cleaned_stocks = []
        for stock in stocks:
            if stock in seen_stocks:
                duplicates.append(f"{stock} (in {seen_stocks[stock]} and {sector})")
            else:
                seen_stocks[stock] = sector
                cleaned_stocks.append(stock)
        
        cleaned_sectors[sector] = cleaned_stocks
    
    if duplicates:
        logging.warning(f"Removed {len(duplicates)} duplicate stocks:")
        for dup in duplicates[:10]:
            logging.warning(f"  - {dup}")
    
    return cleaned_sectors

def assign_primary_sector(symbol, sectors_dict):
    """Assign primary sector based on priority logic"""
    priority_sectors = [
        "Bank", "IT", "Pharma", "Auto", "FMCG", 
        "Metals", "Power", "Oil & Gas", "Telecom",
        "Finance", "Capital Goods", "Chemicals", 
        "Infrastructure", "Cement", "Realty", "Insurance",
        "Diversified", "Aviation", "Retail", "Media",
        "Consumer Durables", "Consumer Services"
    ]
    
    matching_sectors = [
        sector for sector in priority_sectors 
        if symbol in sectors_dict.get(sector, [])
    ]
    
    if matching_sectors:
        return matching_sectors[0]
    
    for sector, stocks in sectors_dict.items():
        if symbol in stocks:
            return sector
    
    return "Unknown"

# Apply cleaning at startup
SECTORS = validate_and_clean_sectors(SECTORS)
INDUSTRY_MAP = {
  "AGRICULTURE": [
    "ARCHIDPLY-EQ",
    "KUANTUM-EQ",
    "PDMJEPAPER-EQ",
    "SYLVANPLY-EQ",
    "BSHSL-EQ",
    "AGRITECH-EQ",
    "TNPL-EQ",
    "WORTHPERI-EQ",
    "STARPAPER-EQ",
    "NATHBIOGEN-EQ",
    "BBTCL-EQ",
    "HARRMALAYA-EQ",
    "SHREYANIND-EQ",
    "SESHAPAPER-EQ",
    "USASEEDS-EQ",
    "GREENPLY-EQ",
    "GREENPANEL-EQ",
    "JKPAPER-EQ",
    "NIRMAN-EQ",
    "WSTCSTPAPR-EQ",
    "CENTURYPLY-EQ",
    "KOTYARK-EQ"
  ],
  "AUTO ANCILLARY": [
    "SHIGAN-EQ",
    "SINTERCOM-EQ",
    "UCAL-EQ",
    "MUNJALSHOW-EQ",
    "ASHOKLEY-EQ",
    "KALYANI-EQ",
    "PVSL-EQ",
    "JTEKTINDIA-EQ",
    "BELRISE-EQ",
    "PRECAM-EQ",
    "URAVIDEF-EQ",
    "PPAP-EQ",
    "TALBROAUTO-EQ",
    "GNA-EQ",
    "EVEREADY-EQ",
    "EXIDEIND-EQ",
    "TATAMOTORS-EQ",
    "JKTYRE-EQ",
    "SUPRAJIT-EQ",
    "FMGOETZE-EQ",
    "SONACOMS-EQ",
    "HINDCOMPOS-EQ",
    "ASKAUTOLTD-EQ",
    "CARRARO-EQ",
    "APOLLOTYRE-EQ",
    "PRICOLLTD-EQ",
    "SANDHAR-EQ",
    "SHANTIGEAR-EQ",
    "ASAL-EQ",
    "MINDACORP-EQ",
    "VARROC-EQ",
    "LANDMARK-EQ",
    "BANCOINDIA-EQ",
    "SUNDRMBRAK-EQ",
    "HITECHGEAR-EQ",
    "RML-EQ",
    "HBLENGINE-EQ",
    "WHEELS-EQ",
    "INDNIPPON-EQ",
    "ARE&M-EQ"
  ],
  "AVIATION": [
    "DREAMFOLKS-EQ",
    "GLOBALVECT-EQ",
    "GMRAIRPORT-EQ",
    "FLYSBS-EQ"
  ],
  "BUILDING MATERIALS": [
    "GSLSU-EQ",
    "AIROLAM-EQ",
    "MDL-EQ",
    "NITCO-EQ",
    "DUCOL-EQ",
    "JSWCEMENT-EQ",
    "KAKATCEM-EQ",
    "PRSMJOHNSN-EQ",
    "WIPL-EQ",
    "HEIDELBERG-EQ",
    "KCP-EQ",
    "NCLIND-EQ",
    "ORIENTCEM-EQ",
    "LAOPALA-EQ",
    "STARCEMENT-EQ",
    "KANSAINER-EQ",
    "GREENLAM-EQ",
    "SAHYADRI-EQ",
    "INM-EQ",
    "ORIENTBELL-EQ",
    "RAMCOIND-EQ",
    "INDIACEM-EQ",
    "NUVOCO-EQ",
    "SOMANYCERA-EQ",
    "BERGEPAINT-EQ",
    "AMBUJACEM-EQ",
    "EVERESTIND-EQ",
    "MANGLMCEM-EQ",
    "JKLAKSHMI-EQ",
    "ASAHIINDIA-EQ",
    "SEJALLTD-EQ"
  ],
  "CHEMICALS": [
    "PAR-EQ",
    "EMMBI-EQ",
    "GAEL-EQ",
    "SRIVASAVI-EQ",
    "MAHICKRA-EQ",
    "TNPETRO-EQ",
    "KRITI-EQ",
    "PENTAGON-EQ",
    "SIKKO-EQ",
    "TOKYOPLAST-EQ",
    "RAIN-EQ",
    "PARAGON-EQ",
    "HPIL-EQ",
    "AVROIND-EQ",
    "KOTHARIPET-EQ",
    "JOCIL-EQ",
    "PRAMARA-EQ",
    "ESFL-EQ",
    "JAICORPLTD-EQ",
    "IVP-EQ",
    "ARVEE-EQ",
    "PURVFLEXI-EQ",
    "CHEMBOND-EQ",
    "HIGREEN-EQ",
    "PLASTIBLEN-EQ",
    "RESPONIND-EQ",
    "MCON-EQ",
    "NOCIL-EQ",
    "HITECHCORP-EQ",
    "IPL-EQ",
    "AVSL-EQ",
    "TAINWALCHM-EQ",
    "LXCHEM-EQ",
    "EPL-EQ",
    "PROLIFE-EQ",
    "TIMETECHNO-EQ",
    "INFINIUM-EQ",
    "SHK-EQ",
    "JAYAGROGN-EQ",
    "CHEMCON-EQ",
    "HUHTAMAKI-EQ",
    "RALLIS-EQ",
    "TIRUMALCHM-EQ",
    "TARSONS-EQ",
    "PPL-EQ",
    "DHARMAJ-EQ",
    "NAHARPOLY-EQ",
    "HERANBA-EQ",
    "DMCC-EQ",
    "MASTER-EQ",
    "PRINCEPIPE-EQ",
    "DVL-EQ",
    "COOLCAPS-EQ",
    "GOCLCORP-EQ",
    "APCOTEXIND-EQ",
    "IEML-EQ",
    "GRCL-EQ",
    "CHEMPLASTS-EQ",
    "IGPL-EQ",
    "SHREEPUSHK-EQ",
    "GOACARBON-EQ",
    "JGCHEM-EQ",
    "HSCL-EQ",
    "MANORG-EQ",
    "SUMICHEM-EQ",
    "DICIND-EQ",
    "HEUBACHIND-EQ",
    "GUJALKALI-EQ",
    "SRHHYPOLTD-EQ",
    "JINDALPOLY-EQ",
    "KIRIINDUS-EQ",
    "CHEMFAB-EQ",
    "ACI-EQ",
    "ROSSARI-EQ",
    "GHCL-EQ",
    "UPL-EQ",
    "JUBLINGREA-EQ",
    "INSECTICID-EQ",
    "TIRUPATI-EQ",
    "AETHER-EQ",
    "FAIRCHEMOR-EQ",
    "SHARDACROP-EQ",
    "TATACHEM-EQ",
    "INDIAGLYCO-EQ",
    "POLYPLEX-EQ"
  ],
  "CONSUMER DURABLES": [
    "KHAITANLTD-EQ",
    "WEL-EQ",
    "ELIN-EQ",
    "ORIENTELEC-EQ",
    "CROMPTON-EQ",
    "BOROLTD-EQ",
    "BAJAJELEC-EQ",
    "PGEL-EQ",
    "TTKPRESTIG-EQ",
    "STOVEKRAFT-EQ"
  ],
  "DIVERSIFIED": [
    "GILLANDERS-EQ",
    "BALMLAWRIE-EQ",
    "ASPINWALL-EQ",
    "SURYAROSNI-EQ",
    "NIRAJISPAT-EQ"
  ],
  "EDUCATION TRAINING": [
    "DHTL-EQ",
    "CPCAP-EQ",
    "ARIHANTACA-EQ",
    "MOXSH-EQ",
    "VERANDA-EQ",
    "LAWSIKHO-EQ"
  ],
  "ENERGY": [
    "GMRP&UI-EQ",
    "MRPL-EQ",
    "IEX-EQ",
    "IOC-EQ",
    "HINDOILEXP-EQ",
    "PTC-EQ",
    "ADANIPOWER-EQ",
    "GAIL-EQ",
    "CESC-EQ",
    "GIPCL-EQ",
    "IGL-EQ",
    "RELINFRA-EQ",
    "ONGC-EQ",
    "NLCINDIA-EQ",
    "PETRONET-EQ",
    "POWERGRID-EQ",
    "PCCL-EQ",
    "GSPL-EQ",
    "IRMENERGY-EQ",
    "BPCL-EQ",
    "NTPC-EQ",
    "ASIANENE-EQ",
    "SEL-EQ",
    "DOLPHIN-EQ",
    "TATAPOWER-EQ",
    "GUJGASLTD-EQ",
    "OIL-EQ",
    "HINDPETRO-EQ",
    "ELLEN-EQ",
    "DEEPINDS-EQ",
    "ANTELOPUS-EQ",
    "JSWENERGY-EQ",
    "JINDRILL-EQ",
    "NAVA-EQ",
    "ATGL-EQ",
    "BFUTILITIE-EQ",
    "CHENNPETRO-EQ",
    "ADANIENSOL-EQ",
    "KKVAPOW-EQ"
  ],
  "ENGINEERING CAPITAL GOODS": [
    "INTLCONV-EQ",
    "SUPREMEINF-EQ",
    "PULZ-EQ",
    "PRESSTONIC-EQ",
    "ATLASCYCLE-EQ",
    "STLTECH-EQ",
    "VETO-EQ",
    "NELCAST-EQ",
    "KECL-EQ",
    "AMEYA-EQ",
    "SERVOTECH-EQ",
    "SCML-EQ",
    "UEL-EQ",
    "ROLEXRINGS-EQ",
    "TEXRAIL-EQ",
    "HECPROJECT-EQ",
    "SOUTHWEST-EQ",
    "DCG-EQ",
    "MAHEPC-EQ",
    "EXICOM-EQ",
    "AVPINFRA-EQ",
    "REPL-EQ",
    "KEL-EQ",
    "MANINFRA-EQ",
    "AISL-EQ",
    "EKC-EQ",
    "INOXWIND-EQ",
    "BIRLACABLE-EQ",
    "MODISONLTD-EQ",
    "PIGL-EQ",
    "IRCON-EQ",
    "WALCHANNAG-EQ",
    "AEROFLEX-EQ",
    "KONSTELEC-EQ",
    "CORDSCABLE-EQ",
    "VMARCIND-EQ",
    "PRIZOR-EQ",
    "AARON-EQ",
    "KNRCON-EQ",
    "HERCULES-EQ",
    "ASHOKA-EQ",
    "LOKESHMACH-EQ",
    "SONAMAC-EQ",
    "TCL-EQ",
    "NITIRAJ-EQ",
    "UNIDT-EQ",
    "ENGINERSIN-EQ",
    "DENEERS-EQ",
    "CASTROLIND-EQ",
    "REFRACTORY-EQ",
    "GREAVESCOT-EQ",
    "IKIO-EQ",
    "NCC-EQ",
    "EFFWA-EQ",
    "SKP-EQ",
    "SPMLINFRA-EQ",
    "STEELCAS-EQ",
    "SWSOLAR-EQ",
    "LIKHITHA-EQ",
    "DCXINDIA-EQ",
    "BHEL-EQ",
    "MARINE-EQ",
    "INOXGREEN-EQ",
    "KABRAEXTRU-EQ",
    "INDOFARM-EQ",
    "RITES-EQ",
    "IFGLEXPOR-EQ",
    "SPRL-EQ",
    "DCI-EQ",
    "BLUEPEBBLE-EQ",
    "ALPHAGEO-EQ",
    "NRBBEARING-EQ",
    "DEEDEV-EQ",
    "SAAKSHI-EQ",
    "ZTECH-EQ",
    "K2INFRA-EQ",
    "SIMPLEXINF-EQ",
    "PNCINFRA-EQ",
    "WINDMACHIN-EQ",
    "APOLLO-EQ",
    "GENUSPOWER-EQ",
    "PRATHAM-EQ",
    "ATMASTCO-EQ",
    "SUPREMEPWR-EQ",
    "MMFL-EQ",
    "TIL-EQ",
    "GVPIL-EQ",
    "STERTOOLS-EQ",
    "JWL-EQ",
    "SPCL-EQ",
    "RVNL-EQ",
    "TRF-EQ",
    "PRAJIND-EQ",
    "EMMIL-EQ",
    "INDIANHUME-EQ",
    "GIRIRAJ-EQ",
    "S&SPOWER-EQ",
    "SANGHVIMOV-EQ",
    "MEGATHERM-EQ",
    "KCEIL-EQ",
    "VGUARD-EQ",
    "DIFFNKG-EQ",
    "ZODIAC-EQ",
    "WINSOL-EQ",
    "RISHABH-EQ",
    "SAHAJSOLAR-EQ",
    "HARSHA-EQ",
    "PANACHE-EQ",
    "BEL-EQ",
    "CIEINDIA-EQ",
    "GUJAPOLLO-EQ",
    "BGRENERGY-EQ",
    "AFCONS-EQ",
    "HPL-EQ",
    "RHIM-EQ",
    "SEMAC-EQ",
    "CYIENTDLM-EQ",
    "UNIPARTS-EQ",
    "MBEL-EQ",
    "ELGIEQUIP-EQ",
    "TGL-EQ",
    "DBL-EQ",
    "TARIL-EQ",
    "GGBL-EQ",
    "VESUVIUS-EQ",
    "IDEAFORGE-EQ",
    "JASH-EQ",
    "VILAS-EQ",
    "RBMINFRA-EQ",
    "ANLON-EQ",
    "IGARASHI-EQ",
    "RULKA-EQ",
    "APS-EQ",
    "HEG-EQ",
    "TRITURBINE-EQ",
    "SGIL-EQ",
    "EMSLIMITED-EQ",
    "AIMTRON-EQ",
    "RKFORGE-EQ",
    "GRAPHITE-EQ",
    "ELECON-EQ",
    "TEMBO-EQ",
    "JKIL-EQ",
    "DIVGIITTS-EQ",
    "WAAREEINDO-EQ",
    "RAMKY-EQ",
    "TDPOWERSYS-EQ",
    "UNIVCABLES-EQ",
    "ICEMAKE-EQ",
    "SILVERTUC-EQ",
    "CGPOWER-EQ",
    "TRANSRAILL-EQ",
    "KALYANIFRG-EQ",
    "KRISHNADEF-EQ",
    "SYRMA-EQ",
    "FINCABLES-EQ",
    "CEMPRO-EQ",
    "KSB-EQ",
    "DENORA-EQ",
    "MACPOWER-EQ",
    "SWELECTES-EQ",
    "SCHNEIDER-EQ",
    "ALPEXSOLAR-EQ",
    "VIVIANA-EQ",
    "KEC-EQ",
    "NELCO-EQ",
    "ALICON-EQ",
    "TITAGARH-EQ",
    "ISGEC-EQ",
    "EMKAYTOOLS-EQ",
    "KIRLOSENG-EQ",
    "CARBORUNIV-EQ",
    "RIIL-EQ",
    "JYOTICNC-EQ",
    "YUKEN-EQ",
    "PITTIENG-EQ",
    "HGINFRA-EQ",
    "RVTH-EQ",
    "SUNDRMFAST-EQ",
    "HAPPYFORGE-EQ"
  ],
  "FMCG": [
    "KOKUYOCMLN-EQ",
    "NAMAN-EQ",
    "TAPIFRUIT-EQ",
    "DHAMPURSUG-EQ",
    "SPECIALITY-EQ",
    "HOACFOODS-EQ",
    "MADHUSUDAN-EQ",
    "DTIL-EQ",
    "GOKULAGRO-EQ",
    "DEVYANI-EQ",
    "GOYALSALT-EQ",
    "UFBL-EQ",
    "PARIN-EQ",
    "EIFFL-EQ",
    "KNAGRI-EQ",
    "APEX-EQ",
    "UTTAMSUGAR-EQ",
    "SULA-EQ",
    "VSTIND-EQ",
    "AWL-EQ",
    "GODAVARIB-EQ",
    "BAJAJCON-EQ",
    "SSFL-EQ",
    "GANESHCP-EQ",
    "CLSEL-EQ",
    "PONNIERODE-EQ",
    "PARAGMILK-EQ",
    "ANNAPURNA-EQ",
    "TBI-EQ",
    "JYOTHYLAB-EQ",
    "AURDIS-EQ",
    "TRIVENI-EQ",
    "DALMIASUG-EQ",
    "SKMEGGPROD-EQ",
    "KRBL-EQ",
    "ITC-EQ",
    "LTFOODS-EQ",
    "AVADHSUGAR-EQ",
    "VIPIND-EQ",
    "KAYA-EQ",
    "VBL-EQ",
    "UNITEDTEA-EQ",
    "BALRAMCHIN-EQ",
    "AARTISURF-EQ",
    "TI-EQ",
    "HERITGFOOD-EQ",
    "ZYDUSWELL-EQ",
    "KRISHIVAL-EQ",
    "DABUR-EQ",
    "EMAMILTD-EQ",
    "MAGADSUGAR-EQ",
    "PATANJALI-EQ",
    "WESTLIFE-EQ",
    "JUBLFOOD-EQ",
    "CELLO-EQ",
    "GODREJAGRO-EQ",
    "SFL-EQ",
    "PKTEA-EQ",
    "MARICO-EQ",
    "AVANTIFEED-EQ",
    "BIKAJI-EQ",
    "SUNDROP-EQ",
    "CCL-EQ",
    "CARYSIL-EQ"
  ],
  "FERTILIZERS": [
    "KHAICHEM-EQ",
    "RCF-EQ",
    "PARADEEP-EQ",
    "GSFC-EQ",
    "ZUARI-EQ",
    "MANGCHEFER-EQ",
    "ARIES-EQ",
    "MBAPL-EQ",
    "CHAMBLFERT-EQ",
    "GNFC-EQ",
    "KRISHANA-EQ",
    "FACT-EQ"
  ],
  "FINANCIAL SERVICES": [
    "J&KBANK-EQ",
    "CANHLIFE-EQ",
    "SBFC-EQ",
    "BEACON-EQ",
    "PNB-EQ",
    "SHREMINVIT-EQ",
    "PALASHSECU-EQ",
    "EDELWEISS-EQ",
    "IRFC-EQ",
    "CANBK-EQ",
    "ASCOM-EQ",
    "BANKINDIA-EQ",
    "SMCGLOBAL-EQ",
    "INDIGRID-EQ",
    "FEDFINA-EQ",
    "UNIONBANK-EQ",
    "SATIN-EQ",
    "SURYODAY-EQ",
    "NXST-EQ",
    "IREDA-EQ",
    "DCBBANK-EQ",
    "GANGESSECU-EQ",
    "BIRLAMONEY-EQ",
    "BANDHANBNK-EQ",
    "JMFINANCIL-EQ",
    "GICHSGFIN-EQ",
    "UGROCAP-EQ",
    "KTKBANK-EQ",
    "SAMMAANCAP-EQ",
    "NIACL-EQ",
    "CONSOFINVT-EQ",
    "MOS-EQ",
    "IITL-EQ",
    "AFSL-EQ",
    "LFIC-EQ",
    "MASKINVEST-EQ",
    "VLSFINANCE-EQ",
    "CUB-EQ",
    "HUDCO-EQ",
    "FEDERALBNK-EQ",
    "DELPHIFX-EQ",
    "INDOSTAR-EQ",
    "KARURVYSYA-EQ",
    "SPANDANA-EQ",
    "TEAMGTY-EQ",
    "DAMCAPITAL-EQ",
    "RELIGARE-EQ",
    "BIRET-EQ",
    "KEYFINSERV-EQ",
    "BANKBARODA-EQ",
    "LTF-EQ",
    "NORTHARC-EQ",
    "PRIMESECU-EQ",
    "MANAPPURAM-EQ",
    "CAPITALSFB-EQ",
    "NAHARCAP-EQ",
    "SRGHFL-EQ",
    "M&MFIN-EQ",
    "MASFIN-EQ",
    "SASTASUNDR-EQ",
    "ABCAPITAL-EQ",
    "JIOFIN-EQ",
    "MONARCH-EQ",
    "5PAISA-EQ",
    "APTUS-EQ",
    "RBLBANK-EQ",
    "FINOPB-EQ",
    "TATACAP-EQ",
    "EMKAY-EQ",
    "CRAMC-EQ",
    "MINDSPACE-EQ",
    "ZAGGLE-EQ",
    "IIFLCAPS-EQ",
    "RECLTD-EQ",
    "CREST-EQ",
    "GICRE-EQ",
    "INDOTHAI-EQ",
    "PFC-EQ",
    "CSBBANK-EQ",
    "REPCOHOME-EQ",
    "STEL-EQ",
    "TMB-EQ",
    "POONAWALLA-EQ",
    "STARHEALTH-EQ",
    "IIFL-EQ",
    "AADHARHFC-EQ",
    "ARSSBL-EQ",
    "FIVESTAR-EQ",
    "LICHSGFIN-EQ",
    "ICICIPRULI-EQ",
    "TSFINV-EQ",
    "SILINV-EQ",
    "SHRIRAMFIN-EQ",
    "HDBFS-EQ",
    "HDFCLIFE-EQ",
    "INDUSINDBK-EQ",
    "INDIANB-EQ",
    "TATAINVEST-EQ",
    "ABSLAMC-EQ",
    "CANFINHOME-EQ",
    "AUBANK-EQ",
    "ISEC-EQ",
    "LICI-EQ",
    "PNBHOUSING-EQ",
    "SBIN-EQ",
    "SBICARD-EQ",
    "NAM-INDIA-EQ"
  ],
  "HEALTHCARE": [
    "ALEMBICLTD-EQ",
    "MEDIORG-EQ",
    "INDSWFTLAB-EQ",
    "HOLMARC-EQ",
    "WALPAR-EQ",
    "LOTUSEYE-EQ",
    "BROOKS-EQ",
    "MAITREYA-EQ",
    "UNIHEALTH-EQ",
    "AMANTA-EQ",
    "THEMISMED-EQ",
    "BAFNAPH-EQ",
    "QUESTLAB-EQ",
    "MADHAVBAUG-EQ",
    "SOTAC-EQ",
    "BLISSGVS-EQ",
    "KOPRAN-EQ",
    "MARKSANS-EQ",
    "PPLPHARMA-EQ",
    "JAGSNPHARM-EQ",
    "WANBURY-EQ",
    "NEPHROCARE-EQ",
    "HIKAL-EQ",
    "ARTEMISMED-EQ",
    "SHALBY-EQ",
    "DCAL-EQ",
    "INDOCO-EQ",
    "SMSPHARMA-EQ",
    "ACCENTMIC-EQ",
    "ADVENZYMES-EQ",
    "SURAKSHA-EQ",
    "NURECA-EQ",
    "GUFICBIO-EQ",
    "MEDICAMEQ-EQ",
    "SHILPAMED-EQ",
    "BIOCON-EQ",
    "SAKAR-EQ",
    "FDC-EQ",
    "PANACEABIO-EQ",
    "VENUSREM-EQ",
    "AKUMS-EQ",
    "UNICHEMLAB-EQ",
    "AARTIDRUGS-EQ",
    "INDRAMEDCO-EQ",
    "INDGN-EQ",
    "GRANULES-EQ",
    "RUBICON-EQ",
    "BLUEJET-EQ",
    "VIMTALABS-EQ",
    "ASTERDM-EQ",
    "AMRUTANJAN-EQ",
    "KIMS-EQ",
    "ANTHEM-EQ",
    "HCG-EQ",
    "ORCHPHARMA-EQ",
    "SUPRIYA-EQ",
    "MEDPLUS-EQ",
    "KRSNAA-EQ",
    "YATHARTH-EQ",
    "STAR-EQ",
    "NATCOPHARM-EQ",
    "AARTIPHARM-EQ",
    "COHANCE-EQ",
    "SAILIFE-EQ",
    "LAURUSLABS-EQ",
    "APLLTD-EQ",
    "ALIVUS-EQ",
    "WINDLAS-EQ",
    "VIJAYA-EQ",
    "ZYDUSLIFE-EQ"
  ],
  "LOGISTICS": [
    "DRSDILIP-EQ",
    "NAVKARCORP-EQ",
    "TVSSCS-EQ",
    "PRLIND-EQ",
    "GPPL-EQ",
    "ZEAL-EQ",
    "SADHAV-EQ",
    "AVG-EQ",
    "TRANSWORLD-EQ",
    "SCI-EQ",
    "VRLLOG-EQ",
    "JSWINFRA-EQ",
    "ABSMARINE-EQ",
    "JITFINFRA-EQ",
    "MAHLOG-EQ",
    "SJLOGISTIC-EQ",
    "DELHIVERY-EQ",
    "CONCOR-EQ",
    "DREDGECORP-EQ",
    "TCIEXP-EQ",
    "SWANDEF-EQ",
    "AEGISLOG-EQ",
    "SVLL-EQ",
    "SEAMECLTD-EQ"
  ],
  "MEDIA ENTERTAINMENT": [
    "ZEEL-EQ",
    "GTPL-EQ",
    "SHEMAROO-EQ",
    "BALAJITELE-EQ",
    "CRAYONS-EQ",
    "VERITAAS-EQ",
    "ENIL-EQ",
    "CMRSL-EQ",
    "TVTODAY-EQ",
    "NAVNETEDUL-EQ",
    "PFOCUS-EQ",
    "SCHAND-EQ",
    "EFACTOR-EQ",
    "DBCORP-EQ",
    "PHANTOMFX-EQ",
    "TIPSFILMS-EQ",
    "DIGIKORE-EQ",
    "TIPSMUSIC-EQ",
    "REPRO-EQ",
    "SUNTV-EQ",
    "BASILIC-EQ",
    "WONDERLA-EQ",
    "SABTNL-EQ"
  ],
  "METALS": [
    "DPEL-EQ",
    "QFIL-EQ",
    "HITECH-EQ",
    "MUKANDLTD-EQ",
    "SAIL-EQ",
    "BEDMUTHA-EQ",
    "MAANALU-EQ",
    "SHANKARA-EQ",
    "JAINAM-EQ",
    "RATNAVEER-EQ",
    "MANAKCOAT-EQ",
    "PRAKASH-EQ",
    "BHARATWIRE-EQ",
    "JINDALSAW-EQ",
    "SHERA-EQ",
    "20MICRONS-EQ",
    "EUROBOND-EQ",
    "DPWIRES-EQ",
    "NATIONALUM-EQ",
    "MWL-EQ",
    "GPIL-EQ",
    "SUNFLAG-EQ",
    "MMP-EQ",
    "BAHETI-EQ",
    "VSSL-EQ",
    "BANSALWIRE-EQ",
    "SURANI-EQ",
    "HINDCOPPER-EQ",
    "KRISHCA-EQ",
    "MOIL-EQ",
    "MIDHANI-EQ",
    "COALINDIA-EQ",
    "MANINDS-EQ",
    "KIOCL-EQ",
    "USHAMART-EQ",
    "HARIOMPIPE-EQ",
    "HINDZINC-EQ",
    "VEDL-EQ",
    "SARDAEN-EQ",
    "GALLANTT-EQ",
    "MAHSEAMLES-EQ",
    "GMDCLTD-EQ",
    "ASHAPURMIN-EQ",
    "JSL-EQ",
    "HINDALCO-EQ",
    "WELCORP-EQ",
    "KSL-EQ",
    "SHYAMMETL-EQ",
    "GANDHITUBE-EQ"
  ],
  "MISCELLANEOUS": [
    "KONTOR-EQ",
    "SHREEOSFM-EQ",
    "PRABHA-EQ",
    "PROPEQUITY-EQ",
    "ROSSTECH-EQ"
  ],
  "PACKAGING": [
    "PYRAMID-EQ",
    "CGRAPHICS-EQ",
    "SATIPOLY-EQ",
    "UFLEX-EQ",
    "AGI-EQ"
  ],
  "REAL ESTATE": [
    "NBCC-EQ",
    "RPPINFRA-EQ",
    "HEMIPROP-EQ",
    "TARC-EQ",
    "DBREALTY-EQ",
    "MASON-EQ",
    "CHAVDA-EQ",
    "VR-EQ",
    "BLAL-EQ",
    "VISHNUINFR-EQ",
    "PANSARI-EQ",
    "CAPACITE-EQ",
    "PURVA-EQ",
    "HUBTOWN-EQ",
    "GEECEE-EQ",
    "KALPATARU-EQ",
    "KOLTEPATIL-EQ",
    "SUNTECK-EQ",
    "MAXESTATES-EQ",
    "WELENT-EQ",
    "RUSTOMJEE-EQ",
    "ARVSMART-EQ",
    "ANANTRAJ-EQ",
    "HOMESFY-EQ",
    "DLF-EQ",
    "PSPPROJECT-EQ",
    "GANESHHOU-EQ",
    "AHLUCONT-EQ"
  ],
  "RETAIL": [
    "RADIOWALLA-EQ",
    "BMETRICS-EQ",
    "GOLDKART-EQ",
    "RGL-EQ",
    "EMIL-EQ",
    "RAJESHEXPO-EQ",
    "TBZ-EQ",
    "ENFUSE-EQ",
    "KALAMANDIR-EQ",
    "FONEBOX-EQ",
    "KHADIM-EQ",
    "NYKAA-EQ",
    "WOMANCART-EQ",
    "CAMPUS-EQ",
    "LGHL-EQ",
    "LIBERTSHOE-EQ",
    "ETERNAL-EQ",
    "THOMASCOTT-EQ",
    "ONDOOR-EQ",
    "KALYANKJIL-EQ",
    "ARVINDFASN-EQ",
    "SHOPERSTOP-EQ",
    "ROCKINGDCE-EQ",
    "GOCOLORS-EQ",
    "KORE-EQ",
    "BIL-EQ",
    "VMART-EQ"
  ],
  "SERVICES": [
    "TOUCHWOOD-EQ",
    "LLOYDS-EQ",
    "SPECTSTM-EQ",
    "AARVI-EQ",
    "PARTYCRUS-EQ",
    "IPSL-EQ",
    "UDS-EQ",
    "UNIVPHOTO-EQ",
    "FELIX-EQ",
    "WINNY-EQ",
    "KAPSTON-EQ",
    "BLS-EQ",
    "SIS-EQ",
    "DRONE-EQ",
    "URBAN-EQ",
    "AWHCL-EQ"
  ],
  "SOFTWARE SERVICES": [
    "ISFT-EQ",
    "NIITLTD-EQ",
    "FIDEL-EQ",
    "APTECHT-EQ",
    "ROXHITECH-EQ",
    "SPARC-EQ",
    "SMARTLINK-EQ",
    "QUICKTOUCH-EQ",
    "XELPMOC-EQ",
    "RELIABLE-EQ",
    "CROWN-EQ",
    "CURAA-EQ",
    "VERTEXPLUS-EQ",
    "CYBERTECH-EQ",
    "AURUM-EQ",
    "TRUST-EQ",
    "MEGASOFT-EQ",
    "ADSL-EQ",
    "DTL-EQ",
    "SAKSOFT-EQ",
    "CADSYS-EQ",
    "ENSER-EQ",
    "TREJHARA-EQ",
    "QUESS-EQ",
    "WIPRO-EQ",
    "SYSTANGO-EQ",
    "NAZARA-EQ",
    "DELAPLEX-EQ",
    "INFOLLION-EQ",
    "ONWARDTEC-EQ",
    "KSOLVES-EQ",
    "FSL-EQ",
    "NIITMTS-EQ",
    "QUICKHEAL-EQ",
    "ALLETEC-EQ",
    "SOFTTECH-EQ",
    "CMSINFO-EQ",
    "BSOFT-EQ",
    "VINSYS-EQ",
    "RSYSTEMS-EQ",
    "LATENTVIEW-EQ",
    "DLINKINDIA-EQ",
    "HGS-EQ",
    "TAC-EQ",
    "MATRIMONY-EQ",
    "INNOVANA-EQ",
    "HAPPSTMNDS-EQ",
    "INFOBEAN-EQ",
    "GENESYS-EQ",
    "MEDIASSIST-EQ",
    "RAMCOSYS-EQ",
    "KRYSTAL-EQ",
    "TANLA-EQ",
    "EMUDHRA-EQ",
    "RATEGAIN-EQ",
    "SYNGENE-EQ",
    "TATATECH-EQ",
    "TECHLABS-EQ",
    "ROUTE-EQ",
    "HEXT-EQ",
    "JUSTDIAL-EQ",
    "63MOONS-EQ",
    "ZENSARTECH-EQ",
    "IZMO-EQ",
    "RPSGVENT-EQ",
    "NEWGEN-EQ",
    "DATAMATICS-EQ",
    "ALLDIGI-EQ",
    "DSSL-EQ",
    "INTELLECT-EQ"
  ],
  "TELECOM": [
    "KAVDEFENCE-EQ",
    "SARTELE-EQ",
    "FROG-EQ",
    "ITI-EQ",
    "INDUSTOWER-EQ",
    "RAILTEL-EQ",
    "TEJASNET-EQ"
  ],
  "TEXTILES": [
    "ZODIACLOTH-EQ",
    "DONEAR-EQ",
    "SOMATEX-EQ",
    "WEIZMANIND-EQ",
    "NAHARINDUS-EQ",
    "JETKNIT-EQ",
    "HIMATSEIDE-EQ",
    "WELSPUNLIV-EQ",
    "GRETEX-EQ",
    "VGL-EQ",
    "SIGNORIA-EQ",
    "DCMNVL-EQ",
    "BOMDYEING-EQ",
    "SHIVATEX-EQ",
    "LEMERITE-EQ",
    "BSL-EQ",
    "NAHARSPING-EQ",
    "VEEKAYEM-EQ",
    "ZENITHEXPO-EQ",
    "CANTABIL-EQ",
    "INDIANCARD-EQ",
    "LOYALTEX-EQ",
    "CPS-EQ",
    "ICIL-EQ",
    "SWARAJ-EQ",
    "KARNIKA-EQ",
    "ARVIND-EQ",
    "NITINSPIN-EQ",
    "PDSL-EQ",
    "DOLLAR-EQ",
    "ABCOTS-EQ",
    "SANGAMIND-EQ",
    "PRECOT-EQ",
    "CENTENKA-EQ",
    "SANATHAN-EQ",
    "FAZE3Q-EQ",
    "KKCL-EQ",
    "RAYMOND-EQ",
    "MANYAVAR-EQ",
    "MONTECARLO-EQ",
    "SPAL-EQ",
    "SIYSIL-EQ",
    "SHREEKARNI-EQ",
    "GARFIBRES-EQ",
    "PASHUPATI-EQ",
    "GOKEX-EQ"
  ],
  "TOURISM HOSPITALITY": [
    "GIRRESORTS-EQ",
    "ORIENTHOT-EQ",
    "VHLTD-EQ",
    "PARKHOTELS-EQ",
    "AHLEAST-EQ",
    "YATRA-EQ",
    "THOMASCOOK-EQ",
    "LEMONTREE-EQ",
    "SAMHI-EQ",
    "RHL-EQ",
    "WTICAB-EQ",
    "KAMATHOTEL-EQ",
    "ASIANHOTNR-EQ",
    "MHRIL-EQ",
    "EIHAHOTELS-EQ",
    "EIHOTEL-EQ",
    "TAJGVK-EQ",
    "ROHLTD-EQ",
    "IRCTC-EQ",
    "CHALET-EQ"
  ],
  "TRADING": [
    "AUSOMENT-EQ",
    "SYNOPTICS-EQ",
    "ASPIRE-EQ",
    "MUFTI-EQ",
    "QMSMEDI-EQ",
    "NEWJAISA-EQ",
    "OMAXAUTO-EQ",
    "STCINDIA-EQ",
    "UNIENTER-EQ",
    "SIDDHIKA-EQ",
    "SLONE-EQ",
    "HEXATRADEX-EQ",
    "MVGJL-EQ",
    "KCK-EQ",
    "SREEL-EQ",
    "REDINGTON-EQ",
    "HONASA-EQ",
    "DYNAMIC-EQ",
    "VINYLINDIA-EQ",
    "CELLECOR-EQ",
    "SHIVAUM-EQ",
    "ESCONET-EQ",
    "RPTECH-EQ",
    "ZUARIIND-EQ",
    "BCONCEPTS-EQ",
    "GPECO-EQ",
    "SIRCA-EQ",
    "MSTCLTD-EQ",
    "DENTALKART-EQ",
    "TVSELECT-EQ",
    "CREATIVE-EQ",
    "JSLL-EQ"
  ]
}

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
# AUTO-RESUME CHECKPOINT SYSTEM
# ============================================================================

CHECKPOINT_FILE = Path("scan_checkpoint.pkl")

def save_checkpoint(completed_stocks, all_results, failed_stocks, trading_style, timeframe, api_provider):
    """Save scan checkpoint to disk"""
    try:
        checkpoint = {
            'completed_stocks': completed_stocks,
            'all_results': all_results,
            'failed_stocks': failed_stocks,
            'trading_style': trading_style,
            'timeframe': timeframe,
            'api_provider': api_provider,
            'timestamp': datetime.now()
        }
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(checkpoint, f)
        logging.info(f"Checkpoint saved: {len(completed_stocks)} stocks completed")
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {str(e)}")

def load_checkpoint():
    """Load scan checkpoint from disk"""
    try:
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, 'rb') as f:
                checkpoint = pickle.load(f)
            age = datetime.now() - checkpoint['timestamp']
            if age.total_seconds() < 3600:  # 1 hour expiry
                logging.info(f"Checkpoint found: {len(checkpoint['completed_stocks'])} stocks already processed")
                return checkpoint
            else:
                logging.info("Checkpoint expired (>1 hour old), starting fresh")
                clear_checkpoint()
        return None
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {str(e)}")
        return None

def clear_checkpoint():
    """Remove checkpoint file"""
    try:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logging.info("Checkpoint cleared")
    except Exception as e:
        logging.error(f"Failed to clear checkpoint: {str(e)}")

def get_scan_progress_info():
    """Get detailed scan progress information"""
    try:
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, 'rb') as f:
                checkpoint = pickle.load(f)
            
            completed = len(checkpoint.get('completed_stocks', []))
            successful = len(checkpoint.get('all_results', []))
            failed = len(checkpoint.get('failed_stocks', []))
            timestamp = checkpoint.get('timestamp', datetime.now())
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60
            
            return {
                'exists': True,
                'completed': completed,
                'successful': successful,
                'failed': failed,
                'success_rate': (successful / completed * 100) if completed > 0 else 0,
                'age_minutes': age_minutes,
                'trading_style': checkpoint.get('trading_style', 'unknown'),
                'timeframe': checkpoint.get('timeframe', 'unknown'),
                'api_provider': checkpoint.get('api_provider', 'unknown')
            }
    except Exception as e:
        logging.error(f"Error reading checkpoint info: {e}")
    
    return {'exists': False}

# ============================================================================
# MARKET BREADTH & SECTOR PERFORMANCE
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
    
    breadth = breadth_data.get('breadth', {})
    total = breadth.get('total', 1)
    advancing = breadth.get('advancing', 0)
    
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
    """Calculate alignment with overall market breadth (±5 points)"""
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
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_technical_screener():
    """Fetch stocks with strong bullish technical indicators"""
    try:
        import time
        nocache = int(time.time() * 1000)
        response = requests.get(
            f"https://brkpoint.in/api/technical-indicators?nocache={nocache}",
            timeout=15,
            headers={"User-Agent": random.choice(USER_AGENTS)}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.warning(f"Technical screener API failed: {str(e)}")
        return None

def filter_bullish_stocks(tech_data, strong_only=True):
    """Filter stocks showing strong bullish signals"""
    if not tech_data or not tech_data.get('success'):
        return []
    
    stocks = tech_data.get('data', [])
    filtered = []
    
    for stock in stocks:
        adx_trend = stock.get('adx_trend', '')
        volume_trend = stock.get('volume_trend', '')
        macd_trend = stock.get('macd_trend', '')
        
        if strong_only:
            # All three must show strong bullish signals
            if ('Strong Bullish' in adx_trend and 
                'Strong Bullish' in volume_trend and 
                macd_trend == 'Bullish'):
                filtered.append({
                    'Symbol': stock.get('tradingsymbol'),
                    'Price': stock.get('live_price'),
                    'RSI': stock.get('rsi'),
                    'ADX': stock.get('adx'),
                    'ADX Trend': adx_trend,
                    'Volume Trend': volume_trend,
                    'MACD Trend': macd_trend,
                    'Stage': stock.get('stage'),
                    'Above EMA200': stock.get('above_ema200'),
                    'Volume Ratio': stock.get('volume_ratio'),
                    'Target': stock.get('next_target'),
                    'Stop Loss': stock.get('stop_loss')
                })
        else:
            # At least 2 out of 3 bullish
            bullish_count = 0
            if 'Bullish' in adx_trend: bullish_count += 1
            if 'Bullish' in volume_trend: bullish_count += 1
            if macd_trend == 'Bullish': bullish_count += 1
            
            if bullish_count >= 2:
                filtered.append({
                    'Symbol': stock.get('tradingsymbol'),
                    'Price': stock.get('live_price'),
                    'RSI': stock.get('rsi'),
                    'ADX': stock.get('adx'),
                    'ADX Trend': adx_trend,
                    'Volume Trend': volume_trend,
                    'MACD Trend': macd_trend,
                    'Stage': stock.get('stage'),
                    'Above EMA200': stock.get('above_ema200'),
                    'Volume Ratio': stock.get('volume_ratio'),
                    'Target': stock.get('next_target'),
                    'Stop Loss': stock.get('stop_loss')
                })
    
    return filtered

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_index_scan():
    """Fetch real-time index prices and changes"""
    try:
        response = requests.get(
            "https://brkpoint.in/api/indexscan",
            timeout=10,
            headers={"User-Agent": random.choice(USER_AGENTS)}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.warning(f"Index scan API failed: {str(e)}")
        return None

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
    """Calculate bonus/penalty based on index alignment (±5 points)"""
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
# NEWS FETCHING
# ============================================================================

def get_security_id_from_symbol(symbol):
    """Map trading symbol to SecurityID for StockEdge API"""
    # Remove -EQ suffix if present
    clean_symbol = symbol.replace('-EQ', '').strip()
    
    # Common stock symbol to SecurityID mapping (partial list)
    # This would ideally be in a database or loaded from a file
    symbol_to_id = {
        'RELIANCE': 6689, 'TCS': 6690, 'HDFCBANK': 6691, 'INFY': 6692,
        'ICICIBANK': 6693, 'HINDUNILVR': 6694, 'ITC': 5535, 'SBIN': 6695,
        'BHARTIARTL': 6696, 'KOTAKBANK': 6697, 'LT': 6698, 'AXISBANK': 6699,
        'ASIANPAINT': 6700, 'MARUTI': 6701, 'SUNPHARMA': 6702, 'TITAN': 6703,
        'ULTRACEMCO': 6704, 'NESTLEIND': 6705, 'BAJFINANCE': 6706, 'WIPRO': 5199,
        'TATASTEEL': 6707, 'HCLTECH': 6708, 'POWERGRID': 6709, 'NTPC': 7760,
        'ONGC': 6710, 'COALINDIA': 6711, 'M&M': 6712, 'TECHM': 6713,
        'TATAMOTORS': 6714, 'INDUSINDBK': 6715, 'ADANIGREEN': 6716, 'DRREDDY': 6717,
        'JSWSTEEL': 6718, 'BRITANNIA': 6719, 'CIPLA': 5128, 'DIVISLAB': 6720,
        'EICHERMOT': 6721, 'HEROMOTOCO': 6722, 'BAJAJFINSV': 6723, 'SHREECEM': 6724,
        'GRASIM': 6725, 'APOLLOHOSP': 6726, 'BPCL': 6727, 'HINDALCO': 6728,
        'ADANIPORTS': 6729, 'TATACONSUM': 6730, 'UPL': 7381, 'SIEMENS': 4902,
        'DLF': 6012, 'BIOCON': 6434, 'LUPIN': 6247, 'MARICO': 5699,
        'VOLTAS': 5838, 'SYMPHONY': 5835, 'MPHASIS': 6350, 'CYIENT': 5462,
        'COFORGE': 7260, 'MASTEK': 7850, 'TRENT': 8185, 'REDINGTON': 8349,
        'NMDC': 7232, 'MOIL': 6889, 'BEML': 5965, 'NCC': 8254,
        'DLF': 6012, 'SOBHA': 7542, 'OMAXE': 8004, 'NHPC': 8183,
        'SJVN': 4901, 'CESC': 6870, 'NAVA': 7731, 'THERMAX': 5528,
        'KOLTEPATIL': 4901, 'BANKBARODA': 6731  # Add more as needed
    }
    
    return symbol_to_id.get(clean_symbol)

def fetch_stock_news(symbol, security_id=None, page=1, page_size=10):
    """Fetch latest news for a stock from StockEdge API"""
    try:
        # If security_id not provided, try to get it from symbol
        if security_id is None:
            security_id = get_security_id_from_symbol(symbol)
            if security_id is None:
                logging.warning(f"No SecurityID mapping found for {symbol}")
                return None
        
        url = f"https://api.stockedge.com/Api/SecurityDashboardApi/GetNewsitemsForSecurity/{security_id}?page={page}&pageSize={page_size}&lang=en"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        news_data = response.json()
        
        if news_data and isinstance(news_data, list):
            return news_data
        return None
        
    except requests.exceptions.Timeout:
        logging.error(f"Timeout fetching news for {symbol}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching news for {symbol}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching news for {symbol}: {e}")
        return None

def analyze_news_sentiment(text):
    """Analyze news sentiment based on keywords - returns (sentiment, emoji, color)"""
    if not text:
        return "Neutral", "⚪", "#808080"
    
    text_lower = text.lower()
    
    # Positive keywords
    positive_words = [
        'profit', 'gain', 'up', 'high', 'growth', 'surge', 'jump', 'rally', 
        'boost', 'strong', 'positive', 'beat', 'exceed', 'record', 'success',
        'bullish', 'upgrade', 'outperform', 'buy', 'breakout', 'momentum',
        'dividend', 'bonus', 'expansion', 'innovation', 'award', 'partnership'
    ]
    
    # Negative keywords
    negative_words = [
        'loss', 'fall', 'down', 'low', 'decline', 'drop', 'crash', 'bearish',
        'weak', 'negative', 'miss', 'underperform', 'sell', 'downgrade', 
        'lawsuit', 'fraud', 'scandal', 'bankruptcy', 'debt', 'warning',
        'concern', 'risk', 'plunge', 'slump', 'cut', 'layoff', 'deficit'
    ]
    
    # Count positive and negative words
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    # Determine sentiment
    if positive_count > negative_count:
        return "Positive", "🟢", "#28a745"
    elif negative_count > positive_count:
        return "Negative", "🔴", "#dc3545"
    else:
        return "Neutral", "⚪", "#6c757d"

def display_stock_news(symbol, max_news=5):
    """Display news for a stock in Streamlit"""
    with st.spinner(f"📰 Fetching latest news for {symbol}..."):
        news = fetch_stock_news(symbol, page_size=max_news)
        
        if news:
            st.markdown(f"#### 📰 Latest News for {symbol}")
            st.caption(f"Found {len(news)} recent news item(s)")
            
            for i, item in enumerate(news[:max_news], 1):
                # Parse date and format it nicely
                try:
                    date_str = item.get('Date', '')
                    if date_str:
                        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        formatted_date = date_obj.strftime('%d %b %Y')
                    else:
                        formatted_date = 'N/A'
                except:
                    formatted_date = item.get('Date', 'N/A')
                
                # Get time
                time_str = item.get('Time', '')
                
                # Create headline with date and time
                headline = item.get('Description', 'No headline')
                date_time_display = f"{formatted_date}"
                if time_str and time_str != "12:00 am":
                    date_time_display += f" at {time_str}"
                
                # Analyze sentiment
                full_text = f"{headline} {item.get('Caption', '')} {item.get('Details', '')}"
                sentiment, emoji, color = analyze_news_sentiment(full_text)
                
                # Create expander with headline and sentiment
                with st.expander(f"{emoji} {headline}", expanded=(i==1)):
                    # Show sentiment badge
                    st.markdown(f"<span style='background-color:{color}; color:white; padding:3px 10px; border-radius:5px; font-size:12px; font-weight:bold;'>{sentiment}</span>", unsafe_allow_html=True)
                    st.write("")  # spacing
                    
                    # Show date and category
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.caption(f"🗓️ {date_time_display}")
                        if item.get('SubSectionName'):
                            st.caption(f"📁 Category: {item.get('SubSectionName')}")
                    
                    with col2:
                        st.caption(f"🏢 Sector: {item.get('SectorName', 'N/A')}")
                        st.caption(f"🏭 Industry: {item.get('IndustryName', 'N/A')}")
                    
                    st.divider()
                    
                    # Show caption if available (summary)
                    if item.get('Caption'):
                        st.info(f"**Summary:** {item.get('Caption')}")
                    
                    # Show full details if available
                    if item.get('Details'):
                        # Remove HTML tags for cleaner display
                        import re
                        import html
                        details = item.get('Details', '')
                        # Simple HTML tag removal (basic)
                        clean_details = re.sub('<[^<]+?>', '', details)
                        clean_details = html.unescape(clean_details).strip()
                        
                        if clean_details and len(clean_details) > 10:
                            st.markdown("**📄 Full Article:**")
                            # Use blockquote style for the article
                            st.markdown(f"> {clean_details}")
        else:
            st.info(f"ℹ️ No news found for {symbol}. News might not be available for this stock.")

# ============================================================================
# API & DATA FETCHING (MODIFIED FOR MULTI-API)
# ============================================================================

def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()

# ============================================================================
# SYMBOL FORMAT UTILITIES
# ============================================================================

def normalize_symbol(symbol, api_provider="SmartAPI"):
    """
    Normalize symbol format based on API provider.
    SmartAPI uses: SYMBOL-EQ
    Dhan uses: SYMBOL (plain)
    """
    # Remove -EQ suffix if present
    base_symbol = symbol.replace("-EQ", "")
    
    if api_provider == "SmartAPI":
        return f"{base_symbol}-EQ"
    else:  # Dhan
        return base_symbol

def get_base_symbol(symbol):
    """Extract base symbol without exchange suffix"""
    return symbol.replace("-EQ", "")

# --- Dhan Specific Setup ---
def get_dhan_client():
    """Initializes and returns the Dhan API client."""
    global _global_dhan_client
    if _global_dhan_client is None:
        if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
            logging.error("Dhan credentials (DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN) missing.")
            return None
        try:
            _global_dhan_client = dhanhq(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
            logging.info("Dhan API client initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Dhan client: {e}")
            return None
    return _global_dhan_client

@st.cache_data(ttl=86400)
def load_dhan_security_id_map():
    """Fetches and caches a mapping from base symbol to Dhan's security_id."""
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    try:
        df = pd.read_csv(url, low_memory=False)
        df.columns = df.columns.str.strip()
        
        logging.info(f"Total rows in Dhan CSV: {len(df)}")
        
        # Use the CORRECT Dhan filtering logic
        # SEM_EXM_EXCH_ID = 'NSE' (exchange)
        # SEM_SEGMENT = 'E' (equity segment)
        nse_eq_df = df[(df['SEM_EXM_EXCH_ID'] == 'NSE') & (df['SEM_SEGMENT'] == 'E')]
        
        logging.info(f"NSE Equity stocks found: {len(nse_eq_df)}")
        
        if nse_eq_df.empty:
            logging.error("No NSE Equity instruments found in Dhan master file!")
            return {}
        
        # Create mapping using trading symbol
        security_map = {}
        for index, row in nse_eq_df.iterrows():
            # Try multiple symbol columns
            symbol = None
            for col in ['SEM_TRADING_SYMBOL', 'SEM_CUSTOM_SYMBOL', 'SM_SYMBOL_NAME']:
                if col in row and pd.notna(row[col]):
                    symbol = str(row[col]).strip()
                    break
            
            sec_id = str(row['SEM_SMST_SECURITY_ID']).strip()
            
            if symbol and sec_id and sec_id != 'nan':
                security_map[symbol] = sec_id
        
        logging.info(f"Loaded {len(security_map)} Dhan NSE Equity instruments.")
        if security_map:
            # Show sample mappings
            sample = list(security_map.items())[:5]
            logging.info(f"Sample mappings: {sample}")
        
        return security_map
    except Exception as e:
        logging.error(f"Error loading Dhan master CSV: {e}", exc_info=True)
        return {}
# --- SmartAPI Specific Setup ---
def get_global_smart_api():
    """Manage global SmartAPI session with auto-refresh"""
    global _global_smart_api, _session_timestamp
    now = time_module.time()
    
    if _global_smart_api is None or (now - _session_timestamp) > SESSION_EXPIRY:
        try:
            _global_smart_api = init_smartapi_client()
            _session_timestamp = now
            if not _global_smart_api:
                logging.error("Failed to create SmartAPI session")
                return None
        except Exception as e:
            logging.error(f"Session creation error: {str(e)}")
            return None
    return _global_smart_api

def init_smartapi_client():
    """Initialize SmartAPI client with authentication"""
    try:
        if not all([CLIENT_ID, PASSWORD, TOTP_SECRET, API_KEYS["Historical"]]):
             logging.error("SmartAPI credentials missing.")
             return None
        smart_api = SmartConnect(api_key=API_KEYS["Historical"])
        totp = pyotp.TOTP(TOTP_SECRET)
        data = smart_api.generateSession(CLIENT_ID, PASSWORD, totp.now())
        
        if data['status']:
            logging.info("SmartAPI session created successfully")
            return smart_api
        else:
            logging.error(f"SmartAPI auth failed: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        logging.error(f"Error initializing SmartAPI: {str(e)}")
        return None

@st.cache_data(ttl=86400)
def load_symbol_token_map():
    """Load instrument token mapping for SmartAPI"""
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        return {entry["symbol"]: entry["token"] for entry in data if "symbol" in entry and "token" in entry}
    except Exception as e:
        logging.warning(f"Failed to load SmartAPI instrument list: {str(e)}")
        return {}

# --- Universal Fetching Logic ---
def retry_with_exponential_backoff(max_retries=5, base_delay=3):
    """Enhanced retry with exponential backoff and jitter"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        if attempt == max_retries:
                            logging.error(f"Rate limit exceeded after {max_retries} attempts")
                            raise
                        sleep_time = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                        logging.warning(f"Rate limited. Waiting {sleep_time:.1f}s (attempt {attempt}/{max_retries})")
                        time_module.sleep(sleep_time)
                    else:
                        raise
                except requests.exceptions.Timeout:
                    if attempt == max_retries:
                        logging.error("Request timeout after retries")
                        raise
                    sleep_time = base_delay * attempt
                    logging.warning(f"Timeout. Retrying in {sleep_time}s...")
                    time_module.sleep(sleep_time)
                except Exception as e:
                    if attempt == max_retries:
                        logging.error(f"Max retries reached: {str(e)}")
                        raise
                    sleep_time = base_delay
                    logging.warning(f"Error: {str(e)}. Retrying in {sleep_time}s...")
                    time_module.sleep(sleep_time)
            
            raise RuntimeError(f"Failed after {max_retries} attempts")
        return wrapper
    return decorator

def calculate_rma(series, period):
    """Wilder's smoothing (RMA) used in ATR calculation"""
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()

@retry_with_exponential_backoff(max_retries=5, base_delay=3)
def _fetch_data_dhan(symbol, period="1y", interval="1d"):
    """Fetches stock data from Dhan API with correct symbol format."""
    dhan = get_dhan_client()
    if not dhan:
        raise ValueError("Dhan client not available.")

    security_map = load_dhan_security_id_map()
    
    # KEY CHANGE: Remove -EQ suffix for Dhan lookup
    base_symbol = get_base_symbol(symbol)
    security_id = security_map.get(base_symbol)
    
    if not security_id:
        logging.warning(f"[Dhan] Security ID not found for {base_symbol} (original: {symbol})")
        return pd.DataFrame()

    interval_map_dhan = {"1d": "D", "1h": "60", "30m": "30", "15m": "15", "5m": "5"}
    api_interval = interval_map_dhan.get(interval)
    if not api_interval:
        raise ValueError(f"Unsupported interval for Dhan: {interval}")
        
    end_date = datetime.now()
    period_map = {"2y": 730, "1y": 365, "6mo": 180, "1mo": 30, "1d": 2}
    days = period_map.get(period, 365)
    start_date = end_date - timedelta(days=days)

    try:
        if interval == "1d":
            response = dhan.historical_daily_data(
                security_id=security_id,
                exchange_segment='NSE_EQ',
                instrument_type='EQUITY',
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d')
            )
        else:
            logging.info(f"Fetching Dhan intraday data for {base_symbol}")
            response = dhan.intraday_data(
                security_id=security_id,
                exchange_segment='NSE_EQ',
                instrument_type='EQUITY',
                interval=api_interval
            )
        
        if not response or response.get('status') != 'success' or 'data' not in response:
            error_msg = response.get('remarks', 'Unknown error') if response else 'No response'
            logging.warning(f"[Dhan] API error for {base_symbol}: {error_msg}")
            return pd.DataFrame()
        
        data_list = response['data']
        if not data_list:
            logging.warning(f"[Dhan] No data returned for {base_symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data_list)
        
        # Log the actual columns we received
        logging.info(f"[Dhan] Columns received for {base_symbol}: {df.columns.tolist()}")
        
        # Flexible column mapping - try multiple possible column names
        column_mapping = {}
        
        # Date/Time column
        for date_col in ['start_time', 'timestamp', 'date', 'time', 'datetime']:
            if date_col in df.columns:
                column_mapping[date_col] = 'Date'
                break
        
        # OHLCV columns
        for open_col in ['open', 'Open', 'OPEN']:
            if open_col in df.columns:
                column_mapping[open_col] = 'Open'
                break
        
        for high_col in ['high', 'High', 'HIGH']:
            if high_col in df.columns:
                column_mapping[high_col] = 'High'
                break
        
        for low_col in ['low', 'Low', 'LOW']:
            if low_col in df.columns:
                column_mapping[low_col] = 'Low'
                break
        
        for close_col in ['close', 'Close', 'CLOSE']:
            if close_col in df.columns:
                column_mapping[close_col] = 'Close'
                break
        
        for vol_col in ['volume', 'Volume', 'VOLUME', 'vol']:
            if vol_col in df.columns:
                column_mapping[vol_col] = 'Volume'
                break
        
        if 'Date' not in column_mapping.values():
            logging.error(f"[Dhan] No date/time column found for {base_symbol}. Columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Rename columns
        df.rename(columns=column_mapping, inplace=True)
        
        # Ensure we have Date column
        if 'Date' not in df.columns:
            logging.error(f"[Dhan] Date column missing after rename for {base_symbol}")
            return pd.DataFrame()
        
        # Convert Date and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Convert numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logging.info(f"[Dhan] Successfully fetched {len(df)} rows for {base_symbol}")
        return df
        
    except Exception as e:
        logging.error(f"[Dhan] Error fetching {base_symbol}: {str(e)}", exc_info=True)
        return pd.DataFrame()
        
@retry_with_exponential_backoff(max_retries=5, base_delay=3)
def _fetch_data_smartapi(symbol, period="1y", interval="1d"):
    """Fetches stock data from SmartAPI with proper rate limiting and date ranges."""
    if "-EQ" not in symbol:
        symbol = f"{symbol.split('.')[0]}-EQ"
    
    smart_api = get_global_smart_api()
    if not smart_api:
        raise ValueError("SmartAPI session unavailable")
    
    # Use proper current date
    end_date = datetime.now().replace(microsecond=0)
    
    interval_map = {
        "1d": "ONE_DAY", 
        "1h": "ONE_HOUR", 
        "30m": "THIRTY_MINUTE", 
        "15m": "FIFTEEN_MINUTE", 
        "5m": "FIVE_MINUTE"
    }
    api_interval = interval_map.get(interval, "ONE_DAY")
    
    # SmartAPI Official Max Days per Interval (from documentation)
    max_days_map = {
        "FIVE_MINUTE": 100,      # 5 min
        "FIFTEEN_MINUTE": 200,   # 15 min
        "THIRTY_MINUTE": 200,    # 30 min
        "ONE_HOUR": 400,         # 1 hour
        "ONE_DAY": 2000          # 1 day
    }
    
    # Get max allowed days for this interval
    max_allowed_days = max_days_map.get(api_interval, 365)
    
    # Calculate days requested
    period_map = {"2y": 730, "1y": 365, "6mo": 180, "1mo": 30, "1d": 1}
    requested_days = period_map.get(period, 365)
    
    # Enforce SmartAPI limits
    days = min(requested_days, max_allowed_days)
    start_date = (end_date - timedelta(days=days)).replace(microsecond=0)
    
    symbol_token_map = load_symbol_token_map()
    symboltoken = symbol_token_map.get(symbol)
    if not symboltoken:
        logging.warning(f"[SmartAPI] Token not found for {symbol}")
        return pd.DataFrame()
    
    # Log request details for debugging
    request_params = {
        "exchange": "NSE", 
        "symboltoken": symboltoken, 
        "interval": api_interval,
        "fromdate": start_date.strftime("%Y-%m-%d %H:%M"), 
        "todate": end_date.strftime("%Y-%m-%d %H:%M")
    }
    
    logging.info(f"[SmartAPI] Requesting {symbol}: {api_interval} from {request_params['fromdate']} to {request_params['todate']}")
    
    # Apply rate limiting before making the API call
    _smartapi_rate_limiter.wait_if_needed()
    
    try:
        historical_data = smart_api.getCandleData(request_params)
    except Exception as e:
        logging.error(f"[SmartAPI] Exception for {symbol}: {str(e)}")
        return pd.DataFrame()
    
    if not historical_data or not historical_data.get('status') or not historical_data.get('data'):
        error_msg = historical_data.get('message', 'No data') if historical_data else 'No response'
        error_code = historical_data.get('errorcode', 'N/A') if historical_data else 'N/A'
        
        # Handle specific error codes
        if error_code == 'AB1004':
            logging.warning(f"[SmartAPI] Rate limit or server error (AB1004) for {symbol} - will retry")
            raise Exception(f"SmartAPI rate limit/server error for {symbol}")
        
        logging.warning(f"[SmartAPI] API error for {symbol}: {error_msg} (Code: {error_code})")
        return pd.DataFrame()
    
    # Create DataFrame
    data = pd.DataFrame(
        historical_data['data'], 
        columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    
    # CRITICAL FIX: Convert Date BEFORE setting as index
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Set index (this removes 'Date' as a column)
    data.set_index('Date', inplace=True)
    
    # VERIFY: Date should now only be an index, not a column
    if 'Date' in data.columns:
        logging.warning(f"[SmartAPI] Duplicate 'Date' column detected, dropping...")
        data = data.drop(columns=['Date'])
    
    return data
    
def fetch_stock_data_cached(symbol, period="1y", interval="1d", api_provider="SmartAPI"):
    """Wrapper for stock data fetching with provider-aware symbol handling."""
    # Normalize symbol for the selected provider
    normalized_symbol = normalize_symbol(symbol, api_provider)
    cache_key = f"{api_provider}_{normalized_symbol}_{period}_{interval}"
    
    try:
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return pd.read_pickle(io.BytesIO(cached_data))
    except Exception as e:
        logging.warning(f"Cache read error for {normalized_symbol}: {str(e)}")
    
    try:
        if api_provider == "Dhan":
            data = _fetch_data_dhan(normalized_symbol, period, interval)
        else:
            data = _fetch_data_smartapi(normalized_symbol, period, interval)
            
        if data.empty:
            return data

        try:
            expire = 86400 if interval == "1d" else 300
            buffer = io.BytesIO()
            data.to_pickle(buffer)
            cache.set(cache_key, buffer.getvalue(), expire=expire)
        except Exception as e:
            logging.warning(f"Cache write error for {normalized_symbol}: {str(e)}")
        
        return data
        
    except Exception as e:
        logging.error(f"Error fetching {normalized_symbol} using {api_provider}: {str(e)}")
        return pd.DataFrame()

def check_api_health(api_provider="SmartAPI"):
    """Verify API session is working for the selected provider."""
    if api_provider == "Dhan":
        try:
            dhan = get_dhan_client()
            if not dhan: return False, "Dhan client not initialized"
            fund_limits = dhan.get_fund_limits()
            if fund_limits and fund_limits.get('status') == 'success':
                return True, "API healthy"
            else:
                return False, f"API check failed: {fund_limits.get('remarks', 'Unknown error')}"
        except Exception as e:
            return False, str(e)
    else: # SmartAPI
        try:
            # This will create session if it doesn't exist
            smart_api = get_global_smart_api()
            
            if not smart_api: 
                return False, "Failed to initialize session"
            
            # SmartAPI uses different attribute names - check multiple possibilities
            # Check for session token (various possible names)
            has_token = (
                hasattr(smart_api, 'auth_token') and smart_api.auth_token or
                hasattr(smart_api, 'access_token') and smart_api.access_token or
                hasattr(smart_api, 'jwtToken') and smart_api.jwtToken or
                hasattr(smart_api, 'sessionToken') and smart_api.sessionToken
            )
            
            # Check for user ID
            has_user_id = (
                hasattr(smart_api, 'userId') and smart_api.userId or
                hasattr(smart_api, 'client_code') and smart_api.client_code or
                hasattr(smart_api, 'clientcode') and smart_api.clientcode
            )
            
            if has_token or has_user_id:
                return True, "Session active"
            
            # Try making an actual API call
            try:
                rms = smart_api.rmsLimit()
                if rms and rms.get('status'):
                    return True, "API healthy"
            except:
                pass
            
            # If we got here, session object exists (it was created successfully)
            # The log shows "SmartAPI session created successfully"
            # So even without finding the token attribute, we know it works
            return True, "Session created"
                
        except Exception as e:
            return False, str(e)
            
# ============================================================================
# DATA VALIDATION & INDICATOR CALCULATION (UNCHANGED)
# ============================================================================
def validate_data(data, required_columns=None, min_length=50):
    if required_columns is None:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if data is None or data.empty or len(data) < min_length: return False
    if [c for c in required_columns if c not in data.columns]: return False
    if data[required_columns].isnull().any().any(): return False
    price_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in data.columns]
    if (data[price_cols] <= 0).any().any(): return False
    return True

def calculate_swing_indicators(data, debug_mode=False):
    """Calculate swing trading indicators matching TradingView defaults with DEBUG"""
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
    
    # 200 EMA - CRITICAL CALCULATION
    df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
    df['Above_EMA200'] = df['Close'] > df['EMA_200']
    
    # 🔍 DEBUG OUTPUT - Always log for swing analysis
    if len(df) >= 200:
        latest_close = df['Close'].iloc[-1]
        latest_ema = df['EMA_200'].iloc[-1]
        
        if pd.notna(latest_ema):
            above_ema = latest_close > latest_ema
            diff_pct = ((latest_close - latest_ema) / latest_ema) * 100
            
            # Always log this critical info
            logging.info(f"{'='*60}")
            logging.info(f"📊 EMA 200 Analysis:")
            logging.info(f"   Symbol: {df.get('Symbol', 'Unknown')}")
            logging.info(f"   Latest Close: ₹{latest_close:.2f}")
            logging.info(f"   EMA 200: ₹{latest_ema:.2f}")
            logging.info(f"   Position: {'ABOVE ✅' if above_ema else 'BELOW ⬇️'} EMA 200")
            logging.info(f"   Distance: {diff_pct:+.2f}%")
            logging.info(f"   Total bars: {len(df)}")
            
            # Extended debug
            if debug_mode:
                logging.info(f"\n   📈 Last 10 Days:")
                for i in range(-10, 0):
                    close = df['Close'].iloc[i]
                    ema = df['EMA_200'].iloc[i]
                    position = "↑" if close > ema else "↓"
                    logging.info(f"     {df.index[i].strftime('%Y-%m-%d')}: Close=₹{close:.2f}, EMA=₹{ema:.2f} {position}")
                
                # Data quality check
                null_count = df['Close'].isnull().sum()
                logging.info(f"\n   🔍 Data Quality:")
                logging.info(f"     Null values in Close: {null_count}")
                logging.info(f"     Date range: {df.index[0]} to {df.index[-1]}")
                logging.info(f"     First valid EMA: {df['EMA_200'].first_valid_index()}")
            
            logging.info(f"{'='*60}\n")
        else:
            logging.error(f"❌ EMA 200 is NaN for symbol! Data length: {len(df)}")
    
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

def calculate_swing_score(df, symbol=None, timeframe='1d', contrarian_mode=False):
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
    
    # Market context adjustments (CORRECTLY IMPLEMENTED)
    confidence_adjustment = 0
    
    if symbol:
        signal_direction = 'bullish' if score > 0 else 'bearish'
        
        # Calculate BASE adjustments (reduced from original ±13 to ±6)
        index_adj = 0
        index_trends = get_index_trend_for_timeframe(timeframe)
        if index_trends:
            relevant_index = get_relevant_index(symbol)
            index_data = index_trends.get(relevant_index)
            index_adj = calculate_index_alignment_score(index_data, signal_direction) * 0.4  # ±5 → ±2
        
        breadth_adj = calculate_market_breadth_alignment(signal_direction) * 0.4  # ±5 → ±2
        
        industry_adj = 0
        industry_data = get_industry_performance(symbol)
        if industry_data:
            industry_adj = calculate_industry_alignment_score(industry_data, signal_direction) * 0.67  # ±3 → ±2
        
        # Total base adjustment: ±6
        base_context_adjustment = index_adj + breadth_adj + industry_adj
        
        # Apply contrarian mode AFTER base reduction
        if contrarian_mode:
            confidence_adjustment = base_context_adjustment * 0.5  # ±6 → ±3
        else:
            confidence_adjustment = base_context_adjustment  # ±6
    
    # Final score
    final_score = score + confidence_adjustment
    
    # Normalization with CORRECT range mapping + Division by Zero Safety
    # Technical range: ~±13, Context: ±6 (normal) or ±3 (contrarian)
    # Total expected range: ±19 (normal) or ±16 (contrarian)
    max_expected = 19 if not contrarian_mode else 16
    
    if final_score >= 0:
        # Map [0, max_expected] → [50, 85] (conservative bullish)
        normalized = 50 + (final_score / max(max_expected, 1)) * 35
    else:
        # Map [-max_expected, 0] → [15, 50] (preserve bearish warnings)
        normalized = 50 + (final_score / max(max_expected, 1)) * 35
    
    normalized = np.clip(normalized, 0, 100)
    return round(normalized, 1)

def calculate_intraday_indicators(data, timeframe='15m'):
    """Enhanced intraday indicators with FIXED date handling"""
    if len(data) < 200:
        return data
    
    df = data.copy()
    
    # Ensure datetime index
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
    # CRITICAL FIX: Create DateOnly column without conflicting with index
    df['DateOnly'] = df.index.date  # Use different name to avoid conflict
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    df['TPV'] = df['Typical_Price'] * df['Volume']
    df['Cumul_TPV'] = df.groupby('DateOnly')['TPV'].cumsum()
    df['Cumul_Vol'] = df.groupby('DateOnly')['Volume'].cumsum()
    df['VWAP'] = df['Cumul_TPV'] / df['Cumul_Vol'].replace(0, np.nan)
    
    df['Deviation_Squared'] = (df['Typical_Price'] - df['VWAP']) ** 2
    df['Cumul_Dev_Sq'] = df.groupby('DateOnly')['Deviation_Squared'].cumsum()
    df['Bar_Count'] = df.groupby('DateOnly').cumcount() + 1
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
    
    df['OR_High'] = df[df['Is_OR']].groupby('DateOnly')['High'].transform('max')
    df['OR_Low'] = df[df['Is_OR']].groupby('DateOnly')['Low'].transform('min')
    
    df['OR_High'] = df.groupby('DateOnly')['OR_High'].transform(lambda x: x.ffill())
    df['OR_Low'] = df.groupby('DateOnly')['OR_Low'].transform(lambda x: x.ffill())
    
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

    # CLEANUP: Drop temporary columns (use DateOnly instead of Date)
    df.drop(columns=['DateOnly', 'Typical_Price', 'Time', 'Is_OR', 'TPV', 'Cumul_TPV', 
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

def calculate_opening_range_score(df):
    """Opening Range Breakout Strategy"""
    score = 0
    if not df['After_OR'].iloc[-1]: return 0
    
    or_breakout_long = df['OR_Breakout_Long'].iloc[-1]
    or_breakout_short = df['OR_Breakout_Short'].iloc[-1]
    failed_breakout = df['Failed_OR_Breakout'].iloc[-1]
    rvol = df['RVOL'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    macd_bullish = df['MACD_Bullish'].iloc[-1]
    or_range = df['OR_Range'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    
    if pd.isna(or_range) or pd.isna(atr) or or_range < (atr * 0.5): return 0
    
    if or_breakout_long:
        score += 5
        if rvol > 2.5: score += 3
        elif rvol > 1.8: score += 2
        elif rvol > 1.5: score += 1
        else: score -= 2
        if macd_bullish: score += 1
        if adx > 20: score += 2
    elif or_breakout_short:
        score -= 5
        if rvol > 2.5: score -= 3
        elif rvol > 1.8: score -= 2
        elif rvol > 1.5: score -= 1
        else: score += 2
        if not macd_bullish: score -= 1
        if adx > 20: score -= 2
    elif failed_breakout:
        if df['High'].shift(1).iloc[-1] > df['OR_High'].iloc[-1]: score -= 3
        elif df['Low'].shift(1).iloc[-1] < df['OR_Low'].iloc[-1]: score += 3
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
        if adx < 20: score += 2
        if rvol > 1.5: score += 1
    elif at_upper_extreme and rsi > 70:
        rsi_strength = (rsi - 70) / 30
        score -= 4 * rsi_strength
        if adx < 20: score -= 2
        if rvol > 1.5: score -= 1
    elif close <= vwap_lower1 and rsi < 40:
        score += 2
    elif close >= vwap_upper1 and rsi > 60:
        score -= 2
    
    if vwap_upper_breakout and rvol > 2.0 and adx > 20:
        score = max(score, 0) + 2
    elif vwap_lower_breakdown and rvol > 2.0 and adx > 20:
        score = min(score, 0) - 2
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
        if ema_crossover: score += 3
        elif ema_bullish: score += 1
        if macd_bullish: score += 1
        if macd_hist_rising: score += 1
        if rvol > 2.0: score += 2
        elif rvol > 1.5: score += 1
        if adx > 25: score += 2
        elif adx > 20: score += 1
    elif close < vwap:
        score -= 2
        if ema_crossunder: score -= 3
        elif not ema_bullish: score -= 1
        if not macd_bullish: score -= 1
        if not macd_hist_rising: score -= 1
        if rvol > 2.0: score -= 2
        elif rvol > 1.5: score -= 1
        if adx > 25: score -= 2
        elif adx > 20: score -= 1
    return score

def calculate_intraday_score(df, symbol=None, timeframe='15m', contrarian_mode=False):
    """Unified intraday scoring with BALANCED market context"""
    regime = detect_intraday_regime(df)
    if regime in ["Pre-Market", "Closing Session", "Unknown", "Last 30 Min (Exit Only)", "Opening Range Formation"]:
        return 50
    if not df['Safe_Hours'].iloc[-1]: return 50
    
    or_score = calculate_opening_range_score(df)
    mean_reversion_score = calculate_vwap_mean_reversion_score(df)
    trend_score = calculate_intraday_trend_score(df)
    
    current_time = df.index[-1].time()
    
    if time(9, 45) <= current_time <= time(11, 0):
        raw_score = or_score if or_score != 0 else trend_score * 0.5
    elif regime in ["Strong Uptrend", "Strong Downtrend"] and not df['Lunch_Hours'].iloc[-1]:
        raw_score = trend_score
    elif df['Prime_Hours'].iloc[-1]:
        raw_score = mean_reversion_score if regime in ["Choppy (VWAP Range)", "Weak Uptrend", "Weak Downtrend"] else trend_score
    else:
        raw_score = trend_score
    
    # Market context adjustments
    confidence_adjustment = 0
    if symbol:
        signal_direction = 'bullish' if raw_score > 0 else 'bearish'
        index_adj, breadth_adj, industry_adj = 0, 0, 0
        
        index_trends = get_index_trend_for_timeframe(timeframe)
        if index_trends:
            relevant_index = get_relevant_index(symbol)
            index_adj = calculate_index_alignment_score(index_trends.get(relevant_index), signal_direction) * 0.4
        
        breadth_adj = calculate_market_breadth_alignment(signal_direction) * 0.4
        industry_data = get_industry_performance(symbol)
        if industry_data:
            industry_adj = calculate_industry_alignment_score(industry_data, signal_direction) * 0.67
        
        base_context_adjustment = index_adj + breadth_adj + industry_adj
        confidence_adjustment = base_context_adjustment * 0.5 if contrarian_mode else base_context_adjustment
    
    final_score = raw_score + confidence_adjustment
    
    # Time modifiers
    if df['Prime_Hours'].iloc[-1]: final_score *= 1.2
    elif df['Lunch_Hours'].iloc[-1]: final_score *= 0.7
    
    # Normalization
    max_expected = 23 if not contrarian_mode else 20
    if final_score >= 0:
        normalized = 50 + (final_score / max(max_expected, 1)) * 40
    else:
        normalized = 50 + (final_score / max(max_expected, 1)) * 40
    
    return round(np.clip(normalized, 0, 100), 1)

# ============================================================================
# POSITION SIZING & RISK MANAGEMENT (UNCHANGED)
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
    max_acceptable_stop = close * (1 - 0.08)
    
    if regime in ["Strong Uptrend", "Weak Uptrend"]:
        stop_loss = max(close - (2 * atr), ema200 * 0.98)
    elif "Consolidation" in regime:
        stop_loss = bb_lower * 0.98
    else:
        stop_loss = close - (1.5 * atr)
    
    stop_loss = round(max(stop_loss, max_acceptable_stop), 2)
    risk = buy_at - stop_loss
    rr_ratio = 3 if regime == "Strong Uptrend" else 2 if "Uptrend" in regime or "Consolidation (Above EMA)" in regime else 1.5
    target = round(buy_at + (risk * rr_ratio), 2)
    
    risk_amount = account_size * risk_pct
    position_size = int(risk_amount / risk) if risk > 0 else 0
    max_position = int((account_size * 0.1) / buy_at)
    position_size = min(position_size, max_position)
    
    return {
        "current_price": round(close, 2), "buy_at": buy_at, "stop_loss": stop_loss, "target": target,
        "position_size": position_size, "rr_ratio": round((target - buy_at) / risk, 2) if risk > 0 else 0,
        "risk_amount": round(position_size * risk, 2), "potential_profit": round(position_size * (target - buy_at), 2)
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
    max_acceptable_stop = close * (1 - 0.03)
    
    if regime == "Strong Uptrend":
        or_stop = or_low - (0.2 * atr) if pd.notna(or_low) else vwap - (0.3 * atr)
        stop_loss = max(vwap - (0.3 * atr), or_stop, close - (1.5 * atr))
    elif "Weak Uptrend" in regime or "Choppy" in regime:
        stop_loss = max(close - (1.0 * atr), vwap_lower1 - (0.2 * atr))
    elif pd.notna(or_low) and df['After_OR'].iloc[-1]:
        stop_loss = or_low - (0.5 * atr)
    else:
        stop_loss = close - (1.5 * atr)
    
    stop_loss = round(max(stop_loss, max_acceptable_stop), 2)
    risk = buy_at - stop_loss
    
    if regime == "Strong Uptrend":
        target = buy_at + (risk * 2.0)
    elif pd.notna(or_high) and df['OR_Breakout_Long'].iloc[-1]:
        target = or_high + (or_high - or_low)
    else:
        target = buy_at + risk
    
    target = round(target, 2)
    
    risk_amount = account_size * risk_pct
    position_size = int(risk_amount / risk) if risk > 0 else 0
    max_position = int((account_size * 0.05) / buy_at)
    position_size = min(position_size, max_position)
    
    current_time = df.index[-1].time()
    minutes_to_close = (15 * 60 + 15) - (current_time.hour * 60 + current_time.minute)
    
    return {
        "current_price": round(close, 2), "buy_at": buy_at, "stop_loss": stop_loss, "target": target,
        "position_size": position_size, "rr_ratio": round((target - buy_at) / risk, 2) if risk > 0 else 0,
        "risk_amount": round(position_size * risk, 2), "potential_profit": round(position_size * (target - buy_at), 2),
        "vwap": round(vwap, 2), "or_high": round(or_high, 2) if pd.notna(or_high) else None,
        "or_low": round(or_low, 2) if pd.notna(or_low) else None, "or_mid": round(or_mid, 2) if pd.notna(or_mid) else None,
        "hours_to_close": round(minutes_to_close / 60, 2)
    }

# ============================================================================
# UNIFIED RECOMMENDATION (UNCHANGED)
# ============================================================================

def generate_recommendation(data, symbol, trading_style='swing', timeframe='1d', account_size=30000, contrarian_mode=False):
    """Generate unified recommendations with COMPLETE MARKET CONTEXT"""
    if trading_style == 'swing':
        df = calculate_swing_indicators(data)
        regime = detect_swing_regime(df)
        score = calculate_swing_score(df, symbol, timeframe, contrarian_mode)
        position = calculate_swing_position(df, account_size)
    else:
        df = calculate_intraday_indicators(data, timeframe)
        regime = detect_intraday_regime(df)
        score = calculate_intraday_score(df, symbol, timeframe, contrarian_mode)
        position = calculate_intraday_position(df, account_size)
    
    market_health, market_signal, market_factors = calculate_market_health_score()
    index_trends = get_index_trend_for_timeframe(timeframe)
    index_context, industry_context = None, None
    
    if index_trends:
        relevant_index = get_relevant_index(symbol)
        index_data = index_trends.get(relevant_index)
        if index_data and 'analysis' in index_data:
            index_context = {'index_name': 'Nifty' if relevant_index == 'nifty' else 'Bank Nifty', **index_data}

    industry_data = get_industry_performance(symbol)
    if industry_data:
        industry_context = industry_data

    if score >= 75: signal = "Strong Buy"
    elif score >= 60: signal = "Buy"
    elif score <= 25: signal = "Strong Sell"
    elif score <= 40: signal = "Sell"
    else: signal = "Hold"
    
    # Build reason
    reasons = []
    close = df['Close'].iloc[-1]
    if trading_style == 'swing':
        reasons.append("Above 200 EMA" if close > df['EMA_200'].iloc[-1] else "Below 200 EMA")
        reasons.append("MACD bullish" if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "MACD bearish")
        if df['RSI'].iloc[-1] < df['RSI_Oversold'].iloc[-1]: reasons.append("RSI oversold")
        elif df['RSI'].iloc[-1] > df['RSI_Overbought'].iloc[-1]: reasons.append("RSI overbought")
    else:
        reasons.append("Above VWAP" if close > df['VWAP'].iloc[-1] else "Below VWAP")
        reasons.append("EMA bullish" if df['EMA_Bullish'].iloc[-1] else "EMA bearish")
        if df['OR_Breakout_Long'].iloc[-1]: reasons.append("OR breakout")
    
    reasons.append(f"Market: {market_signal}")
    
    return {
        "symbol": symbol, "trading_style": trading_style.capitalize(), "timeframe": timeframe,
        "score": score, "signal": signal, "regime": regime, "reason": ", ".join(reasons),
        "index_context": index_context, "industry_context": industry_context,
        "market_health": market_health, "market_signal": market_signal, "market_factors": market_factors,
        "contrarian_mode": contrarian_mode, "processed_data": df, **position
    }

# ============================================================================
# ============================================================================
# BATCH ANALYSIS
# ============================================================================

def analyze_stock_batch(symbol, trading_style='swing', timeframe='1d', contrarian_mode=False, max_retries=3, api_provider="SmartAPI"):
    """Analyze single stock with comprehensive error handling"""
    for attempt in range(max_retries):
        try:
            data = fetch_stock_data_cached(symbol, interval=timeframe, api_provider=api_provider)
            if data.empty or len(data) < 200: return None
            rec = generate_recommendation(data, symbol, trading_style, timeframe, contrarian_mode=contrarian_mode)
            return {
                "Symbol": rec['symbol'], "Score": rec['score'], "Signal": rec['signal'], "Regime": rec['regime'],
                "Current Price": rec['current_price'], "Buy At": rec['buy_at'], "Stop Loss": rec['stop_loss'],
                "Target": rec['target'], "R:R": rec['rr_ratio'], "Position Size": rec['position_size'], "Reason": rec['reason']
            }
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"{symbol}: Attempt {attempt + 1} failed: {e}. Retrying...")
                time_module.sleep(2 * (attempt + 1))
            else:
                logging.error(f"{symbol}: All attempts failed: {e}")
                return None
    return None

def analyze_multiple_stocks(stock_list, trading_style='swing', timeframe='1d', progress_callback=None, resume=False, contrarian_mode=False, api_provider="SmartAPI"):
    """Analyze multiple stocks with enhanced resume capability and robust error handling."""
    global _global_smart_api, _global_dhan_client  # Declare globals at function start
    
    all_results, failed_stocks, completed_stocks = [], [], []
    stock_list = list(dict.fromkeys(stock_list))
    consecutive_failures = 0
    max_consecutive_failures = 10  # Stop if 10 consecutive failures
    
    # Load checkpoint if resuming
    if resume and (checkpoint := load_checkpoint()):
        if (checkpoint.get('trading_style') == trading_style and 
            checkpoint.get('timeframe') == timeframe and 
            checkpoint.get('api_provider') == api_provider):
            all_results = checkpoint['all_results']
            failed_stocks = checkpoint['failed_stocks']
            completed_stocks = checkpoint['completed_stocks']
            logging.info(f"✅ Resuming scan: {len(completed_stocks)} completed, {len(all_results)} successful")
        else:
            logging.warning("Checkpoint parameters don't match current scan. Starting fresh.")
            clear_checkpoint()
    
    # Calculate remaining work
    total_stocks = min(len(stock_list), SCAN_CONFIG["max_stocks_per_scan"])
    
    # CRITICAL FIX: Filter out already completed stocks
    remaining_stocks = [s for s in stock_list if s not in completed_stocks]
    remaining_stocks = remaining_stocks[:total_stocks - len(completed_stocks)]
    
    if not remaining_stocks:
        logging.info(f"All {len(completed_stocks)} stocks already processed!")
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    logging.info(f"Processing {len(remaining_stocks)} remaining stocks (out of {total_stocks} total)")
    
    # Pre-check API health before starting
    api_healthy, api_msg = check_api_health(api_provider)
    if not api_healthy:
        logging.error(f"API not healthy before starting: {api_msg}")
        # Try to reinitialize API
        if api_provider == "SmartAPI":
            global _global_smart_api
            _global_smart_api = None
        elif api_provider == "Dhan":
            global _global_dhan_client  
            _global_dhan_client = None

    try:
        for batch_idx in range(0, len(remaining_stocks), SCAN_CONFIG["batch_size"]):
            batch = remaining_stocks[batch_idx:batch_idx + SCAN_CONFIG["batch_size"]]
            
            # Enhanced session refresh and health check
            if batch_idx > 0 and batch_idx % SCAN_CONFIG["session_refresh_interval"] == 0:
                logging.info(f"🔄 Refreshing {api_provider} session after {batch_idx} stocks...")
                
                if api_provider == "SmartAPI":
                    _global_smart_api = None
                    time_module.sleep(3)
                elif api_provider == "Dhan":
                    _global_dhan_client = None
                    time_module.sleep(2)
                
                # Verify API health after refresh
                api_healthy, api_msg = check_api_health(api_provider)
                if not api_healthy:
                    logging.error(f"API health check failed after refresh: {api_msg}")
                    # Force save checkpoint before potentially failing
                    save_checkpoint(completed_stocks, all_results, failed_stocks, trading_style, timeframe, api_provider)
                    raise RuntimeError(f"API health check failed: {api_msg}")

            for i, symbol in enumerate(batch):
                try:
                    # Calculate progress based on total stocks
                    current_count = len(completed_stocks) + 1
                    progress_pct = min(current_count / total_stocks, 1.0)
                    
                    if progress_callback:
                        progress_callback(progress_pct)
                    
                    logging.info(f"📊 Processing {current_count}/{total_stocks}: {symbol}")
                    
                    # Check for consecutive failures
                    if consecutive_failures >= SCAN_CONFIG["max_consecutive_failures"]:
                        logging.error(f"🛑 Stopping scan due to {consecutive_failures} consecutive failures")
                        save_checkpoint(completed_stocks, all_results, failed_stocks, trading_style, timeframe, api_provider)
                        raise RuntimeError(f"Too many consecutive failures ({consecutive_failures})")
                    
                    # Analyze stock with enhanced error handling
                    result = analyze_stock_batch(symbol, trading_style, timeframe, contrarian_mode, api_provider=api_provider)
                    
                    if result:
                        result['Sector'] = assign_primary_sector(symbol, SECTORS) 
                        all_results.append(result)
                        consecutive_failures = 0  # Reset failure counter on success
                        logging.info(f"✅ {symbol}: Score {result['Score']}")
                    else:
                        failed_stocks.append(symbol)
                        consecutive_failures += 1
                        logging.warning(f"❌ {symbol}: Failed to analyze (consecutive failures: {consecutive_failures})")
                    
                    completed_stocks.append(symbol)
                    
                    # Dynamic checkpoint saving based on scan size
                    checkpoint_interval = SCAN_CONFIG["checkpoint_interval_large"] if len(remaining_stocks) > 100 else SCAN_CONFIG["checkpoint_interval_small"]
                    if len(completed_stocks) % checkpoint_interval == 0:
                        save_checkpoint(completed_stocks, all_results, failed_stocks, trading_style, timeframe, api_provider)
                        logging.info(f"💾 Checkpoint saved: {len(completed_stocks)}/{total_stocks} completed")
                    
                    # Memory cleanup for large scans
                    if len(completed_stocks) % SCAN_CONFIG["memory_cleanup_interval"] == 0:
                        gc.collect()
                        logging.info("🧹 Memory cleanup performed")
                    
                    # Periodic API health check for large scans
                    if len(completed_stocks) % SCAN_CONFIG["api_health_check_interval"] == 0 and len(remaining_stocks) > 50:
                        api_healthy, api_msg = check_api_health(api_provider)
                        if not api_healthy:
                            logging.warning(f"⚠️ API health degraded at stock {len(completed_stocks)}: {api_msg}")
                            # Try to recover by refreshing session
                            if api_provider == "SmartAPI":
                                _global_smart_api = None
                            elif api_provider == "Dhan":
                                _global_dhan_client = None
                            time_module.sleep(2)
                    
                    # Adaptive delay based on consecutive failures
                    base_delay = SCAN_CONFIG["delay_within_batch"]
                    failure_delay = min(consecutive_failures * 0.5, 3)  # Max 3 seconds extra
                    actual_delay = base_delay + failure_delay
                    
                    if i < len(batch) - 1:
                        time_module.sleep(actual_delay)
                    else:
                        time_module.sleep(SCAN_CONFIG["delay_between_batches"] + failure_delay)
                        
                except Exception as stock_error:
                    logging.error(f"💥 Critical error processing {symbol}: {stock_error}")
                    failed_stocks.append(symbol)
                    completed_stocks.append(symbol)
                    consecutive_failures += 1
                    
                    # Save checkpoint on critical errors
                    save_checkpoint(completed_stocks, all_results, failed_stocks, trading_style, timeframe, api_provider)
                    
                    # Don't fail entire scan for individual stock errors
                    continue
        
        # Clear checkpoint on successful completion
        clear_checkpoint()
        logging.info(f"🎉 Scan complete! Processed {len(completed_stocks)} stocks, {len(all_results)} successful")
        
    except Exception as e:
        logging.error(f"💥 Fatal scan error: {e}", exc_info=True)
        save_checkpoint(completed_stocks, all_results, failed_stocks, trading_style, timeframe, api_provider)
        raise

    if not all_results:
        logging.warning("No results found matching criteria")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    # Filter based on trading style
    if trading_style == 'swing':
        df = df[df['Score'] >= 60]
    else:
        df = df[df['Signal'].str.contains('Buy', na=False)]
    
    if df.empty:
        logging.warning("No stocks passed the filter criteria")
        return df
    
    # Diversify results across sectors (max 2 per sector)
    diverse_results = []
    for sector in df['Sector'].unique():
        sector_df = df[df['Sector'] == sector].nlargest(2, 'Score')
        diverse_results.append(sector_df)
    
    if diverse_results:
        final_df = pd.concat(diverse_results).sort_values('Score', ascending=False).head(10)
    else:
        final_df = df.sort_values('Score', ascending=False).head(10)
    
    return final_df

# ============================================================================
# DATABASE & UI (UNCHANGED)
# ============================================================================

def init_database():
    conn = sqlite3.connect('stock_picks.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS picks (date TEXT, symbol TEXT, trading_style TEXT, score REAL, signal TEXT, regime TEXT, current_price REAL, buy_at REAL, stop_loss REAL, target REAL, reason TEXT, PRIMARY KEY (date, symbol, trading_style))''')
    conn.close()

def save_picks(results_df, trading_style):
    conn = sqlite3.connect('stock_picks.db')
    records = [tuple(row) for _, row in results_df.head(5).assign(date=datetime.now().strftime('%Y-%m-%d'), trading_style=trading_style)[['date', 'Symbol', 'trading_style', 'Score', 'Signal', 'Regime', 'Current Price', 'Buy At', 'Stop Loss', 'Target', 'Reason']].iterrows()]
    conn.executemany('INSERT OR REPLACE INTO picks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', records)
    conn.commit()
    conn.close()

# ============================================================================
# PAPER TRADING FUNCTIONS
# ============================================================================

def calculate_brokerage_charges(trade_value, action, trading_style, exchange='NSE'):
    """
    Calculate realistic brokerage and charges for paper trading
    Based on typical Indian discount broker charges (like Zerodha, Angel One)
    
    Returns: dict with total_charges and breakup
    """
    charges = {
        'total_charges': 0,
        'trade_value': trade_value,
        'breakup': []
    }
    
    # 1. Brokerage (₹20 per order or 0.03% for intraday, ₹0 for delivery in most discount brokers)
    if trading_style.lower() == 'intraday':
        brokerage = min(20, trade_value * 0.0003)  # ₹20 or 0.03%, whichever is lower
    else:  # delivery/swing
        brokerage = 0  # Most discount brokers offer ₹0 brokerage for delivery
    
    charges['breakup'].append({
        'name': 'Brokerage',
        'amount': brokerage,
        'percentage': (brokerage / trade_value * 100) if trade_value > 0 else 0
    })
    
    # 2. STT (Securities Transaction Tax)
    # Delivery BUY/SELL: 0.1% on both sides
    # Intraday SELL only: 0.025%
    if trading_style.lower() == 'intraday':
        stt = trade_value * 0.00025 if action == 'SELL' else 0
    else:  # delivery
        stt = trade_value * 0.001  # 0.1% on both buy and sell
    
    charges['breakup'].append({
        'name': 'STT (Securities Transaction Tax)',
        'amount': stt,
        'percentage': (stt / trade_value * 100) if trade_value > 0 else 0
    })
    
    # 3. Exchange Transaction Charges
    # NSE: 0.00325% for equity
    exchange_charges = trade_value * 0.0000325
    charges['breakup'].append({
        'name': 'Exchange Transaction Charges',
        'amount': exchange_charges,
        'percentage': 0.00325
    })
    
    # 4. SEBI Charges (₹10 per crore)
    sebi_charges = (trade_value / 10000000) * 10
    charges['breakup'].append({
        'name': 'SEBI Charges',
        'amount': sebi_charges,
        'percentage': 0.0001
    })
    
    # 5. Stamp Duty
    # 0.015% on buy side only (delivery and intraday)
    stamp_duty = trade_value * 0.00015 if action == 'BUY' else 0
    charges['breakup'].append({
        'name': 'Stamp Duty',
        'amount': stamp_duty,
        'percentage': 0.015 if action == 'BUY' else 0
    })
    
    # 6. GST (18% on brokerage + transaction charges)
    taxable_amount = brokerage + exchange_charges + sebi_charges
    gst = taxable_amount * 0.18
    charges['breakup'].append({
        'name': 'GST (18%)',
        'amount': gst,
        'percentage': (gst / trade_value * 100) if trade_value > 0 else 0
    })
    
    # Calculate total
    charges['total_charges'] = sum(item['amount'] for item in charges['breakup'])
    charges['total_percentage'] = (charges['total_charges'] / trade_value * 100) if trade_value > 0 else 0
    
    return charges

def get_paper_account(user_id='default'):
    """Get paper trading account details"""
    if not supabase:
        return None
    
    try:
        response = supabase.table('paper_account').select('*').eq('user_id', user_id).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logging.error(f"Error fetching paper account: {e}")
        return None

def get_paper_portfolio(user_id='default'):
    """Get current paper trading portfolio"""
    if not supabase:
        return []
    
    try:
        response = supabase.table('paper_portfolio').select('*').eq('user_id', user_id).execute()
        portfolio = response.data if response.data else []
        logging.info(f"Fetched {len(portfolio)} positions from portfolio for user {user_id}")
        for pos in portfolio:
            logging.info(f"  Position: {pos.get('symbol')} - Qty: {pos.get('quantity')} - Style: {pos.get('trading_style')}")
        return portfolio
    except Exception as e:
        logging.error(f"Error fetching portfolio: {e}")
        return []

def get_paper_trades_history(user_id='default', limit=50):
    """Get paper trading history"""
    if not supabase:
        return []
    
    try:
        response = supabase.table('paper_trades').select('*').eq('user_id', user_id).order('timestamp', desc=True).limit(limit).execute()
        return response.data if response.data else []
    except Exception as e:
        logging.error(f"Error fetching trades history: {e}")
        return []

def execute_paper_trade(symbol, action, quantity, price, trading_style, notes='', user_id='default'):
    """Execute a paper trade (BUY or SELL)"""
    if not supabase:
        return False, "Supabase not configured"
    
    try:
        trade_value = quantity * price
        
        # Calculate brokerage and charges
        charges = calculate_brokerage_charges(trade_value, action, trading_style)
        total_charges = charges['total_charges']
        total_amount = trade_value + total_charges  # Total cost including charges
        
        # Get current account
        account = get_paper_account(user_id)
        if not account:
            return False, "Account not found"
        
        if action == 'BUY':
            # Check if enough cash (including charges)
            if account['cash_balance'] < total_amount:
                return False, f"Insufficient funds. Need ₹{total_amount:.2f} (₹{trade_value:.2f} + ₹{total_charges:.2f} charges), have ₹{account['cash_balance']:.2f}"
            
            # Update or insert portfolio
            existing = supabase.table('paper_portfolio').select('*').eq('user_id', user_id).eq('symbol', symbol).eq('trading_style', trading_style).execute()
            
            if existing.data:
                # Update existing position
                old_qty = existing.data[0]['quantity']
                old_invested = existing.data[0]['invested_amount']
                old_avg_price = existing.data[0]['avg_price']
                
                new_qty = old_qty + quantity
                new_invested = old_invested + total_amount
                
                # Calculate new average price based on stock prices only (not including charges)
                # old_cost = old_qty × old_avg_price (pure stock cost)
                # new_cost = quantity × price (new pure stock cost)
                # new_avg = (old_cost + new_cost) / new_qty
                old_stock_cost = old_qty * old_avg_price
                new_stock_cost = quantity * price
                new_avg_price = (old_stock_cost + new_stock_cost) / new_qty
                
                supabase.table('paper_portfolio').update({
                    'quantity': new_qty,
                    'invested_amount': new_invested,  # Total with charges
                    'avg_price': new_avg_price  # Average stock price without charges
                }).eq('user_id', user_id).eq('symbol', symbol).eq('trading_style', trading_style).execute()
            else:
                # Create new position
                # avg_price = pure stock price (no charges)
                # invested_amount = total paid including charges
                # This way P&L calculation works: (current_price × qty) - invested_amount
                insert_result = supabase.table('paper_portfolio').insert({
                    'user_id': user_id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_price': price,  # Store actual stock price, not including charges
                    'invested_amount': total_amount,  # Total paid including charges
                    'trading_style': trading_style,
                    'notes': notes
                }).execute()
                
                # Verify insertion
                if not insert_result.data:
                    logging.error(f"Failed to create position for {symbol}")
                    return False, f"❌ Failed to create position in database"
            
            # Deduct cash
            new_balance = account['cash_balance'] - total_amount
            supabase.table('paper_account').update({'cash_balance': new_balance}).eq('user_id', user_id).execute()
            
            # Record trade (with charges)
            trade_result = supabase.table('paper_trades').insert({
                'user_id': user_id,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'total_amount': total_amount,
                'charges': total_charges,
                'trading_style': trading_style,
                'notes': notes
            }).execute()
            
            return True, f"✅ Bought {quantity} shares of {symbol} at ₹{price:.2f} (+ ₹{total_charges:.2f} charges)"
        
        elif action == 'SELL':
            # Check if position exists
            existing = supabase.table('paper_portfolio').select('*').eq('user_id', user_id).eq('symbol', symbol).eq('trading_style', trading_style).execute()
            
            if not existing.data:
                return False, "No position found to sell"
            
            position = existing.data[0]
            if position['quantity'] < quantity:
                return False, f"Insufficient shares. Have {position['quantity']}, trying to sell {quantity}"
            
            # Calculate P&L (net of charges)
            avg_buy_price = position['avg_price']
            sell_proceeds = trade_value - total_charges  # Deduct charges from proceeds
            cost_basis = avg_buy_price * quantity
            pnl = sell_proceeds - cost_basis
            pnl_percent = (pnl / cost_basis) * 100
            
            # Update or remove position
            new_qty = position['quantity'] - quantity
            if new_qty == 0:
                # Close position
                supabase.table('paper_portfolio').delete().eq('user_id', user_id).eq('symbol', symbol).eq('trading_style', trading_style).execute()
            else:
                # Reduce position
                new_invested = position['invested_amount'] * (new_qty / position['quantity'])
                supabase.table('paper_portfolio').update({
                    'quantity': new_qty,
                    'invested_amount': new_invested
                }).eq('user_id', user_id).eq('symbol', symbol).eq('trading_style', trading_style).execute()
            
            # Add cash (proceeds after charges)
            new_balance = account['cash_balance'] + sell_proceeds
            supabase.table('paper_account').update({'cash_balance': new_balance}).eq('user_id', user_id).execute()
            
            # Record trade with charges
            supabase.table('paper_trades').insert({
                'user_id': user_id,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'total_amount': sell_proceeds,  # Net proceeds
                'charges': total_charges,
                'trading_style': trading_style,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'notes': notes
            }).execute()
            
            pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            return True, f"✅ Sold {quantity} shares of {symbol} at ₹{price:.2f}\n{pnl_emoji} P&L: ₹{pnl:.2f} ({pnl_percent:+.2f}%)\n💰 Net proceeds: ₹{sell_proceeds:.2f} (after ₹{total_charges:.2f} charges)"
        
        return False, "Invalid action"
        
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return False, f"Error: {str(e)}"

def reset_paper_account(user_id='default', initial_balance=100000, clear_trades=False):
    """Reset paper trading account"""
    if not supabase:
        return False, "Supabase not configured"
    
    try:
        # Clear portfolio
        supabase.table('paper_portfolio').delete().eq('user_id', user_id).execute()
        
        # Reset account
        supabase.table('paper_account').update({
            'cash_balance': initial_balance,
            'initial_balance': initial_balance
        }).eq('user_id', user_id).execute()
        
        # Optionally clear trade history
        if clear_trades:
            supabase.table('paper_trades').delete().eq('user_id', user_id).execute()
            return True, f"✅ Account and trade history reset to ₹{initial_balance:,.2f}"
        
        return True, f"✅ Account reset to ₹{initial_balance:,.2f}"
    except Exception as e:
        logging.error(f"Error resetting account: {e}")
        return False, f"Error: {str(e)}"

def display_tradingview_chart(symbol, timeframe='D', height=600):
    """
    Display TradingView mini chart widget (more reliable for NSE stocks)
    
    Args:
        symbol: Stock symbol (e.g., 'SBIN-EQ')
        timeframe: Chart timeframe (D=Daily, 60=1hour, 15=15min, 5=5min)
        height: Chart height in pixels
    """
    # Convert symbol format to TradingView format
    # Remove -EQ suffix if present and try different formats
    clean_symbol = symbol.replace('-EQ', '').replace('-eq', '').strip()
    
    # Try multiple symbol formats for better compatibility
    symbol_formats = [
        f"NSE:{clean_symbol}",           # NSE:SBIN
        f"BSE:{clean_symbol}",           # BSE:SBIN (fallback)
    ]
    
    # Map timeframe to TradingView format
    timeframe_map = {
        '1d': 'D',
        '5m': '5',
        '15m': '15',
        '30m': '30',
        '1h': '60'
    }
    tv_timeframe = timeframe_map.get(timeframe, 'D')
    
    # Use TradingView's simpler mini widget which is more reliable
    tradingview_html = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="height:{height}px;width:100%">
      <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {{
        "autosize": true,
        "symbol": "{symbol_formats[0]}",
        "interval": "{tv_timeframe}",
        "timezone": "Asia/Kolkata",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "allow_symbol_change": true,
        "calendar": false,
        "support_host": "https://www.tradingview.com",
        "studies": [
          "STD;VWAP",
          "STD;MACD",
          "STD;RSI"
        ]
      }}
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    return tradingview_html

def display_enhanced_chart(rec, data):
    """Display enhanced trading chart with detailed levels and annotations"""
    from plotly.subplots import make_subplots
    
    # Create subplots: candlestick + volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{rec['symbol']} - {rec['timeframe']}", "Volume")
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # Add EMAs if available
    if 'EMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['EMA_20'],
            mode='lines',
            name='EMA 20',
            line=dict(color='orange', width=1.5)
        ), row=1, col=1)
    
    if 'EMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['EMA_50'],
            mode='lines',
            name='EMA 50',
            line=dict(color='cyan', width=1.5)
        ), row=1, col=1)
    
    # Add VWAP if available
    if 'VWAP' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['VWAP'],
            mode='lines',
            name='VWAP',
            line=dict(color='yellow', width=2, dash='dash')
        ), row=1, col=1)
    
    # Calculate percentages for annotations
    entry_price = rec['buy_at']
    stop_price = rec['stop_loss']
    target_price = rec['target']
    
    stop_pct = abs((stop_price - entry_price) / entry_price * 100)
    target_pct = abs((target_price - entry_price) / entry_price * 100)
    risk_reward = target_pct / stop_pct if stop_pct > 0 else 0
    
    # Add Opening Range levels if available
    if rec.get('or_high'):
        fig.add_hline(
            y=rec['or_high'],
            line_dash="dot",
            annotation_text=f"OR High: ₹{rec['or_high']:.2f}",
            line_color="lightgreen",
            line_width=1,
            row=1, col=1
        )
        fig.add_hline(
            y=rec['or_low'],
            line_dash="dot",
            annotation_text=f"OR Low: ₹{rec['or_low']:.2f}",
            line_color="lightcoral",
            line_width=1,
            row=1, col=1
        )
    
    # Add Entry Level
    fig.add_hline(
        y=entry_price,
        annotation_text=f"Entry: ₹{entry_price:.2f}",
        line_color="white",
        line_width=2,
        row=1, col=1
    )
    
    # Add Stop Loss Level
    fig.add_hline(
        y=stop_price,
        line_dash="dash",
        annotation_text=f"Stop Loss: ₹{stop_price:.2f} (-{stop_pct:.2f}%)",
        line_color="red",
        line_width=2,
        row=1, col=1
    )
    
    # Add Target Level
    fig.add_hline(
        y=target_price,
        line_dash="dash",
        annotation_text=f"Target: ₹{target_price:.2f} (+{target_pct:.2f}%)",
        line_color="green",
        line_width=2,
        row=1, col=1
    )
    
    # Volume bars with color coding
    colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
              for i in range(len(data))]
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)
    
    # Add annotation box with trade details
    annotation_text = (
        f"<b>Trade Setup</b><br>"
        f"Signal: {rec['signal']}<br>"
        f"Score: {rec['score']}/100<br>"
        f"Risk: {stop_pct:.2f}%<br>"
        f"Reward: {target_pct:.2f}%<br>"
        f"R:R = 1:{risk_reward:.2f}"
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        text=annotation_text,
        showarrow=False,
        font=dict(size=11, color="white"),
        align="left",
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="white",
        borderwidth=1,
        borderpad=8
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def display_intraday_chart(rec, data):
    """Legacy function - redirects to enhanced chart"""
    return display_enhanced_chart(rec, data)

def main():
    init_database()
    st.set_page_config(page_title="StockGenie Pro", layout="wide")
    
    # Check for GitHub updates
    auto_update_check()
    
    # Display update notification if available
    if st.session_state.get('update_available', False):
        # Get full commit hashes for changelog
        local_full = st.session_state.get('local_commit_full', st.session_state.get('local_commit'))
        remote_full = st.session_state.get('remote_commit_full', st.session_state.get('remote_commit'))
        
        # Create a clean update banner
        update_container = st.container()
        with update_container:
            col1, col2 = st.columns([6, 1])
            with col1:
                st.info(f"🔄 **New version available!** Local: `{st.session_state.get('local_commit')}` → Remote: `{st.session_state.get('remote_commit')}`")
            with col2:
                if st.button("🚀 Update Now", type="primary", use_container_width=True, key="update_btn"):
                    with st.spinner("Pulling latest changes..."):
                        success, message = pull_github_updates()
                        if success:
                            st.success("✅ Updated! Reloading app...")
                            st.session_state.update_available = False
                            time_module.sleep(2)
                            st.rerun()
                        else:
                            st.error(f"❌ Update failed: {message}")
            
            # Show changelog in a separate expander below
            if local_full and remote_full:
                changelog = get_changelog(local_full, remote_full)
                if changelog:
                    with st.expander(f"📋 What's New ({len(changelog)} change{'s' if len(changelog) > 1 else ''})", expanded=False):
                        for idx, commit_msg in enumerate(changelog, 1):
                            # Parse commit message for type and description
                            if ':' in commit_msg:
                                commit_type, commit_desc = commit_msg.split(':', 1)
                                # Add emoji based on commit type
                                emoji = {
                                    'feat': '✨', 'fix': '🐛', 'update': '📝', 
                                    'refactor': '♻️', 'style': '🎨', 'docs': '📚',
                                    'perf': '⚡', 'test': '🧪', 'chore': '🔧'
                                }.get(commit_type.strip().lower(), '•')
                                st.markdown(f"{emoji} **{commit_type.strip()}**: {commit_desc.strip()}")
                            else:
                                st.markdown(f"• {commit_msg}")
        
        st.divider()
    
    # Main title with proper spacing
    st.title("📊 StockGenie Pro V2.9")
    st.subheader(f"📅 {datetime.now().strftime('%d %b %Y, %A')}")
    st.markdown("")  # Add spacing
    
    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.title("🔍 Configuration")
    
    api_provider = st.sidebar.selectbox("Data Provider", ["SmartAPI", "Dhan"], index=0)
    
    trading_style = st.sidebar.radio("Trading Style", ["Swing Trading", "Intraday Trading"])
    timeframe = "1d"
    if trading_style == "Intraday Trading":
        timeframe_display = st.sidebar.selectbox("Timeframe", ["5m", "15m", "30m"], index=1)
        timeframe = timeframe_display.replace(" min", "m")
    else:
        timeframe_display = "Daily"

    st.sidebar.divider()
    contrarian_mode = st.sidebar.checkbox("🎯 Contrarian Mode", value=False)
    if contrarian_mode: st.sidebar.info("⚠️ Market context weight reduced by 50%.")
    st.sidebar.divider()
    
    # Sector selection with auto-bullish option
    sector_selection_mode = st.sidebar.radio(
        "Sector Selection Mode", 
        ["Manual Selection", "Auto-Bullish Sectors", "Auto-Bullish + Neutral Sectors"],
        help="Manual: Choose sectors manually | Auto-Bullish: Only bullish sectors | Auto-Bullish + Neutral: Both bullish and neutral sectors"
    )
    
    if sector_selection_mode == "Auto-Bullish Sectors":
        # Get stocks from bullish sectors automatically
        with st.sidebar:
            with st.spinner("🔍 Analyzing sector performance..."):
                bullish_sectors = get_bullish_sectors()
                if bullish_sectors:
                    sector_names = [s['sector'] for s in bullish_sectors[:5]]  # Top 5 bullish sectors
                    st.success(f"📈 Found {len(bullish_sectors)} bullish sectors")
                    st.info("**Top Bullish Sectors:**\n" + "\n".join([f"• {s['sector']} (+{s['change']:.1f}%)" for s in bullish_sectors[:3]]))
                    
                    # Show detailed sector info in expander
                    with st.expander(f"📊 View All {len(bullish_sectors)} Bullish Sectors"):
                        for sector in bullish_sectors:
                            st.write(f"**{sector['sector']}**: +{sector['change']:.2f}% | "
                                   f"{sector['advancing']}/{sector['total']} stocks advancing "
                                   f"({sector['advance_ratio']:.1f}%)")
                    
                    stock_list = get_stocks_from_bullish_sectors(bullish_sectors)
                    st.metric("Selected Stocks", len(stock_list))
                else:
                    st.warning("⚠️ No bullish sectors found, using all sectors")
                    stock_list = get_unique_stock_list(SECTORS)
    
    elif sector_selection_mode == "Auto-Bullish + Neutral Sectors":
        # Get stocks from both bullish and neutral sectors
        with st.sidebar:
            with st.spinner("🔍 Analyzing sector performance..."):
                stock_list, bullish_sectors, neutral_sectors = get_stocks_from_bullish_and_neutral_sectors()
                
                if bullish_sectors or neutral_sectors:
                    st.success(f"📈 Found {len(bullish_sectors)} bullish + {len(neutral_sectors)} neutral sectors")
                    
                    # Show bullish sectors
                    if bullish_sectors:
                        st.info("**🟢 Top Bullish Sectors:**\n" + "\n".join([f"• {s['sector']} (+{s['change']:.1f}%)" for s in bullish_sectors[:3]]))
                    
                    # Show neutral sectors
                    if neutral_sectors:
                        st.info("**🟡 Top Neutral Sectors:**\n" + "\n".join([f"• {s['sector']} ({s['change']:+.1f}%)" for s in neutral_sectors[:3]]))
                    
                    # Show detailed info in expander
                    with st.expander(f"📊 View All Selected Sectors ({len(bullish_sectors + neutral_sectors)} total)"):
                        if bullish_sectors:
                            st.write("**🟢 Bullish Sectors:**")
                            for sector in bullish_sectors:
                                st.write(f"• **{sector['sector']}**: +{sector['change']:.2f}% | "
                                       f"{sector['advancing']}/{sector['total']} advancing "
                                       f"({sector['advance_ratio']:.1f}%)")
                        
                        if neutral_sectors:
                            st.write("**🟡 Neutral Sectors:**")
                            for sector in neutral_sectors:
                                st.write(f"• **{sector['sector']}**: {sector['change']:+.2f}% | "
                                       f"{sector['advancing']}/{sector['total']} advancing "
                                       f"({sector['advance_ratio']:.1f}%)")
                    
                    st.metric("Selected Stocks", len(stock_list))
                else:
                    st.warning("⚠️ No suitable sectors found, using all sectors")
                    stock_list = get_unique_stock_list(SECTORS)
    
    else:
        # Manual sector selection
        selected_sectors = st.sidebar.multiselect("Select Sectors", ["All"] + list(SECTORS.keys()), default=["All"])
        stock_list = get_stock_list_from_sectors(SECTORS, selected_sectors)
    symbol = st.sidebar.selectbox("Select Stock", stock_list, index=0) if stock_list else "SBIN-EQ"
    account_size = st.sidebar.number_input("Account Size (₹)", min_value=10000, value=30000, step=5000)

    st.sidebar.divider()
    st.sidebar.subheader("API Status")
    is_healthy, msg = check_api_health(api_provider)
    
    if is_healthy:
        st.sidebar.success(f"✅ {api_provider}: {msg}")
    else:
        st.sidebar.error(f"❌ {api_provider}: {msg}")
    
    # Telegram alerts section
    st.sidebar.divider()
    st.sidebar.subheader("📱 Telegram Alerts")
    
    telegram_status = TELEGRAM_CONFIG["enabled"] and TELEGRAM_CONFIG["bot_token"] and TELEGRAM_CONFIG["chat_id"]
    if telegram_status:
        st.sidebar.success("✅ Telegram: Connected")
        
        # Alert preferences
        with st.sidebar.expander("⚙️ Alert Settings"):
            alert_on_scan = st.checkbox("Scan Summary", value=TELEGRAM_CONFIG["alert_on_scan"], help="Send summary when scan completes")
            alert_on_high = st.checkbox("High Score Alerts", value=TELEGRAM_CONFIG["alert_on_high_score"], help="Alert for individual high-scoring stocks")
            alert_threshold = st.slider("Alert Threshold", 60, 90, TELEGRAM_CONFIG["alert_threshold"], help="Minimum score for individual alerts")
            
            # Update config
            TELEGRAM_CONFIG["alert_on_scan"] = alert_on_scan
            TELEGRAM_CONFIG["alert_on_high_score"] = alert_on_high
            TELEGRAM_CONFIG["alert_threshold"] = alert_threshold
            
            # Test button
            if st.button("📤 Send Test Alert", use_container_width=True):
                test_msg = f"🤖 <b>StockGenie Pro</b>\n\n✅ Telegram alerts are working!\n⏰ {datetime.now().strftime('%H:%M:%S')}"
                if send_telegram_message(test_msg):
                    st.success("Test alert sent!")
                else:
                    st.error("Failed to send test alert")
    else:
        st.sidebar.warning("⚠️ Telegram: Not configured")
        with st.sidebar.expander("ℹ️ Setup Instructions"):
            st.markdown("""
            **To enable Telegram alerts:**
            
            1. Create a bot with [@BotFather](https://t.me/botfather)
            2. Get your bot token
            3. Start a chat with your bot
            4. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
            5. Add to Streamlit secrets:
               ```
               TELEGRAM_ENABLED = "true"
               TELEGRAM_BOT_TOKEN = "your_token"
               TELEGRAM_CHAT_ID = "your_chat_id"
               ```
            """)
    
    # Auto-update section in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("🔄 Auto-Update")
    
    if AUTO_UPDATE_CONFIG["enabled"]:
        # Show current commit info
        try:
            repo_path = os.path.dirname(os.path.abspath(__file__))
            current_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_path,
                timeout=5
            ).decode('utf-8').strip()
            st.sidebar.caption(f"📌 Current: `{current_commit}`")
        except Exception as e:
            st.sidebar.caption(f"⚠️ Git error: {str(e)}")
            pass
        
        # Display status and check button in same row
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            last_check_time = st.session_state.get('last_update_check')
            if last_check_time:
                time_ago = int((datetime.now() - last_check_time).total_seconds() / 60)
                st.markdown(f"<div style='padding-top: 8px;'>✅ Last check: {time_ago}m ago</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='padding-top: 8px;'>✅ Checks every {AUTO_UPDATE_CONFIG['check_interval']//60}min</div>", unsafe_allow_html=True)
        with col2:
            if st.button("🔍", help="Check for updates now", key="check_updates", use_container_width=True):
                # Show checking status immediately
                status_placeholder = st.sidebar.empty()
                status_placeholder.info("🔄 Checking...")
                
                has_update, local, remote = check_for_github_updates()
                
                if has_update:
                    st.session_state.update_available = True
                    st.session_state.local_commit = local[:7] if local else "unknown"
                    st.session_state.remote_commit = remote[:7] if remote else "unknown"
                    st.session_state.local_commit_full = local  # Store full hash for changelog
                    st.session_state.remote_commit_full = remote  # Store full hash for changelog
                    status_placeholder.success("✅ Update found!")
                    time_module.sleep(1)
                    st.rerun()
                else:
                    status_placeholder.success("✅ Up to date!")
                    time_module.sleep(1)
                    status_placeholder.empty()
    else:
        st.sidebar.caption("⏸️ Auto-update disabled")

    # --- SESSION STATE & TABS ---
    if 'scan_running' not in st.session_state: st.session_state.scan_running = False
    if 'scan_results' not in st.session_state: st.session_state.scan_results = None
    if 'scan_params' not in st.session_state: st.session_state.scan_params = {}

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Analysis", "🔍 Scanner", "🎯 Technical Screener", "🔄 Live Intraday", "💰 Paper Trading", "🌍 Market Dashboard"])


    # --- ANALYSIS TAB ---
    with tab1:
        if st.button("🔍 Analyze Selected Stock"):
            with st.spinner(f"Analyzing {symbol} using {api_provider}..."):
                try:
                    data = fetch_stock_data_cached(symbol, interval=timeframe, api_provider=api_provider)
                    if not data.empty:
                        rec = generate_recommendation(data, symbol, 'swing' if trading_style == "Swing Trading" else 'intraday', timeframe, account_size, contrarian_mode)
                        
                        # Display metrics in columns
                        st.subheader(f"📊 {rec['symbol']} - {rec['trading_style'].upper()} Analysis")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        col1.metric("Score", f"{rec['score']}/100", 
                                   delta="Strong" if rec['score'] >= 70 else "Moderate" if rec['score'] >= 50 else "Weak")
                        col2.metric("Signal", rec['signal'])
                        col3.metric("Regime", rec['regime'])
                        col4.metric("Current Price", f"₹{rec['current_price']}")
                        
                        # Calculate R:R ratio
                        risk = abs(rec['buy_at'] - rec['stop_loss'])
                        reward = abs(rec['target'] - rec['buy_at'])
                        rr_ratio = reward / risk if risk > 0 else 0
                        col5.metric("Risk:Reward", f"1:{rr_ratio:.2f}",
                                   delta="Good" if rr_ratio >= 2 else "Fair" if rr_ratio >= 1.5 else "Low")
                        
                        # Display detailed trade setup
                        st.info(f"**📋 Analysis Reason**: {rec['reason']}")
                        
                        # Check if this is a sell signal
                        is_sell_signal = rec['signal'] in ['Sell', 'Strong Sell']
                        
                        if is_sell_signal:
                            st.warning("⚠️ **SELL SIGNAL - NOT RECOMMENDED FOR LONG ENTRY**")
                            st.error("🚫 This stock shows bearish signals. Avoid buying or consider exiting existing positions.")
                        
                        # Trade setup details in expandable section
                        expander_title = "📍 **Hypothetical Long Trade Setup**" if is_sell_signal else "📍 **Detailed Trade Setup**"
                        with st.expander(expander_title, expanded=not is_sell_signal):
                            if is_sell_signal:
                                st.warning("⚠️ Note: This shows what a long trade would look like, but it's NOT recommended due to bearish signals.")
                            
                            tcol1, tcol2, tcol3 = st.columns(3)
                            
                            with tcol1:
                                st.markdown("### 🎯 Entry & Targets")
                                st.write(f"**Entry Price:** ₹{rec['buy_at']:.2f}")
                                st.write(f"**Target Price:** ₹{rec['target']:.2f}")
                                target_gain = ((rec['target'] - rec['buy_at']) / rec['buy_at'] * 100)
                                st.success(f"**Potential Gain:** +{target_gain:.2f}%")
                            
                            with tcol2:
                                st.markdown("### 🛑 Risk Management")
                                st.write(f"**Stop Loss:** ₹{rec['stop_loss']:.2f}")
                                risk_percent = abs((rec['buy_at'] - rec['stop_loss']) / rec['buy_at'] * 100)
                                st.error(f"**Max Risk:** -{risk_percent:.2f}%")
                                st.write(f"**Risk Amount:** ₹{risk:.2f}/share")
                            
                            with tcol3:
                                st.markdown("### 💰 Position Sizing")
                                shares = int(account_size / rec['buy_at'])
                                total_risk = shares * risk
                                total_reward = shares * reward
                                st.write(f"**Shares to Buy:** {shares}")
                                st.write(f"**Total Investment:** ₹{shares * rec['buy_at']:.2f}")
                                st.write(f"**Max Loss:** ₹{total_risk:.2f}")
                                st.write(f"**Potential Profit:** ₹{total_reward:.2f}")
                        
                        # Display enhanced chart
                        st.markdown("---")
                        fig = display_enhanced_chart(rec, data)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display latest news
                        st.markdown("---")
                        display_stock_news(symbol, max_news=5)
                            
                    else: st.warning("No data available for the selected stock.")
                except Exception as e: st.error(f"❌ Error: {str(e)}")

    # --- SCANNER TAB ---
    with tab2:
        st.markdown("### 📡 Stock Scanner")
        
        # Display scan information
        col1, col2 = st.columns([2, 1])
        with col1:
            if sector_selection_mode == "Auto-Bullish Sectors":
                bullish_sectors_info = get_bullish_sectors()
                if bullish_sectors_info:
                    sector_list = [s['sector'] for s in bullish_sectors_info[:3]]
                    st.info(f"🎯 **Auto-Bullish Mode**: Scanning {len(stock_list)} stocks from {len(bullish_sectors_info)} bullish sectors: {', '.join(sector_list)}{'...' if len(bullish_sectors_info) > 3 else ''}")
                else:
                    st.warning("⚠️ No bullish sectors detected, scanning all sectors")
            elif sector_selection_mode == "Auto-Bullish + Neutral Sectors":
                _, bullish_sectors_info, neutral_sectors_info = get_stocks_from_bullish_and_neutral_sectors()
                if bullish_sectors_info or neutral_sectors_info:
                    total_sectors = len(bullish_sectors_info) + len(neutral_sectors_info)
                    st.info(f"🎯 **Auto-Mixed Mode**: Scanning {len(stock_list)} stocks from {len(bullish_sectors_info)} bullish + {len(neutral_sectors_info)} neutral sectors ({total_sectors} total)")
                else:
                    st.warning("⚠️ No suitable sectors detected, scanning all sectors")
            else:
                if "All" in selected_sectors:
                    st.info(f"📊 **Manual Mode**: Scanning {len(stock_list)} stocks from all sectors")
                else:
                    st.info(f"📊 **Manual Mode**: Scanning {len(stock_list)} stocks from {len(selected_sectors)} selected sectors: {', '.join(selected_sectors[:3])}{'...' if len(selected_sectors) > 3 else ''}")
        
        with col2:
            st.metric("Total Stocks", len(stock_list))
        
        current_scan_params = {"trading_style": trading_style, "timeframe": timeframe, "contrarian_mode": contrarian_mode, "api_provider": api_provider}
        
        can_auto_resume = False
        progress_info = get_scan_progress_info()
        
        # Display checkpoint information if available
        if progress_info['exists']:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Completed", progress_info['completed'])
            with col2:
                st.metric("Success Rate", f"{progress_info['success_rate']:.1f}%")
            with col3:
                st.metric("Age", f"{progress_info['age_minutes']:.1f}m")
            
            if (progress_info['api_provider'] == api_provider and 
                progress_info['trading_style'] == trading_style and 
                progress_info['timeframe'] == timeframe and
                progress_info['age_minutes'] < 60):  # 1 hour expiry
                can_auto_resume = True
                st.success(f"✅ Can resume previous scan ({progress_info['completed']} stocks completed)")
            else:
                st.warning("⚠️ Previous scan found but parameters don't match or expired")

        if not st.session_state.scan_running:
            if st.button("🚀 Start / Resume Scan", type="primary", use_container_width=True):
                if not can_auto_resume: clear_checkpoint()
                st.session_state.scan_running = True
                st.session_state.scan_params = current_scan_params
                st.session_state.scan_results = None
                st.rerun()

        if st.session_state.scan_running:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("⏹️ Cancel Scan (Saves Progress)", use_container_width=True):
                    st.session_state.scan_running = False
                    st.info("🔄 Scan cancelled. Progress saved for resuming later.")
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear & Stop", use_container_width=True):
                    st.session_state.scan_running = False
                    clear_checkpoint()
                    st.warning("🗑️ Scan stopped and progress cleared.")
                    st.rerun()
            
            progress, status_text = st.progress(0), st.empty()
            scan_info = st.empty()
            
            try:
                def update_progress(pct):
                    scan_count = min(len(stock_list), SCAN_CONFIG["max_stocks_per_scan"])
                    progress.progress(pct)
                    current_stock = int(pct * scan_count)
                    status_text.text(f"📊 Scanning... {int(pct*100)}% ({current_stock}/{scan_count})")
                    
                    # Show additional scan info
                    if current_stock > 0:
                        eta_minutes = ((1 - pct) * scan_count * SCAN_CONFIG["delay_within_batch"]) / 60
                        scan_info.info(f"⏱️ ETA: ~{eta_minutes:.0f} minutes | Batch size: {SCAN_CONFIG['batch_size']} | API: {api_provider}")
                
                results = analyze_multiple_stocks(stock_list, 'swing' if trading_style == "Swing Trading" else 'intraday', timeframe, update_progress, can_auto_resume, contrarian_mode, api_provider)
                st.session_state.scan_results = results
                st.session_state.scan_running = False
                clear_checkpoint()
                
                # Send Telegram alert if enabled
                if not results.empty and TELEGRAM_CONFIG["enabled"]:
                    try:
                        scan_time = datetime.now().strftime("%H:%M:%S")
                        send_scan_results_alert(results, trading_style, scan_time)
                        
                        # Send individual alerts for very high scores
                        high_score_stocks = results[results['Score'] >= TELEGRAM_CONFIG['alert_threshold']].head(TELEGRAM_CONFIG['max_alerts_per_scan'])
                        for idx, stock in high_score_stocks.iterrows():
                            send_high_score_alert(stock.to_dict())
                            time_module.sleep(1)  # Small delay between alerts
                    except Exception as e:
                        logging.error(f"Error sending Telegram alerts: {e}")
                
                st.rerun()
            except Exception as e:
                st.session_state.scan_running = False
                st.error(f"❌ Scan failed: {e}")

        if st.session_state.scan_results is not None:
            results = st.session_state.scan_results
            if not results.empty:
                save_picks(results, trading_style)
                st.subheader(f"🏆 Top {trading_style} Picks")
                
                # Style the dataframe with better color contrast
                def style_score(val):
                    """Style score column with readable colors"""
                    try:
                        score = float(val)
                        if score >= 75:
                            return 'background-color: #2d5016; color: #90EE90; font-weight: bold'  # Dark green bg, light green text
                        elif score >= 60:
                            return 'background-color: #1a4d2e; color: #90EE90; font-weight: bold'  # Dark green bg, light green text
                        elif score >= 50:
                            return 'background-color: #3d3d3d; color: #FFD700; font-weight: bold'  # Dark gray bg, gold text
                        else:
                            return 'background-color: #4a1a1a; color: #FF6B6B; font-weight: bold'  # Dark red bg, light red text
                    except:
                        return ''
                
                styled_df = results.style.applymap(style_score, subset=['Score'])
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.warning("⚠️ No stocks met the criteria.")

    # --- TECHNICAL SCREENER TAB ---
    with tab3:
        st.markdown("### 🎯 Technical Screener - Strong Bullish Signals")
        st.caption("Stocks showing Strong Bullish trends in ADX, Volume, and MACD")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            filter_mode = st.radio(
                "Filter Mode:",
                ["Strong Only (All 3 Bullish)", "Moderate (2 out of 3 Bullish)"],
                index=0
            )
        
        if st.button("🔍 Scan Technical Indicators", type="primary", use_container_width=True):
            with st.spinner("Fetching technical data..."):
                tech_data = fetch_technical_screener()
                
                if tech_data:
                    strong_only = "Strong Only" in filter_mode
                    bullish_stocks = filter_bullish_stocks(tech_data, strong_only)
                    
                    if bullish_stocks:
                        df = pd.DataFrame(bullish_stocks)
                        
                        st.success(f"✅ Found {len(bullish_stocks)} stocks matching criteria (out of {tech_data.get('totalCount', 0)} total)")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Avg RSI", f"{df['RSI'].mean():.1f}")
                        col2.metric("Avg ADX", f"{df['ADX'].mean():.1f}")
                        col3.metric("Avg Vol Ratio", f"{df['Volume Ratio'].mean():.2f}")
                        col4.metric("Above EMA200", f"{df['Above EMA200'].sum()}/{len(df)}")
                        
                        # Style the dataframe
                        def highlight_trends(val):
                            if 'Strong Bullish' in str(val):
                                return 'background-color: #90EE90; font-weight: bold'
                            elif 'Bullish' in str(val):
                                return 'background-color: #9ADAD8'
                            elif 'Bearish' in str(val):
                                return 'background-color: #FFB6C6'
                            return ''
                        
                        styled_df = df.style.applymap(
                            highlight_trends, 
                            subset=['ADX Trend', 'Volume Trend', 'MACD Trend']
                        ).format({
                            'Price': '₹{:.2f}',
                            'RSI': '{:.1f}',
                            'ADX': '{:.1f}',
                            'Volume Ratio': '{:.2f}',
                            'Target': '₹{:.2f}',
                            'Stop Loss': '₹{:.2f}'
                        })
                        
                        st.dataframe(styled_df, use_container_width=True, height=500)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"technical_screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"⚠️ No stocks found matching the criteria. Try 'Moderate' filter mode.")
                else:
                    st.error("❌ Failed to fetch technical data. Please try again.")


    # --- LIVE INTRADAY SCANNER TAB ---
    with tab4:
        st.markdown("### 🔄 Live Intraday Scanner")
        st.caption("🎯 Automatically scans stocks from bullish sectors only")
        st.markdown("")  # Add spacing
        
        # Initialize session state for live scanner
        if 'live_scan_active' not in st.session_state:
            st.session_state.live_scan_active = False
        if 'live_scan_results' not in st.session_state:
            st.session_state.live_scan_results = []
        if 'live_scan_alerts' not in st.session_state:
            st.session_state.live_scan_alerts = []
        if 'alert_history' not in st.session_state:
            st.session_state.alert_history = {}
        if 'last_scan_time' not in st.session_state:
            st.session_state.last_scan_time = None
        if 'scan_iteration' not in st.session_state:
            st.session_state.scan_iteration = 0
        if 'scan_status' not in st.session_state:
            st.session_state.scan_status = ""
        
        # Configuration section
        st.markdown("#### ⚙️ Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            scan_timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m"], index=1, key="live_tf")
        with col2:
            alert_threshold = st.slider("Alert Score", 60, 80, 65, key="alert_score")
            LIVE_SCAN_CONFIG['alert_score_threshold'] = alert_threshold
        with col3:
            scan_interval = st.slider("Scan Interval (sec)", 60, 300, 120, step=30, key="scan_int")
            LIVE_SCAN_CONFIG['scan_interval'] = scan_interval
        
        # Display current sector analysis
        st.markdown("")  # Add spacing
        st.markdown("#### 📊 Current Bullish Sectors")
        bullish_sectors = get_bullish_sectors()
        
        # Display bullish sectors
        if bullish_sectors:
            st.markdown(f"**🟢 {len(bullish_sectors)} Bullish Sectors Found**")
            st.markdown("")  # Add spacing
            sector_cols = st.columns(min(len(bullish_sectors), 4))
            for idx, sector in enumerate(bullish_sectors[:4]):
                with sector_cols[idx]:
                    st.metric(
                        label=f"🟢 {sector['sector']}",
                        value=f"{sector['change']:+.2f}%",
                        delta=f"{sector['advance_ratio']:.0f}% advancing"
                    )
        
        # Expandable detailed view with stocks list
        if bullish_sectors:
            # Get stock list from bullish sectors
            stock_list_from_sectors = get_stocks_from_bullish_sectors(bullish_sectors)
            
            with st.expander(f"📋 View All Bullish Sectors & Stocks ({len(bullish_sectors)} sectors, {len(stock_list_from_sectors)} stocks)"):
                for sector in bullish_sectors:
                    st.markdown(f"**• {sector['sector']}**: {sector['change']:+.2f}% | "
                            f"{sector['advancing']}/{sector['total']} advancing ({sector['advance_ratio']:.1f}%)")
                
                st.markdown("")
                st.markdown(f"**📊 {len(stock_list_from_sectors)} Stocks to be scanned:**")
                
                # Display stocks in columns for better readability
                if len(stock_list_from_sectors) > 0:
                    stocks_per_col = 15
                    num_cols = min(4, max(1, (len(stock_list_from_sectors) + stocks_per_col - 1) // stocks_per_col))
                    stock_cols = st.columns(num_cols)
                    
                    for idx, stock in enumerate(stock_list_from_sectors):
                        col_idx = idx % num_cols
                        with stock_cols[col_idx]:
                            st.caption(f"• {stock}")
                else:
                    st.info("No stocks available in the selected sectors.")
        else:
            st.warning("⚠️ No bullish sectors found at the moment")
        
        # Show quick summary info
        if bullish_sectors:
            stock_list_summary = get_stocks_from_bullish_sectors(bullish_sectors)
            st.info(f"📊 **Ready to scan {len(stock_list_summary)} stocks** from {len(bullish_sectors)} bullish sector{'s' if len(bullish_sectors) > 1 else ''}")
        
        st.markdown("")  # Add spacing
        st.divider()
        st.markdown("#### 🎮 Controls")
        
        # Control buttons with better spacing
        col1, col2, col3 = st.columns([3, 3, 2])
        
        with col1:
            if not st.session_state.live_scan_active:
                if st.button("🚀 Start Live Scanner", type="primary", use_container_width=True):
                    if not bullish_sectors:
                        st.error("❌ Cannot start: No bullish sectors found")
                    else:
                        st.session_state.live_scan_active = True
                        st.session_state.scan_iteration = 0
                        st.session_state.live_scan_alerts = []
                        st.session_state.scan_status = "Starting scanner..."
                        st.rerun()
            else:
                if st.button("⏹️ Stop Scanner", type="secondary", use_container_width=True):
                    st.session_state.live_scan_active = False
                    st.session_state.scan_status = "Scanner stopped"
                    st.rerun()
        
        with col2:
            if st.button("🔄 Manual Scan Now", use_container_width=True):
                if not bullish_sectors:
                    st.error("❌ No bullish sectors available")
                else:
                    # Status placeholder
                    status_placeholder = st.empty()
                    
                    def update_status(msg):
                        status_placeholder.info(msg)
                    
                    update_status("🔍 Fetching stocks from bullish sectors...")
                    stock_list = get_stocks_from_bullish_sectors(bullish_sectors)
                    
                    if stock_list:
                        # Show detailed info about what we're scanning
                        sector_names = [s['sector'] for s in bullish_sectors]
                        update_status(f"📊 Found {len(stock_list)} stocks from {len(bullish_sectors)} bullish sectors: {', '.join(sector_names[:3])}{'...' if len(sector_names) > 3 else ''}")
                        results, alerts = live_scan_iteration(
                            stock_list[:50],  # Limit to 50 stocks for manual scan
                            scan_timeframe,
                            api_provider,
                            st.session_state.alert_history,
                            status_callback=update_status
                        )
                        st.session_state.live_scan_results = results
                        st.session_state.live_scan_alerts.extend(alerts)
                        st.session_state.last_scan_time = datetime.now()
                        status_placeholder.success(f"✅ Scan complete! Found {len(results)} opportunities from {len(bullish_sectors)} bullish sectors")
                        time_module.sleep(2)
                        st.rerun()
                    else:
                        status_placeholder.warning("⚠️ No stocks found in bullish sectors")
        
        with col3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.live_scan_results = []
                st.session_state.live_scan_alerts = []
                st.session_state.alert_history = {}
                st.rerun()
        
        # Live scanning logic
        if st.session_state.live_scan_active:
            # Display status
            status_container = st.container()
            with status_container:
                col1, col2, col3 = st.columns(3)
                col1.success("🟢 **Scanner Active**")
                col2.info(f"⏱️ Interval: {scan_interval}s")
                if st.session_state.last_scan_time:
                    next_scan = st.session_state.last_scan_time + timedelta(seconds=scan_interval)
                    time_until = (next_scan - datetime.now()).total_seconds()
                    col3.warning(f"⏳ Next scan in: {max(0, int(time_until))}s")
            
            # Display current status
            if st.session_state.scan_status:
                st.info(f"📢 **Status:** {st.session_state.scan_status}")
            
            # Check if it's time to scan
            should_scan = False
            if st.session_state.last_scan_time is None:
                should_scan = True
            else:
                time_since_last = (datetime.now() - st.session_state.last_scan_time).total_seconds()
                if time_since_last >= scan_interval:
                    should_scan = True
            
            if should_scan:
                # Status placeholder for live updates
                status_placeholder = st.empty()
                
                def update_status(msg):
                    st.session_state.scan_status = msg
                    try:
                        status_placeholder.info(f"📢 {msg}")
                    except Exception:
                        # Silently handle any Streamlit state errors during updates
                        pass
                
                update_status(f"🔍 Starting scan iteration #{st.session_state.scan_iteration + 1}...")
                
                try:
                    # Get fresh bullish sectors only
                    current_bullish = get_bullish_sectors()
                    
                    if current_bullish:
                        update_status(f"📊 Found {len(current_bullish)} bullish sectors")
                        stock_list = get_stocks_from_bullish_sectors(current_bullish)
                        
                        if stock_list:
                            update_status(f"🎯 Scanning {len(stock_list)} stocks from bullish sectors...")
                            results, alerts = live_scan_iteration(
                                stock_list,
                                scan_timeframe,
                                api_provider,
                                st.session_state.alert_history,
                                status_callback=update_status
                            )
                            
                            st.session_state.live_scan_results = results
                            st.session_state.live_scan_alerts.extend(alerts)
                            st.session_state.last_scan_time = datetime.now()
                            st.session_state.scan_iteration += 1
                            
                            if alerts:
                                st.toast(f"🚨 {len(alerts)} new alerts!", icon="🚨")
                                update_status(f"✅ Scan complete! {len(alerts)} new alerts found")
                            else:
                                update_status(f"✅ Scan complete! {len(results)} opportunities found")
                    else:
                        update_status("⚠️ No bullish sectors found, waiting for next scan...")
                except Exception as e:
                    logging.error(f"Error during scan iteration: {e}")
                    update_status(f"❌ Scan error: {str(e)}")
                
                time_module.sleep(2)
                try:
                    st.rerun()
                except Exception as e:
                    logging.warning(f"Rerun failed: {e}")
            else:
                # Auto-refresh every 5 seconds to update countdown
                time_module.sleep(5)
                try:
                    st.rerun()
                except Exception as e:
                    logging.warning(f"Rerun failed: {e}")
        
        # Display alerts section with proper spacing
        st.markdown("")  # Add spacing
        
        # Always show results section if we have data (whether scanner is active or not)
        if st.session_state.live_scan_alerts:
            st.markdown("### 🚨 Recent Alerts")
            alerts_df = pd.DataFrame(st.session_state.live_scan_alerts[-10:])  # Last 10 alerts
            
            # Style alerts
            def highlight_alerts(row):
                if row['Score'] >= 75:
                    return ['background-color: #90EE90'] * len(row)
                elif row['Score'] >= 70:
                    return ['background-color: #FFFACD'] * len(row)
                return [''] * len(row)
            
            styled_alerts = alerts_df.style.apply(highlight_alerts, axis=1)
            st.dataframe(styled_alerts, use_container_width=True, height=300)
            st.markdown("")  # Add spacing
        
        st.divider()
        
        # Display all results - ALWAYS show if we have results
        if st.session_state.live_scan_results and len(st.session_state.live_scan_results) > 0:
            st.markdown("### 📊 Current Scan Results")
            
            try:
                results_df = pd.DataFrame(st.session_state.live_scan_results)
                
                # Show total results found first
                st.info(f"📊 Found **{len(results_df)}** total opportunities in last scan")
                
                # Filter for buy signals only
                buy_signals = results_df[results_df['Signal'].str.contains('Buy', na=False)]
                
                if not buy_signals.empty:
                    # Sort by score
                    buy_signals = buy_signals.sort_values('Score', ascending=False)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Buy Signals", len(buy_signals))
                    col2.metric("Avg Score", f"{buy_signals['Score'].mean():.1f}")
                    col3.metric("Strong Buys", len(buy_signals[buy_signals['Signal'] == 'Strong Buy']))
                    
                    st.markdown("")  # Add spacing
                    
                    # Display top 20 results
                    display_df = buy_signals.head(20)
                    
                    # Style the results
                    def color_score(val):
                        if val >= 75:
                            return 'background-color: #90EE90; font-weight: bold'
                        elif val >= 70:
                            return 'background-color: #FFFACD'
                        elif val >= 65:
                            return 'background-color: #E0E0E0'
                        return ''
                    
                    styled_results = display_df.style.applymap(color_score, subset=['Score'])
                    st.dataframe(styled_results, use_container_width=True, height=500)
                    
                    # Download button
                    csv = buy_signals.to_csv(index=False)
                    st.download_button(
                        label="📥 Download All Results",
                        data=csv,
                        file_name=f"live_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("ℹ️ No buy signals in current scan - all stocks show Hold/Sell signals")
                
                # Display scan stats
                if st.session_state.last_scan_time:
                    st.caption(f"📅 Last scan: {st.session_state.last_scan_time.strftime('%H:%M:%S')} | "
                              f"🔄 Iteration: #{st.session_state.scan_iteration} | "
                              f"🚨 Total alerts: {len(st.session_state.live_scan_alerts)}")
            except Exception as e:
                st.error(f"Error displaying results: {e}")
                logging.error(f"Error displaying scan results: {e}")
        elif st.session_state.get('scan_iteration', 0) > 0:
            st.info("ℹ️ No results yet from the scanner. Waiting for first scan to complete...")


    # --- PAPER TRADING TAB ---
    with tab5:
        st.markdown("### 💰 Paper Trading - Practice Without Risk")
        
        if not supabase:
            st.warning("⚠️ **Paper Trading is not configured**")
            st.info("""
            To enable Paper Trading:
            1. Create a Supabase account at [supabase.com](https://supabase.com)
            2. Create the required tables (see SUPABASE_SETUP.md)
            3. Add your credentials to Streamlit secrets or .env:
               - SUPABASE_URL
               - SUPABASE_KEY
            4. Install supabase: `pip install supabase`
            """)
        else:
            # Get account info
            account = get_paper_account()
            
            if not account:
                st.error("❌ Paper trading account not found. Please check Supabase setup.")
            else:
                # Account summary
                portfolio = get_paper_portfolio()
                
                # Calculate portfolio value
                portfolio_value = 0
                portfolio_pnl = 0
                
                if portfolio:
                    for position in portfolio:
                        try:
                            # Try to fetch current price with multiple attempts
                            current_price = None
                            
                            # Try daily data first
                            try:
                                current_data = fetch_stock_data_cached(
                                    position['symbol'], 
                                    period="5d", 
                                    interval="1d", 
                                    api_provider=api_provider
                                )
                                if not current_data.empty and 'Close' in current_data.columns:
                                    current_price = float(current_data['Close'].iloc[-1])
                            except:
                                pass
                            
                            # Try intraday if daily failed
                            if current_price is None or current_price <= 0:
                                try:
                                    current_data = fetch_stock_data_cached(
                                        position['symbol'], 
                                        period="1d", 
                                        interval="5m", 
                                        api_provider=api_provider
                                    )
                                    if not current_data.empty and 'Close' in current_data.columns:
                                        current_price = float(current_data['Close'].iloc[-1])
                                except:
                                    pass
                            
                            # Use average price as fallback
                            if current_price is None or current_price <= 0:
                                current_price = position['avg_price']
                            
                            position_value = position['quantity'] * current_price
                            position_pnl = position_value - position['invested_amount']
                            portfolio_value += position_value
                            portfolio_pnl += position_pnl
                        except Exception as e:
                            logging.warning(f"Error calculating portfolio value for {position['symbol']}: {e}")
                            # If fetch fails, use invested amount
                            portfolio_value += position['invested_amount']
                
                total_value = account['cash_balance'] + portfolio_value
                total_pnl = total_value - account['initial_balance']
                total_pnl_percent = (total_pnl / account['initial_balance']) * 100 if account['initial_balance'] > 0 else 0
                
                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("💵 Cash Balance", f"₹{account['cash_balance']:,.2f}")
                col2.metric("📊 Portfolio Value", f"₹{portfolio_value:,.2f}")
                col3.metric("💼 Total Value", f"₹{total_value:,.2f}")
                
                pnl_delta = f"{total_pnl_percent:+.2f}%"
                col4.metric("📈 Total P&L", f"₹{total_pnl:,.2f}", delta=pnl_delta)
                col5.metric("🎯 Positions", len(portfolio))
                
                st.divider()
                
                # Trade execution form
                st.markdown("#### 🔄 Execute Trade")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    trade_action = st.radio("Action", ["BUY", "SELL"], horizontal=True)
                    trade_symbol = st.selectbox("Stock", stock_list, key="paper_trade_symbol")
                    trade_quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
                
                with col2:
                    trade_style = st.radio("Trading Style", ["Swing", "Intraday"], horizontal=True, key="paper_trade_style")
                    
                    # Get current price with multiple fallback attempts
                    current_price_available = False
                    suggested_price = 100.0
                    fetch_error = None
                    
                    # Try different intervals and periods
                    fetch_attempts = [
                        ("5d", "1d"),      # 5 days, daily interval
                        ("1d", "5m"),      # 1 day, 5min interval (for intraday)
                        ("1d", "15m"),     # 1 day, 15min interval
                        ("2y", "1d"),      # 2 years, daily (long history)
                    ]
                    
                    for period, interval in fetch_attempts:
                        try:
                            current_data = fetch_stock_data_cached(
                                trade_symbol, 
                                period=period, 
                                interval=interval, 
                                api_provider=api_provider
                            )
                            
                            if current_data is not None and not current_data.empty and 'Close' in current_data.columns:
                                suggested_price = float(current_data['Close'].iloc[-1])
                                if suggested_price > 0:  # Valid price
                                    current_price_available = True
                                    break
                        except Exception as e:
                            fetch_error = str(e)
                            continue
                    
                    # Show current market price with refresh button
                    price_col1, price_col2 = st.columns([4, 1])
                    with price_col1:
                        if current_price_available:
                            st.success(f"📊 **Current Price:** ₹{suggested_price:.2f}")
                        else:
                            st.warning("⚠️ Unable to fetch price - using default")
                            if fetch_error:
                                with st.expander("🔍 Debug Info"):
                                    st.caption(f"Error: {fetch_error}")
                                    st.caption(f"API: {api_provider}")
                    
                    with price_col2:
                        if st.button("🔄", help="Refresh price", key="refresh_price"):
                            st.rerun()
                    
                    trade_price = st.number_input("Price (₹)", min_value=0.01, value=float(suggested_price), step=0.05, format="%.2f")
                    trade_notes = st.text_input("Notes (optional)", key="paper_trade_notes")
                
                # Calculate and display charges
                trade_value = trade_quantity * trade_price
                charges_breakdown = calculate_brokerage_charges(trade_value, trade_action, trade_style.lower())
                total_charges = charges_breakdown['total_charges']
                total_with_charges = trade_value + total_charges if trade_action == 'BUY' else trade_value - total_charges
                
                # Display cost breakdown
                with st.expander(f"💰 Cost Breakdown (Click to expand)", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Trade Value:** ₹{trade_value:,.2f}")
                        st.markdown(f"**Total Charges:** ₹{total_charges:.2f} ({charges_breakdown['total_percentage']:.3f}%)")
                        if trade_action == 'BUY':
                            st.markdown(f"**Total Debit:** ₹{total_with_charges:,.2f}")
                        else:
                            st.markdown(f"**Net Credit:** ₹{total_with_charges:,.2f}")
                    
                    with col2:
                        st.caption("**Charges Breakup:**")
                        for item in charges_breakdown['breakup']:
                            if item['amount'] > 0:
                                st.caption(f"• {item['name']}: ₹{item['amount']:.2f}")
                
                # Summary box
                if trade_action == 'BUY':
                    st.info(f"💰 **Total Cost:** ₹{total_with_charges:,.2f} (₹{trade_value:,.2f} + ₹{total_charges:.2f} charges)")
                else:
                    st.info(f"💰 **Net Proceeds:** ₹{total_with_charges:,.2f} (₹{trade_value:,.2f} - ₹{total_charges:.2f} charges)")
                
                # Validation checks before execution
                can_execute = True
                error_message = None
                
                if trade_action == "BUY":
                    if total_with_charges > account['cash_balance']:
                        can_execute = False
                        error_message = f"⚠️ Insufficient funds! Need ₹{total_with_charges:,.2f}, have ₹{account['cash_balance']:,.2f}"
                elif trade_action == "SELL":
                    # Check if position exists
                    existing_position = None
                    for pos in portfolio:
                        if pos['symbol'] == trade_symbol and pos['trading_style'] == trade_style.lower():
                            existing_position = pos
                            break
                    
                    if not existing_position:
                        can_execute = False
                        error_message = f"⚠️ No {trade_style} position found for {trade_symbol}. Buy first before selling!"
                    elif existing_position['quantity'] < trade_quantity:
                        can_execute = False
                        error_message = f"⚠️ Insufficient shares! You have {existing_position['quantity']} shares, trying to sell {trade_quantity}"
                    else:
                        # Show position info for SELL
                        avg_price = existing_position['avg_price']
                        potential_pnl = (trade_price - avg_price) * trade_quantity
                        potential_pnl_pct = ((trade_price - avg_price) / avg_price) * 100
                        pnl_color = "🟢" if potential_pnl > 0 else "🔴" if potential_pnl < 0 else "⚪"
                        
                        st.info(f"""
                        📊 **Position Details:**
                        - Shares Available: {existing_position['quantity']}
                        - Average Buy Price: ₹{avg_price:.2f}
                        - Potential P&L: {pnl_color} ₹{potential_pnl:,.2f} ({potential_pnl_pct:+.2f}%)
                        """)
                
                if error_message:
                    st.error(error_message)
                
                if st.button(f"✅ Execute {trade_action}", type="primary", use_container_width=True, disabled=not can_execute):
                    with st.spinner(f"Executing {trade_action}..."):
                        success, message = execute_paper_trade(
                            trade_symbol,
                            trade_action,
                            trade_quantity,
                            trade_price,
                            trade_style.lower(),
                            trade_notes
                        )
                        
                        if success:
                            st.success(message)
                            time_module.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
                
                st.divider()
                
                # Current positions with refresh button
                st.markdown("#### 📊 Current Positions")
                
                # Add refresh button for positions
                col_refresh1, col_refresh2 = st.columns([4, 1])
                with col_refresh1:
                    st.caption("Live prices are fetched automatically. Click refresh to update now.")
                with col_refresh2:
                    if st.button("🔄 Refresh Prices", use_container_width=True, key="refresh_positions"):
                        # Clear cache to force fresh data fetch
                        st.cache_data.clear()
                        st.rerun()
                
                if portfolio:
                    positions_data = []
                    
                    # Show fetching status
                    with st.spinner("Fetching current prices..."):
                        for position in portfolio:
                            try:
                                # Try multiple methods to fetch current price
                                current_price = None
                                
                                # Method 1: Try 1 day data
                                try:
                                    current_data = fetch_stock_data_cached(
                                        position['symbol'], 
                                        period="5d", 
                                        interval="1d", 
                                        api_provider=api_provider
                                    )
                                    if not current_data.empty and 'Close' in current_data.columns:
                                        current_price = float(current_data['Close'].iloc[-1])
                                except:
                                    pass
                                
                                # Method 2: Try intraday data if daily failed
                                if current_price is None or current_price <= 0:
                                    try:
                                        current_data = fetch_stock_data_cached(
                                            position['symbol'], 
                                            period="1d", 
                                            interval="5m", 
                                            api_provider=api_provider
                                        )
                                        if not current_data.empty and 'Close' in current_data.columns:
                                            current_price = float(current_data['Close'].iloc[-1])
                                    except:
                                        pass
                                
                                # Fallback to average price if all methods fail
                                if current_price is None or current_price <= 0:
                                    # Use avg_price as fallback for current price
                                    # This is the actual price per share, not including per-share charges
                                    current_price = position['avg_price']
                                    
                            except Exception as e:
                                logging.warning(f"Error fetching price for {position['symbol']}: {e}")
                                current_price = position['avg_price']
                            
                            # Calculate P&L for this position
                            # When current price = avg price (fallback), P&L should be 0 (not negative due to charges)
                            # The charges are already accounted for in invested_amount vs avg_price
                            current_value = position['quantity'] * current_price
                            pnl = current_value - position['invested_amount']
                            pnl_percent = (pnl / position['invested_amount']) * 100 if position['invested_amount'] > 0 else 0
                            
                            # Append this position to the list
                            positions_data.append({
                                'Symbol': position['symbol'],
                                'Style': position['trading_style'].capitalize(),
                                'Qty': position['quantity'],
                                'Avg Price': f"₹{position['avg_price']:.2f}",
                                'Current Price': f"₹{current_price:.2f}",
                                'Invested': f"₹{position['invested_amount']:,.2f}",
                                'Current Value': f"₹{current_value:,.2f}",
                                'P&L': f"₹{pnl:,.2f}",
                                'P&L %': f"{pnl_percent:+.2f}%"
                            })
                    
                    positions_df = pd.DataFrame(positions_data)
                    
                    # Style the dataframe
                    def color_pnl(val):
                        if isinstance(val, str) and '%' in val:
                            num = float(val.replace('%', '').replace('+', ''))
                            if num > 0:
                                return 'background-color: #90EE90; color: black'
                            elif num < 0:
                                return 'background-color: #FFB6C6; color: black'
                        return ''
                    
                    styled_positions = positions_df.style.applymap(color_pnl, subset=['P&L %'])
                    st.dataframe(styled_positions, use_container_width=True)
                else:
                    st.info("📭 No open positions. Start trading to build your portfolio!")
                
                st.divider()
                
                # Trade history
                st.markdown("#### 📜 Recent Trades")
                
                trades = get_paper_trades_history(limit=20)
                
                if trades:
                    trades_data = []
                    
                    for trade in trades:
                        # Get charges (default to 0 if not present in old trades)
                        charges = trade.get('charges', 0)
                        
                        # Format display based on action
                        if trade['action'] == 'BUY':
                            total_display = f"₹{trade['total_amount']:,.2f}"
                            pnl_display = '-'
                            pnl_pct_display = '-'
                            charges_display = f"₹{charges:.2f}" if charges > 0 else '₹0.00'
                        else:  # SELL
                            total_display = f"₹{trade['total_amount']:,.2f}"
                            pnl = trade.get('pnl', 0)
                            pnl_pct = trade.get('pnl_percent', 0)
                            pnl_display = f"₹{pnl:,.2f}"
                            pnl_pct_display = f"{pnl_pct:+.2f}%"
                            charges_display = f"₹{charges:.2f}" if charges > 0 else '₹0.00'
                        
                        # Convert UTC timestamp to IST (UTC+5:30)
                        utc_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                        ist_offset = timedelta(hours=5, minutes=30)
                        ist_time = utc_time + ist_offset
                        
                        trades_data.append({
                            'Time': ist_time.strftime('%Y-%m-%d %H:%M'),
                            'Symbol': trade['symbol'],
                            'Action': trade['action'],
                            'Qty': trade['quantity'],
                            'Price': f"₹{trade['price']:.2f}",
                            'Total': total_display,
                            'Charges': charges_display,
                            'P&L': pnl_display,
                            'P&L %': pnl_pct_display,
                            'Style': trade['trading_style'].capitalize()
                        })
                    
                    trades_df = pd.DataFrame(trades_data)
                    
                    # Color code actions
                    def color_action(val):
                        if val == 'BUY':
                            return 'background-color: #90EE90; color: black; font-weight: bold'
                        elif val == 'SELL':
                            return 'background-color: #FFB6C6; color: black; font-weight: bold'
                        return ''
                    
                    styled_trades = trades_df.style.applymap(color_action, subset=['Action'])
                    st.dataframe(styled_trades, use_container_width=True, height=400)
                    
                    # Download button
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Trade History",
                        data=csv,
                        file_name=f"paper_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("📭 No trade history yet. Execute your first trade above!")
                
                st.divider()
                
                # Reset account
                with st.expander("⚙️ Account Settings"):
                    st.warning("**Reset Account** - This will clear all positions and reset balance")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        reset_balance = st.number_input("Initial Balance (₹)", min_value=10000, value=100000, step=10000)
                    with col2:
                        clear_history = st.checkbox("🗑️ Also clear trade history", value=False)
                    
                    if clear_history:
                        st.error("⚠️ This will permanently delete all trade records!")
                    
                    if st.button("🔄 Reset Account", type="secondary"):
                        confirm_text = "clear all positions and trade history" if clear_history else "clear all positions"
                        if st.checkbox(f"⚠️ I understand this will {confirm_text}"):
                            success, message = reset_paper_account(initial_balance=reset_balance, clear_trades=clear_history)
                            if success:
                                st.success(message)
                                time_module.sleep(1)
                                st.rerun()
                            else:
                                st.error(message)

    # --- MARKET DASHBOARD TAB ---
    with tab6:
        st.subheader("🌍 Market Overview")
        
        # Real-time Index Scanner
        if (index_scan := fetch_index_scan()):
            st.markdown("### 📊 Live Index Prices")
            
            # Separate Indian and Global indices
            indian_indices = [idx for idx in index_scan if idx['index'] in ['NIFTY 50', 'Bank NIFTY', 'Sensex', 'Small Cap', 'Finnifty']]
            global_indices = [idx for idx in index_scan if idx['index'] not in ['NIFTY 50', 'Bank NIFTY', 'Sensex', 'Small Cap', 'Finnifty']]
            
            # Display Indian Indices
            if indian_indices:
                cols = st.columns(len(indian_indices))
                for idx, col in zip(indian_indices, cols):
                    change_color = "🟢" if idx['percentage_change'] > 0 else "🔴" if idx['percentage_change'] < 0 else "⚪"
                    col.metric(
                        label=f"{change_color} {idx['index']}",
                        value=f"{idx['price']:,.2f}",
                        delta=f"{idx['percentage_change']:+.2f}% ({idx['points_change']:+.2f})"
                    )
            
            # Display Global Indices
            if global_indices:
                st.markdown("#### 🌐 Global Markets")
                cols = st.columns(len(global_indices))
                for idx, col in zip(global_indices, cols):
                    change_color = "🟢" if idx['percentage_change'] > 0 else "🔴" if idx['percentage_change'] < 0 else "⚪"
                    col.metric(
                        label=f"{change_color} {idx['index']}",
                        value=f"{idx['price']:,.2f}",
                        delta=f"{idx['percentage_change']:+.2f}% ({idx['points_change']:+.2f})"
                    )
        
        st.divider()
        
        # Index Trend Analysis
        if (index_trends := fetch_index_trend()):
            st.markdown("### 📈 Index Trend Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'nif_min15trend' in index_trends:
                    nifty_data = index_trends['nif_min15trend']
                    st.markdown("#### 📊 NIFTY 50 (15min)")
                    
                    trend = nifty_data['analysis']['15m_trend']
                    adx = nifty_data['analysis']['ADX_analysis']
                    supertrend = "Bullish 🟢" if nifty_data['indicators']['Supertrend'] == 1 else "Bearish 🔴"
                    
                    st.info(f"""
                    **Trend:** {trend}  
                    **ADX:** {adx['value']:.1f} ({adx['strength']}, {adx['direction']})  
                    **Supertrend:** {supertrend}  
                    **RSI:** {nifty_data['indicators']['RSI']:.1f}
                    """)
            
            with col2:
                if 'bnf_min15trend' in index_trends:
                    bnf_data = index_trends['bnf_min15trend']
                    st.markdown("#### 🏦 Bank NIFTY (15min)")
                    
                    trend = bnf_data['analysis']['15m_trend']
                    adx = bnf_data['analysis']['ADX_analysis']
                    supertrend = "Bullish 🟢" if bnf_data['indicators']['Supertrend'] == 1 else "Bearish 🔴"
                    
                    st.info(f"""
                    **Trend:** {trend}  
                    **ADX:** {adx['value']:.1f} ({adx['strength']}, {adx['direction']})  
                    **Supertrend:** {supertrend}  
                    **RSI:** {bnf_data['indicators']['RSI']:.1f}
                    """)
        
        st.divider()
        
        # Market Breadth
        if (breadth_data := fetch_market_breadth()):
            st.markdown("### 📊 Market Breadth")
            breadth = breadth_data.get('breadth', {})
            
            total = breadth.get('total', 1)
            advancing = breadth.get('advancing', 0)
            declining = breadth.get('declining', 0)
            unchanged = breadth.get('unchanged', 0)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Advancing", advancing, f"{(advancing/total*100):.1f}%")
            col2.metric("Declining", declining, f"{(declining/total*100):.1f}%")
            col3.metric("Unchanged", unchanged, f"{(unchanged/total*100):.1f}%")
            
            # AD Ratio
            ad_ratio = advancing / declining if declining > 0 else 0
            ad_signal = "🟢 Bullish" if ad_ratio > 1.5 else "🔴 Bearish" if ad_ratio < 0.7 else "⚪ Neutral"
            col4.metric("A/D Ratio", f"{ad_ratio:.2f}", ad_signal)
        
        st.divider()
        
        # Sector Performance
        if (sector_data := fetch_sector_performance()):
            st.markdown("### 📈 Sector Performance")
            sectors_df = pd.DataFrame(sector_data.get('data', []))
            
            if not sectors_df.empty and 'sector_index' in sectors_df.columns:
                # Select relevant columns
                display_cols = ['sector_index', 'avg_change', 'momentum', 'signal']
                if all(col in sectors_df.columns for col in display_cols):
                    display_df = sectors_df[display_cols].copy()
                    display_df.columns = ['Sector', 'Change %', 'Momentum', 'Signal']
                    
                    # Sort by change
                    display_df = display_df.sort_values('Change %', ascending=False)
                    
                    # Style the dataframe
                    def color_change(val):
                        if isinstance(val, (int, float)):
                            color = 'background-color: #90EE90' if val > 0 else 'background-color: #FFB6C6' if val < 0 else ''
                            return color
                        return ''
                    
                    styled_df = display_df.style.applymap(color_change, subset=['Change %'])
                    st.dataframe(styled_df, use_container_width=True, height=400)

if __name__ == "__main__":
    main()
