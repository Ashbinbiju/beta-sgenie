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
    
    # Skip check if we just performed a manual update
    if st.session_state.get('skip_next_update_check', False):
        st.session_state.skip_next_update_check = False
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
        momentum_sign