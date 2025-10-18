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
import gc
import json
import pickle
from pathlib import Path
from diskcache import Cache
from SmartApi import SmartConnect
import pyotp
import os
from dotenv import load_dotenv
from dhanhq import dhanhq # New import for Dhan

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

API_KEYS = {
    "Historical": "c3C0tMGn",
    "Trading": os.getenv("TRADING_API_KEY"),
    "Market": os.getenv("MARKET_API_KEY")
}

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
    "stocks_per_batch": 10,  # Analyze 10 stocks per batch
    "batch_delay": 5,  # 5 seconds between batches
    "min_sector_change": 0.5,  # Minimum sector change % to consider bullish
    "min_sector_advance_ratio": 60,  # Minimum advance ratio (60%)
    "cooldown_period": 300,  # 5 minutes cooldown for same stock alert
    "alert_score_threshold": 65,  # Minimum score to trigger alert
}

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

def live_scan_iteration(stock_list, timeframe, api_provider, alert_history):
    """Single iteration of live scanner"""
    results = []
    new_alerts = []
    current_time = time_module.time()
    
    # Analyze stocks in batches
    for batch_idx in range(0, len(stock_list), LIVE_SCAN_CONFIG['stocks_per_batch']):
        batch = stock_list[batch_idx:batch_idx + LIVE_SCAN_CONFIG['stocks_per_batch']]
        
        for symbol in batch:
            try:
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
                
                # Small delay between stocks
                time_module.sleep(1)
                
            except Exception as e:
                logging.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Delay between batches
        if batch_idx + LIVE_SCAN_CONFIG['stocks_per_batch'] < len(stock_list):
            time_module.sleep(LIVE_SCAN_CONFIG['batch_delay'])
    
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
        "TMB-EQ", "KTKBANK-EQ", "EQUITASBNK-EQ", "UJJIVANSFB-EQ"
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
    "Bank": [
        "HDFCBANK-EQ", "ICICIBANK-EQ", "SBIN-EQ", "KOTAKBANK-EQ", "AXISBANK-EQ",
        "INDUSINDBK-EQ", "PNB-EQ", "BANKBARODA-EQ", "CANBK-EQ", "UNIONBANK-EQ",
        "IDFCFIRSTB-EQ", "FEDERALBNK-EQ", "RBLBANK-EQ", "BANDHANBNK-EQ", "INDIANB-EQ",
        "BANKINDIA-EQ", "KARURVYSYA-EQ", "CUB-EQ", "J&KBANK-EQ", "DCBBANK-EQ",
        "AUBANK-EQ", "YESBANK-EQ", "IDBI-EQ", "SOUTHBANK-EQ", "CSBBANK-EQ",
        "TMB-EQ", "KTKBANK-EQ", "EQUITASBNK-EQ", "UJJIVANSFB-EQ"
    ],
    "IT": [
        "TCS-EQ", "INFY-EQ", "HCLTECH-EQ", "WIPRO-EQ", "TECHM-EQ", "LTIM-EQ",
        "MPHASIS-EQ", "FSL-EQ", "BSOFT-EQ", "NEWGEN-EQ", "ZENSARTECH-EQ",
        "RATEGAIN-EQ", "TANLA-EQ", "COFORGE-EQ", "PERSISTENT-EQ", "CYIENT-EQ",
        "SONATSOFTW-EQ", "KPITTECH-EQ", "TATAELXSI-EQ", "INTELLECT-EQ",
        "HAPPSTMNDS-EQ", "MASTEK-EQ", "ECLERX-EQ", "NIITLTD-EQ", "RSYSTEMS-EQ",
        "OFSS-EQ", "AURIONPRO-EQ", "DATAMATICS-EQ", "QUICKHEAL-EQ",
        "CIGNITITEC-EQ", "SAGILITY-EQ", "ALLDIGI-EQ"
    ],
    "Finance": [
        "BAJFINANCE-EQ", "BAJAJFINSV-EQ", "SHRIRAMFIN-EQ", "CHOLAFIN-EQ",
        "SBICARD-EQ", "M&MFIN-EQ", "MUTHOOTFIN-EQ", "LICHSGFIN-EQ",
        "POONAWALLA-EQ", "SUNDARMFIN-EQ", "IIFL-EQ", "ABCAPITAL-EQ",
        "LTF-EQ", "CREDITACC-EQ", "MANAPPURAM-EQ", "DHANI-EQ",
        "JMFINANCIL-EQ", "EDELWEISS-EQ", "INDIASHLTR-EQ", "MOTILALOFS-EQ",
        "CDSL-EQ", "BSE-EQ", "MCX-EQ", "ANGELONE-EQ", "PNBHOUSING-EQ",
        "HOMEFIRST-EQ", "AAVAS-EQ", "APTUS-EQ", "RECLTD-EQ", "PFC-EQ",
        "IREDA-EQ", "SMCGLOBAL-EQ", "CHOICEIN-EQ", "KFINTECH-EQ", "MASFIN-EQ",
        "SBFC-EQ", "UGROCAP-EQ", "FUSION-EQ", "PAISALO-EQ", "CAPITALSFB-EQ",
        "NSIL-EQ", "SATIN-EQ", "HDFCAMC-EQ", "UTIAMC-EQ", "ABSLAMC-EQ",
        "360ONE-EQ", "ANANDRATHI-EQ"
    ],
    "Auto": [
        "MARUTI-EQ", "BELRISE-EQ","TATAMOTORS-EQ", "M&M-EQ", "BAJAJ-AUTO-EQ", "HEROMOTOCO-EQ",
        "EICHERMOT-EQ", "TVSMOTOR-EQ", "ASHOKLEY-EQ", "MRF-EQ", "BALKRISIND-EQ",
        "APOLLOTYRE-EQ", "CEATLTD-EQ", "JKTYRE-EQ", "MOTHERSON-EQ", "BHARATFORG-EQ",
        "SUNDRMFAST-EQ", "EXIDEIND-EQ", "BOSCHLTD-EQ", "ENDURANCE-EQ",
        "UNOMINDA-EQ", "ZFCVINDIA-EQ", "GABRIEL-EQ", "SUPRAJIT-EQ",
        "LUMAXTECH-EQ", "FIEMIND-EQ", "SUBROS-EQ", "JAMNAAUTO-EQ",
        "ESCORTS-EQ", "ATULAUTO-EQ", "OLECTRA-EQ", "GREAVESCOT-EQ",
        "SMLISUZU-EQ", "VSTTILLERS-EQ", "MAHSCOOTER-EQ"
    ],
    "Pharma": [
        "SUNPHARMA-EQ", "CIPLA-EQ", "DRREDDY-EQ", "APOLLOHOSP-EQ", "LUPIN-EQ",
        "DIVISLAB-EQ", "AUROPHARMA-EQ", "ALKEM-EQ", "TORNTPHARM-EQ",
        "ZYDUSLIFE-EQ", "IPCALAB-EQ", "GLENMARK-EQ", "BIOCON-EQ",
        "ABBOTINDIA-EQ", "SANOFI-EQ", "PFIZER-EQ", "GLAXO-EQ", "NATCOPHARM-EQ",
        "AJANTPHARM-EQ", "GRANULES-EQ", "LAURUSLABS-EQ", "STAR-EQ",
        "JUBLPHARMA-EQ", "ASTRAZEN-EQ", "WOCKPHARMA-EQ", "FORTIS-EQ",
        "MAXHEALTH-EQ", "METROPOLIS-EQ", "THYROCARE-EQ", "POLYMED-EQ",
        "KIMS-EQ", "LALPATHLAB-EQ", "MEDPLUS-EQ", "ERIS-EQ", "INDOCO-EQ",
        "CAPLIPOINT-EQ", "NEULANDLAB-EQ", "SHILPAMED-EQ", "SUVEN-EQ",
        "AARTIDRUGS-EQ", "PGHL-EQ", "SYNGENE-EQ", "VINATIORGA-EQ", "GLAND-EQ",
        "JBCHEPHARM-EQ", "HCG-EQ", "RAINBOW-EQ", "ASTERDM-EQ", "VIJAYA-EQ",
        "MEDANTA-EQ", "BLISSGVS-EQ", "MOREPENLAB-EQ", "RPGLIFE-EQ"
    ],
    "Metals": [
        "TATASTEEL-EQ", "JSWSTEEL-EQ", "HINDALCO-EQ", "VEDL-EQ", "SAIL-EQ",
        "NMDC-EQ", "HINDZINC-EQ", "NATIONALUM-EQ", "JINDALSTEL-EQ", "MOIL-EQ",
        "APLAPOLLO-EQ", "RATNAMANI-EQ", "JSL-EQ", "WELCORP-EQ",
        "SHYAMMETL-EQ", "MIDHANI-EQ", "GRAVITA-EQ", "SARDAEN-EQ",
        "ASHAPURMIN-EQ", "COALINDIA-EQ"
    ],
    "FMCG": [
        "HINDUNILVR-EQ", "ITC-EQ", "NESTLEIND-EQ", "BRITANNIA-EQ", "DABUR-EQ",
        "GODREJCP-EQ", "COLPAL-EQ", "MARICO-EQ", "PGHH-EQ", "EMAMILTD-EQ",
        "GILLETTE-EQ", "HATSUN-EQ", "JYOTHYLAB-EQ", "BAJAJCON-EQ", "RADICO-EQ",
        "TATACONSUM-EQ", "UNITDSPR-EQ", "CCL-EQ", "AVANTIFEED-EQ", "BIKAJI-EQ",
        "VBL-EQ", "DOMS-EQ", "GODREJAGRO-EQ", "VENKEYS-EQ", "BECTORFOOD-EQ",
        "KRBL-EQ"
    ],
    "Power": [
        "NTPC-EQ", "POWERGRID-EQ", "ADANIPOWER-EQ", "TATAPOWER-EQ",
        "JSWENERGY-EQ", "NHPC-EQ", "SJVN-EQ", "TORNTPOWER-EQ", "CESC-EQ",
        "ADANIENSOL-EQ", "INDIGRID-EQ", "SUZLON-EQ", "BHEL-EQ", "THERMAX-EQ"
    ],
    "Capital Goods": [
        "LT-EQ", "SIEMENS-EQ", "ABB-EQ", "BEL-EQ", "HAL-EQ", "CUMMINSIND-EQ",
        "AIAENG-EQ", "SKFINDIA-EQ", "GRINDWELL-EQ", "TIMKEN-EQ",
        "KSB-EQ", "ELGIEQUIP-EQ", "VOLTAS-EQ", "BLUESTARCO-EQ",
        "HAVELLS-EQ", "DIXON-EQ", "POLYCAB-EQ", "RRKABEL-EQ", "CGPOWER-EQ"
    ],
    "Oil & Gas": [
        "RELIANCE-EQ", "ONGC-EQ", "IOC-EQ", "BPCL-EQ", "HINDPETRO-EQ",
        "GAIL-EQ", "PETRONET-EQ", "OIL-EQ", "IGL-EQ", "MGL-EQ",
        "GUJGASLTD-EQ", "GSPL-EQ"
    ],
    "Chemicals": [
        "PIDILITIND-EQ", "SRF-EQ", "DEEPAKNTR-EQ", "ATUL-EQ", "AARTIIND-EQ",
        "NAVINFLUOR-EQ", "FINEORG-EQ", "ALKYLAMINE-EQ", "BALAMINES-EQ",
        "GUJFLUORO-EQ", "JUBLINGREA-EQ", "GALAXYSURF-EQ", "PCBL-EQ",
        "NOCIL-EQ", "BASF-EQ", "SUDARSCHEM-EQ", "TATACHEM-EQ",
        "COROMANDEL-EQ", "UPL-EQ", "PIIND-EQ", "RALLIS-EQ"
    ],
    "Telecom": [
        "BHARTIARTL-EQ", "IDEA-EQ", "INDUSTOWER-EQ", "TATACOMM-EQ",
        "HFCL-EQ", "TEJASNET-EQ", "STLTECH-EQ", "ITI-EQ"
    ],
    "Infrastructure": [
        "LT-EQ", "IRB-EQ", "NBCC-EQ", "RVNL-EQ", "KEC-EQ",
        "PNCINFRA-EQ", "KNRCON-EQ", "GRINFRA-EQ", "NCC-EQ",
        "HGINFRA-EQ", "ASHOKA-EQ", "IRCON-EQ"
    ],
    "Insurance": [
        "SBILIFE-EQ", "HDFCLIFE-EQ", "ICICIGI-EQ", "ICICIPRULI-EQ",
        "LICI-EQ", "GICRE-EQ", "NIACL-EQ", "STARHEALTH-EQ"
    ],
    "Cement": [
        "ULTRACEMCO-EQ", "SHREECEM-EQ", "AMBUJACEM-EQ", "ACC-EQ",
        "JKCEMENT-EQ", "DALBHARAT-EQ", "RAMCOCEM-EQ", "NUVOCO-EQ",
        "JKLAKSHMI-EQ", "INDIACEM-EQ"
    ],
    "Realty": [
        "DLF-EQ", "GODREJPROP-EQ", "OBEROIRLTY-EQ", "PHOENIXLTD-EQ",
        "PRESTIGE-EQ", "BRIGADE-EQ", "SOBHA-EQ", "SUNTECK-EQ"
    ],
    "Aviation": [
        "INDIGO-EQ", "SPICEJET-EQ"
    ],
    "Retail": [
        "DMART-EQ", "TRENT-EQ", "ABFRL-EQ", "VMART-EQ", "SHOPERSTOP-EQ",
        "BATAINDIA-EQ", "METROBRAND-EQ", "ZOMATO-EQ", "NYKAA-EQ", "TITAN-EQ"
    ],
    "Media": [
        "ZEEL-EQ", "SUNTV-EQ", "TVTODAY-EQ", "DISHTV-EQ",
        "PVR-EQ", "INOXLEISUR-EQ", "SAREGAMA-EQ", "TIPS-EQ"
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
    """Fetches stock data from SmartAPI."""
    if "-EQ" not in symbol:
        symbol = f"{symbol.split('.')[0]}-EQ"
    
    smart_api = get_global_smart_api()
    if not smart_api:
        raise ValueError("SmartAPI session unavailable")
    
    end_date = datetime.now()
    period_map = {"2y": 730, "1y": 365, "6mo": 180, "1mo": 30, "1d": 1}
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
        logging.warning(f"[SmartAPI] Token not found for {symbol}")
        return pd.DataFrame()
    
    historical_data = smart_api.getCandleData({
        "exchange": "NSE", 
        "symboltoken": symboltoken, 
        "interval": api_interval,
        "fromdate": start_date.strftime("%Y-%m-%d %H:%M"), 
        "todate": end_date.strftime("%Y-%m-%d %H:%M")
    })
    
    if not historical_data or not historical_data.get('status') or not historical_data.get('data'):
        error_msg = historical_data.get('message', 'No data') if historical_data else 'No response'
        logging.warning(f"[SmartAPI] API error for {symbol}: {error_msg}")
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
# BACKTESTING
# ============================================================================

def backtest_strategy(data, symbol, trading_style='swing', timeframe='1d', initial_capital=30000, contrarian_mode=False):
    """Backtest strategy with realistic costs"""
    results = {"total_return": 0, "annual_return": 0, "sharpe_ratio": 0, "max_drawdown": 0, "trades": 0, "win_rate": 0, "trades_list": [], "equity_curve": []}
    if len(data) < 200: return results
    
    BROKERAGE, STT, SLIPPAGE = 0.0003, 0.001, 0.0005
    cash, position, entry_price, qty, trades, returns = initial_capital, None, 0, 0, [], []
    
    for i in range(200, len(data)):
        sliced = data.iloc[:i+1]
        try:
            rec = generate_recommendation(sliced, symbol, trading_style, timeframe, cash, contrarian_mode)
            current_price, current_date = data['Close'].iloc[i], data.index[i]
            
            if position and rec['signal'] in ['Sell', 'Strong Sell']:
                exit_price = current_price * (1 - BROKERAGE - STT - SLIPPAGE)
                pnl = (exit_price - entry_price) * qty
                cash += (current_price * qty * (1 - BROKERAGE - STT))
                returns.append(pnl / (entry_price * qty))
                trades.append({"entry_date": entry_date, "exit_date": current_date, "pnl": pnl})
                position = None
            
            if not position and rec['signal'] in ['Buy', 'Strong Buy']:
                entry_price, entry_date = current_price * (1 + BROKERAGE + SLIPPAGE), current_date
                qty = rec['position_size']
                cash -= qty * current_price * (1 + BROKERAGE)
                position = "Long"
            
            equity = cash + (qty * current_price if position else 0)
            results['equity_curve'].append((current_date, equity))
        except Exception: continue
    
    if trades:
        results['trades'] = len(trades)
        results['total_return'] = ((cash + (qty * data['Close'].iloc[-1] if position else 0) - initial_capital) / initial_capital) * 100
        results['win_rate'] = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100
        if returns:
            periods = {'5m': 252*75, '15m': 252*25, '1h': 252*6, '1d': 252}.get(timeframe, 252)
            results['sharpe_ratio'] = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(periods)
            results['annual_return'] = np.mean(returns) * periods * 100
    
    if results['equity_curve']:
        equity_df = pd.DataFrame(results['equity_curve'], columns=['Date', 'Equity'])
        equity_df['Peak'] = equity_df['Equity'].cummax()
        equity_df['DD'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
        results['max_drawdown'] = equity_df['DD'].min() * 100
    
    return results

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

def display_intraday_chart(rec, data):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'))
    if 'VWAP' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP', line=dict(color='blue', width=2)))
    if rec.get('or_high'):
        fig.add_hline(y=rec['or_high'], line_dash="dot", annotation_text="OR High", line_color="green")
        fig.add_hline(y=rec['or_low'], line_dash="dot", annotation_text="OR Low", line_color="red")
    fig.add_hline(y=rec['buy_at'], annotation_text="Entry", line_color="white")
    fig.add_hline(y=rec['stop_loss'], line_dash="dash", annotation_text="Stop", line_color="red")
    fig.add_hline(y=rec['target'], line_dash="dash", annotation_text="Target", line_color="green")
    fig.update_layout(title=f"{rec['symbol']} - {rec['timeframe']} Intraday", height=600, xaxis_rangeslider_visible=False)
    return fig

def main():
    init_database()
    st.set_page_config(page_title="StockGenie Pro", layout="wide")
    st.title("📊 StockGenie Pro V2.9 - Multi-API Analysis")
    st.caption("✨ FIX: Adapted to new Dhan master file column names & improved API health checks.")
    st.subheader(f"📅 {datetime.now().strftime('%d %b %Y, %A')}")
    
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

    # --- SESSION STATE & TABS ---
    if 'scan_running' not in st.session_state: st.session_state.scan_running = False
    if 'scan_results' not in st.session_state: st.session_state.scan_results = None
    if 'scan_params' not in st.session_state: st.session_state.scan_params = {}

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Analysis", "🔍 Scanner", "🎯 Technical Screener", "🔄 Live Intraday", "📊 Backtest", "📜 History", "🌍 Market Dashboard"])


    # --- ANALYSIS TAB ---
    with tab1:
        if st.button("🔍 Analyze Selected Stock"):
            with st.spinner(f"Analyzing {symbol} using {api_provider}..."):
                try:
                    data = fetch_stock_data_cached(symbol, interval=timeframe, api_provider=api_provider)
                    if not data.empty:
                        rec = generate_recommendation(data, symbol, 'swing' if trading_style == "Swing Trading" else 'intraday', timeframe, account_size, contrarian_mode)
                        st.subheader(f"{rec['symbol']} ({rec['trading_style']} - {rec['timeframe']})")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Score", f"{rec['score']}/100")
                        col2.metric("Signal", rec['signal'])
                        col3.metric("Regime", rec['regime'])
                        col4.metric("Current Price", f"₹{rec['current_price']}")
                        st.info(f"**Reason**: {rec['reason']}")
                        
                        fig = display_intraday_chart(rec, data) if trading_style == "Intraday Trading" else go.Figure(go.Candlestick(x=rec['processed_data'].index, open=rec['processed_data']['Open'], high=rec['processed_data']['High'], low=rec['processed_data']['Low'], close=rec['processed_data']['Close']))
                        st.plotly_chart(fig, use_container_width=True)
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
                st.rerun()
            except Exception as e:
                st.session_state.scan_running = False
                st.error(f"❌ Scan failed: {e}")

        if st.session_state.scan_results is not None:
            results = st.session_state.scan_results
            if not results.empty:
                save_picks(results, trading_style)
                st.subheader(f"🏆 Top {trading_style} Picks")
                st.dataframe(results.style.applymap(lambda v: 'background-color: #90EE90' if v >= 75 else 'background-color: #FFFACD' if v >= 60 else '', subset=['Score']), use_container_width=True)
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
        st.caption("🎯 Automatically scans stocks from bullish and neutral sectors")
        
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
        
        # Configuration
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
        st.markdown("#### 📊 Current Market Sectors")
        bullish_sectors = get_bullish_sectors()
        neutral_sectors = get_neutral_sectors()
        
        # Display bullish sectors
        if bullish_sectors:
            st.markdown("**🟢 Bullish Sectors:**")
            sector_cols = st.columns(min(len(bullish_sectors), 4))
            for idx, sector in enumerate(bullish_sectors[:4]):
                with sector_cols[idx]:
                    st.metric(
                        label=f"🟢 {sector['sector']}",
                        value=f"{sector['change']:+.2f}%",
                        delta=f"{sector['advance_ratio']:.0f}% advancing"
                    )
        
        # Display neutral sectors
        if neutral_sectors:
            st.markdown("**🟡 Neutral Sectors:**")
            neutral_cols = st.columns(min(len(neutral_sectors), 4))
            for idx, sector in enumerate(neutral_sectors[:4]):
                with neutral_cols[idx]:
                    st.metric(
                        label=f"🟡 {sector['sector']}",
                        value=f"{sector['change']:+.2f}%",
                        delta=f"{sector['advance_ratio']:.0f}% advancing"
                    )
        
        # Expandable detailed view
        if bullish_sectors or neutral_sectors:
            with st.expander(f"📋 View All Sectors ({len(bullish_sectors)} bullish, {len(neutral_sectors)} neutral)"):
                if bullish_sectors:
                    st.markdown("**🟢 Bullish Sectors:**")
                    for sector in bullish_sectors:
                        st.write(f"• **{sector['sector']}**: {sector['change']:+.2f}% | "
                                f"{sector['advancing']}/{sector['total']} advancing ({sector['advance_ratio']:.1f}%)")
                
                if neutral_sectors:
                    st.markdown("**🟡 Neutral Sectors:**")
                    for sector in neutral_sectors:
                        st.write(f"• **{sector['sector']}**: {sector['change']:+.2f}% | "
                                f"{sector['advancing']}/{sector['total']} advancing ({sector['advance_ratio']:.1f}%)")
        else:
            st.warning("⚠️ No suitable sectors found at the moment")
        
        st.divider()
        
        # Control buttons
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if not st.session_state.live_scan_active:
                if st.button("🚀 Start Live Scanner", type="primary", use_container_width=True):
                    if not bullish_sectors and not neutral_sectors:
                        st.error("❌ Cannot start: No suitable sectors found")
                    else:
                        st.session_state.live_scan_active = True
                        st.session_state.scan_iteration = 0
                        st.session_state.live_scan_alerts = []
                        st.rerun()
            else:
                if st.button("⏹️ Stop Scanner", type="secondary", use_container_width=True):
                    st.session_state.live_scan_active = False
                    st.rerun()
        
        with col2:
            if st.button("🔄 Manual Scan Now", use_container_width=True):
                if not bullish_sectors and not neutral_sectors:
                    st.error("❌ No suitable sectors available")
                else:
                    with st.spinner("Scanning bullish and neutral sectors..."):
                        # Combine bullish and neutral sectors
                        combined_sectors = bullish_sectors + neutral_sectors
                        stock_list = get_stocks_from_bullish_sectors(combined_sectors)
                        if stock_list:
                            results, alerts = live_scan_iteration(
                                stock_list[:50],  # Limit to 50 stocks for manual scan
                                scan_timeframe,
                                api_provider,
                                st.session_state.alert_history
                            )
                            st.session_state.live_scan_results = results
                            st.session_state.live_scan_alerts.extend(alerts)
                            st.session_state.last_scan_time = datetime.now()
                            st.success(f"✅ Scanned {len(stock_list)} stocks from bullish sectors")
                            st.rerun()
        
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
            
            # Check if it's time to scan
            should_scan = False
            if st.session_state.last_scan_time is None:
                should_scan = True
            else:
                time_since_last = (datetime.now() - st.session_state.last_scan_time).total_seconds()
                if time_since_last >= scan_interval:
                    should_scan = True
            
            if should_scan:
                with st.spinner(f"🔍 Running scan iteration #{st.session_state.scan_iteration + 1}..."):
                    # Get fresh bullish and neutral sectors
                    current_bullish = get_bullish_sectors()
                    current_neutral = get_neutral_sectors()
                    
                    if current_bullish or current_neutral:
                        # Combine both sector types
                        combined_sectors = current_bullish + current_neutral
                        stock_list = get_stocks_from_bullish_sectors(combined_sectors)
                        
                        if stock_list:
                            results, alerts = live_scan_iteration(
                                stock_list,
                                scan_timeframe,
                                api_provider,
                                st.session_state.alert_history
                            )
                            
                            st.session_state.live_scan_results = results
                            st.session_state.live_scan_alerts.extend(alerts)
                            st.session_state.last_scan_time = datetime.now()
                            st.session_state.scan_iteration += 1
                            
                            if alerts:
                                st.toast(f"🚨 {len(alerts)} new alerts!", icon="🚨")
                    
                    time_module.sleep(2)
                    st.rerun()
            else:
                # Auto-refresh every 5 seconds to update countdown
                time_module.sleep(5)
                st.rerun()
        
        # Display alerts
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
        
        st.divider()
        
        # Display all results
        if st.session_state.live_scan_results:
            st.markdown("### 📊 Current Scan Results")
            
            results_df = pd.DataFrame(st.session_state.live_scan_results)
            
            # Filter and sort
            results_df = results_df[results_df['Signal'].str.contains('Buy', na=False)]
            results_df = results_df.sort_values('Score', ascending=False).head(20)
            
            if not results_df.empty:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Opportunities", len(results_df))
                col2.metric("Avg Score", f"{results_df['Score'].mean():.1f}")
                col3.metric("Strong Buys", len(results_df[results_df['Signal'] == 'Strong Buy']))
                
                # Style the results
                def color_score(val):
                    if val >= 75:
                        return 'background-color: #90EE90; font-weight: bold'
                    elif val >= 70:
                        return 'background-color: #FFFACD'
                    elif val >= 65:
                        return 'background-color: #E0E0E0'
                    return ''
                
                styled_results = results_df.style.applymap(color_score, subset=['Score'])
                st.dataframe(styled_results, use_container_width=True, height=500)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"live_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("ℹ️ No buy signals in current scan")
            
            # Display scan stats
            if st.session_state.last_scan_time:
                st.caption(f"Last scan: {st.session_state.last_scan_time.strftime('%H:%M:%S')} | "
                          f"Iteration: #{st.session_state.scan_iteration} | "
                          f"Total alerts: {len(st.session_state.live_scan_alerts)}")


    with tab5:
        if st.button("📊 Run Backtest"):
            with st.spinner("Backtesting..."):
                try:
                    data = fetch_stock_data_cached(symbol, period="2y", interval=timeframe, api_provider=api_provider)
                    if not data.empty:
                        results = backtest_strategy(data, symbol, 'swing' if trading_style == "Swing Trading" else 'intraday', timeframe, account_size, contrarian_mode)
                        st.success("✅ Backtest complete")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Return", f"{results['total_return']:.2f}%")
                        col2.metric("Annual Return", f"{results['annual_return']:.2f}%")
                        col3.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                        col4.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
                    else: st.warning("Insufficient data for backtest")
                except Exception as e: st.error(f"❌ Backtest error: {e}")

    # --- HISTORY & MARKET DASHBOARD TABS (UNCHANGED) ---
    with tab6:
        try:
            conn = sqlite3.connect('stock_picks.db')
            history = pd.read_sql_query("SELECT * FROM picks ORDER BY date DESC LIMIT 100", conn)
            conn.close()
            st.dataframe(history, use_container_width=True)
        except Exception as e: st.error(f"❌ Database error: {e}")
        
    with tab7:
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
