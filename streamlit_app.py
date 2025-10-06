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

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
SESSION_EXPIRY = 1800  # 30 minutes for safety

cache = Cache("stock_data_cache")

# Scan configuration
SCAN_CONFIG = {
    "batch_size": 5,
    "delay_within_batch": 2,
    "delay_between_batches": 5,
    "session_refresh_interval": 20,
    "max_stocks_per_scan": 1000
}
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
# FIXED SECTORS (NO DUPLICATES)
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
        "QUICKHEAL-EQ", "CIGNITITEC-EQ", "SAGILITY-EQ", "ALLDIGI-EQ"
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
        "Diversified", "Aviation", "Retail", "Media"
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

def save_checkpoint(completed_stocks, all_results, failed_stocks, trading_style, timeframe):
    """Save scan checkpoint to disk"""
    try:
        checkpoint = {
            'completed_stocks': completed_stocks,
            'all_results': all_results,
            'failed_stocks': failed_stocks,
            'trading_style': trading_style,
            'timeframe': timeframe,
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
# API & DATA FETCHING (ROBUST VERSION)
# ============================================================================

def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()

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
    """Load instrument token mapping"""
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        return {entry["symbol"]: entry["token"] for entry in data if "symbol" in entry and "token" in entry}
    except Exception as e:
        logging.warning(f"Failed to load instrument list: {str(e)}")
        return {}

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
def fetch_stock_data_with_auth(symbol, period="1y", interval="1d"):
    """Fetch stock data from SmartAPI with robust error handling"""
    cache_key = f"{symbol}_{period}_{interval}"
    
    # Try cache first
    try:
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return pd.read_pickle(io.BytesIO(cached_data))
    except Exception as e:
        logging.warning(f"Cache read error for {symbol}: {str(e)}")
    
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
        
        if not historical_data or not historical_data.get('status'):
            error_msg = historical_data.get('message', 'Unknown error') if historical_data else 'No response'
            logging.warning(f"API error for {symbol}: {error_msg}")
            return pd.DataFrame()
        
        if not historical_data.get('data'):
            logging.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        data = pd.DataFrame(
            historical_data['data'],
            columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # Cache with error handling
        try:
            if interval == "1d":
                expire = 86400
            else:
                expire = 300
            
            buffer = io.BytesIO()
            data.to_pickle(buffer)
            cache.set(cache_key, buffer.getvalue(), expire=expire)
        except Exception as e:
            logging.warning(f"Cache write error for {symbol}: {str(e)}")
        
        return data
    
    except Exception as e:
        logging.error(f"Error fetching {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_stock_data_cached(symbol, period="1y", interval="1d"):
    """Wrapper for stock data fetching"""
    return fetch_stock_data_with_auth(symbol, period, interval)

def check_api_health():
    """Verify API session is working"""
    try:
        smart_api = get_global_smart_api()
        if not smart_api:
            return False, "Session not initialized"
        
        test_symbol = "SBIN-EQ"
        symbol_token_map = load_symbol_token_map()
        token = symbol_token_map.get(test_symbol)
        
        if not token:
            return False, "Symbol map not loaded"
        
        return True, "API healthy"
    
    except Exception as e:
        return False, str(e)

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
    """Opening Range Breakout Strategy
    
    Note: Does not use contrarian_mode as it doesn't apply market context.
    Market context adjustments are applied at the parent score level.
    """
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
    """VWAP Mean Reversion Strategy
    
    Note: Does not use contrarian_mode as it doesn't apply market context.
    Market context adjustments are applied at the parent score level.
    """
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
    """Intraday Trend Following
    
    Note: Does not use contrarian_mode as it doesn't apply market context.
    Market context adjustments are applied at the parent score level.
    """
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

def calculate_intraday_score(df, symbol=None, timeframe='15m', contrarian_mode=False):
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
    
    # Market context adjustments (CORRECTLY IMPLEMENTED)
    confidence_adjustment = 0
    
    if symbol:
        signal_direction = 'bullish' if raw_score > 0 else 'bearish'
        
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
    
    # Apply confidence adjustment
    final_score = raw_score + confidence_adjustment
    
    # Time modifiers
    if prime_hours:
        final_score *= 1.2
    elif lunch_hours:
        final_score *= 0.7
    
    # CORRECTED Normalization with proper range mapping
    # Opening Range: ±13 max
    # VWAP Mean Reversion: ±11 max  
    # Trend: ±13 max
    # Context: ±6 (normal) or ±3 (contrarian)
    # Prime time multiplier: 1.2x
    # Actual max: 13 (strategy) + 6 (context) = 19 × 1.2 = 22.8
    max_expected = 23 if not contrarian_mode else 20
    
    if final_score >= 0:
        # Map [0, max_expected] → [50, 90] (slightly less conservative for intraday)
        normalized = 50 + (final_score / max(max_expected, 1)) * 40
    else:
        # Map [-max_expected, 0] → [10, 50] (aggressive warning)
        normalized = 50 + (final_score / max(max_expected, 1)) * 40
    
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
    
    # Build reason (WITHOUT contrarian clutter)
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
    
    # Add market context (CLEAN, no weight mention)
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
        "contrarian_mode": contrarian_mode,
        "processed_data": df, 
        **position
    }

# ============================================================================
# BACKTESTING
# ============================================================================

def backtest_strategy(data, symbol, trading_style='swing', timeframe='1d', initial_capital=30000, contrarian_mode=False):
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
            rec = generate_recommendation(sliced, symbol, trading_style, timeframe, cash, contrarian_mode)
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
# BATCH ANALYSIS WITH ROBUST ERROR HANDLING
# ============================================================================

def analyze_stock_batch(symbol, trading_style='swing', timeframe='1d', contrarian_mode=False, max_retries=3):
    """Analyze single stock with comprehensive error handling"""
    
    for attempt in range(max_retries):
        try:
            data = fetch_stock_data_cached(symbol, interval=timeframe)
            
            if data.empty:
                logging.warning(f"{symbol}: No data available")
                return None
            
            if len(data) < 200:
                logging.warning(f"{symbol}: Insufficient data ({len(data)} bars)")
                return None
            
            rec = generate_recommendation(data, symbol, trading_style, timeframe, contrarian_mode=contrarian_mode)
            
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
        
        except KeyError as e:
            logging.error(f"{symbol}: Missing key {str(e)}")
            return None
        
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"{symbol}: Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time_module.sleep(2 * (attempt + 1))
                continue
            else:
                logging.error(f"{symbol}: All attempts failed: {str(e)}")
                return None
    
    return None

def analyze_multiple_stocks(stock_list, trading_style='swing', timeframe='1d', progress_callback=None, resume=False, contrarian_mode=False):
    """
    Analyze multiple stocks with:
    - Auto-resume capability
    - Batch processing
    - Session refresh
    - Error recovery
    - Memory management
    """
    all_results = []
    failed_stocks = []
    completed_stocks = []
    
    # STEP 1: Remove duplicates from stock_list FIRST
    stock_list = list(dict.fromkeys(stock_list))  # Preserves order, removes duplicates
    
    # Try to load checkpoint if resume is enabled
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint()
        
        if checkpoint:
            # Verify checkpoint matches current scan parameters
            if (checkpoint['trading_style'] == trading_style and 
                checkpoint['timeframe'] == timeframe):
                
                all_results = checkpoint['all_results']
                failed_stocks = checkpoint['failed_stocks']
                completed_stocks = checkpoint['completed_stocks']
                
                logging.info(f"Resuming scan from checkpoint with {len(completed_stocks)} completed stocks")
            else:
                logging.warning("Checkpoint parameters don't match. Starting fresh scan.")
                clear_checkpoint()
    
    batch_size = SCAN_CONFIG["batch_size"]
    
    # STEP 2: Calculate total properly
    total_stocks = min(len(stock_list), SCAN_CONFIG["max_stocks_per_scan"])
    
    # STEP 3: Filter out already completed stocks
    remaining_stocks = [s for s in stock_list if s not in completed_stocks]
    remaining_stocks = remaining_stocks[:total_stocks - len(completed_stocks)]
    
    if not remaining_stocks:
        logging.info("All stocks already processed")
        if all_results:
            return pd.DataFrame(all_results)
        return pd.DataFrame()
    
    logging.info(f"Starting scan of {len(remaining_stocks)} remaining stocks (Total: {total_stocks})")
    
    try:
        for batch_idx in range(0, len(remaining_stocks), batch_size):
            batch = remaining_stocks[batch_idx:batch_idx + batch_size]
            
            # Refresh session periodically
            if batch_idx > 0 and batch_idx % SCAN_CONFIG["session_refresh_interval"] == 0:
                logging.info(f"Refreshing API session at stock {batch_idx + len(completed_stocks)}/{total_stocks}")
                global _global_smart_api, _session_timestamp
                _global_smart_api = None
                time_module.sleep(3)
            
            for i, symbol in enumerate(batch):
                # FIXED PROGRESS CALCULATION
                current_count = len(completed_stocks) + 1  # Number of stocks completed (including current)
                progress_value = min(current_count / total_stocks, 1.0)  # CAP AT 1.0
                
                try:
                    # Update progress with SAFE value
                    if progress_callback:
                        progress_callback(progress_value)
                    
                    logging.info(f"Processing {current_count}/{total_stocks}: {symbol}")
                    
                    result = analyze_stock_batch(symbol, trading_style, timeframe, contrarian_mode, max_retries=3)
                    
                    if result:
                        # Add sector information
                        result['Sector'] = assign_primary_sector(symbol, SECTORS)
                        all_results.append(result)
                    else:
                        failed_stocks.append(symbol)
                    
                    # Mark as completed
                    completed_stocks.append(symbol)
                    
                    # Save checkpoint every 5 stocks
                    if len(completed_stocks) % 5 == 0:
                        save_checkpoint(completed_stocks, all_results, failed_stocks, trading_style, timeframe)
                    
                except KeyboardInterrupt:
                    logging.warning("Scan interrupted by user")
                    save_checkpoint(completed_stocks, all_results, failed_stocks, trading_style, timeframe)
                    raise
                
                except Exception as e:
                    logging.error(f"Critical error analyzing {symbol}: {str(e)}")
                    failed_stocks.append(symbol)
                    completed_stocks.append(symbol)  # Mark as completed to skip on resume
                    
                    # Save checkpoint on error
                    save_checkpoint(completed_stocks, all_results, failed_stocks, trading_style, timeframe)
                    continue
                
                # Adaptive delay
                if i < len(batch) - 1:
                    time_module.sleep(SCAN_CONFIG["delay_within_batch"])
                else:
                    time_module.sleep(SCAN_CONFIG["delay_between_batches"])
            
            # Memory cleanup every batch
            if batch_idx % 10 == 0:
                cleanup_memory()
        
        # Successful completion - clear checkpoint
        clear_checkpoint()
        
    except KeyboardInterrupt:
        # Don't clear checkpoint on interruption
        logging.info("Scan interrupted - checkpoint preserved")
        raise
    
    except Exception as e:
        logging.error(f"Fatal scan error: {str(e)}")
        # Preserve checkpoint for recovery
        save_checkpoint(completed_stocks, all_results, failed_stocks, trading_style, timeframe)
        raise
    
    # Log summary
    logging.info(f"Scan complete: {len(all_results)} successful, {len(failed_stocks)} failed")
    if failed_stocks:
        logging.warning(f"Failed stocks: {', '.join(failed_stocks[:10])}")
    
    if not all_results:
        logging.warning("No results found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    # Filter based on trading style
    if trading_style == 'intraday':
        df = df[df['Signal'].str.contains('Buy', na=False)]
    else:
        df = df[(df['Signal'].str.contains('Buy', na=False)) | (df['Score'] >= 60)]
    
    if df.empty:
        logging.warning("No stocks passed filters")
        return df
    
    # Ensure sector diversity
    diverse_results = []
    
    if 'Sector' in df.columns:
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
    st.title("📊 StockGenie Pro V2.5 - Professional NSE Analysis")
    st.caption("✨ FIXED: Market context weight reduced + Asymmetric scoring + Contrarian mode + Auto-resume")
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
    
    # Contrarian Mode Toggle
    st.sidebar.divider()
    contrarian_mode = st.sidebar.checkbox(
        "🎯 Contrarian Mode",
        value=False,
        help="Reduce market context penalties. Useful for finding counter-trend opportunities."
    )
    
    if contrarian_mode:
        st.sidebar.info("⚠️ Market context weight reduced by 50%. Stock technicals prioritized.")
    
    st.sidebar.divider()
    
    sector_options = ["All"] + list(SECTORS.keys())
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        sector_options,
        default=["All"],
        help="Choose sectors"
    )
    
    # MOVE THIS BLOCK HERE - AFTER selected_sectors IS DEFINED
    stock_list = get_stock_list_from_sectors(SECTORS, selected_sectors)

    if stock_list and len(stock_list) > 0:
        symbol = st.sidebar.selectbox(
            "Select Stock", 
            stock_list, 
            index=0,
            help="Choose a stock to analyze"
        )
    else:
        st.sidebar.error("❌ No stocks available for selected sectors")
        symbol = "SBIN-EQ"  # Fallback
        st.sidebar.info(f"Using default: {symbol}")

    account_size = st.sidebar.number_input(
        "Account Size (₹)", 
        min_value=10000, 
        max_value=10000000, 
        value=30000, 
        step=5000
    )
    
# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Analysis", "🔍 Scanner", "📊 Backtest", "📜 History", "🌍 Market Dashboard"]
)

# TAB 1: Analysis
with tab1:
    # Validate symbol selection
    if symbol is None or symbol == "":
        st.warning("⚠️ Please select a valid stock from the sidebar")
        st.stop()
    
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
                        timeframe, account_size, contrarian_mode
                    )
                    processed_data = rec.get('processed_data', data)
                    
                    # Display Contrarian Mode Status
                    if contrarian_mode:
                        st.info("🎯 **Contrarian Mode Active**: Market context weight reduced by 50%. Stock technicals prioritized.")
                    
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

    
# TAB 2: Scanner (IMPROVED WITH AUTO-RESUME)
with tab2:
    st.markdown("### 📡 Stock Scanner")
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"🔄 Found incomplete scan: {len(checkpoint['completed_stocks'])} stocks already processed")
        with col2:
            if st.button("🗑️ Clear & Start Fresh"):
                clear_checkpoint()
                st.rerun()
    
    # Check API health before scan
    health_status, health_msg = check_api_health()
    if not health_status:
        st.warning(f"⚠️ API Issue: {health_msg}. Will auto-reconnect on next request...")
    
    with st.expander("📋 Scan Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Stocks to scan**: {min(len(stock_list), SCAN_CONFIG['max_stocks_per_scan'])}")
            st.info(f"**Estimated time**: ~{min(len(stock_list), SCAN_CONFIG['max_stocks_per_scan']) * 3} seconds")
        with col2:
            st.info(f"**Trading style**: {trading_style}")
            st.info(f"**Timeframe**: {timeframe_display}")
            # Resume Checkbox
            resume = st.checkbox(
                "Resume from checkpoint", 
                value=True,
                help="Continue previous scan if interrupted"
            )
            if resume and CHECKPOINT_FILE.exists():
                st.success("✅ Checkpoint available")
    
    # Display Contrarian Mode Status
    if contrarian_mode:
        st.info("🎯 **Contrarian Mode Active**: Scanning with reduced market context weight")

    # Validate stock list
    if not stock_list or len(stock_list) == 0:
        st.error("❌ No stocks available to scan. Please select different sectors.")
        st.stop()
    st.info(f"✅ Ready to scan {len(stock_list)} unique stocks")
    
    # Resume or Start Fresh
    col1, col2 = st.columns(2)
    
    with col1:
        scan_button = st.button("🚀 Start Scan", type="primary", use_container_width=True)
    
    with col2:
        resume_enabled = checkpoint is not None and resume
        resume_button = st.button(
            f"▶️ Resume Scan ({len(checkpoint['completed_stocks'])} done)" if checkpoint else "▶️ Resume Scan (No checkpoint)",
            disabled=not resume_enabled,
            use_container_width=True
        )
    
    if scan_button or resume_button:
        resume_scan = resume_button and checkpoint is not None
        
        progress = st.progress(0)
        status_text = st.empty()
        results_placeholder = st.empty()
        
        try:
            if resume_scan:
                status_text.info(f"🔄 Resuming scan from {len(checkpoint['completed_stocks'])} stocks...")
            else:
                status_text.info("🔄 Initializing fresh scan...")
                clear_checkpoint()  # Clear old checkpoint for fresh scan
            
            def update_progress(pct):
                progress.progress(pct)
                scan_count = min(len(stock_list), SCAN_CONFIG['max_stocks_per_scan'])
                status_text.text(f"📊 Scanning... {int(pct*100)}% ({int(pct*scan_count)}/{scan_count} stocks)")
            
            results = analyze_multiple_stocks(
                stock_list,
                'swing' if trading_style == "Swing Trading" else 'intraday',
                timeframe,
                progress_callback=update_progress,
                resume=resume_scan,
                contrarian_mode=contrarian_mode
            )
            
            progress.empty()
            status_text.empty()
            
            if not results.empty:
                save_picks(results, trading_style)
                results_placeholder.success(f"✅ Found {len(results)} opportunities!")
                
                st.subheader(f"🏆 Top {trading_style} Picks (Sector Diversified)")
                
                # IMPROVED STYLING WITH BETTER CONTRAST
                def highlight_score(val):
                    if val >= 75:
                        return 'background-color: #90EE90; color: #000000; font-weight: bold'
                    elif val >= 60:
                        return 'background-color: #FFFACD; color: #000000; font-weight: bold'
                    elif val <= 40:
                        return 'background-color: #FFB6C1; color: #000000; font-weight: bold'
                    else:
                        return 'color: #000000'
                
                styled_df = results.style.applymap(highlight_score, subset=['Score'])
                st.dataframe(styled_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"stock_picks_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    avg_score = results['Score'].mean()
                    st.metric("Average Score", f"{avg_score:.1f}")
            
            else:
                results_placeholder.warning("⚠️ No stocks met the criteria. Try adjusting filters or sectors.")
        
        except KeyboardInterrupt:
            progress.empty()
            status_text.warning("⚠️ Scan paused. Click 'Resume Scan' to continue from where you left off.")
        
        except Exception as e:
            progress.empty()
            status_text.error(f"❌ Scan failed: {str(e)}")
            st.info("💡 Click 'Resume Scan' to retry from the last checkpoint")
            logging.error(f"Scanner error: {str(e)}", exc_info=True)
    
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
                            timeframe, account_size, contrarian_mode
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
