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
import warnings
import sqlite3
from diskcache import Cache
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import pyotp
import os
from dotenv import load_dotenv
import json
import threading
from logzero import logger
import websocket

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
CLIENT_ID = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
API_KEYS = {
    "Historical": os.getenv("HISTORICAL_API_KEY"),
    "Trading": os.getenv("TRADING_API_KEY"),
    "Market": os.getenv("MARKET_API_KEY")
}

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

cache = Cache("stock_data_cache")

# Tooltips and Sectors (unchanged from your code)
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
    "Score": "Measured by RSI, MACD, Ichimoku Cloud, and ATR volatility. Low score = weak signal, high score = strong signal."
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
    "Software & IT Services": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS",
        "MPHASIS.NS", "FSL.NS", "BSOFT.NS", "NEWGEN.NS", "ZENSARTECH.NS",
        "RATEGAIN.NS", "TANLA.NS", "COFORGE.NS", "PERSISTENT.NS", "CYIENT.NS",
        "SONATSOFTW.NS", "KPITTECH.NS", "BIRLASOFT.NS", "TATAELXSI.NS", "MINDTREE.NS",
        "INTELLECT.NS", "HAPPSTMNDS.NS", "MASTEK.NS", "ECLERX.NS", "NIITLTD.NS",
        "RSYSTEMS.NS", "XCHANGING.NS", "OFSS.NS", "AURIONPRO.NS", "DATAMATICS.NS",
        "QUICKHEAL.NS", "CIGNITITEC.NS", "SAGILITY.NS", "ALLSEC.NS"
    ],
    "Finance": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
        "AXISBANK.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS", "SHRIRAMFIN.NS", "CHOLAFIN.NS",
        "SBICARD.NS", "M&MFIN.NS", "MUTHOOTFIN.NS", "LICHSGFIN.NS", "IDFCFIRSTB.NS",
        "AUBANK.NS", "POONAWALLA.NS", "SUNDARMFIN.NS", "IIFL.NS", "ABCAPITAL.NS",
        "L&TFH.NS", "CREDITACC.NS", "MANAPPURAM.NS", "DHANI.NS", "JMFINANCIL.NS",
        "EDELWEISS.NS", "INDIASHLTR.NS", "MOTILALOFS.NS", "CDSL.NS", "BSE.NS",
        "MCX.NS", "ANGELONE.NS", "KARURVYSYA.NS", "RBLBANK.NS", "PNB.NS",
        "CANBK.NS", "UNIONBANK.NS", "IOB.NS", "YESBANK.NS", "UCOBANK.NS",
        "BANKINDIA.NS", "CENTRALBK.NS", "IDBI.NS", "J&KBANK.NS", "DCBBANK.NS",
        "FEDERALBNK.NS", "SOUTHBANK.NS", "CSBBANK.NS", "TMB.NS", "KTKBANK.NS",
        "EQUITASBNK.NS", "UJJIVANSFB.NS", "BANDHANBNK.NS", "SURYODAY.NS", "FSL.NS",
        "PSB.NS", "PFS.NS", "HDFCAMC.NS", "NAM-INDIA.NS", "UTIAMC.NS", "ABSLAMC.NS",
        "360ONE.NS", "ANANDRATHI.NS", "PNBHOUSING.NS", "HOMEFIRST.NS", "AAVAS.NS",
        "APTUS.NS", "RECLTD.NS", "PFC.NS", "IREDA.NS", "SMCGLOBAL.NS", "CHOICEIN.NS",
        "KFINTECH.NS", "CAMSBANK.NS", "MASFIN.NS", "TRIDENT.NS", "SBFC.NS",
        "UGROCAP.NS", "FUSION.NS", "PAISALO.NS", "CAPITALSFB.NS", "NSIL.NS",
        "SATIN.NS", "CREDAGRI.NS"
    ],
    "Automobile & Ancillaries": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
        "EICHERMOT.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "MRF.NS", "BALKRISIND.NS",
        "APOLLOTYRE.NS", "CEATLTD.NS", "JKTYRE.NS", "MOTHERSON.NS", "BHARATFORG.NS",
        "SUNDRMFAST.NS", "EXIDEIND.NS", "AMARAJABAT.NS", "BOSCHLTD.NS", "ENDURANCE.NS",
        "MINDAIND.NS", "WABCOINDIA.NS", "GABRIEL.NS", "SUPRAJIT.NS", "LUMAXTECH.NS",
        "FIEMIND.NS", "SUBROS.NS", "JAMNAAUTO.NS", "SHRIRAMCIT.NS", "ESCORTS.NS",
        "ATULAUTO.NS", "OLECTRA.NS", "GREAVESCOT.NS", "SMLISUZU.NS", "VSTTILLERS.NS",
        "HINDMOTORS.NS", "MAHSCOOTER.NS"
    ],
    "Healthcare": [
        "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "LUPIN.NS",
        "DIVISLAB.NS", "AUROPHARMA.NS", "ALKEM.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS",
        "IPCALAB.NS", "GLENMARK.NS", "BIOCON.NS", "ABBOTINDIA.NS", "SANOFI.NS",
        "PFIZER.NS", "GLAXO.NS", "NATCOPHARM.NS", "AJANTPHARM.NS", "GRANULES.NS",
        "LAURUSLABS.NS", "STAR.NS", "JUBLPHARMA.NS", "ASTRAZEN.NS", "WOCKPHARDT.NS",
        "FORTIS.NS", "MAXHEALTH.NS", "METROPOLIS.NS", "THYROCARE.NS", "POLYMED.NS",
        "KIMS.NS", "NH.NS", "LALPATHLAB.NS", "MEDPLUS.NS", "ERIS.NS", "INDOCO.NS",
        "CAPLIPOINT.NS", "NEULANDLAB.NS", "SHILPAMED.NS", "SUVENPHAR.NS", "AARTIDRUGS.NS",
        "PGHL.NS", "SYNGENE.NS", "VINATIORGA.NS", "GLAND.NS", "JBCHEPHARM.NS",
        "HCG.NS", "RAINBOW.NS", "ASTERDM.NS", "KRSNAA.NS", "VIJAYA.NS", "MEDANTA.NS",
        "NETMEDS.NS", "BLISSGVS.NS", "MOREPENLAB.NS", "RPGLIFE.NS"
    ],
    "Metals & Mining": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "SAIL.NS",
        "NMDC.NS", "HINDZINC.NS", "NALCO.NS", "JINDALSTEL.NS", "MOIL.NS",
        "APLAPOLLO.NS", "RATNAMANI.NS", "JSL.NS", "WELCORP.NS", "TINPLATE.NS",
        "SHYAMMETL.NS", "MIDHANI.NS", "GRAVITA.NS", "SARDAEN.NS", "ASHAPURMIN.NS",
        "JTLIND.NS", "RAMASTEEL.NS", "MAITHANALL.NS", "KIOCL.NS", "IMFA.NS",
        "GMDCLTD.NS", "VISHNU.NS", "SANDUMA.NS"
    ],
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "VARBEV.NS", "BRITANNIA.NS",
        "GODREJCP.NS", "DABUR.NS", "COLPAL.NS", "MARICO.NS", "PGHH.NS",
        "EMAMILTD.NS", "GILLETTE.NS", "HATSUN.NS", "JYOTHYLAB.NS", "BAJAJCON.NS",
        "RADICO.NS", "TATACONSUM.NS", "UNITDSPR.NS", "CCL.NS", "AVANTIFEED.NS",
        "BIKAJI.NS", "PATANJALI.NS", "VBL.NS", "ZOMATO.NS", "DOMS.NS",
        "GODREJAGRO.NS", "SAPPHIRE.NS", "VENKEYS.NS", "BECTORFOOD.NS", "KRBL.NS"
    ],
    "Power": [
        "NTPC.NS", "POWERGRID.NS", "ADANIPOWER.NS", "TATAPOWER.NS", "JSWENERGY.NS",
        "NHPC.NS", "SJVN.NS", "TORNTPOWER.NS", "CESC.NS", "ADANIENSOL.NS",
        "INDIAGRID.NS", "POWERMECH.NS", "KEC.NS", "INOXWIND.NS", "KALPATPOWR.NS",
        "SUZLON.NS", "BHEL.NS", "THERMAX.NS", "GEPIL.NS", "VOLTAMP.NS",
        "TRIL.NS", "TDPOWERSYS.NS", "JYOTISTRUC.NS", "IWEL.NS"
    ],
    "Capital Goods": [
        "LT.NS", "SIEMENS.NS", "ABB.NS", "BEL.NS", "BHEL.NS", "HAL.NS",
        "CUMMINSIND.NS", "THERMAX.NS", "AIAENG.NS", "SKFINDIA.NS", "GRINDWELL.NS",
        "TIMKEN.NS", "KSB.NS", "ELGIEQUIP.NS", "LAKSHMIMACH.NS", "KIRLOSENG.NS",
        "GREAVESCOT.NS", "TRITURBINE.NS", "VOLTAS.NS", "BLUESTARCO.NS", "HAVELLS.NS",
        "DIXON.NS", "KAYNES.NS", "SYRMA.NS", "AMBER.NS", "SUZLON.NS", "CGPOWER.NS",
        "APARINDS.NS", "HBLPOWER.NS", "KEI.NS", "POLYCAB.NS", "RRKABEL.NS",
        "SCHNEIDER.NS", "TDPOWERSYS.NS", "KIRLOSBROS.NS", "JYOTICNC.NS", "DATAPATTNS.NS",
        "INOXWIND.NS", "KALPATPOWR.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "GRSE.NS",
        "POWERMECH.NS", "ISGEC.NS", "HPL.NS", "VTL.NS", "DYNAMATECH.NS", "JASH.NS",
        "GMMPFAUDLR.NS", "ESABINDIA.NS", "CENTURYEXT.NS", "SALASAR.NS", "TITAGARH.NS",
        "VGUARD.NS", "WABAG.NS", "AZAD"
    ],
    "Oil & Gas": [
        "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HPCL.NS", "GAIL.NS",
        "PETRONET.NS", "OIL.NS", "IGL.NS", "MGL.NS", "GUJGASLTD.NS", "GSPL.NS",
        "AEGISCHEM.NS", "CHENNPETRO.NS", "MRPL.NS", "GULFOILLUB.NS", "CASTROLIND.NS",
        "SOTL.NS", "PANAMAPET.NS", "GOCLCORP.NS"
    ],
    "Chemicals": [
        "PIDILITIND.NS", "SRF.NS", "DEEPAKNTR.NS", "ATUL.NS", "AARTIIND.NS",
        "NAVINFLUOR.NS", "VINATIORGA.NS", "FINEORG.NS", "ALKYLAMINE.NS", "BALAMINES.NS",
        "GUJFLUORO.NS", "CLEAN.NS", "JUBLINGREA.NS", "GALAXYSURF.NS", "PCBL.NS",
        "NOCIL.NS", "BASF.NS", "SUDARSCHEM.NS", "NEOGEN.NS", "PRIVISCL.NS",
        "ROSSARI.NS", "LXCHEM.NS", "ANURAS.NS", "JUBLPHARMA.NS", "CHEMCON.NS",
        "DMCC.NS", "TATACHEM.NS", "COROMANDEL.NS", "UPL.NS", "BAYERCROP.NS",
        "SUMICHEM.NS", "PIIND.NS", "DHARAMSI.NS", "EIDPARRY.NS", "CHEMPLASTS.NS",
        "VISHNU.NS", "IGPL.NS", "TIRUMALCHM.NS"
    ],
    "Telecom": [
        "BHARTIARTL.NS", "VODAFONEIDEA.NS", "INDUSTOWER.NS", "TATACOMM.NS",
        "HFCL.NS", "TEJASNET.NS", "STLTECH.NS", "ITI.NS", "ASTEC.NS"
    ],
    "Infrastructure": [
        "LT.NS", "GMRINFRA.NS", "IRB.NS", "NBCC.NS", "RVNL.NS", "KEC.NS",
        "PNCINFRA.NS", "KNRCON.NS", "GRINFRA.NS", "NCC.NS", "HGINFRA.NS",
        "ASHOKA.NS", "SADBHAV.NS", "JWL.NS", "PATELENG.NS", "KALPATPOWR.NS",
        "IRCON.NS", "ENGINERSIN.NS", "AHLUWALIA.NS", "PSPPROJECTS.NS", "CAPACITE.NS",
        "WELSPUNIND.NS", "TITAGARH.NS", "HCC.NS", "MANINFRA.NS", "RIIL.NS",
        "DBREALTY.NS", "JWL.NS"
    ],
    "Insurance": [
        "SBILIFE.NS", "HDFCLIFE.NS", "ICICIGI.NS", "ICICIPRULI.NS", "LICI.NS",
        "GICRE.NS", "NIACL.NS", "STARHEALTH.NS", "BAJAJFINSV.NS", "MAXFIN.NS"
    ],
    "Diversified": [
        "ITC.NS", "RELIANCE.NS", "ADANIENT.NS", "GRASIM.NS", "HINDUNILVR.NS",
        "DCMSHRIRAM.NS", "3MINDIA.NS", "CENTURYPLY.NS", "KFINTECH.NS", "BALMERLAWRI.NS",
        "GODREJIND.NS", "VBL.NS", "BIRLACORPN.NS"
    ],
    "Construction Materials": [
        "ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS", "JKCEMENT.NS",
        "DALBHARAT.NS", "RAMCOCEM.NS", "NUVOCO.NS", "JKLAKSHMI.NS", "BIRLACORPN.NS",
        "HEIDELBERG.NS", "INDIACEM.NS", "PRISMJOHNS.NS", "STARCEMENT.NS", "SAGCEM.NS",
        "DECCANCE.NS", "KCP.NS", "ORIENTCEM.NS", "HIL.NS", "EVERESTIND.NS",
        "VISAKAIND.NS", "BIGBLOC.NS"
    ],
    "Real Estate": [
        "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "PRESTIGE.NS",
        "BRIGADE.NS", "SOBHA.NS", "SUNTECK.NS", "MAHLIFE.NS", "ANANTRAJ.NS",
        "KOLTEPATIL.NS", "PURVA.NS", "ARVSMART.NS", "RUSTOMJEE.NS", "DBREALTY.NS",
        "IBREALEST.NS", "OMAXE.NS", "ASHIANA.NS", "ELDEHSG.NS", "TARC.NS"
    ],
    "Aviation": [
        "INDIGO.NS", "SPICEJET.NS", "AAI.NS", "GMRINFRA.NS"
    ],
    "Retailing": [
        "DMART.NS", "TRENT.NS", "ABFRL.NS", "VMART.NS", "SHOPERSTOP.NS",
        "BATAINDIA.NS", "METROBRAND.NS", "ARVINDFASN.NS", "CANTABIL.NS", "ZOMATO.NS",
        "NYKAA.NS", "MANYAVAR.NS", "ELECTRONICSMRKT.NS", "LANDMARK.NS", "V2RETAIL.NS",
        "THANGAMAYL.NS", "KALYANKJIL.NS", "TITAN.NS"
    ],
    "Miscellaneous": [
        "PIDILITIND.NS", "BSE.NS", "CDSL.NS", "MCX.NS", "NAUKRI.NS",
        "JUSTDIAL.NS", "TEAMLEASE.NS", "QUESS.NS", "SIS.NS", "DELHIVERY.NS",
        "PRUDENT.NS", "MEDIASSIST.NS", "AWFIS.NS", "JUBLFOOD.NS", "DEVYANI.NS",
        "WESTLIFE.NS", "SAPPHIRE.NS", "BARBEQUE.NS", "EASEMYTRIP.NS", "THOMASCOOK.NS",
        "MSTC.NS", "IRCTC.NS", "POLICYBZR.NS", "PAYTM.NS", "INFIBEAM.NS",
        "CARTRADE.NS", "HONASA.NS", "ONE97COMM.NS", "SIGNATURE.NS", "RRKABEL.NS",
        "HMAAGRO.NS", "RKFORGE.NS", "CAMPUS.NS", "SENCO.NS", "CONCORDBIO.NS"
    ]
}

# Utility Functions
def tooltip(label, explanation):
    return f"{label} 📌 ({explanation})"

def retry(max_retries=5, delay=5, backoff_factor=2, jitter=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        retries += 1
                        if retries == max_retries:
                            raise e
                        sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                        st.warning(f"Rate limit hit. Retrying after {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    else:
                        raise e
                except (requests.exceptions.RequestException, ConnectionError) as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                    time.sleep(sleep_time)
        return wrapper
    return decorator

# WebSocket Manager
class WebSocketManager:
    def __init__(self, db_path="stock_picks.db", max_retries=5):
        self.db_path = db_path
        self.max_retries = max_retries
        self.sws = None
        self.auth_token = None
        self.feed_token = None
        self.client_code = CLIENT_ID
        self.subscribed_tokens = set()
        self.running = False
        self.thread = None

    def init_smartapi(self):
        """Initialize SmartAPI session for WebSocket authentication."""
        try:
            smart_api = SmartConnect(api_key=API_KEYS["Market"])
            totp = pyotp.TOTP(TOTP_SECRET)
            data = smart_api.generateSession(CLIENT_ID, PASSWORD, totp.now())
            if data['status']:
                self.auth_token = data['data']['jwtToken']
                self.feed_token = smart_api.getfeedToken()
                self.client_code = CLIENT_ID
                logger.info("SmartAPI session initialized successfully.")
                return True
            else:
                logger.error(f"SmartAPI authentication failed: {data['message']}")
                return False
        except Exception as e:
            logger.error(f"Error initializing SmartAPI: {str(e)}")
            return False

    def get_symbol_tokens(self):
        """Fetch symbol tokens for open positions from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT symbol FROM paper_positions")
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()

            symbol_token_map = load_symbol_token_map()
            token_list = []
            for symbol in symbols:
                token = symbol_token_map.get(symbol)
                if token:
                    token_list.append({"exchangeType": 1, "tokens": [token]})  # NSE: exchangeType 1
            return token_list
        except Exception as e:
            logger.error(f"Error fetching symbol tokens: {str(e)}")
            return []

    def update_portfolio(self, symbol, ltp):
        """Update unrealized P&L for a position based on LTP."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT quantity, avg_price FROM paper_positions WHERE symbol = ?", (symbol,))
            position = cursor.fetchone()
            if position:
                quantity, avg_price = position
                unrealized_pnl = (ltp - avg_price) * quantity
                cursor.execute(
                    "UPDATE paper_positions SET unrealized_pnl = ? WHERE symbol = ?",
                    (unrealized_pnl, symbol)
                )
                total_unrealized_pnl = sum(
                    row[0] for row in cursor.execute("SELECT unrealized_pnl FROM paper_positions").fetchall()
                )
                cursor.execute(
                    "UPDATE paper_portfolio SET unrealized_pnl = ?, total_pnl = realized_pnl + ? WHERE date = ?",
                    (total_unrealized_pnl, total_unrealized_pnl, datetime.now().strftime('%Y-%m-%d'))
                )
                conn.commit()
                logger.info(f"Updated P&L for {symbol}: Unrealized P&L = ₹{unrealized_pnl:.2f}")
            conn.close()
        except Exception as e:
            logger.error(f"Error updating portfolio for {symbol}: {str(e)}")

    def check_stop_loss_target(self, symbol, ltp):
        """Check if LTP hits stop-loss or target and simulate order execution."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT quantity, avg_price, stop_loss, target FROM paper_positions WHERE symbol = ?",
                (symbol,)
            )
            position = cursor.fetchone()
            if position:
                quantity, avg_price, stop_loss, target = position
                if stop_loss and ltp <= stop_loss:
                    logger.info(f"Stop-loss triggered for {symbol} at ₹{ltp}")
                    self.simulate_sell_order(symbol, quantity, ltp, "Stop-Loss")
                elif target and ltp >= target:
                    logger.info(f"Target triggered for {symbol} at ₹{ltp}")
                    self.simulate_sell_order(symbol, quantity, ltp, "Target")
            conn.close()
        except Exception as e:
            logger.error(f"Error checking stop-loss/target for {symbol}: {str(e)}")

    def simulate_sell_order(self, symbol, quantity, price, trigger_type):
        """Simulate a sell order and update positions/portfolio."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT quantity, avg_price FROM paper_positions WHERE symbol = ?", (symbol,))
            position = cursor.fetchone()
            if position:
                current_qty, avg_price = position
                realized_pnl = (price - avg_price) * quantity
                order_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
                order_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                cursor.execute(
                    """
                    INSERT INTO paper_orders (order_id, symbol, order_type, quantity, price, status, order_time, execution_time, strategy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (order_id, symbol, "SELL", quantity, price, "EXECUTED", order_time, order_time, trigger_type)
                )

                new_qty = current_qty - quantity
                if new_qty > 0:
                    cursor.execute(
                        "UPDATE paper_positions SET quantity = ? WHERE symbol = ?",
                        (new_qty, symbol)
                    )
                else:
                    cursor.execute("DELETE FROM paper_positions WHERE symbol = ?", (symbol,))

                cursor.execute(
                    "UPDATE paper_portfolio SET cash_balance = cash_balance + ?, realized_pnl = realized_pnl + ? WHERE date = ?",
                    (quantity * price, realized_pnl, datetime.now().strftime('%Y-%m-%d'))
                )
                conn.commit()
                logger.info(f"Simulated SELL order for {symbol}: {quantity} @ ₹{price} ({trigger_type})")
            conn.close()
        except Exception as e:
            logger.error(f"Error simulating sell order for {symbol}: {str(e)}")

    def on_data(self, wsapp, message):
        """Handle incoming WebSocket data."""
        try:
            # Convert LTP from paisa to rupees if necessary
            ltp = message.get('last_traded_price', 0) / 100 if message.get('last_traded_price') else None
            symbol = message.get('trading_symbol')
            if ltp and symbol:
                logger.info(f"Received tick: {symbol} @ ₹{ltp:.2f}")
                self.update_portfolio(symbol, ltp)
                self.check_stop_loss_target(symbol, ltp)
        except Exception as e:
            logger.error(f"Error processing WebSocket data: {str(e)}")

    def on_open(self, wsapp):
        """Handle WebSocket connection open event."""
        logger.info("WebSocket connection opened")
        token_list = self.get_symbol_tokens()
        if token_list:
            self.sws.subscribe("ws_paper_trading", 1, token_list)  # Mode 1: LTP
            self.subscribed_tokens.update(
                token for sublist in token_list for token in sublist['tokens']
            )
            logger.info(f"Subscribed to tokens: {self.subscribed_tokens}")

    def on_error(self, wsapp, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {str(error)}")

    def on_close(self, wsapp):
        """Handle WebSocket connection close."""
        logger.info("WebSocket connection closed")
        self.running = False

    def start_websocket(self):
        """Start the WebSocket connection in a separate thread."""
        if not self.init_smartapi():
            logger.error("Failed to initialize SmartAPI. WebSocket not started.")
            return

        self.sws = SmartWebSocketV2(
            auth_token=self.auth_token,
            api_key=API_KEYS["Market"],
            client_code=self.client_code,
            feed_token=self.feed_token,
            max_retry_attempt=self.max_retries
        )
        self.sws.on_open = self.on_open
        self.sws.on_data = self.on_data
        self.sws.on_error = self.on_error
        self.sws.on_close = self.on_close

        self.running = True
        self.thread = threading.Thread(target=self.sws.connect)
        self.thread.daemon = True
        self.thread.start()
        logger.info("WebSocket thread started")

    def stop_websocket(self):
        """Stop the WebSocket connection."""
        if self.sws and self.running:
            self.sws.close_connection()
            self.running = False
            self.thread.join()
            logger.info("WebSocket thread stopped")

    def update_subscriptions(self):
        """Update WebSocket subscriptions based on current positions."""
        if not self.running:
            return
        token_list = self.get_symbol_tokens()
        new_tokens = set(token for sublist in token_list for token in sublist['tokens'])
        tokens_to_unsubscribe = self.subscribed_tokens - new_tokens
        tokens_to_subscribe = new_tokens - self.subscribed_tokens

        if tokens_to_unsubscribe:
            self.sws.unsubscribe("ws_paper_trading", 1, [
                {"exchangeType": 1, "tokens": list(tokens_to_unsubscribe)}
            ])
            self.subscribed_tokens -= tokens_to_unsubscribe
            logger.info(f"Unsubscribed from tokens: {tokens_to_unsubscribe}")

        if tokens_to_subscribe:
            self.sws.subscribe("ws_paper_trading", 1, [
                {"exchangeType": 1, "tokens": list(tokens_to_subscribe)}
            ])
            self.subscribed_tokens.update(tokens_to_subscribe)
            logger.info(f"Subscribed to tokens: {tokens_to_subscribe}")

# Paper Trading Class
class PaperTrading:
    def __init__(self, initial_cash= ~

100000):
        self.cash_balance = initial_cash
        self.conn = sqlite3.connect('stock_picks.db')
        self.websocket_manager = WebSocketManager()
        self.init_portfolio()
        self.start_websocket()

    def init_portfolio(self):
        today = datetime.now().strftime('%Y-%m-%d')
        self.conn.execute('''
            INSERT OR IGNORE INTO paper_portfolio (date, cash_balance, total_pnl, realized_pnl, unrealized_pnl)
            VALUES (?, ?, 0, 0, 0)
        ''', (today, self.cash_balance))
        self.conn.commit()

    def start_websocket(self):
        self.websocket_manager.start_websocket()

    def stop_websocket(self):
        self.websocket_manager.stop_websocket()

    def place_order(self, symbol, order_type, quantity, price, strategy, stop_loss=None, target=None):
        try:
            order_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
            order_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            total_cost = quantity * price
            if order_type == "BUY" and total_cost > self.cash_balance:
                st.error(f"⚠️ Insufficient funds. Required: ₹{total_cost}, Available: ₹{self.cash_balance}")
                return False

            self.conn.execute('''
                INSERT INTO paper_orders (order_id, symbol, order_type, quantity, price, status, order_time, execution_time, strategy, stop_loss, target)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (order_id, symbol, order_type, quantity, price, "EXECUTED", order_time, order_time, strategy, stop_loss, target))

            self.update_position(symbol, order_type, quantity, price, strategy, stop_loss, target)

            if order_type == "BUY":
                self.cash_balance -= total_cost
            else:
                self.cash_balance += total_cost

            self.update_portfolio()
            self.websocket_manager.update_subscriptions()
            self.conn.commit()
            st.success(f"✅ {order_type} order for {quantity} shares of {symbol} executed at ₹{price}")
            return True
        except Exception as e:
            st.error(f"⚠️ Error placing order: {str(e)}")
            return False

    def update_position(self, symbol, order_type, quantity, price, strategy, stop_loss, target):
        cursor = self.conn.cursor()
        cursor.execute('SELECT quantity, avg_price FROM paper_positions WHERE symbol = ?', (symbol,))
        position = cursor.fetchone()

        if order_type == "BUY":
            if position:
                current_qty, avg_price = position
                new_qty = current_qty + quantity
                new_avg_price = ((current_qty * avg_price) + (quantity * price)) / new_qty
                cursor.execute('''
                    UPDATE paper_positions SET quantity = ?, avg_price = ?, entry_time = ?, strategy = ?, stop_loss = ?, target = ?
                    WHERE symbol = ?
                ''', (new_qty, new_avg_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), strategy, stop_loss, target, symbol))
            else:
                cursor.execute('''
                    INSERT INTO paper_positions (symbol, quantity, avg_price, entry_time, strategy, stop_loss, target, unrealized_pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                ''', (symbol, quantity, price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), strategy, stop_loss, target))
        else:  # SELL
            if position:
                current_qty, avg_price = position
                new_qty = current_qty - quantity
                if new_qty > 0:
                    cursor.execute('''
                        UPDATE paper_positions SET quantity = ?, avg_price = ?, entry_time = ?, strategy = ?, stop_loss = ?, target = ?
                        WHERE symbol = ?
                    ''', (new_qty, avg_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), strategy, stop_loss, target, symbol))
                else:
                    cursor.execute('DELETE FROM paper_positions WHERE symbol = ?', (symbol,))
                realized_pnl = (price - avg_price) * quantity
                cursor.execute('''
                    UPDATE paper_portfolio SET realized_pnl = realized_pnl + ? WHERE date = ?
                ''', (realized_pnl, datetime.now().strftime('%Y-%m-%d')))
            else:
                st.warning(f"⚠️ No position to sell for {symbol}")

    def update_portfolio(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT symbol, quantity, avg_price FROM paper_positions')
        positions = cursor.fetchall()
        unrealized_pnl = sum(row[0] for row in cursor.execute("SELECT unrealized_pnl FROM paper_positions").fetchall())
        total_pnl = unrealized_pnl + cursor.execute(
            'SELECT realized_pnl FROM paper_portfolio WHERE date = ?',
            (datetime.now().strftime('%Y-%m-%d'),)
        ).fetchone()[0]

        cursor.execute('''
            UPDATE paper_portfolio SET cash_balance = ?, total_pnl = ?, unrealized_pnl = ?
            WHERE date = ?
        ''', (self.cash_balance, total_pnl, unrealized_pnl, datetime.now().strftime('%Y-%m-%d')))

# SmartAPI Client and Market Data
def init_smartapi_client(api_type="Historical"):
    try:
        api_key = API_KEYS.get(api_type)
        if not api_key:
            st.error(f"⚠️ {api_type} API key not found.")
            return None
        smart_api = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(TOTP_SECRET)
        data = smart_api.generateSession(CLIENT_ID, PASSWORD, totp.now())
        if data['status']:
            return smart_api
        else:
            st.error(f"⚠️ SmartAPI {api_type} authentication failed: {data['message']}")
            return None
    except Exception as e:
        st.error(f"⚠️ Error initializing SmartAPI ({api_type}): {str(e)}")
        return None

@retry(max_retries=5, delay=5)
def fetch_market_data(symbol):
    try:
        smart_api = init_smartapi_client(api_type="Market")
        if not smart_api:
            return None
        symbol_token_map = load_symbol_token_map()
        symboltoken = symbol_token_map.get(symbol)
        if not symboltoken:
            st.warning(f"⚠️ Token not found for symbol: {symbol}")
            return None
        market_data = smart_api.getQuote({
            "exchange": "NSE",
            "symboltoken": symboltoken
        })
        if market_data['status']:
            return {
                "ltp": market_data['data']['ltp'] / 100,  # Convert paisa to rupees
                "open": market_data['data']['open'] / 100,
                "high": market_data['data']['high'] / 100,
                "low": market_data['data']['low'] / 100,
                "close": market_data['data']['close'] / 100,
                "volume": market_data['data']['volume'],
                "last_trade_time": market_data['data']['lastTradeTime']
            }
        else:
            st.warning(f"⚠️ No market data for {symbol}: {market_data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        st.warning(f"⚠️ Error fetching market data for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=86400)
def load_symbol_token_map():
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        symbol_token_map = {entry["symbol"]: entry["token"] for entry in data if "symbol" in entry and "token" in entry}
        with open("symbol_token_map.json", "w") as f:
            json.dump(symbol_token_map, f)
        return symbol_token_map
    except Exception as e:
        st.warning(f"⚠️ Failed to load instrument list: {str(e)}")
        if os.path.exists("symbol_token_map.json"):
            with open("symbol_token_map.json", "r") as f:
                return json.load(f)
        return {}

# Database Initialization
def init_database():
    conn = sqlite3.connect('stock_picks.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS daily_picks (
            date TEXT,
            symbol TEXT,
            score REAL,
            current_price REAL,
            buy_at REAL,
            stop_loss REAL,
            target REAL,
            intraday TEXT,
            swing TEXT,
            short_term TEXT,
            long_term TEXT,
            mean_reversion TEXT,
            breakout TEXT,
            ichimoku_trend TEXT,
            pick_type TEXT,
            PRIMARY KEY (date, symbol)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS paper_orders (
            order_id TEXT PRIMARY KEY,
            symbol TEXT,
            order_type TEXT,
            quantity INTEGER,
            price REAL,
            status TEXT,
            order_time TEXT,
            execution_time TEXT,
            strategy TEXT,
            stop_loss REAL,
            target REAL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS paper_positions (
            symbol TEXT PRIMARY KEY,
            quantity INTEGER,
            avg_price REAL,
            entry_time TEXT,
            strategy TEXT,
            stop_loss REAL,
            target REAL,
            unrealized_pnl REAL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS paper_portfolio (
            date TEXT,
            cash_balance REAL,
            total_pnl REAL,
            realized_pnl REAL,
            unrealized_pnl REAL,
            PRIMARY KEY (date)
        )
    ''')
    conn.commit()
    conn.close()

# Stock Data Fetching
@retry(max_retries=5, delay=5)
def fetch_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
        response = session.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        stock_list = [f"{symbol}-EQ" for symbol in nse_data['SYMBOL']]
        return stock_list
    except Exception:
        return list(set([stock for sector in SECTORS.values() for stock in sector]))

@retry(max_retries=5, delay=5)
def fetch_stock_data_with_auth(symbol, period="5y", interval="1d"):
    cache_key = f"{symbol}_{period}_{interval}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return pd.read_pickle(io.BytesIO(cached_data))

    try:
        if "-EQ" not in symbol:
            symbol = f"{symbol.split('.')[0]}-EQ"
        smart_api = init_smartapi_client()
        if not smart_api:
            raise ValueError("SmartAPI client initialization failed")
        end_date = datetime.now()
        if period == "5y":
            start_date = end_date - timedelta(days=5 * 365)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "1mo":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=365)
        interval_map = {
            "1d": "ONE_DAY",
            "1h": "ONE_HOUR",
            "5m": "FIVE_MINUTE",
            "15m": "FIFTEEN_MINUTE"
        }
        api_interval = interval_map.get(interval, "ONE_DAY")
        symbol_token_map = load_symbol_token_map()
        symboltoken = symbol_token_map.get(symbol)
        if not symboltoken:
            st.warning(f"⚠️ Token not found for symbol: {symbol}")
            return pd.DataFrame()
        historical_data = smart_api.getCandleData({
            "exchange": "NSE",
            "symboltoken": symboltoken,
            "interval": api_interval,
            "fromdate": start_date.strftime("%Y-%m-%d %H:%M"),
            "todate": end_date.strftime("%Y-%m-%d %H:%M")
        })
        if historical_data['status'] and historical_data['data']:
            data = pd.DataFrame(historical_data['data'], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            buffer = io.BytesIO()
            data.to_pickle(buffer)
            cache.set(cache_key, buffer.getvalue(), expire=86400)
            return data
        else:
            raise ValueError(f"No data found for {symbol}: {historical_data.get('message', 'Unknown error')}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.warning(f"⚠️ Rate limit exceeded for {symbol}. Skipping...")
            return pd.DataFrame()
        raise e
    except Exception as e:
        st.warning(f"⚠️ Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

@lru_cache(maxsize=1000)
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    return fetch_stock_data_with_auth(symbol, period, interval)

# Technical Analysis and Recommendations (unchanged from your code)
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
    if len(returns) < 30:
        mean_return = returns.mean()
        std_return = returns.std()
        simulation_results = []
        for _ in range(simulations):
            price_series = [data['Close'].iloc[-1]]
            for _ in range(days):
                price = price_series[-1] * (1 + np.random.normal(mean_return, std_return))
                price_series.append(price)
            simulation_results.append(price_series)
        return simulation_results
    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='Normal', rescale=False)
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

def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return entities

def get_trending_stocks():
    pytrends = TrendReq(hl='en-US', tz=360)
    trending = pytrends.trending_searches(pn='india')
    return trending

def calculate_confidence_score(data):
    score = 0
    if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None and data['RSI'].iloc[-1] < 30:
        score += 1
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None and data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
        score += 1
    if 'Ichimoku_Span_A' in data.columns and data['Close'].iloc[-1] is not None and data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]:
        score += 1
    if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None and data['Close'].iloc[-1] is not None:
        atr_volatility = data['ATR'].iloc[-1] / data['Close'].iloc[-1]
        if atr_volatility < 0.02:
            score += 0.5
        elif atr_volatility > 0.05:
            score -= 0.5
    return min(max(score / 3.5, 0), 1)

def assess_risk(data):
    if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None and data['ATR'].iloc[-1] > data['ATR'].mean():
        return "High Volatility Warning"
    else:
        return "Low Volatility"

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

def detect_divergence(data):
    rsi = data['RSI']
    price = data['Close']
    recent_highs = price[-5:].idxmax()
    recent_lows = price[-5:].idxmin()
    rsi_highs = rsi[-5:].idxmax()
    rsi_lows = rsi[-5:].idxmin()
    bullish_div = (recent_lows > rsi_lows) and (price[recent_lows] < price[-1]) and (rsi[rsi_lows] < rsi[-1])
    bearish_div = (recent_highs < rsi_highs) and (price[recent_highs] > price[-1]) and (rsi[rsi_highs] > rsi[-1])
    return "Bullish Divergence" if bullish_div else "Bearish Divergence" if bearish_div else "No Divergence"

def calculate_cmo(close, window=14):
    try:
        diff = close.diff()
        up_sum = diff.where(diff > 0, 0).rolling(window=window).sum()
        down_sum = abs(diff.where(diff < 0, 0)).rolling(window=window).sum()
        cmo = 100 * (up_sum - down_sum) / (up_sum + down_sum)
        return cmo
    except Exception as e:
        st.warning(f"⚠️ Failed to compute custom CMO: {str(e)}")
        return None

def analyze_stock(data):
    if data.empty or len(data) < 27:
        st.warning("⚠️ Insufficient data to compute indicators.")
        columns = [
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50',
            'Upper_Band', 'Middle_Band', 'Lower_Band', 'SlowK', 'SlowD', 'ATR', 'ADX', 'OBV',
            'VWAP', 'Avg_Volume', 'Volume_Spike', 'Parabolic_SAR', 'Fib_23.6', 'Fib_38.2',
            'Fib_50.0', 'Fib_61.8', 'Divergence', 'Ichimoku_Tenkan', 'Ichimoku_Kijun',
            'Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Chikou', 'CMF', 'Donchian_Upper',
            'Donchian_Lower', 'Donchian_Middle', 'Keltner_Upper', 'Keltner_Middle', 'Keltner_Lower',
            'TRIX', 'Ultimate_Osc', 'CMO', 'VPT'
        ]
        for col in columns:
            if col not in data.columns:
                data[col] = None
        return data

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
        for col in missing_cols:
            data[col] = None
        return data

    try:
        rsi_window = optimize_rsi_window(data)
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute RSI: {str(e)}")
        data['RSI'] = None

    try:
        macd = ta.trend.MACD(data['Close'], window_slow=17, window_fast=8, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_hist'] = macd.macd_diff()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute MACD: {str(e)}")
        data['MACD'] = None
        data['MACD_signal'] = None
        data['MACD_hist'] = None

    try:
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Moving Averages: {str(e)}")
        data['SMA_50'] = None
        data['SMA_200'] = None
        data['EMA_20'] = None
        data['EMA_50'] = None

    try:
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband()
        data['Middle_Band'] = bollinger.bollinger_mavg()
        data['Lower_Band'] = bollinger.bollinger_lband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Bollinger Bands: {str(e)}")
        data['Upper_Band'] = None
        data['Middle_Band'] = None
        data['Lower_Band'] = None

    try:
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
        data['SlowK'] = stoch.stoch()
        data['SlowD'] = stoch.stoch_signal()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Stochastic: {str(e)}")
        data['SlowK'] = None
        data['SlowD'] = None

    try:
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute ATR: {str(e)}")
        data['ATR'] = None

    try:
        if len(data) >= 27:
            data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
        else:
            data['ADX'] = None
    except Exception as e:
        st.warning(f"⚠️ Failed to compute ADX: {str(e)}")
        data['ADX'] = None

    try:
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute OBV: {str(e)}")
        data['OBV'] = None

    try:
        data['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['VWAP'] = data['Cumulative_TP'].cumsum() / data['Cumulative_Volume']
    except Exception as e:
        st.warning(f"⚠️ Failed to compute VWAP: {str(e)}")
        data['VWAP'] = None

    try:
        data['Avg_Volume'] = data['Volume'].rolling(window=10).mean()
        data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 1.5)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Volume Spike: {str(e)}")
        data['Volume_Spike'] = None

    try:
        data['Parabolic_SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Parabolic SAR: {str(e)}")
        data['Parabolic_SAR'] = None

    try:
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        data['Fib_23.6'] = high - diff * 0.236
        data['Fib_38.2'] = high - diff * 0.382
        data['Fib_50.0'] = high - diff * 0.5
        data['Fib_61.8'] = high - diff * 0.618
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Fibonacci: {str(e)}")
        data['Fib_23.6'] = None
        data['Fib_38.2'] = None
        data['Fib_50.0'] = None
        data['Fib_61.8'] = None

    try:
        data['Divergence'] = detect_divergence(data)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Divergence: {str(e)}")
        data['Divergence'] = "No Divergence"

    try:
        ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
        data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
        data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
        data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
        data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
        data['Ichimoku_Chikou'] = data['Close'].shift(-26)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Ichimoku: {str(e)}")
        data['Ichimoku_Tenkan'] = None
        data['Ichimoku_Kijun'] = None
        data['Ichimoku_Span_A'] = None
        data['Ichimoku_Span_B'] = None
        data['Ichimoku_Chikou'] = None

    try:
        data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute CMF: {str(e)}")
        data['CMF'] = None

    try:
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
        data['Donchian_Upper'] = donchian.donchian_channel_hband()
        data['Donchian_Lower'] = donchian.donchian_channel_lband()
        data['Donchian_Middle'] = donchian.donchian_channel_mband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Donchian: {str(e)}")
        data['Donchian_Upper'] = None
        data['Donchian_Lower'] = None
        data['Donchian_Middle'] = None

    try:
        keltner = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'], window=20, window_atr=10)
        data['Keltner_Upper'] = keltner.keltner_channel_hband()
        data['Keltner_Middle'] = keltner.keltner_channel_mband()
        data['Keltner_Lower'] = keltner.keltner_channel_lband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Keltner Channels: {str(e)}")
        data['Keltner_Upper'] = None
        data['Keltner_Middle'] = None
        data['Keltner_Lower'] = None

    try:
        data['TRIX'] = ta.trend.TRIXIndicator(data['Close'], window=15).trix()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute TRIX: {str(e)}")
        data['TRIX'] = None

    try:
        data['Ultimate_Osc'] = ta.momentum.UltimateOscillator(
            data['High'], data['Low'], data['Close'], window1=7, window2=14, window3=28
        ).ultimate_oscillator()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Ultimate Oscillator: {str(e)}")
        data['Ultimate_Osc'] = None

    try:
        data['CMO'] = calculate_cmo(data['Close'], window=14)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Chande Momentum Oscillator: {str(e)}")
        data['CMO'] = None

    try:
        data['VPT'] = ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume']).volume_price_trend()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Volume Price Trend: {str(e)}")
        data['VPT'] = None

    return data

def calculate_buy_at(data):
    if data.empty or 'RSI' not in data.columns or data['RSI'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Buy At due to missing or invalid RSI data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    buy_at = last_close * 0.99 if last_rsi < 30 else last_close
    return round(buy_at, 2)

def calculate_stop_loss(data, atr_multiplier=2.5):
    if data.empty or 'ATR' not in data.columns or data['ATR'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Stop Loss due to missing or invalid ATR data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    atr_multiplier = 3.0 if data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25 else 1.5
    stop_loss = last_close - (atr_multiplier * last_atr)
    if stop_loss < last_close * 0.9:
        stop_loss = last_close * 0.9
    return round(stop_loss, 2)

def calculate_target(data, risk_reward_ratio=3):
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        st.warning("⚠️ Cannot calculate Target due to missing Stop Loss data.")
        return None
    last_close = data['Close'].iloc[-1]
    risk = last_close - stop_loss
    adjusted_ratio = min(risk_reward_ratio, 5) if data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25 else min(risk_reward_ratio, 3)
    target = last_close + (risk * adjusted_ratio)
    if target > last_close * 1.2:
        target = last_close * 1.2
    return round(target, 2)

def calculate_buy_at_row(row):
    if pd.notnull(row['RSI']) and row['RSI'] < 30:
        return round(row['Close'] * 0.99, 2)
    return round(row['Close'], 2)

def calculate_stop_loss_row(row, atr_multiplier=2.5):
    if pd.notnull(row['ATR']):
        atr_multiplier = 3.0 if pd.notnull(row['ADX']) and row['ADX'] > 25 else 1.5
        stop_loss = row['Close'] - (atr_multiplier * row['ATR'])
        if stop_loss < row['Close'] * 0.9:
            stop_loss = row['Close'] * 0.9
        return round(stop_loss, 2)
    return None

def calculate_target_row(row, risk_reward_ratio=3):
    stop_loss = calculate_stop_loss_row(row)
    if stop_loss is not None:
        risk = row['Close'] - stop_loss
        adjusted_ratio = min(risk_reward_ratio, 5) if pd.notnull(row['ADX']) and row['ADX'] > 25 else min(risk_reward_ratio, 3)
        target = row['Close'] + (risk * adjusted_ratio)
        if target > row['Close'] * 1.2:
            target = row['Close'] * 1.2
        return round(target, 2)
    return None

def fetch_fundamentals(symbol):
    try:
        smart_api = init_smartapi_client()
        if not smart_api:
            return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}
    except Exception:
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}

def generate_recommendations(data, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold",
        "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "Current Price": None, "Buy At": None,
        "Stop Loss": None, "Target": None, "Score": 0
    }

    if data.empty or len(data) < 27 or 'Close' not in data.columns or data['Close'].iloc[-1] is None:
        st.warning("⚠️ Insufficient data for recommendations.")
        return recommendations

    try:
        recommendations["Current Price"] = float(data['Close'].iloc[-1])
        buy_score = 0
        sell_score = 0

        # RSI-based signals
        if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None and len(data['RSI'].dropna()) >= 1:
            if isinstance(data['RSI'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['RSI'].iloc[-1] <= 20:
                    buy_score += 4
                elif data['RSI'].iloc[-1] < 30:
                    buy_score += 2
                elif data['RSI'].iloc[-1] > 70:
                    sell_score += 2

        # MACD-based signals
        if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None and data['MACD_signal'].iloc[-1] is not None and len(data['MACD'].dropna()) >= 1:
            if isinstance(data['MACD'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['MACD_signal'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
                    buy_score += 1
                elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
                    sell_score += 1

        # Bollinger Bands-based signals
        if 'Close' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns and data['Close'].iloc[-1] is not None and len(data['Lower_Band'].dropna()) >= 1:
            if isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Lower_Band'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Upper_Band'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]:
                    sell_score += 1

        # VWAP-based signals
        if 'VWAP' in data.columns and data['VWAP'].iloc[-1] is not None and data['Close'].iloc[-1] is not None and len(data['VWAP'].dropna()) >= 1:
            if isinstance(data['VWAP'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['VWAP'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] < data['VWAP'].iloc[-1]:
                    sell_score += 1

        # Volume-based signals
        if 'Volume' in data.columns and data['Volume'].iloc[-1] is not None and 'Avg_Volume' in data.columns and data['Avg_Volume'].iloc[-1] is not None and len(data['Volume'].dropna()) >= 2:
            volume_ratio = data['Volume'].iloc[-1] / data['Avg_Volume'].iloc[-1]
            if isinstance(volume_ratio, (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-2], (int, float, np.integer, np.floating)):
                if volume_ratio > 1.5 and data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                    buy_score += 2
                elif volume_ratio > 1.5 and data['Close'].iloc[-1] < data['Close'].iloc[-2]:
                    sell_score += 2
                elif volume_ratio < 0.5:
                    sell_score += 1

        # Volume Spike signals
        if 'Volume_Spike' in data.columns and data['Volume_Spike'].iloc[-1] is not None and len(data['Volume_Spike'].dropna()) >= 1:
            if data['Volume_Spike'].iloc[-1] and data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                buy_score += 2
            elif data['Volume_Spike'].iloc[-1] and data['Close'].iloc[-1] < data['Close'].iloc[-2]:
                sell_score += 2

        # Parabolic SAR signals
        if 'Parabolic_SAR' in data.columns and data['Parabolic_SAR'].iloc[-1] is not None and data['Close'].iloc[-1] is not None and len(data['Parabolic_SAR'].dropna()) >= 1:
            if isinstance(data['Parabolic_SAR'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['Parabolic_SAR'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] < data['Parabolic_SAR'].iloc[-1]:
                    sell_score += 1

        # Ichimoku Cloud signals
        if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns and data['Ichimoku_Span_A'].iloc[-1] is not None and data['Ichimoku_Span_B'].iloc[-1] is not None and len(data['Ichimoku_Span_A'].dropna()) >= 1:
            if isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Ichimoku_Span_B'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                    buy_score += 2
                    recommendations["Ichimoku_Trend"] = "Buy"
                elif data['Close'].iloc[-1] < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                    sell_score += 2
                    recommendations["Ichimoku_Trend"] = "Sell"

        # Divergence signals
        if 'Divergence' in data.columns and data['Divergence'].iloc[-1] is not None:
            if data['Divergence'].iloc[-1] == "Bullish Divergence":
                buy_score += 2
            elif data['Divergence'].iloc[-1] == "Bearish Divergence":
                sell_score += 2

        # Donchian Channels for Breakout
        if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns and data['Donchian_Upper'].iloc[-1] is not None and len(data['Donchian_Upper'].dropna()) >= 1:
            if isinstance(data['Donchian_Upper'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1]:
                    buy_score += 2
                    recommendations["Breakout"] = "Buy"
                elif data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1]:
                    sell_score += 2
                    recommendations["Breakout"] = "Sell"

        # Keltner Channels for Mean Reversion
        if 'Keltner_Upper' in data.columns and 'Keltner_Lower' in data.columns and data['Keltner_Upper'].iloc[-1] is not None and len(data['Keltner_Upper'].dropna()) >= 1:
            if isinstance(data['Keltner_Upper'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Keltner_Lower'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] < data['Keltner_Lower'].iloc[-1]:
                    buy_score += 2
                    recommendations["Mean_Reversion"] = "Buy"
                elif data['Close'].iloc[-1] > data['Keltner_Upper'].iloc[-1]:
                    sell_score += 2
                    recommendations["Mean_Reversion"] = "Sell"

        # Strategy Recommendations
        if buy_score >= 6:
            recommendations["Intraday"] = "Buy"
            recommendations["Swing"] = "Buy"
        elif sell_score >= 6:
            recommendations["Intraday"] = "Sell"
            recommendations["Swing"] = "Sell"

        if buy_score >= 4 and data['SMA_50'].iloc[-1] > data['SMA_200'].iloc[-1]:
            recommendations["Short-Term"] = "Buy"
        elif sell_score >= 4 and data['SMA_50'].iloc[-1] < data['SMA_200'].iloc[-1]:
            recommendations["Short-Term"] = "Sell"

        if buy_score >= 3 and data['Close'].iloc[-1] > data['SMA_200'].iloc[-1]:
            recommendations["Long-Term"] = "Buy"
        elif sell_score >= 3 and data['Close'].iloc[-1] < data['SMA_200'].iloc[-1]:
            recommendations["Long-Term"] = "Sell"

        # Calculate Buy At, Stop Loss, and Target
        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)
        recommendations["Score"] = calculate_confidence_score(data)

    except Exception as e:
        st.warning(f"⚠️ Error generating recommendations: {str(e)}")

    return recommendations

# Backtesting Function
def backtest_strategy(data, strategy="RSI", initial_cash=100000):
    cash = initial_cash
    position = 0
    trades = []
    portfolio_values = []
    buy_price = 0

    for i in range(1, len(data)):
        if strategy == "RSI":
            if data['RSI'].iloc[i] < 30 and cash > 0:
                shares = cash // data['Close'].iloc[i]
                position += shares
                cash -= shares * data['Close'].iloc[i]
                buy_price = data['Close'].iloc[i]
                trades.append(("BUY", data.index[i], shares, buy_price))
            elif data['RSI'].iloc[i] > 70 and position > 0:
                cash += position * data['Close'].iloc[i]
                trades.append(("SELL", data.index[i], position, data['Close'].iloc[i]))
                position = 0
        elif strategy == "MACD":
            if data['MACD'].iloc[i] > data['MACD_signal'].iloc[i] and data['MACD'].iloc[i-1] <= data['MACD_signal'].iloc[i-1] and cash > 0:
                shares = cash // data['Close'].iloc[i]
                position += shares
                cash -= shares * data['Close'].iloc[i]
                buy_price = data['Close'].iloc[i]
                trades.append(("BUY", data.index[i], shares, buy_price))
            elif data['MACD'].iloc[i] < data['MACD_signal'].iloc[i] and data['MACD'].iloc[i-1] >= data['MACD_signal'].iloc[i-1] and position > 0:
                cash += position * data['Close'].iloc[i]
                trades.append(("SELL", data.index[i], position, data['Close'].iloc[i]))
                position = 0

        portfolio_value = cash + position * data['Close'].iloc[i]
        portfolio_values.append({"Date": data.index[i], "Portfolio Value": portfolio_value})

    portfolio_df = pd.DataFrame(portfolio_values)
    portfolio_df.set_index("Date", inplace=True)
    return portfolio_df, trades

# Streamlit Application
def main():
    st.set_page_config(page_title="Stock Analysis & Paper Trading", layout="wide")
    st.title("📈 Stock Analysis & Paper Trading Dashboard")

    # Initialize database and paper trading
    init_database()
    paper_trading = PaperTrading(initial_cash=100000)

    # Sidebar
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Market Overview", "Stock Analysis", "Paper Trading", "Backtesting", "Portfolio"])

    # Market Overview
    if app_mode == "Market Overview":
        st.header("Market Overview")
        stock_list = fetch_nse_stock_list()
        ad_ratio = calculate_advance_decline_ratio(stock_list[:50])
        st.metric("Advance/Decline Ratio", f"{ad_ratio:.2f}")
        trending_stocks = get_trending_stocks()
        st.subheader("Trending Stocks")
        st.write(trending_stocks.head())

    # Stock Analysis
    elif app_mode == "Stock Analysis":
        st.header("Stock Analysis")
        sector = st.selectbox("Select Sector", list(SECTORS.keys()))
        symbol = st.selectbox("Select Stock", SECTORS[sector])
        period = st.selectbox("Select Period", ["1mo", "1y", "5y"])
        interval = st.selectbox("Select Interval", ["5m", "15m", "1h", "1d"])

        data = fetch_stock_data_cached(symbol, period, interval)
        if not data.empty:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data, symbol)

            st.subheader(f"Analysis for {symbol}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(tooltip("Current Price", TOOLTIPS["Score"]), f"₹{recommendations['Current Price']:.2f}")
                st.metric(tooltip("Buy At", "Suggested entry price"), f"₹{recommendations['Buy At']:.2f}" if recommendations['Buy At'] else "N/A")
                st.metric(tooltip("Stop Loss", TOOLTIPS["Stop Loss"]), f"₹{recommendations['Stop Loss']:.2f}" if recommendations['Stop Loss'] else "N/A")
                st.metric(tooltip("Target", "Profit target price"), f"₹{recommendations['Target']:.2f}" if recommendations['Target'] else "N/A")
            with col2:
                st.metric(tooltip("Score", TOOLTIPS["Score"]), f"{recommendations['Score']:.2f}")
                st.write("**Recommendations**")
                st.write(f"- Intraday: {recommendations['Intraday']}")
                st.write(f"- Swing: {recommendations['Swing']}")
                st.write(f"- Short-Term: {recommendations['Short-Term']}")
                st.write(f"- Long-Term: {recommendations['Long-Term']}")

            # Plot Price and Indicators
            fig = px.line(data, x=data.index, y="Close", title=f"{symbol} Price")
            fig.add_scatter(x=data.index, y=data['SMA_50'], name="SMA 50", line=dict(color='orange'))
            fig.add_scatter(x=data.index, y=data['SMA_200'], name="SMA 200", line=dict(color='red'))
            st.plotly_chart(fig)

            fig_rsi = px.line(data, x=data.index, y="RSI", title="RSI")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            st.plotly_chart(fig_rsi)

            # Monte Carlo Simulation
            simulations = monte_carlo_simulation(data)
            sim_df = pd.DataFrame(simulations).T
            fig_mc = px.line(sim_df, title="Monte Carlo Price Simulations")
            st.plotly_chart(fig_mc)

    # Paper Trading
    elif app_mode == "Paper Trading":
        st.header("Paper Trading")
        symbol = st.selectbox("Select Stock", list(set([stock for sector in SECTORS.values() for stock in sector])))
        order_type = st.selectbox("Order Type", ["BUY", "SELL"])
        quantity = st.number_input("Quantity", min_value=1, value=1)
        price = st.number_input("Price", min_value=0.0, value=fetch_market_data(symbol)['ltp'] if fetch_market_data(symbol) else 0.0)
        strategy = st.selectbox("Strategy", ["Manual", "RSI", "MACD", "Breakout"])
        stop_loss = st.number_input("Stop Loss (optional)", min_value=0.0, value=0.0)
        target = st.number_input("Target (optional)", min_value=0.0, value=0.0)

        if st.button("Place Order"):
            stop_loss = stop_loss if stop_loss > 0 else None
            target = target if target > 0 else None
            paper_trading.place_order(symbol, order_type, quantity, price, strategy, stop_loss, target)

        # Display Open Positions
        conn = sqlite3.connect('stock_picks.db')
        positions = pd.read_sql("SELECT * FROM paper_positions", conn)
        if not positions.empty:
            st.subheader("Open Positions")
            st.dataframe(positions)
        conn.close()

    # Backtesting
    elif app_mode == "Backtesting":
        st.header("Backtesting")
        symbol = st.selectbox("Select Stock", list(set([stock for sector in SECTORS.values() for stock in sector])))
        strategy = st.selectbox("Strategy", ["RSI", "MACD"])
        initial_cash = st.number_input("Initial Cash", min_value=1000, value=100000)

        data = fetch_stock_data_cached(symbol, "5y", "1d")
        if not data.empty:
            data = analyze_stock(data)
            portfolio_df, trades = backtest_strategy(data, strategy, initial_cash)
            st.subheader(f"Backtest Results for {symbol} ({strategy})")
            st.line_chart(portfolio_df["Portfolio Value"])
            st.subheader("Trades")
            st.write(pd.DataFrame(trades, columns=["Action", "Date", "Shares", "Price"]))

    # Portfolio
    elif app_mode == "Portfolio":
        st.header("Portfolio")
        conn = sqlite3.connect('stock_picks.db')
        portfolio = pd.read_sql("SELECT * FROM paper_portfolio WHERE date = ?", conn, params=(datetime.now().strftime('%Y-%m-%d'),))
        positions = pd.read_sql("SELECT * FROM paper_positions", conn)
        orders = pd.read_sql("SELECT * FROM paper_orders ORDER BY execution_time DESC", conn)
        conn.close()

        if not portfolio.empty:
            st.metric("Cash Balance", f"₹{portfolio['cash_balance'].iloc[0]:.2f}")
            st.metric("Total P&L", f"₹{portfolio['total_pnl'].iloc[0]:.2f}")
            st.metric("Realized P&L", f"₹{portfolio['realized_pnl'].iloc[0]:.2f}")
            st.metric("Unrealized P&L", f"₹{portfolio['unrealized_pnl'].iloc[0]:.2f}")

        if not positions.empty:
            st.subheader("Open Positions")
            st.dataframe(positions)

        if not orders.empty:
            st.subheader("Order History")
            st.dataframe(orders)

    # Cleanup
    paper_trading.stop_websocket()

if __name__ == "__main__":
    main()