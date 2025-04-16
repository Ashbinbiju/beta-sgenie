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
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Android 11; Mobile; rv:89.0) Gecko/89.0 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 14_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/92.0.902.67 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:92.0) Gecko/20100101 Firefox/92.0",
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
    "TRIX": "Triple Exponential Average - Momentum oscillator with triple smoothing",
    "Ultimate_Osc": "Ultimate Oscillator - Combines short, medium, and long-term momentum",
    "CMO": "Chande Momentum Oscillator - Measures raw momentum (-100 to 100)",
    "VPT": "Volume Price Trend - Tracks trend strength with price and volume",
    "Pivot Points": "Support and resistance levels based on previous day's prices",
    "Heikin-Ashi": "Smoothed candlestick chart to identify trends"
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
        "QUICKHEAL.NS", "CIGNITITEC.NS", "ALLSEC.NS"
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
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "VBL.NS", "BRITANNIA.NS",
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
        "VGUARD.NS", "WABAG.NS"
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

SECTOR_PE_AVG = {
    "Bank": 20, "Software & IT Services": 35, "Finance": 25, "Automobile & Ancillaries": 18,
    "Healthcare": 30, "Metals & Mining": 15, "FMCG": 40, "Power": 12, "Capital Goods": 22,
    "Oil & Gas": 10, "Chemicals": 25, "Telecom": 28, "Infrastructure": 15, "Insurance": 30,
    "Diversified": 20, "Construction Materials": 18, "Real Estate": 25, "Aviation": 22,
    "Retailing": 35, "Miscellaneous": 20
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
                        raise e
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
        return stock_list
    except Exception:
        return list(set([stock for sector in SECTORS.values() for stock in sector]))

def fetch_stock_data_with_auth(symbol, period="5y", interval="1d"):
    try:
        if ".NS" not in symbol:
            symbol += ".NS"
        session = requests.Session()
        session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
        stock = yf.Ticker(symbol, session=session)
        time.sleep(random.uniform(1, 3))
        data = stock.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        return data
    except Exception as e:
        st.warning(f"⚠️ Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

@lru_cache(maxsize=100)
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    return fetch_stock_data_with_auth(symbol, period, interval)

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
            if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns and data['Close'].iloc[-1] is not None and data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
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
    best_window, best_sharpe = 14, -float('inf')
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 100:
        st.warning("⚠️ Insufficient data for RSI optimization (<100 periods). Using default window=14.")
        return best_window
    for window in windows:
        try:
            rsi = ta.momentum.RSIIndicator(data['Close'], window=window).rsi()
            signals = (rsi < 30).astype(int) - (rsi > 70).astype(int)
            strategy_returns = signals.shift(1) * returns
            sharpe = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
            if sharpe > best_sharpe:
                best_sharpe, best_window = sharpe, window
        except Exception as e:
            st.warning(f"⚠️ Failed RSI optimization for window={window}: {str(e)}")
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
        return pd.Series([None] * len(close), index=close.index)

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
            'Pivot': pivot,
            'Support1': support1,
            'Resistance1': resistance1,
            'Support2': support2,
            'Resistance2': resistance2
        }
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Pivot Points: {str(e)}")
        return None

def calculate_heikin_ashi(data):
    try:
        ha_data = pd.DataFrame(index=data.index)
        ha_data['HA_Close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
        ha_data['HA_Open'] = data['Open'].copy()
        for i in range(1, len(ha_data)):
            ha_data['HA_Open'].iloc[i] = (ha_data['HA_Open'].iloc[i-1] + ha_data['HA_Close'].iloc[i-1]) / 2
        ha_data['HA_High'] = pd.concat([data['High'], ha_data['HA_Open'], ha_data['HA_Close']], axis=1).max(axis=1)
        ha_data['HA_Low'] = pd.concat([data['Low'], ha_data['HA_Open'], ha_data['HA_Close']], axis=1).min(axis=1)
        return ha_data[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Heikin-Ashi: {str(e)}")
        return None

def analyze_stock(data):
    if data.empty or len(data) < 100:
        st.warning("⚠️ Insufficient data: less than 100 periods.")
        return None
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
        return None

    try:
        rsi_window = optimize_rsi_window(data)
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute RSI: {str(e)}")
        data['RSI'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_hist'] = macd.macd_diff()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute MACD: {str(e)}")
        data['MACD'] = pd.Series([None] * len(data), index=data.index)
        data['MACD_signal'] = pd.Series([None] * len(data), index=data.index)
        data['MACD_hist'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Moving Averages: {str(e)}")
        data['SMA_50'] = pd.Series([None] * len(data), index=data.index)
        data['SMA_200'] = pd.Series([None] * len(data), index=data.index)
        data['EMA_20'] = pd.Series([None] * len(data), index=data.index)
        data['EMA_50'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband()
        data['Middle_Band'] = bollinger.bollinger_mavg()
        data['Lower_Band'] = bollinger.bollinger_lband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Bollinger Bands: {str(e)}")
        data['Upper_Band'] = pd.Series([None] * len(data), index=data.index)
        data['Middle_Band'] = pd.Series([None] * len(data), index=data.index)
        data['Lower_Band'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
        data['SlowK'] = stoch.stoch()
        data['SlowD'] = stoch.stoch_signal()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Stochastic: {str(e)}")
        data['SlowK'] = pd.Series([None] * len(data), index=data.index)
        data['SlowD'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute ATR: {str(e)}")
        data['ATR'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute ADX: {str(e)}")
        data['ADX'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute OBV: {str(e)}")
        data['OBV'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        data['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['VWAP'] = data['Cumulative_TP'].cumsum() / data['Cumulative_Volume']
    except Exception as e:
        st.warning(f"⚠️ Failed to compute VWAP: {str(e)}")
        data['VWAP'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        data['Avg_Volume'] = data['Volume'].rolling(window=20).mean()
        volume_std = data['Volume'].rolling(window=20).std()
        dynamic_multiplier = 1.5 + (volume_std.iloc[-1] / data['Avg_Volume'].iloc[-1]) if data['Avg_Volume'].iloc[-1] != 0 else 2.0
        data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * dynamic_multiplier)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Volume Spike: {str(e)}")
        data['Volume_Spike'] = pd.Series([False] * len(data), index=data.index)
    
    try:
        data['Parabolic_SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Parabolic SAR: {str(e)}")
        data['Parabolic_SAR'] = pd.Series([None] * len(data), index=data.index)
    
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
        data['Fib_23.6'] = pd.Series([None] * len(data), index=data.index)
        data['Fib_38.2'] = pd.Series([None] * len(data), index=data.index)
        data['Fib_50.0'] = pd.Series([None] * len(data), index=data.index)
        data['Fib_61.8'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        data['Divergence'] = detect_divergence(data)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Divergence: {str(e)}")
        data['Divergence'] = pd.Series(["No Divergence"] * len(data), index=data.index)
    
    try:
        ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
        data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
        data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
        data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
        data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
        data['Ichimoku_Chikou'] = data['Close'].shift(-26)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Ichimoku: {str(e)}")
        data['Ichimoku_Tenkan'] = pd.Series([None] * len(data), index=data.index)
        data['Ichimoku_Kijun'] = pd.Series([None] * len(data), index=data.index)
        data['Ichimoku_Span_A'] = pd.Series([None] * len(data), index=data.index)
        data['Ichimoku_Span_B'] = pd.Series([None] * len(data), index=data.index)
        data['Ichimoku_Chikou'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute CMF: {str(e)}")
        data['CMF'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
        data['Donchian_Upper'] = donchian.donchian_channel_hband()
        data['Donchian_Lower'] = donchian.donchian_channel_lband()
        data['Donchian_Middle'] = donchian.donchian_channel_mband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Donchian: {str(e)}")
        data['Donchian_Upper'] = pd.Series([None] * len(data), index=data.index)
        data['Donchian_Lower'] = pd.Series([None] * len(data), index=data.index)
        data['Donchian_Middle'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        keltner = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'], window=20, window_atr=10)
        data['Keltner_Upper'] = keltner.keltner_channel_hband()
        data['Keltner_Middle'] = keltner.keltner_channel_mband()
        data['Keltner_Low'] = keltner.keltner_channel_lband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Keltner Channels: {str(e)}")
        data['Keltner_Upper'] = pd.Series([None] * len(data), index=data.index)
        data['Keltner_Middle'] = pd.Series([None] * len(data), index=data.index)
        data['Keltner_Low'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        data['TRIX'] = ta.trend.TRIXIndicator(data['Close'], window=15).trix()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute TRIX: {str(e)}")
        data['TRIX'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        data['Ultimate_Osc'] = ta.momentum.UltimateOscillator(
            data['High'], data['Low'], data['Close'], window1=7, window2=14, window3=28
        ).ultimate_oscillator()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Ultimate Oscillator: {str(e)}")
        data['Ultimate_Osc'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        data['CMO'] = calculate_cmo(data['Close'], window=14)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Chande Momentum Oscillator: {str(e)}")
        data['CMO'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        data['VPT'] = ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume']).volume_price_trend()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Volume Price Trend: {str(e)}")
        data['VPT'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        pivot_points = calculate_pivot_points(data)
        if pivot_points:
            data['Pivot'] = pivot_points['Pivot']
            data['Support1'] = pivot_points['Support1']
            data['Resistance1'] = pivot_points['Resistance1']
            data['Support2'] = pivot_points['Support2']
            data['Resistance2'] = pivot_points['Resistance2']
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Pivot Points: {str(e)}")
        data['Pivot'] = pd.Series([None] * len(data), index=data.index)
        data['Support1'] = pd.Series([None] * len(data), index=data.index)
        data['Resistance1'] = pd.Series([None] * len(data), index=data.index)
        data['Support2'] = pd.Series([None] * len(data), index=data.index)
        data['Resistance2'] = pd.Series([None] * len(data), index=data.index)
    
    try:
        ha_data = calculate_heikin_ashi(data)
        if ha_data is not None:
            data['HA_Open'] = ha_data['HA_Open']
            data['HA_High'] = ha_data['HA_High']
            data['HA_Low'] = ha_data['HA_Low']
            data['HA_Close'] = ha_data['HA_Close']
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Heikin-Ashi: {str(e)}")
        data['HA_Open'] = pd.Series([None] * len(data), index=data.index)
        data['HA_High'] = pd.Series([None] * len(data), index=data.index)
        data['HA_Low'] = pd.Series([None] * len(data), index=data.index)
        data['HA_Close'] = pd.Series([None] * len(data), index=data.index)
    
    return data

def calculate_buy_at(data):
    if data is None or 'RSI' not in data.columns or data['RSI'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Buy At due to missing or invalid RSI data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    buy_at = last_close * 0.99 if last_rsi < 30 else last_close
    return round(buy_at, 2)

def calculate_stop_loss(data, atr_multiplier=2.5):
    if data is None or 'ATR' not in data.columns or data['ATR'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Stop Loss due to missing or invalid ATR data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    atr_std = data['ATR'].std() if not data['ATR'].std() == 0 else 1
    atr_multiplier = 3.0 if data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25 else 1.5
    atr_multiplier += (last_atr / atr_std) * 0.5
    atr_stop = last_close - (atr_multiplier * last_atr)
    lookback = 20
    swing_low = data['Low'][-lookback:].min()
    support_level = data['Support1'].iloc[-1] if 'Support1' in data.columns and not pd.isna(data['Support1'].iloc[-1]) else swing_low
    stop_loss = max(atr_stop, support_level * 0.99)
    if stop_loss < last_close * 0.85:
        stop_loss = last_close * 0.85
    return round(stop_loss, 2)

def calculate_target(data, risk_reward_ratio=3):
    if data is None:
        st.warning("⚠️ Cannot calculate Target due to missing data.")
        return None
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        st.warning("⚠️ Cannot calculate Target due to missing Stop Loss data.")
        return None
    last_close = data['Close'].iloc[-1]
    risk = last_close - stop_loss
    adjusted_ratio = min(risk_reward_ratio, 5) if data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 30 else min(risk_reward_ratio, 3)
    target = last_close + (risk * adjusted_ratio)
    max_target = last_close * 1.3 if data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 30 else last_close * 1.2
    if target > max_target:
        target = max_target
    return round(target, 2)

@lru_cache(maxsize=100)
def fetch_fundamentals(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        pe = min(info.get('trailingPE', 100), 100) if info.get('trailingPE', 100) != float('inf') else 100
        sector = next((s for s, stocks in SECTORS.items() if symbol in stocks), "Miscellaneous")
        return {
            'P/E': pe,
            'EPS': info.get('trailingEps', 0),
            'RevenueGrowth': info.get('revenueGrowth', 0),
            'Sector': sector
        }
    except Exception:
        return {'P/E': 100, 'EPS': 0, 'RevenueGrowth': 0, 'Sector': "Miscellaneous"}

def generate_recommendations(data, symbol=None):
    default_recommendations = {
        "Intraday": "N/A", "Swing": "N/A", "Short-Term": "N/A", "Long-Term": "N/A",
        "Mean_Reversion": "N/A", "Breakout": "N/A", "Ichimoku_Trend": "N/A",
        "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None,
        "Score": 0, "Net_Score": 0
    }
    
    if data is None or data.empty or 'Close' not in data.columns or data['Close'].iloc[-1] is None:
        return default_recommendations
    
    buy_score = 0
    sell_score = 0
    conflicts = []
    
    if ('RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]) and 0 <= data['RSI'].iloc[-1] <= 100):
        rsi = data['RSI'].iloc[-1]
        if rsi <= 20:
            buy_score += 2
        elif rsi >= 80:
            sell_score += 2
    
    if ('MACD' in data.columns and 'MACD_signal' in data.columns and 
        not pd.isna(data['MACD'].iloc[-1]) and not pd.isna(data['MACD_signal'].iloc[-1])):
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_signal'].iloc[-1]
        if macd > macd_signal:
            buy_score += 1
        elif macd < macd_signal:
            sell_score += 1
        if ('RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1])):
            rsi = data['RSI'].iloc[-1]
            if rsi < 30 and macd < macd_signal:
                conflicts.append("RSI oversold vs MACD bearish")
                buy_score -= 0.5
            elif rsi > 70 and macd > macd_signal:
                conflicts.append("RSI overbought vs MACD bullish")
                sell_score -= 0.5
    
    if ('MACD_hist' in data.columns and not pd.isna(data['MACD_hist'].iloc[-1]) and 
        not pd.isna(data['MACD_hist'].iloc[-2])):
        hist = data['MACD_hist'].iloc[-1]
        prev_hist = data['MACD_hist'].iloc[-2]
        if hist > 0 and prev_hist < 0:
            buy_score += 1
        elif hist < 0 and prev_hist > 0:
            sell_score += 1
        elif hist > 0 and hist > prev_hist:
            buy_score += 0.5
        elif hist < 0 and hist < prev_hist:
            sell_score += 0.5
    
    if ('Lower_Band' in data.columns and 'Upper_Band' in data.columns and 
        not pd.isna(data['Close'].iloc[-1]) and not pd.isna(data['Lower_Band'].iloc[-1]) and 
        not pd.isna(data['Upper_Band'].iloc[-1])):
        close = data['Close'].iloc[-1]
        lower = data['Lower_Band'].iloc[-1]
        upper = data['Upper_Band'].iloc[-1]
        if close < lower:
            buy_score += 1
        elif close > upper:
            sell_score += 1
    
    if 'VWAP' in data.columns and not pd.isna(data['VWAP'].iloc[-1]) and not pd.isna(data['Close'].iloc[-1]):
        vwap = data['VWAP'].iloc[-1]
        close = data['Close'].iloc[-1]
        if close > vwap:
            buy_score += 1
        elif close < vwap:
            sell_score += 1
    
    if ('Volume' in data.columns and 'Avg_Volume' in data.columns and 
        not pd.isna(data['Volume'].iloc[-1]) and not pd.isna(data['Avg_Volume'].iloc[-1])):
        volume_ratio = data['Volume'].iloc[-1] / data['Avg_Volume'].iloc[-1]
        close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        if volume_ratio > 1.5 and close > prev_close:
            buy_score += 2
        elif volume_ratio > 1.5 and close < prev_close:
            sell_score += 2
        elif volume_ratio < 0.5:
            sell_score += 1
    
    if 'Volume_Spike' in data.columns and not pd.isna(data['Volume_Spike'].iloc[-1]):
        spike = data['Volume_Spike'].iloc[-1]
        close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        if spike and close > prev_close:
            buy_score += 1
        elif spike and close < prev_close:
            sell_score += 1
    
    if 'Divergence' in data.columns:
        divergence = data['Divergence'].iloc[-1]
        if divergence == "Bullish Divergence":
            buy_score += 1
        elif divergence == "Bearish Divergence":
            sell_score += 1
    
    if ('Ichimoku_Tenkan' in data.columns and 'Ichimoku_Kijun' in data.columns and 
        'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns and
        'Ichimoku_Chikou' in data.columns and 
        not pd.isna(data['Ichimoku_Tenkan'].iloc[-1]) and 
        not pd.isna(data['Ichimoku_Kijun'].iloc[-1]) and 
        not pd.isna(data['Ichimoku_Span_A'].iloc[-1]) and
        not pd.isna(data['Ichimoku_Span_B'].iloc[-1]) and
        not pd.isna(data['Ichimoku_Chikou'].iloc[-1])):
        tenkan = data['Ichimoku_Tenkan'].iloc[-1]
        kijun = data['Ichimoku_Kijun'].iloc[-1]
        span_a = data['Ichimoku_Span_A'].iloc[-1]
        span_b = data['Ichimoku_Span_B'].iloc[-1]
        chikou = data['Ichimoku_Chikou'].iloc[-1]
        close = data['Close'].iloc[-1]
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        if tenkan > kijun and close > cloud_top and chikou > close:
            buy_score += 2
            default_recommendations["Ichimoku_Trend"] = "Strong Buy"
        elif tenkan < kijun and close < cloud_bottom and chikou < close:
            sell_score += 2
            default_recommendations["Ichimoku_Trend"] = "Strong Sell"
    
    if 'CMF' in data.columns and not pd.isna(data['CMF'].iloc[-1]):
        cmf = data['CMF'].iloc[-1]
        if cmf > 0:
            buy_score += 1
        elif cmf < 0:
            sell_score += 1
    
    if ('Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns and 
        not pd.isna(data['Donchian_Upper'].iloc[-1]) and not pd.isna(data['Donchian_Lower'].iloc[-1])):
        close = data['Close'].iloc[-1]
        upper = data['Donchian_Upper'].iloc[-1]
        lower = data['Donchian_Lower'].iloc[-1]
        if close > upper:
            buy_score += 1
            default_recommendations["Breakout"] = "Buy"
        elif close < lower:
            sell_score += 1
            default_recommendations["Breakout"] = "Sell"
    
    if ('RSI' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns and 
        not pd.isna(data['RSI'].iloc[-1]) and not pd.isna(data['Lower_Band'].iloc[-1]) and 
        not pd.isna(data['Upper_Band'].iloc[-1])):
        rsi = data['RSI'].iloc[-1]
        close = data['Close'].iloc[-1]
        lower = data['Lower_Band'].iloc[-1]
        upper = data['Upper_Band'].iloc[-1]
        if rsi < 30 and close >= lower:
            buy_score += 2
            default_recommendations["Mean_Reversion"] = "Buy"
        elif rsi > 70 and close >= upper:
            sell_score += 2
            default_recommendations["Mean_Reversion"] = "Sell"
    
    if ('Keltner_Upper' in data.columns and 'Keltner_Low' in data.columns and 
        not pd.isna(data['Keltner_Upper'].iloc[-1]) and not pd.isna(data['Keltner_Low'].iloc[-1])):
        close = data['Close'].iloc[-1]
        upper = data['Keltner_Upper'].iloc[-1]
        lower = data['Keltner_Low'].iloc[-1]
        if close < lower:
            buy_score += 1
        elif close > upper:
            sell_score += 1
    
    if 'TRIX' in data.columns and not pd.isna(data['TRIX'].iloc[-1]) and not pd.isna(data['TRIX'].iloc[-2]):
        trix = data['TRIX'].iloc[-1]
        prev_trix = data['TRIX'].iloc[-2]
        if trix > 0 and trix > prev_trix:
            buy_score += 1
        elif trix < 0 and trix < prev_trix:
            sell_score += 1
    
    if 'Ultimate_Osc' in data.columns and not pd.isna(data['Ultimate_Osc'].iloc[-1]):
        uo = data['Ultimate_Osc'].iloc[-1]
        if uo < 30:
            buy_score += 1
        elif uo > 70:
            sell_score += 1
    
    if 'CMO' in data.columns and not pd.isna(data['CMO'].iloc[-1]):
        cmo = data['CMO'].iloc[-1]
        if cmo < -50:
            buy_score += 1
        elif cmo > 50:
            sell_score += 1
    
    if 'VPT' in data.columns and not pd.isna(data['VPT'].iloc[-1]) and not pd.isna(data['VPT'].iloc[-2]):
        vpt = data['VPT'].iloc[-1]
        prev_vpt = data['VPT'].iloc[-2]
        if vpt > prev_vpt:
            buy_score += 1
        elif vpt < prev_vpt:
            sell_score += 1
    
    if ('Fib_23.6' in data.columns and 'Fib_38.2' in data.columns and 
        not pd.isna(data['Fib_23.6'].iloc[-1]) and not pd.isna(data['Fib_38.2'].iloc[-1])):
        current_price = data['Close'].iloc[-1]
        fib_levels = [data['Fib_23.6'].iloc[-1], data['Fib_38.2'].iloc[-1], 
                      data['Fib_50.0'].iloc[-1], data['Fib_61.8'].iloc[-1]]
        for level in fib_levels:
            if not pd.isna(level) and abs(current_price - level) / current_price < 0.01:
                if current_price > level:
                    buy_score += 1
                else:
                    sell_score += 1
    
    if 'Parabolic_SAR' in data.columns and not pd.isna(data['Parabolic_SAR'].iloc[-1]):
        sar = data['Parabolic_SAR'].iloc[-1]
        close = data['Close'].iloc[-1]
        if close > sar:
            buy_score += 1
        elif close < sar:
            sell_score += 1
    
    if 'OBV' in data.columns and not pd.isna(data['OBV'].iloc[-1]) and not pd.isna(data['OBV'].iloc[-2]):
        obv = data['OBV'].iloc[-1]
        prev_obv = data['OBV'].iloc[-2]
        if obv > prev_obv:
            buy_score += 1
        elif obv < prev_obv:
            sell_score += 1
    
    if ('Pivot' in data.columns and 'Support1' in data.columns and 'Resistance1' in data.columns and 
        not pd.isna(data['Pivot'].iloc[-1]) and not pd.isna(data['Support1'].iloc[-1]) and 
        not pd.isna(data['Resistance1'].iloc[-1])):
        close = data['Close'].iloc[-1]
        pivot = data['Pivot'].iloc[-1]
        support1 = data['Support1'].iloc[-1]
        resistance1 = data['Resistance1'].iloc[-1]
        if abs(close - support1) / close < 0.01:
            buy_score += 1
        elif abs(close - resistance1) / close < 0.01:
            sell_score += 1
    
    if ('HA_Close' in data.columns and 'HA_Open' in data.columns and 
        not pd.isna(data['HA_Close'].iloc[-1]) and not pd.isna(data['HA_Open'].iloc[-1]) and
        not pd.isna(data['HA_Close'].iloc[-2]) and not pd.isna(data['HA_Open'].iloc[-2])):
        ha_close = data['HA_Close'].iloc[-1]
        ha_open = data['HA_Open'].iloc[-1]
        prev_ha_close = data['HA_Close'].iloc[-2]
        prev_ha_open = data['HA_Open'].iloc[-2]
        if ha_close > ha_open and prev_ha_close > prev_ha_open:
            buy_score += 1
        elif ha_close < ha_open and prev_ha_close < prev_ha_open:
            sell_score += 1
    
    if symbol:
        fundamentals = fetch_fundamentals(symbol)
        sector_pe = SECTOR_PE_AVG.get(fundamentals['Sector'], 20)
        if fundamentals['P/E'] < sector_pe * 0.75 and fundamentals['EPS'] > 0:
            buy_score += 2
        elif fundamentals['P/E'] > sector_pe * 1.25 or fundamentals['EPS'] < 0:
            sell_score += 2
        if fundamentals['RevenueGrowth'] > 0.1:
            buy_score += 1
        elif fundamentals['RevenueGrowth'] < 0:
            sell_score += 1
    
    buy_score = min(buy_score, 20)
    sell_score = min(sell_score, 20)
    if buy_score == 0 and sell_score == 0:
        net_score = 0
    else:
        total_signals = max(buy_score + sell_score, 5)
        net_score = (buy_score - sell_score) / total_signals * 5
    if total_signals < 5:
        st.warning(f"Low signal count for {symbol}: buy_score={buy_score}, sell_score={sell_score}")
    if conflicts:
        st.warning(f"Indicator conflicts for {symbol}: {', '.join(conflicts)}")
    
    if net_score >= 3:
        default_recommendations["Intraday"] = "Strong Buy"
        default_recommendations["Swing"] = "Buy" if net_score >= 2 else "Hold"
        default_recommendations["Short-Term"] = "Buy" if net_score >= 1.5 else "Hold"
        default_recommendations["Long-Term"] = "Buy" if net_score >= 1 else "Hold"
    elif net_score >= 1:
        default_recommendations["Intraday"] = "Buy"
        default_recommendations["Swing"] = "Hold"
        default_recommendations["Short-Term"] = "Hold"
        default_recommendations["Long-Term"] = "Hold"
    elif net_score <= -3:
        default_recommendations["Intraday"] = "Strong Sell"
        default_recommendations["Swing"] = "Sell" if net_score <= -2 else "Hold"
        default_recommendations["Short-Term"] = "Sell" if net_score <= -1.5 else "Hold"
        default_recommendations["Long-Term"] = "Sell" if net_score <= -1 else "Hold"
    elif net_score <= -1:
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
    
    return default_recommendations

def analyze_batch(stock_batch):
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                errors.append(f"⚠️ Error processing stock {symbol}: {str(e)}")
    for error in errors:
        st.warning(error)
    return results

def analyze_stock_parallel(symbol):
    data = fetch_stock_data_cached(symbol)
    if not data.empty:
        data = analyze_stock(data)
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
            "Short-Term": recommendations["Short-Term"],
            "Long-Term": recommendations["Long-Term"],
            "Mean_Reversion": recommendations["Mean_Reversion"],
            "Breakout": recommendations["Breakout"],
            "Ichimoku_Trend": recommendations["Ichimoku_Trend"],
            "Score": recommendations["Score"],
            "Net_Score": recommendations["Net_Score"]
        }
    return None

def normalize_scores(scores):
    if not scores:
        return scores
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.5 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]

def plot_stock_data(data, symbol, indicators, show_candlestick=False):
    if data.empty or 'Close' not in data.columns:
        st.error(f"⚠️ No valid data to plot for {symbol}")
        return

    fig = px.line(data, x=data.index, y='Close', title=f"{symbol} Price and Indicators")
    
    if show_candlestick:
        fig.data = []
        fig.add_candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Candlestick'
        )
    
    for indicator in indicators:
        if indicator == 'RSI' and 'RSI' in data.columns:
            fig.add_scatter(x=data.index, y=data['RSI'], name=tooltip("RSI", TOOLTIPS["RSI"]), yaxis='y2')
        elif indicator == 'MACD' and 'MACD' in data.columns:
            fig.add_scatter(x=data.index, y=data['MACD'], name=tooltip("MACD", TOOLTIPS["MACD"]), yaxis='y3')
            fig.add_scatter(x=data.index, y=data['MACD_signal'], name='MACD Signal', yaxis='y3')
        elif indicator == 'Bollinger' and 'Upper_Band' in data.columns:
            fig.add_scatter(x=data.index, y=data['Upper_Band'], name=tooltip("Upper Band", TOOLTIPS["Bollinger"]), line=dict(dash='dash'))
            fig.add_scatter(x=data.index, y=data['Lower_Band'], name=tooltip("Lower Band", TOOLTIPS["Bollinger"]), line=dict(dash='dash'))
            fig.add_scatter(x=data.index, y=data['Middle_Band'], name=tooltip("Middle Band", TOOLTIPS["Bollinger"]))
        elif indicator == 'ATR' and 'ATR' in data.columns:
            fig.add_scatter(x=data.index, y=data['ATR'], name=tooltip("ATR", TOOLTIPS["ATR"]), yaxis='y4')
        elif indicator == 'ADX' and 'ADX' in data.columns:
            fig.add_scatter(x=data.index, y=data['ADX'], name=tooltip("ADX", TOOLTIPS["ADX"]), yaxis='y5')
        elif indicator == 'VWAP' and 'VWAP' in data.columns:
            fig.add_scatter(x=data.index, y=data['VWAP'], name=tooltip("VWAP", TOOLTIPS["VWAP"]), line=dict(color='purple'))
        elif indicator == 'Parabolic_SAR' and 'Parabolic_SAR' in data.columns:
            fig.add_scatter(x=data.index, y=data['Parabolic_SAR'], name=tooltip("Parabolic SAR", TOOLTIPS["Parabolic_SAR"]), mode='markers')
        elif indicator == 'Fib_Retracements' and 'Fib_23.6' in data.columns:
            for level in ['Fib_23.6', 'Fib_38.2', 'Fib_50.0', 'Fib_61.8']:
                if level in data.columns and not data[level].isna().all():
                    fig.add_scatter(x=data.index, y=data[level], name=f"{level.replace('Fib_', '')}%", line=dict(dash='dot'))
        elif indicator == 'Ichimoku' and 'Ichimoku_Span_A' in data.columns:
            fig.add_scatter(x=data.index, y=data['Ichimoku_Tenkan'], name='Tenkan-sen', line=dict(color='blue'))
            fig.add_scatter(x=data.index, y=data['Ichimoku_Kijun'], name='Kijun-sen', line=dict(color='red'))
            fig.add_scatter(x=data.index, y=data['Ichimoku_Span_A'], name='Senkou Span A', line=dict(color='green'))
            fig.add_scatter(x=data.index, y=data['Ichimoku_Span_B'], name='Senkou Span B', line=dict(color='red'))
            fig.add_scatter(x=data.index, y=data['Ichimoku_Chikou'], name='Chikou Span', line=dict(color='purple'))
        elif indicator == 'CMF' and 'CMF' in data.columns:
            fig.add_scatter(x=data.index, y=data['CMF'], name=tooltip("CMF", TOOLTIPS["CMF"]), yaxis='y6')
        elif indicator == 'Donchian' and 'Donchian_Upper' in data.columns:
            fig.add_scatter(x=data.index, y=data['Donchian_Upper'], name='Donchian Upper', line=dict(dash='dash'))
            fig.add_scatter(x=data.index, y=data['Donchian_Lower'], name='Donchian Lower', line=dict(dash='dash'))
            fig.add_scatter(x=data.index, y=data['Donchian_Middle'], name='Donchian Middle')
        elif indicator == 'Keltner' and 'Keltner_Upper' in data.columns:
            fig.add_scatter(x=data.index, y=data['Keltner_Upper'], name='Keltner Upper', line=dict(dash='dash'))
            fig.add_scatter(x=data.index, y=data['Keltner_Low'], name='Keltner Low', line=dict(dash='dash'))
            fig.add_scatter(x=data.index, y=data['Keltner_Middle'], name='Keltner Middle')
        elif indicator == 'TRIX' and 'TRIX' in data.columns:
            fig.add_scatter(x=data.index, y=data['TRIX'], name=tooltip("TRIX", TOOLTIPS["TRIX"]), yaxis='y7')
        elif indicator == 'Ultimate_Osc' and 'Ultimate_Osc' in data.columns:
            fig.add_scatter(x=data.index, y=data['Ultimate_Osc'], name=tooltip("Ultimate Osc", TOOLTIPS["Ultimate_Osc"]), yaxis='y8')
        elif indicator == 'CMO' and 'CMO' in data.columns:
            fig.add_scatter(x=data.index, y=data['CMO'], name=tooltip("CMO", TOOLTIPS["CMO"]), yaxis='y9')
        elif indicator == 'VPT' and 'VPT' in data.columns:
            fig.add_scatter(x=data.index, y=data['VPT'], name=tooltip("VPT", TOOLTIPS["VPT"]), yaxis='y10')
        elif indicator == 'Pivot Points' and 'Pivot' in data.columns:
            for level in ['Pivot', 'Support1', 'Resistance1', 'Support2', 'Resistance2']:
                if level in data.columns and not data[level].isna().all():
                    fig.add_scatter(x=data.index, y=data[level], name=level, line=dict(dash='dot'))
        elif indicator == 'Heikin-Ashi' and 'HA_Close' in data.columns:
            fig.add_candlestick(
                x=data.index,
                open=data['HA_Open'],
                high=data['HA_High'],
                low=data['HA_Low'],
                close=data['HA_Close'],
                name='Heikin-Ashi'
            )

    fig.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="RSI", overlaying='y', side='right', range=[0, 100]),
        yaxis3=dict(title="MACD", overlaying='y', side='right', position=0.95),
        yaxis4=dict(title="ATR", overlaying='y', side='right', position=0.90),
        yaxis5=dict(title="ADX", overlaying='y', side='right', position=0.85),
        yaxis6=dict(title="CMF", overlaying='y', side='right', position=0.80),
        yaxis7=dict(title="TRIX", overlaying='y', side='right', position=0.75),
        yaxis8=dict(title="Ultimate Osc", overlaying='y', side='right', position=0.70),
        yaxis9=dict(title="CMO", overlaying='y', side='right', position=0.65),
        yaxis10=dict(title="VPT", overlaying='y', side='right', position=0.60),
        height=800,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_monte_carlo(data, symbol, simulations=1000, days=30):
    simulation_results = monte_carlo_simulation(data, simulations, days)
    simulation_df = pd.DataFrame(simulation_results).T
    simulation_df.index = pd.date_range(start=data.index[-1], periods=days+1, freq='D')
    
    fig = px.line(simulation_df, title=f"Monte Carlo Simulation for {symbol} ({days} days)")
    fig.add_scatter(x=data.index[-30:], y=data['Close'][-30:], mode='lines', name='Historical Close')
    
    percentiles = simulation_df.quantile([0.1, 0.5, 0.9], axis=1).T
    fig.add_scatter(x=percentiles.index, y=percentiles[0.1], name='10th Percentile', line=dict(dash='dash'))
    fig.add_scatter(x=percentiles.index, y=percentiles[0.5], name='Median', line=dict(color='green'))
    fig.add_scatter(x=percentiles.index, y=percentiles[0.9], name='90th Percentile', line=dict(dash='dash'))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    st.title("📈 Stock Analysis Dashboard")
    
    stock_list = fetch_nse_stock_list()
    stock_list = sorted(list(set(stock_list)))
    
    tab1, tab2, tab3, tab4 = st.tabs(["Single Stock Analysis", "Sector Analysis", "Portfolio Analysis", "Market Overview"])
    
    with tab1:
        st.header("Single Stock Analysis")
        symbol = st.selectbox("Select Stock", stock_list, key="single_stock")
        period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=5)
        interval = st.selectbox("Select Interval", ["1d", "1wk", "1mo"], index=0)
        show_candlestick = st.checkbox("Show Candlestick Chart", value=False)
        
        indicators = st.multiselect(
            "Select Indicators",
            ["RSI", "MACD", "Bollinger", "ATR", "ADX", "VWAP", "Parabolic_SAR", 
             "Fib_Retracements", "Ichimoku", "CMF", "Donchian", "Keltner", 
             "TRIX", "Ultimate_Osc", "CMO", "VPT", "Pivot Points", "Heikin-Ashi"],
            default=["RSI", "MACD", "Bollinger"]
        )
        
        if st.button("Analyze Stock"):
            with st.spinner(f"Fetching data for {symbol}..."):
                data = fetch_stock_data_cached(symbol, period, interval)
                if not data.empty:
                    analyzed_data = analyze_stock(data)
                    if analyzed_data is not None:
                        recommendations = generate_recommendations(analyzed_data, symbol)
                        
                        st.subheader(f"Analysis for {symbol}")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Recommendations**")
                            st.write(f"Intraday: {recommendations['Intraday']}")
                            st.write(f"Swing: {recommendations['Swing']}")
                            st.write(f"Short-Term: {recommendations['Short-Term']}")
                            st.write(f"Long-Term: {recommendations['Long-Term']}")
                            st.write(f"Mean Reversion: {recommendations['Mean_Reversion']}")
                            st.write(f"Breakout: {recommendations['Breakout']}")
                            st.write(f"Ichimoku Trend: {recommendations['Ichimoku_Trend']}")
                        
                        with col2:
                            st.write("**Trade Levels**")
                            st.write(f"Current Price: ₹{recommendations['Current Price']:.2f}" if recommendations['Current Price'] else "Current Price: N/A")
                            st.write(f"Buy At: ₹{recommendations['Buy At']:.2f}" if recommendations['Buy At'] else "Buy At: N/A")
                            st.write(f"Stop Loss: ₹{recommendations['Stop Loss']:.2f}" if recommendations['Stop Loss'] else "Stop Loss: N/A")
                            st.write(f"Target: ₹{recommendations['Target']:.2f}" if recommendations['Target'] else "Target: N/A")
                            st.write(f"Score: {recommendations['Score']}")
                            st.write(f"Net Score: {recommendations['Net_Score']:.2f}")
                        
                        st.subheader("Price and Indicators Chart")
                        plot_stock_data(analyzed_data, symbol, indicators, show_candlestick)
                        
                        st.subheader("Monte Carlo Simulation")
                        plot_monte_carlo(data, symbol)
                        
                        fundamentals = fetch_fundamentals(symbol)
                        st.subheader("Fundamentals")
                        st.write(f"P/E Ratio: {fundamentals['P/E']:.2f}")
                        st.write(f"EPS: {fundamentals['EPS']:.2f}")
                        st.write(f"Revenue Growth: {fundamentals['RevenueGrowth']:.2%}")
                        st.write(f"Sector: {fundamentals['Sector']}")
                        
                        st.subheader("Risk Assessment")
                        st.write(assess_risk(analyzed_data))
                    else:
                        st.error(f"⚠️ Failed to analyze {symbol}")
                else:
                    st.error(f"⚠️ No data available for {symbol}")
    
    with tab2:
        st.header("Sector Analysis")
        sector = st.selectbox("Select Sector", list(SECTORS.keys()))
        max_stocks = st.slider("Max Stocks to Analyze", 5, 50, 20)
        
        if st.button("Analyze Sector"):
            with st.spinner(f"Analyzing {sector} sector..."):
                stock_batch = SECTORS[sector][:max_stocks]
                results = analyze_batch(stock_batch)
                
                if results:
                    df = pd.DataFrame(results)
                    df = df.sort_values(by="Net_Score", ascending=False)
                    
                    st.subheader(f"{sector} Sector Analysis")
                    st.dataframe(df)
                    
                    top_stocks = df.head(5)[['Symbol', 'Net_Score']]
                    fig = px.bar(top_stocks, x='Symbol', y='Net_Score', title=f"Top Stocks in {sector}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"⚠️ No valid data for {sector} sector")
    
    with tab3:
        st.header("Portfolio Analysis")
        portfolio = st.multiselect("Select Stocks for Portfolio", stock_list, default=stock_list[:5])
        weights = st.text_input("Enter Weights (comma-separated, e.g., 0.2,0.3,0.2,0.2,0.1)", "0.2,0.2,0.2,0.2,0.2")
        
        if st.button("Analyze Portfolio"):
            with st.spinner("Analyzing portfolio..."):
                try:
                    weights = [float(w) for w in weights.split(",")]
                    if len(weights) != len(portfolio) or sum(weights) != 1.0:
                        st.error("⚠️ Weights must match the number of stocks and sum to 1")
                    else:
                        portfolio_data = []
                        for symbol in portfolio:
                            data = fetch_stock_data_cached(symbol)
                            if not data.empty:
                                portfolio_data.append(data['Close'])
                        
                        if portfolio_data:
                            portfolio_df = pd.concat(portfolio_data, axis=1)
                            portfolio_df.columns = portfolio
                            portfolio_returns = portfolio_df.pct_change().dropna()
                            weighted_returns = portfolio_returns * weights
                            portfolio_performance = weighted_returns.sum(axis=1)
                            
                            annualized_return = portfolio_performance.mean() * 252
                            annualized_volatility = portfolio_performance.std() * np.sqrt(252)
                            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
                            
                            st.subheader("Portfolio Performance")
                            st.write(f"Annualized Return: {annualized_return:.2%}")
                            st.write(f"Annualized Volatility: {annualized_volatility:.2%}")
                            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
                            
                            fig = px.line(portfolio_df, title="Portfolio Stock Prices")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("⚠️ No valid data for portfolio")
                except ValueError:
                    st.error("⚠️ Invalid weights format")
    
    with tab4:
        st.header("Market Overview")
        if st.button("Generate Market Overview"):
            with st.spinner("Generating market overview..."):
                ad_ratio = calculate_advance_decline_ratio(stock_list[:100])
                trending = get_trending_stocks()
                
                st.subheader("Market Sentiment")
                st.write(f"Advance/Decline Ratio: {ad_ratio:.2f}")
                
                st.subheader("Trending Stocks")
                st.write(trending)
                
                sector_scores = {}
                for sector in SECTORS:
                    stock_batch = SECTORS[sector][:10]
                    results = analyze_batch(stock_batch)
                    if results:
                        scores = [r['Net_Score'] for r in results if r]
                        sector_scores[sector] = np.mean(scores) if scores else 0
                
                sector_df = pd.DataFrame.from_dict(sector_scores, orient='index', columns=['Average Score'])
                sector_df = sector_df.sort_values(by='Average Score', ascending=False)
                
                st.subheader("Sector Performance")
                st.dataframe(sector_df)
                
                fig = px.bar(sector_df, title="Sector Average Scores")
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
