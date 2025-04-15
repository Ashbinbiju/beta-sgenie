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

def calculate_pivot_points(data):
    try:
        high = data['High'].iloc[-1]
        low = data['Low'].iloc[-1]
        close = data['Close'].iloc[-1]
        
        # Standard Pivot Point
        pivot = (high + low + close) / 3
        
        # Support and Resistance Levels
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
        # Create a new DataFrame to store Heikin-Ashi values
        ha_data = pd.DataFrame(index=data.index)
        
        # Calculate HA_Close
        ha_data['HA_Close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
        
        # Initialize HA_Open with the first value from data['Open']
        ha_data['HA_Open'] = data['Open'].copy()
        
        # Compute HA_Open for subsequent rows
        for i in range(1, len(ha_data)):
            ha_data['HA_Open'].iloc[i] = (ha_data['HA_Open'].iloc[i-1] + ha_data['HA_Close'].iloc[i-1]) / 2
        
        # Calculate HA_High and HA_Low
        ha_data['HA_High'] = pd.concat([data['High'], ha_data['HA_Open'], ha_data['HA_Close']], axis=Tube1).max(axis=1)
        ha_data['HA_Low'] = pd.concat([data['Low'], ha_data['HA_Open'], ha_data['HA_Close']], axis=1).min(axis=1)
        
        return ha_data[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Heikin-Ashi: {str(e)}")
        return None

def analyze_stock(data):
    if data.empty or len(data) < 27:
        st.warning("⚠️ Insufficient data to compute indicators.")
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
        data['RSI'] = None
    try:
        macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
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
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
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
        data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 2.0)
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
        data['Pivot'] = None
        data['Support1'] = None
        data['Resistance1'] = None
        data['Support2'] = None
        data['Resistance2'] = None
    try:
        ha_data = calculate_heikin_ashi(data)
        if ha_data is not None:
            data['HA_Open'] = ha_data['HA_Open']
            data['HA_High'] = ha_data['HA_High']
            data['HA_Low'] = ha_data['HA_Low']
            data['HA_Close'] = ha_data['HA_Close']
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Heikin-Ashi: {str(e)}")
        data['HA_Open'] = None
        data['HA_High'] = None
        data['HA_Low'] = None
        data['HA_Close'] = None
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
    atr_multiplier += (last_atr / atr_std) * 0.5  # Adjust based on volatility
    stop_loss = last_close - (atr_multiplier * last_atr)
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
        pe = info.get('trailingPE', 100)
        pe = min(pe, 100) if pe != float('inf') else 100
        return {
            'P/E': pe,
            'EPS': info.get('trailingEps', 0),
            'RevenueGrowth': info.get('revenueGrowth', 0)
        }
    except Exception:
        return {'P/E': 100, 'EPS': 0, 'RevenueGrowth': 0}

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
    
    # RSI (Extreme conditions only)
    if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]) and 0 <= data['RSI'].iloc[-1] <= 100:
        rsi = data['RSI'].iloc[-1]
        if rsi <= 20:
            buy_score += 2
        elif rsi >= 80:
            sell_score += 2
    
    # MACD
    if ('MACD' in data.columns and 'MACD_signal' in data.columns and 
        not pd.isna(data['MACD'].iloc[-1]) and not pd.isna(data['MACD_signal'].iloc[-1])):
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_signal'].iloc[-1]
        if macd > macd_signal:
            buy_score += 1
        elif macd < macd_signal:
            sell_score += 1
    
    # Bollinger Bands
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
    
    # VWAP
    if 'VWAP' in data.columns and not pd.isna(data['VWAP'].iloc[-1]) and not pd.isna(data['Close'].iloc[-1]):
        vwap = data['VWAP'].iloc[-1]
        close = data['Close'].iloc[-1]
        if close > vwap:
            buy_score += 1
        elif close < vwap:
            sell_score += 1
    
    # Volume Analysis
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
    
    # Volume Spikes
    if 'Volume_Spike' in data.columns and not pd.isna(data['Volume_Spike'].iloc[-1]):
        spike = data['Volume_Spike'].iloc[-1]
        close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        if spike and close > prev_close:
            buy_score += 1
        elif spike and close < prev_close:
            sell_score += 1
    
    # Divergence
    if 'Divergence' in data.columns:
        divergence = data['Divergence'].iloc[-1]
        if divergence == "Bullish Divergence":
            buy_score += 1
        elif divergence == "Bearish Divergence":
            sell_score += 1
    
    # Ichimoku Cloud
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
    
    # Chaikin Money Flow
    if 'CMF' in data.columns and not pd.isna(data['CMF'].iloc[-1]):
        cmf = data['CMF'].iloc[-1]
        if cmf > 0:
            buy_score += 1
        elif cmf < 0:
            sell_score += 1
    
    # Donchian Channels
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
    
    # Mean Reversion (RSI-based, standard thresholds)
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
    
    # Keltner Channels
    if ('Keltner_Upper' in data.columns and 'Keltner_Lower' in data.columns and 
        not pd.isna(data['Keltner_Upper'].iloc[-1]) and not pd.isna(data['Keltner_Lower'].iloc[-1])):
        close = data['Close'].iloc[-1]
        upper = data['Keltner_Upper'].iloc[-1]
        lower = data['Keltner_Lower'].iloc[-1]
        if close < lower:
            buy_score += 1
        elif close > upper:
            sell_score += 1
    
    # TRIX
    if 'TRIX' in data.columns and not pd.isna(data['TRIX'].iloc[-1]) and not pd.isna(data['TRIX'].iloc[-2]):
        trix = data['TRIX'].iloc[-1]
        prev_trix = data['TRIX'].iloc[-2]
        if trix > 0 and trix > prev_trix:
            buy_score += 1
        elif trix < 0 and trix < prev_trix:
            sell_score += 1
    
    # Ultimate Oscillator
    if 'Ultimate_Osc' in data.columns and not pd.isna(data['Ultimate_Osc'].iloc[-1]):
        uo = data['Ultimate_Osc'].iloc[-1]
        if uo < 30:
            buy_score += 1
        elif uo > 70:
            sell_score += 1
    
    # Chande Momentum Oscillator
    if 'CMO' in data.columns and not pd.isna(data['CMO'].iloc[-1]):
        cmo = data['CMO'].iloc[-1]
        if cmo < -50:
            buy_score += 1
        elif cmo > 50:
            sell_score += 1
    
    # Volume Price Trend
    if 'VPT' in data.columns and not pd.isna(data['VPT'].iloc[-1]) and not pd.isna(data['VPT'].iloc[-2]):
        vpt = data['VPT'].iloc[-1]
        prev_vpt = data['VPT'].iloc[-2]
        if vpt > prev_vpt:
            buy_score += 1
        elif vpt < prev_vpt:
            sell_score += 1
    
    # Fibonacci Retracements
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
    
    # Parabolic SAR
    if 'Parabolic_SAR' in data.columns and not pd.isna(data['Parabolic_SAR'].iloc[-1]):
        sar = data['Parabolic_SAR'].iloc[-1]
        close = data['Close'].iloc[-1]
        if close > sar:
            buy_score += 1
        elif close < sar:
            sell_score += 1
    
    # OBV
    if 'OBV' in data.columns and not pd.isna(data['OBV'].iloc[-1]) and not pd.isna(data['OBV'].iloc[-2]):
        obv = data['OBV'].iloc[-1]
        prev_obv = data['OBV'].iloc[-2]
        if obv > prev_obv:
            buy_score += 1
        elif obv < prev_obv:
            sell_score += 1
    
    # Pivot Points
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
    
    # Heikin-Ashi
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
    
    # Fundamentals
    if symbol:
        fundamentals = fetch_fundamentals(symbol)
        if fundamentals['P/E'] < 15 and fundamentals['EPS'] > 0:
            buy_score += 2
        elif fundamentals['P/E'] > 30 or fundamentals['EPS'] < 0:
            sell_score += 2
        if fundamentals['RevenueGrowth'] > 0.1:
            buy_score += 1
        elif fundamentals['RevenueGrowth'] < 0:
            sell_score += 1
    
    # Normalized scoring
    total_signals = max(buy_score + sell_score, 5)
    net_score = (buy_score - sell_score) / total_signals * 5
    
    # Set recommendations with narrower thresholds
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
            "Score": recommendations.get("Score", 0),
            "Net_Score": recommendations.get("Net_Score", 0)
        }
    return None

def analyze_all_stocks(stock_list, batch_size=10, progress_callback=None):
    results = []
    total_batches = (len(stock_list) // batch_size) + (1 if len(stock_list) % batch_size != 0 else 0)
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
        if progress_callback:
            progress_callback((i + len(batch)) / len(stock_list))
    
    results_df = pd.DataFrame([r for r in results if r is not None])
    if results_df.empty:
        st.warning("⚠️ No valid stock data retrieved.")
        return pd.DataFrame()
    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    if "Net_Score" not in results_df.columns:
        results_df["Net_Score"] = 0
    if "Current Price" not in results_df.columns:
        results_df["Current Price"] = None
    return results_df.sort_values(by="Net_Score", ascending=False).head(10)

def analyze_intraday_stocks(stock_list, batch_size=10, progress_callback=None):
    results = []
    total_batches = (len(stock_list) // batch_size) + (1 if len(stock_list) % batch_size != 0 else 0)
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
        if progress_callback:
            progress_callback((i + len(batch)) / len(stock_list))
    
    results_df = pd.DataFrame([r for r in results if r is not None])
    if results_df.empty:
        st.warning("⚠️ No valid intraday stock data retrieved.")
        return pd.DataFrame()
    if "Net_Score" not in results_df.columns:
        results_df["Net_Score"] = 0
    return results_df[results_df["Intraday"].isin(["Buy", "Strong Buy"])].sort_values(by="Net_Score", ascending=False).head(5)

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

def display_dashboard(symbol=None, data=None, recommendations=None, selected_stocks=None):
    st.title("📊 StockGenie Pro - NSE Analysis")
    st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")
    
    if st.button("🚀 Generate Daily Top Picks"):
        progress_bar = st.progress(0)
        loading_text = st.empty()
        loading_messages = itertools.cycle([
            "Analyzing trends...", "Fetching data...", "Crunching numbers...",
            "Evaluating indicators...", "Finalizing results..."
        ])
        results_df = analyze_all_stocks(
            selected_stocks,
            batch_size=10,
            progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages)
        )
        progress_bar.empty()
        loading_text.empty()
        if not results_df.empty:
            st.subheader("🏆 Today's Top 10 Stocks")
            for _, row in results_df.iterrows():
                with st.expander(f"{row['Symbol']} - Net Score: {row['Net_Score']}"):
                    current_price = row['Current Price'] if pd.notnull(row['Current Price']) else "N/A"
                    buy_at = row['Buy At'] if pd.notnull(row['Buy At']) else "N/A"
                    stop_loss = row['Stop Loss'] if pd.notnull(row['Stop Loss']) else "N/A"
                    target = row['Target'] if pd.notnull(row['Target']) else "N/A"
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
                    """)
        else:
            st.warning("⚠️ No top picks available due to data issues.")
    
    if st.button("⚡ Generate Intraday Top 5 Picks"):
        progress_bar = st.progress(0)
        loading_text = st.empty()
        loading_messages = itertools.cycle([
            "Scanning intraday trends...", "Detecting buy signals...", "Calculating stop-loss levels...",
            "Optimizing targets...", "Finalizing top picks..."
        ])
        intraday_results = analyze_intraday_stocks(
            selected_stocks,
            batch_size=10,
            progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages)
        )
        progress_bar.empty()
        loading_text.empty()
        if not intraday_results.empty:
            st.subheader("🏆 Top 5 Intraday Stocks")
            for _, row in intraday_results.iterrows():
                with st.expander(f"{row['Symbol']} - Net Score: {row['Net_Score']}"):
                    current_price = row['Current Price'] if pd.notnull(row['Current Price']) else "N/A"
                    buy_at = row['Buy At'] if pd.notnull(row['Buy At']) else "N/A"
                    stop_loss = row['Stop Loss'] if pd.notnull(row['Stop Loss']) else "N/A"
                    target = row['Target'] if pd.notnull(row['Target']) else "N/A"
                    st.markdown(f"""
                    {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                    Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                    Target: ₹{target}  
                    Intraday: {colored_recommendation(row['Intraday'])}  
                    """)
        else:
            st.warning("⚠️ No intraday picks available due to data issues.")
    
    if symbol and data is not None and recommendations is not None:
        st.header(f"📋 {symbol.split('.')[0]} Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = recommendations['Current Price'] if recommendations['Current Price'] is not None else "N/A"
            st.metric(tooltip("Current Price", TOOLTIPS['RSI']), f"₹{current_price}")
        with col2:
            buy_at = recommendations['Buy At'] if recommendations['Buy At'] is not None else "N/A"
            st.metric("Buy At", f"₹{buy_at}")
        with col3:
            stop_loss = recommendations['Stop Loss'] if recommendations['Stop Loss'] is not None else "N/A"
            st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), f"₹{stop_loss}")
        with col4:
            target = recommendations['Target'] if recommendations['Target'] is not None else "N/A"
            st.metric("Target", f"₹{target}")
        
        st.subheader("Recommendations")
        st.write(f"Intraday: {colored_recommendation(recommendations['Intraday'])}")
        st.write(f"Swing: {colored_recommendation(recommendations['Swing'])}")
        st.write(f"Short-Term: {colored_recommendation(recommendations['Short-Term'])}")
        st.write(f"Long-Term: {colored_recommendation(recommendations['Long-Term'])}")
        st.write(f"Mean Reversion: {colored_recommendation(recommendations['Mean_Reversion'])}")
        st.write(f"Breakout: {colored_recommendation(recommendations['Breakout'])}")
        st.write(f"Ichimoku Trend: {colored_recommendation(recommendations['Ichimoku_Trend'])}")
        
        st.subheader("Technical Indicators")
        indicators = {
            "RSI": data['RSI'].iloc[-1] if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]) else "N/A",
            "MACD": data['MACD'].iloc[-1] if 'MACD' in data.columns and not pd.isna(data['MACD'].iloc[-1]) else "N/A",
            "ATR": data['ATR'].iloc[-1] if 'ATR' in data.columns and not pd.isna(data['ATR'].iloc[-1]) else "N/A",
            "VWAP": data['VWAP'].iloc[-1] if 'VWAP' in data.columns and not pd.isna(data['VWAP'].iloc[-1]) else "N/A",
            "Pivot": data['Pivot'].iloc[-1] if 'Pivot' in data.columns and not pd.isna(data['Pivot'].iloc[-1]) else "N/A",
            "HA_Close": data['HA_Close'].iloc[-1] if 'HA_Close' in data.columns and not pd.isna(data['HA_Close'].iloc[-1]) else "N/A"
        }
        for key, value in indicators.items():
            st.write(f"{tooltip(key, TOOLTIPS.get(key, ''))}: {value}")
        
        st.subheader("Price Chart")
        fig = px.line(data, x=data.index, y="Close", title=f"{symbol} Price Trend")
        st.plotly_chart(fig)

if __name__ == "__main__":
    stock_list = fetch_nse_stock_list()
    selected_stocks = st.sidebar.multiselect("Select Stocks or Sectors", 
                                             options=["All NSE Stocks"] + list(SECTORS.keys()), 
                                             default="Bank")
    if "All NSE Stocks" in selected_stocks:
        stocks_to_analyze = stock_list
    else:
        stocks_to_analyze = list(set([stock for sector in selected_stocks for stock in SECTORS.get(sector, [])]))
    
    symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS)", "")
    if symbol:
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            data = analyze_stock(data)
            if data is None:
                st.error(f"⚠️ Insufficient or invalid data for {symbol}")
            else:
                recommendations = generate_recommendations(data, symbol)
                display_dashboard(symbol, data, recommendations, stocks_to_analyze)
        else:
            st.error(f"⚠️ No data found for {symbol}")
