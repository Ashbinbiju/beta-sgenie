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

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Comprehensive list of User Agents
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

# API Keys (Consider moving to environment variables)
ALPHA_VANTAGE_KEY = "TCAUKYUCIDZ6PI57"

# Tooltip explanations
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
}

# Define sectors and their stocks
SECTORS = {
    "top_nse_stocks": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS", "KOTAKBANK.NS",
        "ITC.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "BAJFINANCE.NS",
        "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS",
        "NESTLEIND.NS", "POWERGRID.NS", "TECHM.NS", "M&M.NS", "ONGC.NS",
        "COALINDIA.NS", "NTPC.NS", "ADANIENT.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
        "GRASIM.NS", "DIVISLAB.NS", "BAJAJFINSV.NS", "HINDALCO.NS", "DRREDDY.NS",
        "CIPLA.NS", "BRITANNIA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "SHREECEM.NS",
        "APOLLOHOSP.NS", "TATAMOTORS.NS", "INDUSINDBK.NS", "PIDILITIND.NS", "DMART.NS",
        "SIEMENS.NS", "BHEL.NS", "GAIL.NS", "IOC.NS", "BEL.NS",
        "HAVELLS.NS", "DABUR.NS", "AMBUJACEM.NS", "LUPIN.NS", "BERGEPAINT.NS",
        "UPL.NS", "ACC.NS", "BANKBARODA.NS", "PNB.NS", "GODREJCP.NS",
        "TATAPOWER.NS", "RECLTD.NS", "PFC.NS", "COLPAL.NS", "TRENT.NS",
        "MUTHOOTFIN.NS", "AUROPHARMA.NS", "IGL.NS", "DLF.NS", "CONCOR.NS",
        "ALKEM.NS", "NAUKRI.NS", "JINDALSTEL.NS", "PAGEIND.NS", "BALKRISIND.NS",
        "IDEA.NS", "MFSL.NS", "CHOLAFIN.NS", "L&TFH.NS", "TORNTPHARM.NS",
        "MRF.NS", "BOSCHLTD.NS", "ABB.NS", "ZYDUSLIFE.NS", "HINDPETRO.NS",
        "SBICARD.NS", "INDIGO.NS", "ADANIPORTS.NS", "BAJAJ-AUTO.NS", "VEDL.NS",
        "GODREJPROP.NS", "TATACONSUM.NS", "ICICIPRULI.NS", "OBEROIRLTY.NS", "POLYCAB.NS",
        "PHOENIXLTD.NS", "ASTRAL.NS", "BPCL.NS", "CUMMINSIND.NS", "LICHSGFIN.NS",
        "IDFCFIRSTB.NS", "MCX.NS", "GLENMARK.NS", "SUNTV.NS", "IPCALAB.NS",
        "LTIM.NS", "EXIDEIND.NS", "COROMANDEL.NS", "DEEPAKNTR.NS", "ESCORTS.NS",
        "BANKINDIA.NS", "GSPL.NS", "GNFC.NS", "NMDC.NS", "RAMCOCEM.NS",
        "BANDHANBNK.NS", "APOLLOTYRE.NS", "TATACOMM.NS", "PETRONET.NS", "AUBANK.NS",
        "KANSAINER.NS", "FEDERALBNK.NS", "SRF.NS", "MPHASIS.NS", "JUBLFOOD.NS",
        "VOLTAS.NS", "BIOCON.NS", "CANBK.NS", "RBLBANK.NS", "SYNGENE.NS",
        "DALBHARAT.NS", "YESBANK.NS", "HINDCOPPER.NS", "TVSMOTOR.NS", "UNIONBANK.NS",
        "ASHOKLEY.NS", "INDHOTEL.NS", "ABFRL.NS", "CROMPTON.NS", "LAURUSLABS.NS",
        "BATAINDIA.NS", "SAIL.NS", "GUJGASLTD.NS", "NATCOPHARM.NS", "PRESTIGE.NS",
        "ZEEL.NS", "FORTIS.NS", "LALPATHLAB.NS", "IDFC.NS", "MANAPPURAM.NS",
        "GMRINFRA.NS", "LICINDIA.NS", "STARHEALTH.NS", "PAYTM.NS", "IRCTC.NS",
        "METROPOLIS.NS", "DELHIVERY.NS", "ZOMATO.NS", "NYKAA.NS", "POLICYBZR.NS",
        "ADANIGREEN.NS", "ADANITRANS.NS", "HAL.NS", "IRFC.NS", "RVNL.NS",
        "MAZDOCK.NS", "COCHINSHIP.NS", "BEML.NS", "ENGINERSIN.NS", "KIOCL.NS",
        "NBCC.NS", "RCF.NS", "NLCINDIA.NS", "SJVN.NS", "MOIL.NS",
        "ITI.NS", "HFCL.NS", "TRIDENT.NS", "SUZLON.NS", "JPPOWER.NS",
        "TATACHEM.NS", "IEX.NS", "CDSL.NS", "ANGELONE.NS", "BSE.NS",
        "NAM-INDIA.NS", "UTIAMC.NS", "CAMS.NS", "360ONE.NS", "KFINTECH.NS",
        "CHAMBLFERT.NS", "GNFC.NS", "GSFC.NS", "PARADEEP.NS", "FACT.NS",
        "ATUL.NS", "VINATIORGA.NS", "AARTIIND.NS", "PIIND.NS", "NAVINFLUOR.NS",
        "CUB.NS", "DCBBANK.NS", "EQUITASBNK.NS", "UJJIVANSFB.NS", "SOUTHBANK.NS",
        "KARURVYSYA.NS", "CSBBANK.NS", "TMB.NS", "J&KBANK.NS", "KTKBANK.NS",
        "FSL.NS", "JBMA.NS", "MOTHERSON.NS", "ENDURANCE.NS", "SCHAEFFLER.NS",
        "TIINDIA.NS", "SUNDARMFIN.NS", "SHRIRAMFIN.NS", "AAVAS.NS", "HOMEFIRST.NS",
        "KSB.NS", "GRINDWELL.NS", "ELGIEQUIP.NS", "KIRLOSENG.NS", "TIMKEN.NS",
        "THERMAX.NS", "HONAUT.NS", "GMMPFAUDLR.NS", "SKFINDIA.NS", "ARE&M.NS",
        "EIHOTEL.NS", "CHALET.NS", "LEMONTRIE.NS", "SAPPHIRE.NS", "WONDERLA.NS",
        "CARBORUNIV.NS", "CERA.NS", "KAJARIACER.NS", "SOMANYCERA.NS", "ORIENTELEC.NS",
        "BLUESTARCO.NS", "WHIRLPOOL.NS", "SYMPHONY.NS", "BAJAJELEC.NS", "IFBIND.NS",
        "TTKPRESTIG.NS", "HAWKINCOOK.NS", "BORORENEW.NS", "GREENLAM.NS", "PRINCEPIPE.NS",
        "ROSSARI.NS", "NEOGEN.NS", "SHARDACROP.NS", "FINEORG.NS", "BALAMINES.NS"
    ],
    "Bank": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", 
        "INDUSINDBK.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "UNIONBANK.NS", 
        "IDFCFIRSTB.NS", "FEDERALBNK.NS", "RBLBANK.NS", "BANDHANBNK.NS", "INDIANB.NS", 
        "BANKINDIA.NS", "KARURVYSYA.NS", "CUB.NS", "J&KBANK.NS", "LAKSHVILAS.NS", 
        "DCBBANK.NS", "SYNDIBANK.NS", "ALBK.NS", "ANDHRABANK.NS", "CORPBANK.NS", 
        "ORIENTBANK.NS", "UNITEDBNK.NS", "AUBANK.NS"
    ],
    "Software & IT Services": [
        "TCS.NS", "TECHM.NS", "NETF.NS", "OMEGAIN.NS", "NETLINK.NS", "ACCEL.NS", "MEGRISOFT.NS", 
        "STARCOM.NS", "JETKING.NS", "BRISKTECH.NS", "SATTRIX.NS", "SBSJHVHSV.NS", "SMARTLINK.NS", 
        "CGVAK.NS", "PRITHVI.NS", "DELAPLEX.NS", "AZTEC.NS", "ELNET.NS", "SLONE.NS", "INTRASOFT.NS", 
        "ENFUSE.NS", "ATISHAY.NS", "ESCONET.NS", "TERASOFT.NS", "ADDICTIVE.NS", "SIGMASOLVE.NS", 
        "SILICONRENT.NS", "PEANALYTICS.NS", "DEVINFO.NS", "TREJHARA.NS", "ACESOFT.NS", "SYSTANGO.NS", 
        "ABM.NS", "CAPITALNUM.NS", "ARTIFICIAL.NS", "IZMO.NS", "MOLDTECH.NS", "DANLAW.NS", 
        "CYBERTECH.NS", "ONWARDTEC.NS", "SOFTTECH.NS", "MINDTECK.NS", "VINSYS.NS", "MOSUTIL.NS", 
        "INNOVANA.NS", "APTECH.NS", "ALLETECH.NS", "VEEFIN.NS", "INFOBEAN.NS", "IRIS.NS", "NINTEC.NS", 
        "AURUMPROP.NS", "SILVERT.NS", "TRIDENT.NS", "ALLIEDDIG.NS", "KSOLVES.NS", "KELLTONTEC.NS", 
        "MATRIMONY.NS", "EXPLEOSOL.NS", "DYNACONS.NS", "KERNEX.NS", "ECORECO.NS", "UNICOMMERCE.NS", 
        "BLS.NS", "RAMCOSYS.NS", "ALLDIGI.NS", "QUICKHEAL.NS", "NIIT.NS", "ORIENTTECH.NS", "SAKSOFT.NS", 
        "NUCLEUS.NS", "MOBIKWIK.NS", "HGS.NS", "GENESYS.NS", "RPSGVENT.NS", "MOSCHIP.NS", "DATAMATICS.NS", 
        "63MOONS.NS", "RSYSTEMS.NS", "EMUDHRA.NS", "TANLA.NS", "ROUTE.NS", "RATEGAIN.NS", "NIITMTS.NS", 
        "ZAGGLE.NS", "INTELLECT.NS", "SONATSOFTW.NS", "HAPPSTMNDS.NS", "NAZARA.NS", "CMSINFO.NS", 
        "LATENTVIEW.NS", "ZINKA.NS", "JUSTDIAL.NS", "BSOFT.NS", "NEWGEN.NS", "ZENSARTECH.NS", "FSL.NS", 
        "HEXAWARE.NS", "NYKAA.NS", "PAYTM.NS", "IRCTC.NS", "SWIGGY.NS", "ZOMATO.NS", "WIPRO.NS"
    ],
    "Finance": ["BAJFINANCE.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS"],
    "Automobile & Ancillaries": ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS"],
    "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "Metals & Mining": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS"],
    "Oil&Gas": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS", "OIL.NS", "PETRONET.NS", "MRPL.NS", "CHENNPETRO.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "Power": ["NTPC.NS", "POWERGRID.NS", "ADANIPOWER.NS", "TATAPOWER.NS", "ADANIENSOL.NS", "JSWENERGY.NS", "INDIANREN.NS", "NLCINDIA.NS", "CESC.NS", "RPOWER.NS", "IEX.NS", "NAVA.NS", "INDIGRID.NS", "ACMESOLAR.NS", "RELINFRA.NS", "GMRP&UI.NS", "SWSOLAR.NS", "PTC.NS", "GIPCL.NS", "BFUTILITIE.NS", "RAVINDRA.NS", "DANISH.NS", "APSINDIA.NS", "SUNGARNER.NS"],
    "Capital Goods": ["LT.NS", "BHEL.NS", "SIEMENS.NS"],
    "Chemicals": ["PIDILITIND.NS", "SRF.NS", "UPL.NS"],
    "Telecom": ["BHARTIARTL.NS", "IDEA.NS", "RELIANCE.NS"],
    "Infrastructure": ["ADANIPORTS.NS", "GMRINFRA.NS"],
    "Insurance": ["ICICIGI.NS", "NIACL.NS"],
    "Diversified": ["RODIUM.BO", "FRANKLIN.BO", "ANSALBU.NS", "SHERVANI.BO"],
    "Construction Materials": ["ULTRACEMCO.NS", "ACC.NS", "AMBUJACEM.NS"],
    "Real Estate": ["DLF.NS", "GODREJPROP.NS"],
    "Aviation": ["INDIGO.NS", "SPICEJET.NS"],
    "Retailing": ["DMART.NS", "TRENT.NS"],
    "Miscellaneous": ["ADANIENT.NS", "ADANIGREEN.NS"],
    "Diamond & Jewellery": ["KALYANKJIL.NS", "IGIL.NS", "PNGJL.NS", "RAJESHEXPO.NS", "GOLKUNDIA.BO", "CEENIKEX.BO", "NANAVATI.BO", "UDAYJEW.BO", "MINIDIA.BO", "SENCO.BO", "SKYGOLD.NS", "VAIBHAVGBL.NS", "GOLDIAM.NS", "KHAZANCHI.BO", "TBZ.NS"],
    "Consumer Durables": ["CROMPTON.NS", "PGEL.NS", "WHIRLPOOL.NS", "TTKPRESTIG.NS", "SYMPHONY.NS", "BAJAJELEC.NS", "EMIL.NS", "EPACK.NS", "STOVEKRAFT.NS", "WEL.NS", "RPTECH.NS", "HINDWAREAP.NS", "TIMEX.NS", "D-LINKINDIA.NS", "ICEMAKE.NS", "BUTTERFLY.NS", "CONTROLPR.NS", "TVSELECT.NS", "PEL.NS"],
    "Trading": ["ADANIPOWER.NS"],
    "Hospitality": ["INDHOTEL.NS", "EIHOTEL.NS"],
    "Agri": ["UPL.NS", "PIIND.NS"],
    "Textiles": ["PAGEIND.NS", "RAYMOND.NS"],
    "Industrial Gases & Fuels": ["LINDEINDIA.NS"],
    "Electricals": ["POLYCAB.NS", "KEI.NS"],
    "Alcohol": ["SDBL.NS", "GLOBUSSPR.NS", "TI.NS", "PICCADIL.NS", "ABDL.NS", "RADICO.NS", "UBL.NS", "UNITDSPR.NS"],
    "Logistics": ["CONCOR.NS", "BLUEDART.NS"],
    "Plastic Products": ["SUPREMEIND.NS"],
    "Ship Building": ["ABSMARINE.NS", "GRSE.NS", "COCHINSHIP.NS"],
    "Media & Entertainment": ["ZEEL.NS", "SUNTV.NS"],
    "ETF": ["NIFTYBEES.NS"],
    "Footwear": ["BATAINDIA.NS", "RELAXO.NS"],
    "Manufacturing": ["ASIANPAINT.NS", "BERGEPAINT.NS"],
    "Containers & Packaging": ["AGI.NS", "UFLEX.NS", "JINDALPOLY.NS", "COSMOFIRST.NS", "HUHTAMAKI.NS", "ESTER.NS", "TIRUPATI.NS", "PYRAMID.NS", "BBTCL.NS", "RAJESHCAN.NS", "IDEALTECHO.NS", "PERFECTPAC.NS", "GUJCON.NS", "HCP.NS", "SHETRON.NS"],
    "Paper": ["JKPAPER.NS", "WSTCSTPAPR.NS", "SESHAPAPER.NS", "PDMJEPAPER.NS", "NRAGRINDQ.NS", "RUCHIRA.NS", "SANGALPAPR.NS", "SVJENTERPR.NS", "METROGLOBL.NS", "SHREYANIND.NS", "SUBAMPAPER.NS", "STARPAPER.NS", "PAKKA.NS", "TNPL.NS", "KUANTUM.NS"],
    "Photographic Products": ["JINDALPHOT.NS"]
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
    best_window, best_sharpe = 14, -float('inf')  # Changed default to 14
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
        return data
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
        return data
    
    try:
        rsi_window = optimize_rsi_window(data)
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute RSI: {str(e)}")
        data['RSI'] = None
    try:
        macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)  # Updated to standard parameters
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
        st.warning(f"⚠️ Failed to Stopwatch Loss compute ADX: {str(e)}")
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
        data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 2.0)  # Updated to 2.0x
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

def fetch_fundamentals(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'P/E': info.get('trailingPE', float('inf')),
            'EPS': info.get('trailingEps', 0),
            'RevenueGrowth': info.get('revenueGrowth', 0)
        }
    except Exception:
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}

def generate_recommendations(data, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold", "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None,
        "Score": 0, "Net_Score": 0
    }
    
    if data.empty or 'Close' not in data.columns or data['Close'].iloc[-1] is None:
        print("No valid data available for recommendations.")
        return recommendations
    
    buy_score = 0
    sell_score = 0
    
    print("Starting scoring process...")
    
    # RSI
    if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]) and 0 <= data['RSI'].iloc[-1] <= 100:
        rsi = data['RSI'].iloc[-1]
        print(f"RSI: {rsi}")
        if rsi <= 20:
            buy_score += 4
            print(f"RSI <= 20, buy_score: {buy_score}")
        elif rsi < 30:
            buy_score += 2
            print(f"RSI < 30, buy_score: {buy_score}")
        elif rsi > 70:
            sell_score += 2
            print(f"RSI > 70, sell_score: {sell_score}")
    
    # MACD
    if ('MACD' in data.columns and 'MACD_signal' in data.columns and 
        not pd.isna(data['MACD'].iloc[-1]) and not pd.isna(data['MACD_signal'].iloc[-1])):
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_signal'].iloc[-1]
        print(f"MACD: {macd}, MACD_signal: {macd_signal}")
        if macd > macd_signal:
            buy_score += 1
            print(f"MACD > Signal, buy_score: {buy_score}")
        elif macd < macd_signal:
            sell_score += 1
            print(f"MACD < Signal, sell_score: {sell_score}")
    
    # Bollinger Bands
    if ('Lower_Band' in data.columns and 'Upper_Band' in data.columns and 
        not pd.isna(data['Close'].iloc[-1]) and not pd.isna(data['Lower_Band'].iloc[-1]) and 
        not pd.isna(data['Upper_Band'].iloc[-1])):
        close = data['Close'].iloc[-1]
        lower = data['Lower_Band'].iloc[-1]
        upper = data['Upper_Band'].iloc[-1]
        print(f"Close: {close}, Lower_Band: {lower}, Upper_Band: {upper}")
        if close < lower:
            buy_score += 1
            print(f"Close < Lower_Band, buy_score: {buy_score}")
        elif close > upper:
            sell_score += 1
            print(f"Close > Upper_Band, sell_score: {sell_score}")
    
    # VWAP
    if 'VWAP' in data.columns and not pd.isna(data['VWAP'].iloc[-1]) and not pd.isna(data['Close'].iloc[-1]):
        vwap = data['VWAP'].iloc[-1]
        close = data['Close'].iloc[-1]
        print(f"Close: {close}, VWAP: {vwap}")
        if close > vwap:
            buy_score += 1
            print(f"Close > VWAP, buy_score: {buy_score}")
        elif close < vwap:
            sell_score += 1
            print(f"Close < VWAP, sell_score: {sell_score}")
    
    # Volume Analysis
    if ('Volume' in data.columns and 'Avg_Volume' in data.columns and 
        not pd.isna(data['Volume'].iloc[-1]) and not pd.isna(data['Avg_Volume'].iloc[-1])):
        volume_ratio = data['Volume'].iloc[-1] / data['Avg_Volume'].iloc[-1]
        close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        print(f"Volume Ratio: {volume_ratio}, Close: {close}, Prev Close: {prev_close}")
        if volume_ratio > 1.5 and close > prev_close:
            buy_score += 2
            print(f"Volume Ratio > 1.5 & Price Up, buy_score: {buy_score}")
        elif volume_ratio > 1.5 and close < prev_close:
            sell_score += 2
            print(f"Volume Ratio > 1.5 & Price Down, sell_score: {sell_score}")
        elif volume_ratio < 0.5:
            sell_score += 1
            print(f"Volume Ratio < 0.5, sell_score: {sell_score}")
    
    # Volume Spikes
    if 'Volume_Spike' in data.columns and not pd.isna(data['Volume_Spike'].iloc[-1]):
        spike = data['Volume_Spike'].iloc[-1]
        close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        print(f"Volume_Spike: {spike}, Close: {close}, Prev Close: {prev_close}")
        if spike and close > prev_close:
            buy_score += 1
            print(f"Volume Spike & Price Up, buy_score: {buy_score}")
        elif spike and close < prev_close:
            sell_score += 1
            print(f"Volume Spike & Price Down, sell_score: {sell_score}")
    
    # Divergence
    if 'Divergence' in data.columns:
        divergence = data['Divergence'].iloc[-1]
        print(f"Divergence: {divergence}")
        if divergence == "Bullish Divergence":
            buy_score += 1
            print(f"Bullish Divergence, buy_score: {buy_score}")
        elif divergence == "Bearish Divergence":
            sell_score += 1
            print(f"Bearish Divergence, sell_score: {sell_score}")
    
    # Ichimoku Cloud
    if ('Ichimoku_Tenkan' in data.columns and 'Ichimoku_Kijun' in data.columns and 
        'Ichimoku_Span_A' in data.columns and not pd.isna(data['Ichimoku_Tenkan'].iloc[-1]) and 
        not pd.isna(data['Ichimoku_Kijun'].iloc[-1]) and not pd.isna(data['Ichimoku_Span_A'].iloc[-1])):
        tenkan = data['Ichimoku_Tenkan'].iloc[-1]
        kijun = data['Ichimoku_Kijun'].iloc[-1]
        span_a = data['Ichimoku_Span_A'].iloc[-1]
        close = data['Close'].iloc[-1]
        print(f"Tenkan: {tenkan}, Kijun: {kijun}, Span_A: {span_a}, Close: {close}")
        if tenkan > kijun and close > span_a:
            buy_score += 2
            recommendations["Ichimoku_Trend"] = "Strong Buy"
            print(f"Ichimoku Strong Buy, buy_score: {buy_score}")
        elif tenkan < kijun and close < span_a:
            sell_score += 1
            recommendations["Ichimoku_Trend"] = "Sell"
            print(f"Ichimoku Sell, sell_score: {sell_score}")
    
    # Chaikin Money Flow
    if 'CMF' in data.columns and not pd.isna(data['CMF'].iloc[-1]):
        cmf = data['CMF'].iloc[-1]
        print(f"CMF: {cmf}")
        if cmf > 0:
            buy_score += 1
            print(f"CMF > 0, buy_score: {buy_score}")
        elif cmf < 0:
            sell_score += 1
            print(f"CMF < 0, sell_score: {sell_score}")
    
    # Donchian Channels
    if ('Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns and 
        not pd.isna(data['Donchian_Upper'].iloc[-1]) and not pd.isna(data['Donchian_Lower'].iloc[-1])):
        close = data['Close'].iloc[-1]
        upper = data['Donchian_Upper'].iloc[-1]
        lower = data['Donchian_Lower'].iloc[-1]
        print(f"Close: {close}, Donchian_Upper: {upper}, Donchian_Lower: {lower}")
        if close > upper:
            buy_score += 1
            recommendations["Breakout"] = "Buy"
            print(f"Close > Donchian Upper, buy_score: {buy_score}")
        elif close < lower:
            sell_score += 1
            recommendations["Breakout"] = "Sell"
            print(f"Close < Donchian Lower, sell_score: {sell_score}")
    
    # Mean Reversion
    if ('RSI' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns and 
        not pd.isna(data['RSI'].iloc[-1]) and not pd.isna(data['Lower_Band'].iloc[-1]) and 
        not pd.isna(data['Upper_Band'].iloc[-1])):
        rsi = data['RSI'].iloc[-1]
        close = data['Close'].iloc[-1]
        lower = data['Lower_Band'].iloc[-1]
        upper = data['Upper_Band'].iloc[-1]
        print(f"RSI: {rsi}, Close: {close}, Lower_Band: {lower}, Upper_Band: {upper}")
        if rsi < 30 and close >= lower:
            buy_score += 2
            recommendations["Mean_Reversion"] = "Buy"
            print(f"Mean Reversion Buy, buy_score: {buy_score}")
        elif rsi > 70 and close >= upper:
            sell_score += 2
            recommendations["Mean_Reversion"] = "Sell"
            print(f"Mean Reversion Sell, sell_score: {sell_score}")
    
    # Keltner Channels
    if ('Keltner_Upper' in data.columns and 'Keltner_Lower' in data.columns and 
        not pd.isna(data['Keltner_Upper'].iloc[-1]) and not pd.isna(data['Keltner_Lower'].iloc[-1])):
        close = data['Close'].iloc[-1]
        upper = data['Keltner_Upper'].iloc[-1]
        lower = data['Keltner_Lower'].iloc[-1]
        print(f"Close: {close}, Keltner_Upper: {upper}, Keltner_Lower: {lower}")
        if close < lower:
            buy_score += 1
            print(f"Close < Keltner Lower, buy_score: {buy_score}")
        elif close > upper:
            sell_score += 1
            print(f"Close > Keltner Upper, sell_score: {sell_score}")
    
    # TRIX
    if 'TRIX' in data.columns and not pd.isna(data['TRIX'].iloc[-1]) and not pd.isna(data['TRIX'].iloc[-2]):
        trix = data['TRIX'].iloc[-1]
        prev_trix = data['TRIX'].iloc[-2]
        print(f"TRIX: {trix}, Prev TRIX: {prev_trix}")
        if trix > 0 and trix > prev_trix:
            buy_score += 1
            print(f"TRIX > 0 & Rising, buy_score: {buy_score}")
        elif trix < 0 and trix < prev_trix:
            sell_score += 1
            print(f"TRIX < 0 & Falling, sell_score: {sell_score}")
    
    # Ultimate Oscillator
    if 'Ultimate_Osc' in data.columns and not pd.isna(data['Ultimate_Osc'].iloc[-1]):
        uo = data['Ultimate_Osc'].iloc[-1]
        print(f"Ultimate Oscillator: {uo}")
        if uo < 30:
            buy_score += 1
            print(f"Ultimate Oscillator < 30, buy_score: {buy_score}")
        elif uo > 70:
            sell_score += 1
            print(f"Ultimate Oscillator > 70, sell_score: {sell_score}")
    
    # Chande Momentum Oscillator
    if 'CMO' in data.columns and not pd.isna(data['CMO'].iloc[-1]):
        cmo = data['CMO'].iloc[-1]
        print(f"CMO: {cmo}")
        if cmo < -50:
            buy_score += 1
            print(f"CMO < -50, buy_score: {buy_score}")
        elif cmo > 50:
            sell_score += 1
            print(f"CMO > 50, sell_score: {sell_score}")
    
    # Volume Price Trend
    if 'VPT' in data.columns and not pd.isna(data['VPT'].iloc[-1]) and not pd.isna(data['VPT'].iloc[-2]):
        vpt = data['VPT'].iloc[-1]
        prev_vpt = data['VPT'].iloc[-2]
        print(f"VPT: {vpt}, Prev VPT: {prev_vpt}")
        if vpt > prev_vpt:
            buy_score += 1
            print(f"VPT Rising, buy_score: {buy_score}")
        elif vpt < prev_vpt:
            sell_score += 1
            print(f"VPT Falling, sell_score: {sell_score}")
    
    # Fibonacci Retracements
    if ('Fib_23.6' in data.columns and 'Fib_38.2' in data.columns and 
        not pd.isna(data['Fib_23.6'].iloc[-1]) and not pd.isna(data['Fib_38.2'].iloc[-1])):
        current_price = data['Close'].iloc[-1]
        fib_levels = [data['Fib_23.6'].iloc[-1], data['Fib_38.2'].iloc[-1], 
                      data['Fib_50.0'].iloc[-1], data['Fib_61.8'].iloc[-1]]
        print(f"Close: {current_price}, Fib Levels: {fib_levels}")
        for level in fib_levels:
            if not pd.isna(level) and abs(current_price - level) / current_price < 0.01:
                if current_price > level:
                    buy_score += 1
                    print(f"Close near Fib level {level} (support), buy_score: {buy_score}")
                else:
                    sell_score += 1
                    print(f"Close near Fib level {level} (resistance), sell_score: {sell_score}")
    
    # Parabolic SAR
    if 'Parabolic_SAR' in data.columns and not pd.isna(data['Parabolic_SAR'].iloc[-1]):
        sar = data['Parabolic_SAR'].iloc[-1]
        close = data['Close'].iloc[-1]
        print(f"Close: {close}, Parabolic SAR: {sar}")
        if close > sar:
            buy_score += 1
            print(f"Close > Parabolic SAR, buy_score: {buy_score}")
        elif close < sar:
            sell_score += 1
            print(f"Close < Parabolic SAR, sell_score: {sell_score}")
    
    # OBV
    if 'OBV' in data.columns and not pd.isna(data['OBV'].iloc[-1]) and not pd.isna(data['OBV'].iloc[-2]):
        obv = data['OBV'].iloc[-1]
        prev_obv = data['OBV'].iloc[-2]
        print(f"OBV: {obv}, Prev OBV: {prev_obv}")
        if obv > prev_obv:
            buy_score += 1
            print(f"OBV Rising, buy_score: {buy_score}")
        elif obv < prev_obv:
            sell_score += 1
            print(f"OBV Falling, sell_score: {sell_score}")
    
    # Fundamentals
    if symbol:
        fundamentals = fetch_fundamentals(symbol)
        print(f"Fundamentals: P/E={fundamentals['P/E']}, EPS={fundamentals['EPS']}, RevenueGrowth={fundamentals['RevenueGrowth']}")
        if fundamentals['P/E'] < 15 and fundamentals['EPS'] > 0:
            buy_score += 2
            print(f"P/E < 15 & EPS > 0, buy_score: {buy_score}")
        elif fundamentals['P/E'] > 30 or fundamentals['EPS'] < 0:
            sell_score += 1
            print(f"P/E > 30 or EPS < 0, sell_score: {sell_score}")
        if fundamentals['RevenueGrowth'] > 0.1:
            buy_score += 1
            print(f"Revenue Growth > 10%, buy_score: {buy_score}")
        elif fundamentals['RevenueGrowth'] < 0:
            sell_score += 0.5
            print(f"Revenue Growth < 0%, sell_score: {sell_score}")
    
    # Normalized scoring
    total_signals = buy_score + sell_score
    if total_signals == 0:
        net_score = 0
    else:
        net_score = (buy_score - sell_score) / total_signals * 7
    
    print(f"Buy Score: {buy_score}, Sell Score: {sell_score}, Total Signals: {total_signals}")
    print(f"Normalized Net Score: {net_score:.2f}")
    
    # Set recommendations
    if net_score >= 5:
        recommendations["Intraday"] = "Strong Buy"
        recommendations["Swing"] = "Buy" if net_score >= 3 else "Hold"
        recommendations["Short-Term"] = "Buy" if net_score >= 2 else "Hold"
        recommendations["Long-Term"] = "Buy" if net_score >= 1 else "Hold"
    elif net_score >= 3:
        recommendations["Intraday"] = "Buy"
        recommendations["Swing"] = "Hold"
        recommendations["Short-Term"] = "Hold"
        recommendations["Long-Term"] = "Hold"
    elif net_score <= -5:
        recommendations["Intraday"] = "Strong Sell"
        recommendations["Swing"] = "Sell" if net_score <= -3 else "Hold"
        recommendations["Short-Term"] = "Sell" if net_score <= -2 else "Hold"
        recommendations["Long-Term"] = "Sell" if net_score <= -1 else "Hold"
    elif net_score <= -3:
        recommendations["Intraday"] = "Sell"
        recommendations["Swing"] = "Hold"
        recommendations["Short-Term"] = "Hold"
        recommendations["Long-Term"] = "Hold"
    else:
        recommendations["Intraday"] = "Hold"
    
    recommendations["Current Price"] = float(data['Close'].iloc[-1]) if not pd.isna(data['Close'].iloc[-1]) else None
    recommendations["Buy At"] = calculate_buy_at(data)
    recommendations["Stop Loss"] = calculate_stop_loss(data)
    recommendations["Target"] = calculate_target(data)
    recommendations["Score"] = buy_score - sell_score
    recommendations["Net_Score"] = round(net_score, 2)
    
    print(f"Final Recommendation: {recommendations['Intraday']}")
    return recommendations

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

def colored_recommendation(recommendation):
    if "Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Sell" in recommendation:
        return f"🔴 {recommendation}"
    elif "Hold" in recommendation:
        return f"🟡 {recommendation}"
    else:
        return recommendation

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
                    """, unsafe_allow_html=True)
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
                    """, unsafe_allow_html=True)
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
            st.metric(tooltip("Buy At", "Recommended entry price"), f"₹{buy_at}")
        with col3:
            stop_loss = recommendations['Stop Loss'] if recommendations['Stop Loss'] is not None else "N/A"
            st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), f"₹{stop_loss}")
        with col4:
            target = recommendations['Target'] if recommendations['Target'] is not None else "N/A"
            st.metric(tooltip("Target", "Price target based on risk/reward"), f"₹{target}")
        st.subheader("📈 Trading Recommendations")
        cols = st.columns(4)
        strategy_names = ["Intraday", "Swing", "Short-Term", "Long-Term"]
        for col, strategy in zip(cols, strategy_names):
            with col:
                st.markdown(f"**{strategy}**", unsafe_allow_html=True)
                st.markdown(f"{colored_recommendation(recommendations[strategy])}", unsafe_allow_html=True)
        
        st.subheader("🔍 Specialized Strategies")
        cols_special = st.columns(3)
        special_strategies = ["Mean_Reversion", "Breakout", "Ichimoku_Trend"]
        for col, strategy in zip(cols_special, special_strategies):
            with col:
                st.markdown(f"**{strategy.replace('_', ' ')}**", unsafe_allow_html=True)
                st.markdown(f"{colored_recommendation(recommendations[strategy])}", unsafe_allow_html=True)
        
        st.subheader("📊 Technical Indicators")
        indicator_cols = st.columns(3)
        with indicator_cols[0]:
            rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]) else "N/A"
            st.metric(tooltip("RSI", TOOLTIPS['RSI']), rsi)
            macd = data['MACD'].iloc[-1] if 'MACD' in data.columns and not pd.isna(data['MACD'].iloc[-1]) else "N/A"
            st.metric(tooltip("MACD", TOOLTIPS['MACD']), macd)
        with indicator_cols[1]:
            atr = data['ATR'].iloc[-1] if 'ATR' in data.columns and not pd.isna(data['ATR'].iloc[-1]) else "N/A"
            st.metric(tooltip("ATR", TOOLTIPS['ATR']), atr)
            vwap = data['VWAP'].iloc[-1] if 'VWAP' in data.columns and not pd.isna(data['VWAP'].iloc[-1]) else "N/A"
            st.metric(tooltip("VWAP", TOOLTIPS['VWAP']), vwap)
        with indicator_cols[2]:
            adx = data['ADX'].iloc[-1] if 'ADX' in data.columns and not pd.isna(data['ADX'].iloc[-1]) else "N/A"
            st.metric(tooltip("ADX", TOOLTIPS['ADX']), adx)
            cmf = data['CMF'].iloc[-1] if 'CMF' in data.columns and not pd.isna(data['CMF'].iloc[-1]) else "N/A"
            st.metric(tooltip("CMF", TOOLTIPS['CMF']), cmf)
        
        st.subheader("📉 Price Chart with Key Indicators")
        fig = px.line(data, x=data.index, y='Close', title=f"{symbol} Price Movement")
        fig.add_scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange'))
        fig.add_scatter(x=data.index, y=data['SMA_200'], mode='lines', name='SMA 200', line=dict(color='red'))
        fig.add_scatter(x=data.index, y=data['Upper_Band'], mode='lines', name='Bollinger Upper', line=dict(color='purple', dash='dash'))
        fig.add_scatter(x=data.index, y=data['Lower_Band'], mode='lines', name='Bollinger Lower', line=dict(color='purple', dash='dash'))
        st.plotly_chart(fig)
        
        st.subheader("🔮 Monte Carlo Simulation")
        sim_results = monte_carlo_simulation(data)
        sim_df = pd.DataFrame(sim_results).T
        sim_fig = px.line(sim_df, title=f"{symbol} 30-Day Price Simulation")
        st.plotly_chart(sim_fig)
        
        st.subheader("📊 Risk Assessment")
        risk = assess_risk(data)
        st.write(f"Risk Level: {risk}")
        
        st.subheader("📈 Confidence Score")
        confidence = calculate_confidence_score(data)
        st.write(f"Confidence: {confidence:.2%}")

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
    intraday_df = results_df[results_df['Intraday'].str.contains("Buy")]
    if "Net_Score" not in intraday_df.columns:
        intraday_df["Net_Score"] = 0
    return intraday_df.sort_values(by="Net_Score", ascending=False).head(5)

def update_progress(progress_bar, loading_text, progress, messages):
    progress_bar.progress(min(progress, 1.0))
    loading_text.text(next(messages))

def main():
    st.sidebar.title("🔧 Settings")
    stock_list = fetch_nse_stock_list()
    sector_options = list(SECTORS.keys()) + ["Custom"]
    selected_sector = st.sidebar.selectbox("Select Sector", sector_options)
    
    if selected_sector == "Custom":
        selected_stocks = st.sidebar.multiselect("Select Stocks", stock_list, default=["RELIANCE.NS"])
    else:
        selected_stocks = SECTORS.get(selected_sector, stock_list)
    
    symbol = st.sidebar.selectbox("Select Stock for Detailed Analysis", selected_stocks)
    
    if st.sidebar.button("🔍 Analyze Selected Stock"):
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data, symbol)
            display_dashboard(symbol, data, recommendations, selected_stocks)
        else:
            st.error(f"❌ Failed to fetch data for {symbol}")
    
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("StockGenie Pro provides advanced stock analysis using technical indicators and machine learning for NSE stocks.")

if __name__ == "__main__":
    main()
