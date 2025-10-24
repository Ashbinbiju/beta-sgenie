#!/usr/bin/env python3
"""
Quick test script to verify Zerodha API integrations
"""

import requests
import json

def test_shareholding(symbol='JKPAPER'):
    """Test shareholding API"""
    print(f"\n{'='*60}")
    print(f"Testing Shareholding API for {symbol}")
    print('='*60)
    
    try:
        url = f"https://zerodha.com/markets/stocks/NSE/{symbol}/shareholdings/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': f'https://zerodha.com/markets/stocks/NSE/{symbol}/',
            'Origin': 'https://zerodha.com'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data and 'Shareholdings' in data:
            shareholdings = json.loads(data['Shareholdings'])
            print(f"✅ Successfully fetched shareholding data")
            print(f"Quarters available: {list(shareholdings.keys())[:3]}...")
            
            # Show latest quarter
            latest_quarter = list(shareholdings.keys())[0]
            latest_data = shareholdings[latest_quarter]
            print(f"\nLatest Quarter ({latest_quarter}):")
            print(f"  Promoter: {latest_data.get('Promoter', 0):.2f}%")
            print(f"  FII: {latest_data.get('FII', 0):.2f}%")
            print(f"  DII: {latest_data.get('DII', 0):.2f}%")
            print(f"  Pledge: {latest_data.get('Pledge', 0):.2f}%")
            return True
        else:
            print(f"❌ No shareholding data found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_financials(symbol='JKPAPER'):
    """Test financials API"""
    print(f"\n{'='*60}")
    print(f"Testing Financials API for {symbol}")
    print('='*60)
    
    try:
        url = f"https://zerodha.com/markets/stocks/NSE/{symbol}/financials/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': f'https://zerodha.com/markets/stocks/NSE/{symbol}/',
            'Origin': 'https://zerodha.com'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data:
            print(f"✅ Successfully fetched financial data")
            print(f"Available sections: {list(data.keys())}")
            
            # Parse Profit & Loss
            if 'Profit & Loss' in data:
                pl_data = json.loads(data['Profit & Loss'])
                if 'yearly' in pl_data:
                    latest_year = list(pl_data['yearly'].keys())[0]
                    year_data = pl_data['yearly'][latest_year]
                    print(f"\nProfit & Loss ({latest_year}):")
                    print(f"  Sales: ₹{year_data.get('Sales', 0):.0f} Cr")
                    print(f"  Operating Profit: ₹{year_data.get('Operating Profit', 0):.0f} Cr")
                    print(f"  Net Profit: ₹{year_data.get('Net Profit', 0):.0f} Cr")
            
            # Parse Financial Ratios
            if 'Financial Ratios' in data:
                ratios = json.loads(data['Financial Ratios'])
                latest_year = list(ratios.keys())[0]
                ratio_data = ratios[latest_year]
                print(f"\nFinancial Ratios ({latest_year}):")
                print(f"  EPS: ₹{ratio_data.get('Earnings Per Share (EPS)', 0):.2f}")
                print(f"  NPM: {ratio_data.get('Net Profit Margin', 0):.2f}%")
                print(f"  EV/EBITDA: {ratio_data.get('EV/EBITDA', 0):.2f}")
            
            return True
        else:
            print(f"❌ No financial data found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_technical(symbol='JKPAPER', timeframe='15min'):
    """Test technical analysis API"""
    print(f"\n{'='*60}")
    print(f"Testing Technical Analysis API for {symbol} ({timeframe})")
    print('='*60)
    
    try:
        url = f"https://technicalwidget.streak.tech/api/streak_tech_analysis/?timeFrame={timeframe}&stock=NSE:{symbol}&user_id="
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data and data.get('status') == 1:
            print(f"✅ Successfully fetched technical data")
            
            # Overall signal
            state = data.get('state', 0)
            signal = "BULLISH 🟢" if state == 1 else "BEARISH 🔴" if state == -1 else "NEUTRAL ⚪"
            print(f"\nOverall Signal: {signal}")
            print(f"Win Rate: {data.get('win_pct', 0)*100:.1f}%")
            print(f"Total Signals: {data.get('signals', 0)}")
            
            print(f"\nKey Indicators:")
            print(f"  RSI: {data.get('rsi', 0):.2f}")
            print(f"  MACD: {data.get('macd', 0):.4f}")
            print(f"  ADX: {data.get('adx', 0):.2f}")
            print(f"  CCI: {data.get('cci', 0):.2f}")
            
            print(f"\nMoving Averages:")
            print(f"  SMA20: {data.get('sma20', 0):.2f}")
            print(f"  EMA20: {data.get('ema20', 0):.2f}")
            
            return True
        else:
            print(f"❌ No technical data found or API returned error")
            print(f"Response: {data}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_support_resistance(symbol='JKPAPER', timeframe='5min'):
    """Test support/resistance API"""
    print(f"\n{'='*60}")
    print(f"Testing Support/Resistance API for {symbol} ({timeframe})")
    print('='*60)
    
    try:
        url = "https://mo.streak.tech/api/sr_analysis_multi/"
        
        payload = {
            "time_frame": timeframe,
            "stocks": [f"NSE_{symbol}"],
            "user_broker_id": "ZMS"
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data and 'data' in data and f"NSE_{symbol}" in data['data']:
            sr_data = data['data'][f"NSE_{symbol}"]
            print(f"✅ Successfully fetched support/resistance data")
            
            print(f"\nCurrent Price: ₹{sr_data.get('close', 0):.2f}")
            print(f"Pivot Point: ₹{sr_data.get('pp', 0):.2f}")
            
            print(f"\nResistance Levels:")
            print(f"  R1: ₹{sr_data.get('r1', 0):.2f}")
            print(f"  R2: ₹{sr_data.get('r2', 0):.2f}")
            print(f"  R3: ₹{sr_data.get('r3', 0):.2f}")
            
            print(f"\nSupport Levels:")
            print(f"  S1: ₹{sr_data.get('s1', 0):.2f}")
            print(f"  S2: ₹{sr_data.get('s2', 0):.2f}")
            print(f"  S3: ₹{sr_data.get('s3', 0):.2f}")
            
            return True
        else:
            print(f"❌ No support/resistance data found")
            print(f"Response: {data}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_candlestick(symbol='JKPAPER', timeframe='hour'):
    """Test candlestick data API"""
    print(f"\n{'='*60}")
    print(f"Testing Candlestick Data API for {symbol} ({timeframe})")
    print('='*60)
    
    try:
        url = f"https://technicalwidget.streak.tech/api/candles/?stock=NSE:{symbol}&timeFrame={timeframe}&user_id="
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data and len(data) > 0:
            print(f"✅ Successfully fetched candlestick data")
            print(f"Total Candles: {len(data)}")
            
            # Show latest candle
            latest = data[-1]
            print(f"\nLatest Candle ({latest[0]}):")
            print(f"  Open:   ₹{latest[1]:.2f}")
            print(f"  High:   ₹{latest[2]:.2f}")
            print(f"  Low:    ₹{latest[3]:.2f}")
            print(f"  Close:  ₹{latest[4]:.2f}")
            print(f"  Volume: {latest[5]:,}")
            
            # Calculate some basic stats
            closes = [candle[4] for candle in data[-10:]]
            avg_close = sum(closes) / len(closes)
            print(f"\n10-Candle Average Close: ₹{avg_close:.2f}")
            
            return True
        else:
            print(f"❌ No candlestick data found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_support_resistance(symbol='JKPAPER'):
    """Test support and resistance API"""
    print(f"\n{'='*60}")
    print(f"Testing Support & Resistance API for {symbol}")
    print('='*60)
    
    try:
        url = "https://mo.streak.tech/api/sr_analysis_multi/"
        
        payload = {
            "time_frame": "5min",
            "stocks": [f"NSE_{symbol}"],
            "user_broker_id": "ZMS"
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data and 'data' in data and f"NSE_{symbol}" in data['data']:
            sr_data = data['data'][f"NSE_{symbol}"]
            print(f"✅ Successfully fetched support & resistance data")
            
            print(f"\nCurrent Price: ₹{sr_data.get('close', 0):.2f}")
            print(f"Pivot Point: ₹{sr_data.get('pp', 0):.2f}")
            
            print(f"\nResistance Levels:")
            print(f"  R1: ₹{sr_data.get('r1', 0):.2f}")
            print(f"  R2: ₹{sr_data.get('r2', 0):.2f}")
            print(f"  R3: ₹{sr_data.get('r3', 0):.2f}")
            
            print(f"\nSupport Levels:")
            print(f"  S1: ₹{sr_data.get('s1', 0):.2f}")
            print(f"  S2: ₹{sr_data.get('s2', 0):.2f}")
            print(f"  S3: ₹{sr_data.get('s3', 0):.2f}")
            
            return True
        else:
            print(f"❌ No support/resistance data found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_candle_data(symbol='JKPAPER', timeframe='hour'):
    """Test candlestick data API"""
    print(f"\n{'='*60}")
    print(f"Testing Candlestick Data API for {symbol} ({timeframe})")
    print('='*60)
    
    try:
        url = f"https://technicalwidget.streak.tech/api/candles/?stock=NSE:{symbol}&timeFrame={timeframe}&user_id="
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data and len(data) > 0:
            print(f"✅ Successfully fetched candlestick data")
            print(f"Total candles: {len(data)}")
            
            # Show first and last candle
            first_candle = data[0]
            last_candle = data[-1]
            
            print(f"\nFirst Candle ({first_candle[0]}):")
            print(f"  O: {first_candle[1]:.2f}, H: {first_candle[2]:.2f}, L: {first_candle[3]:.2f}, C: {first_candle[4]:.2f}, V: {first_candle[5]:,}")
            
            print(f"\nLatest Candle ({last_candle[0]}):")
            print(f"  O: {last_candle[1]:.2f}, H: {last_candle[2]:.2f}, L: {last_candle[3]:.2f}, C: {last_candle[4]:.2f}, V: {last_candle[5]:,}")
            
            return True
        else:
            print(f"❌ No candlestick data found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ZERODHA API INTEGRATION TEST")
    print("="*60)
    
    test_symbol = input("\nEnter symbol to test (default: JKPAPER): ").strip().upper() or "JKPAPER"
    
    results = {
        'Shareholding': test_shareholding(test_symbol),
        'Financials': test_financials(test_symbol),
        'Technical': test_technical(test_symbol, '15min'),
        'Support/Resistance': test_support_resistance(test_symbol, '5min'),
        'Candlestick Data': test_candlestick(test_symbol, 'hour')
    }
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    for api, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{api:20s}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! APIs are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    print('='*60 + "\n")
