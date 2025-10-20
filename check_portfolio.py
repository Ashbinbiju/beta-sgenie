"""Check paper trading portfolio in Supabase"""
import os
from supabase import create_client

# Get credentials
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing Supabase credentials in environment")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("=" * 80)
print("📊 PAPER TRADING PORTFOLIO CHECK")
print("=" * 80)

# Check portfolio
print("\n1️⃣ Checking paper_portfolio table...")
try:
    portfolio_response = supabase.table('paper_portfolio').select('*').eq('user_id', 'default').execute()
    portfolio = portfolio_response.data
    
    if portfolio:
        print(f"✅ Found {len(portfolio)} position(s):\n")
        for i, pos in enumerate(portfolio, 1):
            print(f"   Position #{i}:")
            print(f"      Symbol: {pos.get('symbol')}")
            print(f"      Quantity: {pos.get('quantity')}")
            print(f"      Avg Price: ₹{pos.get('avg_price', 0):.2f}")
            print(f"      Invested: ₹{pos.get('invested_amount', 0):,.2f}")
            print(f"      Style: {pos.get('trading_style')}")
            print(f"      Created: {pos.get('created_at', 'N/A')}")
            print()
    else:
        print("⚠️ No positions found in portfolio")
except Exception as e:
    print(f"❌ Error fetching portfolio: {e}")

# Check trades
print("\n2️⃣ Checking paper_trades table (last 10 trades)...")
try:
    trades_response = supabase.table('paper_trades').select('*').eq('user_id', 'default').order('timestamp', desc=True).limit(10).execute()
    trades = trades_response.data
    
    if trades:
        print(f"✅ Found {len(trades)} recent trade(s):\n")
        for i, trade in enumerate(trades, 1):
            print(f"   Trade #{i}:")
            print(f"      Symbol: {trade.get('symbol')}")
            print(f"      Action: {trade.get('action')}")
            print(f"      Quantity: {trade.get('quantity')}")
            print(f"      Price: ₹{trade.get('price', 0):.2f}")
            print(f"      Total: ₹{trade.get('total_amount', 0):,.2f}")
            print(f"      Charges: ₹{trade.get('charges', 0):.2f}")
            print(f"      Style: {trade.get('trading_style')}")
            print(f"      Time: {trade.get('timestamp', 'N/A')}")
            print()
    else:
        print("⚠️ No trades found")
except Exception as e:
    print(f"❌ Error fetching trades: {e}")

# Check account
print("\n3️⃣ Checking paper_account table...")
try:
    account_response = supabase.table('paper_account').select('*').eq('user_id', 'default').execute()
    account = account_response.data
    
    if account:
        acc = account[0]
        print(f"✅ Account found:")
        print(f"      Initial Balance: ₹{acc.get('initial_balance', 0):,.2f}")
        print(f"      Cash Balance: ₹{acc.get('cash_balance', 0):,.2f}")
        print(f"      Created: {acc.get('created_at', 'N/A')}")
    else:
        print("⚠️ No account found")
except Exception as e:
    print(f"❌ Error fetching account: {e}")

print("\n" + "=" * 80)
print("🔍 DIAGNOSIS:")
print("=" * 80)

# Analyze the situation
try:
    portfolio_count = len(portfolio) if portfolio else 0
    trades_count = len(trades) if trades else 0
    
    buy_trades = [t for t in trades if t.get('action') == 'BUY'] if trades else []
    sell_trades = [t for t in trades if t.get('action') == 'SELL'] if trades else []
    
    print(f"\n📊 Summary:")
    print(f"   - Total positions in portfolio: {portfolio_count}")
    print(f"   - Total trades recorded: {trades_count}")
    print(f"   - BUY trades: {len(buy_trades)}")
    print(f"   - SELL trades: {len(sell_trades)}")
    
    # Check for mismatches
    if buy_trades and portfolio_count < len(buy_trades) - len(sell_trades):
        print(f"\n⚠️ ISSUE DETECTED:")
        print(f"   Expected {len(buy_trades) - len(sell_trades)} positions based on trades")
        print(f"   But found only {portfolio_count} positions in portfolio")
        print(f"\n   💡 Possible causes:")
        print(f"      1. Database insert failure (position not created)")
        print(f"      2. Duplicate trading_style causing separate positions")
        print(f"      3. Positions were sold but trades show BUY")
        
        # Check which stocks are in trades but not in portfolio
        trade_symbols = set(t.get('symbol') for t in buy_trades)
        portfolio_symbols = set(p.get('symbol') for p in portfolio) if portfolio else set()
        missing = trade_symbols - portfolio_symbols
        
        if missing:
            print(f"\n   📋 Stocks in BUY trades but NOT in portfolio:")
            for sym in missing:
                print(f"      - {sym}")
    elif portfolio_count > 0:
        print(f"\n✅ Portfolio looks consistent with trade history")
    
except Exception as e:
    print(f"❌ Error during analysis: {e}")

print("\n" + "=" * 80)
