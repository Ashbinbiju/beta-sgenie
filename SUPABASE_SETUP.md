# 📊 Supabase Setup for Paper Trading

## Step 1: Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Sign in or create an account
3. Click "New Project"
4. Choose organization and enter project details:
   - Name: `stockgenie-paper-trading` (or your choice)
   - Database Password: (choose a strong password)
   - Region: Choose closest to your users
5. Click "Create new project" and wait for setup

## Step 2: Create Database Tables

Go to **SQL Editor** in your Supabase dashboard and run these commands:

### 1. Paper Trading Account Table
```sql
-- Store user account information (cash balance, initial capital)
CREATE TABLE paper_account (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT UNIQUE NOT NULL DEFAULT 'default',
    cash_balance NUMERIC(15, 2) DEFAULT 100000.00,
    initial_balance NUMERIC(15, 2) DEFAULT 100000.00,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default account
INSERT INTO paper_account (user_id, cash_balance, initial_balance) 
VALUES ('default', 100000.00, 100000.00)
ON CONFLICT (user_id) DO NOTHING;

-- Add RLS (Row Level Security) policy
ALTER TABLE paper_account ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Enable all access for paper_account" 
ON paper_account FOR ALL 
USING (true) 
WITH CHECK (true);
```

### 2. Paper Portfolio Table
```sql
-- Store current holdings (open positions)
CREATE TABLE paper_portfolio (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL DEFAULT 'default',
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_price NUMERIC(10, 2) NOT NULL,
    invested_amount NUMERIC(15, 2) NOT NULL,
    trading_style TEXT NOT NULL, -- 'swing' or 'intraday'
    entry_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    stop_loss NUMERIC(10, 2),
    target NUMERIC(10, 2),
    notes TEXT,
    UNIQUE(user_id, symbol, trading_style)
);

-- Add RLS policy
ALTER TABLE paper_portfolio ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Enable all access for paper_portfolio" 
ON paper_portfolio FOR ALL 
USING (true) 
WITH CHECK (true);

-- Create index for faster queries
CREATE INDEX idx_portfolio_user_symbol ON paper_portfolio(user_id, symbol);
```

### 3. Paper Trades History Table
```sql
-- Store all trade executions (buy/sell history)
CREATE TABLE paper_trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL DEFAULT 'default',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    symbol TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('BUY', 'SELL')),
    quantity INTEGER NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    total_amount NUMERIC(15, 2) NOT NULL,
    trading_style TEXT NOT NULL,
    pnl NUMERIC(15, 2) DEFAULT 0,
    pnl_percent NUMERIC(8, 2) DEFAULT 0,
    notes TEXT
);

-- Add RLS policy
ALTER TABLE paper_trades ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Enable all access for paper_trades" 
ON paper_trades FOR ALL 
USING (true) 
WITH CHECK (true);

-- Create indexes
CREATE INDEX idx_trades_user ON paper_trades(user_id);
CREATE INDEX idx_trades_symbol ON paper_trades(symbol);
CREATE INDEX idx_trades_timestamp ON paper_trades(timestamp DESC);
```

### 4. Create Function to Update Timestamp
```sql
-- Function to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add trigger to paper_account
CREATE TRIGGER update_paper_account_updated_at 
BEFORE UPDATE ON paper_account 
FOR EACH ROW 
EXECUTE FUNCTION update_updated_at_column();
```

## Step 3: Get API Credentials

1. Go to **Settings** → **API** in Supabase dashboard
2. Copy these values:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **Anon public key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

## Step 4: Configure Environment Variables

### For Local Development (.env):
```env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### For Streamlit Cloud (Secrets):
```toml
SUPABASE_URL = "https://xxxxx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

## Step 5: Install Required Package

Add to `requirements.txt`:
```
supabase==2.3.0
```

Or install locally:
```bash
pip install supabase
```

## Step 6: Test Connection

After setting up, restart your Streamlit app. You should see:
- A new "📈 Paper Trading" tab
- Initial balance of ₹100,000
- Ability to execute virtual trades

## 🔧 Optional: Advanced Setup

### Enable Real-time Updates (Optional)
```sql
-- Enable real-time for paper_portfolio
ALTER PUBLICATION supabase_realtime ADD TABLE paper_portfolio;

-- Enable real-time for paper_trades
ALTER PUBLICATION supabase_realtime ADD TABLE paper_trades;
```

### Create Views for Analytics
```sql
-- View for portfolio performance
CREATE VIEW portfolio_performance AS
SELECT 
    user_id,
    COUNT(*) as total_positions,
    SUM(invested_amount) as total_invested,
    SUM(quantity * avg_price) as current_value
FROM paper_portfolio
GROUP BY user_id;
```

## 🔒 Security Notes

1. The provided RLS policies allow all access - suitable for demo/personal use
2. For multi-user environment, update policies to filter by authenticated user
3. Never commit `.env` file with actual credentials
4. Use Streamlit secrets for cloud deployment

## 📊 Database Schema Overview

```
paper_account
├── id (UUID)
├── user_id (TEXT) - 'default' for single user
├── cash_balance (NUMERIC) - Available cash
├── initial_balance (NUMERIC) - Starting capital
├── created_at (TIMESTAMP)
└── updated_at (TIMESTAMP)

paper_portfolio
├── id (UUID)
├── user_id (TEXT)
├── symbol (TEXT) - Stock symbol
├── quantity (INTEGER) - Number of shares
├── avg_price (NUMERIC) - Average buy price
├── invested_amount (NUMERIC) - Total invested
├── trading_style (TEXT) - 'swing' or 'intraday'
├── entry_date (TIMESTAMP)
├── stop_loss (NUMERIC)
├── target (NUMERIC)
└── notes (TEXT)

paper_trades
├── id (UUID)
├── user_id (TEXT)
├── timestamp (TIMESTAMP)
├── symbol (TEXT)
├── action (TEXT) - 'BUY' or 'SELL'
├── quantity (INTEGER)
├── price (NUMERIC)
├── total_amount (NUMERIC)
├── trading_style (TEXT)
├── pnl (NUMERIC) - Profit/Loss
├── pnl_percent (NUMERIC) - P&L percentage
└── notes (TEXT)
```

## 🚀 Ready to Use!

Once you complete these steps, the Paper Trading feature will be fully functional with cloud-based persistent storage!
