-- ============================================================================
-- AI Stock Genie - Supabase Schema
-- ============================================================================
-- Purpose: Database schema for AI-enhanced stock recommendations
-- Tables: ai_features, ai_predictions, model_metadata, backtest_results
-- Created: 2025-10-20
-- ============================================================================

-- 1. AI Features Table
-- Stores computed features for ML models (cached daily)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ai_features (
    symbol TEXT NOT NULL,
    ts TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    
    -- Price-based features
    returns_1d DOUBLE PRECISION,
    returns_5d DOUBLE PRECISION,
    returns_20d DOUBLE PRECISION,
    volatility_10d DOUBLE PRECISION,
    volatility_30d DOUBLE PRECISION,
    
    -- Technical indicators (from existing system)
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_hist DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,
    ema_9 DOUBLE PRECISION,
    ema_21 DOUBLE PRECISION,
    ema_50 DOUBLE PRECISION,
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    
    -- Volume features
    volume DOUBLE PRECISION,
    volume_ratio_20d DOUBLE PRECISION,
    obv DOUBLE PRECISION,
    vwap DOUBLE PRECISION,
    
    -- Momentum features
    momentum_5d DOUBLE PRECISION,
    momentum_20d DOUBLE PRECISION,
    roc_10d DOUBLE PRECISION,
    
    -- Trend features
    trend_strength DOUBLE PRECISION,
    support_level DOUBLE PRECISION,
    resistance_level DOUBLE PRECISION,
    distance_from_52w_high DOUBLE PRECISION,
    distance_from_52w_low DOUBLE PRECISION,
    
    -- Market context
    sector TEXT,
    sector_strength DOUBLE PRECISION,
    market_regime TEXT, -- 'bullish', 'bearish', 'sideways'
    correlation_to_nifty DOUBLE PRECISION,
    
    -- Sentiment (optional, can be NULL initially)
    sentiment_score DOUBLE PRECISION,
    news_count_7d INTEGER,
    
    -- Metadata
    feature_version TEXT DEFAULT 'v1.0',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    PRIMARY KEY (symbol, ts)
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_ai_features_ts ON ai_features(ts DESC);
CREATE INDEX IF NOT EXISTS idx_ai_features_sector ON ai_features(sector);
CREATE INDEX IF NOT EXISTS idx_ai_features_created ON ai_features(created_at DESC);

-- ============================================================================
-- 2. AI Predictions Table
-- Stores model predictions and recommendations
-- ============================================================================
CREATE TABLE IF NOT EXISTS ai_predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol TEXT NOT NULL,
    ts TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    
    -- Model outputs
    ml_score DOUBLE PRECISION NOT NULL, -- LightGBM probability [0-1]
    ml_direction TEXT, -- 'UP', 'DOWN', 'FLAT'
    ml_confidence DOUBLE PRECISION, -- Model confidence [0-1]
    
    -- Technical score from existing system
    technical_score DOUBLE PRECISION,
    technical_signal TEXT, -- 'Buy', 'Sell', 'Hold'
    
    -- Sentiment (optional)
    sentiment_score DOUBLE PRECISION,
    
    -- Fusion score (combined)
    final_score DOUBLE PRECISION NOT NULL,
    final_signal TEXT NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    
    -- Risk metrics
    predicted_return DOUBLE PRECISION, -- Expected return %
    predicted_volatility DOUBLE PRECISION, -- Expected volatility
    risk_reward_ratio DOUBLE PRECISION,
    max_loss_percentage DOUBLE PRECISION, -- Suggested stop loss
    
    -- Position sizing
    suggested_position_size INTEGER, -- Number of shares
    suggested_allocation_percent DOUBLE PRECISION, -- % of portfolio
    
    -- Explainability
    reasons JSONB, -- {"technical": {...}, "ml": {...}, "sentiment": {...}}
    top_features JSONB, -- Top 5 contributing features
    confidence_breakdown JSONB, -- Breakdown by component
    
    -- Model metadata
    model_version TEXT NOT NULL,
    model_type TEXT DEFAULT 'lightgbm', -- 'lightgbm', 'lstm', 'ensemble'
    
    -- Performance tracking (filled after realization)
    actual_return_1d DOUBLE PRECISION,
    actual_return_5d DOUBLE PRECISION,
    actual_return_10d DOUBLE PRECISION,
    prediction_correct BOOLEAN,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    CONSTRAINT valid_scores CHECK (
        ml_score >= 0 AND ml_score <= 1 AND
        final_score >= 0 AND final_score <= 100
    )
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ai_predictions_symbol ON ai_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_ai_predictions_ts ON ai_predictions(ts DESC);
CREATE INDEX IF NOT EXISTS idx_ai_predictions_signal ON ai_predictions(final_signal);
CREATE INDEX IF NOT EXISTS idx_ai_predictions_score ON ai_predictions(final_score DESC);
CREATE INDEX IF NOT EXISTS idx_ai_predictions_created ON ai_predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_predictions_model ON ai_predictions(model_version);

-- ============================================================================
-- 3. Model Metadata Table
-- Tracks model versions, training history, and performance
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_metadata (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model_version TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL, -- 'lightgbm', 'lstm', 'ensemble'
    
    -- Training info
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    training_period_start DATE,
    training_period_end DATE,
    num_training_samples INTEGER,
    num_stocks_trained INTEGER,
    
    -- Hyperparameters
    hyperparameters JSONB, -- Store all hyperparams as JSON
    
    -- Training metrics
    train_accuracy DOUBLE PRECISION,
    val_accuracy DOUBLE PRECISION,
    train_loss DOUBLE PRECISION,
    val_loss DOUBLE PRECISION,
    
    -- Feature importance
    feature_importance JSONB, -- Top features and their importance scores
    
    -- Backtest performance
    backtest_sharpe DOUBLE PRECISION,
    backtest_win_rate DOUBLE PRECISION,
    backtest_avg_return DOUBLE PRECISION,
    backtest_max_drawdown DOUBLE PRECISION,
    backtest_total_trades INTEGER,
    
    -- Live performance (updated daily)
    live_predictions_count INTEGER DEFAULT 0,
    live_correct_predictions INTEGER DEFAULT 0,
    live_accuracy DOUBLE PRECISION,
    live_avg_return DOUBLE PRECISION,
    
    -- Status
    status TEXT DEFAULT 'active', -- 'active', 'retired', 'testing'
    is_production BOOLEAN DEFAULT false,
    
    -- Notes
    notes TEXT,
    
    -- Metadata
    created_by TEXT DEFAULT 'system',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_model_metadata_version ON model_metadata(model_version);
CREATE INDEX IF NOT EXISTS idx_model_metadata_status ON model_metadata(status);
CREATE INDEX IF NOT EXISTS idx_model_metadata_production ON model_metadata(is_production);
CREATE INDEX IF NOT EXISTS idx_model_metadata_created ON model_metadata(created_at DESC);

-- ============================================================================
-- 4. Backtest Results Table
-- Stores backtest simulation results for different strategies
-- ============================================================================
CREATE TABLE IF NOT EXISTS backtest_results (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    
    -- Identification
    backtest_name TEXT NOT NULL,
    model_version TEXT,
    symbol TEXT, -- NULL for portfolio-level backtest
    
    -- Time period
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    num_trading_days INTEGER,
    
    -- Performance metrics
    total_return DOUBLE PRECISION,
    annualized_return DOUBLE PRECISION,
    cagr DOUBLE PRECISION,
    
    -- Risk metrics
    volatility DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    sortino_ratio DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    max_drawdown_duration_days INTEGER,
    
    -- Trade statistics
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate DOUBLE PRECISION,
    avg_win DOUBLE PRECISION,
    avg_loss DOUBLE PRECISION,
    avg_trade_return DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION, -- sum(wins) / sum(losses)
    expectancy DOUBLE PRECISION,
    
    -- Additional metrics
    avg_trade_duration_days DOUBLE PRECISION,
    max_consecutive_wins INTEGER,
    max_consecutive_losses INTEGER,
    recovery_factor DOUBLE PRECISION, -- net_profit / max_drawdown
    
    -- Capital metrics
    initial_capital DOUBLE PRECISION,
    final_capital DOUBLE PRECISION,
    peak_capital DOUBLE PRECISION,
    
    -- Detailed results (optional)
    trade_log JSONB, -- Array of individual trades
    equity_curve JSONB, -- Array of [date, equity] pairs
    monthly_returns JSONB, -- Monthly return breakdown
    
    -- Backtest parameters
    commission_rate DOUBLE PRECISION DEFAULT 0.0003,
    slippage_rate DOUBLE PRECISION DEFAULT 0.0005,
    position_sizing_method TEXT DEFAULT 'fixed',
    
    -- Metadata
    notes TEXT,
    run_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_backtest_results_model ON backtest_results(model_version);
CREATE INDEX IF NOT EXISTS idx_backtest_results_symbol ON backtest_results(symbol);
CREATE INDEX IF NOT EXISTS idx_backtest_results_dates ON backtest_results(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_backtest_results_sharpe ON backtest_results(sharpe_ratio DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_results_created ON backtest_results(created_at DESC);

-- ============================================================================
-- 5. User Feedback Table (for continuous learning)
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    
    -- Reference
    prediction_id UUID REFERENCES ai_predictions(id),
    user_id TEXT DEFAULT 'default',
    symbol TEXT NOT NULL,
    
    -- Feedback
    action_taken TEXT, -- 'bought', 'sold', 'ignored', 'paper_traded'
    was_successful BOOLEAN,
    actual_pnl DOUBLE PRECISION,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5), -- 1-5 stars
    
    -- Comments
    feedback_text TEXT,
    improvement_suggestions TEXT,
    
    -- Metadata
    feedback_date TIMESTAMP WITH TIME ZONE DEFAULT now(),
    trade_date TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_user_feedback_prediction ON user_feedback(prediction_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_symbol ON user_feedback(symbol);
CREATE INDEX IF NOT EXISTS idx_user_feedback_user ON user_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_date ON user_feedback(feedback_date DESC);

-- ============================================================================
-- Views for Easy Querying
-- ============================================================================

-- Latest predictions by score
CREATE OR REPLACE VIEW v_latest_ai_predictions AS
SELECT DISTINCT ON (symbol)
    symbol,
    final_score,
    final_signal,
    ml_score,
    technical_score,
    sentiment_score,
    predicted_return,
    risk_reward_ratio,
    suggested_position_size,
    reasons,
    model_version,
    created_at
FROM ai_predictions
ORDER BY symbol, created_at DESC;

-- Active model performance
CREATE OR REPLACE VIEW v_active_model_performance AS
SELECT
    model_version,
    model_type,
    trained_at,
    backtest_sharpe,
    backtest_win_rate,
    live_accuracy,
    live_predictions_count,
    status
FROM model_metadata
WHERE status = 'active'
ORDER BY trained_at DESC;

-- Recent backtest summary
CREATE OR REPLACE VIEW v_backtest_summary AS
SELECT
    backtest_name,
    model_version,
    symbol,
    start_date,
    end_date,
    total_return,
    sharpe_ratio,
    win_rate,
    total_trades,
    max_drawdown,
    run_at
FROM backtest_results
ORDER BY run_at DESC
LIMIT 50;

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to calculate prediction accuracy
CREATE OR REPLACE FUNCTION calculate_prediction_accuracy(
    model_ver TEXT,
    lookback_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    accuracy DOUBLE PRECISION,
    total_predictions BIGINT,
    correct_predictions BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        (COUNT(*) FILTER (WHERE prediction_correct = true)::DOUBLE PRECISION / 
         NULLIF(COUNT(*), 0)) * 100 AS accuracy,
        COUNT(*) AS total_predictions,
        COUNT(*) FILTER (WHERE prediction_correct = true) AS correct_predictions
    FROM ai_predictions
    WHERE model_version = model_ver
        AND created_at >= NOW() - (lookback_days || ' days')::INTERVAL
        AND prediction_correct IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- Function to get top performing stocks
CREATE OR REPLACE FUNCTION get_top_ai_picks(
    limit_count INTEGER DEFAULT 10,
    min_score DOUBLE PRECISION DEFAULT 70
)
RETURNS TABLE (
    symbol TEXT,
    final_score DOUBLE PRECISION,
    final_signal TEXT,
    predicted_return DOUBLE PRECISION,
    risk_reward_ratio DOUBLE PRECISION,
    model_version TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT ON (p.symbol)
        p.symbol,
        p.final_score,
        p.final_signal,
        p.predicted_return,
        p.risk_reward_ratio,
        p.model_version
    FROM ai_predictions p
    WHERE p.final_score >= min_score
        AND p.final_signal = 'BUY'
        AND p.created_at >= NOW() - INTERVAL '24 hours'
    ORDER BY p.symbol, p.created_at DESC, p.final_score DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Initial Data / Configuration
-- ============================================================================

-- Insert default model metadata (placeholder)
INSERT INTO model_metadata (
    model_version,
    model_type,
    status,
    is_production,
    notes
) VALUES (
    'v1.0.0-baseline',
    'lightgbm',
    'active',
    true,
    'Initial baseline model using LightGBM on technical features'
) ON CONFLICT (model_version) DO NOTHING;

-- ============================================================================
-- Grants (adjust based on your Supabase RLS policies)
-- ============================================================================

-- For service role: full access (already granted by default)
-- For authenticated users: read-only access to predictions
-- For anon users: no access

-- Example RLS policies (commented out - enable if using Supabase Auth)
/*
ALTER TABLE ai_predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read access to predictions"
ON ai_predictions FOR SELECT
TO authenticated, anon
USING (created_at >= NOW() - INTERVAL '30 days');

CREATE POLICY "Service role full access"
ON ai_predictions FOR ALL
TO service_role
USING (true)
WITH CHECK (true);
*/

-- ============================================================================
-- Maintenance / Cleanup
-- ============================================================================

-- Function to clean old predictions (run monthly)
CREATE OR REPLACE FUNCTION cleanup_old_predictions(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM ai_predictions
    WHERE created_at < NOW() - (days_to_keep || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

-- Usage instructions:
-- 1. Copy this SQL and run it in Supabase SQL Editor
-- 2. Verify tables are created: SELECT * FROM pg_tables WHERE schemaname = 'public';
-- 3. Test views: SELECT * FROM v_latest_ai_predictions LIMIT 5;
-- 4. Test functions: SELECT * FROM get_top_ai_picks(10, 70);
