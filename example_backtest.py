"""
Example: Running AI Backtests
===============================================================================
This script demonstrates how to backtest AI predictions on historical data
===============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_backtest import AIBacktester
from ai_features import FeatureEngineer
from ai_model import StockDirectionModel

# Example configuration
SYMBOLS = ['SBIN', 'RELIANCE', 'TCS', 'INFY', 'HDFCBANK']
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)
INITIAL_CAPITAL = 100000


def generate_mock_predictions():
    """
    Generate mock predictions for demonstration.
    In production, these would come from your trained model.
    """
    print("Generating mock predictions...")
    
    predictions = []
    current_date = START_DATE
    
    while current_date <= END_DATE:
        for symbol in SYMBOLS:
            # Simulate ML prediction
            ml_score = np.random.uniform(0.3, 0.9)
            ml_confidence = np.random.uniform(0.5, 0.95)
            
            if ml_score > 0.7:
                ml_direction = 'UP'
                final_signal = 'BUY'
            elif ml_score < 0.4:
                ml_direction = 'DOWN'
                final_signal = 'SELL'
            else:
                ml_direction = 'FLAT'
                final_signal = 'HOLD'
            
            predictions.append({
                'symbol': symbol,
                'ts': current_date,
                'ml_score': ml_score,
                'ml_direction': ml_direction,
                'ml_confidence': ml_confidence,
                'final_signal': final_signal,
                'model_version': 'v1.0.0-mock'
            })
        
        # Move to next week (weekly predictions)
        current_date += timedelta(days=7)
    
    return pd.DataFrame(predictions)


def generate_mock_price_data():
    """
    Generate mock OHLCV data for demonstration.
    In production, this would come from SmartAPI or cached database.
    """
    print("Generating mock price data...")
    
    price_data = {}
    
    for symbol in SYMBOLS:
        dates = pd.date_range(START_DATE, END_DATE, freq='D')
        
        # Simulate price movement (random walk with drift)
        np.random.seed(hash(symbol) % 10000)
        initial_price = np.random.uniform(100, 2000)
        returns = np.random.normal(0.001, 0.02, len(dates))  # Mean 0.1%, Std 2%
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        df = pd.DataFrame({
            'Open': prices * np.random.uniform(0.98, 1.0, len(dates)),
            'High': prices * np.random.uniform(1.0, 1.02, len(dates)),
            'Low': prices * np.random.uniform(0.98, 1.0, len(dates)),
            'Close': prices,
            'Volume': np.random.uniform(1000000, 5000000, len(dates))
        }, index=dates)
        
        price_data[symbol] = df
    
    return price_data


def run_simple_backtest():
    """Run a simple backtest with mock data."""
    print("="*80)
    print("AI BACKTESTING EXAMPLE")
    print("="*80)
    
    # Step 1: Generate data
    predictions_df = generate_mock_predictions()
    price_data = generate_mock_price_data()
    
    print(f"\n✅ Generated {len(predictions_df)} predictions for {len(SYMBOLS)} symbols")
    print(f"✅ Generated price data for period: {START_DATE.date()} to {END_DATE.date()}")
    
    # Step 2: Initialize backtester
    backtester = AIBacktester(
        initial_capital=INITIAL_CAPITAL,
        commission_rate=0.0003,  # 0.03%
        slippage_rate=0.0005,    # 0.05%
        position_sizing='fixed'
    )
    
    print(f"\n✅ Initialized backtester with ₹{INITIAL_CAPITAL:,.0f} capital")
    
    # Step 3: Run backtest
    print("\n🔄 Running backtest...")
    results = backtester.backtest_predictions(
        predictions_df=predictions_df,
        price_data=price_data,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    print("✅ Backtest complete!")
    
    # Step 4: Display results
    backtester.print_report(results)
    
    # Step 5: Show sample trades
    if results['trade_log']:
        print("\n" + "="*80)
        print("SAMPLE TRADES (First 10)")
        print("="*80)
        trades_df = pd.DataFrame(results['trade_log'])
        
        print(trades_df[['symbol', 'entry_date', 'exit_date', 'pnl', 'pnl_percent', 'confidence']].head(10).to_string(index=False))
    
    return results


def run_comparison_backtest():
    """Compare different position sizing strategies."""
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    
    predictions_df = generate_mock_predictions()
    price_data = generate_mock_price_data()
    
    strategies = {
        'Fixed Size': 'fixed',
        'Confidence-Based': 'confidence',
        'Equal Weight': 'equal_weight'
    }
    
    results_comparison = {}
    
    for strategy_name, sizing_method in strategies.items():
        print(f"\n🔄 Testing {strategy_name} strategy...")
        
        backtester = AIBacktester(
            initial_capital=INITIAL_CAPITAL,
            position_sizing=sizing_method
        )
        
        results = backtester.backtest_predictions(
            predictions_df=predictions_df,
            price_data=price_data
        )
        
        results_comparison[strategy_name] = results['metrics']
    
    # Display comparison
    print("\n" + "="*80)
    print("STRATEGY COMPARISON RESULTS")
    print("="*80)
    
    comparison_df = pd.DataFrame(results_comparison).T
    print(comparison_df[['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades']].to_string())
    
    return results_comparison


def analyze_by_confidence():
    """Analyze performance by confidence buckets."""
    print("\n" + "="*80)
    print("CONFIDENCE ANALYSIS")
    print("="*80)
    
    predictions_df = generate_mock_predictions()
    price_data = generate_mock_price_data()
    
    # Run backtest
    backtester = AIBacktester(initial_capital=INITIAL_CAPITAL)
    results = backtester.backtest_predictions(predictions_df, price_data)
    
    # Analyze trades by confidence
    if results['trade_log']:
        trades_df = pd.DataFrame(results['trade_log'])
        
        # Create confidence buckets
        trades_df['confidence_bucket'] = pd.cut(
            trades_df['confidence'], 
            bins=[0, 0.6, 0.75, 0.9, 1.0],
            labels=['Low (0-60%)', 'Medium (60-75%)', 'High (75-90%)', 'Very High (90%+)']
        )
        
        # Group by confidence
        confidence_stats = trades_df.groupby('confidence_bucket').agg({
            'pnl': ['count', 'mean', 'sum'],
            'pnl_percent': 'mean'
        }).round(2)
        
        print("\nPerformance by Confidence Level:")
        print("-"*80)
        print(confidence_stats.to_string())
        
        # Win rate by confidence
        win_rates = trades_df.groupby('confidence_bucket').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100
        ).round(1)
        
        print("\nWin Rate by Confidence:")
        print("-"*80)
        for bucket, rate in win_rates.items():
            print(f"{bucket}: {rate:.1f}%")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("AI BACKTESTING EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates AI model backtesting capabilities.")
    print("\nNote: Using MOCK data for demonstration purposes.")
    print("In production, use real predictions and price data.")
    print("\n" + "="*80)
    
    print("\n[1/3] Running simple backtest...")
    results = run_simple_backtest()
    
    print("\n" + "="*80)
    input("\n⏸️  Press Enter to continue to strategy comparison...")
    
    print("\n[2/3] Comparing position sizing strategies...")
    comparison = run_comparison_backtest()
    
    print("\n" + "="*80)
    input("\n⏸️  Press Enter to continue to confidence analysis...")
    
    print("\n[3/3] Analyzing by confidence levels...")
    analyze_by_confidence()
    
    print("\n" + "="*80)
    print("✅ ALL BACKTESTS COMPLETE!")
    print("="*80)
    
    print("\n📚 Next Steps:")
    print("  1. Replace mock data with real predictions from your trained model")
    print("  2. Use actual price data from SmartAPI")
    print("  3. Save results to Supabase: backtester.save_results_to_supabase()")
    print("  4. View results in AI Insights tab")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
