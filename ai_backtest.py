"""
AI Backtesting Framework
===============================================================================
Purpose: Validate AI model performance on historical data with walk-forward testing
Author: StockGenie Pro AI Enhancement
Created: 2025-10-20
===============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIBacktester:
    """
    Backtests AI model predictions using walk-forward validation.
    Calculates comprehensive performance metrics including Sharpe, drawdown, win rate.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission_rate: float = 0.0003,
                 slippage_rate: float = 0.0005,
                 position_sizing: str = 'fixed'):
        """
        Initialize backtester with trading parameters.
        
        Args:
            initial_capital: Starting capital in INR
            commission_rate: Brokerage as fraction (0.03%)
            slippage_rate: Slippage as fraction (0.05%)
            position_sizing: 'fixed', 'equal_weight', or 'confidence'
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.position_sizing = position_sizing
        
        # Trading parameters
        self.stt_rate = 0.001  # Securities Transaction Tax (0.1%)
        self.max_positions = 10  # Maximum concurrent positions
        self.position_size = 10000  # Fixed position size in INR
        
    def backtest_predictions(self,
                            predictions_df: pd.DataFrame,
                            price_data: Dict[str, pd.DataFrame],
                            start_date: datetime = None,
                            end_date: datetime = None) -> Dict:
        """
        Backtest AI predictions against actual price movements.
        
        Args:
            predictions_df: DataFrame with columns: symbol, ts, ml_direction, ml_confidence, final_signal
            price_data: Dictionary of {symbol: OHLCV DataFrame}
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with backtest results and metrics
        """
        try:
            # Filter by date range
            if start_date:
                predictions_df = predictions_df[predictions_df['ts'] >= start_date]
            if end_date:
                predictions_df = predictions_df[predictions_df['ts'] <= end_date]
            
            # Sort by timestamp
            predictions_df = predictions_df.sort_values('ts').reset_index(drop=True)
            
            # Initialize tracking
            capital = self.initial_capital
            positions = {}  # {symbol: {'entry_price', 'quantity', 'entry_date', 'confidence'}}
            trade_log = []
            equity_curve = [(predictions_df['ts'].min(), capital)]
            
            # Process each prediction
            for idx, pred in predictions_df.iterrows():
                symbol = pred['symbol']
                pred_date = pred['ts']
                signal = pred.get('final_signal', pred.get('ml_direction', 'HOLD'))
                confidence = pred.get('ml_confidence', 0.5)
                
                # Get price data for this symbol
                if symbol not in price_data:
                    continue
                
                price_df = price_data[symbol]
                
                # Find next available price (after prediction)
                future_prices = price_df[price_df.index > pred_date]
                
                if future_prices.empty:
                    continue
                
                entry_price = future_prices['Close'].iloc[0]
                entry_date = future_prices.index[0]
                
                # BUY Signal
                if signal in ['BUY', 'UP'] and symbol not in positions and len(positions) < self.max_positions:
                    # Calculate position size
                    if self.position_sizing == 'fixed':
                        position_value = self.position_size
                    elif self.position_sizing == 'confidence':
                        position_value = self.position_size * confidence
                    else:  # equal_weight
                        position_value = capital / self.max_positions
                    
                    # Check if we have enough capital
                    if capital < position_value:
                        continue
                    
                    # Apply slippage and commission
                    actual_entry_price = entry_price * (1 + self.slippage_rate)
                    quantity = int(position_value / actual_entry_price)
                    
                    if quantity <= 0:
                        continue
                    
                    total_cost = quantity * actual_entry_price
                    commission = total_cost * self.commission_rate
                    
                    # Enter position
                    positions[symbol] = {
                        'entry_price': actual_entry_price,
                        'quantity': quantity,
                        'entry_date': entry_date,
                        'confidence': confidence,
                        'entry_commission': commission
                    }
                    
                    capital -= (total_cost + commission)
                    
                    logger.debug(f"BUY {symbol}: {quantity} @ ₹{actual_entry_price:.2f} (Confidence: {confidence:.2%})")
                
                # SELL Signal or manage existing position
                elif symbol in positions:
                    position = positions[symbol]
                    
                    # Determine exit condition
                    should_exit = False
                    exit_reason = None
                    
                    # Exit on SELL signal
                    if signal in ['SELL', 'DOWN']:
                        should_exit = True
                        exit_reason = 'signal'
                    
                    # Exit after holding period (e.g., 10 days)
                    days_held = (pred_date - position['entry_date']).days
                    if days_held >= 10:
                        should_exit = True
                        exit_reason = 'holding_period'
                    
                    # Exit logic
                    if should_exit:
                        # Find exit price
                        exit_prices = price_df[price_df.index > pred_date]
                        if exit_prices.empty:
                            continue
                        
                        exit_price = exit_prices['Close'].iloc[0]
                        exit_date = exit_prices.index[0]
                        
                        # Apply slippage, commission, and STT
                        actual_exit_price = exit_price * (1 - self.slippage_rate)
                        exit_value = position['quantity'] * actual_exit_price
                        exit_commission = exit_value * self.commission_rate
                        stt = exit_value * self.stt_rate
                        
                        # Calculate P&L
                        entry_value = position['quantity'] * position['entry_price']
                        total_charges = position['entry_commission'] + exit_commission + stt
                        pnl = (exit_value - entry_value) - total_charges
                        pnl_percent = (pnl / entry_value) * 100
                        
                        # Record trade
                        trade_log.append({
                            'symbol': symbol,
                            'entry_date': position['entry_date'],
                            'exit_date': exit_date,
                            'entry_price': position['entry_price'],
                            'exit_price': actual_exit_price,
                            'quantity': position['quantity'],
                            'pnl': pnl,
                            'pnl_percent': pnl_percent,
                            'holding_days': (exit_date - position['entry_date']).days,
                            'confidence': position['confidence'],
                            'exit_reason': exit_reason,
                            'charges': total_charges
                        })
                        
                        # Update capital
                        capital += exit_value - exit_commission - stt
                        
                        # Remove position
                        del positions[symbol]
                        
                        logger.debug(f"SELL {symbol}: P&L ₹{pnl:.2f} ({pnl_percent:+.2f}%)")
                
                # Update equity curve
                portfolio_value = capital + sum(
                    pos['quantity'] * self._get_current_price(symbol, pred_date, price_data)
                    for symbol, pos in positions.items()
                )
                equity_curve.append((pred_date, portfolio_value))
            
            # Close all remaining positions at end
            for symbol, position in list(positions.items()):
                if symbol in price_data:
                    final_price = price_data[symbol]['Close'].iloc[-1]
                    final_date = price_data[symbol].index[-1]
                    
                    actual_exit_price = final_price * (1 - self.slippage_rate)
                    exit_value = position['quantity'] * actual_exit_price
                    exit_commission = exit_value * self.commission_rate
                    stt = exit_value * self.stt_rate
                    
                    entry_value = position['quantity'] * position['entry_price']
                    total_charges = position['entry_commission'] + exit_commission + stt
                    pnl = (exit_value - entry_value) - total_charges
                    pnl_percent = (pnl / entry_value) * 100
                    
                    trade_log.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': final_date,
                        'entry_price': position['entry_price'],
                        'exit_price': actual_exit_price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'pnl_percent': pnl_percent,
                        'holding_days': (final_date - position['entry_date']).days,
                        'confidence': position['confidence'],
                        'exit_reason': 'end_of_period',
                        'charges': total_charges
                    })
                    
                    capital += exit_value - exit_commission - stt
            
            # Calculate metrics
            metrics = self._calculate_metrics(trade_log, equity_curve, predictions_df)
            
            return {
                'metrics': metrics,
                'trade_log': trade_log,
                'equity_curve': equity_curve,
                'final_capital': capital,
                'num_predictions': len(predictions_df),
                'num_trades': len(trade_log)
            }
        
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            raise
    
    def _get_current_price(self, symbol: str, date: datetime, price_data: Dict) -> float:
        """Get current price for symbol at given date."""
        if symbol not in price_data:
            return 0
        
        df = price_data[symbol]
        future_prices = df[df.index >= date]
        
        if future_prices.empty:
            return df['Close'].iloc[-1] if not df.empty else 0
        
        return future_prices['Close'].iloc[0]
    
    def _calculate_metrics(self, 
                          trade_log: List[Dict], 
                          equity_curve: List[Tuple],
                          predictions_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive backtest metrics."""
        metrics = {}
        
        if not trade_log:
            logger.warning("No trades executed in backtest")
            return {
                'total_return': 0, 'annualized_return': 0, 'sharpe_ratio': 0,
                'max_drawdown': 0, 'win_rate': 0, 'total_trades': 0
            }
        
        # Convert to DataFrames
        trades_df = pd.DataFrame(trade_log)
        equity_df = pd.DataFrame(equity_curve, columns=['date', 'equity'])
        
        # Total return
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        # Time period
        start_date = equity_df['date'].min()
        end_date = equity_df['date'].max()
        days = (end_date - start_date).days
        years = days / 365.25
        
        # Annualized return (CAGR)
        if years > 0:
            cagr = (((final_equity / self.initial_capital) ** (1 / years)) - 1) * 100
        else:
            cagr = 0
        
        # Calculate returns for Sharpe
        equity_df['returns'] = equity_df['equity'].pct_change()
        returns = equity_df['returns'].dropna()
        
        # Sharpe Ratio (assuming risk-free rate = 0)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 1 and negative_returns.std() > 0:
            sortino_ratio = (returns.mean() / negative_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # Maximum Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min() * 100
        
        # Drawdown duration
        drawdown_periods = []
        in_drawdown = False
        drawdown_start = None
        
        for idx, row in equity_df.iterrows():
            if row['drawdown'] < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = row['date']
            elif row['drawdown'] >= 0 and in_drawdown:
                in_drawdown = False
                if drawdown_start:
                    duration = (row['date'] - drawdown_start).days
                    drawdown_periods.append(duration)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Trade statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        avg_trade_return = trades_df['pnl_percent'].mean()
        
        # Profit factor
        total_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Expectancy
        expectancy = (avg_win * win_rate/100) - (abs(avg_loss) * (1 - win_rate/100))
        
        # Consecutive wins/losses
        trades_df['win'] = trades_df['pnl'] > 0
        trades_df['streak'] = (trades_df['win'] != trades_df['win'].shift()).cumsum()
        win_streaks = trades_df[trades_df['win']].groupby('streak').size()
        loss_streaks = trades_df[~trades_df['win']].groupby('streak').size()
        
        max_consecutive_wins = win_streaks.max() if not win_streaks.empty else 0
        max_consecutive_losses = loss_streaks.max() if not loss_streaks.empty else 0
        
        # Recovery factor
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Average holding period
        avg_holding_days = trades_df['holding_days'].mean()
        
        # Prediction accuracy (if available)
        if 'ml_direction' in predictions_df.columns and not trades_df.empty:
            # Match predictions with trades
            correct_predictions = sum(
                1 for _, trade in trades_df.iterrows()
                if trade['pnl'] > 0  # Simplified: profit = correct prediction
            )
            prediction_accuracy = (correct_predictions / total_trades) * 100 if total_trades > 0 else 0
        else:
            prediction_accuracy = None
        
        # Compile metrics
        metrics = {
            # Return metrics
            'total_return': round(total_return, 2),
            'annualized_return': round(cagr, 2),
            'cagr': round(cagr, 2),
            
            # Risk metrics
            'volatility': round(returns.std() * np.sqrt(252) * 100, 2) if len(returns) > 0 else 0,
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_duration_days': int(max_drawdown_duration),
            
            # Trade statistics
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_trade_return': round(avg_trade_return, 2),
            'profit_factor': round(profit_factor, 2),
            'expectancy': round(expectancy, 2),
            
            # Streak statistics
            'max_consecutive_wins': int(max_consecutive_wins),
            'max_consecutive_losses': int(max_consecutive_losses),
            
            # Additional metrics
            'recovery_factor': round(recovery_factor, 2),
            'avg_trade_duration_days': round(avg_holding_days, 1),
            
            # Capital metrics
            'initial_capital': round(self.initial_capital, 2),
            'final_capital': round(final_equity, 2),
            'peak_capital': round(equity_df['peak'].max(), 2),
            
            # Prediction accuracy
            'prediction_accuracy': round(prediction_accuracy, 2) if prediction_accuracy else None,
            
            # Time period
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'num_trading_days': int(days)
        }
        
        return metrics
    
    def save_results_to_supabase(self, 
                                 results: Dict,
                                 supabase_client,
                                 backtest_name: str,
                                 model_version: str = None,
                                 notes: str = None):
        """Save backtest results to Supabase."""
        try:
            # Prepare record
            record = {
                'backtest_name': backtest_name,
                'model_version': model_version,
                'start_date': results['metrics']['start_date'],
                'end_date': results['metrics']['end_date'],
                'num_trading_days': results['metrics']['num_trading_days'],
                
                # Performance metrics
                'total_return': results['metrics']['total_return'],
                'annualized_return': results['metrics']['annualized_return'],
                'cagr': results['metrics']['cagr'],
                
                # Risk metrics
                'volatility': results['metrics']['volatility'],
                'sharpe_ratio': results['metrics']['sharpe_ratio'],
                'sortino_ratio': results['metrics']['sortino_ratio'],
                'max_drawdown': results['metrics']['max_drawdown'],
                'max_drawdown_duration_days': results['metrics']['max_drawdown_duration_days'],
                
                # Trade statistics
                'total_trades': results['metrics']['total_trades'],
                'winning_trades': results['metrics']['winning_trades'],
                'losing_trades': results['metrics']['losing_trades'],
                'win_rate': results['metrics']['win_rate'],
                'avg_win': results['metrics']['avg_win'],
                'avg_loss': results['metrics']['avg_loss'],
                'avg_trade_return': results['metrics']['avg_trade_return'],
                'profit_factor': results['metrics']['profit_factor'],
                'expectancy': results['metrics']['expectancy'],
                
                # Additional metrics
                'avg_trade_duration_days': results['metrics']['avg_trade_duration_days'],
                'max_consecutive_wins': results['metrics']['max_consecutive_wins'],
                'max_consecutive_losses': results['metrics']['max_consecutive_losses'],
                'recovery_factor': results['metrics']['recovery_factor'],
                
                # Capital metrics
                'initial_capital': results['metrics']['initial_capital'],
                'final_capital': results['metrics']['final_capital'],
                'peak_capital': results['metrics']['peak_capital'],
                
                # Detailed results (stored as JSONB)
                'trade_log': results['trade_log'][:100],  # Limit to 100 trades for storage
                'equity_curve': [(str(date), float(value)) for date, value in results['equity_curve']],
                
                # Parameters
                'commission_rate': self.commission_rate,
                'slippage_rate': self.slippage_rate,
                'position_sizing_method': self.position_sizing,
                
                # Notes
                'notes': notes
            }
            
            # Insert to Supabase
            result = supabase_client.table('backtest_results').insert(record).execute()
            
            logger.info(f"Backtest results saved: {backtest_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
            return False
    
    def print_report(self, results: Dict):
        """Print formatted backtest report."""
        metrics = results['metrics']
        
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        
        print(f"\n📅 Period: {metrics['start_date']} to {metrics['end_date']} ({metrics['num_trading_days']} days)")
        print(f"💰 Initial Capital: ₹{metrics['initial_capital']:,.2f}")
        print(f"💵 Final Capital: ₹{metrics['final_capital']:,.2f}")
        
        print("\n" + "-"*80)
        print("PERFORMANCE METRICS")
        print("-"*80)
        print(f"Total Return:        {metrics['total_return']:>10.2f}%")
        print(f"Annualized Return:   {metrics['annualized_return']:>10.2f}%")
        print(f"Volatility:          {metrics['volatility']:>10.2f}%")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:        {metrics['max_drawdown']:>10.2f}%")
        
        print("\n" + "-"*80)
        print("TRADE STATISTICS")
        print("-"*80)
        print(f"Total Trades:        {metrics['total_trades']:>10}")
        print(f"Winning Trades:      {metrics['winning_trades']:>10} ({metrics['win_rate']:.1f}%)")
        print(f"Losing Trades:       {metrics['losing_trades']:>10}")
        print(f"Average Win:         ₹{metrics['avg_win']:>9,.2f}")
        print(f"Average Loss:        ₹{metrics['avg_loss']:>9,.2f}")
        print(f"Profit Factor:       {metrics['profit_factor']:>10.2f}")
        print(f"Expectancy:          ₹{metrics['expectancy']:>9,.2f}")
        
        print("\n" + "-"*80)
        print("ADDITIONAL METRICS")
        print("-"*80)
        print(f"Max Consecutive Wins:   {metrics['max_consecutive_wins']:>7}")
        print(f"Max Consecutive Losses: {metrics['max_consecutive_losses']:>7}")
        print(f"Avg Holding Period:     {metrics['avg_trade_duration_days']:>7.1f} days")
        print(f"Recovery Factor:        {metrics['recovery_factor']:>7.2f}")
        
        if metrics.get('prediction_accuracy'):
            print(f"Prediction Accuracy:    {metrics['prediction_accuracy']:>7.1f}%")
        
        print("\n" + "="*80)
        
        # Trade log summary
        if results['trade_log']:
            print("\nTOP 5 TRADES (by P&L):")
            print("-"*80)
            trades_df = pd.DataFrame(results['trade_log'])
            top_trades = trades_df.nlargest(5, 'pnl')
            
            for _, trade in top_trades.iterrows():
                print(f"{trade['symbol']:<12} P&L: ₹{trade['pnl']:>8,.2f} ({trade['pnl_percent']:>+6.2f}%) | "
                      f"{trade['entry_date'].strftime('%Y-%m-%d')} → {trade['exit_date'].strftime('%Y-%m-%d')}")
            
            print("\n" + "="*80)


# ============================================================================
# Utility Functions
# ============================================================================

def load_backtest_results_from_supabase(supabase_client, limit: int = 10) -> pd.DataFrame:
    """Load recent backtest results from Supabase."""
    try:
        result = supabase_client.table('backtest_results')\
            .select('*')\
            .order('run_at', desc=True)\
            .limit(limit)\
            .execute()
        
        if result.data:
            return pd.DataFrame(result.data)
        else:
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error loading backtest results: {e}")
        return pd.DataFrame()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("AI Backtesting Framework")
    print("=" * 80)
    print("\nThis module provides comprehensive backtesting for AI predictions.")
    print("\nUsage:")
    print("  from ai_backtest import AIBacktester")
    print("  backtester = AIBacktester(initial_capital=100000)")
    print("  results = backtester.backtest_predictions(predictions_df, price_data)")
    print("  backtester.print_report(results)")
    print("\nMetrics calculated:")
    print("  - Returns: Total, Annualized, CAGR")
    print("  - Risk: Sharpe, Sortino, Max Drawdown, Volatility")
    print("  - Trades: Win Rate, Profit Factor, Expectancy")
    print("  - Streaks: Consecutive wins/losses")
    print("\n" + "=" * 80)
