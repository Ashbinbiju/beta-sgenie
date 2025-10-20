"""
AI Feature Engineering Module
===============================================================================
Purpose: Extract ML-ready features from stock data for AI models
Author: StockGenie Pro AI Enhancement
Created: 2025-10-20
===============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import ta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Extracts and computes features for ML models from OHLCV data.
    Features include technical indicators, price patterns, volume analysis,
    and market context.
    """
    
    def __init__(self):
        self.feature_version = "v1.0"
        self.required_history = 200  # Minimum bars needed for feature calculation
        
    def extract_features(self, df: pd.DataFrame, symbol: str, sector: str = None) -> Dict:
        """
        Extract all features from OHLCV dataframe.
        
        Args:
            df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
            symbol: Stock symbol
            sector: Stock sector (optional)
            
        Returns:
            Dictionary of features for the latest timestamp
        """
        if df is None or df.empty or len(df) < 50:
            logger.warning(f"{symbol}: Insufficient data for feature extraction")
            return None
            
        try:
            # Make a copy to avoid modifying original
            data = df.copy()
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                logger.error(f"{symbol}: Missing required OHLCV columns")
                return None
            
            # 1. Price-based features
            features = self._calculate_price_features(data)
            
            # 2. Technical indicators
            features.update(self._calculate_technical_indicators(data))
            
            # 3. Volume features
            features.update(self._calculate_volume_features(data))
            
            # 4. Momentum features
            features.update(self._calculate_momentum_features(data))
            
            # 5. Trend features
            features.update(self._calculate_trend_features(data))
            
            # 6. Support/Resistance levels
            features.update(self._calculate_support_resistance(data))
            
            # 7. Market context
            features.update(self._calculate_market_context(data, symbol, sector))
            
            # Add metadata
            features['symbol'] = symbol
            features['ts'] = datetime.utcnow()
            features['feature_version'] = self.feature_version
            
            return features
            
        except Exception as e:
            logger.error(f"{symbol}: Error extracting features: {e}")
            return None
    
    def _calculate_price_features(self, df: pd.DataFrame) -> Dict:
        """Calculate price-based features."""
        close = df['Close']
        
        features = {
            'returns_1d': close.pct_change(1).iloc[-1] if len(df) > 1 else 0,
            'returns_5d': close.pct_change(5).iloc[-1] if len(df) > 5 else 0,
            'returns_20d': close.pct_change(20).iloc[-1] if len(df) > 20 else 0,
            'volatility_10d': close.pct_change().rolling(10).std().iloc[-1] if len(df) > 10 else 0,
            'volatility_30d': close.pct_change().rolling(30).std().iloc[-1] if len(df) > 30 else 0,
        }
        
        return features
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        features = {}
        
        try:
            # RSI
            rsi_indicator = ta.momentum.RSIIndicator(close=close, window=14)
            features['rsi_14'] = rsi_indicator.rsi().iloc[-1] if len(close) > 14 else 50
            
            # MACD
            macd_indicator = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            features['macd'] = macd_indicator.macd().iloc[-1] if len(close) > 26 else 0
            features['macd_signal'] = macd_indicator.macd_signal().iloc[-1] if len(close) > 26 else 0
            features['macd_hist'] = macd_indicator.macd_diff().iloc[-1] if len(close) > 26 else 0
            
            # ADX
            adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
            features['adx_14'] = adx_indicator.adx().iloc[-1] if len(close) > 14 else 0
            
            # ATR
            atr_indicator = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
            features['atr_14'] = atr_indicator.average_true_range().iloc[-1] if len(close) > 14 else 0
            
            # EMAs
            ema_9 = ta.trend.EMAIndicator(close=close, window=9)
            ema_21 = ta.trend.EMAIndicator(close=close, window=21)
            ema_50 = ta.trend.EMAIndicator(close=close, window=50)
            features['ema_9'] = ema_9.ema_indicator().iloc[-1] if len(df) > 9 else close.iloc[-1]
            features['ema_21'] = ema_21.ema_indicator().iloc[-1] if len(df) > 21 else close.iloc[-1]
            features['ema_50'] = ema_50.ema_indicator().iloc[-1] if len(df) > 50 else close.iloc[-1]
            
            # SMAs
            sma_20 = ta.trend.SMAIndicator(close=close, window=20)
            sma_50 = ta.trend.SMAIndicator(close=close, window=50)
            sma_200 = ta.trend.SMAIndicator(close=close, window=200)
            features['sma_20'] = sma_20.sma_indicator().iloc[-1] if len(df) > 20 else close.iloc[-1]
            features['sma_50'] = sma_50.sma_indicator().iloc[-1] if len(df) > 50 else close.iloc[-1]
            features['sma_200'] = sma_200.sma_indicator().iloc[-1] if len(df) > 200 else close.iloc[-1]
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            # Fill with defaults
            for key in ['rsi_14', 'macd', 'macd_signal', 'macd_hist', 'adx_14', 'atr_14',
                       'ema_9', 'ema_21', 'ema_50', 'sma_20', 'sma_50', 'sma_200']:
                if key not in features:
                    features[key] = 0
        
        return features
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> Dict:
        """Calculate volume-based features."""
        volume = df['Volume']
        close = df['Close']
        
        features = {}
        
        try:
            features['volume'] = volume.iloc[-1]
            
            # Volume ratio (current vs 20-day average)
            avg_volume_20 = volume.rolling(20).mean().iloc[-1] if len(df) > 20 else volume.mean()
            features['volume_ratio_20d'] = volume.iloc[-1] / avg_volume_20 if avg_volume_20 > 0 else 1.0
            
            # OBV
            obv_indicator = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
            features['obv'] = obv_indicator.on_balance_volume().iloc[-1] if len(volume) > 0 else 0
            
            # VWAP - manual calculation as ta library doesn't have built-in VWAP
            if len(df) >= 14:
                typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                vwap_value = (typical_price * volume).rolling(14).sum() / volume.rolling(14).sum()
                features['vwap'] = vwap_value.iloc[-1] if not vwap_value.empty else close.iloc[-1]
            else:
                features['vwap'] = close.iloc[-1]
            
        except Exception as e:
            logger.warning(f"Error calculating volume features: {e}")
            features.update({'volume': 0, 'volume_ratio_20d': 1.0, 'obv': 0, 'vwap': 0})
        
        return features
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> Dict:
        """Calculate momentum indicators."""
        close = df['Close']
        
        features = {}
        
        try:
            # Momentum (price difference)
            features['momentum_5d'] = (close.iloc[-1] - close.iloc[-6]) if len(df) > 5 else 0
            features['momentum_20d'] = (close.iloc[-1] - close.iloc[-21]) if len(df) > 20 else 0
            
            # Rate of Change
            roc_indicator = ta.momentum.ROCIndicator(close=close, window=10)
            features['roc_10d'] = roc_indicator.roc().iloc[-1] if len(close) > 10 else 0
            
        except Exception as e:
            logger.warning(f"Error calculating momentum features: {e}")
            features.update({'momentum_5d': 0, 'momentum_20d': 0, 'roc_10d': 0})
        
        return features
    
    def _calculate_trend_features(self, df: pd.DataFrame) -> Dict:
        """Calculate trend strength and direction."""
        close = df['Close']
        
        features = {}
        
        try:
            # Trend strength (based on ADX and MA alignment)
            if len(df) > 50:
                ema_9_ind = ta.trend.EMAIndicator(close=close, window=9)
                ema_21_ind = ta.trend.EMAIndicator(close=close, window=21)
                ema_50_ind = ta.trend.EMAIndicator(close=close, window=50)
                ema_9 = ema_9_ind.ema_indicator().iloc[-1]
                ema_21 = ema_21_ind.ema_indicator().iloc[-1]
                ema_50 = ema_50_ind.ema_indicator().iloc[-1]
                current_price = close.iloc[-1]
                
                # Calculate alignment score
                if current_price > ema_9 > ema_21 > ema_50:
                    trend_strength = 1.0  # Strong uptrend
                elif current_price < ema_9 < ema_21 < ema_50:
                    trend_strength = -1.0  # Strong downtrend
                else:
                    # Partial alignment
                    alignment = sum([
                        current_price > ema_9,
                        ema_9 > ema_21,
                        ema_21 > ema_50
                    ])
                    trend_strength = (alignment - 1.5) / 1.5  # Normalize to [-1, 1]
            else:
                trend_strength = 0
            
            features['trend_strength'] = trend_strength
            
        except Exception as e:
            logger.warning(f"Error calculating trend features: {e}")
            features['trend_strength'] = 0
        
        return features
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        features = {}
        
        try:
            current_price = close.iloc[-1]
            
            # Recent highs and lows (20-day window)
            if len(df) > 20:
                recent_high = high.iloc[-20:].max()
                recent_low = low.iloc[-20:].min()
                
                features['resistance_level'] = recent_high
                features['support_level'] = recent_low
            else:
                features['resistance_level'] = high.max()
                features['support_level'] = low.min()
            
            # 52-week high/low distance
            if len(df) > 252:
                high_52w = high.iloc[-252:].max()
                low_52w = low.iloc[-252:].min()
            else:
                high_52w = high.max()
                low_52w = low.min()
            
            features['distance_from_52w_high'] = ((current_price - high_52w) / high_52w * 100) if high_52w > 0 else 0
            features['distance_from_52w_low'] = ((current_price - low_52w) / low_52w * 100) if low_52w > 0 else 0
            
        except Exception as e:
            logger.warning(f"Error calculating support/resistance: {e}")
            features.update({
                'resistance_level': 0,
                'support_level': 0,
                'distance_from_52w_high': 0,
                'distance_from_52w_low': 0
            })
        
        return features
    
    def _calculate_market_context(self, df: pd.DataFrame, symbol: str, sector: str = None) -> Dict:
        """Calculate market context features."""
        features = {}
        
        try:
            # Sector (will be populated from external source)
            features['sector'] = sector if sector else 'Unknown'
            
            # Market regime (simplified - based on trend)
            close = df['Close']
            if len(df) > 50:
                sma_50_ind = ta.trend.SMAIndicator(close=close, window=50)
                sma_50 = sma_50_ind.sma_indicator().iloc[-1]
                current_price = close.iloc[-1]
                
                if current_price > sma_50 * 1.02:
                    regime = 'bullish'
                elif current_price < sma_50 * 0.98:
                    regime = 'bearish'
                else:
                    regime = 'sideways'
            else:
                regime = 'sideways'
            
            features['market_regime'] = regime
            
            # Sector strength (placeholder - will be enhanced)
            features['sector_strength'] = 0.5
            
            # Correlation to NIFTY (placeholder - will be enhanced)
            features['correlation_to_nifty'] = 0.5
            
            # Sentiment (placeholder - will be integrated later)
            features['sentiment_score'] = None
            features['news_count_7d'] = None
            
        except Exception as e:
            logger.warning(f"Error calculating market context: {e}")
            features.update({
                'sector': 'Unknown',
                'sector_strength': 0.5,
                'market_regime': 'sideways',
                'correlation_to_nifty': 0.5,
                'sentiment_score': None,
                'news_count_7d': None
            })
        
        return features
    
    def prepare_model_input(self, features: Dict) -> np.ndarray:
        """
        Convert features dictionary to numpy array for model input.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Numpy array of feature values in correct order
        """
        # Define feature order (must match training)
        feature_order = [
            'returns_1d', 'returns_5d', 'returns_20d',
            'volatility_10d', 'volatility_30d',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'adx_14', 'atr_14',
            'ema_9', 'ema_21', 'ema_50',
            'sma_20', 'sma_50', 'sma_200',
            'volume_ratio_20d', 'obv', 'vwap',
            'momentum_5d', 'momentum_20d', 'roc_10d',
            'trend_strength',
            'distance_from_52w_high', 'distance_from_52w_low',
            'sector_strength', 'correlation_to_nifty'
        ]
        
        # Extract values in order, using 0 as default for missing values
        values = []
        for feature_name in feature_order:
            value = features.get(feature_name, 0)
            # Handle None values
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = 0
            values.append(float(value))
        
        return np.array(values).reshape(1, -1)
    
    def batch_extract_features(self, 
                               data_dict: Dict[str, pd.DataFrame],
                               sectors: Dict[str, str] = None) -> pd.DataFrame:
        """
        Extract features for multiple stocks.
        
        Args:
            data_dict: Dictionary of {symbol: dataframe}
            sectors: Dictionary of {symbol: sector}
            
        Returns:
            DataFrame with all features
        """
        all_features = []
        
        for symbol, df in data_dict.items():
            sector = sectors.get(symbol) if sectors else None
            features = self.extract_features(df, symbol, sector)
            
            if features:
                all_features.append(features)
        
        if not all_features:
            return pd.DataFrame()
        
        return pd.DataFrame(all_features)


# ============================================================================
# Utility Functions
# ============================================================================

def save_features_to_supabase(features_df: pd.DataFrame, supabase_client):
    """
    Save features to Supabase ai_features table.
    
    Args:
        features_df: DataFrame with extracted features
        supabase_client: Supabase client instance
    """
    try:
        if features_df.empty:
            logger.warning("No features to save")
            return 0
        
        # Convert DataFrame to list of dicts
        records = features_df.to_dict('records')
        
        # Upsert to Supabase
        result = supabase_client.table('ai_features').upsert(records).execute()
        
        logger.info(f"Saved {len(records)} feature records to Supabase")
        return len(records)
        
    except Exception as e:
        logger.error(f"Error saving features to Supabase: {e}")
        return 0


def load_latest_features_from_supabase(symbols: List[str], supabase_client) -> pd.DataFrame:
    """
    Load latest features from Supabase for given symbols.
    
    Args:
        symbols: List of stock symbols
        supabase_client: Supabase client instance
        
    Returns:
        DataFrame with latest features
    """
    try:
        # Query latest features for each symbol
        result = supabase_client.table('ai_features')\
            .select('*')\
            .in_('symbol', symbols)\
            .order('ts', desc=True)\
            .limit(len(symbols))\
            .execute()
        
        if result.data:
            return pd.DataFrame(result.data)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading features from Supabase: {e}")
        return pd.DataFrame()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("AI Feature Engineering Module")
    print("=" * 80)
    print("\nThis module extracts ML-ready features from stock OHLCV data.")
    print("\nUsage:")
    print("  from ai_features import FeatureEngineer")
    print("  engineer = FeatureEngineer()")
    print("  features = engineer.extract_features(df, symbol='SBIN', sector='Banking')")
    print("  model_input = engineer.prepare_model_input(features)")
    print("\nFeatures extracted:")
    print("  - Price-based: returns, volatility")
    print("  - Technical: RSI, MACD, ADX, ATR, EMAs, SMAs")
    print("  - Volume: volume ratios, OBV, VWAP")
    print("  - Momentum: momentum, ROC")
    print("  - Trend: trend strength, support/resistance")
    print("  - Market context: regime, sector strength")
    print("\n" + "=" * 80)
