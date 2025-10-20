"""
AI Stock Prediction Model - LightGBM Baseline
===============================================================================
Purpose: Train and predict stock direction using LightGBM on technical features
Author: StockGenie Pro AI Enhancement
Created: 2025-10-20
===============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import pickle
import json

# ML imports
try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not installed. Run: pip install lightgbm scikit-learn")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDirectionModel:
    """
    LightGBM-based model for predicting stock price direction.
    Predicts: UP, DOWN, or FLAT for next N days.
    """
    
    def __init__(self, model_version: str = "v1.0.0"):
        self.model_version = model_version
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        self.training_metrics = {}
        self.hyperparameters = {
            'objective': 'multiclass',
            'num_class': 3,  # UP, DOWN, FLAT
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 7,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        
    def prepare_training_data(self, 
                             features_df: pd.DataFrame,
                             price_df: pd.DataFrame,
                             forecast_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from features and price data.
        
        Args:
            features_df: DataFrame with features from ai_features table
            price_df: DataFrame with OHLCV data
            forecast_horizon: Number of days to look ahead for target (default: 5)
            
        Returns:
            X (features), y (labels: 0=DOWN, 1=FLAT, 2=UP)
        """
        try:
            # Merge features with future returns
            data = features_df.copy()
            
            # Calculate future returns for each symbol
            symbols = data['symbol'].unique()
            targets = []
            
            for symbol in symbols:
                symbol_features = data[data['symbol'] == symbol].copy()
                symbol_prices = price_df[price_df['symbol'] == symbol].copy()
                
                if symbol_prices.empty:
                    continue
                
                # Sort by timestamp
                symbol_prices = symbol_prices.sort_values('ts')
                symbol_features = symbol_features.sort_values('ts')
                
                # Calculate future return
                symbol_prices['future_return'] = symbol_prices['close'].pct_change(forecast_horizon).shift(-forecast_horizon)
                
                # Merge on timestamp (approximate matching)
                merged = pd.merge_asof(
                    symbol_features.sort_values('ts'),
                    symbol_prices[['ts', 'future_return']].sort_values('ts'),
                    on='ts',
                    direction='nearest',
                    tolerance=pd.Timedelta('1 hour')
                )
                
                targets.append(merged)
            
            if not targets:
                raise ValueError("No valid training data after merging")
            
            # Combine all symbols
            combined = pd.concat(targets, ignore_index=True)
            
            # Drop rows with missing future returns
            combined = combined.dropna(subset=['future_return'])
            
            # Define labels based on future return
            # DOWN: < -2%, FLAT: -2% to +2%, UP: > +2%
            def classify_return(ret):
                if ret < -0.02:
                    return 0  # DOWN
                elif ret > 0.02:
                    return 2  # UP
                else:
                    return 1  # FLAT
            
            combined['label'] = combined['future_return'].apply(classify_return)
            
            # Select feature columns
            feature_cols = [col for col in combined.columns if col not in [
                'symbol', 'ts', 'future_return', 'label', 'feature_version', 
                'created_at', 'sector', 'market_regime', 'sentiment_score', 'news_count_7d'
            ]]
            
            self.feature_names = feature_cols
            
            X = combined[feature_cols].values
            y = combined['label'].values
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            logger.info(f"Class distribution - DOWN: {np.sum(y==0)}, FLAT: {np.sum(y==1)}, UP: {np.sum(y==2)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              n_splits: int = 5,
              early_stopping_rounds: int = 50) -> Dict:
        """
        Train LightGBM model using time-series cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            n_splits: Number of CV splits
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Dictionary with training metrics
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm scikit-learn")
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            val_scores = []
            train_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create LightGBM datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Train model
                model = lgb.train(
                    self.hyperparameters,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'val'],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0)  # Suppress output
                    ]
                )
                
                # Evaluate
                y_train_pred = model.predict(X_train).argmax(axis=1)
                y_val_pred = model.predict(X_val).argmax(axis=1)
                
                train_acc = accuracy_score(y_train, y_train_pred)
                val_acc = accuracy_score(y_val, y_val_pred)
                
                train_scores.append(train_acc)
                val_scores.append(val_acc)
                
                logger.info(f"Fold {fold+1}/{n_splits} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Train final model on all data
            train_data = lgb.Dataset(X_scaled, label=y)
            self.model = lgb.train(
                self.hyperparameters,
                train_data,
                num_boost_round=500
            )
            
            # Get feature importance
            self.feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importance(importance_type='gain').tolist()
            ))
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            # Store metrics
            self.training_metrics = {
                'train_accuracy': np.mean(train_scores),
                'val_accuracy': np.mean(val_scores),
                'train_std': np.std(train_scores),
                'val_std': np.std(val_scores),
                'num_samples': X.shape[0],
                'num_features': X.shape[1],
                'num_folds': n_splits,
                'model_version': self.model_version,
                'trained_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Training complete - Val Accuracy: {self.training_metrics['val_accuracy']:.4f} ± {self.training_metrics['val_std']:.4f}")
            
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, features: Dict) -> Dict:
        """
        Predict direction and confidence for a single stock.
        
        Args:
            features: Dictionary of features (from FeatureEngineer)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Prepare input
            X = self._prepare_single_input(features)
            X_scaled = self.scaler.transform(X)
            
            # Predict probabilities
            probs = self.model.predict(X_scaled)[0]
            
            # Get prediction and confidence
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])
            
            # Map class to direction
            direction_map = {0: 'DOWN', 1: 'FLAT', 2: 'UP'}
            predicted_direction = direction_map[predicted_class]
            
            # Get top contributing features
            top_features = self._get_top_contributing_features(features, n=5)
            
            result = {
                'ml_direction': predicted_direction,
                'ml_confidence': confidence,
                'ml_score': float(probs[2]),  # Probability of UP (0-1 scale)
                'probabilities': {
                    'DOWN': float(probs[0]),
                    'FLAT': float(probs[1]),
                    'UP': float(probs[2])
                },
                'top_features': top_features,
                'model_version': self.model_version
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'ml_direction': 'FLAT',
                'ml_confidence': 0.33,
                'ml_score': 0.5,
                'probabilities': {'DOWN': 0.33, 'FLAT': 0.34, 'UP': 0.33},
                'top_features': [],
                'model_version': self.model_version,
                'error': str(e)
            }
    
    def _prepare_single_input(self, features: Dict) -> np.ndarray:
        """Prepare single feature dictionary for prediction."""
        values = []
        for feature_name in self.feature_names:
            value = features.get(feature_name, 0)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = 0
            values.append(float(value))
        return np.array(values).reshape(1, -1)
    
    def _get_top_contributing_features(self, features: Dict, n: int = 5) -> List[Dict]:
        """Get top N features contributing to prediction."""
        if not self.feature_importance:
            return []
        
        top_features = []
        for feature_name, importance in list(self.feature_importance.items())[:n]:
            value = features.get(feature_name, 0)
            top_features.append({
                'feature': feature_name,
                'value': float(value) if value is not None else 0,
                'importance': float(importance)
            })
        
        return top_features
    
    def save_model(self, filepath: str):
        """Save model, scaler, and metadata to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'training_metrics': self.training_metrics,
                'hyperparameters': self.hyperparameters,
                'model_version': self.model_version
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str):
        """Load model, scaler, and metadata from file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.feature_importance = model_data.get('feature_importance', {})
            self.training_metrics = model_data.get('training_metrics', {})
            self.hyperparameters = model_data.get('hyperparameters', self.hyperparameters)
            self.model_version = model_data.get('model_version', 'unknown')
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_metadata(self) -> Dict:
        """Get model metadata for storage in Supabase."""
        return {
            'model_version': self.model_version,
            'model_type': 'lightgbm',
            'hyperparameters': self.hyperparameters,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'trained_at': self.training_metrics.get('trained_at'),
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'status': 'active',
            'is_production': True
        }


# ============================================================================
# Utility Functions
# ============================================================================

def save_model_metadata_to_supabase(model: StockDirectionModel, supabase_client):
    """Save model metadata to Supabase model_metadata table."""
    try:
        metadata = model.get_model_metadata()
        
        # Check if version already exists
        existing = supabase_client.table('model_metadata')\
            .select('id')\
            .eq('model_version', metadata['model_version'])\
            .execute()
        
        if existing.data:
            # Update existing
            result = supabase_client.table('model_metadata')\
                .update(metadata)\
                .eq('model_version', metadata['model_version'])\
                .execute()
            logger.info(f"Updated model metadata for {metadata['model_version']}")
        else:
            # Insert new
            result = supabase_client.table('model_metadata')\
                .insert(metadata)\
                .execute()
            logger.info(f"Inserted model metadata for {metadata['model_version']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving model metadata: {e}")
        return False


def load_latest_model_from_supabase(supabase_client) -> Optional[str]:
    """Get the latest active model version from Supabase."""
    try:
        result = supabase_client.table('model_metadata')\
            .select('model_version')\
            .eq('status', 'active')\
            .eq('is_production', True)\
            .order('trained_at', desc=True)\
            .limit(1)\
            .execute()
        
        if result.data:
            return result.data[0]['model_version']
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error loading model version: {e}")
        return None


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("AI Stock Direction Model - LightGBM Baseline")
    print("=" * 80)
    print("\nThis module provides stock direction prediction using LightGBM.")
    print("\nUsage:")
    print("  from ai_model import StockDirectionModel")
    print("  model = StockDirectionModel(model_version='v1.0.0')")
    print("  model.train(X, y)")
    print("  prediction = model.predict(features)")
    print("\nPrediction output:")
    print("  - ml_direction: 'UP', 'DOWN', or 'FLAT'")
    print("  - ml_confidence: Model confidence (0-1)")
    print("  - ml_score: Probability of UP direction (0-1)")
    print("  - probabilities: Full probability distribution")
    print("  - top_features: Top 5 contributing features")
    print("\n" + "=" * 80)
