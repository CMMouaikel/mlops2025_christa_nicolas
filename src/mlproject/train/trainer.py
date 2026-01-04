"""Model training module"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate ML models for trip duration prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.metric = config.get('metric', 'rmse')
        self.model_configs = config.get('models', [])
        self.output_dir = Path(config.get('output_dir', 'outputs/models'))
        self.metrics_dir = Path(config.get('metrics_dir', 'outputs/metrics'))
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.metrics = {}
        self.best_model_name = None
        self.best_model = None
    
    def train(self, df: pd.DataFrame, target_col: str = 'trip_duration') -> Dict[str, Any]:
        """Train multiple models and select the best one.
        
        Args:
            df: Training data with features and target
            target_col: Name of the target column
            
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info("Starting model training")
        
        # Prepare features and target
        X, y = self._prepare_data(df, target_col)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        logger.info(f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
        
        # Train each model
        for model_config in self.model_configs:
            model_name = model_config['name']
            logger.info(f"Training {model_name}...")
            
            model = self._create_model(model_config)
            model.fit(X_train, y_train)
            
            # Evaluate
            train_metrics = self._evaluate(model, X_train, y_train, "train")
            val_metrics = self._evaluate(model, X_val, y_val, "validation")
            
            self.models[model_name] = model
            self.metrics[model_name] = {
                'train': train_metrics,
                'validation': val_metrics,
                'config': model_config['params']
            }
            
            logger.info(f"{model_name} - Validation RMSE: {val_metrics['rmse']:.2f}, "
                       f"MAE: {val_metrics['mae']:.2f}, R2: {val_metrics['r2']:.4f}")
        
        # Select best model
        self._select_best_model()
        
        # Save models and metrics
        self._save_models()
        self._save_metrics()
        
        return {
            'best_model': self.best_model_name,
            'best_metrics': self.metrics[self.best_model_name],
            'all_metrics': self.metrics
        }
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (features, target)
        """
        # Exclude non-feature columns
        exclude_cols = [target_col, 'id', 'pickup_datetime', 'dropoff_datetime', 'store_and_fwd_flag']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Log transform target for better distribution
        y = np.log1p(y)
        
        logger.info(f"Features: {len(feature_cols)} columns")
        logger.debug(f"Feature columns: {feature_cols}")
        
        return X, y
    
    def _create_model(self, model_config: Dict[str, Any]):
        """Create a model instance from configuration.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Initialized model instance
        """
        model_name = model_config['name']
        params = model_config['params']
        
        if model_name == 'random_forest':
            return RandomForestRegressor(**params)
        elif model_name == 'xgboost':
            return XGBRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def _evaluate(self, model, X, y, dataset_name: str) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            model: Trained model
            X: Features
            y: True target values (log-transformed)
            dataset_name: Name of the dataset (for logging)
            
        Returns:
            Dictionary of metrics
        """
        # Predict (log-transformed)
        y_pred_log = model.predict(X)
        
        # Convert back to original scale
        y_true = np.expm1(y)
        y_pred = np.expm1(y_pred_log)
        
        # Calculate metrics on original scale
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Also calculate RMSLE (on log scale)
        rmsle = np.sqrt(mean_squared_error(y, y_pred_log))
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'rmsle': float(rmsle)
        }
    
    def _select_best_model(self) -> None:
        """Select the best model based on validation metric."""
        best_score = float('inf')
        metric_key = 'rmse' if self.metric == 'rmse' else self.metric
        
        for model_name, metrics in self.metrics.items():
            val_score = metrics['validation'][metric_key]
            if val_score < best_score:
                best_score = val_score
                self.best_model_name = model_name
        
        self.best_model = self.models[self.best_model_name]
        logger.info(f"Best model: {self.best_model_name} "
                   f"(validation {metric_key.upper()}: {best_score:.2f})")
    
    def _save_models(self) -> None:
        """Save all trained models to disk."""
        for model_name, model in self.models.items():
            model_path = self.output_dir / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save best model separately
        best_model_path = self.output_dir / "best_model.pkl"
        joblib.dump(self.best_model, best_model_path)
        logger.info(f"Saved best model ({self.best_model_name}) to {best_model_path}")
    
    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        from omegaconf import OmegaConf
        
        # Convert DictConfig to regular dict
        metrics_copy = {}
        for model_name, metrics in self.metrics.items():
            metrics_copy[model_name] = {
                'train': metrics['train'],
                'validation': metrics['validation'],
                'config': OmegaConf.to_container(metrics['config']) if hasattr(metrics['config'], '__dict__') else dict(metrics['config'])
            }
        
        metrics_data = {
            'best_model': self.best_model_name,
            'timestamp': datetime.now().isoformat(),
            'models': metrics_copy
        }
        
        metrics_path = self.metrics_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")


def load_model(model_path: str):
    """Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)
