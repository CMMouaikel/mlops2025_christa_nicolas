"""Batch inference module"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BatchInferenceEngine:
    """Run batch inference on test data."""
    
    def __init__(self, model, output_dir: str = "outputs"):
        """Initialize inference engine.
        
        Args:
            model: Trained model for prediction
            output_dir: Directory to save predictions
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions on input data.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Running predictions on {len(df)} samples")
        
        # Store ID for later
        ids = df['id'].copy()
        
        # Prepare features (same as training)
        X = self._prepare_features(df)
        
        # Predict (model outputs log-transformed values)
        y_pred_log = self.model.predict(X)
        
        # Convert back to original scale
        y_pred = np.expm1(y_pred_log)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'id': ids,
            'trip_duration': y_pred
        })
        
        logger.info(f"Predictions complete. Mean predicted duration: {y_pred.mean():.2f} seconds")
        
        return results
    
    def predict_and_save(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """Run predictions and save to file.
        
        Args:
            df: Input DataFrame with features
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved predictions file
        """
        predictions = self.predict(df)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"{timestamp}_predictions.csv"
        
        output_path = self.output_dir / filename
        predictions.to_csv(output_path, index=False)
        
        logger.info(f"Saved predictions to {output_path}")
        
        return str(output_path)
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Features DataFrame
        """
        # Exclude non-feature columns
        exclude_cols = ['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'store_and_fwd_flag']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        logger.debug(f"Prepared {len(feature_cols)} features for prediction")
        
        return X
