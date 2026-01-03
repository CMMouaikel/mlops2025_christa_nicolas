"""Data loading and validation module"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def load_train_data(file_path: str = "train.csv") -> pd.DataFrame:
    """Load training data from CSV file.
    
    Args:
        file_path: Path to the training CSV file
        
    Returns:
        DataFrame containing training data
    """
    logger.info(f"Loading training data from {file_path}")
    df = pd.read_csv(file_path, parse_dates=['pickup_datetime', 'dropoff_datetime'])
    logger.info(f"Loaded {len(df)} training records")
    return df


def load_test_data(file_path: str = "test.csv") -> pd.DataFrame:
    """Load test data from CSV file.
    
    Args:
        file_path: Path to the test CSV file
        
    Returns:
        DataFrame containing test data
    """
    logger.info(f"Loading test data from {file_path}")
    df = pd.read_csv(file_path, parse_dates=['pickup_datetime'])
    logger.info(f"Loaded {len(df)} test records")
    return df


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
    """
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info(f"Saved data to {file_path} ({len(df)} records)")


def validate_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Validate data schema and basic constraints.
    
    Args:
        df: DataFrame to validate
        is_train: Whether this is training data (has target column)
        
    Returns:
        Validated DataFrame
    """
    required_cols = ['id', 'vendor_id', 'pickup_datetime', 'passenger_count',
                     'pickup_longitude', 'pickup_latitude', 
                     'dropoff_longitude', 'dropoff_latitude']
    
    if is_train:
        required_cols.extend(['dropoff_datetime', 'trip_duration'])
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info("Data validation passed")
    return df
