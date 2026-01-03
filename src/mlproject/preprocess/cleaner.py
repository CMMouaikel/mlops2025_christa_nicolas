"""Data preprocessing and cleaning module"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess and clean NYC Taxi data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config
        self.drop_zero_duration = config.get('drop_zero_duration', True)
        self.max_duration_hours = config.get('max_duration_hours', 24)
        self.speed_threshold_mph = config.get('speed_threshold_mph', 100)
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """Fit preprocessor (no-op for stateless preprocessing).
        
        Args:
            df: Training data
            
        Returns:
            self
        """
        logger.info("Fitting preprocessor (stateless)")
        return self
    
    def transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Transform and clean data.
        
        Args:
            df: Input DataFrame
            is_train: Whether this is training data (has target)
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        initial_count = len(df)
        
        logger.info(f"Starting preprocessing with {initial_count} records")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove invalid coordinates
        df = self._remove_invalid_coordinates(df)
        
        # Remove invalid passenger counts
        df = self._remove_invalid_passenger_counts(df)
        
        if is_train:
            # Remove zero or negative durations
            if self.drop_zero_duration:
                df = df[df['trip_duration'] > 0]
                logger.info(f"Removed {initial_count - len(df)} zero/negative duration trips")
            
            # Remove outlier durations (> max_duration_hours)
            max_duration_seconds = self.max_duration_hours * 3600
            df = df[df['trip_duration'] <= max_duration_seconds]
            logger.info(f"Removed trips longer than {self.max_duration_hours} hours")
            
            # Remove trips with impossibly high average speeds
            df = self._remove_high_speed_outliers(df)
        
        final_count = len(df)
        logger.info(f"Preprocessing complete: {initial_count} -> {final_count} records "
                   f"({100 * (initial_count - final_count) / initial_count:.2f}% removed)")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Fit and transform data.
        
        Args:
            df: Input DataFrame
            is_train: Whether this is training data
            
        Returns:
            Cleaned DataFrame
        """
        return self.fit(df).transform(df, is_train=is_train)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
            # Drop rows with missing critical values
            df = df.dropna(subset=['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                                  'dropoff_longitude', 'dropoff_latitude'])
            
            # Fill missing passenger_count with median
            if df['passenger_count'].isnull().any():
                median_passengers = df['passenger_count'].median()
                df['passenger_count'].fillna(median_passengers, inplace=True)
        
        return df
    
    def _remove_invalid_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records with invalid coordinates (outside NYC bounds).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with valid coordinates
        """
        # Approximate NYC bounds
        nyc_bounds = {
            'min_lat': 40.5,
            'max_lat': 41.0,
            'min_lon': -74.3,
            'max_lon': -73.7
        }
        
        initial_count = len(df)
        
        df = df[
            (df['pickup_latitude'] >= nyc_bounds['min_lat']) &
            (df['pickup_latitude'] <= nyc_bounds['max_lat']) &
            (df['pickup_longitude'] >= nyc_bounds['min_lon']) &
            (df['pickup_longitude'] <= nyc_bounds['max_lon']) &
            (df['dropoff_latitude'] >= nyc_bounds['min_lat']) &
            (df['dropoff_latitude'] <= nyc_bounds['max_lat']) &
            (df['dropoff_longitude'] >= nyc_bounds['min_lon']) &
            (df['dropoff_longitude'] <= nyc_bounds['max_lon'])
        ]
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} records with coordinates outside NYC bounds")
        
        return df
    
    def _remove_invalid_passenger_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records with invalid passenger counts.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with valid passenger counts
        """
        initial_count = len(df)
        df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 9)]
        removed = initial_count - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} records with invalid passenger counts")
        
        return df
    
    def _remove_high_speed_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove trips with impossibly high average speeds.
        
        Args:
            df: Input DataFrame (must have trip_duration)
            
        Returns:
            DataFrame without high-speed outliers
        """
        # Calculate haversine distance for speed check
        df_temp = df.copy()
        df_temp['distance_km'] = self._haversine_distance(
            df_temp['pickup_latitude'].values,
            df_temp['pickup_longitude'].values,
            df_temp['dropoff_latitude'].values,
            df_temp['dropoff_longitude'].values
        )
        
        # Calculate average speed in mph
        df_temp['avg_speed_mph'] = (df_temp['distance_km'] * 0.621371) / (df_temp['trip_duration'] / 3600)
        
        initial_count = len(df)
        df = df[df_temp['avg_speed_mph'] <= self.speed_threshold_mph]
        removed = initial_count - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} records with average speed > {self.speed_threshold_mph} mph")
        
        return df
    
    @staticmethod
    def _haversine_distance(lat1: np.ndarray, lon1: np.ndarray, 
                           lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """Calculate haversine distance between coordinates.
        
        Args:
            lat1: Pickup latitude
            lon1: Pickup longitude
            lat2: Dropoff latitude
            lon2: Dropoff longitude
            
        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
