"""Feature engineering module for NYC Taxi data"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for NYC Taxi trip duration prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature engineer with configuration.
        
        Args:
            config: Configuration dictionary with feature settings
        """
        self.config = config
        self.time_features = config.get('time_features', [])
        self.distance_features = config.get('distance_features', [])
        self.categorical_features = config.get('categorical_features', [])
        self.fitted_encodings_ = {}
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit feature engineer (learn encodings, etc).
        
        Args:
            df: Training data
            
        Returns:
            self
        """
        logger.info("Fitting feature engineer")
        
        # Learn mean encodings for categorical features if needed
        if 'vendor_id' in self.categorical_features and 'trip_duration' in df.columns:
            self.fitted_encodings_['vendor_id_mean'] = df.groupby('vendor_id')['trip_duration'].mean().to_dict()
        
        return self
    
    def transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Transform data by creating features.
        
        Args:
            df: Input DataFrame
            is_train: Whether this is training data
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        logger.info("Starting feature engineering")
        
        # Time-based features
        if self.time_features:
            df = self._create_time_features(df)
        
        # Distance features
        if self.distance_features:
            df = self._create_distance_features(df)
        
        # Categorical features
        df = self._encode_categorical_features(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Fit and transform data.
        
        Args:
            df: Input DataFrame
            is_train: Whether this is training data
            
        Returns:
            DataFrame with engineered features
        """
        return self.fit(df).transform(df, is_train=is_train)
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from pickup_datetime.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time features
        """
        logger.info("Creating time features")
        
        if 'hour' in self.time_features:
            df['hour'] = df['pickup_datetime'].dt.hour
        
        if 'day_of_week' in self.time_features:
            df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        
        if 'month' in self.time_features:
            df['month'] = df['pickup_datetime'].dt.month
        
        if 'is_weekend' in self.time_features:
            df['is_weekend'] = (df['pickup_datetime'].dt.dayofweek >= 5).astype(int)
        
        # Additional useful time features
        df['day'] = df['pickup_datetime'].dt.day
        df['year'] = df['pickup_datetime'].dt.year
        
        # Rush hour indicators
        df['is_rush_hour'] = ((df['pickup_datetime'].dt.hour >= 7) & 
                              (df['pickup_datetime'].dt.hour <= 9) |
                              (df['pickup_datetime'].dt.hour >= 17) & 
                              (df['pickup_datetime'].dt.hour <= 19)).astype(int)
        
        # Night time indicator
        df['is_night'] = ((df['pickup_datetime'].dt.hour >= 22) | 
                          (df['pickup_datetime'].dt.hour <= 5)).astype(int)
        
        return df
    
    def _create_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create distance-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with distance features
        """
        logger.info("Creating distance features")
        
        if 'haversine_distance' in self.distance_features:
            df['haversine_distance'] = haversine_km(
                df['pickup_latitude'].values,
                df['pickup_longitude'].values,
                df['dropoff_latitude'].values,
                df['dropoff_longitude'].values
            )
        
        if 'manhattan_distance' in self.distance_features:
            df['manhattan_distance'] = manhattan_distance(
                df['pickup_latitude'].values,
                df['pickup_longitude'].values,
                df['dropoff_latitude'].values,
                df['dropoff_longitude'].values
            )
        
        # Direction feature
        df['direction'] = bearing(
            df['pickup_latitude'].values,
            df['pickup_longitude'].values,
            df['dropoff_latitude'].values,
            df['dropoff_longitude'].values
        )
        
        # Distance from city center (Times Square: 40.758, -73.9855)
        df['pickup_distance_to_center'] = haversine_km(
            df['pickup_latitude'].values,
            df['pickup_longitude'].values,
            np.full(len(df), 40.758),
            np.full(len(df), -73.9855)
        )
        
        df['dropoff_distance_to_center'] = haversine_km(
            df['dropoff_latitude'].values,
            df['dropoff_longitude'].values,
            np.full(len(df), 40.758),
            np.full(len(df), -73.9855)
        )
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features")
        
        # Simple integer encoding for vendor_id (already 1 or 2)
        if 'vendor_id' in df.columns:
            df['vendor_id'] = df['vendor_id'].astype(int)
        
        # Clip passenger count
        if 'passenger_count' in df.columns:
            df['passenger_count'] = df['passenger_count'].clip(1, 6).astype(int)
        
        return df


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, 
                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculate haversine distance between coordinates in kilometers.
    
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


def manhattan_distance(lat1: np.ndarray, lon1: np.ndarray,
                      lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculate Manhattan (L1) distance in kilometers.
    
    Args:
        lat1: Pickup latitude
        lon1: Pickup longitude
        lat2: Dropoff latitude
        lon2: Dropoff longitude
        
    Returns:
        Manhattan distance in kilometers
    """
    # Approximate: 1 degree latitude ~ 111 km, 1 degree longitude ~ 85 km at NYC latitude
    lat_dist = np.abs(lat2 - lat1) * 111
    lon_dist = np.abs(lon2 - lon1) * 85
    
    return lat_dist + lon_dist


def bearing(lat1: np.ndarray, lon1: np.ndarray,
           lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculate bearing (direction) between two points in degrees.
    
    Args:
        lat1: Pickup latitude
        lon1: Pickup longitude
        lat2: Dropoff latitude
        lon2: Dropoff longitude
        
    Returns:
        Bearing in degrees (0-360)
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    bearing_rad = np.arctan2(y, x)
    bearing_deg = np.degrees(bearing_rad)
    
    return (bearing_deg + 360) % 360
