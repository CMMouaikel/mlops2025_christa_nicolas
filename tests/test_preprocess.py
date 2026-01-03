"""Test data preprocessing functions"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlproject.preprocess.cleaner import DataPreprocessor


@pytest.fixture
def sample_train_data():
    """Create sample training data for testing."""
    return pd.DataFrame({
        'id': ['id1', 'id2', 'id3', 'id4', 'id5'],
        'vendor_id': [1, 2, 1, 2, 1],
        'pickup_datetime': pd.to_datetime([
            '2016-01-01 00:00:00',
            '2016-01-01 01:00:00',
            '2016-01-01 02:00:00',
            '2016-01-01 03:00:00',
            '2016-01-01 04:00:00',
        ]),
        'dropoff_datetime': pd.to_datetime([
            '2016-01-01 00:30:00',
            '2016-01-01 01:30:00',
            '2016-01-01 02:30:00',
            '2016-01-01 03:30:00',
            '2016-01-01 04:30:00',
        ]),
        'passenger_count': [1, 2, 3, 1, 2],
        'pickup_longitude': [-73.99, -73.98, -73.97, -73.96, -73.95],
        'pickup_latitude': [40.75, 40.76, 40.77, 40.78, 40.79],
        'dropoff_longitude': [-73.98, -73.97, -73.96, -73.95, -73.94],
        'dropoff_latitude': [40.76, 40.77, 40.78, 40.79, 40.80],
        'trip_duration': [1800, 1800, 1800, 1800, 1800]  # 30 minutes
    })


@pytest.fixture
def preprocessor_config():
    """Create preprocessor configuration."""
    return {
        'drop_zero_duration': True,
        'max_duration_hours': 24,
        'speed_threshold_mph': 100
    }


def test_preprocessor_initialization(preprocessor_config):
    """Test preprocessor initialization."""
    preprocessor = DataPreprocessor(preprocessor_config)
    assert preprocessor.drop_zero_duration is True
    assert preprocessor.max_duration_hours == 24
    assert preprocessor.speed_threshold_mph == 100


def test_preprocessor_removes_zero_duration(sample_train_data, preprocessor_config):
    """Test that zero duration trips are removed."""
    # Add a zero duration trip
    sample_train_data.loc[5] = [
        'id6', 1, pd.Timestamp('2016-01-01 05:00:00'),
        pd.Timestamp('2016-01-01 05:00:00'), 1,
        -73.99, 40.75, -73.98, 40.76, 0
    ]
    
    preprocessor = DataPreprocessor(preprocessor_config)
    result = preprocessor.fit_transform(sample_train_data, is_train=True)
    
    assert len(result) == 5  # Zero duration trip should be removed
    assert (result['trip_duration'] > 0).all()


def test_preprocessor_handles_missing_values(sample_train_data, preprocessor_config):
    """Test handling of missing values."""
    # Add missing values
    sample_train_data.loc[5] = [
        'id6', 1, pd.Timestamp('2016-01-01 05:00:00'),
        pd.Timestamp('2016-01-01 05:30:00'), np.nan,
        -73.99, 40.75, -73.98, 40.76, 1800
    ]
    
    preprocessor = DataPreprocessor(preprocessor_config)
    result = preprocessor.fit_transform(sample_train_data, is_train=True)
    
    # Should handle missing passenger_count
    assert result['passenger_count'].notna().all()


def test_preprocessor_removes_invalid_coordinates(sample_train_data, preprocessor_config):
    """Test removal of invalid coordinates."""
    # Add trip with invalid coordinates
    sample_train_data.loc[5] = [
        'id6', 1, pd.Timestamp('2016-01-01 05:00:00'),
        pd.Timestamp('2016-01-01 05:30:00'), 1,
        0.0, 0.0, 0.0, 0.0, 1800  # Invalid coordinates
    ]
    
    preprocessor = DataPreprocessor(preprocessor_config)
    result = preprocessor.fit_transform(sample_train_data, is_train=True)
    
    # Invalid coordinate trip should be removed
    assert len(result) == 5
