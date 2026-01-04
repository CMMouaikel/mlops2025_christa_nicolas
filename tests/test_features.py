"""Test feature engineering functions"""

import pytest
import numpy as np
from mlproject.features.engineer import haversine_km, manhattan_distance, bearing


def test_haversine_zero_distance():
    """Test haversine distance with same coordinates."""
    dist = haversine_km(
        np.array([40.758]),
        np.array([-73.9855]),
        np.array([40.758]),
        np.array([-73.9855])
    )
    assert dist[0] == pytest.approx(0.0, abs=0.01)


def test_haversine_known_distance():
    """Test haversine distance with known coordinates."""
    # Approximate distance between two points in NYC
    # Times Square (40.758, -73.9855) to Central Park (40.785, -73.968)
    dist = haversine_km(
        np.array([40.758]),
        np.array([-73.9855]),
        np.array([40.785]),
        np.array([-73.968])
    )
    # Should be approximately 3-4 km
    assert 2.0 < dist[0] < 5.0


def test_manhattan_distance_zero():
    """Test Manhattan distance with same coordinates."""
    dist = manhattan_distance(
        np.array([40.758]),
        np.array([-73.9855]),
        np.array([40.758]),
        np.array([-73.9855])
    )
    assert dist[0] == pytest.approx(0.0, abs=0.01)


def test_manhattan_distance_positive():
    """Test Manhattan distance returns positive values."""
    dist = manhattan_distance(
        np.array([40.758]),
        np.array([-73.9855]),
        np.array([40.785]),
        np.array([-73.968])
    )
    assert dist[0] > 0


def test_bearing_north():
    """Test bearing calculation for north direction."""
    # Moving north should give bearing close to 0 or 360
    b = bearing(
        np.array([40.758]),
        np.array([-73.9855]),
        np.array([40.858]),
        np.array([-73.9855])
    )
    # Should be close to 0 (north) with small tolerance for longitude effect
    assert b[0] < 5 or b[0] > 355


def test_bearing_east():
    """Test bearing calculation for east direction."""
    # Moving east should give bearing close to 90
    b = bearing(
        np.array([40.758]),
        np.array([-73.9855]),
        np.array([40.758]),
        np.array([-73.8855])
    )
    # Should be close to 90 (east)
    assert 85 < b[0] < 95


def test_bearing_range():
    """Test bearing is always in [0, 360) range."""
    b = bearing(
        np.array([40.758, 40.785, 40.720]),
        np.array([-73.9855, -73.968, -74.000]),
        np.array([40.785, 40.720, 40.758]),
        np.array([-73.968, -74.000, -73.9855])
    )
    assert np.all(b >= 0)
    assert np.all(b < 360)
