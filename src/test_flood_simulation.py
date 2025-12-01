"""
Smoke tests for flood_simulation module.
"""
import sys
import os
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flood_simulation import FloodSimulator
from utils import load_config, setup_logging


@pytest.fixture
def simulator():
    """Create a FloodSimulator instance for testing."""
    config = load_config()
    logger = setup_logging(config)
    return FloodSimulator(config, logger)


def test_synthetic_dem_generation(simulator):
    """Test that synthetic DEM generation works."""
    dem = simulator.generate_synthetic_dem(size=(50, 50), elevation_range=(0, 100))
    assert dem.shape == (50, 50)
    assert dem.min() >= 0
    assert dem.max() <= 100
    assert np.isfinite(dem).all()


def test_spatial_rainfall_generation(simulator):
    """Test that spatial rainfall field generation works."""
    dem = simulator.generate_synthetic_dem(size=(50, 50), elevation_range=(0, 100))
    rainfall = simulator.generate_spatial_rainfall(dem, total_mm=100.0, pattern='both', n_hotspots=3)
    assert rainfall.shape == dem.shape
    assert rainfall.min() >= 0
    assert np.isfinite(rainfall).all()


def test_flow_direction_calculation(simulator):
    """Test D8 flow direction calculation."""
    dem = simulator.generate_synthetic_dem(size=(50, 50), elevation_range=(0, 100))
    flow_dir = simulator.calculate_flow_direction_d8(dem)
    assert flow_dir.shape == dem.shape
    assert flow_dir.dtype == np.int8
    # All valid directions should be -1 (sink) or 0-7 (valid direction)
    assert np.all((flow_dir >= -1) & (flow_dir <= 7))


def test_flow_accumulation(simulator):
    """Test flow accumulation calculation."""
    dem = simulator.generate_synthetic_dem(size=(50, 50), elevation_range=(0, 100))
    flow_dir = simulator.calculate_flow_direction_d8(dem)
    accumulation = simulator.calculate_flow_accumulation(flow_dir)
    assert accumulation.shape == flow_dir.shape
    assert np.all(accumulation >= 1)


def test_slope_calculation(simulator):
    """Test slope calculation from DEM."""
    dem = simulator.generate_synthetic_dem(size=(50, 50), elevation_range=(0, 100))
    slope = simulator.calculate_slope(dem)
    assert slope.shape == dem.shape
    assert np.all(slope >= 0.001)  # Ensure minimum slope enforced


def test_rainfall_runoff_simulation_uniform(simulator):
    """Test rainfall-runoff simulation with uniform rainfall."""
    dem = simulator.generate_synthetic_dem(size=(50, 50), elevation_range=(0, 100))
    water_depth = simulator.simulate_rainfall_runoff(dem, rainfall_mm=50.0, duration_hours=1.0)
    assert water_depth.shape == dem.shape
    assert np.isfinite(water_depth).all()
    assert water_depth.max() > 0


def test_rainfall_runoff_simulation_spatial(simulator):
    """Test rainfall-runoff simulation with spatial rainfall field."""
    dem = simulator.generate_synthetic_dem(size=(50, 50), elevation_range=(0, 100))
    rainfall = simulator.generate_spatial_rainfall(dem, total_mm=100.0, pattern='both')
    water_depth = simulator.simulate_rainfall_runoff(dem, rainfall_field=rainfall, duration_hours=1.0)
    assert water_depth.shape == dem.shape
    assert np.isfinite(water_depth).all()


def test_identify_flood_zones(simulator):
    """Test flood zone identification."""
    dem = simulator.generate_synthetic_dem(size=(50, 50), elevation_range=(0, 100))
    water_depth = simulator.simulate_rainfall_runoff(dem, rainfall_mm=50.0, duration_hours=1.0)
    flood_zones = simulator.identify_flood_zones(water_depth, threshold_m=0.1)
    assert flood_zones.shape == water_depth.shape
    assert np.all((flood_zones == 0) | (flood_zones == 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
