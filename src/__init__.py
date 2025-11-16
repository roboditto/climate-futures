"""
Caribbean Climate Impact Simulation & Early Warning System

A comprehensive Python-based climate risk model that predicts heatwaves,
flooding, and storm surge risk for Caribbean islands using environmental
data, machine learning, and simulation engines.

Author: Climate Futures Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Climate Futures Team"

# Import main components for easy access
from .utils import load_config, setup_logging
from .data_preprocessing import ClimateDataLoader
from .features import ClimateFeatureEngineer
from .risk_model import ClimateRiskIndex
from .visualization import ClimateVisualizer
from .alerts import ClimateAlertSystem
from .flood_simulation import FloodSimulator

__all__ = [
    'load_config',
    'setup_logging',
    'ClimateDataLoader',
    'ClimateFeatureEngineer',
    'ClimateRiskIndex',
    'ClimateVisualizer',
    'ClimateAlertSystem',
    'FloodSimulator'
]
