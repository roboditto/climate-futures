"""
Climate Futures System - Utilities Module
Provides common utility functions for the entire system.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing logging settings
    
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get('file', 'logs/climate_system.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to configuration file. If None, uses default path.
    
    Returns:
    --------
    dict
        Configuration dictionary
    """
    if config_path is None:
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        config_path = str(project_root / 'config' / 'config.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories if they don't exist.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing storage paths
    """
    storage = config.get('storage', {})
    paths = [
        storage.get('raw_data_path', 'data/raw/'),
        storage.get('processed_data_path', 'data/processed/'),
        storage.get('models_path', 'data/models/'),
        storage.get('cache_path', 'data/cache/'),
        storage.get('results_path', 'results/'),
        storage.get('reports_path', 'reports/')
    ]
    
    for path in paths:
        os.makedirs(path, exist_ok=True)


def calculate_heat_index(temperature_c: np.ndarray, 
                         relative_humidity: np.ndarray) -> np.ndarray:
    """
    Calculate heat index using the Rothfusz regression.
    
    Parameters:
    -----------
    temperature_c : np.ndarray
        Temperature in Celsius
    relative_humidity : np.ndarray
        Relative humidity (0-100)
    
    Returns:
    --------
    np.ndarray
        Heat index in Celsius
    """
    # Convert to Fahrenheit for calculation
    T = temperature_c * 9.0/5.0 + 32
    RH = relative_humidity
    
    # Rothfusz regression coefficients
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-3
    c6 = -5.481717e-2
    c7 = 1.22874e-3
    c8 = 8.5282e-4
    c9 = -1.99e-6
    
    # Calculate heat index
    HI = (c1 + c2*T + c3*RH + c4*T*RH + c5*T**2 + 
          c6*RH**2 + c7*T**2*RH + c8*T*RH**2 + c9*T**2*RH**2)
    
    # Convert back to Celsius
    heat_index_c = (HI - 32) * 5/9
    
    # For temperatures below 27°C, heat index = temperature
    heat_index_c = np.where(temperature_c < 27, temperature_c, heat_index_c)
    
    return heat_index_c


def calculate_storm_surge_potential(wind_speed_ms: np.ndarray,
                                    pressure_hpa: np.ndarray,
                                    baseline_pressure: float = 1013.25) -> np.ndarray:
    """
    Calculate storm surge potential index.
    
    Parameters:
    -----------
    wind_speed_ms : np.ndarray
        Wind speed in m/s
    pressure_hpa : np.ndarray
        Atmospheric pressure in hPa
    baseline_pressure : float
        Normal atmospheric pressure (default: 1013.25 hPa)
    
    Returns:
    --------
    np.ndarray
        Storm surge potential (0-100 scale)
    """
    # Pressure deficit (lower pressure = higher surge)
    pressure_deficit = baseline_pressure - pressure_hpa
    
    # Wind stress (quadratic relationship)
    wind_stress = wind_speed_ms ** 2
    
    # Combine factors (normalize to 0-100)
    surge_potential = (0.6 * wind_stress + 0.4 * pressure_deficit * 10)
    surge_potential = np.clip(surge_potential / 5.0, 0, 100)  # Scale to 0-100
    
    return surge_potential


def calculate_rainfall_intensity(rainfall_mm: np.ndarray,
                                 duration_hours: float = 1.0) -> np.ndarray:
    """
    Calculate rainfall intensity category.
    
    Parameters:
    -----------
    rainfall_mm : np.ndarray
        Rainfall amount in mm
    duration_hours : float
        Duration in hours
    
    Returns:
    --------
    np.ndarray
        Intensity category (0=none, 1=light, 2=moderate, 3=heavy, 4=extreme)
    """
    intensity_mm_per_hour = rainfall_mm / duration_hours
    
    categories = np.zeros_like(intensity_mm_per_hour, dtype=int)
    categories = np.where(intensity_mm_per_hour > 0.1, 1, categories)    # Light
    categories = np.where(intensity_mm_per_hour > 2.5, 2, categories)    # Moderate
    categories = np.where(intensity_mm_per_hour > 10, 3, categories)     # Heavy
    categories = np.where(intensity_mm_per_hour > 50, 4, categories)     # Extreme
    
    return categories


def create_rolling_features(df: pd.DataFrame, 
                           column: str,
                           windows: list) -> pd.DataFrame:
    """
    Create rolling window features for a time series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    column : str
        Column name to create features from
    windows : list
        List of window sizes (in days)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added rolling features
    """
    result = df.copy()
    
    for window in windows:
        # Rolling mean
        result[f'{column}_rolling_mean_{window}d'] = (
            result[column].rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling std
        result[f'{column}_rolling_std_{window}d'] = (
            result[column].rolling(window=window, min_periods=1).std()
        )
        
        # Rolling max
        result[f'{column}_rolling_max_{window}d'] = (
            result[column].rolling(window=window, min_periods=1).max()
        )
        
        # Rolling min
        result[f'{column}_rolling_min_{window}d'] = (
            result[column].rolling(window=window, min_periods=1).min()
        )
    
    return result


def create_lag_features(df: pd.DataFrame,
                       column: str,
                       lags: list) -> pd.DataFrame:
    """
    Create lagged features for time series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    column : str
        Column name to create lag features from
    lags : list
        List of lag periods (in days)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added lag features
    """
    result = df.copy()
    
    for lag in lags:
        result[f'{column}_lag_{lag}d'] = result[column].shift(lag)
    
    return result


def get_risk_category(risk_score: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get risk category information based on score.
    
    Parameters:
    -----------
    risk_score : float
        Risk score (0-100)
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    dict
        Category information (name, color, range)
    """
    categories = config['risk_index']['categories']
    
    for category in categories:
        min_val, max_val = category['range']
        if min_val <= risk_score < max_val:
            return category
    
    # Return extreme if above all thresholds
    return categories[-1]


def save_model(model: Any, filepath: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : Any
        Trained model object
    filepath : str
        Path to save the model
    logger : logging.Logger, optional
        Logger instance
    """
    import joblib
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    
    if logger:
        logger.info(f"Model saved to {filepath}")


def load_model(filepath: str, logger: Optional[logging.Logger] = None) -> Any:
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
    logger : logging.Logger, optional
        Logger instance
    
    Returns:
    --------
    Any
        Loaded model object
    """
    import joblib
    
    model = joblib.load(filepath)
    
    if logger:
        logger.info(f"Model loaded from {filepath}")
    
    return model


def calculate_anomaly(values: np.ndarray, 
                     baseline_mean: Optional[float] = None,
                     baseline_std: Optional[float] = None) -> np.ndarray:
    """
    Calculate standardized anomaly.
    
    Parameters:
    -----------
    values : np.ndarray
        Values to calculate anomaly for
    baseline_mean : float, optional
        Baseline mean. If None, uses mean of values.
    baseline_std : float, optional
        Baseline standard deviation. If None, uses std of values.
    
    Returns:
    --------
    np.ndarray
        Standardized anomaly
    """
    if baseline_mean is None:
        baseline_mean = float(np.nanmean(values))
    
    if baseline_std is None:
        baseline_std = float(np.nanstd(values))
    
    if baseline_std == 0 or baseline_std is None:
        return np.zeros_like(values)
    
    anomaly = (values - baseline_mean) / baseline_std
    
    return anomaly
