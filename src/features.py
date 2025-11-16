"""
Feature Engineering Module - Days 3-4
Creates scientifically meaningful features for climate prediction models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging


class ClimateFeatureEngineer:
    """
    Handles feature engineering for climate data.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize feature engineer.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        logger : logging.Logger, optional
            Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.feature_config = config.get('features', {})
        self.time_windows = self.feature_config.get('time_windows', [3, 7, 14, 30])
        self.lag_days = self.feature_config.get('lag_days', [1, 2, 3, 7, 14])
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from datetime index.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data with datetime index
        
        Returns:
        --------
        pd.DataFrame
            Data with temporal features
        """
        self.logger.info("Creating temporal features...")
        df_temporal = df.copy()
        
        # Basic temporal features
        df_temporal['year'] = df_temporal.index.year
        df_temporal['month'] = df_temporal.index.month
        df_temporal['day'] = df_temporal.index.day
        df_temporal['day_of_year'] = df_temporal.index.dayofyear
        df_temporal['week_of_year'] = df_temporal.index.isocalendar().week
        df_temporal['day_of_week'] = df_temporal.index.dayofweek
        
        # Cyclical encoding for seasonality (sine/cosine transform)
        df_temporal['month_sin'] = np.sin(2 * np.pi * df_temporal['month'] / 12)
        df_temporal['month_cos'] = np.cos(2 * np.pi * df_temporal['month'] / 12)
        df_temporal['day_of_year_sin'] = np.sin(2 * np.pi * df_temporal['day_of_year'] / 365)
        df_temporal['day_of_year_cos'] = np.cos(2 * np.pi * df_temporal['day_of_year'] / 365)
        
        # Season indicator (Caribbean wet/dry seasons)
        # Wet season: May-November (months 5-11)
        # Dry season: December-April (months 12, 1-4)
        df_temporal['is_wet_season'] = df_temporal['month'].isin([5, 6, 7, 8, 9, 10, 11]).astype(int)
        
        # Hurricane season: June-November
        df_temporal['is_hurricane_season'] = df_temporal['month'].isin([6, 7, 8, 9, 10, 11]).astype(int)
        
        self.logger.info(f"Created {len([c for c in df_temporal.columns if c not in df.columns])} temporal features")
        return df_temporal
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               columns: List[str]) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        columns : list
            Columns to create rolling features for
        
        Returns:
        --------
        pd.DataFrame
            Data with rolling features
        """
        self.logger.info("Creating rolling window features...")
        df_rolling = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in self.time_windows:
                # Rolling mean
                df_rolling[f'{col}_rolling_mean_{window}d'] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std
                df_rolling[f'{col}_rolling_std_{window}d'] = (
                    df[col].rolling(window=window, min_periods=1).std()
                )
                
                # Rolling max
                df_rolling[f'{col}_rolling_max_{window}d'] = (
                    df[col].rolling(window=window, min_periods=1).max()
                )
                
                # Rolling min
                df_rolling[f'{col}_rolling_min_{window}d'] = (
                    df[col].rolling(window=window, min_periods=1).min()
                )
                
                # Rolling sum (useful for rainfall)
                if 'precip' in col.lower() or 'rain' in col.lower():
                    df_rolling[f'{col}_rolling_sum_{window}d'] = (
                        df[col].rolling(window=window, min_periods=1).sum()
                    )
        
        self.logger.info(f"Created rolling features for {len(columns)} columns")
        return df_rolling
    
    def create_lag_features(self, df: pd.DataFrame,
                           columns: List[str]) -> pd.DataFrame:
        """
        Create lagged features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        columns : list
            Columns to create lag features for
        
        Returns:
        --------
        pd.DataFrame
            Data with lag features
        """
        self.logger.info("Creating lag features...")
        df_lag = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in self.lag_days:
                df_lag[f'{col}_lag_{lag}d'] = df[col].shift(lag)
        
        self.logger.info(f"Created lag features for {len(columns)} columns")
        return df_lag
    
    def create_rate_of_change_features(self, df: pd.DataFrame,
                                       columns: List[str]) -> pd.DataFrame:
        """
        Create rate of change (derivative) features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        columns : list
            Columns to create rate of change for
        
        Returns:
        --------
        pd.DataFrame
            Data with rate of change features
        """
        self.logger.info("Creating rate of change features...")
        df_roc = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # 1-day change
            df_roc[f'{col}_change_1d'] = df[col].diff(1)
            
            # 3-day change
            df_roc[f'{col}_change_3d'] = df[col].diff(3)
            
            # 7-day change
            df_roc[f'{col}_change_7d'] = df[col].diff(7)
            
            # Percentage change (avoid division by zero)
            df_roc[f'{col}_pct_change_1d'] = df[col].pct_change(1)
        
        self.logger.info(f"Created rate of change features for {len(columns)} columns")
        return df_roc
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        
        Returns:
        --------
        pd.DataFrame
            Data with interaction features
        """
        self.logger.info("Creating interaction features...")
        df_interact = df.copy()
        
        # Temperature-Humidity interaction (affects heat stress)
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df_interact['temp_humidity_product'] = df['temperature'] * df['humidity']
            df_interact['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 0.001)
        
        # Wind-Pressure interaction (storm indicator)
        if 'wind_speed' in df.columns and 'pressure' in df.columns:
            df_interact['wind_pressure_product'] = df['wind_speed'] * (1013 - df['pressure'])
        
        # Temperature range (daily variability)
        if 'temperature_max' in df.columns and 'temperature_min' in df.columns:
            df_interact['temperature_range'] = df['temperature_max'] - df['temperature_min']
        
        # Precipitation intensity (if we have cumulative measures)
        if 'precipitation' in df.columns:
            # Ratio of current to recent average
            precip_7d_avg = df['precipitation'].rolling(window=7, min_periods=1).mean()
            df_interact['precipitation_intensity_ratio'] = df['precipitation'] / (precip_7d_avg + 0.001)
        
        self.logger.info("Created interaction features")
        return df_interact
    
    def create_climate_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced climate indices.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        
        Returns:
        --------
        pd.DataFrame
            Data with climate indices
        """
        from utils import calculate_heat_index, calculate_storm_surge_potential
        
        self.logger.info("Creating climate indices...")
        df_indices = df.copy()
        
        # Heat Index (if not already calculated)
        if 'heat_index' not in df.columns:
            if 'temperature' in df.columns and 'humidity' in df.columns:
                df_indices['heat_index'] = calculate_heat_index(
                    df['temperature'].values,
                    df['humidity'].values
                )
        
        # Humidity Index (discomfort index)
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df_indices['humidity_index'] = (
                df['temperature'] - 0.55 * (1 - df['humidity']/100) * (df['temperature'] - 14.5)
            )
        
        # Storm Surge Potential
        if 'storm_surge_potential' not in df.columns:
            if 'wind_speed' in df.columns and 'pressure' in df.columns:
                df_indices['storm_surge_potential'] = calculate_storm_surge_potential(
                    df['wind_speed'].values,
                    df['pressure'].values
                )
        
        # Drought Index (simplified using rainfall)
        if 'precipitation' in df.columns:
            # 30-day cumulative rainfall
            precip_30d = df['precipitation'].rolling(window=30, min_periods=1).sum()
            # Standardized Precipitation Index (simplified)
            precip_mean = precip_30d.mean()
            precip_std = precip_30d.std()
            df_indices['drought_index'] = -(precip_30d - precip_mean) / (precip_std + 0.001)
        
        # Wet Bulb Temperature (approximation)
        if 'temperature' in df.columns and 'humidity' in df.columns:
            T = df['temperature']
            RH = df['humidity']
            df_indices['wet_bulb_temp'] = (
                T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) +
                np.arctan(T + RH) - np.arctan(RH - 1.676331) +
                0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH) - 4.686035
            )
        
        self.logger.info("Created climate indices")
        return df_indices
    
    def create_extreme_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary indicators for extreme events.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        
        Returns:
        --------
        pd.DataFrame
            Data with extreme event indicators
        """
        self.logger.info("Creating extreme event indicators...")
        df_extreme = df.copy()
        
        # Extreme heat (> 90th percentile)
        if 'temperature' in df.columns:
            temp_90 = df['temperature'].quantile(0.90)
            df_extreme['is_extreme_heat'] = (df['temperature'] > temp_90).astype(int)
        
        # Extreme rainfall (> 95th percentile)
        if 'precipitation' in df.columns:
            precip_95 = df['precipitation'].quantile(0.95)
            df_extreme['is_extreme_rainfall'] = (df['precipitation'] > precip_95).astype(int)
        
        # High wind (> 85th percentile)
        if 'wind_speed' in df.columns:
            wind_85 = df['wind_speed'].quantile(0.85)
            df_extreme['is_high_wind'] = (df['wind_speed'] > wind_85).astype(int)
        
        # Low pressure (< 15th percentile - storm indicator)
        if 'pressure' in df.columns:
            pressure_15 = df['pressure'].quantile(0.15)
            df_extreme['is_low_pressure'] = (df['pressure'] < pressure_15).astype(int)
        
        # Consecutive dry days
        if 'precipitation' in df.columns:
            dry_days = (df['precipitation'] < 1.0).astype(int)
            df_extreme['consecutive_dry_days'] = (
                dry_days.groupby((dry_days != dry_days.shift()).cumsum()).cumsum()
            )
        
        # Consecutive wet days
        if 'precipitation' in df.columns:
            wet_days = (df['precipitation'] >= 1.0).astype(int)
            df_extreme['consecutive_wet_days'] = (
                wet_days.groupby((wet_days != wet_days.shift()).cumsum()).cumsum()
            )
        
        self.logger.info("Created extreme event indicators")
        return df_extreme
    
    def create_all_features(self, df: pd.DataFrame,
                           include_temporal: bool = True,
                           include_rolling: bool = True,
                           include_lags: bool = True,
                           include_roc: bool = True,
                           include_interactions: bool = True,
                           include_indices: bool = True,
                           include_extremes: bool = True) -> pd.DataFrame:
        """
        Create all features at once.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        include_* : bool
            Flags to control which features to create
        
        Returns:
        --------
        pd.DataFrame
            Data with all features
        """
        self.logger.info("Creating all features...")
        df_features = df.copy()
        
        # Core weather columns to use for feature engineering
        weather_columns = [
            'temperature', 'temperature_max', 'temperature_min',
            'precipitation', 'humidity', 'wind_speed', 'pressure'
        ]
        available_columns = [col for col in weather_columns if col in df.columns]
        
        if include_temporal:
            df_features = self.create_temporal_features(df_features)
        
        if include_indices:
            df_features = self.create_climate_indices(df_features)
        
        if include_rolling:
            df_features = self.create_rolling_features(df_features, available_columns)
        
        if include_lags:
            df_features = self.create_lag_features(df_features, available_columns)
        
        if include_roc:
            df_features = self.create_rate_of_change_features(df_features, available_columns)
        
        if include_interactions:
            df_features = self.create_interaction_features(df_features)
        
        if include_extremes:
            df_features = self.create_extreme_indicators(df_features)
        
        # Remove any infinite values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        self.logger.info(f"Total features created: {len(df_features.columns)}")
        return df_features


def main():
    """
    Example usage of feature engineering module.
    """
    from utils import load_config, setup_logging
    from data_preprocessing import ClimateDataLoader
    
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    # Load processed data
    loader = ClimateDataLoader(config, logger)
    try:
        df = loader.load_processed_data('climate_data_processed.csv')
    except FileNotFoundError:
        logger.info("Processed data not found, generating sample data...")
        df_raw = loader.generate_sample_data(days=730)
        df_clean = loader.clean_data(df_raw)
        df = loader.calculate_derived_metrics(df_clean)
        loader.save_processed_data(df, 'climate_data_processed.csv')
    
    # Initialize feature engineer
    engineer = ClimateFeatureEngineer(config, logger)
    
    # Create all features
    print("Creating features...")
    df_features = engineer.create_all_features(df)
    
    # Save features
    print("Saving features...")
    loader.save_processed_data(df_features, 'climate_features.csv')
    
    print(f"\nFeature engineering complete!")
    print(f"Original columns: {len(df.columns)}")
    print(f"Total columns after feature engineering: {len(df_features.columns)}")
    print(f"\nFirst few rows:")
    print(df_features.head())
    print(f"\nFeature names:")
    for col in sorted(df_features.columns):
        print(f"  - {col}")


if __name__ == "__main__":
    main()
