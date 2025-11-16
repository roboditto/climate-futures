import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json


class ClimateDataLoader:
    """
    Handles loading and preprocessing climate data from various sources.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        logger : logging.Logger, optional
            Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.storage = config.get('storage', {})
        self.data_sources = config.get('data_sources', {})
        
        # Create data directories
        os.makedirs(self.storage.get('raw_data_path', 'data/raw/'), exist_ok=True)
        os.makedirs(self.storage.get('processed_data_path', 'data/processed/'), exist_ok=True)
        os.makedirs(self.storage.get('cache_path', 'data/cache/'), exist_ok=True)
    
    def fetch_nasa_power_data(self, 
                              start_date: str, 
                              end_date: str,
                              latitude: Optional[float] = None,
                              longitude: Optional[float] = None) -> pd.DataFrame:
        """
        Fetch climate data from NASA POWER API.
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYYMMDD format
        end_date : str
            End date in YYYYMMDD format
        latitude : float, optional
            Latitude (uses config default if None)
        longitude : float, optional
            Longitude (uses config default if None)
        
        Returns:
        --------
        pd.DataFrame
            Climate data with datetime index
        """
        self.logger.info(f"Fetching NASA POWER data from {start_date} to {end_date}")
        
        # Use default location if not provided
        if latitude is None:
            latitude = self.config['location']['latitude']
        if longitude is None:
            longitude = self.config['location']['longitude']
        
        # Get parameters to fetch
        parameters = ','.join(self.data_sources['nasa_power']['parameters'])
        
        # Build API URL
        base_url = self.data_sources['nasa_power']['api_url']
        url = (f"{base_url}?"
               f"parameters={parameters}&"
               f"community=AG&"
               f"longitude={longitude}&"
               f"latitude={latitude}&"
               f"start={start_date}&"
               f"end={end_date}&"
               f"format=JSON")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract parameter data
            parameters_data = data['properties']['parameter']
            
            # Convert to DataFrame
            df = pd.DataFrame(parameters_data)
            df.index = pd.to_datetime(df.index, format='%Y%m%d')
            df.index.name = 'date'
            
            # Rename columns for clarity
            column_mapping = {
                'T2M': 'temperature',
                'T2M_MAX': 'temperature_max',
                'T2M_MIN': 'temperature_min',
                'PRECTOTCORR': 'precipitation',
                'RH2M': 'humidity',
                'WS10M': 'wind_speed',
                'PS': 'pressure'
            }
            df.rename(columns=column_mapping, inplace=True)
            
            # Convert pressure from kPa to hPa
            if 'pressure' in df.columns:
                df['pressure'] = df['pressure'] * 10
            
            self.logger.info(f"Successfully fetched {len(df)} days of data")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching NASA POWER data: {e}")
            raise
    
    def generate_sample_data(self, 
                            days: int = 365,
                            start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Generate sample climate data for testing (when real data unavailable).
        
        Parameters:
        -----------
        days : int
            Number of days to generate
        start_date : str, optional
            Start date (YYYY-MM-DD format)
        
        Returns:
        --------
        pd.DataFrame
            Synthetic climate data
        """
        self.logger.info(f"Generating {days} days of sample data")
        
        if start_date is None:
            end_date = datetime.now()
            start_date_dt = end_date - timedelta(days=days)
        else:
            start_date_dt = pd.to_datetime(start_date)
            end_date = start_date_dt + timedelta(days=days)
        
        # Create date range
        dates = pd.date_range(start=start_date_dt, end=end_date, freq='D')
        
        np.random.seed(42)
        
        # Generate realistic Caribbean climate data
        # Temperature: seasonal variation + daily noise
        day_of_year = dates.dayofyear.values  # Convert to numpy array
        temp_seasonal = 27 + 3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temperature = temp_seasonal + np.random.normal(0, 2, len(dates))
        
        # Temperature max/min
        temperature_max = temperature + np.random.uniform(2, 5, len(dates))
        temperature_min = temperature - np.random.uniform(2, 4, len(dates))
        
        # Humidity: 70-90% typical for Caribbean
        humidity = 75 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5, len(dates))
        humidity = np.clip(humidity, 50, 95)
        
        # Rainfall: seasonal with occasional heavy events
        rainfall_base = 2 + 3 * np.sin(2 * np.pi * (day_of_year - 150) / 365)
        rainfall_noise = np.random.gamma(2, 2, len(dates))
        precipitation = rainfall_base * rainfall_noise
        # Add some extreme events
        extreme_events = np.random.random(len(dates)) < 0.05  # 5% chance
        extreme_indices = np.where(extreme_events)[0]
        precipitation[extreme_indices] *= np.random.uniform(5, 15, len(extreme_indices))
        
        # Wind speed: 3-8 m/s typical
        wind_speed = 5 + 2 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 1.5, len(dates))
        wind_speed = np.clip(wind_speed, 0, 25)
        
        # Pressure: around 1013 hPa
        pressure = 1013 + 5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3, len(dates))
        
        # Create DataFrame
        df = pd.DataFrame({
            'temperature': temperature,
            'temperature_max': temperature_max,
            'temperature_min': temperature_min,
            'precipitation': precipitation,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'pressure': pressure
        }, index=dates)
        
        df.index.name = 'date'
        
        self.logger.info(f"Generated sample data with shape {df.shape}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate climate data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw climate data
        
        Returns:
        --------
        pd.DataFrame
            Cleaned data
        """
        self.logger.info("Cleaning data...")
        df_clean = df.copy()
        
        # Remove duplicate dates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # Sort by date
        df_clean = df_clean.sort_index()
        
        # Handle missing values
        # For temperature and pressure: interpolate
        for col in ['temperature', 'temperature_max', 'temperature_min', 'pressure']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].interpolate(method='linear', limit=3)
        
        # For precipitation and wind: forward fill (assume 0 if missing)
        for col in ['precipitation', 'wind_speed']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
        
        # For humidity: interpolate
        if 'humidity' in df_clean.columns:
            df_clean['humidity'] = df_clean['humidity'].interpolate(method='linear', limit=3)
        
        # Validate ranges
        if 'temperature' in df_clean.columns:
            df_clean.loc[df_clean['temperature'] < -10, 'temperature'] = np.nan
            df_clean.loc[df_clean['temperature'] > 50, 'temperature'] = np.nan
        
        if 'humidity' in df_clean.columns:
            df_clean['humidity'] = df_clean['humidity'].clip(0, 100)
        
        if 'precipitation' in df_clean.columns:
            df_clean['precipitation'] = df_clean['precipitation'].clip(0, None)
        
        if 'wind_speed' in df_clean.columns:
            df_clean['wind_speed'] = df_clean['wind_speed'].clip(0, None)
        
        self.logger.info(f"Cleaned data: {len(df_clean)} records")
        return df_clean
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived climate metrics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Cleaned climate data
        
        Returns:
        --------
        pd.DataFrame
            Data with additional derived metrics
        """
        from utils import calculate_heat_index, calculate_storm_surge_potential
        
        self.logger.info("Calculating derived metrics...")
        df_derived = df.copy()
        
        # Heat index
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df_derived['heat_index'] = calculate_heat_index(
                np.asarray(df['temperature'].values),
                np.asarray(df['humidity'].values)
            )
        
        # Temperature anomaly (from mean)
        if 'temperature' in df.columns:
            baseline_mean = df['temperature'].mean()
            baseline_std = df['temperature'].std()
            df_derived['temperature_anomaly'] = (df['temperature'] - baseline_mean) / baseline_std
        
        # Rainfall cumulative (7-day)
        if 'precipitation' in df.columns:
            df_derived['precipitation_7day'] = df['precipitation'].rolling(window=7, min_periods=1).sum()
            df_derived['precipitation_30day'] = df['precipitation'].rolling(window=30, min_periods=1).sum()
        
        # Storm surge potential
        if 'wind_speed' in df.columns and 'pressure' in df.columns:
            df_derived['storm_surge_potential'] = calculate_storm_surge_potential(
                np.asarray(df['wind_speed'].values),
                np.asarray(df['pressure'].values)
            )
        
        # Day of year (for seasonality)
        dt_index = pd.to_datetime(df_derived.index)
        df_derived['day_of_year'] = dt_index.dayofyear
        df_derived['month'] = dt_index.month
        
        self.logger.info(f"Added {len(df_derived.columns) - len(df.columns)} derived metrics")
        return df_derived
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to CSV.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed data
        filename : str
            Output filename
        """
        output_path = os.path.join(
            self.storage.get('processed_data_path', 'data/processed/'),
            filename
        )
        df.to_csv(output_path)
        self.logger.info(f"Saved processed data to {output_path}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Load processed data from CSV.
        
        Parameters:
        -----------
        filename : str
            Filename to load
        
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        filepath = os.path.join(
            self.storage.get('processed_data_path', 'data/processed/'),
            filename
        )
        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
        self.logger.info(f"Loaded data from {filepath}: {df.shape}")
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Climate data
        
        Returns:
        --------
        dict
            Summary statistics
        """
        summary = {
            'records': len(df),
            'date_range': {
                'start': str(df.index.min()),
                'end': str(df.index.max()),
                'days': (df.index.max() - df.index.min()).days
            },
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'statistics': df.describe().to_dict()
        }
        
        return summary


def main():
    """
    Example usage of the data preprocessing module.
    """
    from utils import load_config, setup_logging
    
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    # Initialize data loader
    loader = ClimateDataLoader(config, logger)
    
    # Generate sample data (or fetch from NASA POWER)
    print("Generating sample climate data...")
    df_raw = loader.generate_sample_data(days=730)  # 2 years
    
    # Clean data
    print("Cleaning data...")
    df_clean = loader.clean_data(df_raw)
    
    # Calculate derived metrics
    print("Calculating derived metrics...")
    df_processed = loader.calculate_derived_metrics(df_clean)
    
    # Save processed data
    print("Saving processed data...")
    loader.save_processed_data(df_processed, 'climate_data_processed.csv')
    
    # Get summary
    summary = loader.get_data_summary(df_processed)
    print("\nData Summary:")
    print(f"Records: {summary['records']}")
    print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Columns: {len(summary['columns'])}")
    print("\nFirst few rows:")
    print(df_processed.head())
    print("\nStatistics:")
    print(df_processed.describe())


if __name__ == "__main__":
    main()
