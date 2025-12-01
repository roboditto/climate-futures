"""
Extend processed climate data to today's date by fetching real NASA POWER observations
or generating synthetic observations for missing days, then re-running feature engineering.

Usage:
    python extend_data.py [--days N] [--real]

If `--real` is provided, fetch real data from NASA POWER API.
If `--days` is provided, the script will append up to N days beyond the last record.
Otherwise it will extend the data through today.
"""
import sys
from pathlib import Path
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config, setup_logging
from data_preprocessing import ClimateDataLoader
from features import ClimateFeatureEngineer
import pandas as pd


def main(days: int = 0, use_real_data: bool = False):
    config = load_config()
    logger = setup_logging(config)

    loader = ClimateDataLoader(config, logger)
    fe = ClimateFeatureEngineer(config, logger)

    processed_path = Path(loader.storage.get('processed_data_path', 'data/processed/')) / 'climate_features.csv'

    if processed_path.exists():
        print(f"Loading existing processed data: {processed_path}")
        df_existing = loader.load_processed_data('climate_features.csv')
        last_date = pd_last = df_existing.index.max()
        print(f"Last date in processed data: {last_date}")
    else:
        print("No existing processed data found. Creating new dataset from scratch.")
        df_existing = None
        last_date = None

    today = datetime.now().date()

    if days is not None:
        target_date = (last_date.date() if last_date is not None else today - timedelta(days=days)) + timedelta(days=days)
    else:
        target_date = today

    # Determine start date for generation
    if last_date is None:
        start_date = None
        delta_days = (target_date - (today - timedelta(days=365))).days if target_date else 365
    else:
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        delta_days = (target_date - last_date.date()).days

    if delta_days <= 0:
        print("Processed data already up-to-date. No extension necessary.")
        return

    # Try to fetch real data if requested, otherwise use synthetic
    if use_real_data:
        print(f"Attempting to fetch real NASA POWER data from {start_date}...")
        try:
            # Format dates for NASA POWER API (YYYYMMDD)
            if last_date is not None:
                start_yyyymmdd = (last_date + timedelta(days=1)).strftime('%Y%m%d')
                end_yyyymmdd = (last_date + timedelta(days=delta_days)).strftime('%Y%m%d')
            else:
                start_yyyymmdd = (datetime.now() - timedelta(days=delta_days)).strftime('%Y%m%d')
                end_yyyymmdd = (datetime.now()).strftime('%Y%m%d')
            
            print(f"  Fetching from {start_yyyymmdd} to {end_yyyymmdd}")
            df_new_raw = loader.fetch_nasa_power_data(start_yyyymmdd, end_yyyymmdd)
            print(f"Successfully fetched {len(df_new_raw)} days from NASA POWER")
            df_new_clean = loader.clean_data(df_new_raw)
            df_new_processed = loader.calculate_derived_metrics(df_new_clean)
        except Exception as e:
            print(f"NASA POWER fetch failed ({e}), falling back to synthetic data...")
            print(f"Generating {delta_days} days of synthetic data starting from {start_date}...")
            df_new_raw = loader.generate_sample_data(days=delta_days, start_date=start_date)
            df_new_clean = loader.clean_data(df_new_raw)
            df_new_processed = loader.calculate_derived_metrics(df_new_clean)
    else:
        print(f"Generating {delta_days} days of synthetic data starting from {start_date}...")
        df_new_raw = loader.generate_sample_data(days=delta_days, start_date=start_date)
        df_new_clean = loader.clean_data(df_new_raw)
        df_new_processed = loader.calculate_derived_metrics(df_new_clean)

    # Combine with existing processed data
    if df_existing is not None:
        df_combined = pd.concat([df_existing, df_new_processed])
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')].sort_index()
    else:
        df_combined = df_new_processed

    # Re-run full feature engineering if available
    try:
        df_features = fe.create_all_features(df_combined)
    except Exception as e:
        logger.exception('Feature engineering failed; saving derived metrics instead')
        df_features = df_combined

    loader.save_processed_data(df_features, 'climate_features.csv')
    print(f"Processed data extended and saved to {processed_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, help='Number of days to extend (optional)')
    parser.add_argument('--real', action='store_true', help='Fetch real NASA POWER data instead of synthetic (requires API access)')
    args = parser.parse_args()
    main(days=args.days, use_real_data=args.real)
