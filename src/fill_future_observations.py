"""Fill future observation columns using model predictions (and ERA5 if available).

This script:
- loads data/processed/climate_features.csv
- runs generate_predictions to append a forecast horizon
- fills future-day 'precipitation' using 'rainfall_pred'
- fills 'temperature' and 'humidity' for future days using last-known values or short rolling means
- saves a timestamped backup and writes the updated CSV back to disk

Note: ERA5 regridded arrays are historical (reanalysis) and cannot provide true future values.
"""
import os
import datetime
import pandas as pd
from pathlib import Path

# ensure src is on path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predictions import generate_predictions

CSV_PATH = Path('data/processed/climate_features.csv')
BACKUP_FMT = 'data/processed/climate_features_backup_{ts}.csv'

if not CSV_PATH.exists():
    print('Processed CSV not found:', CSV_PATH)
    raise SystemExit(1)

print('Loading processed features...')
df = pd.read_csv(CSV_PATH, index_col='date', parse_dates=True)

# Minimal system placeholder - generate_predictions has robust fallbacks
system = {'models': {}, 'logger': None, 'risk_calculator': type('R', (), {'calculate_batch_risk': lambda self, df: df})()}

print('Generating predictions (in-memory)...')
df_pred = generate_predictions(df, system, forecast_days=14)

# Determine future rows (date >= today)
today = pd.to_datetime(datetime.date.today()).normalize()
future_mask = df_pred.index.normalize() >= today
future_idx = df_pred.index[future_mask]
print('Future rows to fill:', len(future_idx))

if len(future_idx) == 0:
    print('No future rows to fill. Exiting.')
    raise SystemExit(0)

# Backup original CSV
ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
backup_path = BACKUP_FMT.format(ts=ts)
print('Backing up processed CSV to', backup_path)
os.makedirs(Path(backup_path).parent, exist_ok=True)
df.to_csv(backup_path)

# Fill precipitation from model prediction if available
if 'rainfall_pred' in df_pred.columns:
    vals = df_pred.loc[future_idx, 'rainfall_pred']
    # convert to numeric and replace NaN with 0
    vals = pd.to_numeric(vals, errors='coerce').fillna(0)
    print('Filling precipitation for future rows with rainfall_pred')
    # Ensure df has those indices (append if necessary)
    for idx in future_idx:
        if idx not in df.index:
            df.loc[idx] = [pd.NA] * len(df.columns)
    df = df.sort_index()
    df.loc[future_idx, 'precipitation'] = vals.values
else:
    print('No rainfall_pred found in generated predictions; skipping precipitation fill')

# Fill temperature and humidity from last observed values (or short rolling mean)
for var in ['temperature', 'humidity']:
    if var in df.columns:
        last_valid = df[var].dropna()
        if len(last_valid) == 0:
            fill_val = pd.NA
        else:
            # use 7-day rolling mean of last available observations if possible
            fill_val = float(last_valid.tail(7).mean())
        print(f'Filling {var} for future rows with', fill_val)
        df.loc[future_idx, var] = fill_val

# Sort and save
df = df.sort_index()
print('Writing updated CSV to', CSV_PATH)
df.to_csv(CSV_PATH)
print('Done. Backup at', backup_path)
