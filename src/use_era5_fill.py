"""Populate future-day precipitation from a regridded ERA5 hourly array.

This script loads `data/era5_test_regrid.npy` (shape: time, rows, cols),
aggregates hourly steps into daily sums, computes a spatial mean per day,
and writes those daily values into future rows (dates >= today) of
`data/processed/climate_features.csv`. A timestamped backup is created.
"""
from __future__ import annotations
import os
import math
from datetime import date, datetime
import numpy as np
import pandas as pd


def main():
    csv_path = "data/processed/climate_features.csv"
    era5_path = "data/era5_test_regrid.npy"

    if not os.path.exists(csv_path):
        print(f"Processed CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    today = pd.to_datetime(date.today()).normalize()
    future_mask = df.index.normalize() >= today
    future_idx = df.index[future_mask]
    n_future = len(future_idx)

    if n_future == 0:
        print("No future rows to update.")
        return

    if not os.path.exists(era5_path):
        print(f"ERA5 regrid file not found: {era5_path}. Nothing to apply.")
        return

    arr = np.load(era5_path)
    if arr.ndim != 3:
        print(f"Unexpected ERA5 array shape: {arr.shape}")
        return

    t, r, c = arr.shape
    hours_per_day = 24
    days_available = math.ceil(t / hours_per_day)

    daily_means: list[float] = []
    for d in range(days_available):
        start = d * hours_per_day
        end = min((d + 1) * hours_per_day, t)
        window = arr[start:end]
        # Sum precipitation across the hours to get daily total per grid cell
        daily_sum = np.nansum(window, axis=0)
        # Reduce spatially — use the mean across the grid as a representative value
        spatial_mean = float(np.nanmean(daily_sum))
        if math.isnan(spatial_mean):
            spatial_mean = 0.0
        daily_means.append(spatial_mean)

    if len(daily_means) == 0:
        print("No daily ERA5 data could be computed.")
        return

    # Apply daily_means to future rows in order, repeating if necessary
    for i, idx in enumerate(future_idx):
        val = daily_means[i % len(daily_means)]
        df.at[idx, 'precipitation'] = val

    # Backup and write
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = f"data/processed/climate_features_backup_{timestamp}.csv"
    df.to_csv(backup_path)
    df.to_csv(csv_path)

    print(f"Applied ERA5-derived precipitation to {n_future} future rows.")
    print(f"Backup written to: {backup_path}")


if __name__ == '__main__':
    main()
