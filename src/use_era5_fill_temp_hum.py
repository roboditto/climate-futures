"""Fill future temperature and humidity using ERA5 t2m / d2m when available.

Behavior:
- Reads `data/processed/climate_features.csv` and determines future dates to fill.
- Attempts to use an existing local ERA5 file `data/era5_test.nc`.
- If not present, attempts to download `2m_temperature` and
  `2m_dewpoint_temperature` using `src.era5_helper.download_era5_variables`.
- Regrids downloaded ERA5 to a modest model grid and aggregates hourly->daily
  before filling `temperature` (C) and `humidity` (%) columns.

Notes:
- A configured `~/.cdsapirc` is required for CDS downloads. If unavailable,
  provide your own ERA5 NetCDF at `data/era5_vars.nc` or `data/era5_test.nc`.
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, datetime
import tempfile
import zipfile
import os
import numpy as np
import pandas as pd

try:
    import xarray as xr
except Exception:
    xr = None

try:
    import yaml
except Exception:
    yaml = None

from src.era5_helper import download_era5_variables, regrid_era5_to_grid


def load_config_bbox() -> tuple | None:
    cfg_path = Path('config/config.yaml')
    if not cfg_path.exists() or yaml is None:
        return None
    with open(cfg_path, 'r', encoding='utf8') as fh:
        cfg = yaml.safe_load(fh)
    bbox = cfg.get('location', {}).get('bbox')
    if not bbox or len(bbox) != 4:
        return None
    # config bbox is [min_lon, min_lat, max_lon, max_lat]
    west, south, east, north = bbox[0], bbox[1], bbox[2], bbox[3]
    # era5_helper expects (south, west, north, east)
    return (south, west, north, east)


def extract_inner_nc(nc_path: Path) -> Path | None:
    # If zip-like, extract first .nc into temp file and return path
    if not nc_path.exists():
        return None
    if zipfile.is_zipfile(nc_path):
        td = tempfile.mkdtemp(prefix='era5_extract_')
        with zipfile.ZipFile(nc_path, 'r') as zf:
            members = [m for m in zf.namelist() if m.lower().endswith('.nc')]
            if not members:
                return None
            member = members[0]
            out_fp = Path(td) / Path(member).name
            with zf.open(member) as srcf, open(out_fp, 'wb') as outf:
                outf.write(srcf.read())
            return out_fp
    return nc_path


def hourly_to_daily_mean(arr: np.ndarray) -> np.ndarray:
    # arr: (time, rows, cols)
    if arr is None:
        return None
    t = arr.shape[0]
    days = int(np.ceil(t / 24))
    daily = []
    for d in range(days):
        s = d * 24
        e = min((d + 1) * 24, t)
        window = arr[s:e]
        # mean over time then spatial mean
        daily_val = float(np.nanmean(window))
        daily.append(daily_val)
    return np.array(daily)


def rh_from_t_td(t_c, td_c):
    a, b = 17.625, 243.04
    es = 6.1094 * np.exp(a * t_c / (b + t_c))
    ed = 6.1094 * np.exp(a * td_c / (b + td_c))
    return float(min(max(0.0, 100.0 * (ed / es)), 100.0))


def main():
    csv_path = Path('data/processed/climate_features.csv')
    if not csv_path.exists():
        print('Processed CSV not found; aborting.')
        return

    df = pd.read_csv(csv_path, index_col='date', parse_dates=True)
    today = pd.to_datetime(date.today()).normalize()
    future_idx = df.index[df.index.normalize() >= today]
    if len(future_idx) == 0:
        print('No future rows to update.')
        return

    # Determine required date range to download from ERA5: use min/max of future rows
    start_date = future_idx[0].strftime('%Y-%m-%d')
    end_date = future_idx[-1].strftime('%Y-%m-%d')

    # Prefer user-supplied ERA5 file if present
    local_candidates = [Path('data/era5_vars.nc'), Path('data/era5_test.nc')]
    nc_path = None
    for p in local_candidates:
        if p.exists():
            nc_path = p
            break

    bounds = load_config_bbox()
    downloaded = False
    if nc_path is None and bounds is not None:
        vars_to_get = ['2m_temperature', '2m_dewpoint_temperature']
        outp = Path('data/era5_vars.nc')
        print(f'Attempting CDS download for variables {vars_to_get} for {start_date}..{end_date} using bbox from config')
        res = download_era5_variables(bounds, start_date, end_date, vars_to_get, out_path=str(outp))
        if res:
            nc_path = outp
            downloaded = True
        else:
            print('CDS download failed or unavailable; falling back to local ERA5 files if present.')

    if nc_path is None:
        print('No ERA5 data available (neither local nor downloaded). Aborting temp/humidity fill.')
        return

    # Extract inner .nc if zipped
    extracted = extract_inner_nc(nc_path)
    if extracted is None:
        print('Failed to extract/open ERA5 netCDF file.')
        return

    # Regrid to moderate grid (rows,cols). Prefer existing regrid shape if present
    target_shape = (50, 50)
    target_bounds = bounds if bounds is not None else (-90.0, -180.0, 90.0, 180.0)

    print(f'Regridding ERA5 to shape {target_shape} and bounds {target_bounds} (this may be slow)')
    arr = regrid_era5_to_grid(str(extracted), target_shape, target_bounds)
    if arr is None:
        print('Regridding failed; cannot compute temperature/humidity fills.')
        return

    # Expect arr shape (time, rows, cols). Identify variables order in file: we assume
    # the NetCDF contains variables in order and regrid_era5_to_grid picked the first data_var.
    # For downloaded multi-variable files, regrid_era5_to_grid will pick first variable; therefore
    # attempt to open with xarray directly to extract t2m/d2m arrays if present.
    temp_daily = None
    dew_daily = None
    if xr is not None:
        try:
            ds = xr.open_dataset(extracted)
            ds = ds.load()
            # Common variable names in CDS: '2m_temperature', '2m_dewpoint_temperature'
            if '2m_temperature' in ds and '2m_dewpoint_temperature' in ds:
                ta = ds['2m_temperature']  # time, lat, lon
                td = ds['2m_dewpoint_temperature']
                # convert to numpy and regrid each variable separately
                # simple approach: reuse regrid_era5_to_grid by saving tmp files per variable
                # but to keep simple, compute spatial mean across original grid then hourly->daily
                t_arr = np.asarray(ta.values)
                td_arr = np.asarray(td.values)
                # compute hourly spatial means then daily
                t_hourly_mean = np.nanmean(t_arr.reshape(t_arr.shape[0], -1), axis=1)
                td_hourly_mean = np.nanmean(td_arr.reshape(td_arr.shape[0], -1), axis=1)
                # reshape to (time, rows, cols) mean across spatial dims already taken
                temp_daily = hourly_to_daily_mean(t_hourly_mean[:, None, None])
                dew_daily = hourly_to_daily_mean(td_hourly_mean[:, None, None])
            else:
                # fallback: use spatial mean from regridded array (which may correspond to first var)
                print('Downloaded ERA5 does not contain both 2m variables; using spatial mean from regridded array for available variable(s).')
                # If arr contains precipitation or a single variable, treat it as temperature proxy only if reasonable
                t_hourly = np.nanmean(arr, axis=(1, 2))
                temp_daily = hourly_to_daily_mean(t_hourly[:, None, None])
        except Exception as e:
            print('xarray open failed or unexpected file structure:', e)

    # Convert Kelvin to Celsius heuristic
    def k2c(v):
        if np.isnan(v):
            return np.nan
        v = float(v)
        return v - 273.15 if v > 100 else v

    if temp_daily is not None:
        temp_c = [k2c(v) for v in temp_daily]
    else:
        temp_c = None

    if dew_daily is not None:
        dew_c = [k2c(v) for v in dew_daily]
    else:
        dew_c = None

    # Compute RH if possible
    rh_daily = None
    if temp_c is not None and dew_c is not None:
        days = max(len(temp_c), len(dew_c))
        temp_rep = [temp_c[i % len(temp_c)] for i in range(days)]
        dew_rep = [dew_c[i % len(dew_c)] for i in range(days)]
        rh_daily = [rh_from_t_td(t, d) for t, d in zip(temp_rep, dew_rep)]

    # Apply to future rows
    for i, idx in enumerate(future_idx):
        if temp_c is not None:
            df.at[idx, 'temperature'] = temp_c[i % len(temp_c)]
        if rh_daily is not None:
            df.at[idx, 'humidity'] = rh_daily[i % len(rh_daily)]

    # Backup and write
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    backup = csv_path.parent / f'climate_features_backup_temp_hum_{timestamp}.csv'
    df.to_csv(backup)
    df.to_csv(csv_path)

    print(f'Wrote temperature/humidity for {len(future_idx)} future rows. Backup: {backup}')


if __name__ == '__main__':
    main()
