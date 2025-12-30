"""ERA5 helper utilities

Functions to download ERA5-Land hourly total precipitation via the CDS API
and regrid it to a target model grid (rows,cols) and bounds (south,west,north,east).

These helpers are optional and will log a helpful message if `cdsapi` or
`xarray` are not installed.
"""
from __future__ import annotations
import datetime
import logging
import os
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except Exception:
    cdsapi = None
    CDSAPI_AVAILABLE = False

try:
    import xarray as xr
    import numpy as np
    import zipfile
    import tempfile
    XR_AVAILABLE = True
except Exception:
    xr = None
    np = None
    XR_AVAILABLE = False


def _date_range_days(start_date: str, end_date: str) -> List[datetime.date]:
    sd = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    ed = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    day = sd
    out = []
    while day <= ed:
        out.append(day)
        day += datetime.timedelta(days=1)
    return out


def download_era5_hourly(bounds: Tuple[float, float, float, float], start_date: str, end_date: str,
                         out_path: str = "data/era5_precip.nc") -> Optional[str]:
    """
    Download ERA5-Land hourly total precipitation for the given bbox and date range.

    bounds: (south, west, north, east)
    Dates: 'YYYY-MM-DD'

    Returns path to the downloaded netCDF file or None if CDS API not available.
    """
    if not CDSAPI_AVAILABLE:
        logger.warning("cdsapi not available; cannot download ERA5. Install 'cdsapi' and configure ~/.cdsapirc")
        return None

    south, west, north, east = bounds
    days = _date_range_days(start_date, end_date)
    if len(days) == 0:
        logger.error("No days in requested range")
        return None

    # Build year/month/day lists for the CDS request
    years = sorted({d.year for d in days})
    months = sorted({d.month for d in days})
    days_list = [f"{d.day:02d}" for d in days]

    # CDS expects area as [N, W, S, E]
    area = [north, west, south, east]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    c = cdsapi.Client()
    try:
        logger.info(f"Requesting ERA5-Land total_precipitation for {start_date} to {end_date}")
        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': 'total_precipitation',
                'product_type': 'reanalysis',
                'year': [str(y) for y in years],
                'month': [f"{m:02d}" for m in months],
                'day': days_list,
                'time': [f"{h:02d}:00" for h in range(24)],
                'area': area,
                'format': 'netcdf'
            },
            out_path)
        logger.info(f"ERA5 download saved to {out_path}")
        return out_path
    except Exception as e:
        logger.warning(f"ERA5 download failed: {e}")
        return None


def download_era5_variables(bounds: Tuple[float, float, float, float], start_date: str, end_date: str,
                            variables: list, out_path: str = "data/era5_vars.nc",
                            dataset: str = 'reanalysis-era5-single-levels') -> Optional[str]:
    """
    Download specified ERA5 variables (hourly) for given bbox and date range.

    `variables` should be a list of variable names as expected by the CDS dataset,
    e.g. ['2m_temperature', '2m_dewpoint_temperature'].
    """
    if not CDSAPI_AVAILABLE:
        logger.warning("cdsapi not available; cannot download ERA5 variables.")
        return None

    south, west, north, east = bounds
    days = _date_range_days(start_date, end_date)
    if len(days) == 0:
        logger.error("No days in requested range")
        return None

    years = sorted({d.year for d in days})
    months = sorted({d.month for d in days})
    days_list = [f"{d.day:02d}" for d in days]

    # CDS expects area as [N, W, S, E]
    area = [north, west, south, east]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    c = cdsapi.Client()
    try:
        logger.info(f"Requesting ERA5 variables {variables} for {start_date} to {end_date}")
        req = {
            'product_type': 'reanalysis',
            'year': [str(y) for y in years],
            'month': [f"{m:02d}" for m in months],
            'day': days_list,
            'time': [f"{h:02d}:00" for h in range(24)],
            'area': area,
            'format': 'netcdf'
        }
        # variable key differs by dataset; use provided list
        req['variable'] = variables

        c.retrieve(dataset, req, out_path)
        logger.info(f"ERA5 variables download saved to {out_path}")
        return out_path
    except Exception as e:
        logger.warning(f"ERA5 variables download failed: {e}")
        return None


def regrid_era5_to_grid(nc_path: str, target_shape: Tuple[int, int], target_bounds: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """
    Open ERA5 netCDF (as produced above) and regrid/interpolate hourly precipitation
    to a target rectangular grid.

    Returns numpy array of shape (time, rows, cols) in millimeters.
    """
    if not XR_AVAILABLE:
        logger.warning("xarray/numpy not available; cannot regrid ERA5 data")
        return None

    # Basic file checks
    if not os.path.exists(nc_path):
        logger.error(f'ERA5 file not found: {nc_path}')
        return None
    try:
        size = os.path.getsize(nc_path)
    except Exception:
        size = 0
    if size < 1024:
        logger.error(f'ERA5 file appears too small ({size} bytes): {nc_path}')
        return None
    # If file is a ZIP archive (CDS may return zipped netCDF), extract first .nc member
    if zipfile.is_zipfile(nc_path):
        try:
            tmpdir = tempfile.mkdtemp(prefix='era5_')
            with zipfile.ZipFile(nc_path, 'r') as zf:
                members = [m for m in zf.namelist() if m.lower().endswith('.nc')]
                if not members:
                    logger.error('ZIP archive contains no .nc files')
                    return None
                member = members[0]
                out_fp = os.path.join(tmpdir, os.path.basename(member))
                with zf.open(member) as srcf, open(out_fp, 'wb') as outf:
                    outf.write(srcf.read())
                logger.info(f'Extracted {member} -> {out_fp}')
                nc_path = out_fp
        except Exception as e:
            logger.error(f'Failed to extract ZIP archive: {e}')
            return None

    # Try opening with a list of likely xarray engines to provide clearer errors
    ds = None
    open_errors = {}
    engines_to_try = ['netcdf4', 'h5netcdf', 'scipy', 'cfgrib']
    for eng in engines_to_try:
        try:
            ds = xr.open_dataset(nc_path, engine=eng)
            logger.info(f'Opened ERA5 file with xarray engine "{eng}"')
            break
        except Exception as e:
            open_errors[eng] = str(e)

    if ds is None:
        # If no engine succeeded, log helpful diagnostics
        logger.error('Could not open ERA5 file with available xarray IO engines')
        for eng, err in open_errors.items():
            logger.debug(f'engine {eng}: {err}')
        return None

    # ERA5-Land variable name
    varname = 'total_precipitation'
    if varname not in ds:
        # try common alternatives
        varname = [v for v in ds.data_vars][0]

    da = ds[varname]
    # Convert meters -> mm
    da_mm = da * 1000.0

    # Ensure dims are named latitude/longitude or lat/lon
    lat_name = None
    lon_name = None
    for n in da_mm.dims:
        if 'lat' in n.lower():
            lat_name = n
        if 'lon' in n.lower():
            lon_name = n

    if lat_name is None or lon_name is None:
        logger.error('Could not find latitude/longitude dims in ERA5 file')
        return None

    rows, cols = target_shape
    south, west, north, east = target_bounds

    # Target 1D coordinate arrays (latitude descending if needed)
    target_lats = np.linspace(north, south, rows)
    target_lons = np.linspace(west, east, cols)

    # xarray.interp expects coords in ascending order; ensure monotonic
    if target_lats[0] < target_lats[-1]:
        target_lats = target_lats[::-1]

    try:
        interp = da_mm.interp({lat_name: target_lats, lon_name: target_lons}, method='linear')
    except Exception as e:
        logger.warning(f'Interpolation via xarray failed: {e}; attempting fallback with numpy zoom')
        # Fallback: load into numpy and simple zoom per timestep
        arr = da_mm.values  # time, lat, lon
        time_dim = arr.shape[0]
        out = np.zeros((time_dim, rows, cols), dtype=float)
        from scipy import ndimage
        for t in range(time_dim):
            out[t] = ndimage.zoom(arr[t], (rows / arr.shape[1], cols / arr.shape[2]), order=1)
        return out

    # Ensure result dims ordering: time, lat, lon
    out = np.asarray(interp.values)
    # If lat dimension is descending, xarray handled that; but ensure shape matches (time, rows, cols)
    if out.ndim == 3:
        # If lat ordering reversed compared to target rows, flip first spatial axis
        if interp[lat_name][0] > interp[lat_name][-1]:
            out = out[:, ::-1, :]
        return out
    else:
        logger.error('Unexpected dimensionality from ERA5 interpolation')
        return None
