# Flood Simulation for Jamaica

Enhanced flood simulation module with support for real geospatial data (SRTM DEM, CHIRPS rainfall) and interactive mapping.

## Features

- **Synthetic DEM Generation**: Create realistic terrain with coastal gradients and noise
- **Spatial Rainfall Fields**: Generate orographic and convective rainfall patterns
- **D8 Flow Direction**: Calculate water flow paths on terrain
- **Rainfall-Runoff Simulation**: Model flood propagation using simplified Manning's equation
- **Interactive Maps**: Export Folium-based HTML maps with water depth overlays
- **Real Data Support**:
  - SRTM DEM fetching via `elevation` package
  - CHIRPS daily rainfall downloading and summation
- **Realistic Jamaica Defaults**: Bounding box, elevation range, and event magnitudes

## Quick Start

### Default (Synthetic Data, Non-Interactive)

```bash
python src/flood_simulation.py
```

Produces:

- `results/flood_simulation.png` - static 3-panel visualization (DEM, water depth, flood zones)
- `results/flood_map.html` - interactive Folium map with water depth overlay

### With Real SRTM DEM

Requires `elevation` package (may be slow on first run):

```bash
pip install elevation
python src/flood_simulation.py --use-srtm
```

The script will attempt to download and cache SRTM data to `data/dem_jamaica.tif`.

### With CHIRPS Rainfall Data

Download observed daily rainfall from CHC UCSB (requires network access):

```bash
python src/flood_simulation.py --use-chirps --chirps-start 2024-10-01 --chirps-end 2024-10-10
```

**Note**: CHIRPS files are downloaded on-demand; missing days are skipped gracefully.

### Interactive Plotting (GUI)

Enable Tk-based interactive plot display:

```bash
python src/flood_simulation.py --interactive
```

### Combined: SRTM + CHIRPS + Interactive

```bash
python src/flood_simulation.py --use-srtm --use-chirps --chirps-start 2024-10-01 --chirps-end 2024-10-05 --interactive
```

## CLI Arguments

```bash
usage: flood_simulation.py [-h] [--interactive] [--use-srtm] [--use-chirps] 
                           [--chirps-start CHIRPS_START] [--chirps-end CHIRPS_END]

options:
  -h, --help            show this help message and exit
  --interactive         Enable interactive plotting (GUI). Default: non-interactive (Agg backend).
  --use-srtm            Attempt to fetch SRTM DEM if not already present at data/dem_jamaica.tif
  --use-chirps          Attempt to download CHIRPS daily rainfall data.
  --chirps-start CHIRPS_START
                        CHIRPS download start date (YYYY-MM-DD). Required if --use-chirps.
  --chirps-end CHIRPS_END
                        CHIRPS download end date (YYYY-MM-DD). Required if --use-chirps.
```

## Dependencies

### Core

- `numpy`, `scipy`, `pandas`
- `matplotlib`, `Pillow`
- `folium` (for interactive maps)

### Optional (for real data)

- `rasterio` (for geospatial I/O; required for CHIRPS)
- `elevation` (for SRTM DEM fetching)

Install all:

```bash
pip install -r requirements.txt
pip install elevation
```

## Geospatial Details

### Jamaica Bounds

- **Default**: (south=17.7, west=-78.4, north=18.5, east=-76.1)
- **Elevation Range**: 0–1200 m (synthetic); actual Jamaica: 0–2256 m (Blue Mountain Peak)

### Rainfall Patterns

Spatial rainfall generation uses:

1. **Orographic Effect**: More rain on eastern (windward) slopes and higher elevations (trade wind assumption)
2. **Convective Hotspots**: Random Gaussian-distributed cells (tropical convection)

Default event: ~150 mm over 6 hours (heavy tropical downpour)

### Flood Model

Simplified cellular automata using:

- **D8 flow direction** (each cell flows to steepest neighbor)
- **Manning's equation** (velocity ∝ slope^0.5 × depth^0.67)
- **Infiltration/evaporation** (simplified: 1% per timestep)

**Note**: This is pedagogical. For planning-grade flood mapping, use HEC-RAS 2D, LISFLOOD-FP, or similar hydrodynamic models.

## Testing

Run smoke tests (8 quick validation tests):

```bash
python -m pytest src/test_flood_simulation.py -v
```

Tests cover:

- Synthetic DEM generation
- Spatial rainfall field generation
- Flow direction and accumulation
- Slope calculation
- Rainfall-runoff simulation (uniform and spatial)
- Flood zone identification

## Example Outputs

### Static Visualization

- **Left panel**: Digital Elevation Model (terrain)
- **Middle panel**: Water depth after simulation (meters)
- **Right panel**: Flood risk zones (areas > 5 cm depth)

### Interactive Map

- Folium-based map centered on Jamaica
- Water depth overlay (blue colormap: lighter = shallower, darker = deeper)
- Marker showing max water depth in mm

## Known Limitations

1. **Simplified Physics**: Cellular automata is fast but crude; real flood models solve shallow-water equations.
2. **No Coastline Masking**: Water can flow off the edge of the domain; add a coastline shapefile for realism.
3. **CHIRPS Availability**: Daily files may not exist for all dates; the script logs and skips missing data.
4. **SRTM Processing**: First-time downloads can take minutes; results are cached.
5. **No Drainage Networks**: Rivers and streams are implicit in the flow-direction calculation; they are not explicit objects.

## Future Enhancements

- [ ] Integrate real coastline/watershed shapefiles
- [ ] Add land-use/land-cover infiltration rates
- [ ] Support HEC-RAS 2D model export
- [ ] Parallel computation for larger DEMs
- [ ] Time-series rainfall input
- [ ] Validation against observed flood extents

## ERA5 (Hourly) — Optional: higher-resolution precipitation

You can fetch ERA5-Land hourly total precipitation (useful for time-series simulations) via the
Copernicus Climate Data Store (CDS) API. The project includes `src/era5_helper.py` which provides:

- `download_era5_hourly(bounds, start_date, end_date, out_path)` — downloads a NetCDF of ERA5-Land
  `total_precipitation` for the selected bbox and dates (requires `cdsapi` and a configured `~/.cdsapirc`).
- `regrid_era5_to_grid(nc_path, target_shape, target_bounds)` — interpolates the NetCDF to your model
  grid (rows, cols) and returns a numpy array of shape `(time, rows, cols)` in millimeters.

Quick setup:

1. Install dependencies:

```bash
pip install cdsapi xarray netCDF4
```

2. Configure CDS API credentials: create `~/.cdsapirc` with your CDS key following instructions at
   https://cds.climate.copernicus.eu/api-how-to

3. Example usage (Python):

```python
from era5_helper import download_era5_hourly, regrid_era5_to_grid

bounds = (17.7, -78.4, 18.5, -76.1)  # south, west, north, east
nc = download_era5_hourly(bounds, '2024-10-01', '2024-10-03', out_path='data/era5.nc')
hourly = regrid_era5_to_grid(nc, target_shape=(200,200), target_bounds=bounds)
# hourly.shape -> (time, rows, cols) in mm

# pass to the simulator (time-series)
from flood_simulation import FloodSimulator
sim = FloodSimulator({}, None)
dem = sim.generate_synthetic_dem(size=(200,200))
water = sim.simulate_time_series_runoff(dem, hourly, duration_per_step_hours=1.0)
```

Notes:

- If CDS credentials are not configured the helper warns and the code falls back to CHIRPS or synthetic rainfall.
- ERA5 `total_precipitation` units are meters in the NetCDF; the helper converts to millimeters for the simulator.


## References

```text
- CHIRPS: https://www.chc.ucsb.edu/data/chirps
- SRTM: https://www2.jpl.nasa.gov/srtm/
- Manning's Equation: https://en.wikipedia.org/wiki/Manning_formula
- D8 Algorithm: https://en.wikipedia.org/wiki/Digital_elevation_model#Algorithms
```
