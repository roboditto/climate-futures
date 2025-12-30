import streamlit as st
import os
from flood_simulation import FloodSimulator
from utils import load_config, setup_logging

st.title('Flood Simulation Dashboard (Light)')

config = load_config()
logger = setup_logging(config)

st.sidebar.header('Simulation controls')
use_srtm = st.sidebar.checkbox('Use SRTM DEM')
use_chirps = st.sidebar.checkbox('Use CHIRPS rainfall')
use_numba = st.sidebar.checkbox('Use Numba (if available)')
coastline = st.sidebar.text_input('Coastline shapefile (optional)')
output_dir = st.sidebar.text_input('Output directory', value='results')
country = st.sidebar.text_input('Country', value='Jamaica')
bbox = st.sidebar.text_input('BBox override (south,west,north,east)', value='')

if st.sidebar.button('Run simulation'):
    os.makedirs(output_dir, exist_ok=True)
    # Respect toggles in config
    config.setdefault('flood_simulation', {})['use_numba'] = use_numba
    sim = FloodSimulator(config, logger)

    with st.spinner('Preparing DEM and rainfall...'):
        # Use same country defaults as the CLI main helper
        def _get_defaults():
            if bbox:
                parts = [float(x) for x in bbox.split(',')] if bbox else None
                bbox_override = bbox if parts and len(parts) == 4 else None
            else:
                bbox_override = None
            # mirror the CLI helper from flood_simulation
            # Try to import the module-level helper (now exported by flood_simulation)
            try:
                from flood_simulation import get_country_defaults as _gcd  # type: ignore
                return _gcd(country, bbox_override)
            except Exception:
                # Fallback simple defaults
                return {
                    'bounds': (17.7, -78.4, 18.5, -76.1),
                    'dem_size': (200, 200),
                    'elevation_range': (0, 1200),
                    'rainfall_mm': 150.0,
                    'duration_hours': 6.0
                }

        defaults = _get_defaults()
        dem = sim.generate_synthetic_dem(size=defaults['dem_size'], elevation_range=defaults['elevation_range'])
        rainfall = sim.generate_spatial_rainfall(dem, total_mm=defaults['rainfall_mm'], pattern='both', n_hotspots=5)

    with st.spinner('Running simulation...'):
        water = sim.simulate_rainfall_runoff(dem, duration_hours=defaults['duration_hours'], rainfall_field=rainfall)
        if coastline:
            water = sim.apply_coastline_mask(water, coastline, bounds=defaults['bounds'])

    st.success('Simulation complete')
    img_path = os.path.join(output_dir, 'flood_simulation.png')
    sim.plot_simulation(dem, water, save_path=img_path)
    st.image(img_path, caption='Flood simulation results', use_column_width=True)

    # If folium map exists, offer link/embed
    map_path = os.path.join(output_dir, 'flood_map.html')
    sim.export_map(dem, water, save_html=map_path, bounds=defaults['bounds'])
    if os.path.exists(map_path):
        st.markdown(f"[Open interactive map]({map_path})")
