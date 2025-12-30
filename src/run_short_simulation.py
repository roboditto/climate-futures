import os
import numpy as np
from flood_simulation import FloodSimulator

os.makedirs('results', exist_ok=True)

regrid_path = 'data/era5_test_regrid.npy'
if not os.path.exists(regrid_path):
    print('Regridded ERA5 file not found:', regrid_path)
    raise SystemExit(1)

arr = np.load(regrid_path)
print('Loaded regridded array shape:', arr.shape)

# arr is (time, rows, cols)
T, rows, cols = arr.shape

config = {
    'flood_simulation': {
        'dem_resolution': 30,
        'manning_coefficient': 0.035,
        'timestep': 300,
        'max_iterations': 200,
        'use_numba': False
    }
}

sim = FloodSimulator(config)

dem = sim.generate_synthetic_dem(size=(rows, cols), elevation_range=(0, 500))
print('Generated synthetic DEM of shape', dem.shape)

# Run a short time-series simulation (use hourly slices as provided)
print('Running time-series simulation...')
accumulated = sim.simulate_time_series_runoff(dem, arr, duration_per_step_hours=1.0)
print('Simulation complete. accumulated shape:', accumulated.shape)

# Save outputs
np.save('results/era5_short_sim_accumulated.npy', accumulated)
print('Saved accumulated depth to results/era5_short_sim_accumulated.npy')

# Plot and save
sim.plot_simulation(dem, accumulated, save_path='results/era5_short_sim.png', country='ERA5 test')
print('Saved plot to results/era5_short_sim.png')
