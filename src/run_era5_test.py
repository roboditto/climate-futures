import os
import sys
from era5_helper import download_era5_hourly, regrid_era5_to_grid

bounds = (17.7, -78.4, 18.5, -76.1)
start = '2024-10-01'
end = '2024-10-02'
out = 'data/era5_test.nc'

os.makedirs('data', exist_ok=True)

print('Starting ERA5 download...')
nc = download_era5_hourly(bounds, start, end, out_path=out)
print('download returned:', nc)
if nc:
    try:
        print('Regridding to small grid (50x50)')
        arr = regrid_era5_to_grid(nc, target_shape=(50,50), target_bounds=bounds)
        if arr is not None:
            print('Regrid complete. shape =', arr.shape)
            import numpy as np
            np.save('data/era5_test_regrid.npy', arr)
            print('Saved regridded array to data/era5_test_regrid.npy')
        else:
            print('Regrid returned None')
    except Exception as e:
        print('Regrid failed:', e)
else:
    print('No netCDF downloaded; skipping regrid')
