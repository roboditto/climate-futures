from era5_helper import regrid_era5_to_grid

nc='data/era5_test.nc'
bounds=(17.7, -78.4, 18.5, -76.1)
print('Calling regrid on', nc)
arr = regrid_era5_to_grid(nc, target_shape=(50,50), target_bounds=bounds)
print('regrid result:', None if arr is None else ('shape', arr.shape))
if arr is not None:
    import numpy as np
    np.save('data/era5_test_regrid.npy', arr)
    print('Saved to data/era5_test_regrid.npy')
