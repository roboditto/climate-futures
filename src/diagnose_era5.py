import os
import traceback
p = 'data/era5_test.nc'
print('PATH ->', p)
print('exists', os.path.exists(p))
try:
    print('size', os.path.getsize(p))
except Exception as e:
    print('size err', e)

if os.path.exists(p):
    try:
        with open(p, 'rb') as fh:
            hdr = fh.read(256)
        print('\nFirst 256 bytes (hex):')
        print(hdr[:128].hex())
        print('\nFirst 256 bytes (repr):')
        print(repr(hdr[:256]))
    except Exception as e:
        print('Could not read header bytes:', e)

try:
    import xarray as xr
    print('xarray version', xr.__version__)
    try:
        print('xarray engines:', xr.backends.list_engines())
    except Exception as e:
        print('list_engines err', e)
except Exception as e:
    print('xarray import failed', e)

engines = ['netcdf4','h5netcdf','scipy','cfgrib']
for eng in engines:
    try:
        import xarray as xr
        print('\nTrying engine', eng)
        ds = xr.open_dataset(p, engine=eng)
        print('Opened with', eng, 'dims:', ds.dims)
        print('vars:', list(ds.data_vars))
        ds.close()
    except Exception as e:
        print('engine', eng, 'failed:', type(e).__name__)
        traceback.print_exc()

# Try default open
try:
    import xarray as xr
    print('\nTrying default engine')
    ds = xr.open_dataset(p)
    print('Opened default dims:', ds.dims)
    print('vars:', list(ds.data_vars))
    ds.close()
except Exception as e:
    print('default open failed:', type(e).__name__)
    traceback.print_exc()

# Try netCDF4 lower-level
try:
    from netCDF4 import Dataset
    print('\nTrying netCDF4.Dataset')
    ds = Dataset(p)
    print('netCDF4 variables:', list(ds.variables.keys()))
    for k in ds.variables:
        print(k, ds.variables[k].shape, getattr(ds.variables[k],'dtype', ''))
    ds.close()
except Exception as e:
    print('netCDF4 open failed:', type(e).__name__)
    traceback.print_exc()

print('\nDone')
