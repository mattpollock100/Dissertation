#%%
import xarray as xr
import numpy as np
import regionmask

#%%
file1_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF/TRACE/TRACE_PRECC.nc'
file2_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF/TRACE/TRACE_PRECL.nc'

ds1 = xr.open_dataset(file1_path)
ds2 = xr.open_dataset(file2_path)

#%%
summed_precip = ds1['PRECC'] + ds2['PRECL']


new_ds = summed_precip.to_dataset(name='PRECIP')


new_ds['PRECIP'].attrs = ds1['PRECC'].attrs

#%%
output_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF/TRACE/TRACE_PRECIP_Final.nc'
new_ds.to_netcdf(output_path)


ds1.close()
ds2.close()

"""
#%%
data_hist = ds1['PRECC']



weights = data_hist[0,:,:]
weights = np.cos(np.deg2rad(ds1.lat))
weights.name = "weights"

mask = regionmask.defined_regions.ar6.all.mask(data_hist)
region_weights=weights.where(mask == 9 ,0)

#%%
data_slice_1=ds1['PRECC'][0:6000,:,:]
mean_1= data_slice_1.weighted(region_weights).mean(("lat","lon")) * 86400000


#%%
data_slice_2 = ds2['PRECL'][0:6000,:,:]
mean_2 = data_slice_2.weighted(region_weights).mean(("lat","lon")) * 86400000

# %%
#convert metres per second to mm per day
conversion_factor = 1000 * 24 * 60 * 60
"""