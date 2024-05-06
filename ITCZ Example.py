import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Load the NetCDF data
ds = xr.open_dataset('path_to_your_precipitation_file.nc')

# Assume the precipitation data variable is named 'precip'
precip_data = ds['precip']

# Focus on a specific longitudinal range if necessary
precip_data = precip_data.sel(lon=slice(-180, 180))  # Modify as needed for your dataset

# Calculate the zonal mean precipitation
zonal_mean_precip = precip_data.mean(dim='lon')

# Find the latitude of maximum precipitation for each time step
max_precip_lat = zonal_mean_precip.idxmax(dim='lat')

# Plot the latitude of maximum precipitation over time
max_precip_lat.plot()
plt.title('ITCZ Position (Latitude) Over Time')
plt.xlabel('Time')
plt.ylabel('Latitude')
plt.show()
