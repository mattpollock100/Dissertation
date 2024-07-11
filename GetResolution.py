#%%

import netCDF4 as nc
import numpy as np
import ModelParams


from ModelParams import *

from CommonFunctions import convert_dates

#%%
path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
plot_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Plots/'

for model_to_use in all_models:
    
    sub_path = model_to_use['sub_path']
    file = model_to_use['file']
    variable_name = model_to_use['variable_name']
    model_end_year = model_to_use['model_end_year']
    conversion_factor = model_to_use['conversion_factor']

    # Open the NetCDF file
    file_path = path + sub_path + file
    dataset = nc.Dataset(file_path, 'r')

    # Extract latitude and longitude
    lat = dataset.variables['lat'][:]
    lon = dataset.variables['lon'][:]

    # Calculate resolution
    lat_resolution = np.abs(lat[45] - lat[44])
    lon_resolution = np.abs(lon[45] - lon[44])

    print(sub_path + file)
    print(f'Latitude Resolution: {lat_resolution} degrees')
    print(f'Longitude Resolution: {lon_resolution} degrees')
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    # Close the dataset
    dataset.close()
# %%

# Open the NetCDF file
file_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF/TRACE/TRACE_TAS.nc'
dataset = nc.Dataset(file_path, 'r')

# Extract latitude and longitude
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]

# Calculate resolution
lat_resolution = np.abs(lat[45] - lat[44])
lon_resolution = np.abs(lon[45] - lon[44])

print(sub_path + file)
print(f'Latitude Resolution: {lat_resolution} degrees')
print(f'Longitude Resolution: {lon_resolution} degrees')
print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
# %%
