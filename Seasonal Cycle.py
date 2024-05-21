#%%
print('Importing Libraries')
import matplotlib.pyplot
import xarray
import numpy as np
import cartopy
import matplotlib
import matplotlib.pyplot as plt


# Add a couple of deep down individual functions.
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.ticker import AutoMinorLocator

# used to create area averages over AR6 regions.
import regionmask

import netCDF4 as nc
from netCDF4 import num2date

import cftime

import matplotlib.cm as cm

from scipy.interpolate import griddata

#%%
print('Setup Variables')

from ModelParams import *

model_to_use = IPSL_CM6_Temp

sub_path = model_to_use['sub_path']
file = model_to_use['file']
variable_name = model_to_use['variable_name']
conversion_factor = model_to_use['conversion_factor']
y_min = model_to_use['y_min']
y_max = model_to_use['y_max']
convert_dates = model_to_use['convert_dates']
model_end_year = model_to_use['model_end_year']

region_number = 9

cmap = cm.get_cmap('Reds')

chunk_years = 500


path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
plot_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Plots/'

filename = path + sub_path + file

#%%
dataset = xarray.open_dataset(filename,decode_times=False)

# Start date
date_size = dataset.time.shape[0]
start_year = (model_end_year - (date_size / 12))

# Number of periods
periods = date_size

# Generate the dates
dates_xarray = [cftime.DatetimeProlepticGregorian(year=start_year + i // 12, month=i % 12 + 1, day=1) for i in range(periods)]

dataset['time'] = dates_xarray


data_hist = dataset[variable_name]

weights = data_hist[0,:,:]
weights = np.cos(np.deg2rad(dataset.lat))
weights.name = "weights"

#regionmask.defined_regions.ar6.all.plot(text_kws=dict(color="#67000d", fontsize=7, bbox=dict(pad=0.2, color="w")))
#regionmask.defined_regions.ar6.all

mask = regionmask.defined_regions.ar6.all.mask(data_hist)
region_weights=weights.where(mask == region_number ,0)

#Loop through time in chunks 
max_time = data_hist.time.shape[0]
chunk_size = chunk_years * 12

output = []
output_var = []
output_year = []

for i in range(0,max_time,chunk_size):
    print(f"Slicing {i}")
    data_slice = data_hist[i:i+chunk_size,:,:]
    
    
    region_slice_mean = data_slice.weighted(region_weights).mean(("lat","lon"))  * conversion_factor
    average_seasonal_cycle = region_slice_mean.groupby('time.month').mean('time')
    year_variance = region_slice_mean.groupby('time.year').mean('time').var().values.item()

    output.append(average_seasonal_cycle)

    output_var.append(year_variance)

    output_year.append(int(250 + i / 12))

# %%
# Create a new figure
fig, ax = plt.subplots()


# Iterate over the list of arrays
for i, arr in enumerate(output):
    # Get a color from the colormap
    color = cmap((i+1) / len(output))
    max = round(arr.max().values.item(),2)
    min = round(arr.min().values.item(),2)

    # Plot the array with the color
    plt.plot(arr, label=f'Year: {250 + 500 * i} Max: {max} Min: {min} Range: {round(max - min,2)}', color=color)

# Add a legend
ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center')

ax.set_xticks(range(0, 12))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


# Show the plot
plt.show()
# %%
fig, ax = plt.subplots()

plt.plot(output_year,output_var,color='red')
plt.show()

# %%
