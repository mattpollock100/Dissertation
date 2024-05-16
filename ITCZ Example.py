#Look at plotting zonal averages of precipitation across a range of latitudes.
#This will give you a sense of the ITCZ position over time.
# Plot them for different time slices on the same chart to see how the ITCZ moves over time.
#Given it's a zonal average, you can also calculate the latitude of maximum precipitation for each time step.
#But the max will be at a discrete point that may not represent the exact ITCZ position.
#Instead is it possible to transform the data to a smooth curve and then find the latitude of maximum precipitation 
#using interpolation or curve fitting techniques, or by calculating the center of mass of the curve.
#or fourier transform


#%%
print('Importing Libraries')
import matplotlib.pyplot
import xarray
import numpy
import cartopy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm



# Add a couple of deep down individual functions.
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.ticker import AutoMinorLocator

# used to create area averages over AR6 regions.
import regionmask

import netCDF4 as nc
from netCDF4 import num2date

import cftime

#%%
from ModelParams import *

model = IPSL_CM5_Precip #IPSL_CM5_Precip, MPI_ESM_Precip, IPSL_CM6_Precip

#%%
print('Opening File')

path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
plot_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Plots/'


sub_path = model['sub_path']
file = model['file']
variable_name = model['variable_name']
conversion_factor = model['conversion_factor']
model_end_year = model['model_end_year']

filename = path + sub_path + file

model_name = sub_path.replace('/', '')

initial_lon_coords = [-90, -80] 
#Nino1+2: 90W-80W
#Peru (roughly): 80W-70W

lon_string = str(abs(initial_lon_coords[0])) + 'W' + '-' + str(abs(initial_lon_coords[1])) + 'W'

lat_coords = [15, -15]
lat_string = str(lat_coords[0]) + 'N' + '-' + str(abs(lat_coords[1])) + 'S'

chunk_years= 500

seasons = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']

cmap = cm.get_cmap('Blues')

#%%
dataset = xarray.open_dataset(filename,decode_times=False)

date_size = dataset.time.shape[0]
start_year = (model_end_year - (date_size / 12))
periods = date_size
dates_xarray = [cftime.DatetimeProlepticGregorian(year=start_year + i // 12, month=i % 12 + 1, day=1) for i in range(periods)]
dataset['time'] = dates_xarray

#focus on the region around the equator relevant for the ITCZ
data_hist = dataset[variable_name].sel(lat=slice(lat_coords[0], lat_coords[1]))

#convert longitude to 0-360 if needed
lon_conversion = 360
if min(data_hist[0,:,:].lon) < 0:
    lon_conversion = 0
lon_coords = [x + lon_conversion for x in initial_lon_coords]

#Loop through time in chunks 
max_time = data_hist.time.shape[0]
chunk_size = chunk_years * 12

# Focus on a specific longitudinal range
region_slice = data_hist.sel(lon=slice(lon_coords[0], lon_coords[1]))  # Modify as needed for your dataset

chunk_size = chunk_years * 12 

for season in seasons:
    fig, ax = plt.subplots()

    for i in range(0,periods,chunk_size):
        print(f"Slicing {i}")
        data_slice = region_slice[i:i+chunk_size,:,:]
        
        
        if season == 'Annual':
            region_slice_mean = data_slice.mean('lon').mean('time')  * conversion_factor
        else:    
            region_slice_mean = data_slice.where(data_slice.time.dt.season == season).dropna(dim='time').mean('lon').mean('time')  * conversion_factor
        

        # Get the latitudes and values of region_slice_mean
        latitudes = region_slice_mean.lat.values
        values = region_slice_mean.values

        # Calculate the center of mass
        centre_of_mass = numpy.average(latitudes, weights=values)
        centre_of_mass = round(centre_of_mass, 2)
        year = int(i/12 + chunk_years / 2)
        
        color = cmap(i / periods)
        ax.plot(latitudes, values, label=f"Time: {year} CofM: {round(centre_of_mass, 2)}", color=color)

    title = f"{model_name} {season} {lat_string} {lon_string}"
    ax.set_title(title)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Precipitation')
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center')
    

    plot_file_name = title + '.png'
    plt.savefig(plot_path + plot_file_name, bbox_inches='tight')
    plt.show()
    plt.close()

# %%
