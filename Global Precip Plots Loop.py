

#%%
print('Importing Libraries')
import matplotlib.pyplot
import xarray
import numpy
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

#%%
print('Opening File')

path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
#sub_path ='/IPSL_CM6/'
#file = 'TR6AV-Sr02_20000101_79991231_1M_precip.nc'
sub_path = '/MPI_ESM/'
file = 'pr_Amon_MPI_ESM_TRSF_slo0043_100101_885012.nc'

filename = path + sub_path + file


use_weights = True
conversion_factor = 86400 #for precipitation data in kg m-2 s-1 to mm/day


dataset = xarray.open_dataset(filename,decode_times=False)



#%%
print('Initial Data Tweaks')
time_var = dataset.time
# Convert the time values to dates
dates = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
# Convert the dates to a format that xarray can understand
dates_xarray = [cftime.DatetimeNoLeap(date.year - 6000, date.month, date.day) for date in dates]
# Update the time variable in the dataset
dataset['time'] = dates_xarray

#Change this variable to the variable you want to plot
data_hist = dataset.precip 


#%%
#Function to make maps
print('define function to plot maps')
def map_precip(data):
    matplotlib.pyplot.figure(figsize=(10,7))
    proj=cartopy.crs.Robinson(central_longitude=-85)
    ax = matplotlib.pyplot.subplot(111, projection=proj)
    #ax.set_extent([-180, 180, -180, 180], crs=cartopy.crs.PlateCarree())

    # do the plot
    data.plot.pcolormesh(ax=ax, transform=cartopy.crs.PlateCarree(), cmap = 'BrBG', levels=numpy.linspace(0,15,41), cbar_kwargs={'label':'Precipitation $(mm day^{-1})$'})
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS)

# %%
#Loop through time in chunks of 500 yrs
max_time = data_hist.time.shape[0]
chunk_size = 500 * 12
for i in range(0,max_time,chunk_size):
    data_slice = data_hist[i:i+chunk_size,:,:]
    data_slice_mean = data_slice.where(data_slice.time.dt.season == "JJA").dropna(dim='time').mean(dim='time') * conversion_factor
    map_precip(data_slice_mean)

# %%
