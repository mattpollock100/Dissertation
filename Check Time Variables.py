#%%
print('Importing Libraries')
import matplotlib.pyplot
import xarray
import numpy
import cartopy
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

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


#MPI ends in 1850


#%%
dataset = xarray.open_dataset(filename,decode_times=True)
#time_var_temp = dataset.time
#calendar = dataset.time.calendar
#time_var = (dataset.time - dataset.time[0]) * 3153.6
#convert time_var to integers
#time_var_int = numpy.round(time_var).astype(int)

#data_date = '2000-01-01'
#start_date = pd.to_datetime(data_date)
#dataset['time']  = start_date + pd.to_timedelta(time_var_int, unit='s')
# Convert the time values to dates
#dates = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
#dates = num2date((time_var[:]), units='seconds since 2000-01-01 00:00:00', calendar=calendar)

# Convert the dates to a format that xarray can understand
#dates_xarray = [cftime.DatetimeProlepticGregorian(date.year - 7850, date.month, min(28, date.day)) for date in dates]
# Update the time variable in the dataset

#dates_xarray = dates

# Start date
start_year = (2000 - 7850)

# Number of periods
periods = 7850 * 12

# Generate the dates
dates_xarray = [cftime.DatetimeProlepticGregorian(year=start_year + i // 12, month=i % 12 + 1, day=1) for i in range(periods)]

dataset['time'] = dates_xarray

data_hist = dataset['pr']

weights = data_hist[0,:,:]
weights = numpy.cos(numpy.deg2rad(dataset.lat))
weights.name = "weights"

#%%
data_slice = data_hist[0:6000,:,:]

data_slice_weighted = data_slice.weighted(weights).mean(("lat","lon"))
data_slice_weighted_seasonal = data_slice_weighted.where(data_slice_weighted.time.dt.season == 'DJF').dropna(dim='time')
data_slice_weighted_seasonal.plot()
# %%
for date in dates:
    print([date.year, date.month, date.day])

#%%
for date in dates_xarray:
    print([date.year, date.month, date.day])

# %%
#length of dates_xarray
len(dates_xarray)

#%%

# Start date
start_year = 2000

# Number of periods
periods = 7850 * 12

# Generate the dates
dates_force = [cftime.DatetimeProlepticGregorian(year=start_year + i // 12, month=i % 12 + 1, day=1) for i in range(periods)]
# %%
