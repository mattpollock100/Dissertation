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
print('Setup Variables')



IPSL_CM6_Precip = {'sub_path' : '/IPSL_CM6/', 
                   'file' : 'TR6AV-Sr02_20000101_79991231_1M_precip.nc', 
                   'variable_name' : 'precip',
                   'conversion_factor' : 86400,
                   'y_min' : 0,
                   'y_max' : 10,
                   'convert_dates' : 1}

IPSL_CM6_Temp = {'sub_path' : '/IPSL_CM6/', 
                 'file' : 'TR6AV-Sr02_20000101_79991231_1M_t2m.nc', 
                 'variable_name' : 'tas',
                 'conversion_factor' : 1,
                 'y_min' : 295,
                 'y_max' : 300,
                 'convert_dates' : 1}

MPI_ESM_Precip = {'sub_path' : '/MPI_ESM/',
                  'file' : 'pr_Amon_MPI_ESM_TRSF_slo0043_100101_885012.nc',
                  'variable_name' : 'pr',
                  'conversion_factor' : 86400,
                  'y_min' : 0,
                  'y_max' : 10,
                  'convert_dates' : 2}

all_models = {'IPSL_CM6_Precip' : IPSL_CM6_Precip,
              'IPSL_CM6_Temp': IPSL_CM6_Temp,
              'MPI_ESM_Precip' : MPI_ESM_Precip}

model_to_use = 'MPI_ESM_Precip'

sub_path = all_models[model_to_use]['sub_path']
file = all_models[model_to_use]['file']
variable_name = all_models[model_to_use]['variable_name']
conversion_factor = all_models[model_to_use]['conversion_factor']
y_min = all_models[model_to_use]['y_min']
y_max = all_models[model_to_use]['y_max']
convert_dates = all_models[model_to_use]['convert_dates']

chunk_years = 500
region_number = 9
roll_avg_years = 1

season = "DJF"

#IPSL ends in 1990
#
#%%
print('Opening File')

path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'

filename = path + sub_path + file

dataset = xarray.open_dataset(filename,decode_times=False)

#%%
print('Initial Data Tweaks')
if convert_dates == 1:
    time_var = dataset.time
    # Convert the time values to dates
    dates = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    #dates = num2date((time_var[:]) * 3153.6, units='seconds since 2000-01-01 00:00:00', calendar=time_var.calendar)

    # Convert the dates to a format that xarray can understand
    dates_xarray = [cftime.DatetimeNoLeap(date.year - 6000, date.month, date.day) for date in dates]
    # Update the time variable in the dataset
    dataset['time'] = dates_xarray

if convert_dates == 2:
    # Start date
    date_size = dataset.time.shape[0]
    start_year = (1850 - (date_size / 12))

    # Number of periods
    periods = date_size

    # Generate the dates
    dates_xarray = [cftime.DatetimeProlepticGregorian(year=start_year + i // 12, month=i % 12 + 1, day=1) for i in range(periods)]

    dataset['time'] = dates_xarray

#%%
print('Create weights for region')

#Change this variable to the variable you want to plot
data_hist = dataset[variable_name]


weights = data_hist[0,:,:]
weights = numpy.cos(numpy.deg2rad(dataset.lat))
weights.name = "weights"

#regionmask.defined_regions.ar6.all.plot(text_kws=dict(color="#67000d", fontsize=7, bbox=dict(pad=0.2, color="w")))
#regionmask.defined_regions.ar6.all

mask = regionmask.defined_regions.ar6.all.mask(data_hist)
region_weights=weights.where(mask == region_number ,0)


#%%
print('Define function for linear plot')
def line_plot_precip(data, title):
    fig, ax = plt.subplots(figsize=(15,4))
    data.plot(ax=ax, label = 'Region', color = 'cornflowerblue')
    plt.title(title)
    plt.ylim(y_min, y_max)
    plt.show

#%%
print('Select time slices and plot')

#Loop through time in chunks 
max_time = data_hist.time.shape[0]
chunk_size = chunk_years * 12
output = []
for i in range(0,max_time,chunk_size):
    print(f"Slicing {i}")
    data_slice = data_hist[i:i+chunk_size,:,:]
    
    #region_slice_mean = region_slice.where(data_slice.time.dt.season == "JJA").dropna(dim='time').mean(dim='time') * conversion_factor
    region_slice_mean = data_slice.where(data_slice.time.dt.season == season).dropna(dim='time').weighted(region_weights).mean(("lat","lon"))  * conversion_factor
    average = region_slice_mean.mean().values.item()
    variance = region_slice_mean.var().values.item()
    
    #create a running average 
    #N.B. Remember if a season is selected there will be less than 12m in a year 
    region_slice_rolling = region_slice_mean.rolling(time= roll_avg_years * 3, center = True).mean()
    title = f"{season} Average temperature: {average}, Variance of temperature: {variance}"
    line_plot_precip(region_slice_rolling, title)
    output.append((int(i/12),round(average,2),round(variance,4)))

print(output)
# %%
