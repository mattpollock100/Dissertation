

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
sub_path ='/TRACE/'
file = 'TRACE_TAS.nc'
#sub_path = '/MPI_ESM/'
#file = 'pr_Amon_MPI_ESM_TRSF_slo0043_100101_885012.nc'

filename = path + sub_path + file


use_weights = True
conversion_factor = 1 #for precipitation data in kg m-2 s-1 to mm/day


dataset = xarray.open_dataset(filename,decode_times=False)



#%%
print('Initial Data Tweaks')
#time_var = dataset.time
# Convert the time values to dates
#dates = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
#dates = num2date((time_var[:]) * 3153.6, units='seconds since 2000-01-01 00:00:00', calendar=time_var.calendar)

# Convert the dates to a format that xarray can understand
#dates_xarray = [cftime.DatetimeNoLeap(date.year - 6000, date.month, date.day) for date in dates]
# Update the time variable in the dataset
#dataset['time'] = dates_xarray


#%%
#Change this variable to the variable you want to plot
data_hist = dataset.TREFHT
#data_hist = dataset.precip 
#data_hist = dataset.pr

#Select time slice
data_hist = data_hist[0:6000,:,:]

weights = data_hist[0,:,:]
weights = numpy.cos(numpy.deg2rad(dataset.lat))
weights.name = "weights"

#Weight the data by grid size if required
if use_weights:

    data_hist_wgtd = data_hist.weighted(weights)
else:
    data_hist_wgtd = data_hist

#%%
#Calculate the global mean
global_mean_data = data_hist_wgtd.mean(("lat","lon"))

global_mean_data = global_mean_data * conversion_factor

fig, ax = plt.subplots(figsize=(15,4))

global_mean_data.plot(ax=ax, label = 'rainfall', color = 'cornflowerblue')

plt.show()

#%%

#Plot just for the NWS region

####NEED TO REFINE THIS CODE AS IT TAKES weights AND THEN APPLIES MASK
#SO WEIGHTS COS LAT WEIGHTS ARE ALREADY APPLIED

regionmask.defined_regions.ar6.all.plot(text_kws=dict(color="#67000d", fontsize=7, bbox=dict(pad=0.2, color="w")))
regionmask.defined_regions.ar6.all

mask = regionmask.defined_regions.ar6.all.mask(data_hist_wgtd)
NWS_weights=weights.where(mask == 9 ,0)

NWS_hist = data_hist_wgtd.weighted(NWS_weights).mean(("lat","lon")) 
NWS_hist = NWS_hist * conversion_factor

fig, ax = plt.subplots(figsize=(15,4))
NWS_hist.plot(ax=ax, label = 'NWS', color = 'cornflowerblue')
plt.show

#%%
#Look at seasonality

NWS_hist_DJF = NWS_hist.where(NWS_hist.time.dt.season == "DJF")
NWS_hist_MAM = NWS_hist.where(NWS_hist.time.dt.season == "MAM")
NWS_hist_JJA = NWS_hist.where(NWS_hist.time.dt.season == "JJA")
NWS_hist_SON = NWS_hist.where(NWS_hist.time.dt.season == "SON")

fig, ax = plt.subplots(figsize=(15,4))
NWS_hist_DJF.plot(ax=ax, label = 'NWS DJF', color = 'cornflowerblue')
NWS_hist_MAM.plot(ax=ax, label = 'NWS MAM', color = 'darkgreen')
NWS_hist_JJA.plot(ax=ax, label = 'NWS JJA', color = 'darkorange')
NWS_hist_SON.plot(ax=ax, label = 'NWS SON', color = 'darkred')
plt.show


#%%
#Plot the data on a map
NWS_hist_regional = data_hist_wgtd.where(NWS_weights)

NWS_hist_regional_DJF = NWS_hist_regional.where(NWS_hist_regional.time.dt.season == "DJF").mean("time") * conversion_factor
NWS_hist_regional_MAM = NWS_hist_regional.where(NWS_hist_regional.time.dt.season == "MAM").mean("time") * conversion_factor
NWS_hist_regional_JJA = NWS_hist_regional.where(NWS_hist_regional.time.dt.season == "JJA").mean("time") * conversion_factor
NWS_hist_regional_SON = NWS_hist_regional.where(NWS_hist_regional.time.dt.season == "SON").mean("time") * conversion_factor

NWS_hist_regional_DJF = NWS_hist_regional_DJF.rename('DJF')


matplotlib.pyplot.figure(figsize=(10,7))
proj=cartopy.crs.LambertConformal(central_longitude=-85)
ax = matplotlib.pyplot.subplot(111, projection=proj)
ax.set_extent([-90, -60, -20, 10], crs=cartopy.crs.PlateCarree())

# do the plot
NWS_hist_regional_SON.plot.pcolormesh(ax=ax, transform=cartopy.crs.PlateCarree(), cmap='BrBG')
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)

#%%
#Look at lat of max precipitation


precip_data = data_hist_wgtd.sel(lon=slice(-120, -50))  
precip_data_season = precip_data.where(precip_data.time.dt.season == "JJA").dropna(dim='time')

# Calculate the zonal mean precipitation
zonal_mean_precip = precip_data_season.mean(dim='lon') 
zonal_mean_precip = zonal_mean_precip.rolling(time=30, center=True).mean()
# Find the latitude of maximum precipitation for each time step
max_precip_lat = zonal_mean_precip.idxmax(dim='lat')

# Plot the latitude of maximum precipitation over time
max_precip_lat.plot()
plt.title('ITCZ Position (Latitude) Over Time')
plt.xlabel('Time')
plt.ylabel('Latitude')
plt.show()






# %%
