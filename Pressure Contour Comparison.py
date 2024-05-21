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

from scipy.interpolate import griddata

#%%
print('Setup Variables')

from ModelParams import *

model_to_use = MPI_ESM_PSL

sub_path = model_to_use['sub_path']
file = model_to_use['file']
variable_name = model_to_use['variable_name']
conversion_factor = model_to_use['conversion_factor']
y_min = model_to_use['y_min']
y_max = model_to_use['y_max']
convert_dates = model_to_use['convert_dates']
model_end_year = model_to_use['model_end_year']

region_number = 9

chunk_years = 500

seasons = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']

path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
plot_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Plots/'

filename = path + sub_path + file

dataset = xarray.open_dataset(filename,decode_times=False)

# Start date
date_size = dataset.time.shape[0]
start_year = (model_end_year - (date_size / 12))

# Number of periods
periods = date_size

# Generate the dates
dates_xarray = [cftime.DatetimeProlepticGregorian(year=start_year + i // 12, month=i % 12 + 1, day=1) for i in range(periods)]

dataset['time'] = dates_xarray

#select region of interest - need to convert longitude to 0-360 if needed - add that code.
data_hist = dataset[variable_name].sel(lat=slice(10, -30), lon=slice(275, 330))


season = 'Annual'
months_in_year = 3
if season == 'Annual': months_in_year = 12

chunk_size = chunk_years * 12
output = []


#Change to start at chunk_size ?? Or think about how to do the baseline
data_slice_start = data_hist[0:6000,:,:]
data_slice_end = data_hist[data_hist.shape[0] - 6000:data_hist.shape[0],:,:]

def get_smooth(data_slice):
    if season == 'Annual':
        region_slice_mean = data_slice.mean('time')  * conversion_factor
    else:    
        region_slice_mean = data_slice.where(data_slice.time.dt.season == season).dropna(dim='time').mean('time')  * conversion_factor
    
    #anomalies = region_slice_mean - baseline_mean

    #map_plot(region_slice_mean,'cividis', str(i/12) + ' ' + season)

    #smooth region_slice_mean using inverse distance weighting
    # Get the coordinates of the data points
    lon, lat = np.meshgrid(region_slice_mean.lon, region_slice_mean.lat)

    # Flatten the data and the coordinates
    lon_flat = lon.flatten()
    lat_flat = lat.flatten()
    data_flat = region_slice_mean.values.flatten()

    # Create a grid for the interpolated data
    lon_grid, lat_grid = np.mgrid[lon.min():lon.max():100j, lat.min():lat.max():100j]

    # Interpolate the data
    region_slice_mean_smooth = griddata((lon_flat, lat_flat), data_flat, (lon_grid, lat_grid), method='cubic').T

    # Create new latitude and longitude arrays with finer granularity
    lat_fine = np.linspace(region_slice_mean.lat.min(), region_slice_mean.lat.max(), region_slice_mean_smooth.shape[0])
    lon_fine = np.linspace(region_slice_mean.lon.min(), region_slice_mean.lon.max(), region_slice_mean_smooth.shape[1])
    
    # Create a DataArray from the NumPy array with the new latitude and longitude arrays
    region_slice_mean_smooth_da = xarray.DataArray(
        region_slice_mean_smooth,
        coords=[('lat', lat_fine), ('lon', lon_fine)],
        name='Pressure'
    )

    # Convert the DataArray to a numpy array and get the indices of the max and min values
    max_pressure_idx = np.unravel_index(region_slice_mean_smooth_da.values.argmax(), region_slice_mean_smooth_da.shape)
    min_pressure_idx = np.unravel_index(region_slice_mean_smooth_da.values.argmin(), region_slice_mean_smooth_da.shape)

    max_lat = region_slice_mean_smooth_da.lat[max_pressure_idx[0]].values
    max_lon = region_slice_mean_smooth_da.lon[max_pressure_idx[1]].values - 360.0

    min_lat = region_slice_mean_smooth_da.lat[min_pressure_idx[0]].values
    min_lon = region_slice_mean_smooth_da.lon[min_pressure_idx[1]].values - 360.0

    coords_string = f'Max: {max_lat:.2f}°, {max_lon:.2f}°  Min: {min_lat:.2f}°, {min_lon:.2f}°'
    title = sub_path.replace('/','') + ' ' + season + ' Smoothed'
    
    return region_slice_mean_smooth_da, title, coords_string

smoothed_data_start, title_start, coords_string_start = get_smooth(data_slice_start)
smoothed_data_end, title_end, coords_string_end = get_smooth(data_slice_end)

matplotlib.pyplot.figure(figsize=(10,7))
proj=cartopy.crs.Robinson(central_longitude=-85)
ax = matplotlib.pyplot.subplot(111, projection=proj)
ax.set_extent([-85, -30, -30, 10], crs=cartopy.crs.PlateCarree())

# do the plot
#img = data.plot.pcolormesh(ax=ax, transform=cartopy.crs.PlateCarree(), cmap = 'cividis', cbar_kwargs={'label':'Sea Level Pressure $(Pa)$'})

#levels=numpy.linspace(0,15,41), 

gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,
                    linewidth=1, color='gray', alpha=0.5, linestyle='-',
                    xlocs=range(-180, 181, 10), ylocs=range(-90, 91, 10))

gl.top_labels = False
gl.right_labels = False

ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)

contour_levels = np.arange(100000, 103000, 200)


contours_start = ax.contour(smoothed_data_start.lon, smoothed_data_start.lat, smoothed_data_start, colors='black', levels=contour_levels, transform=cartopy.crs.PlateCarree())
#ax.clabel(contours_start, inline=True, fontsize=8)

contours_end = ax.contour(smoothed_data_end.lon, smoothed_data_end.lat, smoothed_data_end, colors='red', levels=contour_levels, transform=cartopy.crs.PlateCarree())
ax.clabel(contours_end, inline=True, fontsize=8)

#plt.title(title)
#plt.suptitle(subtitle)

#plt.savefig(plot_path + title.replace(' ','_') + '.png')
plt.show()
