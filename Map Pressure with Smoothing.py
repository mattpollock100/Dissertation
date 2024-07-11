#%%
print('Importing Libraries')
import matplotlib.pyplot
import xarray
import numpy as np
import cartopy
import matplotlib
import matplotlib.pyplot as plt

#libraries to calculate area of a contour
import cartopy.crs as ccrs
from shapely.geometry import Polygon
from shapely.ops import transform
from functools import partial
import pyproj

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

#Look at a model of precip
#model_to_use = IPSL_CM6_Precip

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

#only look at one for now
seasons = ['Annual']

path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
plot_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Plots/'

filename = path + sub_path + file

#%%
#Function to make maps
print('define function to plot maps')
def map_plot(data, cmap='BrBG', title = 'Title', subtitle = ''):
    matplotlib.pyplot.figure(figsize=(10,7))
    proj=cartopy.crs.Robinson(central_longitude=-85)
    ax = matplotlib.pyplot.subplot(111, projection=proj)
    # set extent for Peru
    # ax.set_extent([-85, -30, -30, 10], crs=cartopy.crs.PlateCarree())
    # set extent for the South Pacific Anticyclone
    ax.set_extent([-120, -70, -10, -50], crs=cartopy.crs.PlateCarree())
    # do the plot
    img = data.plot.pcolormesh(ax=ax, transform=cartopy.crs.PlateCarree(), cmap = cmap, cbar_kwargs={'label':'Sea Level Pressure $(Pa)$'})
    
    #levels=numpy.linspace(0,15,41), 

    gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='-',
                      xlocs=range(-180, 181, 10), ylocs=range(-90, 91, 10))

    gl.top_labels = False
    gl.right_labels = False

    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS)
    
    contour_levels = np.arange(100000, 103000, 200)
    #contour_levels = np.linspace(0, 20, 5)

    contours = ax.contour(data.lon, data.lat, data, colors='black', levels=contour_levels, transform=cartopy.crs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=8)

    plt.title(title)
    plt.suptitle(subtitle)
    
    #plt.savefig(plot_path + title.replace(' ','_') + '.png')
    plt.show()

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

#select region of interest - need to convert longitude to 0-360 if needed - add that code.
#data_hist = dataset[variable_name].sel(lat=slice(10, -30), lon=slice(275, 330))
#data_hist = dataset[variable_name].sel(lat=slice(10, -30), lon=slice(-85, -30))

#Region for the South Pacific AntiCyclone
data_hist = dataset[variable_name].sel(lat=slice(-10, -60), lon=slice(230, 300))

#%%

for season in seasons:
    months_in_year = 3
    if season == 'Annual': months_in_year = 12

    chunk_size = chunk_years * 12
    output = []
    """
    #Create Baseline
    baseline = data_hist[0:chunk_size,:,:]

    if season == 'Annual':
        baseline_mean = baseline.mean('time') * conversion_factor
    else:
        baseline_mean = baseline.where(baseline.time.dt.season == season).dropna(dim='time').mean('time') * conversion_factor

    map_plot(baseline_mean,'viridis', 'Baseline ' + season)
    """
    year_output = []
    area_output = []
    #Change to start at chunk_size ?? Or think about how to do the baseline
    for i in range(0,periods,chunk_size):
        print(f"Slicing {i}")
        data_slice = data_hist[i:i+chunk_size,:,:]
        
        
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
        title = sub_path.replace('/','') + ' ' + str(int(i/12)) + ' ' + season + ' Smoothed'
        map_plot(region_slice_mean_smooth_da,'cividis', title, coords_string)

        # Calculate the area within the 1022 hPa contour

        data = region_slice_mean_smooth_da

        # Step 1: Extract Contour Paths
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()})
        contour = ax.contour(data.lon, data.lat, data, levels=[102200], transform=ccrs.PlateCarree())
        paths = contour.collections[0].get_paths()
        plt.close(fig)
        # Step 2: Convert Paths to Shapely Polygons
        polygons = [Polygon(path.vertices) for path in paths if path.vertices.size > 0]

        # Step 3: Project Polygons to an Equal-Area CRS
        # Example projection: Albers Equal Area used here, you might want to choose one that suits your region
        proj = partial(
            pyproj.transform,
            pyproj.Proj(init='epsg:4326'),  # source coordinate system (lat/lon)
            pyproj.Proj(proj='aea', lat_1=-40, lat_2=-20)  # Albers Equal Area
        )

        projected_polygons = [transform(proj, polygon) for polygon in polygons]

        # Step 4: Calculate the Area
        areas = [polygon.area for polygon in projected_polygons]
        total_area = sum(areas) / 1000000  # Convert to square kilometers

        print(f"Total area within the 102200 Pa contour: {total_area} km2")

        year_output.append(i/12 + chunk_years / 2)
        area_output.append(total_area)
# %%

# Plotting
plt.plot(year_output, area_output)

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Area (km²)')
plt.title('Area within the 102200 Pa contour over time')

# Display the plot
plt.show()
# %%
