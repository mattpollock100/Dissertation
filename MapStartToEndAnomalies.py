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

from ModelParams import *

#%%
#Function to make maps
print('define function to plot maps')
def map_plot(data, cmap='BrBG', title = 'Title', label = 'Label', vmin=None, vmax=None):
    matplotlib.pyplot.figure(figsize=(10,7))
    proj=cartopy.crs.Robinson(central_longitude=-85)
    ax = matplotlib.pyplot.subplot(111, projection=proj)
    ax.set_extent([-85, -70, -15, 13], crs=cartopy.crs.PlateCarree())

    # do the plot
    data.plot.pcolormesh(ax=ax, transform=cartopy.crs.PlateCarree(), cmap = cmap, cbar_kwargs={'label':label}, vmin=vmin, vmax=vmax)
    
    #levels=numpy.linspace(0,15,41), 

    gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='-',
                      xlocs=range(-180, 181, 10), ylocs=range(-90, 91, 10))

    gl.top_labels = False
    gl.right_labels = False

    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS)
    ax.title.set_text(title)
    plt.show()
#%%
print('Setup Variables')

models = [IPSL_CM5_Precip, IPSL_CM6_Precip, MPI_ESM_Precip, TRACE_Precip]
for model_to_use in models:


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

    time_frame = 6000

    seasons = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']
    seasons = ['DJF']
    path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
    plot_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Plots/'

    filename = path + sub_path + file

    model_name = sub_path.replace('/','')




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
    data_hist_all = dataset[variable_name]

    #if data_hist_all.lon.min() < 0:
    #    data_hist = data_hist_all.sel(lat=slice(10, -20), lon=slice(-90, -30))
    #else:
    #    data_hist = data_hist_all.sel(lat=slice(10, -20), lon=slice(270, 330))


    weights = data_hist_all[0,:,:]
    weights = numpy.cos(numpy.deg2rad(dataset.lat * 0)) # set all weights to 1.0
    weights.name = "weights"

    regionmask.defined_regions.ar6.all.plot(text_kws=dict(color="#67000d", fontsize=7, bbox=dict(pad=0.2, color="w")))
    regionmask.defined_regions.ar6.all

    mask = regionmask.defined_regions.ar6.all.mask(data_hist_all)
    region_weights=weights.where(mask == region_number ,0)

    data_hist = data_hist_all



    for season in seasons:
        months_in_year = 3
        if season == 'Annual': months_in_year = 12

        chunk_size = chunk_years * 12
        output = []

        #Create Baseline
        start_point = periods - time_frame * 12
        baseline = data_hist[start_point:start_point + chunk_size,:,:]

        if season == 'Annual':
            baseline_mean = baseline.weighted(region_weights).mean('time') * conversion_factor
        else:
            baseline_mean = baseline.where(baseline.time.dt.season == season).dropna(dim='time').weighted(region_weights).mean('time') * conversion_factor

        map_plot(baseline_mean,'Blues', model_name + ' Baseline (' + season + ')', 'Precipitation (mm day$^{-1}$)')

        #Change to start at chunk_size ?? Or think about how to do the baseline
    

        data_slice = data_hist[periods - chunk_size:periods,:,:]
        
        
        if season == 'Annual':
            region_slice_mean = data_slice.mean('time')  * conversion_factor
        else:    
            region_slice_mean = data_slice.where(data_slice.time.dt.season == season).dropna(dim='time').mean('time')  * conversion_factor
        
        anomalies = region_slice_mean - baseline_mean

        map_plot(anomalies,'BrBG', model_name + ' Changes (' + season + ')', 'Precipitation Change (mm day$^{-1}$)', -2.5, 2.5)



        # %%
