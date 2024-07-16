#%%
print('Importing Libraries')
import matplotlib.pyplot
import xarray
import numpy as np
import cartopy
import matplotlib
import matplotlib.pyplot as plt

import scipy.ndimage as ndimage


# Add a couple of deep down individual functions.
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.ticker import AutoMinorLocator

# used to create area averages over AR6 regions.
import regionmask

import netCDF4 as nc
from netCDF4 import num2date
import seaborn as sns

import cftime

#%%
#Open file

from ModelParams import *

from CommonFunctions import find_enso_events, convert_dates


path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
plot_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Plots/'
mask_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/ENSO Masks/'

output_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Output Data/'

all_model_pairs =  [[MPI_ESM_SST, MPI_ESM_Precip], [MPI_ESM_SST, MPI_ESM_Temp], 
                    [IPSL_CM5_Temp, IPSL_CM5_Temp], [IPSL_CM5_Temp, IPSL_CM5_Precip],
                    [IPSL_CM6_Temp, IPSL_CM6_Temp], [IPSL_CM6_Temp, IPSL_CM6_Precip],
                    [TRACE_TS, TRACE_Temp], [TRACE_TS, TRACE_Precip]]


#[MPI_ESM_SST, MPI_ESM_Precip], [MPI_ESM_SST, MPI_ESM_Temp], 
#[IPSL_CM5_Temp, IPSL_CM5_Temp], [IPSL_CM5_Temp, IPSL_CM5_Precip]

temp_model_pairs = [[MPI_ESM_SST, MPI_ESM_Temp], 
                    [IPSL_CM5_Temp, IPSL_CM5_Temp], 
                    [IPSL_CM6_Temp, IPSL_CM6_Temp],
                    [TRACE_TS, TRACE_Temp]]

precip_model_pairs =  [[MPI_ESM_SST, MPI_ESM_Precip], 
                    [IPSL_CM5_Temp, IPSL_CM5_Precip],
                    [IPSL_CM6_Temp, IPSL_CM6_Precip],
                    [TRACE_TS, TRACE_Precip]]


all_model_pairs = temp_model_pairs

#For Mid-Holocene
#i_dic = {'MPI_ESM' : 24000,
#        'IPSL_CM5' : 0,
#        'IPSL_CM6' : 0,
#        'TRACE' : 24000}

#For Pre-Industrial
i_dic = {'MPI_ESM' : 90000,
        'IPSL_CM5' : 60000,
        'IPSL_CM6' : 60000,
        'TRACE' : 90000}

enso_type = '12'

region_number = 9

chunk_years= 500
climatology_years = 50

threshold_34 = 0.5
threshold_12 = 0.5

months_threshold_34 = 6 #12 to compare to extended ENSOs
months_threshold_12 = 6 


coords_34 = [190, 240]
coords_12 = [270, 280]

roll_avg_years = 1



month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


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
for model_pair in all_model_pairs:

    enso_model_to_use = model_pair[0]
    sub_path = enso_model_to_use['sub_path']
    file = enso_model_to_use['file']
    variable_name = enso_model_to_use['variable_name']
    model_end_year = enso_model_to_use['model_end_year']

    i  = i_dic[sub_path.replace("/","")]

    # Convert the time values to dates
    enso_dataset = xarray.open_dataset(path + sub_path + file)
    enso_dataset, periods = convert_dates(enso_dataset, model_end_year)
    enso_data_hist = enso_dataset[variable_name]

    #Convert lon coords to -180 -to +180 if required
    if enso_data_hist.lon.min() < 0:
        coords_34 = [coord - 360 for coord in coords_34]
        coords_12 = [coord - 360 for coord in coords_12]

    #Loop through time in chunks 
    max_time = enso_data_hist.time.shape[0]
    chunk_size = chunk_years * 12
    output_34 = []
    output_12 = []
    output = []

    window_size = climatology_years * 12
    
    model_to_use= model_pair[1]
    sub_path = model_to_use['sub_path']
    file = model_to_use['file']
    variable_name = model_to_use['variable_name']
    model_end_year = model_to_use['model_end_year']
    conversion_factor = model_to_use['conversion_factor']

    # Convert the time values to dates
    dataset = xarray.open_dataset(path + sub_path + file)
    dataset, periods = convert_dates(dataset, model_end_year)
    data_hist = dataset[variable_name]

    data_hist = dataset[variable_name]


    weights = data_hist[0,:,:]
    weights = np.cos(np.deg2rad(dataset.lat * 0)) #Set all weigths to one for a map
    weights.name = "weights"

    #regionmask.defined_regions.ar6.all.plot(text_kws=dict(color="#67000d", fontsize=7, bbox=dict(pad=0.2, color="w")))
    #regionmask.defined_regions.ar6.all

    mask = regionmask.defined_regions.ar6.all.mask(data_hist)
    region_weights=weights.where(mask == region_number ,0)

    months_in_year = 12



          
    mask_file_name = mask_path + sub_path.replace("/","") + "_" + str(i) + '_ENSO' + enso_type + '_Mask.npy'
    
    mask_34 = np.load(mask_file_name)

    data_slice = data_hist[i:i+chunk_size,:,:]
    
    mask_34_data_slice = data_slice[mask_34, :, :]
    
    region_slice_mean = data_slice.weighted(region_weights).mean('time')  * conversion_factor
    mask_34_region_slice_mean = mask_34_data_slice.weighted(region_weights).mean('time') * conversion_factor

    mask_34_anomalies = mask_34_region_slice_mean - region_slice_mean

    model_name = sub_path.replace("/","")

    time_frame = 'Pre-Industrial'
    if i < 30000:
        time_frame = 'Mid-Holocene'
    
    title = model_name + ' ' + time_frame + ' Niño' + enso_type + ' Temp Anomalies'
    label = 'Temp Anomalies (K)'
    cmap = 'coolwarm'
    vmax = 1.0
    vmin = -vmax

    if False:
        title = model_name + ' ' + time_frame + ' Niño' + enso_type + ' Precip Anomalies'
        label = 'Precipitation Anomalies mm (mm day$^{-1}$)'
        cmap = 'BrBG'
        vmax = 3.0
        vmin = -vmax

    map_plot(mask_34_anomalies, cmap, title, label, vmin, vmax)

        #average = region_slice_mean.mean().values.item()
        #average_34 = region_slice_mean.where(mask_34).dropna(dim='time').mean().values.item()
        #average_not_34 = region_slice_mean.where(1 - mask_34).dropna(dim='time').mean().values.item()

        

# %%

