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

import cftime

#%%
#Open file

from ModelParams import *

from CommonFunctions import find_enso_events, convert_dates


###This code looks at ENSO anomalies with the climatology removed.
###It can be used both to produce maps for each time slice of the ENSO anomaly


path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
plot_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Plots/'
mask_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/ENSO Masks/'

all_model_pairs =  [[MPI_ESM_SST, MPI_ESM_Precip], [MPI_ESM_SST, MPI_ESM_Temp], 
                    [IPSL_CM5_Temp, IPSL_CM5_Temp], [IPSL_CM5_Temp, IPSL_CM5_Precip],
                    [IPSL_CM6_Temp, IPSL_CM6_Temp], [IPSL_CM6_Temp, IPSL_CM6_Precip]]


#DELETE THIS JUST FOR TESTING
all_model_pairs = [[MPI_ESM_SST, MPI_ESM_Temp]]

region_number = 9

chunk_years= 500
climatology_years = 500   #####Note this is taking the climatology of the WHOLE slice

threshold_34 = 0.5
threshold_12 = 0.5

months_threshold_34 = 6 #12 to compare to extended ENSOs
months_threshold_12 = 6 


coords_34 = [190, 240]
coords_12 = [270, 280]

roll_avg_years = 1

lat_slice = [5, -35]
lon_slice = [270, 300]


#%%
#Function to make maps
print('define function to plot maps')
def map_plot(data, cmap='BrBG', title = 'Title'):
    matplotlib.pyplot.figure(figsize=(10,7))
    proj=cartopy.crs.Robinson(central_longitude=-85)
    ax = matplotlib.pyplot.subplot(111, projection=proj)
    ax.set_extent([-90, -60, -35, 5], crs=cartopy.crs.PlateCarree())

    # do the plot
    data.plot.pcolormesh(ax=ax, transform=cartopy.crs.PlateCarree(), cmap = cmap, cbar_kwargs={'label':'Temp Anomaly $(K)$'})
    
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

    # Convert the time values to dates
    enso_dataset = xarray.open_dataset(path + sub_path + file)
    enso_dataset, periods = convert_dates(enso_dataset, model_end_year)
    enso_data_hist = enso_dataset[variable_name]

    #Convert lon coords to -180 -to +180 if required
    if enso_data_hist.lon.min() < 0:
        coords_34 = [coord - 360 for coord in coords_34]
        coords_12 = [coord - 360 for coord in coords_12]
        lon_slice = [coord - 360 for coord in lon_slice]

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

    weights = data_hist[0,:,:]
    weights = np.cos(np.deg2rad(dataset.lat))
    weights.name = "weights"

    #regionmask.defined_regions.ar6.all.plot(text_kws=dict(color="#67000d", fontsize=7, bbox=dict(pad=0.2, color="w")))
    #regionmask.defined_regions.ar6.all

    mask = regionmask.defined_regions.ar6.all.mask(data_hist)
    region_weights=weights.where(mask == region_number ,0)

    months_in_year = 12
    
    for i in range(0,max_time,chunk_size):
        print(f"Slicing {i}")

        mask_file_name = mask_path + sub_path.replace("/","") + "_" + str(i) + '_ENSO34_Mask.npy'
        mask_enso = np.load(mask_file_name)
        mask_enso_3D = mask_enso[:, None, None]

        #select a broad slice of the region to reduce computational load
        data_slice_raw = data_hist[i:i+chunk_size,:,:].sel(lat=slice(lat_slice[0], lat_slice[1]), lon=slice(lon_slice[0], lon_slice[1])) 
        
        climatology = data_slice_raw.groupby('time.month').apply(
            lambda x: x.rolling(time=window_size, center=True, min_periods=1).mean())
       
        data_slice = data_slice_raw - climatology

        anomalies_enso = data_slice.where(mask_enso_3D).dropna(dim='time').mean('time')

        #map_plot(data = anomalies_enso, cmap = 'bwr', title = str(i) + ' ENSO Anomalies')

        
        region_slice_mean = data_slice.weighted(region_weights).mean(("lat","lon"))  * conversion_factor
        average = region_slice_mean.mean().values.item()
        average_34 = region_slice_mean.where(mask_enso).dropna(dim='time').mean().values.item()
        average_not_34 = region_slice_mean.where(1 - mask_enso).dropna(dim='time').mean().values.item()

        #create a running average 
        #N.B. Remember if a season is selected there will be less than 12m in a year 
        region_slice_rolling = region_slice_mean.rolling(time= roll_avg_years * months_in_year, center = True).mean()
        output.append((int(i/12),round(average,2), round(average_34,2) ,round(average_not_34,2)))
        



    #Plot statistics
    x = [(t[0] + chunk_years / 2 ) for t in output]
    y_1 = [t[1] for t in output]
    y_2 = [t[2] for t in output]
    y_3 = [t[3] for t in output]


    fig, ax1 = plt.subplots()


    # Plot the first variable on the first y-axis
    ax1.plot(x, y_1, 'b-', label = 'Total')
    ax1.plot(x, y_2, 'r-', label = 'ENSO')
    ax1.plot(x, y_3, 'g-', label = 'Not ENSO')

    ax1.set_xlabel('Model Years')
    ax1.set_ylabel(variable_name)

    #add legend to the plot
    ax1.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center')

    plot_file_name = sub_path.replace("/","") + "_" + str(region_number) + "_" + variable_name + "_" + "ENSO_34_climatology.png"

    plt.savefig(plot_path + plot_file_name)
    plt.show()
    plt.close('all')



# %%
