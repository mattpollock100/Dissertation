#%%
print('Importing Libraries')
import matplotlib.pyplot
import xarray
import numpy as np
import cartopy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

from CommonFunctions import convert_dates

path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
plot_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Plots/'

mask_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/ENSO Masks/'

all_models = [MPI_ESM_SST, IPSL_CM5_Temp, IPSL_CM6_Temp, TRACE_TS]

all_models = [TRACE_TS]

chunk_years= 500

lat_slice = [5, -5]
initial_lon_slice = [170, 270]
average_over = 'lat'

#lat_slice = [0, -10]
#initial_lon_slice = [240, 260]
#average_over = 'lon'

cmap = cm.get_cmap('Reds')

season = 'JJA'

use_enso_mask = 'all' #enso, not_enso, all

print('-----------' + season + ' ' + use_enso_mask + '----------')
for model_to_use in all_models:
    
    sub_path = model_to_use['sub_path']
    file = model_to_use['file']
    variable_name = model_to_use['variable_name']
    model_end_year = model_to_use['model_end_year']
    conversion_factor = model_to_use['conversion_factor']

    if model_to_use == TRACE_TS:
        lat_slice = [lat_slice[1], lat_slice[0]]

    # Convert the time values to dates
    dataset = xarray.open_dataset(path + sub_path + file)
    dataset, periods = convert_dates(dataset, model_end_year)
    data_hist = dataset[variable_name]

    weights = data_hist[0,:,:]
    weights = np.cos(np.deg2rad(dataset.lat))
    weights.name = "weights"

    if data_hist.lon.min() < 0:
        lon_slice = [coord - 360 for coord in initial_lon_slice]
    else:
        lon_slice = initial_lon_slice

    #Loop through time in chunks 
    max_time = data_hist.time.shape[0]
    chunk_size = chunk_years * 12
    year_output = []
    gradient_output = []

    fig, ax = plt.subplots()

    for i in range(0,max_time,chunk_size):
        print(f"Slicing {i}")
        data_slice = data_hist[i:i+chunk_size,:,:]

        mask_file_name = mask_path + sub_path.replace("/","") + "_" + str(i) + '_ENSO34_Mask.npy'
        mask_enso = np.load(mask_file_name)
        mask_enso_3D = mask_enso[:, None, None]

        if use_enso_mask == 'enso':
            data_slice_enso_mask = data_slice.where(mask_enso_3D == 1)
        elif use_enso_mask == 'not_enso':
            data_slice_enso_mask = data_slice.where(mask_enso_3D == 0)
        else:
            data_slice_enso_mask = data_slice
        
        if season == 'Annual':
            region_slice = data_slice_enso_mask.sel(lat=slice(lat_slice[0], lat_slice[1]), 
                                        lon=slice(lon_slice[0], lon_slice[1]))
        else: 
            region_slice = data_slice_enso_mask.sel(lat=slice(lat_slice[0], lat_slice[1]), 
                                        lon=slice(lon_slice[0], lon_slice[1])).where(data_slice.time.dt.season == season).dropna(dim='time')  
        


        region_slice_weighted = region_slice.weighted(weights)

        region_slice_mean = region_slice_weighted.mean(average_over).mean('time') * conversion_factor
        
        if average_over == 'lat':
            coordinates = region_slice_mean.lon.values
        elif average_over == 'lon':
            coordinates = region_slice_mean.lat.values



        values = region_slice_mean.values

        year = int(i/12 + chunk_years / 2)

        color = cmap(i / periods)

        #Find the max value and its coordinate
        max_value = max(values)
        max_value_index = np.where(values == max_value)
        coordinate_of_max_value = coordinates[max_value_index][0]

        #Find the min value and its coordinate
        min_value = min(values)
        min_value_index = np.where(values == min_value)
        coordinate_of_min_value = coordinates[min_value_index][0]
        
        year_output.append(year)
        gradient_output.append(max_value - min_value)

        #round max and min values to 2dp
        max_value = round(max_value, 2)
        min_value = round(min_value, 2)
        gradient = round(max_value - min_value, 2)

        coordinate_of_max_value = round(coordinate_of_max_value, 2)
        coordinate_of_min_value = round(coordinate_of_min_value, 2)

        label = "Time " + str(year) + " Max: " + str(max_value) + " Min: " + str(min_value) + " Gradient: " + str(gradient) + " Max Coord: " + str(coordinate_of_max_value) + " Min Coord: " + str(coordinate_of_min_value)

        if coordinates[0] > 0:  
            coordinates = [coord - 360 for coord in coordinates]

        
        if False:
            print('skipping plot')
        else:
            if average_over == 'lat':
                ax.plot(coordinates, values, 
                        label=label,
                        color=color)
            elif average_over == 'lon':
                ax.plot(values, coordinates, 
                        label=label,
                        color=color)
            
    model_name = sub_path.replace("/","")
    title = model_name + ' ' + season + ' SST Gradient'
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('SST (K)')
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center')


    plot_file_name = title + '.png'
    plt.savefig(plot_path + plot_file_name, bbox_inches='tight')
    plt.show()
    plt.close()
    

#%%
year_output
# %%
gradient_output


# %%
