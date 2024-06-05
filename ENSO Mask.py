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


path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
plot_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Plots/'

all_model_pairs = [[MPI_ESM_SST, MPI_ESM_Precip]]

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



#%%
def line_plot_precip(data, title):
    fig, ax = plt.subplots(figsize=(15,4))
    data.plot(ax=ax, label = 'Region', color = 'cornflowerblue')
    plt.title(title)
    plt.ylim(y_min, y_max)
    plt.show

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

    #Loop through time in chunks 
    max_time = enso_data_hist.time.shape[0]
    chunk_size = chunk_years * 12
    output_34 = []
    output_12 = []
    output = []

    window_size = climatology_years * 12
    
    max_time = chunk_size * 2 #####Delete this just for testing

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
    weights = np.cos(np.deg2rad(dataset.lat))
    weights.name = "weights"

    #regionmask.defined_regions.ar6.all.plot(text_kws=dict(color="#67000d", fontsize=7, bbox=dict(pad=0.2, color="w")))
    #regionmask.defined_regions.ar6.all

    mask = regionmask.defined_regions.ar6.all.mask(data_hist)
    region_weights=weights.where(mask == region_number ,0)

    months_in_year = 12

    for i in range(0,max_time,chunk_size):
        print(f"Slicing {i}")
        enso_data_slice = enso_data_hist[i:i+chunk_size,:,:]
        
        nino_34 = enso_data_slice.sel(lat=slice(5, -5), lon=slice(coords_34[0], coords_34[1]))  
        nino_12 = enso_data_slice.sel(lat=slice(0, -10), lon=slice(coords_12[0], coords_12[1]))

        climatology_34 = nino_34.groupby('time.month').apply(
            lambda x: x.rolling(time=window_size, center=True, min_periods=1).mean())

        climatology_12 = nino_12.groupby('time.month').apply(
            lambda x: x.rolling(time=window_size, center=True, min_periods=1).mean())

        anomalies_34 = nino_34 - climatology_34
        anomalies_12 = nino_12 - climatology_12

        anomalies_mean_34 = anomalies_34.mean(['lat', 'lon'])

        anomalies_mean_12 = anomalies_12.mean(['lat', 'lon'])

        count_34, mask_34 = find_enso_events(anomalies_mean_34.dropna('time'), threshold_34, months_threshold_34)
        count_12, mask_12 = find_enso_events(anomalies_mean_12.dropna('time'), threshold_12, months_threshold_12)

        output_34.append((int(i/12),count_34))
        output_12.append((int(i/12),count_12))


        data_slice = data_hist[i:i+chunk_size,:,:]
            
        region_slice_mean = data_slice.weighted(region_weights).mean(("lat","lon"))  * conversion_factor
        average = region_slice_mean.mean().values.item()
        average_34 = region_slice_mean.where(mask_34).dropna(dim='time').mean().values.item()
        average_not_34 = region_slice_mean.where(1 - mask_34).dropna(dim='time').mean().values.item()

        #create a running average 
        #N.B. Remember if a season is selected there will be less than 12m in a year 
        region_slice_rolling = region_slice_mean.rolling(time= roll_avg_years * months_in_year, center = True).mean()
        #title = f"{season} Average temperature: {average}, Variance of temperature: {variance}"
        #line_plot_precip(region_slice_rolling, title)
        output.append((int(i/12),round(average,2), round(average_34,2) ,round(average_not_34,2)))


# %%
