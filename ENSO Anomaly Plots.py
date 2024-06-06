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

all_models = [MPI_ESM_SST, IPSL_CM5_Temp, IPSL_CM6_Temp]


chunk_years= 500
climatology_years = 50

threshold_34 = 0.5
threshold_12 = 0.5

months_threshold_34 = 6 #12 to compare to extended ENSOs
months_threshold_12 = 6 


coords_34 = [190, 240]
coords_12 = [270, 280]

#%%
# Define a function to plot statistics
def plot_enso_anomaly_stats(output, title):
    fig, ax1 = plt.subplots()

    # Plot the first sub-element on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Model Year')
    ax1.set_ylabel('Count', color=color)
    ax1.plot([x[0] for x in output], [x[1][0] for x in output], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the second sub-element
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean Anomaly', color=color)
    ax2.plot([x[0] for x in output], [x[1][1] for x in output], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Create a third y-axis for the third sub-element
    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Mean Length', color=color)
    ax3.plot([x[0] for x in output], [x[1][2] for x in output], color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(title)
    plot_file_name = title + '.png'
    #plt.savefig(plot_path + plot_file_name)
    plt.show()

#%%

for model_to_use in all_models:

    sub_path = model_to_use['sub_path']
    file = model_to_use['file']
    variable_name = model_to_use['variable_name']
    model_end_year = model_to_use['model_end_year']

    # Convert the time values to dates

    dataset = xarray.open_dataset(path + sub_path + file)

    dataset, periods = convert_dates(dataset, model_end_year)

    data_hist = dataset[variable_name]

    #Convert lon coords to -180 -to +180 if required
    if data_hist.lon.min() < 0:
        coords_34 = [coord - 360 for coord in coords_34]
        coords_12 = [coord - 360 for coord in coords_12]

    #Loop through time in chunks 
    max_time = data_hist.time.shape[0]
    chunk_size = chunk_years * 12
    output_34 = []
    output_12 = []

    window_size = climatology_years * 12
    
    for i in range(0,max_time,chunk_size):
        print(f"Slicing {i}")
        data_slice = data_hist[i:i+chunk_size,:,:]
        
        #NEED TO DOUBLE CHECK THE ACTUAL DEFINITION OF THE ENSO REGIONS
        nino_34 = data_slice.sel(lat=slice(5, -5), lon=slice(coords_34[0], coords_34[1]))  
        nino_12 = data_slice.sel(lat=slice(0, -10), lon=slice(coords_12[0], coords_12[1]))

        climatology_34 = nino_34.groupby('time.month').apply(
            lambda x: x.rolling(time=window_size, center=True, min_periods=1).mean())

        climatology_12 = nino_12.groupby('time.month').apply(
            lambda x: x.rolling(time=window_size, center=True, min_periods=1).mean())

        anomalies_34 = nino_34 - climatology_34
        anomalies_12 = nino_12 - climatology_12

        anomalies_mean_34 = anomalies_34.mean(['lat', 'lon'])
        anomalies_mean_34.plot()
        plt.title('ENSO 3.4 SST Anomalies')
        plt.ylabel('Temperature Anomaly (°C)')
        plt.show()

        anomalies_mean_12 = anomalies_12.mean(['lat', 'lon'])
        anomalies_mean_12.plot()
        plt.title('ENSO 1+2 SST Anomalies')
        plt.ylabel('Temperature Anomaly (°C)')
        plt.show()

        count_34, mask_34 = find_enso_events(anomalies_mean_34.dropna('time'), threshold_34, months_threshold_34)
        count_12, mask_12 = find_enso_events(anomalies_mean_12.dropna('time'), threshold_12, months_threshold_12)

        output_34.append((int(i/12),count_34))
        output_12.append((int(i/12),count_12))




    plot_enso_anomaly_stats(output_34, sub_path.replace('/','') + ' ENSO 3.4 Anomaly Stats')
    plot_enso_anomaly_stats(output_12, sub_path.replace('/','') + ' ENSO 1+2 Anomaly Stats')




# %%
