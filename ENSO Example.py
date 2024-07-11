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
path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
plot_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Plots/'
output_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Output Data/'

MPI_ESM_SST =   {'sub_path' : '/MPI_ESM/',
                'file' : 'sst_Amon_MPI_ESM_TRSF_slo0043_100101_885012.nc', 
                'variable_name' : 'sst',
                'coords_34' : [190, 240],
                'coords_12' : [270, 280],
                'model_end_year' : 1850}

IPSL_CM5_TAS =   {'sub_path' : '/IPSL_CM5/',
                'file' : 'tas_Amon_TR5AS_combined.nc', 
                'variable_name' : 'tas',
                'coords_34' : [-170, -40],
                'coords_12' : [-90, -80],
                'model_end_year' : 1990}

IPSL_CM6_TAS =   {'sub_path' : '/IPSL_CM6/',
                'file' : 'TR6AV-Sr02_20000101_79991231_1M_t2m.nc', 
                'variable_name' : 'tas',
                'coords_34' : [-170, -40],
                'coords_12' : [-90, -80],
                'model_end_year' : 1990}



model_to_use = IPSL_CM6_TAS

sub_path = model_to_use['sub_path']
file = model_to_use['file']
variable_name = model_to_use['variable_name']
coords_34 = model_to_use['coords_34']
coords_12 = model_to_use['coords_12']
model_end_year = model_to_use['model_end_year']


chunk_years= 500
climatology_years = 50

threshold_34 = 0.5
threshold_12 = 0.5

months_threshold_34 = 6
months_threshold_12 = 6

dataset = xarray.open_dataset(path + sub_path + file)

window_size = climatology_years * 12

#%%
# Convert the time values to dates
date_size = dataset.time.shape[0]
start_year = (model_end_year - (date_size / 12))
periods = date_size
dates_xarray = [cftime.DatetimeProlepticGregorian(year=start_year + i // 12, month=i % 12 + 1, day=1) for i in range(periods)]
dataset['time'] = dates_xarray


#%%
# Define a function to find the number of ENSO events in a dataset
def find_enso_events(data, threshold, months_threshold):
    # Create a boolean array where True indicates that the anomaly is above 0.5
    above_threshold = data > threshold

    # Label contiguous True regions
    labeled_array, num_features = ndimage.label(above_threshold)

    # Count the size of each labeled region
    region_sizes = np.bincount(labeled_array.ravel())

    # Count the number of regions that have a size of 6 or more
    num_large_regions = np.count_nonzero(region_sizes >= months_threshold)

    avg_anomalies = []
    lengths = []

    for i in range(1, num_features + 1):
        # Get the size of the region
        size = np.count_nonzero(labeled_array == i)

        # If the size is greater than or equal to months_threshold
        if size >= months_threshold:
            # Append the size to lengths
            lengths.append(size)

            # Calculate the average anomaly in the region and append it to avg_anomalies
            avg_anomaly = data[labeled_array == i].mean()
            avg_anomalies.append(avg_anomaly)

    return [num_large_regions, round(np.mean(avg_anomalies),3), round(np.mean(lengths),1)]
    
    
#%%

#STILL NEED TO WEIGHT IT??

#NEED TO CHANGE THE VARIABLE NAME FOR EACH DATA SET
data_hist = dataset[variable_name]

#Loop through time in chunks 
max_time = data_hist.time.shape[0]
chunk_size = chunk_years * 12
output_34 = []
output_12 = []

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
    #anomalies_mean_34.plot()
    #plt.title('ENSO 3.4 SST Anomalies')
    #plt.ylabel('Temperature Anomaly (°C)')
    #plt.show()

    anomalies_mean_12 = anomalies_12.mean(['lat', 'lon'])
    #anomalies_mean_12.plot()
    #plt.title('ENSO 1+2 SST Anomalies')
    #plt.ylabel('Temperature Anomaly (°C)')
    #plt.show()

    count_34 = find_enso_events(anomalies_mean_34.dropna('time'), threshold_34, months_threshold_34)
    count_12 = find_enso_events(anomalies_mean_12.dropna('time'), threshold_12, months_threshold_12)

    output_34.append((int(i/12),count_34))
    output_12.append((int(i/12),count_12))

#%%
# Create a figure and a set of subplots
def plot_anomaly_stats(output, title):
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

# %%

plot_anomaly_stats(output_34, sub_path.replace('/','') + ' ENSO 3.4 Anomaly Stats')
plot_anomaly_stats(output_12, sub_path.replace('/','') + ' ENSO 1+2 Anomaly Stats')




# %%

#save output variables to be loaded later, as .npy files


output_34_years = [x[0] for x in output_34]
output_34_count = [x[1][0] for x in output_34]
output_34_mean_anomaly = [x[1][1] for x in output_34]
output_34_mean_length = [x[1][2] for x in output_34]



np.save(output_path + sub_path.replace('/','') + '_enso_34_years.npy', output_34_years)
np.save(output_path + sub_path.replace('/','') + '_enso_34_count.npy', output_34_count)
np.save(output_path + sub_path.replace('/','') + '_enso_34_mean_anomaly.npy', output_34_mean_anomaly)
np.save(output_path + sub_path.replace('/','') + '_enso_34_mean_length.npy', output_34_mean_length)

output_12_years = [x[0] for x in output_12]
output_12_count = [x[1][0] for x in output_12]
output_12_mean_anomaly = [x[1][1] for x in output_12]
output_12_mean_length = [x[1][2] for x in output_12]

np.save(output_path + sub_path.replace('/','') + '_enso_12_years.npy', output_12_years)
np.save(output_path + sub_path.replace('/','') + '_enso_12_count.npy', output_12_count)
np.save(output_path + sub_path.replace('/','') + '_enso_12_mean_anomaly.npy', output_12_mean_anomaly)
np.save(output_path + sub_path.replace('/','') + '_enso_12_mean_length.npy', output_12_mean_length)



# %%
