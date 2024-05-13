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
sub_path = '/MPI_ESM/'
file ='sst_Amon_MPI_ESM_TRSF_slo0043_100101_885012.nc'

dataset = xarray.open_dataset(path + sub_path + file)

chunk_years= 500
climatology_years = 50

threshold_34 = 0.5
threshold_12 = 0.5

months_threshold_34 = 6
months_threshold_12 = 6

model_end_year = 1850

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

    # Initialize lists to store average anomalies and lengths
    avg_anomalies = []
    lengths = []

    # Loop over each region
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

    return [num_large_regions, avg_anomalies.mean(), lengths.mean()]
    #print(f"The anomaly stays above 0.5 for 6 periods or more {num_large_regions} times.")

#%%

#PROBABLY STILL NEED TO WEIGHT IT

# Select the relevant enso regions
data_hist = dataset['sst']

#Loop through time in chunks 
max_time = data_hist.time.shape[0]
chunk_size = chunk_years * 12
output = []
for i in range(0,max_time,chunk_size):
    print(f"Slicing {i}")
    data_slice = data_hist[i:i+chunk_size,:,:]
    #data_slice = data_hist[0:6000,:,:]

    #NEED TO DOUBLE CHECK THE ACTUAL DEFINITION OF THE ENSO REGIONS
    #nino_34 = data_slice.sel(lat=slice(-5, 5), lon=slice(190, 240))  
    nino_34 = data_slice.sel(lat=slice(5, -5), lon=slice(190, 240))  
    nino_12 = data_slice.sel(lat=slice(0, -10), lon=slice(270, 280))

    # Define the size of the rolling window, e.g., 5 years (12 months * 5)
    window_size = climatology_years * 12

    # Calculate the rolling climatology (mean for each calendar month over a running window)
    # 'center=True' ensures the window is centered on the month being averaged
    climatology_34 = nino_34.rolling(time=window_size, center=True).mean()
    climatology_12 = nino_12.rolling(time=window_size, center=True).mean()

    # Calculate anomalies by subtracting the rolling climatology from the full dataset
    anomalies_34 = nino_34 - climatology_34
    anomalies_12 = nino_12 - climatology_12

    # Plot the time series of the area-averaged anomalies
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

    count_34 = find_enso_events(anomalies_mean_34.dropna('time'), threshold_34, months_threshold_34)
    count_12 = find_enso_events(anomalies_mean_12.dropna('time'), threshold_12, months_threshold_12)

    output.append((int(i/12),count_34,count_12))



