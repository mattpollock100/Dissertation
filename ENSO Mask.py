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
                    [IPSL_CM6_Temp, IPSL_CM6_Temp], [IPSL_CM6_Temp, IPSL_CM6_Precip]]


#[MPI_ESM_SST, MPI_ESM_Precip], [MPI_ESM_SST, MPI_ESM_Temp], 
#[IPSL_CM5_Temp, IPSL_CM5_Temp], [IPSL_CM5_Temp, IPSL_CM5_Precip]


#DELETE THIS JUST FOR TESTING
all_model_pairs = [[TRACE_TS, TRACE_Temp], [TRACE_TS, TRACE_Precip]]

enso_type = '34'

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

        #CAreful here, setting to true will create new masks that take a long time
        if False:
            enso_data_slice = enso_data_hist[i:i+chunk_size,:,:]
            
            nino_34 = enso_data_slice.sel(lat=slice(-5, 5), lon=slice(coords_34[0], coords_34[1]))  
            nino_12 = enso_data_slice.sel(lat=slice(-10, 0), lon=slice(coords_12[0], coords_12[1]))

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
        
            mask_file_name_34 = mask_path + sub_path.replace("/","") + "_" + str(i) + '_ENSO34_Mask.npy'
            mask_file_name_12 = mask_path + sub_path.replace("/","") + "_" + str(i) + '_ENSO12_Mask.npy'
            np.save(mask_file_name_34, mask_34)
            np.save(mask_file_name_12, mask_12)
            
        mask_file_name = mask_path + sub_path.replace("/","") + "_" + str(i) + '_ENSO' + enso_type + '_Mask.npy'
        
        mask_34 = np.load(mask_file_name)

        data_slice = data_hist[i:i+chunk_size,:,:]
        
        
        region_slice_mean = data_slice.weighted(region_weights).mean(("lat","lon"))  * conversion_factor
        average = region_slice_mean.mean().values.item()
        average_34 = region_slice_mean.where(mask_34).dropna(dim='time').mean().values.item()
        average_not_34 = region_slice_mean.where(1 - mask_34).dropna(dim='time').mean().values.item()

        #get the month that ENSO most commonly starts in
        
        int_arr = mask_34.astype(int)
        # Calculate the difference between subsequent elements
        diff_arr = np.diff(int_arr, prepend=0)
        # Convert back to boolean array
        start_month_34 = (diff_arr == 1)

        masked_data = region_slice_mean.where(start_month_34)
        dropped_na = masked_data.dropna(dim='time')
        month = dropped_na.time.dt.month

        #get the most common month
        month_array = month.values
        month_strings = [month_names[i-1] for i in month_array]

        sns.countplot(month_strings, order=month_names)
        plt.xlabel('Count')
        plt.ylabel('Month')
        plt.title('Histogram of Months ' + str(i) + sub_path.replace('/',''))
        plt.show()

        #create a running average 
        #N.B. Remember if a season is selected there will be less than 12m in a year 
        region_slice_rolling = region_slice_mean.rolling(time= roll_avg_years * months_in_year, center = True).mean()
        #title = f"{season} Average temperature: {average}, Variance of temperature: {variance}"
        #line_plot_precip(region_slice_rolling, title)
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

    plot_file_name = sub_path.replace("/","") + "_" + str(region_number) + "_" + variable_name + "_" + "ENSO_" + enso_type + ".png"

    #plt.savefig(plot_path + plot_file_name)
    plt.show()
    plt.close('all')

    #Save the output
    model_name = sub_path.replace("/","")
    
    np.save(output_path + model_name + "_" + enso_type + variable_name + "_years_ENSO_Impact.npy", x)
    np.save(output_path + model_name + "_" + enso_type + variable_name + "_total_ENSO_Impact.npy", y_1)
    np.save(output_path + model_name + "_" + enso_type + variable_name + "_ENSO_ENSO_Impact.npy", y_2)
    np.save(output_path + model_name + "_" + enso_type + variable_name + "_not_ENSO_ENSO_Impact.npy", y_3)


    


# %%

