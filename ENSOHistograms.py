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

all_model_pairs =  [[MPI_ESM_SST, MPI_ESM_Temp], 
                    [IPSL_CM5_Temp, IPSL_CM5_Temp],
                    [IPSL_CM6_Temp, IPSL_CM6_Temp],
                    [TRACE_TS, TRACE_Temp]]


#DELETE THIS JUST FOR TESTING
all_model_pairs = [[MPI_ESM_SST, MPI_ESM_Temp]]

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


i_dic = {'IPSL_CM5': [0, 60000],
         'MPI_ESM': [24000, 84000],
         'IPSL_CM6': [0, 60000],
         'TRACE': [24000, 90000]}



#%%
for model_pair in all_model_pairs:

    MAM_pct_output = []
    DJF_pct_output = []
    SON_pct_output = []
    JJA_pct_output = []
    year_output = []

    enso_model_to_use = model_pair[0]
    sub_path = enso_model_to_use['sub_path']
    file = enso_model_to_use['file']
    variable_name = enso_model_to_use['variable_name']
    model_end_year = enso_model_to_use['model_end_year']

    model_name = sub_path.replace("/","")
    i_range = i_dic[model_name]

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

    for i in i_range:
        print(f"Slicing {i}")
        period = 'Mid-Holocene'
        if i > 30000: period = 'Pre-Industrial'

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
        plt.title(model_name + ' ' + period + ' Nino' + enso_type + ' Start Month')
        plt.show()

        MAM_count = np.sum(np.isin(month_array, [3, 4, 5]))
        MAM_pct = MAM_count / len(month_array)

        DJF_count = np.sum(np.isin(month_array, [12, 1, 2]))
        DJF_pct = DJF_count / len(month_array)

        JJA_count = np.sum(np.isin(month_array, [6, 7, 8]))
        JJA_pct = JJA_count / len(month_array)

        SON_count = np.sum(np.isin(month_array, [9, 10, 11]))
        SON_pct = SON_count / len(month_array)

        MAM_pct_output.append(MAM_pct)
        DJF_pct_output.append(DJF_pct)
        JJA_pct_output.append(JJA_pct)
        SON_pct_output.append(SON_pct)
        year_output.append(int(i/12))

    

# %%
