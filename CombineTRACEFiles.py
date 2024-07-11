#%%
import xarray as xr
import os
import pandas as pd
import cftime
from CommonFunctions import convert_dates
#%%
# Step 2: Specify the directory containing your NetCDF files
variable = 'TS'

main_directory = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF/TRACE/'
sub_directory = variable + '/'
temp_directory = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF/TRACE/temp7/'

directory = main_directory + sub_directory
netcdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nc')]

last_end_year = 1950
total_years_bp = 8000

#%%
# Ensure the temporary directory exists
if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)

#%%
preprocessed_files = []

start_year = last_end_year - total_years_bp
for file_path in netcdf_files:
    ds_raw = xr.open_dataset(file_path, use_cftime=True)

    hyphen_index = file_path.find('-')
    year_BP = file_path[hyphen_index+1:hyphen_index+6]
    #if year_BP == '1990C':
    #    end_year = last_end_year 
    #else:
    #    end_year = last_end_year - int(year_BP) - (2*i)
    end_year = start_year + (ds_raw.time.shape[0] / 12 )
    ds, periods = convert_dates(ds_raw, end_year)
    print(end_year)
    temp_file_path = os.path.join(temp_directory, os.path.basename(file_path))
    ds.to_netcdf(temp_file_path)
    preprocessed_files.append(temp_file_path)
    ds.close()
    ds_raw.close()
    start_year = end_year
#%%
# Step 4: Open and concatenate the preprocessed files
combined_ds = xr.open_mfdataset(preprocessed_files, combine='by_coords')

#%%
# Step 5: Save the combined dataset to a new NetCDF file
combined_ds.to_netcdf(main_directory + 'TRACE_' + variable + '.nc')
combined_ds.close()

# %%
