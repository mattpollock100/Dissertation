#%%
print('Importing Libraries')
import matplotlib.pyplot
import xarray
import numpy
import cartopy
import matplotlib
import matplotlib.pyplot as plt


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
#Define Files
path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
sub_path = '/IPSL_CM5/'
variables = ['psl', 'tas', 'ts']

file_names = ['_Amon_TR5AS-Vlr01_200001_299912.nc', 
              '_Amon_TR5AS-Vlr01_300001_399912.nc',
              '_Amon_TR5AS-Vlr01_400001_499912.nc',
              '_Amon_TR5AS-Vlr01_500001_599912.nc',
              '_Amon_TR5AS-Vlr01_600001_699912.nc',
              '_Amon_TR5AS-Vlr01_700001_799912.nc']

# %%
for var in variables:
    file_paths = [path + sub_path + var + file for file in file_names]

    dss = [xarray.open_dataset(file_path) for file_path in file_paths]

    ds_combined = xarray.concat(dss, dim = 'time')


    ds_combined.to_netcdf(path + sub_path + var + '_Amon_TR5AS_combined.nc')
    # %%
