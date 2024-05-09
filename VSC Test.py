
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

regionmask.defined_regions.ar6.all

path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
sub_path ='/IPSL_CM6/'
file = 'TR6AV-Sr02_20000101_79991231_1M_precip.nc'

filename = path + sub_path + file
#dataset = nc.Dataset(path + sub_path + file, "r")
#print(dataset)
dataset = xarray.open_dataset(filename,decode_times=False)

precip_hist = dataset.precip

weights = precip_hist[0,:,:]
weights = numpy.cos(numpy.deg2rad(dataset.lat))
weights.name = "weights"

precip_hist_wgtd = precip_hist.weighted(weights)

global_mean_precip = precip_hist_wgtd.mean(("lat","lon"))

fig, ax = matplotlib.pyplot.subplots(figsize=(15,4))

global_mean_precip.plot(ax=ax, label = 'rainfall', color = 'cornflowerblue')

print("got there in the end")
"""
precip = dataset.variables["precip"]

weights= precip[0,:,:] #using tas to copy over metadata  #select the 1st timestep
weights = numpy.cos(numpy.deg2rad(dataset.variables["lat"]))
#weights.name = "weights"
    
#Make some field with the weighting applied
precip_wgtd=precip.weighted(weights)


"""
"""
# Iterate through all variables
for var in dataset.variables:
   
    variable = dataset.variables[var]

    
    # Print common details of the variable
    print(f"Variable: {var}")
    print(f"Dimensions: {variable.dimensions}")
    print(f"Shape: {variable.shape}")
    print(f"Data type: {variable.dtype}")
    
    
    # Iterate and print all attributes of the variable
    for attr_name in variable.ncattrs():
        attr_value = variable.getncattr(attr_name)
        print(f"{attr_name}: {attr_value}")
    
    #New line 
    print("\n")
    
    
# Close the dataset
dataset.close()
"""


