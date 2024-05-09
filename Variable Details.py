#%%
path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/NetCDF'
sub_path ='/IPSL_CM5/'
file = 'pr_Amon_TR5AS-Vlr01_200001_299912.nc'
#sub_path = '/MPI_ESM/'
#file = 'pr_Amon_MPI_ESM_TRSF_slo0043_100101_885012.nc'

filename = path + sub_path + file
from netCDF4 import Dataset

dataset = Dataset(filename, mode='r')

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
# %%
