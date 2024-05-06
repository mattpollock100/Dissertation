import xarray as xr
import matplotlib.pyplot as plt

# Load the NetCDF data
ds = xr.open_dataset('path_to_your_file.nc')

# Select SST and slice the ENSO 3.4 region
enso34_sst = ds['SST'].sel(lat=slice(-5, 5), lon=slice(190, 240))  # Adjust for longitude as needed

# Define the size of the rolling window, e.g., 5 years (12 months * 5)
window_size = 12 * 5

# Calculate the rolling climatology (mean for each calendar month over a running window)
# 'center=True' ensures the window is centered on the month being averaged
climatology = enso34_sst.rolling(time=window_size, center=True).mean()

# Calculate anomalies by subtracting the rolling climatology from the full dataset
anomalies = enso34_sst - climatology

# Plot the time series of the area-averaged anomalies
anomalies_mean = anomalies.mean(['lat', 'lon'])
anomalies_mean.plot()
plt.title('ENSO 3.4 SST Anomalies with Running Climatology')
plt.ylabel('Temperature Anomaly (°C)')
plt.show()