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


from scipy.stats import skew, kurtosis, probplot, shapiro
import statsmodels.api as sm
from scipy.stats import norm

from diptest import diptest

from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc

#%%


from ModelParams import *

from CommonFunctions import find_enso_events, convert_dates


#%%

mask_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/ENSO Masks/'
ENSO_type = 'ENSO34'
model = 'MPI_ESM'
max_steps = 90000
anomaly_range = 1.5

i = 12000
j = 18000

# %%
#Load variable from npy file

data1 = np.load(mask_path + model + '_' + str(i) + '_' + ENSO_type + '_Anomalies.npy')
data2 = np.load(mask_path + model + '_' + str(j) + '_' + ENSO_type + '_Anomalies.npy')
data = np.concatenate((data1, data2), axis=0)

data_clean = data

#%%
# Perform the Shapiro-Wilk test
stat, p = shapiro(data)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# Interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

# %%

# Calculate mean and standard deviation
mu, std = norm.fit(data_clean)

# Plot histogram
plt.hist(data_clean, bins=20, density=True, alpha=0.6, color='g')

# Plot Gaussian distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()

# %%
#See how far from a normal distribution it it with a Q-Q plot


# Calculate skewness and kurtosis
data_skewness = skew(data_clean)
data_kurtosis = kurtosis(data_clean, fisher=True)  # Fisher=True returns excess kurtosis

print(f"Skewness: {data_skewness}")
print(f"Kurtosis: {data_kurtosis}")

# Generate Q-Q plot
fig = plt.figure()
ax = fig.add_subplot(111)
probplot(data_clean, dist="norm", plot=ax)
plt.show()

# %%
#Hartigans Dip Test for multimodality

result = diptest(data_clean)
print(f"Dip Statistic: {result[0]}")
print(f"p-value: {result[1]}")


#%%
#scale by the values expected from a normal distribution to see where the differences are



# Step 1: Calculate the histogram of your data
count, bins = np.histogram(data_clean, bins=20, density=True)

# Step 2: Generate a normal distribution with the same mean and std as your data
mean, std = np.mean(data_clean), np.std(data_clean)
pdf = norm.pdf(bins, mean, std)

# Step 3: Calculate the expected count in each bin for the normal distribution
# The expected count is the PDF value times the total count times the bin width
bin_widths = np.diff(bins)
expected_count = pdf[:-1] * len(data_clean) * bin_widths

# Step 4: Divide the actual count by the expected count for each bin
# To match the histogram output, we need to normalize the actual counts first
actual_count_normalized = count * len(data_clean) * bin_widths
ratio = actual_count_normalized / expected_count

# Plotting the ratio
plt.bar(bins[:-1], ratio, width=bin_widths, alpha=0.7)
plt.xlabel('SST Anomaly')
plt.ylabel('Ratio (Actual / Expected)')
plt.title('Ratio of Actual to Expected Counts in Each Bin')
plt.show()
# %%
# Test for Mean Reversion



# Hurst Exponent
H, c, data_hurst = compute_Hc(data_clean, kind='change', simplified=True)
print(f"Hurst Exponent: {H}")


# Interpretation
if H < 0.5:
    print("The series is mean-reverting.")
elif H > 0.5:
    print("The series is trending.")
else:
    print("The series is a random walk.")

# %%

i = 12000
j = 18000

# %%
#Load variable from npy file
chunk_size = 6000
max_time = 90000
wing_hurst_output =[]
central_hurst_output = []
year_output = []
anomaly_range = 1.25
for i in range(0,max_time,chunk_size*2):
    j = i + chunk_size
    data1 = np.load(mask_path + model + '_' + str(i) + '_' + ENSO_type + '_Anomalies.npy')
    data2 = np.load(mask_path + model + '_' + str(j) + '_' + ENSO_type + '_Anomalies.npy')
    data = np.concatenate((data1, data2), axis=0)

    data_clean = data

    # Check stats for central range
    data_range = data[(data > -anomaly_range) & (data < anomaly_range)]
    data_range_clean = data_range[~np.isnan(data_range)]

    # Calculate skewness and kurtosis
    data_skewness = skew(data_range_clean)
    data_kurtosis = kurtosis(data_range_clean, fisher=True)  # Fisher=True returns excess kurtosis

    print('For centre of distribution:')
    print(f"Skewness: {data_skewness}")
    print(f"Kurtosis: {data_kurtosis}")

    H, c, data_hurst = compute_Hc(data_range_clean, kind='change', simplified=True)
    print(f"Hurst Exponent: {H}")
    central_hurst_output.append(H)

    data_range = data[(data > anomaly_range)]
    data_range_clean = data_range[~np.isnan(data_range)]

    # Calculate skewness and kurtosis
    data_skewness = skew(data_range_clean)
    data_kurtosis = kurtosis(data_range_clean, fisher=True)  # Fisher=True returns excess kurtosis

    print('For wings of distribution:')
    print(f"Skewness: {data_skewness}")
    print(f"Kurtosis: {data_kurtosis}")

    H, c, data_hurst = compute_Hc(data_range_clean, kind='change', simplified=True)
    print(f"Hurst Exponent: {H}")

    year_output.append(i/12 + 250)
    wing_hurst_output.append(H)
# %%
plt.plot(year_output, central_hurst_output)
plt.plot(year_output, wing_hurst_output)

# %%
