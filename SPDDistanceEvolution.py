#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import scipy.stats as stats
#%%
path = "C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/SPD/"
all_path = "C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/spd_result_all_peruvian_coast_30.csv"


distances = ['0_20', '20_40', '40_60', '60_80', '80_100', '100_99999']

step = 20
time_step = 500

cmap = cm.get_cmap('Reds')

#%%
#normalise the SPDs


spd_all = pd.read_csv(all_path)
spd_norm_all = spd_all["PrDens"] / spd_all["PrDens"].sum()
#calendar_all = [2000 - BP for BP in spd_all["calBP"]]
calendar_all = np.array([2000 - BP for BP in spd_all["calBP"]])

spd_sum = pd.Series(0, index=spd_all["PrDens"].index)
spd_sum_distance = pd.Series(0, index=spd_all["PrDens"].index)
c = 1
for distance in distances:
    integration_results = []
    year_results = []
    average_distance = int(distance.split("_")[0]) + step/2

    spd_path = path + "spd_result_" + distance + ".csv"
    spd = pd.read_csv(spd_path)
    
    spd_normalised = spd["PrDens"] / spd_norm_all
    
    spd_sum_distance = spd_sum_distance + spd_normalised * average_distance
    spd_sum = spd_sum + spd_normalised
    
    #integrate spd_normalised in 1000 year intervals
    for i in range(-4000,2000,time_step):
        # Define the start and end of the current interval
        interval_start = i
        interval_end = i + time_step
        year_results.append(interval_start + time_step/2)

        # Filter the data for the current interval
        mask = (calendar_all >= interval_start) & (calendar_all < interval_end)
        
        x_interval = np.array(calendar_all)[mask]
        y_interval = np.array(spd_normalised)[mask]
        
        # Check if we have enough data points to perform integration
        if len(x_interval) > 1:
            # Calculate the area under the curve for the current interval
            area = np.trapz(y_interval, x_interval)
        else:
            # Not enough data points to integrate
            area = np.nan  # or 0, depending on how you want to handle this case
        
        # Store the result
        integration_results.append(area)
    color = cmap(c*30)
    c = c + 1
    #get a 500yr running average for spd_normalised
    spd_average = pd.Series(spd_normalised).rolling(window=2000).mean()
    plt.plot(calendar_all, spd_average, label=distance, color = color)
    
# %%
spd_final = spd_sum_distance / spd_sum
plt.plot(calendar_all, spd_final, color = 'black', label="All Sites")
#add line of best fit
m, b = np.polyfit(calendar_all, spd_final, 1)
plt.plot(calendar_all, m*np.array(calendar_all) + b, color = 'black', linestyle='dashed')
plt.xlabel("Year")
plt.ylabel("Distance")
plt.title("Weigthed Average Distance From Coast")

# %%

# Example data
x = calendar_all
y = spd_final

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Calculate the 95% confidence interval for the slope
alpha = 0.01  # 95% confidence level
n = len(x)
t_value = stats.t.ppf(1 - alpha/2, df=n-2)  # Two-tailed t-distribution
slope_conf_interval = (slope - t_value * std_err, slope + t_value * std_err)

print(f"Slope: {slope}")
print(f"99% Confidence Interval for the Slope: {slope_conf_interval}")

# Plotting the data and the line of best fit
plt.plot(x, y, color = 'black', label='Data Points')
plt.plot(x, intercept + slope * x, color = 'red', linestyle = '--', label='Line of Best Fit')
plt.fill_between(x, intercept + slope_conf_interval[0] * x, intercept + slope_conf_interval[1] * x, color='gray', alpha=0.7, label='99% Confidence Interval')
plt.xlabel('Year')
plt.ylabel('Distance (km)')
plt.legend()
plt.title('Weigthed Average Distance From Coast')
plt.show()

# %%
