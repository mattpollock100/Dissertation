#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro

#%%
# Parameters
a = 0.8  # Feedback coefficient
b = 0.0  # Constant term
sigma = 0.5  # Standard deviation of Gaussian noise
num_steps = 10000  # Number of time steps

# Initialize state array
x = np.zeros(num_steps)

# Initial state
x[0] = 0.0

# Generate the time series
for t in range(1, num_steps):
    epsilon_t = np.random.normal(0, sigma)  # Gaussian noise
    x[t] = a * x[t-1] + b + epsilon_t

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, label='State $x_t$')
plt.title('Simple Linear System with Feedback')
plt.xlabel('Time Step')
plt.ylabel('State $x_t$')
plt.legend()
plt.grid(True)
plt.show()

#%%
#Plot a histogram of the data x
plt.figure(figsize=(10, 6))
plt.hist(x, bins=20, density=True)

# %%
#shapiro-wilkes test on x for normality


# Perform the Shapiro-Wilk test for normality
test_statistic, p_value = shapiro(x)

print(f"Shapiro-Wilk Test Statistic: {test_statistic}, p-value: {p_value}")

if p_value > 0.05:
    print("Data appears to be normally distributed.")
else:
    print("Data does not appear to be normally distributed.")
# %%
#Get mean and std deviation of x
mean = np.mean(x)
std_dev = np.std(x)
print(f"Mean: {mean}, Standard Deviation: {std_dev}")
# %%

# %%
