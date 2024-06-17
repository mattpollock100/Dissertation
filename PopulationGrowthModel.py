
#%%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Specify the directory you want to use
directory = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/ENSO Masks/'
model = 'IPSL_CM6'  #IPSL_CM6, MPI_ESM
enso = 'ENSO12'
filename = model + '_0_' + enso + '_Mask.npy'

n_max = 66000
n_step = 6000

#doubling time in years
doubling_time  = 6000

boost = 2.0

#monthly growth rate from doubling time
monthly_growth_rate = ( (2.0 ** (1/(12*doubling_time))) - 1.0 ) * boost

decline_rate = monthly_growth_rate * 2.0 
recovery_rate =  decline_rate * 2.0 

carry_capacity = 10

initial_population = 1


data = np.load(os.path.join(directory, filename))
population = np.zeros(data.shape)
population[0] = initial_population
growth_rate = np.zeros(data.shape)
growth_rate[0] = monthly_growth_rate

population_output = []
year_output = []

population_output.append(initial_population)
year_output.append(0)


#for each cell in the dataframe, calculate the population at each time step
#if  FALSE, population increases by standard logistic growth model
#if TRUE, growth rate is reduced by decline_rate


for i in range(1, len(data)):
    if data[i] == True:
        growth_rate[i] = growth_rate[i-1] - decline_rate
    else:
        growth_rate[i] = min(monthly_growth_rate, growth_rate[i-1] + recovery_rate)

    population[i] = population[i-1] + growth_rate[i] * population[i-1] * (1 - population[i-1]/carry_capacity)  


population_output.append(population[i])
year_output.append(500)

initial_population = population[i]
initial_growth_rate = growth_rate[i]


for n in range(n_step, n_max + n_step, n_step):
    filename = model + '_' + str(n) + '_' + enso + '_Mask.npy'
    data = np.load(os.path.join(directory, filename))
    population = np.zeros(data.shape)
    population[0] = initial_population
    growth_rate = np.zeros(data.shape)
    growth_rate[0] = initial_growth_rate

    for i in range(1, len(data)):
        if data[i] == True:
            growth_rate[i] = growth_rate[i-1] - decline_rate
        else:
            growth_rate[i] = min(monthly_growth_rate, growth_rate[i-1] + recovery_rate)

        population[i] = population[i-1] + growth_rate[i] * population[i-1] * (1 - population[i-1]/carry_capacity)  


    population_output.append(population[i])
    year_output.append(n/12 + 500)

    initial_population = population[i]
    inital_growth_rate = growth_rate[i]





#plt.figure(figsize=(10, 6))
#plt.plot(year_output, population_output)
#plt.xlabel('Year')
#plt.ylabel('Population')
#plt.title('Population vs Year')
#plt.grid(True)
#plt.show()


population_output_np = np.array(population_output)
population_change = np.diff(population_output_np) / population_output_np[:-1] * 100

year_output = [year - n_max/12 - 750 for year in year_output]

plt.figure(figsize=(10, 6))
plt.plot(year_output[1:], population_change)
plt.xlabel('Year BP')
plt.ylabel('Population Change per 500 Years (%)')
plt.title(model + ' Population Change vs Year ' + enso)
plt.grid(True)
plt.show()



# %%


from scipy.optimize import curve_fit

# Define a piecewise linear function
def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0, x >= x0], [lambda x: k1*x + y0 - k1*x0, lambda x: k2*x + y0 - k2*x0])

# Generate some synthetic data for illustration
x = year_output[1:]
yn = population_change

# Initial guess for the parameters
p0 = [5, np.mean(yn), 2, -1.5]

# Fit the data
popt, pcov = curve_fit(piecewise_linear, x, yn, p0)

# Plot the data and the fit
xd = np.linspace(0, 10, 1000)
plt.plot(x, yn, "o")
plt.plot(xd, piecewise_linear(xd, *popt), "-")
plt.show()

# Print the optimized parameters
print("Optimized parameters:", popt)
# %%
