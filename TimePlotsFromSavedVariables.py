#%%
import numpy as np
import matplotlib.pyplot as plt
from ModelParams import *
#%%
data_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Output Data'
data_sub_path = '/TempPrecipRegions/'



temp_models = [IPSL_CM5_Temp,  MPI_ESM_Temp,  IPSL_CM6_Temp, TRACE_Temp]
precip_models = [IPSL_CM5_Precip, MPI_ESM_Precip, IPSL_CM6_Precip, TRACE_Precip]


chunk_years = 500

#temp_colors = { 'IPSL_CM5': (165.0/255.0, 15.0/255.0, 21.0/255.0), 
#                'IPSL_CM6': (222.0/255.0, 45.0/255.0, 38.0/255.0), 
#                'MPI_ESM': (251.0/255.0, 106.0/255.0, 74.0/255.0),                 
#                'TRACE': (252.0/255.0, 174.0/255.0, 145.0/255.0)}

temp_colors = { 'IPSL_CM5': 'red',
                'IPSL_CM6': 'orange',
                'MPI_ESM': 'green',                 
                'TRACE': 'blue'}

precip_colors = { 'IPSL_CM5': 'red',
                'IPSL_CM6': 'orange',
                'MPI_ESM': 'green',                 
                'TRACE': 'blue'}

region_dict = {  9: 'NWS',
                 10: 'NSA'}



#%%

region = 9
region_string = region_dict[region]
seasons = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']
seasons = ['Annual']
for season in seasons:

    #create a figure to plot subsequent plots on
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    if season == 'Annual':
        ax2 = ax.twinx()

    for model in temp_models:
        model_name = model['sub_path'].replace('/','')
        variable_name = model['variable_name']
        model_end_year = model['model_end_year']

        output = np.load(data_path + data_sub_path + model_name + '_' + str(region) + '_' + variable_name + '_' + season + '.npy')
        model_years = [t[0] for t in output]
        temperature = [t[1] for t in output]
        variance = [t[2] for t in output]

        #print first and last value in output
        print(model_name + ' ' + str(model_years[0]) + ' ' + str(temperature[0]) + ' ' + str(variance[0]))
        
        print(model_name + ' ' + str(model_years[4]) + ' ' + str(temperature[4]) + ' ' + str(variance[0]))
        print(model_name + ' ' + str(model_years[5]) + ' ' + str(temperature[5]) + ' ' + str(variance[0]))
        
        
        print(model_name + ' ' + str(model_years[-1]) + ' ' + str(temperature[-1]) + ' ' + str(variance[-1]))
        
        simulation_length = np.max(model_years)

        calendar_years = [model_end_year - simulation_length + year for year in model_years]

        color = temp_colors[model_name]
        ax.plot(calendar_years, temperature, label=model_name, color=color)

        if season == 'Annual':
            ax2.plot(calendar_years, variance, label=model_name + ' Variance', linestyle='dashed', color=color)


    #label y-axis
    ax.set_ylabel('Temperature (K)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)  

    if season == 'Annual':
        ax2.set_ylabel('Variance', fontsize=16)
        ax2.tick_params(axis='both', labelsize=14)  # Set font size for x and y-axis tick labels on ax

    #label x-axis
    ax.set_xlabel('Calendar Year', fontsize=16)

    #set title
    ax.set_title(season + ' Temperature for ' + region_string, fontsize=20)
    ax.legend(fontsize=16)
    plt.show()
# %%

region = 10
region_string = region_dict[region]
seasons = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']

for season in seasons:

    #create a figure to plot subsequent plots on
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    if season == 'Annual':
        ax2 = ax.twinx()

    for model in precip_models:
        model_name = model['sub_path'].replace('/','')
        variable_name = model['variable_name']
        model_end_year = model['model_end_year']

        output = np.load(data_path + data_sub_path + model_name + '_' + str(region) + '_' + variable_name + '_' + season + '.npy')
        model_years = [t[0] for t in output]
        temperature = [t[1] for t in output]
        variance = [t[2] for t in output]

        #print first and last value in output
        #print(model_name + ' ' + str(model_years[0]) + ' ' + str(temperature[0]) + ' ' + str(variance[0]))
        
        #print(model_name + ' ' + str(model_years[4]) + ' ' + str(temperature[4]) + ' ' + str(variance[0]))
        #print(model_name + ' ' + str(model_years[5]) + ' ' + str(temperature[5]) + ' ' + str(variance[0]))
        
        
        #print(model_name + ' ' + str(model_years[-1]) + ' ' + str(temperature[-1]) + ' ' + str(variance[-1]))
        
        simulation_length = np.max(model_years)

        calendar_years = [model_end_year - simulation_length + year for year in model_years]

        color = temp_colors[model_name]
        ax.plot(calendar_years, temperature, label=model_name, color=color)

        if season == 'Annual':
            ax2.plot(calendar_years, variance, label=model_name + ' Variance', linestyle='dashed', color=color)


    #label y-axis
    ax.set_ylabel('Precipitation (mm day$^{-1}$)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)  

    if season == 'Annual':
        ax2.set_ylabel('Variance', fontsize=16)
        ax2.tick_params(axis='both', labelsize=14)  # Set font size for x and y-axis tick labels on ax
        ax.legend(fontsize=16, bbox_to_anchor=(0.2, 0.2))
    else:
        #ax.set_ylim(2.0, 8.0)
        ax.legend(fontsize=16)

    #label x-axis
    ax.set_xlabel('Calendar Year', fontsize=16)

    #set title
    ax.set_title(season + ' Precipitation for ' + region_string, fontsize=20)
    
    plt.show()
# %%
