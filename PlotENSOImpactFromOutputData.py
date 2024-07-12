#%%
import numpy as np
import matplotlib.pyplot as plt
from ModelParams import *
#%%
data_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Output Data/ENSOImpact/'

enso_type = '12'

IPSL_CM5_precip_years = np.load(data_path + 'IPSL_CM5_' + enso_type + 'pr_years_ENSO_Impact.npy')
IPSL_CM5_precip_total = np.load(data_path + 'IPSL_CM5_' + enso_type + 'pr_total_ENSO_Impact.npy')
IPSL_CM5_precip_ENSO = np.load(data_path + 'IPSL_CM5_' + enso_type + 'pr_ENSO_ENSO_Impact.npy')
IPSL_CM5_precip_not_ENSO = np.load(data_path + 'IPSL_CM5_' + enso_type + 'pr_not_ENSO_ENSO_Impact.npy')

IPSL_CM6_precip_years = np.load(data_path + 'IPSL_CM6_' + enso_type + 'precip_years_ENSO_Impact.npy')
IPSL_CM6_precip_total = np.load(data_path + 'IPSL_CM6_' + enso_type + 'precip_total_ENSO_Impact.npy')
IPSL_CM6_precip_ENSO = np.load(data_path + 'IPSL_CM6_' + enso_type + 'precip_ENSO_ENSO_Impact.npy')
IPSL_CM6_precip_not_ENSO = np.load(data_path + 'IPSL_CM6_' + enso_type + 'precip_not_ENSO_ENSO_Impact.npy')

MPI_ESM_precip_years = np.load(data_path + 'MPI_ESM_' + enso_type + 'pr_years_ENSO_Impact.npy')
MPI_ESM_precip_total = np.load(data_path + 'MPI_ESM_' + enso_type + 'pr_total_ENSO_Impact.npy')
MPI_ESM_precip_ENSO = np.load(data_path + 'MPI_ESM_' + enso_type + 'pr_ENSO_ENSO_Impact.npy')
MPI_ESM_precip_not_ENSO = np.load(data_path + 'MPI_ESM_' + enso_type + 'pr_not_ENSO_ENSO_Impact.npy')

TRACE_precip_years = np.load(data_path + 'TRACE_' + enso_type + 'PRECIP_years_ENSO_Impact.npy')
TRACE_precip_total = np.load(data_path + 'TRACE_' + enso_type + 'PRECIP_total_ENSO_Impact.npy')
TRACE_precip_ENSO = np.load(data_path + 'TRACE_' + enso_type + 'PRECIP_ENSO_ENSO_Impact.npy')
TRACE_precip_not_ENSO = np.load(data_path + 'TRACE_' + enso_type + 'PRECIP_not_ENSO_ENSO_Impact.npy')

# %%
def calendar_years(model_years, end_year):
    time_frame = model_years[-1]
    calendar_years = [end_year - time_frame +year for year in model_years]

    return calendar_years

IPSL_CM5_calendar = calendar_years(IPSL_CM5_precip_years, 1990)
IPSL_CM6_calendar = calendar_years(IPSL_CM6_precip_years, 1990)
MPI_ESM_calendar = calendar_years(MPI_ESM_precip_years, 1850)
TRACE_calendar = calendar_years(TRACE_precip_years, 1950)

#%%
def plot_precip_impact(calendar, total, enso, model_name, enso_type):
    fig, ax1 = plt.subplots()

    ax1.plot(calendar, total, label = ' Total', color = 'blue', linestyle=':')
    ax1.plot(calendar, enso, label = ' El Niño', color = 'blue')

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Precipitation (mm day$^{-1}$)')
    ax1.set_title(model_name + ' Niño' + enso_type + ' Impact on Precipitation')
    
    ax2 = ax1.twinx()

    anomaly = enso - total
    ax2.plot(calendar, anomaly, label = model_name + ' Anomaly', color = 'black', linestyle='--')

    ax2.set_ylabel('Anomaly (mm day$^{-1}$)')
# %%
plot_precip_impact(IPSL_CM5_calendar, IPSL_CM5_precip_total, IPSL_CM5_precip_ENSO, 'IPSL CM5', enso_type)
plot_precip_impact(IPSL_CM6_calendar, IPSL_CM6_precip_total, IPSL_CM6_precip_ENSO, 'IPSL CM6', enso_type)
plot_precip_impact(MPI_ESM_calendar, MPI_ESM_precip_total, MPI_ESM_precip_ENSO, 'MPI ESM', enso_type)
plot_precip_impact(TRACE_calendar, TRACE_precip_total, TRACE_precip_ENSO, 'TRACE', enso_type)

# %%

IPSL_CM5_temp_years = np.load(data_path + 'IPSL_CM5_' + enso_type + 'tas_years_ENSO_Impact.npy')
IPSL_CM5_temp_total = np.load(data_path + 'IPSL_CM5_' + enso_type + 'tas_total_ENSO_Impact.npy')
IPSL_CM5_temp_ENSO = np.load(data_path + 'IPSL_CM5_' + enso_type + 'tas_ENSO_ENSO_Impact.npy')
IPSL_CM5_temp_not_ENSO = np.load(data_path + 'IPSL_CM5_' + enso_type + 'tas_not_ENSO_ENSO_Impact.npy')

IPSL_CM6_temp_years = np.load(data_path + 'IPSL_CM6_' + enso_type + 'tas_years_ENSO_Impact.npy')
IPSL_CM6_temp_total = np.load(data_path + 'IPSL_CM6_' + enso_type + 'tas_total_ENSO_Impact.npy')
IPSL_CM6_temp_ENSO = np.load(data_path + 'IPSL_CM6_' + enso_type + 'tas_ENSO_ENSO_Impact.npy')
IPSL_CM6_temp_not_ENSO = np.load(data_path + 'IPSL_CM6_' + enso_type + 'tas_not_ENSO_ENSO_Impact.npy')

MPI_ESM_temp_years = np.load(data_path + 'MPI_ESM_' + enso_type + 'tas_years_ENSO_Impact.npy')
MPI_ESM_temp_total = np.load(data_path + 'MPI_ESM_' + enso_type + 'tas_total_ENSO_Impact.npy')
MPI_ESM_temp_ENSO = np.load(data_path + 'MPI_ESM_' + enso_type + 'tas_ENSO_ENSO_Impact.npy')
MPI_ESM_temp_not_ENSO = np.load(data_path + 'MPI_ESM_' + enso_type + 'tas_not_ENSO_ENSO_Impact.npy')

TRACE_temp_years = np.load(data_path + 'TRACE_' + enso_type + 'TREFHT_years_ENSO_Impact.npy')
TRACE_temp_total = np.load(data_path + 'TRACE_' + enso_type + 'TREFHT_total_ENSO_Impact.npy')
TRACE_temp_ENSO = np.load(data_path + 'TRACE_' + enso_type + 'TREFHT_ENSO_ENSO_Impact.npy')
TRACE_temp_not_ENSO = np.load(data_path + 'TRACE_' + enso_type + 'TREFHT_not_ENSO_ENSO_Impact.npy')

#%%
IPSL_CM5_calendar = calendar_years(IPSL_CM5_temp_years, 1990)
IPSL_CM6_calendar = calendar_years(IPSL_CM6_temp_years, 1990)
MPI_ESM_calendar = calendar_years(MPI_ESM_temp_years, 1850)
TRACE_calendar = calendar_years(TRACE_temp_years, 1950)

#%%
def plot_temp_impact(calendar, total, enso, model_name, enso_type):
    fig, ax1 = plt.subplots()

    ax1.plot(calendar, total, label = ' Total', color = 'red', linestyle=':')
    ax1.plot(calendar, enso, label = ' El Niño', color = 'red')

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title(model_name + ' Niño' + enso_type + ' Impact on Temperature')
    
    ax2 = ax1.twinx()

    anomaly = enso - total
    ax2.plot(calendar, anomaly, label = model_name + ' Anomaly', color = 'black', linestyle='--')

# %%
plot_temp_impact(IPSL_CM5_calendar, IPSL_CM5_temp_total, IPSL_CM5_temp_ENSO, 'IPSL CM5', enso_type)
plot_temp_impact(IPSL_CM6_calendar, IPSL_CM6_temp_total, IPSL_CM6_temp_ENSO, 'IPSL CM6', enso_type)
plot_temp_impact(MPI_ESM_calendar, MPI_ESM_temp_total, MPI_ESM_temp_ENSO, 'MPI ESM', enso_type)
plot_temp_impact(TRACE_calendar, TRACE_temp_total, TRACE_temp_ENSO, 'TRACE', enso_type)