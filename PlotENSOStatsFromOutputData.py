#%%
import xarray
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#%%
output_path = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Output Data/ENSOStats/'

enso_type = '12'

IPSL_CM5_count = np.load(output_path + 'IPSL_CM5_enso_' + enso_type + '_count.npy')
IPSL_CM5_years = np.load(output_path + 'IPSL_CM5_enso_' + enso_type + '_years.npy')
IPSL_CM5_mean_anomlay = np.load(output_path + 'IPSL_CM5_enso_' + enso_type + '_mean_anomaly.npy')
IPSL_CM5_mean_length = np.load(output_path + 'IPSL_CM5_enso_' + enso_type + '_mean_length.npy')

IPSL_CM6_count = np.load(output_path + 'IPSL_CM6_enso_' + enso_type + '_count.npy')
IPSL_CM6_years = np.load(output_path + 'IPSL_CM6_enso_' + enso_type + '_years.npy')
IPSL_CM6_mean_anomlay = np.load(output_path + 'IPSL_CM6_enso_' + enso_type + '_mean_anomaly.npy')
IPSL_CM6_mean_length = np.load(output_path + 'IPSL_CM6_enso_' + enso_type + '_mean_length.npy')

MPI_ESM_count = np.load(output_path + 'MPI_ESM_enso_' + enso_type + '_count.npy')
MPI_ESM_years = np.load(output_path + 'MPI_ESM_enso_' + enso_type + '_years.npy')
MPI_ESM_mean_anomlay = np.load(output_path + 'MPI_ESM_enso_' + enso_type + '_mean_anomaly.npy')
MPI_ESM_mean_length = np.load(output_path + 'MPI_ESM_enso_' + enso_type + '_mean_length.npy')

TRACE_count = np.load(output_path + 'TRACE_enso_' + enso_type + '_count.npy')
TRACE_years = np.load(output_path + 'TRACE_enso_' + enso_type + '_years.npy')
TRACE_mean_anomlay = np.load(output_path + 'TRACE_enso_' + enso_type + '_mean_anomaly.npy')
TRACE_mean_length = np.load(output_path + 'TRACE_enso_' + enso_type + '_mean_length.npy')
# %%
def calendar_years(model_years, end_year):
    time_frame = model_years[-1]
    calendar_years = [end_year - time_frame +year for year in model_years]

    return calendar_years

IPSL_CM5_calendar = calendar_years(IPSL_CM5_years, 1990)
IPSL_CM6_calendar = calendar_years(IPSL_CM6_years, 1990)
MPI_ESM_calendar = calendar_years(MPI_ESM_years, 1850)
TRACE_calendar = calendar_years(TRACE_years, 1950)

#%%
def plot_enso_stats(calendar, count, mean_anomaly, mean_length, model_name, enso_type):
    fig, ax1 = plt.subplots()

    # Plot the first sub-element on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Calendar Year')
    ax1.set_ylabel('Count', color=color)
    ax1.plot(calendar, count, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))


    # Create a second y-axis for the second sub-element
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean Anomaly (K)', color=color)
    ax2.plot(calendar, mean_anomaly, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Create a third y-axis for the third sub-element
    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Mean Length (months)', color=color)
    ax3.plot(calendar, mean_length, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    title = model_name + ' El Niño' + enso_type
    fig.tight_layout()
    plt.title(title)
    plot_file_name = title + '.png'
    #plt.savefig(plot_path + plot_file_name)
    plt.show()
# %%
plot_enso_stats(IPSL_CM5_calendar, IPSL_CM5_count, IPSL_CM5_mean_anomlay, IPSL_CM5_mean_length, 'IPSL CM5', enso_type)


plot_enso_stats(IPSL_CM6_calendar, IPSL_CM6_count, IPSL_CM6_mean_anomlay, IPSL_CM6_mean_length, 'IPSL CM6', enso_type)


plot_enso_stats(MPI_ESM_calendar, MPI_ESM_count, MPI_ESM_mean_anomlay, MPI_ESM_mean_length, 'MPI ESM', enso_type)


#TRACE_count = np.delete(TRACE_count, 0)
plot_enso_stats(TRACE_calendar, TRACE_count, TRACE_mean_anomlay, TRACE_mean_length, 'TRACE', enso_type)

# %%
