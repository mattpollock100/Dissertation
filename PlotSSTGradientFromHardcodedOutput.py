
#%%
#Annual numbers

MPI_ESM_year = [250,
 750,
 1250,
 1750,
 2250,
 2750,
 3250,
 3750,
 4250,
 4750,
 5250,
 5750,
 6250,
 6750,
 7250,
 7750]

MPI_ESM_gradient = [3.3969979024494137,
 3.3687416947669817,
 3.3592010933404026,
 3.3047507227710184,
 3.316111003778019,
 3.164202320017637,
 3.1090089186728846,
 3.040166195801305,
 2.925799807140095,
 2.8543538360008256,
 2.733248307214069,
 2.687804614895583,
 2.548773013371431,
 2.5033861367531927,
 2.3735253573920545,
 2.30546181418606]

IPSL_CM5_year = [250, 750, 1250, 1750, 2250, 2750, 3250, 3750, 4250, 4750, 5250, 5750]

IPSL_CM5_gradient = [2.6076965,
 2.6999207,
 2.6504822,
 2.6090393,
 2.53479,
 2.5462952,
 2.4473877,
 2.4144287,
 2.3762207,
 2.2822266,
 2.248291,
 2.1541748]

IPSL_CM6_year = [250, 750, 1250, 1750, 2250, 2750, 3250, 3750, 4250, 4750, 5250, 5750]

IPSL_CM6_gradient = [3.3367615,
 3.3414001,
 3.3404846,
 3.3024597,
 3.2821655,
 3.2529602,
 3.2155762,
 3.1879883,
 3.104126,
 3.0446472,
 2.9544373,
 2.8874512]

TRACE_year = [250,
 750,
 1250,
 1750,
 2250,
 2750,
 3250,
 3750,
 4250,
 4750,
 5250,
 5750,
 6250,
 6750,
 7250,
 7750,
 8250]

TRACE_gradient = [1.993770395914737,
 2.0335282491047906,
 1.9598536580403447,
 1.8412242635091047,
 1.7320646362304615,
 1.8532895507812555,
 1.7899254353841343,
 1.8581119283040266,
 1.7087273457844958,
 1.7445200297037786,
 1.683502339680956,
 1.6781346333821716,
 1.5812407430013309,
 1.6064023132324223,
 1.6572754720052103,
 1.5604537556965852,
 1.6860253651936432]

 #%%
def calendar_years(model_years, end_year):
    time_frame = model_years[-1]
    calendar_years = [end_year - time_frame +year for year in model_years]

    return calendar_years

def get_anomalies(data, zero_point):
    anomalies = [data_point - data[zero_point] for data_point in data]
    return anomalies

#%%
IPSL_CM5_calendar = calendar_years(IPSL_CM5_year, 1990)
IPSL_CM6_calendar = calendar_years(IPSL_CM6_year, 1990)
MPI_ESM_calendar = calendar_years(MPI_ESM_year, 1850)
TRACE_calendar = calendar_years(TRACE_year, 1950)


#%%
from matplotlib import pyplot as plt

colors = { 'IPSL_CM5': 'red',
                'IPSL_CM6': 'orange',
                'MPI_ESM': 'green',                 
                'TRACE': 'blue'}

fig, ax = plt.subplots()

ax.plot(IPSL_CM5_calendar, IPSL_CM5_gradient, label = 'IPSL_CM5', color = colors['IPSL_CM5'])
ax.plot(IPSL_CM6_calendar, IPSL_CM6_gradient, label = 'IPSL_CM6', color = colors['IPSL_CM6'])
ax.plot(MPI_ESM_calendar, MPI_ESM_gradient, label = 'MPI_ESM', color = colors['MPI_ESM'])
ax.plot(TRACE_calendar, TRACE_gradient, label = 'TRACE', color = colors['TRACE'])

ax.set_xlabel('Year')
ax.set_ylabel('SST Gradient (K)')

#show legend
ax.legend()

#add title
ax.set_title('SST Gradient')

# %%
