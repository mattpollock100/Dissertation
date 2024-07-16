
#%%
#Data from ENSO Masks.py

IPSL_CM5_year = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
IPSL_CM5_MAM_pct = [0.3114754098360656,
 0.3835616438356164,
 0.45121951219512196,
 0.4634146341463415,
 0.46987951807228917,
 0.5280898876404494,
 0.4523809523809524,
 0.47126436781609193,
 0.5384615384615384,
 0.5,
 0.45454545454545453,
 0.5057471264367817]

IPSL_CM6_year  = IPSL_CM5_year

IPSL_CM6_MAM_pct = [0.5443037974683544,
 0.5,
 0.4864864864864865,
 0.4657534246575342,
 0.4050632911392405,
 0.5063291139240507,
 0.4146341463414634,
 0.4567901234567901,
 0.44155844155844154,
 0.4523809523809524,
 0.42857142857142855,
 0.5111111111111111]

MPI_ESM_year = [0,
 500,
 1000,
 1500,
 2000,
 2500,
 3000,
 3500,
 4000,
 4500,
 5000,
 5500,
 6000,
 6500,
 7000,
 7500]

MPI_ESM_DJF_pct = [0.43209876543209874,
 0.3411764705882353,
 0.41025641025641024,
 0.34177215189873417,
 0.4027777777777778,
 0.3333333333333333,
 0.3670886075949367,
 0.28205128205128205,
 0.35443037974683544,
 0.3466666666666667,
 0.23076923076923078,
 0.3246753246753247,
 0.22077922077922077,
 0.17333333333333334,
 0.16176470588235295,
 0.25]

TRACE_year = [0,
 500,
 1000,
 1500,
 2000,
 2500,
 3000,
 3500,
 4000,
 4500,
 5000,
 5500,
 6000,
 6500,
 7000,
 7500,
 8000]

TRACE_JJA_pct = [0.47058823529411764,
 0.4330708661417323,
 0.47244094488188976,
 0.5620437956204379,
 0.5255474452554745,
 0.5333333333333333,
 0.5362318840579711,
 0.5109489051094891,
 0.5136986301369864,
 0.472,
 0.45864661654135336,
 0.4647887323943662,
 0.5384615384615384,
 0.4626865671641791,
 0.54421768707483,
 0.48905109489051096,
 0.5333333333333333]

  #%%
def calendar_years(model_years, end_year):
    time_frame = model_years[-1]
    calendar_years = [end_year - time_frame +year for year in model_years]

    return calendar_years


#%%
IPSL_CM5_calendar = calendar_years(IPSL_CM5_year, 1990)
IPSL_CM6_calendar = calendar_years(IPSL_CM6_year, 1990)
MPI_ESM_calendar = calendar_years(MPI_ESM_year, 1850)
TRACE_calendar = calendar_years(TRACE_year, 1950)

#%%
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

colors = { 'IPSL_CM5': 'red',
                'IPSL_CM6': 'orange',
                'MPI_ESM': 'green',                 
                'TRACE': 'blue'}

fig, ax = plt.subplots()

ax.plot(IPSL_CM5_calendar, IPSL_CM5_MAM_pct, label = 'IPSL_CM5 (MAM)', color = colors['IPSL_CM5'])
ax.plot(IPSL_CM6_calendar, IPSL_CM6_MAM_pct, label = 'IPSL_CM6 (MAM)', color = colors['IPSL_CM6'])
ax.plot(MPI_ESM_calendar, MPI_ESM_DJF_pct, label = 'MPI_ESM (DJF)', color = colors['MPI_ESM'])
ax.plot(TRACE_calendar, TRACE_JJA_pct, label = 'TRACE (JJA)', color = colors['TRACE'])



ax.set_xlabel('Year')
ax.set_ylabel('Percentage')

#show legend
ax.legend()

#add title
ax.set_title('Percentage of Niño34 events occuring during highest frequency season')

def to_percentage(x, pos):
    return f'{100 * x:.0f}%'

# Create formatter
formatter = FuncFormatter(to_percentage)

# Get current axis
ax = plt.gca()

# Set the formatter for the y-axis
ax.yaxis.set_major_formatter(formatter)

plt.show()

# %%