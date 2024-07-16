#%%
#Data from Map Pressure with Smoothing.py

MPI_ESM_year = [250.0,
 750.0,
 1250.0,
 1750.0,
 2250.0,
 2750.0,
 3250.0,
 3750.0,
 4250.0,
 4750.0,
 5250.0,
 5750.0,
 6250.0,
 6750.0,
 7250.0,
 7750.0]

MPI_ESM_102200_area = [2642681.7812599493,
 2164286.280046056,
 2035487.629926801,
 1977742.1299341267,
 1599843.8313833603,
 1612041.1565796488,
 1358735.3883354296,
 1364609.2979624209,
 960920.3739830395,
 936772.8489864754,
 735153.8152424983,
 357741.2900086526,
 225139.5722962208,
 178557.0876564252,
 143040.24521661783,
 141263.6606539438]

MPI_ESM_102000_area = [7841817.327934047,
 7573585.342270594,
 7503248.0630447045,
 7456473.9842403,
 7104558.6986274365,
 7113602.830557472,
 6807791.329743936,
 6775448.4824080495,
 6397938.966801293,
 6250883.957037802,
 6055835.087592482,
 5565382.736390308,
 5425916.498235104,
 5164621.170264461,
 5062012.504942066,
 5052892.9812780395]

TRACE_year = [250.0,
 750.0,
 1250.0,
 1750.0,
 2250.0,
 2750.0,
 3250.0,
 3750.0,
 4250.0,
 4750.0,
 5250.0,
 5750.0,
 6250.0,
 6750.0,
 7250.0,
 7750.0,
 8250.0]


TRACE_102000_area = [2692746.570612651,
 2599719.894922434,
 2362084.449194294,
 2267204.7715483485,
 2281170.0693081203,
 1932752.029520336,
 1938379.0311094464,
 1923158.58688028,
 1715571.5213941252,
 1759356.127291754,
 1689613.062778701,
 1735705.6903843395,
 1836468.4997846182,
 1890727.121896262,
 1758465.8302611804,
 872374.0654428706,
 1154460.0773226274]

 
  #%%
def calendar_years(model_years, end_year):
    time_frame = model_years[-1]
    calendar_years = [end_year - time_frame +year for year in model_years]

    return calendar_years


#%%

MPI_ESM_calendar = calendar_years(MPI_ESM_year, 1850)
TRACE_calendar = calendar_years(TRACE_year, 1950)

#%%
from matplotlib import pyplot as plt

#%%
colors = { 'IPSL_CM5': 'red',
                'IPSL_CM6': 'orange',
                'MPI_ESM': 'green',                 
                'TRACE': 'blue'}

fig, ax = plt.subplots()

ax2 = ax.twinx()

#ax.plot(IPSL_CM5_calendar, IPSL_CM5_MAM_pct, label = 'IPSL_CM5 (MAM)', color = colors['IPSL_CM5'])
#ax.plot(IPSL_CM6_calendar, IPSL_CM6_MAM_pct, label = 'IPSL_CM6 (MAM)', color = colors['IPSL_CM6'])
ax.plot(MPI_ESM_calendar, MPI_ESM_102000_area, label = 'MPI_ESM', color = colors['MPI_ESM'])
ax2.plot(TRACE_calendar, TRACE_102000_area, label = 'TRACE', color = colors['TRACE'])



ax.set_xlabel('Year')
ax.set_ylabel('Area (km²), MPI')

ax2.set_ylabel('Area (km²), TraCE')

# Extract handles and labels for both axes
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Combine handles and labels
combined_handles = handles1 + handles2
combined_labels = labels1 + labels2

# Create a single legend on the first axis (or whichever you prefer)
ax.legend(combined_handles, combined_labels)


#add title
ax.set_title('Area with Pressure Greater Than 1020hPa in the SPAC')

plt.show()

# %%