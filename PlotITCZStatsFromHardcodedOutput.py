#%%

#IPSL_CM5

IPSL_CM5_year = [250, 750, 1250, 1750, 2250, 2750, 3250, 3750, 4250, 4750, 5250, 5750]
IPSL_CM5_CoM = [1.0127380801486292, 0.9315752557857789, 0.9181959699474251, 0.8732731771712895, 0.8627347327172216, 0.8371114809412946, 0.8553403519515514, 0.8044910906483042, 0.7891488144151811, 0.8323063218151203, 0.8469093043043802, 0.8812060822534226]
IPSL_CM5_max = [5.553311714287792, 5.5364364716881385, 5.572272496917238, 5.56566573407728, 5.537141792813983, 5.582274129720649, 5.577357454844693, 5.601506932310374, 5.6389125165616925, 5.709252366315161, 5.733679328194393, 5.828551488161224]
IPSL_CM5_lat = [5.789473745557997, 5.931721257733869, 5.903271755298695, 5.960170760169044, 6.017069765039393, 6.0455192674745675, 5.988620262604218, 5.988620262604218, 6.0455192674745675, 5.960170760169044, 5.988620262604218, 5.931721257733869]


#IPSL_CM6


IPSL_CM6_year = [250, 750, 1250, 1750, 2250, 2750, 3250, 3750, 4250, 4750, 5250, 5750]
IPSL_CM6_CoM = [1.2349085903277353, 1.1825242327746714, 1.1425768848974966, 1.105640807525215, 1.0450561976789114, 1.0180941894340636, 1.028600215913266, 0.9912888523780857, 1.0431766289641815, 1.0647846234597684, 1.1031017146747977, 1.120135422995532]
IPSL_CM6_max = [5.671062332568826, 5.61730061125347, 5.563928996625117, 5.533246883246821, 5.48962020233911, 5.416455482327067, 5.417256398022899, 5.400013751292606, 5.433611802047212, 5.440405497721277, 5.48719074782065, 5.533404015463521]
IPSL_CM6_lat =[5.178276763663993, 5.178276763663993, 5.178276763663993, 5.178276763663993, 5.178276763663993, 5.178276763663993, 5.178276763663993, 5.178276763663993, 5.178276763663993, 5.150361525045859, 5.150361525045859, 5.122446286427724]



#MPI_ESM

MPI_ESM_year = [250, 750, 1250, 1750, 2250, 2750, 3250, 3750, 4250, 4750, 5250, 5750, 6250, 6750, 7250, 7750]
MPI_ESM_CoM = [5.845936483967234, 5.806584431609509, 5.820129686666003, 5.7890062674425415, 5.799605385125146, 5.760556415534185, 5.731823085494867, 5.706295315313601, 5.601573443688456, 5.645917984212722, 5.562728044078053, 5.568127181654999, 5.536973067660776, 5.516438271156653, 5.4487192346768305, 5.385394063708484]
MPI_ESM_max = [7.572992403706883, 7.492659999477442, 7.4338237749920095, 7.4211404693634, 7.33366241281828, 7.280346373294777, 7.133732128160886, 7.178259458526296, 7.045945461417749, 7.087808168057468, 7.019731957106081, 7.048285808021163, 7.066356669501061, 7.074880409987054, 7.124658339148082, 7.1207425809824425]
MPI_ESM_lat = [7.239783216504906, 7.239783216504906, 7.267790114827942, 7.267790114827942, 7.295797013150978, 7.323803911474014, 7.323803911474014, 7.295797013150978, 7.295797013150978, 7.295797013150978, 7.295797013150978, 7.295797013150978, 7.239783216504906, 7.211776318181869, 7.183769419858833, 7.211776318181869]



#TRACE


TRACE_year = [250, 750, 1250, 1750, 2250, 2750, 3250, 3750, 4250, 4750, 5250, 5750, 6250, 6750, 7250, 7750, 8250]
TRACE_CoM = [0.5288842665178644, 0.3567872899806467, 0.5950298321929746, 0.7773805985294935, 0.9409042259666094, 0.7456226915793805, 0.821892561055194, 0.7236938479489532, 0.8272145812848111, 0.7618703431791796, 0.8001892221238033, 0.6510577042010867, 0.7903834272078397, 0.7620349696774722, 0.6815317918097435, 0.6655844813097888, 0.44851492923272623]
TRACE_max = [4.706722316182467, 4.562988227129826, 4.71455323519064, 4.832171780752083, 4.910947663768551, 4.790808237006575, 4.822641350859699, 4.714301903156738, 4.854052100255296, 4.725455390050254, 4.786930933911301, 4.73530380392463, 4.852174033032407, 4.813427219095372, 4.829003810074005, 4.662642167887943, 4.505023713681288]
TRACE_lat = [5.81188978597938, 5.733877842543415, 5.915905710560667, 6.045925616287275, 6.149941540868561, 6.123937559723239, 6.201949503159204, 6.201949503159204, 6.253957465449847, 6.305965427740491, 6.305965427740491, 6.227953484304526, 6.201949503159204, 6.227953484304526, 6.123937559723239, 6.253957465449847, 6.045925616287275]

#%%
def calendar_years(model_years, end_year):
    time_frame = model_years[-1]
    calendar_years = [end_year - time_frame +year for year in model_years]

    return calendar_years

def get_anomalies(data, zero_point):
    anomalies = [data_point - data[zero_point] for data_point in data]
    return anomalies

#%%
def plot_with_ci(x, y, color, label, ax, ci):
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Calculate the 95% confidence interval for the slope
    alpha = 1 - ci  # 95% confidence level
    n = len(x)
    t_value = stats.t.ppf(1 - alpha/2, df=n-2)  # Two-tailed t-distribution
    slope_conf_interval = (slope - t_value * std_err, slope + t_value * std_err)

    print(f"Slope: {slope}")
    print(f"Confidence Interval for the Slope: {slope_conf_interval}")

    # Plotting the data and the line of best fit
    plt.plot(x, y, color = color, label=label)
    plt.plot(x, intercept + slope * x, color = color, linestyle = '--')
    plt.fill_between(x, intercept + slope_conf_interval[0] * x, intercept + slope_conf_interval[1] * x, color=color, alpha=0.1)


#%%
IPSL_CM5_calendar = np.array(calendar_years(IPSL_CM5_year, 1990))
IPSL_CM6_calendar = np.array(calendar_years(IPSL_CM6_year, 1990))
MPI_ESM_calendar = np.array(calendar_years(MPI_ESM_year, 1850))
TRACE_calendar = np.array(calendar_years(TRACE_year, 1950))

IPSL_CM5_lat_anom = get_anomalies(IPSL_CM5_lat, 0)
IPSL_CM6_lat_anom = get_anomalies(IPSL_CM6_lat, 0)
MPI_ESM_lat_anom = get_anomalies(MPI_ESM_lat, 4)
TRACE_lat_anom = get_anomalies(TRACE_lat, 5)

IPSL_CM5_CoM_anom = get_anomalies(IPSL_CM5_CoM, 0)
IPSL_CM6_CoM_anom = get_anomalies(IPSL_CM6_CoM, 0)
MPI_ESM_CoM_anom = get_anomalies(MPI_ESM_CoM, 4)
TRACE_CoM_anom = get_anomalies(TRACE_CoM, 5)

# %%
#plot all calendar and anomalies on one chart
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


#%%

colors = { 'IPSL_CM5': 'red',
                'IPSL_CM6': 'orange',
                'MPI_ESM': 'green',                 
                'TRACE': 'blue'}

fig, ax = plt.subplots()

ax.plot(IPSL_CM5_calendar, IPSL_CM5_lat_anom, label = 'IPSL_CM5', color = colors['IPSL_CM5'])
ax.plot(IPSL_CM6_calendar, IPSL_CM6_lat_anom, label = 'IPSL_CM6', color = colors['IPSL_CM6'])
ax.plot(MPI_ESM_calendar, MPI_ESM_lat_anom, label = 'MPI_ESM', color = colors['MPI_ESM'])
ax.plot(TRACE_calendar, TRACE_lat_anom, label = 'TRACE', color = colors['TRACE'])

#show legend
ax.legend()

#add title
ax.set_title('Latitude of Maximum Precipitation (Anomaly vs 6000BP)')
# %%
fig, ax = plt.subplots()

ci = 0.95

plot_with_ci(IPSL_CM5_calendar, IPSL_CM5_CoM_anom, label = 'IPSL_CM5', color = colors['IPSL_CM5'], ax = ax, ci = ci)
plot_with_ci(IPSL_CM6_calendar, IPSL_CM6_CoM_anom, label = 'IPSL_CM6', color = colors['IPSL_CM6'], ax = ax, ci = ci)
plot_with_ci(MPI_ESM_calendar, MPI_ESM_CoM_anom, label = 'MPI_ESM', color = colors['MPI_ESM'], ax = ax, ci = ci)
plot_with_ci(TRACE_calendar, TRACE_CoM_anom, label = 'TRACE', color = colors['TRACE'], ax = ax, ci = 0)

#show legend
ax.legend()

#add title
ax.set_title('Lat of Precipitation Centre of Mass (Anomaly vs 6000BP)')
# %%
