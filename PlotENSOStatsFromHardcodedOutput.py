#%%

#Data from StatsOnENSOAnomalies.py

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
 7250.0]

TRACE_skew = [-0.17012417840079358,
 -0.08495321926167704,
 -0.1314025354401032,
 -0.07528882437123136,
 -0.1522201693912448,
 -0.08339209091655894,
 -0.12183968929699934,
 -0.015614193187997083,
 -0.053983297031834694,
 0.04751899165454431,
 -0.1126544199937749,
 -0.03172556879176252,
 0.008334434536382981,
 0.10677278823201103,
 -0.04770609078248399]

TRACE_kurtosis = [-0.15027349233279175,
 -0.015430355533748052,
 0.0034854184747175054,
 -0.11599790703773216,
 -0.10449362896434922,
 -0.033917577626147555,
 -0.23841827706523144,
 0.06857233605805124,
 0.05434205056539154,
 0.31340062037095073,
 0.16843381192357088,
 0.002456575757338708,
 0.10439658227428561,
 0.2628152261761678,
 0.02726912105832202]

TRACE_hurst = [0.3956276921159582,
 0.41098237709307217,
 0.4167851586335674,
 0.4798124769215592,
 0.3772553807274899,
 0.40998390639685883,
 0.4249376001196042,
 0.3921810073046705,
 0.4149031056028736,
 0.3961760839613112,
 0.36651408207645536,
 0.41974028514276,
 0.3824148302279204,
 0.40041400572804253,
 0.3503763821284371]

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
 7250.0]

MPI_ESM_skew = [-0.08859747108291416,
 -0.006092563891149331,
 0.019655926075497487,
 -0.05923632582407907,
 0.06897739538145821,
 0.07473734258738648,
 -0.07736656635403441,
 -0.11432252089164763,
 0.22278255448271517,
 0.13118786476625477,
 0.13747106727722574,
 0.09366540263821382,
 0.07755711136675404,
 0.27334994766667725,
 0.1364708077235745]

MPI_ESM_kurtosis = [0.08659469187852586,
 0.10896101832978466,
 0.1279863053458694,
 -0.08156080839234736,
 0.597591107557796,
 0.02924269279427616,
 0.36912850748491133,
 0.0082165210700067,
 0.33617518640168953,
 0.6046164356378716,
 0.3974974098906423,
 0.1565685868105664,
 0.07071425362422223,
 0.6449368020486106,
 0.035449502701268454]

MPI_ESM_hurst = [0.37460446646629714,
 0.3945507412378817,
 0.32109187363519603,
 0.38096730266036255,
 0.3981633008725263,
 0.4214037828415488,
 0.40594893752629874,
 0.396252394145921,
 0.33846663794749043,
 0.3824542955295944,
 0.3665686814296589,
 0.45258763025627413,
 0.31185770371912824,
 0.3762232671586649,
 0.3507366408309336]

IPSL_CM5_year = [250.0,
 750.0,
 1250.0,
 1750.0,
 2250.0,
 2750.0,
 3250.0,
 3750.0,
 4250.0,
 4750.0,
 5250.0]

IPSL_CM5_skew = [-0.13766876075041107,
 0.16571280047423487,
 0.14919280213041275,
 0.07158639084452562,
 0.24424200333220192,
 0.0808935534999522,
 0.12189860612731264,
 0.07325010108744788,
 0.16919210299955229,
 0.20250991540908966,
 0.21291770196560772]

IPSL_CM5_kurtosis = [0.5451853306719836,
 0.2594049074640492,
 0.1899466889621806,
 0.2009827193673157,
 0.14351955707747255,
 0.13355337259007882,
 0.02589363760025698,
 -0.08154864299134301,
 -0.030020765324339838,
 0.28015275984654364,
 0.13398393304066447]

IPSL_CM5_hurst = [0.43845139076151524,
 0.4049095905846157,
 0.4057224848451088,
 0.42794395559937504,
 0.42849176762609625,
 0.3665829891439567,
 0.3680697187618245,
 0.35094928243234536,
 0.40341199296971847,
 0.37698663073152777,
 0.48139620761131763]

IPSL_CM6_year = [250.0,
 750.0,
 1250.0,
 1750.0,
 2250.0,
 2750.0,
 3250.0,
 3750.0,
 4250.0,
 4750.0,
 5250.0]

IPSL_CM6_skew = [0.09630163303012462,
 0.07815274665683287,
 0.2130314875059603,
 0.14565458184912247,
 0.2463100681854674,
 0.3940535531508962,
 0.14662317492650676,
 0.2991602580076354,
 0.05888937446472977,
 0.06799245052467694,
 0.17060350371238303]

IPSL_CM6_kurtosis = [0.365420106012881,
 0.26724379366085405,
 0.27360437196177534,
 0.4829298432896869,
 0.2452015957666105,
 0.511859338077477,
 0.23493174622188473,
 0.2921336473598477,
 0.2155929220970738,
 0.09847186150618814,
 0.06658063698144723]

IPSL_CM6_hurst = [0.3966808638948744,
 0.4299746735410489,
 0.4749197962013634,
 0.3213885919208064,
 0.47100789369541596,
 0.4475833909581751,
 0.44699399626112934,
 0.32366843901605,
 0.393110682404426,
 0.42130542891146955,
 0.3949480310281245]

 
 #%%
def calendar_years(model_years, end_year):
    time_frame = model_years[-1]
    calendar_years = [end_year - time_frame +year for year in model_years]

    return calendar_years

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


#%%
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats

#%%
IPSL_CM5_data = IPSL_CM5_skew
IPSL_CM6_data = IPSL_CM6_skew
MPI_ESM_data = MPI_ESM_skew
TRACE_data = TRACE_skew

colors = { 'IPSL_CM5': 'red',
                'IPSL_CM6': 'orange',
                'MPI_ESM': 'green',                 
                'TRACE': 'blue'}

fig, ax = plt.subplots()

plot_with_ci(IPSL_CM5_calendar, IPSL_CM5_data, colors['IPSL_CM5'], 'IPSL_CM5', ax, 0.905)
#plot_with_ci(IPSL_CM6_calendar, IPSL_CM6_data, colors['IPSL_CM6'], 'IPSL_CM6', ax, 0.95)
#plot_with_ci(MPI_ESM_calendar, MPI_ESM_data, colors['MPI_ESM'], 'MPI_ESM', ax, 0.95)
#plot_with_ci(TRACE_calendar, TRACE_data, colors['TRACE'], 'TRACE', ax, 0.95)
"""
ax.plot(IPSL_CM5_calendar, IPSL_CM5_data, label = 'IPSL_CM5', color = colors['IPSL_CM5'])
ax.plot(IPSL_CM6_calendar, IPSL_CM6_data, label = 'IPSL_CM6', color = colors['IPSL_CM6'])
ax.plot(MPI_ESM_calendar, MPI_ESM_data, label = 'MPI_ESM', color = colors['MPI_ESM'])
ax.plot(TRACE_calendar, TRACE_data, label = 'TRACE', color = colors['TRACE'])

#add line of best fit for each model


IPSL_CM5_slope, IPSL_CM5_intercept, _, _, _ = linregress(IPSL_CM5_calendar, IPSL_CM5_data)
IPSL_CM5_fit = [IPSL_CM5_slope*year + IPSL_CM5_intercept for year in IPSL_CM5_calendar]
ax.plot(IPSL_CM5_calendar, IPSL_CM5_fit, color = colors['IPSL_CM5'], linestyle='--')

IPSL_CM6_slope, IPSL_CM6_intercept, _, _, _ = linregress(IPSL_CM6_calendar, IPSL_CM6_data)
IPSL_CM6_fit = [IPSL_CM6_slope*year + IPSL_CM6_intercept for year in IPSL_CM6_calendar]
ax.plot(IPSL_CM6_calendar, IPSL_CM6_fit, color = colors['IPSL_CM6'], linestyle='--')

MPI_ESM_slope, MPI_ESM_intercept, _, _, _ = linregress(MPI_ESM_calendar, MPI_ESM_data)
MPI_ESM_fit = [MPI_ESM_slope*year + MPI_ESM_intercept for year in MPI_ESM_calendar]
ax.plot(MPI_ESM_calendar, MPI_ESM_fit, color = colors['MPI_ESM'], linestyle='--')

TRACE_slope, TRACE_intercept, _, _, _ = linregress(TRACE_calendar, TRACE_data)
TRACE_fit = [TRACE_slope*year + TRACE_intercept for year in TRACE_calendar]
ax.plot(TRACE_calendar, TRACE_fit, color = colors['TRACE'], linestyle='--')
"""

ax.set_xlabel('Year')
ax.set_ylabel('Skew')

#show legend
ax.legend()

#add title
ax.set_title('Skew of Anomaly Distribution')

# %%