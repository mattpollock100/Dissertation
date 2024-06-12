#%%
import matplotlib.pyplot as plt
import pandas as pd

far_path = "C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/spd_result_far.csv"
close_path = "C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/spd_result_close.csv"

spd_far = pd.read_csv(far_path)
spd_close = pd.read_csv(close_path)


# %%
#plot the PrDens from spd_far and spd_close vs calBP on the same graph

plt.plot(spd_far["calBP"], spd_far["PrDens"], label="Non-Coastal")
plt.plot(spd_close["calBP"], spd_close["PrDens"], label="Coastal")
plt.legend()

#reverse the x-axis
plt.gca().invert_xaxis()
# %%

#create a new plot of the difference Far - Close
plt.plot(spd_far["calBP"], spd_far["PrDens"] - spd_close["PrDens"], label="Non-Coastal - Coastal", color='black')
plt.gca().invert_xaxis()
#add a horizontal line at 0
plt.axhline(0, color='black', lw=1)

#shade above the horizontal line as pale orange, below the line as pale blue
plt.fill_between(spd_far["calBP"], spd_far["PrDens"] - spd_close["PrDens"], 0, 
                 where=(spd_far["PrDens"] - spd_close["PrDens"]) >= 0, 
                 facecolor='orange', alpha=0.5)
plt.fill_between(spd_far["calBP"], spd_far["PrDens"] - spd_close["PrDens"], 0, 
                 where=(spd_far["PrDens"] - spd_close["PrDens"]) < 0, 
                 facecolor='blue', alpha=0.5)

# %%
