#%%

import pandas as pd
import folium

from folium.plugins import MarkerCluster

#%%
# Load the CSV file into a DataFrame
#df_raw = pd.read_csv("C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/RadioCarbonDatesPeru.csv")

df_raw = pd.read_csv("C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/sites_close_to_peruvian_coast.csv")

#remove the lines which dont have a lat or long
df = df_raw.dropna(subset=['Lat', 'Long'])

#%%



# Create a map centered at the mean latitude and longitude values
m = folium.Map(location=[df['Lat'].mean(), df['Long'].mean()], zoom_start=10)

# Create a MarkerCluster
marker_cluster = MarkerCluster().add_to(m)

# Add a marker for each latitude and longitude to the MarkerCluster
for _, row in df.iterrows():
    folium.Marker([row['Lat'], row['Long']]).add_to(marker_cluster)

# Display the map
m
# %%
