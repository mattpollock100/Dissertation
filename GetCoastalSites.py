#%%
print('import the libraries')
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

#%%
print('define the parameters')
#max_distance_km = 50
distance_step = 20
distance_limit = 100
# Load the sites CSV file
sites = pd.read_csv("C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/RadioCarbonDatesPeru.csv")

# Convert sites DataFrame to a GeoDataFrame
sites_gdf = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites.Long, sites.Lat))

#%%
print('load the coastline shapefile')
# Load the coastline shapefile
coastline_gdf = gpd.read_file("C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/Shape Files/ne_10m_coastline.shp")

# Ensure both GeoDataFrames use the same coordinate reference system (CRS)
sites_gdf = sites_gdf.set_crs("EPSG:4326")
coastline_gdf = coastline_gdf.to_crs("EPSG:4326")

# Convert the GeoDataFrames to a UTM CRS
#sites_gdf = sites_gdf.set_crs("EPSG:32718")  # UTM Zone 18S
#coastline_gdf = coastline_gdf.to_crs("EPSG:32718")  # UTM Zone 18S

# Calculate the minimum distance from each site to the coastline
sites_gdf['distance_to_coast'] = sites_gdf.geometry.apply(lambda site: coastline_gdf.distance(site).min())

#%%
print('loop through the distances')
for d in range(0, distance_limit + distance_step, distance_step):
    # Filter the sites based on the maximum distance to the coastline
    print(str(d) +'km')
    min_distance_km = d
    max_distance_km = d + distance_step
    if d == distance_limit:
        max_distance_km = 99999
    sites_slice = sites_gdf[(sites_gdf['distance_to_coast'] < max_distance_km / 111) & 
                            (sites_gdf['distance_to_coast'] >= min_distance_km / 111)]  # Convert km to lat/lon degrees

    # Drop the geometry column if not needed
    sites_slice = sites_slice.drop(columns=['geometry'])

    # Save the filtered sites to a new CSV file
    sites_slice.to_csv('C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/SPD/sites_slice_' + str(min_distance_km) + '_' + str(max_distance_km) + '.csv', index=False)
    print(sites_slice.shape[0])
#%%
if False:
    # Filter the sites based on the maximum distance to the coastline
    sites_close_to_coast = sites_gdf[sites_gdf['distance_to_coast'] <= max_distance_km / 111]  # Convert km to lat/lon degrees

    # Drop the geometry column if not needed
    sites_close_to_coast = sites_close_to_coast.drop(columns=['geometry'])

    # Save the filtered sites to a new CSV file
    sites_close_to_coast.to_csv('C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/sites_close_to_peruvian_coast_' + str(max_distance_km) + '.csv', index=False)


#%%
if False:
    # Filter the sites based on the maximum distance to the coastline
    sites_far_from_coast = sites_gdf[sites_gdf['distance_to_coast'] > max_distance_km / 111]  # Convert km to lat/lon degrees

    # Drop the geometry column if not needed
    sites_far_from_coast = sites_far_from_coast.drop(columns=['geometry'])

    # Save the filtered sites to a new CSV file
    sites_far_from_coast.to_csv('C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/sites_far_from_peruvian_coast_' + str(max_distance_km) + '.csv', index=False)

# %%
