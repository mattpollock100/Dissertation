#%%
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

#%%
# Step 1: Read the CSV file into a pandas DataFrame
csv_file = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/RadioCarbonDates_FromManuel.csv'
sites_df = pd.read_csv(csv_file)

# Ensure your CSV has 'latitude' and 'longitude' columns, otherwise adjust column names accordingly
sites_df['geometry'] = sites_df.apply(lambda row: Point(row['Lon'], row['Lat']), axis=1)
sites_gdf = gpd.GeoDataFrame(sites_df, geometry='geometry')

# Step 2: Read the shapefile of Peru boundaries into a GeoDataFrame
shapefile = 'C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/Shape Files/Peru/peru_administrative_boundaries_national_polygon.shp'
peru_gdf = gpd.read_file(shapefile)

# Set the CRS for sites_gdf assuming the coordinates are in WGS84
sites_gdf.set_crs(epsg=4326, inplace=True)

# Ensure both GeoDataFrames have the same coordinate reference system (CRS)
if sites_gdf.crs != peru_gdf.crs:
    sites_gdf = sites_gdf.to_crs(peru_gdf.crs)

# Step 3: Perform spatial join to filter sites within Peru
sites_within_peru = gpd.sjoin(sites_gdf, peru_gdf, how='inner', op='within')

# Step 4: Output the filtered sites
filtered_sites_df = pd.DataFrame(sites_within_peru.drop(columns='geometry'))
filtered_sites_df.to_csv('C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/RadioCarbonDates_FromManuel_PeruOnly.csv', index=False)

# If you want to keep the geometry, you can save it as a new shapefile
#sites_within_peru.to_file('filtered_archaeological_sites.shp')

# %%
