import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load fishing data from CSV and limit to the first 20 entries
fishing_df = pd.read_csv('/Users/pavly/Downloads/1_-_2023-05-01T00_00_00.000Z2024-05-01T00_00_00.000Z/layer-activity-data-0/public-global-fishing-effort-v20231026.csv').head(20)

# Ensure the correct types for Lat and Lon, if not automatically detected
fishing_df['Lat'] = pd.to_numeric(fishing_df['Lat'], errors='coerce')
fishing_df['Lon'] = pd.to_numeric(fishing_df['Lon'], errors='coerce')

# Load ports data from CSV
ports_df = pd.read_csv('/Users/pavly/Downloads/26c10160-03bd-11ef-8dfc-7dff0eda9712/ports_persian_gulf.csv')

# Ensure the correct types for Lat and Lon in ports data
ports_df['lat'] = pd.to_numeric(ports_df['lat'], errors='coerce')
ports_df['lon'] = pd.to_numeric(ports_df['lon'], errors='coerce')

# Create GeoDataFrames
fishing_gdf = gpd.GeoDataFrame(fishing_df, geometry=gpd.points_from_xy(fishing_df.Lon, fishing_df.Lat))
ports_gdf = gpd.GeoDataFrame(ports_df, geometry=gpd.points_from_xy(ports_df.lon, ports_df.lat))

# Set the number of clusters and apply KMeans clustering
num_clusters = 10  # Adjust the number of clusters here
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(fishing_gdf[['Lat', 'Lon']])
fishing_gdf['cluster'] = kmeans.labels_

# Create a base map centered around the mean location of the fishing data
map_fishing = folium.Map(location=[fishing_gdf['Lat'].mean(), fishing_gdf['Lon'].mean()], zoom_start=7)

# Add HeatMap for fishing spots
HeatMap(data=fishing_gdf[['Lat', 'Lon']], radius=20).add_to(map_fishing)

# Color map for clusters
colors = plt.get_cmap('viridis', num_clusters)

# Add clustered points to the map with color coding
for idx, row in fishing_gdf.iterrows():
    folium.CircleMarker(
        location=[row['Lat'], row['Lon']],
        radius=5,
        fill=True,
        color=colors(row['cluster']),
        fill_color=colors(row['cluster']),
        popup=f"Cluster: {row['cluster']}"
    ).add_to(map_fishing)

# Draw red lines between each node to visualize full mesh connectivity
for i, point_i in fishing_gdf.iterrows():
    for j, point_j in fishing_gdf.iterrows():
        if i != j:
            folium.PolyLine(locations=[(point_i['Lat'], point_i['Lon']), (point_j['Lat'], point_j['Lon'])],
                            color='grey', weight=0.5, opacity=0.5).add_to(map_fishing)

# Draw grey lines from each fishing spot to every port
for _, fish_row in fishing_gdf.iterrows():
    for _, port_row in ports_gdf.iterrows():
        folium.PolyLine(locations=[(fish_row['Lat'], fish_row['Lon']), (port_row['lat'], port_row['lon'])],
                        color='red', weight=1, opacity=0.5).add_to(map_fishing)

# Add ports to the map with markers
for idx, row in ports_gdf.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=f"{row['port_name']}",
        icon=folium.Icon(color='red', icon='anchor')
    ).add_to(map_fishing)

# Save or show the map
map_fishing.save('fishing_ports_connectivity_map.html')
print("Map with fishing spots and detailed connectivity has been created.")


# Extracting cluster centers
cluster_centers = kmeans.cluster_centers_

# Creating a DataFrame for cluster centers
centers_df = pd.DataFrame(cluster_centers, columns=['Latitude', 'Longitude'])

# Saving the DataFrame to a CSV file
# centers_df.to_csv('cluster_centers.csv', index=False)

print("Cluster centers have been saved to 'cluster_centers.csv'.")

# Danger zone coordinates
danger_zone_coords = [
    [24.56, 51.6],
    [24.7, 52.28],
    [24.348, 52.38],
    [24.19, 51.9]
]

# Add danger zone as a red polygon
folium.Polygon(
    locations=danger_zone_coords,
    color='red',  # Line color
    fill=True,
    fill_color='red',
    fill_opacity=0.5,  # Semi-transparent fill
    popup='Danger Zone'
).add_to(map_fishing)

# Save the map
map_fishing.save('fishing_ports_connectivity_map.html')
print("Map with fishing spots, detailed connectivity, and danger zone has been created.")