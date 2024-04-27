import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

fishing_df = pd.read_csv('public-global-fishing-effort-v20231026.csv').head(20)
fishing_df['Lat'] = pd.to_numeric(fishing_df['Lat'], errors='coerce')
fishing_df['Lon'] = pd.to_numeric(fishing_df['Lon'], errors='coerce')

ports_df = pd.read_csv('ports_persian_gulf.csv')

ports_df['lat'] = pd.to_numeric(ports_df['lat'], errors='coerce')
ports_df['lon'] = pd.to_numeric(ports_df['lon'], errors='coerce')

fishing_gdf = gpd.GeoDataFrame(fishing_df, geometry=gpd.points_from_xy(fishing_df.Lon, fishing_df.Lat))
ports_gdf = gpd.GeoDataFrame(ports_df, geometry=gpd.points_from_xy(ports_df.lon, ports_df.lat))

num_clusters = 10 
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(fishing_gdf[['Lat', 'Lon']])
fishing_gdf['cluster'] = kmeans.labels_

map_fishing = folium.Map(location=[fishing_gdf['Lat'].mean(), fishing_gdf['Lon'].mean()], zoom_start=7)


HeatMap(data=fishing_gdf[['Lat', 'Lon']], radius=20).add_to(map_fishing)


colors = plt.get_cmap('viridis', num_clusters)

for idx, row in fishing_gdf.iterrows():
    folium.CircleMarker(
        location=[row['Lat'], row['Lon']],
        radius=5,
        fill=True,
        color=colors(row['cluster']),
        fill_color=colors(row['cluster']),
        popup=f"Cluster: {row['cluster']}"
    ).add_to(map_fishing)

for i, point_i in fishing_gdf.iterrows():
    for j, point_j in fishing_gdf.iterrows():
        if i != j:
            folium.PolyLine(locations=[(point_i['Lat'], point_i['Lon']), (point_j['Lat'], point_j['Lon'])],
                            color='grey', weight=0.5, opacity=0.5).add_to(map_fishing)

for _, fish_row in fishing_gdf.iterrows():
    for _, port_row in ports_gdf.iterrows():
        folium.PolyLine(locations=[(fish_row['Lat'], fish_row['Lon']), (port_row['lat'], port_row['lon'])],
                        color='red', weight=1, opacity=0.5).add_to(map_fishing)

for idx, row in ports_gdf.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=f"{row['port_name']}",
        icon=folium.Icon(color='red', icon='anchor')
    ).add_to(map_fishing)

map_fishing.save('fishing_ports_connectivity_map.html')
print("Map with fishing spots and detailed connectivity has been created.")


cluster_centers = kmeans.cluster_centers_

centers_df = pd.DataFrame(cluster_centers, columns=['Latitude', 'Longitude'])

centers_df.to_csv('cluster_centers.csv', index=False)

print("Cluster centers have been saved to 'cluster_centers.csv'.")

danger_zone_coords = [
    [24.56, 51.6],
    [24.7, 52.28],
    [24.348, 52.38],
    [24.19, 51.9]
]

folium.Polygon(
    locations=danger_zone_coords,
    color='red',  
    fill=True,
    fill_color='red',
    fill_opacity=0.5, 
    popup='Danger Zone'
).add_to(map_fishing)


map_fishing.save('fishing_ports_connectivity_map.html')
print("Map with fishing spots, detailed connectivity, and danger zone has been created.")
