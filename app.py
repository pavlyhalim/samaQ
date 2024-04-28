import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from streamlit_folium import folium_static

def load_data(num_entries):
    fishing_df = pd.read_csv('data/public-global-fishing-effort-v20231026.csv').head(num_entries)
    ports_df = pd.read_csv('data/ports_persian_gulf.csv')
    return fishing_df, ports_df

def process_data(fishing_df, ports_df):
    fishing_df['Lat'] = pd.to_numeric(fishing_df['Lat'], errors='coerce')
    fishing_df['Lon'] = pd.to_numeric(fishing_df['Lon'], errors='coerce')
    ports_df['lat'] = pd.to_numeric(ports_df['lat'], errors='coerce')
    ports_df['lon'] = pd.to_numeric(ports_df['lon'], errors='coerce')
    fishing_gdf = gpd.GeoDataFrame(fishing_df, geometry=gpd.points_from_xy(fishing_df.Lon, fishing_df.Lat))
    ports_gdf = gpd.GeoDataFrame(ports_df, geometry=gpd.points_from_xy(ports_df.lon, ports_df.lat))
    return fishing_gdf, ports_gdf

def save_ports(ports_df):
    ports_df.to_csv('data/ports_persian_gulf.csv', index=False)

def create_map(fishing_gdf, ports_gdf, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(fishing_gdf[['Lat', 'Lon']])
    fishing_gdf['cluster'] = kmeans.labels_
    map_fishing = folium.Map(location=[fishing_gdf['Lat'].mean(), fishing_gdf['Lon'].mean()], zoom_start=7)
    HeatMap(data=fishing_gdf[['Lat', 'Lon']], radius=20).add_to(map_fishing)
    colors = plt.get_cmap('viridis', num_clusters)

    cluster_centers = kmeans.cluster_centers_
    for center in cluster_centers:
        folium.CircleMarker(
            location=center,
            radius=7,
            color='black',
            fill=True,
            fill_color='black',
            popup='Cluster Center'
        ).add_to(map_fishing)

    for idx, row in fishing_gdf.iterrows():
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=5,
            fill=True,
            color=colors(row['cluster']),
            fill_color=colors(row['cluster']),
            popup=f"Cluster: {row['cluster']}"
        ).add_to(map_fishing)

    for center in cluster_centers:
        for _, point in fishing_gdf.iterrows():
            folium.PolyLine(locations=[center, [point['Lat'], point['Lon']]], color='grey', weight=0.5).add_to(map_fishing)
        for _, port in ports_gdf.iterrows():
            folium.PolyLine(locations=[center, [port['lat'], port['lon']]], color='red', weight=1).add_to(map_fishing)

    for idx, row in ports_gdf.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=row['port_name'],
            icon=folium.Icon(color='red', icon='anchor')
        ).add_to(map_fishing)

    return map_fishing

def app():
    st.title('Fishing Activity and Port Connectivity Map')
    num_entries = st.sidebar.number_input('Number of Data Entries', min_value=1, max_value=100, value=20, step=1)
    fishing_df, ports_df = load_data(num_entries)
    fishing_gdf, ports_gdf = process_data(fishing_df, ports_df)
    
    num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=20, value=10)

    port_name = st.sidebar.text_input('Port Name', key='1')
    lat = st.sidebar.number_input('Latitude', format="%.5f", key='2')
    lon = st.sidebar.number_input('Longitude', format="%.5f", key='3')

    if st.sidebar.button('Add Port'):
        if port_name and lat and lon:
            new_port = pd.DataFrame([[port_name, lat, lon]], columns=['port_name', 'lat', 'lon'])
            ports_df = pd.concat([ports_df, new_port], ignore_index=True)
            save_ports(ports_df)
            st.sidebar.success('Port added and saved successfully!')
            fishing_gdf, ports_gdf = process_data(fishing_df, ports_df)

    map_fishing = create_map(fishing_gdf, ports_gdf, num_clusters)
    folium_static(map_fishing)

if __name__ == '__main__':
    app()
