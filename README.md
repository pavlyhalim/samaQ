
#SamaQ

## Introduction
This Streamlit application visualizes global fishing efforts and port connectivity using geospatial data analysis and interactive mapping. It uses data clustering to identify high-activity fishing areas and allows users to dynamically add new port information.

## Requirements
- Python 3.8+
- streamlit
- pandas
- geopandas
- folium
- scikit-learn
- matplotlib

## Installation
First, clone the repository or download the project. Then install the required Python libraries:
```bash
pip install streamlit pandas geopandas folium scikit-learn matplotlib
```

## Usage
Run the application by navigating to the project directory and executing:
```bash
streamlit run app.py
```
The web interface allows users to interact with the data, visualize fishing activities, and manage port information.

## Features
- Interactive mapping of fishing activities and port locations.
- Clustering of fishing data to identify key fishing zones.
- Capability to add new ports through the user interface.

## Data Handling
Data for fishing activities and ports is loaded from CSV files. Users can add new ports, which are saved back to the CSV to persist changes across sessions.

## License
This project is released under the MIT License.
