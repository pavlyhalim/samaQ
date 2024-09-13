Quantum Annealing solver for the NYUAD Quantum Hackathon 2024.

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

## Quantum Solution Implementation
As part of this project, we also explored a quantum computing approach to solve the Traveling Salesman Problem (TSP). This implementation utilizes a Quadratic Unconstrained Binary Optimization (QUBO) formulation of the problem, which is suitable for quantum annealers.

### Key Components
- **QUBO Matrix Construction**: The TSP is converted into a QUBO matrix, where each entry represents the cost associated with traveling between cities, adjusted by penalties to enforce constraints.
- **Quantum Solver**: We employ a classical simulation of a quantum QUBO solver. This involves testing all possible states to find the one that minimizes the objective function, simulating what a quantum annealer would do.
- **Ising Model Conversion**: The QUBO formulation is converted into an Ising model. This is done to prepare the problem for solving on quantum annealers, which typically solve problems expressed in terms of Ising models.

### Technologies Used
- `networkx`: For graph operations and structures.
- `numpy`: For matrix operations and numerical calculations.
- `dimod`: A library provided by D-Wave Systems that supports the creation and manipulation of QUBO and Ising models.

This quantum solution showcases the potential of quantum computing to solve optimization problems efficiently, providing a cutting-edge alternative to classical algorithms.

## License
This project is released under Apache License
