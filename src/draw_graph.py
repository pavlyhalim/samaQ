# Script to draw graph
# QHack 2024
# Julien-Pierre Houle

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import itertools
from collections import defaultdict

import dimod
from dimod import BQM
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from neal import SimulatedAnnealingSampler
from dimod.reference.samplers import ExactSolver

from bqm_utils import graph_viz, tsp_viz
#%%

# Load the data from file
lat_cluster, lon_cluster = np.loadtxt("data/cluster_centers.csv", delimiter=",", skiprows=1).T
lat_port, lon_port = np.loadtxt("data/ports_persian_gulf.csv", delimiter=",", skiprows=1, usecols=[1,2]).T
lats = np.array(list(lat_cluster) + list(lat_port))
lons = np.array(list(lon_cluster) + list(lon_port))


#%%
G1 = nx.Graph()
for i in range(len(lats)):
    for j in range(len(lons)):
        if i != j:
            print('Distance:', np.sqrt((lats[i]-lats[j])**2 + (lons[i]-lons[j])**2))
            G1.add_weighted_edges_from({ (i, j, np.sqrt((lats[i]-lats[j])**2 + (lons[i]-lons[j])**2)) })


dict_pos = {}
for idx in range(len(lats)):
    dict_pos[idx] = (lons[idx], lats[idx])



nx.draw(G1, pos=dict_pos, with_labels=False, alpha=1.0, style='--', node_color='white', edgecolors='k')
plt.show()



#%%


def tsp_bqm(G, P):
    """ Create a Binary Quadratic Model from a graph.

    Args:
        G (_type_): _description_
        P (_type_): _description_

    Returns:
        BQM object defining the graph.
    """
    N = len(G.nodes) # nb of nodes
    bqm = BQM("BINARY")
    for i in range(N):
        for j in range(N):
            if i != j:
                for t in range(N - 1):
                    bqm.add_quadratic(f"x_{i}_{t}", f"x_{j}_{t+1}", G[i][j]["weight"])

                # Remember that we were assuming N=0 in the sum
                bqm.add_quadratic(f"x_{i}_{N-1}", f"x_{j}_{0}", G[i][j]["weight"])
    # Add the first constraint
    for t in range(N):
        c1 = [(f"x_{i}_{t}", 1) for i in range(N)]  # coefficient list
        bqm.add_linear_equality_constraint(c1, constant=-1, lagrange_multiplier=P)
    # Add the second constraint
    for i in range(N):
        c2 = [(f"x_{i}_{t}", 1) for t in range(N)]
        bqm.add_linear_equality_constraint(c2, constant=-1, lagrange_multiplier=P)
    return bqm
    

def is_sample_feasible(sample, N):
    """ Make sure that the graph can be solved. """
    for i in range(N):
        if sum(sample[f"x_{i}_{t}"] for t in range(N)) != 1:
            return False
    for t in range(N):
        if sum(sample[f"x_{i}_{t}"] for i in range(N)) != 1:
            return False
    return True


def sample_to_path(sample, N):
    path = []
    for t in range(N):
        for i in range(N):
            if sample[f"x_{i}_{t}"] == 1:
                path.append(i)
    return path



N = len(G1.nodes)

print('Quantum Calculation In Progress...')
bqm = tsp_bqm(G1, 2000) # bqm formulation

sampler = SimulatedAnnealingSampler() # quantum solver
sampleset = sampler.sample(bqm, num_reads=1000)

first_sample = sampleset.first.sample
is_sample_feasible(first_sample, N)

path = sample_to_path(first_sample,N)
tsp_viz(G1, path)
plt.show()
plt.savefig("path.png")

print(f"Path: {path}")
print(f"Total cost: {sampleset.first.energy}")


# %%

nx.draw(G1, pos=dict_pos, with_labels=False, alpha=1, style='--', node_color='white', edgecolors='k')
# plt.savefig(f"figs/graph_fishing_spots", transparent=True)
# plt.show()


edges_sol = [(path[i], path[i+1]) for i in range(len(path[:-1]))]
edges_sol.append((path[-1], path[0]))


nx.draw_networkx_edges(G1, pos=dict_pos, edgelist=edges_sol, edge_color='r', width=4)
plt.savefig(f"figs/graph_solution", transparent=True)
plt.show()

# %%
