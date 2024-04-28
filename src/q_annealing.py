#%%
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

# from bqm_utils import graph_viz, tsp_viz

#%%
# # Parameters
# nodes = 8

# # G1 = nx.Graph()
# # G1.add_weighted_edges_from(
# #     {(0, 1, 0.1), (0, 2, 0.5), (0, 3, 0.1), (1, 2, 0.1), (1, 3, 0.5), (2, 3, 0.1)}
# # )

# G1 = nx.complete_graph(nodes)
# for u, v in G1.edges():
#     G1[u][v]["weight"] = np.random.randint(1, 50)
# graph_viz(G1)
# N = len(G1.nodes)



# %%

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





bqm = tsp_bqm(G1, 7) # bqm formulation

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
