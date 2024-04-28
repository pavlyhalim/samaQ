# Implementation of the QUBO solver for the TSP problem
# Julien-Pierre Houle
# Team 1 - NYUAD Quantum Hackathon


#%%
# from bqm_utils import tsp_viz
import networkx as nx
import numpy as np
import dimod

# G = nx.Graph()
# G.add_weighted_edges_from(
#     [(0, 1, 10), (1, 0, 15), (0, 2, 7), (2, 0, 14), (1, 2, 9), (2, 1, 8)]
# )
# path = [0,2,1]
# tsp_viz(G,path)


#%%

def qubo_solver(Q_matrix):
    """ Classical QUBO solver, where each permutation is try. """

    possible_values = {}
    # A list of all the possible permutations for x vector
    vec_permutations = itertools.product([0, 1], repeat=Q_matrix.shape[0])    
    
    for permutation in vec_permutations:
        x = np.array([[var] for var in permutation])         # Converts the permutation into a column vector
        value = (x.T).dot(Q_matrix).dot(x)
        possible_values[value[0][0]] = x                     # Adds the value and its vector to the dictionary
         
    min_value = min(possible_values.keys())                  # Lowest value of the objective function
    opt_vector = tuple(possible_values[min_value].T[0])      # Optimum vector x that produces the lowest value
     
    return f"The vector {opt_vector} minimizes the objective function to a value of {min_value}."







def tsp_matrix(n, W, P):
    # Initialize an empty matrix
    Q = np.zeros((n*n,n*n))

    for i in range(n*n):
        for j in range(i,n*n):
            # Diagonals
            if i==j:
                Q[i][j]=-2*P
            else:
                # If share the same node or time point
                if i//n == j//n or i%n == j%n:
                    Q[i][j]=2*P
                # If entries in row i and column j correspond to consecutive time points
                elif j%n == i%n +1 or i%n == n-1 and j%n == 0:
                    Q[i][j]=W[i//n][j//n]
                elif j%n == i%n -1 or i%n == 0  and j%n == n-1:
                    Q[i][j]=W[j//n][i//n]
    return Q


nodes = 3 # number of nodes
W = np.array([[0,10,7],
             [15,0,9],
             [14,8,0]])

Q = tsp_matrix(nodes, W, P=20)

# %%
# Convert the QUBO formulation to Ising model
dict_coefs = {}
for i in range(nodes**2):
    for j in range(nodes**2):
        dict_coefs[(i, j)] = Q[i, j]

ising_lin, ising_quad, ising_ens = dimod.utilities.qubo_to_ising(dict_coefs)
# %%
# Get the hamiltonian from the Ising terms

hamil_ising = np.zeros((nodes**2, nodes**2))

for k in ising_lin: # filling the linear terms
    hamil_ising[k, k] = ising_lin[k]

for k in ising_quad: # filling the quadratics terms
    hamil_ising[k[0], k[1]] = ising_quad[(k[0], k[1])]

print(hamil_ising)
# %%
