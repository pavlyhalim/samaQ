# VQE quantum solver to minimize the Hamiltonian

#%%

import numpy as np

# from qiskit.opflow import Z, I, X

from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# from qiskit.providers.aer import QasmSimulator  
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA

# runtime imports
# from qiskit_ibm_runtime import QiskitRuntimeService, Session
# from qiskit_ibm_runtime import EstimatorV2 as Estimator

# To run on hardware, select the backend with the fewest number of jobs in the queue
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.least_busy(operational=True, simulator=False)
backend.name


# runtime imports
# from qiskit_ibm_runtime import QiskitRuntimeService, Session
# from qiskit_ibm_runtime import EstimatorV2 as Estimator

#%%
# To run on hardware, select the backend with the fewest number of jobs in the queue
# service = QiskitRuntimeService(channel="ibm_quantum")
# backend = service.least_busy(operational=True, simulator=False)



hamil = np.array([[31.5 , 10.  , 10.  , 10.  ,  2.5 ,  3.75, 10.  ,  1.75,  3.5 ],
       [ 0.  , 31.5 , 10.  ,  3.75, 10.  ,  2.5 ,  3.5 , 10.  ,  1.75],
       [ 0.  ,  0.  , 31.5 ,  2.5 ,  3.75, 10.  ,  1.75,  3.5 , 10.  ],
       [ 0.  ,  0.  ,  0.  , 30.5 , 10.  , 10.  , 10.  ,  2.25,  2.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  , 30.5 , 10.  ,  2.  , 10.  ,  2.25],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  , 30.5 ,  2.25,  2.  , 10.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , 29.5 , 10.  , 10.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , 29.5 , 10.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , 29.5 ]])

pauli_list = []

for i in range(hamil.shape[0]):
    for j in range(hamil.shape[1]):
        if i == j:
            pauli_list.append(("Z", [i], hamil[i, j]))
        else:
            pauli_list.append(("ZZ", [i, j], hamil[i, j]))

H_op = SparsePauliOp.from_sparse_list(pauli_list, num_qubits=9)


# %%
# you can swap this for a real quantum device and keep the rest of the code the same!
backend = QasmSimulator() 
optimizer = COBYLA(maxiter=200)
ansatz = RealAmplitudes(3, reps=2)
ansatz.decompose().draw("mpl", style="iqp")

ansatz_isa.draw(output="mpl", idle_wires=False, style="iqp")

hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)


from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)

ansatz_isa = pm.run(ansatz)

def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    return energy


h = 0.25  # or whatever value you have for h
H = -(Z ^ Z) - h * ((X ^ I) + (I ^ X))



#%%
# set the algorithm
vqe = VQE(ansatz, optimizer, quantum_instance=backend)


# run it with the Hamiltonian we defined above
result = vqe.compute_minimum_eigenvalue(PrimitiveOp(H_op))  


# print the result (it contains lot's of information)
print(result) 

# %%
