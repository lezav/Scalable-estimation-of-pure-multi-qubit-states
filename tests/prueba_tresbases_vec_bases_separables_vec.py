import numpy as np
from core.tresbases import bases_2_3, bases_separables
from core.tresbases_vec import bases_separables_vec
from core.utils import estado, estado_sep


n_qubits = 2
dim = int(2**n_qubits)
v_a = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
v_fase = np.array([0, np.pi/2])
v_b = np.sqrt(1 - v_a**2)
# elegimos la base separable
base_diag_vec, bases_vec = bases_separables_vec(dim, v_a, v_b, v_fase)
base_diag, bases = bases_separables(dim, v_a, v_b, v_fase)


base_k = 0
base_qubits = 1
A = bases_vec[:, :, 0, base_qubits, base_k]
for k in range(1, n_qubits):
    A = np.kron(A, bases_vec[:, :, k, base_qubits, base_k])

# comparamos los resultados calculados de forma eficiente con los calculados
# normalmente
A - bases[:, :, base_qubits, base_k]
