# codecell
import numpy as np
from core.tresbases_vec import bases_2_3, bases_separables_vec, tomography_vec
from core.utils import estado, estado_sep, fidelidad_vec, dot_prod_vec
import time

n_qubits = 3
dim = int(2**n_qubits)
nu_exp = 2**23
psi_sistema = estado(dim, 1, seed=None)
psi_sistema = psi_sistema/np.linalg.norm(psi_sistema)
# 8 bases
v_a = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
v_fase = np.array([0, np.pi/2])
v_b = np.sqrt(1 - v_a**2)
n_bases = v_b.shape[0]
# elegimos la base separable
base_diag_vec, bases_sep_vec = bases_separables_vec(dim, v_a, v_b, v_fase)
# calculamos probabilidades
start = time.time()
prob_sep_vec = np.zeros((dim, n_qubits, n_bases))
prob_diag_vec = fidelidad_vec(psi_sistema, base_diag_vec, nu_exp = nu_exp)
for k in range(n_bases):
    for j in range(n_qubits):
        prob_sep_vec[:, j, k] = fidelidad_vec(psi_sistema,
                                              bases_sep_vec[:, :, :, j, k],
                                              nu_exp=nu_exp)
end = time.time()
# print("time", end - start)
# tomografia
fid = 0
start = time.time()
for k in range(4):
    psi = tomography_vec(prob_diag_vec, prob_sep_vec, bases_sep_vec)
    # fid = fid + np.abs(np.dot(psi_sistema.conj().T, psi))**2
end = time.time()
print("time", end - start)
print("fid", np.abs(np.dot(psi_sistema.conj().T, psi))**2)
