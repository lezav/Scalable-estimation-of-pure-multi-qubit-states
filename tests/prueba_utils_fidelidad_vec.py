# %% codecell
# %load_ext autoreload
# %autoreload 2
import numpy as np
from core.tresbases_vec import bases_2_3, bases_separables_vec
from core.tresbases import bases_separables
from core.utils import estado, estado_sep, fidelidad_vec, fidelidad
import time

n_qubits = 10
n_bases = 2
nu_exp = 100000
dim = int(2**n_qubits)
psi_sistema = estado(dim, 1, seed=None)
v_a = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
v_fase = np.array([0, np.pi/2])
v_b = np.sqrt(1 - v_a**2)
# elegimos la base separable
base_diag_vec, bases_sep_vec = bases_separables_vec(dim, v_a, v_b, v_fase)
base_diag, bases_sep = bases_separables(dim, v_a, v_b, v_fase)
# calculamos la diferencia en tiempo
start1 = time.time()
fid = fidelidad(psi_sistema, bases_sep[:, :, 0, 1])
end1 = time.time()

start2 = time.time()
fid_vec = fidelidad_vec(psi_sistema, bases_sep_vec[:, :, :, 0, 1])
end2 = time.time()

print("normal", end1 - start1, "eficiente", end2 - start2)

# calculamos si los valores son correctos
# bases normales
fid = np.zeros((dim, n_qubits, n_bases))
fid_0 = fidelidad(psi_sistema, base_diag)
for k in range(n_bases):
    for j in range(n_qubits):
        fid[:, j, k] = fidelidad(psi_sistema, bases_sep[:, :, j, k],
                                 nu_exp = nu_exp)

# bases eficientes
prob_sep_vec = np.zeros((dim, n_qubits, n_bases))
prob_diag_vec = fidelidad_vec(psi_sistema, base_diag_vec)
for k in range(n_bases):
    for j in range(n_qubits):
        prob_sep_vec[:, j, k] = fidelidad_vec(psi_sistema,
                                              bases_sep_vec[:, :, :, j, k],
                                              nu_exp=nu_exp)

print("dif prob sim", fid[:, 1, 1] -  prob_sep_vec[:, 1, 1])
