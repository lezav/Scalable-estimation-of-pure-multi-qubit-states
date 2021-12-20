# %% codecell
# %load_ext autoreload
# %autoreload 2
import numpy as np
from core.tresbases import tomography, bases_separables
from core.utils import estado, fidelidad
import time
# preparamos las bases y el estado
n_qubits = 2
dim = int(2**n_qubits)
psi_sistema = estado(dim, 1, seed=None)
psi_sistema = psi_sistema/np.linalg.norm(psi_sistema)
v_a = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
v_b = np.sqrt(1 - v_a**2)
v_fase = np.array([0, np.pi/2])
base_diag, bases_sep = bases_separables(dim, v_a, v_b, v_fase)

# medimos el estado usando las bases
n_bases = bases_sep.shape[3]
fid = np.zeros((dim, n_qubits, n_bases))
fid_0 = fidelidad(psi_sistema, base_diag, nu_exp = None)
for k in range(n_bases):
    for j in range(n_qubits):
        fid[:, j, k] = fidelidad(psi_sistema, bases_sep[:, :, j, k], nu_exp = None)


# las bases deben tener 4 dimensiones. dim x dim x n_qubits x n_bases
# n_qubits es el numero de bases separables que equivalen a una base entralazada
# y n_bases es el numero de bases entrelazadas equivalente.
# Deben estar ordenadas en potencias descendentes de 2. Por ejemplo, en
# bases[:, :, 0, :] deben estar los proyectores de dimension 2^n_qubits, y
# de ahi descander hasta bases[:, :, n_qubits, :] con proyectores de dimension 2
# #tomografia
start = time.time()
psi, lamb = tomography(fid_0, fid, bases_sep, pur=True)
end = time.time()
# print(end - start)
print("fid", np.abs(np.dot(psi.conj(), psi_sistema))**2)
