# %% codecell
# %load_ext autoreload
# %autoreload 2
import numpy as np
from core.tresbases import tomography, bases_ent
from core.utils import estado, fidelidad
import time
# preparamos las bases y el estado
n_qubits = 2
dim = int(2**n_qubits)

psi_sistema = estado(dim, 1, seed=10)
psi_sistema = psi_sistema*np.exp(-1j*np.angle(psi_sistema[0]))
# caracteristicas de las bases
v_a = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
v_b = np.sqrt(1 - v_a**2)
v_fase = np.array([0, np.pi/2])
base_diag, bases_ent = bases_ent(dim, v_a, v_b, v_fase)
n_bases = v_a.shape[0]
# definimos los array donde se almacenan los datos y simulamos mediciones
fid = np.zeros((dim, n_bases))
fid_0 = fidelidad(psi_sistema, base_diag, nu_exp = 8200000)
for k in range(n_bases):
    fid[:, k] = fidelidad(psi_sistema, bases_ent[:, :, k], nu_exp = 8200000)

# las bases deben tener entrelazadas deben tener 3 dimensiones. dim x dim x n_bases
# Las primeras columnas de la base deben tener los proyectores de dimension dim,
# mientras que los de dimension 2 deben estar al final. Por ejemplo, para dos
# qubits, una base seria
# [x x x 0]
# [x x x 0]
# [x x 0 x]
# [x x 0 x]

# tomografia
start = time.time()
psi, lamb = tomography(fid_0, fid, bases_ent, pur=True)
end = time.time()
print("tiempo", end - start)
print("fid", np.abs(np.dot(psi.conj(), psi_sistema))**2)
