# %% codecell
# %load_ext autoreload
# %autoreload 2
import numpy as np
from core.tresbases_vec import simulacion_vec, bases_separables_vec
from core.utils import estado, estado_sep
import time
from joblib import Parallel, delayed
# preparamos las bases y el estado
n_qubits = 10
nu_exp = 2**13
dim = int(2**n_qubits)
nu_estados = 100
psi_sistema = estado(dim, nu_estados, seed=None)
psi_sistema = psi_sistema*np.exp(-1j*np.angle(psi_sistema[0]))
# 2, 4 u 8 bases
v_a = np.array([1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)])
v_fase = np.array([0, np.pi/2, np.pi/3, np.pi/4])
v_b = np.sqrt(1 - v_a**2)
# elegimos la base entrelazada o la separable
base_diag_vec, bases_sep_vec = bases_separables_vec(dim, v_a, v_b, v_fase)

datos = np.zeros((9, 3, nu_estados, 2)) #comentar luego de la primera iteracion
start = time.time()
datos[n_qubits-2, int(np.log2(v_b.shape[0])) - 1, :, :] = Parallel(n_jobs=8)(
    delayed(simulacion_vec)(psi_sistema[:, i:i+1],
    base_diag_vec, bases_sep_vec, nu_exp)
    for i in range(nu_estados)
    )

end = time.time()
print("tiempo", end - start)
print("tabla fids")
print(datos[:, :, :, 0].mean(2))
