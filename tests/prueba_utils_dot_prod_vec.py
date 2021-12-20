# %% codecell
# %load_ext autoreload
# %autoreload 2
import numpy as np
from core.tresbases import bases_2_3, bases_separables
from core.tresbases_vec import bases_separables_vec
from core.utils import estado, estado_sep, dot_prod_vec
import time

n_qubits = 2
dim = int(2**n_qubits)
psi_sis = estado(dim, 1, seed=None)
# definimos los parametros de las bases sobre las que queremos calcular
# el producto punto
v_a = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
v_fase = np.array([0, np.pi/2])
v_b = np.sqrt(1 - v_a**2)
# elegimos la base entrelazada o la separable
base_diag_vec, bases_vec = bases_separables_vec(dim, v_a, v_b, v_fase)
base_diag, bases = bases_separables(dim, v_a, v_b, v_fase)
# comparamos el producto punto usando la forma normal y la eficiente
a = 1
b = 1
dt_vec = dot_prod_vec(psi_sis, bases_vec[:, :, :, a, b], n_qubits)
dt = (bases[:, :, a, b].conj().T@psi_sis).reshape(-1)

print("dif dot_prod",  dt_vec - dt)
