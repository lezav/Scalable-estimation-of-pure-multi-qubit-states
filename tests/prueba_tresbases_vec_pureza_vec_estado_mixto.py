# %% codecell
# %load_ext autoreload
# %autoreload 2
import numpy as np
from core.tresbases_vec import bases_separables_vec, pureza_vec
from core.tresbases import bases_separables
from core.utils import estado, estado_sep, fidelidad_vec, fidelidad, fid_teo_dens, fid_sim_dens
import time

n_qubits = 8
dim = int(2**n_qubits)
nu_exp = 2**13
lamb = 0.05
psi_sistema = estado(dim, 1, seed=None)
psi_sistema = psi_sistema/np.linalg.norm(psi_sistema)
dens = np.kron(psi_sistema.conj().T, psi_sistema)
dens = (1 -lamb)*dens + lamb*np.eye(dim)/dim
# 8 bases
v_a = np.array([1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)])
v_fase = np.array([0, np.pi/2, np.pi/3, np.pi/4])
v_b = np.sqrt(1 - v_a**2)
n_bases = v_b.shape[0]
# elegimos la base separable
base_diag_vec, bases_sep_vec = bases_separables_vec(dim, v_a, v_b, v_fase)
base_diag, bases_sep = bases_separables(dim, v_a, v_b, v_fase)
# calculamos probabilidades
start = time.time()
prob_sep_vec = np.zeros((dim, n_qubits, n_bases))
prob_diag_vec = fid_sim_dens(dens, base_diag, nu_exp)
for k in range(n_bases):
    for j in range(n_qubits):
        prob_sep_vec[:, j, k] = fid_sim_dens(dens, bases_sep[:, :, j, k], nu_exp)
end = time.time()
# print("time", end - start)
# calculo de purezas
start = time.time()
for k in range(1):
    lamb_est, data_like = pureza_vec(prob_diag_vec, prob_sep_vec, bases_sep_vec,
                                     True)
end = time.time()
print("time", end - start)
print("lambda_original", lamb,  "lambda_est",
      np.median(lamb_est[lamb_est != -1]))
