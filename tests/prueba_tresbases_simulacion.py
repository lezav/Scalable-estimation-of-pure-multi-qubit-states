import numpy as np
from core.tresbases import simulacion, bases_separables, bases_ent
from core.utils import estado, estado_sep
import time
from joblib import Parallel, delayed
# repite la tomografia para un conjunto de estados aleatorios y calcula
# fidelidades
# preparamos las bases y el estado
n_qubits = 3
nu_exp = 2**13
dim = int(2**n_qubits)
nu_estados = 10
psi_sistema = estado(dim, nu_estados, seed=None)
psi_sistema = psi_sistema*np.exp(-1j*np.angle(psi_sistema[0]))
# elegimos las caracteristicas de las bases
v_a = np.array([1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)])
v_fase = np.array([0, np.pi/2, np.pi/3, np.pi/4])
v_b = np.sqrt(1 - v_a**2)
# elegimos la base entrelazada o la separable
base_diag, bases = bases_separables(dim, v_a, v_b, v_fase)
# base_diag, bases = bases_ent(dim, v_a, v_b, v_fase)

datos = np.zeros((9, nu_estados, 2)) #comentar luego de la primera iteracion
start = time.time()
datos[n_qubits-2, :, :] = Parallel(n_jobs=1)(
    delayed(simulacion)(psi_sistema[:, i:i+1],
    base_diag, bases, nu_exp)
    for i in range(nu_estados)
    )
end = time.time()
# mostramos fidelidad
print("tiempo", end - start)
print("tabla fids")
print(datos[:, :, 0].mean(1))
