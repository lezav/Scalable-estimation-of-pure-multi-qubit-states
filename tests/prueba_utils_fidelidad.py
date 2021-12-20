# %%codecell
from core.utils import fidelidad, estado, gram_schmidt
import numpy as np
#from qiskit import*
#IBMQ.save_account('7333b1fd962586554e308dcb057011514d13e47933e1823e0cfc9bacee382510888ac2039495ed6f5d4aebf1dade75c423d9d8be80dc9d54fde4750c74ca5e5d')
#IBMQ.load_account()
"""
Test de funcionamiento funcion fidelidad
"""
def f(x, y):
    # fidelidad teorica
    return np.absolute(np.dot(x.conj().T, y))**2

DIMENSION = 10
base = estado(DIMENSION, DIMENSION)
base = gram_schmidt(base)
psi_sistema = estado(10, 1)
psi_sistema = psi_sistema/np.linalg.norm(psi_sistema)

print(fidelidad(psi_sistema, base).shape)
print([f(base[:,k], psi_sistema)[0] for k in range(DIMENSION)] )

#run using python -m tests.prueba_utils_fidelidad
