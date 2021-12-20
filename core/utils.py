import numpy as np
import scipy

def gram_schmidt(matriz):
    """
    from a matrix of n vectors of dimension n calculate a base using
    Gram-Schmidt algorithm.
    IN
    matriz: d x d matrix
    OUT
    Q: d x d orthogonal matrix
    """
    R = np.zeros((matriz.shape[1], matriz.shape[1]),dtype=complex)
    Q = np.zeros(matriz.shape, dtype=complex)
    for k in range(0, matriz.shape[1]):
	    R[k, k] = np.sqrt(np.dot(matriz[:, k].conj(), matriz[:, k]))
	    Q[:, k] = matriz[:, k]/R[k, k]
	    for j in range(k+1, matriz.shape[1]):
		    R[k, j] = np.dot(Q[:, k].conj(), matriz[:, j])
		    matriz[:, j] = matriz[:, j] - R[k, j]*Q[:, k]
    return Q


def fidelidad(psi_sis, base, nu_exp = None):
    """
    Calcula la fidelidad entre el estado del sistema psi_sis y
    el conjunto de proyectores base
    IN
        psi_sis: d x 1 vector.
        base: d x n_base matrix. d is de dimension of each vector, and n_base
                the number of states in the base.
        nu_exp: int. Numero de mediciones. Solo si tipo = sim
    OUT
        prob: d array. Projections between the state of
             the system psi_sys and the set of psi_est
    """

    base = base/np.linalg.norm(base, axis = 0) #verifies normalization
    dim = psi_sis.shape[0]                     #system dimension
    n_base = int(base.size/base.shape[0])         #states in the base
    if n_base < dim:
        base = np.c_[base, np.random.rand(dim,dim-n_base)*1j]
        base = gram_schmidt(base)
    if nu_exp == None:
        return fid_teo(psi_sis, base)
    else:
        return fid_sim(psi_sis, base, nu_exp)


def fid_teo(psi_sis, base):
    """
    Calcula la fidelidad teorica entre el estado del sistema psi_sis
    y la base
    IN
        psi_sis: d x 1 complex vector.
        base: d x d, a 2-d array. A base asociated to a estimator.
    OUT
        prob: d array. Projections of psi_sis on the base.
    """

    prob = np.absolute(np.dot(psi_sis.conj().T, base))**2
    prob = (prob/np.sum(prob)).real
    return prob[0]


def fid_sim(psi_sis, base, nu_exp):
    """
    Calcula la fidelidad simulada con nu_exp cuentas entre el estado del sistema
    psi_sis y la base
    IN
        psi_sis: d x 1 complex vector.
        base: d x d, a 2-d array. A base asociated to a estimator.
        nu_exp: int. Number of counts asociated to a experiment.
    OUT
        prob: d array. Projections of psi_sis on the base.
    """

    prob = np.absolute(np.dot(psi_sis.conj().T, base))**2
    prob = (prob/np.sum(prob)).real
    prob = np.random.multinomial(nu_exp, prob[0])/nu_exp
    prob = (prob/np.sum(prob)).real
    return prob


def fidelidad_vec(psi_sis, base, nu_exp = None):
    """
    Calcula la fidelidad entre el estado del sistema psi_sis y la base
    IN
        psi_sis: d x 1 vector.
        base: 2 x 2 x n_qubits. Las primeras dos dimensiones contienen matrices
             de 2 x 2 cuyo producto kronecker entrega la base completa.
        nu_exp: int. Numero de mediciones. Solo si tipo = sim
    OUT
        prob: d array. Projections between the state psi_sis and the base
    """

    dim = psi_sis.shape[0]                     #system dimension
    n_qubits = int(np.log2(dim))
    if nu_exp == None:
        return fid_teo_vec(psi_sis, base, n_qubits)
    else:
        return fid_sim_vec(psi_sis, base, n_qubits, nu_exp)


def fid_teo_vec(psi_sis, base, n_qubits):
    """
    Calcula la fidelidad teorica entre el estado del sistema psi_sis
    y la base
    IN
        psi_sis: d x 1 complex vector.
        base: 2 x 2 x n_qubits. Las primeras dos dimensiones contienen matrices
             de 2 x 2 cuyo producto kronecker entrega la base completa.
        n_qubits: int. number of qubits in the system.
    OUT
        prob: d array. Projections of psi_sis on the base.
    """

    x = psi_sis
    for j in range(n_qubits-1, -1, -1):
        x = x.reshape(2, -1, order="F")
        x = base[:, :, j].conj().T@x
        x = x.T
        x = x.reshape(-1, order="F")
    x = x.reshape(-1, order="F")
    return abs(x)**2


def fid_sim_vec(psi_sis, base, n_qubits, nu_exp):
    """
    Calcula la fidelidad simulada con nu_exp cuentas entre el estado del sistema
    psi_sis y la base
    IN
        psi_sis: d x 1 complex vector.
        base: 2 x 2 x n_qubits. Las primeras dos dimensiones contienen matrices
             de 2 x 2 cuyo producto kronecker entrega la base completa.
        n_qubits: int. number of qubits in the system.
    OUT
        prob: d array. Projections of psi_sis on the base.
    """

    x = psi_sis
    for j in range(n_qubits-1, -1, -1):
        x = x.reshape(2, -1, order="F")
        x = base[:, :, j].conj().T@x
        x = x.T
        x = x.reshape(-1, order="F")
    prob =  abs(x)**2
    prob = np.random.multinomial(nu_exp, prob)/nu_exp
    prob = (prob/np.sum(prob)).real
    return prob


def dot_prod_vec(psi_sis, base, n_qubits):
    """
    Calcula la producto punto entre el estado psi_sis y la base eficientemente
    IN
        psi_sis: d x 1 complex vector.
        base: 2 x 2 x n_qubits. Las primeras dos dimensiones contienen matrices
             de 2 x 2 cuyo producto kronecker entrega la base completa.
        n_qubits: int. Number of qubits in the system.
    OUT
        x: d array. dot producto between psi_sis y base, < base | psi_sis >.
    """

    x = psi_sis
    for j in range(n_qubits-1, -1, -1):
        x = x.reshape(2, -1, order="F")
        x = base[:, :, j].conj().T@x
        x = x.T
        x = x.reshape(-1, order="F")
    x = x.reshape(-1, order="F")
    return x


def estado(dim, n_par, seed = None):
    np.random.seed(seed)
    psi = (np.random.normal(loc=0.0, scale=1.0,
           size=(dim, n_par))
           + np.random.normal(loc=0.0, scale=1.0,
           size=(dim, n_par))*1j)
    psi = psi/np.linalg.norm(psi, axis=0)
    return psi


def estado_sep(dim, n_par, seed=None):
    np.random.seed(seed)
    n_qubits = int(np.log2(dim))
    psi = np.zeros((dim, n_par), dtype=complex)
    psi[0:2, :] = estado(2, n_par)
    for j in range(n_par):
        for k in range(1, n_qubits):
            psi[0:2**(k+1), j] = np.kron(psi[0:2**k, j],
                                         estado(2, 1).reshape(-1))
    return psi

#
def fid_teo_dens(psi_sis, base):
    """
    Calcula la fidelidad teorica entre el estado del sistema psi_sis
    y la base
    IN
        psi_sis: d x d complex array.
        base: d x d, a 2-d array. A base asociated to a estimator.
    OUT
        prob: par real vector. These are the projections
                    of psi_sis on the base that contains psi_est.
    """
    dim = psi_sis.shape[0]
    prob = np.zeros((dim))
    for k in range(dim):
        prob[k] = np.absolute(np.dot(base[:, k:k+1].conj().T,
                              np.dot(psi_sis, base[:, k:k+1])))
    prob = (prob/np.sum(prob)).real
    return prob


def fid_sim_dens(psi_sis, base, nu_exp):
    """
    Calcula la fidelidad teorica entre el estado del sistema psi_sis
    y la base
    IN
        psi_sis: d x d complex array.
        base: d x d, a 2-d array. A base asociated to a estimator.
    OUT
        prob: par real vector. These are the projections
                    of psi_sis on the base that contains psi_est.
    """
    dim = psi_sis.shape[0]
    prob = np.zeros((dim))
    for k in range(dim):
        prob[k] = np.absolute(np.dot(base[:, k:k+1].conj().T,
                              np.dot(psi_sis, base[:, k:k+1])))
    prob = (prob/np.sum(prob)).real
    prob = np.random.multinomial(nu_exp, prob)/nu_exp
    prob = (prob/np.sum(prob)).real
    return prob
