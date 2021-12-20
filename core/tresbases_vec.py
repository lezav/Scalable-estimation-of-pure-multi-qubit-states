import numpy as np
from scipy import linalg
from core.utils import fidelidad_vec, dot_prod_vec


def bases_2_3(a, b, fase):
    """
    Bases basicas en dimension 2 y 3.
    IN
        a: float. Coeficiente acompañando a |0>.
        b: float. Coeficiente acompañando a |1>.
        fase: float. Fase acompañando a |1>.
    OUT
        B_2: array 2 x 2.
        B_3: array 3 x 3
    """
    #aseguramos que las bases sean complejas.
    a = a + 0.*1.j
    b = b + 0.*1.j
    B_3 = np.array([[a, np.exp(1j*fase)*b, 0.*1.j],
                     [a*b, - np.exp(1j*fase)*a*a, np.exp(1j*fase)*b],
                     [b*b, - np.exp(1j*fase)*b*a, -np.exp(1j*fase)*a]]).T
    B_2 = np.array([[a, np.exp(1j*fase)*b],
                     [b, - np.exp(1j*fase)*a]]).T
    return B_2, B_3


def bases_separables_vec(dim, v_a, v_b, v_fase):
    """
    Genera log2(dim) x n_bases bases separables
    IN:
        dim: int. Dimension del estado a reconstruir
        v_a, v_b: arrays. Coeficientes de los estados de las bases medidas
        v_fase: array. Angulos relativos de los estados de la base.
    OUT
        base_0: array n_qubits x n_qubits x n_qubits. Base diagonal
        bases_sep: array n_qubits x n_qubits x n_qubits x n_qubits x n_bases.
                   Entrega las matrices con las que hay que calcular el producto
                   tensorial para construir bases_sep. Estan almacenadas
                   en el tercer indice "m".
    """
    n_qubits = int(np.log2(dim))
    n_bases = v_fase.shape[0]
    b_0 = np.array([[1, 0], [0, 1]], dtype="complex")
    base_0 = np.dstack([b_0]*n_qubits)
    bases_sep_vec = np.stack([np.stack([np.stack([b_0]*n_qubits,
                                             axis=-1)]*n_qubits,
                                   axis=-1)]*n_bases, axis=-1)
    for k in range(n_bases):
        B_2, B_3 = bases_2_3(v_a[k], v_b[k], v_fase[k])
        for j in range(n_qubits):
            for m in range(n_qubits-j-1, n_qubits):
                bases_sep_vec[:, :, m, j, k] = B_2
    return base_0, bases_sep_vec[:, :, :, ::-1, :]


def tomography_vec(prob_diag_vec, prob_sep_vec, bases_sep_vec):
    """
    Tomografia tres bases para estados en cualquier dimension
    IN
        prob_diag_vec: array dim. Contiene las mediciones de la base estandar
        prob_sep_vec: array dim x n_qubits x n_bases. Contiene las mediciones de
                     las bases separables. Tenemos n_bases conjuntos de n_qubits
                     bases separables
        bases_sep_vec: array 2 x 2 x nqubits x nqubits x n_bases. Bases de qubits
                      tal que su producto tensorial sobre la tercera dimension
                      entrega las bases separables a medir
    OUT
        psi_sis: array dim x 1. Estado del sistema.
    """

    dim, n_qubits, n_bases = prob_sep_vec.shape
    # comenzamos llenando todas las hojas
    psi_list = [np.zeros((2**(j+1), 2**(n_qubits-j)), dtype="complex")
                for j in range(n_qubits)]
    psi_list.append(np.zeros((2**n_qubits, 1), dtype="complex"))
    for k in range(2**(n_qubits-1)):
        psi_list[0][:, 2*k] = np.array([np.sqrt(prob_diag_vec[2*k]), 0])
        psi_list[0][:, 2*k+1] = np.array([0, np.sqrt(prob_diag_vec[2*k + 1])])

    for lv in range(n_qubits-1, -1, -1):
        for k in range(2**lv):
            psi_j = psi_list[n_qubits-lv-1][:, 2*k]
            psi_k = psi_list[n_qubits-lv-1][:, 2*k+1]
            n_qubits_eff = n_qubits - lv
            slice = 2**(n_qubits_eff)
            prob = prob_sep_vec[slice*k:slice*(k+1), lv, :].reshape(-1, order="F")
            proyectores = bases_sep_vec[:, :, lv:, lv, :]
            psi_n = block_n_vec(psi_j, psi_k, prob, proyectores, n_qubits_eff,
                                n_bases)
            pad = np.zeros(psi_n.shape[0])
            if lv != 0:
                if k%2 == 0:
                    psi_n =  np.concatenate([psi_n, pad])
                else:
                    psi_n =  np.concatenate([pad, psi_n])
            psi_list[n_qubits-lv][:, k] = psi_n
    psi_sis = psi_list[-1]
    return psi_sis


def block_n_vec(psi_j, psi_k, prob, proyectores, n_qubits_eff, n_bases):
    """
    Reconstruye un subestado en dimension dim usando subestados en
    dimension k y j.
    IN
        psi_j, psi_k: arrays. Subestados que hay que acoplar.
        prob: array slice*n_bases. Una probabilidad por cada proyector.
        proyectores: array dim x slice*n_bases. Proyectores de las bases medidas
    OUT
        psi_n: array. Subestado de la union de psi_k y psi_j.
    """

    # si uno de los dos subestados es cero no calculamos nada
    if np.all(psi_k == 0) | np.all(psi_j == 0):
        return psi_k +  psi_j

    n_eqs_bas = 2**proyectores.shape[2]
    dot_j = np.zeros((n_eqs_bas*n_bases), dtype="complex")
    dot_k = np.zeros((n_eqs_bas*n_bases), dtype="complex")
    for r in range(n_bases):
        dot_j[r*n_eqs_bas:(r+1)*n_eqs_bas] = dot_prod_vec(
            psi_j,
            proyectores[:, :, :, r],
            n_qubits_eff
            )
        dot_k[r*n_eqs_bas:(r+1)*n_eqs_bas] = dot_prod_vec(
            psi_k,
            proyectores[:, :, :, r],
            n_qubits_eff
            )
    p_tilde = (prob - np.abs(dot_j)**2
                   - np.abs(dot_k)**2)
    X = dot_k*(dot_j.conj())
    eqs = np.zeros((n_eqs_bas*n_bases, 2))
    eqs[:, 0] = np.real(X)
    eqs[:, 1] = - np.imag(X)
    exp_fase = np.dot(linalg.pinv2(eqs), p_tilde)
    exp_fase = exp_fase[0] + 1j*exp_fase[1]
    exp_fase = exp_fase/np.linalg.norm(exp_fase)
    # Si el sistema de ecuaciones se indetermina hacemos la fase 0
    if np.isnan(exp_fase):
        exp_fase = 1
    psi_n = psi_j + psi_k*exp_fase
    return psi_n


def simulacion_vec(psi_sistema, base_diag_vec, bases_sep_vec, nu_exp):
    """
    Simula tomografia calculando productos puntos de forma eficiente.
    IN
        psi_sistema: array n_qubits x 1. Estado de prueba
        bases_sep_vec: array 2 x 2 x nqubits x nqubits x n_bases. Bases de qubits
                      tal que su producto tensorial sobre la tercera dimension
                      entrega las bases separables a medir
        bases_sep_vec: array 2 x 2 x nqubits. Bases de qubits
                       tal que su producto tensorial sobre la tercera dimension
                       entrega la base canonica a medir
        nu_exp: int. Numero de experimentos para estimar las probabilidades

    """
    n_qubits, n_bases = bases_sep_vec.shape[3:]
    prob_sep_vec = np.zeros((2**n_qubits, n_qubits, n_bases))
    prob_diag_vec = fidelidad_vec(psi_sistema, base_diag_vec, nu_exp = nu_exp)
    for k in range(n_bases):
        for j in range(n_qubits):
            prob_sep_vec[:, j, k] = fidelidad_vec(psi_sistema,
                bases_sep_vec[:, :, :, j, k],
                nu_exp=nu_exp
                )
    psi = tomography_vec(prob_diag_vec, prob_sep_vec, bases_sep_vec)
    fid = (np.abs(np.dot(psi.conj().T, psi_sistema))**2)[0]
    return fid


def pureza_vec(prob_diag_vec, prob_sep_vec, bases_sep_vec, like=False):
    """
    Calcula los coeficientes lambda para estados cuasi puros
    IN
        prob_diag_vec: array dim. Contiene las mediciones de la base estandar
        prob_sep_vec: array dim x n_qubits x n_bases. Contiene las mediciones de
                     las bases separables. Tenemos n_bases conjuntos de n_qubits
                     bases separables
        bases_sep_vec: array 2 x 2 x nqubits x nqubits x n_bases. Bases de qubits
                      tal que su producto tensorial sobre la tercera dimension
                      entrega las bases separables a medir
    OUT
        lamb: array dim//2. Coeficientes de mixtura.
    """

    dim, n_qubits, n_bases = prob_sep_vec.shape
    lamb = np.zeros((dim//2))
    data_like = np.zeros((dim//2, 3))
    # seleccionamos los proyectores y las probabilidades adecuadas
    for k in range(2**(n_qubits-1)):
        p_0 = prob_diag_vec[2*k]
        p_1 = prob_diag_vec[2*k + 1]
        psi_j = np.array([np.sqrt(p_0), 0])
        psi_k = np.array([0, np.sqrt(p_1)])
        n_qubits_eff = 1
        slice = 2
        prob = prob_sep_vec[slice*k:slice*(k+1),
                            n_qubits-1, :].reshape(-1, order="F")
        proyectores = bases_sep_vec[:, :, n_qubits-1:, n_qubits-1, :]
        # esta es la parte que realiza el calculo de la pureza
        if (p_0 == 0) | (p_1 == 0):
            lamb[k] = -1
            data_like[k, :] = -1
        else:
            n_eqs_bas = 2**proyectores.shape[2]
            dot_j = np.zeros((n_eqs_bas*n_bases), dtype="complex")
            dot_k = np.zeros((n_eqs_bas*n_bases), dtype="complex")
            dot_jx = np.zeros((n_eqs_bas*n_bases), dtype="complex")
            dot_kx = np.zeros((n_eqs_bas*n_bases), dtype="complex")
            for r in range(n_bases):
                dot_j[r*n_eqs_bas:(r+1)*n_eqs_bas] = dot_prod_vec(
                    psi_j,
                    proyectores[:, :, :, r],
                    n_qubits_eff
                    )
                dot_k[r*n_eqs_bas:(r+1)*n_eqs_bas] = dot_prod_vec(
                    psi_k,
                    proyectores[:, :, :, r],
                    n_qubits_eff
                    )
                dot_jx[r*n_eqs_bas:(r+1)*n_eqs_bas] = dot_prod_vec(
                    np.array([0, 1]),
                    proyectores[:, :, :, r],
                    n_qubits_eff
                    )
                dot_kx[r*n_eqs_bas:(r+1)*n_eqs_bas] = dot_prod_vec(
                    np.array([1, 0]),
                    proyectores[:, :, :, r],
                    n_qubits_eff
                    )
            p_tilde = (prob - np.abs(dot_j)**2 - np.abs(dot_k)**2)
            X = dot_kx*(dot_jx.conj())
            eqs = np.zeros((n_eqs_bas*n_bases, 2))
            eqs[:, 0] = np.real(X)
            eqs[:, 1] = - np.imag(X)
            exp_fase = np.dot(linalg.pinv2(eqs), p_tilde)
            exp_fase = exp_fase[0] + 1j*exp_fase[1]
            data_like[k, :] = np.abs(exp_fase)**2, p_0, p_1
            lamb[k] = dim/2*(p_0 + p_1 - np.sqrt((p_0 - p_1)**2
                             + np.abs(exp_fase)**2))
    if like==True:
        return lamb, data_like[data_like[:, 0]!=-1]
    else:
        return lamb
