import numpy as np
from scipy import linalg
from core.utils import fidelidad, dot_prod_vec

def bases_2_3(a, b, fase):
    """
    Bases basicas en dimension 2 y 3.
    IN
        a: real. Coeficiente acompañando a |0>.
        b: real. Coeficiente acompañando a |1>.
        fase: real. Fase acompañando a |1>.
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


def bases_separables(dim, v_a, v_b, v_fase):
    """
    Genera log2(dim) x n_bases bases separables
    IN:
        dim: int. Dimension del estado a reconstruir
        v_a, v_b: arrays. Coeficientes de los estados de las bases medidas
        v_fase: array. Angulos relativos de los estados de la base.
    OUT
        base_0: array dim x dim. Base diagonal
        bases_sep: array dim x dim x n_qubits x n_bases. Bases separables
    """
    n_qubits = int(np.log2(dim))
    n_bases = v_fase.shape[0]
    base_0 = np.identity(dim) + 0.*1j
    b_0 = np.array([[1, 0], [0, 1]])
    bases_sep = np.zeros((dim, dim, n_qubits, n_bases)) + 0.*1j
    for k in range(n_bases):
        B_2, B_3 = bases_2_3(v_a[k], v_b[k], v_fase[k])
        for j in range(n_qubits):
            A = B_2
            for m in range(n_qubits-j-1):
                A = np.kron(b_0, A)
            for m in range(j):
                A = np.kron(A, B_2)
            bases_sep[:, :, j, k] = A
    return base_0, bases_sep[:, :, ::-1, :]


def tree(dim, a, b, fase):
    """
    Construye las bases de acuerdo al esquema mostrado en el articulo,
    y las almacena en base_vec_d_1. Cada vector [:, k] de esta representa
    un proyector, con el indice k un nodo del arbol binario. Completa
    la ultima mitad de base_vec_d_1 con los proyectores en dimension 2 y 3
    y luego une el proyector 2*i con el 2*i+1 en otro proyector que
    se almacena en i. Continua el proceso con cada nodo hasta llegar
    a la raiz.
    IN
        dim: int. Dimension de la base deseada.
        a: real. Coeficiente acompañando a |0>.
        b: real. Coeficiente acompañando a |1>.
        fase: real. Fase acompañando a |1>.
    OUT
        base_vec_d_1, base_vec_d: arrays dimension x dimension. La primera es
        la base a medir y la segunda es necesaria para calcular
        los productos puntos en el algoritmo de tomografia.
    """
    #vemos si la dimension es par o impar y determinamos el numero de hojas
    # y el de nodos del arbol completo.
    n_leaves, n_nodes, par = n_leaves_nodes(dim)
    B_2, B_3 = bases_2_3(a, b, fase)  #bases en dimension 2 y 3.
    if dim == 2:
        return B_2, B_2
    elif dim == 3:
        return B_3, B_3
    #base_vec_d_1 almacena los vectores de la base a medir, los con
    #indice n-1 en la bases B_n.
    base_vec_d_1 = np.zeros((dim, dim)) + 0.*1j
    #base_vec_d almacena los vectores indice n de las bases B_n.
    base_vec_d = np.zeros((dim, dim)) + 0.*1j
    idx = 0
    #agregamos los subestados de dimension 2 y 3
    for k in range(int(n_nodes//2+1), int(n_nodes)):
        base_vec_d_1[2*idx:2*idx+2,k] = B_2[:,0]
        #el ultimo vector es cero para que no de problemas al
        #aplicar el algoritmo
        base_vec_d[2*idx:2*idx+2,k] = B_2[:,1]
        idx = idx+1
    #si es par la ultima hoja tiene dimension 2.
    if par:
        base_vec_d_1[2*idx:2*idx+2,k+1] = B_2[:,0]
        base_vec_d[2*idx:2*idx+2,k+1] = B_2[:,1]
    else:
        base_vec_d_1[2*idx:2*idx+3,k+1:k+3] = B_3[:,0:2]
        base_vec_d[2*idx:2*idx+3,k+1:k+2] = B_3[:,2].reshape(-1, 1)
    #construye los nodos internos desde las hojas. En un arbol completo
    #el padre es el nodo int(i//2)
    for i in range(int(n_nodes), int(1), -2):
        base_vec_d_1[:, int(i//2)] = (a*base_vec_d[:,i-1]
                                      + b*np.exp(1j*fase)*base_vec_d[:,i])
        base_vec_d[:, int(i//2)] = (b*base_vec_d[:,i-1]
                                    - a*np.exp(1j*fase)*base_vec_d[:,i])
    base_vec_d_1[:, 0] = base_vec_d[:, 1]
    return base_vec_d_1, base_vec_d


def bases_ent(dim, v_a, v_b, v_fase):
    """
    Construye las 3 bases de acuerdo al esquema mostrado en el articulo.
    IN:
        dim: int. Dimension del estado a reconstruir
        v_a, v_b: arrays. Coeficientes de los estados de las bases medidas
        v_fase: array. Angulos relativos de los estados de la base.
    """
    n_bases = v_fase.shape[0]
    base_0 = np.identity(dim) + 0.*1j
    d_bases = np.zeros((dim, dim, n_bases)) + 0.*1j
    d_bases_res = np.zeros((dim, dim, n_bases)) + 0.*1j
    for k in range(n_bases):
        d_bases[:, :, k], d_bases_res[:, :, k] = tree(dim, v_a[k],
                                                      v_b[k], v_fase[k])
    return base_0, d_bases


def tomography(prob_diag, prob_sep, bases_sep, pur=False):
    """
    Tomografia tres bases para estados en cualquier dimension
    IN
        prob_diag: array dim x 1. Contiene las mediciones de la base estandar
        prob_sep: array dim x n_qubits x n_bases o array dim x n_bases.
                Contiene las mediciones de las bases separables o las entrelazadas.
                Por cada base entrelazada tenemos n_qubits
                bases separables. n_bases es el numero de bases entrelazadas.
        proyectores: array dim x dim x nqubits x n_bases. Bases separables a medir.
        pureza: cat. Si es True entrega tambien la pureza.
    OUT
        psi_sis: array dim x 1. Estado del sistema.
    """
    # vemos si las bases son separables o entrelazadas
    ent = len(bases_sep.shape) < 4
    #
    dim = prob_diag.shape[0]
    n_qubits = int(np.log2(dim))
    n_bases = bases_sep.shape[1]/dim
    n_leaves, n_nodes, par = n_leaves_nodes(dim)
    psi = np.zeros((dim, dim)) + 0.*1j
    lamb = np.zeros((dim//2))
    # comenzamos llenando todas las hojas
    idx = 0
    for k in range(int(n_nodes//2+1), int(n_nodes)+1):
        nivel = int(np.log2(k)) # nivel del arbol. En este caso el ultimo
        psi_j = np.array([np.sqrt(prob_diag[2*idx]), 0])
        psi_k = np.array([0, np.sqrt(prob_diag[2*idx+1])])
        if ent:
            prob = prob_sep[k, :]
            proyectores = bases_sep[2*idx:2*idx+2, k, :]
            psi[2*idx:2*idx+2, k] = block_n(psi_j, psi_k, prob, proyectores)
        else:
            prob = prob_sep[2*idx:2*idx+2, nivel, :].reshape(-1, order="F")
            proyectores = bases_sep[2*idx:2*idx+2, 2*idx:2*idx+2,
                                    nivel, :].reshape(2, -1, order="F")
            psi[2*idx:2*idx+2, k] = block_n(psi_j, psi_k, prob, proyectores)
        #calculamos la pureza
        lamb[idx] = pureza(prob_diag[2*idx], prob_diag[2*idx + 1], prob,
                      proyectores, dim)
        idx = idx + 1
    # llenamos los otros niveles del arbol
    for i in range(int(n_nodes), 1, -2):
        idx = int(i//2)
        if ent:
            proyectores = bases_sep[:, idx, :]
            prob = prob_sep[idx, :].reshape(-1, order="F")
            psi[:, idx] = block_n(psi[:, i-1], psi[:, i], prob, proyectores)
        else:
            nivel = int(np.log2(idx))
            d = int(2**(n_qubits - nivel)) # numero de nodos en el nivel
            j = 2**(nivel + 1) - idx
            # intervalo donde estan los proyectores asociados al nodo
            slice = range(dim - int(d*(j-1)) - d, dim - int(d*(j-1)))
            # seleccionamos los proyectores slice de las matrices asociadas a nivel
            proyectores = bases_sep[:, slice, nivel, :].reshape(dim, -1, order="F")
            prob = prob_sep[slice, nivel, :].reshape(-1, order="F")
            psi[:, idx] = block_n(psi[:, i-1], psi[:, i], prob, proyectores)
    psi = psi[:,1]/np.linalg.norm(psi[:,1])
    if pur:
        return psi, np.abs(np.mean(lamb[lamb != -1]))
    else:
        return psi


def block_n(psi_j, psi_k, prob, proyectores):
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
    n_bases = proyectores.shape[1]
    p_tilde = (prob - np.abs(np.dot(proyectores.conj().T,psi_j))**2
               - np.abs(np.dot(proyectores.conj().T,psi_k))**2)
    X = np.dot(proyectores.conj().T, psi_k)*np.dot(psi_j.conj(), proyectores)
    eqs = np.zeros((n_bases, 2))
    eqs[:, 0] = np.real(X)
    eqs[:, 1] = - np.imag(X)
    ## la fase no tiene norma 1.
    exp_fase = np.dot(linalg.pinv2(eqs), p_tilde)
    exp_fase = exp_fase[0] + 1j*exp_fase[1]
    exp_fase = exp_fase/np.linalg.norm(exp_fase)
    # Si el sistema de ecuaciones se indetermina hacemos la fase 0
    if np.isnan(exp_fase):
        exp_fase = 1

    psi_n = psi_j + psi_k*exp_fase
    return psi_n


def n_leaves_nodes(dim):
    """
    Dependiendo de la dimension entrega el numero de nodos y
    hojas del arbol.
    IN
        dim: int. Dimension del sistema.
    OUT
        n_leaves: int. Numero de hojas que deberia tener el
                 arbol de acuerdo al articulo.
        n_nodes: int. Numero total de nodos del arbol. Se relaciona de
                es forma con n_leaves ya que tiene que el arbol tiene que
                ser binario completo.
        par: boolean. True si la dimension es par.
    """
    if dim%2 == 0:
        n_leaves = dim/2
        par = True
    else:
        n_leaves = (dim-1)/2
        par = False
    n_nodes = 2*n_leaves - 1
    return n_leaves, n_nodes, par


def simulacion(psi_sistema, base_diag, bases, nu_exp):
    """
    Simula tomografia
    IN
        psi_sistema: array n_qubits x 1. Estado de prueba
        bases: array dim x dim x n_qubits x n_bases o dim x dim x n_bases. Bases
        nu_exp: int. Numero de experimentos para estimar las probabilidades
    OUT
        fid: float. fidelidad del estado preparado con el estimado.
        lam: float. coeficiente de mixtura.

    """
    dim = psi_sistema.shape[0]
    n_qubits = int(np.log2(dim))
    ent = len(bases.shape) < 4

    if ent:
        n_bases = bases.shape[2]
        fid = np.zeros((dim, n_bases))
        fid_0 = fidelidad(psi_sistema, base_diag, nu_exp=nu_exp)
        for k in range(n_bases):
            fid[:, k] = fidelidad(psi_sistema, bases[:, :, k], nu_exp=nu_exp)
    else:
        n_bases = bases.shape[3]
        fid = np.zeros((dim, n_qubits, n_bases))
        fid_0 = fidelidad(psi_sistema, base_diag, nu_exp=nu_exp)
        for k in range(n_bases):
            for j in range(n_qubits):
                fid[:, j, k] = fidelidad(psi_sistema, bases[:, :, j, k],
                                         nu_exp=nu_exp)
    # las bases deben tener 4 dimensiones. dim x dim x n_qubits x n_bases
    # n_qubits es el numero de bases separables que equivalen a una base entralazada
    # y n_bases es el numero de bases entrelazadas equivalente.
    # Deben estar ordenadas en potencias descendentes de 2. Por ejemplo, en
    # bases[:, :, 0, :] deben estar los proyectores de dimension 2^n_qubits, y
    # de ahi descander hasta bases[:, :, n_qubits, :] con proyectores de dimension 2
    psi, lamb = tomography(fid_0, fid, bases, pur=True)
    fid = (np.abs(np.dot(psi.conj(), psi_sistema))**2)[0]
    return np.array([fid, lamb])


def pureza(p_0, p_1, prob, proyectores, dim):
    """
    Calcula la pureza de un estado en dimension dim usando subestados en
    dimension k y j.
    IN
        psi_j, psi_k: arrays. Subestados que hay que acoplar.
        prob: array slice*n_bases. Una probabilidad por cada proyector.
        proyectores: array dim x slice*n_bases. Proyectores de las bases medidas
    OUT
        lambda: array dim/2. Parametro de pureza.
    """

    # si uno de los dos subestados es cero no calculamos nada
    if (p_0 == 0) | (p_1 == 0):
        return -1
    psi_j = np.array([np.sqrt(p_0), 0])
    psi_k = np.array([0, np.sqrt(p_1)])
    n_bases = proyectores.shape[1]
    p_tilde = (prob - np.abs(np.dot(proyectores.conj().T, psi_j))**2
               - np.abs(np.dot(proyectores.conj().T, psi_k))**2)
    # p_tilde = (prob - np.abs(np.dot(proyectores.conj().T, np.array([1, 0])))**2
    #            - np.abs(np.dot(proyectores.conj().T, np.array([0, 1])))**2)
    X = np.dot(proyectores.conj().T, np.array([0, 1]))*np.dot(np.array([1, 0]).conj(), proyectores)
    eqs = np.zeros((n_bases, 2))
    eqs[:, 0] = np.real(X)
    eqs[:, 1] = - np.imag(X)
    # hasta aca todo igual
    ## la fase no tiene norma 1.
    exp_fase = np.dot(linalg.pinv2(eqs), p_tilde)
    exp_fase = exp_fase[0] + 1j*exp_fase[1]
    lamb = dim/2*(p_0 + p_1 - np.sqrt((p_0 - p_1)**2 + np.abs(exp_fase)**2))
    return lamb
