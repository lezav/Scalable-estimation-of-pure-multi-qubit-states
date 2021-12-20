import numpy as np
from core.tresbases import bases_2_3


a = 1./6
b = np.sqrt(1 - a**2)
fase = np.pi/4
B_2, B_3 = bases_2_3(a, b, fase)


print(np.dot(B_3[:, 1].T.conj(), B_3[:, 2]))
