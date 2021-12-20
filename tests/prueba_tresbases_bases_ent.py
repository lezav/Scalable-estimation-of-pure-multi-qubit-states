import numpy as np
from core.tresbases import bases_ent

dim = 4
v_a = np.array([1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)])
v_b = np.sqrt(1 - v_a**2)
v_fase = np.array([0, np.pi/2, np.pi/4 ])
base_0, d_bases = bases_ent(dim, v_a, v_b, v_fase)

print(d_bases[:, :, 2])
