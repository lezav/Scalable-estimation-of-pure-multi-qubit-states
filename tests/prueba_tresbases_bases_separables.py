# %% codecell
# %load_ext autoreload
# %autoreload 2
import numpy as np
from core.tresbases import bases_separables, bases_2_3


dim = 4
v_a = np.array([1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)])
v_b = np.sqrt(1 - v_a**2)
v_fase = np.array([0, np.pi/2, np.pi/4 ])

b_1 = (1/np.sqrt(2))*np.array( [[1, 1], [1, -1]] )
base_diag, bases_sep = bases_separables(dim, v_a, v_b, v_fase)

print(bases_sep[:, :, 0, 0])
