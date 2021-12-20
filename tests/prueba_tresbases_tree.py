# %% codecell
# %load_ext autoreload
# %autoreload 2
import numpy as np
from core.tresbases import tree, bases_2_3

DIMENSION = 4
a = 1./np.sqrt(2)
b = 1./np.sqrt(2)
base_1, base_2 = tree(DIMENSION, a, b, 0)
print(base_1)
