import numpy as np
from core.utils import estado_sep, estado

psi_1 = estado_sep(1024, 100, seed = None)

np.linalg.norm(psi_1, axis=0)
