from typing import Union
import numpy as np
import scipy.sparse as sp

# common type for shapes
Shape = Union[int, tuple[int, ...]]

# common type for dense and sparse matrices
Matrix = Union[np.ndarray, sp.spmatrix]
# Matrix = Union[np.ndarray, sp.spmatrix, float]
