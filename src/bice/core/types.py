"""Common type aliases used throughout the package."""

import numpy as np
import numpy.typing
import scipy.sparse as sp

# common type for shapes
Shape = int | tuple[int, ...]

# Common type for Arrays, e.g. the vector of unknowns
Array = numpy.typing.NDArray[np.float64 | np.complexfloating]

# Objects that can be coerced into an Array
ArrayLike = numpy.typing.ArrayLike

# Common type for dense and sparse matrices
Matrix = np.ndarray | sp.spmatrix
