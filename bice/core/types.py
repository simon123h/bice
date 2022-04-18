from typing import Union
import numpy as np
import scipy.sparse as sp
import numpy.typing

# common type for shapes
Shape = Union[int, tuple[int, ...]]

# Common type for Arrays, e.g. the vector of unknowns
Array = numpy.typing.NDArray

# Objects that can be coerced into an Array
ArrayLike = numpy.typing.ArrayLike

# Common type for dense and sparse matrices
Matrix = Union[np.ndarray, sp.spmatrix]
