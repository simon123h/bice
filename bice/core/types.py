from typing import Tuple, Union

import numpy as np
import numpy.typing
import scipy.sparse as sp

# common type for shapes
Shape = Union[int, Tuple[int, ...]]

# Common type for Arrays, e.g. the vector of unknowns
# TODO: also support complex values?
Array = numpy.typing.NDArray[np.float64]

# Objects that can be coerced into an Array
ArrayLike = numpy.typing.ArrayLike

# Common type for dense and sparse matrices
Matrix = Union[np.ndarray, sp.spmatrix]
