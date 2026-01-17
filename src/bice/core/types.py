from typing import Tuple, Union

import numpy as np
import numpy.typing
import scipy.sparse as sp

# common type for shapes
Shape = Union[int, Tuple[int, ...]]

# Common type for Arrays, e.g. the vector of unknowns
Array = numpy.typing.NDArray[Union[np.float64, np.complexfloating, np.bool_]]

# Objects that can be coerced into an Array
ArrayLike = numpy.typing.ArrayLike

# Common type for dense and sparse matrices
Matrix = Union[np.ndarray, sp.spmatrix]
