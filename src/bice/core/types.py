"""Common type aliases used throughout the package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import numpy.typing
import scipy.sparse as sp

if TYPE_CHECKING:
    import matplotlib.axes

# common type for shapes
Shape: TypeAlias = int | tuple[int, ...]

# Common type for Arrays, e.g. the vector of unknowns
Array: TypeAlias = numpy.typing.NDArray[np.float64 | np.complexfloating]

# Type for purely real-valued arrays (e.g. grid points)
RealArray: TypeAlias = numpy.typing.NDArray[np.float64]

# Type for complex-valued arrays
ComplexArray: TypeAlias = numpy.typing.NDArray[np.complexfloating]

# Objects that can be coerced into an Array
ArrayLike: TypeAlias = numpy.typing.ArrayLike

# Common type for dense and sparse matrices
Matrix: TypeAlias = np.ndarray | sp.spmatrix

# Common type for matplotlib axes
if TYPE_CHECKING:
    Axes: TypeAlias = matplotlib.axes.Axes
else:
    Axes: TypeAlias = Any

# Dictionary for serialized data
DataDict: TypeAlias = dict[str, Any]
