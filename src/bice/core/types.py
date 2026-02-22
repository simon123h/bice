"""Common type aliases used throughout the package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing
import scipy.sparse as sp

if TYPE_CHECKING:
    import matplotlib.axes

# common type for shapes
type Shape = int | tuple[int, ...]

# Common type for Arrays, e.g. the vector of unknowns
type Array = numpy.typing.NDArray[np.float64 | np.complexfloating]

# Type for purely real-valued arrays (e.g. grid points)
type RealArray = numpy.typing.NDArray[np.float64]

# Type for complex-valued arrays
type ComplexArray = numpy.typing.NDArray[np.complexfloating]

# Objects that can be coerced into an Array
type ArrayLike = numpy.typing.ArrayLike

# Common type for dense and sparse matrices
type Matrix = np.ndarray | sp.spmatrix

# Common type for matplotlib axes
if TYPE_CHECKING:
    type Axes = matplotlib.axes.Axes
else:
    type Axes = Any

# Dictionary for serialized data
type DataDict = dict[str, Any]
