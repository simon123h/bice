"""
Partial Differential Equation (PDE) discretization schemes.

This package provides base classes for different spatial discretization
methods, such as finite differences and pseudospectral methods.
"""

from .finite_differences import FiniteDifferencesEquation
from .pde import PartialDifferentialEquation
from .pseudospectral import PseudospectralEquation

__all__ = [
    "PartialDifferentialEquation",
    "FiniteDifferencesEquation",
    "PseudospectralEquation",
]
