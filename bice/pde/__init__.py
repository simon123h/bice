
from .pde import PartialDifferentialEquation
from .finite_differences import FiniteDifferencesEquation
from .pseudospectral import PseudospectralEquation
from .collocation import CollocationEquation


__all__ = [
    "PartialDifferentialEquation",
    "FiniteDifferencesEquation",
    "PseudospectralEquation",
    "CollocationEquation"
]
