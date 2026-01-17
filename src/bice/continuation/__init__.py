"""
Continuation and bifurcation tracking functionality.

This package provides classes for path continuation (natural, pseudo-arclength)
and bifurcation analysis (branch switching, tracking).
"""

from .bifurcations import BifurcationConstraint
from .constraints import ConstraintEquation, TranslationConstraint, VolumeConstraint
from .continuation_steppers import NaturalContinuation, PseudoArclengthContinuation
from .timeperiodic import TimePeriodicOrbitHandler

__all__ = [
    "NaturalContinuation",
    "PseudoArclengthContinuation",
    "ConstraintEquation",
    "VolumeConstraint",
    "TranslationConstraint",
    "BifurcationConstraint",
    "TimePeriodicOrbitHandler",
]
