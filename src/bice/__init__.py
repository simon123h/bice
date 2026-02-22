"""
BICE: Bifurcation Continuation Engine.

A numerical path continuation and bifurcation analysis package written in Python.
It provides tools for tracking branches of solutions to nonlinear algebraic
equations, identifying bifurcation points, and performing time-stepping for
evolutionary PDEs.
"""

from . import time_steppers
from .core import (
    BifurcationDiagram,
    Branch,
    EigenSolver,
    Equation,
    EquationGroup,
    MyNewtonSolver,
    NewtonSolver,
    Problem,
    Profiler,
    Solution,
    profile,
)

__all__ = [
    "Problem",
    "Equation",
    "EquationGroup",
    "Solution",
    "Branch",
    "BifurcationDiagram",
    "MyNewtonSolver",
    "NewtonSolver",
    "EigenSolver",
    "profile",
    "Profiler",
    "time_steppers",
]
