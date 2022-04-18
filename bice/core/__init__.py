"""
The 'core' package contains all of bice's basic functionality.
"""

from .equation import Equation, EquationGroup
from .problem import Problem
from .profiling import Profiler, profile
from .solution import BifurcationDiagram, Branch, Solution
from .solvers import EigenSolver, MyNewtonSolver, NewtonSolver

__all__ = [
    'Problem',
    'Equation', 'EquationGroup',
    'Solution', 'Branch', 'BifurcationDiagram',
    'MyNewtonSolver', 'NewtonSolver', 'EigenSolver',
    'profile', 'Profiler'
]
