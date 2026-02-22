"""Sociable unit tests for Solution, Branch and BifurcationDiagram classes."""

import numpy as np
import pytest

from bice.core.equation import Equation
from bice.core.problem import Problem
from bice.core.solution import BifurcationDiagram, Branch, Solution


class SimpleEquation(Equation):
    def rhs(self, u):
        return -u


def test_solution_creation():
    prob = Problem()
    prob.add_equation(SimpleEquation())
    prob.u = np.array([1.0])

    sol = Solution(prob)
    assert sol.p == 0.0
    assert sol.norm == pytest.approx(1.0)
    assert isinstance(sol.data, dict)
    assert "SimpleEquation.u" in sol.data


def test_branch_management():
    branch = Branch()
    assert branch.is_empty()

    prob = Problem()
    prob.add_equation(SimpleEquation())

    for i in range(5):
        prob.u = np.array([float(i)])
        sol = Solution(prob)
        branch.add_solution_point(sol)

    assert len(branch.solutions) == 5
    assert not branch.is_empty()
    np.testing.assert_allclose(branch.norm_vals(), [0.0, 1.0, 2.0, 3.0, 4.0])


def test_bifurcation_detection():
    branch = Branch()

    # Create a sequence of solutions where eigenvalues change sign
    # Solution 1: 1 unstable EV
    sol1 = Solution()
    sol1.nunstable_eigenvalues = 1
    branch.add_solution_point(sol1)

    # Solution 2: 0 unstable EVs (Stable)
    sol2 = Solution()
    sol2.nunstable_eigenvalues = 0
    branch.add_solution_point(sol2)

    assert sol2.is_bifurcation()
    assert sol2.neigenvalues_crossed == -1
    assert sol2.bifurcation_type() == "-"


def test_bifurcation_diagram():
    bd = BifurcationDiagram()
    assert len(bd.branches) == 1  # Active branch created by default

    branch1 = bd.active_branch
    sol = Solution()
    branch1.add_solution_point(sol)

    assert bd.current_solution() is sol

    new_branch = bd.new_branch()
    assert bd.active_branch is new_branch
    assert len(bd.branches) == 2
