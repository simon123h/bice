"""Sociable unit tests for Solution, Branch and BifurcationDiagram classes."""

import numpy as np
import pytest

from bice.core.equation import Equation
from bice.core.problem import Problem
from bice.core.solution import BifurcationDiagram, Branch, Solution
from bice.core.types import Array


class SimpleEquation(Equation):
    def rhs(self, u: Array) -> Array:
        return -u


def test_solution_creation() -> None:
    prob = Problem()
    prob.add_equation(SimpleEquation())
    prob.u = np.array([1.0])

    sol = Solution(prob)
    assert sol.p == 0.0
    assert sol.norm == pytest.approx(1.0)
    assert isinstance(sol.data, dict)
    assert "SimpleEquation.u" in sol.data


def test_branch_management() -> None:
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


def test_bifurcation_detection() -> None:
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


def test_bifurcation_detection_multiple_points() -> None:
    """Test bifurcation detection across multiple points."""
    branch = Branch()

    # Solution 1: First point, no previous info
    sol1 = Solution()
    sol1.nunstable_eigenvalues = 0
    branch.add_solution_point(sol1)
    assert not sol1.is_bifurcation()

    # Solution 2: Second point, still 0 unstable EVs
    sol2 = Solution()
    sol2.nunstable_eigenvalues = 0
    branch.add_solution_point(sol2)
    assert not sol2.is_bifurcation()

    # Solution 3: Third point, bifurcation! (0 -> 1)
    sol3 = Solution()
    sol3.nunstable_eigenvalues = 1
    branch.add_solution_point(sol3)
    assert sol3.is_bifurcation()
    assert sol3.neigenvalues_crossed == 1
    assert sol3.bifurcation_type() == "+"

    # Solution 4: Fourth point, still 1 unstable EV
    sol4 = Solution()
    sol4.nunstable_eigenvalues = 1
    branch.add_solution_point(sol4)
    assert not sol4.is_bifurcation()
    assert sol4.neigenvalues_crossed == 0


def test_neigenvalues_crossed_first_point() -> None:
    """Test that the first point in a branch does not report a crossing."""
    branch = Branch()
    sol1 = Solution()
    sol1.nunstable_eigenvalues = 1
    branch.add_solution_point(sol1)

    assert sol1.neigenvalues_crossed is None
    assert not sol1.is_bifurcation()


def test_branch_data_mask_preservation() -> None:
    """
    Regression test: Ensure Branch.data preserves masks for filtered views.

    If np.asarray is used instead of np.asanyarray, the mask is stripped and
    the plotting logic sees every point as a bifurcation.
    """
    branch = Branch()

    # 3 points: [Normal, Bifurcation, Normal]
    # Point 0: 0 unstable EVs
    # Point 1: 1 unstable EV (Bifurcation!)
    # Point 2: 1 unstable EV (No crossing)
    unstable_evs = [0, 1, 1]
    for i, nev in enumerate(unstable_evs):
        sol = Solution()
        sol.p = float(i)
        sol.norm = 1.0
        sol.nunstable_eigenvalues = nev
        branch.add_solution_point(sol)

    # Filter for only bifurcations
    p_bif, n_bif = branch.data(only="bifurcations")

    # Verify they are masked arrays
    assert np.ma.isMaskedArray(p_bif)
    assert np.ma.isMaskedArray(n_bif)

    # Verify mask positions: [Masked, Unmasked, Masked]
    # (Since we hid everything that is NOT a bifurcation)
    expected_mask = [True, False, True]
    np.testing.assert_array_equal(p_bif.mask, expected_mask)

    # Check that np.asarray would have failed this test
    stripped = np.asarray(p_bif)
    assert not np.ma.isMaskedArray(stripped)


def test_no_bifurcation_detection() -> None:
    """Test that points are NOT detected as bifurcations if unstable EVs don't change."""
    branch = Branch()

    sol1 = Solution()
    sol1.nunstable_eigenvalues = 0
    branch.add_solution_point(sol1)

    sol2 = Solution()
    sol2.nunstable_eigenvalues = 0
    branch.add_solution_point(sol2)

    assert not sol2.is_bifurcation()
    assert sol2.neigenvalues_crossed == 0
    assert sol2.bifurcation_type() == ""


def test_bifurcation_diagram() -> None:
    bd = BifurcationDiagram()
    assert len(bd.branches) == 1  # Active branch created by default

    branch1 = bd.active_branch
    sol = Solution()
    branch1.add_solution_point(sol)

    assert bd.current_solution() is sol

    new_branch = bd.new_branch()
    assert bd.active_branch is new_branch
    assert len(bd.branches) == 2
