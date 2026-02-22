"""Sociable unit tests for continuation and constraints."""

import numpy as np
import pytest

from bice.continuation.constraints import VolumeConstraint
from bice.continuation.continuation_steppers import NaturalContinuation
from bice.core.equation import Equation
from bice.core.problem import Problem


class ParameterizedEquation(Equation):
    """du/dt = -u + p"""

    def __init__(self):
        super().__init__(shape=(1,))
        self.p = 0.0

    def rhs(self, u):
        return -u + self.p


def test_natural_continuation():
    prob = Problem()
    eq = ParameterizedEquation()
    prob.add_equation(eq)
    prob.u = np.array([0.0])
    prob.continuation_parameter = (eq, "p")

    stepper = NaturalContinuation(ds=0.1)
    prob.continuation_stepper = stepper

    # First point: p=0, u=0
    prob.newton_solve()
    assert eq.p == 0.0
    np.testing.assert_allclose(prob.u, [0.0])

    # Perform continuation step
    prob.continuation_step()

    # Should have p=0.1, and after Newton solve u=0.1
    assert eq.p == pytest.approx(0.1)
    np.testing.assert_allclose(prob.u, [0.1], atol=1e-5)


def test_volume_constraint():
    """Test VolumeConstraint which enforces integral of u = constant."""
    prob = Problem()
    # Simple equation where we want to enforce integral
    # Normally used with PDEs, but let's test with a 2D system
    # eq1: du1/dt = -u1 + source
    # eq2: du2/dt = -u2 + source
    # constraint: u1 + u2 = fixed_volume

    class MultiVarEq(Equation):
        def __init__(self):
            super().__init__(shape=(2,))
            self.source = 0.0  # This will be the Lagrange multiplier from constraint

        def rhs(self, u):
            # source acts on both variables
            return -u + self.source

    eq = MultiVarEq()
    prob.add_equation(eq)

    # Volume constraint on eq
    # It adds one degree of freedom (the source/Lagrange multiplier)
    vc = VolumeConstraint(eq)
    vc.fixed_volume = 10.0
    prob.add_equation(vc)

    # Total DOFs: 2 (from eq) + 1 (from vc) = 3
    assert prob.ndofs == 3

    # Initial guess
    prob.u = np.array([5.0, 5.0, 0.0])

    # Solve
    prob.newton_solve()

    # At steady state: -u1 + source = 0, -u2 + source = 0 => u1 = u2 = source
    # Volume constraint (parametric with fixed_volume=10):
    # trapezoid(u, x) = 10. Since x=[0, 1], trapz is essentially the sum/mean for these discrete points.
    # For N=2 points at x=0, 1, trapz(u, [0, 1]) = 0.5 * (u1 + u2) * (1-0) = 0.5 * (u1 + u2)
    # 0.5 * (u1 + u2) = 10 => u1 + u2 = 20 => 2 * source = 20 => source = 10
    # So u1=10, u2=10, source=10
    np.testing.assert_allclose(prob.u, [10.0, 10.0, 10.0], atol=1e-5)
