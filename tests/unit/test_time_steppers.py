"""Sociable unit tests for time stepper classes."""

import numpy as np
import pytest

from bice.core.equation import Equation
from bice.core.problem import Problem
from bice.time_steppers.bdf import BDF2
from bice.time_steppers.runge_kutta import RungeKutta4
from bice.time_steppers.time_steppers import Euler, ImplicitEuler


class DecayEquation(Equation):
    """du/dt = -u. Analytical solution: u(t) = u(0) * exp(-t)."""

    def __init__(self, shape=(1,)):
        super().__init__(shape=shape)

    def rhs(self, u):
        return -u

    def jacobian(self, u):
        return np.array([[-1.0]])


def test_euler_step():
    prob = Problem()
    prob.add_equation(DecayEquation())
    prob.u = np.array([1.0])
    dt = 0.1
    prob.time_stepper = Euler(dt=dt)

    prob.time_step()
    # Euler: u_new = u_old + dt * f(u_old) = 1.0 + 0.1 * (-1.0) = 0.9
    np.testing.assert_allclose(prob.u, [0.9])
    assert prob.time == pytest.approx(0.1)


def test_implicit_euler_step():
    prob = Problem()
    prob.add_equation(DecayEquation())
    prob.u = np.array([1.0])
    dt = 0.1
    prob.time_stepper = ImplicitEuler(dt=dt)

    prob.time_step()
    # Implicit Euler: u_new = u_old + dt * f(u_new)
    # u_new = 1.0 + 0.1 * (-u_new) => u_new (1 + 0.1) = 1.0 => u_new = 1.0 / 1.1 approx 0.90909
    np.testing.assert_allclose(prob.u, [1.0 / 1.1])


def test_rk4_step():
    prob = Problem()
    prob.add_equation(DecayEquation())
    prob.u = np.array([1.0])
    dt = 0.1
    prob.time_stepper = RungeKutta4(dt=dt)

    prob.time_step()
    # RK4 is much more accurate. Analytical: exp(-0.1) approx 0.904837
    expected = np.exp(-0.1)
    np.testing.assert_allclose(prob.u, [expected], atol=1e-5)


def test_bdf2_step():
    prob = Problem()
    prob.add_equation(DecayEquation())
    prob.u = np.array([1.0])
    dt = 0.1
    prob.time_stepper = BDF2(dt=dt)

    # BDF2 needs history. First step might fallback to something else or use available history.
    # Implementation check:
    # u_1 = problem.u
    # u_2 = problem.history.u(1) if problem.history.length > 1 else u_1

    prob.time_step()
    # If no history, it uses u_1 for both u_1 and u_2 in the formula?
    # Actually, BDF2 step 1 with same values usually gives a reasonable first step.
    assert prob.u[0] < 1.0

    # Let's do a second step to see it uses history
    u_after_step1 = prob.u.copy()
    prob.time_step()
    assert prob.u[0] < u_after_step1[0]
    assert prob.history.length == 2
