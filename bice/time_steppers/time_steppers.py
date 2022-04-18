
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bice.core.problem import Problem


class TimeStepper:
    """
    Abstract base class for all time-steppers.
    Specifies attributes and methods that all time-steppers should have.
    """

    # constructor
    def __init__(self, dt: float = 1e-2) -> None:
        #: the time step size
        self.dt = dt

    # # calculate the time derivative of the unknowns for a given problem
    # def get_dudt(self, problem, u):
    #     raise NotImplementedError(
    #         "Method 'get_dudt' not implemented for this time-stepper!")

    # perform a timestep on a problem
    def step(self, problem: 'Problem') -> None:
        raise NotImplementedError(
            "'TimeStepper' is an abstract base class - do not use for actual time-stepping!")


class Euler(TimeStepper):
    """
    Explicit Euler scheme
    """

    # perform timestep
    def step(self, problem: 'Problem') -> None:
        problem.u += self.dt * problem.rhs(problem.u)
        problem.time += self.dt


class ImplicitEuler(TimeStepper):
    """
    Implicit Euler scheme
    """

    def step(self, problem: 'Problem') -> None:
        # advance in time
        problem.time += self.dt
        # obtain the mass matrix
        M = problem.mass_matrix()

        def f(u):
            # assemble the system
            return problem.rhs(u) - M.dot(u - problem.u) / self.dt

        def J(u):
            # Jacobian of the system
            return problem.jacobian(u) - M / self.dt
        # solve it with a Newton solver
        # TODO: detect if Newton solver failed and reject step
        problem.u = problem.newton_solver.solve(f, problem.u, J)
