from .time_steppers import RungeKutta4
from .continuation_steppers import PseudoArclengthContinuation
import numpy as np


class Problem():
    """
    All algebraic problems inherit from the 'Problem' class.
    It is an aggregate of (one or many) governing algebraic equations,
    initial and boundary conditions, constraints, etc. Also, it provides
    all the basic properties, methods and routines for treating the problem,
    e.g., time-stepping, solvers or plain analysis of the solution.
    Custom problems should be implemented as children of this class.
    """

    # Constructor: initialize basic ar
    def __init__(self):
        # the list of governing equations in this problem
        self.eq = []
        # The vector of unknowns (NumPy array)
        self.u = None
        # Time variable
        self.time = 0
        # The time-stepper for integration in time
        self.time_stepper = RungeKutta4(dt=1e-2)
        # The continuation stepper for parameter continuation
        self.continuation_stepper = PseudoArclengthContinuation()

    # The dimension of the system
    @property
    def dim(self):
        return self.u.size

    # Calculate the right-hand side of the system 0 = rhs(u)
    def rhs(self, u):
        raise NotImplementedError(
            "No right-hand side (rhs) implemented for this problem!")

    # Calculate the Jacobian of the system J = d rhs(u) / du for the unknowns u.
    # 'eps' is the step size used for the central FD scheme
    def jacobian(self, u, eps=1e-10):
        # default implementation: calculate Jacobian with finite differences
        J = np.zeros([self.dim, self.dim], dtype=np.float)
        for i in range(self.dim):
            u1 = u.copy()
            u2 = u.copy()
            u1[i] += eps
            u2[i] -= eps
            f1 = self.rhs(u1)
            f2 = self.rhs(u2)
            J[:, i] = (f1 - f2) / (2 * eps)
        return J

    # Integrate in time with the assigned time-stepper
    def time_step(self):
        # perform timestep according to current scheme
        self.time_stepper.step(self)

    # Solve the equation rhs(u) = 0 for u with the assigned linear solver
    # Perform a parameter continuation step, w.r.t the parameter defined by
    # self.continuation_stepper.get_parameter/set_parameter

    def continuation_step(self):
        self.continuation_stepper.step(self)
