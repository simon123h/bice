import numpy as np
from .time_steppers import RungeKutta4
from .continuation_steppers import PseudoArclengthContinuation
from .solvers import NewtonSolver, EigenSolver
from .solution import Solution, BifurcationDiagram


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
        # The Newton solver for finding roots of equations
        self.newton_solver = NewtonSolver()
        # The eigensolver for eigenvalues and -vectors
        self.eigen_solver = EigenSolver()
        # The bifurcation diagram of the problem holds all branches and their solutions
        self.bifurcation_diagram = BifurcationDiagram()
        # how small does an eigenvalue need to be in order to be counted as 'zero'?
        self.eigval_zero_tolerance = 1e-6

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

    # Solve the system rhs(u) = 0 for u with Newton's method
    def newton_solve(self):
        # TODO: check for convergence
        self.u = self.newton_solver.solve(self.rhs, self.u)

    # Calculate the eigenvalues and eigenvectors of the Jacobian
    # optional argument k: number of requested eigenvalues
    def solve_eigenproblem(self, k=None):
        return self.eigen_solver.solve(self.jacobian(self.u), k)

    # Integrate in time with the assigned time-stepper
    def time_step(self):
        # perform timestep according to current scheme
        self.time_stepper.step(self)

    # Perform a parameter continuation step, w.r.t the parameter defined by
    # self.continuation_stepper.get_continuation_parameter/set_continuation_parameter()
    def continuation_step(self):
        # get the current branch in the bifurcation diagram
        branch = self.bifurcation_diagram.current_branch()
        # if the branch is empty, add initial point
        if branch.is_empty():
            sol = Solution(self)
            branch.add_solution_point(sol)
            # solve the eigenproblem
            sol.eigenvalues, sol.eigenvectors = self.solve_eigenproblem()
        # save the sign of the jacobian
        jac_sign = np.linalg.slogdet(self.jacobian(self.u))[0]
        # perform the step with a continuation stepper
        self.continuation_stepper.step(self)
        # add the solution to the branch
        sol = Solution(self)
        branch.add_solution_point(sol)
        # detect sign change in jacobian
        jac_sign *= np.linalg.slogdet(self.jacobian(self.u))[0]
        # if desired or Jac sign changed, solve the eigenproblem
        if self.continuation_stepper.always_check_eigenvalues or jac_sign < 0:
            sol.eigenvalues, sol.eigenvectors = self.solve_eigenproblem()
        # return the solution object
        return sol

    # create a new branch in the bifurcation diagram and prepare for a new continuation
    def new_branch(self):
        # create a new branch in the bifurcation diagram
        self.bifurcation_diagram.new_branch()
        # reset the settings and storage of the continuation stepper
        self.continuation_stepper.factory_reset()

    # the default norm of the solution, used for bifurcation diagrams
    def norm(self):
        # defaults to the L2-norm
        return np.linalg.norm(self.u)

    # save the current solution to disk
    def save(self, filename):
        np.savetxt(filename, self.u)

    # load the current solution from disk
    def load(self, filename):
        self.u = np.loadtxt(filename)
