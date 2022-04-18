from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from bice.core.profiling import profile
from bice.core.types import Array

if TYPE_CHECKING:
    from bice.core.problem import Problem


class ContinuationStepper:
    """
    Abstract base class for all parameter continuation-steppers.
    Specifies attributes and methods that all continuation-steppers should have.
    """

    # constructor
    def __init__(self, ds: float = 1e-3) -> None:
        #: continuation step size
        self.ds = ds

    def step(self, problem: 'Problem') -> None:
        """Perform a continuation step on a problem"""
        raise NotImplementedError(
            "'ContinuationStepper' is an abstract base class - "
            "do not use for actual parameter continuation!")

    def factory_reset(self) -> None:
        """
        Reset the continuation-stepper parameters & storage to default, e.g.,
        when starting off a new solution point, switching branches or
        switching the principal continuation parameter
        """
        # TODO: is this method even needed?
        pass


class NaturalContinuation(ContinuationStepper):
    """
    Natural parameter continuation stepper
    """

    def step(self, problem: 'Problem') -> None:
        """Perform a continuation step on a problem"""
        # update the parameter value
        p = problem.get_continuation_parameter()
        problem.set_continuation_parameter(p + self.ds)
        # solve the problem with a Newton solver
        problem.newton_solve()


class PseudoArclengthContinuation(ContinuationStepper):
    """
    Pseudo-arclength parameter continuation stepper
    """

    def __init__(self, ds: float = 1e-3) -> None:
        super().__init__(ds)
        #: convergence tolerance for the newton solver in the continuation step
        self.convergence_tolerance = 1e-8
        #: maximum number of newton iterations for solving
        self.max_newton_iterations = 30
        #: should the step size be adapted while stepping?
        self.adapt_stepsize = True
        #: the desired number of newton iterations for solving,
        #: step size is adapted if we over/undershoot this number
        self.ndesired_newton_steps = 3
        #: the actual number of newton iterations taken in the last continuation step
        self.nnewton_iter_taken = None
        #: ds decreases by this factor when less than desired_newton_steps are performed
        self.ds_decrease_factor = 0.5
        #: ds increases by this factor when more than desired_newton_steps are performed
        self.ds_increase_factor = 1.1
        #: maximum step size
        self.ds_max = 1e0
        #: minimum step size
        self.ds_min = 1e-9
        #: Rescale the parameter constraint, for numerical stability.
        #: May be decreased, e.g. for very sharp folds.
        self.parameter_arc_length_proportion = 1
        #: finite-difference for calculating parameter derivatives
        self.fd_epsilon = 1e-10

    @profile
    def step(self, problem: 'Problem') -> None:
        """Perform a continuation step on a problem"""
        p = problem.get_continuation_parameter()
        u = problem.u
        N = u.size
        # save the old variables
        u_old, p_old = u.copy(), p
        # check if we know at least the two previous continuation points
        if problem.history.length > 1 and problem.history.type == "continuation":
            # if yes, we can approximate the tangent in phase-space from the history points
            # TODO: use higher order polynomial predictor?
            tangent = np.append(u - problem.history.u(-1),
                                p - problem.history.continuation_parameter(-1))
            # normalize tangent and adjust sign with respect to continuation direction
            tangent /= np.linalg.norm(tangent) * \
                np.sign(problem.history.step_size(-1))
        else:
            # else, we need to calculate the tangent from extended Jacobian in (u, parameter)-space
            jac = problem.jacobian(u)
            # TODO: detect if jacobian is sparse and decide whether to use np / sp methods
            if not sp.issparse(jac):
                jac = sp.coo_matrix(jac)
            # last column of extended jacobian: d(rhs)/d(parameter), calculate it with FD
            problem.set_continuation_parameter(p - self.fd_epsilon)
            rhs_1 = problem.rhs(u)
            problem.set_continuation_parameter(p + self.fd_epsilon)
            rhs_2 = problem.rhs(u)
            drhs_dp = (rhs_2 - rhs_1) / (2. * self.fd_epsilon)
            problem.set_continuation_parameter(p)
            jac = sp.hstack((jac, drhs_dp.reshape((N, 1))))
            zero = np.zeros(N+1)
            zero[N] = 1  # for solvability, determines length of tangent vector
            jac = sp.vstack((jac, zero.reshape((1, N+1))))
            # compute tangent by solving (jac)*tangent=0 and normalize
            tangent = self._linear_solve(
                jac, zero, problem.settings.use_sparse_matrices)
            tangent /= np.linalg.norm(tangent)
            # make sure that the tangent points in positive parameter direction
            tangent *= np.sign(tangent[-1])
        # make initial guess: u -> u + ds * tangent
        u = u + self.ds * tangent[:N]
        p = p + self.ds * tangent[N]

        converged = False
        count = 0
        while not converged and count < self.max_newton_iterations:
            # build extended jacobian in (u, parameter)-space
            problem.set_continuation_parameter(p)
            jac = problem.jacobian(u)
            if not sp.issparse(jac):
                jac = sp.coo_matrix(jac)
            # last column of extended jacobian: d(rhs)/d(parameter) - calculate with FD
            problem.set_continuation_parameter(p - self.fd_epsilon)
            rhs_1 = problem.rhs(u)
            problem.set_continuation_parameter(p + self.fd_epsilon)
            rhs_2 = problem.rhs(u)
            problem.set_continuation_parameter(p)
            drhs_dp = (rhs_2 - rhs_1) / (2. * self.fd_epsilon)
            jac_ext = sp.hstack((jac, drhs_dp.reshape((N, 1))))
            # last row of extended jacobian: tangent vector
            jac_ext = sp.vstack((jac_ext, tangent.reshape((1, N+1))))
            # extended rhs: model's rhs & arclength condition
            arclength_condition = (u - u_old).dot(tangent[:N]) + (p - p_old) * \
                tangent[N] * self.parameter_arc_length_proportion - self.ds
            rhs_ext = np.append(problem.rhs(u), arclength_condition)
            # solving (jac_ext) * du_ext = rhs_ext for du_ext will now give the new solution
            du_ext = self._linear_solve(
                jac_ext, rhs_ext, problem.settings.use_sparse_matrices)
            u -= du_ext[:N]
            p -= du_ext[N]
            # TODO: use max(rhs_ext(u)) < tol as convergence check, as in other solvers?
            # update counter and check for convergence
            count += 1
            converged = np.linalg.norm(du_ext) < self.convergence_tolerance

        # update number of steps taken
        self.nnewton_iter_taken = count

        if converged:
            # system converged to new solution, assign the new values
            problem.u = u
            problem.set_continuation_parameter(p)
        else:
            # we didn't converge, reset to old values :-/
            problem.u = u_old
            problem.set_continuation_parameter(p_old)
            # if step size is already minimal, throw an error
            if abs(self.ds) < self.ds_min:
                raise np.linalg.LinAlgError(
                    f"Newton solver did not converge after {count} iterations!")
            # else, retry with a smaller step size
            self.ds /= 2
            print(
                f"Newton solver did not converge, trying again with ds = {self.ds:.3e}")
            return self.step(problem)

        # adapt step size
        if self.adapt_stepsize:
            if count > self.ndesired_newton_steps and abs(self.ds) > self.ds_min:
                # decrease step size
                self.ds = max(
                    abs(self.ds)*self.ds_decrease_factor, self.ds_min)*np.sign(self.ds)
            elif count < self.ndesired_newton_steps:
                # increase step size
                self.ds = min(
                    abs(self.ds)*self.ds_increase_factor, self.ds_max)*np.sign(self.ds)

    def _linear_solve(self, A, b: Array, use_sparse_matrices: bool = False):
        """Solve the linear system A*x = b for x and return x"""
        # if desired, convert A to sparse matrix
        if use_sparse_matrices and not sp.issparse(A):
            A = sp.csr_matrix(A)
        # use either a solver for sparse matrices...
        if sp.issparse(A):
            return sp.linalg.spsolve(sp.csr_matrix(A), b)
        # ...or simply the one for full rank matrices
        return np.linalg.solve(A, b)
