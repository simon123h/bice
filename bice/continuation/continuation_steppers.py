import numpy as np
import scipy.optimize
import scipy.sparse as sp


class ContinuationStepper:
    """
    Abstract base class for all parameter continuation-steppers.
    Specifies attributes and methods that all continuation-steppers should have.
    """

    # constructor
    def __init__(self, ds=1e-3):
        # continuation step size
        self.ds = ds

    def step(self, problem):
        """Perform a continuation step on a problem"""
        raise NotImplementedError(
            "'ContinuationStepper' is an abstract base class - "
            "do not use for actual parameter continuation!")

    def factory_reset(self):
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

    def step(self, problem):
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

    def __init__(self, ds=1e-3):
        super().__init__(ds)
        # convergence tolerance for the newton solver in the continuation step
        self.convergence_tolerance = 1e-8
        # maximum number of newton iterations for solving
        self.max_newton_iterations = 30
        # should the step size be adapted while stepping?
        self.adapt_stepsize = True
        # the desired number of newton iterations for solving,
        # step size is adapted if we over/undershoot this number
        self.ndesired_newton_steps = 3
        # the actual number of newton iterations taken in the last continuation step
        self.nnewton_iter_taken = None
        # ds decreases by this factor when less than desired_newton_steps are performed
        self.ds_decrease_factor = 0.5
        # ds increases by this factor when more than desired_newton_steps are performed
        self.ds_increase_factor = 1.1
        # maximum step size
        self.ds_max = 1e0
        # minimum step size
        self.ds_min = 1e-9
        # rescale the parameter constraint, for numerical stability
        # may be decreased for, e.g., very sharp folds
        self.parameter_arc_length_proportion = 1
        # finite-difference for calculating parameter derivatives
        self.fd_epsilon = 1e-10

    def step(self, problem):
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
            zero[N] = 1  # for solvability
            jac = sp.vstack((jac, zero.reshape((1, N+1))))
            # compute tangent by solving (jac)*tangent=0 and normalize
            tangent = self._linear_solve(
                jac, zero, problem.settings.use_sparse_matrices)
            tangent /= np.linalg.norm(tangent)
            # NOTE: if we multiply tangent with sign(tangent-1),
            #       we could make sure that it points in positive parameter direction
        # make initial guess: u -> u + ds * tangent
        u = u + self.ds * tangent[:N]
        p = p + self.ds * tangent[N]

        def f(up):
            u = up[:-1]
            p = up[-1]
            # extended rhs: model's rhs & arclength condition
            arclength_condition = (u - u_old).dot(tangent[:N]) + (p - p_old) * \
                tangent[N] * self.parameter_arc_length_proportion - self.ds
            return np.append(problem.rhs(u), arclength_condition)

        # build extended jacobian in (u, parameter)-space
        def J(up):
            u = up[:-1]
            p = up[-1]
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
            return sp.vstack((jac_ext, tangent.reshape((1, N+1))))

        up = np.append(u, p)
        up = problem.newton_solver.solve(f, up, J)
        u = up[:-1]
        p = up[-1]

        # update number of steps taken
        self.nnewton_iter_taken = problem.newton_solver.niterations
        count = self.nnewton_iter_taken

        converged = True
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
                    "Newton solver did not converge after {:d} iterations!".format(count))
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

    # Solve the linear system A*x = b for x and return x
    def _linear_solve(self, A, b, use_sparse_matrices=False):
        if use_sparse_matrices or sp.issparse(A):
            # use either solver for sparse matrices...
            A = sp.csr_matrix(A)
            return sp.linalg.spsolve(A, b)
        # ...or simply the one for full rank matrices
        return np.linalg.solve(A, b)

