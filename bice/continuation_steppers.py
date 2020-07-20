import numpy as np
import scipy.optimize


class ContinuationStepper:
    """
    Abstract base class for all parameter continuation-steppers.
    Specifies attributes and methods that all continuation-steppers should have.
    """

    # constructor
    def __init__(self, ds=1e-3):
        self.ds = ds

    # perform a continuation step on a problem
    def step(self, problem):
        raise NotImplementedError(
            "'ContinuationStepper' is an abstract base class - do not use for actual parameter continuation!")

    # getter for the continuation parameter (override in order to set parameter)
    def get_parameter(self):
        raise NotImplementedError(
            "Overwrite this method with a getter for the principal continuation parameter!")
    # setter for the continuation parameter (override in order to set parameter)

    def set_parameter(self, val):
        raise NotImplementedError(
            "Overwrite this method with a setter for the principal continuation parameter!")


class NaturalContinuation(ContinuationStepper):
    """
    Natural parameter continuation stepper
    """

    # perform continuation step
    def step(self, problem):
        # update the parameter value
        p = problem.get_parameter()
        problem.set_parameter(p + self.ds)
        # solve it with a Newton solver
        # TODO: detect if Newton solver failed and reject step
        problem.u = scipy.optimize.newton_krylov(problem.rhs, problem.u)


class PseudoArclengthContinuation(ContinuationStepper):
    """
    Pseudo-arclength parameter continuation stepper
    """

    def __init__(self, ds=1e-3):
        super().__init__(ds)
        # if the norm of the step in the newton loop is below this threshold, the method has converged
        self.convergence_epsilon = 1e-10
        # maximum number of newton iterations for solving
        self.max_newton_iterations = 30
        # should the step size be adapted while stepping?
        self.adapt_stepsize = True
        # the desired number of newton iterations for solving, step size is adapted if we over/undershoot this number
        self.ndesired_newton_steps = 8
        # the actual number of newton iterations taken in the last continuation step
        self.nnewton_iter_taken = None
        # ds 'penalty' factor when less nthan desired_newton_steps are performed
        self.ds_decrease_factor = 0.5
        # ds 'penalty' factor when more nthan desired_newton_steps are performed
        self.ds_increase_factor = 1.05
        # maximum step size
        self.ds_max = 1e0
        # minimum step size
        self.ds_min = 1e-9
        # rescale the parameter constraint, for numerical stability
        # may be decreased for, e.g., very sharp folds
        self.constraint_scale = 1
        # finite-difference for calculating parameter derivatives
        self.fd_epsilon = 1e-10
        # stores the tangent between steps in order to use it for the next step
        self.tangent = None

    # perform continuation step

    def step(self, problem):
        p = problem.get_parameter()
        u = problem.u
        N = u.size
        # save old variables
        u_old = u.copy()
        p_old = p
        if self.tangent is not None:
            # simply get tangent from difference between last steps
            # NOTE: is it a good idea to always reuse the tangent or should we be able to switch this off?
            tangent = self.tangent
        else:
            # calculate tangent from extended Jacobian in (u, parameter)-space
            jac = problem.jacobian(u)
            # last column of extended jacobian: d(rhs)/d(parameter), calculate it with FD
            problem.set_parameter(p - self.fd_epsilon)
            rhs_1 = problem.rhs(u)
            problem.set_parameter(p + self.fd_epsilon)
            rhs_2 = problem.rhs(u)
            drhs_dp = (rhs_2 - rhs_1) / (2. * self.fd_epsilon)
            problem.set_parameter(p)
            jac = np.concatenate((jac, drhs_dp.reshape((N, 1))), axis=1)
            zero = np.zeros(N+1)
            zero[N] = 1  # for solvability
            jac = np.concatenate((jac, zero.reshape((1, N+1))), axis=0)
            # compute tangent by solving (jac)*tangent=0 and normalize
            tangent = np.linalg.solve(jac, zero)
            tangent /= np.linalg.norm(tangent)
        # make initial guess: u -> u + ds * tangent
        u = u + self.ds * tangent[:N]
        p = p + self.ds * tangent[N]

        converged = False
        count = 0
        while not converged and count < self.max_newton_iterations:
            # build extended jacobian in (vars, parameter)-space
            problem.set_parameter(p)
            rhs = problem.rhs(u)
            jac = problem.jacobian(u)
            # last column of extended jacobian: d(rhs)/d(parameter) - calculate with FD
            problem.set_parameter(p - self.fd_epsilon)
            rhs_1 = problem.rhs(u)
            problem.set_parameter(p + self.fd_epsilon)
            rhs_2 = problem.rhs(u)
            problem.set_parameter(p)
            drhs_dp = (rhs_2 - rhs_1) / (2. * self.fd_epsilon)
            jac_ext = np.concatenate((jac, drhs_dp.reshape((N, 1))), axis=1)
            # last row of extended jacobian: tangent vector
            jac_ext = np.concatenate(
                (jac_ext, tangent.reshape((1, N+1))), axis=0)
            # extended rhs: model's rhs + arclength condition
            rhs_ext = (u - u_old).dot(tangent[:N]) + (p - p_old) * \
                tangent[N] * self.constraint_scale - self.ds
            rhs_ext = np.append(rhs, rhs_ext)
            # solving (jac_ext) * du_ext = rhs_ext for du_ext will now give the new solution
            du_ext = np.linalg.solve(jac_ext, rhs_ext)
            u = u - du_ext[:N]
            p = p - du_ext[N]
            # update counter and check for convergence
            count += 1
            converged = np.linalg.norm(du_ext) < self.convergence_epsilon

        # update number of steps taken
        self.nnewton_iter_taken = count

        if converged:
            # system converged to new solution
            # assign the new values
            problem.u = u
            problem.set_parameter(p)
            # approximate tangent for the following step
            self.tangent = np.append(u - u_old, p-p_old)
            self.tangent /= np.linalg.norm(self.tangent)
        else:
            # we didn't converge :-(
            # reset to old values
            problem.u = u_old
            problem.set_parameter(p_old)
            # TODO: not converged, throw error

        # adapt step size
        if self.adapt_stepsize:
            if count > self.ndesired_newton_steps and abs(self.ds) > self.ds_min:
                # decrease step size
                sign = -1 if self.ds < 0 else 1
                self.ds = max(
                    abs(self.ds)*self.ds_decrease_factor, self.ds_min)*sign
                # redo continuation step
                # self.u = u_old
                # problem.set_parameter(p_old)
                # self.step(problem)
            elif count < self.ndesired_newton_steps:
                sign = -1 if self.ds < 0 else 1
                self.ds = min(
                    abs(self.ds)*self.ds_increase_factor, self.ds_max)*sign
