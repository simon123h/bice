import numpy as np
import scipy.optimize


class ContinuationStepper:
    """
    Abstract base class for all parameter continuation-steppers.
    Specifies attributes and methods that all continuation-steppers should have.
    """

    # constructor
    def __init__(self, ds=1e-2):
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
        problem.u = scipy.optimize.newton(problem.rhs, problem.u)


class PseudoArclengthContinuation(ContinuationStepper):
    """
    Pseudo-arclength parameter continuation stepper
    """

    def __init__(self, ds):
        super().__init__(ds)
        # rescale the parameter constraint, for numerical stability
        # may be decreased for, e.g., very sharp folds
        self.constraint_scale = 1

    # perform continuation step
    def step(self, problem):
        p = problem.get_parameter()
        u = problem.u
        count = 0
        N = u.size
        # TODO: use tangent vector from previous run?
        # build extended jacobian in (u, parameter)-space for tangent calculation
        jac = problem.jacobian(u)
        # last column of extended jacobian: d(rhs)/d(parameter) - calculate with FD
        eps = 1e-10
        problem.set_parameter(p - eps)
        rhs_1 = problem.rhs(u)
        problem.set_parameter(p + eps)
        rhs_2 = problem.rhs(u)
        drhs_dp = (rhs_2 - rhs_1) / (2. * eps)
        problem.set_parameter(p)
        jac = np.concatenate((jac, drhs_dp.reshape((N, 1))), axis=1)
        zero = np.zeros(N+1)
        zero[N] = 1  # for solvability
        jac = np.concatenate((jac, zero.reshape((1, N+1))), axis=0)
        # compute tangent by solving (jac)*tangent=0
        tangent = np.linalg.solve(jac, zero)
        # normalize
        tangent /= np.linalg.norm(tangent)
        # save old variables
        u_old = u
        p_old = p
        # make initial guess: u -> u + ds * tangent
        u = u + self.ds * tangent[:N]
        p = p + self.ds * tangent[N]
        # initial guess for the step in (u, p)-space
        dU_ext = np.append(u - u_old, p - p_old)

        # assemble the extended system: (jac_ext) * u_ext = rhs_ext
        def f(u_ext):
            # build extended jacobian in (vars, parameter)-space
            u = u_ext[:N]
            p = u_ext[N]
            problem.set_parameter(p)
            rhs = problem.rhs(u)
            jac = problem.jacobian(u)
            # last column of extended jacobian: d(rhs)/d(parameter) - calculate with FD
            problem.set_parameter(p - eps)
            rhs_1 = problem.rhs(u)
            problem.set_parameter(p + eps)
            rhs_2 = problem.rhs(u)
            drhs_dp = (rhs_2 - rhs_1) / (2. * eps)
            jac_ext = np.concatenate((jac, drhs_dp.reshape((N, 1))), axis=1)
            # last row of extended jacobian: tangent vector
            jac_ext = np.concatenate(
                (jac_ext, tangent.reshape((1, N+1))), axis=0)
            # extended rhs: model's rhs + arclength condition
            rhs_ext = np.sum((u - u_old) * tangent[:N]) + (
                p - p_old) * tangent[N] * self.constraint_scale - self.ds
            rhs_ext = np.append(rhs, rhs_ext)
            # solving (jac_ext) * dU_ext = rhs_ext for dU_ext will now give the new solution
            return jac.dot(u_ext) - rhs_ext

        # solve it with a Newton solver
        # TODO: detect if Newton solver failed and reject step
        dU_ext = scipy.optimize.newton(f, dU_ext)

        # assign the new values
        problem.u = u_old + dU_ext[:N]
        problem.set_parameter(p_old + dU_ext[N])
