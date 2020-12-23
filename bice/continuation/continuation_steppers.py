import numpy as np
import scipy.optimize


class ContinuationStepper:
    """
    Abstract base class for all parameter continuation-steppers.
    Specifies attributes and methods that all continuation-steppers should have.
    """

    # constructor
    def __init__(self, ds=1e-3):
        # continuation step size
        self.ds = ds

    # perform a continuation step on a problem
    def step(self, problem):
        raise NotImplementedError(
            "'ContinuationStepper' is an abstract base class - "
            "do not use for actual parameter continuation!")

    # reset the continuation-stepper parameters & storage to default, e.g.,
    # when starting off a new solution point, switching branches or
    # switching the principal continuation parameter
    def factory_reset(self):
        pass


class NaturalContinuation(ContinuationStepper):
    """
    Natural parameter continuation stepper
    """

    # perform continuation step
    def step(self, problem):
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

    # perform continuation step
    def step(self, problem):
        p = problem.get_continuation_parameter()
        u = problem.u
        N = u.size
        # save the old variables
        u_old, p_old = u.copy(), p
        # check if we know at least the two previous continuation points
        if problem.history.length > 1 and problem.history.type == "continuation":
            # if yes, we can approximate the tangent in phase-space from the history points
            tangent = np.append(u - problem.history.u(-1),
                                p - problem.history.continuation_parameter(-1))
            # normalize tangent and adjust sign with respect to continuation direction
            tangent /= np.linalg.norm(tangent) * \
                np.sign(problem.history.step_size(-1))
        else:
            # else, we need to calculate the tangent from extended Jacobian in (u, parameter)-space
            jac = problem.jacobian(u)
            # last column of extended jacobian: d(rhs)/d(parameter), calculate it with FD
            problem.set_continuation_parameter(p - self.fd_epsilon)
            rhs_1 = problem.rhs(u)
            problem.set_continuation_parameter(p + self.fd_epsilon)
            rhs_2 = problem.rhs(u)
            drhs_dp = (rhs_2 - rhs_1) / (2. * self.fd_epsilon)
            problem.set_continuation_parameter(p)
            jac = np.concatenate((jac, drhs_dp.reshape((N, 1))), axis=1)
            zero = np.zeros(N+1)
            zero[N] = 1  # for solvability
            jac = np.concatenate((jac, zero.reshape((1, N+1))), axis=0)
            # compute tangent by solving (jac)*tangent=0 and normalize
            tangent = self._linear_solve(
                jac, zero, problem.settings.use_sparse_matrices)
            tangent /= np.linalg.norm(tangent)
            # NOTE: if we multiply tangent with sign(tangent-1),
            #       we could make sure that it points in positive parameter direction
        # make initial guess: u -> u + ds * tangent
        u = u + self.ds * tangent[:N]
        p = p + self.ds * tangent[N]

        converged = False
        count = 0
        # TODO: use problem.newton_solver instead of homebrewed newton solver
        while not converged and count < self.max_newton_iterations:
            # build extended jacobian in (u, parameter)-space
            problem.set_continuation_parameter(p)
            jac = problem.jacobian(u)
            # last column of extended jacobian: d(rhs)/d(parameter) - calculate with FD
            problem.set_continuation_parameter(p - self.fd_epsilon)
            rhs_1 = problem.rhs(u)
            problem.set_continuation_parameter(p + self.fd_epsilon)
            rhs_2 = problem.rhs(u)
            problem.set_continuation_parameter(p)
            drhs_dp = (rhs_2 - rhs_1) / (2. * self.fd_epsilon)
            jac_ext = np.concatenate((jac, drhs_dp.reshape((N, 1))), axis=1)
            # last row of extended jacobian: tangent vector
            jac_ext = np.concatenate(
                (jac_ext, tangent.reshape((1, N+1))), axis=0)
            # extended rhs: model's rhs & arclength condition
            arclength_condition = (u - u_old).dot(tangent[:N]) + (p - p_old) * \
                tangent[N] * self.parameter_arc_length_proportion - self.ds
            rhs_ext = np.append(problem.rhs(u), arclength_condition)
            # solving (jac_ext) * du_ext = rhs_ext for du_ext will now give the new solution
            du_ext = self._linear_solve(jac_ext, rhs_ext)
            u -= du_ext[:N]
            p -= du_ext[N]
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
            # and throw error
            # TODO: we could also try again with a smaller step size, unless ds is already minimal
            raise np.linalg.LinAlgError(
                "Newton solver did not converge after {:d} iterations!".format(count))

        # adapt step size
        if self.adapt_stepsize:
            if count > self.ndesired_newton_steps and abs(self.ds) > self.ds_min:
                # decrease step size
                self.ds = max(
                    abs(self.ds)*self.ds_decrease_factor, self.ds_min)*np.sign(self.ds)
                # redo continuation step
                # self.u = u_old
                # problem.set_continuation_parameter(p_old)
                # self.step(problem)
            elif count < self.ndesired_newton_steps:
                self.ds = min(
                    abs(self.ds)*self.ds_increase_factor, self.ds_max)*np.sign(self.ds)

    # Solve the linear system A*x = b for x and return x
    def _linear_solve(self, A, b, use_sparse_matrices=False):
        if use_sparse_matrices:
            # use either solver for sparse matrices...
            return scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(A), b)
        # ...or simply the one for full rank matrices
        return np.linalg.solve(A, b)


class DeflatedContinuation(ContinuationStepper):
    """
    Deflated continuation stepper
    """

    def __init__(self, ds=1e-3):
        super().__init__(ds)
        # list of known solutions, that will be suppressed using the deflation operator
        self.known_solutions = []
        # the order of the norm that will be used for the deflation operator
        self.norm_ord = 2

    # deflation operator for given u
    def deflation_operator(self, u):
        return 1. / np.prod([np.linalg.norm(uk - u, ord=self.norm_ord)
                             for uk in self.known_solutions])

    # Jacobian of deflation operator for given u
    def deflation_operator_jac(self, u):
        # TODO: implement
        return 0

    # perform deflated continuation step
    def step(self, problem):
        # clear list of known solutions
        self.known_solutions = []

        # deflated rhs: multiplication of original rhs with deflation operator
        def deflated_rhs(u):
            # multiply with rhs and return
            return self.deflation_operator(u) * problem.rhs(u)

        # Jacobian of deflated rhs
        def deflated_jacobian(u):
            # obtain rhs, jacobian, deflation operator and operator derivative
            rhs = problem.rhs(u)
            jac = problem.jacobian(u)
            def_op = self.deflation_operator(u)
            ddef_op = self.deflation_operator_jac(u)
            return scipy.sparse.diags(ddef_op * rhs) + def_op * jac

        # save initial guess
        u0 = problem.u  # TODO: is there a better choice for the initial guess?

        # search for unknown solutions in a loop, will break when no solution found
        while True:
            # try to solve problem with Newton solver, starting from u0
            try:
                # call Newton solver with deflated rhs
                # TODO: include deflated Jacobian
                u_new = problem.newton_solver.solve(deflated_rhs, u0)
                # add new solution to known solutions
                self.known_solutions.append(u_new)
            except scipy.optimize.NoConvergence:  # TODO: support other Newton solver's exceptions
                # did not converge! Possibly, there are no unknown solutions left
                break

        # check if any solutions were found
        if len(self.known_solutions) == 0:
            # TODO: raise exception?
            print("Warning: deflated continuation found no solutions at all!")

        # increase value of continuation parameter by ds
        p = problem.get_continuation_parameter()
        problem.set_continuation_parameter(p + self.ds)
