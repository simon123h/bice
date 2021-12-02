import numpy as np
import scipy.sparse as sp
import scipy.optimize
import scipy.linalg
from .profiling import profile


class AbstractNewtonSolver:
    """
    Abstract base class for all Newton solvers.
    Newton solver find the root a (possibly high-dimensional, nonlinear) function f(u) = 0 using
    a stepping procedure. An initial guess u0 for the solution needs to be supplied. Also giving the
    Jacobian J = df/du will speed up the computation and is even required for some solvers.
    """

    def __init__(self):
        # maximum number of steps during solve
        self.max_iterations = 100
        # absolute convergence tolerance for norm or residuals
        self.convergence_tolerance = 1e-8
        # how verbose should the solving be? 0 = quiet, larger numbers = print more details
        self.verbosity = 0
        # internal storage for the number of iterations taken during last solve
        self._iteration_count = None

    def solve(self, f, u0, jac):
        """solve the system f(u) = 0 with the initial guess u0 and the Jacobian jac(u)"""
        raise NotImplementedError(
            "'AbstractNewtonSolver' is an abstract base class - do not use for actual solving!")

    @property
    def niterations(self):
        """access to the number of iterations taken in the last Newton solve"""
        return self._iteration_count

    def norm(self, residuals):
        """the norm used for checking the residuals for convergence"""
        return np.max(residuals)

    def throw_no_convergence_error(self, res=None):
        """throw an error when the solver failed to converge"""
        if res is None:
            res = ""
        else:
            res = " Max. residuals: {:.2e}".format(res)
        if self.niterations is None:
            it = ""
        else:
            it = " after {:d} iterations".format(self.niterations)
        name = type(self).__name__
        raise np.linalg.LinAlgError(
            name + " did not converge" + it + "!" + res)


class MyNewtonSolver(AbstractNewtonSolver):
    """Reference implementation of a simple 'text book' Newton solver"""

    @profile
    def solve(self, f, u0, jac):
        self._iteration_count = 0
        u = u0
        while self._iteration_count < self.max_iterations:
            # do a classical Newton step
            J = jac(u)
            if sp.issparse(J):
                du = sp.linalg.spsolve(J, f(u))
            else:
                du = np.linalg.solve(J, f(u))
            u -= du
            self._iteration_count += 1
            # calculate the norm of the residuals
            err = self.norm(f(u))
            # print some info on the step, if desired
            if self.verbosity > 1:
                print("Newton step #{:d}, max. residuals: {:.2e}".format(
                    self._iteration_count, err))
            # if system converged to new solution, return solution
            if err < self.convergence_tolerance:
                if self.verbosity > 0:
                    print("MyNewtonSolver solver converged after",
                          self._iteration_count, "iterations, error:", err)
                return u
        # if we didn't converge, throw an error
        self.throw_no_convergence_error(err)


class NewtonSolver(AbstractNewtonSolver):
    """
    A Newton solver that uses scipy.optimize.root for solving.
    The method (algorithm) to be used can be adjusted with the attribute 'method'.
    NOTE: does not work with sparse Jacobians, but converts it to a dense matrix instead! 
    """

    def __init__(self):
        super().__init__()
        # choose from the different methods of scipy.optimize.root
        # NOTE: method = "krylov" might be faster, but then we can use NewtonKrylovSolver directly
        self.method = "hybr"

    @profile
    def solve(self, f, u0, jac=None):
        # methods that do not use the Jacobian, but use an approximation
        inexact_methods = ["krylov", "broyden1", "broyden2",
                           "diagbroyden", "anderson", "linearmixing", "excitingmixing"]
        # wrapper for the Jacobian

        def jac_wrapper(u):
            # sparse matrices are not supported by scipy's root method :-/ convert to dense
            j = jac(u)
            if sp.issparse(j):
                return j.toarray()
            return j
        # check if Jacobian is required by the method
        if jac is None or self.method in inexact_methods:
            jac_wrapper = None
        # solve!
        opt_result = scipy.optimize.root(
            f, u0, jac=jac_wrapper, method=self.method, tol=self.convergence_tolerance)
        # fetch number of iterations, residuals and status
        err = self.norm(opt_result.fun)
        self._iteration_count = opt_result.nit if 'nit' in opt_result.keys() else opt_result.nfev
        # if we didn't converge, throw an error
        if not opt_result.success:
            self.throw_no_convergence_error(err)
        if self.verbosity > 0:
            print("NewtonSolver converged after",
                  self._iteration_count, "iterations, error:", err)
        # return the result vector
        return opt_result.x


class NewtonKrylovSolver(AbstractNewtonSolver):
    """
    A Newton solver using the highly efficient Krylov subspace method for Jacobian approximation.
    If provided, the known Jacobian is inverted: M = J^-1 and then M*J*a = M*b is solved instead
    of J*a = b, because M*J is likely closer to the identity, cf.:
    https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
    """

    def __init__(self):
        super().__init__()
        # choose from the different methods of scipy.optimize.root
        self.approximate_jacobian = True

    @profile
    def solve(self, f, u0, jac=None):
        # some options
        options = {
            'disp': self.verbosity > 1,  # print the results of each step?
            'maxiter': self.max_iterations,
            'fatol': self.convergence_tolerance
        }
        # the inverse of the Jacobian M = J^-1 at initial guess u
        # increases performance of the krylov method
        if jac is not None and not self.approximate_jacobian:
            # compute incomplete LU decomposition of Jacobian
            J_ilu = sp.linalg.spilu(sp.csc_matrix(jac(u0)))
            M = sp.linalg.LinearOperator(shape=jac.shape, matvec=J_ilu.solve)
            options.update({'jac_options': {'inner_M': M}})

        # solve!
        opt_result = scipy.optimize.root(f, u0,
                                         method="krylov",
                                         options=options)
        # fetch number of iterations, residuals and status
        err = self.norm(opt_result.fun)
        self._iteration_count = opt_result.nit
        # if we didn't converge, throw an error
        if not opt_result.success:
            self.throw_no_convergence_error(err)
        if self.verbosity > 0:
            print("NewtonKrylovSolver converged after",
                  self._iteration_count, "iterations, error:", err)
        # return the result vector
        return opt_result.x


class EigenSolver:
    """
    A wrapper to the powerful iterative Eigensolver ARPACK,
    that finds eigenvalues and eigenvectors of an eigenproblem.
    """

    def __init__(self):
        # The shift used for the shift-invert method in the iterative eigensolver.
        # If shift != None, the eigensolver will find the eigenvalues near the
        # value of the shift first
        self.shift = 0.0
        # store results of the latest computation
        # NOTE: this is currently only needed for plotting the problem
        self.latest_eigenvalues = None
        self.latest_eigenvectors = None

    def solve(self, A, M=None, k=None):
        if k is None:
            # if no number of values was specified, use a direct eigensolver for computing all eigenvalues
            eigenvalues, eigenvectors = scipy.linalg.eig(A, M)
        else:
            # else: compute only the largest k eigenvalues with an iterative eigensolver
            # this iterative eigensolver relies on ARPACK (Arnoldi method)
            # A: matrix of which we compute the eigenvalues
            # k: number of eigenvalues to compute in iterative method
            # M: mass matrix for generized eigenproblem A*x=w*M*x
            # which: order of eigenvalues to compute ('LM' = largest magnitude is default and the fastest)
            # sigma: Find eigenvalues near sigma using shift-invert mode.
            # v0: Starting vector for iteration. Default: random.
            #     This may not be deterministic, since it is random! For this reason, pde2path uses a [1,...,1]-vector
            # For more info, see the documentation:
            # https://docs.scipy.org/doc/scipy/reference/generated/sp.linalg.eigs.html
            eigenvalues, eigenvectors = sp.linalg.eigs(
                A, k=k, M=M, sigma=self.shift, which='LM')
        # sort by largest eigenvalue (largest real part) and filter infinite eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        idx = idx[np.isfinite(eigenvalues[idx])]
        # store and return
        self.latest_eigenvalues = eigenvalues[idx]
        self.latest_eigenvectors = eigenvectors.T[idx]
        return (self.latest_eigenvalues, self.latest_eigenvectors)
