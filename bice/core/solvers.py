import numpy as np
import scipy.sparse as sp
import scipy.optimize
import scipy.linalg
from .profiling import profile


class MyNewtonSolver:
    def __init__(self):
        self.max_newton_iterations = 30
        self.convergence_tolerance = 1e-8
        self.iteration_count = 0
        self.verbosity = 0

    @profile
    def solve(self, f, u, J):
        self.iteration_count = 0
        while self.iteration_count < self.max_newton_iterations:
            jac = J(u)
            if scipy.sparse.issparse(jac):
                du = scipy.sparse.linalg.spsolve(jac, f(u))
            else:
                du = np.linalg.solve(jac, f(u))
            u -= du
            self.iteration_count += 1
            # if system converged to new solution, return solution
            err = np.linalg.norm(du)  # TODO: use maximum norm
            if self.verbosity > 1:
                print("Newton step:", self.iteration_count, ", error:", err)
            if err < self.convergence_tolerance:
                if self.verbosity > 0:
                    print("Newton solver converged after",
                          self.iteration_count, "iterations, error:", err)
                return u
        # if we didn't converge, throw an error
        raise np.linalg.LinAlgError(
            "Newton solver did not converge after {:d} iterations!".format(self.iteration_count))


class NewtonSolver:
    def __init__(self):
        self.method = "hybr"  # TODO: how does method "hybr" perform?
        self.verbosity = 0
        self.iteration_count = None

    @profile
    def solve(self, f, u, J=None):
        # methods that do not use the Jacobian, but use an approximation
        inexact_methods = ["krylov", "broyden2", "anderson"]
        # wrapper for the Jacobian

        def jac(u):
            # sparse matrices are not supported by scipy's root method :-/ convert to dense
            j = J(u)
            if scipy.sparse.issparse(j):
                return j.toarray()
            return j
        # check if Jacobian is required by the method
        if J is None or self.method in inexact_methods:
            jac = None
        # solve!
        opt_result = scipy.optimize.root(f, u, jac=jac, method=self.method)
        # fetch number of iterations
        try:
            self.iteration_count = opt_result.nit
        except AttributeError:
            self.iteration_count = opt_result.nfev
        # if we didn't converge, throw an error
        if not opt_result.success and False:  # TODO: should check for success
            raise np.linalg.LinAlgError(
                "Newton solver did not converge after {:d} iterations!".format(self.iteration_count))
        # return the result vector
        return opt_result.x


class NewtonKrylovSolver:
    def __init__(self):
        self.verbosity = 0
        self.iteration_count = 0

    @profile
    def solve(self, f, u, J=None):
        # some options
        options = {
            'disp': self.verbosity > 0,  # print the results of each step?
            # 'maxiter': ...
            # 'fatol': ...  # absolute converge tolerance
        }
        # the inverse of the Jacobian M = J^-1 at initial guess u
        # increases performance of the krylov method
        if J is not None:
            M = sp.linalg.inv(sp.csc_matrix(J(u)))
            options.update({'jac_options': {'inner_M': M}})

        # solve!
        opt_result = scipy.optimize.root(f, u,
                                         method="krylov",
                                         options=options)
        # fetch number of iterations
        self.iteration_count = opt_result.nit
        # if we didn't converge, throw an error
        if not opt_result.success:
            raise np.linalg.LinAlgError(
                "Newton solver did not converge after {:d} iterations!".format(self.iteration_count))
        # return the result vector
        return opt_result.x


class EigenSolver:

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
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(
                A, k=k, M=M, sigma=self.shift, which='LM')
        # sort by largest eigenvalue (largest real part) and filter infinite eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        idx = idx[np.isfinite(eigenvalues[idx])]
        # store and return
        self.latest_eigenvalues = eigenvalues[idx]
        self.latest_eigenvectors = eigenvectors.T[idx]
        return (self.latest_eigenvalues, self.latest_eigenvectors)
