import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.linalg


class MyNewtonSolver:
    def __init__(self):
        self.max_newton_iterations = 30
        self.convergence_tolerance = 1e-8
        self.iteration_count = 0
        self.verbosity = 0

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
            err = np.linalg.norm(du)
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
    # TODO: catch errors, get number of iterations...
    def __init__(self):
        self.method = "hybr"

    def solve(self, f, u, J=None):
        if J is None or self.method == "krylov":
            # if Jacobian is not given, use krylov approximation
            return scipy.optimize.newton_krylov(f, u)
        # TODO: sparse matrices are not supported by scipy's root method :-/

        def jac(u):
            j = J(u)
            if scipy.sparse.issparse(j):
                j = j.toarray()
            return j
        return scipy.optimize.root(f, u, jac=jac, method=self.method).x


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
