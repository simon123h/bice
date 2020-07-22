import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.linalg


class NewtonSolver:
    # TODO: find an optimal solver
    # TODO: catch errors, get number of iterations...
    def solve(self, f, u):
        return scipy.optimize.newton(f, u)


class EigenSolver:

    def solve(self, A, M=None, k=None):
        # TODO: find an optimal solver
        if k is None:
            # if no number of values was specified, use a direct eigensolver for computing all eigenvalues
            eigenvalues, eigenvectors = scipy.linalg.eig(A, M)
        else:
            # else: compute only the largest k eigenvalues with an iterative eigensolver
            # this iterative eigensolver relies on ARPACK (Arnoldi method)
            # which = 'LR' --> largest real part first
            # TODO: optimize arguments for iterative eigensolver
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(A, k, M=M, which='LM')
        # sort by largest eigenvalue (largest real part) and filter infinite eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        idx = idx[np.isfinite(eigenvalues[idx])]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors.T[idx]
        return (eigenvalues, eigenvectors)
