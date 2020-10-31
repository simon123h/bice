import numpy as np
import scipy.sparse
from bice.core.equation import Equation
from bice.core.profiling import profile


class FiniteElementEquation(Equation):
    """
    The FiniteElementEquation is a subclass of the general Equation
    and provides some useful routines that are needed for implementing
    ODEs/PDEs in a Finite-Element approach.
    It defaults to using periodic boundaries.
    A FiniteElementEquation relates to a mesh: it has nodes/elements and
    can build the related matrices.
    """

    def __init__(self, shape=None):
        super().__init__(shape)
        # the mesh
        self.mesh = None
        # FEM mass matrix
        self.M = None
        # FEM stiffness matrix
        self.laplace = None
        # FEM first order derivative matrices
        self.nabla = None

    # return the coordinate vector from the nodes
    @property
    def x(self):
        return np.array([n.x for n in self.mesh.nodes]).T

    # the number of unknowns per independent variable in the equation
    @property
    def dim(self):
        return self.shape[-1]

    @dim.setter
    def dim(self, d):
        self.reshape(self.shape[:-1] + (d,))

    # the number of independent variables in the equation
    @property
    def nvariables(self):
        if len(self.shape) == 1:
            return 1
        return self.shape[0]

    # assemble the matrices of the FEM operators
    @profile
    def build_FEM_matrices(self):
        # number of nodes
        N = len(self.mesh.nodes)
        # mass matrix
        self.M = np.zeros((N, N))
        # stiffness matrix
        self.laplace = np.zeros((N, N))
        # first order derivative matrices
        self.nabla = [np.zeros((N, N)) for d in range(self.mesh.dim)]
        # store the global indices of the nodes
        for i, n in enumerate(self.mesh.nodes):
            n.index = i
        # for every element
        for element in self.mesh.elements:
            # spatial integration loop
            for s, weight in element.integration_points:
                # premultiply weight with coordinate transformation determinant
                weight *= element.transformation_det
                # evaluate the shape functions, test functions are identical to shape functions
                test = shape = element.shape(s)
                dtestdx = dshapedx = element.dshapedx(s)
                # loop over every node i, j of the element and add contributions to the integral
                for i, ni in enumerate(element.nodes):
                    for j, nj in enumerate(element.nodes):
                        # mass matrix
                        self.M[ni.index, nj.index] += shape[i] * \
                            test[j] * weight
                        # stiffness matrix
                        self.laplace[ni.index,
                                     nj.index] -= dshapedx[i].dot(dtestdx[j]) * weight
                        # first order derivative matrices
                        for d in range(self.mesh.dim):
                            self.nabla[d][ni.index, nj.index] += dshapedx[i][d] * \
                                test[j] * weight
                        # TODO: also include a matrix for the Dirichlet conditions?

        # convert matrices to CSR-format (compressed sparse row)
        # for efficiency of arithmetic operations
        self.M = scipy.sparse.csr_matrix(self.M)
        self.laplace = scipy.sparse.csr_matrix(self.laplace)
        self.nabla = [scipy.sparse.csr_matrix(n) for n in self.nabla]

    # Assemble the residuals from an external definition of the integrand in
    # res = \int dx residual_definition(x, u dudx, test, dtestdx),
    # similar to building the matrices
    # The residuals will have to be assembled using this method if they cannot
    # be computed with the FEM matrices alone.
    @profile
    def assemble_residuals(self, residuals_definition, u):
        # empty result variable
        res = np.zeros(self.shape).T
        # store the global indices of the nodes
        for i, n in enumerate(self.mesh.nodes):
            n.index = i
        # for every element
        for element in self.mesh.elements:
            # spatial integration loop
            for s, weight in element.integration_points:
                # premultiply weight with coordinate transformation determinant
                weight *= element.transformation_det
                # evaluate the shape functions, test functions are identical to shape functions
                test = shape = element.shape(s)
                dtestdx = dshapedx = element.dshapedx(s)
                # interpolate...
                #  ...spatial coordinates
                x = sum([n.x*s for n, s in zip(element.nodes, shape)])
                #  ...unknowns
                uu = sum([u[..., n.index]*s for n,
                          s in zip(element.nodes, shape)])
                # ...spatial derivative of unknowns: dudx = sum_{nodes} u * dshapedx
                dudx = np.array([sum([u[..., n.index]*ds[d] for n, ds in zip(
                    element.nodes, dshapedx)]) for d in range(self.mesh.dim)]).T
                # for every node in the element
                for i, node in enumerate(element.nodes):
                    # calculate residual contribution
                    residual_contribution = residuals_definition(
                        x, uu, dudx, test[i], dtestdx[i])
                    # cancel the contributions of pinned values (Dirichlet conditions)
                    if node.pinned_values:
                        residual_contribution[list(node.pinned_values)] = 0
                    # accumulate the weighted residual contributions to the integral
                    res[node.index] += residual_contribution * weight
        # return the result
        return res.T

    # evaluate the integral of a given function I = \int f(u) dx on the mesh
    @profile
    def integrate(self, f, u):
        # integral will be accumulated in this variable
        result = 0
        # store the global indices of the nodes
        for i, n in enumerate(self.mesh.nodes):
            n.index = i
        # for every element
        for element in self.mesh.elements:
            # spatial integration loop
            for s, weight in element.integration_points:
                # premultiply weight with coordinate transformation determinant
                weight *= element.transformation_det
                # evaluate the shape functions
                shape = element.shape(s)
                # interpolate spatial coordinates and unknowns
                x = sum([n.x*s for n, s in zip(element.nodes, shape)])
                uu = sum([u[..., n.index]*s for n,
                          s in zip(element.nodes, shape)])
                # accumulate weighted values to the integral
                result += f(x, uu) * weight
        # return the integral value
        return result

    # adapt the mesh to the variables
    @profile
    def adapt(self):
        # calculate error estimate
        error_estimate = self.refinement_error_estimate()
        # store the unknowns in the nodes
        self.copy_unknowns_to_nodal_values()
        # adapt the mesh
        self.mesh.adapt(error_estimate)
        # restore the unknowns from the (interpolated) node values
        self.copy_nodal_values_to_unknowns()
        # re-build the FEM matrices
        self.build_FEM_matrices()

    # estimate the error made in each node
    @profile
    def refinement_error_estimate(self):
        # calculate integral of curvature:
        # error = | \int d^2 u / dx^2 * test(x) dx |
        # NOTE: overwrite this method, if different weights of the curvatures are needed
        curvature = 0
        for n in range(self.nvariables):
            u = self.u if len(self.shape) == 1 else self.u[n]
            curvature += np.abs(self.laplace.dot(u))
        return curvature

    # copy the unknowns u to the values in the mesh nodes
    @profile
    def copy_unknowns_to_nodal_values(self, u=None, include_history=True):
        if u is None:
            u = self.u
        # optionally append the history of unknowns to u
        if include_history:
            u = np.array([u] + self.u_history)
        # loop over the nodes
        for n, node in enumerate(self.mesh.nodes):
            # assign the unknowns to the node
            node.u = u[..., n]

    # copy the values of the nodes to the equation's unknowns
    @profile
    def copy_nodal_values_to_unknowns(self, include_history=True):
        # number of nodes
        N = len(self.mesh.nodes)
        # if the number of nodes changed...
        if N != self.dim:
            # update the dimension of the vector of unknowns
            self.dim = N
            # if the equation belongs to a group of equations, redo it's mapping of the unknowns
            if self.group is not None:
                self.group.map_unknowns()
        # new empty array for the unknowns
        u = np.zeros(self.shape)
        # if needed, also for the history of unknowns
        if include_history:
            hist_len = len(self.u_history)
            u = np.array([u] + [u.copy() for t in range(hist_len)])
        # loop over the nodes
        for n, node in enumerate(self.mesh.nodes):
            # make sure the node has storage for the unknowns
            if node.u is None:
                raise Exception(
                    "Node #{:d} at x={} has no unknowns assigned! "
                    "Unable to copy them to equation.u!".format(n, node.x))
            # store the node's unknowns in the new u
            u[..., n] = node.u
        # optionally split the history of unknowns again
        if include_history:
            self.u_history = list(u[1:])
            u = u[0]
        # assign the new unknowns to the equation
        self.u = u
