import math
import numpy as np
from .equation import Equation
from .profiling import profile
import scipy.sparse

# isoparametric element formulation with numerical integration


class FiniteElementEquation(Equation):
    """
    The FiniteElementEquation is a subclass of the general Equation
    and provides some useful routines that are needed for implementing
    ODEs/PDEs in a Finite-Element approach.
    It defaults to using periodic boundaries.
    A FiniteElementEquation relates to a mesh: it has nodes/elements and can build the related matrices
    """

    def __init__(self, shape=(1,)):
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
        # transpose so node index comes first
        uT = u.T
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
                u = sum([uT[n.index]*s for n, s in zip(element.nodes, shape)])
                # ...spatial derivative of unknowns: dudx = sum_{nodes} u * dshapedx
                dudx = np.array([sum(
                    [uT[n.index]*ds[d] for n, ds in zip(element.nodes, dshapedx)]) for d in range(self.mesh.dim)]).T
                # for every node in the element
                for i, node in enumerate(element.nodes):
                    # calculate residual contribution
                    residual_contribution = residuals_definition(
                        x, u, dudx, test[i], dtestdx[i])
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
        # transpose so node index comes first
        uT = u.T
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
                u = sum([uT[n.index]*s for n, s in zip(element.nodes, shape)])
                # accumulate weighted values to the integral
                result += f(x, u) * weight
        # return the integral value
        return result

    # adapt the mesh to the variables
    @profile
    def adapt(self):
        # store the unknowns in the nodes
        self.copy_unknowns_to_nodal_values()
        # calculate error estimate
        error_estimate = self.refinement_error_estimate()
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
        # TODO: use weighted curvature of all variables, not just the sum
        curvature = 0
        for n in range(self.nvariables):
            u = self.u if len(self.shape) == 1 else self.u[n]
            curvature += np.abs(self.laplace.dot(u))
        return curvature

    # copy the unknowns u to the values in the mesh nodes
    @profile
    def copy_unknowns_to_nodal_values(self, u=None):
        if u is None:
            u = self.u
        # transpose for correct shape
        u = u.T
        # loop over the nodes
        for n, node in enumerate(self.mesh.nodes):
            # assign the unknowns to the node, transpose for correct shape
            node.u = u[n]

    # copy the values of the nodes to the equation's unknowns
    @profile
    def copy_nodal_values_to_unknowns(self):
        # number of nodes
        N = len(self.mesh.nodes)
        # if the number of nodes changed...
        if self.u is None or N != self.dim:
            # store reference to the equation's problem
            problem = self.problem
            # remove the equation from the problem (if assigned)
            if problem is not None:
                problem.remove_equation(self)
            # create a new array of correct size, values will be filled later
            self.dim = N
            self.u = np.zeros(self.shape)
            # re-add the equation to the problem
            if problem is not None:
                problem.add_equation(self)
        # now, we'll fill self.u with the values from the nodes
        # new empty array for the unknowns
        u = np.zeros(self.shape).T
        # loop over the nodes
        for n, node in enumerate(self.mesh.nodes):
            # make sure the node has storage for the unknowns
            if node.u is None:
                raise Exception(
                    "Node #{:d} at x={} has no unknowns assigned! Unable to copy them to equation.u!".format(n, node.x))
            # store the node's unknowns in the new u
            u[n] = node.u
        # assign the new unknowns to the equation
        self.u = np.array(u).T


class Node:
    """
    Simple node class for nodes in meshes,
    stores a position in space, the unknowns, the related elements and more.
    """

    def __init__(self, x):
        # the spatial coordinates
        self.x = x
        # storage for the unknowns (not always up to date!)
        self.u = None
        # the elements that this node belongs to
        self.elements = []
        # indices of the values that are pinned (e.g. by a Dirichlet condition)
        self.pinned_values = set()
        # does the node lie on a boundary?
        self.is_boundary_node = False
        # Flags for marking the node for (un)refinement
        self.can_be_unrefined = False
        self.should_be_refined = False

    # keep the value at a specific index pinned by a Dirichlet condition
    def pin(self, index):
        self.pinned_values.add(index)


class Element:
    """
    Base class for all finite elements
    """

    def __init__(self, nodes):
        # spatial dimension
        self.dim = nodes[0].x.size
        # the nodes that build the element
        self.nodes = nodes
        # add self to element list in nodes
        for node in self.nodes:
            node.elements.append(self)
        # transformation matrix from local coordinates x to local coordinates s: x = T * s
        # will be overwritten for specific element geometries
        self.transformation_matrix = np.eye(self.dim)
        # inverse of the transformation matrix: s = T^-1 * x
        # will be overwritten for specific element geometries
        self.transformation_matrix_inv = np.eye(self.dim)
        # determinant of the transformation matrix:
        # will be overwritten for specific element geometries
        self.transformation_det = 1

    # called when the element is removed from the mesh
    def purge(self):
        # remove the element from element list in nodes
        for node in self.nodes:
            node.elements.remove(self)

    # returns a list of all shape functions of this element
    def shape(self, s):
        pass

    # returns a list of all shape function derivatives with
    # respect to the local coordinate s of this element
    def dshape(self, s):
        pass

    # returns a list of all shape function derivatives with
    # respect to the global coordinate x of this element
    def dshapedx(self, s):
        return self.dshape(s).dot(self.transformation_matrix_inv)


class Element1d(Element):
    """
    One-dimensional elements with linear shape functions
    """

    def __init__(self, nodes):
        super().__init__(nodes)
        self.x0 = nodes[0].x[0]
        self.x1 = nodes[1].x[0]
        # transformation matrix from local to global coordinates
        self.transformation_matrix = np.array([
            [self.x1-self.x0]
        ])
        # inverse of transformation_matrix
        self.transformation_matrix_inv = 1 / self.transformation_matrix
        # corresponding determinant of transformation
        self.transformation_det = self.x1-self.x0
        # exact polynomial integration using Gaussian quadrature
        # see: https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Gau%C3%9F-Legendre-Integration
        # and: https://link.springer.com/content/pdf/bbm%3A978-3-540-32609-0%2F1.pdf
        a = 1. / 2.
        b = np.sqrt(1./3.) / 2
        w = 1. / 2.
        self.integration_points = [
            (np.array([a-b]), w),
            (np.array([a+b]), w)
        ]

    # 1d linear triangle shape functions
    def shape(self, s):
        s = s[0]
        return np.array([1-s, s])

    # 1d linear shape functions within the element
    def dshape(self, s):
        return np.array([[-1], [1]])

    # returns a list of all shape function derivatives with
    # respect to the global coordinate x of this element
    def dshapedx(self, s):
        return self.dshape(s) / self.transformation_det


class TriangleElement2d(Element):
    """
    Two-dimensional triangular elements with linear shape functions
    """

    @profile
    def __init__(self, nodes):
        # calculate edge lengths of the triangle
        edge_lengths = [np.linalg.norm(
            nodes[(i+1) % 3].x-nodes[i].x) for i in range(3)]
        # sort nodes so that the longest edge comes first
        max_index = edge_lengths.index(max(edge_lengths))
        nodes = nodes[max_index:] + nodes[:max_index]
        self.edge_lengths = edge_lengths[max_index:] + edge_lengths[:max_index]
        # call parent constructor
        super().__init__(nodes)
        # abbreviations for the coordinates
        self.x0 = nodes[0].x[0]
        self.y0 = nodes[0].x[1]
        self.x1 = nodes[1].x[0]
        self.y1 = nodes[1].x[1]
        self.x2 = nodes[2].x[0]
        self.y2 = nodes[2].x[1]
        # transformation matrix from local to global coordinates
        self.transformation_matrix = np.array([
            [self.x1-self.x0, self.x2-self.x0],
            [self.y1-self.y0, self.y2-self.y0]
        ])
        # corresponding determinant of transformation
        self.transformation_det = (
            self.x1-self.x0)*(self.y2-self.y0)-(self.x2-self.x0)*(self.y1-self.y0)
        # calculate invert of transformation matrix
        self.transformation_matrix_inv = np.array([[self.transformation_matrix[1, 1], -self.transformation_matrix[0, 1]],
                                                   [-self.transformation_matrix[1, 0], self.transformation_matrix[0, 0]]]) / self.transformation_det

        # exact polynomial integration using Gaussian quadrature
        # see: https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Gau%C3%9F-Legendre-Integration
        # and: https://link.springer.com/content/pdf/bbm%3A978-3-540-32609-0%2F1.pdf
        w = 1. / 6.
        self.integration_points = [
            (np.array([0.5, 0.5]), w),
            (np.array([0.0, 0.5]), w),
            (np.array([0.5, 0.0]), w)
        ]

    # 2d linear triangle shape functions
    def shape(self, s):
        sx, sy = s
        # three shape functions, counter-clockwise order of nodes
        return np.array([1-sx-sy, sx, sy])

    # 2d linear shape functions within the element
    def dshape(self, s):
        # (dshape_dsx, dshape_dsy)
        return np.array([
            [-1, -1],
            [1, 0],
            [0, 1]
        ])

    # returns a list of all shape function derivatives with
    # respect to the global coordinate x of this element
    # TODO: are these also correct or should we go with the default implementation?
    def dshapedx(self, s):
        return np.array([
            [self.y1-self.y2, self.x2-self.x1],
            [self.y2-self.y0, self.x0-self.x2],
            [self.y0-self.y1, self.x1-self.x0]
        ]) / self.transformation_det

    # if the orientation is positive/negative, the triangle is oriented anticlockwise/clockwise
    def orientation(self):
        return np.linalg.det(np.array([
            [self.x0, self.y0, 1],
            [self.x1, self.y1, 1],
            [self.x2, self.y2, 1]
        ]))

    # calculate the area of the triangle
    def area(self):
        return abs(0.5*(((self.x1-self.x0)*(self.y2-self.y0))-(
            (self.x2-self.x0)*(self.y1-self.y0))))

    # calculate the corner angles of the triangle
    def angles(self):
        a1 = self.nodes[1].x - self.nodes[0].x
        a2 = self.nodes[2].x - self.nodes[0].x
        b1 = self.nodes[0].x - self.nodes[1].x
        b2 = self.nodes[2].x - self.nodes[1].x
        ang = [
            abs(math.acos(np.dot(a1, a2) /
                          np.linalg.norm(a1)/np.linalg.norm(a2))),
            abs(math.acos(np.dot(b1, b2) /
                          np.linalg.norm(b1)/np.linalg.norm(b2)))
        ]
        ang.append(180 - ang[0] - ang[1])
        return ang


class Mesh:
    """
    Base class for meshes
    """

    def __init__(self):
        # spatial dimension of the mesh
        self.dim = None
        # storage for the nodes
        self.nodes = []
        # storage for the elements
        self.elements = []
        # error thresholds for refinement
        self.min_refinement_error = 1e-5
        self.max_refinement_error = 1e-3
        # minimal edge length of an element, for mesh adaption
        self.min_element_dx = 1e-9
        # maximal edge length of an element, for mesh adaption
        self.max_element_dx = 1e9

    # adapt mesh to the values given by the unknowns u
    def adapt(self, error_estimate):
        pass


class OneDimMesh(Mesh):
    """
    One-dimensional mesh from elements with linear shape functions
    """

    def __init__(self, N, L, L0=0):
        # call parent constructor
        super().__init__()
        # spatial dimension
        self.dim = 1
        # generate x
        x = np.linspace(L0, L0+L, N, endpoint=True)
        # add equidistant nodes
        for i in range(N):
            node = Node(x=np.array([x[i]]))
            self.nodes.append(node)
            # mark boundary nodes
            if i in [0, N-1]:
                node.is_boundary_node = True
        # generate the elements
        for i in range(N-1):
            nodes = [self.nodes[i], self.nodes[i+1]]
            self.elements.append(Element1d(nodes))

    @profile
    def adapt(self, error_estimate):
        # check the errors for each node and store whether they should be (un)refined
        for node, error in zip(self.nodes, error_estimate):
            node.can_be_unrefined = abs(error) < self.min_refinement_error
            node.should_be_refined = abs(error) > self.max_refinement_error

        # unrefinement loop
        i = 0
        while i < len(self.elements)-1:
            # store reference to nodes
            node_l = self.elements[i].nodes[0]
            node_m = self.elements[i].nodes[1]
            node_r = self.elements[i+1].nodes[1]
            # unrefine if all three nodes call for unrefinement
            if node_l.can_be_unrefined and node_m.can_be_unrefined and node_r.can_be_unrefined:
                # check if element has maximum size already
                if node_r.x[0] - node_l.x[0] >= 0.5 * self.max_element_dx:
                    break
                # delete the old elements
                self.elements.pop(i).purge()
                self.elements.pop(i).purge()
                # delete middle node
                self.nodes.remove(node_m)
                # create new element
                self.elements.insert(i, Element1d([node_l, node_r]))
                # this element should not be unrefined any further (for now)
                node_l.should_be_refined = False
            i += 1

        # refinement loop
        i = 0
        while i < len(self.elements):
            # store reference to the nodes
            node_l = self.elements[i].nodes[0]
            node_r = self.elements[i].nodes[1]
            # refine if any of the nodes was marked for refinement
            if node_l.should_be_refined or node_r.should_be_refined:
                # check if element has minimal size already
                if node_r.x[0] - node_l.x[0] <= 2 * self.min_element_dx:
                    break
                # generate new node in the middle and insert after node_l
                node_m = Node((node_l.x+node_r.x)/2)
                n = self.nodes.index(node_l)
                self.nodes.insert(n+1, node_m)
                # interpolate the unknowns
                node_m.u = (node_l.u + node_r.u)/2
                # delete old element
                self.elements.pop(i).purge()
                # generate two new elements and insert at the position of the old element
                self.elements.insert(i, Element1d([node_l, node_m]))
                self.elements.insert(i+1, Element1d([node_m, node_r]))
                # skip refinement of the newly created element
                i += 1
            i += 1


class TriangleMesh(Mesh):
    """
    Two-dimensional rectangular mesh with
    triangular elements and linear shape functions
    """

    def __init__(self, Nx, Ny, Lx, Ly, Lx0=0, Ly0=0):
        # call parent constructor
        super().__init__()
        # spatial dimension
        self.dim = 2
        # generate x,y-space
        x = np.linspace(Lx0, Lx0+Lx, Nx, endpoint=True)
        y = np.linspace(Ly0, Ly0+Ly, Ny, endpoint=True)
        # add equidistant nodes
        for i in range(Nx):
            for j in range(Ny):
                node = Node(x=np.array([x[i], y[j]]))
                self.nodes.append(node)
                # mark boundary nodes
                if i in [0, Nx-1] or j in [0, Ny-1]:
                    node.is_boundary_node = True

        # generate the elements
        for i in range(Nx-1):
            for j in range(Ny-1):
                # just like setting up a rectangular mesh, but we
                # divide each rectangle into two triangular elements
                # using counter-clockwise order of the nodes
                nodes = [
                    self.nodes[i*Nx+j],
                    self.nodes[(i+1)*Nx+j],
                    self.nodes[i*Nx+(j+1)],
                    self.nodes[(i+1)*Nx+(j+1)],
                    self.nodes[i*Nx+j]
                ]
                offs = (i+j) % 2
                nodes_a = [nodes[i] for i in (0, 1, 2+offs)]
                nodes_b = [nodes[i] for i in (1-offs, 3, 2)]
                self.elements.append(TriangleElement2d(nodes_a))
                self.elements.append(TriangleElement2d(nodes_b))

    @profile
    def adapt(self, error_estimate):
        # check the errors for each node and store whether they should be (un)refined
        for node, error in zip(self.nodes, error_estimate):
            node.can_be_unrefined = abs(error) < self.min_refinement_error
            node.should_be_refined = abs(error) > self.max_refinement_error

        # unrefinement loop, using node fusion
        i = 0
        while i < len(self.elements):
            # store reference to the current element
            elem1 = self.elements[i]
            # store reference to the nodes
            node_a = elem1.nodes[0]
            node_b = elem1.nodes[1]
            node_c = elem1.nodes[2]
            # unrefine if all of the nodes were marked for unrefinement
            if node_a.can_be_unrefined and node_b.can_be_unrefined and node_c.can_be_unrefined:
                # find the nodes with smallest distance
                min_index = elem1.edge_lengths.index(min(elem1.edge_lengths))
                node_b = elem1.nodes[min_index]
                node_c = elem1.nodes[(min_index+1) % 3]
                # check if they're both boundary nodes
                if node_b.is_boundary_node or node_c.is_boundary_node:
                    # TODO: we could deal with this case, though...
                    i += 1
                    continue
                # find the element that shares the edge: node_b--node_c
                elem2 = [
                    e for e in node_b.elements if node_c in e.nodes and e is not elem1][0]
                # generate a new node in the middle
                x_m = (node_b.x + node_c.x) / 2
                node_m = Node(x_m)
                node_m.can_be_unrefined = False
                node_m.should_be_refined = False
                # interpolate the unknowns
                node_m.u = (node_b.u + node_c.u) / 2
                # we will now join the nodes node_b and node_c
                # all the elements with node_b and node_c will be recreated using node_m
                # in order to abort in case of bad mesh quality, we'll save the replacements
                # to a list first and will (maybe) apply them later
                replacement_elements = []
                aborted = False  # flag whether the unrefinement is aborted
                # for each element that surrounds the principal nodes
                for node in [node_b, node_c]:
                    for element in node.elements:
                        if element in [elem1, elem2]:
                            continue
                        # calculate orientation of the old element
                        orientation_old = element.orientation()
                        # create the new nodes and element
                        new_nodes = [
                            n if n != node else node_m for n in element.nodes]
                        new_element = TriangleElement2d(new_nodes)
                        # store the replacement in the replacements-list
                        replacement_elements.append((element, new_element))
                        # calculate orientation of the old element
                        orientation_new = new_element.orientation()
                        # check whether the orientation of the triangle flipped
                        if orientation_old * orientation_new < 0:
                            # if yes, abort the current unrefinement
                            aborted = True
                        # check whether the angles would become very small
                        if min(new_element.angles()) < 0.2:
                            aborted = True
                    # break if unrefinement is aborted
                    if aborted:
                        break
                # apply the unrefinement, unless aborted
                if not aborted:
                    # destroy the collapsed elements
                    self.elements.pop(i).purge()  # elem1
                    i2 = self.elements.index(elem2)
                    self.elements.pop(i2).purge()  # elem2
                    # overwrite the surrounding elements
                    for old_e, new_e in replacement_elements:
                        index = self.elements.index(old_e)
                        old_e.purge()
                        self.elements[index] = new_e
                    # insert the new middle node
                    n = self.nodes.index(node_b)
                    self.nodes.insert(n, node_m)
                    # delete the nodes node_b and node_c
                    self.nodes.remove(node_b)
                    self.nodes.remove(node_c)
                else:
                    # purge the new elements
                    for old_e, new_e in replacement_elements:
                        new_e.purge()
            i += 1

        # refinement loop
        i = 0
        while i < len(self.elements):
            # store reference to the nodes
            node_a = self.elements[i].nodes[0]
            node_b = self.elements[i].nodes[1]
            node_c = self.elements[i].nodes[2]
            # refine if all of the nodes were marked for refinement
            if node_a.should_be_refined and node_b.should_be_refined and node_c.should_be_refined:
                # if both node_a and node_b were at a border, node_m is also at a border
                if node_a.is_boundary_node and node_b.is_boundary_node:
                    # if we don't unrefine boundary nodes, we should not refine them either
                    i += 1
                    continue
                # check if element has minimal size already
                if np.linalg.norm(node_a.x - node_b.x) <= 2 * self.min_element_dx:
                    i += 1
                    continue
                # generate new node in the middle of the first two nodes (longest edge)
                x_m = (node_a.x + node_b.x) / 2
                node_m = Node(x_m)
                node_m.should_be_refined = False
                # and insert after node_a
                n = self.nodes.index(node_a)
                self.nodes.insert(n+1, node_m)
                # interpolate the unknowns
                node_m.u = (node_a.u + node_b.u) / 2
                # delete old element
                self.elements.pop(i).purge()
                # generate two new elements and insert at the position of the old element
                self.elements.insert(
                    i, TriangleElement2d([node_c, node_a, node_m]))
                self.elements.insert(
                    i+1, TriangleElement2d([node_b, node_c, node_m]))
                # else, find the element, that shares the edge: node_a--node_b
                neighbor_element = [
                    e for e in node_a.elements if e in node_b.elements]
                # if no neighboring element was found, we must be at a border
                if neighbor_element:
                    neighbor_element = neighbor_element[0]
                    neighbor_node = [
                        n for n in neighbor_element.nodes if n not in [node_a, node_b]][0]
                    index = self.elements.index(neighbor_element)
                    # delete old element
                    self.elements.pop(index).purge()
                    # generate two new elements and insert at the position of the old element
                    self.elements.insert(
                        index, TriangleElement2d([neighbor_node, node_m, node_a]))
                    self.elements.insert(
                        index+1, TriangleElement2d([node_b, node_m, neighbor_node]))
                # skip refinement of the newly created elements
                i += 2
            i += 1
