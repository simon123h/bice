import math
import numpy as np
from .equation import Equation
from scipy.sparse import lil_matrix

# isoparametric element formulation with numerical integration


class FiniteElementEquation(Equation):
    """
    The FiniteElementEquation is a subclass of the general Equation
    and provides some useful routines that are needed for implementing
    ODEs/PDEs in a Finite-Element approach.
    It defaults to using periodic boundaries.
    A FiniteElementEquation relates to a mesh: it has nodes/elements and can build the related matrices
    """

    def __init__(self):
        super().__init__()
        # the mesh
        self.mesh = None
        # number of values per node in the mesh
        self.nvalue = 1
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
    def build_FEM_matrices(self):
        # number of nodes
        N = len(self.mesh.nodes)
        # spatial dimension
        dim = len(self.x)
        # mass matrix
        self.M = lil_matrix((N, N))
        # stiffness matrix
        self.laplace = lil_matrix((N, N))
        # first order derivative matrices
        self.nabla = []
        for d in range(dim):
            self.nabla.append(lil_matrix((N, N)))
        # store the global indices of the nodes
        for i, n in enumerate(self.mesh.nodes):
            n.index = i
        # for every element
        for element in self.mesh.elements:
            # spatial integration loop
            for x, weight in element.integration_points:
                # premultiply weight with coordinate transformation determinant
                det = element.transformation_det
                weight *= det
                # evaluate the shape functions
                shape = element.shape(x)
                dshape = element.dshape(x)
                # evaluate the test functions
                test = element.test(x)
                dtest = element.dtest(x)
                # loop over every node i, j of the element
                for i, ni in enumerate(element.nodes):
                    for j, nj in enumerate(element.nodes):
                        # integral contributions
                        self.M[ni.index, nj.index] += shape[i] * \
                            test[j] * weight
                        for d in range(dim):
                            self.laplace[ni.index, nj.index] -= dshape[d][i] * \
                                dtest[d][j] * weight / det / det
                            self.nabla[d][ni.index, nj.index] += dshape[d][i] * \
                                test[j] * weight / det

        # convert matrices to CSR-format (compressed sparse row)
        # for efficiency of arithmetic operations
        self.M = self.M.tocsr()
        self.laplace = self.laplace.tocsr()
        self.nabla = [n.tocsr() for n in self.nabla]

    # adapt the mesh to the variables
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
    def refinement_error_estimate(self):
        # calculate curvature
        # TODO: use weighted curvature of all value indices
        curvature = self.laplace.dot(self.nodal_values(0))
        # store curvature in nodes
        for node, c in zip(self.mesh.nodes, curvature):
            # get maximal distance to neighbour node
            max_dx = max([e.max_len for e in node.elements])
            # the error per node is curvature * max_dx
            node.error = abs(c * max_dx)
        # return the error per node
        return np.array([node.error for node in self.mesh.nodes])

    # copy the unknowns u to the values in the mesh nodes
    def copy_unknowns_to_nodal_values(self, u=None):
        if u is None:
            u = self.u
        # loop over the nodes
        i = 0
        for node in self.mesh.nodes:
            # make sure the node has storage for the unknowns
            if node.u is None:
                node.u = np.zeros(self.nvalue)
            # loop over the number of values per node
            for n in range(self.nvalue):
                # exclude pinned values
                if n not in node.pinned_values:
                    # write the value to the node and increment counter
                    node.u[n] = u[i]
                    i += 1

    # copy the values of the nodes to the equation's unknowns
    def copy_nodal_values_to_unknowns(self):
        # calculate the number of unknown values / degrees of freedom
        N = len(self.mesh.nodes) * self.nvalue - \
            sum([len(n.pinned_values) for n in self.mesh.nodes])
        # if the number of unknowns changed...
        if self.u is None or N != self.u.size:
            # store reference to the equation's problem
            problem = self.problem
            # remove the equation from the problem (if assigned)
            if problem is not None:
                problem.remove_equation(self)
            # create a new array of correct size, values will be filled later
            self.u = np.zeros(N)
            # re-add the equation to the problem
            if problem is not None:
                problem.add_equation(self)
        # now, we'll fill self.u with the values from the nodes
        # loop over the nodes
        i = 0
        for node in self.mesh.nodes:
            # make sure the node has storage for the unknowns
            if node.u is None:
                node.u = np.zeros(self.nvalue)
            # loop over the number of values per node
            for n in range(self.nvalue):
                # exclude pinned values
                if n not in node.pinned_values:
                    # write the value to the equation and increment counter
                    self.u[i] = node.u[n]
                    i += 1

    # return the vector of nodal values with specific index
    def nodal_values(self, index):
        return np.array([node.u[index] for node in self.mesh.nodes])


class Node:
    """
    simple node class for nodes in meshes,
    basically stores a position in space
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

    # keep the value at a specific index pinned (remove it from dof)
    def pin(self, index):
        self.pinned_values.add(index)


class Element:
    """
    base class for all finite elements
    """

    def __init__(self, nodes):
        # spatial dimension
        self.dim = nodes[0].x.size
        # the nodes that build the element
        self.nodes = nodes
        # add self to element list in nodes
        for node in self.nodes:
            node.elements.append(self)
        # maximum distance between two nodes within the element
        self.max_len = 0
        for node1 in self.nodes:
            for node2 in self.nodes:
                dx = np.linalg.norm(node1.x - node2.x)
                self.max_len = max(self.max_len, dx)

    # called when the element is removed from the mesh
    def purge(self):
        # remove the element from element list in nodes
        for node in self.nodes:
            node.elements.remove(self)

    # returns a list of all shape functions of this element
    def shape(self, s):
        pass

    # returns a list of all shape function derivatives of this element
    def dshape(self, s):
        pass

    # returns a list of all test functions of this element
    def test(self, x):
        return self.shape(x)

    # returns a list of all test function derivatives of this element
    def dtest(self, x):
        return self.dshape(x)


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
        return [1-s, s]

    # 1d linear shape functions within the element
    def dshape(self, s):
        return [[-1, 1]]


class TriangleElement2d(Element):
    """
    Two-dimensional triangular elements with linear shape functions
    """

    def __init__(self, nodes):
        super().__init__(nodes)
        # sort so that the longest edge comes first
        while np.linalg.norm(nodes[1].x-nodes[0].x) < np.linalg.norm(nodes[2].x-nodes[1].x) or np.linalg.norm(nodes[1].x-nodes[0].x) < np.linalg.norm(nodes[0].x-nodes[2].x):
            nodes.append(nodes.pop(0))
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
        return [1-sx-sy, sx, sy]

    # 2d linear shape functions within the element
    def dshape(self, s):
        # (dshape_dsx, dshape_dsy)
        return [
            [-1, 1, 0],
            [-1, 0, 1]
        ]

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


class RectangleElement2d(Element):
    """
    Two-dimensional rectangular elements with linear shape functions
    """

    def __init__(self, nodes):
        super().__init__(nodes)
        self.x0 = self.nodes[0].x[0]
        self.y0 = self.nodes[0].x[1]
        self.x1 = self.nodes[2].x[0]
        self.y1 = self.nodes[2].x[1]
        # transformation matrix from local to global coordinates
        self.transformation_matrix = np.array([
            [self.x1-self.x0, 0],
            [0, self.y1-self.y0]
        ])
        # corresponding determinant of transformation
        self.transformation_det = (self.x1-self.x0)*(self.y1-self.y0)
        # exact polynomial integration using Gaussian quadrature
        # see: https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Gau%C3%9F-Legendre-Integration
        # and: https://link.springer.com/content/pdf/bbm%3A978-3-540-32609-0%2F1.pdf
        a = 1. / 2.
        b = np.sqrt(1./3.) / 2
        w = 1. / 2.
        self.integration_points = [
            (np.array([a-b, a-b]), w),
            (np.array([a+b, a-b]), w),
            (np.array([a-b, a+b]), w),
            (np.array([a+b, a+b]), w)
        ]

    # 2d linear rectangle shape functions
    def shape(self, s):
        sx, sy = s
        # four shape functions, counter-clockwise order of nodes
        return [(1-sx)*(1-sy), (1-sx)*sy, sx*sy, sx*(1-sy)]

    # 2d linear shape functions within the element
    def dshape(self, s):
        sx, sy = s
        return [
            [sy-1, -sy, sy, 1-sy],
            [sx-1, 1-sx, sx, -sx]
        ]


class Mesh:
    """
    Base class for meshes
    """

    def __init__(self):
        # storage for the nodes
        self.nodes = []
        # storage for the elements
        self.elements = []
        # storage for the values of each node: 2d array of shape (#nodes, nvalue)
        self.values = None
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
        # generate x
        x = np.linspace(L0, L0+L, N, endpoint=True)
        # add equidistant nodes
        for i in range(N):
            node = Node(x=np.array([x[i]]))
            self.nodes.append(node)
            # mark boundary nodes
            if i == 0 or i == N-1:
                node.is_boundary_node = True
        # generate the elements
        for i in range(N-1):
            nodes = [self.nodes[i], self.nodes[i+1]]
            self.elements.append(Element1d(nodes))

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
        # generate x,y-space
        x = np.linspace(Lx0, Lx0+Lx, Nx, endpoint=True)
        y = np.linspace(Ly0, Lx0+Ly, Ny, endpoint=True)
        # add equidistant nodes
        for i in range(Nx):
            for j in range(Ny):
                node = Node(x=np.array([x[i], y[j]]))
                self.nodes.append(node)
                # mark boundary nodes
                if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                    node.is_boundary_node = True

        # generate the elements
        for i in range(Nx-1):
            for j in range(Ny-1):
                # just like setting up a rectangular mesh, but we
                # divide each rectangle into two triangular elements
                # using counter-clockwise order of the nodes
                nodes = [
                    self.nodes[i*Nx+j],
                    self.nodes[(i+1)*Nx+(j+1)],
                    self.nodes[(i+1)*Nx+j]
                ]
                self.elements.append(TriangleElement2d(nodes))
                nodes = [
                    self.nodes[i*Nx+j],
                    self.nodes[i*Nx+(j+1)],
                    self.nodes[(i+1)*Nx+(j+1)]
                ]
                self.elements.append(TriangleElement2d(nodes))

    def adapt(self, error_estimate):
        # check the errors for each node and store whether they should be (un)refined
        for node, error in zip(self.nodes, error_estimate):
            node.can_be_unrefined = abs(error) < self.min_refinement_error
            node.should_be_refined = abs(error) > self.max_refinement_error

        # unrefinement loop, using node fusion
        i = 0
        while i < len(self.elements):
            # store reference to the nodes
            node_a = self.elements[i].nodes[0]
            node_b = self.elements[i].nodes[1]
            node_c = self.elements[i].nodes[2]
            # unrefine if all of the nodes were marked for unrefinement
            if node_a.can_be_unrefined and node_b.can_be_unrefined and node_c.can_be_unrefined:
                # find the nodes with smallest distance
                my_nodes = [node_a, node_b, node_c, node_a]
                dist = [np.linalg.norm(my_nodes[i+1].x-my_nodes[i].x)
                        for i in range(3)]
                index = dist.index(min(dist))
                node_b = my_nodes[index]
                node_c = my_nodes[index+1]
                # for now, we assume it is node_b and node_c
                # check if they're both boundary nodes
                if node_b.is_boundary_node or node_c.is_boundary_node:
                    # TODO: we could deal with this case, though...
                    i += 1
                    continue
                # store reference to the current element
                elem1 = self.elements[i]
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
                # for each element that surrounds the pricipal nodes
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
                # check if element has minimal size already
                if self.elements[i].max_len <= 2 * self.min_element_dx:
                    break
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
                # if both node_a and node_b were at a border, node_m is also at a border
                if node_a.is_boundary_node and node_b.is_boundary_node:
                    node_m.is_boundary_node = True
                else:
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
                            index, TriangleElement2d([neighbor_node, node_a, node_m]))
                        self.elements.insert(
                            index+1, TriangleElement2d([node_b, neighbor_node, node_m]))
                # skip refinement of the newly created elements
                i += 2
            i += 1


class RectangleMesh(Mesh):
    """
    Two-dimensional rectangular mesh with
    rectangular elements and linear shape functions
    """

    def __init__(self, Nx, Ny, Lx, Ly, Lx0=0, Ly0=0):
        # call parent constructor
        super().__init__()
        # generate x,y-space
        x = np.linspace(Lx0, Lx0+Lx, Nx, endpoint=True)
        y = np.linspace(Ly0, Lx0+Ly, Ny, endpoint=True)
        # add equidistant nodes
        for i in range(Nx):
            for j in range(Ny):
                pos = np.array([x[i], y[j]])
                self.nodes.append(Node(pos))
        # generate the elements
        for i in range(Nx-1):
            for j in range(Ny-1):
                # using counter-clockwise order of the nodes
                nodes = [
                    self.nodes[i*Nx+j],
                    self.nodes[i*Nx+(j+1)],
                    self.nodes[(i+1)*Nx+(j+1)],
                    self.nodes[(i+1)*Nx+j]
                ]
                self.elements.append(RectangleElement2d(nodes))
