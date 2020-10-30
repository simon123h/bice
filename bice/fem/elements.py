import math
import numpy as np
from bice.core.profiling import profile


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
        # storage for the history of the unknowns
        self.u_history = []
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
