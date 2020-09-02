import numpy as np
from .equation import Equation

# isoparametric element formulation with numerical integration


class MyFiniteElementEquation(Equation):
    """
    The FiniteElementEquation is a subclass of the general Equation
    and provides some useful routines that are needed for implementing
    ODEs/PDEs in a Finite-Element approach.
    It defaults to using periodic boundaries.
    A FiniteElementEquation relates to a mesh: it has nodes/elements and can build the related matrices
    """

    def __init__(self):
        super().__init__()
        # space
        self.x = None
        # nodes
        self.nodes = []
        # elements
        self.elements = []
        # FEM mass matrix
        self.M = None
        # FEM stiffness matrix
        self.laplace = None
        # FEM first order derivative matrices
        self.nabla = None

    # setup a mesh of size L with N points. N and L can be lists for multi-dimensional meshes

    def setup_mesh(self, N, L):
        # get dimension of the mesh
        try:
            dim = len(N)
        except TypeError as te:
            dim = 1
        # generate a 1d mesh
        if dim == 1:
            # generate x
            self.x = [np.linspace(0, L, N, endpoint=True)]
            # add equidistant nodes
            for i in range(N):
                x = np.array([self.x[0][i]])
                self.nodes.append(Node(x))
            # generate the elements
            for i in range(N-1):
                nodes = [self.nodes[i], self.nodes[i+1]]
                self.elements.append(Element1d(nodes))
        # generate a 2d mesh
        elif dim == 2:
            # generate x
            Nx, Ny = N
            Lx, Ly = L
            self.x = [np.linspace(0, Lx, Nx, endpoint=True),
                      np.linspace(0, Ly, Ny, endpoint=True)]
            # add equidistant nodes
            for i in range(Nx):
                for j in range(Ny):
                    x = np.array([self.x[0][i], self.x[1][j]])
                    self.nodes.append(Node(x))
            # generate the elements
            for i in range(Nx-1):
                for j in range(Ny-1):
                    # note the counter-clockwise order of the nodes
                    nodes = [
                        self.nodes[i*Nx+j],
                        self.nodes[i*Nx+(j+1)],
                        self.nodes[(i+1)*Nx+(j+1)],
                        self.nodes[(i+1)*Nx+j]
                    ]
                    self.elements.append(Element2d(nodes))
        else:
            raise AttributeError(
                "We have no routines for n-dim. meshes with n>2, yet!")

    # assemble the matrices of the FEM operators
    def build_FEM_matrices(self):
        # number of nodes
        N = len(self.nodes)
        # spatial dimension
        dim = len(self.x)
        # mass matrix
        self.M = np.zeros((N, N))
        # stiffness matrix
        self.laplace = np.zeros((N, N))
        # first order derivative matrices
        self.nabla = []
        for d in range(dim):
            self.nabla.append(np.zeros((N, N)))
        # store the global indices of the nodes
        for i, n in enumerate(self.nodes):
            n.index = i
        # for every element
        for element in self.elements:
            # spatial integration loop
            for x, weight in element.integration_points:
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
                                dtest[d][j] * weight
                            self.nabla[d][ni.index, nj.index] += dshape[d][i] * \
                                test[j] * weight


class Node:

    def __init__(self, x):
        self.x = x


class Element:

    def __init__(self, nodes):
        self.dim = nodes[0].x.size
        self.nodes = nodes

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

# one-dimensional elements with linear shape functions


class Element1d(Element):

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
        a = 1./2.
        b = np.sqrt(1./3.) / 2
        w = self.transformation_det / 2
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


# two-dimensional triangular elements with linear shape functions
class TElement2d(Element):

    def __init__(self, nodes):
        super().__init__(nodes)
        self.n0 = nodes[0]
        self.n1 = nodes[1]
        self.n2 = nodes[2]
        self.x0 = self.n0.x[0]
        self.y0 = self.n0.x[1]
        self.x1 = self.n1.x[0]
        self.y1 = self.n1.x[1]
        self.x2 = self.n2.x[0]
        self.y2 = self.n2.x[1]
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
        w = self.transformation_det / 6
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

# two-dimensional rectangular elements with linear shape functions
class QElement2d(Element):

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
        a = 1./2.
        b = np.sqrt(1./3.) / 2
        w = self.transformation_det / 2
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
