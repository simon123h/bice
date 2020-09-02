import numpy as np
from .equation import Equation

# isoparametric element formulation with numerical integration


# A Finite-Element-Equation relates to a mesh: it has nodes/elements and can build the related matrices
class MyFiniteElementEquation(Equation):

    def __init__(self):
        self.nodes = []
        self.elements = []
        self.x = None

    # setup a mesh of size L with N points. N and L can be lists for multi-dimensional meshes
    def setup_mesh(self, N, L):
        # get dimension of the mesh
        dim = len(N)
        # check if discretization and size have the same
        if len(N) != len(L):
            raise AttributeError(
                "Mismatch in dimension! Mesh discretization must have same dimension as Mesh size.")
        # generate a 1d mesh
        if dim == 1:
            # generate x
            self.x = [np.linspace(0, L, N)]
            # add equidistant nodes
            for i in range(N):
                self.nodes.append(Node(self.x[i]))
            # generate the elements
            for i in range(N-1):
                nodes = [self.nodes[i], self.nodes[i+1]]
                self.elements.append(Element1d(nodes))
        # generate a 2d mesh
        if dim == 2:
            # generate x
            Nx, Ny = N
            Lx, Ly = L
            self.x = [np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny)]
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
        # dimension
        N = len(self.nodes)
        # mass matrix
        M = np.zeros((N, N))
        # stiffness matrix
        K = np.zeros((N, N))
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
                        M[ni.index, nj.index] += shape[i] * test[j] * weight
                        for d in range(self.dim):
                            K[ni.index, nj.index] += dshape[i][d] * \
                                dtest[j][d] * weight


class Node:

    def __init__(self, x):
        self.x = x


class Element:

    def __init__(self, nodes):
        self.dim = nodes[0].x.size
        self.nodes = nodes

    # returns a list of all shape functions of this element
    def shape(self, x):
        pass

    # returns a list of all shape function derivatives of this element
    def dshape(self, x):
        pass

    # returns a list of all test functions of this element
    def test(self, x):
        return self.shape(x)

    # returns a list of all test function derivatives of this element
    def dtest(self, x):
        return self.dshape(x)


class Element1d(Element):

    def __init__(self, nodes):
        super().__init__(nodes)
        self.x0 = nodes[0].x[0]
        self.x1 = nodes[1].x[0]
        self.dx = self.x1 - self.x0
        # exact polynomial integration using Gaussian quadrature
        # see: https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Gau%C3%9F-Legendre-Integration
        a = np.array([(self.x0 + self.x1) / 2])
        b = np.array([np.sqrt(1./3.) * self.dx / 2])
        self.integration_points = [(a + b, 1), (a - b, 1)]

    # 1d linear shape functions
    def shape(self, x):
        s = (x[0] - self.x0) / self.dx  # local coordinate
        return [1-s, s]

    # 1d linear shape functions within the element
    def dshape(self, x):
        ds_dx = 1 / self.dx  # local coordinate derivative
        return [[-ds_dx, ds_dx]]


class Element2d(Element):

    def __init__(self, nodes):
        super().__init__(nodes)
        self.x0 = nodes[0].x[0]
        self.x1 = nodes[1].x[0]
        self.y0 = nodes[0].x[0]
        self.y1 = nodes[1].x[0]
        self.dx = self.x1 - self.x0
        self.dy = self.y1 - self.x0
        # exact polynomial integration using Gaussian quadrature
        # see: https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Gau%C3%9F-Legendre-Integration
        ax = (self.x0 + self.x1) / 2
        bx = np.sqrt(1./3.) * self.dx / 2
        ay = (self.y0 + self.y1) / 2
        by = np.sqrt(1./3.) * self.dy / 2
        self.integration_points = [
            (np.array([ax-bx, ay-by]), 1),
            (np.array([ax+bx, ay-by]), 1),
            (np.array([ax-bx, ay+by]), 1),
            (np.array([ax+bx, ay+by]), 1)
        ]

    # 2d linear shape functions
    def shape(self, x):
        # local coordinates
        sx = (x[0] - self.x0) / self.dx
        sy = (x[1] - self.y0) / self.dy
        # again, note the counter-clockwise order of the nodes
        return [(1-sx)*(1-sy), (1-sx)*sy, sx*sy, sx*(1-sy)]

    # 2d linear shape functions within the element
    def dshape(self, x):
        # local coordinates
        sx = (x[0] - self.x0) / self.dx
        sy = (x[1] - self.y0) / self.dy
        # local coordinate derivatives
        dsx_dx = 1 / self.dx
        dsy_dy = 1 / self.dy
        # derivatives
        # again, note the counter-clockwise order of the nodes
        return [
            [-dsx_dx*(1-sy), -dsx_dx*sy, dsx_dx*sy, dsx_dx*(1-sy)],
            [(1-sx)*dsy_dy, (1-sx)*dsy_dy, sx*dsy_dy, -sx*dsy_dy]
        ]
