import matplotlib
matplotlib.use("GTK3Agg")

import numpy as np
import scipy.sparse as scs
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.optimize as sco


class FEMelement:
    def __init__(self, nx, L):
        self.L = L
        self.nx = nx
        self.x = np.linspace(-L, L, nx, endpoint=False)
        self.dx = self.dxs()
        self.setmats()

    def dxs(self):
        xex = np.append(self.x, -self.x[0])
        return xex[1:]-xex[:-1]

    def setmats(self):
        N = self.nx
        dx = self.dxs()

        self.K = np.zeros((N, N))
        np.fill_diagonal(self.K, -1./np.roll(dx, 1))
        self.K = np.roll(self.K, -2, axis=1)
        np.fill_diagonal(self.K, -1./dx)
        self.K = np.roll(self.K, 1, axis=1)
        np.fill_diagonal(self.K, 1./dx+1./np.roll(dx, 1))
        self.K = -self.K

        self.D = 0.5*np.roll(np.eye(N), -1, axis=1)-0.5 * \
            np.roll(np.eye(N), 1, axis=1)

        self.M = np.eye(N)
        np.fill_diagonal(self.M, np.roll(dx, 1)/6.)
        self.M = np.roll(self.M, -2, axis=1)
        np.fill_diagonal(self.M, dx/6.)
        self.M = np.roll(self.M, 1, axis=1)
        np.fill_diagonal(self.M, (dx+np.roll(dx, 1))/3.)

    def qlmat(self, c):
        N = self.nx
        dx = self.dxs()

        K = np.zeros((N, N))
        np.fill_diagonal(K, -0.5*(np.roll(c, 1)+c)/np.roll(dx, 1))
        K = np.roll(K, -2, axis=1)
        np.fill_diagonal(K, -0.5*(c+np.roll(c, -1))/dx)
        K = np.roll(K, 1, axis=1)
        np.fill_diagonal(K, 0.5*(np.roll(c, -1)+c)/dx+0.5 *
                         (np.roll(c, 1)+c)/np.roll(dx, 1))
        return K

    def rhs2(self, u):
        return np.matmul(self.D, np.sin(np.pi/self.L*self.x)) + np.matmul(self.M, u)

    def rhs(self, u):
        c = self.x**2.
        c = np.ones(len(self.x))
        c = np.sin(np.pi/self.L*self.x)
        return np.matmul(self.qlmat(c), c) + np.matmul(self.M, u)


test = FEMelement(200, 750)
print(test.K)
dx = test.dxs()
test.x -= 10*dx[0]*np.sin(2.*np.pi/test.L*test.x)
inis = np.zeros(200)
plt.plot(test.x, inis, '.')
plt.show()
test.setmats()

sol = sco.newton_krylov(test.rhs, inis)
# plt.plot(test.x,np.sin(np.pi/test.L*test.x)/np.amax(np.sin(np.pi/test.L*test.x)))
# plt.plot(test.x,sol/np.amax(sol),'+')
# plt.plot(test.x,np.pi/test.L*np.cos(np.pi/test.L*test.x))
# print(np.amax(np.abs(np.amax(sol)-np.amax((np.pi/test.L)*np.cos(np.pi/test.L*test.x)))))
# print(np.amax(np.abs(np.amax(sol)-np.amax(-(np.pi/test.L)**2.*np.sin(np.pi/test.L*test.x)))))
# plt.plot(test.x,(np.pi/test.L)**2.*np.sin(np.pi/test.L*test.x))

plt.plot(test.x, (np.pi/test.L)**2.*np.cos(np.pi/test.L*test.x)
         ** 2.-(np.pi/test.L)**2.*np.sin(np.pi/test.L*test.x)**2.)
plt.plot(test.x, sol, '.')
plt.show()
