# -*- coding: utf-8 -*-
# Pseudospektralverfahren fÃ¼r Cahn-Hilliard in 2D
# d/dt c = - kappa laplace^2 c - \laplace c + laplace (cÂ³)

import numpy as np
import matplotlib.pyplot as plt


Lx = 128  # Physikalische LÃ¤nge des Gebietes in x-Richtung
Ly = 128  # Physikalische LÃ¤nge des Gebietes in y-Richtung
Nx = 128  # Anzahl Diskretisierungspunkte in x-Richtung
Ny = 128  # Anzahl Diskretisierungspunkte in y-Richtung

x, y = np.meshgrid(np.arange(Nx) * Lx/Nx, np.arange(Ny) * Ly/Ny)  # x-Array
kx, ky = np.meshgrid(np.fft.fftfreq(Nx, Lx/(Nx*2.0*np.pi)),
                     np.fft.fftfreq(Ny, Ly/(Ny*2.0*np.pi)))
ksquare = kx*kx + ky*ky

#c = -0.2*np.exp(-((x-Lx/2.0)*(x-Lx/2.0)+(y-Ly/2.0)*(y-Ly/2.0))/50)
c = (np.random.random((Nx, Ny))-0.5)*0.02
kappa = 1.2

t = 0.0
h = 0.005
T_End = 1000
N_t = int(T_End / h)

plotEveryNth = 200


def rhs(c):
    ck = np.fft.fft2(c)  # Fouriertransformation
    c3k = np.fft.fft2(c*c*c)
    result_k = (-kappa*ksquare*ksquare + ksquare)*ck - ksquare*c3k
    result = np.fft.ifft2(result_k).real
    return result


# plt.ion()

plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.pcolormesh(x, y, c, vmin=-1, vmax=1)
plt.colorbar()

for i in range(N_t):
    k1 = rhs(c)
    k2 = rhs(c+h/2.0*k1)
    k3 = rhs(c+h/2.0*k2)
    k4 = rhs(c+h*k3)
    c = c+h/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)

    if(i % plotEveryNth == 0):
        print("Step %04d" % (i/plotEveryNth))
        plt.cla()
        plt.pcolormesh(x, y, c, vmin=-1, vmax=1)
        plt.colorbar()
        # plt.show()
        # plt.pause(0.1)
        filename = "Cahn-Hilliard-2D%04d.png" % (i/plotEveryNth)
        plt.savefig(filename)
        plt.close()
