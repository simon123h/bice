#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("../..")
from bice import Problem, Equation, PseudospectralEquation
from bice.time_steppers import BDF2, RungeKuttaFehlberg45
from bice.constraints import TranslationConstraint, VolumeConstraint
from bice.profiling import profile, Profiler


class activePhaseFieldCrystalEquation(PseudospectralEquation):
    r"""
    Pseudospectral implementation of the 2-dimensional active PFC equation
    \partial_t \psi &= \Delta (r - (q_c^2 + \Delta)^2)\psi + (\psi + \bar\phi)^3) - v_0 \nabla P
    \partial_t P &= C_1\Delta P - D_r C_1 P - v_o \nabla \psi
    """

    def __init__(self, Nx, Ny, Lx, Ly):
        super().__init__(shape=(2, Nx*Ny))
        # parameters
        self.r = -1.5
        self.phi0 = -0.8
        self.qc = 1.
        self.v0 = 0.1694
        self.C1 = 0.1
        self.Dr = 0.5
        self.x = [np.linspace(-Lx/2, Lx/2, Nx), np.linspace(-Ly/2, Ly/2, Ny)]
        self.build_kvectors()
        self.u = np.load


