#!/usr/bin/python3
from src.demo_problems import SwiftHohenberg
from src.time_stepper import RungeKuttaFehlberg45
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

she = SwiftHohenberg(512, 240)
she.time_stepper = RungeKuttaFehlberg45()
she.time_stepper.dt = 1e-3
she.time_stepper.adaptive_timesteps = False
she.time_stepper.error_dt = 1e1

fig, ax = plt.subplots()
plotevery = 500
n = 0
while True:
    if n % plotevery == 0:
        ax.plot(she.x, she.u)
        # u_k = np.fft.rfft(she.u)
        # ax.plot(she.k, np.abs(u_k))
        fig.savefig("out/img/{:05d}.svg".format(n//plotevery))
        ax.clear()
        print("Step #{:05d}".format(n//plotevery))
        print("dt:   {:}".format(she.time_stepper.dt))
    n += 1
    she.time_step()
    if np.max(she.u) > 1e12:
        break

