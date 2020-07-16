#!/usr/bin/python3
from src.demo_problems import SwiftHohenberg
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

she = SwiftHohenberg(512, 64*np.pi)
she.time_stepper.dt = 1e-3

fig, ax = plt.subplots()
for n in range(100):
    ax.plot(she.x, she.u)
    plt.show()
    fig.savefig("out/img/{:05d}.svg".format(n))
    ax.clear()
    she.time_step()
    print("Step #{:05d}".format(n))

