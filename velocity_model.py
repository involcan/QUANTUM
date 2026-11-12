"""
Velocity model on the normalized domain [0, 1] x [0, 1].

shortest_path.py expects two functions:
    velocity(x, y) -> v           velocity (> 0)
    gradient(x, y) -> (dvx, dvy)  velocity gradient (a scalar |grad v| is also accepted)

Both must accept scalars and NumPy arrays (element-wise).
"""

import numpy as np

V0 = 2.0          # background velocity
DV = -0.0          # amplitude of the Gaussian anomaly
XC, YC = 0.5, 0.5 # anomaly centre
SIGMA = 0.4       # anomaly width


def velocity(x, y):
    r2 = (x - XC) ** 2 + (y - YC) ** 2
    return V0 + DV * np.exp(-r2 / SIGMA ** 2)


def gradient(x, y):
    r2 = (x - XC) ** 2 + (y - YC) ** 2
    g = DV * np.exp(-r2 / SIGMA ** 2) * (-2.0 / SIGMA ** 2)
    return g * (x - XC), g * (y - YC)
