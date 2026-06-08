from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
from SOALIB import soalib as sb
import pandas as pd
import matplotlib.pyplot as plt

"""
This TEST simulates a flexible 2-body pendulum with spherical joints 
under the influence of gravity.
"""

# ----- Body Parameters -----
klOO1 = np.array([1, 0, 0]).reshape(3, 1)
klOO2 = np.array([0, 1, 0]).reshape(3, 1)
H_type1 = "spherical"
H_type2 = "spherical"
E, G, c, rho, n_nd, n_md = 230e7, 80e7, 0.02, 7850, 11, 20
w, h = 0.04, 0.06

# ----- Setup Bodies -----
# Body 1
j1 = Joint(klOO1, H_type1)
r1 = Rigid_Properties(rho, w, h)
f1 = Flex_Properties(E, G, c, n_nd, n_md, mode_selection={
                     "bending_xy": 2,
                     "bending_xz": 2})

# Body 2
j2 = Joint(klOO2, H_type2)
r2 = Rigid_Properties(rho, w, h)
f2 = Flex_Properties(E, G, c, n_nd, n_md, mode_selection={
                     "bending_xy": 2,
                     "bending_xz": 2})

# ----- Setup Bodies -----          
b1 = SOABody(j1, r1, f1)
b2 = SOABody(j2, r2, f2)

# ----- Setup Serial-Chain -----
bodies = [b1, b2]
system = MultibodySystem(bodies)
tf = 10
dt = 0.01
sim = Simulation(system, tf, dt)

# ----- Setup Animation -----
sim.set_camera_ver(0)
sim.set_camera_hor(90)
sim.set_camera_speed(0.5)
sim.show_COM_frames(0.2)
sim.IntegrateSystem("Radau")
sim.animate_nodes()