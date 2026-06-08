from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
from SOALIB import soalib as sb
import pandas as pd
import matplotlib.pyplot as plt

"""
This TEST simulates a flexible 2m beam under the influence of 
gravity and an initial impulse force along the z-axis of 200N.
"""

# ----- Body Parameters -----
klOO1 = np.array([2, 0, 0]).reshape(3, 1)
H_type1 = "fixed"
E, G, c, rho, n_nd, n_md = 230e7, 80e7, 0.02, 7850, 11, 20
w, h = 0.04, 0.06

# ----- Setup Bodies -----
# Body 1
j1 = Joint(klOO1, H_type1)
r1 = Rigid_Properties(rho, w, h)
f1 = Flex_Properties(E, G, c, n_nd, n_md, mode_selection={
                     "bending_xy": 2,
                     "bending_xz": 2})

# ----- Setup Body -----          
b1 = SOABody(j1, r1, f1)

# ----- Loads -----
F_impulse = np.array([0, 0, 0, 0, 0, 2e2]).reshape(6, 1)
b1.set_impulse_force(0, 0.25, F_impulse, node=-1)

# ----- Setup Serial-Chain -----
bodies = [b1]
system = MultibodySystem(bodies)
tf = 10
dt = 0.01
sim = Simulation(system, tf, dt)

# ----- Setup Animation -----
sim.set_camera_ver(0)
sim.set_camera_hor(-90)
sim.set_camera_speed(0)
sim.show_COM_frames(0.2)
sim.IntegrateSystem("Radau")
sim.animate_nodes()