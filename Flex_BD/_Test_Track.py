from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
from SOALIB import soalib as sb
import pandas as pd
import matplotlib.pyplot as plt

klOO1 = np.array([1, 0, 0]).reshape(3, 1)
klOO2 = np.array([1, 0, 0]).reshape(3, 1)
klOO3 = np.array([1, 0, 0]).reshape(3, 1)
H_type1 = "revy"
H_type2 = "revy"
H_type3 = "revy"

# n_md_max = (n_nd - 1) * 3
E, G, c, rho, n_nd, n_md = 1e9, 5e6, 0.49, 1.4e3, 11, 20

w, h = 0.3, 0.08

j1 = Joint(klOO1, H_type1)
r1 = Rigid_Properties(rho, w, h)
f1 = Flex_Properties(E, G, c, n_nd, n_md, mode_selection={
                     "bending_xy": 2,
                     "bending_xz": 2,
                     "axial_x": 1})

j2 = Joint(klOO2, H_type2)
r2 = Rigid_Properties(rho, w, h)
f2 = Flex_Properties(E, G, c, n_nd, n_md, mode_selection={
                     "bending_xy": 2,
                     "bending_xz": 2,
                     "axial_x": 1})

j3 = Joint(klOO3, H_type3)
r3 = Rigid_Properties(rho, w, h)
f3 = Flex_Properties(E, G, c, n_nd, n_md, mode_selection={
                     "bending_xy": 2,
                     "bending_xz": 2,
                     "axial_x": 1})

F_axial = 1e4
f1.set_constant_axial_load(F_axial)
f2.set_constant_axial_load(F_axial)
f3.set_constant_axial_load(F_axial)

b1 = SOABody(j1, r1, f1)
b2 = SOABody(j2, r2, f2)
b3 = SOABody(j3, r3, f3)


# b1.set_initial_beta0(1)
# b1.set_initial_eta0(np.array([0, 0, 0, 0, 0, 0, 0]).reshape(7, 1))

PIe = b1.flex.PI_end

print(np.linalg.norm(PIe[3, :]))
print(np.linalg.norm(PIe[4, :]))
print(np.linalg.norm(PIe[5, :]))

K = b1.flex.K_fl
M = b1.flex.M_fl

k_TS, c_TS = 1e2, 1e2

b1.set_TS(k_TS, c_TS, 0)
b2.set_TS(k_TS, c_TS, 0)
b3.set_TS(k_TS, c_TS, 0)

#F_ext1 = np.array([0, 0, 0, F_axial, 0, 0]).reshape(6, 1)
#b1.set_F_ext(node=-1, F_ext=F_ext1)

F_ext2 = np.array([0, 0, 0, 0, 0, 6e3]).reshape(6, 1)
b2.set_impulse_force(0.5, 0.25, F_ext2)

k_z = 2e6
c_z = 1e4

b1.set_global_axial_force(F_axial)
b1.set_z_spring(k_z, c_z, node=-1)

bodies = [b1, b2, b3]

system = MultibodySystem(bodies)
system.set_gravity(False)

tf = 2
dt = 0.01

sim = Simulation(system, tf, dt)

sim.set_camera_ver(0)
sim.set_camera_hor(-90)
sim.set_camera_speed(0)
sim.set_ani_dt(0.01)
sim.show_COM_frames(0.2)
sim.set_max_step(0.01)

sim.IntegrateSystem("Radau")

save_dir = r"C:\Users\jepp6\OneDrive - Aarhus universitet\Dokumenter\Noter\6. Semester\Bachelor Projekt\BachelorCode\Renders"

# sim.animate_nodes(filename="Test1", save_dir=save_dir)
sim.animate_nodes()