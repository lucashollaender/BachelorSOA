from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
from SOALIB import soalib as sb
import pandas as pd

klOO1 = np.array([1, 0, 0]).reshape(3, 1)
klOO2 = np.array([1, 0, 0]).reshape(3, 1)
<<<<<<< HEAD
H_type1 = "spherical"
H_type2 = "fixed"
=======
H_type1 = "revy"
H_type2 = "revy"
>>>>>>> efde51e3fd21c6adfdae9d017c451779b8187fa2

# n_md_max = (n_nd - 1) * 3
E, G, c, rho, n_nd, n_md = 230e12, 80e12, 0.0, 7850, 2, 1

w, h = 0.04, 0.04

j1 = Joint(klOO1, H_type1)
r1 = Rigid_Properties(rho, w, h)
f1 = Flex_Properties(E, G, c, n_nd, n_md)

j2 = Joint(klOO2, H_type2)
r2 = Rigid_Properties(rho, w, h)
f2 = Flex_Properties(E, G, c, n_nd, n_md)

b1 = SOABody(j1, r1, f1)
b2 = SOABody(j2, r2, f2)

#b1.set_initial_beta0(1)
#b1.set_initial_eta0(np.array([0, 0, 0, 0, 0, 0, 0]).reshape(7, 1))

PIe = b1.flex.PI_end

print(np.linalg.norm(PIe[3, :]))
print(np.linalg.norm(PIe[4, :]))
print(np.linalg.norm(PIe[5, :]))

K = b1.flex.K_fl
M = b1.flex.M_fl

<<<<<<< HEAD
F_ext1 = np.array([0, 0, 0, 0, 1e5, 0]).reshape(6, 1)
# b1.set_F_ext(F_ext1)
F_ext2 = np.array([1e4, 0, 0, 0, 0, 0]).reshape(6, 1)
b2.set_F_ext(F_ext2)
# b1.set_initial_beta0(2)


# eta0 = np.vstack([np.array([5]), np.zeros((n_md-1, 1))]).reshape(6, 1)
# eta0 = np.array([0, 0, 0, 0, 10, 0]).reshape(6, 1)
# b1.set_initial_eta0(eta0)
=======
#F_ext1 = np.array([0, 0, 0, 0, 1e5, 0]).reshape(6, 1)
#b1.set_F_ext(F_ext1)
#F_ext2 = np.array([1e4, 0, 0, 0, 0, 0]).reshape(6, 1)
#b2.set_F_ext(F_ext2)
#b1.set_initial_beta0(2)

#eta0 = np.vstack([np.array([5]), np.zeros((n_md-1, 1))]).reshape(6, 1)
#eta0 = np.array([0, 0, 0, 0, 10, 0]).reshape(6, 1)
#b1.set_initial_eta0(eta0)
>>>>>>> efde51e3fd21c6adfdae9d017c451779b8187fa2

bodies = [b1, b2, b2]

system = MultibodySystem(bodies)

tf = 5
dt = 0.01

sim = Simulation(system, tf, dt)

sim.set_camera_ver(0)
sim.set_camera_hor(90)
sim.set_camera_speed(0)
sim.set_ani_dt(0.01)

sim.IntegrateSystem("Radau")

save_dir = r"C:\Users\jepp6\OneDrive - Aarhus universitet\Dokumenter\Noter\6. Semester\Bachelor Projekt\BachelorCode\Renders"

# sim.animate_nodes(filename="FlexOORotMissing", save_dir=save_dir)
sim.animate_nodes()

"""
print("eigval")
print(pd.DataFrame(b1.flex.eigval))
print("K_fl")
print(pd.DataFrame(b1.flex.K_fl))
print("M_fl_red")
print(pd.DataFrame(b1.flex.M_fl[-6:, -6:]))
#print("M_fl")
#print(pd.DataFrame(b1.flex.M_fl))
"""
# Problems:
# Rotation due to deformation at tip node
# If revz and z load, then force seem to be applied rotated. Works fine for "fixed" joint
