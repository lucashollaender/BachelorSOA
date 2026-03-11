from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
from SOALIB import soalib as sb
import pandas as pd

L = 5
H_type1 = "revz"
H_type2 = "spherical"

m = 1
klOC = np.array([2.5, 0, 0])

# n_md_max = (n_nd - 1) * 3

E, G, c, rho, n_nd, n_md = 230e9, 80e9, 0.02, 7850, 6, 4

w, h = 0.1, 0.1

j1 = Joint(L, H_type1)
r = Rigid_Properties(rho, klOC, w, h)
f = Flex_Properties(E, G, c, n_nd, n_md)
b1 = SOABody(j1, r, f)

j2 = Joint(L, H_type2)
r = Rigid_Properties(rho, klOC, w, h)
f = Flex_Properties(E, G, c, n_nd, n_md)
b2 = SOABody(j2, r, f)

PIe = b1.flex.PI_end

print(np.linalg.norm(PIe[3, :]))
print(np.linalg.norm(PIe[4, :]))
print(np.linalg.norm(PIe[5, :]))

K = b1.flex.K_fl
M = b1.flex.M_fl

F_ext = np.array([0, 0, 0, -1e5, 0, 0]).reshape(6, 1)
# b1.set_F_ext(F_ext)
b1.set_initial_beta0(2)

bodies = [b1, b2, b1]

system = MultibodySystem(bodies)

tf = 1
dt = 0.001

sim = Simulation(system, tf, dt)

sim.set_camera_ver(90)
sim.set_camera_hor(90)
sim.set_camera_speed(0)
sim.set_ani_dt(0.01)

sim.IntegrateSystem()

save_dir = r"C:\Users\jepp6\OneDrive - Aarhus universitet\Dokumenter\Noter\6. Semester\Bachelor Projekt\BachelorCode\Renders"

sim.animate_nodes()

"""
print("PI")
print(pd.DataFrame(b1.flex.PI))
print("eigval")
print(pd.DataFrame(b1.flex.eigval))
print("K_fl")
print(pd.DataFrame(b1.flex.K_fl))
print("M_fl_red")
print(pd.DataFrame(b1.flex.M_fl[-6:, -6:]))
#print("M_fl")
#print(pd.DataFrame(b1.flex.M_fl))
"""
