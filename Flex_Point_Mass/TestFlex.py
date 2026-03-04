from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
from SOALIB import soalib as sb
import pandas as pd

L = 5

klOO = L * np.array([1, 0, 0])
H_type1 = "fixed"

m = 1
CkJk = np.array([1, 1, 0.1])
klOC = np.array([0, 0, 2.5])

# n_md_max = (n_nd - 1) * 3

E, G, rho, n_nd, n_md = 230e9, 80e9, 7850, 20, 20

w, h = 0.1, 0.1

j1 = Joint(klOO, H_type1)
r = Rigid_Properties(rho, klOC, w, h)
f = Flex_Properties(E, G, n_nd, n_md)
b1 = SOABody(j1, r, f)

PIe = b1.flex.PI_end

print(np.linalg.norm(PIe[3, :]))
print(np.linalg.norm(PIe[4, :]))
print(np.linalg.norm(PIe[5, :]))

K = b1.flex.K_fl
M = b1.flex.M_fl

F_ext = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
b1.set_F_ext(F_ext)

bodies = [b1]

system = MultibodySystem(bodies)

tf = 20
dt = 0.01

sim = Simulation(system, tf, dt)

# sim.camera_speed(0.5)
sim.camera_ver(0)
sim.camera_hor(0)

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
print("M_fl")
print(pd.DataFrame(b1.flex.M_fl))
"""
