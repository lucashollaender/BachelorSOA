from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
from SOALIB import soalib as sb
import pandas as pd

L = 1
H_type1 = "revy"
H_type2 = "revy"
H_type3 = "fixed"

klOC = np.array([L/2, 0, 0])

E, G, c, rho, n_nd, n_md = 230e9, 80e9, 0.02, 7850, 6, 3
w, h = 0.1, 0.1

j1 = Joint(L, H_type1)
r1 = Rigid_Properties(rho, klOC, w, h)
f1 = Flex_Properties(E, G, c, n_nd, n_md)
b1 = SOABody(j1, r1, f1)

j2 = Joint(L, H_type2)
r2 = Rigid_Properties(rho, klOC, w, h)
f2 = Flex_Properties(E, G, c, n_nd, n_md)
b2 = SOABody(j2, r2, f2)

j3 = Joint(L, H_type3)
r3 = Rigid_Properties(rho, klOC, w, h)
f3 = Flex_Properties(E, G, c, n_nd, n_md)
b3 = SOABody(j3, r3, f3)

PIe = b1.flex.PI_end

print(np.linalg.norm(PIe[3, :]))
print(np.linalg.norm(PIe[4, :]))
print(np.linalg.norm(PIe[5, :]))

F_ext = np.array([0, 0, 0, 0, -1e5, 0]).reshape(6, 1)
b1.set_F_ext(F_ext)

b1.set_initial_beta0(np.array([[0.1]]))
b2.set_initial_beta0(np.array([[0.0]]))

bodies = [b1, b2, b3]
system = MultibodySystem(bodies)

tf = 1
dt = 0.01

sim = Simulation(system, tf, dt)
sim.setting.solver = "BDF"

sim.camera_ver(90)
sim.camera_hor(90)
sim.camera_speed(0)

sim.IntegrateSystem()
sim.animate_nodes()

print("M_fl_red")
print(pd.DataFrame(b1.flex.M_fl[-6:, -6:]))
