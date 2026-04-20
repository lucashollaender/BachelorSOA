from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Axial load test
# -------------------------------------------------
L = 0.60

# Material
E = 1e7
G = 3.8e6
c = 0.02
rho = 1000

# Use enough modes if you want axial deformation represented
n_nd = 8
n_md = 8

# Geometry
w = 0.1
h = 0.1

# -------------------------------------------------
# Fixed beam
# -------------------------------------------------
j1 = Joint(L, "fixed")
r1 = Rigid_Properties(rho, w, h)
f1 = Flex_Properties(E, G, c, n_nd, n_md)
b1 = SOABody(j1, r1, f1)

# IMPORTANT:
# fixed joint => theta, beta, tau must have size 0
b1.set_initial_theta0(np.zeros((0, 1)))
b1.set_initial_beta0(np.zeros((0, 1)))
b1.set_tau(np.zeros((0, 1)))

# External axial force in +x
# F_ext = [Mx, My, Mz, Fx, Fy, Fz]^T
F_ax = 1e4
F_ext1 = np.array([0.0, 0.0, 0.0, F_ax, 0.0, 0.0]).reshape(6, 1)
b1.set_F_ext(F_ext1)

# -------------------------------------------------
# Build system
# -------------------------------------------------
bodies = [b1]
system = MultibodySystem(bodies)

tf = 2.0
dt = 0.001

sim = Simulation(system, tf, dt)

sim.set_camera_ver(90)
sim.set_camera_hor(45)
sim.set_camera_speed(0)
sim.set_ani_dt(0.02)

sim.IntegrateSystem("BDF")

# -------------------------------------------------
# Post-processing
# -------------------------------------------------
nodal_pos = sim.nNodalPos()
time = np.array(sim.data.time)

tip_xyz = np.array([frame[-1][-1].flatten() for frame in nodal_pos])

x_tip = tip_xyz[:, 0]
y_tip = tip_xyz[:, 1]
z_tip = tip_xyz[:, 2]

x_extension = x_tip - x_tip[0]

plt.figure(figsize=(9, 5))
plt.plot(time, x_extension, label="axial extension")
plt.plot(time, y_tip - y_tip[0], label="y drift")
plt.plot(time, z_tip - z_tip[0], label="z drift")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.title("Fixed beam under axial x-load")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(tip_xyz[:, 0], tip_xyz[:, 1], tip_xyz[:, 2], lw=2)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("Tip trajectory")
plt.tight_layout()
plt.show()

print("Initial tip position [m]:")
print(tip_xyz[0])

print("Final tip position [m]:")
print(tip_xyz[-1])

print("Final axial extension [m]:")
print(x_extension[-1])

sim.animate_nodes()
