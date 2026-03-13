from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
import matplotlib.pyplot as plt

# Pendulum test
L = 0.60

# Settings
E = 230e9
G = 80e9
c = 0.30
rho = 7850

# nodes/modes
n_nd = 4
n_md = 1

# Geometry
w = 0.025
h = 0.025

# -------------------------------------------------
# Create bodies
# -------------------------------------------------

# Moving tip body: revolute about y
j1 = Joint(L, "revy")
r1 = Rigid_Properties(rho, w, h)
f1 = Flex_Properties(E, G, c, n_nd, n_md)
b1 = SOABody(j1, r1, f1)

# Moving tip body: revolute about y
j2 = Joint(L, "revy")
r2 = Rigid_Properties(rho, w, h)
f2 = Flex_Properties(E, G, c, n_nd, n_md)
b2 = SOABody(j2, r2, f2)

# Fixed base body
j3 = Joint(L, "fixed")
r3 = Rigid_Properties(rho, w, h)
f3 = Flex_Properties(E, G, c, n_nd, n_md)
b3 = SOABody(j3, r3, f3)

# IC

b1.set_initial_theta0(np.array([[0.0]]))
b1.set_initial_beta0(np.array([[0.0]]))

# No joint torque input
b1.set_tau(np.array([[0.0]]))

# Moment about y
F_ext1 = np.array([0.0, 10, 0.0, 0.0, 0.0, 0.0]).reshape(6, 1)
F_ext2 = np.array([0.0, 50, 0.0, 0.0, 0.0, 0.0]).reshape(6, 1)
b1.set_F_ext(F_ext1)
b2.set_F_ext(F_ext2)

# -------------------------------------------------
# Build system
# -------------------------------------------------

bodies = [b1, b2, b3]
system = MultibodySystem(bodies)

tf = 10
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

# frame[-1][-1] = last node of outermost tip body
tip_xyz = np.array([frame[-1][-1].flatten() for frame in nodal_pos])

plt.figure(figsize=(9, 5))
plt.plot(time, tip_xyz[:, 0], label="x_tip")
plt.plot(time, tip_xyz[:, 1], label="y_tip")
plt.plot(time, tip_xyz[:, 2], label="z_tip")
plt.xlabel("Time [s]")
plt.ylabel("Tip position [m]")
plt.title("Two-body revy motion from external moment")
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

print("Final tip position [m]:")
print(tip_xyz[-1])

sim.animate_nodes()
