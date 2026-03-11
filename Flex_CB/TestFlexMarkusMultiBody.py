from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
import matplotlib.pyplot as plt


N_MOVING_BODIES = 4
L = 0.60
klOC = np.array([L/2, 0, 0])

E = 230e9
G = 80e9
c = 0.50
rho = 7850
n_nd = 4
n_md = 2
w = 0.025
h = 0.025

joint_types = ["revz"] * N_MOVING_BODIES + ["fixed"]
initial_theta = [0.10, 0.07, 0.04, 0]
initial_beta = [0.80, 0.50, 0.20, 0]

bodies = []
for joint_type in joint_types:
    joint = Joint(L, joint_type)
    rigid = Rigid_Properties(rho, klOC, w, h)
    flex = Flex_Properties(E, G, c, n_nd, n_md)
    body = SOABody(joint, rigid, flex)
    bodies.append(body)

for k in range(N_MOVING_BODIES):
    bodies[k].set_initial_theta0(np.array([[initial_theta[k]]]))
    bodies[k].set_initial_beta0(np.array([[initial_beta[k]]]))

system = MultibodySystem(bodies)

tf = 10
dt = 0.002

sim = Simulation(system, tf, dt)
sim.setting.solver = "BDF"

sim.set_camera_ver(90)
sim.set_camera_hor(90)
sim.set_camera_speed(0)
sim.set_ani_dt(0.01)

sim.IntegrateSystem()

# Collect histories
states = sim.get_state()
time = np.array(sim.data.time)

joint_angle_hist = []
for k in range(N_MOVING_BODIES):
    joint_angle_hist.append(
        np.array([state.Theta[k].item() for state in states])
    )

nodal_pos = sim.nNodalPos()
tip_xyz = np.array([frame[-1][-1].flatten() for frame in nodal_pos])

# Plot joint angles
plt.figure(figsize=(8, 4))
for k in range(N_MOVING_BODIES):
    plt.plot(time, joint_angle_hist[k], label=f"Joint {k+1}")
plt.xlabel("Time [s]")
plt.ylabel("Joint angle [rad]")
plt.title("Multi-body pendulum-style chain: joint angles")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot tip trajectory in the plane of motion
plt.figure(figsize=(5, 5))
plt.plot(tip_xyz[:, 0], tip_xyz[:, 1])
plt.xlabel("Tip x-position [m]")
plt.ylabel("Tip y-position [m]")
plt.title("Tip trajectory")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Final joint angles [rad]:")
for k in range(N_MOVING_BODIES):
    print(f"  Joint {k+1}: {joint_angle_hist[k][-1]:.6f}")

print("Final tip position [m]:")
print(tip_xyz[-1])

# Uncomment when you want the 3D animation as well:
sim.animate_nodes()
