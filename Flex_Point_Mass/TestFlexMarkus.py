from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
import matplotlib.pyplot as plt

L = 3

klOC = np.array([L/2, 0, 0])

E, G, c, rho, n_nd, n_md = 230e9, 80e9, 0, 7850, 6, 4
w, h = 0.1, 0.1

j1 = Joint(L, "fixed")
r1 = Rigid_Properties(rho, klOC, w, h)
f1 = Flex_Properties(E, G, c, n_nd, n_md)
b1 = SOABody(j1, r1, f1)

PIe = b1.flex.PI_end


print(np.linalg.norm(PIe[3, :]))
print(np.linalg.norm(PIe[4, :]))
print(np.linalg.norm(PIe[5, :]))

F_ext = np.array([0, 0, 0, 0, -1e5, 0]).reshape(6, 1)
b1.set_F_ext(F_ext)

bodies = [b1]
system = MultibodySystem(bodies)

tf = 2
dt = 0.01

sim = Simulation(system, tf, dt)

sim.set_camera_ver(90)
sim.set_camera_hor(90)
sim.set_camera_speed(0)
sim.set_ani_dt(0.01)

sim.setting.solver = "BDF"
sim.IntegrateSystem()

nodal_pos = sim.nNodalPos()
time = np.array(sim.data.time)
end_node_y = np.array([frame[0][-1][1, 0] for frame in nodal_pos])

plt.figure(figsize=(8, 4))
plt.plot(time, end_node_y)
plt.xlabel("Time [s]")
plt.ylabel("End-node y-position [m]")
plt.title("Single flexible body response to applied force")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Final end-node y-position: {end_node_y[-1]:.6e} m")

sim.animate_nodes()
