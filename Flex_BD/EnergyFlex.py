import numpy as np
import matplotlib.pyplot as plt

# Import classes from your new flexible framework
from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties

# Setup: 2-Body Spherical Pendulum
klOO1 = np.array([1, 0, 0]).reshape(3, 1)
klOO2 = np.array([0, 1, 0]).reshape(3, 1)
H_type1 = "spherical"
H_type2 = "spherical"

# Flexible Parameters
E, G = 230e9, 80e9
c = 0.0
rho = 7850
n_nd, n_md = 10, 0
w, h = 0.04, 0.06

# Properties for Body 1
j1 = Joint(klOO1, H_type1)
r1 = Rigid_Properties(rho, w, h)
f1 = Flex_Properties(E, G, c, n_nd, n_md)
b1 = SOABody(j1, r1, f1)

# Properties for Body 2
j2 = Joint(klOO2, H_type2)
r2 = Rigid_Properties(rho, w, h)
f2 = Flex_Properties(E, G, c, n_nd, n_md)
b2 = SOABody(j2, r2, f2)

# Build system
bodies = [b1, b2]
system = MultibodySystem(bodies)

# Time properties
tf = 5
dt = 0.01

# Simulate
sim = Simulation(system, tf, dt)
sim.IntegrateSystem("Radau")
sim.set_camera_speed(0)
sim.set_tol(1e-10, 1e-12)

sim.animate_nodes()

# Fetch integrated data
states = sim.get_state()
V_list = sim.get_V_fl()
nodal_pos = sim.nNodalPos()
t_vector = sim.data.time

nt = len(t_vector)
n = len(bodies)
g = 9.81

# Lists to track energy over time
KE_list = []
PE_list = []
E_list = []

# Energy Evaluation Loop
for i in range(nt):
    state = states[i]
    V_fl = V_list[i]
    nodes_i = nodal_pos[i] 

    total_KE = 0
    total_PE = 0

    # 1 & 2. Calculate KE and Elastic PE
    for k in range(n):
        body = bodies[k]

        # --- Kinetic Energy ---
        M_fl = body.flex.M_fl
        V_k = V_fl[k]
        KE_k = 0.5 * (V_k.T @ M_fl @ V_k).item()
        total_KE += KE_k

        # --- Elastic Potential Energy ---
        eta = state.Eta[k]
        K_fl_elast = body.flex.K_fl[0:body.flex.n_md, 0:body.flex.n_md]
        PE_elast_k = 0.5 * (eta.T @ K_fl_elast @ eta).item()
        total_PE += PE_elast_k

    # 3. Calculate Gravitational PE 
    for idx, body_nodes in enumerate(nodes_i):
        body_index = n - 1 - idx
        body = bodies[body_index]
        n_nodes = body.flex.n_nd

        m_e = body.m / (n_nodes - 1)
        m_nd = np.full(n_nodes, m_e)
        m_nd[0] = m_e / 2
        m_nd[-1] = m_e / 2

        for j in range(n_nodes):
            z = body_nodes[j][2].item()
            total_PE += m_nd[j] * g * z

    KE_list.append(total_KE)
    PE_list.append(total_PE)
    E_list.append(total_KE + total_PE)

# ----------------------------------------------------
# Plotting the Energy Conservation (Dual Y-Axis)
# ----------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))

plt.title("Energy Analysis of Flexible 2-Body Pendulum (No External Load)", fontsize=14)

# ---- Primary axis (KE + PE) ----
ax1.set_xlabel("Time (s)", fontsize=14)
ax1.set_ylabel("Kinetic & Potential Energy (J)", fontsize=14)

l1, = ax1.plot(t_vector, KE_list, linestyle='--', linewidth=2,
               color='red', label="Kinetic Energy")
l2, = ax1.plot(t_vector, PE_list, linestyle='--', linewidth=2,
               color='green', label="Total Potential Energy")

ax1.grid(True)
ax1.set_xlim(t_vector[0], t_vector[-1])

# ---- Secondary axis (Total Energy) ----
ax2 = ax1.twinx()
ax2.set_ylabel("Total Energy (J)", fontsize=14, color='blue')

l3, = ax2.plot(t_vector, E_list, linewidth=3,
               color='blue', label="Total Energy")

ax2.tick_params(axis='y', labelcolor='blue')

# ---- Zero reference line ----
ax1.axhline(0, color='black', linewidth=1, alpha=0.5)

# ----------------------------------------------------
# SYMMETRIC, USER-CONTROLLED AXIS SCALING
# ----------------------------------------------------
scale1 = 1.5   # scaling for KE + PE axis
scale2 = 5  # scaling for total energy axis

def symmetric_ylim(data, scale):
    ymax = np.max(np.abs(data))
    if ymax == 0:
        ymax = 1.0  # avoid degenerate axis
    return (-scale * ymax, scale * ymax)

# Apply limits
ax1.set_ylim(*symmetric_ylim(np.concatenate([KE_list, PE_list]), scale1))
ax2.set_ylim(*symmetric_ylim(E_list, scale2))

# ----------------------------------------------------
# Legend (combined)
# ----------------------------------------------------
lines = [l1, l2, l3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=12)

fig.tight_layout()
plt.show()