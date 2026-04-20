from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation
import numpy as np
from SOALIB import soalib as sb
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Setup
klOO = np.array([1, 0, 0])
H_type1 = "spherical"
H_type2 = "spherical"

m = 1
CkJk = np.array([1/12, 1/12, 1/12])
klOC = np.array([0.5, 0, 0])

j1 = Joint(klOO, H_type1)
i = Inertia(m, CkJk, klOC)
b1 = SOABody(j1, i)

j2 = Joint(klOO, H_type2)
i = Inertia(m, CkJk, klOC)
b2 = SOABody(j2, i)

b1.set_initial_theta0(sb.get_quat_from_degrees(0, 0, 90))

bodies = [b1, b2]
system = MultibodySystem(bodies)

# Time
tf = 10
dt = 0.01
nt = int(tf/dt + 1)

sim = Simulation(system, tf, dt)
sim.camera_speed(1)
sim.IntegrateSystem()
sim.animate()

# Mass matrix at COM
n = len(system.bodies)
MC = [None] * n

for j in range(n):
    # Fetching the corrected self.MC from the Inertia class
    MC[j] = system.bodies[j].inertia.MC

# Velocities
V_list = sim.get_V()
nt = len(V_list) # Total number of time steps

# Position
Pos_list = sim.get_pos()
g = 9.81

# List to store the total energy
KE_list = []
PE_list = []
E_list = []
P_list = []
W_list = []

accumulated_work = 0

# Energy
for i in range(nt):
    total_KE = 0
    total_PE = 0
    
    for j in range(n):
        # Kinetic energy
        klOC = system.bodies[j].inertia.klOC
        VC_j = sb.phi(klOC).T @ V_list[i][j]
        KE_j = 0.5 * (VC_j.T @ MC[j] @ VC_j).item()
        total_KE += KE_j

        # Potential energy
        m = system.bodies[j].inertia.m
        z = (Pos_list[i][j][2].item() + Pos_list[i][j+1][2].item()) / 2
        PE_j = z*g*m
        total_PE += PE_j

    KE_list.append(total_KE)
    PE_list.append(total_PE)
    E_list.append(total_KE + total_PE)

t_vector = sim.data.time

plt.figure(figsize=(10, 6))
# Plot actual total energy
plt.plot(t_vector, E_list, label="Total Energy (Kinetic + Potential)", color='blue', linewidth=3)

plt.title("Energy Analysis of 2-Body-Pendulum", fontsize=14)
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Total Energy (J)", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()