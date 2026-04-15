from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation
import numpy as np
from SOALIB import soalib as sb
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Setup
klOO = np.array([0, 0, 1])
H_type1 = "revx"
H_type2 = "revx"

m = 1
CkJk = np.array([1/12, 1/12, 1/12])
klOC = np.array([0, 0, 0.5])

j1 = Joint(klOO, H_type1)
i = Inertia(m, CkJk, klOC)
b1 = SOABody(j1, i)

j2 = Joint(klOO, H_type2)
i = Inertia(m, CkJk, klOC)
b2 = SOABody(j2, i)

b1.set_initial_theta0(0)
b2.set_initial_theta0(np.pi/2)

F_ext = [np.array([0, 0, 0, 0, 5, 0]).reshape(6, 1)]
klOB = [np.array([0, 0, 0.5]).reshape(3, 1)]
b1.set_F_ext(F_ext, klOB)

bodies = [b1, b2]
system = MultibodySystem(bodies)

# Time
tf = 10
dt = 0.01
nt = int(tf/dt + 1)

sim = Simulation(system, tf, dt)
sim.camera_speed(0)
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
    total_Power = 0
    
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

        # Work
        F_ext_j = system.bodies[j].force.sum_phi_F_ext
        V_j = V_list[i][j]
        Power_j = (V_j.T @ F_ext_j).item()
        total_Power += Power_j

    KE_list.append(total_KE)
    PE_list.append(total_PE)
    E_list.append(total_KE + total_PE)
    P_list.append(total_Power)

t_vector = sim.data.time

W_list = cumulative_trapezoid(P_list, x=t_vector, initial=0)

# Calculate the theoretical energy curve (Initial Energy + Work Done)
E_initial = E_list[0]
Theoretical_Energy = [E_initial + w for w in W_list]

plt.figure(figsize=(10, 6))
# Plot actual total energy
plt.plot(t_vector, E_list, label="Total Energy (Kinetic + Potential)", color='blue', linewidth=3)

# Plot theoretical energy (should perfectly overlap the blue line)
plt.plot(t_vector, Theoretical_Energy, label="Theoretical Energy (Initial + Work)", color='orange', linestyle='--', linewidth=2)

plt.title("Energy Analysis of 2-Body-Pendulum (5N load)", fontsize=14)
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Energy or Work (J)", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()



