from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation
import numpy as np
from SOALIB import soalib as sb
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def run_simulation_no_force():
    # Body setup
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

    # Simulation
    sim = Simulation(system, tf, dt)
    sim.camera_speed(1)
    sim.IntegrateSystem()
    #sim.animate()

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
    return t_vector, E_list

def run_simulation_ext_force():
    # Body setup
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

    # External force
    F_ext1 = [np.array([0, 1, -3, 0, -2, 1.5]).reshape(6, 1)]
    klOB1 = [np.array([0.5, 0, 0]).reshape(3, 1)]
    b1.set_F_ext(F_ext1, klOB1)

    F_ext2 = [np.array([0, -1, 0.5, 0, -1, 2]).reshape(6, 1)]
    klOB2 = [np.array([0.5, 0, 0]).reshape(3, 1)]
    b2.set_F_ext(F_ext2, klOB2)

    bodies = [b1, b2]
    system = MultibodySystem(bodies)

    # Time
    tf = 10
    dt = 0.01
    nt = int(tf/dt + 1)

    # Simulation
    sim = Simulation(system, tf, dt)
    sim.camera_speed(0)
    sim.IntegrateSystem()
    #sim.animate()

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
            Power_j = (F_ext_j.T @ V_j).item()
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

    return t_vector, E_list, Theoretical_Energy

# Results
t_no_force, E_no_force = run_simulation_no_force()
t_ext_force, E_ext_force, Theoretical_Energy = run_simulation_ext_force()
"""
# Plotting
plt.figure(figsize=(10, 6))
    
# Standard Pendulum Energy
plt.plot(t_no_force, E_no_force, label="Total Energy (No External Force)", color='green', linewidth=3)
    
# Pendulum with External Force Energy
plt.plot(t_ext_force, E_ext_force, label="Total Energy (External Force)", color='blue', linewidth=3)
    
# Theoretical Energy of External Force Pendulum
plt.plot(t_ext_force, Theoretical_Energy, label="Theoretical Energy (Initial + Work)", color='orange', linestyle='--', linewidth=2)

plt.title("Energy Analysis of 2-Body-Pendulums", fontsize=14)
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Energy or Work (J)", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()
"""

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Standard Pendulum Energy on Left y-axis
line1 = ax1.plot(t_no_force, E_no_force, label="Total Energy (No External Force)", color='green', linewidth=3)
ax1.set_xlabel("Time (s)", fontsize=14)
ax1.set_ylabel("Total Energy (J)", fontsize=14, color='green')
ax1.tick_params(axis='y', labelcolor='green')
ax1.grid(True)

# Create Right y-axis sharing the same x-axis
ax2 = ax1.twinx()

# External Force Energy on Right y-axis
line2 = ax2.plot(t_ext_force, E_ext_force, label="Total Energy (External Force)", color='blue', linewidth=3)
line3 = ax2.plot(t_ext_force, Theoretical_Energy, label="Theoretical Energy (External Force)", color='orange', linestyle='--', linewidth=2)
ax2.set_ylabel("Total Energy with Work (J)", fontsize=14, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# --- ALIGN ZERO LINES ---
# 1. Lock left y-axis minimum
y1_min = -2e-6

# 2. Get current data bounds to ensure no data is cut off
_, y1_max = ax1.get_ylim()
y2_min, y2_max = ax2.get_ylim()

# Ensure we aren't dealing with inverted bounds
y1_max = max(y1_max, 0)
y2_max = max(y2_max, 0)
y2_min = min(y2_min, 0)

# 3. Calculate required max-to-abs(min) ratios
r1 = y1_max / abs(y1_min) 

if y2_min < 0:
    r2 = y2_max / abs(y2_min)
    
    if r1 >= r2:
        # Left axis needs a larger ratio. Apply r1 to right axis.
        # Expand right axis upwards to avoid cutting off its bottom data.
        y2_max = r1 * abs(y2_min)
    else:
        # Right axis needs a larger ratio. Apply r2 to left axis.
        # Expand left axis upwards.
        y1_max = r2 * abs(y1_min)
else:
    # Right axis has no negative data natively.
    # Force a negative bottom bound to match the left axis's ratio.
    y2_min = -y2_max / r1 if r1 != 0 else -0.02

# Apply the aligned bounds
ax1.set_ylim(y1_min, y1_max)
ax2.set_ylim(y2_min, y2_max)
# ------------------------

# Combine legends from both axes into one cohesive box
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=12, loc='upper left')

plt.title("Energy Analysis of 2-Body-Pendulum", fontsize=15)
plt.show()