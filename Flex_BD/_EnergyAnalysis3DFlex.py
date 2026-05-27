import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Use Flexible Architecture Imports
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from SOALIB import soalib as sb

def run_simulation_no_force():
    # Body setup
    klOO = np.array([1, 0, 0]).reshape(3, 1)
    H_type1 = "spherical"
    H_type2 = "spherical"

    # Properties to match m = 1kg exactly as done in _RigidFlexComparison.py
    w, h = 0.1, 0.1
    rho = 100.0

    # Flex properties 
    E, G, c = 230e9, 80e9, 0
    n_nd = 3 # 3 nodes -> node 1 is exactly at the COM (0.5)
    n_md = 0 # Zero modes -> Body behaves completely rigidly

    j1 = Joint(klOO, H_type1)
    r1 = Rigid_Properties(rho, w, h)
    f1 = Flex_Properties(E, G, c, n_nd, n_md)
    b1 = SOABody(j1, r1, f1)

    j2 = Joint(klOO, H_type2)
    r2 = Rigid_Properties(rho, w, h)
    f2 = Flex_Properties(E, G, c, n_nd, n_md)
    b2 = SOABody(j2, r2, f2)

    b1.set_initial_theta0(sb.get_quat_from_degrees(0, 0, 90))

    bodies = [b1, b2]
    system = MultibodySystem(bodies)

    # Time
    tf = 10
    dt = 0.01

    # Simulation
    sim = Simulation(system, tf, dt)
    sim.set_camera_speed(1)
    
    # We use RK45 settings matching the rigid/flex comparison script
    sim.set_max_step(dt)
    sim.set_tol(1e-8, 1e-10)
    sim.IntegrateSystem("Radau")

    # Extract Results
    t_vector = sim.data.time
    V_list = sim.get_V_fl() # Spatial velocities
    Pos_list = sim.nNodalPos()
    g = 9.81
    n = len(bodies)

    # List to store the total energy
    KE_list = []
    PE_list = []
    E_list = []

    # Energy
    for i in range(len(t_vector)):
        total_KE = 0
        total_PE = 0
        
        for j in range(n):
            body = system.bodies[j]
            V_j = V_list[i][j]
            M_fl = body.flex.M_fl
            
            # Kinetic energy (0.5 * V^T * M * V mapped to the body frame)
            KE_j = 0.5 * (V_j.T @ M_fl @ V_j).item()
            total_KE += KE_j

            # Potential energy
            m = body.m
            # Using node 1 which represents the COM (s = 0.5)
            z = Pos_list[i][j][1][2].item() 
            PE_j = z * g * m
            total_PE += PE_j

        KE_list.append(total_KE)
        PE_list.append(total_PE)
        E_list.append(total_KE + total_PE)

    return t_vector, E_list

def run_simulation_ext_force():
    # Body setup
    klOO = np.array([1, 0, 0]).reshape(3, 1)
    H_type1 = "spherical"
    H_type2 = "spherical"

    # Properties to match m = 1kg exactly as done in _RigidFlexComparison.py
    w, h = 0.1, 0.1
    rho = 100.0

    # Flex properties 
    E, G, c = 230e6, 80e6, 0.02
    n_nd = 3 # 3 nodes -> node 1 is exactly at the COM (0.5)
    n_md = 0 # Zero modes -> Body behaves completely rigidly

    j1 = Joint(klOO, H_type1)
    r1 = Rigid_Properties(rho, w, h)
    f1 = Flex_Properties(E, G, c, n_nd, n_md)
    b1 = SOABody(j1, r1, f1)

    j2 = Joint(klOO, H_type2)
    r2 = Rigid_Properties(rho, w, h)
    f2 = Flex_Properties(E, G, c, n_nd, n_md)
    b2 = SOABody(j2, r2, f2)

    b1.set_initial_theta0(sb.get_quat_from_degrees(0, 0, 90))

    # External forces
    F_ext1 = np.array([0, 1, -3, 0, -2, 1.5]).reshape(6, 1)
    b1.set_F_ext(node=1, F_ext=F_ext1) # Node 1 is COM

    F_ext2 = np.array([0, -1, 0.5, 0, -1, 2]).reshape(6, 1)
    b2.set_F_ext(node=1, F_ext=F_ext2) # Node 1 is COM

    bodies = [b1, b2]
    system = MultibodySystem(bodies)

    # Time
    tf = 10
    dt = 0.01

    # Simulation
    sim = Simulation(system, tf, dt)
    sim.set_camera_speed(0)

    # We use RK45 settings matching the rigid/flex comparison script
    sim.set_max_step(dt)
    sim.set_tol(1e-8, 1e-10)
    sim.IntegrateSystem("Radau")

    # Extract Results
    t_vector = sim.data.time
    V_list = sim.get_V_fl()
    Pos_list = sim.nNodalPos()
    g = 9.81
    n = len(bodies)

    # List to store the total energy
    KE_list = []
    PE_list = []
    E_list = []
    P_list = []

    # Energy
    for i in range(len(t_vector)):
        total_KE = 0
        total_PE = 0
        total_Power = 0
        
        for j in range(n):
            body = system.bodies[j]
            V_j = V_list[i][j]
            M_fl = body.flex.M_fl
            
            # Kinetic energy
            KE_j = 0.5 * (V_j.T @ M_fl @ V_j).item()
            total_KE += KE_j

            # Potential energy
            m = body.m
            z = Pos_list[i][j][1][2].item()
            PE_j = z * g * m
            total_PE += PE_j

            # Work
            # Ext force mapped natively into root body coordinates 
            F_ext_j = body.get_F_ext_term(sim.data.state[i], t_vector[i])
            Power_j = (F_ext_j.T @ V_j).item()
            total_Power += Power_j

        KE_list.append(total_KE)
        PE_list.append(total_PE)
        E_list.append(total_KE + total_PE)
        P_list.append(total_Power)

    W_list = cumulative_trapezoid(P_list, x=t_vector, initial=0)

    # Calculate the theoretical energy curve (Initial Energy + Work Done)
    E_initial = E_list[0]
    Theoretical_Energy = [E_initial + w for w in W_list]

    return t_vector, E_list, Theoretical_Energy

# Results
t_no_force, E_no_force = run_simulation_no_force()
t_ext_force, E_ext_force, Theoretical_Energy = run_simulation_ext_force()

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
y1_min = -2e-8

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

plt.title(r"Energy Analysis of 2-Body-Pendulum ($n_{md}=0$)", fontsize=15)
plt.show()