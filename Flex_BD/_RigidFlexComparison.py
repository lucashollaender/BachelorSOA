import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. IMPORTS (Aliased to avoid conflicts)
# ==========================================
# Flex Library Imports
from Body_Properties import Joint as FlexJoint, Rigid_Properties, Flex_Properties
from SOABody import SOABody as FlexSOABody
from MultibodySystem import MultibodySystem as FlexMultibodySystem
from Simulation import Simulation as FlexSimulation

# Rigid Library Imports (Assuming it's saved as RigidForwardSOA.py)
from SOALIB.RigidForwardSOA import Joint as RigidJoint, Inertia, SOABody as RigidSOABody 
from SOALIB.RigidForwardSOA import MultibodySystem as RigidMultibodySystem, Simulation as RigidSimulation

# ==========================================
# 2. SHARED PARAMETERS
# ==========================================
# Kinematics
klOO1 = np.array([1, 0, 0]).reshape(3, 1)
klOO2 = np.array([0, 1, 0]).reshape(3, 1)
H_type1 = "spherical"
H_type2 = "spherical"

# Mass & Inertia Targeting (m = rho * w * h * L)
# We want exactly 1 kg. Since L1 = L2 = 1, we can set w=0.1, h=0.1 (A=0.01) and rho=100.
w, h = 0.1, 0.1
rho = 100.0 
m_target = 1.0  # (100 * 0.1 * 0.1 * 1) = 1.0 kg

# Flex-specific parameters (dummy values since n_md=0)
E, G, c = 230e6, 80e6, 0.02
n_nd = 2

# Simulation settings
tf = 10
dt = 0.01

# ==========================================
# 3. FLEX IMPLEMENTATION (n_md = 0)
# ==========================================
print("Setting up Flex Simulation (n_md=0)...")

# Body 1
flex_j1 = FlexJoint(klOO1, H_type1)
flex_r1 = Rigid_Properties(rho, w, h)
flex_f1 = Flex_Properties(E, G, c, n_nd, n_md=0)
flex_b1 = FlexSOABody(flex_j1, flex_r1, flex_f1)

# Body 2
flex_j2 = FlexJoint(klOO2, H_type2)
flex_r2 = Rigid_Properties(rho, w, h)
flex_f2 = Flex_Properties(E, G, c, n_nd, n_md=0)
flex_b2 = FlexSOABody(flex_j2, flex_r2, flex_f2)

# Run Flex Sim
flex_system = FlexMultibodySystem([flex_b1, flex_b2])
flex_sim = FlexSimulation(flex_system, tf, dt)
flex_sim.set_max_step(dt)
flex_sim.set_tol(1e-8, 1e-10)
flex_sim.IntegrateSystem("RK45") # Use RK45 to match the rigid solver
#flex_sim.animate_nodes()

# ==========================================
# 4. RIGID IMPLEMENTATION
# ==========================================
print("Setting up Rigid Simulation...")

# Calculate identical CkJk to the Flex model to ensure matching inertia matrices
L1 = np.linalg.norm(klOO1)
L2 = np.linalg.norm(klOO2)
CkJk1 = np.array([1/12 * m_target * (h**2 + w**2), 1/12 * m_target * (h**2 + L1**2), 1/12 * m_target * (w**2 + L1**2)])
CkJk2 = np.array([
    1/12 * m_target * (h**2 + L2**2),  # X-axis
    1/12 * m_target * (h**2 + w**2),   # Y-axis (Longitudinal)
    1/12 * m_target * (w**2 + L2**2)   # Z-axis
])
klOC1 = klOO1 / 2
klOC2 = klOO2 / 2

# Body 1
rigid_j1 = RigidJoint(klOO1, H_type1)
rigid_i1 = Inertia(m_target, CkJk1, klOC1)
rigid_b1 = RigidSOABody(rigid_j1, rigid_i1)

# Body 2
rigid_j2 = RigidJoint(klOO2, H_type2)
rigid_i2 = Inertia(m_target, CkJk2, klOC2)
rigid_b2 = RigidSOABody(rigid_j2, rigid_i2)

# Run Rigid Sim
rigid_system = RigidMultibodySystem([rigid_b1, rigid_b2])
rigid_sim = RigidSimulation(rigid_system, tf, dt)
rigid_sim.IntegrateSystem()
#rigid_sim.animate()

# ==========================================
# 5. DATA EXTRACTION & COMPARISON
# ==========================================
print("Extracting data and plotting...")

t_flex = flex_sim.data.time
t_rigid = rigid_sim.data.time

labels = ['Angular x', 'Angular y', 'Angular z',
          'Linear x', 'Linear y', 'Linear z']

for body_idx in range(2):

    # Extract accelerations
    accel_flex = np.array([
        a[body_idx][-6:].flatten()
        for a in flex_sim.get_alpha_fl()
    ])

    accel_rigid = np.array([
        a[body_idx].flatten()
        for a in rigid_sim.get_alpha()
    ])

    # Assign colors based on the body index
    if body_idx == 0:
        flex_color = 'b'           # Blue
        flex_style = '-'           # Solid line
        rigid_color = 'r'          # Red
        rigid_style = '--'         # Dashed (original)
    else:
        flex_color = 'darkgreen'   # Dark green
        flex_style = '-'           # Solid line
        rigid_color = 'y'          # Yellow
        rigid_style = ':'          # Dotted line

    # Create figure
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    # Apply saved subplot spacing
    fig.subplots_adjust(
        left=0.12,
        bottom=0.224,
        right=0.88,
        top=0.906,
        wspace=0.505,
        hspace=0.565
    )

    fig.suptitle(
        rf"Comparison of Body {body_idx+1} Spatial Accelerations:"
        rf" Flex ($n_{{md}}$=0) vs. Rigid Forward SOA",
        fontsize=14
    )

    # Plot components
    for i in range(6):
        ax = axs[i//3, i%3]

        # Plot Flex
        ax.plot(
            t_flex,
            accel_flex[:, i],
            color=flex_color,
            linestyle=flex_style,
            linewidth=2,
            label=r'Flex ($n_{md}$=0)'
        )

        # Plot Rigid
        ax.plot(
            t_rigid,
            accel_rigid[:, i],
            color=rigid_color,
            linestyle=rigid_style,
            linewidth=2,
            label='Rigid SOA'
        )

        ax.set_title(labels[i])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r"Acceleration $[rad/s^2]$ or $[m/s^2]$")
        ax.grid(True)

        if np.linalg.norm(accel_flex[:, i]) < 1e-4:
            ax.set_ylim(-20, 20)

        if i == 0:
            ax.legend()

    plt.show()