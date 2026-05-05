import numpy as np
import matplotlib.pyplot as plt

# Imports based on your provided flex files
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation

# ==========================================
# 1. TANK TRACK PARAMETERS
# ==========================================
N_links = 2          # Number of track links
L_link = 0.8          # Length of each link [m]
w, h = 0.05, 0.01     # Width and height of the track plates [m]
rho = 7850            # Density of steel [kg/m^3]

# Flex parameters (n_md = 0 makes it act as a rigid link within the flex framework)
E, G, c = 210e9, 80e9, 0.2
n_nd = 10  # Minimal nodes since we aren't using flexible modes
n_md = 20

# Torsional Spring Parameters
# The track naturally wants to be straight (theta0_TS = 0)
k_TS = 50.0  # Torsional stiffness [Nm/rad]
c_TS = 5.0   # Torsional damping [Nms/rad]

# Initialize in a loop: dividing 360 degrees (2*pi) evenly among the links
theta_init = (2 * np.pi) / N_links 

# ==========================================
# 2. BUILD THE CHAIN
# ==========================================
bodies = []

for i in range(N_links):
    # Vector from joint to joint (Local X-axis)
    klOO = np.array([L_link, 0, 0]).reshape(3, 1)
    
    # Create Properties
    j = Joint(klOO, "revy")
    r = Rigid_Properties(rho, w, h)
    f = Flex_Properties(E, G, c, n_nd, n_md, mode_selection={
                     "bending_xy": 2,
                     "bending_xz": 2,
                     "axial_x": 1})
    
    # Initialize Body
    b = SOABody(j, r, f)
    
    # Set the joint angle so it curves into a circle
    #b.set_initial_theta0(np.array([theta_init]))
    
    # Add torsional springs to resist bending (simulates the track tension)
    b.set_TS(k_TS, c_TS, theta0_TS=0)
    
    bodies.append(b)

# ==========================================
# 3. APPLY EXTERNAL FORCES
# ==========================================
# Let's apply an external force to the middle link to simulate an 
# upward push from the ground or a bogie wheel hitting a rock.
# The spatial force vector is [Mx, My, Mz, Fx, Fy, Fz]^T. 
# Your gravity is defined in +Z, so "up" is -Z.
F_ext_ground = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
F_ext_axial = np.array([0, 0, 0, 100000, 0, 0]).reshape(6, 1)


# Apply to the tip (node = -1) of the middle link
mid_index = N_links // 2
bodies[mid_index].set_F_ext(node=9, F_ext=F_ext_ground)
bodies[-1].set_F_ext(node=9, F_ext=F_ext_ground)

# ==========================================
# 4. SIMULATION SETUP & EXECUTION
# ==========================================
print(f"Constructing system with {N_links} track links...")
system = MultibodySystem(bodies)
system.set_gravity(False)

# Time parameters
tf = 10
dt = 0.01

sim = Simulation(system, tf, dt)

# Set camera to view the XZ plane (2D view for revy joints)
sim.set_camera_ver(0)    # 0 elevation
sim.set_camera_hor(-90)  # Looking down the Y-axis

sim.IntegrateSystem("Radau")

# Render the result
print("Rendering animation...")
save_dir = r"C:\Users\jepp6\OneDrive - Aarhus universitet\Dokumenter\Noter\6. Semester\Bachelor Projekt\BachelorCode\Renders"
#sim.animate_nodes(filename="TankTrack_2D", save_dir=save_dir)
sim.animate_nodes()