import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as Rot
from SOALIB import soalib as sb
from scipy.integrate import solve_ivp
import os

# Increase limit to 100 MB (default is 20)
plt.rcParams['animation.embed_limit'] = 1000


def get_coriolis(V, beta, H):
    # Takes spatial velocity, beta, and hinge map and returns coriolis acceleration.
    
    # H.T @ beta performs the matrix multiplication (H' * beta)
    deltaV = H.T @ beta
    
    # Standard spatial cross-product formulation
    # cor = skew6(V)*deltaV - spatialBar(deltaV)*deltaV
    cor = sb.skew6(V) @ deltaV - sb.bar6(deltaV) @ deltaV
    
    return cor

def get_gyroscopic_force(V, M):
    # Takes spatial velocity V and spatial inertia M and returns the gyroscopic force term.
    
    # Equivalent to spatialBar(V) * M * V
    gyr = sb.bar6(V) @ M @ V
    
    return gyr

def scatter_kinematics(Theta, Beta, klOO, Mk, H):
    # Step 1 and 2 of ATBI (first scatter sweep): Takes generalized vectors 
    # and returns the system-level vectors with a scatter loop

    # Number of bodies
    n = len(Theta)

    # Set up lists
    X = [None] * n
    V = [None] * n
    a = [None] * n
    b = [None] * n

    # Loop backwards from n-1 down to 0
    for k in range(n - 1, -1, -1):

        # Relative orientation with quaternion and X-vector setup
        if np.size(Theta[k]) == 1 and H[k][0, 0] == 1: # revx
            ang = Theta[k].item()
            q = np.array([[np.sin(ang/2)], [0], [0], [np.cos(ang/2)]])
            X[k] = np.vstack((q, klOO[k]))
        
        elif np.size(Theta[k]) == 1 and H[k][0, 1] == 1: # revy
            ang = Theta[k].item()
            q = np.array([[0], [np.sin(ang/2)], [0], [np.cos(ang/2)]])
            X[k] = np.vstack((q, klOO[k]))
        
        elif np.size(Theta[k]) == 1 and H[k][0, 2] == 1: # revz
            ang = Theta[k].item()
            q = np.array([[0], [0], [np.sin(ang/2)], [np.cos(ang/2)]])
            X[k] = np.vstack((q, klOO[k]))

        elif np.size(Theta[k]) == 4: # spherical
            q = Theta[k].reshape(4, 1)
            q = q / np.linalg.norm(q)
            X[k] = np.vstack((q, klOO[k]))
        
        elif np.size(Theta[k]) == 7: # free
            q = np.array(Theta[k][0:4]).reshape(-1, 1)
            q = q / np.linalg.norm(q)
            X[k] = Theta[k].reshape(7, 1)
        
        elif np.size(Theta[k]) == 0: # fixed
            q = np.array([[0], [0], [0], [1]])
            X[k] = np.vstack((q, klOO[k]))

        if k == n - 1:
            # Scatter loop (Tip of the chain)
            V[k] = H[k].T @ Beta[k]
        else:
            # Rotation matrix
            R6 = sb.q2R(q.flatten(), 6)

            # Scatter loop
            V[k] = R6.T @ sb.phi(klOO[k+1]).T @ V[k+1] + H[k].T @ Beta[k]

        # Coriolis term
        a[k] = get_coriolis(V[k], Beta[k], H[k])

        # Gyroscopic term
        b[k] = get_gyroscopic_force(V[k], Mk[k])
    
    return X, V, a, b

def gather_ATBI(a, b, X, Mk, H, tau, F_ext):
    # Step 3 of ATBI (gather sweep): Takes generalized forces, Coriolis-, gyroscopic
    # terms, X-vector and system configuration and returns G and nu parameters

    # Number of bodies
    n = len(a)

    # Setup lists
    P_plus = [None] * n
    xi_plus = [None] * n
    G = [None] * n
    nu = [None] * n

    for k in range(n):

        # External force
        klBO = F_ext[k][0:3]
        F = F_ext[k][3:9]

        if k == 0:
            # Gather loop for k = 0 (Base Case)
            P = Mk[k]
            D = H[k] @ P @ H[k].T
            G[k] = (P @ H[k].T) @ np.linalg.inv(D)
            tau_bar = np.eye(6) - G[k] @ H[k]
            P_plus[k] = tau_bar @ P
            xi = P @ a[k] + b[k] - sb.phi(klBO) @ F
            epsilon = tau[k] - H[k] @ xi
            nu[k] = np.linalg.solve(D, epsilon)
            xi_plus[k] = xi + G[k] @ epsilon
            
        else:
            # Unpacking X-vector
            q = X[k-1][0:4]
            klOO = X[k][4:7]

            # Rotation
            R6 = sb.q2R(q.flatten(), 6)
            
            # Gather loop for k > 0
            P = sb.phi(klOO) @ R6 @ P_plus[k-1] @ R6.T @ sb.phi(klOO).T + Mk[k]
            D = H[k] @ P @ H[k].T
            G[k] = (P @ H[k].T) @ np.linalg.inv(D)
            tau_bar = np.eye(6) - G[k] @ H[k]
            P_plus[k] = tau_bar @ P
            xi = sb.phi(klOO) @ R6 @ xi_plus[k-1] + P @ a[k] + b[k] - sb.phi(klBO) @ F
            epsilon = tau[k] - H[k] @ xi
            nu[k] = np.linalg.solve(D, epsilon)
            xi_plus[k] = xi + G[k] @ epsilon

    return G, nu

def scatter_ATBI(a, X, G, nu, H):
    # Step 4 of ATBI (second scatter sweep): Takes Coriolis term, X-vector, G,
    # nu and hinge map, H and returns generalized acceleration, gamma

    # Number of bodies
    n = len(a)

    # Setup of list
    alpha = [None] * n
    alpha_plus = [None] * n
    gamma = [None] * n

    # Spatial gravity
    g = np.array([0, 0, 0, 0, 0, 9.81]).reshape(6, 1)
    
    # Spatial gravity rotation setup
    Ri = [None] * (n + 1)
    Ri[-1] = np.eye(6)

    # Loop backwards from n-1 down to 0
    for k in range(n - 1, -1, -1):

        # Unpacking rotation
        q = X[k][0:4]

        # Rotation
        R6 = sb.q2R(q.flatten(), 6)

        # Spatial gravity rotation
        Ri[k] =   Ri[k+1]  @ R6

        if k == n - 1:
            # Scatter loop (Tip of the chain)
            nu_bar = nu[k] - (G[k].T @ Ri[k].T @ g)
            gamma[k] = nu_bar
            alpha[k] = H[k].T @ gamma[k] + a[k]
        
        else:
            # Hinge vector
            klOO = X[k+1][4:7]
        
            # Scatter loop
            alpha_plus[k] = R6.T @ sb.phi(klOO).T @ alpha[k+1]
            nu_bar = nu[k] - (G[k].T @ Ri[k].T @ g)
            gamma[k] = nu_bar - G[k].T @ alpha_plus[k]
            alpha[k] = alpha_plus[k] + H[k].T @ gamma[k] + a[k]

    return gamma

def ATBI(Theta, Beta, klOO, Mk, H, tau, F_ext):
    # Takes theta, beta, klOO, Mk and H and returns gamma
    
    # 1 and 2) Scatter loop to find X=[q, klOO], velocity, V, corriolis acceleration, a and gyroscopic force, b
    X, V, a, b = scatter_kinematics(Theta, Beta, klOO, Mk, H)

    # 3) Gather loop to find G and nu
    G, nu = gather_ATBI(a, b, X, Mk, H, tau, F_ext)

    # 4) Scatter loop to find gamma
    gamma = scatter_ATBI(a, X, G, nu, H)

    return gamma

def unpack_state(S, n, H_type):
    # Takes the state vector S and returns lists of Theta and Beta
    
    # Ensure S is flat
    S = S.flatten() 
    
    theta_vec = S[0:4*n]
    beta_vec = S[4*n:7*n]


    Theta = []
    Beta = []

    # Tracker
    x = 0

    # Change to list with n arrays
    # Theta
    for k in range(n):
        if H_type[k] in ["revx", "revy", "revz"]:
            theta = np.array(S[x]).reshape(1, 1)
            x = x + 1
        elif H_type[k] == "spherical":
            theta = np.array(S[x : x + 4]).reshape(4, 1)
            x = x + 4
        elif H_type[k] == "free":
            theta = np.array(S[x : x + 7]).reshape(7, 1)
            x = x + 7
        elif H_type[k] == "fixed":
            theta = np.zeros((0, 1))
            x = x + 0
        Theta.append(theta)
    
    # Beta
    for k in range(n):
        if H_type[k] in ["revx", "revy", "revz"]:
            beta = np.array(S[x]).reshape(1, 1)
            x = x + 1
        elif H_type[k] == "spherical":
            beta = np.array(S[x : x + 3]).reshape(3, 1)
            x = x + 3
        elif H_type[k] == "free":
            beta = np.array(S[x : x + 7]).reshape(6, 1)
            x = x + 6
        elif H_type[k] == "fixed":
            beta = np.zeros((0, 1))
            x = x + 0
        Beta.append(beta)

    return Theta, Beta

def EOM(S, n, klOO, Mk, H_type, H, tau, F_ext):
    # Takes state, S, number of links, n, hinge pos, klOO,
    # spatial in hinge inertia, Mk and hinge map, H and returns 
    # generalized acceleration, alpha

    # Unpacking of state, S
    Theta, Beta = unpack_state(S, n, H_type)

    # Find gamma
    gamma = ATBI(Theta, Beta, klOO, Mk, H, tau, F_ext)

    # Assemble S_dot
    Theta_dot = []

    for k in range(n):
        if H_type[k] in ["revx", "revy", "revz"]:
            theta_dot = Beta[k]
        elif H_type[k] == "spherical":
            theta_dot = sb.quat_derivative(Theta[k], Beta[k]).reshape(4, 1)
        elif H_type[k] == "free":
            q = sb.quat_derivative(Theta[k][0:4], Beta[k][0:3]).reshape(4, 1)
            v = Beta[k][3:6]
            theta_dot = np.vstack([q, v])
        elif H_type[k] == "fixed":
            theta_dot = np.zeros((0, 1))
        Theta_dot.append(theta_dot)

    Theta_dot = np.vstack(Theta_dot)
    Beta_dot = np.vstack(gamma)

    S_dot = np.vstack((Theta_dot, Beta_dot)).flatten()

    return S_dot

def nBodyPos(t, X, klOO_B):
    # Takes time vector, t and X-vector [q, klOO]^T and returns hinge positions
    
    # Number of bodies and time steps
    n = len(X[0])
    nt = len(t)

    # Setup hinge position list
    penPos = []

    for i in range(nt):

        kpos = [None] * (n + 1)

        kpos[n] = np.zeros((3, 1)) 
        Ri = np.eye(3)
        
        for k in range(n - 1, -1, -1):
            # Unpancking X-vector
            q = X[i][k][0:4]    # Quaternion: k+1 to k
            klOO = X[i][k][4:7] # O_k to O_k-1^+

            # Rotation
            Ri = Ri @ sb.q2R(q.flatten(), 3)

            # Hinge position
            kpos[k] = kpos[k+1] + Ri @ klOO

        # Rotation of base body k = -1
        q = X[i][-1][0:4]
        R_base = sb.q2R(q.flatten(), 3)

        # This will account for "free" base body hinge   
        kpos[-1] = kpos[-2] - R_base @ klOO_B[-1]

        # Add to pendulum position list, penPos
        penPos.append(kpos)

    return penPos
            
def nBodySim(t, X, klOO, save_anim=False, filename="simulation.mp4"):
    # Takes X-vector list and returns simulation

    # Number of bodies
    n = len(X[0])

    # Number of time steps and dt
    nt = len(t)
    dt = t[1] - t[0]

    # Get position for each time step
    penPos = nBodyPos(t, X, klOO)

    # Initialize animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Determine Axis Limits
    all_points = []

    for i in range(0, nt, 10): # Sample every 10th frame for speed
        for body in penPos[i]:
            all_points.append(body.flatten()) # Flatten (3,1) to (3,)

    all_points = np.array(all_points)
    max_range = np.abs(all_points).max()
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, 0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"n-Body Pendulum with ({len(penPos[0]) - 1} Bodies)")

    # Create n colored lines (one per link)
    cmap = plt.cm.get_cmap('tab10', n)  # scalable colormap
    lines = []

    for i in range(n):
        line, = ax.plot([], [], [], '-', lw=4, markersize=4, color=cmap(i))
        lines.append(line)
    
    joint_dots, = ax.plot([], [], [], 'ko', markersize=4)

    ax.plot([0], [0], [0], 'o', color='gray', markersize=6)

    # Initialize the timer text (placed in top-left corner)
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    def update(frame_idx):
        current_state = penPos[frame_idx]

        # Extract joint positions
        xs = [float(body[0][0]) for body in current_state]
        ys = [float(body[1][0]) for body in current_state]
        zs = [float(body[2][0]) for body in current_state]

        # Update each link separately
        for i in range(n):
            lines[i].set_data(xs[i:i+2], ys[i:i+2])
            lines[i].set_3d_properties(zs[i:i+2])

        joint_dots.set_data(xs, ys)
        joint_dots.set_3d_properties(zs)

        # Update timer
        time_text.set_text(f'Time: {t[frame_idx]:.2f} s')

        ax.view_init(elev=20, azim=frame_idx * 0.1)
    
        return (*lines, joint_dots, time_text)

    # Create Animation
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(penPos), 
        interval=dt*1000, 
        blit=False
                        )

    if save_anim:
        print("Rendering animation to HTML... (This may take a minute)")
        
        # Use the 'html' writer
        save_dir = r"C:\Users\jepp6\OneDrive - Aarhus universitet\Dokumenter\Noter\6. Semester\Bachelor Projekt\BachelorCode\Renders"
        os.makedirs(save_dir, exist_ok=True)

        fullpath = os.path.join(save_dir, filename)

        with open(fullpath, "w") as f:
            f.write(anim.to_jshtml())
            
        print(f"Saved to {fullpath}")

    else:
        plt.show()

def initialize_n_bodies(n, theta0, beta0, H_type):
    # Initializes state and hinge for an n-body system.

    # Parameters:
    # theta0: List of n initial quaternions
    # beta0: List of n angular velocities quaternions
    # H_type: List of hinge types
    
    # Initial condition
    theta_stacked = np.vstack(theta0)
    beta_stacked = np.vstack(beta0)
        
    S0 = np.vstack((theta_stacked, beta_stacked))

    # Hinge maps
    H = [None] * n

    for i in range(n):
        H[i] = sb.hinge_map(H_type[i])
    return S0, H

def get_quat_from_degrees(x, y, z):
    # Takes angles, x, y and z and returns quaternion

    r = Rot.from_euler('xyz', [x, y, z], degrees=True)
    q = np.array(r.as_quat()).reshape(4, 1)

    return q

def nBodySOA(n, theta0, beta0, L, m, CkJk, H_type, tau, F_ext, tf, dt, SIM, filename):
    # Takes number of links, length, mass and inertia, n, L, m, Cklk 
    # for initial condition, S0 and returns animation of time t.
    
    klOO = [None] * n
    Mk = [None] * n

    S0, H = initialize_n_bodies(n, theta0, beta0, H_type)

    for i in range(n):

        # kl(O_k, O+_k-1): Hinge pos
        klOO[i] = np.array([[0],
                        [0],
                        [L[i]]])
        
        # kl(O_k, C_k): Centroid pos
        klOC = np.array([[0],
                        [0],
                        [L[i]/2]])
        
        # M(C): Spatial inertia centroid
        MC = np.block([
        [np.diag(CkJk[i]), np.zeros((3, 3))],
        [np.zeros((3, 3)), m[i]*np.eye(3)]
        ])

        # M(k): Spatial inertia hinge
        Mk[i] = sb.phi(klOC) @ MC @ sb.phi(klOC).T

    # Time control
    ts = 0
    t_span = np.linspace(ts, tf, int(tf/dt) + 1)
    nt = len(t_span)

    # ODE Solver
    sol = solve_ivp(
    fun=lambda t, y: EOM(y, n, klOO, Mk, H_type, H, tau, F_ext),
    t_span=(ts, tf),
    y0=S0.squeeze(),
    t_eval=t_span,
    method='RK45'
    )

    print("Integration using SOA was succesful!")

    # Extract results to match [t, y] format
    t_out = sol.t
    y_out = sol.y.T

    X_list = []

    # Find X-vector for each time step
    for i in range(len(t_out)):

        # Unpack state, s
        Theta, Beta = unpack_state(y_out[i], n, H_type)
        
        # Kinematic scatter loop to find X
        X, _, _, _ = scatter_kinematics(Theta, Beta, klOO, Mk, H)
        
        # Store X for this time step
        X_list.append(X)

    # Simulation of system
    if SIM == "yes" or SIM == "Yes":
        if filename != "":
            nBodySim(t_out, X_list, klOO, save_anim=True, filename = filename + ".html")
        else:
            nBodySim(t_out, X_list, klOO, save_anim=False, filename = "filename.html")
            print("Simulation was succesful")

    return t_out, y_out, X_list

#################### Manual setup ####################
n = 4                         # Number of bodies
L = np.array([6, 4, 2, 4])    # Length of bodies
m = np.array([2, 1, 1, 1])    # Mass of bodies
H_type = ["revx", "revy", "fixed", "spherical"]

# Second moment of inertia of bodies
CkJk1 = np.array([1, 1, 0.1])
CkJk2 = np.array([1, 1, 0.1])
CkJk3 = np.array([1, 1, 0.1])
CkJk4 = np.array([1, 1, 0.1])
CkJk = [CkJk1, CkJk2, CkJk3, CkJk4]

# Initial condition: Rotation: get_quat_from_degrees(x, y, z)
theta01 = np.deg2rad(30)
theta02 = np.deg2rad(75)
theta03 = np.zeros((0, 1))
theta04 = get_quat_from_degrees(135, 0, 0)

theta0 = [theta01, theta02, theta03, theta04]

# Initial condition: Angular velocity: get_quat_from_degrees(x, y, z)
beta01 = np.array([0]).reshape(1, 1)
beta02 = np.array([0]).reshape(1, 1)
beta03 = np.zeros((0, 1))
beta04 = np.array([0, 0, 0]).reshape(3, 1)

beta0 = [beta01, beta02, beta03, beta04]

# Generalized forces
tau1 = np.array([0]).reshape(1, 1)
tau2 = np.array([0]).reshape(1, 1)
tau3 = np.zeros((0, 1))
tau4 = np.array([0, 0, 0]).reshape(3, 1)

tau = [tau1, tau2, tau3, tau4]

# External forces F_ext = [klOO; F]
klBO1 = np.array([0, 0, 6]).reshape(3, 1)
F1 = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
F_ext1 = np.vstack((klBO1, F1))

klBO2 = np.array([0, 0, 6]).reshape(3, 1)
F2 = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
F_ext2 = np.vstack((klBO2, F2))

klBO3 = np.array([0, 0, 6]).reshape(3, 1)
F3 = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
F_ext3 = np.vstack((klBO3, F3))

klBO4 = np.array([0, 0, 4]).reshape(3, 1)
F4 = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
F_ext4 = np.vstack((klBO4, F4))

F_ext = [F_ext1, F_ext2, F_ext3, F_ext4]

# Time
tf = 10
dt = 0.01

# Run program: nBodySOA(n, L, m, CkJk, S0, tau, tf, dt, SIM, filename)
nBodySOA(n, theta0, beta0, L, m, CkJk, H_type, tau, F_ext, tf, dt, "Yes", "123")