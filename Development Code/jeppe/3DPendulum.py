import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as Rot
from SOALIB import soalib as sb
from scipy.integrate import solve_ivp

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
        
        # Extracting current values
        theta = Theta[k]
        beta = Beta[k]

        # Relative orientation with quaternion and X-vector setup        
        q = theta.reshape(4, 1)
        X[k] = np.vstack((q, klOO[k]))

        if k == n - 1:
            # Scatter loop (Tip of the chain)
            V[k] = H.T @ beta
        else:
            # Rotation matrix
            R6 = sb.q2R(q.flatten(), 6)

            # Scatter loop
            V[k] = R6.T @ sb.phi(klOO[k+1]).T @ V[k+1] + H.T @ beta

        # Coriolis term
        a[k] = get_coriolis(V[k], beta, H)

        # Gyroscopic term
        b[k] = get_gyroscopic_force(V[k], Mk[k])
    
    return X, V, a, b

def gather_ATBI(a, b, X, Mk, H):
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

        if k == 0:
            # Gather loop for k = 0 (Base Case)
            P = Mk[k]
            D = H @ P @ H.T
            G[k] = (P @ H.T) @ np.linalg.inv(D)
            tau_bar = np.eye(6) - G[k] @ H
            P_plus[k] = tau_bar @ P
            xi = P @ a[k] + b[k]
            epsilon = - H @ xi
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
            D = H @ P @ H.T
            G[k] = (P @ H.T) @ np.linalg.inv(D)
            tau_bar = np.eye(6) - G[k] @ H
            P_plus[k] = tau_bar @ P
            xi = sb.phi(klOO) @ R6 @ xi_plus[k-1] + P @ a[k] + b[k]
            epsilon = - H @ xi
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
            alpha[k] = H.T @ gamma[k] + a[k]
        
        else:
            # Hinge vector
            klOO = X[k+1][4:7]
        
            # Scatter loop
            alpha_plus[k] = R6.T @ sb.phi(klOO).T @ alpha[k+1]
            nu_bar = nu[k] - (G[k].T @ Ri[k].T @ g)
            gamma[k] = nu_bar - G[k].T @ alpha_plus[k]
            alpha[k] = alpha_plus[k] + H.T @ gamma[k] + a[k]

    return gamma

def ATBI(Theta, Beta, klOO, Mk, H):
    # Takes theta, beta, klOO, Mk and H and returns gamma
    
    # 1 and 2) Scatter loop to find X=[q, klOO], velocity, V, corriolis acceleration, a and gyroscopic force, b
    X, V, a, b = scatter_kinematics(Theta, Beta, klOO, Mk, H)

    # 3) Gather loop to find G and nu
    G, nu = gather_ATBI(a, b, X, Mk, H)

    # 4) Scatter loop to find gamma
    gamma = scatter_ATBI(a, X, G, nu, H)

    return gamma

def unpack_state(S, n):
    # Takes the state vector S and returns lists of Theta and Beta
    
    # Ensure S is flat
    S = S.flatten() 
    
    theta_vec = S[0:4*n]
    beta_vec = S[4*n:7*n]

    # Change to list with n arrays
    Theta = [theta_vec[i*4 : (i+1)*4].reshape(4, 1) for i in range(n)]
    Beta = [beta_vec[i*3 : (i+1)*3].reshape(3, 1) for i in range(n)]
    
    return Theta, Beta

def EOM(S, n, klOO, Mk, H):
    # Takes state, S, number of links, n, hinge pos, klOO,
    # spatial in hinge inertia, Mk and hinge map, H and returns 
    # generalized acceleration, alpha

    # Unpacking of state, S
    Theta, Beta = unpack_state(S, n)

    # Normalizing quaternions
    for i in range(n):
        Theta[i] = Theta[i] / np.linalg.norm(Theta[i])

    # Setup derivative of state vector
    Theta_dot = np.zeros((4*n, 1))
    Beta_dot = np.zeros((3*n, 1))

    # Find gamma
    gamma = ATBI(Theta, Beta, klOO, Mk, H)

    for i in range(n):
        Theta_dot[i*4 : (i+1)*4] = sb.quat_derivative(Theta[i], Beta[i]).reshape((4, 1))
        Beta_dot[i*3 : (i+1)*3] = gamma[i]

    S_dot = np.vstack((Theta_dot, Beta_dot)).flatten()

    return S_dot

def nBodyPendumlumPos(t, X):
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

        # Add to pendulum position list, penPos
        penPos.append(kpos)

    return penPos
            
def nPendulumSim(t, X, save_anim=False, filename="simulation.mp4"):
    # Takes X-vector list and returns simulation

    # Number of bodies
    n = len(X[0])

    # Number of time steps and dt
    nt = len(t)
    dt = t[1] - t[0]

    # Get position for each time step
    penPos = nBodyPendumlumPos(t, X)

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

    # Initialize the Line Object
    line, = ax.plot([], [], [], 'o-', lw=2, markersize=4)
    
    # Initialize the timer text (placed in top-left corner)
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    # Line update function
    def update(frame_idx):
        current_state = penPos[frame_idx]
        # Unpack the column vectors:
        xs = [float(body[0][0]) for body in current_state]
        ys = [float(body[1][0]) for body in current_state]
        zs = [float(body[2][0]) for body in current_state]

        line.set_data(xs, ys)
        line.set_3d_properties(zs)

        # Update timer text
        time_text.set_text(f'Time: {t[frame_idx]:.2f} s')

        return line, time_text

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
        with open(filename, "w") as f:
            f.write(anim.to_jshtml())
            
        print(f"Saved to {filename}")

    else:
        plt.show()

def nBodyPen(n, L, m, CkJk, S0, tf, dt, SIM, filename):
    # Takes number of links, length, mass and inertia, n, L, m, Cklk 
    # for initial condition, S0 and returns animation of time t.
    
    klOO = [None] * n
    Mk = [None] * n

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

    # H(k): Hinge map
    H = sb.hinge_map("spherical")

    # Time control
    ts = 0
    t_span = np.linspace(ts, tf, int(tf/dt) + 1)
    nt = len(t_span)

    # ODE Solver
    sol = solve_ivp(
    fun=lambda t, y: EOM(y, n, klOO, Mk, H),
    t_span=(ts, tf),
    y0=S0.squeeze(),
    t_eval=t_span,
    method='RK45'
    )

    # Extract results to match [t, y] format
    t_out = sol.t
    y_out = sol.y.T

    X_list = []

    # Find X-vector for each time step
    for i in range(len(t_out)):

        # Unpack state, s
        Theta, Beta = unpack_state(y_out[i], n)
        
        # Kinematic scatter loop to find X
        X, _, _, _ = scatter_kinematics(Theta, Beta, klOO, Mk, H)
        
        # Store X for this time step
        X_list.append(X)
    
    # Simulation of system
    if SIM == "yes" or SIM == "Yes":
        if filename != "":
            nPendulumSim(t_out, X_list, save_anim=True, filename = filename + ".html")
        else:
            nPendulumSim(t_out, X_list, save_anim=False, filename = "filename.html")

    return t_out, y_out, X_list

def initialize_n_bodies(n, theta0, beta0, L, CkJk, m, type):
    # Initializes state and properties for an n-body system.

    # Parameters:
    # n: Number of bodies.
    # theta0: List of n initial quaternions
    # beta0: List of n angular velocities quaternions
    # L: Length of the bodies (same for all)
    # CkJk: Moment of inertia [Ixx, Iyy, Izz] (same for all)
    # m: Mass of the bodies (same for all)
    
    if type == "auto":
        # Auto setup

        # Create array of masses and lengths
        m = np.full(n, m)
        L = np.full(n, L)
        
        # Create list of inertia arrays
        inertia_array = np.array(CkJk)
        CkJk = [inertia_array for _ in range(n)]

        theta_stacked = np.vstack(theta0[0:n])
        beta_stacked = np.zeros((3*n, 1))
        
    elif type == "manual":
        # Manual setup
        theta_stacked = np.vstack(theta0)
        beta_stacked = np.vstack(beta0)    
        
    S0 = np.vstack((theta_stacked, beta_stacked))

    return S0, L, CkJk, m, n

def get_quat_from_degrees(x, y, z):
    # Takes angles, x, y and z and returns quaternion

    r = Rot.from_euler('xyz', [x, y, z], degrees=True)
    q = np.array(r.as_quat()).reshape(4, 1)

    return q

#################### Manual setup ####################
n_m = 2                   # Number of bodies
L_m = np.array([5, 5])    # Length of bodies
m_m = np.array([1, 1])    # Mass of bodies

CkJk1 = np.array([1, 1, 0.1])
CkJk2 = np.array([1, 1, 0.1])
CkJk_m = [CkJk1, CkJk2]   # Second moment of inertia of bodies

# Initial condition: Rotation and angular velocity
theta01 = np.array([np.sin(np.pi/12), 0, 0, np.cos(np.pi/12)]).reshape(4, 1)
theta02 = np.array([0, 0, 0, 1]).reshape(4, 1)
beta01 = np.zeros((3, 1))
beta02 = np.zeros((3, 1))

theta0_m = [theta01, theta02]
beta0_m = [beta01, beta02]

# initialize_n_bodies(n, theta0, beta0 (only for "manual"),  L, CkJk, m):
# S0, L, CkJk, m, n = initialize_n_bodies(n_m, theta0_m, beta0_m, L_m, CkJk_m, m_m, "manual")

#################### Auto setup #################### (For large number of bodies):
n_a = 2            # Number of bodies
L_a = 5            # Length of bodies
m_a = 1            # Mass of bodies
CkJk_a = np.array([1, 1, 0.1])    # Second moment of inertia of bodies

# Initial condition: Rotation: get_quat_from_degrees(x, y, z)
theta01 = get_quat_from_degrees(20, 0, 0)
theta02 = get_quat_from_degrees(-135, 0, 0)
theta03 = get_quat_from_degrees(90, -10, 0)
theta04 = get_quat_from_degrees(-45, 90, 0)

theta0_a = [theta01, theta02, theta03, theta04]

# initialize_n_bodies(n, theta0, beta0 (only for "manual"),  L, CkJk, m):
S0, L, CkJk, m, n = initialize_n_bodies(n_a, theta0_a, 0, L_a, CkJk_a, m_a, "auto")

# Time
tf = 10
dt = 0.01

# Specify: 
SIM = "Yes" # Want simulation?
filename = "" # Want to render .mp4? ("") gives no render.

# Run program
nBodyPen(n, L, m, CkJk, S0, tf, dt, SIM, filename)