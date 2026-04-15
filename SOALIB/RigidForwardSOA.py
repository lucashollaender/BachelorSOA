import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as Rot
from SOALIB import soalib as sb
from scipy.integrate import solve_ivp
import copy
import os
import matplotlib as mpl

# Increase limit to 100 MB (default is 20)
plt.rcParams['animation.embed_limit'] = 1000

class Joint:
# Joint class with H_type, H and klOO
    def __init__(self, klOO, H_type: str):
        # Parameters
        self.type = H_type
        self.H = sb.hinge_map(H_type)
        self.klOO = klOO.reshape(3, 1)

    # Unpacking size
    def theta_size(self):
        return {
            "revx": 1,
            "revy": 1,
            "revz": 1,
            "spherical": 4,
            "free": 7,
            "fixed": 0
        }[self.type]

    def beta_size(self):
        return {
            "revx": 1,
            "revy": 1,
            "revz": 1,
            "spherical": 3,
            "free": 6,
            "fixed": 0
        }[self.type]

class Inertia:
# Inertia class with m, CkJk and klOC
    def __init__(self, m, CkJk, klOC):
        # Parameters
        self.m = m
        self.CkJk = CkJk
        self.klOC = klOC.reshape(3, 1)

        # Spatial inertia
        MC = np.block([
            [np.diag(CkJk), np.zeros((3, 3))],
            [np.zeros((3, 3)), m * np.eye(3)]
        ])
        self.Mk = sb.phi(klOC) @ MC @ sb.phi(klOC).T
        self.MC = MC

class SOABody:
# SOAbody class
    class Force:
        def __init__(self, joint: Joint):
            self.tau = np.zeros((joint.beta_size(), 1))
            self.sum_phi_F_ext = np.zeros((6, 1))
        
    class InitialCondition:
        def __init__(self, joint: Joint):
            # Setup of initial conditions (assumes identity rotation and no initial velocity)
            self.theta0 = np.zeros((joint.theta_size(), 1))
            if np.size(self.theta0) == 4:
                self.theta0[-1] = 1
            elif np.size(self.theta0) == 7:
                self.theta0[3] = 1
            self.beta0 = np.zeros((joint.beta_size(), 1))
    
    def __init__(self, joint: Joint, inertia: Inertia):
        self.joint = joint
        self.inertia = inertia
        self.force = self.Force(self.joint)
        self.initialcondition = self.InitialCondition(self.joint)
    
    def set_tau(self, tau):
        self.force.tau = tau
    
    def set_F_ext(self, F_ext, klBO):
        F = np.zeros((6, 1))
        for i in range(len(F_ext)):
            F = F + sb.phi(klBO[i]) @ F_ext[i]
        self.force.sum_phi_F_ext = F

    def set_initial_theta0(self, theta0):
        self.initialcondition.theta0 = theta0

    def set_initial_beta0(self, beta0):
        self.initialcondition.beta0 = beta0

class SystemState:
# State of system class
    def __init__(self, theta, beta):
        # Parameters
        self.Theta = theta
        self.Beta = beta

    # Packing of state, S: Two lists to column vector
    def pack(self):
        return np.vstack([*self.Theta, *self.Beta]).flatten()

    # Unpacking of state, S: Column vector to two lists
    @staticmethod
    def unpack(S, joints):
        S = S.flatten()
        Theta, Beta = [], []
        idx = 0

        for k in joints:
            sz = k.theta_size()
            Theta.append(S[idx:idx + sz].reshape(sz, 1))
            idx += sz

        for k in joints:
            sz = k.beta_size()
            Beta.append(S[idx:idx + sz].reshape(sz, 1))
            idx += sz

        return SystemState(Theta, Beta)

class ATBI:
# ATBI class with bodies
    def __init__(self, bodies):
        # Parameters
        self.bodies = bodies
        self.n = len(bodies)

    def coriolis(self, V, beta, H):
        deltaV = H.T @ beta
        return sb.skew6(V) @ deltaV - sb.bar6(deltaV) @ deltaV

    def gyroscopic(self, V, M):
        return sb.bar6(V) @ M @ V
    
    def theta2X(self, theta, joint_type, klOO):
        if joint_type == "revx":
            ang = theta.item()
            q = np.array([[np.sin(ang/2)], [0], [0], [np.cos(ang/2)]])
            return np.vstack((q, klOO)), q

        elif joint_type == "revy":
            ang = theta.item()
            q = np.array([[0], [np.sin(ang/2)], [0], [np.cos(ang/2)]])
            return np.vstack((q, klOO)), q

        elif joint_type == "revz":
            ang = theta.item()
            q = np.array([[0], [0], [np.sin(ang/2)], [np.cos(ang/2)]])
            return np.vstack((q, klOO)), q
         
        elif joint_type == "spherical":
            q = theta.reshape(4, 1) 
            return np.vstack((q, klOO)), q

        elif joint_type == "free":
            q = theta[0:4].reshape(4, 1) 
            v = theta[4:7].reshape(3, 1)
            return np.vstack((q, klOO)), q

        elif joint_type == "fixed":
            q = np.array([[0],[0],[0],[1]])
            return np.vstack((q, klOO)), q

    def scatter_kinematics(self, state: SystemState):
        
        # Number of bodies
        n = len(self.bodies)

        # Set up lists
        X = [None] * n
        V = [None] * n
        a = [None] * n
        b = [None] * n

        for k in reversed(range(self.n)):
            # Parameters of the body
            body = self.bodies[k]
            theta = state.Theta[k]
            beta = state.Beta[k]
            H = body.joint.H
            Mk = body.inertia.Mk

            # Build X
            X[k], q = self.theta2X(theta, body.joint.type, body.joint.klOO)
            
            if k == self.n - 1:
                V[k] = H.T @ beta
            else:
                R6 = sb.q2R(q.flatten(), 6)
                V[k] = R6.T @ sb.phi(X[k+1][4:7]).T @ V[k+1] + H.T @ beta

            a[k] = self.coriolis(V[k], beta, H)
            b[k] = self.gyroscopic(V[k], Mk)    

        return X, V, a, b
        
    def gather_ATBI(self, a, b, X):
        # Step 3 of ATBI (gather sweep): Takes generalized forces, Coriolis-, gyroscopic
        # terms, X-vector and system configuration and returns G and nu parameters

        # Number of bodies
        n = len(self.bodies)

        # Setup lists
        P_plus = [None] * n
        xi_plus = [None] * n
        G = [None] * n
        nu = [None] * n

        for k in range(n):
            # Parameters of the body
            body = self.bodies[k]
            H = body.joint.H
            Mk = body.inertia.Mk
            sum_phi_F_ext = body.force.sum_phi_F_ext
            tau = body.force.tau

            if k == 0:
                # Gather loop for k = 0 (Base Case)
                P = Mk
                D = H @ P @ H.T
                G[k] = np.linalg.solve(D.T, (P @ H.T).T).T
                tau_bar = np.eye(6) - G[k] @ H
                P_plus[k] = tau_bar @ P
                xi = P @ a[k] + b[k] - sum_phi_F_ext
                epsilon = tau - H @ xi
                nu[k] = np.linalg.solve(D, epsilon)
                xi_plus[k] = xi + G[k] @ epsilon
                
            else:
                # Unpacking X-vector
                q = X[k-1][0:4]
                klOO = X[k][4:7]

                # Rotation
                R6 = sb.q2R(q.flatten(), 6)
                
                # Gather loop for k > 0
                P = sb.phi(klOO) @ R6 @ P_plus[k-1] @ R6.T @ sb.phi(klOO).T + Mk
                D = H @ P @ H.T
                G[k] = np.linalg.solve(D.T, (P @ H.T).T).T
                tau_bar = np.eye(6) - G[k] @ H
                P_plus[k] = tau_bar @ P
                xi = sb.phi(klOO) @ R6 @ xi_plus[k-1] + P @ a[k] + b[k] - sum_phi_F_ext
                epsilon = tau - H @ xi
                nu[k] = np.linalg.solve(D, epsilon)
                xi_plus[k] = xi + G[k] @ epsilon

        return G, nu

    def scatter_ATBI(self, a, X, G, nu):
        # Step 4 of ATBI (second scatter sweep): Takes Coriolis term, X-vector, G,
        # nu and hinge map, H and returns generalized acceleration, gamma

        # Number of bodies
        n = len(self.bodies)

        # Setup of list
        alpha = [None] * n
        alpha_plus = [None] * n
        gamma = [None] * n

        # Spatial gravity
        g = np.array([0, 0, 0, 0, 0, 9.81]).reshape(6, 1)

        # Loop backwards from n-1 down to 0
        for k in range(n - 1, -1, -1):
            # Parameters of the body
            body = self.bodies[k]
            H = body.joint.H

            # Unpacking rotation
            q = X[k][0:4]

            # Rotation
            R6 = sb.q2R(q.flatten(), 6)

            if k == n - 1:
                # Scatter loop (Base of the chain)
                alpha_plus[k] = R6.T @ g
                gamma[k] = nu[k] - G[k].T @ alpha_plus[k]
                alpha[k] = alpha_plus[k] + H.T @ gamma[k] + a[k]
            
            else:
                # Hinge vector
                klOO = X[k+1][4:7]

                # Scatter loop
                alpha_plus[k] = R6.T @ sb.phi(klOO).T @ alpha[k+1]
                gamma[k] = nu[k] - G[k].T @ alpha_plus[k]
                alpha[k] = alpha_plus[k] + H.T @ gamma[k] + a[k]

        return gamma, alpha

class MultibodySystem:
    def __init__(self, bodies):
        self.bodies = bodies
        self.ATBI = ATBI(bodies)
        Theta_0 = [b.initialcondition.theta0 for b in bodies]
        Beta_0 = [b.initialcondition.beta0 for b in bodies]
        self.S0 = SystemState(Theta_0, Beta_0).pack()
        
    def EOM(self, t, S):
            state = SystemState.unpack(S.reshape(-1, 1), [b.joint for b in self.bodies])

            # Normalize quaternions
            for k, body in enumerate(self.bodies):
                if body.joint.type in ["spherical", "free"]:
                    q = state.Theta[k][0:4]
                    state.Theta[k][0:4] = q / np.linalg.norm(q)

            X, V, a, b = self.ATBI.scatter_kinematics(state)
            G, nu = self.ATBI.gather_ATBI(a, b, X)
            gamma, alpha = self.ATBI.scatter_ATBI(a, X, G, nu)

            Theta_dot = []
            for k, body in enumerate(self.bodies):
                if body.joint.type.startswith("rev"):
                    Theta_dot.append(state.Beta[k].reshape(1, 1))
                elif body.joint.type == "spherical":
                    Theta_dot.append(sb.quat_derivative(state.Theta[k], state.Beta[k]).reshape(4, 1))
                elif body.joint.type == "free":
                    qdot = sb.quat_derivative(state.Theta[k][0:4], state.Beta[k][0:3]).reshape(4, 1)
                    Theta_dot.append(np.vstack([qdot, state.Beta[k][3:6]]).reshape(7, 1))
                elif body.joint.type == "fixed":
                    Theta_dot.append(np.zeros((0,1)))

            S_dot = np.vstack([*Theta_dot, *gamma]).flatten()
            return S_dot

class Simulation:
    class Data:
        def __init__(self):
            self.time, self.state, self.X_list, self.V_list, self.a_list, self.b_list, self.alpha_list, self.pos = [], [], [], [], [], [], [], []

    class Setting:
        def __init__(self):
            self.camera_speed = 0
            self.camera_ver = 20
            self.camera_hor = 0

    def __init__(self, system: MultibodySystem, tf, dt):
        self.system = system
        self.data = self.Data()
        self.setting = self.Setting()
        self.tf = tf
        self.dt = dt

    def IntegrateSystem(self):
        t_eval = np.linspace(0, self.tf, int(self.tf/self.dt)+1)

        sol = solve_ivp(
            fun=self.system.EOM,
            t_span=(0, self.tf),
            y0=self.system.S0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10
        )

        print("Integration successful!")
    
        # Extract results to match [t, y] format
        self.data.time = sol.t
        states = sol.y.T

        # Find X-vector for each time step
        for i in range(len(self.data.time)):

            # Unpack state            
            current_state = SystemState.unpack(states[i].reshape(-1, 1), [b.joint for b in self.system.bodies])
            
            # Kinematic scatter loop to find X
            X, V, a, b = self.system.ATBI.scatter_kinematics(current_state)
            G, nu = self.system.ATBI.gather_ATBI(a, b, X)
            gamma, alpha = self.system.ATBI.scatter_ATBI(a, X, G, nu)

            # Add to list
            self.data.state.append(current_state)
            self.data.X_list.append(X)
            self.data.V_list.append(V)
            self.data.a_list.append(a)
            self.data.b_list.append(b)
            self.data.alpha_list.append(alpha)

    # Call functions for data
    def get_state(self):
        return self.data.state
    def get_X(self):
        return self.data.X_list
    def get_V(self):
        return self.data.V_list
    def get_a(self):
        return self.data.a_list
    def get_b(self):
        return self.data.b_list
    def get_alpha(self):
        return self.data.alpha_list

    # Settings    
    def camera_speed(self, x):
        self.setting.camera_speed = x

    def camera_ver(self, x):
        self.setting.camera_ver = x

    def camera_hor(self, x):
        self.setting.camera_hor = x

    def nBodyPos(self):
        # Takes time vector, t and X-vector [q, klOO]^T and returns hinge positions            

        t = self.data.time
        X = self.data.X_list
        klOO_B = [b.joint.klOO for b in self.system.bodies]

        # Number of bodies and time steps
        n = len(X[0])
        nt = len(t)

        # Setup hinge position list
        penPos = []

        dxyz = np.zeros((3, 1))

        for i in range(nt):
            # Account for possible free BASE hinge
            if  self.system.bodies[-1].joint.type == "free":
                theta_base_free = self.data.state[i].Theta[-1]
                dxyz = theta_base_free[4:7]

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
            kpos = kpos - dxyz
            penPos.append(kpos)

        return penPos

    def get_pos(self):
        self.data.pos = self.nBodyPos()
        return self.data.pos

    def animate(self, filename="", save_dir =""):
        # Takes X-vector list and returns simulation
        
        t = self.data.time
        X = self.data.X_list

        # Number of bodies
        n = len(X[0])

        # Number of time steps and dt
        nt = len(t)
        dt = t[1] - t[0]

        # Get position for each time step
        penPos = self.nBodyPos()

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
        cmap = mpl.colormaps['tab10']
        colors = cmap(np.linspace(0, 1, n))        
        lines = []

        for i in range(n):
            line, = ax.plot([], [], [], '-', lw=4, markersize=4, color=colors[i])
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

            ax.view_init(elev=self.setting.camera_ver, azim=frame_idx * self.setting.camera_speed * 40 * dt + self.setting.camera_hor)
        
            return (*lines, joint_dots, time_text)
    
        # Create Animation
        anim = FuncAnimation(
        fig, 
        update, 
        frames=len(penPos), 
        interval=dt*1000, 
        blit=False)

        if filename != "":            
            print("Rendering animation to HTML... (This may take a minute)")
            
            filename = filename + ".html"

            # Use the 'html' writer
            os.makedirs(save_dir, exist_ok=True)

            fullpath = os.path.join(save_dir, filename)

            with open(fullpath, "w") as f:
                f.write(anim.to_jshtml())
            
            print("Renedering of animation: Done!")
            print(f"Saved to {fullpath}")

        else:
            plt.show()

""" ------ File Setup ------ """
# Remember to run this:
# from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation

""" ------ Body Setup ------ """
# *** Body Parameters ****
# klOO:     Hinge position (row vector)
# H_type:   Hinge type (string)
# m:        Mass (scalar)
# CkJk:     Inertia (row vector)
# klOC:     COM position (row vector)

# *** Create Body ***
# joint = Joint(<klOO>, <H_type>)
# inertia = Inertia(<m>, <CkJk>, <klOC>)
# body = SOABody(<joint>, <inertia>)

""" ------ Body Attributes ------ """ 
# If not specified program assumes zero column vectors
# theta0, beta0, tau, F_ext ---> column vectors

# *** Initial condition ***
# body.set_initial_theta0(<theta0>)   //   <theta0> ---> column vector
#       --->   "revx/y/z" use: theta0 = np.deg2rad(theta_x/y/z)
#       --->   "spherical" use: theta0 = q0 = sb.get_quat_from_degrees(theta_x, theta_y, theta_z)
#       --->   "free" use: theta0 = np.vstack([q0, l]), where l is the initial linear displacement (l = [l_x, l_y, l_z])
#       --->   "fixed" use: theta0 cannot be specified
# body.set_initial_beta0(<beta0>)   //   <beta0> ---> column vector 
#       --->   "revx/y/z" use: beta0 = omega_x/y/z
#       --->   "spherical" use: beta0 = np.array([omega_x, omega_y, omega_z]).reshape(3, 1)
#       --->   "free" use: beta0 = np.array([omega_x, omega_y, omega_z, v_x, v_y, v_z]), where v is the initial linear velocity (v = [v_x, v_y, v_z])
#       --->   "fixed" use: beta0 cannot be specified

# *** Forces ***
# body.set_tau(<tau>)   //   <tau> ---> column vector, np.array([<tau>]).reshape(nDOF, 1)

# body.set_F_ext(<F_ext>, <klBO>)   //   <F_ext>, <klBO> ---> lists of same length 
#       --->   F_ext is a list of column vectors (6, 1) with external forces
#       --->   klBO is a list of row vectors (1, 3) with the external forces' appliying position

""" ------ System Setup and Simulation ------ """ 
# *** Multibody System ***
# system = MultibodySystem(bodies)
#       --->   bodies = [body_1, body_2, ..., body_n], list of bodies created above (tip: b_1 and base: b_n)

# *** Simulation Setup ***
# sim = Simulation(system, tf, dt)
#       --->   system as created above
#       --->   tf, length of simulation
#       --->   dt, time step size

""" ------ Camera Settings ------ """ 
# sim.camera_speed(x)
#       --->   x, number from -1..1 defining the speed in both directions (zero if not changed)

# sim.camera_hor(x)
#       --->   x, number from 0..360 defining the camera rotation around z-axis (zero if not changed)

# sim.camera_ver(x)
#       --->   x, number from -90..90 defining the camera rotation around x-axis (20 if not changed)

""" ------ Parameter Call ------ """
# *** Parameter Call ***   //   Get parameter for each body for each time step
# sim.get_<parameter>()
#       --->   <parameter> = [state, X, pos, V, alpha, a, b]

""" ------ Animate or Render Animation ------ """
# *** Animate ***
# sim.animate()

# *** Render Animation to HTML-file ***   // <file_name>, r<file_path> ---> strings
# sim.animate(<file_name>, <file_path>)
#       --->   <file_name>, name of the render file
#       --->   <file_path>, copy the file path of the folder you want to save the HTML-file in.
#               nb! You must add the letter <r> in front of file path string as:
#               file_path = r"C:\Users\jepp6\OneDrive..."
#               Choose another folder than the GIT-Hub synchronize folder, since the file will
#               be to big and result in a "commit" error.