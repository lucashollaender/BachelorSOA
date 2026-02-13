import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as Rot
from SOALIB import soalib as sb
from scipy.integrate import solve_ivp

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
        with open(filename, "w") as f:
            f.write(anim.to_jshtml())
            
        print(f"Saved to {filename}")

    else:
        plt.show()

class Joint:
    def __init__(self, joint_type: str):
        self.type = joint_type
        self.H = sb.hinge_map(joint_type)

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
    
class RigidBody:
# Rigid body class
    def __init__(self, L, m, CkJk, klOO, klOC, joint: Joint):
        self.length = L
        self.mass = m
        self.klOO = klOO
        self.klOC = klOC
        self.joint = joint

        # Spatial inertia
        MC = np.block([
            [np.diag(CkJk), np.zeros((3, 3))],
            [np.zeros((3, 3)), m * np.eye(3)]
        ])
        self.Mk = sb.phi(klOC) @ MC @ sb.phi(klOC).T

class SystemState:
    def __init__(self, thetas, betas):
        self.Theta = thetas
        self.Beta = betas

    def pack(self):
        return np.vstack([*self.Theta, *self.Beta]).flatten()

    @staticmethod
    def unpack(S, joints):
        S = S.flatten()
        Theta, Beta = [], []
        idx = 0

        for j in joints:
            sz = j.theta_size()
            Theta.append(S[idx:idx+sz].reshape(-1, 1))
            idx += sz

        for j in joints:
            sz = j.beta_size()
            Beta.append(S[idx:idx+sz].reshape(-1, 1))
            idx += sz

        return SystemState(Theta, Beta)

class ATBIEngine:
    def __init__(self, bodies):
        self.bodies = bodies
        self.n = len(bodies)

    def coriolis(self, V, beta, H):
        deltaV = H.T @ beta
        return sb.skew6(V) @ deltaV - sb.bar6(deltaV) @ deltaV

    def gyroscopic(self, V, M):
        return sb.bar6(V) @ M @ V
    
    def _theta_to_X(self, theta, joint_type, klOO):
        if joint_type == "revx":
            ang = theta.item()
            q = np.array([[np.sin(ang/2)], [0], [0], [np.cos(ang/2)]])
            return np.vstack((q, klOO))

        elif joint_type == "revy":
            ang = theta.item()
            q = np.array([[0], [np.sin(ang/2)], [0], [np.cos(ang/2)]])
            return np.vstack((q, klOO))

        elif joint_type == "revz":
            ang = theta.item()
            q = np.array([[0], [0], [np.sin(ang/2)], [np.cos(ang/2)]])
            return np.vstack((q, klOO))
         
        elif joint_type == "spherical":
            q = theta.reshape(4, 1) / np.linalg.norm(q)
            return np.vstack((q, klOO))

        elif joint_type == "free":
            q = theta[0:4].reshape(4, 1) / np.linalg.norm(q)
            v = theta[4:7].reshape(3, 1)
            return np.vstack((q, v))

        elif joint_type == "fixed":
            q = np.array([[0],[0],[0],[1]])
            return np.vstack((q, klOO))

    def scatter_kinematics(self, state: SystemState):
        X, V, a, b = [None]*self.n, [None]*self.n, [None]*self.n, [None]*self.n

        for k in reversed(range(self.n)):
            body = self.bodies[k]
            theta = state.Theta[k]
            beta = state.Beta[k]
            H = body.joint.H[k]

            # Build X (same logic you already have)
            X[k] = self._theta_to_X(theta, body.joint.type, body.klOO)
            q = X[k][0:4]
            

            if k == self.n - 1:
                V[k] = H @ beta
            else:
                R6 = sb.q2R(q.flatten(), 6)
                V[k] = (
                    R6.T
                    @ sb.phi(self.bodies[k+1].klOO).T
                    @ V[k+1]
                    + body.joint.H.T @ beta
                )

            a[k] = self.coriolis(V[k], beta, body.joint.H)
            b[k] = self.gyroscopic(V[k], body.Mk)

    

        return X, V, a, b
    
class MultibodySystem:
    def __init__(self, bodies, tau, F_ext):
        self.bodies = bodies
        self.tau = tau
        self.F_ext = F_ext
        self.engine = ATBIEngine(bodies)

    def eom(self, t, S):
        state = SystemState.unpack(S, [b.joint for b in self.bodies])
        X, V, a, b = self.engine.scatter_kinematics(state)
        G, nu = self.engine.gather_ATBI(a, b, X, self.tau, self.F_ext)
        gamma = self.engine.scatter_ATBI(a, X, G, nu)

        Theta_dot = []
        for k, body in enumerate(self.bodies):
            if body.joint.type.startswith("rev"):
                Theta_dot.append(state.Beta[k])
            elif body.joint.type == "spherical":
                Theta_dot.append(sb.quat_derivative(state.Theta[k], state.Beta[k]))
            elif body.joint.type == "free":
                qdot = sb.quat_derivative(state.Theta[k][:4], state.Beta[k][:3])
                Theta_dot.append(np.vstack([qdot, state.Beta[k][3:6]]))
            else:
                Theta_dot.append(np.zeros((0,1)))

        Sdot = np.vstack([*Theta_dot, *gamma]).flatten()
        return Sdot

class Simulator:
    def __init__(self, system: MultibodySystem, S0, tf, dt):
        self.system = system
        self.S0 = S0
        self.tf = tf
        self.dt = dt

    def run(self):
        t_eval = np.linspace(0, self.tf, int(self.tf/self.dt)+1)

        sol = solve_ivp(
            fun=self.system.eom,
            t_span=(0, self.tf),
            y0=self.S0,
            t_eval=t_eval
        )

        return sol.t, sol.y.T

class Visualizer:
    def __init__(self, bodies):
        self.bodies = bodies

    def animate(self, t, X_list):
        nBodySim(t, X_list, [b.klOO for b in self.bodies])

bodies = []
for i in range(n):
    joint = Joint(H_type[i])
    body = RigidBody(L[i], m[i], CkJk[i], joint)
    bodies.append(body)

system = MultibodySystem(bodies, tau, F_ext)

state0 = SystemState(theta0, beta0).pack()
sim = Simulator(system, state0, tf, dt)

t, y = sim.run()

