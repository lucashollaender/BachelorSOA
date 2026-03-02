import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import os
from MultibodySystem import MultibodySystem
from SystemState import SystemState
from SOALIB import soalib as sb

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
        
        # Increase limit to 100 MB (default is 20)
        plt.rcParams['animation.embed_limit'] = 1000

    def IntegrateSystem(self):
        t_eval = np.linspace(0, self.tf, int(self.tf/self.dt)+1)

        sol = solve_ivp(
            fun=self.system.EOM,
            t_span=(0, self.tf),
            y0=self.system.S0,
            t_eval=t_eval,
            method="RK45"
        )

        print("Integration successful!")
    
        # Extract results to match [t, y] format
        self.data.time = sol.t
        states = sol.y.T

        # Find X-vector for each time step
        for i in range(len(self.data.time)):

            # Unpack state            
            current_state = SystemState.unpack(states[i].reshape(-1, 1), [b.joint for b in self.system.bodies], [b.flex for b in self.system.bodies])
            
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