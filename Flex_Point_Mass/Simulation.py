import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import os
from MultibodySystem import MultibodySystem
from SystemState import SystemState
from SOALIB import soalib as sb
import pandas as pd


class Simulation:
    class Data:
        def __init__(self):
            self.time, self.state, self.X_list, self.V_list, self.a_list, self.b_list, self.alpha_list, self.pos = [
            ], [], [], [], [], [], [], []

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
            current_state = SystemState.unpack(
                states[i].reshape(-1, 1), [b.joint for b in self.system.bodies], [b.flex for b in self.system.bodies])

            # Kinematic scatter loop to find X
            X, V, a_fl, b_fl = self.system.ATBI.scatter_kinematics(
                current_state)
            G_pr, nu_pr, nu_m, g_fl = self.system.ATBI.gather_ATBI(
                current_state, a_fl, b_fl, X, self.data.time[i])
            _, _, alpha = self.system.ATBI.scatter_ATBI(
                a_fl, X, G_pr, nu_pr, nu_m, g_fl)

            # Add to list
            self.data.state.append(current_state)
            self.data.X_list.append(X)
            self.data.V_list.append(V)
            self.data.a_list.append(a_fl)
            self.data.b_list.append(b_fl)
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

    def nNodalPos(self):
        t = self.data.time
        X = self.data.X_list

        # Access body 1 (index 0) parameters
        body = self.system.bodies[0]
        n_nd = body.flex.n_nd
        PI = body.flex.PI
        L = body.rigid.L
        L_elem = L / (n_nd - 1)

        nt = len(t)
        nodal_pos = []

        for i in range(nt):
            state = self.data.state[i]
            eta = state.Eta[0]

            # Base translation (account for free joint if applicable)
            dxyz = np.zeros((3, 1))
            if body.joint.type == "free":
                dxyz = state.Theta[0][4:7].reshape(3, 1)

            nodes_at_t = []

            for j in range(n_nd):
                # 1. Undeformed position in local frame (Structural analysis assumes beam along Z-axis)
                p_und = np.array([[j * L_elem], [0], [0]])

                # 2. Translational deformation for node j
                # (Translations are stored at indices j*6+3 to j*6+6 in the PI matrix)
                # Define a small tolerance based on your dt
                tolerance = 1e-6

                # Check if the current time is very close to an integer
                x = 0
                if i % 20 == 0 and j == n_nd-1 and x == 1:
                    print(pd.DataFrame(eta))
                    print(pd.DataFrame(PI[j*6+3 : j*6+6, :]))
                
                u_j = PI[j*6+3: j*6+6, :] @ eta

                # 3. Global position of node j
                p_glob = dxyz + (p_und + u_j)
                nodes_at_t.append(p_glob)

            nodal_pos.append(nodes_at_t)

        return nodal_pos

    def animate_nodes(self, filename="", save_dir=""):
        # Takes nodal position list and returns 3D simulation of the flexible beam

        t = self.data.time
        dt = t[1] - t[0]

        # Get nodal positions for the flexible beam
        nodal_pos = self.nNodalPos()

        if not nodal_pos:
            print("Error: No nodal position data found. Did you run the integration?")
            return

        nt = len(nodal_pos)
        n_nd = len(nodal_pos[0])

        # Initialize animation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Determine Axis Limits dynamically based on node movement
        all_points = []
        # Sample frames to speed up boundary calculation
        step = max(1, nt // 50)
        for i in range(0, nt, step):
            for node in nodal_pos[i]:
                all_points.append(node.flatten())

        all_points = np.array(all_points)
        max_range = np.abs(all_points).max()

        # Prevent zero-range if the beam doesn't move
        if max_range == 0:
            max_range = 1.0

        ax.set_xlim(-2, 7)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Flexible Beam Animation ({n_nd} Nodes)")

        # Create a single line representing the continuous flexible beam
        beam_line, = ax.plot([], [], [], '-', lw=4,
                             color='blue', marker='o', markersize=4)

        # Plot origin for reference
        ax.plot([0], [0], [0], 'o', color='gray', markersize=6)

        # Initialize the timer text
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

        def update(frame_idx):
            current_nodes = nodal_pos[frame_idx]

            # Extract x, y, z coordinates for all nodes at the current frame
            xs = [float(node[0][0]) for node in current_nodes]
            ys = [float(node[1][0]) for node in current_nodes]
            zs = [float(node[2][0]) for node in current_nodes]

            # Update the beam line data
            beam_line.set_data(xs, ys)
            beam_line.set_3d_properties(zs)

            # Update timer
            time_text.set_text(f'Time: {t[frame_idx]:.2f} s')

            # Handle camera rotation settings
            ax.view_init(elev=self.setting.camera_ver,
                         azim=frame_idx * self.setting.camera_speed * 40 * dt + self.setting.camera_hor)

            return beam_line, time_text

        # Create Animation
        anim = FuncAnimation(
            fig,
            update,
            frames=nt,
            interval=dt*1000,
            blit=False)

        if filename != "":
            print("Rendering animation to HTML... (This may take a minute)")
            filename = filename + ".html"

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fullpath = os.path.join(save_dir, filename)
            else:
                fullpath = filename

            with open(fullpath, "w") as f:
                f.write(anim.to_jshtml())

            print("Rendering of animation: Done!")
            print(f"Saved to {fullpath}")

        else:
            plt.show()

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
            if self.system.bodies[-1].joint.type == "free":
                theta_base_free = self.data.state[i].Theta[-1]
                dxyz = theta_base_free[4:7]

            kpos = [None] * (n + 1)

            kpos[n] = np.zeros((3, 1))
            Ri = np.eye(3)

            for k in range(n - 1, -1, -1):
                # Unpancking X-vector
                q = X[i][k][0:4]    # Quaternion: k+1 to k
                klOO = X[i][k][4:7]  # O_k to O_k-1^+

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

    def animate(self, filename="", save_dir=""):
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

        for i in range(0, nt, 10):  # Sample every 10th frame for speed
            for body in penPos[i]:
                all_points.append(body.flatten())  # Flatten (3,1) to (3,)

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
            line, = ax.plot([], [], [], '-', lw=4,
                            markersize=4, color=colors[i])
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

            ax.view_init(elev=self.setting.camera_ver, azim=frame_idx *
                         self.setting.camera_speed * 40 * dt + self.setting.camera_hor)

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
