import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
import os
from MultibodySystem import MultibodySystem
from SystemState import SystemState
from SOALIB import soalib as sb
import pandas as pd
from tqdm import tqdm


class Simulation:
    class Data:
        def __init__(self):
            self.time, self.state, self.X_list, self.V_fl_list, self.a_fl_list, self.b_fl_list, self.alpha_fl_list, self.pos = [
            ], [], [], [], [], [], [], []

    class Setting:
        def __init__(self):
            self.camera_speed = 0
            self.camera_ver = 20
            self.camera_hor = 0
            self.solver = "RK4"

    def __init__(self, system: MultibodySystem, tf, dt):
        self.system = system
        self.data = self.Data()
        self.setting = self.Setting()
        self.tf = tf
        self.dt = dt
        self.setting.ani_dt = dt

        # Increase limit to 100 MB (default is 20)
        plt.rcParams['animation.embed_limit'] = 1000

    def IntegrateSystem(self, solver="RK4"):
        self.setting.solver = solver
        # print("Integrating...")

        # Progress bar
        pbar = tqdm(total=100, desc=f"Integration ({solver})", unit="%")
        original_EOM = self.system.EOM

        last_percent = -1

        def tracked_EOM(t, S):
            nonlocal last_percent

            percent = int((t / self.tf) * 100 - 1)

            if percent > last_percent:
                pbar.update(percent - last_percent)
                last_percent = percent

            return original_EOM(t, S)

        self.system.EOM = tracked_EOM

        if self.setting.solver == "RK4":

            Y, t = sb.integrate_RK4(self.system, 0, self.tf, self.dt)

            self.data.time = t
            states = Y.T

        else:

            t_eval = np.linspace(0, self.tf, int(self.tf/self.dt)+1)

            sol = solve_ivp(
                fun=self.system.EOM,
                t_span=(0, self.tf),
                y0=self.system.S0,
                t_eval=t_eval,
                method=self.setting.solver,
                atol=1e-8,
                rtol=1e-10
            )

            # Checking if integration actually succeeds
            if not sol.success:
                self.system.EOM = original_EOM
                pbar.close()
                raise RuntimeError(f"Integration failed: {sol.message}")

            self.data.time = sol.t
            states = sol.y.T

        self.system.EOM = original_EOM
        pbar.close()
        print("Integration successful!")

        # Find X-vector for each time step
        dt0 = self.setting.ani_dt

        if dt0 >= self.dt:
            scale = int(dt0 / self.dt)
            nt = int(len(self.data.time) / scale)
            self.data.time = np.linspace(0, self.data.time[-1], nt)
        else:
            print("Error! Invalid animation time step! (ani_dt < sim_dt)")

        for i in range(len(self.data.time)):
            j = i * scale
            # Unpack state
            current_state = SystemState.unpack(
                states[j].reshape(-1, 1), [b.joint for b in self.system.bodies], [b.flex for b in self.system.bodies])

            # Kinematic scatter loop to find X
            X, V_fl, a_fl, b_fl = self.system.ATBI.scatter_kinematics(
                current_state)
            G_pr, nu_pr, nu_m, g_fl = self.system.ATBI.gather_ATBI(
                current_state, a_fl, b_fl, X, self.data.time[i])
            _, _, alpha_fl = self.system.ATBI.scatter_ATBI(
                a_fl, X, G_pr, nu_pr, nu_m, g_fl)

            # Add to list
            self.data.state.append(current_state)
            self.data.X_list.append(X)
            self.data.V_fl_list.append(V_fl)
            self.data.a_fl_list.append(a_fl)
            self.data.b_fl_list.append(b_fl)
            self.data.alpha_fl_list.append(alpha_fl)

    # Call functions for data
    def get_state(self):
        return self.data.state

    def get_X(self):
        return self.data.X_list

    def get_V_fl(self):
        return self.data.V_fl_list

    def get_a_fl(self):
        return self.data.a_fl_list

    def get_b_fl(self):
        return self.data.b_fl_list

    def get_alpha_fl(self):
        return self.data.alpha_fl_list

    # Settings
    def set_camera_speed(self, x):
        self.setting.camera_speed = x

    def set_camera_ver(self, x):
        self.setting.camera_ver = x

    def set_camera_hor(self, x):
        self.setting.camera_hor = x

    def set_ani_dt(self, x):
        self.setting.ani_dt = x

    def nNodalPos(self):
        t = self.data.time
        X = self.data.X_list
        n = len(self.system.bodies)

        nt = len(t)
        nodal_pos = []

        for i in range(nt):
            # Current state
            state = self.data.state[i]

            # End node (O_-1+)
            last_end = np.zeros((3, 1))

            R_i = np.eye(3)
            R_n = np.eye(3)

            nodes_i = []

            for k in range(n - 1, -1, -1):
                # Current body parameters
                eta = state.Eta[k]
                body = self.system.bodies[k]
                n_nd = body.flex.n_nd
                PI = body.flex.PI
                L_elem = body.flex.L_elem
                klOO_nd = body.flex.klOO_nd

                nodes_k = []

                # Unpancking X-vector
                q = X[i][k][0:4]    # Quaternion: k+1 to k

                # Rotation
                R_i = R_i @ sb.q2R(q.flatten(), 3)

                for j in range(n_nd):
                    # Undeformed position in local frame
                    pos_und = klOO_nd[j]

                    # Translational deformation for node j
                    # (Translations are stored at indices j*6+3 to j*6+6 in the PI matrix)
                    u_j = PI[j*6+3: j*6+6, :] @ eta

                    # Global position of node j (of body k)
                    p_glob = last_end + R_i @ (pos_und + u_j)
                    nodes_k.append(p_glob)

                # Rotation of last node
                R_n_vec = PI[-6:-3, :] @ eta
                R_n = Rotation.from_rotvec(R_n_vec.flatten()).as_matrix()

                R_i = R_i @ R_n

                last_end = nodes_k[-1]
                nodes_i.append(nodes_k)

            nodal_pos.append(nodes_i)

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
        n_bodies = len(nodal_pos[0])

        # Initialize animation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Determine Axis Limits dynamically based on node movement
        all_points = []
        # Sample frames to speed up boundary calculation
        step = max(1, nt // 50)
        for i in range(0, nt, step):
            for body_nodes in nodal_pos[i]:
                for node in body_nodes:
                    all_points.append(node.flatten())

        all_points = np.array(all_points)
        if len(all_points) > 0:
            max_range = np.abs(all_points).max()
        else:
            max_range = 1.0

        # Prevent zero-range if the beam doesn't move
        if max_range == 0:
            max_range = 1.0

        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Flexible n-Body Animation ({n_bodies} Bodies)")

        # Create n colored lines (one per flexible link)
        cmap = mpl.colormaps['tab10']
        colors = cmap(np.linspace(0, 1, n_bodies))
        lines = []

        for i in range(n_bodies):
            line, = ax.plot([], [], [], '-', lw=4, color=colors[i])
            lines.append(line)

        # Scatter plot for the nodes
        node_dots, = ax.plot([], [], [], 'ko', markersize=4)

        # Plot origin for reference
        ax.plot([0], [0], [0], 'o', color='gray', markersize=6)

        # Initialize the timer text
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

        # Camera setting
        camera_initialized = False

        def update(frame_idx):
            nonlocal camera_initialized

            current_state = nodal_pos[frame_idx]

            all_xs, all_ys, all_zs = [], [], []

            # Update each body separately
            for b_idx in range(n_bodies):
                body_nodes = current_state[b_idx]

                xs = [float(node[0][0]) for node in body_nodes]
                ys = [float(node[1][0]) for node in body_nodes]
                zs = [float(node[2][0]) for node in body_nodes]

                lines[b_idx].set_data(xs, ys)
                lines[b_idx].set_3d_properties(zs)

                all_xs.extend(xs)
                all_ys.extend(ys)
                all_zs.extend(zs)

            node_dots.set_data(all_xs, all_ys)
            node_dots.set_3d_properties(all_zs)

            # Update timer
            time_text.set_text(f'Time: {t[frame_idx]:.2f} s')

            # Camera control
            if self.setting.camera_speed == 0 and frame_idx == 0 and camera_initialized == False:
                ax.view_init(elev=self.setting.camera_ver,
                             azim=self.setting.camera_hor)
                camera_initialized = True
            elif self.setting.camera_speed != 0:
                ax.view_init(elev=self.setting.camera_ver,
                             azim=self.setting.camera_hor + frame_idx * self.setting.camera_speed * 40 * dt)

            return (*lines, node_dots, time_text)

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

    def get_pos(self):
        self.data.pos = self.nBodyPos()
        return self.data.pos
