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
from matplotlib.lines import Line2D

# Increase limit to 100 MB (default is 20)
plt.rcParams['animation.embed_limit'] = 1000

class Simulation:
    class Data:
        def __init__(self):
            self.time, self.state, self.X_list, self.V_fl_list, self.a_fl_list, self.b_fl_list, self.alpha_fl_list, self.pos, self.R_i_list = [
            ], [], [], [], [], [], [], [], []

    class Setting:
        def __init__(self):
            self.camera_speed = 0
            self.camera_ver = 20
            self.camera_hor = 0
            self.solver = "RK4"
            self.atol = 1e-3
            self.rtol = 1e-6
            self.show_com_frames = False
            self.frame_scale = 0.5
            self.max_step = np.inf
            self.xlim = None
            self.ylim = None
            self.zlim = None

    def __init__(self, system: MultibodySystem, tf, dt):
        self.system = system
        self.data = self.Data()
        self.setting = self.Setting()
        self.tf = tf
        self.dt = dt
        self.setting.ani_dt = dt

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

        elif self.setting.solver == "BE":
            Y, t = sb.integrate_backward_euler(
                self.system,
                0,
                self.tf,
                self.dt,
                tol=1e-8,
                max_iter=20
            )

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
                atol=self.setting.atol,
                rtol=self.setting.rtol,
                max_step=self.setting.max_step
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
            X, R3_list, V_fl, a_fl, b_fl, pos, pos_dot, R_i = self.system.ATBI.scatter_kinematics(
                current_state)
            G_pr, nu_pr, nu_m, g_fl, P_pr_plus, z_pr_plus = self.system.ATBI.gather_ATBI(
                current_state, a_fl, b_fl, X, R3_list, pos, pos_dot, R_i, self.data.time[i])
            _, _, alpha_fl, F_int = self.system.ATBI.scatter_ATBI(
                a_fl, X, R3_list, G_pr, nu_pr, nu_m, g_fl, P_pr_plus, z_pr_plus)

            # Add to list
            self.data.state.append(current_state)
            self.data.X_list.append(X)
            self.data.V_fl_list.append(V_fl)
            self.data.a_fl_list.append(a_fl)
            self.data.b_fl_list.append(b_fl)
            self.data.alpha_fl_list.append(alpha_fl)
            self.data.pos.append(pos)
            self.data.R_i_list.append(R_i)

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
    
    def get_pos(self):
        return self.data.pos

    # Settings
    def set_xlim(self, xmin, xmax):
        self.setting.xlim = (xmin, xmax)

    def set_ylim(self, ymin, ymax):
        self.setting.ylim = (ymin, ymax)

    def set_zlim(self, zmin, zmax):
        self.setting.zlim = (zmin, zmax)
    
    def set_camera_speed(self, x):
        self.setting.camera_speed = x

    def set_camera_ver(self, x):
        self.setting.camera_ver = x

    def set_camera_hor(self, x):
        self.setting.camera_hor = x

    def set_ani_dt(self, x):
        self.setting.ani_dt = x

    def set_tol(self, atol, rtol):
        self.setting.atol = atol
        self.setting.rtol = rtol

    def set_max_step(self, max_step):
        self.setting.max_step = max_step

    # COM Coordinate Frames
    def show_COM_frames(self, scale="auto", auto_scale=0.5):
        self.setting.show_com_frames = True
        self.setting.frame_scale = scale

    def get_com_frame_data(self, frame_idx, auto_scale=None):
        state = self.data.state[frame_idx]
        R_i_list = self.data.R_i_list[frame_idx]
        pos_list = self.data.pos[frame_idx]     
        n = len(self.system.bodies)

        lines_data = []

        if self.setting.frame_scale == "auto" and auto_scale is not None:
            scale_x, scale_y, scale_z = auto_scale
        else:
            scale_x = scale_y = scale_z = self.setting.frame_scale

        scale_mat = np.diag([scale_x, scale_y, scale_z])

        for k in range(n - 1, -1, -1):
            body = self.system.bodies[k]
            eta = state.Eta[k]
            n_nd = body.flex.n_nd
            PI = body.flex.PI
            
            mid_idx = n_nd // 2

            p_glob = pos_list[k][mid_idx]

            R_mid_vec = PI[mid_idx*6: mid_idx*6+3, :] @ eta
            R_mid = Rotation.from_rotvec(R_mid_vec.flatten()).as_matrix()
            
            R_glob = R_i_list[k] @ R_mid

            x_end = p_glob + scale_mat @ R_glob @ np.array([[1.0], [0.0], [0.0]])
            y_end = p_glob + scale_mat @ R_glob @ np.array([[0.0], [1.0], [0.0]])
            z_end = p_glob + scale_mat @ R_glob @ np.array([[0.0], [0.0], [1.0]])

            lines_data.append((p_glob, x_end, y_end, z_end))

        return lines_data

    def animate_nodes(self, filename="", save_dir=""):
        # Takes nodal position list and returns 3D simulation of the flexible beam

        t = self.data.time
        dt = t[1] - t[0]

        # Get nodal positions for the flexible beam
        nodal_pos = self.data.pos

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

        # X Axis
        if self.setting.xlim is not None:
            ax.set_xlim(self.setting.xlim[0], self.setting.xlim[1])
        else:
            ax.set_xlim(-max_range, max_range)

        # Y Axis
        if self.setting.ylim is not None:
            ax.set_ylim(self.setting.ylim[0], self.setting.ylim[1])
        else:
            ax.set_ylim(-max_range, max_range)

        # Z Axis
        if self.setting.zlim is not None:
            ax.set_zlim(self.setting.zlim[0], self.setting.zlim[1])
        else:
            ax.set_zlim(-max_range, max_range)

        span_x = ax.get_xlim()[1] - ax.get_xlim()[0]
        span_y = ax.get_ylim()[1] - ax.get_ylim()[0]
        span_z = ax.get_zlim()[1] - ax.get_zlim()[0]
        
        calculated_auto_scale = (0.1 * span_x, 0.1 * span_y, 0.1 * span_z)

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

        # COM Coordinate Frames
        com_lines = []
        if self.setting.show_com_frames:
            for i in range(n_bodies):
                # RGB lines for X, Y, Z axes
                line_x, = ax.plot([], [], [], 'r-', lw=2)
                line_y, = ax.plot([], [], [], 'g-', lw=2)
                line_z, = ax.plot([], [], [], 'b-', lw=2)
                com_lines.extend([line_x, line_y, line_z])
            
            legend_elements = [
                Line2D([0], [0], color='r', lw=2, label='X-Axis'),
                Line2D([0], [0], color='g', lw=2, label='Y-Axis'),
                Line2D([0], [0], color='b', lw=2, label='Z-Axis')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

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

            # COM Coordinate Frames
            if self.setting.show_com_frames:
                # ---> THIS IS THE NEW LINE <---
                frame_data = self.get_com_frame_data(frame_idx, auto_scale=calculated_auto_scale)
                
                for b_idx, (p_glob, x_end, y_end, z_end) in enumerate(frame_data):
                    # X Axis (Red)
                    com_lines[b_idx*3].set_data([p_glob[0, 0], x_end[0, 0]], [p_glob[1, 0], x_end[1, 0]])
                    com_lines[b_idx*3].set_3d_properties([p_glob[2, 0], x_end[2, 0]])

                    # Y Axis (Green)
                    com_lines[b_idx*3+1].set_data([p_glob[0, 0], y_end[0, 0]], [p_glob[1, 0], y_end[1, 0]])
                    com_lines[b_idx*3+1].set_3d_properties([p_glob[2, 0], y_end[2, 0]])

                    # Z Axis (Blue)
                    com_lines[b_idx*3+2].set_data([p_glob[0, 0], z_end[0, 0]], [p_glob[1, 0], z_end[1, 0]])
                    com_lines[b_idx*3+2].set_3d_properties([p_glob[2, 0], z_end[2, 0]])

            # Camera control
            if self.setting.camera_speed == 0 and frame_idx == 0 and camera_initialized == False:
                ax.view_init(elev=self.setting.camera_ver,
                             azim=self.setting.camera_hor)
                camera_initialized = True
            elif self.setting.camera_speed != 0:
                ax.view_init(elev=self.setting.camera_ver,
                             azim=self.setting.camera_hor + frame_idx * self.setting.camera_speed * 40 * dt)

            return (*lines, node_dots, time_text, *com_lines)

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
