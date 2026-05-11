from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
from SOALIB import soalib as sb
import pandas as pd
import matplotlib.pyplot as plt
import os

ADAMS_HINGE1_FILE = "Hinge1_ang_acc_alu.tab"
ADAMS_HINGE2_FILE = "Hinge2_ang_acc_alu.tab"


def read_adams_tab(filename):
    start_idx = None

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                float(parts[0])
                float(parts[1])
                start_idx = i
                break
            except ValueError:
                pass

    if start_idx is None:
        raise ValueError(f"Could not find numeric data in {filename}")

    df = pd.read_csv(
        filename,
        sep=r"\s+",
        skiprows=start_idx,
        header=None,
        names=["TIME", "Q"],
        engine="python"
    )

    return df


klOO1 = np.array([1, 0, 0]).reshape(3, 1)
klOO2 = np.array([1, 0, 0]).reshape(3, 1)

H_type1 = "revy"
H_type2 = "revy"

# n_md_max = (n_nd - 1) * 3
# steel
# E, G, c, rho, n_nd, n_md = 230e9, 80e9, 0.02, 7801, 10, 6

# Alu
E, G, c, rho, n_nd, n_md = 7.17e10, 2.7e10, 0.02, 2740, 10, 6

w, h = 0.04, 0.04

j1 = Joint(klOO1, H_type1)
r1 = Rigid_Properties(rho, w, h)
f1 = Flex_Properties(E, G, c, n_nd, n_md)

j2 = Joint(klOO2, H_type2)
r2 = Rigid_Properties(rho, w, h)
f2 = Flex_Properties(E, G, c, n_nd, n_md)

b1 = SOABody(j1, r1, f1)
b2 = SOABody(j2, r2, f2)

bodies = [b1, b2]

system = MultibodySystem(bodies)

tf = 10
dt = 0.01

sim = Simulation(system, tf, dt)
sim.IntegrateSystem("Radau")


# =================================================
# Extract flexible SOA hinge angular accelerations
# =================================================
# For the flexible implementation:
# scatter_kinematics -> X, V, a_fl, b_fl
# gather_ATBI        -> G_pr, nu_pr, nu_m, g_fl
# scatter_ATBI       -> theta_ddot, eta_ddot, alpha_fl
#
# theta_ddot[k] is the generalized hinge acceleration.
# For a revolute joint this is the angular acceleration
# about the joint axis, in rad/s^2.

t_flex = sim.data.time
states = sim.get_state()

theta_ddot_body1 = []
theta_ddot_body2 = []

for i, state in enumerate(states):
    t_i = t_flex[i]

    X, V, a_fl, b_fl = system.ATBI.scatter_kinematics(state)

    G_pr, nu_pr, nu_m, g_fl = system.ATBI.gather_ATBI(
        state, a_fl, b_fl, X, t_i
    )

    theta_ddot, eta_ddot, alpha_fl = system.ATBI.scatter_ATBI(
        a_fl, X, G_pr, nu_pr, nu_m, g_fl
    )

    theta_ddot_body1.append(float(theta_ddot[0].flatten()[0]))
    theta_ddot_body2.append(float(theta_ddot[1].flatten()[0]))

theta_ddot_body1 = np.array(theta_ddot_body1)
theta_ddot_body2 = np.array(theta_ddot_body2)


# =================================================
# Read Adams data
# =================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

adams1_path = os.path.join(script_dir, ADAMS_HINGE1_FILE)
adams2_path = os.path.join(script_dir, ADAMS_HINGE2_FILE)

adams1 = read_adams_tab(adams1_path)
adams2 = read_adams_tab(adams2_path)

# =================================================
# Interpolate Adams onto SOA time grid
# =================================================

adams1_q = np.interp(t_flex, adams1["TIME"], adams1["Q"])
adams2_q = np.interp(t_flex, adams2["TIME"], adams2["Q"])

# =================================================
# Cut off Adams/SOA startup transient
# =================================================

t_start_plot = 0.02  # Adams has some weird values in the beginning - we dont want these
mask = t_flex >= t_start_plot

t_plot = t_flex[mask]

theta_ddot_body1_plot = theta_ddot_body1[mask]
theta_ddot_body2_plot = theta_ddot_body2[mask]

adams1_q_plot = adams1_q[mask]
adams2_q_plot = adams2_q[mask]

err1 = theta_ddot_body1_plot - adams1_q_plot
err2 = theta_ddot_body2_plot - adams2_q_plot


# =================================================
# Plot SOA vs Adams and errors
# =================================================

fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

axs[0].plot(t_plot, theta_ddot_body1_plot, label="Flex SOA")
axs[0].plot(t_plot, adams1_q_plot, "--", label="Adams")
axs[0].set_ylabel(r"$\ddot{\theta}_1$ [rad/s$^2$]")
axs[0].set_title("Hinge 1 Angular Acceleration")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(t_plot, err1)
axs[1].set_ylabel("SOA - Adams")
axs[1].set_title("Hinge 1 Error")
axs[1].grid(True)

axs[2].plot(t_plot, theta_ddot_body2_plot, label="Flex SOA")
axs[2].plot(t_plot, adams2_q_plot, "--", label="Adams")
axs[2].set_ylabel(r"$\ddot{\theta}_2$ [rad/s$^2$]")
axs[2].set_title("Hinge 2 Angular Acceleration")
axs[2].grid(True)
axs[2].legend()

axs[3].plot(t_plot, err2)
axs[3].set_xlabel("Time [s]")
axs[3].set_ylabel("SOA - Adams")
axs[3].set_title("Hinge 2 Error")
axs[3].grid(True)

plt.tight_layout()
plt.show()
