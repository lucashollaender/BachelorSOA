import numpy as np
from SOALIB import soalib as sb
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
import pandas as pd
from scipy.spatial.transform import Rotation as R, Slerp
import n_body_pend as nbp
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import time

# -----------------------------
# Settings you actually tweak
# -----------------------------
n = 4

# Physical params
link_vec = np.array([0.0, 0.0, -5.0])
com_vec = np.array([0.0, 0.0, -2.5])
J_diag = np.array([1.0, 1.0, 0.1])
mass = 1.0

# Simulation time (physics)
t0, tf = 0.0, 20.0
base_angle_deg = 60
step_deg = 20

# Playback controls
real_time_seconds = 20.0   # make the whole animation last ~10s on screen
target_frames = 600        # ~60 fps for 10s (lower = faster compile)

# ODE solver tolerances (looser = faster)
rtol = 1e-6
atol = 1e-8

# -----------------------------
# Build system + initial state
# -----------------------------
sys = nbp.build_system_data(n, link_vec, com_vec, J_diag, mass)

S0 = nbp.initial_condition(n, base_angle_deg, step_deg, axis=(0, 1, 0))
print(S0[:4*n].reshape(n, 4))

# Choose only as many output points as you need for the animation
t_eval = np.linspace(t0, tf, target_frames)

sol = solve_ivp(
    fun=lambda t, S: nbp.odefun(t, S, sys, n),
    t_span=(t0, tf),
    y0=S0,
    t_eval=t_eval,
    rtol=rtol,
    atol=atol,
    method="RK45",
)

Y = sol.y.T
Nt = Y.shape[0]

# --- Precompute joint positions (exactly Nt frames) ---
p_hist = np.zeros((Nt, n + 1, 3))
for i in range(Nt):
    theta = Y[i, :4*n].reshape(n, 4)

    P = nbp.forward_kinematics_points(theta, sys, transpose_rel=True)

    # Anchor base at origin
    P = P - P[-1]
    # Base->tip ordering on screen
    P = P[::-1]

    p_hist[i] = P

# --- Plot setup ---
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
(line,) = ax.plot([], [], [], "o-", lw=2)

mins = p_hist.reshape(-1, 3).min(axis=0)
maxs = p_hist.reshape(-1, 3).max(axis=0)
center = (mins + maxs) / 2
span = (maxs - mins).max() * 0.6 + 1e-9

ax.set_xlim(center[0] - span, center[0] + span)
ax.set_ylim(center[1] - span, center[1] + span)
ax.set_zlim(center[2] - span, center[2] + span)
ax.set_box_aspect([1, 1, 1])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


def update(frame):
    P = p_hist[frame]
    line.set_data(P[:, 0], P[:, 1])
    line.set_3d_properties(P[:, 2])
    return (line,)


# Make the whole animation last ~10 seconds real time
interval_ms = max(1, int(1000 * real_time_seconds / Nt))

ani = FuncAnimation(
    fig, update,
    frames=Nt,
    interval=interval_ms,
    blit=False,
    repeat=True
)

plt.show()
