import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R, Slerp
import n_body_pend as nbp
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import time

# Defining data
n = 4
link_vec = np.array([0.0, 0.0, -5.0])
com_vec = np.array([0.0, 0.0, -2.5])
J_diag = np.array([1, 1, 0.1])
mass = 1.0

sys = nbp.build_system_data(n, link_vec, com_vec, J_diag, mass)

S = nbp.initial_condition(n)

t0, tf = 0.0, 10.0
t_eval = np.linspace(t0, tf, 1001)

S0 = nbp.initial_condition(n, angle_deg=45, axis=(0, 1, 0))  # rotate about y

sol = solve_ivp(
    fun=lambda t, S: nbp.odefun(t, S, sys, n),
    t_span=(t0, tf),
    y0=S0,
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10,
)

Y = sol.y.T
Nt = Y.shape[0]

p_hist = np.zeros((Nt, n + 1, 3))

for i in range(Nt):
    theta = Y[i, :4*n].reshape(n, 4)
    p_hist[i] = nbp.joint_positions(theta, sys)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")

line, = ax.plot([], [], [], "o-", lw=2)

# axis limits (adjust if needed)
Ltot = np.sum([np.linalg.norm(l) for l in sys["L"]])
ax.set_xlim(-Ltot, Ltot)
ax.set_ylim(-Ltot, Ltot)
ax.set_zlim(-Ltot, 0.5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


def update(frame):
    P = p_hist[frame]
    line.set_data(P[:, 0], P[:, 1])
    line.set_3d_properties(P[:, 2])
    return line,


ani = FuncAnimation(fig, update, frames=Nt, interval=30)

plt.show()
