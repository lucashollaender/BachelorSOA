import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
import pandas as pd
from matplotlib.animation import FuncAnimation
import time


def skew(z):
    z = np.asarray(z).reshape(3,)
    return np.array([
        [0.0,    -z[2],  z[1]],
        [z[2],    0.0,  -z[0]],
        [-z[1],   z[0],  0.0]
    ])


def R6(R: np.ndarray):
    R = np.asarray(R)
    return np.block([
        [R,            np.zeros((3, 3))],
        [np.zeros((3, 3)), R]
    ])


def skew6(z):
    z = np.asarray(z).reshape(6,)
    omega = z[0:3]
    v = z[3:6]
    return np.block([
        [skew(omega),           np.zeros((3, 3))],
        [skew(v),           skew(omega)]
    ])


def bar6(V):
    V = np.asarray(V).reshape(6,)
    omega = V[0:3]
    v = V[3:6]
    return np.block([
        [skew(omega),           skew(v)],
        [np.zeros((3, 3)),   skew(omega)]
    ])


def phi(l):
    l = np.asarray(l).reshape(3,)
    return np.block([
        [np.eye(3),          skew(l)],
        [np.zeros((3, 3)),   np.eye(3)]
    ])


def quat_derivative(q, omega):
    """
    Quaternion derivative:
        qdot = 0.5 * [[-skew(omega), omega],
                      [-omega^T,        0]] @ q

    Assumes q = [qx, qy, qz, qw]^T (vector part first, scalar last).
    omega: shape (3,)
    q: shape (4,)
    returns qdot: shape (4,)
    """
    q = np.asarray(q).reshape(4,)
    omega = np.asarray(omega).reshape(3,)

    Omega = np.block([
        [-skew(omega),        omega.reshape(3, 1)],
        [-omega.reshape(1, 3), np.zeros((1, 1))]
    ])

    return 0.5 * (Omega @ q)


def quat_to_rotmat(q):
    """
    Convert quaternion to rotation matrix.
    Assumes q = [qx, qy, qz, qw]^T (vector part first, scalar last).
    q: shape (4,)
    returns R: shape (3, 3)
    """

    q = np.asarray(q).reshape(4,)
    quat = pq.Quaternion(q[3], q[0], q[1], q[2])  # pq uses scalar first
    return quat.rotation_matrix


def R_accumulated(R):
    """
    Computes the world-frame orientation of each link.
    R: list of local rotation matrices [R0, R1, R2, R3] 
       where R[k] is rotation relative to parent (k+1).
    Returns: list of accumulated matrices [R_acc0, R_acc1, R_acc2, R_acc3]
    """
    n = len(R)
    R_acc = [None] * n

    # Start from the base (last link)
    # The last link's world orientation is just its rotation relative to the fixed base
    current_R = np.eye(3)

    # Iterate backward from n-1 down to 0
    for k in range(n - 1, -1, -1):
        # Accumulate: R_world = R_parent_world @ R_local
        current_R = current_R @ R[k]
        R_acc[k] = current_R

    return R_acc


def scatter(R, L, H, beta, M):
    """
    SCATTER: Calculates spatial velocity and acceleration in a scatter loop.

    Inputs:
        R     : list of 3x3 rotation matrices
        L     : list of 3-vectors (link vectors)
        H     : list of hinge maps (shape 6x1, 6x3, etc.)
        beta  : list of joint velocities (scalars or vectors)
        M     : list of 6x6 spatial inertia matrices

    Outputs:
        V           : list of spatial velocities (6,)
        a_corolis   : list of Coriolis terms (6,)
        b_gyro      : list of gyroscopic terms (6,)
    """

    n = len(R)

    # Allocate lists (MATLAB cells → Python lists)
    V = [None] * (n + 1)
    a_corolis = [None] * n
    b_gyro = [None] * n

    # V{n+1} = zeros(6,1)
    V[n] = np.zeros(6)

    # Scatter loop: for k = n:-1:1
    for k in range(n - 1, -1, -1):

        # --- Velocity ---
        V[k] = (
            phi(R[k].T @ L[k]).T
            @ R6(R[k].T)
            @ V[k + 1]
            + H[k].T @ beta[k]
        )

        # --- Coriolis term ---
        a_corolis[k] = (
            skew6(V[k]) @ (H[k].T @ beta[k])
            - bar6(H[k].T @ beta[k]) @ (H[k].T @ beta[k])
        )

        # --- Gyroscopic term ---
        b_gyro[k] = bar6(V[k]) @ M[k] @ V[k]

    return V, a_corolis, b_gyro


def ATBI(R, beta, tau, H, M, L):
    """
    Articulated-Body / Two-Body Inverse dynamics-style algorithm (ATBI)

    Inputs:
        R    : list of 3x3 rotation matrices, length n
        beta : list of joint velocities (scalar or small vector), length n
        tau  : list of joint torques (scalar or small vector), length n
        H    : list of motion subspace matrices, length n
               (often shape (1,6) or (3,6) in MATLAB; i.e. maps spatial->joint)
        M    : list of 6x6 spatial inertia matrices, length n
        L    : list of 3-vectors (link vectors), length n

    Outputs:
        thetaddot : list of joint accelerations, length n
        V         : list of spatial velocities, length n+1 (V[n] is base/parent)
        alpha     : list of spatial accelerations, length n+1 (alpha[n] = 0)
    """

    # Spatial velocities + coriolis + gyro
    V, a_corolis, b_gyro = scatter(R, L, H, beta, M)

    n = len(R)

    # -------- Gather sweep init --------
    Pplus = [None] * n
    xi_plus = [None] * n
    G = [None] * n
    v = [None] * n
    vbar = [None] * n
    g = np.array([
        0.0, 0.0, 0.0,    # angular part
        0.0, 0.0, 9.81   # linear part
    ])

    Pplus[0] = np.zeros_like(M[0])          # 6x6
    xi_plus[0] = np.zeros_like(b_gyro[0])   # 6,

    # -------- Scatter sweep init --------
    alpha = [None] * (n + 1)
    alpha[n] = np.zeros(6)                  # alpha{n+1} = 0
    alphaplus = [None] * n
    thetaddot = [None] * n

    # -------- Gather loop: k = 1..n (MATLAB) -> k = 0..n-1 (Python) --------
    for k in range(n):
        if k == 0:
            Pk = M[k]
            xi = Pk @ a_corolis[k] + b_gyro[k]
        else:
            # MATLAB: phi(L{k-1})*R6(R{k-1})*Pplus{k-1}*R6(R{k-1}')*phi(L{k-1})'+M{k}
            X = phi(L[k - 1]) @ R6(R[k - 1])
            Pk = X @ Pplus[k - 1] @ X.T + M[k]

            # MATLAB: phi(L{k-1})*R6(R{k-1})*xi_plus{k-1}+Pk*a_corolis{k}+b_gyro{k}
            xi = X @ xi_plus[k - 1] + Pk @ a_corolis[k] + b_gyro[k]

        # Dk = H{k} * Pk * H{k}'
        Dk = H[k] @ Pk @ H[k].T

        G[k] = (Pk @ H[k].T) @ np.linalg.inv(Dk)

        # taubar = I - G*H
        taubar = np.eye(Pk.shape[0]) - G[k] @ H[k]

        # Pplus{k} = taubar * Pk
        Pplus[k] = taubar @ Pk

        # e = tau - H{k} * xi
        e = tau[k] - (H[k] @ xi)

        # v{k} = Dk \ e
        v[k] = np.linalg.solve(Dk, e)

        # xi_plus{k} = xi + G{k} * e
        xi_plus[k] = xi + G[k] @ e

    # -------- Scatter loop: k = n:-1:1 (MATLAB) -> k = n-1..0 (Python) --------
    R_acc = R_accumulated(R)  # The loop now "knows" orientations for all k
    for k in range(n - 1, -1, -1):
        alphaplus[k] = phi(R[k].T @ L[k]).T @ R6(R[k].T) @ alpha[k + 1]

        vbar[k] = v[k] - (G[k].T @ R6(R) @ g)

        thetaddot[k] = vbar[k] - (G[k].T @ alphaplus[k])

        alpha[k] = alphaplus[k] + (H[k].T @ thetaddot[k]) + a_corolis[k]

    return thetaddot, V, alpha


def odefun(t, S, SystemData):
    """
    Equivalent of MATLAB:
      dS = odefun(t, S, SystemData)

    S shape: (32,)
    Returns dS shape: (32,)
    """
    n = 4
    S = np.asarray(S)
    dS = np.zeros(28)

    # --- unpack state (MATLAB is 1-indexed; Python is 0-indexed) ---
    theta1 = S[0:4]          # (4,)  [qx,qy,qz,qw]
    theta2 = S[4:8]            # scalar
    theta3 = S[8:12]            # scalar
    theta4 = S[12:16]         # (4,)

    beta1 = S[16:19]         # (3,)
    beta2 = S[19:22]            # scalar
    beta3 = S[22:25]            # scalar
    beta4 = S[25:28]         # (3,)

    beta = [beta1, beta2, beta3, beta4]

    # --- torques (you said ignore generalized forces => set tau=0) ---
    # Match your MATLAB tau shapes:
    tau = [
        np.zeros(3),   # H1 (Spherical)
        np.zeros(3),   # H2 (Spherical)
        np.zeros(3),   # H3 (Spherical)
        np.zeros(3),   # H4 (Spherical)
    ]

    # --- rotations ---
    R = [None] * n

    # full quaternion"
    R[0] = quat_to_rotmat(theta1)
    R[1] = quat_to_rotmat(theta2)
    R[2] = quat_to_rotmat(theta3)
    R[3] = quat_to_rotmat(theta4)

    # --- System data ---
    H = SystemData["H"]
    M = SystemData["M"]
    L = list(SystemData["L"])     # copy so we can modify

    # --- generalized accelerations ---
    thetaddot, _, _ = ATBI(R, beta, tau, H, M, L)

    # --- assemble dS ---
    dS[0:4] = quat_derivative(theta1, beta1)          # qdot for H1
    dS[4:8] = quat_derivative(theta2, beta2)        # qdot for H2
    dS[8:12] = quat_derivative(theta3, beta3)        # qdot for H3
    dS[12:16] = quat_derivative(theta4, beta4)        # qdot for H4
    dS[16:19] = thetaddot[0]                            # (3,)
    dS[19:22] = thetaddot[1]                            # scalar
    dS[22:25] = thetaddot[2]                            # (3,)
    dS[25:28] = thetaddot[3]                            # (3,)
    return dS


def build_system_data():
    # -------- H hinge maps (MATLAB cell -> Python list) --------
    H = [
        np.hstack([np.eye(3), np.zeros((3, 3))]),          # 3x6
        np.hstack([np.eye(3), np.zeros((3, 3))]),          # 3x6
        np.hstack([np.eye(3), np.zeros((3, 3))]),          # 3x6
        np.hstack([np.eye(3), np.zeros((3, 3))]),          # 3x6
    ]

    # -------- Link/hinge positions (MATLAB cell -> list of 3-vectors) --------
    # L_OtoO: "hinge position I have to add p" -> keep L[4] as placeholder (will overwrite with p at runtime)
    L_OtoO = [
        np.array([0.0, 0.0, -5.0]),
        np.array([0.0, 0.0, -5.0]),
        np.array([0.0, 0.0, -5.0]),
        np.array([0.0, 0.0, -5.0]),
    ]

    # Center of mass offsets from hinge frame
    L_HtoC = [
        np.array([0.0, 0.0, -2.5]),
        np.array([0.0, 0.0, -2.5]),
        np.array([0.0, 0.0, -2.5]),
        np.array([0.0, 0.0, -2.5]),
    ]

    # -------- Inertia tensors J (MATLAB cell -> Python list) --------
    J = [None] * 4
    J[0] = np.diag(np.array([3.542, 3.542, 0.4167]))
    J[1] = np.diag(np.array([3.542, 3.542, 0.4167]))
    J[2] = np.diag(np.array([3.542, 3.542, 0.4167]))
    J[3] = np.diag(np.array([3.542, 3.542, 0.4167]))
    # Masses
    m = np.array([10.0, 10.0, 10.0, 10.0])

    # -------- Spatial inertia M (list of 6x6) --------
    M = [None] * 4
    for i in range(4):
        r = L_HtoC[i]
        rx = skew(r)
        M[i] = np.block([
            [J[i] - m[i] * (rx @ rx),   m[i] * rx],
            [-m[i] * rx,                m[i] * np.eye(3)]
        ])

    # "struct" equivalent
    SystemData = {
        "H": H,
        "L": L_OtoO,
        "M": M,
    }

    return SystemData


# --- initial conditions ---
theta_initial = np.array([
    # theta1 quaternion [qx qy qz qw]
    np.sin(np.pi/12), 0.0, 0.0, np.cos(np.pi/12),
    0.0, 0.0, 0.0, 1.0,          # theta2 quaternion
    0.0, 0.0, 0.0, 1.0,          # theta3 quaternion
    0.0, 0.0, 0.0, 1.0,          # theta4 quaternion
], dtype=float)

beta_initial = np.zeros(12, dtype=float)  # 12 generalized velocities

S0 = np.concatenate([theta_initial, beta_initial])  # length 28

# --- integrate (ode45 ~ RK45) ---
t_span = (0.0, 20.0)  # time span

SystemData = build_system_data()

sol = sp.integrate.solve_ivp(
    fun=lambda t, S: odefun(t, S, SystemData),
    t_span=t_span,
    y0=S0,
    method="RK45",     # closest to MATLAB ode45
    rtol=1e-4,
    atol=1e-7,
    dense_output=True  # like ode45 giving you interpolation
)

# If you want t and S sampled like MATLAB output arrays:
t = sol.t                        # time points chosen by solver
S = sol.y.T                      # shape (len(t), 28)
print(sol.success, sol.message)
print("t shape:", t.shape, "S shape:", S.shape)


def forward_kinematics_points(quats_xyzw, link_vecs):
    """
    quats_xyzw: list of 4 quaternions (each [x,y,z,w]) for one timestep
    link_vecs: list of 4 link vectors (3,) in each hinge's local frame
    returns points array shape (5,3): base + 4 joints/end
    """
    pts = [np.zeros(3)]
    R = np.eye(3)

    for i in range(4):
        R = R @ quat_to_rotmat(quats_xyzw[i])     # accumulate orientation
        # step along current local link
        pts.append(pts[-1] + R @ link_vecs[i])
    return np.vstack(pts)


def animate_chain(S, t=None, stride=1):
    """
    S shape: (T, 28) but quaternions at:
      hinge1: cols 0:4
      hinge2: cols 4:8
      hinge3: cols 8:12
      hinge4: cols 12:16

    t: array of time points (optional). If provided, animation runs at real-time speed.
    stride: animate every 'stride' frames (e.g. 2,5,10 for faster)
    """

    # ---- Define link vectors (EDIT THESE) ----
    # 4 links, each in its own hinge local frame.
    # Example: all links length 5 along local +z:
    link_length = -5.0
    link_vecs = [
        np.array([0.0, 0.0, link_length]),
        np.array([0.0, 0.0, link_length]),
        np.array([0.0, 0.0, link_length]),
        np.array([0.0, 0.0, link_length]),
    ]

    # Use only needed columns (0:16 are quats; rest ignored for geometry here)
    S = np.asarray(S)
    T = S.shape[0]

    # Precompute points for speed (optional but makes animation smooth)
    frames = range(0, T, stride)
    all_pts = []
    all_times = []
    for k in frames:
        q1 = S[k, 0:4]
        q2 = S[k, 4:8]
        q3 = S[k, 8:12]
        q4 = S[k, 12:16]
        pts = forward_kinematics_points([q1, q2, q3, q4], link_vecs)
        all_pts.append(pts)
        all_times.append(t[k] if t is not None else k)
    all_pts = np.array(all_pts)  # shape (F, 5, 3)
    all_times = np.array(all_times)  # shape (F,)

    # Determine plot limits from trajectory
    mins = all_pts.reshape(-1, 3).min(axis=0)
    maxs = all_pts.reshape(-1, 3).max(axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() * 0.6 + 1e-6

    # Setup figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("4-hinge quaternion animation")

    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])

    # Line + points
    (line,) = ax.plot([], [], [], "-o", linewidth=2, markersize=5)

    # Timer and frame info text
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=11,
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Store timing info on figure for access in update function
    fig.ani_data = {
        'start_time': None,
        'frame_times': all_times,
        'current_frame': 0
    }

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        txt.set_text("")
        return line, txt

    def update(i):
        pts = all_pts[i]  # (5,3)
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        sim_time = all_times[i]
        total_time = all_times[-1]
        progress = (i / (len(all_pts) - 1)) * 100 if len(all_pts) > 1 else 0
        txt.set_text(f"Simulation Time: {sim_time:.3f}s / {total_time:.3f}s\n"
                     f"Frame: {i}/{len(all_pts)-1}\n"
                     f"Progress: {progress:.1f}%")
        return line, txt

    # Calculate intervals for real-time playback
    if t is not None:
        # Calculate time deltas between frames
        time_deltas = np.diff(all_times)
        # Calculate frame intervals (in milliseconds)
        # We'll use these to adjust animation speed
        frame_intervals = (time_deltas * 1000).astype(int)
        # Minimum interval is 16ms (~60fps), maximum is meaningful
        frame_intervals = np.clip(frame_intervals, 5, 100)

        # Create custom animation with variable frame rates
        ani = FuncAnimation(fig, update, frames=len(all_pts), init_func=init,
                            blit=False, interval=16, repeat=True)  # Base interval

        # Store frame intervals for later use
        ani.frame_intervals = frame_intervals
        ani.frame_times = all_times

    else:
        # Use fixed 30ms per frame if no time data
        ani = FuncAnimation(fig, update, frames=len(all_pts), init_func=init,
                            blit=False, interval=30, repeat=True)

    plt.show()
    return ani


ani = animate_chain(S, t=t, stride=2)
