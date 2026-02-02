import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R, Slerp
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
    v= z[3:6]
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
    R_inertial = np.eye(3)
    Ri = [None] * n
    for k in range(n - 1, -1, -1):
        R_inertial =  R[k] @ R_inertial 
        Ri[k]= R_inertial
    
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
    alpha[n] = np.zeros(6)            
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
    for k in range(n - 1, -1, -1):
        alphaplus[k] = phi(R[k].T @ L[k]).T @ R6(R[k].T) @ alpha[k + 1]

        vbar[k]=v[k]- (G[k].T @ R6(Ri[k].T)@g) 
        thetaddot[k] = vbar[k] - (G[k].T @ alphaplus[k])

        alpha[k] = alphaplus[k] + (H[k].T @ thetaddot[k]) + a_corolis[k]

    return thetaddot, V, alpha

def odefun(t, S, SystemData):
    """
    General n-link version for spherical joints only.

    State layout:
      S[0:4n]      = quaternions for each link, stacked as [qx,qy,qz,qw] (n of them)
      S[4n:7n]     = angular generalized velocities (omega) stacked as 3-vectors (n of them)

    Returns:
      dS with same layout:
        quaternion derivatives, then angular accelerations
    """
    S = np.asarray(S, dtype=float)

    # infer n from system data
    n = len(SystemData["H"])

    expected_len = 7 * n
    if S.size != expected_len:
        raise ValueError(f"S has length {S.size}, expected {expected_len} for n={n} (7*n).")

    dS = np.zeros_like(S)

    # --- unpack ---
    Q = S[0:4*n].reshape(n, 4)      # (n,4) quaternions [x,y,z,w]
    Beta = S[4*n:7*n].reshape(n, 3) # (n,3) omegas

    # optional safety: normalize quaternions (prevents drift)
    # Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)

    beta_list = [Beta[i] for i in range(n)]

    # --- torques (all zero as you said) ---
    tau_list = [np.zeros(3) for _ in range(n)]

    # --- rotations ---
    R = [quat_to_rotmat(Q[i]) for i in range(n)]  # each 3x3

    # --- system data ---
    H = SystemData["H"]
    M = SystemData["M"]
    L = SystemData["L"]

    # --- generalized accelerations ---
    thetaddot, _, _ = ATBI(R, beta_list, tau_list, H, M, L)
    # Expect thetaddot to be a list of (3,) for spherical joints

    # --- assemble dS ---
    # quaternion derivatives
    for i in range(n):
        dS[4*i:4*i+4] = quat_derivative(Q[i], Beta[i])

    # angular accelerations
    for i in range(n):
        dS[4*n + 3*i : 4*n + 3*i + 3] = thetaddot[i]

    return dS

def build_system_data(
    n: int,
    link_vec=np.array([0.0, 0.0, -5.0]),
    com_vec=np.array([0.0, 0.0, -2.5]),
    J_diag=np.array([1, 1, 0.1]),
    mass: float = 1.0,
):
    """
    Build SystemData for n identical links.
    Each link:
      H[k] = [I3 0]  (3x6 spherical hinge map as you used)
      L[k] = link_vec
      r_com[k] = com_vec
      J[k] = diag(J_diag)
      m[k] = mass
      M[k] computed as spatial inertia about hinge using r_com
    """

    # --- H: list of (3x6) hinge maps ---
    #H_one = np.hstack([np.eye(3), np.zeros((3, 3))])  # 3x6
    #H = [H_one.copy() for _ in range(n)]
    H = [None] * n
    H[0]= np.hstack([np.zeros((6, 6))]) 
    H[1]= np.hstack([np.eye(3), np.zeros((3, 3))])
    # --- L: link vectors (3,) ---
    link_vec = np.asarray(link_vec, dtype=float).reshape(3,)
    L = [link_vec.copy() for _ in range(n)]

    # --- CoM offsets from hinge (3,) ---
    com_vec = np.asarray(com_vec, dtype=float).reshape(3,)
    L_HtoC = [com_vec.copy() for _ in range(n)]

    # --- Inertia tensors (3x3) ---
    J_diag = np.asarray(J_diag, dtype=float).reshape(3,)
    J_one = np.diag(J_diag)
    J = [J_one.copy() for _ in range(n)]

    # --- Masses ---
    m = np.full(n, float(mass), dtype=float)

    # --- Spatial inertia (6x6 each) ---
    M = []
    for i in range(n):
        r = L_HtoC[i]
        rx = skew(r)
        Mi = np.block([
            [J[i] - m[i] * (rx @ rx),   m[i] * rx],
            [-m[i] * rx,                m[i] * np.eye(3)]
        ])
        M.append(Mi)

    return {"H": H, "L": L, "M": M}

def make_initial_state(n):
    # first quaternion: your existing one (30° about x)
    q1 = np.array([np.sin(np.pi/4), 0.0, 0.0, np.cos(np.pi/4)], dtype=float)
    #q2 = np.array([0.0, np.sin(np.pi/12), 0.0, np.cos(np.pi/12)], dtype=float)
    qI = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    # stack n quaternions: first is q1, remaining are identity
    Q0 = np.tile(qI, n)          # length 4n
    Q0[0:4] = q1                 # overwrite first quaternion
    #Q0[4:8] = q2                 # overwrite second quaternion

    # generalized velocities: 3 per body (spherical)
    Beta0 = np.zeros(3 * n, dtype=float)

    # full initial state
    S0 = np.concatenate([Q0, Beta0])   # length 7n
    return S0

n = 2
SystemData = build_system_data(n=n)

S0 = make_initial_state(n)

sol = sp.integrate.solve_ivp(
    fun=lambda t, S: odefun(t, S, SystemData),
    t_span=(0.0, 20.0),
    y0=S0,
    method="RK45",
    rtol=1e-4,
    atol=1e-7,
    dense_output=True
)

t = sol.t
S = sol.y.T
print(sol.success, sol.message)
print("t shape:", t.shape, "S shape:", S.shape)  # S should be (len(t), 7n)

def forward_kinematics_points(quats_xyzw, link_vecs, n=None):
    """
    quats_xyzw: sequence of quaternions (n,4) or flat array length 4*n
    link_vecs: sequence of n link vectors (3,) in each hinge's local frame
    n: optional number of links to use (defaults to len(quats))

    returns points array shape (n+1,3): base + n joints/ends
    """
    quats = np.asarray(quats_xyzw)
    # accept flat or (n,4)
    if quats.ndim == 1:
        if quats.size % 4 != 0:
            raise ValueError("quats_xyzw must have length multiple of 4 when flat")
        quats = quats.reshape(-1, 4)
    if n is None:
        n = quats.shape[0]
    else:
        if n > quats.shape[0]:
            raise ValueError(f"Requested n={n} larger than provided quaternions ({quats.shape[0]})")

    if len(link_vecs) < n:
        raise ValueError(f"link_vecs must have at least {n} entries")

    pts = [np.zeros(3)]
    R = np.eye(3)

    for i in range(n):
        R = R @ quat_to_rotmat(quats[i])     # accumulate orientation
        pts.append(pts[-1] + R @ link_vecs[i])    # step along current local link
    return np.vstack(pts)

def animate_chain(S, t=None, stride=1, n=None, link_vecs=None, smooth=True, target_fps=30, speed=30.0):
    """
    S shape: (T, 7*n) expected for this n-link spherical-joint model (n quaternions + 3n omegas)

    t: array of time points (optional). If provided and smooth=True, we resample quaternions to a uniform
       time grid at `target_fps` using slerp for smooth animation.
    stride: animate every 'stride' frames from the original samples before resampling
    n: explicit number of links (overrides inference)
    link_vecs: optional list of n link vectors
    smooth: enable slerp resampling to `target_fps`
    target_fps: desired frames per second for smooth playback
    speed: playback speed multiplier (>1.0 plays faster, <1.0 plays slower)
    """

    S = np.asarray(S)
    T = S.shape[0]

    # Determine n
    if n is None:
        if S.shape[1] % 7 != 0:
            raise ValueError("Cannot infer n from S; provide n explicitly")
        n = S.shape[1] // 7

    # default link vectors if not provided
    if link_vecs is None:
        link_vecs = [np.array([0.0, 0.0, -5.0]) for _ in range(n)]

    # Original sample times
    if t is None:
        t_in = np.arange(T)
    else:
        t_in = np.asarray(t)

    # Optionally resample to uniform high-rate grid using slerp
    if smooth and t is not None and len(t_in) > 1:
        n_frames = int(max(2, np.ceil((t_in[-1] - t_in[0]) * target_fps)))
        t_uniform = np.linspace(t_in[0], t_in[-1], n_frames)

        # original quaternions shape (T, n, 4)
        quats_orig = S[:, 0:4 * n].reshape(T, n, 4)

        # pre-allocate
        quats_uniform = np.zeros((n_frames, n, 4))

        for i_link in range(n):
            rots = R.from_quat(quats_orig[:, i_link, :])
            slerp = Slerp(t_in, rots)
            rots_u = slerp(t_uniform)
            quats_uniform[:, i_link, :] = rots_u.as_quat()

        all_times = t_uniform
        all_pts = [forward_kinematics_points(quats_uniform[k], link_vecs, n=n) for k in range(n_frames)]
        all_pts = np.array(all_pts)
        interval_ms = int(round(1000.0 / target_fps / float(speed)))
        interval_ms = max(1, interval_ms)

    else:
        # Use original (possibly strided) frames
        frames = range(0, T, stride)
        all_pts = []
        all_times = []
        for k in frames:
            quats = S[k, 0:4 * n].reshape(n, 4)
            pts = forward_kinematics_points(quats, link_vecs, n=n)
            all_pts.append(pts)
            all_times.append(t_in[k] if t is not None else k)
        all_pts = np.array(all_pts)
        all_times = np.array(all_times)
        # choose interval from median dt and scale by speed
        if len(all_times) > 1:
            dt_ms = int(max(1.0, np.median(np.diff(all_times)) * 1000.0 * stride))
        else:
            dt_ms = max(1, 30 * stride)
        interval_ms = int(max(1, float(dt_ms) / float(speed)))

    # Determine plot limits from trajectory
    mins = all_pts.reshape(-1, 3).min(axis=0)
    maxs = all_pts.reshape(-1, 3).max(axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() * 0.6 + 1e-6

    # Setup figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"{n}-link quaternion animation")

    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])

    # Line + points (will plot n+1 points)
    (line,) = ax.plot([], [], [], "-o", linewidth=2, markersize=5)

    # Timer and frame info text
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=11, 
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        txt.set_text("")
        return line, txt

    def update(i):
        pts = all_pts[i]
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        sim_time = all_times[i]
        total_time = all_times[-1]
        progress = (i / (len(all_pts) - 1)) * 100 if len(all_pts) > 1 else 0
        txt.set_text(f"Simulation Time: {sim_time:.3f}s / {total_time:.3f}s\n"
                     f"Frame: {i}/{len(all_pts)-1}\n"
                     f"Progress: {progress:.1f}%")
        return line, txt

    ani = FuncAnimation(fig, update, frames=len(all_pts), init_func=init,
                        blit=False, interval=interval_ms, repeat=True)

    plt.show()
    return ani
ani = animate_chain(S, t=t, stride=2, n=n)
