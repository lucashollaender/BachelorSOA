import numpy as np
from SOALIB import soalib as sb
# Function defining initial conditions for the n-body pendulum


def initial_condition(n):
    """
       Returns theta0 of size (4*n,) where each body has identity quaternion
       meaning it is aligned with the world frame (x-axis points along +x).

       Quaternion convention: [x, y, z, w]
       """

    # First hinge fixed => all other hinges are spherical
    # All hinges are initial positioned in the x-direction
    # position
    p0 = np.array([0, 0, 0])  # fixed hinge pos
    q = np.array([0, 0, 0, 1])
    theta0 = np.tile(q, n)

    # Velocity
    # First hinge is defined with 6 DOF, the rest are 3
    beta0 = np.zeros(6+3*n,)

    # state vector

    S0 = np.concatenate([p0, theta0, beta0])
    return S0


def build_system_data(n, link_vec, com_vec, J_diag, mass):
    """
    Build SystemData for n identical links.
    """
    # Convert inputs (works for tuples, lists, np arrays)
    link_vec = np.asarray(link_vec, dtype=float).reshape(3,)
    com_vec = np.asarray(com_vec,  dtype=float).reshape(3,)
    J_diag = np.asarray(J_diag,   dtype=float).reshape(3,)
    mass = float(mass)
    # Hinge maps: first link is fixed - rest are spherical
    H_fixed = sb.hinge_map("fixed")
    H_spherical = sb.hinge_map("spherical")
    H = [H_fixed.copy()] + [H_spherical.copy() for _ in range(n - 1)]

    # Geometry
    L = [link_vec.copy() for _ in range(n)]
    L_HtoC = [com_vec.copy() for _ in range(n)]

    # Inertia tensors
    J_one = np.diag(J_diag)
    J = [J_one.copy() for _ in range(n)]

    # Masses
    m = np.full(n, mass)

    # Spatial inertia
    M = []
    for i in range(n):
        r = L_HtoC[i]
        rx = sb.skew(r)

        Mi = np.block([
            [J[i] - m[i] * (rx @ rx),   m[i] * rx],
            [-m[i] * rx,                m[i] * np.eye(3)]
        ])
        M.append(Mi)

    return {"H": H, "L": L, "M": M}


def unpack_state_blocks(S, n):
    """
    Unpack flat state S into:
      p0 : (3,)
      Q  : list length n, each quaternion (4,)
      beta_blocks: list length n+1
          beta_blocks[0] : (6,) base spatial velocity V0
          beta_blocks[i+1] : (3,) omega_i for body i
    """
    S = np.asarray(S, dtype=float).reshape(-1)

    # --- theta ---
    n_theta = 3 + 4*n
    theta = S[:n_theta]
    p0 = theta[:3]
    Qmat = theta[3:].reshape(n, 4)
    Q = [Qmat[i].copy() for i in range(n)]

    # --- beta ---
    beta_flat = S[n_theta:]
    expected = 6 + 3*n
    if beta_flat.size != expected:
        raise ValueError(
            f"Expected beta length {expected}, got {beta_flat.size}")

    beta_blocks = [beta_flat[:6].copy()] + [
        beta_flat[6 + 3*i: 6 + 3*(i+1)].copy() for i in range(n)
    ]

    return p0, Q, beta_blocks


def pack_state(p0_dot, Q_dot, beta_dot):
    """
    Pack derivatives back into flat dS.

    p0_dot : (3,)
    Q_dot  : list length n, each (4,)
    beta_dot : flat (6 + 3*n,)
    """
    theta_dot = np.concatenate([p0_dot] + Q_dot)  # (3 + 4*n,)
    return np.concatenate([theta_dot, beta_dot])  # (7*n + 9,)


def normalize_quats_list(Q):
    """Normalize all quaternions in list Q in-place-like (returns new list)."""
    Qn = []
    for q in Q:
        q = np.asarray(q, dtype=float).reshape(4,)
        nq = np.linalg.norm(q)
        if nq == 0.0:
            raise ValueError("Quaternion has zero norm.")
        Qn.append(q / nq)
    return Qn


def odefun(t, S, sys, n):
    S = np.asarray(S, dtype=float).reshape(-1)
    # Firstly we unpack the state vector

    theta = S[:4*n].reshape(n, 4)
    beta = S[4*n:]

    # initiating solution vectors
    theta_dot = np.zeros_like(theta)
    beta_dot = np.zeros_like(beta)

    Q, beta = unpack_state_blocks(S, n)

    # --- theta_dot from angular velocities ---
    Q_dot = [None] * n

    # quaternion derivatives
    for i in range(n):
        if i == 0:
            # base block is 6 entries: [omega(3), v(3)] (assumed)
            omega_i = beta[0][:3]
        else:
            # spherical blocks are already omega(3)
            # note: beta_blocks[1] corresponds to body 0 omega if you chose that layout
            omega_i = beta[i]

        Q_dot[i] = sb.quat_derivative(Q[i], omega_i)

    # pack theta_dot
    theta_dot = np.concatenate(Q_dot)

    # --- torques (all zero as you said) ---
    tau_list = [np.zeros(3) for _ in range(n)]

    # --- rotations ---
    R = [sb.q2R(((Q[i]) for i in range(n)), 3)]  # each 3x3

    # --- system data ---
    H = sys["H"]
    M = sys["M"]
    L = sys["L"]

    # --- generalized accelerations ---
    beta_dot, _, _ = ATBI(R, beta_list, tau_list, H, M, L)

    dS = np.concatenate(theta_dot, beta_dot)

    return dS
