import numpy as np
from SOALIB import soalib as sb
# Function defining initial conditions for the n-body pendulum


def initial_condition(n):
    """
       Returns theta0 of size (4*n,) where each body has identity quaternion
       meaning it is aligned with the world frame (x-axis points along +x).

       Quaternion convention: [x, y, z, w]
       """

    # All hinges are spherical
    # All hinges are initial positioned in the x-direction
    # position
    q = np.array([0, 0, 0, 1])
    theta0 = np.tile(q, n)

    # Velocity
    # All hinges are 3 DOF
    beta0 = np.zeros(3*n,)

    # state vector

    S0 = np.concatenate([theta0, beta0])
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

    # Hinge maps: All spherical
    H_spherical = sb.hinge_map("spherical")
    H = [H_spherical.copy() for _ in range(n)]

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
    beta = S[4*n:].reshape(n, 3)

    # Normalizing quats
    theta = np.array(normalize_quats_list(theta))

    # theta_dot from angular velocities and initial quaternions
    Q_dot = [None] * n

    # quaternion derivatives
    for i in range(n):
        omega_i = beta[i]  # Angular velocity
        Q_dot[i] = sb.quat_derivative(theta[i], omega_i)

    # pack theta_dot
    theta_dot = np.concatenate(Q_dot)

    # All generalized forces are 0
    tau_list = [np.zeros(3) for _ in range(n)]

    # 3x3 Rotation matrices
    R = [sb.q2R(theta[i], 3) for i in range(n)]

    # --- generalized accelerations ---
    beta_dot, _, _ = ATBI(theta, beta, tau_list, sys)

    beta_dot = np.asarray(beta_dot, dtype=float)
    dS = np.concatenate([theta_dot, beta_dot.reshape(-1)])
    dS = np.concatenate([theta_dot, beta_dot.reshape(-1)])

    return dS


def scatter1_ATBI(theta, beta, sys):

    # Convert theta,beta into numpy arrasýs
    theta = np.asarray(theta, dtype=float)
    beta = np.asarray(beta,  dtype=float)

    # Define n from theta
    n = theta.shape[0]

    # Insert error if sizes dont match
    assert theta.shape == (n, 4)
    assert beta.shape == (n, 3)

    # initiate lists
    V = [None]*n
    a = [None]*n
    b = [None]*n

    V_parent = np.zeros(6)
    for k in range(n-1, -1, -1):

        # Correct quaternions and general velocities
        q_rel = theta[k]
        beta_k = beta[k]

        # Tranpose the hinge map
        HT = sys["H"][k].T

        # Handling the rotation matrices
        pRc = sb.q2R(q_rel, 6)  # Rotation matrix child=>parent
        cRp = pRc.T           # Transposing to get rotation from parent => child

        # Getting the correct position vector
        # position vector rotated to child frame
        L = sb.q2R(q_rel, 3).T @ sys["L"][k]

        # Rigid body matrix
        phi = sb.phi(L).T  # Transpose as we are dealing with velocities

        # Velocity recursion
        V_k = phi @ cRp @ V_parent + HT @ beta_k

        # coriolis acc
        a[k] = sb.skew6(V_k) @ HT @ beta_k

        # Gyroscopic
        M_k = sys["M"][k]
        b[k] = sb.bar6(V_k)@M_k@V_k

        # shift recursion
        V[k] = V_k
        V_parent = V_k

    return V, a, b


def ATBI(theta, beta, tau, sys):
    """
    Docstring for ATBI

    :param theta: List of quaternions
    :param beta: list of generalized velocities
    :param tau_list: list of generalized forces
    :param H: List of hinge maps
    :param M: list of spatial inertia
    :param L: list of position vectors
    """
    # Define n from theta
    n = theta.shape[0]
    # Call scatter to get velocity, gyro and coriolis
    V, a, b = scatter1_ATBI(theta, beta, sys)

    # unpacking sys
    H = sys["H"]
    M = sys["M"]
    L = sys["L"]

    # Computing the rotation from interial to body
    R_inertial = np.eye(6)
    Ri = [None] * n
    for k in range(n - 1, -1, -1):
        R_inertial = R_inertial  @ sb.q2R(theta[k], 6)
        Ri[k] = R_inertial

    # initiate lists
    G = [None]*n
    Pp = [None]*n
    nu = [None]*n
    xip = [None]*n
    vbar = [None] * n
    g = np.array([
        0.0, 0.0, 0.0,    # angular part
        0.0, 0.0, 9.81   # linear part
    ])

    # Initiating child values
    Pp_child = np.zeros((6, 6))
    xip_child = np.zeros(6)

    for k in range(n):
        if k == 0:
            Pk = M[k]
            xi = Pk @ a[k] + b[k]
        else:
            X = sb.phi(L[k - 1]) @ sb.q2R(theta[k-1], 6)
            Pk = X @ Pp_child @ X.T + M[k]
            xi = X @ xip_child + Pk @ a[k] + b[k]

        Dk = H[k] @ Pk @ H[k].T
        G[k] = (Pk @ H[k].T) @ np.linalg.inv(Dk)
        tau_bark = np.eye(6) - G[k] @ H[k]
        Pp[k] = tau_bark @ Pk
        epsk = tau[k] - (H[k] @ xi)
        nu[k] = np.linalg.solve(Dk, epsk)
        xip[k] = xi + G[k] @ epsk

        Pp_child = Pp[k]
        xip_child = xip[k]

    # Second scatter sweep in ATBI
    alpha_p = [None] * n
    nu_bar = [None]*n
    gamma = [None] * n
    alpha = [None] * n

    alpha_parent = np.zeros(6)
    for k in range(n - 1, -1, -1):
        q_rel = theta[k]

        pRc = sb.q2R(q_rel, 6)  # Rotation matrix child=>parent
        cRp = pRc.T           # Transposing to get rotation from parent => child

        # Getting the correct position vector
        # position vector rotated to child frame
        Lk = sb.q2R(q_rel, 3).T @ L[k]

        # Rigid body matrix
        phi = sb.phi(Lk).T  # Transpose as we are dealing with velocities

        # Last scatter sweep
        alpha_p[k] = phi @ cRp @ alpha_parent
        nu_bar[k] = nu[k] - G[k].T @ (Ri[k].T @ g)
        gamma[k] = nu_bar[k] - G[k].T @ alpha_p[k]
        alpha[k] = alpha_p[k] + H[k].T @ gamma[k] + a[k]

        alpha_parent = alpha[k]

    return gamma, V, alpha
