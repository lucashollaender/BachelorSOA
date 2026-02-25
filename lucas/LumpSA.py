import numpy as np
import scipy as sp
import scipy.linalg as la
import pandas as pd

# ============================================================
# Helpers: DOF mapping and BC reduction
# ============================================================
DOF = {"u": 0, "v": 1, "w": 2, "thx": 3, "thy": 4, "thz": 5}

def timoshenko_beam_3d_stiffness(E, G, A, L, Iy, Iz, J, phi_y, phi_z):
    """
    12x12 local stiffness matrix matching the form in the provided image.

    DOF order:
    [u1, v1, w1, thx1, thy1, thz1, u2, v2, w2, thx2, thy2, thz2]
    """

    K = np.zeros((12, 12), dtype=float)

    # --- Axial ---
    k_ax = E * A / L
    K[0, 0] =  k_ax
    K[0, 6] = -k_ax
    K[6, 0] = -k_ax
    K[6, 6] =  k_ax

    # --- Torsion ---
    k_tor = G * J / L
    K[3, 3]   =  k_tor
    K[3, 9]   = -k_tor
    K[9, 3]   = -k_tor
    K[9, 9]   =  k_tor

    # --- Bending associated with v and thz (uses Iz, phi_y) ---
    kvv = 12 * E * Iz / (L**3 * (1 + phi_y))
    kvt =  6 * E * Iz / (L**2 * (1 + phi_y))
    ktt = (4 + phi_y) * E * Iz / (L * (1 + phi_y))
    ktt2 = (2 - phi_y) * E * Iz / (L * (1 + phi_y))

    # indices: v1=1, thz1=5, v2=7, thz2=11
    v1, thz1, v2, thz2 = 1, 5, 7, 11

    K[v1, v1]     =  kvv
    K[v1, thz1]   =  kvt
    K[v1, v2]     = -kvv
    K[v1, thz2]   =  kvt

    K[thz1, v1]   =  kvt
    K[thz1, thz1] =  ktt
    K[thz1, v2]   = -kvt
    K[thz1, thz2] =  ktt2

    K[v2, v1]     = -kvv
    K[v2, thz1]   = -kvt
    K[v2, v2]     =  kvv
    K[v2, thz2]   = -kvt

    K[thz2, v1]   =  kvt
    K[thz2, thz1] =  ktt2
    K[thz2, v2]   = -kvt
    K[thz2, thz2] =  ktt

    # --- Bending associated with w and thy (uses Iy, phi_z) ---
    kww = 12 * E * Iy / (L**3 * (1 + phi_z))
    kwt =  6 * E * Iy / (L**2 * (1 + phi_z))
    kyy = (4 + phi_z) * E * Iy / (L * (1 + phi_z))
    kyy2 = (2 - phi_z) * E * Iy / (L * (1 + phi_z))

    # indices: w1=2, thy1=4, w2=8, thy2=10
    w1, thy1, w2, thy2 = 2, 4, 8, 10

    K[w1, w1]     =  kww
    K[w1, thy1]   = -kwt
    K[w1, w2]     = -kww
    K[w1, thy2]   = -kwt

    K[thy1, w1]   = -kwt
    K[thy1, thy1] =  kyy
    K[thy1, w2]   =  kwt
    K[thy1, thy2] =  kyy2

    K[w2, w1]     = -kww
    K[w2, thy1]   =  kwt
    K[w2, w2]     =  kww
    K[w2, thy2]   =  kwt

    K[thy2, w1]   = -kwt
    K[thy2, thy1] =  kyy2
    K[thy2, w2]   =  kwt
    K[thy2, thy2] =  kyy

    return K

def assemble_1d_chain_4nodes(K_e_list, conn, ndof_per_node=6):
    """
    Assemble global stiffness for a 1D chain.

    K_e_list: list of 12x12 element stiffness matrices (one per element)
    conn:     list of (node_i, node_j) connectivity using 1-based node numbers
              e.g. [(1,2),(2,3),(3,4)]
    """
    n_nodes = 4
    ndof = n_nodes * ndof_per_node
    K_global = np.zeros((ndof, ndof))

    for e, (ni, nj) in enumerate(conn):
        Ke = K_e_list[e]

        # convert node numbers (1-based) to global dof indices (0-based)
        dofs_i = list(range((ni-1)*ndof_per_node, ni*ndof_per_node))
        dofs_j = list(range((nj-1)*ndof_per_node, nj*ndof_per_node))
        edofs = dofs_i + dofs_j  # 12 dofs for this element

        # assemble: add Ke into K_global at the edofs positions
        for a in range(12):
            A = edofs[a]
            for b in range(12):
                B = edofs[b]
                K_global[A, B] += Ke[a, b]

    return K_global

def gdof(node: int, dof_name: str, ndpn: int = 6) -> int:
    """(1-based node, dof name) -> 0-based global DOF index"""
    return (node - 1) * ndpn + DOF[dof_name]

def sym(A):
    """Force exact symmetry (helps numerical stability)."""
    return 0.5 * (A + A.T)

def apply_bc_elimination(K, M, constrained_dofs):
    """
    Step 1: Apply Dirichlet BCs by eliminating rows/cols.
    Returns: K_red, M_red, free_dofs, constrained_dofs_sorted
    """
    K = sym(K)
    M = sym(M)

    n = K.shape[0]
    constrained = np.array(sorted(set(constrained_dofs)), dtype=int)
    all_dofs = np.arange(n, dtype=int)
    free = np.setdiff1d(all_dofs, constrained)

    K_red = K[np.ix_(free, free)]
    M_red = M[np.ix_(free, free)]
    return K_red, M_red, free, constrained

# ============================================================
# Core: static condensation of massless rotational DOFs
# ============================================================
def static_condense_rotations(K_red, M_red, free_dofs, ndpn=6, rot_reg=0.0):
    """
    Steps 2 & 3:
      - Partition reduced DOFs into translations (t) and rotations (r)
      - Condense rotations out of stiffness:
            Kc = Ktt - Ktr * inv(Krr) * Krt
        and mass becomes simply Mtt (rotational blocks assumed zero)

    Inputs:
      K_red, M_red : matrices AFTER BC elimination
      free_dofs    : global DOF indices corresponding to K_red/M_red
      rot_reg      : optional small regularization added to Krr diagonal if needed

    Returns:
      Kc, Mtt, maps, and factors needed for recovery of rotations
    """
    K_red = sym(K_red)
    M_red = sym(M_red)

    # Identify which reduced DOFs are translational vs rotational
    # (based on global DOF index mod 6)
    t_mask = np.array([(d % ndpn) in (0, 1, 2) for d in free_dofs], dtype=bool)
    r_mask = np.array([(d % ndpn) in (3, 4, 5) for d in free_dofs], dtype=bool)

    t_idx = np.where(t_mask)[0]  # indices into reduced system
    r_idx = np.where(r_mask)[0]

    # Partition reduced matrices
    Ktt = K_red[np.ix_(t_idx, t_idx)]
    Ktr = K_red[np.ix_(t_idx, r_idx)]
    Krt = K_red[np.ix_(r_idx, t_idx)]
    Krr = K_red[np.ix_(r_idx, r_idx)]

    Mtt = M_red[np.ix_(t_idx, t_idx)]
    # Rotational mass assumed to be ~0 (point-mass model)

    # Regularize Krr slightly if requested (sometimes helpful if Krr is ill-conditioned)
    if rot_reg and rot_reg > 0:
        Krr = Krr + rot_reg * np.eye(Krr.shape[0])

    # Condensation: Kc = Ktt - Ktr * inv(Krr) * Krt
    # Use solve() rather than inverse for numerical stability:
    # Solve Krr * X = Krt  => X = inv(Krr)*Krt
    X = la.solve(Krr, Krt, assume_a="sym")  # (r x t)
    Kc = Ktt - Ktr @ X                      # (t x t)

    Kc = sym(Kc)
    Mtt = sym(Mtt)

    maps = {
        "t_idx": t_idx,
        "r_idx": r_idx,
        "t_global": free_dofs[t_idx],  # global dofs kept in eigenproblem
        "r_global": free_dofs[r_idx],  # global dofs condensed out
    }

    # Store what we need to recover rotations: lambda = -inv(Krr)*Krt*gamma = -X*gamma
    recovery = {"X": X}  # X = inv(Krr)*Krt

    return Kc, Mtt, maps, recovery

# ============================================================
# Eigen solve + recover full 6-DOF mode shapes
# ============================================================
def solve_condensed_eigen(K, M, constrained_dofs, nmodes=6, ndpn=6, rot_reg=0.0, mass_normalize=True):
    """
    Full pipeline:
      1) BC elimination on K,M
      2) Partition & static condensation of rotational DOFs
      3) Solve condensed generalized eigenproblem
      4) Recover rotational components
      5) Expand to full global DOF size (constrained DOFs = 0)
    """
    ndof_total = K.shape[0]

    # Step 1
    K_red, M_red, free_dofs, constrained_sorted = apply_bc_elimination(K, M, constrained_dofs)

    # Step 2-3
    Kc, Mtt, maps, recovery = static_condense_rotations(K_red, M_red, free_dofs, ndpn=ndpn, rot_reg=rot_reg)

    # Step 4: solve Kc * gamma = lambda * Mtt * gamma, lambda = omega^2
    # Use eigh for symmetric generalized eigenproblem; requires Mtt SPD (it should be if translational masses > 0)
    n_available = Kc.shape[0]
    if nmodes > n_available:
        nmodes = n_available

    evals, gammas = la.eigh(Kc, Mtt, subset_by_index=(0, nmodes - 1))
    evals = np.maximum(evals, 0.0)

    omega = np.sqrt(evals)         # rad/s
    freq = omega / (2 * np.pi)     # Hz

    # Step 5: recover rotations for each mode: lambda = -X * gamma
    X = recovery["X"]              # (r x t)
    lambdas = -X @ gammas          # (r x nmodes)

    # Build reduced full vector (t + r) in reduced DOF ordering
    # reduced DOFs are ordered as free_dofs; within that, we place t and r components back in their spots
    modes_red = np.zeros((len(free_dofs), nmodes))
    modes_red[maps["t_idx"], :] = gammas
    modes_red[maps["r_idx"], :] = lambdas

    # Expand to full global DOFs (constrained DOFs are zero)
    modes_full = np.zeros((ndof_total, nmodes))
    modes_full[free_dofs, :] = modes_red
    modes_full[constrained_sorted, :] = 0.0

    # Optional: mass normalize with FULL M (common in dynamics)
    if mass_normalize:
        M_sym = sym(M)
        for i in range(nmodes):
            mi = modes_full[:, i].T @ (M_sym @ modes_full[:, i])
            if mi > 0:
                modes_full[:, i] /= np.sqrt(mi)

    return {
        "omega": omega,                 # rad/s
        "freq_hz": freq,                # Hz
        "omega2": evals,                # eigenvalues (ω²)
        "gamma": gammas,                # translational mode shapes
        "lambda_rot": lambdas,          # rotational mode shapes
        "modes_full": modes_full        # full 6DOF modes
    }

def mass_matrix_pointmass(nd,rho,A,L):
    n_elms = nd - 1
    L_e = L / n_elms  # element length

    # Lumped nodal masses from element mass m_e = rho*A*L_e
    m_e = rho * A * L_e
    m = np.full(nd, m_e)
    m[0]  = m_e / 2
    m[-1] = m_e / 2
    zero3 = np.zeros((3, 3))
    I3 = np.eye(3)

    blocks = []
    for i in range(nd):
        blocks.append(m[i] * I3)  # translational inertia
        blocks.append(zero3)      # rotational inertia = 0 (point-mass assumption)

    M = la.block_diag(*blocks)
    return M

if __name__ == "__main__":

    # ----------------------------
    # Model / mesh
    # ----------------------------

    # ----------------------------
    # Build global M (24x24)
    # DOF per node: [u v w thx thy thz]
    # So each node contributes: diag(m_i*I3, 0_3)
    # ----------------------------

    # ----------------------------
    # Build global K (24x24)
    # ----------------------------
    E = 210e9
    G = 80e9
    A = 0.01
    Iy = 1e-6
    Iz = 2e-6
    J  = 5e-7
    L=5
    phi_y = 0.2
    phi_z = 0.3
    n_nd=4
    n_elms = n_nd - 1
    L_e = L / n_elms
    M=mass_matrix_pointmass(n_nd,9000,A,5)
    # element stiffness list (3 elements)
    K_e_list = []
    for e in range(n_nd-1):
        Ke = timoshenko_beam_3d_stiffness(E, G, A, L_e, Iy, Iz, J, phi_y, phi_z)
        K_e_list.append(Ke)

    # connectivity for chain 1-2-3-4
    conn = [(1, 2), (2, 3), (3, 4)]

    # Assemble global stiffness
    K = assemble_1d_chain_4nodes(K_e_list, conn, ndof_per_node=6)

    # ----------------------------
    # Boundary conditions
    # Node 1 fully fixed
    # ----------------------------
    constrained = [gdof(1, d) for d in ["u", "v", "w", "thx", "thy", "thz"]]

    # ----------------------------
    # Condensed eigen-solution (translations kept, rotations condensed)
    # ----------------------------
    results = solve_condensed_eigen(
        K, M,
        constrained_dofs=constrained,
        nmodes=6,
        ndpn=6,
        rot_reg=0.0,          # you can try rot_reg=1e-12*np.trace(K)/K.shape[0] if Krr is ill-conditioned
        mass_normalize=True
    )

    print("\nEigenvalues:")
    df0 = pd.DataFrame(results["freq_hz"])
    print(df0)

    print("\nTranslational mode shapes γ:")
    df = pd.DataFrame(results["gamma"])
    print(df)

    print("\nRotational mode shapes λ:")
    df1 = pd.DataFrame(results["lambda_rot"])
    print(df1)
