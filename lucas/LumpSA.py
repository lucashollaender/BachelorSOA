import numpy as np
import scipy as sp
import scipy.linalg as la
import pandas as pd
from SOALIB import soalib as sb

def assemble_1d_chain(K_e_list, conn, n_nodes):
    """
    Assemble global stiffness for a 1D chain.

    K_e_list: list of 12x12 element stiffness matrices (one per element)
    conn:     list of (node_i, node_j) connectivity using 1-based node numbers
              e.g. [(1,2),(2,3),(3,4)]
    """
    ndof_per_node=6
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


def sym(A):
    """Force exact symmetry (helps numerical stability)."""
    return 0.5 * (A + A.T)

# ============================================================
# Eigen solve + recover full 6-DOF mode shapes
# ============================================================
def solve_condensed_eigen_reduced(Kred, Mred, nmodes=6, ndpn=6, rot_reg=0.0, mass_normalize=True):
    """
    Solve eigenproblem from already-BC-reduced matrices (fixed DOFs removed).
    Condenses out rotations (massless) and recovers them.
    Returns gamma (translations), lambda_rot (rotations), and modes_red6 (full reduced 6DOF modes).
    """
    Kred = sym(Kred)
    Mred = sym(Mred)

    n = Kred.shape[0]
    red_dofs = np.arange(n)  # reduced system DOFs are 0..n-1

    # translation vs rotation by position mod 6
    r_mask = np.array([(d % ndpn) in (0, 1, 2) for d in red_dofs], dtype=bool)
    t_mask = ~r_mask

    t_idx = np.where(t_mask)[0]
    r_idx = np.where(r_mask)[0]

    Ktt = Kred[np.ix_(t_idx, t_idx)]
    Ktr = Kred[np.ix_(t_idx, r_idx)]
    Krt = Kred[np.ix_(r_idx, t_idx)]
    Krr = Kred[np.ix_(r_idx, r_idx)]
    Mtt = Mred[np.ix_(t_idx, t_idx)]

    if rot_reg and rot_reg > 0:
        Krr = Krr + rot_reg * np.eye(Krr.shape[0])

    # Condense rotations
    X = la.solve(Krr, Krt, assume_a="sym")   # X = inv(Krr)*Krt
    Kc = sym(Ktt - Ktr @ X)
    Mtt = sym(Mtt)

    # Solve condensed generalized EVP
    n_available = Kc.shape[0]
    nmodes = min(nmodes, n_available)
    evals, gammas = la.eigh(Kc, Mtt, subset_by_index=(0, nmodes - 1))
    evals = np.maximum(evals, 0.0)

    omega = np.sqrt(evals)
    freq = omega / (2*np.pi)

    # Recover rotations
    lambdas = -X @ gammas

    # Put back into reduced 6DOF ordering
    modes_red6 = np.zeros((n, nmodes))
    modes_red6[t_idx, :] = gammas
    modes_red6[r_idx, :] = lambdas

    # Optional normalization using reduced M
    if mass_normalize:
        for i in range(nmodes):
            mi = modes_red6[:, i].T @ (Mred @ modes_red6[:, i])
            if mi > 0:
                modes_red6[:, i] /= np.sqrt(mi)

    return {
        "freq_hz": freq,
        "omega": omega,
        "omega2": evals,
        "gamma": gammas,
        "lambda_rot": lambdas,
        "modes_red6": modes_red6,  # size (18 x nmodes) for your case
        "t_idx": t_idx,
        "r_idx": r_idx
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
        blocks.append(zero3)  # translational inertia
        blocks.append(m[i] * I3)      # rotational inertia = 0 (point-mass assumption)

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
    h=0.5
    w=0.5
    J  = 5e-7
    L=5
    n_nd=4
    n_elms = n_nd - 1
    L_e = L / n_elms
    M=mass_matrix_pointmass(n_nd,9000,h*w,L)
    # element stiffness list (3 elements)
    K_e_list = []
    for e in range(n_nd-1):
        Ke = sb.get_stiff_mat_rect_3D(h, w, L, E, G)
        K_e_list.append(Ke)

    # connectivity for chain 1-2-3-4
    conn = [(1, 2), (2, 3), (3, 4)]

    # Assemble global stiffness
    K = assemble_1d_chain(K_e_list, conn, n_nd)

    Kred = K[6:, 6:]
    Mred = M[6:, 6:]

    results = solve_condensed_eigen_reduced(Kred, Mred, nmodes=6, ndpn=6)

    print("\nEigenvalues:")
    df0 = pd.DataFrame(results["freq_hz"])
    print(df0)

    print("\nTranslational mode shapes γ:")
    df = pd.DataFrame(results["gamma"])
    print(df)

    print("\nRotational mode shapes λ:")
    df1 = pd.DataFrame(results["lambda_rot"])
    print(df1)
