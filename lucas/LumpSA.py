import numpy as np
import scipy as sp
import scipy.linalg as la
import pandas as pd
from SOALIB import soalib as sb

def solve_condensed_eigen_reduced(Kred, Mred, nmodes=6, ndpn=6, rot_reg=0.0, mass_normalize=False):
    """
    Solve eigenproblem from already-BC-reduced matrices (fixed DOFs removed).
    Condenses out rotations (massless) and recovers them.
    Returns gamma (translations), lambda_rot (rotations), and modes_red6 (full reduced 6DOF modes).
    """

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

    # Condense rotations
    X = la.solve(Krr, Krt, assume_a="sym")   # X = inv(Krr)*Krt
    Kc = Ktt - Ktr @ X

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

def modalIntegrals_zero(nd,rho,A,L,gamma):
    n_elms = nd - 1
    L_e = L / n_elms
    lkOkj = [i * L_e for i in range(nd)] 
    m_e = rho * A * L_e
    m = np.full(nd, m_e)
    m[0]  = m_e / 2
    m[-1] = m_e / 2
    for i in range(nd):
        sum_p0+=m[i]*lkOkj[i]
        p0=1/m_e*sum_p0

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
    E = 230e9
    G = 80e9
    h=0.1
    w=0.1
    rho=7850
    L=5
    n_nd=4
    n_elms = n_nd - 1
    L_e = L / n_elms
    M=mass_matrix_pointmass(n_nd,rho,h*w,L)
    # element stiffness list (3 elements)
    k=sb.get_stiff_mat_rect_3D(h, w, L, E, G)   
    # Global stiffness matrix setup
    K_st = np.zeros((6*n_nd, 6*n_nd))

    for i in range(n_nd - 1):
        k_i = np.zeros((6*n_nd, 6*n_nd))
        k_i[i*6:i*6+12, i*6:i*6+12] = k
        K_st = K_st + k_i

    Kred = K_st[6:, 6:]
    Mred = M[6:, 6:]

    results = solve_condensed_eigen_reduced(Kred, Mred, nmodes=7, ndpn=6)

    print("\nEigenvalues:")
    df0 = pd.DataFrame(results["omega2"])
    print(df0)

    print("\nTranslational mode shapes γ:")
    df = pd.DataFrame(results["gamma"])
    print(df)

    print("\nRotational mode shapes λ:")
    df1 = pd.DataFrame(results["lambda_rot"])
    print(df1)
