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

def modalIntegrals_zero(nd,md,rho,A,L,gamma):
    n_elms = nd - 1
    L_e = L / n_elms
    lkOkj = [np.array([0.0, 0.0, i * L_e]) for i in range(nd)]
    print(lkOkj)
    m_e = rho * A * L_e
    m_tot=rho * A *L
    m = np.full(nd, m_e)
    m[0]  = m_e / 2
    m[-1] = m_e / 2
    gamma = np.vstack((np.zeros((3, gamma.shape[1])), gamma))
    sum_j0 = np.zeros((3,3))
    sum_p0 = np.zeros((3))
    for i in range(nd):
        sum_p0+=m[i]*lkOkj[i]
        sum_j0+=m[i]*sb.skew(lkOkj[i])@sb.skew(lkOkj[i])
    p0=1/m_tot*sum_p0
    J0=-sum_j0
    F0 = np.zeros((3, md))
    E0 = np.zeros((3, md))

    for j in range(md):
        for i in range(nd):
            gamma_i = gamma[3*i:3*(i+1), j]          # (3,)
            F0[:, j] += m[i] * (sb.skew(lkOkj[i]) @ gamma_i)
            E0[:, j] += m[i] * gamma_i

    G0 = np.zeros((md, md), dtype=gamma.dtype)

    for r in range(md):
        for s in range(md):
            G0sum = 0.0
            for i in range(nd):
                g_r = gamma[3*i:3*(i+1), r]   # (3,)
                g_s = gamma[3*i:3*(i+1), s]   # (3,)
                # "scalar" product for node i:
                G0sum += m[i] * np.dot(g_r.T, g_s)
            G0[r, s] = G0sum
    return G0, E0, F0, p0, J0, m_tot

def build_Mfl(G, F0, E0, J0, p0, m_tot):
    md = G.shape[0]
    I3 = np.eye(3)

    p0 = np.asarray(p0).reshape(3, 1)      # (3,1)

    # Allocate full matrix
    Mfl = np.zeros((md + 6, md + 6), dtype=G.dtype)

    # --- Top-left block: G ---
    Mfl[:md, :md] = G

    # --- Top-middle / Top-right: conjugate-transpose of F0/E0 ---
    # F0: (3,md) -> F0^H: (md,3)
    Mfl[:md, md:md+3] = F0.conj().T
    Mfl[:md, md+3:md+6] = E0.conj().T

    # --- Middle-left / Bottom-left ---
    Mfl[md:md+3, :md] = F0
    Mfl[md+3:md+6, :md] = E0

    # --- Middle-middle: J0 ---
    Mfl[md:md+3, md:md+3] = J0

    # --- Middle-right:  m * p0  (3x1) lives in a 3x3 block in your picture
    # The figure shows "m(k) p0^k" as a 3x3 block; in many texts this means m*skew(p0).
    # But if your assignment defines it as an outer product or diagonal placement, adjust here.
    # The most common is: m * skew(p0) for coupling between rotation and translation.
    #
    # If they literally want a 3x3 matrix built from p0: use m * sb.skew(p0).
    #
    # I'll implement the common rigid-body coupling: m*skew(p0)
    Mfl[md:md+3, md+3:md+6] = m_tot * sb.skew(p0)

    # --- Bottom-middle: -m * p0 coupling ---
    Mfl[md+3:md+6, md:md+3] = -m_tot * sb.skew(p0)

    # --- Bottom-right: m I3 ---
    Mfl[md+3:md+6, md+3:md+6] = m_tot * I3

    return Mfl

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

    G0, E0, F0, p0, J0, m_tot=modalIntegrals_zero(n_nd,7,rho,h*w,L,results["gamma"])

    Gprint=pd.DataFrame(G0)
    Eprint=pd.DataFrame(E0)
    Fprint=pd.DataFrame(F0)
    pprint=pd.DataFrame(p0)
    Jprint=pd.DataFrame(J0)

    print("G")
    print(Gprint)
    print("E")
    print(Eprint)
    print("F")
    print(Fprint)
    print("p")
    print(pprint)
    print("J")
    print(Jprint)

    M_fl=build_Mfl(G0, F0, E0, J0, p0, m_tot)
    mflprint=pd.DataFrame(M_fl)
    print(mflprint)

    Pi=results["modes_red6"]
    Pi = np.vstack((np.zeros((6, Pi.shape[1])), Pi))
    piprint=pd.DataFrame(Pi)
    