import numpy as np
import scipy as sp
import scipy.linalg as la
import pandas as pd
from SOALIB import soalib as sb

def get_stiff_mat_rect_3D(h, w, L, E, G):
        # b < a // h < w, w = y, h = z
        # Parameters
    a = w / 2
    b = h / 2

        # Book -> code
        # x -> z
        # y -> x
        # z -> y

        # Rectangular cross-section
    k_y = 1.2
    k_z = k_y
    K = a * b**3 * (16/3 - 3.36 * a / b * (1 - a**4 / (12 * b**4)))
    A = w*h
    I_y = w * h**3 / 12
    I_z = h * w**3 / 12

        # Factors
    phi_y = 12 * E * I_z * k_y / (A * G * L**2)
    phi_z = 12 * E * I_y * k_z / (A * G * L**2)
    S = G * K / L

        # X
    X = A * E / L

        # Y
    Y_1 = 12 * E * I_z / ((1 + phi_y) * L**3)
    Y_2 = 6 * E * I_z / ((1 + phi_y) * L**2)
    Y_3 = (4 + phi_y) * E * I_z / ((1 + phi_y) * L)
    Y_4 = (2 - phi_y) * E * I_z / ((1 + phi_y) * L)

        # Z
    Z_1 = 12 * E * I_y / ((1 + phi_z) * L**3)
    Z_2 = 6 * E * I_y / ((1 + phi_z) * L**2)
    Z_3 = (4 + phi_z) * E * I_y / ((1 + phi_z) * L)
    Z_4 = (2 - phi_z) * E * I_y / ((1 + phi_z) * L)

        # Stiffness matrix
    diag = [None] * 6

    diag[0] = np.array([X, Y_1, Z_1, S, Z_3, Y_3, X, Y_1, Z_1, S, Z_3, Y_3])
    diag[1] = np.array([0, 0, -Z_2, 0, 0, -Y_2, 0, 0, Z_2, 0])
    diag[2] = np.array([0, Y_2, 0, 0, Z_2, 0, 0, -Y_2])
    diag[3] = np.array([-X, -Y_1, -Z_1, -S, Z_4, Y_4])
    diag[4] = np.array([0, 0, -Z_2, 0])
    diag[5] = np.array([0, Y_2])

    k = np.diag(diag[0], k=0)

    for i in range(1, 6):
        k = k + np.diag(diag[i], k=-2*i) + np.diag(diag[i], k=2*i)

        # Change so rotations first is along
    perm = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

    k_perm = k[np.ix_(perm, perm)]

    return k_perm

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
    lkOkj = [np.array([i * L_e, 0.0, 0.0]) for i in range(nd)]
    md = gamma.shape[1]
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
    p1 = np.zeros((3, md))
    sum_j1 = np.zeros((3,3))
    J1=np.zeros((3,3*md))
    for j in range(md):
        sum_p1 = np.zeros(3)
        sum_j1 = np.zeros((3,3))
        for i in range(nd):
            gamma_i = gamma[3*i:3*(i+1), j]
            F0[:, j] += m[i] * sb.skew(lkOkj[i]) @ gamma_i
            E0[:, j] += m[i] * gamma_i
            sum_p1 += m[i]*gamma_i
            sum_j1 += m[i]*sb.skew(gamma_i)@sb.skew(lkOkj[i])
        p1[:, j] += 1/m_tot * sum_p1
        J1[:, 3*j:3*(j+1)] = -sum_j1

    G0 = np.zeros((md, md), dtype=gamma.dtype)
    sum_j2 = np.zeros((3,3))
    J2=np.zeros((3*md,3*md))
    sum_F1 = np.zeros((3, 1))
    F1= np.zeros((3*md, md))
    for r in range(md):
        for s in range(md):
            sum_j2 = np.zeros((3,3))
            sum_F1 = np.zeros(3)
            G0sum = 0.0
            for i in range(nd):
                g_r = gamma[3*i:3*(i+1), r]
                g_s = gamma[3*i:3*(i+1), s]
                # "scalar" product for node i:
                G0sum += m[i] * np.dot(g_r.T, g_s)
                sum_j2 += m[i]*sb.skew(g_r)@sb.skew(g_s)
                sum_F1 += m[i]*sb.skew(g_r)@g_s
            G0[r, s] = G0sum
            J2[3*r:3*(r+1), 3*s:3*(s+1)] = -sum_j2
            F1[3*r:3*(r+1), s] = -sum_F1
    return G0, E0, F0, p0, J0, m_tot, p1, J1, J2, F1

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
    k=get_stiff_mat_rect_3D(h, w, L_e, E, G)   
    # Global stiffness matrix setup
    K_st = np.zeros((6*n_nd, 6*n_nd))

    for i in range(n_nd - 1):
        k_i = np.zeros((6*n_nd, 6*n_nd))
        k_i[i*6:i*6+12, i*6:i*6+12] = k
        K_st = K_st + k_i

    Kred = K_st[6:, 6:]
    Mred = M[6:, 6:]

    results = solve_condensed_eigen_reduced(Kred, Mred, nmodes=4, ndpn=6)
    print(pd.DataFrame(K_st))
    print("\nEigenvalues:")
    df0 = pd.DataFrame(results["omega2"])
    print(df0)

    print("\nTranslational mode shapes γ:")
    df = pd.DataFrame(results["gamma"])
    print(df)

    print("\nRotational mode shapes λ:")
    df1 = pd.DataFrame(results["lambda_rot"])
    print(df1)

    G0, E0, F0, p0, J0, m_tot, p1, J1, J2, F1=modalIntegrals_zero(n_nd,rho,h*w,L,results["gamma"])

    Gprint=pd.DataFrame(G0)
    Eprint=pd.DataFrame(E0)
    Fprint=pd.DataFrame(F0)
    pprint=pd.DataFrame(p0)
    Jprint=pd.DataFrame(J0)
    print("p0")
    print(pd.DataFrame(p0))
    print("E0")
    print(pd.DataFrame(E0))
    print("F0")
    print(pd.DataFrame(F0))
    print("J1")
    print(pd.DataFrame(J1))
    print("p1")
    print(pd.DataFrame(p1))
    print("J2")
    print(pd.DataFrame(J2))
    print("F1")
    print(pd.DataFrame(F1))

    """
    M_fl=build_Mfl(G0, F0, E0, J0, p0, m_tot)
    mflprinttot=pd.DataFrame(M_fl)
    mflprint=pd.DataFrame(M_fl[-6:,-6:])
    print("Mfl")
    print(mflprint)
    print(mflprinttot)

    Pi=results["modes_red6"]
    Pi = np.vstack((np.zeros((6, Pi.shape[1])), Pi))

    Kmodal=Pi.T@K_st@Pi
    """