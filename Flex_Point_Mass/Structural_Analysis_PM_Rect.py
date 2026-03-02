import numpy as np
import scipy.linalg as la
import pandas as pd
from SOALIB import soalib as sb
from .Body_Properties import Joint, Rigid_Properties, Flex_Properties


class Structural_Analysis_PM_Rect:
    """
    Class:
    Structural analysis of 3D rectangular cantilever beam with point mass assumption.

    Call:
    Structural_Analysis_PM_Rect(klOO, rho, E, G, n_nd, n_md)

    Constraint:
    Width, w >= Height, h
    """

    def get_stiff_mat_rect_3D(self):
        # b < a // h < w
        # Parameters
        w = self.w
        h = self.h
        a = w
        b = h
        L = self.L
        E = self.E
        G = self.G

        # Book -> code
        # x -> z
        # y -> x
        # z -> y

        # Rectangular cross-section
        k_x = 1.2
        k_y = k_x
        K = a * b**3 * (16/3 - 3.36 * a / b * (1 - a**4 / (12 * b**4)))
        A = self.A
        I_x = w * h**3 / 12
        I_y = h * w**3 / 12

        # Factors
        phi_x = 12 * E * I_y * k_x / (A * G * L**2)
        phi_y = 12 * E * I_x * k_y / (A * G * L**2)
        S = G * K / L

        # Z
        Z = A * E / L

        # X
        X_1 = 12 * E * I_y / ((1 + phi_x) * L**3)
        X_2 = 6 * E * I_y / ((1 + phi_x) * L**2)
        X_3 = (4 + phi_x) * E * I_y / ((1 + phi_x) * L)
        X_4 = (2 - phi_x) * E * I_y / ((1 + phi_x) * L)

        # Y
        Y_1 = 12 * E * I_x / ((1 + phi_y) * L**3)
        Y_2 = 6 * E * I_x / ((1 + phi_y) * L**2)
        Y_3 = (4 + phi_y) * E * I_x / ((1 + phi_y) * L)
        Y_4 = (2 - phi_y) * E * I_x / ((1 + phi_y) * L)

        # Stiffness matrix
        diag = [None] * 6

        diag[0] = np.array(
            [Z, X_1, Y_1, S, Y_3, X_3, Z, X_1, Y_1, S, Y_3, X_3])
        diag[1] = np.array([0, 0, -Y_2, 0, 0, -X_2, 0, 0, Y_2, 0])
        diag[2] = np.array([0, X_2, 0, 0, Y_2, 0, 0, -X_2])
        diag[3] = np.array([-Z, -X_1, -Y_1, -S, Y_4, X_4])
        diag[4] = np.array([0, 0, -Y_2, 0])
        diag[5] = np.array([0, X_2])

        k = np.diag(diag[0], k=0)

        for i in range(1, 6):
            k = k + np.diag(diag[i], k=-2*i) + np.diag(diag[i], k=2*i)

        # Change so rotations first is along
        perm = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

        k_perm = k[np.ix_(perm, perm)]

        return k_perm

    def get_K_st(self):

        # Get nodal stiffness
        k = self.get_stiff_mat_rect_3D()

        # Global stiffness matrix setup
        K_st = np.zeros((6*self.n_nd, 6*self.n_nd))

        for i in range(self.n_nd - 1):
            k_i = np.zeros((6*self.n_nd, 6*self.n_nd))
            k_i[i*6:i*6+12, i*6:i*6+12] = k
            K_st = K_st + k_i

        return K_st

    def get_M_nd(self):
        # Nodal masses
        m_e = self.m_e

        m = np.full(self.n_nd, m_e)
        m[-1], m[0] = m_e / 2, m_e / 2

        # Store nodal masses and lenghts
        self.m_nd = m

        block = []
        for i in range(self.n_nd):
            block.append(np.zeros((3, 3)))
            block.append(m[i] * np.eye(3, 3))

        M = la.block_diag(*block)

        return M

    def get_PI(self):

        # Fixed BC
        K_st = self.K_st[6:, 6:]
        M_nd = self.M_nd[6:, 6:]

        # Rearranging of M and K
        index = np.zeros((1, 0))

        for i in range(self.n_elem):
            index_add = np.linspace(i * 6 + 3, i * 6 + 5, 3).reshape(1, 3)
            index = np.hstack([index, index_add])

        for i in range(self.n_elem):
            index_add = np.linspace(i * 6, i * 6 + 2, 3).reshape(1, 3)
            index = np.hstack([index, index_add])

        index = index.flatten().astype(int)
        K = K_st[np.ix_(index, index)]
        M = M_nd[np.ix_(index, index)]

        # Find K_tt, K_rr, K_tr, K_rt and M_c
        sz = M.shape[0]
        sz2 = int(sz / 2)

        K_tt = K[0:sz2, 0:sz2]
        K_rr = K[sz2:sz, sz2:sz]
        K_tr = K[0:sz2, sz2:sz]
        K_rt = K[sz2:sz, 0:sz2]

        M_c = M[0:sz2, 0:sz2]

        # Find K_e (np.linalg.inv(K_rr) * K_rt)
        X = la.solve(K_rr, K_rt, assume_a="sym")
        K_c = K_tt - K_tr @ X

        print(np.linalg.norm(M_c))
        print(np.linalg.norm(K_c))

        # Solve eigenvalue problem for Pi_t (Mass normalized!)
        eigval, PI_t = la.eigh(K_c, M_c, subset_by_index=(0, self.n_md - 1))

        # Store eigen values
        self.eigval = eigval

        # Compute rotational part of PI
        PI_r = - X @ PI_t

        # Store PI_r and PI_t for modal integrals
        self.PI_r = np.vstack([np.zeros((3, self.n_md)), PI_r])
        self.PI_t = np.vstack([np.zeros((3, self.n_md)), PI_t])

        # PI setup
        PI = np.zeros((2 * PI_t.shape[0] + 6, PI_t.shape[1]))
        for i in range(self.n_elem):
            PI[i * 6 + 6:i * 6 + 9, :] = PI_r[i * 3:i * 3 + 3, :]
            PI[i * 6 + 9:i * 6 + 12, :] = PI_t[i * 3:i * 3 + 3, :]

        return PI

    def get_K_fl(self):
        # Initialize K
        K = np.zeros((self.n_md + 6, self.n_md + 6))
        K[0:self.n_md, 0:self.n_md] = self.PI.T @ self.K_st @ self.PI

        return K

    def get_Modal_Int(self):
        # Parameters
        m = self.m
        m_nd = self.m_nd.reshape(-1, 1)
        L_elem = self.L_elem
        PI_t = self.PI_t
        n_md = self.n_md
        n_nd = self.n_nd

        # Initialize sums
        p_0_sum = np.zeros((3, 1))
        CkJk_0_sum = np.zeros((3, 3))
        F_0_sum = np.zeros((3, n_md))
        G_0_sum = np.zeros((n_md, n_md))
        E_0_sum = np.zeros((3, n_md))

        for i in range(n_nd):
            # Parameters
            klkO = np.array([0, 0, i * L_elem]).reshape(3, 1)
            klkO_skew = sb.skew(klkO)

            # Compute sums
            p_0_sum += m_nd[i] * klkO

            CkJk_0_sum += - m_nd[i] * klkO_skew @ klkO_skew

            for r in range(n_md):
                F_0_sum[:, r] += m_nd[i] * \
                    klkO_skew @ PI_t[i * 3: i * 3 + 3, r]
                E_0_sum[:, r] += m_nd[i] * PI_t[i * 3: i*3 + 3, r]

                for s in range(n_md):
                    G_0_sum[r, s] += m_nd[i] * PI_t[i * 3: i *
                                                    3 + 3, r].T @ PI_t[i * 3: i*3 + 3, s]

        # Store modal integrals
        self.p_0 = 1/m * p_0_sum
        self.CkJk_0 = CkJk_0_sum
        self.F_0 = F_0_sum
        self.G_0 = G_0_sum
        self.E_0 = E_0_sum

    def get_M_fl(self):
        # Collect modal integrals
        self.get_Modal_Int()

        # Parameters
        p_0_skew = sb.skew(self.p_0)
        CkJk_0 = self.CkJk_0
        F_0 = self.F_0
        G_0 = self.G_0
        E_0 = self.E_0

        m = self.m

        # Build M
        rw1 = np.hstack([G_0, F_0.T, E_0.T])
        rw2 = np.hstack([F_0, CkJk_0, m * p_0_skew])
        rw3 = np.hstack([E_0, -m * p_0_skew, m * np.eye(3)])

        return np.vstack([rw1, rw2, rw3])

    def __init__(self, joint: Joint, rigid: Rigid_Properties, flex: Flex_Properties):
        self.w = float(joint.klOO[0].flatten()[0])
        self.h = float(joint.klOO[1].flatten()[0])
        self.L = float(joint.klOO[2].flatten()[0])
        self.A = rigid.w * rigid.h
        self.m = rigid.rho * self.A * self.L
        self.E = flex.E
        self.G = flex.G
        self.n_nd = flex.n_nd
        self.n_md = flex.n_md
        self.n_elem = self.n_nd - 1
        self.L_elem = self.L / self.n_elem
        self.m_e = rigid.rho * self.A * self.L_elem

        # Structural analysis
        self.K_st = self.get_K_st()
        self.M_nd = self.get_M_nd()
        self.PI = self.get_PI()
        self.K_fl = self.get_K_fl()
        self.M_fl = self.get_M_fl()
