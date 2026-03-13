import numpy as np
import scipy.linalg as la
import pandas as pd
from SOALIB import soalib as sb
from Body_Properties import Joint, Rigid_Properties, Flex_Properties


class Structural_Analysis_CB_Rect:
    """
    Class:
    Structural analysis of 3D rectangular cantilever beam with point mass assumption.

    Call:
    Structural_Analysis_PM_Rect(klOO, rho, E, G, n_nd, n_md)

    Constraint:
    Width, w >= Height, h
    """

    def get_stiff_mat_rect_3D(self):
        # b < a // h < w, w = y, h = z
        # Parameters
        w = self.w
        h = self.h
        a = w / 2
        b = h / 2
        L = self.L_elem
        E = self.E
        G = self.G

        # Book -> code
        # x -> z
        # y -> x
        # z -> y

        # Rectangular cross-section
        k_y = 1.2
        k_z = k_y
        K = a * b**3 * (16/3 - 3.36 * a / b * (1 - a**4 / (12 * b**4)))
        A = self.A
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

    def get_M_st(self):
        # Nodal masses
        m_e = self.m_e

        m = np.full(self.n_nd, m_e)
        m[-1], m[0] = m_e / 2, m_e / 2

        # Store nodal masses and lenghts
        self.m_nd = m
        
        J_x = 1/12 * self.h * self.w * (self.h**2 + self.w**2)
        I_y = 1/12 * self.h**3 * self.w
        I_z = 1/12 * self.h * self.w**3
        L_elem = self.L_elem
        A = self.A
        rho = self.rho

        # Stiffness matrix
        diag = [None] * 6

        diag[0] = np.array([1/3, 13/35+6*I_z/(5*A*L_elem**2), 13/35+6*I_y/(5*A*L_elem**2), J_x/(3*A), L_elem**2/105+2*I_y/(15*A), L_elem**2/105+2*I_z/(15*A), 1/3, 13/35+6*I_z/(5*A*L_elem**2), 13/35+6*I_y/(5*A*L_elem**2), J_x/(3*A), L_elem**2/105+2*I_y/(15*A), L_elem**2/105+2*I_z/(15*A)])
        diag[1] = np.array([0, 0, -11*L_elem/210-I_y/(10*A*L_elem), 0, 0, 13*L_elem/420-I_z/(10*A*L_elem), 0, 0, 11*L_elem/210+I_y/(10*A*L_elem), 0])
        diag[2] = np.array([0, 11*L_elem/210+I_z/(10*A*L_elem), 0, 0, -13*L_elem/420+I_y/(10*A*L_elem), 0, 0, -11*L_elem/210-I_z/(10*A*L_elem)])
        diag[3] = np.array([1/6, 9/70-6*I_z/(5*A*L_elem**2), 9/70-6*I_y/(5*A*L_elem**2), J_x/(6*A), -L_elem**2/140-I_y/(30*A), -L_elem**2/140-I_z/(30*A)])
        diag[4] = np.array([0, 0, 13*L_elem/420-I_y/(10*A*L_elem), 0])
        diag[5] = np.array([0, -13*L_elem/420+I_z/(10*A*L_elem)])

        M = np.diag(diag[0], k=0)

        for i in range(1, 6):
            M += np.diag(diag[i], k=-2*i) + np.diag(diag[i], k=2*i)
        
        M = rho * A * L_elem * M

        # Change so rotations first is along
        perm = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        M_perm = M[np.ix_(perm, perm)]

        # Global stiffness matrix setup
        M_st = np.zeros((6*self.n_nd, 6*self.n_nd))

        for i in range(self.n_nd - 1):
            M_i = np.zeros((6*self.n_nd, 6*self.n_nd))
            M_i[i*6:i*6+12, i*6:i*6+12] = M_perm
            M_st += M_i

        return M_st

    def get_PI(self):

        M = self.M_st
        K = self.K_st
        # Indexing B(boundary) I(interior) Follows Adams flex notation
        boundary_nodes = [0]          # Cantilever beam
        B = []
        for i in boundary_nodes:
            B.extend(range(6*i, 6*i+6))
        
        all_dofs = list(range(6*self.n_nd))
        I = [k for k in all_dofs if k not in B]
        K_BB = K[np.ix_(B, B)]
        K_BI = K[np.ix_(B, I)]
        K_IB = K[np.ix_(I, B)]
        K_II = K[np.ix_(I, I)]
        M_II = M[np.ix_(I, I)]

        # We compute constraint modes from static deformation shape
        PI_b = - la.solve(K_II, K_IB)

        # Solve eigenvalue problem for Pi_t (Mass normalized!)
        eig_e, PI_e = la.eigh(K_II, M_II, subset_by_index=(0, self.n_md - 1))

        # Extract PI_e
        self.PI_e = np.vstack([np.zeros((6, self.n_md)), PI_e])

        PI_c = np.block([
        [np.eye(6), np.zeros((6, self.n_md))],
        [PI_b, PI_e]])

        M_n = PI_c.T @ M @ PI_c
        K_n = PI_c.T @ K @ PI_c

        omega2, PI_n = la.eigh(K_n, M_n)

        # Store eigenvalues
        self.omega2 = omega2[6:]
        self.omega = np.sqrt(self.omega2)

        """
        print("PI_n")
        print(pd.DataFrame(PI_n))
        print("PI_c")
        print(pd.DataFrame(PI_c))
        """
        
        PI_n = PI_n[:, 6:]

        PI = PI_c @ PI_n

        # Extract lambda_ and gamma
        self.gamma = np.zeros((3*self.n_nd, self.n_md))
        self.lambda_ = np.zeros((3*self.n_nd, self.n_md))
        for i in range(self.n_nd):
            self.lambda_[i*3:i*3+3, :] = self.PI_e[i*6:i*6+3, :]
            self.gamma[i*3:i*3+3, :] = self.PI_e[i*6+3:i*6+6, :]

        return self.PI_e

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
        n_md = self.n_md
        n_nd = self.n_nd
        lambda_ = self.lambda_
        gamma = self.gamma
        #J_nd = m_nd * np.array([1/12 * (self.w**2 + self.h**2), 1/12 * (self.w**2 + self.h**2)])
        #p_nd = 

        # Initialize sums
        p_0_sum = np.zeros((3, 1))
        p_1_sum = np.zeros((3, n_md))
        CkJk_0_sum = np.zeros((3, 3))
        CkJk_1_sum = np.zeros((3, 3*n_md))
        CkJk_2_sum = np.zeros((3*n_md, 3*n_md))
        F_0_sum = np.zeros((3, n_md))
        F_1_sum = np.zeros((3*n_md, n_md))
        G_0_sum = np.zeros((n_md, n_md))
        E_0_sum = np.zeros((3, n_md))

        for i in range(n_nd):
            # Parameters
            klkO = np.array([i * L_elem, 0, 0]).reshape(3, 1)
            klkO_skew = sb.skew(klkO)

            # Compute sums
            p_0_sum += m_nd[i] * klkO

            CkJk_0_sum += - m_nd[i] * klkO_skew @ klkO_skew

            for r in range(n_md):
                F_0_sum[:, r] += m_nd[i] * \
                    klkO_skew @ gamma[i * 3: i * 3 + 3, r]
                E_0_sum[:, r] += m_nd[i] * gamma[i * 3: i*3 + 3, r]
                p_1_sum[:, r] += m_nd[i] * gamma[i * 3: i*3 + 3, r]
                CkJk_1_sum[:, 3 * r: 3 * r + 3] += m_nd[i] * \
                    sb.skew(gamma[i * 3: i*3 + 3, r]) @ klkO_skew
                for s in range(n_md):
                    G_0_sum[r, s] += m_nd[i] * gamma[i * 3: i * 3 + 3, r].T @ gamma[i * 3: i*3 + 3, s]
                    CkJk_2_sum[3*r:3*r+3, 3*s:3*s+3] += m_nd[i] * sb.skew(gamma[i * 3: i *
                                                                               3 + 3, r]) @ sb.skew(gamma[i * 3: i*3 + 3, s])
                    F_1_sum[3*r:3*r+3, s] += m_nd[i] * sb.skew(gamma[i * 3: i *
                                                                    3 + 3, r]) @ gamma[i * 3: i*3 + 3, s]

        # Store modal integrals
        self.p_0 = 1/m * p_0_sum
        self.p_1 = 1/m * p_1_sum
        self.CkJk_0 = CkJk_0_sum
        #self.CkJk_0[0, 0] = self.CkJk[0]
        self.CkJk_1 = - CkJk_1_sum
        self.CkJk_2 = - CkJk_2_sum
        self.F_0 = F_0_sum
        self.F_1 = - F_1_sum
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

    def get_C_fl(self):
        
        # Damping setup
        zeta = self.c * np.ones(self.n_md)

        C_eta = np.diag(2.0 * zeta * self.omega)

        C_fl = np.zeros_like(self.M_fl)
        C_fl[:self.n_md, :self.n_md] = C_eta

        return C_fl

    def __init__(self, joint: Joint, rigid: Rigid_Properties, flex: Flex_Properties):
        self.klOC = joint.klOC
        self.w = rigid.w
        self.h = rigid.h
        self.L = rigid.L
        self.A = rigid.w * rigid.h
        self.rho = rigid.rho
        self.m = self.rho * self.A * self.L
        self.E = flex.E
        self.G = flex.G
        self.c = flex.c
        self.n_nd = flex.n_nd
        self.n_md = flex.n_md
        self.CkJk = rigid.CkJk
        self.n_elem = self.n_nd - 1
        self.L_elem = self.L / self.n_elem
        self.m_e = self.rho * self.A * self.L_elem

        # Structural analysis
        self.K_st = self.get_K_st()
        self.M_st = self.get_M_st()
        self.PI = self.get_PI()
        """
        print("PI")
        print(pd.DataFrame(self.PI))
        print("omega")
        print(pd.DataFrame(self.omega))
        """
        self.K_fl = self.get_K_fl()
        self.M_fl = self.get_M_fl()
        self.C_fl = self.get_C_fl()
