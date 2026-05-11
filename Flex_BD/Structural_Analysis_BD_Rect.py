import numpy as np
import scipy.linalg as la
import pandas as pd
from SOALIB import soalib as sb
from Body_Properties import Joint, Rigid_Properties, Flex_Properties


class Structural_Analysis_BD_Rect:
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

        diag[0] = np.array(
            [X, Y_1, Z_1, S, Z_3, Y_3, X, Y_1, Z_1, S, Z_3, Y_3])
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
        L_e = self.L_elem
        m_e = self.m_e
        n_nd = self.n_nd

        # nodal masses
        m = np.full(self.n_nd, m_e)
        m[0] = m_e / 2
        m[-1] = m_e / 2
        self.m_nd = m

        M_blocks = []
        J_list = [None] * n_nd
        p_list = [None] * n_nd

        for i in range(n_nd):
            p = np.zeros((3, 1))
            if i == 0:
                p = np.array([(1 / 4) * L_e, 0, 0]
                             ).reshape(3, 1)   # left end node
            elif i == n_nd - 1:
                p = np.array([- (1 / 4) * L_e, 0, 0]
                             ).reshape(3, 1)   # right end node

            # centroidal inertia of assigned nodal lump
            # interior nodes get full element length, end nodes get half length
            L_slice = L_e if (0 < i < n_nd - 1) else L_e / 2

            Jc = (m[i] / 12.0) * np.diag([
                self.w**2 + self.h**2,    # about x
                L_slice**2 + self.h**2,   # about y
                L_slice**2 + self.w**2    # about z
            ])

            # shift centroidal inertia to node origin: J = Jc - m skew(p)@skew(p)
            J = Jc - m[i] * sb.skew(p) @ sb.skew(p)

            # full 6x6 spatial inertia block
            Mj = np.block([
                [J,              m[i] * sb.skew(p)],
                [-m[i] * sb.skew(p), m[i] * np.eye(3)]
            ])

            M_blocks.append(Mj)
            J_list[i] = J
            p_list[i] = p

        self.J_list = J_list
        self.p_list = p_list

        M_st = la.block_diag(*M_blocks)
        return M_st

    def identify_mode_labels(self):
        labels = []

        for r in range(self.PI_e.shape[1]):
            pie = self.PI_e[:, r]

            rx = pie[0::6]   # rot_x
            ry = pie[1::6]   # rot_y
            rz = pie[2::6]   # rot_z
            ux = pie[3::6]   # x translation
            uy = pie[4::6]   # y translation
            uz = pie[5::6]   # z translation

            torsion_x = np.linalg.norm(self.L * rx)
            axial_x = np.linalg.norm(ux)
            bending_xy = np.linalg.norm(uy) + np.linalg.norm(self.L * rz)
            bending_xz = np.linalg.norm(uz) + np.linalg.norm(self.L * ry)

            scores = {
                "torsion_x": torsion_x,
                "axial_x": axial_x,
                "bending_xy": bending_xy,
                "bending_xz": bending_xz,
            }

            label = max(scores, key=scores.get)

            labels.append({
                "mode": r + 1,
                "freq_hz": self.omega[r] / (2*np.pi),
                "label": label,
                "score": scores[label],
            })

        return labels

    def get_PI(self):
        # Parameters
        n_nd = self.n_nd
        n_md_compute = self.n_md_compute
        mode_selection = self.mode_selection
        M_st = self.M_st
        K_st = self.K_st

        # Zero modes (Rigid body)
        if n_md_compute == 0:
            self.PI_e = np.zeros((6 * n_nd, 0))
            self.PI = self.PI_e
            self.omega2 = np.array([])
            self.omega = np.array([])
            self.modes = []
            self.n_md = 0
            self.gamma = np.zeros((3 * n_nd, 0))
            self.lambda_ = np.zeros((3 * n_nd, 0))
            return self.PI_e

        # Apply boundary conditions (fixed at root node 0)
        bnd_nodes = [0]
        dof_bnd = []
        for i in bnd_nodes:
            dof_bnd.extend(range(6 * i, 6 * i + 6))

        dof_all = list(range(6 * n_nd))
        dof_int = [i for i in dof_all if i not in dof_bnd]

        # Interior partition of mass and stiffness matrices
        K_int = K_st[np.ix_(dof_int, dof_int)]
        M_int = M_st[np.ix_(dof_int, dof_int)]

        # Compute candidate fixed-boundary modes
        n_md_cand = min(n_md_compute, K_int.shape[0])

        # Solve generalized eigenvalue problem for interior DOFs
        eig_val, PI_int = la.eigh(
            K_int, M_int,
            subset_by_index=(0, n_md_cand - 1)
        )

        # Reconstruct full mode shape matrix (zero displacement at boundary)
        PI_full = np.vstack([np.zeros((6, n_md_cand)), PI_int])

        # Temporary full storage used for mode classification
        self.PI_e = PI_full
        self.omega2 = eig_val
        self.omega = np.sqrt(np.maximum(eig_val, 0.0))

        # Identify mode labels
        all_modes = self.identify_mode_labels()

        # Mode selection
        if mode_selection is not None:
            used_count = {label: 0 for label in mode_selection}
            keep_idx = []

            for i, mode in enumerate(all_modes):
                label = mode["label"]
                if label in mode_selection and used_count[label] < mode_selection[label]:
                    keep_idx.append(i)
                    used_count[label] += 1
        else:
            keep_idx = list(range(n_md_cand))

        if len(keep_idx) == 0:
            raise ValueError("No modes matched the selection criteria!")

        # Reduce all modal quantities to the selected set
        self.PI_e = PI_full[:, keep_idx]
        self.PI = self.PI_e
        self.omega2 = eig_val[keep_idx]
        self.omega = np.sqrt(np.maximum(self.omega2, 0.0))
        self.modes = [all_modes[i] for i in keep_idx]
        self.n_md = len(keep_idx)

        # Rebuild lambda and gamma mode matrices using the selected modes only
        self.gamma = np.zeros((3 * n_nd, self.n_md))
        self.lambda_ = np.zeros((3 * n_nd, self.n_md))

        for i in range(n_nd):
            self.lambda_[i * 3: i * 3 + 3, :] = self.PI_e[i * 6: i * 6 + 3, :]
            self.gamma[i * 3: i * 3 + 3,
                       :] = self.PI_e[i * 6 + 3: i * 6 + 6, :]

        return self.PI_e

    def get_K_fl(self):
        # Initialize K
        K = np.zeros((self.n_md + 6, self.n_md + 6))
        K[0:self.n_md, 0:self.n_md] = self.PI.T @ self.K_st @ self.PI

        return K

    def get_Modal_Int(self):
        # Parameters
        m = self.m
        m_nd = self.m_nd  # .reshape(-1, 1)
        L_elem = self.L_elem
        n_md = self.n_md
        n_nd = self.n_nd
        lambda_ = self.lambda_
        gamma = self.gamma
        J = self.J_list
        p = self.p_list

        # Initialize sums
        p_0_sum = np.zeros((3, 1))
        p_1_sum = np.zeros((3, n_md))
        J_0_sum = np.zeros((3, 3))
        J_1_sum = np.zeros((3, 3*n_md))
        J_2_sum = np.zeros((3*n_md, 3*n_md))
        F_0_sum = np.zeros((3, n_md))
        F_1_sum = np.zeros((3*n_md, n_md))
        G_0_sum = np.zeros((n_md, n_md))
        E_0_sum = np.zeros((3, n_md))
        S_1_sum = np.zeros((3, 3*n_md))

        for i in range(n_nd):
            # Parameters
            klkO = np.array([i * L_elem, 0, 0]).reshape(3, 1)
            klkO_skew = sb.skew(klkO)
            p_skew = sb.skew(p[i])

            # Compute sums
            p_0_sum += m_nd[i] * (p[i] + klkO)
            J_0_sum += J[i] - m_nd[i] * \
                (klkO_skew @ klkO_skew + p_skew @ klkO_skew + klkO_skew @ p_skew)

            for r in range(n_md):
                gamma_r = gamma[i * 3: i * 3 + 3, r]
                lambda_r = lambda_[i * 3: i * 3 + 3, r]

                F_0_sum[:, r] += J[i] @ lambda_r + m_nd[i] * \
                    (klkO_skew + p_skew) @ gamma_r - \
                    m_nd[i] * klkO_skew @ p_skew @ lambda_r
                E_0_sum[:, r] += m_nd[i] * (gamma_r - p_skew @ lambda_r)
                p_1_sum[:, r] += m_nd[i] * gamma_r
                J_1_sum[:, 3 * r: 3 * r + 3] += m_nd[i] * \
                    sb.skew(gamma_r) @ (klkO_skew + p_skew)
                S_1_sum[:, 3 * r: 3 * r + 3] += sb.skew(
                    m_nd[i] * p_skew @ lambda_r) @ klkO_skew - J[i] @ sb.skew(lambda_r)
                for s in range(n_md):
                    gamma_s = gamma[i * 3: i*3 + 3, s]
                    lambda_s = lambda_[i * 3: i * 3 + 3, s]

                    G_0_sum[r, s] += lambda_r.T @ J[i] @ lambda_s + m_nd[i] * (
                        lambda_r.T @ p_skew @ gamma_s + lambda_s.T @ p_skew @ gamma_r + gamma_r.T @ gamma_s)
                    # J_2_sum[3*r:3*r+3, 3*s:3*s+3] += m_nd[i] * sb.skew(gamma_r) @ sb.skew(gamma_s)
                    # F_1_sum[3*r:3*r+3, s] += m_nd[i] * sb.skew(gamma_r) @ gamma_s

        # Store modal integrals
        self.p_0 = 1/m * p_0_sum
        self.p_1 = 1/m * p_1_sum
        self.J_0 = J_0_sum
        self.J_1 = J_1_sum
        # self.J_2 = - J_2_sum
        self.F_0 = F_0_sum
        self.F_1 = F_1_sum
        self.G_0 = G_0_sum
        self.E_0 = E_0_sum
        self.S_1 = S_1_sum

    def get_M_fl(self):
        # Collect modal integrals
        self.get_Modal_Int()

        # Parameters
        p_0_skew = sb.skew(self.p_0)
        J_0 = self.J_0
        F_0 = self.F_0
        G_0 = self.G_0
        E_0 = self.E_0

        m = self.m

        # Build M
        rw1 = np.hstack([G_0, F_0.T, E_0.T])
        rw2 = np.hstack([F_0, J_0, m * p_0_skew])
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
        self.klOO = joint.klOO
        self.klOC = joint.klOC
        self.w = rigid.w
        self.h = rigid.h
        self.L = rigid.L
        self.A = rigid.A
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

        # Mode selection
        self.n_md_compute = flex.n_md
        self.mode_selection = flex.mode_selection

        # Structural analysis
        self.K_st = self.get_K_st()
        self.M_st = self.get_M_st()
        self.PI = self.get_PI()
        self.K_fl = self.get_K_fl()
        self.M_fl = self.get_M_fl()
        self.C_fl = self.get_C_fl()
