# merged_flex_soa.py
"""
Merged flexible-body SOA implementation.

Generated from:
    - Body_Properties.py
    - SystemState.py
    - Structural_Analysis_BD_Rect.py
    - SOABody.py
    - ATBI_Flex_Grav.py
    - MultibodySystem.py
    - Simulation.py

Notes:
    - SOALIB helper functions are intentionally not duplicated.
    - Keep your existing SOALIB package available on the Python path.
    - This merged file uses: from SOALIB import soalib as sb
"""

import os

import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from SOALIB import soalib as sb

# Increase limit to 100 MB+ for embedded animations.
plt.rcParams["animation.embed_limit"] = 1000




# =============================================================================
# Body_Properties.py
# =============================================================================

class Joint:
    # Joint class with H_type, H and klOO
    def __init__(self, klOO, H_type: str):
        # Parameters
        self.type = H_type
        self.H = sb.hinge_map(H_type)
        self.klOO = klOO
        self.L = np.linalg.norm(klOO)
        self.klOC = klOO / 2

    # Unpacking size
    def theta_size(self):
        return {
            "revx": 1,
            "revy": 1,
            "revz": 1,
            "spherical": 4,
            "free": 7,
            "fixed": 0
        }[self.type]

    def beta_size(self):
        return {
            "revx": 1,
            "revy": 1,
            "revz": 1,
            "spherical": 3,
            "free": 6,
            "fixed": 0
        }[self.type]


class Rigid_Properties:
    # Inertia class with m, CkJk and klOC
    def __init__(self, rho, w, h):
        # Parameters
        self.rho = rho
        self.CkJk = [None]
        self.Mk = [None]
        self.w = w
        self.h = h
        self.L = [None]

    def get_Mk(self, m, CkJk, klOC):
        CkJk = self.CkJk

        # Rigid spatial inertia
        MC = np.block([
            [np.diag(CkJk), np.zeros((3, 3))],
            [np.zeros((3, 3)), m * np.eye(3)]
        ])
        return sb.phi(klOC) @ MC @ sb.phi(klOC).T


class Flex_Properties:
    def __init__(self, E, G, c, n_nd, n_md, selected_mode_labels=None, mode_quota=None):
        self.E = E
        self.G = G
        self.c = c
        self.n_nd = n_nd
        # n_md is the number of candidate modes to compute before optional selection.
        # If a selector is used, this value is overwritten by the number of kept modes.
        self.n_md = n_md
        self.n_elem = self.n_nd - 1

        # Optional mode selection:
        #   selected_mode_labels=["bending_xy", "bending_xz"] keeps all computed modes
        #       with those labels.
        #   mode_quota={"bending_xy": 3, "bending_xz": 3, "axial_x": 0}
        #       keeps the first N modes per label, ordered by frequency.
        self.selected_mode_labels = selected_mode_labels
        self.mode_quota = mode_quota
        self.modes = []

        self.K_fl = [None]
        self.M_fl = [None]
        self.C_fl = [None]
        self.omega2 = [None]
        self.omega = [None]
        self.PI = [None]
        self.PI_end = [None]

    def set_PI(self, PI):
        self.PI = PI

    def set_K_fl(self, K_fl):
        self.K_fl = K_fl

    def set_M_fl(self, M_fl):
        self.M_fl = M_fl


# =============================================================================
# SystemState.py
# =============================================================================

class SystemState:
    # State of system class
    def __init__(self, theta, beta, eta, eta_dot):
        # Parameters
        self.Theta = theta
        self.Beta = beta
        self.Eta = eta
        self.Eta_dot = eta_dot

    # Packing of state, S: Four lists to column vector
    def pack(self):
        return np.vstack([*self.Theta, *self.Beta, *self.Eta, *self.Eta_dot]).flatten()

    # Unpacking of state, S: Column vector to two lists
    @staticmethod
    def unpack(S, joints, flexs):
        S = S.flatten()
        Theta, Beta, Eta, Eta_dot = [], [], [], []
        idx = 0

        for k in joints:
            sz = k.theta_size()
            Theta.append(S[idx:idx + sz].reshape(sz, 1))
            idx += sz

        for k in joints:
            sz = k.beta_size()
            Beta.append(S[idx:idx + sz].reshape(sz, 1))
            idx += sz

        for k in flexs:
            sz = k.n_md
            Eta.append(S[idx:idx + sz].reshape(sz, 1))
            idx += sz

        for k in flexs:
            sz = k.n_md
            Eta_dot.append(S[idx:idx + sz].reshape(sz, 1))
            idx += sz

        return SystemState(Theta, Beta, Eta, Eta_dot)


# =============================================================================
# Structural_Analysis_BD_Rect.py
# =============================================================================

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
                p = np.array([(1 / 4) * L_e, 0, 0]).reshape(3, 1)   # left end node
            elif i == n_nd - 1:
                p = np.array([ - (1 / 4) * L_e, 0, 0]).reshape(3, 1)   # right end node

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

    def get_PI(self):
        M = self.M_st
        K = self.K_st

        boundary_nodes = [0]          # Cantilever beam root node
        B = []
        for i in boundary_nodes:
            B.extend(range(6*i, 6*i+6))

        all_dofs = list(range(6*self.n_nd))
        I = [k for k in all_dofs if k not in B]

        K_II = K[np.ix_(I, I)]
        M_II = M[np.ix_(I, I)]

        # Compute candidate fixed-boundary modes.
        n_candidates = min(self.n_md_compute, K_II.shape[0])
        if n_candidates <= 0:
            raise ValueError("n_md must be at least 1 before mode selection.")

        eig_e, PI_e_int = la.eigh(
            K_II, M_II,
            subset_by_index=(0, n_candidates - 1)
        )

        # Full nodal vector: root boundary DOFs are fixed, so they are zero.
        PI_e_full = np.vstack([np.zeros((6, n_candidates)), PI_e_int])

        # Temporary full storage is used for mode classification.
        self.PI_e = PI_e_full
        self.omega2 = eig_e
        self.omega = np.sqrt(np.maximum(eig_e, 0.0))

        all_modes = self.identify_mode_labels()

        # Choose which modes to keep. The eigenvectors are frequency ordered by la.eigh.
        if self.mode_quota is not None:
            used_count = {label: 0 for label in self.mode_quota}
            keep = []

            for i, mode in enumerate(all_modes):
                label = mode["label"]
                if label in self.mode_quota and used_count[label] < self.mode_quota[label]:
                    keep.append(i)
                    used_count[label] += 1

        elif self.selected_mode_labels is not None:
            selected = set(self.selected_mode_labels)
            keep = [
                i for i, mode in enumerate(all_modes)
                if mode["label"] in selected
            ]

        else:
            keep = list(range(n_candidates))

        if len(keep) == 0:
            raise ValueError("No modes matched the selected labels / quota.")

        # Reduce all modal quantities to the selected set.
        self.PI_e = PI_e_full[:, keep]
        self.PI = self.PI_e
        self.omega2 = eig_e[keep]
        self.omega = np.sqrt(np.maximum(self.omega2, 0.0))
        self.modes = [all_modes[i] for i in keep]
        self.n_md = len(keep)

        # Rebuild lambda and gamma using the selected modes only.
        self.gamma = np.zeros((3*self.n_nd, self.n_md))
        self.lambda_ = np.zeros((3*self.n_nd, self.n_md))

        for i in range(self.n_nd):
            self.lambda_[i*3:i*3+3, :] = self.PI_e[i*6:i*6+3, :]
            self.gamma[i*3:i*3+3, :] = self.PI_e[i*6+3:i*6+6, :]

        return self.PI_e

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

            # Simple amplitude measures. Rotations are multiplied by L so
            # their scale is comparable to translational amplitudes.
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

    def get_K_fl(self):
        # Initialize K
        K = np.zeros((self.n_md + 6, self.n_md + 6))
        K[0:self.n_md, 0:self.n_md] = self.PI.T @ self.K_st @ self.PI

        return K

    def get_Modal_Int(self):
        # Parameters
        m = self.m
        m_nd = self.m_nd #.reshape(-1, 1)
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
        S_2_sum = np.zeros((3*n_md, 3*n_md))
        R_1_sum = np.zeros((3, n_md))
        K_1_sum = np.zeros((3*n_md, 3))
        K_2_sum = np.zeros((3*n_md, 3*n_md))
        T_1_sum = np.zeros((3*n_md, n_md))
        T_2_sum = np.zeros((3*n_md, n_md))
        W_1_sum = np.zeros((3*n_md, n_md))
        W_2_sum = np.zeros((3*n_md, n_md))

        for i in range(n_nd):
            # Parameters
            klkO = np.array([i * L_elem, 0, 0]).reshape(3, 1)
            klkO_skew = sb.skew(klkO)
            p_skew = sb.skew(p[i])

            # Compute sums
            p_0_sum += m_nd[i] * (p[i] + klkO)
            J_0_sum += J[i] - m_nd[i] * (klkO_skew @ klkO_skew + p_skew @ klkO_skew + klkO_skew @ p_skew)

            for r in range(n_md):
                gamma_r = gamma[i * 3: i * 3 + 3, r]
                lambda_r = lambda_[i * 3: i * 3 + 3, r]

                F_0_sum[:, r] += J[i] @ lambda_r + m_nd[i] * (klkO_skew + p_skew) @ gamma_r - m_nd[i] * klkO_skew @ p_skew @ lambda_r
                E_0_sum[:, r] += m_nd[i] * (gamma_r - p_skew @ lambda_r)
                p_1_sum[:, r] += m_nd[i] * gamma_r
                J_1_sum[:, 3 * r: 3 * r + 3] += m_nd[i] * sb.skew(gamma_r) @ (klkO_skew + p_skew)
                S_1_sum[:, 3 * r: 3 * r + 3] += sb.skew(m_nd[i] * p_skew @ lambda_r) @ klkO_skew - J[i] @ sb.skew(lambda_r)
                R_1_sum[:, r] += J[i] @ lambda_r
                K_1_sum[3*r:3*r+3, :] += 2*klkO_skew@ sb.skew(m_nd[i]*p_skew@lambda_r)- J[i]@sb.skew(lambda_r)+ sb.skew(lambda_r)@J[i]

                for s in range(n_md):
                    gamma_s = gamma[i * 3: i*3 + 3, s]
                    lambda_s = lambda_[i * 3: i * 3 + 3, s]

                    G_0_sum[r, s] += lambda_r.T @ J[i] @ lambda_s + m_nd[i] * (lambda_r.T @ p_skew @ gamma_s + lambda_s.T @ p_skew @ gamma_r+ gamma_r.T @ gamma_s)
                    J_2_sum[3*r:3*r+3, 3*s:3*s+3] += m_nd[i] * sb.skew(gamma_r) @ sb.skew(gamma_s)
                    F_1_sum[3*r:3*r+3, s] += m_nd[i] * sb.skew(gamma_s) @ (gamma_r-p_skew @ lambda_r)
                    S_2_sum[3*r:3*r+3, 3*s:3*s+3] += sb.skew(m_nd[i]* p_skew @ lambda_r ) @ sb.skew(gamma_s)
                    K_2_sum[3*r:3*r+3, 3*s:3*s+3] += 2* sb.skew(gamma_s) @ sb.skew(m_nd[i]* p_skew @ lambda_r )
                    T_1_sum[3*r:3*r+3, s] += (m_nd[i]*sb.skew(gamma_r)@p_skew + J[i]@sb.skew(lambda_r) )@lambda_s
                    T_2_sum[3*r:3*r+3, s] += (m_nd[i]*p_skew@ sb.skew(gamma_r)-sb.skew(lambda_r)@J[i] )@lambda_s
                    W_1_sum[3*r:3*r+3, s] += m_nd[i]*sb.skew(lambda_r)@p_skew@gamma_s
                    W_2_sum[3*r:3*r+3, s] += sb.skew(lambda_r)@ J[i]@ lambda_s   

        # Store modal integrals
        self.p_0 = 1/m * p_0_sum
        self.p_1 = 1/m * p_1_sum
        self.J_0 = -J_0_sum
        self.J_1 = -J_1_sum #-
        self.J_2 = -J_2_sum
        self.F_0 = F_0_sum
        self.F_1 = F_1_sum
        self.G_0 = G_0_sum
        self.E_0 = E_0_sum
        self.S_1 = S_1_sum
        self.S_2 = S_2_sum
        self.R_1 = R_1_sum
        self.K_1 = K_1_sum
        self.K_2 = K_2_sum
        self.T_1 = T_1_sum
        self.T_2 = T_2_sum
        self.W_1 = W_1_sum
        self.W_2 = W_2_sum

    def get_M_fl(self):
        self.get_Modal_Int()

        p_0_skew = sb.skew(self.p_0)

        J_0 = self.J_0
        F_0 = self.F_0
        G_0 = self.G_0
        E_0 = self.E_0

        m = self.m

        rw1 = np.hstack([G_0, F_0.T, E_0.T])
        rw2 = np.hstack([F_0, J_0, m * p_0_skew])
        rw3 = np.hstack([E_0, -m * p_0_skew, m * np.eye(3)])

        return np.vstack([rw1, rw2, rw3])

    def get_C_fl(self):

        # Damping setup
        zeta = self.c * np.ones(self.n_md)

        C_eta = np.diag(2.0 * zeta * self.omega)

        C_fl = np.zeros_like(self.K_fl)
        C_fl[:self.n_md, :self.n_md] = C_eta

        return C_fl

    def __init__(self, joint: Joint, rigid: Rigid_Properties, flex: Flex_Properties):
        self.klOO = joint.klOO
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
        # Keep the requested number as candidate count. self.n_md is later
        # overwritten by the number of selected modes.
        self.n_md_compute = flex.n_md
        self.selected_mode_labels = flex.selected_mode_labels
        self.mode_quota = flex.mode_quota
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


# =============================================================================
# SOABody.py
# =============================================================================

# Increase limit to 100 MB (default is 20)

class SOABody:
    # SOAbody class
    class Force:
        def __init__(self, joint: Joint):
            self.tau = np.zeros((joint.beta_size(), 1))
            # list of external nodal loads
            # each load is {"node": int, "fun": callable}
            self.external_loads = []

    class InitialCondition:
        def __init__(self, joint: Joint, flex: Flex_Properties):
            # Setup of initial conditions (assumes identity rotation and no initial velocity)
            self.theta0 = np.zeros((joint.theta_size(), 1))
            if np.size(self.theta0) == 4:
                self.theta0[-1] = 1
            elif np.size(self.theta0) == 7:
                self.theta0[3] = 1
            self.beta0 = np.zeros((joint.beta_size(), 1))

            # Setup of initial conditions for eta and eta_dot
            self.eta0 = np.zeros((flex.n_md, 1))
            self.eta_dot0 = np.zeros((flex.n_md, 1))

    def __init__(self, joint: Joint, rigid: Rigid_Properties, flex: Flex_Properties):
        # Import classes
        self.joint = joint
        self.rigid = rigid
        self.flex = flex
        self.force = self.Force(self.joint)
        rigid.A = rigid.h * rigid.w
        rigid.L = joint.L
        flex.L_elem = joint.L / flex.n_elem
        flex.klOO_nd = [j * (joint.klOO / flex.n_elem) for j in range(flex.n_nd)]
        self.m = rigid.rho * rigid.A * joint.L
        self.rigid.CkJk = np.array([1/12 * self.m * (rigid.h**2 + rigid.w**2), 1/12 * self.m * (
            rigid.h**2 + joint.L**2), 1/12 * self.m * (rigid.w**2 + joint.L**2)])
        rigid.Mk = rigid.get_Mk(self.m, self.rigid.CkJk, self.joint.klOC)

        # Structural analysis is PI == [None] (Point mass: Rectangular cross section)
        if self.flex.PI == [None]:
            body_analysis = Structural_Analysis_BD_Rect(joint, rigid, flex)

            """
            print("p_0")
            print(pd.DataFrame(body_analysis.p_0))
            print("p_1")
            print(pd.DataFrame(body_analysis.p_1))
            print("CkJk_0")
            print(pd.DataFrame(body_analysis.CkJk_0))
            print("CkJk_1")
            print(pd.DataFrame(body_analysis.CkJk_1))
            print("CkJk_2")
            print(pd.DataFrame(body_analysis.CkJk_2))
            print("F_0")
            print(pd.DataFrame(body_analysis.F_0))
            print("F_1")
            print(pd.DataFrame(body_analysis.F_1))
            print("G_0")
            print(pd.DataFrame(body_analysis.G_0))
            print("E_0")
            print(pd.DataFrame(body_analysis.E_0))
            print("omega^2")
            print(pd.DataFrame(body_analysis.omega2))
            print("PI_t")
            print(pd.DataFrame(body_analysis.PI_t))
            print("K_st")
            print(pd.DataFrame(body_analysis.K_st))
            """

            # PI
            self.flex.PI = body_analysis.PI
            self.flex.PI_e = body_analysis.PI_e
            self.flex.PI_end = body_analysis.PI[-6:, :]
            self.flex.omega2 = body_analysis.omega2
            self.flex.omega = body_analysis.omega
            self.flex.modes = body_analysis.modes
            self.flex.n_md = body_analysis.n_md

            # Stiffness, damping and mass matrix
            self.flex.K_fl = body_analysis.K_fl
            self.flex.C_fl = body_analysis.C_fl

            # Modal integral for gyroscopic force
            self.flex.F_0 = body_analysis.F_0
            self.flex.G_0 = body_analysis.G_0
            self.flex.p_0 = body_analysis.p_0
            self.flex.S_1 = body_analysis.S_1
            self.flex.S_2 = body_analysis.S_2
            self.flex.F_1 = body_analysis.F_1
            self.flex.J_0 = body_analysis.J_0
            self.flex.J_1 = body_analysis.J_1
            self.flex.J_2 = body_analysis.J_2
            self.flex.T_1 = body_analysis.T_1
            self.flex.T_2 = body_analysis.T_2
            self.flex.W_1 = body_analysis.W_1
            self.flex.W_2 = body_analysis.W_2
            self.flex.p_1 = body_analysis.p_1
            self.flex.E_0 = body_analysis.E_0
            self.flex.R_1 = body_analysis.R_1
            self.flex.K_1 = body_analysis.K_1
            self.flex.K_2 = body_analysis.K_2



        # Initial conditions must be created after optional mode selection,
        # because flex.n_md may have changed.
        self.initialcondition = self.InitialCondition(joint, self.flex)

        # D_m invers (offline)
        H_M_fl = np.hstack([np.eye(self.flex.n_md, self.flex.n_md), np.zeros((self.flex.n_md, 6))])
        A_fl = sb.get_A(self.flex.PI_end, self.joint.klOO)
        #self.flex.L_fl = la.inv(H_M_fl @ self.flex.M_fl @ H_M_fl.T)
        #zeta = H_M_fl @ A_fl
        #self.flex.U_fl = self.flex.L_fl @ zeta
        #self.flex.D_fl = zeta.T @ self.flex.U_fl

    def set_tau(self, tau):
        self.force.tau = tau

    def add_F_ext(self, node=-1, F_ext=None, F_fun=None):
        """
        Add external spatial force/moment at a selected node.

        Usage:
        b1.add_F_ext(F_ext)                    # constant tip force
        b1.add_F_ext(node=-1, F_ext=F_ext)     # constant tip force
        b1.add_F_ext(node=3, F_ext=F_ext)      # constant force at node 3
        b1.add_F_ext(node=3, F_fun=my_fun)     # time-dependent force
        """

        # Backward compatibility:
        # b1.add_F_ext(F_ext)
        if F_ext is None and F_fun is None:
            arr = np.asarray(node)
            if arr.size == 6:
                F_ext = arr.reshape(6, 1)
                node = -1
            else:
                raise ValueError("Provide either F_ext or F_fun.")

        # If constant force is supplied, wrap it in a function.
        if F_fun is None:
            F_const = np.asarray(F_ext, dtype=float).reshape(6, 1)

            def constant_force_fun(t, state, body):
                return F_const

            force_function = constant_force_fun
        # If a function is supplied, use it directly.
        else:
            force_function = F_fun

        self.force.external_loads.append({
        "node": int(node),
        "fun": force_function
        })

    def set_initial_theta0(self, theta0):
        self.initialcondition.theta0 = theta0

    def set_initial_beta0(self, beta0):
        self.initialcondition.beta0 = beta0

    def set_initial_eta0(self, eta0):
        self.initialcondition.eta0 = eta0

    def set_initial_eta_dot0(self, eta_dot0):
        self.initialcondition.eta_dot0 = eta_dot0

    def get_D_m_inv(self, Gamma, x):
        # Calculate D_m_inv
        if x == 0:
            Dminv = self.flex.L_fl
        elif x == 1:
            #Gamma_inv = la.inv(Gamma)
            #Dminv = self.flex.L_fl - la.solve((Gamma_inv + self.flex.D_fl).T, self.flex.U_fl.T).T @ self.flex.U_fl.T
            Dminv = self.flex.L_fl - self.flex.U_fl @ la.solve((np.eye(6, 6) + Gamma @ self.flex.D_fl), Gamma) @ self.flex.U_fl.T
        return Dminv


# =============================================================================
# ATBI_Flex_Grav.py
# =============================================================================

class ATBI_Flex:
    # ATBI class with bodies
    def __init__(self, bodies):
        # Parameters
        self.bodies = bodies
        self.n = len(bodies)

    def coriolis(self, V, beta, H):
        deltaV = H.T @ beta
        return sb.skew6(V) @ deltaV - sb.bar6(deltaV) @ deltaV

    def coriolis_BD(self, V_k, V_p, beta, H, klOO, R3):
        deltaV = H.T @ beta

        a01 = sb.skew(V_k[0:3]) @ deltaV[0:3]
        a02 = sb.skew(R3.T @V_p[0:3]) @ sb.skew(R3.T @V_p[0:3]) @ klOO

        return np.vstack([a01, a02])

    def gyroscopic(self, V, M):
        return sb.bar6(V) @ M @ V

    def gyroscopic_BD(self, body, V_r, m, eta, eta_dot):
        n_md = body.flex.n_md

        # modal integrals
        p_0 = body.flex.p_0
        F_1 = body.flex.F_1
        J_0 = body.flex.J_0
        J_1 = body.flex.J_1
        J_2 = body.flex.J_2
        S_1 = body.flex.S_1
        S_2 = body.flex.S_2
        p_1 = body.flex.p_1
        E = body.flex.E_0
        R_1 = body.flex.R_1
        K_1 = body.flex.K_1
        K_2 = body.flex.K_2
        T_1 = body.flex.T_1
        T_2 = body.flex.T_2
        W_1 = body.flex.W_1
        W_2 = body.flex.W_2

        # First we compine K_1 and K_2 and assemble Q with T_1, T_2, W_1 and W_2
        K = K_1 + K_2 @ np.kron(eta.reshape(n_md, 1), np.eye(3))
        Q_sum = np.zeros((3*n_md, n_md))

        for r in range(n_md):
            for s in range(n_md):

                T1_sr = T_1[3*s:3*s+3, r:r+1]
                T2_sr = T_2[3*s:3*s+3, r:r+1]
                W1_rs = W_1[3*r:3*r+3, s:s+1]
                W2_sr = W_2[3*s:3*s+3, r:r+1]
                F1_sr = F_1[3*s:3*s+3, r:r+1]

                Q_sum[3*r:3*r+3, s:s+1] = (
                T1_sr
                + T2_sr
                + W1_rs
                + W2_sr
                + 2.0 * F1_sr
                )

        omega = V_r[0:3, :]

        b_eta_0 = np.zeros((n_md, 1))
        b_eta_1 = np.zeros((n_md, 1))
        b_omega_0 = sb.skew(omega) @ J_0 @ omega
        b_v_0 = m * sb.skew(omega) @ sb.skew(omega) @ p_0

        for r in range(n_md):
            # zeroth-order term:
            # -omega.T @ (S1(r) + J1(r)) @ omega
            S1_r = S_1[:, 3*r:3*r+3]
            J1_r = J_1[:, 3*r:3*r+3]

            b_eta_0[r, 0] = -(
            omega.T @ (S1_r + J1_r) @ omega
            ).item()

            # first-order term:
            # -omega.T sum_s [ Q(r,s) eta_dot_s + (S2(r,s)+J2(r,s)) @ omega * eta_s ]
            inside_sum = np.zeros((3, 1))

            for s in range(n_md):
                Q_rs = Q_sum[3*r:3*r+3, s:s+1]

                S2_rs = S_2[3*r:3*r+3, 3*s:3*s+3]
                J2_rs = J_2[3*r:3*r+3, 3*s:3*s+3]

                eta_s = float(eta[s, 0])
                eta_dot_s = float(eta_dot[s, 0])

                inside_sum += (
                Q_rs * eta_dot_s
                + (S2_rs + J2_rs) @ omega * eta_s
                )

            b_eta_1[r, 0] = -(omega.T @ inside_sum).item()

        b_eta = b_eta_0 + b_eta_1

        b_omega_1 = np.zeros((3, 1))
        omega_tilde = sb.skew(omega)

        for r in range(n_md):
            eta_r = float(eta[r, 0])
            eta_dot_r = float(eta_dot[r, 0])

            J1_r = J_1[:, 3*r:3*r+3]          # J1(r), shape (3,3)
            K_r  = K[3*r:3*r+3, :]        # K(r),  shape (3,3)

            
            R1_r = R_1[:, r:r+1]   # R1(r), shape (3,1)

            b_omega_1 += (
            omega_tilde @ (J1_r + J1_r.T) @ omega * eta_r
            + (2.0 * J1_r.T + K_r) @ omega * eta_dot_r
            + omega_tilde @ R1_r * eta_dot_r
            )
        b_omega=b_omega_0+b_omega_1

        b_v_1=sb.skew(omega) @ (m*sb.skew(omega) @ p_1 @ eta + 2*E @ eta_dot)
        b_v=b_v_0+b_v_1
        return np.vstack([b_eta, b_omega, b_v])
    
    def M_fl_eta(self, body, eta):
        n_md = body.flex.n_md
        m = body.m

        eta = np.asarray(eta, dtype=float).reshape(n_md, 1)

        p_0 = body.flex.p_0
        p_1 = body.flex.p_1

        F_0 = body.flex.F_0
        F_1 = body.flex.F_1

        G_0 = body.flex.G_0
        E_0 = body.flex.E_0

        J_0 = body.flex.J_0
        J_1 = body.flex.J_1

        # p(eta) = p0 + p1 eta
        p_eta_skew = sb.skew(p_0) + sb.skew(p_1 @ eta)

        # F(eta) has shape (3, n_md)
        F_eta = F_0.copy()

        for r in range(n_md):
            F_r = F_0[:, r:r+1].copy()

            for s in range(n_md):
                F1_rs = F_1[3*r:3*r+3, s:s+1]   # shape (3,1)
                F_r += F1_rs * float(eta[s, 0])

            F_eta[:, r:r+1] = F_r

        # J(eta) has shape (3,3)
        J_eta = J_0.copy()

        for r in range(n_md):
            J1_r = J_1[:, 3*r:3*r+3]
            J_eta += (J1_r + J1_r.T) * float(eta[r, 0])


        row1 = np.hstack([
            G_0,        # (n_md, n_md)
            F_eta.T,   # (n_md, 3)
            E_0.T      # (n_md, 3)
        ])

        row2 = np.hstack([
            F_eta,              # (3, n_md)
            J_eta,              # (3, 3)
            m * p_eta_skew      # (3, 3)
        ])

        row3 = np.hstack([
            E_0,                # (3, n_md)
            -m * p_eta_skew,    # (3, 3)
            m * np.eye(3)       # (3, 3)
        ])

        M = np.vstack([row1, row2, row3])

        return M

    def theta2X(self, theta, joint_type, klOO):
        if joint_type == "revx":
            ang = theta.item()
            q = np.array([[np.sin(ang/2)], [0], [0], [np.cos(ang/2)]])
            return np.vstack((q, klOO)), q

        elif joint_type == "revy":
            ang = theta.item()
            q = np.array([[0], [np.sin(ang/2)], [0], [np.cos(ang/2)]])
            return np.vstack((q, klOO)), q

        elif joint_type == "revz":
            ang = theta.item()
            q = np.array([[0], [0], [np.sin(ang/2)], [np.cos(ang/2)]])
            q = q / np.linalg.norm(q)
            # print(ang)
            return np.vstack((q, klOO)), q

        elif joint_type == "spherical":
            q = theta.reshape(4, 1)
            q = q / np.linalg.norm(q)
            return np.vstack((q, klOO)), q

        elif joint_type == "free":
            q = theta[0:4].reshape(4, 1)
            v = theta[4:7].reshape(3, 1)
            return np.vstack((q, klOO)), q

        elif joint_type == "fixed":
            q = np.array([[0], [0], [0], [1]])
            return np.vstack((q, klOO)), q
        
    def get_node_position(self, body, node):
        """
        Position vector from body reference/inboard node to selected node.
        Assumes straight beam discretized uniformly along body.joint.klOO.
        """
        n_nd = body.flex.n_nd

        if node < 0:
            node = n_nd + node

        if node < 0 or node >= n_nd:
            raise IndexError(f"node={node} outside valid range 0...{n_nd-1}")

        s = node / (n_nd - 1)
        return s * body.joint.klOO.reshape(3, 1)


    def external_force_term(self, body, state, t):
        """
        Returns flexible generalized external force contribution:
        [modal part; rigid spatial part]
        """
        n_md = body.flex.n_md
        F_ext_term = np.zeros((n_md + 6, 1))

        PI_full = body.flex.PI  # shape: (6*n_nd, n_md)

        for load in body.force.external_loads:
            node = load["node"]
            F_j = load["fun"](t, state, body).reshape(6, 1)

            # Mode-shape block for selected node
            PI_j = PI_full[6*node : 6*node + 6, :]

            # Vector from body reference to selected node
            r_j = self.get_node_position(body, node)

            F_ext_term += np.vstack([
            PI_j.T @ F_j,
            sb.phi(r_j) @ F_j
            ])

        return F_ext_term

    def scatter_kinematics(self, state: SystemState):

        # Number of bodies
        n = len(self.bodies)

        # Set up lists
        X = [None] * n
        V = [None] * n
        A_fl = [None] * n
        V_f = [None] * n
        V_r = [None] * n
        a_fl = [None] * n
        b_fl = [None] * n

        for k in reversed(range(n)):
            # Parameters of the body
            body = self.bodies[k]
            theta = state.Theta[k]
            beta = state.Beta[k]
            eta = state.Eta[k]
            eta_dot = state.Eta_dot[k]
            H = body.joint.H
            Mk = body.rigid.Mk
            PI = body.flex.PI_end
            n_md = body.flex.n_md

            # Build X
            X[k], q = self.theta2X(theta, body.joint.type, body.joint.klOO)

            # Build A: NB! Typo in text?!?!
            R3 = sb.q2R(q.flatten(), 3)
            A_fl[k] = sb.get_A(PI, X[k][4:7])

            if k == n - 1:
                V_f[k] = eta_dot
                V_r[k] = H.T @ beta

                a_fl[k] = self.coriolis_BD(V_r[k], np.zeros((6, 1)), beta, H, np.zeros((3, 1)), R3)
            else:
                R6 = sb.q2R(q.flatten(), 6)
                #R_tot = sb.get_R_tot(R6, n_md)

                V_f[k] = eta_dot
                V_r[k] = R6.T @ A_fl[k+1].T @ V[k+1] + H.T @ beta

                a_fl[k] = self.coriolis_BD(V_r[k], V_r[k+1], beta, H, R3.T @ X[k+1][4:7], R3)

            # Coriolis
            #a_fl[k] = np.vstack([np.zeros((n_md, 1)), self.coriolis(V_r[k], beta, H)])

            # Gyroscopic
            #b_fl[k] = np.vstack([np.zeros((n_md, 1)), self.gyroscopic(V_r[k], Mk)])
            b_fl[k] = self.gyroscopic_BD(body, V_r[k], body.m, eta, eta_dot)

            V[k] = np.vstack([V_f[k], V_r[k]])

        return X, V, a_fl, b_fl

    def gather_ATBI(self, state: SystemState, a_fl, b_fl, X, t):
        # Step 3 of ATBI (gather sweep): Takes generalized forces, Coriolis-, gyroscopic
        # terms, X-vector and system configuration and returns G and nu parameters

        # Number of bodies
        n = len(self.bodies)

        # Setup lists
        P_pr_plus = [None] * n
        D_m = [None] * n
        g_fl = [None] * n
        P_pr = [None] * n
        D_pr = [None] * n
        G_pr = [None] * n
        nu_m = [None] * n
        nu_pr = [None] * n
        z_pr_plus = [None] * n

        for k in range(n):
            # Parameters of the body
            body = self.bodies[k]
            H_B = body.joint.H
            tau_pr = body.force.tau
            eta = state.Eta[k]
            eta_dot = state.Eta_dot[k]
            M_fl = self.M_fl_eta(body, eta)
            K_fl = body.flex.K_fl
            C_fl = body.flex.C_fl
            PI = body.flex.PI_end
            n_md = body.flex.n_md
            H_M_fl = np.hstack([np.eye(n_md, n_md), np.zeros((n_md, 6))])
            klOO = X[k][4:7]
            F_ext_term = self.external_force_term(body, state, t)
            """
            # External force
            # F_ext_term = np.zeros((b_fl[k].shape[0], 1))
            # F_ext_term = np.exp(- 5 * t) * np.vstack([PI.T @ F_ext, sb.phi(klOO) @ F_ext])

            F_ext_term = np.vstack([PI.T @ F_ext, sb.phi(klOO) @ F_ext])

            
            if t <= 0.25:
                F_ext_term = np.vstack([PI.T @ F_ext, sb.phi(klOO) @ F_ext])
            """

            if k == 0:
                # Gather loop for k = 0 (Base Case)
                # 13.6
                Gamma_fl = np.zeros((0, 6))
                P_fl = M_fl
                D_m[k] = H_M_fl @ P_fl @ H_M_fl.T
                mu_fl = P_fl[-6:, :] @ H_M_fl.T
                D_m_inv = la.solve(D_m[k], np.eye(n_md), assume_a="sym")
                g_fl[k] = mu_fl @ D_m_inv
                P_pr[k] = P_fl[-6:, -6:] - g_fl[k] @ mu_fl.T
                D_pr[k] = H_B @ P_pr[k] @ H_B.T
                G_pr[k] = P_pr[k] @ la.solve(D_pr[k].T, H_B).T
                tau_pr_bar = np.eye(6, 6) - G_pr[k] @ H_B
                P_pr_plus[k] = tau_pr_bar @ P_pr[k]

                # 13.7
                z = b_fl[k] + K_fl @ np.vstack([eta, np.zeros((6, 1))]) - \
                    F_ext_term + C_fl @ np.vstack([eta_dot, np.zeros((6, 1))])
                eps_m = - z[0:n_md]  # tau_m (assumed to be zero): dim(n_md, 1)
                nu_m[k] = D_m_inv @ eps_m

                z_pr = z[-6:] + g_fl[k] @ eps_m + P_pr[k] @ a_fl[k][-6:]
                eps_pr = tau_pr - H_B @ z_pr
                nu_pr[k] = la.solve(D_pr[k], eps_pr)
                z_pr_plus[k] = z_pr + G_pr[k] @ eps_pr

            else:
                # 13.6
                # Unpacking X-vector
                q = X[k-1][0:4]

                # Rotation
                R6 = sb.q2R(q.flatten(), 6)

                A_fl = sb.get_A(PI, klOO)

                # Gather loop for k > 0
                Gamma_fl = R6 @ P_pr_plus[k-1] @ R6.T  # ?!?!?!?
                P_fl = A_fl @ Gamma_fl @ A_fl.T + M_fl
                D_m[k] = H_M_fl @ P_fl @ H_M_fl.T
                mu_fl = P_fl[-6:, :] @ H_M_fl.T
                #D_m_inv = body.get_D_m_inv(Gamma_fl, 1)
                D_m_inv = la.solve(D_m[k], np.eye(n_md), assume_a="sym")
                g_fl[k] = mu_fl @ D_m_inv
                P_pr[k] = P_fl[-6:, -6:] - g_fl[k] @ mu_fl.T
                D_pr[k] = H_B @ P_pr[k] @ H_B.T
                G_pr[k] = P_pr[k] @ la.solve(D_pr[k].T, H_B).T
                tau_pr_bar = np.eye(6, 6) - G_pr[k] @ H_B
                P_pr_plus[k] = tau_pr_bar @ P_pr[k]

                # 13.7
                z = A_fl @ R6 @ z_pr_plus[k-1] + b_fl[k] + K_fl @ np.vstack([eta, np.zeros(
                    (6, 1))]) - F_ext_term + C_fl @ np.vstack([eta_dot, np.zeros((6, 1))])
                eps_m = - z[0:n_md]  # tau_m (assumed to be zero): dim(n_md, 1)
                nu_m[k] = D_m_inv @ eps_m

                z_pr = z[-6:] + g_fl[k] @ eps_m + P_pr[k] @ a_fl[k][-6:]
                eps_pr = tau_pr - H_B @ z_pr
                nu_pr[k] = la.solve(D_pr[k], eps_pr)
                z_pr_plus[k] = z_pr + G_pr[k] @ eps_pr

        return G_pr, nu_pr, nu_m, g_fl

    def scatter_ATBI(self, a_fl, X, G_pr, nu_pr, nu_m, g_fl):
        # Step 4 of ATBI (second scatter sweep): Takes Coriolis term, X-vector, G,
        # nu and hinge map, H and returns generalized acceleration, gamma

        # Number of bodies
        n = len(self.bodies)

        # Setup of list
        alpha_fl = [None] * n
        theta_ddot = [None] * n
        eta_ddot = [None] * n
        A_fl = [None] * n

        # Spatial gravity
        #g = np.array([0, 0, 0, 0, 0, 9.81]).reshape(6, 1)
        g = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)

        # Spatial gravity rotation setup
        Ri = [None] * (n + 1)
        Ri[-1] = np.eye(6)

        # Loop backwards from n-1 down to 0
        for k in range(n - 1, -1, -1):
            # Parameters of the body
            body = self.bodies[k]
            H_B = body.joint.H
            n_md = body.flex.n_md
            PI = body.flex.PI_end

            # Unpacking rotation
            q = X[k][0:4]

            # Rotation
            R3 = sb.q2R(q.flatten(), 3)
            R6 = sb.q2R(q.flatten(), 6)
            #R_tot = sb.get_R_tot(R6, n_md)

            A_fl[k] = sb.get_A(PI, X[k][4:7])

            if k == n - 1:
                alpha_base = R6.T @ g
                theta_ddot[k] = nu_pr[k] - G_pr[k].T @ alpha_base
                alpha_pr = alpha_base + H_B.T @ theta_ddot[k] + a_fl[k][-6:]
                eta_ddot[k] = nu_m[k] - g_fl[k].T @ alpha_pr
                alpha_fl[k] = np.vstack([eta_ddot[k], alpha_pr])

            else:
                # Scatter loop
                alpha_pr_plus = R6.T @ A_fl[k+1].T @ alpha_fl[k+1]
                theta_ddot[k] = nu_pr[k] - G_pr[k].T @ alpha_pr_plus
                alpha_pr = alpha_pr_plus + H_B.T @ theta_ddot[k] + a_fl[k][-6:]
                eta_ddot[k] = nu_m[k] - g_fl[k].T @ alpha_pr
                alpha_fl[k] = np.vstack([eta_ddot[k], alpha_pr])

        return theta_ddot, eta_ddot, alpha_fl


# =============================================================================
# MultibodySystem.py
# =============================================================================

class MultibodySystem:
    def __init__(self, bodies):
        self.bodies = bodies
        self.ATBI = ATBI_Flex(bodies)
        Theta_0 = [b.initialcondition.theta0 for b in bodies]
        Beta_0 = [b.initialcondition.beta0 for b in bodies]
        Eta_0 = [b.initialcondition.eta0 for b in bodies]
        Eta_dot_0 = [b.initialcondition.eta_dot0 for b in bodies]
        self.S0 = SystemState(Theta_0, Beta_0, Eta_0, Eta_dot_0).pack()

    def EOM(self, t, S):
        state = SystemState.unpack(
            S.reshape(-1, 1), [b.joint for b in self.bodies], [b.flex for b in self.bodies])

        # Normalize quaternions
        for k, body in enumerate(self.bodies):
            if body.joint.type in ["spherical", "free"]:
                q = state.Theta[k][0:4]
                state.Theta[k][0:4] = q / np.linalg.norm(q)

        X, V, a_fl, b_fl = self.ATBI.scatter_kinematics(state)
        G_pr, nu_pr, nu_m, g_fl = self.ATBI.gather_ATBI(
            state, a_fl, b_fl, X, t)
        theta_ddot, eta_ddot, alpha_fl = self.ATBI.scatter_ATBI(
            a_fl, X, G_pr, nu_pr, nu_m, g_fl)

        # Uncomment for at se hvad der blower up. Lige nu(theta_ddot, nu_pr og alpha_fl)
        """if t > 2.35:
            print("\n--- DEBUG t =", t, "---")
            for k in range(len(self.bodies)):
                print("k =", k)
                print("theta =", state.Theta[k].flatten())
                print("beta =", state.Beta[k].flatten())
                print("eta =", state.Eta[k].flatten())
                print("eta_dot =", state.Eta_dot[k].flatten())
                print("theta_ddot =", theta_ddot[k].flatten())
                print("eta_ddot =", eta_ddot[k].flatten())
                print("nu_pr =", nu_pr[k].flatten())
                print("nu_m =", nu_m[k].flatten())
                print("alpha_fl norm =", np.linalg.norm(alpha_fl[k]))
        if t > 2.38:
            raise RuntimeError("Stopped for debug near instability")"""
        Theta_dot, Eta_dot_list = [], []
        for k, body in enumerate(self.bodies):
            if body.joint.type.startswith("rev"):
                Theta_dot.append(state.Beta[k].reshape(1, 1))
            elif body.joint.type == "spherical":
                Theta_dot.append(sb.quat_derivative(
                    state.Theta[k], state.Beta[k]).reshape(4, 1))
            elif body.joint.type == "free":
                qdot = sb.quat_derivative(
                    state.Theta[k][0:4], state.Beta[k][0:3]).reshape(4, 1)
                Theta_dot.append(
                    np.vstack([qdot, state.Beta[k][3:6]]).reshape(7, 1))
            elif body.joint.type == "fixed":
                Theta_dot.append(np.zeros((0, 1)))

            Eta_dot_list.append(state.Eta_dot[k].reshape(body.flex.n_md, 1))

        S_dot = np.vstack([*Theta_dot, *theta_ddot, *
                          Eta_dot_list, *eta_ddot]).flatten()

        return S_dot


# =============================================================================
# Simulation.py
# =============================================================================

class Simulation:
    class Data:
        def __init__(self):
            self.time, self.state, self.X_list, self.V_list, self.a_list, self.b_list, self.alpha_list, self.pos = [
            ], [], [], [], [], [], [], []

    class Setting:
        def __init__(self):
            self.camera_speed = 0
            self.camera_ver = 20
            self.camera_hor = 0
            self.solver = "RK4"

    def __init__(self, system: MultibodySystem, tf, dt):
        self.system = system
        self.data = self.Data()
        self.setting = self.Setting()
        self.tf = tf
        self.dt = dt
        self.setting.ani_dt = dt

        # Increase limit to 100 MB (default is 20)

    def IntegrateSystem(self, solver="RK4"):
        self.setting.solver = solver
        # print("Integrating...")

        # Progress bar
        pbar = tqdm(total=100, desc=f"Integration ({solver})", unit="%")
        original_EOM = self.system.EOM

        last_percent = -1

        def tracked_EOM(t, S):
            nonlocal last_percent

            percent = int((t / self.tf) * 100 - 1)

            if percent > last_percent:
                pbar.update(percent - last_percent)
                last_percent = percent

            return original_EOM(t, S)

        self.system.EOM = tracked_EOM

        if self.setting.solver == "RK4":

            Y, t = sb.integrate_RK4(self.system, 0, self.tf, self.dt)

            self.data.time = t
            states = Y.T

        else:

            t_eval = np.linspace(0, self.tf, int(self.tf/self.dt)+1)

            sol = solve_ivp(
                fun=self.system.EOM,
                t_span=(0, self.tf),
                y0=self.system.S0,
                t_eval=t_eval,
                method=self.setting.solver,
                rtol=1e-8,
                atol=1e-10,
                max_step=0.05
            )

            # Checking if integration actually succeeds
            if not sol.success:
                self.system.EOM = original_EOM
                pbar.close()
                raise RuntimeError(f"Integration failed: {sol.message}")

            self.data.time = sol.t
            states = sol.y.T

        self.system.EOM = original_EOM
        pbar.close()
        print("Integration successful!")

        # Find X-vector for each time step
        dt0 = self.setting.ani_dt

        if dt0 >= self.dt:
            scale = int(dt0 / self.dt)
            nt = int(len(self.data.time) / scale)
            self.data.time = np.linspace(0, self.data.time[-1], nt)
        else:
            print("Error! Invalid animation time step! (ani_dt < sim_dt)")

        for i in range(len(self.data.time)):
            j = i * scale
            # Unpack state
            current_state = SystemState.unpack(
                states[j].reshape(-1, 1), [b.joint for b in self.system.bodies], [b.flex for b in self.system.bodies])

            # Kinematic scatter loop to find X
            X, V, a_fl, b_fl = self.system.ATBI.scatter_kinematics(
                current_state)
            G_pr, nu_pr, nu_m, g_fl = self.system.ATBI.gather_ATBI(
                current_state, a_fl, b_fl, X, self.data.time[i])
            _, _, alpha = self.system.ATBI.scatter_ATBI(
                a_fl, X, G_pr, nu_pr, nu_m, g_fl)

            # Add to list
            self.data.state.append(current_state)
            self.data.X_list.append(X)
            self.data.V_list.append(V)
            self.data.a_list.append(a_fl)
            self.data.b_list.append(b_fl)
            self.data.alpha_list.append(alpha)

    # Call functions for data
    def get_state(self):
        return self.data.state

    def get_X(self):
        return self.data.X_list

    def get_V(self):
        return self.data.V_list

    def get_a(self):
        return self.data.a_list

    def get_b(self):
        return self.data.b_list

    def get_alpha(self):
        return self.data.alpha_list

    # Settings
    def set_camera_speed(self, x):
        self.setting.camera_speed = x

    def set_camera_ver(self, x):
        self.setting.camera_ver = x

    def set_camera_hor(self, x):
        self.setting.camera_hor = x

    def set_ani_dt(self, x):
        self.setting.ani_dt = x

    def nNodalPos(self):
        t = self.data.time
        X = self.data.X_list
        n = len(self.system.bodies)

        nt = len(t)
        nodal_pos = []

        for i in range(nt):
            # Current state
            state = self.data.state[i]

            # End node (O_-1+)
            last_end = np.zeros((3, 1))

            R_i = np.eye(3)
            R_n = np.eye(3)

            nodes_i = []

            for k in range(n - 1, -1, -1):
                # Current body parameters
                eta = state.Eta[k]
                body = self.system.bodies[k]
                n_nd = body.flex.n_nd
                PI = body.flex.PI
                L_elem = body.flex.L_elem
                klOO_nd = body.flex.klOO_nd

                nodes_k = []

                # Unpancking X-vector
                q = X[i][k][0:4]    # Quaternion: k+1 to k

                # Rotation
                R_i = R_i @ sb.q2R(q.flatten(), 3)

                for j in range(n_nd):
                    # Undeformed position in local frame
                    pos_und = klOO_nd[j]

                    # Translational deformation for node j
                    # (Translations are stored at indices j*6+3 to j*6+6 in the PI matrix)
                    u_j = PI[j*6+3: j*6+6, :] @ eta

                    # Global position of node j (of body k)
                    p_glob = last_end + R_i @ (pos_und + u_j)
                    nodes_k.append(p_glob)

                # Rotation of last node
                R_n_vec = PI[-6:-3, :] @ eta
                R_n = Rotation.from_rotvec(R_n_vec.flatten()).as_matrix()

                R_i = R_i @ R_n

                last_end = nodes_k[-1]
                nodes_i.append(nodes_k)

            nodal_pos.append(nodes_i)

        return nodal_pos

    def animate_nodes(self, filename="", save_dir=""):
        # Takes nodal position list and returns 3D simulation of the flexible beam

        t = self.data.time
        dt = t[1] - t[0]

        # Get nodal positions for the flexible beam
        nodal_pos = self.nNodalPos()

        if not nodal_pos:
            print("Error: No nodal position data found. Did you run the integration?")
            return

        nt = len(nodal_pos)
        n_bodies = len(nodal_pos[0])

        # Initialize animation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Determine Axis Limits dynamically based on node movement
        all_points = []
        # Sample frames to speed up boundary calculation
        step = max(1, nt // 50)
        for i in range(0, nt, step):
            for body_nodes in nodal_pos[i]:
                for node in body_nodes:
                    all_points.append(node.flatten())

        all_points = np.array(all_points)
        if len(all_points) > 0:
            max_range = np.abs(all_points).max()
        else:
            max_range = 1.0

        # Prevent zero-range if the beam doesn't move
        if max_range == 0:
            max_range = 1.0

        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Flexible n-Body Animation ({n_bodies} Bodies)")

        # Create n colored lines (one per flexible link)
        cmap = mpl.colormaps['tab10']
        colors = cmap(np.linspace(0, 1, n_bodies))
        lines = []

        for i in range(n_bodies):
            line, = ax.plot([], [], [], '-', lw=4, color=colors[i])
            lines.append(line)

        # Scatter plot for the nodes
        node_dots, = ax.plot([], [], [], 'ko', markersize=4)

        # Plot origin for reference
        ax.plot([0], [0], [0], 'o', color='gray', markersize=6)

        # Initialize the timer text
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

        # Camera setting
        camera_initialized = False

        def update(frame_idx):
            nonlocal camera_initialized

            current_state = nodal_pos[frame_idx]

            all_xs, all_ys, all_zs = [], [], []

            # Update each body separately
            for b_idx in range(n_bodies):
                body_nodes = current_state[b_idx]

                xs = [float(node[0][0]) for node in body_nodes]
                ys = [float(node[1][0]) for node in body_nodes]
                zs = [float(node[2][0]) for node in body_nodes]

                lines[b_idx].set_data(xs, ys)
                lines[b_idx].set_3d_properties(zs)

                all_xs.extend(xs)
                all_ys.extend(ys)
                all_zs.extend(zs)

            node_dots.set_data(all_xs, all_ys)
            node_dots.set_3d_properties(all_zs)

            # Update timer
            time_text.set_text(f'Time: {t[frame_idx]:.2f} s')

            # Camera control
            if self.setting.camera_speed == 0 and frame_idx == 0 and camera_initialized == False:
                ax.view_init(elev=self.setting.camera_ver,
                             azim=self.setting.camera_hor)
                camera_initialized = True
            elif self.setting.camera_speed != 0:
                ax.view_init(elev=self.setting.camera_ver,
                             azim=self.setting.camera_hor + frame_idx * self.setting.camera_speed * 40 * dt)

            return (*lines, node_dots, time_text)

        # Create Animation
        anim = FuncAnimation(
            fig,
            update,
            frames=nt,
            interval=dt*1000,
            blit=False)

        if filename != "":
            print("Rendering animation to HTML... (This may take a minute)")
            filename = filename + ".html"

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fullpath = os.path.join(save_dir, filename)
            else:
                fullpath = filename

            with open(fullpath, "w") as f:
                f.write(anim.to_jshtml())

            print("Rendering of animation: Done!")
            print(f"Saved to {fullpath}")

        else:
            plt.show()

    def get_pos(self):
        self.data.pos = self.nBodyPos()
        return self.data.pos

if __name__ == "__main__":
    L=1
    klOO1 = np.array([1, 0, 0]).reshape(3, 1)
    klOO2 = np.array([1, 0, 0]).reshape(3, 1)
    H_type1 = "fixed"
    H_type2 = "revy"

    # n_md_max = (n_nd - 1) * 3

    E, G,c, rho, n_nd, n_md = 1.93e9, 6.902e8,0.2, 1300, 10, 20

    w, h = 0.04, 0.04

    j1 = Joint(klOO1, H_type1)
    r1 = Rigid_Properties(rho, w, h)
    f1 = Flex_Properties(
    E, G, c, n_nd, n_md,
    mode_quota={
        "bending_xy": 1,
        "bending_xz": 1,
        "axial_x": 1
    })

    j2 = Joint(klOO2, H_type2)
    r2 = Rigid_Properties(rho, w, h)
    f2 = Flex_Properties(E, G, c, n_nd, n_md,mode_quota={
        "bending_xy": 1,
        "bending_xz": 1,
        "axial_x": 0
    })
    b1 = SOABody(j1, r1, f1)
    b2 = SOABody(j2, r2, f2)

    def decaying_tip_force(t, state, body):
        F0 = 100.0 * np.exp(-5.0 * t)
        return np.array([0, 0, 0, 0, 0, -F0]).reshape(6, 1)
    
    def pulse_force(t, state, body):
        if  1<= t <= 1.09:
            return np.array([0, 0, 0, 0, 0, 100]).reshape(6, 1)
        else:
            return np.zeros((6, 1)) 

    #F_tip = np.array([0, 0, 0, 1e7, 0, 0]).reshape(6, 1)
    #b1.add_F_ext(node=b1.flex.n_nd - 1, F_ext=F_tip)
    b1.add_F_ext(node=b1.flex.n_nd // 2, F_fun=pulse_force)
    
    
    #b1.add_F_ext(node=b1.flex.n_nd - 2, F_fun=decaying_tip_force)

    

    #F_ext2 = np.array([0, 0, 0, 1e3, 0, 0]).reshape(6, 1)
    #b2.set_F_ext(F_ext2)
    #b1.set_initial_beta0(40)


    #eta0 = np.vstack([np.array([5]), np.zeros((n_md-1, 1))]).reshape(6, 1)
    #eta0 = np.array([0, 0, 0, 0, 10, 0]).reshape(6, 1)
    #b1.set_initial_eta0(eta0)

    bodies = [b1]
    print("Mode classification:")
    print(pd.DataFrame(b1.flex.modes).to_string(index=False))
    print("n_md used =", b1.flex.n_md)

    system = MultibodySystem(bodies)

    tf = 2
    dt = 0.01
    sim = Simulation(system, tf, dt)

    sim.set_camera_ver(0)
    sim.set_camera_hor(90)
    sim.set_camera_speed(0)
    sim.set_ani_dt(0.01)

    sim.IntegrateSystem("Radau")

    #save_dir = r"C:\Users\jepp6\OneDrive - Aarhus universitet\Dokumenter\Noter\6. Semester\Bachelor Projekt\BachelorCode\Renders"

    # sim.animate_nodes(filename="FlexOORotMissing", save_dir=save_dir)
    sim.animate_nodes()
    """
    state = sim.get_state()
    body = sim.system.bodies[0]

    PI_end = body.flex.PI_end
    ux_hist = []
    eta_hist=[] 

    for s in state:
        eta = s.Eta[0]
        eta_hist.append(eta.flatten())
        u = PI_end@ eta
        ux_hist.append(u[3, 0])   # x deformation

    ux_hist = np.array(ux_hist)
    eta_hist = np.array(eta_hist)
    t = np.asarray(sim.data.time)
    plt.plot(t,ux_hist)
    plt.show()

    plt.plot(t,eta_hist)
    plt.show()

    print(pd.DataFrame(eta_hist))
    """
    
    
    # FFT analysis
    state = sim.get_state()
    body = sim.system.bodies[0]

    PI = body.flex.PI
    mid_node = body.flex.n_nd // 2

    uz_mid_hist = []

    for s in state:
        eta = s.Eta[0]
        u_mid = PI[6*mid_node : 6*mid_node + 6, :] @ eta
        uz_mid_hist.append(u_mid[5, 0])   # z deformation

    uz_mid_hist = np.array(uz_mid_hist)

    from scipy.fft import rfft, rfftfreq
    t = np.asarray(sim.data.time)
    y_all = np.asarray(uz_mid_hist)

    # Only use data after t = 3.1 s
    mask = t >= 1.1
    t_fft = t[mask]
    y = y_all[mask]

    # Optional check
    print(pd.DataFrame({"t": t_fft, "uz_mid": y}))

    # remove static offset after cutting the signal
    #y = y - np.mean(y)

    # window
    window = np.hanning(len(y))
    y_win = y * window

    # time step
    dt = t_fft[1] - t_fft[0]

    freq = rfftfreq(len(y_win), dt)
    amp = np.abs(rfft(y_win))

    plt.figure()
    plt.plot(freq, amp)
    plt.xlim(0, 20)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    