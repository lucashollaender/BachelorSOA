import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from SOALIB import soalib as sb
from scipy.integrate import solve_ivp
from Structural_Analysis_BD_Rect import Structural_Analysis_BD_Rect
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import pandas as pd


class SOABody:

    # ----- Forces -----
    class Force:
        def __init__(self, joint: Joint):
            self.tau = np.zeros((joint.beta_size(), 1))
            self.F_ext = []
            self.k_TS = 0
            self.c_TS = 0
            self.theta0_TS = 0

    def set_tau(self, tau):
        self.force.tau = tau

    def set_F_ext(self, node=-1, F_ext=None, F_fun=None):
        """
        Add external spatial force/moment at a selected node.

        How to call:
        b1.set_F_ext(F_ext)                    # constant tip force
        b1.set_F_ext(node=-1, F_ext=F_ext)     # constant tip force
        b1.set_F_ext(node=3, F_ext=F_ext)      # constant force at node 3
        b1.set_F_ext(node=3, F_fun=my_fun)     # time-dependent force
        """

        if F_ext is None and F_fun is None:
            arr = np.asarray(node)
            if arr.size == 6:
                F_ext = arr.reshape(6, 1)
                node = -1
            else:
                raise ValueError("Provide either F_ext or F_fun.")

        # Constant F_ext
        if F_fun is None:
            F_const = np.asarray(F_ext, dtype=float).reshape(6, 1)

            def constant_force_fun(t, state):
                return F_const

            force_function = constant_force_fun
        # Function F_ext in terms of time, t
        else:
            force_function = F_fun

        self.force.F_ext.append({
            "node": int(node),
            "fun": force_function
        })

    def get_node_position(self, node):
        """
        Position vector from body reference/inboard node to selected node.
        Assumes straight beam discretized uniformly along body.joint.klOO.
        """
        n_nd = self.flex.n_nd

        if node < 0:
            node = n_nd + node

        if node < 0 or node >= n_nd:
            raise IndexError(f"node={node} outside valid range 0...{n_nd-1}")

        s = node / (n_nd - 1)
        return s * self.joint.klOO.reshape(3, 1)

    def get_F_ext_term(self, state, t):
        n_md = self.flex.n_md
        F_ext_term = np.zeros((n_md + 6, 1))

        PI = self.flex.PI  # shape: (6*n_nd, n_md)

        for load in self.force.F_ext:
            node = load["node"]

            if node < 0:
                node = self.flex.n_nd + node

            F_j = load["fun"](t, state).reshape(6, 1)

            # Mode-shape block for selected node
            PI_j = PI[6*node: 6*node + 6, :]

            # Vector from body reference to selected node
            r_j = self.get_node_position(node)

            F_ext_term += np.vstack([PI_j.T @ F_j, sb.phi(r_j) @ F_j])

        return F_ext_term

    def set_TS(self, k_TS, c_TS, theta0_TS):
        self.force.k_TS = k_TS
        self.force.c_TS = c_TS
        self.force.theta0_TS = theta0_TS

    def get_TS_term(self, theta, beta):
        k_TS = self.force.k_TS
        c_TS = self.force.c_TS
        theta0_TS = self.force.theta0_TS
        joint_type = self.joint.type

        # Torsional spring
        if joint_type.startswith("rev"):
            return - k_TS * (theta - theta0_TS) - c_TS * beta
        elif joint_type == "spherical":
            return np.zeros((3, 1))  # Not implemented for spherical joint
        elif joint_type == "fixed":
            return np.zeros((0, 1))

    def set_impulse_force(self, ts, dt, F_impulse, node=-1):
        # Define the time-dependent function
        def impulse_fun(t, state):
            if ts <= t <= (ts + dt):
                # If you want the exponential decay you had previously, you could do:
                # return np.exp(-5 * (t - ts)) * F_vec
                return F_impulse
            else:
                return np.zeros((6, 1))

        # Pass the function to the existing external force handler
        self.set_F_ext(node=node, F_fun=impulse_fun)

    # ----- Coriolis acceleration and gyroscopic force -----
    # Rigid SOA:
    def coriolis(self, V, beta, H, n_md):
        deltaV = H.T @ beta

        b_eta = np.zeros((n_md, 1))
        b_r = sb.skew6(V) @ deltaV - sb.bar6(deltaV) @ deltaV
        return np.vstack([b_eta, b_r])

    def gyroscopic(self, V, M):
        return sb.bar6(V) @ M @ V

    # Using Modal Integrals
    def coriolis_BD(self, V_k, V_p, beta, H, klOO, R3):
        deltaV = H.T @ beta

        a01 = sb.skew(V_k[0:3]) @ deltaV[0:3]
        a02 = sb.skew(R3.T @ V_p[0:3]) @ sb.skew(R3.T @ V_p[0:3]) @ klOO

        return np.vstack([a01, a02])

    def gyroscopic_BD(self, body, V_r, m):
        n_md = body.flex.n_md

        # Modal integrals
        p_0 = body.flex.p_0
        # F_1 = body.flex.F_1
        J_0 = body.flex.J_0
        J_1 = body.flex.J_1
        S_1 = body.flex.S_1

        omega = V_r[0:3, :].reshape(3, 1)

        # b_eta = np.zeros((n_md, 1))

        # for i in range(n_md):
        # b_eta[i] = - omega.T @ (S_1[:, 3 * i: 3 * i + 3] + J_1[:, 3 * i: 3 * i + 3]) @ omega

        # A[m, :, :] = S_1[:, 3*m:3*m+3] + J_1[:, 3*m:3*m+3]
        A = (S_1 + J_1).reshape(3, n_md, 3).transpose(1, 0, 2)

        # b_eta[m] = - omega.T @ A[m] @ omega
        b_eta = -(omega.T @ A @ omega).reshape(n_md, 1)

        omega_skew = sb.skew(omega)

        return np.vstack([b_eta, omega_skew @ J_0 @ omega, m * omega_skew @ omega_skew @ p_0])

    # ----- Initial Conditions -----
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

    def set_initial_theta0(self, theta0):
        self.initialcondition.theta0 = theta0

    def set_initial_beta0(self, beta0):
        self.initialcondition.beta0 = beta0

    def set_initial_eta0(self, eta0):
        self.initialcondition.eta0 = eta0

    def set_initial_eta_dot0(self, eta_dot0):
        self.initialcondition.eta_dot0 = eta_dot0

    # ----- Modal Rotation -----
    def get_R_initial(self, joint):
        u_hat = joint.klOO_u
        i_hat = np.array([1, 0, 0])

        if np.allclose(u_hat, i_hat):
            return np.eye(3)
        elif np.allclose(u_hat, -i_hat):
            return np.diag([-1, -1, 1])

        v = np.cross(i_hat, u_hat)
        c = np.dot(i_hat, u_hat)

        v_skew = sb.skew(v)

        R = np.eye(3) + v_skew + (v_skew @ v_skew) * (1 / (1 + c))
        return R

    def get_body_analysis_rot(self, body_analysis):
        # klOO rotation
        R = self.get_R_initial(self.joint)
        R6 = la.block_diag(R, R)
        R_nodes = la.block_diag(*[R6 for _ in range(self.flex.n_nd)])
        R_modal = la.block_diag(np.eye(body_analysis.n_md), R6)

        # PI
        self.flex.PI = R_nodes @ body_analysis.PI
        self.flex.PI_e = R_nodes @ body_analysis.PI_e
        self.flex.PI_end = R6 @ body_analysis.PI[-6:, :]
        self.flex.omega2 = body_analysis.omega2
        self.flex.omega = body_analysis.omega

        # Mode selection
        self.flex.modes = body_analysis.modes
        self.flex.n_md = body_analysis.n_md

        # Stiffness, damping and mass matrix
        self.flex.K_fl = R_modal @ body_analysis.K_fl @ R_modal.T
        self.flex.M_fl = R_modal @ body_analysis.M_fl @ R_modal.T
        self.flex.C_fl = R_modal @ body_analysis.C_fl @ R_modal.T

        # Modal integral for gyroscopic force
        self.flex.p_0 = R @ body_analysis.p_0
        self.flex.J_0 = R @ body_analysis.J_0 @ R.T

        S_1_rot = np.zeros_like(body_analysis.S_1)
        J_1_rot = np.zeros_like(body_analysis.J_1)
        F_1_rot = np.zeros_like(body_analysis.F_1)

        for i in range(self.flex.n_md):
            S_1_rot[:, 3*i:3*i+3] = R @ body_analysis.S_1[:, 3*i:3*i+3] @ R.T
            J_1_rot[:, 3*i:3*i+3] = R @ body_analysis.J_1[:, 3*i:3*i+3] @ R.T
            F_1_rot[3*i:3*i+3, :] = R @ body_analysis.F_1[3*i:3*i+3, :]

        self.flex.S_1 = S_1_rot
        self.flex.J_1 = J_1_rot
        self.flex.F_1 = F_1_rot

    # ----- D_m Inverse Offline Computation -----
    def get_D_m_offline(self):
        # D_m inverse (offline)
        if self.flex.n_md == 0:
            # Rigid Body
            self.flex.L_fl = np.zeros((0, 0))
            self.flex.U_fl = np.zeros((0, 6))
            self.flex.D_fl = np.zeros((6, 6))
        else:
            # Standard flexible body formulation
            H_M_fl = np.hstack(
                [np.eye(self.flex.n_md), np.zeros((self.flex.n_md, 6))])
            A_fl = sb.get_A(self.flex.PI_end, self.joint.klOO)
            self.flex.L_fl = la.inv(H_M_fl @ self.flex.M_fl @ H_M_fl.T)
            zeta = H_M_fl @ A_fl
            self.flex.U_fl = self.flex.L_fl @ zeta
            self.flex.D_fl = zeta.T @ self.flex.U_fl

    def get_D_m_inv(self, Gamma, type):
        # Calculate D_m_inv
        if self.flex.n_md == 0:
            return np.zeros((0, 0))

        if type == "tip":
            Dminv = self.flex.L_fl
        elif type == "not_tip":
            Dminv = self.flex.L_fl - self.flex.U_fl @ la.solve(
                (np.eye(6, 6) + Gamma @ self.flex.D_fl), Gamma) @ self.flex.U_fl.T
        return Dminv

    def __init__(self, joint: Joint, rigid: Rigid_Properties, flex: Flex_Properties):
        # Import Classes
        self.joint = joint
        self.rigid = rigid
        self.flex = flex
        self.force = self.Force(self.joint)
        rigid.A = rigid.h * rigid.w
        rigid.L = joint.L
        flex.L_elem = joint.L / flex.n_elem
        flex.klOO_nd = [j * (joint.klOO / flex.n_elem)
                        for j in range(flex.n_nd)]
        self.m = rigid.rho * rigid.A * joint.L
        self.rigid.CkJk = np.array([1/12 * self.m * (rigid.h**2 + rigid.w**2), 1/12 * self.m * (
            rigid.h**2 + joint.L**2), 1/12 * self.m * (rigid.w**2 + joint.L**2)])
        rigid.Mk = rigid.get_Mk(self.m, self.rigid.CkJk, self.joint.klOC)
        joint.klOO_u = joint.klOO.flatten() / joint.L

        # Structural Analysis
        body_analysis = Structural_Analysis_BD_Rect(joint, rigid, flex)
        # Modal Rotation
        self.get_body_analysis_rot(body_analysis)

        # Initial Condition Setup
        self.initialcondition = self.InitialCondition(joint, self.flex)

        # D_m Inverse Offline Computation
        self.get_D_m_offline()


"""
Without rotation to klOO:
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

            # PI
            self.flex.PI = body_analysis.PI
            self.flex.PI_e = body_analysis.PI_e
            self.flex.PI_end = body_analysis.PI[-6:, :]
            self.flex.omega2 = body_analysis.omega2
            self.flex.omega = body_analysis.omega

            # Mode selection
            self.flex.modes = body_analysis.modes
            self.flex.n_md = body_analysis.n_md

            # Stiffness, damping and mass matrix
            self.flex.K_fl = body_analysis.K_fl
            self.flex.M_fl = body_analysis.M_fl
            self.flex.C_fl = body_analysis.C_fl

            # Modal integral for gyroscopic force
            self.flex.p_0 = body_analysis.p_0
            self.flex.S_1 = body_analysis.S_1
            self.flex.F_1 = body_analysis.F_1
            self.flex.J_0 = body_analysis.J_0
            self.flex.J_1 = body_analysis.J_1

        self.initialcondition = self.InitialCondition(joint, self.flex)

        # D_m inverse (offline)
        if self.flex.n_md == 0:
            # Rigid Body
            self.flex.L_fl = np.zeros((0, 0))
            self.flex.U_fl = np.zeros((0, 6))
            self.flex.D_fl = np.zeros((6, 6))
        else:
            # Standard flexible body formulation
            H_M_fl = np.hstack([np.eye(self.flex.n_md), np.zeros((self.flex.n_md, 6))])
            A_fl = sb.get_A(self.flex.PI_end, self.joint.klOO)
            self.flex.L_fl = la.inv(H_M_fl @ self.flex.M_fl @ H_M_fl.T)
            zeta = H_M_fl @ A_fl
            self.flex.U_fl = self.flex.L_fl @ zeta
            self.flex.D_fl = zeta.T @ self.flex.U_fl

"""
