import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
import scipy.linalg as la
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as Rot
from SOALIB import soalib as sb
from scipy.integrate import solve_ivp
import copy
import os
import matplotlib as mpl

# Increase limit to 100 MB (default is 20)
plt.rcParams['animation.embed_limit'] = 1000

class Joint:
# Joint class with H_type, H and klOO
    def __init__(self, klOO, H_type: str):
        # Parameters
        self.type = H_type
        self.H = sb.hinge_map(H_type)
        self.klOO = klOO.reshape(3, 1)

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

class Inertia:
# Inertia class with m, CkJk and klOC
    def __init__(self, m, CkJk, klOC):
        # Parameters
        self.m = m
        self.CkJk = CkJk
        self.klOC = klOC.reshape(3, 1)

        # Spatial inertia
        MC = np.block([
            [np.diag(CkJk), np.zeros((3, 3))],
            [np.zeros((3, 3)), m * np.eye(3)]
        ])
        self.Mk = sb.phi(klOC) @ MC @ sb.phi(klOC).T        

class Flex:
    def __init__(self, E, G, rho, n_nd, n_md):
        self.E = E
        self.G = G
        self.rho = rho
        self.n_nd = n_nd
        self.n_md = n_md
        self.n_elem = self.n_nd - 1
        self.L_elem = [None]
        self.m_nd = [None]
        self.K_st = [None]
        self.M_nd = [None]
        self.K = [None]
        self.M = [None]
        self.eigval = [None]
        self.PI_r = [None]
        self.PI_t = [None]
        self.PI = [None]
        self.p_0 = [None]
        self.CkJk_0 = [None]
        self.F_0 = [None]
        self.E_0 = [None]
        self.G_0 = [None]
    
    def set_PI(self, PI):
        self.PI = PI

class SOABody:
# SOAbody class
    class Force:
        def __init__(self, joint: Joint):
            self.tau = np.zeros((joint.beta_size(), 1))
            self.sum_phi_F_ext = np.zeros((6, 1))
        
    class InitialCondition:
        def __init__(self, joint: Joint):
            # Setup of initial conditions (assumes identity rotation and no initial velocity)
            self.theta0 = np.zeros((joint.theta_size(), 1))
            if np.size(self.theta0) == 4:
                self.theta0[-1] = 1
            elif np.size(self.theta0) == 7:
                self.theta0[3] = 1
            self.beta0 = np.zeros((joint.beta_size(), 1))
    
    def get_K_st(self):
        
        # Get nodal stiffness
        k = sb.get_stiff_mat_rect_3D(self.h, self.w, self.L, self.flex.E, self.flex.G)

        # Global stiffness matrix setup
        K_st = np.zeros((6*self.flex.n_nd, 6*self.flex.n_nd))

        for i in range(self.flex.n_nd - 1):
            k_i = np.zeros((6*self.flex.n_nd, 6*self.flex.n_nd))
            k_i[i*6:i*6+12, i*6:i*6+12] = k
            K_st = K_st + k_i
        
        return K_st
    
    def get_M_nd(self):
        # Nodal masses
        L_elem = self.L / self.flex.n_elem
        m_e = self.flex.rho * self.A * L_elem

        m = np.full(self.flex.n_nd, m_e)
        m[-1], m[0] = m_e / 2, m_e / 2

        # Store nodal masses and lenghts
        self.flex.m_nd = m
        self.flex.L_elem = L_elem

        block = []
        for i in range(self.flex.n_nd):
            block.append(np.zeros((3, 3)))
            block.append(m[i] * np.eye(3, 3))
        
        M = la.block_diag(*block)

        return M
    
    def get_PI(self):

        # Fixed BC
        K_st = self.flex.K_st[6:, 6:]
        M_nd = self.flex.M_nd[6:, 6:]

        # Rearranging of M and K
        index = np.zeros((1, 0))

        for i in range(self.flex.n_elem):
            index_add = np.linspace(i * 6 + 3, i * 6 + 5, 3).reshape(1, 3)
            index = np.hstack([index, index_add])

        for i in range(self.flex.n_elem):
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
        X = la.solve(K_rr, K_rt, assume_a = "sym")
        K_c = K_tt - K_tr @ X

        print(np.linalg.norm(M_c))
        print(np.linalg.norm(K_c))

        # Solve eigenvalue problem for Pi_t (Mass normalized!)
        eigval, PI_t = la.eigh(K_c, M_c, subset_by_index = (0, self.flex.n_md - 1))

        # Store eigen values
        self.flex.eigval = eigval

        # Compute rotational part of PI
        PI_r = - X @ PI_t

        # Store PI_r and PI_t for modal integrals
        self.flex.PI_r = np.vstack([np.zeros((3, self.flex.n_md)), PI_r])
        self.flex.PI_t = np.vstack([np.zeros((3, self.flex.n_md)), PI_t])

        # PI setup
        PI = np.zeros((2 * PI_t.shape[0] + 6, PI_t.shape[1]))
        for i in range(self.flex.n_elem):
            PI[i * 6 + 6:i * 6 + 9, :] = PI_r[i * 3:i * 3 + 3, :]
            PI[i * 6 + 9:i * 6 + 12, :] = PI_t[i * 3:i * 3 + 3, :]
        
        return PI
    
    def get_Modal_Int(self):
        # Parameters
        m = self.m
        m_nd = self.flex.m_nd.reshape(-1, 1)
        L_elem = self.flex.L_elem
        PI_t = self.flex.PI_t
        n_md = self.flex.n_md
        n_nd = self.flex.n_nd

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
                F_0_sum[:, r] += m_nd[i] * klkO_skew @ PI_t[i * 3: i * 3 + 3, r]
                E_0_sum[:, r] += m_nd[i] * PI_t[i * 3: i*3 + 3, r]

                for s in range(n_md):
                    G_0_sum[r, s] += m_nd[i] * PI_t[i * 3: i*3 + 3, r].T @ PI_t[i * 3: i*3 + 3, s]

        # Store modal integrals
        self.flex.p_0 = 1/m * p_0_sum
        self.flex.CkJk_0 = CkJk_0_sum
        self.flex.F_0 = F_0_sum
        self.flex.G_0 = G_0_sum
        self.flex.E_0 = E_0_sum

    def get_M(self):
        # Parameters
        p_0_skew = sb.skew(self.flex.p_0)
        CkJk_0 = self.flex.CkJk_0
        F_0 = self.flex.F_0
        G_0 = self.flex.G_0
        E_0 = self.flex.E_0

        m = self.inertia.m

        # Build M
        rw1 = np.hstack([G_0, F_0.T, E_0.T])
        rw2 = np.hstack([F_0, CkJk_0, m * p_0_skew])
        rw3 = np.hstack([E_0, -m * p_0_skew, m * np.eye(3)])

        return np.vstack([rw1, rw2, rw3])

    def get_D_m_inv(self, Gamma):
        # Parameters
        n_md = self.flex.n_md 

        H_M_fl = np.hstack([np.eye(n_md, n_md), np.zeros((n_md, 6))])
        M_fl = self.flex.M
        PI = self.flex.PI
        A_fl = sb.get_A(PI, self.joint.klOO)

        L_fl = la.inv(H_M_fl @ M_fl @ H_M_fl)
        zeta = H_M_fl @ A_fl
        U_fl = L_fl @ zeta
        D_fl = zeta.T @ U_fl
        Gamma_inv = la.inv(Gamma)

        return L_fl - la.solve((Gamma_inv + D_fl).T, U_fl.T).T @ U_fl.T


    def __init__(self, joint: Joint, inertia: Inertia, flex: Flex, h, w):
        self.joint = joint
        self.inertia = inertia
        self.flex = flex
        self.force = self.Force(self.joint)
        self.initialcondition = self.InitialCondition(self.joint)
        self.h = h
        self.w = w
        self.A = h * w
        self.L = np.linalg.norm(self.joint.klOO)
        self.m = flex.rho * self.A * self.L

        # Intermidiate stiffness and mass matrix
        self.flex.K_st = self.get_K_st()
        self.flex.M_nd = self.get_M_nd()

        # PI
        if self.flex.PI == [None]:
            self.flex.PI = self.get_PI()

        # Stiffness and mass matrix
        self.flex.K = np.zeros((self.flex.n_md + 6, self.flex.n_md + 6))
        self.flex.K[0:self.flex.n_md, 0:self.flex.n_md] = self.flex.PI.T @ self.flex.M_nd @ self.flex.PI

        self.get_Modal_Int()
        self.flex.M = self.get_M()
    
    def set_tau(self, tau):
        self.force.tau = tau
    
    def set_F_ext(self, F_ext, klBO):
        F = np.zeros((6, 1))
        for i in range(len(F_ext)):
            F = F + sb.phi(klBO[i]) @ F_ext[i]
        self.force.sum_phi_F_ext = F

    def set_initial_theta0(self, theta0):
        self.initialcondition.theta0 = theta0

    def set_initial_beta0(self, beta0):
        self.initialcondition.beta0 = beta0

class SystemState:
# State of system class
    def __init__(self, theta, beta):
        # Parameters
        self.Theta = theta
        self.Beta = beta

    # Packing of state, S: Two lists to column vector
    def pack(self):
        return np.vstack([*self.Theta, *self.Beta]).flatten()

    # Unpacking of state, S: Column vector to two lists
    @staticmethod
    def unpack(S, joints):
        S = S.flatten()
        Theta, Beta = [], []
        idx = 0

        for k in joints:
            sz = k.theta_size()
            Theta.append(S[idx:idx + sz].reshape(sz, 1))
            idx += sz

        for k in joints:
            sz = k.beta_size()
            Beta.append(S[idx:idx + sz].reshape(sz, 1))
            idx += sz

        return SystemState(Theta, Beta)

class ATBI:
# ATBI class with bodies
    def __init__(self, bodies):
        # Parameters
        self.bodies = bodies
        self.n = len(bodies)

    def coriolis(self, V, beta, H):
        deltaV = H.T @ beta
        return sb.skew6(V) @ deltaV - sb.bar6(deltaV) @ deltaV

    def gyroscopic(self, V, M):
        return sb.bar6(V) @ M @ V
    
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
            return np.vstack((q, klOO)), q
         
        elif joint_type == "spherical":
            q = theta.reshape(4, 1) 
            return np.vstack((q, klOO)), q

        elif joint_type == "free":
            q = theta[0:4].reshape(4, 1) 
            v = theta[4:7].reshape(3, 1)
            return np.vstack((q, klOO)), q

        elif joint_type == "fixed":
            q = np.array([[0],[0],[0],[1]])
            return np.vstack((q, klOO)), q

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
            eta_dot = state.Eta_dot[k]
            H = body.joint.H
            Mk = body.inertia.Mk
            PI = body.flex.PI
            K_fl = body.flex.K
            M_fl = body.flex.M
            n_md = body.flex.n_md

            # Build X
            X[k], q = self.theta2X(theta, body.joint.type, body.joint.klOO)

            # Build A: NB! Typo in text?!?!
            A_fl[k] = sb.get_A(PI, X[k][4:7])
            
            if k == n - 1:
                V_f[k] = eta_dot
                V_r[k] = H.T @ beta - PI @ eta_dot
            else:
                R6 = sb.q2R(q.flatten(), 6)
                R_tot = sb.get_R_tot(R6, n_md)
                
                V_f[k] = eta_dot
                V_r[k] = R_tot.T @ A_fl[k+1].T @ V[k+1] + H.T @ beta

            a_fl[k] = np.vstack([np.zeros((n_md, 1)), self.coriolis(V_r[k], beta, H)])
            b_fl[k] = np.vstack([np.zeros((n_md, 1)), self.gyroscopic(V_r[k], Mk)])

            V[k] = np.vstack([V_f[k], V_r[k]])

        return X, V, a_fl, b_fl
        
    def gather_ATBI(self, state: SystemState, a_fl, b_fl, X):
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
            Mk = body.inertia.Mk
            sum_phi_F_ext = body.force.sum_phi_F_ext
            tau_pr = body.force.tau
            eta = state.Eta
            M_fl = body.flex.M
            K = body.flex.K
            PI = body.flex.PI
            n_md = body.flex.n_md
            H_M_fl = np.hstack([np.eye(n_md, n_md), np.zeros((n_md, 6))])

            if k == 0:
                # Gather loop for k = 0 (Base Case)
                # 13.6
                Gamma = np.zeros((6, 6))
                P_fl = M_fl
                D_m[k] = H_M_fl @ P_fl @ H_M_fl.T
                mu_fl = P_fl[-6:, :] @ H_M_fl.T
                D_m_inv = body.get_D_m_inv(Gamma)
                g_fl[k] = mu_fl @ D_m_inv
                P_pr[k] = P_fl[-6:, -6:] - g_fl[k] @ mu_fl.T
                D_pr[k] = H_B @ P_pr[k] @ H_B.T
                G_pr[k] = P_pr[k] @ la.solve(D_pr[k].T, H_B).T
                tau_pr_bar = np.eye(6, 6) - G_pr[k] @ H_B
                P_pr_plus[k] = tau_pr_bar @ P_pr[k]

                # 13.7
                z = b_fl + K @ np.vstack([eta, np.zeros((6, 1))])
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
                klOO = X[k][4:7]

                # Rotation
                R6 = sb.q2R(q.flatten(), 6)
                R_tot = sb.get_R_tot(R6, n_md)

                A_fl = sb.get_A(PI, X[k][4:7])
                
                # Gather loop for k > 0
                Gamma_fl = R_tot @ P_pr_plus[k-1] @ R_tot.T # ?!?!?!?
                P_fl = A_fl @ Gamma_fl @ A_fl.T + M_fl
                D_m[k] = H_M_fl @ P_fl @ H_M_fl.T
                mu_fl = P_fl[-6:, :] @ H_M_fl.T
                D_m_inv = body.get_D_m_inv(Gamma)
                g_fl[k] = mu_fl @ D_m_inv
                P_pr[k] = P_fl[-6:, -6:] - g_fl[k] @ mu_fl.T
                D_pr[k] = H_B @ P_pr[k] @ H_B.T
                G_pr[k] = P_pr[k] @ la.solve(D_pr[k].T, H_B).T
                tau_pr_bar = np.eye(6, 6) - G_pr[k] @ H_B
                P_pr_plus[k] = tau_pr_bar @ P_pr[k]

                # 13.7
                z =  A_fl @ R6 @ z_pr_plus[k-1] + b_fl + K @ np.vstack([eta, np.zeros((6, 1))])
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
        g = np.array([0, 0, 0, 0, 0, 9.81]).reshape(6, 1)
        
        # Spatial gravity rotation setup
        Ri = [None] * (n + 1)
        Ri[-1] = np.eye(6)

        # Loop backwards from n-1 down to 0
        for k in range(n - 1, -1, -1):
            # Parameters of the body
            body = self.bodies[k]
            H_B = body.joint.H
            PI = body.flex.PI

            # Unpacking rotation
            q = X[k][0:4]

            # Rotation
            R6 = sb.q2R(q.flatten(), 6)

            # Spatial gravity rotation
            Ri[k] =   Ri[k+1]  @ R6

            A_fl[k] = sb.get_A(PI, X[k][4:7])

            if k == n - 1:
                # Scatter loop (Tip of the chain)
                theta_ddot[k] = nu_pr[k]
                alpha_pr = H_B @ theta_ddot[k] + a_fl[k][-6:]
                eta_ddot[k] = nu_m[k] - g_fl[k].T @ alpha_pr
                alpha_fl[k] = np.vstack([eta_ddot[k], alpha_pr])
            
            else:
                # Scatter loop
                alpha_pr_plus = A_fl[k+1].T @ R6 @ alpha_fl[k+1]
                theta_ddot[k] = nu_pr[k] - G_pr[k].T @ alpha_pr_plus
                alpha_pr = H_B @ theta_ddot[k] + a_fl[k][-6:]
                eta_ddot[k] = nu_m[k] - g_fl[k].T @ alpha_pr
                alpha_fl[k] = np.vstack([eta_ddot[k], alpha_pr])

        return theta_ddot, eta_ddot, alpha_fl

class MultibodySystem:
    def __init__(self, bodies):
        self.bodies = bodies
        self.ATBI = ATBI(bodies)
        Theta_0 = [b.initialcondition.theta0 for b in bodies]
        Beta_0 = [b.initialcondition.beta0 for b in bodies]
        self.S0 = SystemState(Theta_0, Beta_0).pack()
        
    def EOM(self, t, S):
            state = SystemState.unpack(S.reshape(-1, 1), [b.joint for b in self.bodies])

            # Normalize quaternions
            for k, body in enumerate(self.bodies):
                if body.joint.type in ["spherical", "free"]:
                    q = state.Theta[k][0:4]
                    state.Theta[k][0:4] = q / np.linalg.norm(q)

            X, V, a, b = self.ATBI.scatter_kinematics(state)
            G, nu = self.ATBI.gather_ATBI(a, b, X)
            gamma, alpha = self.ATBI.scatter_ATBI(a, X, G, nu)

            Theta_dot = []
            for k, body in enumerate(self.bodies):
                if body.joint.type.startswith("rev"):
                    Theta_dot.append(state.Beta[k].reshape(1, 1))
                elif body.joint.type == "spherical":
                    Theta_dot.append(sb.quat_derivative(state.Theta[k], state.Beta[k]).reshape(4, 1))
                elif body.joint.type == "free":
                    qdot = sb.quat_derivative(state.Theta[k][0:4], state.Beta[k][0:3]).reshape(4, 1)
                    Theta_dot.append(np.vstack([qdot, state.Beta[k][3:6]]).reshape(7, 1))
                elif body.joint.type == "fixed":
                    Theta_dot.append(np.zeros((0,1)))

            S_dot = np.vstack([*Theta_dot, *gamma]).flatten()
            return S_dot

class Simulation:
    class Data:
        def __init__(self):
            self.time, self.state, self.X_list, self.V_list, self.a_list, self.b_list, self.alpha_list, self.pos = [], [], [], [], [], [], [], []

    class Setting:
        def __init__(self):
            self.camera_speed = 0
            self.camera_ver = 20
            self.camera_hor = 0

    def __init__(self, system: MultibodySystem, tf, dt):
        self.system = system
        self.data = self.Data()
        self.setting = self.Setting()
        self.tf = tf
        self.dt = dt

    def IntegrateSystem(self):
        t_eval = np.linspace(0, self.tf, int(self.tf/self.dt)+1)

        sol = solve_ivp(
            fun=self.system.EOM,
            t_span=(0, self.tf),
            y0=self.system.S0,
            t_eval=t_eval,
            method="RK45"
        )

        print("Integration successful!")
    
        # Extract results to match [t, y] format
        self.data.time = sol.t
        states = sol.y.T

        # Find X-vector for each time step
        for i in range(len(self.data.time)):

            # Unpack state            
            current_state = SystemState.unpack(states[i].reshape(-1, 1), [b.joint for b in self.system.bodies])
            
            # Kinematic scatter loop to find X
            X, V, a, b = self.system.ATBI.scatter_kinematics(current_state)
            G, nu = self.system.ATBI.gather_ATBI(a, b, X)
            gamma, alpha = self.system.ATBI.scatter_ATBI(a, X, G, nu)

            # Add to list
            self.data.state.append(current_state)
            self.data.X_list.append(X)
            self.data.V_list.append(V)
            self.data.a_list.append(a)
            self.data.b_list.append(b)
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
    def camera_speed(self, x):
        self.setting.camera_speed = x

    def camera_ver(self, x):
        self.setting.camera_ver = x

    def camera_hor(self, x):
        self.setting.camera_hor = x

    def nBodyPos(self):
        # Takes time vector, t and X-vector [q, klOO]^T and returns hinge positions            

        t = self.data.time
        X = self.data.X_list
        klOO_B = [b.joint.klOO for b in self.system.bodies]

        # Number of bodies and time steps
        n = len(X[0])
        nt = len(t)

        # Setup hinge position list
        penPos = []

        dxyz = np.zeros((3, 1))

        for i in range(nt):
            # Account for possible free BASE hinge
            if  self.system.bodies[-1].joint.type == "free":
                theta_base_free = self.data.state[i].Theta[-1]
                dxyz = theta_base_free[4:7]

            kpos = [None] * (n + 1)

            kpos[n] = np.zeros((3, 1)) 
            Ri = np.eye(3)
            
            for k in range(n - 1, -1, -1):
                # Unpancking X-vector
                q = X[i][k][0:4]    # Quaternion: k+1 to k
                klOO = X[i][k][4:7] # O_k to O_k-1^+

                # Rotation
                Ri = Ri @ sb.q2R(q.flatten(), 3)

                # Hinge position
                kpos[k] = kpos[k+1] + Ri @ klOO

            # Rotation of base body k = -1
            q = X[i][-1][0:4]
            R_base = sb.q2R(q.flatten(), 3)

            # This will account for "free" base body hinge   
            kpos[-1] = kpos[-2] - R_base @ klOO_B[-1]

            # Add to pendulum position list, penPos
            kpos = kpos - dxyz
            penPos.append(kpos)

        return penPos

    def get_pos(self):
        self.data.pos = self.nBodyPos()
        return self.data.pos

    def animate(self, filename="", save_dir =""):
        # Takes X-vector list and returns simulation
        
        t = self.data.time
        X = self.data.X_list

        # Number of bodies
        n = len(X[0])

        # Number of time steps and dt
        nt = len(t)
        dt = t[1] - t[0]

        # Get position for each time step
        penPos = self.nBodyPos()

        # Initialize animation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Determine Axis Limits
        all_points = []

        for i in range(0, nt, 10): # Sample every 10th frame for speed
            for body in penPos[i]:
                all_points.append(body.flatten()) # Flatten (3,1) to (3,)

        all_points = np.array(all_points)
        max_range = np.abs(all_points).max()
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, 0)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"n-Body Pendulum with ({len(penPos[0]) - 1} Bodies)")

        # Create n colored lines (one per link)
        cmap = mpl.colormaps['tab10']
        colors = cmap(np.linspace(0, 1, n))        
        lines = []

        for i in range(n):
            line, = ax.plot([], [], [], '-', lw=4, markersize=4, color=colors[i])
            lines.append(line)
        
        joint_dots, = ax.plot([], [], [], 'ko', markersize=4)

        ax.plot([0], [0], [0], 'o', color='gray', markersize=6)

        # Initialize the timer text (placed in top-left corner)
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

        def update(frame_idx):
            current_state = penPos[frame_idx]

            # Extract joint positions
            xs = [float(body[0][0]) for body in current_state]
            ys = [float(body[1][0]) for body in current_state]
            zs = [float(body[2][0]) for body in current_state]

            # Update each link separately
            for i in range(n):
                lines[i].set_data(xs[i:i+2], ys[i:i+2])
                lines[i].set_3d_properties(zs[i:i+2])

            joint_dots.set_data(xs, ys)
            joint_dots.set_3d_properties(zs)

            # Update timer
            time_text.set_text(f'Time: {t[frame_idx]:.2f} s')

            ax.view_init(elev=self.setting.camera_ver, azim=frame_idx * self.setting.camera_speed * 40 * dt + self.setting.camera_hor)
        
            return (*lines, joint_dots, time_text)
    
        # Create Animation
        anim = FuncAnimation(
        fig, 
        update, 
        frames=len(penPos), 
        interval=dt*1000, 
        blit=False)

        if filename != "":            
            print("Rendering animation to HTML... (This may take a minute)")
            
            filename = filename + ".html"

            # Use the 'html' writer
            os.makedirs(save_dir, exist_ok=True)

            fullpath = os.path.join(save_dir, filename)

            with open(fullpath, "w") as f:
                f.write(anim.to_jshtml())
            
            print("Renedering of animation: Done!")
            print(f"Saved to {fullpath}")

        else:
            plt.show()

""" ------ File Setup ------ """
# Remember to run this:
# from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation

""" ------ Body Setup ------ """
# *** Body Parameters ****
# klOO:     Hinge position (row vector)
# H_type:   Hinge type (string)
# m:        Mass (scalar)
# CkJk:     Inertia (row vector)
# klOC:     COM position (row vector)

# *** Create Body ***
# joint = Joint(<klOO>, <H_type>)
# inertia = Inertia(<m>, <CkJk>, <klOC>)
# body = SOABody(<joint>, <inertia>)

""" ------ Body Attributes ------ """ 
# If not specified program assumes zero column vectors
# theta0, beta0, tau, F_ext ---> column vectors

# *** Initial condition ***
# body.set_initial_theta0(<theta0>)   //   <theta0> ---> column vector
#       --->   "revx/y/z" use: theta0 = np.deg2rad(theta_x/y/z)
#       --->   "spherical" use: theta0 = q0 = sb.get_quat_from_degrees(theta_x, theta_y, theta_z)
#       --->   "free" use: theta0 = np.vstack([q0, l]), where l is the initial linear displacement (l = [l_x, l_y, l_z])
#       --->   "fixed" use: theta0 cannot be specified
# body.set_initial_beta0(<beta0>)   //   <beta0> ---> column vector 
#       --->   "revx/y/z" use: beta0 = omega_x/y/z
#       --->   "spherical" use: beta0 = np.array([omega_x, omega_y, omega_z]).reshape(3, 1)
#       --->   "free" use: beta0 = np.array([omega_x, omega_y, omega_z, v_x, v_y, v_z]), where v is the initial linear velocity (v = [v_x, v_y, v_z])
#       --->   "fixed" use: beta0 cannot be specified

# *** Forces ***
# body.set_tau(<tau>)   //   <tau> ---> column vector, np.array([<tau>]).reshape(nDOF, 1)

# body.set_F_ext(<F_ext>, <klBO>)   //   <F_ext>, <klBO> ---> lists of same length 
#       --->   F_ext is a list of column vectors (6, 1) with external forces
#       --->   klBO is a list of row vectors (1, 3) with the external forces' appliying position

""" ------ System Setup and Simulation ------ """ 
# *** Multibody System ***
# system = MultibodySystem(bodies)
#       --->   bodies = [body_1, body_2, ..., body_n], list of bodies created above (tip: b_1 and base: b_n)

# *** Simulation Setup ***
# sim = Simulation(system, tf, dt)
#       --->   system as created above
#       --->   tf, length of simulation
#       --->   dt, time step size

""" ------ Camera Settings ------ """ 
# sim.camera_speed(x)
#       --->   x, number from -1..1 defining the speed in both directions (zero if not changed)

# sim.camera_hor(x)
#       --->   x, number from 0..360 defining the camera rotation around z-axis (zero if not changed)

# sim.camera_ver(x)
#       --->   x, number from -90..90 defining the camera rotation around x-axis (20 if not changed)

""" ------ Parameter Call ------ """
# *** Parameter Call ***   //   Get parameter for each body for each time step
# sim.get_<parameter>()
#       --->   <parameter> = [state, X, pos, V, alpha, a, b]

""" ------ Animate or Render Animation ------ """
# *** Animate ***
# sim.animate()

# *** Render Animation to HTML-file ***   // <file_name>, r<file_path> ---> strings
# sim.animate(<file_name>, <file_path>)
#       --->   <file_name>, name of the render file
#       --->   <file_path>, copy the file path of the folder you want to save the HTML-file in.
#               nb! You must add the letter <r> in front of file path string as:
#               file_path = r"C:\Users\jepp6\OneDrive..."
#               Choose another folder than the GIT-Hub synchronize folder, since the file will
#               be to big and result in a "commit" error.

# To DOOOOOOOOOOOOO
# Mass import in inertia is not consistent with rho