import numpy as np
import scipy.linalg as la
from SOALIB import soalib as sb
from SystemState import SystemState
import pandas as pd
from scipy.spatial.transform import Rotation

class ATBI_Flex:
    # ATBI class with bodies
    def __init__(self, bodies):
        # Parameters
        self.bodies = bodies
        self.n = len(bodies)

        # Operators
        self.A_fl = [None] * self.n

        for k in range(self.n):
            body = bodies[k]
            self.A_fl[k] = sb.get_A(body.flex.PI_end, body.joint.klOO)

    def scatter_kinematics(self, state: SystemState):
        # Step 1 of ATBI (scatter sweep): Takes state and returns velocities, pose, Coriolis- and gyroscopic
        # terms

        # Number of bodies
        n = self.n

        # Set up lists
        X = [None] * n
        V = [None] * n
        V_f = [None] * n
        V_r = [None] * n
        a_fl = [None] * n
        b_fl = [None] * n
        A_fl = self.A_fl
        R3_list = [None] * n

        # Track
        pos = [None] * n
        pos_dot = [None] * n
        R_i = [None] * (n+1)
        R_i[n] = np.eye(3)

        for k in range(n - 1, -1, -1):
            # Parameters of the body
            body = self.bodies[k]
            joint = body.joint
            theta = state.Theta[k]
            beta = state.Beta[k]
            eta = state.Eta[k]
            eta_dot = state.Eta_dot[k]
            H = body.joint.H
            Mk = body.rigid.Mk
            PI = body.flex.PI_end
            n_md = body.flex.n_md

            # Build X
            X[k], q = joint.get_theta2X(theta)

            if k == n - 1:
                # Base body
                # Rotation
                R_j = sb.q2R(q.flatten(), 3) 
                R3_n = np.eye(3)
                R3 = R3_n @ R_j
                R3_list[k] = R3

                V_f[k] = eta_dot
                V_r[k] = H.T @ beta

                # Coriolis Acceleration (base)
                a_fl[k] = body.coriolis_BD(V_r[k], np.zeros(
                    (6, 1)), beta, H, np.zeros((3, 1)), R3)
                
                # Last node position of last body 
                last_end = np.zeros((3, 1))
                last_end_dot = np.zeros((3, 1))
            else:
                # Rotation
                R3_j = sb.q2R(q.flatten(), 3)
                R3 = R3_n @ R3_j
                R6 = sb.get_R6(R3)
                R3_list[k] = R3
                
                V_f[k] = eta_dot
                V_r[k] = R6.T @ A_fl[k+1].T @ V[k+1] + H.T @ beta

                # Coriolis Acceleration
                a_fl[k] = body.coriolis_BD(
                    V_r[k], V_r[k+1], beta, H, R3.T @ X[k+1][4:7], R3)

                # Last node position and velocity of last body
                last_end = pos[k+1][-1]
                last_end_dot = pos_dot[k+1][-1]

            # Gyroscopic Force
            b_fl[k] = body.gyroscopic_BD(body, V_r[k], body.m)

            # Rigid formulations
            # a_fl[k] = body.coriolis(V_r[k], beta, H, n_md)
            # b_fl[k] = np.vstack([np.zeros((n_md, 1)), body.gyroscopic(V_r[k], body.flex.M_fl[-6:, -6:])])

            V[k] = np.vstack([V_f[k], V_r[k]])

            pos[k], pos_dot[k], R_i[k] = body.get_track_kin(last_end, last_end_dot, R_i[k+1], R3, V[k][-6:, 0], eta, eta_dot)

            # Last end rotation
            R3_n_vec = PI[0:3, :] @ eta 
            R3_n = Rotation.from_rotvec(R3_n_vec.flatten()).as_matrix()
        
        return X, R3_list, V, a_fl, b_fl, pos, pos_dot, R_i

    def gather_ATBI(self, state: SystemState, a_fl, b_fl, X, R3_list, pos, pos_dot, R_i, t):
        # Step 2 of ATBI (gather sweep): Takes generalized forces, Coriolis-, gyroscopic
        # terms, X-vector and system configuration and returns G and nu parameters

        # Number of bodies
        n = self.n

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
        A_fl = self.A_fl

        for k in range(n):
            # Parameters of the body
            body = self.bodies[k]
            H_B = body.joint.H
            tau_pr = body.force.tau
            theta = state.Theta[k]
            beta = state.Beta[k]
            eta = state.Eta[k]
            eta_dot = state.Eta_dot[k]
            M_fl = body.flex.M_fl
            K_fl = body.flex.K_fl
            C_fl = body.flex.C_fl
            PI = body.flex.PI_end
            n_md = body.flex.n_md
            # H_M_fl = np.hstack([np.eye(n_md, n_md), np.zeros((n_md, 6))])

            # Applied loads and springs
            F_ext_term = body.get_F_ext_term(state, t)
            tau_TS_term = body.get_TS_term(theta, beta)

            # Track
            F_ext_term += body.get_global_forces_term(pos[k], pos_dot[k], R_i[k])

            if k == 0:
                # Gather loop for k = 0 (Tip)
                Gamma_fl = np.zeros((0, 6))
                P_fl = M_fl
                # D_m[k] = H_M_fl @ P_fl @ H_M_fl.T
                # mu_fl = P_fl[-6:, :] @ H_M_fl.T
                D_m[k] = P_fl[:n_md, :n_md]
                mu_fl = P_fl[-6:, :n_md]
                D_m_inv = body.get_D_m_inv(Gamma_fl, "tip")
                g_fl[k] = mu_fl @ D_m_inv
                P_pr[k] = P_fl[-6:, -6:] - g_fl[k] @ mu_fl.T
                D_pr[k] = H_B @ P_pr[k] @ H_B.T
                G_pr[k] = P_pr[k] @ la.solve(D_pr[k].T, H_B).T
                tau_pr_bar = np.eye(6, 6) - G_pr[k] @ H_B
                P_pr_plus[k] = tau_pr_bar @ P_pr[k]

                z = b_fl[k] +  K_fl @ np.vstack([eta, np.zeros((6, 1))]) - \
                    F_ext_term + C_fl @ np.vstack([eta_dot, np.zeros((6, 1))])
                eps_m = - z[0:n_md]  # tau_m (assumed to be zero): dim(n_md, 1)
                nu_m[k] = D_m_inv @ eps_m

                z_pr = z[-6:] + g_fl[k] @ eps_m + P_pr[k] @ a_fl[k][-6:]
                eps_pr = tau_pr - H_B @ z_pr + tau_TS_term
                nu_pr[k] = la.solve(D_pr[k], eps_pr)
                z_pr_plus[k] = z_pr + G_pr[k] @ eps_pr

            else:
                # Rotation
                R6 = sb.get_R6(R3_list[k-1])

                # Gather loop for k > 0 (Not tip)
                Gamma_fl = R6 @ P_pr_plus[k-1] @ R6.T
                P_fl = A_fl[k] @ Gamma_fl @ A_fl[k].T + M_fl
                # D_m[k] = H_M_fl @ P_fl @ H_M_fl.T
                # mu_fl = P_fl[-6:, :] @ H_M_fl.T
                D_m[k] = P_fl[:n_md, :n_md]
                mu_fl = P_fl[-6:, :n_md]
                # D_m_inv = la.solve(D_m[k], np.eye(n_md), assume_a="sym")
                D_m_inv = body.get_D_m_inv(Gamma_fl, "not_tip")
                g_fl[k] = mu_fl @ D_m_inv
                P_pr[k] = P_fl[-6:, -6:] - g_fl[k] @ mu_fl.T
                D_pr[k] = H_B @ P_pr[k] @ H_B.T
                G_pr[k] = P_pr[k] @ la.solve(D_pr[k].T, H_B).T
                tau_pr_bar = np.eye(6, 6) - G_pr[k] @ H_B
                P_pr_plus[k] = tau_pr_bar @ P_pr[k]

                z = A_fl[k] @ R6 @ z_pr_plus[k-1] + b_fl[k] + K_fl @ np.vstack([eta, np.zeros((6, 1))]) - \
                    F_ext_term + C_fl @ np.vstack([eta_dot, np.zeros((6, 1))])
                eps_m = - z[0:n_md]  # tau_m (assumed to be zero): dim(n_md, 1)
                nu_m[k] = D_m_inv @ eps_m

                z_pr = z[-6:] + g_fl[k] @ eps_m + P_pr[k] @ a_fl[k][-6:]
                eps_pr = tau_pr - H_B @ z_pr + tau_TS_term
                nu_pr[k] = la.solve(D_pr[k], eps_pr)
                z_pr_plus[k] = z_pr + G_pr[k] @ eps_pr

        return G_pr, nu_pr, nu_m, g_fl, P_pr_plus, z_pr_plus

    def scatter_ATBI(self, a_fl, X, R3_list, G_pr, nu_pr, nu_m, g_fl, P_pr_plus, z_pr_plus):
        # Step 3 of ATBI (second scatter sweep): Takes found parameters in gather sweep and returns generalized acceleration and eta_ddot

        # Number of bodies
        n = self.n

        # Setup of list
        alpha_fl = [None] * n
        theta_ddot = [None] * n
        eta_ddot = [None] * n
        F_int = [None] * n
        A_fl = self.A_fl

        # Loop backwards from n-1 down to 0
        for k in range(n - 1, -1, -1):
            # Parameters of the body
            body = self.bodies[k]
            H_B = body.joint.H
            R6 = sb.get_R6(R3_list[k])

            if k == n - 1:
                alpha_pr_plus = np.zeros((6, 1))
                alpha_base = R6.T @ self.g
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
            
            # Inter-link-forces
            F_int[k] = P_pr_plus[k] @ alpha_pr_plus + z_pr_plus[k]
            #print(k)
            #print(F_int[k][3, 0])

        return theta_ddot, eta_ddot, alpha_fl, F_int