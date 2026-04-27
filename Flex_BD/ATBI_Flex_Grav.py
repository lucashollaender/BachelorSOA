import numpy as np
import scipy.linalg as la
from SOALIB import soalib as sb
from SystemState import SystemState
import pandas as pd


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
        a02 = sb.skew(V_p[0:3]) @ sb.skew(V_p[0:3]) @ klOO

        return np.vstack([a01, R3.T @ a02])
    
    def gyroscopic(self, V, M):
        return sb.bar6(V) @ M @ V

    def gyroscopic_BD(self, body, V_r, m):
        n_md = body.flex.n_md

        # modal integrals
        p_0 = body.flex.p_0
        F_1 = body.flex.F_1
        J_0 = body.flex.J_0
        J_1 = body.flex.J_1
        S_1 = body.flex.S_1

        omega = V_r[0:3, :]

        b_eta = np.zeros((n_md, 1))

        for i in range(n_md):
            b_eta[i] = - omega.T @ (S_1[:, 3 * i: 3 * i + 3] @ J_1[:, 3 * i: 3 * i + 3]) @ omega

        return np.vstack([b_eta, sb.skew(omega) @ J_0 @ omega, m * sb.skew(omega) @ sb.skew(omega) @ p_0])

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
            #print(ang)
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
            b_fl[k] = self.gyroscopic_BD(body, V_r[k], body.m)

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
            F_ext = body.force.F_ext
            tau_pr = body.force.tau
            eta = state.Eta[k]
            eta_dot = state.Eta_dot[k]
            M_fl = body.flex.M_fl
            K_fl = body.flex.K_fl
            C_fl = body.flex.C_fl
            PI = body.flex.PI_end
            n_md = body.flex.n_md
            H_M_fl = np.hstack([np.eye(n_md, n_md), np.zeros((n_md, 6))])
            klOO = X[k][4:7]

            # External force
            # F_ext_term = np.zeros((b_fl[k].shape[0], 1))
            #F_ext_term = np.exp(- 5 * t) * np.vstack([PI.T @ F_ext, sb.phi(klOO) @ F_ext])

            F_ext_term = np.vstack([PI.T @ F_ext, sb.phi(klOO) @ F_ext])

            """
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
                D_m_inv = body.get_D_m_inv(Gamma_fl, 0)
                g_fl[k] = mu_fl @ D_m_inv
                P_pr[k] = P_fl[-6:, -6:] - g_fl[k] @ mu_fl.T
                D_pr[k] = H_B @ P_pr[k] @ H_B.T
                G_pr[k] = P_pr[k] @ la.solve(D_pr[k].T, H_B).T
                tau_pr_bar = np.eye(6, 6) - G_pr[k] @ H_B
                P_pr_plus[k] = tau_pr_bar @ P_pr[k]

                # 13.7
                z = b_fl[k] + K_fl @ np.vstack([eta, np.zeros((6, 1))]) - F_ext_term + C_fl @ np.vstack([eta_dot, np.zeros((6, 1))])
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
                D_m_inv = body.get_D_m_inv(Gamma_fl, 1)
                g_fl[k] = mu_fl @ D_m_inv
                P_pr[k] = P_fl[-6:, -6:] - g_fl[k] @ mu_fl.T
                D_pr[k] = H_B @ P_pr[k] @ H_B.T
                G_pr[k] = P_pr[k] @ la.solve(D_pr[k].T, H_B).T
                tau_pr_bar = np.eye(6, 6) - G_pr[k] @ H_B
                P_pr_plus[k] = tau_pr_bar @ P_pr[k]

                # 13.7
                z = A_fl @ R6 @ z_pr_plus[k-1] + b_fl[k] + K_fl @ np.vstack([eta, np.zeros((6, 1))]) - F_ext_term + C_fl @ np.vstack([eta_dot, np.zeros((6, 1))])
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