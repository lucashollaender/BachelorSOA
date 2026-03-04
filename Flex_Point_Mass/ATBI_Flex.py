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
            eta_dot = state.Eta_dot[k]
            H = body.joint.H
            Mk = body.rigid.Mk
            PI = body.flex.PI_end
            n_md = body.flex.n_md

            # Build X
            X[k], q = self.theta2X(theta, body.joint.type, body.joint.klOO)

            # Build A: NB! Typo in text?!?!
            R3 = sb.q2R(q.flatten(), 3)
            A_fl[k] = sb.get_A(PI, R3.T @ X[k][4:7])

            if k == n - 1:
                V_f[k] = eta_dot
                V_r[k] = H.T @ beta
            else:
                R6 = sb.q2R(q.flatten(), 6)
                R_tot = sb.get_R_tot(R6, n_md)

                V_f[k] = eta_dot
                V_r[k] = A_fl[k+1].T @ R_tot.T @ V[k+1] + H.T @ beta

            a_fl[k] = np.vstack(
                [np.zeros((n_md, 1)), self.coriolis(V_r[k], beta, H)])
            b_fl[k] = np.vstack(
                [np.zeros((n_md, 1)), self.gyroscopic(V_r[k], Mk)])

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
            M_fl = body.flex.M_fl
            K_fl = body.flex.K_fl
            PI = body.flex.PI_end
            n_md = body.flex.n_md
            H_M_fl = np.hstack([np.eye(n_md, n_md), np.zeros((n_md, 6))])
            klOO = X[k][4:7]

            # External force
            F_ext_term = np.zeros((b_fl[k].shape[0], 1))

            if t <= 0.25:
                F_ext_term = np.vstack([PI.T @ F_ext, sb.phi(klOO) @ F_ext])

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
                z = b_fl[k] + \
                    K_fl @ np.vstack([eta, np.zeros((6, 1))]) - F_ext_term
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
                z = A_fl @ R6 @ z_pr_plus[k-1] + b_fl[k] + \
                    K_fl @ np.vstack([eta, np.zeros((6, 1))]) - F_ext_term
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

            # Spatial gravity rotation
            Ri[k] = Ri[k+1]  @ R6

            A_fl[k] = sb.get_A(PI, R3.T @ X[k][4:7])

            if k == n - 1:
                # Scatter loop (Tip of the chain)
                theta_ddot[k] = nu_pr[k]
                alpha_pr = H_B.T @ theta_ddot[k] + a_fl[k][-6:]
                eta_ddot[k] = nu_m[k] - g_fl[k].T @ alpha_pr
                alpha_fl[k] = np.vstack([eta_ddot[k], alpha_pr])

            else:
                # Scatter loop
                R_tot = sb.get_R_tot(R6, n_md)

                alpha_pr_plus = A_fl[k+1].T @ R_tot.T @ alpha_fl[k+1]
                theta_ddot[k] = nu_pr[k] - G_pr[k].T @ alpha_pr_plus
                alpha_pr = alpha_pr_plus + H_B.T @ theta_ddot[k] + a_fl[k][-6:]
                eta_ddot[k] = nu_m[k] - g_fl[k].T @ alpha_pr
                alpha_fl[k] = np.vstack([eta_ddot[k], alpha_pr])

        return theta_ddot, eta_ddot, alpha_fl
