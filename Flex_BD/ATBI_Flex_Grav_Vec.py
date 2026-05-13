import numpy as np
import scipy.linalg as la
from SOALIB import soalib as sb
from SystemState import SystemState

class ATBI_Flex:
    def __init__(self, bodies):
        # Parameters
        self.bodies = bodies
        self.n = len(bodies)

        # Spatial gravity
        self.g = np.array([0, 0, 0, 0, 0, 9.81]).reshape(6, 1)

        # Precompute static operators and slices
        self.A_fl = [None] * self.n
        self.K_fl_trunc = [None] * self.n
        self.C_fl_trunc = [None] * self.n
        
        for k in range(self.n):
            body = bodies[k]
            # Precompute constant structural mapping
            self.A_fl[k] = sb.get_A(body.flex.PI_end, body.joint.klOO)
            
            # Pre-slice K_fl and C_fl to avoid multiplying by zeros in the hot loop
            n_md = body.flex.n_md
            if n_md > 0:
                self.K_fl_trunc[k] = body.flex.K_fl[:, :n_md]
                self.C_fl_trunc[k] = body.flex.C_fl[:, :n_md]
            else:
                self.K_fl_trunc[k] = np.zeros((6, 0))
                self.C_fl_trunc[k] = np.zeros((6, 0))

        # Pass-to-pass state cache to avoid redundant recalculations
        self.R6_list = [None] * self.n

        # PRE-ALLOCATION of lists to eliminate memory thrashing in the hot loop
        self.X = [None] * self.n
        self.V = [None] * self.n
        self.V_f = [None] * self.n
        self.V_r = [None] * self.n
        self.a_fl = [None] * self.n
        self.b_fl = [None] * self.n

        self.P_pr_plus = [None] * self.n
        self.D_m = [None] * self.n
        self.g_fl = [None] * self.n
        self.P_pr = [None] * self.n
        self.D_pr = [None] * self.n
        self.G_pr = [None] * self.n
        self.nu_m = [None] * self.n
        self.nu_pr = [None] * self.n
        self.z_pr_plus = [None] * self.n

        self.alpha_fl = [None] * self.n
        self.theta_ddot = [None] * self.n
        self.eta_ddot = [None] * self.n

    def scatter_kinematics(self, state: SystemState):
        n = self.n

        for k in reversed(range(n)):
            body = self.bodies[k]
            joint = body.joint
            theta = state.Theta[k]
            beta = state.Beta[k]
            eta_dot = state.Eta_dot[k]
            H = body.joint.H

            # Build X and rotations
            self.X[k], q = joint.get_theta2X(theta)
            q_flat = q.flatten()
            
            # Compute R3 and R6 ONLY ONCE per step and cache R6
            R3 = sb.q2R(q_flat, 3)
            R6 = sb.q2R(q_flat, 6)
            self.R6_list[k] = R6

            if k == n - 1:
                # Base body
                self.V_f[k] = eta_dot
                self.V_r[k] = H.T @ beta
                self.a_fl[k] = body.coriolis_BD(self.V_r[k], np.zeros((6, 1)), beta, H, np.zeros((3, 1)), R3)
            else:
                self.V_f[k] = eta_dot
                self.V_r[k] = R6.T @ self.A_fl[k+1].T @ self.V[k+1] + H.T @ beta
                self.a_fl[k] = body.coriolis_BD(self.V_r[k], self.V_r[k+1], beta, H, R3.T @ self.X[k+1][4:7], R3)

            self.b_fl[k] = body.gyroscopic_BD(body, self.V_r[k], body.m)
            self.V[k] = np.vstack([self.V_f[k], self.V_r[k]])

        return self.X, self.V, self.a_fl, self.b_fl

    def gather_ATBI(self, state: SystemState, a_fl, b_fl, X, t):
        n = self.n

        for k in range(n):
            body = self.bodies[k]
            H_B = body.joint.H
            tau_pr = body.force.tau
            theta = state.Theta[k]
            beta = state.Beta[k]
            eta = state.Eta[k]
            eta_dot = state.Eta_dot[k]
            M_fl = body.flex.M_fl
            n_md = body.flex.n_md
            
            # Retrieve precomputed truncated stiffness/damping
            K_eta = self.K_fl_trunc[k] @ eta if n_md > 0 else 0
            C_etadot = self.C_fl_trunc[k] @ eta_dot if n_md > 0 else 0

            F_ext_term = body.get_F_ext_term(state, t)
            tau_TS_term = body.get_TS_term(theta, beta)

            if k == 0:
                Gamma_fl = np.zeros((0, 6))
                P_fl = M_fl
                self.D_m[k] = P_fl[:n_md, :n_md]
                mu_fl = P_fl[-6:, :n_md]
                D_m_inv = body.get_D_m_inv(Gamma_fl, "tip")
                self.g_fl[k] = mu_fl @ D_m_inv
                self.P_pr[k] = P_fl[-6:, -6:] - self.g_fl[k] @ mu_fl.T
                self.D_pr[k] = H_B @ self.P_pr[k] @ H_B.T
                
                # REVERTED TO SAFE SCIPY SOLVE
                self.G_pr[k] = self.P_pr[k] @ la.solve(self.D_pr[k].T, H_B).T
                
                tau_pr_bar = np.eye(6, 6) - self.G_pr[k] @ H_B
                self.P_pr_plus[k] = tau_pr_bar @ self.P_pr[k]

                # Optimized Z calculation (no np.vstack or zero padding)
                z = b_fl[k] + K_eta - F_ext_term + C_etadot
                
                eps_m = - z[0:n_md]
                self.nu_m[k] = D_m_inv @ eps_m

                z_pr = z[-6:] + self.g_fl[k] @ eps_m + self.P_pr[k] @ a_fl[k][-6:]
                eps_pr = tau_pr - H_B @ z_pr + tau_TS_term
                
                # REVERTED TO SAFE SCIPY SOLVE
                self.nu_pr[k] = la.solve(self.D_pr[k], eps_pr)
                self.z_pr_plus[k] = z_pr + self.G_pr[k] @ eps_pr

            else:
                # Use cached R6 from step 1 instead of recomputing
                R6 = self.R6_list[k-1]
                A_fl = self.A_fl[k]

                Gamma_fl = R6 @ self.P_pr_plus[k-1] @ R6.T 
                P_fl = A_fl @ Gamma_fl @ A_fl.T + M_fl
                
                self.D_m[k] = P_fl[:n_md, :n_md]
                mu_fl = P_fl[-6:, :n_md]
                D_m_inv = body.get_D_m_inv(Gamma_fl, "not_tip")
                self.g_fl[k] = mu_fl @ D_m_inv
                self.P_pr[k] = P_fl[-6:, -6:] - self.g_fl[k] @ mu_fl.T
                self.D_pr[k] = H_B @ self.P_pr[k] @ H_B.T
                
                # REVERTED TO SAFE SCIPY SOLVE
                self.G_pr[k] = self.P_pr[k] @ la.solve(self.D_pr[k].T, H_B).T
                
                tau_pr_bar = np.eye(6, 6) - self.G_pr[k] @ H_B
                self.P_pr_plus[k] = tau_pr_bar @ self.P_pr[k]

                # Optimized Z calculation
                z = A_fl @ R6 @ self.z_pr_plus[k-1] + b_fl[k] + K_eta - F_ext_term + C_etadot
                
                eps_m = - z[0:n_md]
                self.nu_m[k] = D_m_inv @ eps_m

                z_pr = z[-6:] + self.g_fl[k] @ eps_m + self.P_pr[k] @ a_fl[k][-6:]
                eps_pr = tau_pr - H_B @ z_pr + tau_TS_term
                
                # REVERTED TO SAFE SCIPY SOLVE
                self.nu_pr[k] = la.solve(self.D_pr[k], eps_pr)
                self.z_pr_plus[k] = z_pr + self.G_pr[k] @ eps_pr

        return self.G_pr, self.nu_pr, self.nu_m, self.g_fl

    def scatter_ATBI(self, a_fl, X, G_pr, nu_pr, nu_m, g_fl):
        n = self.n

        for k in range(n - 1, -1, -1):
            body = self.bodies[k]
            H_B = body.joint.H
            
            # Use precomputed cached variables! 
            R6 = self.R6_list[k]
            
            if k == n - 1:
                alpha_base = R6.T @ self.g
                self.theta_ddot[k] = nu_pr[k] - G_pr[k].T @ alpha_base
                alpha_pr = alpha_base + H_B.T @ self.theta_ddot[k] + a_fl[k][-6:]
                self.eta_ddot[k] = nu_m[k] - g_fl[k].T @ alpha_pr
                self.alpha_fl[k] = np.vstack([self.eta_ddot[k], alpha_pr])
            else:
                alpha_pr_plus = R6.T @ self.A_fl[k+1].T @ self.alpha_fl[k+1]
                self.theta_ddot[k] = nu_pr[k] - G_pr[k].T @ alpha_pr_plus
                alpha_pr = alpha_pr_plus + H_B.T @ self.theta_ddot[k] + a_fl[k][-6:]
                self.eta_ddot[k] = nu_m[k] - g_fl[k].T @ alpha_pr
                self.alpha_fl[k] = np.vstack([self.eta_ddot[k], alpha_pr])

        return self.theta_ddot, self.eta_ddot, self.alpha_fl