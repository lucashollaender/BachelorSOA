import numpy as np
import pandas as pd
from SOALIB import soalib as sb
from SystemState import SystemState
from ATBI_Flex_Grav import ATBI_Flex


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
