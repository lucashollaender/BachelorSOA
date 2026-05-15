import numpy as np
import pandas as pd
from SOALIB import soalib as sb
from SystemState import SystemState
from ATBI_Flex_Grav_Track import ATBI_Flex

class MultibodySystem:
    def __init__(self, bodies):
        self.bodies = bodies
        self.joints = [b.joint for b in self.bodies]
        self.flexs = [b.flex for b in self.bodies]
        self.ATBI = ATBI_Flex(bodies)
        Theta_0 = [b.initialcondition.theta0 for b in bodies]
        Beta_0 = [b.initialcondition.beta0 for b in bodies]
        Eta_0 = [b.initialcondition.eta0 for b in bodies]
        Eta_dot_0 = [b.initialcondition.eta_dot0 for b in bodies]
        self.S0 = SystemState(Theta_0, Beta_0, Eta_0, Eta_dot_0).pack()
        self.ATBI.g = np.array([0, 0, 0, 0, 0, 9.81]).reshape(6, 1)

    def set_gravity(self, gravOnOff):
        if gravOnOff:
            self.ATBI.g = np.array([0, 0, 0, 0, 0, 9.81]).reshape(6, 1)
        else:
            self.ATBI.g = np.zeros((6, 1))

    def EOM(self, t, S):
        state = SystemState.unpack(S.reshape(-1, 1), self.joints, self.flexs)
        
        for k, body in enumerate(self.bodies):
            if body.joint.type in ["spherical"]:
                q = state.Theta[k][0:4]
                state.Theta[k][0:4] = q / np.linalg.norm(q)
            
        X, V, a_fl, b_fl, pos, pos_dot, R_i = self.ATBI.scatter_kinematics(state)
        G_pr, nu_pr, nu_m, g_fl = self.ATBI.gather_ATBI(
            state, a_fl, b_fl, X, pos, pos_dot, R_i, t)
        theta_ddot, eta_ddot, alpha_fl = self.ATBI.scatter_ATBI(
            a_fl, X, G_pr, nu_pr, nu_m, g_fl)

        # S_dot setup 
        Theta_dot, Eta_dot_list = [], []
        for k, body in enumerate(self.bodies):
            Theta_dot.append(body.joint.get_theta_dot(state.Theta[k], state.Beta[k]))
            Eta_dot_list.append(state.Eta_dot[k].reshape(body.flex.n_md, 1))

        S_dot = np.vstack([*Theta_dot, *theta_ddot, *Eta_dot_list, *eta_ddot]).flatten()

        return S_dot
