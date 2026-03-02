import numpy as np
import pandas as pd

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