import numpy as np
import scipy as sp
import scipy.linalg as la
from SOALIB import soalib as sb

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

class Rigid_Properties:
# Inertia class with m, CkJk and klOC
    def __init__(self, rho, klOC):
        # Parameters
        self.rho = rho
        self.klOC = klOC.reshape(3, 1)
        self.CkJk = [None]
        self.Mk = [None]
        self.w = [None]
        self.h = [None]
        self.L = [None]
    
    def get_Mk(self, m, CkJk):
        klOC = self.klOC
        CkJk = self.CkJk

        # Rigid spatial inertia
        MC = np.block([
            [np.diag(CkJk), np.zeros((3, 3))],
            [np.zeros((3, 3)), m * np.eye(3)]
        ])
        return sb.phi(klOC) @ MC @ sb.phi(klOC).T


class Flex_Properties:
    def __init__(self, E, G, n_nd, n_md):
        self.E = E
        self.G = G
        self.n_nd = n_nd
        self.n_md = n_md
        self.n_elem = self.n_nd - 1
        self.K_fl = [None]
        self.M_fl = [None]
        self.eigval = [None]
        self.PI = [None]
        self.PI_end = [None]
    
    def set_PI(self, PI):
        self.PI = PI
    
    def set_K_fl(self, K_fl):
        self.K_fl = K_fl

    def set_M_fl(self, M_fl):
        self.M_fl = M_fl