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
        self.klOO = klOO
        self.L = np.linalg.norm(klOO)
        self.klOC = klOO / 2

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
    
    def get_theta2X(self, theta):
        klOO = self.klOO

        if self.type == "revx":
            ang = theta.item()
            q = np.array([[np.sin(ang/2)], [0], [0], [np.cos(ang/2)]])
            return np.vstack((q, klOO)), q

        elif self.type == "revy":
            ang = theta.item()
            q = np.array([[0], [np.sin(ang/2)], [0], [np.cos(ang/2)]])
            return np.vstack((q, klOO)), q

        elif self.type == "revz":
            ang = theta.item()
            q = np.array([[0], [0], [np.sin(ang/2)], [np.cos(ang/2)]])
            return np.vstack((q, klOO)), q

        elif self.type == "spherical":
            q = theta.reshape(4, 1)
            return np.vstack((q, klOO)), q

        elif self.type == "fixed":
            q = np.array([[0], [0], [0], [1]])
            return np.vstack((q, klOO)), q

    def get_theta_dot(self, Theta, Beta):
        if self.type.startswith("rev"):
            return Beta.reshape(1, 1)
        elif self.type == "spherical":
            return sb.quat_derivative(Theta, Beta).reshape(4, 1)
        elif self.type == "fixed":
            return np.zeros((0, 1))

class Rigid_Properties:
    # Inertia class with m, CkJk and klOC
    def __init__(self, rho, w, h):
        # Parameters
        self.rho = rho
        self.CkJk = [None]
        self.Mk = [None]
        self.w = w
        self.h = h
        self.L = [None]

    def get_Mk(self, m, CkJk, klOC):
        CkJk = self.CkJk

        # Rigid spatial inertia
        MC = np.block([
            [np.diag(CkJk), np.zeros((3, 3))],
            [np.zeros((3, 3)), m * np.eye(3)]
        ])
        return sb.phi(klOC) @ MC @ sb.phi(klOC).T


class Flex_Properties:
    def __init__(self, E, G, c, n_nd, n_md, mode_selection=None):
        self.E = E
        self.G = G
        self.c = c
        self.n_nd = n_nd
        self.n_md = n_md
        self.n_elem = self.n_nd - 1

        # Mode selection
        self.mode_selection = mode_selection
        self.modes = []