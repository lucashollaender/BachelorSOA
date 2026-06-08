import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import os
import pandas as pd

from SOALIB import soalib as sb


# ============================================================
# SystemState
# ============================================================

class SystemState:

    def __init__(self, theta, beta, eta, eta_dot):
        self.Theta = theta
        self.Beta = beta
        self.Eta = eta
        self.Eta_dot = eta_dot

    def pack(self):
        return np.vstack([*self.Theta, *self.Beta, *self.Eta, *self.Eta_dot]).flatten()

    @staticmethod
    def unpack(S, joints, flexs):

        S = S.flatten()

        Theta = []
        Beta = []
        Eta = []
        Eta_dot = []

        idx = 0

        for k in joints:
            sz = k.theta_size()
            Theta.append(S[idx:idx+sz].reshape(sz, 1))
            idx += sz

        for k in joints:
            sz = k.beta_size()
            Beta.append(S[idx:idx+sz].reshape(sz, 1))
            idx += sz

        for k in flexs:
            sz = k.n_md
            Eta.append(S[idx:idx+sz].reshape(sz, 1))
            idx += sz

        for k in flexs:
            sz = k.n_md
            Eta_dot.append(S[idx:idx+sz].reshape(sz, 1))
            idx += sz

        return SystemState(Theta, Beta, Eta, Eta_dot)


# ============================================================
# Body properties
# ============================================================

class Joint:

    def __init__(self, L, H_type):

        self.type = H_type
        self.H = hinge_map(H_type)

        self.L = L
        self.klOO = np.array([L, 0, 0]).reshape(3, 1)

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

    def __init__(self, rho, klOC, w, h):

        self.rho = rho
        self.klOC = klOC.reshape(3, 1)

        self.w = w
        self.h = h

        self.CkJk = None
        self.Mk = None
        self.L = None

    def get_Mk(self, m, CkJk):

        klOC = self.klOC

        MC = np.block([
            [np.diag(CkJk), np.zeros((3, 3))],
            [np.zeros((3, 3)), m*np.eye(3)]
        ])

        return phi(klOC) @ MC @ phi(klOC).T


class Flex_Properties:

    def __init__(self, E, G, c, n_nd, n_md):

        self.E = E
        self.G = G
        self.c = c

        self.n_nd = n_nd
        self.n_md = n_md
        self.n_elem = n_nd-1

        self.K_fl = None
        self.M_fl = None
        self.C_fl = None

        self.PI = None
        self.PI_end = None

        self.omega = None
        self.omega2 = None


# ============================================================
# SOABody
# ============================================================

class SOABody:

    class Force:

        def __init__(self, joint):

            self.tau = np.zeros((joint.beta_size(), 1))
            self.F_ext = np.zeros((6, 1))

    class InitialCondition:

        def __init__(self, joint, flex):

            self.theta0 = np.zeros((joint.theta_size(), 1))

            if np.size(self.theta0) == 4:
                self.theta0[-1] = 1

            elif np.size(self.theta0) == 7:
                self.theta0[3] = 1

            self.beta0 = np.zeros((joint.beta_size(), 1))

            self.eta0 = np.zeros((flex.n_md, 1))
            self.eta_dot0 = np.zeros((flex.n_md, 1))

    def __init__(self, joint, rigid, flex):

        self.joint = joint
        self.rigid = rigid
        self.flex = flex

        self.force = self.Force(joint)
        self.initialcondition = self.InitialCondition(joint, flex)

        rigid.A = rigid.h*rigid.w
        rigid.L = joint.L

        flex.L_elem = joint.L/flex.n_elem

        self.m = rigid.rho*rigid.A*joint.L

        self.rigid.CkJk = np.array([
            1/12*self.m*(rigid.h**2+rigid.w**2),
            1/12*self.m*(rigid.h**2+joint.L**2),
            1/12*self.m*(rigid.w**2+joint.L**2)
        ])

        rigid.Mk = rigid.get_Mk(self.m, self.rigid.CkJk)

        if self.flex.PI is None:

            body_analysis = Structural_Analysis_PM_Rect(rigid, flex)

            self.flex.PI = body_analysis.PI
            self.flex.PI_end = body_analysis.PI[-6:, :]

            self.flex.K_fl = body_analysis.K_fl
            self.flex.M_fl = body_analysis.M_fl
            self.flex.C_fl = body_analysis.C_fl

            self.flex.omega2 = body_analysis.omega2
            self.flex.omega = body_analysis.omega

        H_M_fl = np.hstack(
            [np.eye(self.flex.n_md), np.zeros((self.flex.n_md, 6))])

        A_fl = get_A(self.flex.PI_end, self.joint.klOO)

        self.flex.L_fl = la.inv(H_M_fl @ self.flex.M_fl @ H_M_fl.T)

        zeta = H_M_fl @ A_fl
        self.flex.U_fl = self.flex.L_fl @ zeta
        self.flex.D_fl = zeta.T @ self.flex.U_fl

    def get_D_m_inv(self, Gamma, x):

        if x == 0:
            return self.flex.L_fl

        Gamma_inv = la.inv(Gamma)

        return self.flex.L_fl - \
            la.solve((Gamma_inv+self.flex.D_fl).T,
                     self.flex.U_fl.T).T @ self.flex.U_fl.T


# ============================================================
# Structural analysis (unchanged)
# ============================================================

class Structural_Analysis_PM_Rect:

    # (your structural code goes here unchanged)
    # I omitted it here because it is ~400 lines
    # You can paste it exactly as you already have it

    pass


# ============================================================
# ATBI_Flex
# ============================================================

class ATBI_Flex:

    def __init__(self, bodies):

        self.bodies = bodies
        self.n = len(bodies)

    def coriolis(self, V, beta, H):

        deltaV = H.T @ beta

        return skew6(V) @ deltaV - bar6(deltaV) @ deltaV

    def gyroscopic(self, V, M):

        return bar6(V) @ M @ V

    def theta2X(self, theta, joint_type, klOO):

        if joint_type == "revz":

            ang = theta.item()

            q = np.array([
                [0],
                [0],
                [np.sin(ang/2)],
                [np.cos(ang/2)]
            ])

            return np.vstack((q, klOO)), q

        elif joint_type == "fixed":

            q = np.array([[0], [0], [0], [1]])

            return np.vstack((q, klOO)), q

        else:
            raise ValueError("Joint type not implemented here")


# ============================================================
# MultibodySystem
# ============================================================

class MultibodySystem:

    def __init__(self, bodies):

        self.bodies = bodies
        self.ATBI = ATBI_Flex(bodies)

        Theta0 = [b.initialcondition.theta0 for b in bodies]
        Beta0 = [b.initialcondition.beta0 for b in bodies]
        Eta0 = [b.initialcondition.eta0 for b in bodies]
        Eta_dot0 = [b.initialcondition.eta_dot0 for b in bodies]

        self.S0 = SystemState(
            Theta0,
            Beta0,
            Eta0,
            Eta_dot0
        ).pack()

    def EOM(self, t, S):

        state = SystemState.unpack(
            S.reshape(-1, 1),
            [b.joint for b in self.bodies],
            [b.flex for b in self.bodies]
        )

        # rest of your EOM function unchanged


# ============================================================
# Simulation
# ============================================================

class Simulation:

    class Data:

        def __init__(self):

            self.time = []
            self.state = []
            self.X_list = []

    def __init__(self, system, tf, dt):

        self.system = system
        self.data = self.Data()

        self.tf = tf
        self.dt = dt

    def IntegrateSystem(self):

        t_eval = np.linspace(0, self.tf, int(self.tf/self.dt)+1)

        sol = solve_ivp(
            fun=self.system.EOM,
            t_span=(0, self.tf),
            y0=self.system.S0,
            t_eval=t_eval,
            method="BDF"
        )

        print(sol.message)

        self.data.time = sol.t


# ============================================================
# Test example
# ============================================================

if __name__ == "__main__":

    L = 5
    H_type = "revz"

    rho = 7850
    E = 230e9
    G = 80e9

    n_nd = 6
    n_md = 5

    w = 0.1
    h = 0.1

    j = Joint(L, H_type)

    r = Rigid_Properties(rho, np.array([2.5, 0, 0]), w, h)

    f = Flex_Properties(E, G, 0.2, n_nd, n_md)

    b = SOABody(j, r, f)

    bodies = [b]

    system = MultibodySystem(bodies)

    sim = Simulation(system, 1, 0.01)

    sim.IntegrateSystem()
