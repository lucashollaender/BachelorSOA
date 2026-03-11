import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from SOALIB import soalib as sb
from scipy.integrate import solve_ivp
from Flex_CB.Structural_Analysis_CB_Rect import Structural_Analysis_CB_Rect
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import pandas as pd

# Increase limit to 100 MB (default is 20)
plt.rcParams['animation.embed_limit'] = 1000

class SOABody:
    # SOAbody class
    class Force:
        def __init__(self, joint: Joint):
            self.tau = np.zeros((joint.beta_size(), 1))
            self.F_ext = np.zeros((6, 1))

    class InitialCondition:
        def __init__(self, joint: Joint, flex: Flex_Properties):
            # Setup of initial conditions (assumes identity rotation and no initial velocity)
            self.theta0 = np.zeros((joint.theta_size(), 1))
            if np.size(self.theta0) == 4:
                self.theta0[-1] = 1
            elif np.size(self.theta0) == 7:
                self.theta0[3] = 1
            self.beta0 = np.zeros((joint.beta_size(), 1))

            # Setup of initial conditions for eta and eta_dot
            self.eta0 = np.zeros((flex.n_md, 1))
            self.eta_dot0 = np.zeros((flex.n_md, 1))

    def __init__(self, joint: Joint, rigid: Rigid_Properties, flex: Flex_Properties):
        # Import classes
        self.joint = joint
        self.rigid = rigid
        self.flex = flex
        self.force = self.Force(self.joint)
        self.initialcondition = self.InitialCondition(joint, flex)
        rigid.A = rigid.h * rigid.w
        rigid.L = joint.L
        flex.L_elem = joint.L / flex.n_elem
        self.m = rigid.rho * rigid.A * joint.L
        self.rigid.CkJk = np.array([1/12 * self.m * (rigid.h**2 + rigid.w**2), 1/12 * self.m * (
            rigid.h**2 + joint.L**2), 1/12 * self.m * (rigid.w**2 + joint.L**2)])
        rigid.Mk = rigid.get_Mk(self.m, self.rigid.CkJk)

        # Structural analysis is PI == [None] (Point mass: Rectangular cross section)
        if self.flex.PI == [None]:
            body_analysis = Structural_Analysis_CB_Rect(rigid, flex)

            """
            print("p_0")
            print(pd.DataFrame(body_analysis.p_0))
            print("p_1")
            print(pd.DataFrame(body_analysis.p_1))
            print("CkJk_0")
            print(pd.DataFrame(body_analysis.CkJk_0))
            print("CkJk_1")
            print(pd.DataFrame(body_analysis.CkJk_1))
            print("CkJk_2")
            print(pd.DataFrame(body_analysis.CkJk_2))
            print("F_0")
            print(pd.DataFrame(body_analysis.F_0))
            print("F_1")
            print(pd.DataFrame(body_analysis.F_1))
            print("G_0")
            print(pd.DataFrame(body_analysis.G_0))
            print("E_0")
            print(pd.DataFrame(body_analysis.E_0))
            print("omega^2")
            print(pd.DataFrame(body_analysis.omega2))
            print("PI_t")
            print(pd.DataFrame(body_analysis.PI_t))
            print("K_st")
            print(pd.DataFrame(body_analysis.K_st))
            """

            # PI
            self.flex.PI = body_analysis.PI
            self.flex.PI_end = body_analysis.PI[-6:, :]
            self.flex.omega2 = body_analysis.omega2
            self.flex.omega = body_analysis.omega

            # Stiffness, damping and mass matrix
            self.flex.K_fl = body_analysis.K_fl
            self.flex.M_fl = body_analysis.M_fl
            self.flex.C_fl = body_analysis.C_fl

            # Modal integral for gyroscopic force
            self.flex.p_1 = body_analysis.p_1
            self.flex.F_1 = body_analysis.F_1
            self.flex.CkJk_1 = body_analysis.CkJk_1
            self.flex.CkJk_2 = body_analysis.CkJk_2

        # D_m invers (offline)
        H_M_fl = np.hstack([np.eye(self.flex.n_md, self.flex.n_md), np.zeros((self.flex.n_md, 6))])
        A_fl = sb.get_A(self.flex.PI_end, self.joint.klOO)
        self.flex.L_fl = la.inv(H_M_fl @ self.flex.M_fl @ H_M_fl.T)
        zeta = H_M_fl @ A_fl
        self.flex.U_fl = self.flex.L_fl @ zeta
        self.flex.D_fl = zeta.T @ self.flex.U_fl

    def set_tau(self, tau):
        self.force.tau = tau

    def set_F_ext(self, F_ext):
        self.force.F_ext = F_ext

    def set_initial_theta0(self, theta0):
        self.initialcondition.theta0 = theta0

    def set_initial_beta0(self, beta0):
        self.initialcondition.beta0 = beta0

    def get_D_m_inv(self, Gamma, x):
        # Calculate D_m_inv
        if x == 0:
            Dminv = self.flex.L_fl
        elif x == 1:
            #Gamma_inv = la.inv(Gamma)
            #Dminv = self.flex.L_fl - la.solve((Gamma_inv + self.flex.D_fl).T, self.flex.U_fl.T).T @ self.flex.U_fl.T
            Dminv = self.flex.L_fl - self.flex.U_fl @ la.solve((np.eye(6, 6) + Gamma @ self.flex.D_fl), Gamma) @ self.flex.U_fl.T
        return Dminv
