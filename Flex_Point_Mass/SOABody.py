import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from SOALIB import soalib as sb
from scipy.integrate import solve_ivp
from .Structural_Analysis_PM_Rect import Structural_Analysis_PM_Rect
from .Body_Properties import Joint, Rigid_Properties, Flex_Properties

# Increase limit to 100 MB (default is 20)
plt.rcParams['animation.embed_limit'] = 1000


class SOABody:
    # SOAbody class
    class Force:
        def __init__(self, joint: Joint):
            self.tau = np.zeros((joint.beta_size(), 1))
            self.sum_phi_F_ext = np.zeros((6, 1))

    class InitialCondition:
        def __init__(self, joint: Joint):
            # Setup of initial conditions (assumes identity rotation and no initial velocity)
            self.theta0 = np.zeros((joint.theta_size(), 1))
            if np.size(self.theta0) == 4:
                self.theta0[-1] = 1
            elif np.size(self.theta0) == 7:
                self.theta0[3] = 1
            self.beta0 = np.zeros((joint.beta_size(), 1))

            # Setup of initial conditions for eta and eta_dot
            self.eta0 = np.zeros((6, 1))
            self.eta_dot0 = np.zeros((6, 1))

    def __init__(self, joint: Joint, rigid: Rigid_Properties, flex: Flex_Properties):
        # Import classes
        self.joint = joint
        self.rigid = rigid
        self.flex = flex
        self.force = self.Force(self.joint)
        self.initialcondition = self.InitialCondition(self.joint)
        rigid.w = float(joint.klOO[0].flatten()[0])
        rigid.h = float(joint.klOO[1].flatten()[0])
        rigid.L = float(joint.klOO[2].flatten()[0])
        rigid.A = rigid.h * rigid.w
        self.m = rigid.rho * rigid.A * rigid.L
        rigid.Mk = rigid.get_Mk(self.m)

        # Structural analysis is PI == [None] (Point mass: Rectangular cross section)
        if self.flex.PI == [None]:
            body_analysis = Structural_Analysis_PM_Rect(joint, rigid, flex)

            # PI
            self.flex.PI = body_analysis.PI
            self.flex.PI_end = body_analysis.PI[-6, :]
            self.flex.eigval = body_analysis.eigval

            # Stiffness and mass matrix
            self.flex.K_fl = body_analysis.K_fl
            self.flex.M_fl = body_analysis.M_fl

    def set_tau(self, tau):
        self.force.tau = tau

    def set_F_ext(self, F_ext, klBO):
        F = np.zeros((6, 1))
        for i in range(len(F_ext)):
            F = F + sb.phi(klBO[i]) @ F_ext[i]
        self.force.sum_phi_F_ext = F

    def set_initial_theta0(self, theta0):
        self.initialcondition.theta0 = theta0

    def set_initial_beta0(self, beta0):
        self.initialcondition.beta0 = beta0

    def get_D_m_inv(self, Gamma):
        # Parameters
        n_md = self.flex.n_md

        H_M_fl = np.hstack([np.eye(n_md, n_md), np.zeros((n_md, 6))])
        M_fl = self.flex.M
        PI = self.flex.PI
        A_fl = sb.get_A(PI, self.joint.klOO)

        L_fl = la.inv(H_M_fl @ M_fl @ H_M_fl)
        zeta = H_M_fl @ A_fl
        U_fl = L_fl @ zeta
        D_fl = zeta.T @ U_fl
        Gamma_inv = la.inv(Gamma)

        return L_fl - la.solve((Gamma_inv + D_fl).T, U_fl.T).T @ U_fl.T
