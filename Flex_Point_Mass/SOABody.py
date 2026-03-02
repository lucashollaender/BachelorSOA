import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from SOALIB import soalib as sb
from scipy.integrate import solve_ivp
from Structural_Analysis_PM_Rect import Structural_Analysis_PM_Rect
from Body_Properties import Joint, Rigid_Properties, Flex_Properties

# Increase limit to 100 MB (default is 20)
plt.rcParams['animation.embed_limit'] = 1000

class SOABody:
# SOAbody class
    class Force:
        def __init__(self, joint: Joint):
            self.tau = np.zeros((joint.beta_size(), 1))
            self.sum_phi_F_ext = np.zeros((6, 1))
        
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
        rigid.w = float(joint.klOO[0].flatten()[0])
        rigid.h = float(joint.klOO[1].flatten()[0])
        rigid.L = float(joint.klOO[2].flatten()[0])
        rigid.A = rigid.h * rigid.w
        self.m = rigid.rho * rigid.A * rigid.L
        self.rigid.CkJk = np.array([1/12 * self.m * (rigid.h**2 + rigid.L**2), 1/12 * self.m * (rigid.w**2 + rigid.L**2), 1/12 * self.m * (rigid.h**2 + rigid.w**2)])
        rigid.Mk = rigid.get_Mk(self.m, self.rigid.CkJk)

        # Structural analysis is PI == [None] (Point mass: Rectangular cross section)
        if self.flex.PI == [None]:
            body_analysis = Structural_Analysis_PM_Rect(joint, rigid, flex)
            
            # PI
            self.flex.PI = body_analysis.PI
            self.flex.PI_end = body_analysis.PI[-6:, :]
            self.flex.eigval = body_analysis.eigval

            # Stiffness and mass matrix
            self.flex.K_fl = body_analysis.K_fl
            self.flex.M_fl = body_analysis.M_fl
        
        # D_m invers (offline)
        H_M_fl = np.hstack([np.eye(self.flex.n_md, self.flex.n_md), np.zeros((self.flex.n_md, 6))])
        A_fl = sb.get_A(self.flex.PI_end, self.joint.klOO)
        self.flex.L_fl = la.inv(H_M_fl @ self.flex.M_fl @ H_M_fl.T)
        zeta = H_M_fl @ A_fl
        self.flex.U_fl = self.flex.L_fl @ zeta
        self.flex.D_fl = zeta.T @ self.flex.U_fl

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
    
    def get_D_m_inv(self, Gamma, x):
        # Calculate D_m_inv
        if x == 0:
            Dminv = self.flex.L_fl
        elif x == 1:
            Gamma_inv = la.inv(Gamma)
            Dminv = self.flex.L_fl - la.solve((Gamma_inv + self.flex.D_fl).T, self.flex.U_fl.T).T @ self.flex.U_fl.T
        return Dminv