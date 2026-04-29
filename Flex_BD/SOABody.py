import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from SOALIB import soalib as sb
from scipy.integrate import solve_ivp
from Structural_Analysis_BD_Rect import Structural_Analysis_BD_Rect
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import pandas as pd

# Increase limit to 100 MB (default is 20)
plt.rcParams['animation.embed_limit'] = 1000

class SOABody:
    # SOAbody class
    class Force:
        def __init__(self, joint: Joint):
            self.tau = np.zeros((joint.beta_size(), 1))
            self.F_ext = []
            self.k_TS = 0
            self.c_TS = 0
            self.theta0_TS = 0

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
        rigid.A = rigid.h * rigid.w
        rigid.L = joint.L
        flex.L_elem = joint.L / flex.n_elem
        flex.klOO_nd = [j * (joint.klOO / flex.n_elem) for j in range(flex.n_nd)]
        self.m = rigid.rho * rigid.A * joint.L
        self.rigid.CkJk = np.array([1/12 * self.m * (rigid.h**2 + rigid.w**2), 1/12 * self.m * (
            rigid.h**2 + joint.L**2), 1/12 * self.m * (rigid.w**2 + joint.L**2)])
        rigid.Mk = rigid.get_Mk(self.m, self.rigid.CkJk, self.joint.klOC)

        # Structural analysis is PI == [None] (Point mass: Rectangular cross section)
        if self.flex.PI == [None]:
            body_analysis = Structural_Analysis_BD_Rect(joint, rigid, flex)

            # PI
            self.flex.PI = body_analysis.PI
            self.flex.PI_e = body_analysis.PI_e
            self.flex.PI_end = body_analysis.PI[-6:, :]
            self.flex.omega2 = body_analysis.omega2
            self.flex.omega = body_analysis.omega

            # Mode selection
            self.flex.modes = body_analysis.modes
            self.flex.n_md = body_analysis.n_md

            # Stiffness, damping and mass matrix
            self.flex.K_fl = body_analysis.K_fl
            self.flex.M_fl = body_analysis.M_fl
            self.flex.C_fl = body_analysis.C_fl

            # Modal integral for gyroscopic force
            self.flex.p_0 = body_analysis.p_0
            self.flex.S_1 = body_analysis.S_1
            self.flex.F_1 = body_analysis.F_1
            self.flex.J_0 = body_analysis.J_0
            self.flex.J_1 = body_analysis.J_1

        self.initialcondition = self.InitialCondition(joint, self.flex)

        # D_m inverse (offline)
        if self.flex.n_md == 0:
            # Rigid Body
            self.flex.L_fl = np.zeros((0, 0))
            self.flex.U_fl = np.zeros((0, 6))
            self.flex.D_fl = np.zeros((6, 6))
        else:
            # Standard flexible body formulation
            H_M_fl = np.hstack([np.eye(self.flex.n_md), np.zeros((self.flex.n_md, 6))])
            A_fl = sb.get_A(self.flex.PI_end, self.joint.klOO)
            self.flex.L_fl = la.inv(H_M_fl @ self.flex.M_fl @ H_M_fl.T)
            zeta = H_M_fl @ A_fl
            self.flex.U_fl = self.flex.L_fl @ zeta
            self.flex.D_fl = zeta.T @ self.flex.U_fl

    def set_tau(self, tau):
        self.force.tau = tau
    
    def set_F_ext(self, node=-1, F_ext=None, F_fun=None):
        """
        Add external spatial force/moment at a selected node.

        How to call:
        b1.set_F_ext(F_ext)                    # constant tip force
        b1.set_F_ext(node=-1, F_ext=F_ext)     # constant tip force
        b1.set_F_ext(node=3, F_ext=F_ext)      # constant force at node 3
        b1.set_F_ext(node=3, F_fun=my_fun)     # time-dependent force
        """

        if F_ext is None and F_fun is None:
            arr = np.asarray(node)
            if arr.size == 6:
                F_ext = arr.reshape(6, 1)
                node = -1
            else:
                raise ValueError("Provide either F_ext or F_fun.")

        # Constant F_ext
        if F_fun is None:
            F_const = np.asarray(F_ext, dtype=float).reshape(6, 1)

            def constant_force_fun(t, state, body):
                return F_const

            force_function = constant_force_fun
        # Function F_ext in terms of time, t
        else:
            force_function = F_fun

        self.force.F_ext.append({
        "node": int(node),
        "fun": force_function
        })

    def set_TS(self, k_TS, c_TS, theta0_TS):
        self.force.k_TS = k_TS
        self.force.c_TS = c_TS
        self.force.theta0_TS = theta0_TS

    def set_initial_beta0(self, beta0):
        self.initialcondition.beta0 = beta0

    def set_initial_eta0(self, eta0):
        self.initialcondition.eta0 = eta0
    
    def set_initial_eta_dot0(self, eta_dot0):
        self.initialcondition.eta_dot0 = eta_dot0

    def get_D_m_inv(self, Gamma, type):
        # Calculate D_m_inv
        if self.flex.n_md == 0:
            return np.zeros((0, 0))
            
        if type == "tip":
            Dminv = self.flex.L_fl
        elif type == "not_tip":
            Dminv = self.flex.L_fl - self.flex.U_fl @ la.solve((np.eye(6, 6) + Gamma @ self.flex.D_fl), Gamma) @ self.flex.U_fl.T
        return Dminv