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
            body_analysis = Structural_Analysis_PM_Rect(joint.klOO, rigid.rho, flex.E, flex.G, flex.n_nd, flex.n_md)
            
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

""" ------ File Setup ------ """
# Remember to run this:
# from <SOALIB.>RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation

""" ------ Body Setup ------ """
# *** Body Parameters ****
# klOO:     Hinge position (row vector)
# H_type:   Hinge type (string)
# m:        Mass (scalar)
# CkJk:     Inertia (row vector)
# klOC:     COM position (row vector)

# *** Create Body ***
# joint = Joint(<klOO>, <H_type>)
# inertia = Inertia(<m>, <CkJk>, <klOC>)
# body = SOABody(<joint>, <inertia>)

""" ------ Body Attributes ------ """ 
# If not specified program assumes zero column vectors
# theta0, beta0, tau, F_ext ---> column vectors

# *** Initial condition ***
# body.set_initial_theta0(<theta0>)   //   <theta0> ---> column vector
#       --->   "revx/y/z" use: theta0 = np.deg2rad(theta_x/y/z)
#       --->   "spherical" use: theta0 = q0 = sb.get_quat_from_degrees(theta_x, theta_y, theta_z)
#       --->   "free" use: theta0 = np.vstack([q0, l]), where l is the initial linear displacement (l = [l_x, l_y, l_z])
#       --->   "fixed" use: theta0 cannot be specified
# body.set_initial_beta0(<beta0>)   //   <beta0> ---> column vector 
#       --->   "revx/y/z" use: beta0 = omega_x/y/z
#       --->   "spherical" use: beta0 = np.array([omega_x, omega_y, omega_z]).reshape(3, 1)
#       --->   "free" use: beta0 = np.array([omega_x, omega_y, omega_z, v_x, v_y, v_z]), where v is the initial linear velocity (v = [v_x, v_y, v_z])
#       --->   "fixed" use: beta0 cannot be specified

# *** Forces ***
# body.set_tau(<tau>)   //   <tau> ---> column vector, np.array([<tau>]).reshape(nDOF, 1)

# body.set_F_ext(<F_ext>, <klBO>)   //   <F_ext>, <klBO> ---> lists of same length 
#       --->   F_ext is a list of column vectors (6, 1) with external forces
#       --->   klBO is a list of row vectors (1, 3) with the external forces' appliying position

""" ------ System Setup and Simulation ------ """ 
# *** Multibody System ***
# system = MultibodySystem(bodies)
#       --->   bodies = [body_1, body_2, ..., body_n], list of bodies created above (tip: b_1 and base: b_n)

# *** Simulation Setup ***
# sim = Simulation(system, tf, dt)
#       --->   system as created above
#       --->   tf, length of simulation
#       --->   dt, time step size

""" ------ Camera Settings ------ """ 
# sim.camera_speed(x)
#       --->   x, number from -1..1 defining the speed in both directions (zero if not changed)

# sim.camera_hor(x)
#       --->   x, number from 0..360 defining the camera rotation around z-axis (zero if not changed)

# sim.camera_ver(x)
#       --->   x, number from -90..90 defining the camera rotation around x-axis (20 if not changed)

""" ------ Parameter Call ------ """
# *** Parameter Call ***   //   Get parameter for each body for each time step
# sim.get_<parameter>()
#       --->   <parameter> = [state, X, pos, V, alpha, a, b]

""" ------ Animate or Render Animation ------ """
# *** Animate ***
# sim.animate()

# *** Render Animation to HTML-file ***   // <file_name>, r<file_path> ---> strings
# sim.animate(<file_name>, <file_path>)
#       --->   <file_name>, name of the render file
#       --->   <file_path>, copy the file path of the folder you want to save the HTML-file in.
#               nb! You must add the letter <r> in front of file path string as:
#               file_path = r"C:\Users\jepp6\OneDrive..."
#               Choose another folder than the GIT-Hub synchronize folder, since the file will
#               be to big and result in a "commit" error.

# To DOOOOOOOOOOOOO:
# Shouldnt the zeros come first in K_fl? In Structural_Analysis_PM_Rect 