import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as Rot
from SOALIB import soalib as sb

def nBodyPen(n, L, m, CkJk, S0, t):
    # Takes number of links, length, mass and inertia, n, L, m, Cklk 
    # for initial condition, S0 and returns animation of time t.
    
    # kl(O_k, O+_k-1): Hinge pos
    klOO = np.array([0],
                    [0],
                    [L])
    
    # kl(O_k, C_k): Centroid pos
    klOC = np.array([0],
                    [0],
                    [L/2])
    
    # M(C): Spatial inertia centroid
    MC = np.block([
    [np.diag(CkJk), np.zeros((3, 3))],
    [np.zeros((3, 3)), m*np.eye(3)]
    ])

    # M(k): Spatial inertia hinge
    Mk = phi(klOC) @ MC @ phi(klOC).T

    # H(k): Hinge map
    H = [np.eye(3), np.zeros(3)]

    # S: Spatial state vector

    EOM(S, n, klOO, klOC, Mk, H)

    return S






def EOM(S, n, klOO, klOC, Mk, H):
    # Takes state, S, number of links, n, hinge pos, klOO, centroid pos,
    # klOC, spatial in hinge inertia, Mk and hinge map, H and returns 
    # generalized acceleration, alpha

    # Define derivative of state vector
    Theta_dot = np.zeros((4*n, 1))
    Beta_dot = np.zeros((3*n, 1))

    # Unpacking of S
    Theta = S[0:4*n]
    Beta = S[4*n:7*n]

    for i in range(n):
        q = Theta(S)
        Theta_dot[i*4:i*4+4] = theta_dot()
        Beta_dot[i*3:i*3+3] =


def theta_dot(theta, beta):
    # Takes theta (4-element quaternion) and beta (3-element angular velocity)
    # and returns theta_dot.

    td = 0.5 * np.block([[-sb.skew(beta), beta], 
                         [beta.T, 0]]) @ theta
    
    return 

"""
skew(z): Returns the 3x3 skew-symmetric matrix of a 3-vector (cross-product operator).
R6(R): Constructs a 6x6 spatial rotation matrix by placing the 3x3 rotation R on the diagonal blocks.
skew6(z): Returns the 6x6 spatial cross-product matrix for motion vectors (twists).
bar6(V): Returns the 6x6 spatial cross-product matrix (dual) for force vectors (wrenches).
phi(l): Returns the 6x6 rigid body spatial translation matrix for a frame shift by vector l.
quat_derivative(q, omega): Computes the time derivative of quaternion q given angular velocity omega.
quat_to_rotmat(q): Converts a vector-first quaternion [x, y, z, w] into a 3x3 rotation matrix.
"""

