import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
from SOALIB import soalib as sb
 
def phi_fl(Pi, l_k_plus_1_k, l_Okplus_k):
    # l_k_plus_1_k length between the 2 hinges
    # l_Okplus_k length from bodyframe k to O_K_plus
    nm = Pi.shape[1]
    return np.block([
        [np.zeros((nm, nm)), Pi.T @ sb.phi(l_Okplus_k)],
        [np.zeros((6, nm)), sb.phi(l_k_plus_1_k)]    ])

def H_fl(Pi, l_Ok_k, H):

    # Force H to be 2D: (rv,6) instead of (6,)
    if H.ndim == 1:
        H = H.reshape(1, -1)

    rv = H.shape[0]
    nm = Pi.shape[1]

    Phi = np.asarray(sb.phi(l_Ok_k))  # should be (6,6)

    Pi_B = Phi.T @ Pi                 # (6,nm)
    H_B  = H @ Phi                    # (rv,6)

    return np.block([
        [np.eye(nm),          -Pi_B.T],
        [np.zeros((rv, nm)),  H_B   ]
    ])

H = sb.hinge_map("spherical")
l1=[1,2,3]
l2=[4,5,6]
Pi = np.ones((6,4))
print(H_fl(Pi,l1,H))