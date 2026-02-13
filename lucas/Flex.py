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

def H_fl(Pi,l_Ok_k,H):
    rv = H.shape[0]
    nm = Pi.shape[1]
    Pi_B=sb.phi(l_Ok_k).T @ Pi
    H_B=H @ sb.phi(l_Ok_k)
    return np.block([
        [np.eye(nm), Pi_B.T],
        [np.zeros((rv,nm)),  H_B]
    ])


H = sb.hinge_map("fixed")
l1=[1,2,3]
l2=[4,5,6]
Pi = np.ones((6,4))
print(H_fl(Pi,l1,H))