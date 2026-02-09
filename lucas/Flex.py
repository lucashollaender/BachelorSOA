import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
from SOALIB import soalib as sb
 
def phi_fl(l_k_plus_1_k, l_Okplus_k,nm):
    # l_k_plus_1_k length between the 2 hinges
    # l_Okplus_k length from bodyframe k to O_K_plus
    Pi = np.ones((6,nm))
    return np.block([
        [np.zeros((nm, nm)), Pi.T @ sb.phi(l_Okplus_k)],
        [np.zeros((6, nm)), sb.phi(l_k_plus_1_k)]    ])

def H_fl(l_Ok_k,nm):
    Pi = np.ones((6,nm))
    H = sb.hinge_map("fixed")

l1=[1,2,3]
l2=[4,5,6]
print(phi_fl(l1,l2,4))