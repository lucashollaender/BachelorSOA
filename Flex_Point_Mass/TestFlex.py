from FlexForwardSOA import Joint, Rigid_Properties, Flex_Properties, SOABody, MultibodySystem, Simulation
import numpy as np
from SOALIB import soalib as sb
import pandas as pd
import sys
import os

h = 0.1
w = 0.3
L = 5

klOO = np.array([h, w, L])
H_type1 = "spherical"

m = 1
CkJk = np.array([1, 1, 0.1])
klOC = np.array([0, 0, 2.5])

E, G, rho, n_nd, n_md = 230e9, 80e9, 7850, 100, 6

j1 = Joint(klOO, H_type1)
r = Rigid_Properties(rho, CkJk, klOC)
f = Flex_Properties(E, G, n_nd, n_md)
b1 = SOABody(j1, r, f)

bodies = [b1, b1]

system = MultibodySystem(bodies)

print("PI")
print(pd.DataFrame(b1.flex.PI))
print("eigval")
print(pd.DataFrame(b1.flex.eigval))
print("K_fl")
print(pd.DataFrame(b1.flex.K_fl))
print("M_fl")
print(pd.DataFrame(b1.flex.M_fl))