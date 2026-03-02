from Flex_Point_Mass.SOABody import SOABody
from Flex_Point_Mass.MultibodySystem import MultibodySystem
from Flex_Point_Mass.Simulation import Simulation
from Flex_Point_Mass.Body_Properties import Joint, Rigid_Properties, Flex_Properties
from SOALIB import soalib as sb
import numpy as np
import pandas as pd

h = 0.1
w = 0.3
L = 5

klOO = np.array([h, w, L])
H_type1 = "spherical"

m = 1
CkJk = np.array([1, 1, 0.1])
klOC = np.array([0, 0, 2.5])

E, G, rho, n_nd, n_md = 230e9, 80e9, 7850, 4, 7

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
