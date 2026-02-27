from SOALIB.FlexForwardSOA import Joint, Inertia, Flex, SOABody, MultibodySystem, Simulation
import numpy as np
from SOALIB import soalib as sb
import pandas as pd

klOO = np.array([0, 0, 5])
H_type1 = "spherical"

m = 1
CkJk = np.array([1, 1, 0.1])
klOC = np.array([0, 0, 2.5])

E, G, rho, n_nd, n_md = 230e9, 80e9, 7850, 4, 7

h = 0.1
w = 0.3

j1 = Joint(klOO, H_type1)
i = Inertia(m, CkJk, klOC)
f = Flex(E, G, rho, n_nd, n_md)
b1 = SOABody(j1, i, f, h, w)

print(pd.DataFrame(b1.flex.PI))
print(pd.DataFrame(b1.flex.eigval))

print(pd.DataFrame(b1.flex.M))