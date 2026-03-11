import numpy as np
import matplotlib.pyplot as plt
from Merged_Flex_Project import Structural_Analysis_PM_Rect
from Merged_Flex_Project import Rigid_Properties, Flex_Properties, SOABody, Joint   # wherever defined
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.interpolate import interp1d
# -----------------------------
# Create beam
# -----------------------------
H1="fixed"
L = 1
klOC = np.array([L/2, 0, 0])
# n_md_max = (n_nd - 1) * 3
E, G, rho, n_nd, n_md = 2.1e9, 8e10, 7850, 4, 9 
w, h = 0.04, 0.04
rigid=Rigid_Properties(rho, klOC, w, h)

flex = Flex_Properties(E, G, n_nd, n_md)
j1 = Joint(L, H1)
b1 = SOABody(j1,rigid,flex)
beam = Structural_Analysis_PM_Rect(rigid, flex)

PI_t = beam.PI_t
eigval = beam.eigval

frequencies = np.sqrt(eigval) / (2*np.pi)

n_nodes = beam.n_nd
L = beam.L
n_modes = PI_t.shape[1]

x = np.linspace(0, L, n_nodes)
x_dense = np.linspace(0, L, 200)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X_all, Y_all, Z_all = [], [], []

for m in range(n_modes):

    u = PI_t[0::3, m]
    v = PI_t[1::3, m]
    w = PI_t[2::3, m]

    fu = interp1d(x, u, kind='cubic')
    fv = interp1d(x, v, kind='cubic')
    fw = interp1d(x, w, kind='cubic')

    X = x_dense + fu(x_dense)
    Y = fv(x_dense)
    Z = fw(x_dense)

    ax.plot(X, Y, Z, label=f"Mode {m+1}")

    X_all.extend(X)
    Y_all.extend(Y)
    Z_all.extend(Z)

# --- enforce equal axis scaling ---
X_all = np.array(X_all)
Y_all = np.array(Y_all)
Z_all = np.array(Z_all)

max_range = np.array([
    X_all.max()-X_all.min(),
    Y_all.max()-Y_all.min(),
    Z_all.max()-Z_all.min()
]).max() / 2

mid_x = (X_all.max()+X_all.min()) / 2
mid_y = (Y_all.max()+Y_all.min()) / 2
mid_z = (Z_all.max()+Z_all.min()) / 2

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Beam Mode Shapes")
ax.legend()

plt.show()