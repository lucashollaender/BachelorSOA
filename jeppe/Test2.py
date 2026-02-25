from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation
import numpy as np
from SOALIB import soalib as sb

h = 4
w = 10
L = 2
E = 230 * 1e9
G = 120 * 1e9


k = sb.get_stiff_mat_rect_3D(h, w, L, E, G)

print(k)

print(np.size(k))