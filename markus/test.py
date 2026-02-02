from SOALIB import soalib as sb
import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
import pandas as pd
from matplotlib.animation import FuncAnimation
import time
A = np.array([1, 2, 3])
print(A)

print(sb.skew(A))

quat = np.array([0, 0, 0, 1]).T

print(sb.quat_to_rotmat(quat))

V = np.array([1, 2, 3, 4, 5, 6]).T

print(sb.bar6(V))
