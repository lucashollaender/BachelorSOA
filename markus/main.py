import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R, Slerp
import n_body_pend as nbp
import time

# Defining data
n = 2
link_vec = np.array([0.0, 0.0, -5.0]),
com_vec = np.array([0.0, 0.0, -2.5]),
J_diag = np.array([1, 1, 0.1]),
mass = 1.0

sys = nbp.build_system_data(n, link_vec, com_vec, J_diag, mass)

S = nbp.initial_condition(n)
print(S)
