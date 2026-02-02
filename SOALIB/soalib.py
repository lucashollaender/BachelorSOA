# Soa Library
import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
from scipy.spatial.transform import Rotation as R
import pandas as pd
from matplotlib.animation import FuncAnimation
import time


def skew(z):
    z = np.asarray(z).reshape(3,)
    return np.array([
        [0.0,    -z[2],  z[1]],
        [z[2],    0.0,  -z[0]],
        [-z[1],   z[0],  0.0]
    ])


def q2R(q, n):
    # Takes a quaternion vector [x, y, z, w] and returns an n x n matrix (3 or 6).

    # Create rotation object from quaternion [x, y, z, w]
    rot_matrix = R.from_quat(q).as_matrix()

    if n == 3:
        return rot_matrix

    elif n == 6:
        # Create a 3x3 zero matrix
        z = np.zeros((3, 3))
        # Stack blocks: [R, 0]
        #               [0, R]
        return np.block([
            [rot_matrix, z],
            [z, rot_matrix]
        ])

    else:
        raise ValueError("n must be 3 or 6")


def skew6(z):
    z = np.asarray(z).reshape(6,)
    omega = z[0:3]
    v = z[3:6]
    return np.block([
        [skew(omega),           np.zeros((3, 3))],
        [skew(v),           skew(omega)]
    ])


def bar6(V):
    V = np.asarray(V).reshape(6,)
    omega = V[0:3]
    v = V[3:6]
    return np.block([
        [skew(omega),           skew(v)],
        [np.zeros((3, 3)),   skew(omega)]
    ])


def phi(l):
    l = np.asarray(l).reshape(3,)
    return np.block([
        [np.eye(3),          skew(l)],
        [np.zeros((3, 3)),   np.eye(3)]
    ])


def quat_derivative(q, omega):
    """
    Quaternion derivative:
        qdot = 0.5 * [[-skew(omega), omega],
                      [-omega^T,        0]] @ q

    Assumes q = [qx, qy, qz, qw]^T (vector part first, scalar last).
    omega: shape (3,)
    q: shape (4,)
    returns qdot: shape (4,)
    """
    q = np.asarray(q).reshape(4,)
    omega = np.asarray(omega).reshape(3,)

    Omega = np.block([
        [-skew(omega),        omega.reshape(3, 1)],
        [-omega.reshape(1, 3), np.zeros((1, 1))]
    ])

    return 0.5 * (Omega @ q)


def hinge_map(x):
    """
    Returns the hinge map (SOA) as a 3x6 matrix based on joint type.
    """
    if x == "spherical":
        H = np.hstack((
            np.zeros((3, 3)),
            np.eye(3)
        ))

    elif x == "fixed":
        H = np.zeros((6, 6))

    elif x == "free":
        H = np.eye(6)

    elif x == "revx":
        H = np.array([1, 0, 0, 0, 0, 0])

    elif x == "revy":
        H = np.array([0, 1, 0, 0, 0, 0])

    elif x == "revz":
        H = np.array([0, 0, 1, 0, 0, 0])
    else:
        raise ValueError(f"Unknown joint type: {x}")

    return H
