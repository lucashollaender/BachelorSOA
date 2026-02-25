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
            np.eye(3),
            np.zeros((3, 3))
        ))

    elif x == "fixed":
        H = np.zeros((0, 6))

    elif x == "free":
        H = np.eye(6)

    elif x == "revx":
        H = np.array([1, 0, 0, 0, 0, 0]).reshape(1, 6)

    elif x == "revy":
        H = np.array([0, 1, 0, 0, 0, 0]).reshape(1, 6)

    elif x == "revz":
        H = np.array([0, 0, 1, 0, 0, 0]).reshape(1, 6)
    else:
        raise ValueError(f"Unknown joint type: {x}")

    return H

def get_quat_from_degrees(x, y, z):
    # Takes angles, x, y and z and returns quaternion

    r = R.from_euler('xyz', [x, y, z], degrees=True)
    q = np.array(r.as_quat()).reshape(4, 1)

    return q

def get_stiff_mat_rect_3D(h, w, L, E, G):
    
    # h > w
    a = w
    b = h

    # Book -> code
    # x -> z
    # y -> x
    # z -> y

    # Rectangular cross-section
    k_x = 1.2
    k_y = k_x
    K = a * b**3 * (16/3 - 3.36 * a / b * (1 - a**4 / (12 * b**4)))
    A = h*w
    I_x = w * h**3 / 12
    I_y = h * w**3 / 12

    # Factors
    phi_x = 12 * E * I_y * k_x / (A * G * L**2)
    phi_y = 12 * E * I_x * k_y / (A * G * L**2)
    S = G * K / L

    # Z    
    Z = A * E / L

    # X
    X_1 = 12 * E * I_y / ((1 + phi_x) * L**3)
    X_2 = 6 * E * I_y / ((1 + phi_x) * L**2)
    X_3 = (4 + phi_x) * E * I_y / ((1 + phi_x) * L)
    X_4 = (2 - phi_x) * E * I_y / ((1 + phi_x) * L)

    # Y
    Y_1 = 12 * E * I_x / ((1 + phi_y) * L**3)
    Y_2 = 6 * E * I_x / ((1 + phi_y) * L**2)
    Y_3 = (4 + phi_y) * E * I_x / ((1 + phi_y) * L)
    Y_4 = (2 - phi_y) * E * I_x / ((1 + phi_y) * L)

    # Stiffness matrix
    diag = [None] * 6

    diag[0] = np.array([Z, X_1, Y_1, S, Y_3, X_3, Z, X_1, Y_1, S, Y_3, X_3])
    diag[1] = np.array([0, 0, -Y_2, 0, 0, -X_2, 0, 0, Y_2])
    diag[2] = np.array([0, X_2, 0, 0, Y_2, 0, 0, -X_2])
    diag[3] = np.array([-Z, -X_1, -Y_1, -S, Y_4, X_4])
    diag[4] = np.array([0, 0, -Y_2, 0])
    diag[5] = np.array([0, X_2])

    k = np.diag(diag[0], k=0)

    for i in range(1, 6):
        k = k + np.diag(diag[i], k=-2*i) + np.diag(diag[i], k=2*i)
    
    # Change so rotations first is along
    perm = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    
    k_perm = k[np.ix_(perm, perm)]

    return k_perm

