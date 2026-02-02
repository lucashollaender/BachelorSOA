# Soa Library
import numpy as np


def skew(z):
    z = np.asarray(z).reshape(3,)
    return np.array([
        [0.0,    -z[2],  z[1]],
        [z[2],    0.0,  -z[0]],
        [-z[1],   z[0],  0.0]
    ])


def R6(R: np.ndarray):
    R = np.asarray(R)
    return np.block([
        [R,            np.zeros((3, 3))],
        [np.zeros((3, 3)), R]
    ])


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


def quat_to_rotmat(q):
    """
    Convert quaternion to rotation matrix.
    Assumes q = [qx, qy, qz, qw]^T (vector part first, scalar last).
    q: shape (4,)
    returns R: shape (3, 3)
    """

    q = np.asarray(q).reshape(4,)
    quat = pq.Quaternion(q[3], q[0], q[1], q[2])  # pq uses scalar first
    return quat.rotation_matrix
