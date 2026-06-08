# Soa Library
import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pq
import scipy as sp
from scipy.spatial.transform import Rotation as R
import pandas as pd
from matplotlib.animation import FuncAnimation
import time
from scipy.optimize import root


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
    l = np.asarray(l).flatten()
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


def get_A(PI_end, klOO):
    return np.vstack([PI_end.T, phi(klOO)])


def get_R_tot(R6, n_md):
    rw1 = np.hstack([np.eye(n_md, n_md), np.zeros((n_md, 6))])
    rw2 = np.hstack([np.zeros((6, n_md)), R6])
    return np.vstack([rw1, rw2])


def get_R6(R3):
    z3 = np.zeros((3, 3))
    rw1 = np.hstack([R3, z3])
    rw2 = np.hstack([z3, R3])
    return np.vstack([rw1, rw2])


def integrate_RK4(system, t0, tf, dt):
    """
    Choose dt such that (tf-t0)/dt is an integer
    """
    nt = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, nt)

    y0 = system.S0

    n = len(y0)

    Y = np.zeros((n, nt))
    Y[:, 0] = y0

    for i in range(nt - 1):

        y = Y[:, i]
        ti = t[i]

        k1 = system.EOM(ti, y)
        k2 = system.EOM(ti + dt/2, y + dt/2 * k1)
        k3 = system.EOM(ti + dt/2, y + dt/2 * k2)
        k4 = system.EOM(ti + dt, y + dt * k3)

        Y[:, i+1] = y + dt * (k1/6 + (k2 + k3)/3 + k4/6)

        if not np.all(np.isfinite(Y[:, i+1])):
            raise FloatingPointError(
                f"RK4 produced invalid state at step {i+1}, t={t[i+1]}"
            )

    return Y, t


def integrate_backward_euler(system, t0, tf, dt, tol=1e-8, max_iter=20):
    nt = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, nt)

    y0 = system.S0.copy()
    n = len(y0)

    Y = np.zeros((n, nt))
    Y[:, 0] = y0

    for i in range(nt - 1):
        y_old = Y[:, i]
        t_next = t[i + 1]

        # Explicit Euler predictor
        y_guess = y_old + dt * system.EOM(t[i], y_old)

        def residual(y_next):
            return y_next - y_old - dt * system.EOM(t_next, y_next)

        sol = root(
            residual,
            y_guess,
            method="hybr",
            options={
                "xtol": tol,
                "maxfev": max_iter * (n + 1),
            }
        )

        if not sol.success:
            raise RuntimeError(
                f"Backward Euler failed at step {i+1}, "
                f"t={t_next:.6f}: {sol.message}"
            )

        Y[:, i + 1] = sol.x

        if not np.all(np.isfinite(Y[:, i + 1])):
            raise FloatingPointError(
                f"Backward Euler produced invalid state at step {i+1}, "
                f"t={t_next:.6f}"
            )

    return Y, t
