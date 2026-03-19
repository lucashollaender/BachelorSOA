import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
from Structural_Analysis_CB_Rect import Structural_Analysis_CB_Rect
import pandas as pd

# Settings
L = 1.0

E = 1e7
G = 3.8e6
c = 0.02
rho = 1000

n_nd = 8
n_md = 7

w = 0.04
h = 0.04   # non-square section helps separate bending pairs


# Properties
j1 = Joint(L, "fixed")
r1 = Rigid_Properties(rho, w, h)
f1 = Flex_Properties(E, G, c, n_nd, n_md)

# Structural_Analysis_CB_Rect expects these to exist on rigid
r1.A = r1.w * r1.h
r1.L = L

m = rho * r1.A * L
r1.CkJk = np.array([
    (m / 12.0) * (h**2 + w**2),
    (m / 12.0) * (h**2 + L**2),
    (m / 12.0) * (w**2 + L**2),
])

# Get numerical mode shapes
analysis = Structural_Analysis_CB_Rect(j1, r1, f1)
PI_e = analysis.PI_e
# e

# Computing natural frequencies
K = analysis.K_st
M = analysis.M_st

boundary_nodes = [0]
B = []
for i in boundary_nodes:
    B.extend(range(6 * i, 6 * i + 6))

all_dofs = list(range(6 * analysis.n_nd))
I = [k for k in all_dofs if k not in B]

K_II = K[np.ix_(I, I)]
M_II = M[np.ix_(I, I)]

eig_e, _ = la.eigh(K_II, M_II, subset_by_index=(0, n_md - 1))
omega_e = np.sqrt(eig_e)

print("Fixed-interface natural frequencies from PI_e [Hz]:")
print(np.round(omega_e / (2 * np.pi), 6))


# Node coordinates
x_nodes = np.linspace(0.0, L, n_nd)
xi = x_nodes / L


# Analytical mode shapes

# Euler-Bernoulli cantilever bending roots
betaL_bending = np.array([
    1.87510407,
    4.69409113,
    7.85475744,
    10.99554073,
    14.13716839,
    17.27875953,
    20.42035225,
    23.56194490
])


def normalize_mode(y):
    y = np.asarray(y, dtype=float).copy()
    ymax = np.max(np.abs(y))
    if ymax < 1e-14:
        return y
    y /= ymax
    if y[-1] < 0:
        y = -y
    return y


def phi_bending_cantilever(xi, mode_index):
    lam = betaL_bending[mode_index]
    sigma = (np.cosh(lam) + np.cos(lam)) / (np.sinh(lam) + np.sin(lam))
    y = (
        np.cosh(lam * xi)
        - np.cos(lam * xi)
        - sigma * (np.sinh(lam * xi) - np.sin(lam * xi))
    )
    return normalize_mode(y)


def phi_axial_cantilever(x, L, mode_index):
    n = mode_index + 1
    y = np.sin((2 * n - 1) * np.pi * x / (2 * L))
    return normalize_mode(y)


def phi_torsion_cantilever(x, L, mode_index):
    n = mode_index + 1
    y = np.sin((2 * n - 1) * np.pi * x / (2 * L))
    return normalize_mode(y)


def mac(a, b):
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    denom = (a.T @ a) * (b.T @ b)
    if np.abs(denom).item() < 1e-14:
        return 0.0
    return (((a.T @ b) ** 2) / denom).item()


# -------------------------------------------------
# Extract nodal components from PI_e
# node j ordering:
# [theta_x, theta_y, theta_z, u_x, u_y, u_z]
# -------------------------------------------------
def get_mode_components(PI, mode_idx, n_nd):
    theta_x = np.array([PI[j * 6 + 0, mode_idx] for j in range(n_nd)])
    theta_y = np.array([PI[j * 6 + 1, mode_idx] for j in range(n_nd)])
    theta_z = np.array([PI[j * 6 + 2, mode_idx] for j in range(n_nd)])

    u_x = np.array([PI[j * 6 + 3, mode_idx] for j in range(n_nd)])
    u_y = np.array([PI[j * 6 + 4, mode_idx] for j in range(n_nd)])
    u_z = np.array([PI[j * 6 + 5, mode_idx] for j in range(n_nd)])

    return theta_x, theta_y, theta_z, u_x, u_y, u_z


# -------------------------------------------------
# Classify mode type
# -------------------------------------------------
def classify_mode(PI, mode_idx, n_nd):
    theta_x, theta_y, theta_z, u_x, u_y, u_z = get_mode_components(
        PI, mode_idx, n_nd)

    amp_torsion = la.norm(theta_x)
    amp_axial = la.norm(u_x)
    amp_by = la.norm(u_y)
    amp_bz = la.norm(u_z)

    # dominant shape used for plotting
    if amp_torsion > 2.0 * max(amp_axial, amp_by, amp_bz):
        mode_type = "torsion"
        num_shape = normalize_mode(theta_x)

    elif amp_axial > 2.0 * max(amp_by, amp_bz) and amp_axial > 0.25 * amp_torsion:
        mode_type = "axial"
        num_shape = normalize_mode(u_x)

    else:
        if amp_by >= amp_bz:
            mode_type = "bending-y"   # deflection in y, bending about z
            num_shape = normalize_mode(u_y)
        else:
            mode_type = "bending-z"   # deflection in z, bending about y
            num_shape = normalize_mode(u_z)

    info = {
        "amp_torsion": amp_torsion,
        "amp_axial": amp_axial,
        "amp_by": amp_by,
        "amp_bz": amp_bz,
    }

    return mode_type, num_shape, info


# -------------------------------------------------
# Best analytical match for each family
# -------------------------------------------------
def best_reference(mode_type, num_shape, x_nodes, xi, n_try):
    if mode_type in ["bending-y", "bending-z"]:
        refs = np.column_stack([
            phi_bending_cantilever(xi, k)
            for k in range(min(n_try, len(betaL_bending)))
        ])
        labels = [f"Bending shape {k+1}" for k in range(refs.shape[1])]

    elif mode_type == "axial":
        refs = np.column_stack([
            phi_axial_cantilever(x_nodes, L, k)
            for k in range(n_try)
        ])
        labels = [f"Axial shape {k+1}" for k in range(refs.shape[1])]

    else:  # torsion
        refs = np.column_stack([
            phi_torsion_cantilever(x_nodes, L, k)
            for k in range(n_try)
        ])
        labels = [f"Torsion shape {k+1}" for k in range(refs.shape[1])]

    macs = np.array([mac(num_shape, refs[:, j]) for j in range(refs.shape[1])])
    j_best = int(np.argmax(macs))
    ref = refs[:, j_best].copy()

    if np.dot(num_shape, ref) < 0:
        ref *= -1

    return ref, labels[j_best], macs[j_best]


# -------------------------------------------------
# Print classification summary
# -------------------------------------------------
mode_types = []
num_shapes = []

print("\nMode classification summary")
print(
    "mode | type       | freq [Hz] | ||u_y||   | ||u_z||   | ||u_x||   | ||theta_x||")
print("-" * 82)

for r in range(n_md):
    mode_type, num_shape, info = classify_mode(PI_e, r, n_nd)
    mode_types.append(mode_type)
    num_shapes.append(num_shape)

    print(
        f"{r+1:>4d} | "
        f"{mode_type:<10s} | "
        f"{omega_e[r]/(2*np.pi):>9.4f} | "
        f"{info['amp_by']:>9.3e} | "
        f"{info['amp_bz']:>9.3e} | "
        f"{info['amp_axial']:>9.3e} | "
        f"{info['amp_torsion']:>11.3e}"
    )


# -------------------------------------------------
# Plot classified shapes against best analytical match
# -------------------------------------------------
fig, axes = plt.subplots(n_md, 1, figsize=(8, 2.5 * n_md), sharex=True)
if n_md == 1:
    axes = [axes]

for r, ax in enumerate(axes):
    num_shape = num_shapes[r]
    mode_type = mode_types[r]

    ref, ref_label, best_mac = best_reference(
        mode_type, num_shape, x_nodes, xi, n_md)

    ax.plot(x_nodes, num_shape, "o-", label=f"PI_e mode {r+1} ({mode_type})")
    ax.plot(x_nodes, ref, "--", label=f"{ref_label}, MAC={best_mac:.4f}")
    ax.set_ylabel("normalized")
    ax.grid(True)
    ax.legend(loc="best")

axes[-1].set_xlabel("x [m]")
plt.suptitle("PI_e mode labeling: bending-y / bending-z / axial / torsion")
plt.tight_layout()
plt.show()
