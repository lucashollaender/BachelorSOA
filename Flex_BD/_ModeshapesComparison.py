import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
from Structural_Analysis_BD_Rect import Structural_Analysis_BD_Rect
import pandas as pd

# Settings
L = 1
klOO1 = np.array([L, 0, 0]).reshape(3, 1)

E = 1.93e9
G = 6.902e9
c = 0.02
rho = 1300

n_nd = 10
n_md = 10

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
analysis = Structural_Analysis_BD_Rect(j1, r1, f1)
PI_e = analysis.PI_e

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
    23.56194490,
    26.70353756,
    29.84513021
])

# Section properties used for analytical frequencies
A = w * h
I_y = w * h**3 / 12.0
I_z = h * w**3 / 12.0
J_p = I_y + I_z
a = w / 2.0
b = h / 2.0
K_t = a * b**3 * (16/3 - 3.36 * a / b * (1 - a**4 / (12 * b**4)))


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


def analytical_frequency_hz(mode_type, mode_index):
    n = mode_index + 1

    if mode_type == "bending-y":
        betaL = betaL_bending[mode_index]
        omega = (betaL**2) * np.sqrt(E * I_z / (rho * A * L**4))

    elif mode_type == "bending-z":
        betaL = betaL_bending[mode_index]
        omega = (betaL**2) * np.sqrt(E * I_y / (rho * A * L**4))

    elif mode_type == "axial":
        k_n = (2 * n - 1) * np.pi / (2 * L)
        omega = k_n * np.sqrt(E / rho)

    elif mode_type == "torsion":
        k_n = (2 * n - 1) * np.pi / (2 * L)
        omega = k_n * np.sqrt(G * K_t / (rho * J_p))

    else:
        raise ValueError(f"Unknown mode type: {mode_type}")

    return omega / (2 * np.pi)


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
        PI, mode_idx, n_nd
    )

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
analytical_freqs = []

print("\nMode classification summary")
print(
    "mode | type       | num freq [Hz] | ana freq [Hz] | ||u_y||   | ||u_z||   | ||u_x||   | ||theta_x||"
)
print("-" * 100)

family_counters = {
    "bending-y": 0,
    "bending-z": 0,
    "axial": 0,
    "torsion": 0,
}

for r in range(n_md):
    mode_type, num_shape, info = classify_mode(PI_e, r, n_nd)
    mode_types.append(mode_type)
    num_shapes.append(num_shape)

    family_index = family_counters[mode_type]
    f_analytical = analytical_frequency_hz(mode_type, family_index)
    analytical_freqs.append(f_analytical)
    family_counters[mode_type] += 1

    print(
        f"{r+1:>4d} | "
        f"{mode_type:<10s} | "
        f"{omega_e[r]/(2*np.pi):>13.4f} | "
        f"{f_analytical:>13.4f} | "
        f"{info['amp_by']:>9.3e} | "
        f"{info['amp_bz']:>9.3e} | "
        f"{info['amp_axial']:>9.3e} | "
        f"{info['amp_torsion']:>11.3e}"
    )


# -------------------------------------------------
# Plot classified shapes against best analytical match
# More visible version for 10 modes
# -------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

ncols = 2
nrows = int(np.ceil(n_md / ncols))

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(15, 3.8 * nrows),
    sharex=True,
    constrained_layout=True
)

axes = np.array(axes).reshape(-1)

for r in range(n_md):
    ax = axes[r]
    num_shape = num_shapes[r]
    mode_type = mode_types[r]

    ref, ref_label, best_mac = best_reference(
        mode_type, num_shape, x_nodes, xi, n_md
    )

    ax.plot(
        x_nodes,
        num_shape,
        "o-",
        linewidth=2.8,
        markersize=6.5,
        label=f"Numerical mode {r+1}"
    )

    ax.plot(
        x_nodes,
        ref,
        "--",
        linewidth=2.6,
        label=f"{ref_label}, MAC={best_mac:.4f}"
    )

    ax.axhline(0.0, linestyle=":", linewidth=1.2)
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.set_xlim(0.0, L)
    ax.set_ylim(-1.15, 1.15)

    ax.set_title(
        f"Mode {r+1}: {mode_type}\n"
        f"Num = {omega_e[r]/(2*np.pi):.3f} Hz, "
        f"Ana = {analytical_freqs[r]:.3f} Hz"
    )

    ax.legend(loc="best", frameon=True)

# Hide unused axes if any
for k in range(n_md, len(axes)):
    axes[k].axis("off")

# x-label only on bottom row
for ax in axes[-ncols:]:
    ax.set_xlabel("x [m]")

fig.suptitle(
    "Modes: bending-y / bending-z / axial / torsion",
    fontsize=15
)

plt.show()
