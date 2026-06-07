import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from SOABody import SOABody
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
from Structural_Analysis_PM_Rect import Structural_Analysis_PM_Rect


# -------------------------------------------------
# Beam setup
# Use w != h for clean validation (avoids degenerate bending pairs)
# -------------------------------------------------
L = 1.0
klOC = np.array([L/2, 0, 0])

E = 230e9
G = 80e9
c = 0.02
rho = 7850
n_nd = 8
n_md = 4

w = 0.10
h = 0.06   # deliberately non-square for validation

j1 = Joint(L, "fixed")
r1 = Rigid_Properties(rho, klOC, w, h)
f1 = Flex_Properties(E, G, c, n_nd, n_md)
b1 = SOABody(j1, r1, f1)

# Fresh structural-analysis object so we can access K_st, M_nd, etc.
analysis = Structural_Analysis_PM_Rect(r1, f1)

PI = analysis.PI            # full nodal mode matrix used in body
omega = analysis.omega
omega2 = analysis.omega2

print("Natural frequencies [Hz]:")
print(omega / (2*np.pi))


# -------------------------------------------------
# Rebuild the reduced matrices used in get_PI()
# -------------------------------------------------
K_st = analysis.K_st[6:, 6:]
M_nd = analysis.M_nd[6:, 6:]

index = np.zeros((1, 0))
for i in range(analysis.n_elem):
    index_add = np.linspace(i * 6 + 3, i * 6 + 5, 3).reshape(1, 3)
    index = np.hstack([index, index_add])

for i in range(analysis.n_elem):
    index_add = np.linspace(i * 6, i * 6 + 2, 3).reshape(1, 3)
    index = np.hstack([index, index_add])

index = index.flatten().astype(int)
K = K_st[np.ix_(index, index)]
M = M_nd[np.ix_(index, index)]

sz = M.shape[0]
sz2 = sz // 2

K_tt = K[0:sz2, 0:sz2]
K_rr = K[sz2:sz, sz2:sz]
K_tr = K[0:sz2, sz2:sz]
K_rt = K[sz2:sz, 0:sz2]
M_c = M[0:sz2, 0:sz2]

X = la.solve(K_rr, K_rt, assume_a="sym")
K_c = K_tt - K_tr @ X

# Remove the prepended fixed-node zeros from stored PI_t / PI_r
PI_t_red = analysis.PI_t[3:, :]
PI_r_red = analysis.PI_r[3:, :]


# -------------------------------------------------
# Quantitative checks
# -------------------------------------------------
orth_M = PI_t_red.T @ M_c @ PI_t_red
orth_K = PI_t_red.T @ K_c @ PI_t_red
condense_err = la.norm(PI_r_red + X @ PI_t_red)

print("\nBoundary-condition / normalization checks")
print("||PI(root node)|| =", la.norm(PI[0:6, :]))
print("||PI_t^T M_c PI_t - I|| =", la.norm(orth_M - np.eye(n_md)))
print("||PI_t^T K_c PI_t - diag(omega^2)|| =",
      la.norm(orth_K - np.diag(omega2)))
print("||PI_r + X PI_t|| =", condense_err)


# -------------------------------------------------
# Analytical Euler-Bernoulli cantilever mode shapes
# Good reference for slender beams when validating centerline shape
# -------------------------------------------------
betaL = np.array([1.87510407, 4.69409113, 7.85475744, 10.99554073])


def phi_cantilever(xi, mode_index):
    lam = betaL[mode_index]
    sigma = (np.cosh(lam) + np.cos(lam)) / (np.sinh(lam) + np.sin(lam))
    y = np.cosh(lam * xi) - np.cos(lam * xi) - sigma * (
        np.sinh(lam * xi) - np.sin(lam * xi)
    )
    y = y / np.max(np.abs(y))
    return y


x_nodes = np.linspace(0.0, L, n_nd)
xi = x_nodes / L


# -------------------------------------------------
# Extract transverse centerline shape from numerical PI
# For each node j:
# translational rows are PI[j*6+3 : j*6+6, mode]
# -------------------------------------------------
def get_transverse_shape(PI, mode_idx, n_nd):
    U = np.zeros((n_nd, 3))
    for j in range(n_nd):
        U[j, :] = PI[j*6 + 3: j*6 + 6, mode_idx]

    # Beam axis is x, so use transverse y-z part
    A = U[:, 1:3]

    # Principal transverse direction (robust if mode is rotated in yz-plane)
    if la.norm(A) < 1e-14:
        # fallback for unexpected axial mode
        u = U[:, 0].copy()
    else:
        _, _, Vt = la.svd(A, full_matrices=False)
        u = A @ Vt[0, :]

    # normalize and fix sign at tip
    u = u / np.max(np.abs(u))
    if u[-1] < 0:
        u = -u
    return u


def mac(a, b):
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    return (((a.T @ b)**2) / ((a.T @ a) * (b.T @ b))).item()


# -------------------------------------------------
# MAC matrix: numerical modes vs analytical cantilever shapes
# -------------------------------------------------
num_shapes = np.column_stack([
    get_transverse_shape(PI, r, n_nd) for r in range(n_md)
])

ana_shapes = np.column_stack([
    phi_cantilever(xi, r) for r in range(n_md)
])

MAC = np.zeros((n_md, n_md))
for i in range(n_md):
    for j in range(n_md):
        MAC[i, j] = mac(num_shapes[:, i], ana_shapes[:, j])

print("\nMAC matrix (numerical vs analytical cantilever shapes)")
print(np.round(MAC, 4))


# -------------------------------------------------
# Plot comparison
# -------------------------------------------------
fig, axes = plt.subplots(n_md, 1, figsize=(7, 2.2*n_md), sharex=True)

if n_md == 1:
    axes = [axes]

for r, ax in enumerate(axes):
    u_num = num_shapes[:, r]

    # best analytical match
    j_best = np.argmax(MAC[r, :])
    u_ana = ana_shapes[:, j_best].copy()

    # sign alignment
    if np.dot(u_num, u_ana) < 0:
        u_ana *= -1

    ax.plot(x_nodes, u_num, "o-", label=f"Numerical mode {r+1}")
    ax.plot(x_nodes, u_ana, "--",
            label=f"Analytical cantilever shape {j_best+1}")
    ax.set_ylabel("normalized")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("x [m]")
plt.suptitle("Cantilever mode-shape validation")
plt.tight_layout()
plt.show()
