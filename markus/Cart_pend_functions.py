import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ----------------------------
# Rotation utilities
# ----------------------------
def A_matrix(phi):
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([[c, -s],
                     [s,  c]])


def B_matrix(phi):
    s = np.sin(phi)
    c = np.cos(phi)
    return np.array([[-s, -c],
                     [c, -s]])


# ----------------------------
# Joint building blocks
# ----------------------------
def apos_joint(q, qdot, s_local, C, request):
    r = q[:2]
    th = q[2]
    thd = qdot[2]

    if request == "PHI":
        return r + A_matrix(th) @ s_local - C

    elif request == "PHI_q":
        return np.hstack((np.eye(2), (B_matrix(th) @ s_local).reshape(2, 1)))

    elif request == "gamma":
        return (thd**2) * (A_matrix(th) @ s_local)

    else:
        raise ValueError("Unknown request")


def rev_joint(q1, q1dot, q2, q2dot, s1, s2, request):
    r1 = q1[:2]
    th1 = q1[2]
    th1d = q1dot[2]

    r2 = q2[:2]
    th2 = q2[2]
    th2d = q2dot[2]

    if request == "PHI":
        return r1 + A_matrix(th1) @ s1 - r2 - A_matrix(th2) @ s2

    elif request == "PHI_q":
        left = np.hstack((np.eye(2),  (B_matrix(th1) @ s1).reshape(2, 1)))
        right = np.hstack((-np.eye(2), -(B_matrix(th2) @ s2).reshape(2, 1)))
        return np.hstack((left, right))

    elif request == "gamma":
        return (th1d**2) * (A_matrix(th1) @ s1) - (th2d**2) * (A_matrix(th2) @ s2)

    else:
        raise ValueError("Unknown request")


# ----------------------------
# System assembly
# ----------------------------
def build_mass_matrix(params):
    N = params["N"]
    m = params["m"]
    L = params["L"]
    J = m * L**2 / 12.0

    M = np.zeros((3 * N, 3 * N))
    for i in range(N):
        idx = slice(3 * i, 3 * i + 3)
        M[idx, idx] = np.diag([m, m, J])
    return M


def phi(q, params):
    N = params["N"]
    C1 = params["C1"]
    sA = params["sA"]
    sB = params["sB"]

    out = np.zeros(2 * N)

    q1 = q[0:3]
    out[0:2] = apos_joint(q1, np.zeros(3), sA, C1, "PHI")

    row = 2
    for i in range(N - 1):
        qi = q[3 * i:3 * i + 3]
        qj = q[3 * (i + 1):3 * (i + 1) + 3]
        out[row:row + 2] = rev_joint(qi, np.zeros(3),
                                     qj, np.zeros(3), sB, sA, "PHI")
        row += 2

    return out


def phi_q(q, params):
    N = params["N"]
    C1 = params["C1"]
    sA = params["sA"]
    sB = params["sB"]

    Jac = np.zeros((2 * N, 3 * N))

    q1 = q[0:3]
    Jac[0:2, 0:3] = apos_joint(q1, np.zeros(3), sA, C1, "PHI_q")

    row = 2
    for i in range(N - 1):
        qi = q[3 * i:3 * i + 3]
        qj = q[3 * (i + 1):3 * (i + 1) + 3]
        Jac[row:row + 2, 3 * i:3 * i + 6] = rev_joint(
            qi, np.zeros(3), qj, np.zeros(3), sB, sA, "PHI_q"
        )
        row += 2

    return Jac


def gamma(q, qdot, params):
    N = params["N"]
    C1 = params["C1"]
    sA = params["sA"]
    sB = params["sB"]

    out = np.zeros(2 * N)

    q1 = q[0:3]
    q1d = qdot[0:3]
    out[0:2] = apos_joint(q1, q1d, sA, C1, "gamma")

    row = 2
    for i in range(N - 1):
        qi = q[3 * i:3 * i + 3]
        qid = qdot[3 * i:3 * i + 3]
        qj = q[3 * (i + 1):3 * (i + 1) + 3]
        qjd = qdot[3 * (i + 1):3 * (i + 1) + 3]
        out[row:row + 2] = rev_joint(qi, qid, qj, qjd, sB, sA, "gamma")
        row += 2

    return out


def applied_forces(params):
    N = params["N"]
    m = params["m"]
    g = params["g"]

    Qa = np.zeros(3 * N)
    for i in range(N):
        Qa[3 * i + 1] = -m * g
    return Qa


def odefun(t, y, params):
    N = params["N"]
    q = y[:3 * N]
    qdot = y[3 * N:]

    M = params["M"]
    Jac = phi_q(q, params)
    gam = gamma(q, qdot, params)
    Qa = params["Qa"]

    KKT = np.block([
        [M, Jac.T],
        [Jac, np.zeros((2 * N, 2 * N))]
    ])
    rhs = np.concatenate((Qa, gam))
    sol = np.linalg.solve(KKT, rhs)

    qdd = sol[:3 * N]
    return np.concatenate((qdot, qdd))


# ----------------------------
# Initial condition
# ----------------------------
def build_consistent_initial_state(params, theta_init):
    N = params["N"]
    sA = params["sA"]
    sB = params["sB"]
    C1 = params["C1"]

    q0 = np.zeros(3 * N)

    A1 = A_matrix(theta_init[0])
    r1 = C1 - A1 @ sA
    q0[0:3] = [r1[0], r1[1], theta_init[0]]

    prev_B = r1 + A1 @ sB
    for i in range(1, N):
        Ai = A_matrix(theta_init[i])
        ri = prev_B - Ai @ sA
        q0[3 * i:3 * i + 3] = [ri[0], ri[1], theta_init[i]]
        prev_B = ri + Ai @ sB

    qdot0 = np.zeros(3 * N)
    return np.concatenate((q0, qdot0))


# ----------------------------
# Animation helpers
# ----------------------------
def get_link_endpoints(q, params):
    N = params["N"]
    sA = params["sA"]
    sB = params["sB"]

    endpoints = []
    for i in range(N):
        qi = q[3 * i:3 * i + 3]
        r = qi[:2]
        th = qi[2]
        A = r + A_matrix(th) @ sA
        B = r + A_matrix(th) @ sB
        endpoints.append((A, B))
    return endpoints


def animate_solution(t, Y, params, step=5):
    N = params["N"]
    q_hist = Y[:3 * N, :]

    total_length = N * params["L"]
    margin = 0.2 * total_length

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlim(-total_length - margin, total_length + margin)
    ax.set_ylim(-total_length - margin, total_length + margin)
    ax.set_title("N-body Pendulum")

    ax.plot(params["C1"][0], params["C1"][1], "^", markersize=10, color="gray")

    lines = []
    for _ in range(N):
        line, = ax.plot([], [], "-", lw=2)
        lines.append(line)

    trace_x = []
    trace_y = []
    trace_line, = ax.plot([], [], "k--", lw=1)

    def update(frame):
        k = min(frame * step, q_hist.shape[1] - 1)
        q = q_hist[:, k]
        endpoints = get_link_endpoints(q, params)

        for i, (A, B) in enumerate(endpoints):
            lines[i].set_data([A[0], B[0]], [A[1], B[1]])

        last_B = endpoints[-1][1]
        trace_x.append(last_B[0])
        trace_y.append(last_B[1])
        trace_line.set_data(trace_x, trace_y)

        return lines + [trace_line]

    nframes = max(1, q_hist.shape[1] // step)

    # draw first frame explicitly
    update(0)

    ani = FuncAnimation(
        fig,
        update,
        frames=nframes,
        interval=30,
        blit=False,      # more robust than blit=True
        repeat=True
    )

    return fig, ani

# ----------------------------
# Case setup
# ----------------------------


def make_params(N, L=1.0, m=1.0, g=9.82):
    params = {
        "N": N,
        "L": L,
        "m": m,
        "g": g,
        "C1": np.array([0.0, 0.0]),
        "sA": np.array([-L / 2, 0.0]),
        "sB": np.array([L / 2, 0.0]),
    }
    params["M"] = build_mass_matrix(params)
    params["Qa"] = applied_forces(params)
    return params


def solve_case(N, L=1.0, m=1.0, g=9.82, T_end=10.0,
               theta_init=None, rtol=1e-7, atol=1e-8, max_step=0.01):
    params = make_params(N, L=L, m=m, g=g)

    if theta_init is None:
        theta_init = -np.pi / 2 + np.deg2rad(np.linspace(20, 5, N))
    else:
        theta_init = np.asarray(theta_init, dtype=float)
        if len(theta_init) != N:
            raise ValueError(f"theta_init must have length {N}")

    y0 = build_consistent_initial_state(params, theta_init)

    t0 = time.perf_counter()
    sol = solve_ivp(
        fun=lambda t, y: odefun(t, y, params),
        t_span=(0.0, T_end),
        y0=y0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        max_step=max_step
    )
    elapsed = time.perf_counter() - t0

    return sol, params, elapsed


# ----------------------------
# Public run modes
# ----------------------------
def run_single_case(N=10, L=1.0, m=1.0, g=9.82, T_end=10.0,
                    theta_init=None, animate=True):
    sol, params, elapsed = solve_case(
        N=N, L=L, m=m, g=g, T_end=T_end, theta_init=theta_init
    )

    print(f"N = {N}")
    print(f"Solve time = {elapsed:.4f} s")
    print(f"Success = {sol.success}")
    print(f"Message = {sol.message}")

    t = sol.t
    Y = sol.y

    phi_hist = np.array([phi(Y[:3 * N, k], params)
                        for k in range(Y.shape[1])]).T
    constraint_error = np.linalg.norm(phi_hist, axis=0)
    print(f"Max constraint error = {np.max(constraint_error):.3e}")

    if animate:
        fig, ani = animate_solution(t, Y, params, step=5)
        return sol, params, elapsed, ani

    return sol, params, elapsed


def run_benchmark(N_values, L=1.0, m=1.0, g=9.82, T_end=10.0,
                  n_runs=2, theta_init=None, theta_init_fn=None):
    solve_times = []

    for N in N_values:
        runtimes = []

        if theta_init_fn is not None:
            this_theta = np.asarray(theta_init_fn(N), dtype=float)
        elif theta_init is not None:
            this_theta = np.asarray(theta_init, dtype=float)
            if len(this_theta) != N:
                raise ValueError(f"theta_init must have length {N} for N={N}")
        else:
            this_theta = None

        for _ in range(n_runs):
            _, _, elapsed = solve_case(
                N=N, L=L, m=m, g=g, T_end=T_end, theta_init=this_theta
            )
            runtimes.append(elapsed)

        avg_time = np.mean(runtimes)
        solve_times.append(avg_time)
        print(f"N = {N:2d}, avg solve time = {avg_time:.4f} s")

    plt.figure(figsize=(7, 5))
    plt.plot(N_values, solve_times, "o-", linewidth=2)
    plt.xlabel("Number of bodies, N")
    plt.ylabel("Computational time [s]")
    plt.title("Computational time vs number of bodies")
    plt.grid(True)
    plt.show()

    return np.array(N_values), np.array(solve_times)
