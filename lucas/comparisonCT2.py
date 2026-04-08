from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Worker-safe helper functions
# ---------------------------

def zero_theta_init(N):
    import numpy as np
    return np.zeros(N)

def cartesian_worker(k, L, m, g, T_end, n_runs):
    import numpy as np
    from markus.Cart_pend_functions import run_benchmark

    N_vals, solve = run_benchmark(
        N_values=[k],
        L=L,
        m=m,
        g=g,
        T_end=T_end,
        n_runs=n_runs,
        theta_init_fn=zero_theta_init,
    )

    return int(N_vals[0]), float(np.asarray(solve)[0])

def soa_worker(k, n_samples, tf, dt, klOO, H_type, m_body, CkJk, klOC):
    import numpy as np
    from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation
    from SOALIB.ticToc import TicToc

    sample_times = []

    for _ in range(n_samples):
        bodies = []

        # Use range(k) if you want exactly k bodies
        for i_body in range(k):
            j = Joint(klOO, H_type)
            i = Inertia(m_body, CkJk, klOC)
            b = SOABody(j, i)

            if i_body == 0:
                b.set_initial_theta0(np.pi / 2)

            bodies.append(b)

        bodies.reverse()
        system = MultibodySystem(bodies)
        sim = Simulation(system, tf, dt)

        timer = TicToc(False)
        timer.tic()
        sim.IntegrateSystem()
        sample_times.append(timer.toc())

    return int(k), float(np.mean(sample_times))


# ---------------------------
# Main benchmark
# ---------------------------

def main():
    # Common N values
    N_values = list(range(1, 140, 1))

    # Cartesian settings
    L = 1.0
    m_cart = 1.0
    g = 9.82
    T_end = 5
    cart_runs = 3

    # SOA settings
    n_samples = 3
    tf = 5
    dt = 0.01

    klOO = np.array([0, 0, 1])
    H_type = "revx"
    m_body = 1.0
    CkJk = np.array([1/12, 1/12, 1/12])
    klOC = np.array([0, 0, 0.5])

    max_workers = min(os.cpu_count() or 1, len(N_values))
    ctx = mp.get_context("spawn")   # cross-platform safe

    # ---------------------------
    # Run Cartesian in parallel
    # ---------------------------
    cart_results = {}

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = [
            ex.submit(cartesian_worker, k, L, m_cart, g, T_end, cart_runs)
            for k in N_values
        ]

        for fut in as_completed(futures):
            k, t = fut.result()
            cart_results[k] = t
            print(f"Cartesian N={k}: done")

    # ---------------------------
    # Run SOA in parallel
    # ---------------------------
    soa_results = {}

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = [
            ex.submit(soa_worker, k, n_samples, tf, dt, klOO, H_type, m_body, CkJk, klOC)
            for k in N_values
        ]

        for fut in as_completed(futures):
            k, t = fut.result()
            soa_results[k] = t
            print(f"SOA N={k}: done")

    # Sort results back into arrays
    N = np.array(N_values)
    solvetime = np.array([cart_results[k] for k in N], dtype=float)
    dt_n = np.array([soa_results[k] for k in N], dtype=float)

    # Save results
    df = pd.DataFrame({
        "N": N,
        "cartesian_time_s": solvetime,
        "soa_time_s": dt_n
    })
    df.to_csv("benchmark_results.csv", index=False)
    print("Saved benchmark_results.csv")

    # Plot SOA only
    plt.figure()
    plt.plot(N, dt_n, marker='o')
    plt.xlabel("Number of bodies, N")
    plt.ylabel("Average computation time [s]")
    plt.title("SOA: Average Computation Time vs Number of Bodies")
    plt.grid(True)
    plt.xlim(1, max(N))
    plt.ylim(0, max(dt_n) * 1.1)
    plt.show()

    # Comparison plot
    plt.figure(figsize=(8, 5))
    plt.plot(N, solvetime, marker='o', linewidth=2.5, markersize=7, label='Cartesian formulation')
    plt.plot(N, dt_n, marker='s', linewidth=2.5, markersize=7, label='SOA formulation')
    plt.xlabel("Number of bodies / pendulums, N", fontsize=11)
    plt.ylabel("Average computation time [s]", fontsize=11)
    plt.title("Computation time comparison", fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()