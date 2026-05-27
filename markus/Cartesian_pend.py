from Cart_pend_functions import run_single_case, run_benchmark
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time

print("numpy:", np.__version__)
print("scipy:", sp.__version__)


def bench_solve(n, reps=2000):
    A = np.random.randn(n, n)
    A = A.T @ A + np.eye(n)   # make it well-conditioned SPD-ish
    b = np.random.randn(n)

    t0 = time.perf_counter()
    for _ in range(reps):
        np.linalg.solve(A, b)
    dt = (time.perf_counter() - t0) / reps
    print(f"n={n}, avg solve = {dt:.6e} s")


bench_solve(95)
bench_solve(99)
bench_solve(100)

if __name__ == "__main__":
    MODE = "single"

    if MODE == "single":
        N = 20
        theta_init = 0 * np.ones(N)

        result = run_single_case(
            N=N,
            L=1,
            m=1,
            g=9.82,
            T_end=20.0,
            theta_init=theta_init,
            animate=True
        )

        if len(result) == 4:
            sol, params, elapsed, ani = result
        else:
            sol, params, elapsed = result

        plt.show()

    elif MODE == "benchmark":
        run_benchmark(
            N_values=[1],
            L=1.0,
            m=1.0,
            g=9.82,
            T_end=10.0,
            n_runs=1,
            theta_init_fn=lambda N: np.zeros(N)
        )
    else:
        raise ValueError("MODE must be 'single' or 'benchmark'")
