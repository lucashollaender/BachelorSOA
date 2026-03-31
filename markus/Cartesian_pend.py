from Cart_pend_functions import run_single_case, run_benchmark
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    MODE = "benchmark"

    if MODE == "single":
        N = 30
        theta_init = 0 * np.ones(N)

        result = run_single_case(
            N=N,
            L=1,
            m=1,
            g=9.82,
            T_end=10.0,
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
            N_values=[1, 2, 3, 5, 8, 10, 12, 15, 20, 50, 100, 200],
            L=1.0,
            m=1.0,
            g=9.82,
            T_end=10.0,
            n_runs=3,
            theta_init_fn=lambda N: np.zeros(N)
        )

    else:
        raise ValueError("MODE must be 'single' or 'benchmark'")
