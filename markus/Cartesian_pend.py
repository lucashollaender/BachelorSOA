from Cart_pend_functions import run_single_case, run_benchmark
import matplotlib.pyplot as plt
import numpy as np
import scipy

if __name__ == "__main__":
    MODE = "single"

    if MODE == "single":
        N = 10
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
            n_runs=3,
            theta_init_fn=lambda N: np.zeros(N)
        )
    else:
        raise ValueError("MODE must be 'single' or 'benchmark'")
