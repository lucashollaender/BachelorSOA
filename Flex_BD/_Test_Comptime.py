from Body_Properties import Joint, Rigid_Properties, Flex_Properties
from Simulation import Simulation
from MultibodySystem import MultibodySystem
from SOABody import SOABody

import time
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------
def build_flexible_soa_model(n_bodies=2):
    """
    Builds an n-body flexible SOA model.
    Uses Simulation.IntegrateSystem("Radau") later, same as before.
    """

    # Material parameters
    E = 7.17e10
    G = 2.7e10
    c = 0.02
    rho = 2740

    # Beam cross-section
    w = 0.06
    h = 0.04

    # Flexible discretization
    n_nd = 20
    n_md = 10

    # Simulation settings
    tf = 10
    dt = 0.01

    def make_body():
        klOO = np.array([1.0, 0.0, 0.0]).reshape(3, 1)

        joint = Joint(klOO, "revy")
        rigid = Rigid_Properties(rho, w, h)
        flex = Flex_Properties(E, G, c, n_nd, n_md, mode_selection={
            "bending_xy": 1,
            "beding_xz": 1, })

        return SOABody(joint, rigid, flex)

    # Important: create a NEW body object for each body
    bodies = [make_body() for _ in range(n_bodies)]

    system = MultibodySystem(bodies)

    sim = Simulation(system, tf, dt)
    # sim.set_max_step()
    sim.set_tol(atol=1e-4, rtol=1e-3)

    return sim


# ---------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------
def time_soa_simulation(n_bodies, solver="Radau"):
    """
    Builds and integrates the SOA model.
    Uses sim.IntegrateSystem(solver), same as your original script.
    """

    total_wall_start = time.perf_counter()
    total_cpu_start = time.process_time()

    # Model-build timing
    build_wall_start = time.perf_counter()
    build_cpu_start = time.process_time()

    sim = build_flexible_soa_model(n_bodies)

    build_wall = time.perf_counter() - build_wall_start
    build_cpu = time.process_time() - build_cpu_start

    # Integration timing
    integration_wall_start = time.perf_counter()
    integration_cpu_start = time.process_time()

    sim.IntegrateSystem(solver)

    integration_wall = time.perf_counter() - integration_wall_start
    integration_cpu = time.process_time() - integration_cpu_start

    total_wall = time.perf_counter() - total_wall_start
    total_cpu = time.process_time() - total_cpu_start

    return {
        "n_bodies": n_bodies,
        "solver": solver,
        "build_wall": build_wall,
        "build_cpu": build_cpu,
        "integration_wall": integration_wall,
        "integration_cpu": integration_cpu,
        "total_wall": total_wall,
        "total_cpu": total_cpu,
    }


def print_timing_report(results):
    def cpu_load(cpu_time, wall_time):
        if wall_time <= 0.0:
            return float("nan")
        return 100.0 * cpu_time / wall_time

    print("\nFinished")
    print(f"Bodies = {results['n_bodies']}")
    print(f"Solver = {results['solver']}")

    print("\nIntegration only:")
    print(f"Elapsed time     = {results['integration_wall']:.6f} s")
    print(f"CPU time         = {results['integration_cpu']:.6f} s")
    print(
        f"Average CPU load = "
        f"{cpu_load(results['integration_cpu'], results['integration_wall']):.2f} %"
    )

    print("\nModel build only:")
    print(f"Elapsed time     = {results['build_wall']:.6f} s")
    print(f"CPU time         = {results['build_cpu']:.6f} s")
    print(
        f"Average CPU load = "
        f"{cpu_load(results['build_cpu'], results['build_wall']):.2f} %"
    )

    print("\nTotal run: model build + integration")
    print(f"Elapsed time     = {results['total_wall']:.6f} s")
    print(f"CPU time         = {results['total_cpu']:.6f} s")
    print(
        f"Average CPU load = "
        f"{cpu_load(results['total_cpu'], results['total_wall']):.2f} %"
    )


# ---------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------
if __name__ == "__main__":

    solver = "Radau"

    n_body_values = np.arange(1, 7)

    SOAElapsed = []

    for n_bodies in n_body_values:
        results = time_soa_simulation(
            n_bodies=n_bodies,
            solver=solver,
        )

        print_timing_report(results)

        # Use total elapsed wall time: model build + integration
        SOAElapsed.append(results["total_wall"])

    # Adams results
    AdamsElapsed = [3.31, 3.41, 3.51, 3.87, 3.99, 4.09]

    # Plot elapsed time
    plt.figure()
    plt.plot(n_body_values, AdamsElapsed[0:6], "o-", label="Adams elapsed")
    plt.plot(n_body_values, SOAElapsed, "o-", label="SOA elapsed")

    plt.xlabel("Number of bodies")
    plt.ylabel("Elapsed time [s]")
    plt.title(f"Elapsed time comparison, solver = {solver}")
    plt.grid(True)
    plt.legend()
    plt.show()
