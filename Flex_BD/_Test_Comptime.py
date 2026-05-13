from Body_Properties import Joint, Rigid_Properties, Flex_Properties
from Simulation import Simulation
from MultibodySystem import MultibodySystem
from SOABody import SOABody
import os
import sys
import time
import types
import importlib.util
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------
def build_two_body_flexible_soa_model():
    """
    Builds the same type of 2-body flexible SOA model used in AdamsComparison.py.
    Adjust tf, dt, n_nd, n_md, material data, etc. below if needed.
    """

    # Geometry: two 1 m flexible bodies connected by revolute-y joints
    klOO1 = np.array([1.0, 0.0, 0.0]).reshape(3, 1)
    klOO2 = np.array([1.0, 0.0, 0.0]).reshape(3, 1)

    H_type1 = "revy"
    H_type2 = "revy"

    # Aluminium-like parameters, same as your AdamsComparison.py
    E = 7.17e10
    G = 2.7e10
    c = 0.02
    rho = 2740

    # Beam cross-section
    w = 0.04
    h = 0.04

    # Flexible discretization
    n_nd = 10     # number of nodes
    n_md = 6     # number of retained flexible modes

    j1 = Joint(klOO1, H_type1)
    r1 = Rigid_Properties(rho, w, h)
    f1 = Flex_Properties(E, G, c, n_nd, n_md)

    j2 = Joint(klOO2, H_type2)
    r2 = Rigid_Properties(rho, w, h)
    f2 = Flex_Properties(E, G, c, n_nd, n_md)

    b1 = SOABody(j1, r1, f1)
    b2 = SOABody(j2, r2, f2)

    system = MultibodySystem([b1, b2, b1, b1, b1])

    # Match AdamsComparison.py style
    tf = 10
    dt = 0.01

    sim = Simulation(system, tf, dt)
    sim.set_max_step(dt)

    # sim.IntegrateSystem("Radau")
    # sim.animate_nodes()

    # Optional: set solver tolerances. Uncomment if you want a stricter test.
    # sim.set_tol(atol=1e-8, rtol=1e-10)

    return sim


# ---------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------
def time_soa_simulation(solver="Radau"):
    """
    Builds and integrates the SOA model, measuring both total time and
    integration-only time.
    """

    # Total time includes model construction + integration
    total_wall_start = time.perf_counter()
    total_cpu_start = time.process_time()

    # Model-build timing
    build_wall_start = time.perf_counter()
    build_cpu_start = time.process_time()

    sim = build_two_body_flexible_soa_model()

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
    print(f"Solver = {results['solver']}")

    print("\nIntegration only:")
    print(f"Elapsed time     = {results['integration_wall']:.6f} s")
    print(f"CPU time         = {results['integration_cpu']:.6f} s")
    print(
        f"Average CPU load = {cpu_load(results['integration_cpu'], results['integration_wall']):.2f} %")

    print("\nModel build only:")
    print(f"Elapsed time     = {results['build_wall']:.6f} s")
    print(f"CPU time         = {results['build_cpu']:.6f} s")
    print(
        f"Average CPU load = {cpu_load(results['build_cpu'], results['build_wall']):.2f} %")

    print("\nTotal run: model build + integration")
    print(f"Elapsed time     = {results['total_wall']:.6f} s")
    print(f"CPU time         = {results['total_cpu']:.6f} s")
    print(
        f"Average CPU load = {cpu_load(results['total_cpu'], results['total_wall']):.2f} %")


if __name__ == "__main__":
    results = time_soa_simulation(solver="Radau")
    print_timing_report(results)

AdamsElapsed = [3.31, 3.41, 3.51, 3.87, 3.99, 4.09]
SOAElapsed = [2.48, 14.63, 44.39, 74.73, 179.34, 520.77]

# X-vector for plotting
X = np.arange(len(AdamsElapsed))

plt.plot(X, AdamsElapsed, label='Adams')
plt.plot(X, SOAElapsed, label='SOA')

plt.xlabel('Bodies')
plt.ylabel('Computation time')
plt.legend()
plt.show()
