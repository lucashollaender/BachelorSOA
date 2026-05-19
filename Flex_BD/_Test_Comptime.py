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
    """

    # Material parameters (alu)
    E = 7.17e10
    G = 2.7e10
    c = 0.02
    rho = 2740

    # Beam cross-section
    w = 0.06
    h = 0.04

    # Flexible discretization
    n_nd = 11
    n_md = 20

    # Simulation settings
    tf = 5
    dt = 0.01

    def make_body():
        klOO = np.array([1.0, 0.0, 0.0]).reshape(3, 1)

        joint = Joint(klOO, "revy")
        rigid = Rigid_Properties(rho, w, h)
        flex = Flex_Properties(
            E,
            G,
            c,
            n_nd,
            n_md,
            mode_selection={
                "bending_xy": 2,
                "bending_xz": 2,
                "axial_x": 1,
            },
        )

        return SOABody(joint, rigid, flex)

    # Important: create a NEW body object for each body
    bodies = [make_body() for _ in range(n_bodies)]

    # Initial position
    initial_revy_angle = np.deg2rad(85)

    # Only rotate the base body, otherwise the chain bends
    bodies[-1].set_initial_theta0(np.array([[initial_revy_angle]]))

    system = MultibodySystem(bodies)

    sim = Simulation(system, tf, dt)
    sim.set_tol(atol=1e-4, rtol=1e-3)

    return sim


# ---------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------
def time_soa_simulation(n_bodies, solver="Radau"):
    """
    Builds and integrates the SOA model.
    Reports elapsed wall time only.
    """

    total_start = time.perf_counter()

    # Model-build timing
    build_start = time.perf_counter()
    sim = build_flexible_soa_model(n_bodies)
    build_elapsed = time.perf_counter() - build_start

    # Integration timing
    integration_start = time.perf_counter()
    sim.IntegrateSystem(solver)
    integration_elapsed = time.perf_counter() - integration_start

    total_elapsed = time.perf_counter() - total_start

    return {
        "n_bodies": n_bodies,
        "solver": solver,
        "build_elapsed": build_elapsed,
        "integration_elapsed": integration_elapsed,
        "total_elapsed": total_elapsed,
    }


def print_timing_report(results):
    print("\nFinished")
    print(f"Bodies = {results['n_bodies']}")
    print(f"Solver = {results['solver']}")

    print("\nIntegration only:")
    print(f"Elapsed time = {results['integration_elapsed']:.6f} s")

    print("\nModel build only:")
    print(f"Elapsed time = {results['build_elapsed']:.6f} s")

    print("\nTotal run: model build + integration")
    print(f"Elapsed time = {results['total_elapsed']:.6f} s")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":

    solver = "Radau"

    # Choose:
    # "timing"     -> run computation-time benchmark and compare with Adams
    # "simulation" -> run and animate one corresponding simulation
    run_mode = "simulation"

    # Used only when run_mode == "simulation"
    n_bodies = 3

    # Used only when run_mode == "timing"
    n_body_values = np.arange(1, 11)

    # Adams elapsed-time results
    AdamsElapsed = [3.30, 3.36, 3.47, 3.59, 3.74, 3.78, 4.12, 4.14, 4.17, 4.35]

    if run_mode == "timing":

        SOAElapsed = []

        for n_bodies_i in n_body_values:
            results = time_soa_simulation(
                n_bodies=n_bodies_i,
                solver=solver,
            )

            print_timing_report(results)

            # Use total elapsed time: model build + integration
            SOAElapsed.append(results["integration_elapsed"])

        # Plot elapsed time comparison
        plt.figure()
        plt.plot(
            n_body_values,
            AdamsElapsed[:len(n_body_values)],
            "o-",
            label="Adams elapsed",
        )
        plt.plot(
            n_body_values,
            SOAElapsed,
            "o-",
            label="SOA elapsed",
        )

        plt.xlabel("Number of bodies")
        plt.ylabel("Elapsed time [s]")
        plt.title(f"Elapsed time comparison, solver = {solver}")
        plt.grid(True)
        plt.legend()
        plt.show()

    elif run_mode == "simulation":

        sim = build_flexible_soa_model(n_bodies=n_bodies)

        sim.IntegrateSystem(solver)

        sim.set_camera_hor(90)
        sim.set_camera_ver(0)

        sim.animate_nodes()

    else:
        raise ValueError("run_mode must be either 'timing' or 'simulation'")
