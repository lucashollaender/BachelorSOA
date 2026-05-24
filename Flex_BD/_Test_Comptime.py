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
def build_flexible_soa_model(n_bodies=2, mode_selection=None):
    """
    Builds an n-body flexible SOA model with a specified mode selection.
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

    # Total number of modes from selected mode groups
    n_md = sum(mode_selection.values())

    # Simulation settings
    tf = 1
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
            mode_selection=mode_selection,
        )

        return SOABody(joint, rigid, flex)

    # Important: create a NEW body object for each body
    bodies = [make_body() for _ in range(n_bodies)]

    # Initial position
    initial_revy_angle = np.deg2rad(0)

    # Only rotate the base body, otherwise the chain bends initially
    bodies[-1].set_initial_theta0(np.array([[initial_revy_angle]]))

    system = MultibodySystem(bodies)

    sim = Simulation(system, tf, dt)
    sim.set_tol(atol=1e-4, rtol=1e-3)

    return sim


# ---------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------
def time_soa_simulation(n_bodies, mode_selection, label, solver="Radau"):
    """
    Builds and integrates the SOA model.
    Reports elapsed wall time only.
    """

    total_start = time.perf_counter()

    # Model-build timing
    build_start = time.perf_counter()
    sim = build_flexible_soa_model(
        n_bodies=n_bodies,
        mode_selection=mode_selection,
    )
    build_elapsed = time.perf_counter() - build_start

    # Integration timing
    integration_start = time.perf_counter()
    sim.IntegrateSystem(solver)
    integration_elapsed = time.perf_counter() - integration_start

    total_elapsed = time.perf_counter() - total_start

    return {
        "n_bodies": n_bodies,
        "mode_selection": mode_selection,
        "label": label,
        "solver": solver,
        "build_elapsed": build_elapsed,
        "integration_elapsed": integration_elapsed,
        "total_elapsed": total_elapsed,
    }


def print_timing_report(results):
    print("\nFinished")
    print(f"Bodies = {results['n_bodies']}")
    print(f"Modes  = {results['label']}")
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
    # "timing"     -> run computation-time benchmark
    # "simulation" -> run and animate one simulation
    run_mode = "timing"

    # Used only when run_mode == "simulation"
    n_bodies = 3

    simulation_mode_selection = {
        "bending_xy": 2,
        "bending_xz": 2,
        "axial_x": 1,
    }

    # Used only when run_mode == "timing"
    n_body_values = np.arange(1, 11)

    # Only two mode selections
    mode_cases = [
        {
            "label": "2 bending + 1 axial",
            "mode_selection": {
                "bending_xy": 1,
                "bending_xz": 1,
                "axial_x": 1,
            },
        },
        {
            "label": "4 bending + 1 axial",
            "mode_selection": {
                "bending_xy": 2,
                "bending_xz": 2,
                "axial_x": 1,
            },
        },
    ]

    if run_mode == "timing":

        timing_results = {}

        for case in mode_cases:

            label = case["label"]
            mode_selection = case["mode_selection"]

            SOAElapsed = []

            print("\n" + "=" * 60)
            print(f"Running benchmark for: {label}")
            print(f"Mode selection: {mode_selection}")
            print("=" * 60)

            for n_bodies_i in n_body_values:

                results = time_soa_simulation(
                    n_bodies=n_bodies_i,
                    mode_selection=mode_selection,
                    label=label,
                    solver=solver,
                )

                print_timing_report(results)

                # Use integration elapsed time only
                SOAElapsed.append(results["integration_elapsed"])

            timing_results[label] = SOAElapsed

        # -------------------------------------------------------------
        # Plot computation time vs number of bodies
        # -------------------------------------------------------------
        plt.figure()

        for label, elapsed_values in timing_results.items():
            plt.plot(
                n_body_values,
                elapsed_values,
                "o-",
                label=label,
            )

        plt.xlabel("Number of bodies")
        plt.ylabel("Integration elapsed time [s]")
        plt.title(
            f"SOA computation time for selected modes, solver = {solver}")
        plt.grid(True)
        plt.legend(title="Mode selection")
        plt.tight_layout()
        plt.show()

    elif run_mode == "simulation":

        sim = build_flexible_soa_model(
            n_bodies=n_bodies,
            mode_selection=simulation_mode_selection,
        )

        sim.IntegrateSystem(solver)

        sim.set_camera_hor(90)
        sim.set_camera_ver(0)

        sim.animate_nodes()

    else:
        raise ValueError("run_mode must be either 'timing' or 'simulation'")
