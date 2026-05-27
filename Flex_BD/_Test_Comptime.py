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
def build_flexible_soa_model(n_bodies=2, mode_selection=None, n_md=20):
    """
    Builds an n-body flexible SOA model with a specified mode selection.

    If mode_selection is None:
        The first n_md modes are used.
        These should be the lowest-frequency modes if your structural analysis
        sorts the eigenmodes by frequency.
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

    # Simulation settings
    tf = 0.5
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
    sim.set_tol(atol=1e-2, rtol=1e-4)

    return sim


# ---------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------
def time_soa_simulation(
    n_bodies,
    mode_selection,
    label,
    solver="Radau",
    n_md=20,
):
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
        n_md=n_md,
    )
    build_elapsed = time.perf_counter() - build_start

    # Integration timing
    integration_start = time.perf_counter()
    sim.IntegrateSystem(solver)
    integration_elapsed = time.perf_counter() - integration_start

    total_elapsed = time.perf_counter() - total_start

    return {
        "n_bodies": n_bodies,
        "n_md": n_md,
        "mode_selection": mode_selection,
        "label": label,
        "solver": solver,
        "build_elapsed": build_elapsed,
        "integration_elapsed": integration_elapsed,
        "total_elapsed": total_elapsed,
    }


def average_soa_simulation_time(
    n_bodies,
    mode_selection,
    label,
    solver="Radau",
    n_runs=2,
    n_md=20,
):
    """
    Runs the same SOA simulation n_runs times and returns averaged timings.
    Each run rebuilds the model from scratch before integration.
    """

    run_results = []

    for run_idx in range(n_runs):
        print(f"\nRun {run_idx + 1}/{n_runs}")

        results = time_soa_simulation(
            n_bodies=n_bodies,
            mode_selection=mode_selection,
            label=label,
            solver=solver,
            n_md=n_md,
        )

        print_timing_report(results)
        run_results.append(results)

    build_times = np.array([r["build_elapsed"] for r in run_results])
    integration_times = np.array(
        [r["integration_elapsed"] for r in run_results]
    )
    total_times = np.array([r["total_elapsed"] for r in run_results])

    averaged_results = {
        "n_bodies": n_bodies,
        "n_md": n_md,
        "mode_selection": mode_selection,
        "label": label,
        "solver": solver,
        "n_runs": n_runs,
        "build_elapsed": np.mean(build_times),
        "integration_elapsed": np.mean(integration_times),
        "total_elapsed": np.mean(total_times),
        "build_elapsed_std": np.std(build_times, ddof=1) if n_runs > 1 else 0.0,
        "integration_elapsed_std": np.std(integration_times, ddof=1) if n_runs > 1 else 0.0,
        "total_elapsed_std": np.std(total_times, ddof=1) if n_runs > 1 else 0.0,
        "all_runs": run_results,
    }

    return averaged_results


def print_timing_report(results):
    print("\nFinished")
    print(f"Bodies = {results['n_bodies']}")
    print(f"Modes  = {results['label']}")
    print(f"n_md   = {results['n_md']}")
    print(f"Solver = {results['solver']}")

    print("\nIntegration only:")
    print(f"Elapsed time = {results['integration_elapsed']:.6f} s")

    print("\nModel build only:")
    print(f"Elapsed time = {results['build_elapsed']:.6f} s")

    print("\nTotal run: model build + integration")
    print(f"Elapsed time = {results['total_elapsed']:.6f} s")


def print_average_timing_report(results):
    print("\nAverage over repeated runs")
    print(f"Bodies = {results['n_bodies']}")
    print(f"Modes  = {results['label']}")
    print(f"n_md   = {results['n_md']}")
    print(f"Solver = {results['solver']}")
    print(f"Runs   = {results['n_runs']}")

    print("\nIntegration only:")
    print(
        f"Average elapsed time = {results['integration_elapsed']:.6f} s "
        f"(+/- {results['integration_elapsed_std']:.6f} s)"
    )

    print("\nModel build only:")
    print(
        f"Average elapsed time = {results['build_elapsed']:.6f} s "
        f"(+/- {results['build_elapsed_std']:.6f} s)"
    )

    print("\nTotal run: model build + integration")
    print(
        f"Average elapsed time = {results['total_elapsed']:.6f} s "
        f"(+/- {results['total_elapsed_std']:.6f} s)"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":

    solver = "Radau"
    n_runs = 2

    # Choose:
    # "timing"      -> run computation-time benchmark vs number of bodies
    # "mode_timing" -> run computation-time benchmark vs number of modes
    # "simulation"  -> run and animate one simulation
    run_mode = "mode_timing"

    # Used only when run_mode == "simulation"
    n_bodies = 3

    simulation_mode_selection = {
        "bending_xy": 2,
        "bending_xz": 2,
        "axial_x": 1,
    }

    simulation_n_md = 20

    # Used only when run_mode == "timing"
    n_body_values = np.arange(1, 9)

    # Mode-selection cases for the normal timing benchmark
    mode_cases = [
        {
            "label": "2 bending",
            "mode_selection": {
                "bending_xy": 1,
                "bending_xz": 1,
            },

        },
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

    # Used only when run_mode == "mode_timing"
    # Number of bodies is constant for each curve.
    constant_body_cases = [2, 4, 6]

    # Number of first/lowest-frequency modes to include.
    # No manual mode_selection is used in this benchmark.
    n_mode_values = np.arange(1, 20)

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

                results = average_soa_simulation_time(
                    n_bodies=n_bodies_i,
                    mode_selection=mode_selection,
                    label=label,
                    solver=solver,
                    n_runs=n_runs,
                    n_md=20,
                )

                print_average_timing_report(results)

                # Use average integration elapsed time only
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
        plt.ylabel("Average computation time [s]")
        plt.title("SOA Computation Time")
        plt.grid(True)
        plt.legend(title="Mode selection")
        plt.tight_layout()
        plt.show()

    elif run_mode == "mode_timing":

        mode_timing_results = {}

        for n_bodies_i in constant_body_cases:

            SOAElapsed = []

            print("\n" + "=" * 60)
            print(f"Running mode benchmark for {n_bodies_i} bodies")
            print("Mode selection: None")
            print("Using first n_md modes / lowest-frequency modes")
            print("=" * 60)

            for n_md_i in n_mode_values:

                label = f"First {n_md_i} modes"

                results = average_soa_simulation_time(
                    n_bodies=n_bodies_i,
                    mode_selection=None,
                    label=label,
                    solver=solver,
                    n_runs=n_runs,
                    n_md=n_md_i,
                )

                print_average_timing_report(results)

                # Use average integration elapsed time only
                SOAElapsed.append(results["integration_elapsed"])

            mode_timing_results[n_bodies_i] = SOAElapsed

        # -------------------------------------------------------------
        # Plot computation time vs number of modes
        # -------------------------------------------------------------
        plt.figure()

        for n_bodies_i, elapsed_values in mode_timing_results.items():
            plt.plot(
                n_mode_values,
                elapsed_values,
                "o-",
                label=f"{n_bodies_i} bodies",
            )

        plt.xlabel("Number of modes")
        plt.ylabel("Average computation time [s]")
        plt.title("SOA Computation Time")
        plt.grid(True)
        plt.legend(title="Number of bodies")
        plt.tight_layout()
        plt.show()

    elif run_mode == "simulation":

        sim = build_flexible_soa_model(
            n_bodies=n_bodies,
            mode_selection=simulation_mode_selection,
            n_md=simulation_n_md,
        )

        sim.IntegrateSystem(solver)

        sim.set_camera_hor(90)
        sim.set_camera_ver(0)

        sim.animate_nodes()

    else:
        raise ValueError(
            "run_mode must be either 'timing', 'mode_timing', or 'simulation'"
        )
