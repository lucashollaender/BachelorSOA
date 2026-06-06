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
    Builds an n-body flexible SOA model.

    If mode_selection is None:
        The first n_md lowest-frequency modes are used from all available modes.

    If mode_selection is given:
        The first n_md lowest-frequency modes are selected from the allowed
        mode pool defined by mode_selection.
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
    # "timing"          -> computation-time benchmark vs number of bodies
    # "mode_timing"     -> computation-time benchmark vs number of modes from selected mode pool
    # "mode_timing_all" -> computation-time benchmark vs number of modes with no mode selection
    # "simulation"      -> run and animate one simulation
    run_mode = "simulation"

    # -----------------------------------------------------------------
    # Simulation settings
    # -----------------------------------------------------------------
    # Used only when run_mode == "simulation"
    n_bodies = 20

    simulation_mode_selection = {
        "bending_xy": 1,
        "bending_xz": 1,
        "axial_x": 1,
    }

    simulation_n_md = 20

    # -----------------------------------------------------------------
    # Timing vs number of bodies
    # -----------------------------------------------------------------
    # Used only when run_mode == "timing"
    n_body_values = np.arange(1, 9)

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

    # -----------------------------------------------------------------
    # Mode timing using selected mode pool
    # -----------------------------------------------------------------
    # Used only when run_mode == "mode_timing"
    # This defines the allowed mode pool.
    # The benchmark selects the first n_md lowest-frequency modes from this pool.
    mode_timing_case = {
        "label": "8 bending_xy + 10 bending_xz + 2 axial",
        "mode_selection": {
            "bending_xy": 8,
            "bending_xz": 10,
            "axial_x": 2,
        },
    }

    # Number of bodies to test for selected-pool mode timing
    mode_timing_body_values = np.array([2])

    # Run with 1, 2, 3, ..., 20 modes from the selected mode pool
    n_mode_values = np.arange(
        1, sum(mode_timing_case["mode_selection"].values()) + 1)

    # -----------------------------------------------------------------
    # Mode timing using all available modes
    # -----------------------------------------------------------------
    # Used only when run_mode == "mode_timing_all"
    # No mode_selection is used.
    # The benchmark selects the first n_md lowest-frequency modes overall.
    mode_timing_all_label = "All modes"
    mode_timing_all_mode_selection = None

    # Number of bodies to test for all-mode timing
    mode_timing_all_body_values = np.array([2])

    # Run with 1, 2, 3, ..., 20 lowest-frequency modes overall
    n_mode_values_all = np.arange(1, 30)

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

        label = mode_timing_case["label"]
        mode_selection = mode_timing_case["mode_selection"]
        total_modes = sum(mode_selection.values())

        mode_timing_results = {}

        print("\n" + "=" * 60)
        print("Running mode-number benchmark")
        print(f"Mode pool: {label}")
        print(f"Mode selection pool: {mode_selection}")
        print(f"Maximum selected modes: {total_modes}")
        print("=" * 60)

        for n_bodies_i in mode_timing_body_values:

            SOAElapsed = []

            print("\n" + "=" * 60)
            print(f"Running benchmark for {n_bodies_i} bodies")
            print("=" * 60)

            for n_md_i in n_mode_values:

                current_label = f"First {n_md_i} modes from pool"

                results = average_soa_simulation_time(
                    n_bodies=n_bodies_i,
                    mode_selection=mode_selection,
                    label=current_label,
                    solver=solver,
                    n_runs=n_runs,
                    n_md=n_md_i,
                )

                print_average_timing_report(results)

                # Use average integration elapsed time only
                SOAElapsed.append(results["integration_elapsed"])

            mode_timing_results[n_bodies_i] = SOAElapsed

        # -------------------------------------------------------------
        # Plot computation time vs number of selected modes
        # -------------------------------------------------------------
        plt.figure()

        for n_bodies_i, elapsed_values in mode_timing_results.items():
            plt.plot(
                n_mode_values,
                elapsed_values,
                "o-",
                label=f"{n_bodies_i} bodies",
            )

        plt.xlabel("Number of selected modes")
        plt.ylabel("Average computation time [s]")
        plt.title("SOA Computation Time vs Number of Modes")
        plt.grid(True)
        plt.legend(title="Number of bodies")
        plt.tight_layout()
        plt.show()

    elif run_mode == "mode_timing_all":

        label = mode_timing_all_label
        mode_selection = mode_timing_all_mode_selection

        mode_timing_all_results = {}

        print("\n" + "=" * 60)
        print("Running mode-number benchmark with no mode selection")
        print("Mode selection: None")
        print("Using first n_md lowest-frequency modes overall")
        print("=" * 60)

        for n_bodies_i in mode_timing_all_body_values:

            SOAElapsed = []

            print("\n" + "=" * 60)
            print(f"Running benchmark for {n_bodies_i} bodies")
            print("=" * 60)

            for n_md_i in n_mode_values_all:

                current_label = f"First {n_md_i} modes"

                results = average_soa_simulation_time(
                    n_bodies=n_bodies_i,
                    mode_selection=mode_selection,
                    label=current_label,
                    solver=solver,
                    n_runs=n_runs,
                    n_md=n_md_i,
                )

                print_average_timing_report(results)

                # Use average integration elapsed time only
                SOAElapsed.append(results["integration_elapsed"])

            mode_timing_all_results[n_bodies_i] = SOAElapsed

        # -------------------------------------------------------------
        # Plot computation time vs number of selected modes
        # -------------------------------------------------------------
        plt.figure()

        for n_bodies_i, elapsed_values in mode_timing_all_results.items():
            plt.plot(
                n_mode_values_all,
                elapsed_values,
                "o-",
                label=f"{n_bodies_i} bodies",
            )

        plt.xlabel("Number of selected modes")
        plt.ylabel("Average computation time [s]")
        plt.title("SOA Computation Time vs Number of Modes")
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
            "run_mode must be either 'timing', 'mode_timing', "
            "'mode_timing_all', or 'simulation'"
        )
