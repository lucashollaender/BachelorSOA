from pathlib import Path

import numpy as np
import pandas as pd

from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties


def make_link(
    length=1.0,
    joint_type="revy",
    rho=2700.0,
    w=0.02,
    h=0.02,
    E=9.0e8,
    G=2.6e7,
    c=0.1,
    n_nd=8,
    n_md=3,
):
    """Create one flexible SOA beam link initially along the local x-axis."""
    klOO = np.array([length, 0.0, 0.0]).reshape(3, 1)

    joint = Joint(klOO, joint_type)
    rigid = Rigid_Properties(rho, w, h)
    flex = Flex_Properties(E, G, c, n_nd, n_md)

    body = SOABody(joint, rigid, flex)

    # Zero applied joint torque.
    tau = np.zeros((joint.beta_size(), 1))
    body.set_tau(tau)

    return body


def write_node_csv(sim: Simulation, filename: str | Path) -> pd.DataFrame:
    """
    Export nodal positions from a completed Simulation to a long-format CSV.

    Uses sim.nNodalPos(), which returns data ordered from base-to-tip because it
    loops internally from body n-1 down to body 0.
    """
    filename = Path(filename)
    nodal_pos = sim.nNodalPos()
    times = np.asarray(sim.data.time)

    rows = []
    n_bodies = len(sim.system.bodies)

    for frame, (t, bodies_at_t) in enumerate(zip(times, nodal_pos)):
        for chain_link, nodes in enumerate(bodies_at_t):
            # Internal body index corresponding to this base-to-tip chain link.
            body_index = n_bodies - 1 - chain_link

            for node_id, p in enumerate(nodes):
                p = np.asarray(p).reshape(3,)
                rows.append(
                    {
                        "time": float(t),
                        "frame": int(frame),
                        "link": int(chain_link),          # 0 = base link, 1 = tip link
                        "body_index": int(body_index),    # internal index in sim.system.bodies
                        "node": int(node_id),
                        "x": float(p[0]),
                        "y": float(p[1]),
                        "z": float(p[2]),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    return df


def main():
    # -----------------------------
    # Build 2-link pendulum
    # -----------------------------
    base_link = make_link(length=1.0, joint_type="revy")
    tip_link = make_link(length=1.0, joint_type="revy")

    # Optional initial angles. Both zero means both links start horizontal.
    # Gravity then creates a torque about the y-axis.
    base_link.set_initial_theta0(np.array([0.0]).reshape(1, 1))
    tip_link.set_initial_theta0(np.array([0.0]).reshape(1, 1))

    # Optional initial angular velocities.
    base_link.set_initial_beta0(np.array([0.0]).reshape(1, 1))
    tip_link.set_initial_beta0(np.array([0.0]).reshape(1, 1))

    # IMPORTANT: in this implementation, the last body is the base body.
    bodies = [tip_link, base_link]

    system = MultibodySystem(bodies)
    system.set_gravity(True)

    # -----------------------------
    # Time setup
    # -----------------------------
    tf = 5.0
    dt = 0.001
    ani_dt = 0.01      # CSV output interval

    sim = Simulation(system, tf, dt)
    sim.set_ani_dt(ani_dt)

    # Use same style as your C++ test. If RK4 becomes unstable, try "BE" or "Radau".
    sim.IntegrateSystem("RK4")

    # -----------------------------
    # Export node locations
    # -----------------------------
    out_file = Path("simulation_nodes_2_links_python.csv")
    df = write_node_csv(sim, out_file)

    print(f"Wrote {out_file.resolve()}")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()
