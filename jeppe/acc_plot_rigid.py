from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def read_adams_tab(filename):
    start_idx = None

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                float(parts[0])
                float(parts[1])
                start_idx = i
                break
            except ValueError:
                pass

    if start_idx is None:
        raise ValueError(f"Could not find numeric data in {filename}")

    df = pd.read_csv(
        filename,
        sep=r"\s+",
        skiprows=start_idx,
        header=None,
        names=["TIME", "Q"],
        engine="python"
    )
    return df


# -----------------------------
# Build rigid pendulum model
# -----------------------------
klOO = np.array([0, 0, 1])
H_type1 = "revx"
H_type2 = "revx"

m = 1
CkJk = np.array([1/12, 1/12, 1/12])
klOC = np.array([0, 0, 0.5])

j1 = Joint(klOO, H_type1)
i1 = Inertia(m, CkJk, klOC)
b1 = SOABody(j1, i1)

j2 = Joint(klOO, H_type2)
i2 = Inertia(m, CkJk, klOC)
b2 = SOABody(j2, i2)

b1.set_initial_theta0(0)
b2.set_initial_theta0(90/180*np.pi)

bodies = [b1, b2]
system = MultibodySystem(bodies)

tf = 10
dt = 0.01

sim = Simulation(system, tf, dt)
sim.camera_speed(0)
sim.IntegrateSystem()

# -----------------------------
# Extract rigid joint accelerations gamma
# -----------------------------
t_rigid = sim.data.time
states = sim.get_state()

gamma_body1 = []
gamma_body2 = []

for state in states:
    X, V, a, b = system.ATBI.scatter_kinematics(state)
    G, nu = system.ATBI.gather_ATBI(a, b, X)
    gamma, alpha = system.ATBI.scatter_ATBI(a, X, G, nu)

    gamma_body1.append(float(gamma[0].flatten()[0]))
    gamma_body2.append(float(gamma[1].flatten()[0]))

gamma_body1 = np.array(gamma_body1)
gamma_body2 = np.array(gamma_body2)

# -----------------------------
# Read Adams data
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

adams1 = read_adams_tab(os.path.join(script_dir, "PendulumAdamsHinge1.tab"))
adams2 = read_adams_tab(os.path.join(script_dir, "PendulumAdamsHinge2.tab"))

# -----------------------------
# Plot both in one figure
# -----------------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].plot(t_rigid, gamma_body1, label="Rigid joint acceleration body 1")
axs[0].plot(adams1["TIME"], adams1["Q"], "--", label="AdamsHinge1")
axs[0].set_ylabel("Joint acceleration")
axs[0].set_title("AdamsHinge1 vs joint acceleration of body 1")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(t_rigid, gamma_body2, label="Rigid joint acceleration body 2")
axs[1].plot(adams2["TIME"], adams2["Q"], "--", label="AdamsHinge2")
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Joint acceleration")
axs[1].set_title("AdamsHinge2 vs joint acceleration of body 2")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
