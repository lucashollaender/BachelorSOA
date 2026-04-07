#SOA
from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation
from SOALIB.ticToc import TicToc
import numpy as np
from SOALIB import soalib as sb
import matplotlib.pyplot as plt

#Cartesian
from markus.Cart_pend_functions import run_single_case, run_benchmark
import pandas as pd

N, solvetime =run_benchmark( N_values= list(range(130, 160, 4)), L=1.0, m=1.0, g=9.82, T_end=5, n_runs=2, theta_init_fn=lambda N: np.zeros(N) )

#SOA

# Test setup
n = list(range(130, 160, 4))
n_samples = 2 # <-- number of runs per n

# Simulation setups
tf = 5
dt = 0.01

# Body setup
klOO = np.array([0, 0, 1])
H_type = "revx"
m = 1
CkJk = np.array([1/12, 1/12, 1/12])
klOC = np.array([0, 0, 0.5])

dt_n = np.zeros(len(n))

for z, k in enumerate(n):
    sample_times = []

    for s in range(n_samples):
        bodies = []  # <-- reset bodies EACH sample!

        for i_body in range(k + 1):
            j = Joint(klOO, H_type)
            i = Inertia(m, CkJk, klOC)
            b = SOABody(j, i)

            if i_body == 0:
                b.set_initial_theta0(90/180*np.pi)

            bodies.append(b)

        bodies.reverse()
        system = MultibodySystem(bodies)
        sim = Simulation(system, tf, dt)

        timer = TicToc(False)
        timer.tic()
        sim.IntegrateSystem()
        sample_times.append(timer.toc())

    # Average over samples
    dt_n[z] = np.mean(sample_times)

    print(f"Integration of n={k}: Done! (avg over {n_samples} runs)")


plt.figure()
plt.plot(n, dt_n, marker='o')
plt.xlabel("Number of bodies, n")
plt.ylabel("Average computation time [s]")
plt.title("Average Computation Time vs Number of Bodies")
plt.grid(True)

plt.xlim(1, max(n))
plt.ylim(0, max(dt_n)*1.1)

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(N, solvetime, marker='o', linewidth=2.5, markersize=7, label='Cartesian formulation')
plt.plot(n, dt_n, marker='s', linewidth=2.5, markersize=7, label='SOA formulation')

plt.xlabel("Number of bodies / pendulums, N", fontsize=11)
plt.ylabel("Average computation time [s]", fontsize=11)
plt.title("Computation time comparison", fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(N)
plt.show()

# Save summary data
df = pd.DataFrame({
    "N": N,
    "cartesian_time_s": solvetime,
    "soa_time_s": dt_n
})

df.to_csv("benchmark_results130150.csv", index=False)
print("Saved benchmark_results130150.csv")