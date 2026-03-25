from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation
from SOALIB.ticToc import TicToc
import numpy as np
from SOALIB import soalib as sb
import matplotlib.pyplot as plt

# Test setup
n_max = 12
n_samples = 1  # <-- number of runs per n

# Simulation setup
tf = 10
dt = 0.01

# Body setup
klOO = np.array([0, 0, 1])
H_type = "revx"
m = 1
CkJk = np.array([1/12, 1/12, 1/12])
klOC = np.array([0, 0, 0.5])

dt_n = np.zeros((n_max,))

for k in range(n_max):
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

        if k == 2:
            sim_save = sim

        timer = TicToc(False)
        timer.tic()
        sim.IntegrateSystem()
        sample_times.append(timer.toc())

    # Average over samples
    dt_n[k] = np.mean(sample_times)

    print(f"Integration of n={k+1}: Done! (avg over {n_samples} runs)")

# Plotting
n_values = np.arange(1, n_max + 1)

plt.figure()
plt.plot(n_values, dt_n, marker='o')
plt.xlabel("Number of bodies, n")
plt.ylabel("Average computation time [s]")
plt.title("Average Computation Time vs Number of Bodies")
plt.grid(True)

plt.xlim(0, n_max)
plt.ylim(0, max(dt_n)*1.1)

plt.show()

sim_save.IntegrateSystem()
sim_save.animate()