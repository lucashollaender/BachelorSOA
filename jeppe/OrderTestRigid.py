from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation
from SOALIB.ticToc import TicToc
import numpy as np
from SOALIB import soalib as sb
import matplotlib.pyplot as plt

# Test setup
n_max = 10

# Simulation setup
tf = 10
dt = 0.01

# Body setup
klOO = np.array([0, 0, 5])
H_type = "spherical"
m = 1
CkJk = np.array([1, 1, 0.1])
klOC = np.array([0, 0, 2.5])

bodies = []
dt_n = np.zeros((n_max,))

for k in range(n_max):
    # Add body to tip
    j = Joint(klOO, H_type)
    i = Inertia(m, CkJk, klOC)
    b = SOABody(j, i)
    b.set_initial_theta0(sb.get_quat_from_degrees(-37, -25, 0))
    bodies.append(b)
    system = MultibodySystem(bodies)
    sim = Simulation(system, tf, dt)

    # Timer
    timer = TicToc(False)
    timer.tic()
    sim.IntegrateSystem()
    dt_n[k] = timer.toc()

    print(f"Integration of n={k+1}: Done!")

# Plotting
# Number of bodies (1..n_max)
n_values = np.arange(1, n_max + 1)

plt.figure()
plt.plot(n_values, dt_n, marker='o')
plt.xlabel("Number of bodies, n")
plt.ylabel("Computation time [s]")
plt.title("Computation Time vs Number of Bodies")
plt.grid(True)

plt.xlim(0, n_max)
plt.ylim(0, max(dt_n)*1.1)

plt.show()