from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation
import numpy as np
from SOALIB import soalib as sb

klOO = np.array([1, 0, 0])
H_type1 = "free"
H_type2 = "free"

m = 2
CkJk = np.array([0.00053, 0.16693, 0.16693])
klOC = np.array([0.5, 0, 0])

j1 = Joint(klOO, H_type1)
i = Inertia(m, CkJk, klOC)
b1 = SOABody(j1, i)

j2 = Joint(klOO, H_type2)
i = Inertia(m, CkJk, klOC)
b2 = SOABody(j2, i)
"""
j3 = Joint(klOO, H_type3)
i = Inertia(m, CkJk, klOC)
b3 = SOABody(j3, i)
"""

b1.set_initial_theta0(0)
# b2.set_initial_theta0(90/180*np.pi)

# b1.set_initial_beta0(np.array([0, 0 , 0]).reshape(3, 1))

bodies = [b1, b2]

system = MultibodySystem(bodies)

tf = 10
dt = 0.01

sim = Simulation(system, tf, dt)

sim.camera_speed(0)
sim.camera_hor(90)
sim.camera_ver(0)

sim.IntegrateSystem()

save_dir = r"C:\Users\jepp6\OneDrive - Aarhus universitet\Dokumenter\Noter\6. Semester\Bachelor Projekt\BachelorCode\Renders"

sim.animate()
