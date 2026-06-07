from  RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation
import numpy as np
from SOALIB import soalib as sb

"""
This TEST simulates a rigid 3-body pendulum with spherical joints 
under the influence of gravity.
"""

# ----- Body Parameters -----
klOO1 = np.array([1, 0, 0])
klOO2 = np.array([0, 1, 0])
klOO3 = np.array([0, -1, 0])
H_type = "spherical"
m = 2
CkJk = np.array([0.00053, 0.16693, 0.16693])
klOC = np.array([0.5, 0, 0])
i = Inertia(m, CkJk, klOC)

# ----- Setup Bodies -----
# Body 1
j1 = Joint(klOO1, H_type)
b1 = SOABody(j1, i)

#Body 2
j2 = Joint(klOO2, H_type)
b2 = SOABody(j2, i)

# Body 3
j3 = Joint(klOO3, H_type)
b3 = SOABody(j3, i)

# ----- Setup Serial-Chain -----
bodies = [b1, b2, b3]
system = MultibodySystem(bodies)
tf = 10
dt = 0.01
sim = Simulation(system, tf, dt)

# ----- Setup Animation -----
sim.camera_speed(0.5)
sim.camera_hor(90)
sim.camera_ver(0)
sim.IntegrateSystem()
sim.animate()