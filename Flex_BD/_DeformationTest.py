from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
from SOALIB import soalib as sb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Remember to turn on constant external force

# Parameters
klOO = np.array([3, 0, 0]).reshape(3, 1)
H_type = "fixed"
w, h, rho = 0.1, 0.1, 7850
j1 = Joint(klOO, H_type)
r1 = Rigid_Properties(rho, w, h)
E, G, c = 230e9, 80e9, 0.2

n = 5

# Deformation list
u = []
u_e = []
node_list = []

F_ext1 = np.array([0, 0, 0, 0, 0, -1e4]).reshape(6, 1)

# Timoshenko
L = np.linalg.norm(klOO)
A = w * h
I = w * h**3 / 12
k = 5/6
F = F_ext1[5, 0]

u_T = F*L**3/(3*E*I) + F*L/(k*G*A)
print(u_T)

n_nd = 2
for i in range(1, n+1, 1):
    n_nd = 2**i

    max_modes = (n_nd - 1) * 6

    # Body setup
    f1 = Flex_Properties(E, G, c, n_nd, max_modes) #, mode_selection={"bending_xy": 10, "bending_xz": 10})
    b1 = SOABody(j1, r1, f1)

    # Force
    b1.set_F_ext(F_ext1)

    bodies = [b1]
    system = MultibodySystem(bodies)
    system.set_gravity(False)

    # Time setup
    tf = 2
    dt = 0.01

    # Simulation
    print(f"Nodes: {n_nd}")
    sim = Simulation(system, tf, dt)
    #sim.set_tol(1e-8, 1e-10)
    sim.IntegrateSystem("Radau")

    # Find deformation
    eta = sim.data.state[-1].Eta[0]
    body = sim.system.bodies[0]
    PI = body.flex.PI[-3:, :]

    u_i = PI @ eta
    u_z = u_i[2, 0]
    u_error = u_z - u_T
    u.append(u_z)
    u_e.append(u_error)
    node_list.append(n_nd)

plt.figure()

plt.plot(node_list, u, 'o-', label="SOA solution")
plt.axhline(u_T, linestyle='--', label="Timoshenko Beam")
#plt.plot(node_list, u_e, 'o-', label="Deformation error")

plt.xlabel("Number of nodes, n_{nd}")
plt.ylabel("Tip deformation [mm]")
#plt.ylim([-10*max(u_e), 10*max(u_e)])
plt.title("Convergence study of deformation")

# Integer ticks on x-axis
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# Scientific notation on y-axis
#plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

print(u_T)

plt.legend()
plt.grid()

plt.show()
