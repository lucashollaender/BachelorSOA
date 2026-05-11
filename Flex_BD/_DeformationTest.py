from SOABody import SOABody
from MultibodySystem import MultibodySystem
from Simulation import Simulation
from Body_Properties import Joint, Rigid_Properties, Flex_Properties
import numpy as np
from SOALIB import soalib as sb
import pandas as pd
import matplotlib.pyplot as plt

# Remember to turn on constant external force

# Parameters
L = 5
H_type = "fixed"
klOC = np.array([2.5, 0, 0])
w, h, rho = 0.1, 0.1, 7850
j1 = Joint(L, H_type)
r1 = Rigid_Properties(rho, w, h)
E, G, c = 230e9, 80e9, 0.2

n = 10
n_md = 6

# Deformation list
u = [None]
node_list = [None]

for i in range(3, n+1, 1):
    n_nd = i

    # Body setup
    f1 = Flex_Properties(E, G, c, n_nd, n_md)
    b1 = SOABody(j1, r1, f1)

    # Force
    F_ext1 = np.array([0, 0, 0, 0, 0, 1e3]).reshape(6, 1)
    b1.set_F_ext(F_ext1)

    bodies = [b1]
    system = MultibodySystem(bodies)

    # Time setup
    tf = 2
    dt = 0.01

    # Simulation
    sim = Simulation(system, tf, dt)
    sim.IntegrateSystem("Radau")

    print(f"Nodes: {i}")

    # Find deformation
    eta = sim.data.state[-1].Eta[0]
    body = sim.system.bodies[0]
    PI = body.flex.PI[-3:, :]

    u_i = PI @ eta
    u_z = u_i[2, 0]
    u.append(u_z)
    node_list.append(i)

# Euler-Bernoulli deformation
I_x = w * h**3 / 12
u_EB = F_ext1[5, 0] * L**3 / (3 * E * I_x)

# Timoshenko
A = w * h
I = w * h**3 / 12
k = 5/6
F = F_ext1[5, 0]

u_T = F*L**3/(3*E*I) + F*L/(k*G*A)

plt.figure()

plt.plot(node_list, u, 'o-', label="SOA solution")
# plt.axhline(u_EB, linestyle='--', label="Euler-Bernoulli")
plt.axhline(u[-1], linestyle='--', label="Timoshenko")

plt.xlabel("Number of nodes")
plt.ylabel("Tip deformation (z)")
plt.title("Convergence study of deformation")
plt.legend()
plt.grid()

plt.show()
