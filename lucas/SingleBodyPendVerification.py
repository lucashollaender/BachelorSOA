#SOA
from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation
from SOALIB.ticToc import TicToc
import numpy as np
from SOALIB import soalib as sb
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
#SOA

# Test setup
n = 1

# Simulation setups
tf = 10
dt = 0.001

# Body setup
klOO = np.array([0, 0, -1])
H_type = "revx"
m = 1
CkJk = np.array([1/12, 1/12, 1/12])
klOC = np.array([0, 0, -0.5])

j = Joint(klOO, H_type)
i = Inertia(m, CkJk, klOC)
b = SOABody(j, i)
b.set_initial_theta0(np.pi/2)
bodies = [b]
system = MultibodySystem(bodies)
sim = Simulation(system, tf, dt)
sim.IntegrateSystem()
sim.animate()
state = sim.get_state()

theta = np.array([s.Theta[0][0, 0] for s in state])
print(pd.DataFrame(theta))
time=sim.data.time

# Parameters
m = 1
l = 1
I_O = m * l**2 / 3
g = 9.82
d = l / 2

# ODE system:
# y[0] = theta
# y[1] = theta_dot
def pendulum_ode(t, y):
    theta, theta_dot = y
    theta_ddot = -(m * g * d / I_O) * np.sin(theta)
    return [theta_dot, theta_ddot]

# Time span
t0 = 0
tf = 10
dt = 0.001
t_eval = np.arange(t0, tf + dt, dt)

# Initial conditions
theta0 = np.pi / 2      # 90 degrees
theta_dot0 = 0.0
y0 = [theta0, theta_dot0]

# Solve
sol = solve_ivp(pendulum_ode, [t0, tf], y0, t_eval=t_eval)

# Extract solution
tt = sol.t
Theta = sol.y[0]

# Plot theta
plt.plot(time,theta)
plt.plot(tt, Theta,'--' ,label='theta(t)')
plt.xlabel('Time [s]')
plt.ylabel('Theta [rad]')
plt.title('Pendulum angle vs time')
plt.grid(True)
plt.show()

print(pd.DataFrame(theta))
print(pd.DataFrame(Theta))
print(pd.DataFrame(tt))
print(pd.DataFrame(time))
plt.plot(time, np.abs(Theta)-np.abs(theta), label='theta(t)')
plt.show()