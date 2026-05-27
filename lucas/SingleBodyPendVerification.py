#SOA
from SOALIB.RigidForwardSOA import Joint, Inertia, SOABody, MultibodySystem, Simulation
from SOALIB.ticToc import TicToc
import numpy as np
from SOALIB import soalib as sb
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

#SOA

# Test setup
n = 1

# Simulation setups
tf = 30
dt = 0.01

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
g = 9.81
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
tf = 30
dt = 0.01
t_eval = np.arange(t0, tf + dt, dt)

# Initial conditions
theta0 = np.pi / 2      # 90 degrees
theta_dot0 = 0.0
y0 = [theta0, theta_dot0]

# Solve
sol = solve_ivp(
    pendulum_ode,
    [t0, tf],
    y0,
    t_eval=t_eval,
    method='Radau'
)

# Extract solution
tt = sol.t
Theta = sol.y[0]


fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(time, theta, linewidth=2.5, label='SOA formulation')
ax.plot(tt, Theta, '--', linewidth=2.5, label='Compound pendulum')

ax.set_xlabel('Time [s]', fontsize=12)
ax.set_ylabel(r'Angle $\theta$ [rad]', fontsize=12)
ax.set_title('Pendulum Angle Comparison', fontsize=14, fontweight='bold', pad=10)

ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', frameon=True, fontsize=11)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis='both', labelsize=11)
fig.tight_layout()

plt.show()

print(pd.DataFrame(theta))
print(pd.DataFrame(Theta))
print(pd.DataFrame(tt))
print(pd.DataFrame(time))

error = Theta - theta
abs_error = np.abs(error)

plt.figure(figsize=(10, 5))
plt.plot(time, abs_error, linewidth=2, label='|Θ_compound(t) - θ_soa(t)|')

plt.title('Absolute Error Between SOA Pendulum and Compound Pendulum', fontsize=14)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Absolute Error [rad]', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

def get_period_from_peaks(t, x):
    t = np.asarray(t).ravel()
    x = np.asarray(x).ravel()

    # ignore tiny wiggles
    prominence = 0.05 * np.ptp(x)

    peaks, _ = find_peaks(x, prominence=prominence)

    # include t=0 if the signal starts at a peak
    if len(x) > 1 and x[0] > x[1]:
        peaks = np.r_[0, peaks]

    peak_times = t[peaks]

    if len(peak_times) < 2:
        return np.nan, peak_times, peaks

    period = np.mean(np.diff(peak_times))
    return period, peak_times, peaks


T_theta, peak_times_theta, peaks_theta = get_period_from_peaks(time, theta)
T_Theta, peak_times_Theta, peaks_Theta = get_period_from_peaks(tt, Theta)

print("Period of theta  =", T_theta)
print("Peak times theta =", peak_times_theta)

print("Period of Theta  =", T_Theta)
print("Peak times Theta =", peak_times_Theta)