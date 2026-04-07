import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
# --- Load and combine data ---
df1 = pd.read_csv("benchmark_results.csv")
df2 = pd.read_csv("benchmark_results130150.csv")

df = pd.concat([df1, df2], ignore_index=True)

# Optional: remove duplicate N rows if both files overlap
# Keeps the last occurrence
df = df.drop_duplicates(subset="N", keep="last")

# Sort so plots look correct
df = df.sort_values("N").reset_index(drop=True)

N = df["N"].to_numpy()
cart = df["cartesian_time_s"].to_numpy()
soa = df["soa_time_s"].to_numpy()
#cart=cart[-100:]
#soa=soa[-100:]

# --- Fit models ---
# SOA: linear fit  a*N + b
soa_coef = np.polyfit(N, soa, 1)
soa_fit = np.polyval(soa_coef, N)

# Cartesian: cubic fit  a*N^3 + b*N^2 + c*N + d
cart_coef = np.polyfit(N, cart, 3)
cart_fit = np.polyval(cart_coef, N)

# --- R^2 function ---
def r2_score(y, yfit):
    ss_res = np.sum((y - yfit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res/ss_tot

soa_r2 = r2_score(soa, soa_fit)
cart_r2 = r2_score(cart, cart_fit)

# --- Print equations ---
print("SOA linear fit:")
print(f"t(N) = {soa_coef[0]:.6e} * N + {soa_coef[1]:.6e}")
print(f"R^2 = {soa_r2:.6f}")

print("\nCartesian cubic fit:")

print(
    f"t(N) = {cart_coef[0]:.6e} * N^3 + "
    f"{cart_coef[1]:.6e} * N^2 + "
    f"{cart_coef[2]:.6e} * N + "
    f"{cart_coef[3]:.6e}")
print(f"R^2 = {cart_r2:.6f}")

# --- Smooth curves for plotting ---
N_smooth = np.linspace(N.min(), N.max(), 300)
soa_smooth = np.polyval(soa_coef, N_smooth)
cart_smooth = np.polyval(cart_coef, N_smooth)

# --- Plot ---
plt.figure(figsize=(9, 6))

plt.plot(N, soa, 'o', label='SOA data')
#plt.plot(N_smooth, soa_smooth, '-', label=f'SOA linear fit (R²={soa_r2:.4f})')

plt.plot(N, cart, 's', label='Cartesian data')
#plt.plot(N_smooth, cart_smooth, '--', label=f'Cartesian cubic fit (R²={cart_r2:.4f})')

plt.xlabel("N")
plt.ylabel("Computation time [s]")
plt.title("Computation time SOA vs 2D-Cartesian")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Power-law fit: t = c * N^p
def power_law_fit(x, y):
    logx = np.log(x)
    logy = np.log(y)
    p, logc = np.polyfit(logx, logy, 1)
    c = np.exp(logc)
    yfit = c * x**p
    r2 = r2_score(y, yfit)
    return c, p, yfit, r2

c_soa, p_soa, soa_power_fit, soa_power_r2 = power_law_fit(N, soa)
c_cart, p_cart, cart_power_fit, cart_power_r2 = power_law_fit(N, cart)

print(f"SOA power fit: t = {c_soa:.6e} * N^{p_soa:.4f}, R^2={soa_power_r2:.6f}")
print(f"Cartesian power fit: t = {c_cart:.6e} * N^{p_cart:.4f}, R^2={cart_power_r2:.6f}")