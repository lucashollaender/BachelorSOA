from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -------------------------------------------------
# Files
# -------------------------------------------------
file_1 = Path("fft_response.csv")
file_2 = Path("fft_response0axial.csv")

label_1 = "With axial load"
label_2 = "Without axial load"

# -------------------------------------------------
# Settings
# -------------------------------------------------
f_min_plot = 0
f_max_plot = 100
amp_floor = 1e-6

# -------------------------------------------------
# Load data
# -------------------------------------------------
df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)

# -------------------------------------------------
# Global normalization
# -------------------------------------------------
global_max = max(df1["amplitude"].max(), df2["amplitude"].max())

df1["amplitude_global_norm"] = df1["amplitude"] / global_max
df2["amplitude_global_norm"] = df2["amplitude"] / global_max


# -------------------------------------------------
# Helper function for peak detection
# -------------------------------------------------
def get_main_peaks(df, amp_col, f_min=1, f_max=300, min_height=1e-7, distance=10):
    mask = (df["frequency_hz"] >= f_min) & (df["frequency_hz"] <= f_max)

    f = df.loc[mask, "frequency_hz"].to_numpy()
    a = df.loc[mask, amp_col].to_numpy()

    peaks, _ = find_peaks(
        a,
        height=min_height,
        distance=distance
    )

    return f[peaks], a[peaks]

# -------------------------------------------------
# Plot: globally normalized amplitude
# -------------------------------------------------
plt.figure(figsize=(10, 5.5))

plt.plot(
    df1["frequency_hz"],
    df1["amplitude_global_norm"],
    linewidth=2.0,
    label=label_1
)

plt.plot(
    df2["frequency_hz"],
    df2["amplitude_global_norm"],
    linewidth=2.0,
    linestyle="--",
    label=label_2
)

for df in [df1, df2]:
    f_peaks, a_peaks = get_main_peaks(
        df,
        amp_col="amplitude_global_norm",
        min_height=1e-5,
        distance=10
    )

    for f, a in zip(f_peaks, a_peaks):
        plt.plot(f, a, "o", markersize=4)
        plt.text(
            f,
            a * 1.05,
            f"{f:.1f} Hz",
            rotation=90,
            fontsize=8,
            ha="center",
            va="bottom"
        )

plt.xlim(f_min_plot, f_max_plot)
plt.ylim(bottom=0)

plt.xlabel("Frequency [Hz]", fontsize=12)
plt.ylabel("Globally normalized amplitude [-]", fontsize=12)
plt.title("FFT comparison of flexible-body response", fontsize=14)

plt.grid(True, linestyle=":", linewidth=0.8)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig("fft_comparison_global_normalized_amplitude.png", dpi=300)
plt.show()

# -------------------------------------------------
# Plot: globally normalized amplitude, semilog y-axis
# -------------------------------------------------

# Apply floor to avoid log(0)
df1["amplitude_global_norm_log"] = np.maximum(
    df1["amplitude_global_norm"],
    amp_floor
)

df2["amplitude_global_norm_log"] = np.maximum(
    df2["amplitude_global_norm"],
    amp_floor
)

plt.figure(figsize=(10, 5.5))

plt.semilogy(
    df1["frequency_hz"],
    df1["amplitude_global_norm_log"],
    linewidth=2.0,
    label=label_1
)

plt.semilogy(
    df2["frequency_hz"],
    df2["amplitude_global_norm_log"],
    linewidth=2.0,
    linestyle="--",
    label=label_2
)

for df in [df1, df2]:
    f_peaks, a_peaks = get_main_peaks(
        df,
        amp_col="amplitude_global_norm",
        min_height=1e-5,
        distance=10
    )

    for f, a in zip(f_peaks, a_peaks):
        a_plot = max(a, amp_floor)

        plt.plot(f, a_plot, "o", markersize=4)
        plt.text(
            f,
            a_plot * 1.3,
            f"{f:.1f} Hz",
            rotation=90,
            fontsize=8,
            ha="center",
            va="bottom"
        )

plt.xlim(f_min_plot, f_max_plot)
plt.ylim(amp_floor, 1.2)

plt.xlabel("Frequency [Hz]", fontsize=12)
plt.ylabel("Globally normalized amplitude [-]", fontsize=12)
plt.title("FFT comparison of flexible-body response, semilog scale", fontsize=14)

plt.grid(True, which="both", linestyle=":", linewidth=0.8)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig("fft_comparison_global_normalized_amplitude_semilog.png", dpi=300)
plt.show()