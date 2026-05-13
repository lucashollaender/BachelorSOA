import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
import numpy as np

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
f_max_plot = 300
amp_floor = 1e-11

# -------------------------------------------------
# Load data
# -------------------------------------------------
df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)

required_cols = {"frequency_hz", "amplitude", "amplitude_normalized"}
for file, df in [(file_1, df1), (file_2, df2)]:
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{file} is missing columns: {missing}")

# Clip normalized amplitude for semilog plotting
df1["amplitude_normalized_log"] = np.maximum(
    df1["amplitude_normalized"].to_numpy(),
    amp_floor
)
df2["amplitude_normalized_log"] = np.maximum(
    df2["amplitude_normalized"].to_numpy(),
    amp_floor
)

# -------------------------------------------------
# Helper function for peak detection
# -------------------------------------------------
def get_main_peaks(df, amp_col, f_min=1, f_max=300, min_height=1e-6, distance=20):
    mask = (df["frequency_hz"] >= f_min) & (df["frequency_hz"] <= f_max)
    f = df.loc[mask, "frequency_hz"].to_numpy()
    a = df.loc[mask, amp_col].to_numpy()

    peaks, _ = find_peaks(a, height=min_height, distance=distance)
    return f[peaks], a[peaks]

# -------------------------------------------------
# Semilog plot: normalized amplitude
# -------------------------------------------------
plt.figure(figsize=(10, 5.5))

plt.semilogy(
    df1["frequency_hz"],
    df1["amplitude_normalized_log"],
    linewidth=2.0,
    label=label_1
)

plt.semilogy(
    df2["frequency_hz"],
    df2["amplitude_normalized_log"],
    linewidth=2.0,
    linestyle="--",
    label=label_2
)

for df in [df1, df2]:
    f_peaks, a_peaks = get_main_peaks(
        df,
        amp_col="amplitude_normalized",
        min_height=1e-6
    )

    for f, a in zip(f_peaks, a_peaks):
        if a >= amp_floor:
            plt.plot(f, max(a, amp_floor), "o", markersize=4)
            plt.text(
                f,
                max(a * 1.4, amp_floor * 1.4),
                f"{f:.1f} Hz",
                rotation=90,
                fontsize=8,
                ha="center",
                va="bottom"
            )

plt.xlim(f_min_plot, f_max_plot)
plt.ylim(amp_floor, 1.2)

plt.xlabel("Frequency [Hz]", fontsize=12)
plt.ylabel("Normalized amplitude [-]", fontsize=12)
plt.title("FFT comparison of flexible-body response", fontsize=14)

plt.grid(True, which="both", linestyle=":", linewidth=0.8)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig("fft_comparison_semilog_normalized.png", dpi=300)

plt.show()

# -------------------------------------------------
# Linear plot: raw amplitude
# -------------------------------------------------
plt.figure(figsize=(10, 5.5))

plt.plot(
    df1["frequency_hz"],
    df1["amplitude"],
    linewidth=2.0,
    label=label_1
)

plt.plot(
    df2["frequency_hz"],
    df2["amplitude"],
    linewidth=2.0,
    linestyle="--",
    label=label_2
)

for df in [df1, df2]:
    f_peaks, a_peaks = get_main_peaks(
        df,
        amp_col="amplitude",
        min_height=0.0001 * df["amplitude"].max()
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
plt.ylabel("Amplitude [-]", fontsize=12)
plt.title("FFT comparison of flexible-body response", fontsize=14)

plt.grid(True, linestyle=":", linewidth=0.8)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig("fft_comparison_linear_amplitude.png", dpi=300)

plt.show()