from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -------------------------------------------------
# Files
# -------------------------------------------------
input_file = Path("fft_nodal_dofs.csv")

# -------------------------------------------------
# Settings
# -------------------------------------------------
f_max_plot = 350
f_min_peak = 0.5
f_max_peak = f_max_plot

peak_min_relative_height = 0.01
peak_min_relative_prominence = 0.01
peak_min_distance_hz = 1.2
n_peaks_to_mark = 5

# Optional theoretical eigenfrequencies
# Leave empty if you only want detected FFT peaks
theoretical_freqs = []
# theoretical_freqs = [10.2, 63.8, 178.5]  # example

# -------------------------------------------------
# Load FFT data
# -------------------------------------------------
fft_data = pd.read_csv(input_file)
freq = fft_data["frequency_hz"].to_numpy()

# -------------------------------------------------
# Peak detection helper
# -------------------------------------------------
def detect_peaks(signal_name):
    amp = fft_data[f"{signal_name}_amp"].to_numpy()

    valid = (freq >= f_min_peak) & (freq <= f_max_peak)
    amp_valid = amp[valid]

    df = freq[1] - freq[0]
    peak_min_distance_samples = max(
        1,
        int(np.ceil(peak_min_distance_hz / df))
    )

    if np.max(amp_valid) > 0:
        min_height = peak_min_relative_height * np.max(amp_valid)
        min_prominence = peak_min_relative_prominence * np.max(amp_valid)
    else:
        min_height = 0
        min_prominence = 0

    peaks, _ = find_peaks(
        amp_valid,
        height=min_height,
        prominence=min_prominence,
        distance=peak_min_distance_samples
    )

    valid_indices = np.where(valid)[0]
    peak_indices = valid_indices[peaks]

    peak_indices = sorted(
        peak_indices,
        key=lambda i: amp[i],
        reverse=True
    )

    peak_indices = peak_indices[:n_peaks_to_mark]

    peak_rows = []

    for i in peak_indices:
        peak_rows.append({
            "signal": signal_name,
            "frequency_hz": freq[i],
            "amplitude": amp[i]
        })

    return peak_rows

# -------------------------------------------------
# Detect peaks
# -------------------------------------------------
signals = [
    "u_x",
    "u_y",
    "u_z",
    "theta_x",
    "theta_y",
    "theta_z"
]

peak_rows = []

for signal in signals:
    peak_rows.extend(detect_peaks(signal))

peak_data = pd.DataFrame(peak_rows)
peak_data.to_csv("fft_detected_peaks_raw_from_csv.csv", index=False)

print("Saved detected peaks to fft_detected_peaks_raw_from_csv.csv")
print(peak_data)

# -------------------------------------------------
# Plot 1: translational DOFs, raw amplitude
# -------------------------------------------------
plt.figure(figsize=(10, 5.5))

translation_signals = ["u_x", "u_y", "u_z"]

for name in translation_signals:
    plt.plot(
        freq,
        fft_data[f"{name}_amp"],
        linewidth=2.0,
        label=rf"${name}$"
    )

    signal_peaks = peak_data[peak_data["signal"] == name]

    plt.scatter(
        signal_peaks["frequency_hz"],
        signal_peaks["amplitude"],
        s=35,
        marker="o"
    )

    for _, row in signal_peaks.iterrows():
        plt.annotate(
            f"{row['frequency_hz']:.1f} Hz",
            xy=(row["frequency_hz"], row["amplitude"]),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
            rotation=45
        )

for f in theoretical_freqs:
    plt.axvline(f, linestyle=":", linewidth=0.8, alpha=0.5)

plt.xlim(0, f_max_plot)
plt.ylim(bottom=0)

plt.xlabel("Frequency [Hz]", fontsize=12)
plt.ylabel("FFT amplitude [-]", fontsize=12)
plt.title("FFT of translational deformation at tip node", fontsize=14)

plt.grid(True, linestyle=":", linewidth=0.8)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig("fft_translations_raw_from_csv.png", dpi=300)
plt.show()

# -------------------------------------------------
# Plot 2: rotational DOFs, raw amplitude
# -------------------------------------------------
plt.figure(figsize=(10, 5.5))

rotation_signals = ["theta_x", "theta_y", "theta_z"]

for name in rotation_signals:
    label = name.replace("theta", r"\theta")

    plt.plot(
        freq,
        fft_data[f"{name}_amp"],
        linewidth=2.0,
        label=rf"${label}$"
    )

    signal_peaks = peak_data[peak_data["signal"] == name]

    plt.scatter(
        signal_peaks["frequency_hz"],
        signal_peaks["amplitude"],
        s=35,
        marker="o"
    )

    for _, row in signal_peaks.iterrows():
        plt.annotate(
            f"{row['frequency_hz']:.1f} Hz",
            xy=(row["frequency_hz"], row["amplitude"]),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
            rotation=45
        )

for f in theoretical_freqs:
    plt.axvline(f, linestyle=":", linewidth=0.8, alpha=0.5)

plt.xlim(0, f_max_plot)
plt.ylim(bottom=0)

plt.xlabel("Frequency [Hz]", fontsize=12)
plt.ylabel("FFT amplitude [-]", fontsize=12)
plt.title("FFT of rotational deformation at tip node", fontsize=14)

plt.grid(True, linestyle=":", linewidth=0.8)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig("fft_rotations_raw_from_csv.png", dpi=300)
plt.show()