from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")   # Force popup window backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


project_dir = Path(__file__).resolve().parents[1]
csv_path = project_dir / "simulation_nodes_5_links.csv"

nodes = pd.read_csv(csv_path)

nodes = nodes[np.isfinite(nodes[["x", "y", "z"]]).all(axis=1)].copy()

if nodes.empty:
    raise RuntimeError("No valid node data found.")

frames = sorted(nodes["frame"].unique())
bodies = sorted(nodes["body"].unique())

print(f"Loaded {len(nodes)} valid node rows")
print(f"Frames: {len(frames)}")
print(f"Bodies: {bodies}")

x_min, x_max = nodes["x"].min(), nodes["x"].max()
y_min, y_max = nodes["y"].min(), nodes["y"].max()
z_min, z_max = nodes["z"].min(), nodes["z"].max()

cx = 0.5 * (x_min + x_max)
cy = 0.5 * (y_min + y_max)
cz = 0.5 * (z_min + z_max)

span = max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-6)
half = 0.55 * span

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(cx - half, cx + half)
ax.set_ylim(cy - half, cy + half)
ax.set_zlim(cz - half, cz + half)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("Beam animation")

lines = {}

for body in bodies:
    line, = ax.plot([], [], [], "-o", linewidth=2, markersize=4, label=f"Body {body}")
    lines[body] = line

ax.legend()

time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)


def update(i):
    frame = frames[i]
    frame_data = nodes[nodes["frame"] == frame]

    if not frame_data.empty:
        t = frame_data["t"].iloc[0]
        time_text.set_text(f"t = {t:.4f} s")

    for body in bodies:
        data = frame_data[frame_data["body"] == body].sort_values("node")

        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        z = data["z"].to_numpy()

        lines[body].set_data(x, y)
        lines[body].set_3d_properties(z)

    return list(lines.values()) + [time_text]


ani = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=40,
    blit=False,
    repeat=True
)

plt.show(block=True)