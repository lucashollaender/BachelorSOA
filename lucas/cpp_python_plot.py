import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# File paths
# -------------------------------------------------
cpp_file = Path(r"C:\Project\BachelorSOA\cpp\soa_cpp_port_headeronly\simulation_nodes_2_links.csv")
py_file = Path(r"C:\Project\BachelorSOA\simulation_nodes_2_links_python.csv")

# -------------------------------------------------
# Load C++ CSV
# C++ columns: t, frame, body, node, x, y, z, vx, vy, vz
# -------------------------------------------------
cpp = pd.read_csv(cpp_file)
cpp = cpp.rename(columns={"t": "time"})
cpp = cpp[["time", "body", "node", "z"]].copy()

# -------------------------------------------------
# Load Python CSV
# Expected columns from the Python test file:
# time, frame, link, body_index, node, x, y, z
# -------------------------------------------------
py = pd.read_csv(py_file)

if "body" not in py.columns:
    if "link" in py.columns:
        py["body"] = py["link"]
    elif "body_index" in py.columns:
        py["body"] = py["body_index"]
    else:
        raise KeyError("Python CSV must contain either 'body', 'link', or 'body_index'.")

if "time" not in py.columns:
    if "t" in py.columns:
        py = py.rename(columns={"t": "time"})
    else:
        raise KeyError("Python CSV must contain either 'time' or 't'.")

py = py[["time", "body", "node", "z"]].copy()

# Make sure values are numeric
for df in [cpp, py]:
    for col in ["time", "body", "node", "z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["time", "body", "node", "z"], inplace=True)

# -------------------------------------------------
# Two plots: end-node z-coordinate for body 0 and body 1
# -------------------------------------------------
for body_id in [0, 1]:
    cpp_body = cpp[cpp["body"] == body_id]
    # Python body order is reversed compared to C++
    py_body = py[py["body"] == 1 - body_id]

    if cpp_body.empty:
        print(f"No C++ data found for body {body_id}")
        continue
    if py_body.empty:
        print(f"No Python data found for body {body_id}")
        continue

    cpp_end_node = cpp_body["node"].max()
    py_end_node = py_body["node"].max()

    cpp_tip = cpp_body[cpp_body["node"] == cpp_end_node]
    py_tip = py_body[py_body["node"] == py_end_node]

    plt.figure()
    plt.plot(cpp_tip["time"], cpp_tip["z"], label="C++")
    plt.plot(py_tip["time"], py_tip["z"], "--", label="Python")
    plt.xlabel("Time [s]")
    plt.ylabel("End-node z-coordinate [m]")
    plt.title(f"Body {body_id}: end-node z-coordinate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

plt.show()
