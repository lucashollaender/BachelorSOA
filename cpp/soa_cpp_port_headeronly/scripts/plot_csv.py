import pandas as pd
import matplotlib.pyplot as plt

# Wide file: one row per output frame, state variables + all node positions.
df = pd.read_csv("simulation_wide.csv")

# Example: plot tip node z displacement of body 0.
# Change node index if your body has a different number of nodes.
tip_z_cols = [c for c in df.columns if c.startswith("body0_node") and c.endswith("_z")]
tip_z_col = tip_z_cols[-1]

plt.figure()
plt.plot(df["t"], df[tip_z_col])
plt.xlabel("Time [s]")
plt.ylabel(f"{tip_z_col} [m]")
plt.title("Tip z-position")
plt.grid(True)
plt.show()

# Long/tidy file: useful for animations or grouping by body/node.
nodes = pd.read_csv("simulation_nodes.csv")
last_node = nodes[(nodes["body"] == 0) & (nodes["node"] == nodes[nodes["body"] == 0]["node"].max())]

plt.figure()
plt.plot(last_node["t"], last_node["z"])
plt.xlabel("Time [s]")
plt.ylabel("z [m]")
plt.title("Tip z-position from tidy node CSV")
plt.grid(True)
plt.show()
