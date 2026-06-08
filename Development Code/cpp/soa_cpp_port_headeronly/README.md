# SOA / multibody C++ port

This is a C++17/Eigen port of the uploaded Python multibody simulation code.
The port focuses on numerical simulation and CSV export rather than C++ animation.
Plotting is intended to be done later in Python with pandas/Matplotlib.

## Equivalent libraries

- Python `numpy` / `scipy.linalg` -> C++ `Eigen`
- Python `solve_ivp` / custom RK4 -> included fixed-step RK4 and backward Euler
- Python `pandas` CSV workflows -> C++ `std::ofstream` CSV export, then Python `pandas.read_csv`
- Python `matplotlib` animation -> not ported; use `simulation_nodes.csv` from Python

## Files

- `include/soa/SoaCpp.hpp` — header-only C++ port containing the multibody classes
- `examples/example_simulation.cpp` — minimal example that writes CSV files
- `scripts/plot_csv.py` — example Python plotting script
- `CMakeLists.txt` — build configuration

## Build

Install Eigen3, then:

```bash
mkdir build
cd build
cmake ..
cmake --build .
./example_simulation
```

The example writes:

- `simulation_wide.csv`: one row per time step; includes time, full state vector, node positions, and node velocities.
- `simulation_nodes.csv`: tidy/long format with columns `t,frame,body,node,x,y,z,vx,vy,vz`.

## Plot in Python

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simulation_wide.csv")
plt.plot(df["t"], df["body0_node7_z"])
plt.xlabel("Time [s]")
plt.ylabel("Tip z [m]")
plt.show()
```

Or use the tidy file:

```python
nodes = pd.read_csv("simulation_nodes.csv")
tip = nodes[(nodes.body == 0) & (nodes.node == nodes[nodes.body == 0].node.max())]
plt.plot(tip.t, tip.z)
plt.show()
```

## Notes

The original Python code imports a local `SOALIB.soalib` module that was not included with the upload. I recreated the required functions in the C++ `soa` namespace, including spatial transforms, quaternion conversion, hinge maps, RK4, and rigid/spatial algebra helpers.

The C++ example is only a template. Replace the example body definitions, initial conditions, loads, springs, and track/earth settings with the same values you used in Python.

## MSYS2 note

MSYS2 may install Eigen 5.x. This project pins Eigen 3.4.0 through CMake FetchContent to avoid MinGW linker errors with Eigen internals. If automatic download fails, manually download Eigen 3.4.0 and place the extracted folder here:

```text
third_party/eigen-3.4.0/Eigen/Dense
```

Then rebuild with:

```powershell
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
cmake -S . -B build -G Ninja
cmake --build build
```


## MSYS2 note

This package uses Eigen as a header-only dependency. The CMake file fetches only the Eigen 3.4.0 headers and does not configure Eigen's tests or build targets.
