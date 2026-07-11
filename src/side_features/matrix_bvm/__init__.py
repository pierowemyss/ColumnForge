"""Matrix BVM -- boundary-value column sizing & feasibility (side module).

A self-contained conceptual-design solver (see MatBVM_blueprint.md, v4). It
builds a *difference-point chain* for any S-section topology, marches composition
profiles inward from each product end in their stable direction, anchors interior
sections (continuation, else saddle-pinch manifolds), connects adjacent profiles
by closest approach in full R^(C-1), places feeds by operating-line crossover and
draws by purity target, finds R_min and minimum E/F from pinch conditions, and
sweeps (R, S, E/F) to explore designs. Its output -- stages per section, feed/
draw locations, and full profiles -- is the warm start handed to FreeColumn's
existing rigorous MESH solver. It does NOT converge MESH itself.

Thermodynamics is consumed from FreeColumn's `core` layer through the thin
`thermo_adapter` only. Kernels are pure functions over NumPy arrays (C-port
friendly); no Python objects live in the marching hot loop.

Module map (blueprint Sec 18):
    problem        feeds/draws/entrainer/spec -> overall balance (x_D,x_B,D,B)
    thermo_adapter ThermoProvider interface + FreeColumn wrapper
    sections       difference-point chain (Delta_k, delta_k) + operating lines
    march          equilibrium + operating-line stepping, stable-direction
    anchor         product ends, continuation, saddle-pinch manifold launch
    connect        closest-approach connection in full R^(C-1) -> stage counts
    place          feed operating-line crossover, side-draw purity target
    pinch          fixed-point + eigen classification -> R_min, min E/F
    reactive       reaction-invariant transformed-composition marching
    diagnostics    classified infeasibility
    driver         sweep (R,S,E/F), build the design map, size a column
    handoff        package stage counts + profiles for the rigorous solver
    api            size_column / feasibility_map / to_solver
"""

# Bootstrap: this module imports FreeColumn's thermo as `core.*` (the repo's
# cross-package convention, resolved by launch.py adding src/python to the path).
# Ensure that path is present so the package is importable on its own.
import os as _os
import sys as _sys

_SRC_PY = _os.path.normpath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "python"))
if _SRC_PY not in _sys.path:
    _sys.path.insert(0, _SRC_PY)

__all__ = [
    "problem", "thermo_adapter", "sections", "march", "anchor", "connect",
    "place", "pinch", "reactive", "diagnostics", "driver", "handoff", "api",
]
