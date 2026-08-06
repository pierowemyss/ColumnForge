"""BVM -- boundary-value column sizing & feasibility (side module).

A self-contained conceptual-design solver (see MatBVM_blueprint.md, v4). It
builds a *difference-point chain* for any S-section topology, marches composition
profiles inward from each product end in their stable direction, anchors interior
sections (continuation, else saddle-pinch manifolds), connects adjacent profiles
by closest approach in full R^(C-1), places feeds by operating-line crossover and
draws by purity target, finds R_min and minimum E/F from pinch conditions, and
sweeps (R, S, E/F) to explore designs. Its output -- stages per section, feed/
draw locations, and full profiles -- is the warm start handed to ColumnForge's
existing rigorous MESH solver. It does NOT converge MESH itself.

Thermodynamics is consumed from ColumnForge's `core` layer through the thin
`thermo_adapter` only. Kernels are pure functions over NumPy arrays (C-port
friendly); no Python objects live in the marching hot loop.

Module map (blueprint Sec 18):
    problem        feeds/draws/entrainer/spec -> overall balance (x_D,x_B,D,B)
    thermo_adapter ThermoProvider interface + ColumnForge wrapper
    sections       difference-point chain (Delta_k, delta_k) + operating lines
    march          equilibrium + operating-line stepping, stable-direction
    anchor         product ends, and the three ways to start an INTERIOR section
                   (`Problem.anchor_method`): saddle manifolds, the saddle's ray
                   end, or continuation off a neighbour
    connect        closest-approach connection in full R^(C-1) -> stage counts
    place          feed operating-line crossover, side-draw purity target
    pinch          fixed-point + eigen classification -> R_min, min E/F
    bodies         rectification bodies + hull distance. Shared with RBM (which
                   re-exports it) and used here to PRUNE which arm of which
                   saddle an interior section can be on before anything is
                   marched -- the hull is an over-approximation, so it cannot
                   choose, only rule out
    reactive       reaction-invariant transformed-composition marching
    splits         the C-2 free non-key distillate splits: Fenske seed, pinch-
                   equation solve, marched-profile solve, and `spectrum` -- the
                   one-parameter family of designs indexed by feed position
    diagnostics    classified infeasibility
    driver         sweep (R,S,E/F), build the design map, size a column
    handoff        package stage counts + profiles for the rigorous solver
    api            size_column / feasibility_map / to_solver
"""

# This package imports ColumnForge's thermo as `core.*` (the repo's cross-package
# convention, resolved by launch.py adding src/python to the path). It is imported
# as `side_features.bvm` with src on the path -- see launch.py.

__all__ = [
    "problem", "thermo_adapter", "sections", "march", "anchor", "connect",
    "place", "pinch", "bodies", "reactive", "splits", "diagnostics", "driver",
    "handoff",
    "api",
]
