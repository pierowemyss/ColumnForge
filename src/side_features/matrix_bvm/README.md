# Matrix BVM

A **boundary-value column sizing & feasibility** side module for FreeColumn,
implementing the difference-point-chain method of `MatBVM_blueprint.md` (v4).

Given a separation and an operating point `(R, S, E/F)`, Matrix BVM answers:
**is the split feasible, and if so how many equilibrium stages does each section
need, and where do the feeds and draws go?** It does this by building a
*difference-point chain* for the column topology, marching composition profiles
inward from each product end, connecting adjacent profiles by closest approach
in full composition space, and iterating over reflux / boilup / entrainer ratio.

Matrix BVM is a **conceptual-design / sizing** method. It does **not** converge
the rigorous MESH system — its output (stages per section + full profiles) is the
**warm start** handed to FreeColumn's existing rigorous solver
(`core.column_solvers.solve_bubble_point`), sharply cutting that solver's burden.

## What it is (and isn't)

The genuinely "Matrix" content lives in three places classic ternary BVM handles
only by accident of low dimension:

1. **Connection** is closest approach in full `R^(C-1)`, not a geometric
   curve-crossing (which only exists at `C = 3`). See `connect.py`.
2. **Interior sections** (`S > 2`) have no product anchor; they are anchored by
   continuation or, when strongly pinched, at a **saddle pinch** via its
   invariant manifolds. See `anchor.py`.
3. **Feed / draw placement** is an operating-line crossover / purity target,
   computed, not guessed. See `place.py`.

The "matrix" is the **Jacobian eigenstructure at the pinches** (`pinch.py`) — not
the big block matrix of the rigorous MESH solve.

## Module map (blueprint §18)

| module | role |
|---|---|
| `problem.py` | feeds/draws/entrainer/spec → overall balance `(x_D, x_B, D, B)` |
| `thermo_adapter.py` | the `ThermoProvider` interface + `FreeColumnThermo` wrapper |
| `sections.py` | the difference-point chain `(Δ_k, δ_k)` + operating-line coeffs |
| `march.py` | equilibrium + operating-line stepping, stable-direction selection, Murphree efficiency |
| `anchor.py` | product ends, continuation, saddle-pinch manifold launch |
| `connect.py` | closest-approach connection in full `R^(C-1)` → stage counts |
| `place.py` | feed operating-line crossover, side-draw purity target |
| `pinch.py` | fixed-point + eigenvalue classification → `R_min`, min `E/F` |
| `reactive.py` | reaction-invariant transformed-composition variables |
| `diagnostics.py` | classified infeasibility (names the offending section/pinch) |
| `driver.py` | size a column, sweep `(R, S, E/F)`, build the design map |
| `handoff.py` | package stages + profiles for the rigorous solver |
| `api.py` | `size_column` / `feasibility_map` / `to_solver` |

Kernels are pure functions over NumPy arrays (C-port friendly); no Python objects
live in the marching hot loop.

## API

```python
from thermo_adapter import FreeColumnThermo
from problem import build_problem
import api

tp = FreeColumnThermo(antoine, gamma_fn=gamma_fn, phi_fn=phi_fn)   # §17 provider (SRK optional)
prob = build_problem(comps, feeds=[(z, F, q)], pressure=P,
                     lk=0, hk=1, rec_lk=0.98, rec_hk=0.02)

design = api.size_column(prob, tp, R=4.0)                  # -> design dict
if design["feasible"]:
    N   = design["N_total"]           # total stages (an OUTPUT)
    fs  = design["feed_stages"]        # section boundaries (stage indices)
    col = design["column"]             # x, y, T, liquid_flow, vapor_flow, feed_stage
    Rmin = design["R_min"]
    init = api.to_solver(design)       # warm start for the rigorous MESH solver
else:
    for f in design["findings"]:       # classified reasons (§11)
        print(f.cls, f.section, f.detail)

fmap = api.feasibility_map(prob, tp, R_grid=[1, 2, 4, 8])  # feasibility + stages grid
```

- **`size_column(prob, provider, R, S=None, EF=None) → design`** — size at one
  operating point; attaches `R_min` (and `EF_min` in extractive mode).
- **`feasibility_map(prob, provider, R_grid, S_grid=None, EF_grid=None) → map`** —
  feasibility (bool grid) + stage count (int grid, `-1` where infeasible).
- **`to_solver(design) → init_state`** — plain warm-start dict (§12).

**Conventions.** Stage 0 = distillate (top), matching the FreeColumn GUI.
Components are listed light → heavy; `lk < hk` index into that list. Strictly
non-distributing components are kept at a `1e-4` trace in each product so profiles
can leave a simplex face (heavies amplify downward in the rectifying section).

## ThermoProvider contract (§17)

The module consumes FreeColumn thermo through a narrow adapter — it never
reimplements VLE/enthalpy. A provider supplies:

```
K(x, T, P)        -> (N, C)      equilibrium ratios y = K x
bubble(x, P)      -> (y, T)      conjugate vapour + stage T   (stripping march)
dew(y, P)         -> (x, T)      conjugate liquid + stage T   (rectifying march)
bubble_T/dew_T    -> T
Psat(T) / K       -> vapour pressure / K-values
h_L, h_V          -> molar enthalpies (only for energy-corrected flows)
```

`FreeColumnThermo` wraps `core.thermodynamics` (Antoine/PLXANT `Psat`, γ via any
FreeColumn activity model, optional γ–φ). Default flow model is **constant molar
overflow**; an energy-corrected variant can update section flows from the shared
enthalpy functions.

## Handoff to the rigorous solver (§12)

`to_solver(design)` returns a plain dict — `n_stages`, `feed_stage`,
`draw_stages`, `R`, `D`, `B`, `pressure`, `comps`, and the warm-start profiles
`x0 (N,C)`, `y0`, `T0`, `L0`, `V0` (stage 0 = top). `solve_bubble_point` consumes
`x0`/`T0` directly through its warm-start hook, converging in materially fewer
iterations than a cold start (see `tests/test_validation.py`).

## Running

```bash
# each kernel is runnable and asserts its own sanity
python sections.py && python march.py && python connect.py && python driver.py

# the validation suite (blueprint §19), headless
QT_QPA_PLATFORM=offscreen python -m pytest tests/ -q
```

## Known ceilings (marked `ponytail:` in the source)

- **Extractive / strongly-pinched interior sections** run through the saddle-pinch
  machinery, but exact literature stage counts need finer invariant-manifold
  tracing than the current forward-map launch. Feasibility and min-`E/F` *trends*
  are captured; three-digit stage counts for extractive designs are not the goal.
- **Reactive is NOT consumed by the sizing loop.** `reactive.py` provides the
  Ung–Doherty reaction-invariant transform (validated for invariance), but
  `size_column` does not march in transformed coordinates — it raises
  `NotImplementedError` if a `Problem.reactions` set is supplied rather than
  silently ignoring it. Transformed-space marching (physical VLE as the stagewise
  closure) is the upgrade path; the GUI exposes no reactive input.
- **`R_min` / min-`E/F`** come from bisection on the connection boundary (robust,
  equivalent to pinch tangency) rather than a direct pinch-tangency solve. The
  bisection coarse-pre-scans first so a spurious low-`R` feasibility island (from
  the local connection tolerance) is not mistaken for the minimum.
- **`dew()` uses a γ(y) proxy**, not a self-consistent γ(x) fixed point. The
  audit-preferred γ(x) dew was implemented and reverted: for the stiff
  MEOH/DMC/EG multicomp reference it has a second (EG-heavy) root the rectifying
  march jumps to (T→1700 K, blow-up), breaking the reference contract. SRK
  fugacities (`phi_fn`, wired) keep the proxy march on the physical branch.
  Upgrade path: branch-continuation dew seeded from the previous stage's liquid.
- **Murphree efficiency** (`Problem.efficiency`, GUI spin) inflates the ideal-stage
  march per direction. For sloppy difference-point splits whose rectifying section
  already pinches deep (e.g. the multicomp reference, ~45 *ideal* stages), stacking
  `E<1` on top over-counts; the reference stage count there is matched by the
  ideal march. Efficiency is validated on cleaner columns (extractive, BTX).
- **Columns with more than three sections** (multiple interior sections) size the
  enclosing two-section problem; full N-section assembly is not wired yet.
