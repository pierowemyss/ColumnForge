# FreeColumn — Distillation Column Solver

[![CI](https://github.com/pierowemyss/FreeColumn/actions/workflows/ci.yml/badge.svg)](https://github.com/pierowemyss/FreeColumn/actions/workflows/ci.yml)

A PySide6 (Qt6) desktop app for designing and rating chemical-engineering
distillation columns, styled after Aspen Plus's RadFrac. Everything —
GUI, thermodynamics, and every solver — is pure Python (NumPy/SciPy), no
compiled dependency required to run it.

![Initialization tab — species and thermodynamics setup](docs/img/initialization-tab.png)

*Screenshot placeholders throughout this README mark where a run of the
actual app belongs — see [Screenshots](#screenshots).*

## Why this exists

Commercial column simulators (Aspen Plus, HYSYS, ChemCAD) are black boxes:
you get an answer, not the Newton iteration that produced it. FreeColumn is
the opposite bet — every solver is legible, self-testing Python you can step
through, from the Antoine fit to the final tridiagonal solve. It was built to
answer a specific question honestly: *what does it actually take to converge
a multicomponent column, end to end?* Three independent solver paths (a
classical Wang-Henke MESH solve, a Boston-Sullivan Inside-Out solve, and a
Naphtali-Sandholm boundary-value sizing tool) exist so their answers can be
checked against each other, not just against a textbook.

## Quick Start

```bash
pip install -r requirements.txt        # PySide6, numpy, scipy, matplotlib
python launch.py                       # run the GUI (canonical entry point)
```

`launch.py` puts `src/python` on the path and calls `gui.main_window.main()`.
To run a module directly, replicate that path:
`PYTHONPATH=src/python python -m gui.main_window`.

## Architecture

FreeColumn is three layers. The GUI never does numerical work; it builds a
plain-data description of the column and hands it to a solver.

```
┌─────────────────────────────────────────────────────────────────┐
│  gui/                                                            │
│    WindowState  ──single source of truth for the whole session   │
│    tabs/  (Initialization → Specifications → Simulation →        │
│            Results → Modules)          panels/   state/          │
└───────────────────────────┬───────────────────────────────────────┘
                             │ WindowState.build_solver_input()
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  core/                                                            │
│    SolverInput  ──canonical column: open per-stage NumPy arrays  │
│    thermodynamics.py  (Antoine/PLXANT, K-values, activity/EOS)   │
│    column_solvers.py  (Bubble-Point, Inside-Out)                 │
│    dof.py / operating_specs.py  (spec ledger → (R, D))           │
│    material_balance.py / enthalpy.py / shortcut.py / flash.py    │
└───────────────────────────┬───────────────────────────────────────┘
                             │ (BVM sizes → warm-starts →)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  side_features/matrix_bvm/   Naphtali-Sandholm boundary-value    │
│                               sizing (Modules tab)                │
│  side_features/freeRCM/      residue-curve maps (Modules tab)    │
└─────────────────────────────────────────────────────────────────┘
```

`core` never imports `gui` — it's a standalone library you could drive from
a script or a Jupyter notebook with no Qt installed. Every `core` module also
carries a runnable self-check (`python -m core.column_solvers` and friends),
so correctness isn't only asserted by the pytest suite; each file vouches for
itself.

## The solvers

| | method | answers | scope |
|---|---|---|---|
| **Bubble-Point** | Wang-Henke MESH | rigorous tray-by-tray profile | CMO, total condenser |
| **Inside-Out (HYSIM)** | Boston-Sullivan two-tier | same, faster on stiff systems | CMO or full energy balance |
| **BVM** | Naphtali-Sandholm difference-point chain | *is this split feasible, and how many stages does it need?* | sizing/feasibility, energy balance, reactive stages |
| **Shortcut (FUG)** | Fenske-Underwood-Gilliland-Kirkbride | back-of-envelope N, R<sub>min</sub>, feed stage | screening tool, constant α |

### Bubble-Point — Wang-Henke MESH

The textbook rigorous solve: **M**aterial balance, **E**quilibrium,
**S**ummation, **H**eat balance, solved stage-by-stage until temperatures
stop moving. FreeColumn assembles the per-component material balance as a
tridiagonal system in liquid composition and solves it with the Thomas
algorithm — one linear solve per component, per outer iteration:

```
Material balance (stage n, component i):

  -V(n-1) K(i,n-1) x(i,n-1)  +  [L(n) + V(n)] x(i,n)  -  L(n+1) x(i,n+1)
      = F(n) z(i,n)

Equilibrium:      y(i,n) = K(i,n) x(i,n),   K = gamma * Psat(T) * phi / P

Summation:        sum_i  K(i,n) x(i,n)  =  1     →  solved for T(n) (bubble point)
```

Outer loop: assemble the tridiagonal system at the current `T` profile →
Thomas-solve for `x` → bubble-point each stage for a new `T` → repeat until
`max|ΔT| < tol`. Constant molar overflow (CMO) by default; an optional
`flows_hook` seam lets an energy balance replace the CMO flow assumption
without touching the assembly code.

### Inside-Out (HYSIM) — two-tier solve

The trick Boston & Sullivan used to make rigorous solves converge on stiff,
highly non-ideal systems: don't call the expensive thermodynamics (activity
model, Antoine, EOS) every inner iteration. Split into an **outer** loop that
refreshes rigorous K-values and freezes them into per-stage relative
volatilities, and a cheap **inner** loop that iterates only those frozen
ratios against the material balance:

```
OUTER (expensive thermo, once per pass):
  K(i,n)   = gamma(i,n) * Psat_i(T_n) * phi_i / P        — rigorous
  Kb(n)    = geometric mean of K(:,n)                     — per-stage base K
  alpha(i,n) = K(i,n) / Kb(n)                              — frozen below

INNER (cheap, no thermo calls, iterated to `tol`):
  x        = tridiagonal solve at K = alpha * Kb           — material balance
  Kb_new   = 1 / sum_i( alpha(i,n) * x(i,n) )              — bubble constraint
  repeat until |Kb_new - Kb| / Kb < tol

Refresh T from rigorous bubble-point on the new x, repeat OUTER until
max|ΔT| < tol.
```

Because the inner loop reuses one frozen `alpha`, most of the iteration
avoids the thermo evaluation that dominates cost on multicomponent,
non-ideal mixtures — the same reason production simulators default to it.

### BVM — Naphtali-Sandholm difference-point chain

BVM answers a different question than the two solvers above: not *converge
this column*, but *is this split even reachable, and with how many stages?*
Given an operating point `(R, S, E/F)`, it builds a **difference-point
chain** — one difference point Δ<sub>k</sub> per column section — and marches
composition profiles inward from each product end until adjacent profiles
meet:

```
Difference point (mass balance, defines the operating line for section k):

  Δ_k  =  (V_k y_k − L_k x_k) / (V_k − L_k)     — net flow composition,
                                                    collinear with every
                                                    (x, y) pair the section
                                                    passes

Stage-by-stage march (equilibrium + operating-line step, per section):

  y_stage = K(x, T, P) · x                       — equilibrium
  x_next  = operating line through Δ_k and y_stage
```

Two profiles are **connected** when they come within tolerance of each
other in full `R^(C−1)` composition space (not a 2-D curve crossing — that
only exists for ternaries, which is why classical textbook BVM is
ternary-only). The stage count where that connection happens *is* the
design output, not an input. Feasibility, R<sub>min</sub>, feed/draw
placement, and reactive-stage support (Ung-Doherty transform) all build on
top of this chain; see `src/side_features/matrix_bvm/README.md` for the
full module-by-module writeup. Module map:

```
problem.py    → feeds/draws/spec → overall balance (x_D, x_B, D, B)
sections.py   → difference-point chain (Δ_k, δ_k) + operating-line coeffs
march.py      → equilibrium + operating-line stepping, Murphree efficiency
anchor.py     → product ends, continuation, saddle-pinch manifold launch
connect.py    → closest-approach connection in full R^(C-1) → stage counts
place.py      → feed operating-line crossover, side-draw purity target
pinch.py      → fixed-point + eigenvalue classification → R_min, min E/F
reactive.py   → reaction-invariant transformed composition variables
diagnostics.py→ classified infeasibility (names the offending section/pinch)
driver.py     → size a column, sweep (R, S, E/F), build the design map
handoff.py    → package stages + profiles as a rigorous-solver warm start
api.py        → size_column / feasibility_map / to_solver (public entry points)
```

A sized BVM column can be sent straight to Bubble-Point as a **warm start**
(`api.to_solver`), converging in a fraction of the cold-start iterations —
the whole reason a sizing tool and a rigorous solver share a repository.

### Shortcut (FUG)

The pencil-and-paper method, kept honest as closed-form correlations rather
than a solve — the "where do I even start" tool that seeds a rigorous run:

```
Fenske (minimum stages, total reflux):
  Nmin = ln[ (xLK/xHK)_D · (xHK/xLK)_B ] / ln(alpha_LK,HK)

Underwood (minimum reflux):
  sum_i  alpha_i z_i / (alpha_i − θ)  =  1 − q        (root θ between the keys)
  Rmin + 1  =  sum_i  alpha_i xD_i / (alpha_i − θ)

Gilliland/Molokanov (actual stages at operating reflux R):
  X = (R − Rmin)/(R + 1),   Y = 1 − exp[ ((1+54.4X)/(11+117.2X)) · (X−1)/√X ]
  N = (Y + Nmin) / (1 − Y)

Kirkbride (feed-stage location, Ns = stripping stages, Nr = rectifying):
  Ns/Nr = [ (B/D)(xF,HK/xF,LK)(xB,LK/xD,HK)^2 ] ^ 0.206
```

## Thermodynamics

Every solver goes through one seam: `K = gamma(x, T) · Psat(T) · phi(y, T, P) / P`.
Swapping a model means swapping what plugs into that seam — solvers never
change.

- **Vapour pressure**: Antoine (`log10 Psat = A − B/(T+C)`) or Aspen extended
  Antoine / PLXANT (`ln Psat = C1 + C2/(C3+T) + C4·T + C5·ln(T) + C6·T^C7`,
  dispatched automatically on coefficient-matrix width).
- **Activity coefficients** (non-ideal liquids): **NRTL**, **Wilson**,
  **UNIQUAC**, two-suffix **Margules**, and **UNIFAC** group-contribution
  (needs no binary parameters — built from the bundled group-interaction
  database). Each model raises a user-facing error when its parameters are
  missing rather than silently falling back to ideal.
- **Equation of state** (vapour-phase fugacity): **SRK**, for pressure
  effects on relative volatility (validated on a 4-atm depropanizer).
- **Component database**: 78 curated species (`core/data/components.json`) —
  Antoine/PLXANT fits, Tc/Pc/ω, Cp, latent heat — searchable by name, alias,
  CAS, or formula, one click to load into a column. Every record is gated by
  a physical-consistency test (Antoine reproduces Tb within 1 K, ΔHvap
  matches Clausius-Clapeyron within 12%). 7 curated NRTL binary pairs ship
  with it, gated against known azeotropes (ethanol/water,
  2-propanol/water, acetone/chloroform, acetone/methanol).
- **Enthalpy**: constant liquid Cp + Watson-corrected latent heat
  (`h_vap(T) = h_liq(T) + ΔHvap,Tb · ((Tc−T)/(Tc−Tb))^0.38`), the shared
  seam behind the Inside-Out energy balance, enthalpy-based feed quality,
  and condenser subcooling.

![Thermodynamics — activity model and binary interaction table](docs/img/thermodynamics-subtab.png)

## Modules tab

Standalone tools that share the session's species/thermo but don't need a
full column defined:

| module | what it does |
|---|---|
| **RCM** | residue-curve maps (the preserved predecessor app, `side_features/freeRCM/`) |
| **BVM** | size/feasibility per above — size at one `R`, sweep a design map, send a warm start to the rigorous solver |
| **Shortcut (FUG)** | Fenske/Underwood/Gilliland/Kirkbride report + stages-vs-reflux curve |
| **Txy/Pxy** | binary bubble/dew loci at fixed P or T, plus an azeotrope table (singular-point classification) |
| **Pure Components** | browse/search the 78-species database, plot Psat(T), load straight into the column |
| **Phase EQ** | isothermal / vapour-fraction flash on the loaded species — a live test bench for whichever thermo model is selected |

![Modules tab — BVM design map](docs/img/modules-bvm.png)

## Column setup and results

- **Species & thermodynamics**: pick VLE/activity/EOS models, edit binary
  interaction tables, search or hand-enter component properties.
- **Column/streams/condenser/reboiler** configuration with a live
  degrees-of-freedom status (`core/dof.py`) — the app tells you exactly how
  many more specs it needs, and which kinds are valid under the active flow
  model, before you hit run.
- **Interactive column diagram** (Specifications → Column Overview): stages,
  feeds, draws, and interheater/intercooler modules on one canvas.
- **Complex topology**: interheaters/intercoolers carry a signed duty the
  energy balance consumes as a real per-stage term.
- **Threaded solves**: every solver runs on a QThread with live
  iteration/residual progress and a real Abort.
- **Results**: composition/temperature/pressure/flow/K-value/enthalpy
  profile plots, a McCabe-Thiele diagram for binary columns, a
  component-aware data table, product-stream summary with mass-balance
  closure, and CSV export. Display units (°C/K/°F, kmol/h/kg/h, kW/MW/kJ/h)
  are chosen independently of solver-internal units.
- **Save/Load**: the full session persists to `.colx` — versioned JSON, no
  pickle (see `docs/adr/0001-json-colx-and-one-model-two-projections.md`).

![Results tab — composition profile and stream summary](docs/img/results-tab.png)

## Project Structure

```
freeColumn/
├── src/
│   ├── python/
│   │   ├── core/          # thermodynamics, solvers, dof, material/energy balance
│   │   │   └── data/      # components.json, unifac_groups.json
│   │   ├── gui/
│   │   │   ├── tabs/      # Initialization / Specifications / Simulation / Results / Modules
│   │   │   ├── panels/    # reusable config panels (species, streams, condenser, ...)
│   │   │   ├── modules/   # BVM, FUG, Txy/Pxy, Pure Components, Phase EQ widgets
│   │   │   ├── state/     # WindowState (single source of truth) + .colx persistence
│   │   │   └── theme/     # Qt stylesheet
│   │   └── tests/         # headless pytest suite (+ tests/validation/)
│   ├── side_features/
│   │   ├── matrix_bvm/    # Naphtali-Sandholm BVM solver (own README + tests/)
│   │   └── freeRCM/       # preserved predecessor (residue curve maps)
│   └── native/            # C/Fortran sources (nifco.f90, column_solver.c) — not built yet
├── legacy/                # pre-src/ prototype scripts, reference only
├── docs/                  # ADRs + archived audits/plans
└── launch.py              # GUI entry point
```

## Testing

The solver and state layers are Qt-free and self-checking. 97 tests, run
headless:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest src/python/tests/ src/side_features/matrix_bvm/tests/
```

`src/python/tests/validation/` is the acceptance gate for solver changes —
BTX ideal, a depropanizer through the PLXANT path, ethanol/water vs its
known NRTL azeotrope, methanol/water vs Perry's VLE data. Every `core`
module is also runnable standalone as a self-check, e.g.
`PYTHONPATH=src/python python -m core.column_solvers`. CI (GitHub Actions)
runs the full suite plus `pyflakes` on Python 3.11 and 3.12.

## Conventions worth knowing

- **Stage 0 = distillate (top)** everywhere in the GUI and result profiles;
  solvers may use a different internal ordering, converted at the boundary.
- **Nothing is silently ignored**: every value you can enter is either
  consumed by the active solver or visibly greyed out with a "not consumed
  yet" tooltip.
- `.colx` files are versioned JSON — no pickle, so an old save stays
  readable by a newer build and vice versa.

See `CLAUDE.md` for the full architecture/conventions writeup and
`AGENTS.md` for code style.

## Screenshots

Placeholders above point at `docs/img/*.png` — drop screenshots of the
running app there with matching filenames and they'll render in place:

- `docs/img/initialization-tab.png` / `thermodynamics-subtab.png`
- `docs/img/modules-bvm.png`
- `docs/img/results-tab.png`

## Roadmap / not yet built

- Native C/Fortran acceleration (`src/native/` sources exist but aren't
  compiled or bound yet).
- A full energy balance in the core Bubble-Point/Inside-Out solvers is
  opt-in for Inside-Out; Bubble-Point is CMO-only (BVM has its own energy
  balance already).
- A full 3-section extractive BVM (interior-section stage counts are
  feasibility-grade, not yet literature-exact — see the "known ceilings"
  section of `src/side_features/matrix_bvm/README.md`).
- Full roadmap: `PLAN_2026-07-06_one-year-roadmap.md`.
