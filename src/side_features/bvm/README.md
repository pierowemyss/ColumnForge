# BVM

A **boundary-value column sizing & feasibility** side module for ColumnForge,
after Levy, Van Dongen & Doherty (1985) and Doherty & Malone, *Conceptual Design
of Distillation Systems* (2001), generalised to `C > 3` and to multi-section
topologies.

Given a separation and an operating point `(R, S, E/F)`, BVM answers: **is the
split feasible, and if so how many equilibrium stages does each section need, and
where do the feeds and draws go?** It builds a _difference-point chain_ for the
column topology, marches composition profiles inward from each product end,
locates the junction where adjacent profiles meet, and iterates over reflux /
boilup / entrainer ratio.

BVM is a **conceptual-design / sizing** method. It does **not** converge the
rigorous MESH system — its output (stages per section + full profiles) is the
**warm start** handed to `core.column_solvers.solve_bubble_point`.

## What it is (and isn't)

Three places where this goes beyond what textbook ternary BVM needs:

1. **Connection** is a liquid-profile intersection in full `R^(C-1)`, not a
   planar curve crossing (which only exists at `C = 3`). See `connect.py` and
   [Section connection](#section-connection-the-junction-test) below.
2. **Interior sections** (`S > 2`) have no product anchor. There are three ways
   to start one, chosen by `Problem.anchor_method` (the panel's "Interior
   anchor" combo — there is no `auto`, because they do not agree on `r_max`):
   `saddle` (the invariant manifolds through the section's **saddle pinch**, the
   default), `ray` (march inward from the far end of the stable eigendirection,
   the rectification body's S vertex) and `continuation` (launch from stages of
   the neighbouring profiles that lie inside this section). See `anchor.py`.
   Which arm of which saddle is narrowed first by `bodies.py`, geometrically and
   without marching — but only narrowed: a convex hull ties too often, and its
   ranking can invert against the marched answer (`docs/adr/0004`).
3. **Feed / draw placement** is an operating-line crossover / purity target,
   computed, not guessed. See `place.py`.

Pinch classification (`pinch.py`) is by the **Jacobian eigenstructure** of the
stage map, not by the block matrix of a rigorous MESH solve.

## Module map

| module              | role                                                                                   |
| ------------------- | -------------------------------------------------------------------------------------- |
| `problem.py`        | feeds/draws/entrainer/spec → overall balance `(x_D, x_B, D, B)`                        |
| `thermo_adapter.py` | the `ThermoProvider` interface + `ColumnForgeThermo` wrapper                           |
| `sections.py`       | the difference-point chain `(Δ_k, δ_k)` + operating-line coeffs                        |
| `march.py`          | equilibrium + operating-line stepping, stable-direction selection, Murphree efficiency |
| `anchor.py`         | product ends, and the three ways to start an interior section (`Problem.anchor_method`) |
| `connect.py`        | liquid-profile intersection in full `R^(C-1)` → stage counts (see below)               |
| `place.py`          | feed operating-line crossover, side-draw purity target                                 |
| `pinch.py`          | fixed-point + eigenvalue classification → `R_min`, min `E/F`                           |
| `bodies.py`         | rectification bodies + hull distance; shared with RBM, used here to **prune** which arm of which saddle an interior section can be on |
| `reactive.py`       | reactive columns: transformed compositions + the reaction-equilibrium stage closure     |
| `diagnostics.py`    | classified infeasibility (names the offending section/pinch)                           |
| `driver.py`         | size a column, sweep `(R, S, E/F)`, build the design map                               |
| `handoff.py`        | package stages + profiles for the rigorous solver                                      |
| `api.py`            | `size_column` / `feasibility_map` / `to_solver`                                        |

Kernels are pure functions over NumPy arrays (C-port friendly); no Python objects
live in the marching hot loop.

## Section connection (the junction test)

Everything BVM outputs — feasibility, stage counts, feed location, `R_min` —
comes out of one test in `connect.py`, so it is worth stating precisely.

**The criterion.** Two sections meet at a feed when their marched **liquid**
profiles intersect: there exist continuous stage coordinates `xi_R`, `xi_S` with

```
x_R(xi_R) == x_S(xi_S)                                                     (X)
```

That is an intersection, not a proximity. A small distance between two profiles
is not sufficient — two curves can run close for many stages without a column
profile existing that passes from one to the other.

**Why the liquid and not the vapour.** `y = K(x) x` is a function of `x`, so
liquid equality implies vapour equality; the converse fails wherever `dy/dx` is
small, and that is where a vapour-space test quietly goes wrong. On
ethane/propane/n-butane at `R = 0.175, E = 0.5` the two *vapour* curves cross
exactly at `(xi_R, xi_S) = (6.31, 12.04)` while the liquids there are far apart;
the real liquid crossing is at `(10.17, 6.17)`, four stages further down. A
rigorous MESH sweep over the feed stage puts the optimum at 10–11, and the vapour
answer of 7 gave away 2 points of light-key recovery. The same compression made
`R_min` ~22 % low (vapour gap 0.019 against a tolerance that reached 0.10, while
the liquids were 0.072 apart). Measured on the liquid, that column's `R_min` is
~0.139, against 0.134 from the RBM module and 0.147 from Underwood.

**The feed jump is a different distance.** A feed does jump the liquid
composition between the stages either side of it — `x_{f-1}` on the upper section
against `x_f` on the lower one, 0.11 in entrainer across the main feed of
ethanol/water/EG. That jump is real and is reported as `gap_liquid`. It does not
prevent (X), because `x_f` lies on *both* curves: the vapour leaving the feed
stage is what the section above puts its operating line on, so `x_f = dew(y_f)`
whichever section computes it. On BTX the crossing lands at ~1e-16 at every
reflux tested.

**How it is located.**

0. Trim each profile's **pinch tail** (`travel_end`). A profile that has stopped
   moving needs infinite stages to get anywhere, so its crawl is not somewhere a
   feed can be placed; trimming also stops the scan matching against a pile-up of
   near-coincident points. The cut is where the per-stage step falls below the
   crossing tolerance and stays there — a fraction-of-largest-step rule was tried
   first and cut off a genuine crossing.
1. All-pairs **segment/segment closest approach** over the travelling parts
   (clamped standard formulation), giving `dmin` and fractional positions on both
   profiles.
2. Read the stage counts off the crossing. The crossing index on the **upper**
   profile *is* the feed stage, so `nA` (the last stage above the feed) is one
   less, and `nB` is the crossing index on the lower profile. Both stay
   fractional; the caller rounds.
3. Require the crossing point to lie **inside the simplex** — a closest approach
   reached outside it is an artefact of extrapolated segments.

**The tolerance ladder**, which is where the honesty lives:

| case                        | rule                                                                 | flag          |
| --------------------------- | -------------------------------------------------------------------- | ------------- |
| `C <= 3`, product-anchored  | true crossing, `dmin <= 1e-6`                                        | exact         |
| `C >= 4`                    | near miss within one stage of travel, step capped at `STEP_CAP = 0.05` | `approximate` |
| saddle-launched interior arm | same near-miss rule (`strict=False`)                                 | `approximate` |

`C >= 4` is not a tuning choice. Two 1-D curves in the `(C-1)`-simplex carry 2
free coordinates against `C-1` equations, so the system is over-determined and an
exact crossing generically does not exist — the quaternary reference case sits at
8.4e-3 and the `C = 6` one at 0.035, at every reflux. The missing degrees of
freedom are the non-key distillate splits (`problem.free_split_indices`), held at
a trace-floor guess; `splits.solve_free_splits`, which is meant to solve them, does not
converge at `C >= 4`. Until that is replaced those junctions are accepted within
one stage and flagged `approximate`, and **callers must not read that flag as a
crossing**. (This is also why the RBM module exists: bodies are up to
`(C-1)`-dimensional and intersect generically at any `C`.)

The interior-arm case is the same concession for a different reason: a
saddle-launched arm is a manifold branch, not a product-anchored march, and on
2-propanol/water/EG no arm reaches both neighbours exactly (0.082 at the lower
junction, creeping only to 0.068 by `E/F = 2`). Demanding a crossing there would
reject every extractive design the arm construction can produce.

`CROSS_TOL = 1e-6` has three orders of margin on both sides: a real transversal
intersection lands at ~1e-16, and the nearest near-miss measured across BTX,
C2–C4 and the extractive case is 5e-3.

**Murphree efficiency** does not widen the tolerance at `C <= 3`. It used to
divide it, doubling the accepted gap at `E = 0.5`, which is half of why `R_min`
came out low. It still scales the near-miss path, where there is no crossing to
tighten onto and the bridge is an equilibrium stage's reach.

**Reported, never gated on:**

- `residual_vapour` — the junction equation `a x_{f-1} + b == y_f`. At `E = 1`
  liquid equality implies it (~1e-3 on BTX, all interpolation error). Below
  `E = 1` it does not, and what it measures is a real inconsistency in the
  marching model, not a bad junction: the rectifying march computes `x_{f-1}`
  assuming stage `f` carries rectifying flows when it carries stripping ones.
  Gating on it is what put the feed four stages too high.
- `gap_liquid` — the feed jump described above. Supposed to be non-zero.

**Cost.** The scan is `O(N*M)` over a few hundred points per profile; the
marching dominates. A two-pointer walk is the upgrade path if that changes.

## API

```python
from thermo_adapter import ColumnForgeThermo
from problem import build_problem
import api

tp = ColumnForgeThermo(antoine, gamma_fn=gamma_fn, phi_fn=phi_fn)   # (SRK optional)
prob = build_problem(comps, feeds=[(z, F, q)], pressure=P,
                     lk=0, hk=1, rec_lk=0.98, rec_hk=0.02,
                     dP=0.0,            # per-stage pressure drop, Psat unit
                     eps_stage=1e-2)    # floor on the junction tolerance

design = api.size_column(prob, tp, R=4.0)                  # -> design dict
if design["feasible"]:
    N   = design["N_total"]           # total stages (an OUTPUT)
    fs  = design["feed_stages"]        # section boundaries (stage indices)
    col = design["column"]             # x, y, T, liquid_flow, vapor_flow, feed_stage
    Rmin = design["R_min"]
    init = api.to_solver(design)       # warm start for the rigorous MESH solver
else:
    for f in design["findings"]:       # classified reasons
        print(f.cls, f.section, f.detail)

fmap = api.feasibility_map(prob, tp, R_grid=[1, 2, 4, 8])  # feasibility + stages grid
```

- **`size_column(prob, provider, R, S=None, EF=None) → design`** — size at one
  operating point; attaches `R_min` (and `EF_min` in extractive mode).
- **`feasibility_map(prob, provider, R_grid, S_grid=None, EF_grid=None) → map`** —
  feasibility (bool grid) + stage count (int grid, `-1` where infeasible).
- **`to_solver(design) → init_state`** — plain warm-start dict.

**Pressure.** `pressure` is the CONDENSER pressure; `dP` ramps it down the
column. Each march records the pressure it was evaluated at (`prof["P"]`), and
`connect` boils the lower profile at its own. The reboiler end is
`pressure + dP*(N-1)` — a fixed point, since `N` is the method's output, closed by
one refinement pass in `size_column`. `dP = 0` skips it and leaves the column flat.

**Conventions.** Stage 0 = distillate (top), matching the ColumnForge GUI.
Components are listed light → heavy; `lk < hk` index into that list. Strictly
non-distributing components are kept at a `1e-4` trace in each product so profiles
can leave a simplex face (heavies amplify downward in the rectifying section).

## ThermoProvider contract

The module consumes ColumnForge thermo through a narrow adapter — it never
reimplements VLE/enthalpy. A provider supplies:

```
K(x, T, P)        -> (N, C)      equilibrium ratios y = K x
bubble(x, P)      -> (y, T)      conjugate vapour + stage T   (stripping march)
dew(y, P, x_seed) -> (x, T)      conjugate liquid + stage T   (rectifying march)
bubble_T/dew_T    -> T
Psat(T) / K       -> vapour pressure / K-values
h_L, h_V          -> molar enthalpies (only for energy-corrected flows)
```

`ColumnForgeThermo` wraps `core.thermodynamics` (Antoine/PLXANT `Psat`, γ via any
ColumnForge activity model, optional γ–φ). Default flow model is **constant molar
overflow**; an energy-corrected variant can update section flows from the shared
enthalpy functions.

## Reactive distillation

Set `Problem.reactions` (a `reactive.Reactions`: stoichiometry, reference
component, `keq_fn`) and `size_column` runs the whole method in **Ung–Doherty
transformed compositions**. Two things make that cheap: `X_ref ≡ 0`, so one
reaction over `C` components is a `(C-1)`-component problem; and the transform is
reaction-invariant, so `march`/`sections`/`connect`/`pinch`/`place` are used
unchanged. `ReactiveThermo` presents `bubble`/`dew`/`bubble_T` in those
coordinates, putting each stage liquid at chemical equilibrium
(`prod_i (γ_i x_i)^ν_i = Keq(T)`, rooted in the reaction extent at the mixture's
own bubble point) before flashing it with the real VLE. Transformed flows carry
`F̄ = F·denom(x)`, which is what makes the transformed component balance close
exactly; for a total condenser `R̄ = R`.

The design dict gains `reactive=True` and a `physical` block: real compositions,
temperatures and the **reaction extent per stage**.

Ceilings, all enforced rather than assumed:

- **One equilibrium reaction, ideal stages, every stage catalytic.** A Murphree
  stage is an affine blend of vapour compositions and the transform is rational,
  so `efficiency < 1` raises. So do multiple reactions, extractive + reactive,
  and side draws.
- **The transform must stay inside the simplex.** `X_i ≥ 0` for every physical
  composition exactly when each `ν_i` has the opposite sign to `ν_ref`, i.e. a
  **one-product** reaction with the product as reference (etherification,
  hydration, hydrogenation). A two-product reaction — any esterification, where
  ester and water are both products — has no such reference, and the coordinates
  go negative precisely at the two ends a column drives to. `reactive.simplex_safe`
  detects it, warns, and adds a `leaves_simplex` finding instead of mis-sizing.
- **At least three components must survive the reduction**, so one reaction needs
  `C ≥ 4` (MTBE synthesis has its inert n-butane). With two, every profile lies on
  the same line and closest approach is degenerate: `dmin ≈ 0` for many segment
  pairs, so the junction — and with it `N_total` and the feed stage — is arbitrary.
  Refused with that reason. The same degeneracy applies to a genuinely binary
  *non-reactive* column, which belongs to McCabe-Thiele rather than BVM.
- **No rigorous handoff.** `to_solver` output would be converged by a MESH solver
  with no reaction terms, i.e. a different column; the GUI disables the button.
- Products come out **on the reaction-equilibrium surface** (the condenser and
  reboiler are reactive too), so a spec like "essentially pure ester distillate"
  is not attainable in this model — that is what a non-reactive rectifying section
  above the reaction zone is for, and it is not wired yet.

## Handoff to the rigorous solver

`to_solver(design)` returns a plain dict — `n_stages`, `feed_stage`,
`feed_stages`, `draw_stages`, `R`, `D`, `B`, `pressure`, `comps`, and the
warm-start profiles `x0 (N,C)`, `y0`, `T0`, `L0`, `V0` (stage 0 = top).
`solve_bubble_point` consumes `x0`/`T0` directly through its warm-start hook,
converging in materially fewer iterations than a cold start (see
`tests/test_validation.py`).

`feed_stages` is a list because an extractive design has two: entrainer above,
main feed below. The GUI builds a `SolverInput` with one solver feed per BVM
feed (`bvm_module._handoff_feeds`) plus the session's `phi_fn` and the per-stage
pressure — handing over a single pooled feed, or a D sized for `F + E` against a
column carrying only `F`, is a different column.

## Running

```bash
# each kernel is runnable and asserts its own sanity
python sections.py && python march.py && python connect.py && python driver.py

# the validation suite, headless
QT_QPA_PLATFORM=offscreen python -m pytest tests/ -q
```

<!-- ## Known ceilings (marked `ponytail:` in the source) -->
<!---->
<!-- - **Extractive / strongly-pinched interior sections** run through the saddle-pinch -->
<!--   machinery, but exact literature stage counts need finer invariant-manifold -->
<!--   tracing than the current forward-map launch. Feasibility and min-`E/F` _trends_ -->
<!--   are captured; three-digit stage counts for extractive designs are not the goal. -->
<!-- - **Reactive is NOT consumed by the sizing loop.** `reactive.py` provides the -->
<!--   Ung–Doherty reaction-invariant transform (validated for invariance), but -->
<!--   `size_column` does not march in transformed coordinates — it raises -->
<!--   `NotImplementedError` if a `Problem.reactions` set is supplied rather than -->
<!--   silently ignoring it. Transformed-space marching (physical VLE as the stagewise -->
<!--   closure) is the upgrade path; the GUI exposes no reactive input. -->
<!-- - **`R_min` / min-`E/F`** come from bisection on the connection boundary (robust, -->
<!--   equivalent to pinch tangency) rather than a direct pinch-tangency solve. The -->
<!--   bisection coarse-pre-scans first so a spurious low-`R` feasibility island (from -->
<!--   the local connection tolerance) is not mistaken for the minimum. -->
<!-- - **`dew()` uses a γ(y) proxy**, not a self-consistent γ(x) fixed point. A -->
<!--   self-consistent γ(x) dew was implemented and reverted: for the stiff -->
<!--   MEOH/DMC/EG multicomp reference it has a second (EG-heavy) root the rectifying -->
<!--   march jumps to (T→1700 K, blow-up), breaking the reference contract. SRK -->
<!--   fugacities (`phi_fn`, wired) keep the proxy march on the physical branch. -->
<!--   Upgrade path: branch-continuation dew seeded from the previous stage's liquid. -->
<!-- - **Murphree efficiency** (`Problem.efficiency`, GUI spin) inflates the ideal-stage -->
<!--   march per direction. For sloppy difference-point splits whose rectifying section -->
<!--   already pinches deep (the multicomp reference: ~47 _ideal_ stages ≈ the 45-stage -->
<!--   eff-0.5 MESH column), the ideal march already lands the real-column count, so -->
<!--   stacking `E<1` on top roughly _doubles_ it (~93). Such columns size feasibly at -->
<!--   any efficiency, but match the MESH stage count at `E=1`; use `E<1` for cleanly -->
<!--   pinched columns (extractive → 37 at eff 0.5, ≈ MESH 48; BTX). Both are validated. -->
<!-- - **Columns with more than three sections** (multiple interior sections) size the -->
<!--   enclosing two-section problem; full N-section assembly is not wired yet. -->
