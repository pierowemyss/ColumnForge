# Matrix BVM

A universal **feasibility solver + MESH-initialization framework** for staged
separation columns, implemented per `MatBVM_blueprint.md`. It builds the full
Naphtali–Sandholm residual system `R(U) = 0` in component-flow variables, a
structured initial guess `U⁰`, and offers a damped-Newton / continuation solve
on top — but its primary product is the **feasibility report and `U⁰`** you can
hand to any external nonlinear solver.

It is a self-contained side module: it consumes FreeColumn's thermodynamics
(`core.thermodynamics`) through a thin adapter and never reimplements VLE or
enthalpy.

## Formulation

Stages are numbered `0 .. N-1`, top → bottom; liquid flows down, vapour up.
Per-stage unknowns are component flows plus temperature (plus reaction extents),
packed with a **constant stride** so the tridiagonal kernels port to C:

```
block i = [ l_i0..l_i,C-1 | v_i0..v_i,C-1 | T_i | ξ_i0..ξ_i,R-1 ]   length m = 2C+1+R
L_i = Σ_j l_ij   V_i = Σ_j v_ij   x_ij = l_ij/L_i   y_ij = v_ij/V_i
```

Equation rows in each block:

| rows | equation | form |
|------|----------|------|
| `0 .. C-1`      | material balance | `(1+rₗ)l_ij + (1+r_v)v_ij − l_{i-1,j} − v_{i+1,j} − f_ij − Σ_r ν_rj ξ_ir` |
| `C .. 2C-1`     | equilibrium (cleared) | `K_ij·l_ij·V_i − v_ij·L_i` |
| `2C`            | energy balance | N–S enthalpy balance (**replaced by the terminal spec** on stages `0`, `N-1`) |
| `2C+1 .. 2C+R`  | reaction closure | kinetic `ξ − holdup·k·∏x^order` or equilibrium `∏x^ν − K_eq` (`ξ=0` on non-reactive stages) |

Boundaries: `l_{-1}=0`, `v_N=0`. So the top stage's vapour `v_0` **is** the
(partial-condenser) distillate of rate `D`, and the bottom stage's liquid
`l_{N-1}` is the bottoms.

**Squareness (why it stays block-tridiagonal).** The base system is exactly
`2C+1` equations/unknowns per stage. The two terminal energy balances are
*replaced* by the reflux/boilup-family operating specs; the freed
condenser/reboiler duties are **recovered** afterward. Nothing is appended, so
the system stays square and block-tridiagonal, and the required-spec count is the
ordinary MESH design-DoF — borrowed from `core.dof.DoFAnalyzer`. Multifeed, side
draws, pumparounds and inter-heaters are parameter changes (non-zero array
entries / sidedraw ratios), never new equation types.

## Modules

| module | role |
|--------|------|
| `problem`        | topology + specs + a square DOF ledger (`build_problem`, `Problem`, `OpSpec`, `Reactions`) |
| `thermo_adapter` | `ThermoProvider` interface + `FreeColumnThermo` wrapper |
| `residual`       | `R(U)` assembly and `pack`/`unpack`/`mass_balance_residual` |
| `jacobian`       | analytic block-tridiagonal `A_i, B_i, C_i` (+ `fd_jacobian` oracle) |
| `linsolve`       | `block_thomas` — block Thomas / LU sweep, O(N·m³) |
| `initializer`    | `U⁰`: FUG split → BVM trajectory stepping → CMO flows → bubble-T |
| `newton`         | LM-damped, projected, backtracking Newton (`newton`, `recover_duties`) |
| `continuation`   | `thermodynamic_homotopy` (ideal→real) + `parameter_homotopy` |
| `diagnostics`    | `classify` / `assess` — failure classes + offending stages |
| `api`            | `assess_feasibility`, `initialize`, `converge`, external-solver hooks |

## Quick start

```python
import numpy as np
from thermo_adapter import FreeColumnThermo
from problem import build_problem, OpSpec
from api import assess_feasibility, converge

abc = np.array([(6.90565, 1211.033, 220.79),
                (6.95464, 1344.8, 219.48),
                (6.99052, 1453.43, 215.31)])          # benzene/toluene/xylene, mmHg·°C
tp = FreeColumnThermo(abc)                            # optional gamma_fn=<NRTL closure>

prob = build_problem(
    n_stages=16, comps=["benzene", "toluene", "xylene"],
    feeds=[(8, 100.0, [0.4, 0.35, 0.25])],            # (stage, flow, z[, T_feed])
    pressure=760.0, provider=tp,
    top_spec=OpSpec("reflux_ratio", 3.0),
    bottom_spec=OpSpec("bottoms_rate", 60.0))

# feasibility first — report + structured guess, no solve
fa = assess_feasibility(prob, tp)
print(fa["feasible"], fa["findings"])

# convergence (offered, not the point)
sol = converge(prob, tp)
print(sol["converged"], sol["xD"], sol["xB"], sol["condenser_duty"])
```

`converge` returns a profile dict oriented **bottom → top** (index 0 =
reboiler/bottoms), matching `core.column_solvers`, plus `condenser_duty`,
`reboiler_duty`, and a `mass_balance` audit.

### Spec kinds

- top: `reflux_ratio`, `reflux_rate`, `distillate_rate`, `dist_purity` (with `comp=`)
- bottom: `boilup_ratio`, `boilup_rate`, `bottoms_rate`, `bottoms_purity` (with `comp=`)

LK/HK recovery is deliberately **not** a spec: it is a projection used to shape
the initializer (blueprint §10), never a governing equation.

## ThermoProvider contract

Subclass `thermo_adapter.ThermoProvider` (or reuse `FreeColumnThermo`). All
methods take stage-major arrays — `x`/`y` are `(N, C)`, `T`/`P` are `(N,)` — and
return:

```
K(x,T,P)      -> (N, C)        dK_dx(x,T,P) -> (N, C, C)   # dK_j/dx_k
dK_dT(x,T,P)  -> (N, C)        dK_dP(x,T,P) -> (N, C) | None
h_L(x,T)      -> (N,)          h_V(y,T)     -> (N,)
dhL_dx(x,T)   -> (N, C)        dhL_dT(x,T)  -> (N,)
dhV_dy(y,T)   -> (N, C)        dhV_dT(y,T)  -> (N,)
bubble_T(x,P) -> scalar        dew_T(y,P)   -> scalar      Psat(T) -> (C,)
```

The solver **does not assume where a derivative came from** — analytic,
complex-step or finite-difference are all valid. `FreeColumnThermo` uses central
finite differences (the `core` NRTL closure casts to real, so complex-step can't
thread through it) and an analytic `dK_dP = −K/P`. Enthalpies are a constant-`Cp`
sensible term plus a Clausius–Clapeyron latent heat; `Cp` and `Tref` are the
calibration knobs (real heat-capacity data slots straight in).

## External solvers

Matrix BVM hands the raw pieces to any nonlinear solver:

```python
from api import residual_fn, jacobian_fn, dense_jacobian_fn, initialize
f  = residual_fn(prob, tp)          # U -> R
J  = jacobian_fn(prob, tp)          # U -> (A, B, C) block diagonals
Jd = dense_jacobian_fn(prob, tp)    # U -> dense J   (for scipy.optimize.root)
U0 = initialize(prob, tp)
```

Row-scale the residual (see `newton._row_scale`) before feeding a generic solver
— the energy row is ~10⁶ larger than the material rows.

## Convergence notes

The energy-coupled MESH is stiff from a CMO cold start (the energy imbalance is
~10⁶). `newton` handles it with **adaptive Levenberg–Marquardt / pseudo-transient
damping** on the diagonal blocks, a positivity projection (instead of
fraction-to-boundary, which a trace component would otherwise lock), and Armijo
backtracking on the scaled merit. When Newton still stalls on a strongly
non-ideal column, `converge` falls back to the ideal→real thermodynamic homotopy.
Near a minimum-reflux pinch the last digits crawl (a vanishing component sits at
the flow floor) — that is where the feasibility diagnostics, not tighter
tolerances, are the right tool.

## Running the checks

```bash
python run_checks.py                 # every module's _demo self-check
python tests/test_validation.py      # Section-15 cross-cutting validation
# or: pytest tests/
```

Both need `src/python` on the path for `core` (the package `__init__` and the
runners add it automatically). The kernels validated: analytic Jacobian vs FD to
~1e-9, block Thomas vs dense solve to 1e-10, mass-balance closure on every
returned state, agreement with `core.column_solvers.solve_bubble_point`,
homotopy rescue of a Newton-stalling NRTL case, and a converged reactive column
that conserves atoms.
