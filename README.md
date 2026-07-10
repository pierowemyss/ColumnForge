# FreeColumn — Distillation Column Solver

A PySide6 (Qt6) GUI for solving chemical-engineering distillation columns, with an
Aspen Plus-inspired interface. Four solver paths are implemented in pure Python:
**Bubble-Point** (Wang-Henke MESH) and **Inside-Out (HYSIM)** rigorous solvers,
**BVM** (Boundary Value Method — feasibility/preliminary design), and
**Matrix BVM** (Naphtali-Sandholm with a real energy balance).

## Quick Start

```bash
pip install -r requirements.txt        # PySide6, numpy, scipy, matplotlib
python launch.py                       # run the GUI (canonical entry point)
```

`launch.py` puts `src/python` on the path and calls `gui.main_window.main()`. To run a
module directly, replicate that: `PYTHONPATH=src/python python -m gui.main_window`.

## Features

- **Column setup**: define species and thermodynamics, configure column/streams/
  condenser/reboiler, with a live degrees-of-freedom status and an interactive
  column diagram.
- **Four solvers**:
  - *Bubble-Point* — rigorous tray-by-tray MESH solve (CMO, total condenser).
  - *Inside-Out (HYSIM)* — two-tier solve (outer rigorous K / relative volatilities,
    inner frozen-α material balances), emitting per-stage flows, K-values and a profile.
  - *BVM (preliminary)* — section-curve feasibility + stage profile (recovery or direct
    xD/xB spec mode; approximate extractive support). Modules tab.
  - *Matrix BVM* — Naphtali-Sandholm block-tridiagonal Newton with an energy balance
    (condenser/reboiler duties, reactive-stage support). Modules tab.
- **Thermodynamics**: ideal (Antoine + Raoult) by default, with an **NRTL** activity
  model for non-ideal mixtures (other activity/EOS models are future drop-ins).
- **Auto-saturate** streams to their bubble-/dew-point temperature.
- **Results**: composition/temperature plots (plus flow/K-value/enthalpy profiles from the
  Inside-Out solver), a component-aware data table, and CSV export.
- **Save / Load** the full configuration to `.colx` (versioned JSON).

## Project Structure

```
freeColumn/
├── src/
│   ├── python/
│   │   ├── core/          # thermodynamics, column_solvers, material_balance, dof, data_structures
│   │   ├── gui/           # PySide6 app: main_window + tabs/, panels/, state/, modules/
│   │   └── tests/         # headless pytest suite
│   ├── side_features/
│   │   ├── bvm/           # section-marching BVM solver
│   │   ├── matrix_bvm/    # Naphtali-Sandholm solver (own tests/)
│   │   └── freeRCM/       # preserved predecessor (residue curve maps)
│   └── native/            # C/Fortran sources (nifco.f90, column_solver.c) — not built yet
├── legacy/                # pre-src/ prototype scripts, reference only
├── docs/                  # ADRs + archived audits/plans
└── launch.py              # GUI entry point
```

## Testing

The solver/state layers are Qt-free and self-checking. Run the full suite headless:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest src/python/tests/ src/side_features/matrix_bvm/tests/
```

Each core module also has a runnable self-check, e.g.
`PYTHONPATH=src/python python -m core.column_solvers`.

## Development

See `CLAUDE.md` for architecture and conventions, `AGENTS.md` for code style, and
`PLAN_2026-07-06_one-year-roadmap.md` for the current roadmap to v1.0.

## Not yet built

- Native C/Fortran acceleration (`src/native/` sources exist but aren't compiled or bound).
- Additional activity/EOS models beyond NRTL; a full energy balance in the core
  Bubble-Point/Inside-Out solvers (Matrix BVM has one); a full 3-section extractive BVM.
- Component property database (Month 2 of the roadmap) — species are entered manually.
