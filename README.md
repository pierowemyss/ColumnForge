# FreeColumn — Distillation Column Solver

A PySide6 (Qt6) GUI for solving chemical-engineering distillation columns, with an
Aspen Plus-inspired interface. Three solver paths are implemented in pure Python:
**BVM** (Boundary Value Method — feasibility/preliminary design), a rigorous
**Bubble-Point** (Wang-Henke MESH) solver, and an **Inside-Out (HYSIM)** solver.

## Quick Start

```bash
pip install -r requirements.txt        # PySide6, numpy, scipy, matplotlib
python launch.py                       # run the GUI (canonical entry point)
```

`launch.py` puts `src/python` on the path and calls `gui.main_window.main()`. To run a
module directly, replicate that: `PYTHONPATH=src/python python -m gui.main_window`.

## Features

- **Column setup**: define species and thermodynamics, configure column/streams/
  condenser/reboiler and side modules, with a live degrees-of-freedom status and an
  interactive column diagram.
- **Three solvers**, selectable on the Simulation tab:
  - *BVM (preliminary)* — section-curve feasibility + stage profile (recovery or direct
    xD/xB spec mode; approximate extractive support).
  - *Bubble-Point* — rigorous tray-by-tray MESH solve (CMO, total condenser).
  - *Inside-Out (HYSIM)* — two-tier solve (outer rigorous K / relative volatilities,
    inner frozen-α material balances), emitting per-stage flows, K-values and a profile.
- **Thermodynamics**: ideal (Antoine + Raoult) by default, with an **NRTL** activity
  model for non-ideal mixtures (other activity/EOS models are future drop-ins).
- **Auto-saturate** streams to their bubble-/dew-point temperature.
- **Results**: composition/temperature plots (plus flow/K-value/enthalpy profiles from the
  Inside-Out solver), a component-aware data table, and CSV export.
- **Save / Load** the full configuration to `.colx`.

## Project Structure

```
freeColumn/
├── src/
│   ├── python/
│   │   ├── core/          # thermodynamics, column_solvers, material_balance, dof, data_structures
│   │   ├── gui/           # PySide6 app: main_window + tabs/, panels/, state/, modules/
│   │   ├── side_features/ # bvm/ (the BVM solver) + freeRCM/ (preserved predecessor)
│   │   └── tests/         # headless pytest suite (run with python3.12)
│   └── native/            # C/Fortran sources (nifco.f90, column_solver.c) — not built this round
└── launch.py              # GUI entry point
```

## Testing

The solver/state layers are Qt-free and self-checking. Run the suite (use Python 3.12,
where the project's packages live):

```bash
cd src/python && PYTHONPATH=. python3.12 -m pytest tests/ -q
```

Each core module also has a runnable self-check, e.g. `python3.12 -m core.column_solvers`
or `python3.12 -m core.thermodynamics`.

## Development

See `AGENTS.md` for code style, naming, and docstring conventions, and `progress.md` for
the running change log.

## Not yet built

- Native C/Fortran acceleration (`src/native/` sources exist but aren't compiled or bound).
- Additional activity/EOS models beyond NRTL; a full energy balance / duties in Inside-Out
  (currently CMO flows); a full 3-section extractive BVM.
- Some UI placeholders marked `(#)` (UNIFAC estimation, Pure Components / Phase EQ panels).
