# Finish FreeColumn — Implementation Plan

## Context

The GUI is essentially complete: every tab, panel, and widget exists and lays out
correctly. What's missing is the wiring between the UI and the logic that already
works. Today there are **two disconnected worlds**:

- A **working BVM solver** (`side_features/bvm/solver.py` + `gui/modules/bvm_module.py`)
  that gathers inputs from `window_state`, solves, and plots — but only reachable
  from the **Modules** tab.
- The **main workflow** (Specifications → Simulation → Results) whose Run button is a
  UI-only stub (`main_window.run_simulation`, line 404: *"no solver wired yet"*) and
  whose Results tab is entirely placeholders (no plot, empty table, static summary,
  fake hardcoded CSV export).

A third world, `core/thermodynamics.py` + `core/column_solvers.py` (the rigorous
"HYSIM Inside-Out" path), is 100% stubs/TODOs — a separate numerics project.

**Decisions (from the user):** wire **BVM** into the main flow first and keep going in
a loop toward the full solver; **keep solver knobs in the Modules/BVM widget** (don't
duplicate them into Specifications); **Save/Load is in scope** and must round-trip the
full configuration. The HYSIM core is wanted eventually, staged at the end.

**Outcome:** Run becomes a real solve, Results shows real plots/tables/CSV, configs
save and reload, and the rigorous solver is built incrementally afterward.

Work is staged so each phase is independently shippable and leaves a runnable check —
suitable for looping one phase at a time.

---

## Phase 1 — Make Run actually solve (BVM)

**Goal:** the Simulation Run button drives the existing BVM solver and produces a
result object in shared state.

Reuse, don't rebuild. The solve logic already exists in `bvm_module`
(`_gather_inputs` → `bound_val_method` → `build_column_profile`). Expose it headlessly:

- `gui/modules/bvm_module.py`: add `def solve(self) -> dict` on `BVMModuleWidget` that
  runs `_gather_inputs()` → `bound_val_method()` → `build_column_profile()` and returns
  the profile dict (raises `ValueError` with a user message on incomplete setup, as
  `_gather_inputs` already does). `_on_run`/`_on_build_profile` can call it too, so the
  plot path and the headless path share one implementation.
- `gui/tabs/modules_tab.py`: add `def ensure_bvm(self) -> BVMModuleWidget` that builds
  the widget if `self.bvm_widget is None` (extract from `_launch_bvm`) and returns it,
  without forcing a tab switch. Used by the main Run.
- `gui/state/window_state.py`: add `self.results = None` (init + `clear` + the second
  reset path near line 484). This is the single slot the Results tab reads.
- `gui/main_window.py` `run_simulation` (replace the stub at ~388–405): after the
  existing `_check_specification()` gate, call `self.modules_tab.ensure_bvm().solve()`
  inside try/except. On `ValueError`, show the message in a `QMessageBox.warning` and
  point the user at Modules → BVM to set the knobs (this is the "BVM not configured yet"
  prompt). On success, store the profile in `self.window_state.results`, switch to the
  Results tab (index 3), and call `self.results_tab.update_results(...)` with a
  normalized dict.

Normalize the BVM profile (`{x, y, T, feed_stage, n_stages, n_rect, n_strip, found,
message, intersection, comps}`) into the shape `results_tab.update_results` already
expects (`status`, `stages`, `iterations`, `runtime`, `data`). Keep the raw profile in
`window_state.results` for the richer Results rendering in Phase 2.

`# ponytail:` mark that Abort is cosmetic — BVM solves synchronously in well under a
second, so there's no long-running task to interrupt; real cancellation only matters
once the rigorous solver (Phase 5) runs long.

**Critical files:** `gui/modules/bvm_module.py`, `gui/tabs/modules_tab.py`,
`gui/state/window_state.py`, `gui/main_window.py`.

**Check:** extend the existing `bvm_module._demo()` to also call the new `solve()` and
assert it returns a `found`, multi-stage profile. Runs headless (already does Qt
offscreen via `QApplication([])`).

---

## Phase 2 — Real Results tab

**Goal:** Results renders the profile instead of placeholder labels.

`gui/tabs/results_tab.py`:

- Replace `self.plot_placeholder` (`QLabel`) with a matplotlib `FigureCanvas` — copy the
  pattern already used in `bvm_module` (`Figure` + `backend_qtagg.FigureCanvas`). Plot
  the stage profile: liquid mole fraction `x[:, j]` vs stage per component, plus `T` on a
  twin axis, with the feed stage marked (mirror `bvm_module._plot_profile`'s right panel
  — reuse, don't reinvent).
- Make the table **component-count-aware**: headers are currently hardcoded `x₁/x₂/x₃`.
  Build columns from `result["comps"]` (`Stage, T, <comp names…>`).
- Wire the controls that are currently dead: connect `view_combo` →
  `_on_view_changed` (declared but never connected) and `data_combo` → a redraw, so
  Plot/Table and the data selector actually switch.
- `update_results`: drive the plot + table + summary from `window_state.results`
  (n_stages, feed_stage, found/message, rect/strip split) rather than the placeholder
  text.

`gui/main_window.py` `export_results` (~365–386): replace the hardcoded
`350.0,0.33,0.33,0.34` loop with a real CSV written from `window_state.results`
(`Stage, T, x per component`). Guard for "no results yet".

**Critical files:** `gui/tabs/results_tab.py`, `gui/main_window.py`.

**Check:** a small `test_results_export.py` (or extend the bvm demo): build a profile,
write CSV to a temp path, assert the row count == `n_stages` and the header column
count == `len(comps) + 2`. No Qt needed for the export path if CSV building is factored
into a plain function.

---

## Phase 3 — Save / Load round-trip

**Goal:** `.colx` persists and restores the whole configuration, not just name +
solver_mode.

Today `main_window._do_save` (~349) only sets `config.name` and `config.solver_mode`;
species, streams, condenser/reboiler, modules, specs, pressure are dropped on save and
absent on load. `core/data_structures.py` already has `save_to_file`/`load_from_file`
(JSON) — reuse them; the gap is populating the payload.

- `gui/state/window_state.py`: add `to_dict()` / `load_from_dict()` covering species,
  streams (type, stage, flow, composition), `condenser_config`, `reboiler_config`,
  `modules`, `specs` (the structured `Spec` list), `pressure`, `pressure_drop`,
  `light_key_index`, `solver_mode`, and the BVM knobs if persisting those (store the
  Modules/BVM widget values via `ensure_bvm`, since the knobs live there per the design
  decision). Dataclasses → use `dataclasses.asdict` where the configs are dataclasses;
  `Spec` is a frozen dataclass so it serializes directly.
- `main_window._do_save`: stash `window_state.to_dict()` into the config payload before
  `save_to_file`. `load_config`: after `load_from_file`, call
  `window_state.load_from_dict(...)` then refresh all tabs (`init_tab`, `specs_tab`,
  `modules_tab`, `results_tab.clear_results()`), mirroring `new_config`.

`# ponytail:` if `ColumnConfiguration` can't carry an arbitrary dict, add one
`state: dict` field rather than mapping every attribute onto bespoke config fields —
shortest diff, and the GUI is the source of truth anyway.

**Critical files:** `gui/state/window_state.py`, `gui/main_window.py`,
possibly `core/data_structures.py` (one field).

**Check:** `test_state_roundtrip.py` — build a `WindowState` with 3 species + feed +
condenser/reboiler + a spec, `to_dict()` → `load_from_dict()` into a fresh state, assert
species/streams/specs/pressure equal. Pure logic, no Qt.

---

## Phase 4 — BVM completeness + polish

Smaller gaps that make the BVM path feel finished:

- **Direct spec mode** (`bvm_module._on_run` currently rejects it with *"not wired
  yet"*). `core.material_balance.matbal_direct` already exists — wire `spec_mode="direct"`
  through `_gather_inputs`/`solve` using xD/xB inputs (add the two spins to the BVM panel;
  they belong there per the knobs-stay-in-BVM decision).
- **Preferences dialog** (`main_window.show_preferences`, stub at 412). YAGNI unless
  wanted — a minimal dialog for solver tolerance/max-iter, or delete the menu/toolbar
  entries. Decide at execution; default is to drop it until the rigorous solver needs
  settings.
- **Abort:** leave cosmetic for BVM; revisit in Phase 5.

**Check:** extend `bvm_module._demo()` with a direct-mode `solve()` asserting a feasible
profile.

---

## Phase 5 — Rigorous solver (HYSIM / Inside-Out) — the long loop

This is the large numerics effort, built incrementally so each step is checkable. All of
`core/thermodynamics.py` and `core/column_solvers.py` are stubs today.

1. **Thermo first (reuse what exists).** `side_features/bvm/solver.py` already computes
   Antoine `Psat` and ideal K-values; lift that into `core/thermodynamics.py` as the
   `VLECalculator` Psat/K methods (replace the `1.0` placeholders). Add bubble-point T
   solve (single-stage) with a `scipy.optimize.brentq` root on Σyᵢ−1.
2. **Bubble-Point column method** in `core/column_solvers.py` (the simplest rigorous
   tray-by-tray method) before the full Inside-Out — gives a working rigorous Run with
   far less machinery.
3. **Inside-Out** proper (the `solveColumn` the original TODOs reference): outer loop on
   simplified K/H models, inner loop on MESH. Stage behind the bubble-point method.
4. Wire the **Simulation tab solver combo** (`HYSIM Inside-Out / Newton-Raphson /
   Bubble-Point / Inside-Out`) to dispatch to whichever methods exist; BVM stays as the
   feasibility/preliminary option. Make Abort real here (run the solver in a `QThread` or
   check a cancel flag between outer iterations — `# ponytail:` cancel flag first, thread
   only if the UI actually freezes).

**Check:** each method gets an `assert`-based `__main__` self-check against a known
binary/ternary case (e.g. benzene/toluene) — converged Σx=Σy=1 per stage, monotonic
composition profile. The BVM demo already establishes the reference feasibility.

---

## Cross-cutting verification

- Per-phase: the runnable check listed above (all headless / `assert`-based, no test
  framework — consistent with the existing `_demo()`/`__main__` self-checks in
  `dof.py`, `material_balance.py`, `window_state.py`, `bvm/solver.py`).
- `pyflakes` + `python -m py_compile` on touched files after each phase.
- Manual GUI smoke (user's environment, PySide6 installed):
  `python launch.py` → define species + feed (Initialization), configure BVM (Modules),
  hit **Run** → Results shows a profile → **Export CSV** → **Save**, **New**, **Open**
  restores the configuration.
- `# ponytail:` the offscreen `MainWindow` construct test can't run in this sandbox
  (PySide6 not installed here); the user runs `python launch.py` to confirm the Qt path.

## Deliverable

Per the original request, copy this plan into the project as `FINISH_PLAN.md` (project
root, alongside `CLAUDE.md`) during execution so it lives with the repo — plan mode only
permits editing the plan file itself right now.

## Scope notes (ponytail)

- **Reused, not rebuilt:** the BVM solver, its gather logic, the matplotlib plot pattern,
  `matbal_direct`, `data_structures.save_to_file/load_from_file`, the Antoine/K code for
  the rigorous thermo.
- **Deferred deliberately:** rigorous Inside-Out (Phase 5, staged behind bubble-point),
  Preferences dialog (likely deleted), real Abort/threading (only when a solver runs
  long enough to need it).
- **Shortest-diff bias:** main Run reuses the Modules/BVM widget rather than duplicating
  knobs into Specifications (per the user's decision); Save/Load stuffs one `state` dict
  rather than mapping every field onto bespoke config attributes.
