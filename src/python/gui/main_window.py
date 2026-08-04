#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Window - ColumnForge Column Solver GUI
Tabbed interface with comprehensive simulation workflow

Author: Piero Wemyss
"""

import math
import sys

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QMessageBox, QFileDialog, QInputDialog
)
from PySide6.QtGui import QAction

from .tabs.initialization_tab import InitializationTab
from .tabs.specifications_tab import SpecificationsTab
from .tabs.simulation_tab import SimulationTab
from .tabs.results_tab import ResultsTab
from .tabs.modules_tab import ModulesTab
from .state.window_state import WindowState
from .state.persistence import save_colx, load_colx


class _AbortedResolve(Exception):
    """A trial solve inside the operating-point root-find was cancelled.
    Carries that solve's (partial) profile so the run finishes as an abort."""

    def __init__(self, profile):
        super().__init__("Aborted.")
        self.profile = profile


class MainWindow(QMainWindow):
    """Main application window with tabbed interface for column simulation"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ColumnForge - Column Solver")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)

        self.window_state = WindowState()

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

        self.statusBar().showMessage("Ready")

    def _setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.tab_widget = QTabWidget(self)
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabsClosable(False)

        self.init_tab = InitializationTab(self)
        self.specs_tab = SpecificationsTab(self)
        self.sim_tab = SimulationTab(self)
        self.results_tab = ResultsTab(self)
        self.modules_tab = ModulesTab(self)
        
        # Set window state on tabs
        self.init_tab.set_window_state(self.window_state)
        self.specs_tab.set_window_state(self.window_state)
        self.sim_tab.set_window_state(self.window_state)
        self.modules_tab.set_window_state(self.window_state)
        self.results_tab.set_window_state(self.window_state)

        # Connect species changes to refresh specs tab
        self.init_tab.speciesChanged.connect(self.specs_tab.refresh)

        # Keep the Simulation tab's mirror thermo combos in lock-step with the
        # Initialization tab's (both write the same window_state config).
        self.init_tab.thermoChanged.connect(self.sim_tab.refresh_thermo)
        self.sim_tab.thermoChanged.connect(self.init_tab.refresh_thermo)

        self.tab_widget.addTab(self.init_tab, "Initialization")
        self.tab_widget.addTab(self.specs_tab, "Specifications")
        self.tab_widget.addTab(self.sim_tab, "Simulation")
        self.tab_widget.addTab(self.results_tab, "Results")
        self.tab_widget.addTab(self.modules_tab, "Modules")

        main_layout.addWidget(self.tab_widget)

        self.toolbar = self.addToolBar("Main")
        self.toolbar.setMovable(False)

        from .theme.iconset import icon

        new_act = QAction(icon("new"), "New", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self.new_config)
        self.toolbar.addAction(new_act)

        open_act = QAction(icon("open"), "Open", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self.load_config)
        self.toolbar.addAction(open_act)

        save_act = QAction(icon("save"), "Save", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self.save_config)
        self.toolbar.addAction(save_act)

        self.toolbar.addSeparator()

        run_act = QAction(icon("run"), "Run", self)
        run_act.setShortcut("F5")
        run_act.triggered.connect(self.run_simulation)
        self.toolbar.addAction(run_act)

        self.toolbar.addSeparator()

        prefs_act = QAction(icon("settings"), "Preferences", self)
        prefs_act.triggered.connect(self.show_preferences)
        self.toolbar.addAction(prefs_act)

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        new_act = QAction("New", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self.new_config)
        file_menu.addAction(new_act)

        open_act = QAction("Open...", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self.load_config)
        file_menu.addAction(open_act)

        save_act = QAction("Save", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self.save_config)
        file_menu.addAction(save_act)

        save_as_act = QAction("Save As...", self)
        save_as_act.setShortcut("Ctrl+Shift+S")
        save_as_act.triggered.connect(self.save_config_as)
        file_menu.addAction(save_as_act)

        file_menu.addSeparator()

        export_act = QAction("Export Results...", self)
        export_act.triggered.connect(self.export_results)
        file_menu.addAction(export_act)

        file_menu.addSeparator()

        exit_act = QAction("Exit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # ponytail: no Edit/View menus — Undo/Redo lands with the Month-11 UX
        # pass; text widgets already handle cut/copy/paste natively.
        help_menu = menubar.addMenu("Help")
        about_act = QAction("About ColumnForge", self)
        about_act.triggered.connect(self.show_about)
        help_menu.addAction(about_act)

    def _connect_signals(self):
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        # In-tab Run/Abort buttons (were emitting into the void)
        self.sim_tab.runSimulation.connect(self.run_simulation)
        self.sim_tab.abortSimulation.connect(self.abort_simulation)
        # Results-tab Export CSV = File -> Export Results
        self.results_tab.export_btn.clicked.connect(self.export_results)

    def _on_tab_changed(self, index: int):
        if index == 2:
            can_run = self._check_specification()
            self.sim_tab.set_running(False)
            # gate AFTER set_running, which would otherwise re-enable the button
            self.sim_tab.run_btn.setEnabled(can_run)
        elif index == 4:
            self.modules_tab.refresh()
        self.window_state.current_tab = index

    def _check_specification(self) -> bool:
        icon, message, can_run = self.window_state.get_specification_status()
        self.statusBar().showMessage(message)
        return can_run

    def new_config(self):
        """Create a new column configuration."""
        reply = QMessageBox.question(
            self, "New Configuration",
            "Create a new column configuration? Any unsaved changes will be lost.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.window_state.create_new_column()

            self.init_tab.clear()
            self.specs_tab.clear()
            self.sim_tab.clear()
            self.results_tab.clear_results()

            self.statusBar().showMessage("New column configuration created")
            self.tab_widget.setCurrentIndex(0)

    def load_config(self):
        """Load column configuration from *.colx file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Column Configuration", "",
            "ColumnForge Files (*.colx);;All Files (*)"
        )

        if filename:
            try:
                state = load_colx(filename)
                self.window_state.load_from_dict(state)
                # repopulate every tab from the restored state
                self.init_tab.set_window_state(self.window_state)
                self.specs_tab.set_window_state(self.window_state)
                self.sim_tab.set_window_state(self.window_state)
                self.modules_tab.set_window_state(self.window_state)
                self.results_tab.clear_results()

                self.statusBar().showMessage(f"Loaded {filename}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Load Error",
                    f"Failed to load configuration:\n{str(e)}"
                )

    def save_config(self):
        """Save column configuration to file"""
        if self.window_state.file_path:
            self._do_save(self.window_state.file_path)
        else:
            self.save_config_as()

    def save_config_as(self):
        """Save column configuration to a new file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Column Configuration", "",
            "ColumnForge Files (*.colx);;All Files (*)"
        )

        if filename:
            if not filename.endswith('.colx'):
                filename += '.colx'
            self._do_save(filename)

    def _do_save(self, filepath: str):
        """Perform the actual save operation."""
        try:
            save_colx(filepath, self.window_state.to_dict(),
                      name="ColumnForge Configuration")
            self.window_state.set_file_path(filepath)
            self.statusBar().showMessage(f"Saved to {filepath}")
        except Exception as e:
            QMessageBox.critical(
                self, "Save Error",
                f"Failed to save configuration:\n{str(e)}"
            )

    def export_results(self):
        """Export simulation results to CSV"""
        profile = self.window_state.results
        if not profile:
            QMessageBox.information(
                self, "Export Results",
                "No results yet — run a simulation first."
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if filename:
            if not filename.endswith('.csv'):
                filename += '.csv'

            try:
                import csv
                from .tabs.results_tab import profile_to_csv_rows
                ws = self.window_state
                mws = [getattr(ws.species.get(c), "mw", None)
                       for c in profile.get("comps", [])]
                with open(filename, 'w', newline='') as f:
                    csv.writer(f).writerows(profile_to_csv_rows(
                        profile, units=getattr(ws, "display_units", None), mws=mws))
                self.statusBar().showMessage(f"Results exported to {filename}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error",
                    f"Failed to export results:\n{str(e)}"
                )

    def run_simulation(self):
        """Run the column simulation via the selected rigorous MESH solver."""
        method = self.sim_tab.solver_combo.currentText()
        can_run = self._check_specification()

        if not can_run:
            icon, message, _ = self.window_state.get_specification_status()
            QMessageBox.warning(
                self, "Cannot Run Simulation",
                f"{message}\n\nPlease complete the column specification before running."
            )
            return

        if getattr(self, "_solver_thread", None) and self._solver_thread.isRunning():
            return                                     # one solve at a time

        # Widget reads happen here on the GUI thread; the job itself (gather,
        # operating-point resolution, and the final solve) runs on the worker,
        # which reports its ValueErrors back through _on_solver_failed.
        try:
            job = self._make_solver_job(method)
        except ValueError as exc:
            self.sim_tab.set_running(False)
            self.statusBar().showMessage("Run failed")
            QMessageBox.warning(self, "Cannot Run Simulation", str(exc))
            return
        except Exception:
            import traceback
            self._on_solver_failed("Configuration failed",
                                   traceback.format_exc(), False)
            return
        self.statusBar().showMessage("Solving...")
        self._start_solver(job)

    def _make_solver_job(self, method):
        """A thread-safe callable (report, cancel) -> profile for `method`.
        All Qt widget access happens in here, on the GUI thread. Only the two
        rigorous MESH solvers are dispatched here; BVM/FUG live in Modules."""
        if method == "Bubble-Point" or "Inside-Out" in method:
            from core.column_solvers import solve_bubble_point, solve_inside_out
            solver = (solve_bubble_point if method == "Bubble-Point"
                      else solve_inside_out)
            cfg = self.sim_tab.get_solver_config()      # the only widget reads

            def job(report, cancel):
                # Gather (incl. the operating-point root-find, which re-solves
                # the column for every purity/recovery spec) runs here, off the
                # GUI thread — it used to block the UI for minutes. report/cancel
                # go in too, so the root-find's trial solves drive the progress
                # bar and honour Abort instead of looking hung.
                #
                # Progress is reported in *work units*: the resolve's trial-solve
                # budget, then the final solve. One monotonic sweep either way —
                # reporting the trial solves' own iteration counts made the bar
                # loop 0-100% once per trial.
                stats = {}
                try:
                    si, knobs = self._gather_rigorous_inputs(
                        cfg=cfg, method=method, report=report, cancel=cancel,
                        stats=stats)
                except _AbortedResolve as ab:
                    return ab.profile
                base = stats.get("budget", 0)          # 0 when no column solves
                span = int(knobs["max_iter"])
                total = base + span
                tol = float(knobs["tol"])
                seen = {}

                def final_report(it, res):
                    # Iteration count alone is a bad ruler: both solvers converge
                    # well short of max_iter, so the bar would stall at whatever
                    # fraction they happened to need and then snap to 100. They
                    # converge geometrically, so the *log* residual closing on tol
                    # is the honest measure of how far along we are.
                    r0 = seen.setdefault("r0", max(res, tol * 10.0))
                    frac = 0.0
                    if res > 0.0 and r0 > tol:
                        frac = math.log(r0 / max(res, tol)) / math.log(r0 / tol)
                    done = base + max(it, int(span * min(1.0, max(0.0, frac))))
                    report(min(done, total), total, res)

                warm = stats.get("warm", {})   # last trial's profile, if any
                run = stats.get("solver", solver)   # side-section wrapper, if any
                return run(si, cancel=cancel, report=final_report,
                           x0=warm.get("x"), T0=warm.get("T"), **knobs)
            return job
        raise ValueError(
            f"{method} is not implemented yet — choose Bubble-Point or "
            "Inside-Out.")

    def _start_solver(self, job):
        import time
        from PySide6.QtCore import QThread, QTimer
        from .solver_worker import SolverWorker

        self._solve_t0 = time.monotonic()
        self._solver_iters = 0
        self._solver_pct = 0
        if not hasattr(self, "_elapsed_timer"):
            self._elapsed_timer = QTimer(self)
            self._elapsed_timer.setInterval(250)
            self._elapsed_timer.timeout.connect(self._tick_elapsed)

        self._solver_worker = SolverWorker(job)
        self._solver_thread = QThread(self)
        self._solver_worker.moveToThread(self._solver_thread)
        self._solver_thread.started.connect(self._solver_worker.run)
        self._solver_worker.progress.connect(self._on_solver_progress)
        self._solver_worker.finished.connect(self._on_solver_finished)
        self._solver_worker.failed.connect(self._on_solver_failed)
        for sig in (self._solver_worker.finished, self._solver_worker.failed):
            sig.connect(self._solver_thread.quit)
        self._solver_thread.finished.connect(self._solver_worker.deleteLater)
        self.sim_tab.set_running(True)
        self._elapsed_timer.start()
        self._solver_thread.start()

    def _elapsed(self):
        import time
        return time.monotonic() - getattr(self, "_solve_t0", time.monotonic())

    def _tick_elapsed(self):
        self.sim_tab.set_progress(self.sim_tab.progress_bar.value(),
                                  self._solver_iters, self._elapsed())

    def _on_solver_progress(self, done, total, residual):
        # (done, total) are the job's work units: for a run whose specs need an
        # operating-point resolve they span the trial solves *and* the final
        # solve, so the bar sweeps once, forward, over the whole run.
        self._solver_iters = done
        # A residual that ticks back up must not walk the bar backwards.
        pct = max(self._solver_pct, min(99, int(100 * done / max(1, total))))
        self._solver_pct = pct
        self.sim_tab.set_progress(pct, done, self._elapsed())

    def _on_solver_finished(self, profile):
        self._elapsed_timer.stop()
        self.window_state.results = profile
        self.sim_tab.set_running(False)
        self.sim_tab.set_progress(100, profile.get("iterations",
                                                   self._solver_iters),
                                  self._elapsed())
        self.sim_tab.set_status(profile.get("message", "Solved"))
        warns = self._antoine_range_warnings(profile)
        summary = self._normalize_results(profile)
        if warns:
            summary["status"] += "  |  WARNING: " + "; ".join(warns)
        self.results_tab.update_results(summary)
        self.tab_widget.setCurrentIndex(3)
        msg = (f"Solved: {profile['n_stages']} stages, "
               f"feed at stage {profile['feed_stage']}.")
        if warns:
            msg += "  ⚠ Antoine fit used outside its range — see Results status."
        self.statusBar().showMessage(msg)

    def _on_solver_failed(self, message, tb, user_error):
        if hasattr(self, "_elapsed_timer"):
            self._elapsed_timer.stop()
        self.sim_tab.set_running(False)
        self.statusBar().showMessage("Run failed")
        self.sim_tab.set_status("Run failed", is_error=True)
        if user_error:
            QMessageBox.warning(self, "Cannot Run Simulation", message)
        else:
            # Solver bugs (LinAlgError, KeyError, ...) must not crash the app.
            import logging
            logging.getLogger(__name__).error("Simulation run failed\n%s", tb)
            box = QMessageBox(QMessageBox.Critical, "Run Failed",
                              f"The solver raised an unexpected error:\n"
                              f"{message}", parent=self)
            box.setDetailedText(tb)
            box.exec()

    def _antoine_range_warnings(self, profile):
        """Species whose Antoine fit was evaluated outside its validity range
        by the solved temperature profile (degC). Empty when ranges are unknown
        or the vle_model isn't Antoine."""
        tc = self.window_state.thermodynamics_config
        T = profile.get("T")
        if tc.vle_model != "Antoine" or T is None or len(T) == 0:
            return []
        tlo, thi = float(min(T)), float(max(T))
        warns = []
        for name in self.window_state.species:
            p = tc.component_params.get(name)
            if p is None or p.antoine_tmin is None or p.antoine_tmax is None:
                continue
            if tlo < p.antoine_tmin - 0.5 or thi > p.antoine_tmax + 0.5:
                warns.append(
                    f"{name}: column T {tlo:.0f}–{thi:.0f} °C exceeds "
                    f"Antoine fit range {p.antoine_tmin:.0f}–"
                    f"{p.antoine_tmax:.0f} °C")
        return warns

    @staticmethod
    def _normalize_results(profile: dict) -> dict:
        """Map a column profile to the ResultsTab.update_results summary schema.
        Profiles are top -> bottom; stage numbers are 0-based from the top."""
        x, T = profile["x"], profile["T"]
        rows = [
            [i, round(float(T[i]), 2)] + [round(float(v), 4) for v in x[i]]
            for i in range(profile["n_stages"])
        ]
        return {
            "status": profile.get("message", "Solved"),
            "stages": profile["n_stages"],
            "iterations": profile.get("iterations", "—"),  # BVM marches directly
            "runtime": "< 1 s",
            "data": rows,
        }

    def _gather_rigorous_inputs(self, cfg=None, method=None,
                                report=None, cancel=None, stats=None):
        """Build the canonical SolverInput for the rigorous solvers from
        window_state + the Simulation tab (Phase 0: one config path).

        `cfg`/`method` are the Simulation tab's knobs; pass them in (read on the
        GUI thread) to run this whole gather — including the operating-point
        root-find, which can cost dozens of column solves — on a worker thread.
        Omit them and they're read from the widgets here, for headless callers.
        `report`/`cancel` are the worker's hooks: the root-find ticks `report`
        once per trial solve (so a long resolve shows progress) and raises
        _AbortedResolve when a trial is cancelled. `stats`, if given, comes back
        carrying the resolve's progress budget so the caller can continue the
        same monotonic sweep through the final solve — and `stats["solver"]`, the
        solver callable the final run must use (the plain solver, or one wrapped
        to converge the side-section tear).

        Every configurable value flows through here: all feed streams (stage,
        flow, composition, thermal quality from the entered temperature), side
        draws, the pressure profile from top pressure + per-stage drop, the
        condenser type, and the activity model. Raises ValueError (with a
        user-facing message) when the setup is incomplete.
        Returns (solver_input, solver_knobs_dict).
        """
        import numpy as np
        from .state.window_state import StreamType, CondenserType

        ws = self.window_state
        order = ws.get_species_names()
        if len(order) < 2:
            raise ValueError("Need at least 2 species (Initialization tab).")

        N = int(ws.num_stages)

        def _stage_internal(gui_stage, what):
            """GUI stages are 0-based from the top; solvers count 1=top."""
            s = int(gui_stage)
            if not (0 <= s <= N - 1):
                raise ValueError(
                    f"{what} stage {s} is outside the column (0..{N - 1}, "
                    "0 = distillate).")
            return s + 1

        feeds = []
        for s in ws.streams.values():
            if s.stream_type != StreamType.FEED:
                continue
            if not s.flow or not s.composition:
                raise ValueError(f"Feed '{s.id}' needs a flow rate and composition.")
            z = np.array([s.composition.get(nm, 0.0) for nm in order], float)
            if abs(z.sum() - 1.0) > 0.05 or z.sum() <= 0.0:
                raise ValueError(
                    f"Feed '{s.id}' composition sums to {z.sum():.4f}, not 1 — "
                    "fix it on the Streams page.")
            if abs(z.sum() - 1.0) > 1e-6:
                # normalize-on-solve for near-1 sums, and say so
                self.statusBar().showMessage(
                    f"Feed '{s.id}' composition summed to {z.sum():.4f}; "
                    "normalized to 1 for this run.")
                z = z / z.sum()
            q = ws.feed_quality(s, order)
            feeds.append((_stage_internal(s.stage if s.stage is not None else 10,
                                          f"Feed '{s.id}'"),
                          float(s.flow), z, q))
        if not feeds:
            raise ValueError("At least one feed stream is required.")

        draws = []
        for s in ws.streams.values():
            if s.stream_type == StreamType.SIDESTREAM and s.flow:
                stage = _stage_internal(s.stage if s.stage is not None else 10,
                                        f"Side draw '{s.id}'")
                flow = float(s.flow)
                if getattr(s, "phase", "liquid") == "vapor":
                    draws.append((stage, 0.0, flow))
                else:
                    draws.append((stage, flow, 0.0))
        W = sum(d[1] + d[2] for d in draws)

        antoine = ws.thermodynamics_config.psat_params(order)
        # window_state pressures are bar; the thermo layer works in the Psat
        # fit's unit. pressure_drop is per stage, growing top -> bottom, and the
        # returned profile is 1=top .. N=bottom (solver-internal ordering).
        to_unit = ws.thermodynamics_config.pressure_in_psat_unit
        P_top = to_unit(ws.pressure)
        dP = to_unit(ws.pressure_drop) if ws.pressure_drop else 0.0
        pressure = P_top + dP * np.arange(N)

        gamma_fn = ws.build_gamma_fn(order)
        phi_fn = ws.build_phi_fn(order)
        flows_hook = ws.build_energy_hook(order)   # None unless energy_balance on
        condenser = ws.condenser_config.condenser_type.value.lower()
        fixed_R = 0.0 if ws.condenser_config.condenser_type == CondenserType.NONE else None

        # Resolve the operating specs the user set (reflux, boilup, rates, a
        # product purity, a key recovery, ...) down to (R, D) — Aspen-style.
        # Side-draw rates are their own answer and don't enter the root-find.
        from core.operating_specs import resolve_operating_point
        from core.dof import OPERATING_KINDS, SpecKind
        from core.column_solvers import solve_bubble_point, solve_inside_out
        from core.solver_input import build_solver_input
        ops = [s for s in ws.collect_specs() if s.kind in OPERATING_KINDS
               or s.kind == SpecKind.SIDEDRAW_RATE]
        # Duty specs are entered in kW; the resolver compares them to the energy
        # balance's kJ/h duties — convert (kJ/h = kW / KJH_TO_KW).
        from dataclasses import replace as _replace
        from core.thermodynamics import KJH_TO_KW
        from core.dof import ENERGY_ONLY
        ops = [_replace(s, value=s.value / KJH_TO_KW) if s.kind in ENERGY_ONLY else s
               for s in ops]
        n_free = 1 if fixed_R is not None else 2
        n_ops = sum(1 for s in ops if s.kind != SpecKind.SIDEDRAW_RATE)
        if n_ops != n_free:
            raise ValueError(
                f"This column needs exactly {n_free} operating spec(s) besides "
                "side-draw rates (e.g. reflux ratio + a distillate rate, purity, "
                f"or recovery). You have {n_ops} — see the Specifications DoF "
                "status.")

        # Subcooling ΔT (total condenser) — only the energy balance consumes it;
        # a delta, so °C/K units coincide (see condenser panel note).
        subcool = float(ws.condenser_config.subcooling_temp or 0.0)

        # Interheater/intercooler modules → per-stage duty (kW entered → kJ/h).
        # These are known heat terms in the energy balance (si.duty[]); ignored
        # under CMO, so require the energy balance rather than silently drop them.
        duties = [(_stage_internal(gs, "Interheater"), q_kw / KJH_TO_KW)
                  for gs, q_kw in ws.interheater_duties()]
        if duties and flows_hook is None:
            raise ValueError(
                "Interheater/intercooler duties need the energy balance "
                "(Initialization → Flow Model). Under constant molar overflow "
                "they would be silently ignored.")

        # Pumparounds: (draw, return, rate, duty). Stages GUI 0-based -> solver
        # 1-based; duty kW -> kJ/h. The cooling is an energy-balance term (folded
        # into si.duty at build), so it needs the energy balance like interheaters.
        pumparounds = [(_stage_internal(ds, "Pumparound draw"),
                        _stage_internal(rs, "Pumparound return"),
                        rate, q_kw / KJH_TO_KW)
                       for ds, rs, rate, q_kw in ws.pumparounds()]
        if pumparounds and flows_hook is None:
            raise ValueError(
                "Pumparound cooling needs the energy balance (Initialization → "
                "Flow Model). Under constant molar overflow the duty is ignored.")

        # Side strippers/rectifiers: a real draw plus a torn return feed. They
        # work under CMO (their ratio spec sets the split), so no energy-balance
        # guard. F_total/z_mixed below stay the *external* feed — the return is a
        # recycle and must not enter a recovery/purity denominator.
        from core.side_sections import SideSection, make_side_solver
        sections = [
            SideSection(id=mid, kind=kind,
                        draw_stage=_stage_internal(ds, f"'{mid}' draw"),
                        return_stage=_stage_internal(rs, f"'{mid}' return"),
                        rate=rate, ratio=ratio, n_stages=nst)
            for mid, kind, ds, rs, rate, ratio, nst in ws.side_sections()]
        for s in sections:
            # Both ends must be real trays: stage 1 is the condenser and stage N
            # the reboiler, neither of which can host a section draw or return.
            if not (2 <= s.draw_stage <= N - 1 and 2 <= s.return_stage <= N - 1):
                raise ValueError(
                    f"'{s.id}': draw and return stages must be interior trays "
                    f"(1 to {N - 2} in Stage-0-is-distillate numbering).")

        F_total = sum(f[1] for f in feeds)
        z_mixed = sum(f[1] * f[2] for f in feeds) / F_total
        # Section draw leaves the column, section return comes back in: the net
        # removal is the side product, so B = F_external - D - W still closes.
        W += sum(s.product_flow for s in sections)

        def _build_si(R, D):
            si_feeds = list(feeds) + [
                (s.return_stage, s.return_flow, s.return_comp, s.return_q)
                for s in sections if s.return_comp is not None]
            si_draws = list(draws) + [(s.draw_stage, *s.draw_rates())
                                      for s in sections]
            return build_solver_input(
                n_stages=N, comps=order, feeds=si_feeds, draws=si_draws,
                duties=duties,
                pumparounds=pumparounds, R=R, D=D, pressure=pressure,
                antoine=antoine, gamma_fn=gamma_fn, phi_fn=phi_fn,
                condenser=condenser, subcooling=subcool)

        if cfg is None:
            cfg = self.sim_tab.get_solver_config()   # honor the Simulation tab knobs
        knobs = dict(
            max_iter=int(cfg["max_iterations"]), tol=float(cfg["tolerance"]),
            efficiency=float(getattr(ws, "stage_efficiency", 1.0)))

        # Resolve implicit specs with the SAME efficiency and solver as the
        # final run — an operating point found for an E=1 column misses purity
        # targets when the real column runs at E<1.
        if method is None:
            method = self.sim_tab.solver_combo.currentText()
        is_inside_out = "Inside-Out" in method
        rigorous = solve_inside_out if is_inside_out else solve_bubble_point
        # The energy balance is an Inside-Out feature (its flows_hook seam); the
        # Wang-Henke bubble-point path stays CMO. Fold the hook into knobs only
        # when it applies, so both operating-point resolution and the final run
        # use it consistently.
        if is_inside_out and flows_hook is not None:
            knobs["flows_hook"] = flows_hook

        # Side sections: converge the return tear inside every solve, so the
        # operating-point root-find sees the real column instead of one whose
        # sections do nothing. The sections keep their torn composition between
        # calls, so only the first trial pays the full tear.
        if sections:
            rigorous = make_side_solver(
                rigorous, sections, lambda si: _build_si(si.R, si.D))
        if stats is not None:
            stats["solver"] = rigorous       # the final run must use this one

        # Each trial solve of the root-find is a full column solve. Report one tick
        # per trial (not per inner iteration — that swept the bar 0-100% once per
        # trial and queued thousands of cross-thread signals), and turn a cancelled
        # trial into a real abort: least_squares would otherwise happily keep
        # root-finding on the half-solved profiles an aborted solve returns.
        # ponytail: RESOLVE_BUDGET is a nominal work unit (least_squares' 60-eval
        # cap plus its finite differences), not a promise; the bar clamps, so a
        # long resolve parks near the hand-off instead of overrunning.
        RESOLVE_BUDGET = 120
        n_solves = [0]
        warm = {}                # previous trial's profile; neighbouring (R, D)
                                 # differ by a finite-difference step, so it skips
                                 # the slow front-placement phase of a cold start
        if stats is not None:
            stats["warm"] = warm     # the final solve warm-starts from it too

        def _trial_solve(R, D):
            n_solves[0] += 1
            if stats is not None:
                stats["budget"] = RESOLVE_BUDGET
            prof = rigorous(_build_si(R, D), cancel=cancel,
                            x0=warm.get("x"), T0=warm.get("T"), **knobs)
            if prof.get("message") == "Aborted.":
                raise _AbortedResolve(prof)
            if float(prof.get("residual", math.inf)) < 1e-2:
                # only a (near-)converged profile is a good seed — warm-starting
                # from a chaotic max-iter trial poisons every trial after it
                warm["x"], warm["T"] = prof["x"], prof["T"]
            if report is not None:
                report(min(n_solves[0], RESOLVE_BUDGET),
                       RESOLVE_BUDGET + int(knobs["max_iter"]),
                       float(prof.get("residual", 0.0)))
            return prof

        R, D = resolve_operating_point(
            ops, F_total, z_mixed, solve_fn=_trial_solve,
            lk=ws.light_key_index, hk=ws.heavy_key_index,
            side_draw_total=W, fixed_R=fixed_R)

        return _build_si(float(R), float(D)), knobs

    def _solve_bubble_point(self) -> dict:
        """Run the rigorous bubble-point (Wang-Henke) solver."""
        from core.column_solvers import solve_bubble_point
        stats = {}
        si, knobs = self._gather_rigorous_inputs(method="Bubble-Point",
                                                 stats=stats)
        return stats.get("solver", solve_bubble_point)(si, **knobs)

    def _solve_inside_out(self) -> dict:
        """Run the Inside-Out solver, with the Abort flag as cancel hook."""
        from core.column_solvers import solve_inside_out
        stats = {}
        si, knobs = self._gather_rigorous_inputs(method="Inside-Out",
                                                 stats=stats)
        self._abort_flag = False
        return stats.get("solver", solve_inside_out)(
            si, **knobs, cancel=lambda: getattr(self, "_abort_flag", False))

    def abort_simulation(self):
        """Abort a running simulation: flips the worker's cancel flag, which the
        solvers check every (outer) iteration. The run then finishes with an
        'Aborted.' profile through the normal finished path."""
        self._abort_flag = True                        # legacy direct-call path
        worker = getattr(self, "_solver_worker", None)
        if worker is not None:
            worker.cancel()
        self.statusBar().showMessage("Aborting…")

    def show_preferences(self):
        """Edit the default rigorous-solver iteration limit and tolerance, which
        are the Simulation tab's solver knobs.
        # ponytail: QInputDialog instead of a bespoke dialog — two values, two
        # prompts; build a proper form only if Preferences grows more fields."""
        sim = self.sim_tab
        it, ok = QInputDialog.getInt(
            self, "Preferences", "Max solver iterations:",
            sim.max_iter_spin.value(), sim.max_iter_spin.minimum(),
            sim.max_iter_spin.maximum())
        if not ok:
            return
        tol, ok = QInputDialog.getDouble(
            self, "Preferences", "Convergence tolerance:",
            sim.tolerance_spin.value(), sim.tolerance_spin.minimum(),
            sim.tolerance_spin.maximum(), sim.tolerance_spin.decimals())
        if not ok:
            return
        sim.max_iter_spin.setValue(it)
        sim.tolerance_spin.setValue(tol)
        self.statusBar().showMessage("Solver preferences updated.")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About ColumnForge",
            "ColumnForge - Column Solver\n\n"
            "A comprehensive GUI-based column solver for chemical "
            "engineering applications.\n\n"
            "Version 1.0.0\n"
            "Author: Piero Wemyss"
        )

    def _stop_solver_thread(self):
        """Cancel and join a running solve. Qt deletes the worker with the window;
        if the thread is still in run() at that point its next emit hits a dead
        C++ object and takes the process down with it."""
        thread = getattr(self, "_solver_thread", None)
        if thread is None or not thread.isRunning():
            return
        self.abort_simulation()
        thread.quit()
        thread.wait(10000)

    def closeEvent(self, event):
        """Handle close event with unsaved changes check"""
        if self.window_state.is_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before exiting?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                self.save_config()
            elif reply != QMessageBox.Discard:
                event.ignore()
                return
        self._stop_solver_thread()      # never leave a solve running into teardown
        event.accept()


def _setup_logging():
    """Log to ~/.columnforge/columnforge.log (roadmap Month 3): solver failures
    from the Run handler land here with tracebacks."""
    import logging
    import logging.handlers
    import os
    log_dir = os.path.join(os.path.expanduser("~"), ".columnforge")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "columnforge.log"),
        maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logging.getLogger(__name__).info("ColumnForge started")


def _install_excepthook():
    """Keep an unhandled exception in a Qt slot from aborting the process.

    Qt calls slots from C++, so an exception escaping one doesn't unwind into
    main() — the default hook prints and the app dies with no dialog. Log it
    and stay alive instead; the run handler already does this for solver
    failures, this is the same treatment for everything else.
    """
    import logging
    import traceback

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.getLogger(__name__).error("Unhandled exception\n%s", text)
        try:
            box = QMessageBox()
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle("Unexpected error")
            box.setText("Something went wrong, but ColumnForge is still running.\n"
                        "Details are in ~/.columnforge/columnforge.log.")
            box.setDetailedText(text)
            box.exec()
        except Exception:
            pass          # a failed dialog must not re-enter the hook

    sys.excepthook = hook


def main():
    _setup_logging()
    _install_excepthook()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    from .theme import load_theme
    from .theme.mpl_style import apply as apply_mpl_style
    app.setStyleSheet(load_theme())      # one app-wide dark theme
    apply_mpl_style()                    # matplotlib figures match the shell
    from .table_edit import install as install_table_edit
    app._table_assist = install_table_edit(app)   # Ctrl+C/V + editor commits

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
