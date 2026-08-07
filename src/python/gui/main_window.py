#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Window - ColumnForge Column Solver GUI
Tabbed interface with comprehensive simulation workflow

Author: Piero Wemyss
"""

import sys

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QMessageBox, QFileDialog
)
from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QDesktopServices

from .tabs.initialization_tab import InitializationTab
from .tabs.specifications_tab import SpecificationsTab
from .tabs.simulation_tab import SimulationTab
from .tabs.results_tab import ResultsTab
from .tabs.modules_tab import ModulesTab
from .state.window_state import WindowState
from .state.persistence import save_colx, load_colx


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
        # The Specifications tab owns which column is active; the Simulation
        # tab's per-column method override has to follow it.
        self.specs_tab.specsChanged.connect(self.sim_tab.refresh_columns)

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
        """Export simulation results to CSV.

        With more than one column every profile goes into the one file, each
        block preceded by its column id — a per-column file set would leave the
        user reassembling a flowsheet by hand.
        """
        ws = self.window_state
        results = ws.results or {}
        if not results:
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
                units = getattr(ws, "display_units", None)
                rows = []
                for cid, profile in results.items():
                    mws = [getattr(ws.species.get(c), "mw", None)
                           for c in profile.get("comps", [])]
                    if len(results) > 1:
                        if rows:
                            rows.append([])
                        rows.append([f"Column: {cid}"])
                    rows.extend(profile_to_csv_rows(profile, units=units, mws=mws))
                with open(filename, 'w', newline='') as f:
                    csv.writer(f).writerows(rows)
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
            cfg = self.sim_tab.get_solver_config()      # the only widget reads

            def job(report, cancel):
                # The gather AND every column's operating-point root-find run
                # here, off the GUI thread — they used to block the UI for
                # minutes. report/cancel go in too, so a long solve shows
                # progress and honours Abort instead of looking hung.
                #
                # Progress is one work unit per unit-solve, one monotonic sweep
                # over the whole flowsheet. Per-MESH-iteration ticks are
                # deliberately not forwarded: doing that swept the bar 0-100%
                # once per trial and queued thousands of cross-thread signals.
                return self._solve_flowsheet(cfg=cfg, method=method,
                                             report=report, cancel=cancel)
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

    def _on_solver_finished(self, result):
        """`result` is a core.flowsheet.FlowsheetResult — one entry per column."""
        self._elapsed_timer.stop()
        ws = self.window_state
        ws.results = {cid: ur.profile for cid, ur in result.units.items()}
        ws.flowsheet_result = result          # streams, products, tear residual
        profile = self._active_profile(result)
        self.sim_tab.set_running(False)
        self.sim_tab.set_progress(100, profile.get("iterations",
                                                   self._solver_iters),
                                  self._elapsed())
        self.sim_tab.set_status(result.message or profile.get("message", "Solved"))
        warns = self._antoine_range_warnings(profile)
        summary = self._normalize_results(profile)
        summary["status"] = result.message or summary["status"]
        if warns:
            summary["status"] += "  |  WARNING: " + "; ".join(warns)
        self.results_tab.set_flowsheet_result(result)
        self.results_tab.update_results(summary)
        self.tab_widget.setCurrentIndex(3)
        if len(result.units) > 1:
            msg = (f"Solved {len(result.units)} columns"
                   + (f", recycle tear {result.tear_residual:.1e} after "
                      f"{result.tear_passes} passes" if result.tear_ids else "")
                   + ".")
        else:
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

    def _gather_flowsheet(self, cfg=None, method=None):
        """Project window_state onto a core.flowsheet.Flowsheet. No solves.

        This is pure translation: every column becomes a Unit, every
        ws.connections entry a Connection, and the flowsheet-global species and
        thermodynamics become the closures every Unit shares. The
        operating-point root-find that used to live here now runs inside
        `core.flowsheet.solve_flowsheet`, once per unit per tear pass, because a
        column's inlet — and therefore the (R, D) that meets its specs — moves
        while a recycle converges.

        Splitting it that way also makes the gather headlessly testable, which
        it was not while it took dozens of column solves to return.

        `cfg`/`method` are the Simulation tab's knobs; pass them in (read on the
        GUI thread) so the solve can run on a worker thread. Omit them and
        they're read from the widgets here, for headless callers.

        Every configurable value flows through here: all feed streams (stage,
        flow, composition, thermal quality from the entered temperature), side
        draws, the pressure profile from top pressure + per-stage drop, the
        condenser type, and the activity model. Raises ValueError (with a
        user-facing message) when the setup is incomplete.
        Returns (flowsheet, solver_knobs_dict).
        """
        import numpy as np
        from core.flowsheet import Flowsheet, Unit
        from core.dof import OPERATING_KINDS, SpecKind, ENERGY_ONLY
        from core.side_sections import SideSection
        from core.thermodynamics import KJH_TO_KW
        from dataclasses import replace as _replace
        from .state.window_state import StreamType, CondenserType

        ws = self.window_state
        order = ws.get_species_names()
        if len(order) < 2:
            raise ValueError("Need at least 2 species (Initialization tab).")

        antoine = ws.thermodynamics_config.psat_params(order)
        gamma_fn = ws.build_gamma_fn(order)
        phi_fn = ws.build_phi_fn(order)
        flows_hook = ws.build_energy_hook(order)   # None unless energy_balance on
        to_unit = ws.thermodynamics_config.pressure_in_psat_unit

        # Which columns are fed by another column: a brand-new column's empty
        # default Feed stream is not an error if a connection supplies it.
        fed_by = {c.dst for c in ws.connections}

        units = {}
        for cid, col in ws.columns.items():
            N = int(col.num_stages)

            def _stage_internal(gui_stage, what, _N=N, _cid=cid):
                """GUI stages are 0-based from the top; solvers count 1=top."""
                s = int(gui_stage)
                if not (0 <= s <= _N - 1):
                    raise ValueError(
                        f"{_cid}: {what} stage {s} is outside the column "
                        f"(0..{_N - 1}, 0 = distillate).")
                return s + 1

            feeds = []
            for s in col.streams.values():
                if s.stream_type != StreamType.FEED:
                    continue
                if not s.flow and not s.composition and cid in fed_by:
                    continue          # supplied by a connection, not by hand
                if not s.flow or not s.composition:
                    raise ValueError(
                        f"{cid}: feed '{s.id}' needs a flow rate and composition.")
                z = np.array([s.composition.get(nm, 0.0) for nm in order], float)
                if abs(z.sum() - 1.0) > 0.05 or z.sum() <= 0.0:
                    raise ValueError(
                        f"{cid}: feed '{s.id}' composition sums to {z.sum():.4f}, "
                        "not 1 — fix it on the Streams page.")
                if abs(z.sum() - 1.0) > 1e-6:
                    # normalize-on-solve for near-1 sums, and say so
                    self.statusBar().showMessage(
                        f"Feed '{s.id}' composition summed to {z.sum():.4f}; "
                        "normalized to 1 for this run.")
                    z = z / z.sum()
                q = ws.feed_quality(s, order, col=col)
                feeds.append((_stage_internal(s.stage if s.stage is not None else 10,
                                              f"feed '{s.id}'"),
                              float(s.flow), z, q))
            if not feeds and cid not in fed_by:
                raise ValueError(
                    f"{cid}: at least one feed stream is required (or a "
                    "connection from another column).")

            # Side draws keep their stream id as their flowsheet port key, so a
            # connection drawn from one survives the stream being renamed.
            draws = []
            for s in col.streams.values():
                if s.stream_type == StreamType.SIDESTREAM and s.flow:
                    stage = _stage_internal(s.stage if s.stage is not None else 10,
                                            f"side draw '{s.id}'")
                    flow = float(s.flow)
                    if getattr(s, "phase", "liquid") == "vapor":
                        draws.append((s.id, stage, 0.0, flow))
                    else:
                        draws.append((s.id, stage, flow, 0.0))

            # window_state pressures are bar; the thermo layer works in the Psat
            # fit's unit. pressure_drop is per stage, growing top -> bottom, and
            # the profile is 1=top .. N=bottom (solver-internal ordering).
            P_top = to_unit(col.pressure)
            dP = to_unit(col.pressure_drop) if col.pressure_drop else 0.0
            pressure = P_top + dP * np.arange(N)

            # Interheater/intercooler modules → per-stage duty (kW → kJ/h).
            # These are known heat terms in the energy balance (si.duty[]);
            # ignored under CMO, so require the energy balance rather than
            # silently drop them.
            duties = [(_stage_internal(gs, "interheater"), q_kw / KJH_TO_KW)
                      for gs, q_kw in col.interheater_duties()]
            if duties and flows_hook is None:
                raise ValueError(
                    f"{cid}: interheater/intercooler duties need the energy "
                    "balance (Initialization → Flow Model). Under constant "
                    "molar overflow they would be silently ignored.")

            # Pumparounds: (draw, return, rate, duty). Stages GUI 0-based ->
            # solver 1-based; duty kW -> kJ/h. The cooling is an energy-balance
            # term (folded into si.duty at build), so it needs the energy
            # balance like interheaters.
            pumparounds = [(_stage_internal(ds, "pumparound draw"),
                            _stage_internal(rs, "pumparound return"),
                            rate, q_kw / KJH_TO_KW)
                           for ds, rs, rate, q_kw in col.pumparounds()]
            if pumparounds and flows_hook is None:
                raise ValueError(
                    f"{cid}: pumparound cooling needs the energy balance "
                    "(Initialization → Flow Model). Under constant molar "
                    "overflow the duty is ignored.")

            # Side strippers/rectifiers: a real draw plus a torn return feed.
            # They work under CMO (their ratio spec sets the split), so no
            # energy-balance guard. Their tear nests inside the flowsheet's.
            sections = [
                SideSection(id=mid, kind=kind,
                            draw_stage=_stage_internal(ds, f"'{mid}' draw"),
                            return_stage=_stage_internal(rs, f"'{mid}' return"),
                            rate=rate, ratio=ratio, n_stages=nst)
                for mid, kind, ds, rs, rate, ratio, nst in col.side_sections()]
            for s in sections:
                # Both ends must be real trays: stage 1 is the condenser and
                # stage N the reboiler, neither of which can host a draw or
                # a return.
                if not (2 <= s.draw_stage <= N - 1 and 2 <= s.return_stage <= N - 1):
                    raise ValueError(
                        f"{cid}: '{s.id}' draw and return stages must be interior "
                        f"trays (1 to {N - 2} in Stage-0-is-distillate numbering).")

            # Operating specs, with duty specs converted kW -> kJ/h to match the
            # energy balance's units, exactly as the single-column path did.
            ops = [s for s in col.collect_specs()
                   if s.kind in OPERATING_KINDS or s.kind == SpecKind.SIDEDRAW_RATE]
            ops = [_replace(s, value=s.value / KJH_TO_KW)
                   if s.kind in ENERGY_ONLY else s for s in ops]
            n_free = 1 if col.condenser_config.condenser_type == CondenserType.NONE else 2
            n_ops = sum(1 for s in ops if s.kind != SpecKind.SIDEDRAW_RATE)
            if n_ops != n_free:
                raise ValueError(
                    f"{cid} needs exactly {n_free} operating spec(s) besides "
                    "side-draw rates (e.g. reflux ratio + a distillate rate, "
                    f"purity, or recovery). It has {n_ops} — see the "
                    "Specifications DoF status.")

            units[cid] = Unit(
                id=cid, n_stages=N, pressure=pressure, antoine=antoine,
                specs=[s for s in ops if s.kind in OPERATING_KINDS],
                feeds=feeds, draws=draws, duties=duties,
                pumparounds=pumparounds, sections=sections,
                condenser=col.condenser_config.condenser_type.value.lower(),
                # Subcooling ΔT (total condenser) — only the energy balance
                # consumes it; a delta, so °C/K units coincide.
                subcooling=float(col.condenser_config.subcooling_temp or 0.0),
                efficiency=float(col.stage_efficiency or 1.0),
                method=col.method, flows_hook=flows_hook,
                lk=col.light_key_index, hk=col.heavy_key_index,
                node_pos=col.node_pos)

        if cfg is None:
            cfg = self.sim_tab.get_solver_config()   # honor the Simulation tab knobs
        knobs = dict(max_iter=int(cfg["max_iterations"]), tol=float(cfg["tolerance"]))
        if method is None:
            method = self.sim_tab.solver_combo.currentText()

        fs = Flowsheet(units=units, connections=list(ws.connections),
                       comps=list(order), default_method=method,
                       gamma_fn=gamma_fn, phi_fn=phi_fn)
        return fs, knobs

    def _solve_flowsheet(self, cfg=None, method=None, report=None, cancel=None):
        """Gather and solve every column, converging any recycle. The one path —
        a single column is a one-unit flowsheet with nothing torn."""
        from core.flowsheet import solve_flowsheet
        fs, knobs = self._gather_flowsheet(cfg=cfg, method=method)
        return solve_flowsheet(fs, knobs=knobs, report=report, cancel=cancel)

    def _gather_rigorous_inputs(self, cfg=None, method=None,
                                report=None, cancel=None, stats=None):
        """The active column's resolved SolverInput + knobs.

        Kept for callers that want one column's solver input rather than a whole
        flowsheet result. It now goes through the flowsheet, so the answer is
        the same one a run produces — there is no second configuration path.
        """
        fs, knobs = self._gather_flowsheet(cfg=cfg, method=method)
        res = self._run_flowsheet(fs, knobs, report=report, cancel=cancel)
        cid = self.window_state.active_column_id
        ur = res.units.get(cid)
        if ur is None:
            raise ValueError(f"No result for column '{cid}'.")
        if stats is not None:
            stats["result"] = res
        return ur.si, knobs

    def _run_flowsheet(self, fs, knobs, report=None, cancel=None):
        from core.flowsheet import solve_flowsheet
        return solve_flowsheet(fs, knobs=knobs, report=report, cancel=cancel)

    def _active_profile(self, res):
        """The profile the single-column callers expect out of a flowsheet run."""
        cid = self.window_state.active_column_id
        ur = res.units.get(cid) or next(iter(res.units.values()), None)
        if ur is None:
            raise ValueError("The flowsheet produced no results.")
        return ur.profile

    def _solve_bubble_point(self) -> dict:
        """Run the rigorous bubble-point (Wang-Henke) solver."""
        return self._active_profile(self._solve_flowsheet(method="Bubble-Point"))

    def _solve_inside_out(self) -> dict:
        """Run the Inside-Out solver, with the Abort flag as cancel hook."""
        self._abort_flag = False
        return self._active_profile(self._solve_flowsheet(
            method="Inside-Out",
            cancel=lambda: getattr(self, "_abort_flag", False)))

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
        """Settings that belong to the install, not to the open case.

        Max iterations and tolerance used to live here; they are per-case values
        owned by the Simulation tab and saved in the `.colx`, so editing them
        from Preferences quietly modified the case you had open. Everything left
        here is stored in QSettings and survives File -> New.
        """
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
            QLabel, QPushButton, QVBoxLayout,
        )
        from core.units import DUTY, FLOW, TEMPERATURE, DisplayUnits
        from core import nifco
        from .app_settings import (
            apply_nifco, beta_enabled, default_units, log_dir, log_level,
            nifco_enabled, set_beta_enabled, set_default_units, set_log_level,
            set_nifco_enabled,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Preferences")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()

        units = default_units()
        unit_combos = {}
        for field, choices, label in (("temperature", TEMPERATURE, "Temperature:"),
                                      ("flow", FLOW, "Flow:"),
                                      ("duty", DUTY, "Duty:")):
            combo = QComboBox(dlg)
            combo.addItems(list(choices))
            combo.setCurrentText(getattr(units, field))
            form.addRow(label, combo)
            unit_combos[field] = combo
        units_hint = QLabel("Default display units for new cases. A saved case "
                            "keeps the units it was saved with.", dlg)
        units_hint.setProperty("hint", True)
        units_hint.setWordWrap(True)
        form.addRow("", units_hint)

        log_combo = QComboBox(dlg)
        log_combo.addItems(["INFO", "DEBUG"])
        log_combo.setCurrentText(log_level())
        log_combo.setToolTip("Verbosity of ~/.columnforge/columnforge.log. "
                             "Takes effect on the next launch.")
        form.addRow("Log level:", log_combo)

        open_log = QPushButton("Open log folder", dlg)
        open_log.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(log_dir())))
        form.addRow("", open_log)

        have_nifco = nifco.available()
        nifco_check = QCheckBox("Enable NIFCO thermo", dlg)
        nifco_check.setChecked(nifco_enabled() and have_nifco)
        nifco_check.setEnabled(have_nifco)
        nifco_check.setToolTip(
            "Evaluate NRTL / Wilson / UNIQUAC / UNIFAC / Margules in the "
            "compiled Fortran kernel (src/native/nifco2.f90) instead of NumPy. "
            "Identical equations and identical numbers — speed only."
            if have_nifco else
            "Not consumed yet: no compiled library in src/native/lib. Build one "
            "with `make -C src/native`, then re-open Preferences.")
        form.addRow("", nifco_check)

        beta_check = QCheckBox("Enable beta features", dlg)
        beta_check.setChecked(beta_enabled())
        beta_check.setToolTip(
            "Shows features that work but are not settled enough to be the "
            "default: currently the multi-column flowsheet editor. Solver "
            "results are unaffected.")
        form.addRow("", beta_check)

        hint = QLabel("Beta: multi-column flowsheets — several columns joined "
                      "by streams, with recycles.", dlg)
        hint.setProperty("hint", True)
        hint.setWordWrap(True)
        form.addRow("", hint)

        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                   parent=dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return

        set_default_units(DisplayUnits(
            **{f: c.currentText() for f, c in unit_combos.items()}))
        set_log_level(log_combo.currentText())
        set_nifco_enabled(nifco_check.isChecked())
        apply_nifco()
        was = beta_enabled()
        set_beta_enabled(beta_check.isChecked())
        if beta_check.isChecked() != was:
            self.refresh_beta()
            self.statusBar().showMessage(
                "Beta features " + ("enabled." if beta_check.isChecked()
                                    else "disabled."))
        else:
            self.statusBar().showMessage("Preferences updated.")

    def refresh_beta(self):
        """Show or hide the beta UI without a restart."""
        for tab in (self.specs_tab, self.sim_tab, self.results_tab):
            refresh = getattr(tab, "refresh_beta", None)
            if refresh is not None:
                refresh()

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
    from the Run handler land here with tracebacks. Verbosity is a Preferences
    setting, read once at startup."""
    import logging
    import logging.handlers
    import os
    from .app_settings import log_dir, log_level
    handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir(), "columnforge.log"),
        maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level()))
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
    from .app_settings import apply_nifco
    apply_nifco()                        # compiled thermo, if built and asked for
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
