#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Window - FreeColumn Column Solver GUI
Tabbed interface with comprehensive simulation workflow

Author: Piero Wemyss
"""

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


class MainWindow(QMainWindow):
    """Main application window with tabbed interface for column simulation"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FreeColumn - Column Solver")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)

        self.window_state = WindowState()

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #2d2d2d;
                color: #cccccc;
                border-top: 1px solid #444444;
            }
        """)
        self.statusBar().showMessage("Ready")

    def _setup_ui(self):
        central_widget = QWidget(self)
        central_widget.setStyleSheet("""
            QWidget {
                background-color: #2d2d2d;
                color: #cccccc;
            }
        """)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.tab_widget = QTabWidget(self)
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabsClosable(False)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                background-color: #2d2d2d;
                border: 1px solid #444444;
            }
            QTabBar::tab {
                background-color: #1a1a1a;
                color: #888888;
                padding: 10px 20px;
                margin-right: 2px;
                border: 1px solid #333333;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #2d2d2d;
                color: #ffffff;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background-color: #333333;
                color: #cccccc;
            }
        """)

        self.init_tab = InitializationTab(self)
        self.specs_tab = SpecificationsTab(self)
        self.sim_tab = SimulationTab(self)
        self.results_tab = ResultsTab(self)
        self.modules_tab = ModulesTab(self)
        
        # Set window state on tabs
        self.init_tab.set_window_state(self.window_state)
        self.specs_tab.set_window_state(self.window_state)
        self.modules_tab.set_window_state(self.window_state)
        self.results_tab.set_window_state(self.window_state)

        # Connect species changes to refresh specs tab
        self.init_tab.speciesChanged.connect(self.specs_tab.refresh)

        self.tab_widget.addTab(self.init_tab, "Initialization")
        self.tab_widget.addTab(self.specs_tab, "Specifications")
        self.tab_widget.addTab(self.sim_tab, "Simulation")
        self.tab_widget.addTab(self.results_tab, "Results")
        self.tab_widget.addTab(self.modules_tab, "Modules")

        main_layout.addWidget(self.tab_widget)

        self.toolbar = self.addToolBar("Main")
        self.toolbar.setMovable(False)
        self.toolbar.setStyleSheet("""
            QToolBar {
                spacing: 5px;
                padding: 3px;
                background-color: #111111;
                border-bottom: 1px solid #333333;
            }
            QToolBar QToolButton {
                min-width: 60px;
                padding: 5px 10px;
                color: #888888;
                background-color: transparent;
                border: none;
            }
            QToolBar QToolButton:hover {
                background-color: #333333;
                color: #cccccc;
            }
            QToolBar QToolButton:pressed {
                background-color: #444444;
            }
        """)

        new_act = QAction("New", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self.new_config)
        self.toolbar.addAction(new_act)

        open_act = QAction("Open", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self.load_config)
        self.toolbar.addAction(open_act)

        save_act = QAction("Save", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self.save_config)
        self.toolbar.addAction(save_act)

        self.toolbar.addSeparator()

        run_act = QAction("Run", self)
        run_act.setShortcut("F5")
        run_act.triggered.connect(self.run_simulation)
        self.toolbar.addAction(run_act)

        self.toolbar.addSeparator()

        prefs_act = QAction("Preferences", self)
        prefs_act.triggered.connect(self.show_preferences)
        self.toolbar.addAction(prefs_act)

    def _setup_menu(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #2d2d2d;
                color: #cccccc;
                border-bottom: 1px solid #444444;
                padding: 2px;
            }
            QMenuBar::item:selected {
                background-color: #3d3d3d;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #444444;
            }
            QMenu::item:selected {
                background-color: #3d3d3d;
            }
        """)

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

        edit_menu = menubar.addMenu("Edit")
        undo_act = QAction("Undo", self)
        undo_act.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_act)

        redo_act = QAction("Redo", self)
        redo_act.setShortcut("Ctrl+Y")
        edit_menu.addAction(redo_act)

        edit_menu.addSeparator()

        cut_act = QAction("Cut", self)
        cut_act.setShortcut("Ctrl+X")
        edit_menu.addAction(cut_act)

        copy_act = QAction("Copy", self)
        copy_act.setShortcut("Ctrl+C")
        edit_menu.addAction(copy_act)

        paste_act = QAction("Paste", self)
        paste_act.setShortcut("Ctrl+V")
        edit_menu.addAction(paste_act)

        view_menu = menubar.addMenu("View")
        zoom_in_act = QAction("Zoom In", self)
        zoom_in_act.setShortcut("Ctrl++")
        view_menu.addAction(zoom_in_act)

        zoom_out_act = QAction("Zoom Out", self)
        zoom_out_act.setShortcut("Ctrl+-")
        view_menu.addAction(zoom_out_act)

        reset_view_act = QAction("Reset View", self)
        reset_view_act.setShortcut("Ctrl+0")
        view_menu.addAction(reset_view_act)

        help_menu = menubar.addMenu("Help")
        docs_act = QAction("Documentation", self)
        docs_act.setShortcut("F1")
        help_menu.addAction(docs_act)

        about_act = QAction("About FreeColumn", self)
        about_act.triggered.connect(self.show_about)
        help_menu.addAction(about_act)

    def _connect_signals(self):
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        # In-tab Run/Abort buttons (were emitting into the void)
        self.sim_tab.runSimulation.connect(self.run_simulation)
        self.sim_tab.abortSimulation.connect(self.abort_simulation)

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
            "FreeColumn Files (*.colx);;All Files (*)"
        )

        if filename:
            try:
                state = load_colx(filename)
                self.window_state.load_from_dict(state)
                # repopulate every tab from the restored state
                self.init_tab.set_window_state(self.window_state)
                self.specs_tab.set_window_state(self.window_state)
                self.modules_tab.set_window_state(self.window_state)
                self.results_tab.clear_results()

                QMessageBox.information(
                    self, "Load Successful",
                    f"Column configuration loaded from:\n{filename}"
                )
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
            "FreeColumn Files (*.colx);;All Files (*)"
        )

        if filename:
            if not filename.endswith('.colx'):
                filename += '.colx'
            self._do_save(filename)

    def _do_save(self, filepath: str):
        """Perform the actual save operation."""
        try:
            save_colx(filepath, self.window_state.to_dict(),
                      name="FreeColumn Configuration")
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
                with open(filename, 'w', newline='') as f:
                    csv.writer(f).writerows(profile_to_csv_rows(profile))
                self.statusBar().showMessage(f"Results exported to {filename}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error",
                    f"Failed to export results:\n{str(e)}"
                )

    def run_simulation(self):
        """Run the column simulation via the BVM solver."""
        can_run = self._check_specification()

        if not can_run:
            icon, message, _ = self.window_state.get_specification_status()
            QMessageBox.warning(
                self, "Cannot Run Simulation",
                f"{message}\n\nPlease complete the column specification before running."
            )
            return

        # ponytail: solvers here run synchronously in well under a second, so there
        # is no long task to interrupt — Abort stays cosmetic until a solver runs
        # long enough to need real cancellation.
        method = self.sim_tab.solver_combo.currentText()
        self.statusBar().showMessage("Solving...")
        try:
            if method == "Bubble-Point":
                profile = self._solve_bubble_point()
            elif "Inside-Out" in method or method.startswith("HYSIM"):
                profile = self._solve_inside_out()
            elif method.startswith("BVM"):
                profile = self.modules_tab.ensure_bvm().solve()
            else:
                raise ValueError(
                    f"{method} is not implemented yet — choose Bubble-Point, "
                    "HYSIM Inside-Out, or BVM (preliminary)."
                )
        except ValueError as exc:
            self.sim_tab.set_running(False)
            self.statusBar().showMessage("Run failed")
            QMessageBox.warning(self, "Cannot Run Simulation", str(exc))
            return

        self.window_state.results = profile
        self.sim_tab.set_running(False)
        self.results_tab.update_results(self._normalize_results(profile))
        self.tab_widget.setCurrentIndex(3)
        self.statusBar().showMessage(
            f"Solved: {profile['n_stages']} stages, feed at stage {profile['feed_stage']}."
        )

    @staticmethod
    def _normalize_results(profile: dict) -> dict:
        """Map a BVM column profile to the ResultsTab.update_results schema."""
        x, T = profile["x"], profile["T"]
        rows = [
            [i + 1, round(float(T[i]), 2)] + [round(float(v), 4) for v in x[i]]
            for i in range(profile["n_stages"])
        ]
        return {
            "status": profile.get("message", "Solved"),
            "stages": profile["n_stages"],
            "iterations": profile.get("iterations", "—"),  # BVM marches directly
            "runtime": "< 1 s",
            "data": rows,
        }

    def _gather_rigorous_inputs(self) -> dict:
        """Pull and validate the common inputs for the rigorous solvers
        (bubble-point and Inside-Out) from window_state + the Simulation tab.
        Raises ValueError (with a user-facing message) when setup is incomplete."""
        import numpy as np
        from .state.window_state import StreamType

        ws = self.window_state
        order = ws.get_species_names()
        if len(order) < 2:
            raise ValueError("Need at least 2 species (Initialization tab).")

        feed = next((s for s in ws.streams.values()
                     if s.stream_type == StreamType.FEED), None)
        if feed is None or not feed.flow or not feed.composition:
            raise ValueError("Feed stream needs a flow rate and composition.")
        zF = np.array([feed.composition.get(nm, 0.0) for nm in order], float)
        if abs(zF.sum() - 1.0) > 1e-3:
            raise ValueError(f"Feed composition sums to {zF.sum():.4f}, not 1.")

        antoine = ws.thermodynamics_config.psat_params(order)  # (N,3) Antoine or (N,7) PLXANT

        N = int(ws.num_stages)
        feed_stage = min(max(1, int(ws.feed_stage)), N)
        F = float(feed.flow)
        P = float(ws.pressure)
        gamma_fn = ws.build_gamma_fn(order)

        # Resolve whatever 2 operating specs the user set (reflux, boilup, rates,
        # a product purity, a key recovery, ...) down to (R, D) — Aspen-style.
        from core.operating_specs import resolve_operating_point
        from core.dof import OPERATING_KINDS
        from core.column_solvers import solve_bubble_point
        ops = [s for s in ws.collect_specs() if s.kind in OPERATING_KINDS]
        if len(ops) != 2:
            raise ValueError(
                "A simple column needs exactly 2 operating specs (e.g. reflux "
                "ratio + a distillate rate, purity, or recovery). "
                f"You have {len(ops)} — see the Specifications DoF status.")

        def _solve_fn(R, D):
            return solve_bubble_point(zF, F, antoine, order, N=N,
                                      feed_stage=feed_stage, R=R, D=D, P=P,
                                      gamma_fn=gamma_fn)

        R, D = resolve_operating_point(
            ops, F, zF, solve_fn=_solve_fn,
            lk=ws.light_key_index, hk=ws.heavy_key_index)

        cfg = self.sim_tab.get_solver_config()   # honor the Simulation tab knobs
        return dict(zF=zF, F=F, antoine=antoine, order=order,
                    N=N, feed_stage=feed_stage, R=float(R), D=float(D),
                    P=P, max_iter=int(cfg["max_iterations"]),
                    tol=float(cfg["tolerance"]), gamma_fn=gamma_fn)

    def _solve_bubble_point(self) -> dict:
        """Run the rigorous bubble-point (Wang-Henke) solver."""
        from core.column_solvers import solve_bubble_point
        g = self._gather_rigorous_inputs()
        return solve_bubble_point(g["zF"], g["F"], g["antoine"], g["order"],
                                  N=g["N"], feed_stage=g["feed_stage"], R=g["R"],
                                  D=g["D"], P=g["P"], max_iter=g["max_iter"],
                                  tol=g["tol"], gamma_fn=g["gamma_fn"])

    def _solve_inside_out(self) -> dict:
        """Run the Inside-Out (HYSIM) solver, with the Abort flag as cancel hook."""
        from core.column_solvers import solve_inside_out
        g = self._gather_rigorous_inputs()
        self._abort_flag = False
        return solve_inside_out(g["zF"], g["F"], g["antoine"], g["order"],
                                N=g["N"], feed_stage=g["feed_stage"], R=g["R"],
                                D=g["D"], P=g["P"], max_iter=g["max_iter"],
                                tol=g["tol"], gamma_fn=g["gamma_fn"],
                                cancel=lambda: getattr(self, "_abort_flag", False))

    def abort_simulation(self):
        """Abort a running simulation. Sets the cancel flag the Inside-Out solver
        checks between outer iterations.
        # ponytail: solves run synchronously sub-second, so the flag only bites
        # once a solver runs in a QThread — wire threading when a run is slow
        # enough to need mid-run cancellation."""
        self._abort_flag = True
        self.sim_tab.set_running(False)
        self.statusBar().showMessage("Simulation aborted")

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
            self, "About FreeColumn",
            "FreeColumn - Column Solver\n\n"
            "A comprehensive GUI-based column solver for chemical "
            "engineering applications.\n\n"
            "Version 1.0.0\n"
            "Author: Piero Wemyss"
        )

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
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
