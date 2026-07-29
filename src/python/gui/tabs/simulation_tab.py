from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QProgressBar,
    QPushButton, QGroupBox, QFormLayout, QSpinBox, QStackedWidget
)
from PySide6.QtCore import Signal

from core.data_structures import SolverMode
from gui.theme import set_state
from gui.panels.sub_tab_bar import SubTabBar
from gui.panels.sci_spin_box import SciDoubleSpinBox
from gui.tabs.initialization_tab import (
    VLE_MODELS, ACTIVITY_MODELS, EOS_MODELS,
    IMPLEMENTED_VLE, IMPLEMENTED_ACTIVITY, IMPLEMENTED_EOS,
    _grey_unimplemented,
)


class SimulationTab(QWidget):
    """Simulation tab with solver method selection and run controls."""

    runSimulation = Signal()
    abortSimulation = Signal()
    thermoChanged = Signal()   # mirror combos changed the thermo model

    def __init__(self, parent=None):
        super().__init__(parent)

        self.window_state = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left: Solver Method Column (LHS Sub-Tab Column)
        self.sub_tab_bar = SubTabBar(self)
        self.sub_tab_bar.addTab("Solver Method")
        self.sub_tab_bar.tabClicked.connect(self._on_sub_tab_changed)
        main_layout.addWidget(self.sub_tab_bar)

        # Right: Solver Control Pane
        self.stack = QStackedWidget(self)
        
        solver_page = QWidget()
        solver_layout = QVBoxLayout(solver_page)
        solver_layout.setSpacing(15)
        solver_layout.setContentsMargins(10, 10, 10, 10)

        # Top Row: Solver-specific options
        options_group = QGroupBox("Solver Options")
        options_layout = QFormLayout(options_group)

        self.solver_combo = QComboBox(self)
        # Only the rigorous MESH solvers live here; BVM and FUG are feasibility/
        # shortcut methods and now have their own widgets in the Modules tab.
        self.solver_combo.addItems([
            "Inside-Out",      # rigorous, inner/outer (implemented) — default
            "Bubble-Point",            # rigorous, CMO + ideal VLE (implemented)
        ])
        options_layout.addRow("Method:", self.solver_combo)

        self.max_iter_spin = QSpinBox(self)
        self.max_iter_spin.setRange(10, 10000)
        self.max_iter_spin.setValue(500)
        options_layout.addRow("Max Iterations:", self.max_iter_spin)

        self.tolerance_spin = SciDoubleSpinBox(self)
        self.tolerance_spin.setRange(1e-10, 1e-1)
        self.tolerance_spin.setValue(1e-7)
        options_layout.addRow("Tolerance:", self.tolerance_spin)

        solver_layout.addWidget(options_group)

        # Thermodynamics mirror — same models as the Initialization tab, kept in
        # lock-step through window_state so the model can be tuned without
        # tab-hopping. See _on_thermo_changed / refresh_thermo.
        thermo_group = QGroupBox("Thermodynamics")
        thermo_layout = QFormLayout(thermo_group)

        self.vle_combo = QComboBox(self)
        self.vle_combo.addItems(VLE_MODELS)
        _grey_unimplemented(self.vle_combo, IMPLEMENTED_VLE)
        self.vle_combo.currentTextChanged.connect(self._on_thermo_changed)
        thermo_layout.addRow("Vapor Pressure:", self.vle_combo)

        self.activity_combo = QComboBox(self)
        self.activity_combo.addItems(ACTIVITY_MODELS)
        _grey_unimplemented(self.activity_combo, IMPLEMENTED_ACTIVITY)
        self.activity_combo.currentTextChanged.connect(self._on_thermo_changed)
        thermo_layout.addRow("Activity Coefficient:", self.activity_combo)

        self.eos_combo = QComboBox(self)
        self.eos_combo.addItems(EOS_MODELS)
        _grey_unimplemented(self.eos_combo, IMPLEMENTED_EOS)
        self.eos_combo.currentTextChanged.connect(self._on_thermo_changed)
        thermo_layout.addRow("Equation of State:", self.eos_combo)

        solver_layout.addWidget(thermo_group)

        # Bottom Row: Run/Abort + Progress
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(20)

        # Run/Abort buttons (vertical stack)
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        from gui.theme.iconset import icon
        self.run_btn = QPushButton(icon("run"), "Run")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.setMinimumHeight(40)
        button_layout.addWidget(self.run_btn)

        self.abort_btn = QPushButton(icon("abort"), "Abort")
        self.abort_btn.setObjectName("abort_btn")
        self.abort_btn.setMinimumHeight(40)
        self.abort_btn.setEnabled(False)
        button_layout.addWidget(self.abort_btn)

        bottom_layout.addLayout(button_layout)

        # Progress Info
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Iterations:"))
        self.iter_label = QLabel("0")
        iter_layout.addWidget(self.iter_label)
        progress_layout.addLayout(iter_layout)

        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time Elapsed:"))
        self.time_label =QLabel("0.0 s")
        time_layout.addWidget(self.time_label)
        progress_layout.addLayout(time_layout)

        # Status message
        self.status_label = QLabel("Ready")
        set_state(self.status_label, "neutral")
        progress_layout.addWidget(self.status_label)

        bottom_layout.addWidget(progress_group, 2)

        solver_layout.addLayout(bottom_layout)
        solver_layout.addStretch()

        self.stack.addWidget(solver_page)
        main_layout.addWidget(self.stack)

    def _on_sub_tab_changed(self, index: int):
        """Handle sub-tab change."""
        self.stack.setCurrentIndex(index)
        self.sub_tab_bar.setCurrentIndex(index)

    def _connect_signals(self):
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.abort_btn.clicked.connect(self._on_abort_clicked)
        self.solver_combo.currentTextChanged.connect(self._on_method_changed)

    def _on_run_clicked(self):
        """Handle Run button click."""
        self.run_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.status_label.setText("Running...")
        self.progress_bar.setValue(0)
        self.runSimulation.emit()

    def _on_abort_clicked(self):
        """Handle Abort button click."""
        self.abort_btn.setEnabled(False)
        self.status_label.setText("Aborted")
        self.abortSimulation.emit()

    def set_running(self, is_running: bool):
        """Update UI state for running simulation."""
        if is_running:
            self.run_btn.setEnabled(False)
            self.abort_btn.setEnabled(True)
            self.status_label.setText("Running...")
        else:
            self.run_btn.setEnabled(True)
            self.abort_btn.setEnabled(False)

    def set_progress(self, value: int, iterations: int = 0, elapsed: float = 0):
        """Update progress display."""
        self.progress_bar.setValue(value)
        self.iter_label.setText(str(iterations))
        self.time_label.setText(f"{elapsed:.1f} s")

    def set_status(self, message: str, is_error: bool = False):
        """Set status message."""
        self.status_label.setText(message)
        set_state(self.status_label, "error" if is_error else "ok")

    def set_window_state(self, window_state):
        """Give the tab the shared state and sync the mirror combos to it."""
        self.window_state = window_state
        self.refresh_thermo()
        self.refresh_method()

    # window_state.solver_mode <-> the Method combo. It is persisted in the
    # .colx, so a file saved on Bubble-Point must come back on Bubble-Point.
    _MODE_TO_METHOD = {SolverMode.HYSIM: "Inside-Out",
                       SolverMode.BUBBLE_POINT: "Bubble-Point"}

    def refresh_method(self):
        """Re-sync the Method combo from window_state.solver_mode."""
        if not self.window_state:
            return
        # BVM is sized in the Modules tab and has no rigorous entry here; leave
        # the combo alone rather than silently rewriting a BVM case's mode.
        method = self._MODE_TO_METHOD.get(self.window_state.solver_mode)
        if method:
            self.solver_combo.blockSignals(True)
            self.solver_combo.setCurrentText(method)
            self.solver_combo.blockSignals(False)

    def _on_method_changed(self, method: str):
        if self.window_state:
            self.window_state.solver_mode = (
                SolverMode.HYSIM if "Inside-Out" in method
                else SolverMode.BUBBLE_POINT)
            self.window_state.is_modified = True

    def _on_thermo_changed(self):
        """Write the mirror combos through to the shared thermo config and let
        the Initialization tab re-sync."""
        if self.window_state:
            tc = self.window_state.thermodynamics_config
            tc.vle_model = self.vle_combo.currentText()
            tc.activity_model = self.activity_combo.currentText()
            tc.eos_model = self.eos_combo.currentText()
            self.window_state.is_modified = True
            self.thermoChanged.emit()

    def refresh_thermo(self):
        """Re-sync the mirror combos from window_state (signals blocked to avoid
        a feedback loop with the Initialization tab)."""
        if not self.window_state:
            return
        tc = self.window_state.thermodynamics_config
        for combo, value in ((self.vle_combo, tc.vle_model),
                             (self.activity_combo, tc.activity_model),
                             (self.eos_combo, tc.eos_model)):
            combo.blockSignals(True)
            combo.setCurrentText(value)
            combo.blockSignals(False)

    def get_solver_config(self) -> dict:
        """Get current solver configuration."""
        return {
            "solver_method": self.solver_combo.currentText(),
            "max_iterations": self.max_iter_spin.value(),
            "tolerance": self.tolerance_spin.value(),
        }

    def clear(self):
        """Clear simulation state."""
        self.progress_bar.setValue(0)
        self.iter_label.setText("0")
        self.time_label.setText("0.0 s")
        self.status_label.setText("Ready")
        self.run_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
