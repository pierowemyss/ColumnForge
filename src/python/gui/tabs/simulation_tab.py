from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QProgressBar,
    QPushButton, QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QStackedWidget
)
from PySide6.QtCore import Signal

from gui.panels.sub_tab_bar import SubTabBar


class SimulationTab(QWidget):
    """Simulation tab with solver method selection and run controls."""

    runSimulation = Signal()
    abortSimulation = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._setup_ui()
        self._setup_styles()
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
        self.solver_combo.addItems([
            "Bubble-Point",            # rigorous, CMO + ideal VLE (implemented)
            "BVM (preliminary)",       # feasibility / profile, via the BVM module
            "Inside-Out (HYSIM)",      # rigorous, inner/outer (implemented)
            "Shortcut (FUG)",          # Fenske-Underwood-Gilliland design report
        ])
        options_layout.addRow("Method:", self.solver_combo)

        self.max_iter_spin = QSpinBox(self)
        self.max_iter_spin.setRange(10, 10000)
        self.max_iter_spin.setValue(500)
        options_layout.addRow("Max Iterations:", self.max_iter_spin)

        self.tolerance_spin = QDoubleSpinBox(self)
        self.tolerance_spin.setRange(1e-10, 1e-1)
        self.tolerance_spin.setDecimals(8)
        self.tolerance_spin.setValue(1e-7)
        options_layout.addRow("Tolerance:", self.tolerance_spin)

        solver_layout.addWidget(options_group)

        # Bottom Row: Run/Abort + Progress
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(20)

        # Run/Abort buttons (vertical stack)
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        self.run_btn = QPushButton("Run")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("font-weight: bold;")
        button_layout.addWidget(self.run_btn)

        self.abort_btn = QPushButton("Abort")
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
        self.status_label.setStyleSheet("font-weight: bold;")
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

    def _setup_styles(self):
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton#run_btn {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            QPushButton#abort_btn {
                background-color: #f44336;
                color: white;
            }
            QPushButton:checked {
                background-color: #666666;
            }
        """)

    def _connect_signals(self):
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.abort_btn.clicked.connect(self._on_abort_clicked)

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
        if is_error:
            self.status_label.setStyleSheet("font-weight: bold; color: red;")
        else:
            self.status_label.setStyleSheet("font-weight: bold; color: green;")

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
