"""Phase EQ — isothermal / vapour-fraction flash on the loaded species.

A live test-bench for every thermo model: enter a feed composition, pick a spec
pair (T & P, or P & vapour fraction), and flash via core.flash. Results show
beta, the solved T, the x/y/K table and an x-vs-y bar chart. Honesty label and
missing-pair note come from the shared session_models helper.
"""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
)

from core import flash
from gui.plotting import CompactNavigationToolbar
from .module_thermo import session_models
from ..panels.unit_combo_box import UnitComboBox
from ..panels.sci_spin_box import SciDoubleSpinBox, fmt


class PhaseEQModuleWidget(QWidget):
    SPECS = ["T & P", "P & vapour fraction"]

    def __init__(self, window_state=None, parent=None):
        super().__init__(parent)
        self.window_state = window_state
        self._build_ui()
        if self.window_state:
            self.refresh()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # ---- left: inputs --------------------------------------------
        left = QVBoxLayout()
        feed_group = QGroupBox("Feed composition (mole basis, normalised)")
        fg = QVBoxLayout(feed_group)
        self.feed_table = QTableWidget(0, 2)
        self.feed_table.setHorizontalHeaderLabels(["species", "z"])
        self.feed_table.horizontalHeader().setStretchLastSection(True)
        fg.addWidget(self.feed_table)
        left.addWidget(feed_group)

        spec_group = QGroupBox("Specification")
        form = QFormLayout(spec_group)
        self.spec_combo = QComboBox()
        self.spec_combo.addItems(self.SPECS)
        self.spec_combo.currentTextChanged.connect(self._on_spec_changed)
        form.addRow("Spec pair:", self.spec_combo)

        self.p_units = UnitComboBox(unit_type="pressure")
        self.p_units.setValue(1.0)
        self.p_units.setUnit("atm")
        form.addRow("Pressure:", self.p_units)

        self.t_units = UnitComboBox(unit_type="temperature")
        self.t_units.setValue(90.0)
        form.addRow("Temperature:", self.t_units)

        self.beta_spin = SciDoubleSpinBox()
        self.beta_spin.setRange(0.0, 1.0)
        self.beta_spin.setValue(0.5)
        form.addRow("Vapour fraction β:", self.beta_spin)
        left.addWidget(spec_group)

        self.model_label = QLabel("")
        self.model_label.setWordWrap(True)
        left.addWidget(self.model_label)

        self.solve_btn = QPushButton("Solve flash")
        self.solve_btn.clicked.connect(self._solve)
        left.addWidget(self.solve_btn)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        left.addStretch()
        layout.addLayout(left, 1)

        # ---- right: results ------------------------------------------
        right = QVBoxLayout()
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["species", "x", "y", "K"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.result_table, 1)
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
        right.addWidget(self.toolbar)
        right.addWidget(self.canvas, 1)
        layout.addLayout(right, 1)

        self._on_spec_changed(self.spec_combo.currentText())

    def _on_spec_changed(self, spec):
        by_beta = spec.startswith("P &")
        self.t_units.setEnabled(not by_beta)
        self.beta_spin.setEnabled(by_beta)

    def refresh(self):
        order = self.window_state.get_species_names()
        # keep any already-entered z values
        prev = self._read_feed()
        self.feed_table.setRowCount(0)
        for nm in order:
            row = self.feed_table.rowCount()
            self.feed_table.insertRow(row)
            name_item = QTableWidgetItem(nm)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.feed_table.setItem(row, 0, name_item)
            z0 = prev.get(nm, 1.0)
            self.feed_table.setItem(row, 1, QTableWidgetItem(f"{z0:g}"))
        self._update_label()

    def _read_feed(self):
        out = {}
        for row in range(self.feed_table.rowCount()):
            nm = self.feed_table.item(row, 0)
            zc = self.feed_table.item(row, 1)
            if nm is None:
                continue
            try:
                out[nm.text()] = float(zc.text()) if zc else 0.0
            except ValueError:
                out[nm.text()] = 0.0
        return out

    def _update_label(self):
        order = self.window_state.get_species_names()
        try:
            _, _, _, label, note = session_models(self.window_state, order)
        except ValueError as exc:
            self.model_label.setText(str(exc))
            return
        self.model_label.setText(label + ("  —  " + note if note else ""))

    def _solve(self):
        if not self.window_state:
            return
        order = self.window_state.get_species_names()
        if len(order) < 2:
            self.status.setText("Need at least 2 species (Initialization tab).")
            return
        feed = self._read_feed()
        z = np.array([feed.get(nm, 0.0) for nm in order], float)
        if z.sum() <= 0.0:
            self.status.setText("Enter a feed composition.")
            return
        z = z / z.sum()
        try:
            antoine, gamma_fn, phi_fn, label, note = session_models(
                self.window_state, order)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        P = self.window_state.thermodynamics_config.pressure_in_psat_unit(
            self.p_units.valueInSI())
        try:
            if self.spec_combo.currentText().startswith("P &"):
                r = flash.flash_PbetaT(z, P, self.beta_spin.value(),
                                       antoine, gamma_fn, phi_fn)
            else:
                T_degC = self.t_units.valueInSI() - 273.15
                r = flash.flash_TP(z, T_degC, P, antoine, gamma_fn, phi_fn)
        except (ValueError, ZeroDivisionError) as exc:
            self.status.setText(f"Flash failed: {exc}")
            return
        self._show_result(order, r, note)

    def _show_result(self, order, r, note):
        self.result_table.setRowCount(0)
        for i, nm in enumerate(order):
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            for col, v in enumerate([nm, fmt(r.x[i]), fmt(r.y[i]),
                                     fmt(r.K[i])]):
                self.result_table.setItem(row, col, QTableWidgetItem(v))
        self.status.setText(
            f"phase: {r.phase} · β = {r.beta:.4f} · T = {r.T:.2f} °C"
            + ("  —  " + note if note else ""))
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        idx = np.arange(len(order))
        ax.bar(idx - 0.2, r.x, width=0.4, label="x (liquid)")
        ax.bar(idx + 0.2, r.y, width=0.4, label="y (vapour)")
        ax.set_xticks(idx)
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("mole fraction")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
