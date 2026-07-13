"""Txy / Pxy / xy binary VLE diagram module (+ azeotrope table).

The everyday VLE-inspection tool. Picks two species from the loaded set (or
adds one from the bundled database), sweeps the binary at the session's
selected thermo models, and draws the bubble/dew loci or the y-x curve. The
azeotrope table below classifies the singular points (pure ends + azeotropes)
via gui.plotting.singular_points. Ternary residue-curve maps live in the
dedicated RCM module.

All math is in core.vle_diagrams / gui.plotting; this file is only the widget.
"""
import csv

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QFileDialog,
)

from core import vle_diagrams
from gui.plotting import CompactNavigationToolbar, singular_points
from .module_thermo import session_models
from ..panels.unit_combo_box import UnitComboBox
from ..panels.species_search_dialog import SpeciesSearchDialog


class TxyModuleWidget(QWidget):
    MODES = ["Txy (fixed P)", "Pxy (fixed T)", "xy (fixed P)"]

    def __init__(self, window_state=None, parent=None):
        super().__init__(parent)
        self.window_state = window_state
        self._last = None                    # last diagram dict for CSV export
        self._build_ui()
        if self.window_state:
            self.refresh()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Top row: controls column (left) and plot column (right), side by side.
        top_row = QHBoxLayout()

        controls = QGroupBox("Binary pair & conditions")
        controls.setMaximumWidth(420)
        form = QFormLayout(controls)
        pair_row = QHBoxLayout()
        self.c1_combo = QComboBox()
        self.c2_combo = QComboBox()
        db_btn = QPushButton("From database…")
        db_btn.clicked.connect(self._add_from_db)
        pair_row.addWidget(self.c1_combo)
        pair_row.addWidget(QLabel(" / "))
        pair_row.addWidget(self.c2_combo)
        pair_row.addWidget(db_btn)
        pair_row.addStretch()
        form.addRow("Components:", self._wrap(pair_row))

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.MODES)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        form.addRow("Mode:", self.mode_combo)

        self.p_units = UnitComboBox(unit_type="pressure")
        self.p_units.setValue(1.0)
        self.p_units.setUnit("atm")
        form.addRow("Pressure:", self.p_units)

        self.t_units = UnitComboBox(unit_type="temperature")
        self.t_units.setValue(78.0)
        form.addRow("Temperature:", self.t_units)

        self.model_label = QLabel("")
        self.model_label.setWordWrap(True)
        form.addRow("Models:", self.model_label)
        # keep the controls column pinned to the top of its side
        controls_col = QVBoxLayout()
        controls_col.addWidget(controls)
        controls_col.addStretch()
        top_row.addLayout(controls_col, 0)

        # Right: plot column (buttons, toolbar, canvas, status).
        plot_col = QVBoxLayout()
        btn_row = QHBoxLayout()
        self.plot_btn = QPushButton("Plot")
        self.plot_btn.clicked.connect(self._plot)
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self._export_csv)
        self.export_btn.setEnabled(False)
        btn_row.addWidget(self.plot_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addStretch()
        plot_col.addLayout(btn_row)

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
        plot_col.addWidget(self.toolbar)
        plot_col.addWidget(self.canvas, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        plot_col.addWidget(self.status)
        top_row.addLayout(plot_col, 1)
        layout.addLayout(top_row, 1)

        # Table below, spanning the full width.
        az_group = QGroupBox("Singular points (pure components & azeotropes)")
        az_layout = QVBoxLayout(az_group)
        self.az_table = QTableWidget(0, 4)
        self.az_table.setHorizontalHeaderLabels(
            ["composition", "T (°C)", "type", "model"])
        self.az_table.horizontalHeader().setStretchLastSection(True)
        az_layout.addWidget(self.az_table)
        layout.addWidget(az_group)

        self._on_mode_changed(self.mode_combo.currentText())

    @staticmethod
    def _wrap(inner_layout):
        w = QWidget()
        w.setLayout(inner_layout)
        return w

    def _on_mode_changed(self, mode):
        is_pxy = mode.startswith("Pxy")
        self.t_units.setEnabled(is_pxy)
        self.p_units.setEnabled(not is_pxy)

    # -------------------------------------------------------------- session
    def refresh(self):
        """Repopulate combos from the loaded species and refresh the label."""
        if not self.window_state:
            return
        order = self.window_state.get_species_names()
        for combo, default in ((self.c1_combo, 0), (self.c2_combo, 1)):
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(order)
            idx = combo.findText(keep)
            combo.setCurrentIndex(idx if idx >= 0
                                  else min(default, len(order) - 1))
            combo.blockSignals(False)
        self._update_label()

    def _add_from_db(self):
        if not self.window_state:
            return
        dlg = SpeciesSearchDialog(
            self, existing_names=self.window_state.get_species_names())
        if dlg.exec() and dlg.selected_name:
            from core import component_db
            component_db.load_into(self.window_state, dlg.selected_name)
            self.refresh()
            # point the second combo at the freshly added species
            idx = self.c2_combo.findText(component_db.get(
                dlg.selected_name)["name"])
            if idx >= 0:
                self.c2_combo.setCurrentIndex(idx)

    def _order(self):
        return [self.c1_combo.currentText(), self.c2_combo.currentText()]

    def _pressure_psat_unit(self):
        p_bar = self.p_units.valueInSI()
        return self.window_state.thermodynamics_config.pressure_in_psat_unit(
            p_bar)

    def _update_label(self):
        try:
            _, _, _, label, note = session_models(self.window_state, self._order())
        except ValueError as exc:
            self.model_label.setText(str(exc))
            return
        self.model_label.setText(label + ("  —  " + note if note else ""))

    # ----------------------------------------------------------- plotting
    def _plot(self):
        if not self.window_state:
            return
        mode = self.mode_combo.currentText()
        order = self._order()
        if order[0] == order[1]:
            self.status.setText("Pick two different components.")
            return
        try:
            antoine, gamma_fn, phi_fn, label, note = session_models(
                self.window_state, order)
        except ValueError as exc:
            self.status.setText(str(exc))
            return

        try:
            if mode.startswith("Pxy"):
                T_degC = self.t_units.valueInSI() - 273.15
                data = vle_diagrams.diagram("Pxy", antoine, gamma_fn, phi_fn,
                                            T=T_degC)
            else:
                P = self._pressure_psat_unit()
                key = "Txy" if mode.startswith("Txy") else "xy"
                data = vle_diagrams.diagram(key, antoine, gamma_fn, phi_fn, P=P)
        except (ValueError, ZeroDivisionError) as exc:
            self.status.setText(f"Could not compute: {exc}")
            return

        self._last = data
        self.export_btn.setEnabled(True)
        self._draw(data, order, mode, label, note)
        self._fill_az_table(order, antoine, gamma_fn, label)

    def _draw(self, data, order, mode, label, note):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x1, y1, z = data["x1"], data["y1"], data["z"]
        if mode.startswith("xy"):
            ax.plot(x1, y1, "-", label="equilibrium")
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
            ax.set_xlabel(f"x ({order[0]})")
            ax.set_ylabel(f"y ({order[0]})")
            ax.set_ylim(0, 1)
        else:
            zlab = "T (°C)" if data["zlabel"] == "T" else "P (Psat unit)"
            ax.plot(x1, z, "-", label="bubble")
            ax.plot(y1, z, "-", label="dew")
            ax.set_xlabel(f"x, y ({order[0]})")
            ax.set_ylabel(zlab)
            for az in data["azeotropes"]:
                ax.axvline(az["x1"], color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_title(f"{order[0]} / {order[1]}  ({label})", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
        msg = f"{mode}: {len(data['azeotropes'])} azeotrope(s)."
        self.status.setText(msg + ("  " + note if note else ""))

    def _fill_az_table(self, order, antoine, gamma_fn, label):
        P = self._pressure_psat_unit()
        try:
            pts = singular_points(P, antoine, order, gamma_fn=gamma_fn)
        except Exception:
            pts = []
        self.az_table.setRowCount(0)
        for p in pts:
            comp = ", ".join(f"{order[i]}:{p['x'][i]:.3f}"
                             for i in range(len(order)) if p["x"][i] > 1e-3)
            row = self.az_table.rowCount()
            self.az_table.insertRow(row)
            kind = p["kind"] + (" (pure)" if p["pure"] else " (azeotrope)")
            for col, txt in enumerate(
                    [comp, f"{p['T']:.2f}", kind, label]):
                self.az_table.setItem(row, col, QTableWidgetItem(txt))

    def _export_csv(self):
        if not self._last:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export diagram CSV", "vle_diagram.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(vle_diagrams.to_csv_rows(self._last))
        self.status.setText(f"Saved {path}")
