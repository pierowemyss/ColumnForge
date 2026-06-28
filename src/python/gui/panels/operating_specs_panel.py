"""Aspen-style operating specification slots.

A simple column needs N operating specs (2 for the basic total-condenser +
reboiler case). Rather than scatter reflux on the condenser and boilup on the
reboiler, this panel lets you add any mix of operating variables — reflux ratio,
reboil ratio, distillate/bottoms rate, D:F, a product purity, a key recovery —
and the solver resolves them to the column's operating point. The DoF status
(driven by core.dof) tells you when you've supplied the right number.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QComboBox, QDoubleSpinBox, QHeaderView, QLabel
)
from PySide6.QtCore import Signal

from core.dof import Spec, SpecKind, OPERATING_KINDS

# Stable display order for the kind dropdown.
_KINDS = [k for k in SpecKind if k in OPERATING_KINDS]
_PURITY = {SpecKind.DIST_PURITY, SpecKind.BOTTOMS_PURITY}

# Specs that are physical fractions in [0, 1]; everything else is a non-negative
# ratio or flow rate. Keeps the spinbox from accepting e.g. a 5.0 mole fraction.
_FRACTION = _PURITY | {
    SpecKind.LK_RECOVERY, SpecKind.HK_RECOVERY,
    SpecKind.DF_RATIO, SpecKind.BF_RATIO,
}


class OperatingSpecsPanel(QWidget):
    """Variable-length list of operating specs; DoF count enforced upstream."""

    specsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._species = []
        self._loading = False

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.addWidget(QLabel(
            "Specify the column's operating variables (any combination the DoF "
            "allows). The solver holds these exactly."))

        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Specification", "Value", "Component"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        self.add_btn = QPushButton("Add spec")
        self.add_btn.clicked.connect(lambda: self._add_row())
        self.remove_btn = QPushButton("Remove selected")
        self.remove_btn.clicked.connect(self._remove_selected)
        buttons.addWidget(self.add_btn)
        buttons.addWidget(self.remove_btn)
        buttons.addStretch()
        layout.addLayout(buttons)

    # --- row construction -------------------------------------------------
    def _add_row(self, kind=None, value=0.0, component=-1):
        r = self.table.rowCount()
        self.table.insertRow(r)

        kind_combo = QComboBox()
        for k in _KINDS:
            kind_combo.addItem(k.value, k)
        if kind is not None:
            kind_combo.setCurrentIndex(_KINDS.index(kind))
        kind_combo.currentIndexChanged.connect(lambda _i, row=r: self._on_kind_changed(row))
        self.table.setCellWidget(r, 0, kind_combo)

        value_spin = QDoubleSpinBox()
        value_spin.setDecimals(4)
        value_spin.valueChanged.connect(self._emit)
        self.table.setCellWidget(r, 1, value_spin)
        self._apply_value_range(r)
        value_spin.setValue(value)

        comp_combo = QComboBox()
        comp_combo.addItems(self._species)
        if 0 <= component < comp_combo.count():
            comp_combo.setCurrentIndex(component)
        comp_combo.currentIndexChanged.connect(self._emit)
        self.table.setCellWidget(r, 2, comp_combo)

        self._sync_component_enabled(r)
        if not self._loading:
            self._emit()

    def _on_kind_changed(self, row):
        self._sync_component_enabled(row)
        self._apply_value_range(row)
        self._emit()

    def _apply_value_range(self, row):
        """Fractions clamp to [0, 1]; ratios/rates are non-negative."""
        kind = self.table.cellWidget(row, 0).currentData()
        spin = self.table.cellWidget(row, 1)
        spin.setRange(0.0, 1.0 if kind in _FRACTION else 1e9)

    def _sync_component_enabled(self, row):
        """A component choice only matters for the purity specs."""
        kind = self.table.cellWidget(row, 0).currentData()
        self.table.cellWidget(row, 2).setEnabled(kind in _PURITY)

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self._emit()

    def _emit(self):
        if not self._loading:
            self.specsChanged.emit()

    # --- data in/out ------------------------------------------------------
    def set_species(self, names):
        """Refresh the component dropdowns, preserving each row's selection."""
        self._species = list(names)
        for r in range(self.table.rowCount()):
            combo = self.table.cellWidget(r, 2)
            keep = combo.currentIndex()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self._species)
            if 0 <= keep < combo.count():
                combo.setCurrentIndex(keep)
            combo.blockSignals(False)

    def get_specs(self):
        specs = []
        for r in range(self.table.rowCount()):
            kind = self.table.cellWidget(r, 0).currentData()
            value = self.table.cellWidget(r, 1).value()
            comp = self.table.cellWidget(r, 2).currentIndex() if kind in _PURITY else -1
            # unit_ref = kind name: one spec per kind, matching the condenser/
            # reboiler editors so the two views never double-count a shared spec.
            specs.append(Spec(kind, value, unit_ref=kind.name, component=comp))
        return specs

    def set_specs(self, specs):
        self._loading = True
        self.table.setRowCount(0)
        for s in specs:
            if s.kind in OPERATING_KINDS:
                self._add_row(s.kind, s.value, s.component)
        self._loading = False
