"""Aspen-style operating specification slots.

A simple column needs N operating specs (2 for the basic total-condenser +
reboiler case). Rather than scatter reflux on the condenser and boilup on the
reboiler, this panel lets you add any mix of operating variables — reflux ratio,
reboil ratio, distillate/bottoms rate, D:F, a product purity, a key recovery —
and the solver resolves them to the column's operating point. The DoF status
(driven by core.dof) tells you when you've supplied the right number.

Only two kinds actually need a component, so there's no permanent Component
column: the Value cell grows a small combo just for the rows that use one —
a component picker for purity specs, and the light/heavy key picker for the
recovery specs (which writes the shared window_state key indices).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QComboBox, QHeaderView, QLabel
)
from PySide6.QtCore import Signal

from core.dof import Spec, SpecKind, OPERATING_KINDS
from .sci_spin_box import SciDoubleSpinBox

# Stable display order for the kind dropdown.
_KINDS = [k for k in SpecKind if k in OPERATING_KINDS]
_PURITY = {SpecKind.DIST_PURITY, SpecKind.BOTTOMS_PURITY}
_KEY_RECOVERY = {SpecKind.LK_RECOVERY, SpecKind.HK_RECOVERY}

# Specs that are physical fractions in [0, 1]; everything else is a non-negative
# ratio or flow rate. Keeps the spinbox from accepting e.g. a 5.0 mole fraction.
_FRACTION = _PURITY | {
    SpecKind.LK_RECOVERY, SpecKind.HK_RECOVERY,
    SpecKind.DF_RATIO, SpecKind.BF_RATIO,
}

# Duty specs are entered in kW and can be negative (a condenser removes heat).
# Stored/emitted in kW; main_window converts to the solver's kJ/h basis.
_DUTY = {SpecKind.CONDENSER_DUTY, SpecKind.REBOILER_DUTY}


class OperatingSpecsPanel(QWidget):
    """Variable-length list of operating specs; DoF count enforced upstream."""

    specsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._species = []
        self._loading = False
        self.window_state = None

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.addWidget(QLabel(
            "Specify the column's operating variables (any combination the DoF "
            "allows). The solver holds these exactly."))

        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Specification", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
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

    def set_window_state(self, ws):
        """Needed so the recovery rows can read/write the shared key indices."""
        self.window_state = ws

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

        # Composite value cell: spin box + an optional trailing combo whose role
        # depends on the kind (component for purity, key for recovery).
        cell = QWidget()
        h = QHBoxLayout(cell)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        spin = SciDoubleSpinBox()
        spin.valueChanged.connect(self._emit)
        combo = QComboBox()
        combo.addItems(self._species)
        combo.currentIndexChanged.connect(lambda _i, row=r: self._on_combo_changed(row))
        h.addWidget(spin, 1)
        h.addWidget(combo)
        cell.spin = spin
        cell.combo = combo
        self.table.setCellWidget(r, 1, cell)

        self._apply_value_range(r)
        spin.setValue(value)
        if 0 <= component < combo.count():
            combo.setCurrentIndex(component)
        self._sync_value_widget(r)
        if not self._loading:
            self._emit()

    def _cell(self, row):
        return self.table.cellWidget(row, 1)

    def _on_kind_changed(self, row):
        self._apply_value_range(row)
        self._sync_value_widget(row)
        self._emit()

    def _on_combo_changed(self, row):
        kind = self.table.cellWidget(row, 0).currentData()
        if kind in _KEY_RECOVERY and self.window_state:
            idx = self._cell(row).combo.currentIndex()
            if idx >= 0:
                if kind == SpecKind.LK_RECOVERY:
                    self.window_state.light_key_index = idx
                else:
                    self.window_state.heavy_key_index = idx
                self.window_state.is_modified = True
        self._emit()

    def _apply_value_range(self, row):
        """Fractions clamp to [0, 1]; duties are signed kW; the rest are
        non-negative ratios/rates."""
        kind = self.table.cellWidget(row, 0).currentData()
        spin = self._cell(row).spin
        if kind in _FRACTION:
            spin.setRange(0.0, 1.0); spin.setSuffix("")
        elif kind in _DUTY:
            spin.setRange(-1e9, 1e9); spin.setSuffix(" kW")
        else:
            spin.setRange(0.0, 1e9); spin.setSuffix("")

    def _sync_value_widget(self, row):
        """Show the trailing combo only for the kinds that need one, and point a
        recovery row's combo at the current global key."""
        kind = self.table.cellWidget(row, 0).currentData()
        combo = self._cell(row).combo
        if kind in _PURITY:
            combo.setVisible(True)
        elif kind in _KEY_RECOVERY:
            combo.setVisible(True)
            self._set_combo_to_key(row, kind)
        else:
            combo.setVisible(False)

    def _set_combo_to_key(self, row, kind):
        if not self.window_state:
            return
        combo = self._cell(row).combo
        if kind == SpecKind.LK_RECOVERY:
            idx = getattr(self.window_state, "light_key_index", 0) or 0
        else:
            idx = getattr(self.window_state, "heavy_key_index", None)
            if idx is None:
                lk = getattr(self.window_state, "light_key_index", 0) or 0
                idx = min(lk + 1, combo.count() - 1)
        combo.blockSignals(True)
        if 0 <= idx < combo.count():
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

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
        """Refresh the component/key dropdowns, preserving each row's selection."""
        self._species = list(names)
        for r in range(self.table.rowCount()):
            combo = self._cell(r).combo
            keep = combo.currentIndex()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self._species)
            if 0 <= keep < combo.count():
                combo.setCurrentIndex(keep)
            combo.blockSignals(False)
            self._sync_value_widget(r)   # recovery rows re-track the global key

    def get_specs(self):
        specs = []
        for r in range(self.table.rowCount()):
            kind = self.table.cellWidget(r, 0).currentData()
            cell = self._cell(r)
            value = cell.spin.value()
            comp = cell.combo.currentIndex() if kind in _PURITY else -1
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
