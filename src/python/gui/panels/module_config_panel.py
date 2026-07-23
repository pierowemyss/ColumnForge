from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QGridLayout, QSpinBox
)
from PySide6.QtCore import Signal

from .sci_spin_box import SciDoubleSpinBox


# Per module type: which rows are shown, how they're labelled, and the one-line
# hint under the form. Everything listed here is consumed by a solver — a row the
# solver ignores is not shown at all (no greyed-out dead fields).
_TYPES = {
    "Interreboiler": {
        "rows": ("stage", "duty"),
        "labels": {"stage": "Stage (0 = top):",
                   "duty": "Duty (kW, + heat / − cool):"},
        "hint": "A signed heat term on one stage. Needs the energy balance "
                "(Initialization → Flow Model); under CMO it would be ignored.",
    },
    "Pumparound": {
        "rows": ("stage", "return_stage", "rate", "duty"),
        "labels": {"stage": "Draw stage (0 = top):",
                   "return_stage": "Return stage (above draw):",
                   "rate": "Circulation rate (kmol/h):",
                   "duty": "Cooler duty (kW removed):"},
        "hint": "Liquid drawn, cooled, and returned higher up — no product "
                "leaves. Needs the energy balance.",
    },
    "Side Stripper": {
        "rows": ("stage", "return_stage", "rate", "num_stages", "ratio"),
        "labels": {"stage": "Draw stage (0 = top):",
                   "return_stage": "Vapour return stage (above draw):",
                   "rate": "Liquid draw rate (kmol/h):",
                   "num_stages": "Stripper stages:",
                   "ratio": "Boilup ratio (V/B):"},
        "hint": "Liquid drawn to a reboiled side column: its bottoms is a side "
                "product, its overhead vapour returns above the draw.",
    },
    "Side Rectifier": {
        "rows": ("stage", "return_stage", "rate", "num_stages", "ratio"),
        "labels": {"stage": "Draw stage (0 = top):",
                   "return_stage": "Liquid return stage (below draw):",
                   "rate": "Vapour draw rate (kmol/h):",
                   "num_stages": "Rectifier stages:",
                   "ratio": "Reflux ratio (L/D):"},
        "hint": "Vapour drawn to a condensed side column: its distillate is a "
                "side product, its liquid returns below the draw.",
    },
}


class ModuleConfigPanel(QWidget):
    """Configure one side module. The form is type-driven: only the fields the
    chosen module type actually feeds into the solver are shown."""

    configChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.module_types = list(_TYPES)
        self._loading = False

        self._setup_ui()
        self._connect_signals()
        self._on_type_changed(self.type_combo.currentText())

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Module Type:"))
        self.type_combo = QComboBox(self)
        self.type_combo.addItems(self.module_types)
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()
        main_layout.addLayout(type_layout)

        # One grid of (label, widget) rows, shown/hidden per type.
        grid = QGridLayout()
        grid.setColumnStretch(2, 1)
        self.rows = {}

        def add_row(key, widget):
            row = len(self.rows)
            label = QLabel("", self)
            grid.addWidget(label, row, 0)
            grid.addWidget(widget, row, 1)
            self.rows[key] = (label, widget)

        # Stages are 0-based from the top; 0 = distillate, matching feeds/draws.
        self.stage_spin = QSpinBox(self)
        self.stage_spin.setRange(0, 199)
        add_row("stage", self.stage_spin)

        self.return_stage_spin = QSpinBox(self)
        self.return_stage_spin.setRange(0, 199)
        add_row("return_stage", self.return_stage_spin)

        self.rate_spin = SciDoubleSpinBox(self)
        self.rate_spin.setRange(0, 1e9)
        self.rate_spin.setDecimals(3)
        add_row("rate", self.rate_spin)

        self.num_stages_spin = QSpinBox(self)
        self.num_stages_spin.setRange(1, 50)
        add_row("num_stages", self.num_stages_spin)

        # Boilup (V/B) for a stripper, reflux (L/D) for a rectifier — one spin,
        # relabelled; get_config maps it back onto the right stored field.
        self.ratio_spin = SciDoubleSpinBox(self)
        self.ratio_spin.setRange(0, 1000)
        self.ratio_spin.setDecimals(4)
        add_row("ratio", self.ratio_spin)

        self.duty_spin = SciDoubleSpinBox(self)
        self.duty_spin.setRange(-1e9, 1e9)
        self.duty_spin.setDecimals(3)
        add_row("duty", self.duty_spin)

        main_layout.addLayout(grid)

        self.hint_label = QLabel("", self)
        self.hint_label.setWordWrap(True)
        self.hint_label.setProperty("hint", True)
        main_layout.addWidget(self.hint_label)

        main_layout.addStretch()

    def _connect_signals(self):
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        for _, widget in self.rows.values():
            widget.valueChanged.connect(self._on_value_changed)

    def _on_type_changed(self, module_type: str):
        spec = _TYPES.get(module_type, _TYPES["Interreboiler"])
        for key, (label, widget) in self.rows.items():
            shown = key in spec["rows"]
            label.setText(spec["labels"].get(key, ""))
            label.setVisible(shown)
            widget.setVisible(shown)
        self.hint_label.setText(spec["hint"])
        self._on_value_changed()

    def _on_value_changed(self):
        if not self._loading:
            self.configChanged.emit()

    def _is(self, *types) -> bool:
        return self.type_combo.currentText() in types

    def set_config(self, config: dict):
        """Set module configuration from a dictionary."""
        self._loading = True                 # one configChanged, not one per field
        try:
            module_type = config.get("type", "Interreboiler")
            index = self.type_combo.findText(module_type)
            if index >= 0:
                self.type_combo.setCurrentIndex(index)

            self.stage_spin.setValue(config.get("stage", 1))
            self.num_stages_spin.setValue(config.get("num_stages") or 4)
            self.duty_spin.setValue(config.get("duty") or 0)
            self.return_stage_spin.setValue(config.get("return_stage") or 0)
            self.rate_spin.setValue(config.get("rate") or 0)
            ratio = (config.get("reflux_ratio") if self._is("Side Rectifier")
                     else config.get("boilup_ratio"))
            self.ratio_spin.setValue(ratio or 0)
            # inside the guard: loading a config re-labels the form but must not
            # echo a configChanged back at whoever is loading it
            self._on_type_changed(self.type_combo.currentText())
        finally:
            self._loading = False

    def _visible(self, key, widget):
        """Value of a row, or None when this module type doesn't have it.
        Keyed off the type table, not Qt visibility — a panel sitting on a hidden
        stack page is invisible but its values are still real."""
        spec = _TYPES.get(self.type_combo.currentText(), _TYPES["Interreboiler"])
        return widget.value() if key in spec["rows"] else None

    def get_config(self) -> dict:
        """Get module configuration as a dictionary. Fields the current type does
        not show come back None, so nothing invisible reaches the solver."""
        ratio = self._visible("ratio", self.ratio_spin)
        return {
            "type": self.type_combo.currentText(),
            "stage": self.stage_spin.value(),
            "num_stages": self._visible("num_stages", self.num_stages_spin),
            "boilup_ratio": ratio if self._is("Side Stripper") else None,
            "reflux_ratio": ratio if self._is("Side Rectifier") else None,
            "duty": self._visible("duty", self.duty_spin) or None,
            "return_stage": self._visible("return_stage", self.return_stage_spin),
            "rate": self._visible("rate", self.rate_spin) or None,
        }

    def get_specs(self) -> list:
        """Return list of active specifications for DoF tracking."""
        cfg = self.get_config()
        specs = [f"Stage: {cfg['stage']}"]
        if cfg["num_stages"]:
            specs.append(f"Stages: {cfg['num_stages']}")
        if cfg["boilup_ratio"]:
            specs.append(f"Boilup ratio: {cfg['boilup_ratio']:.4f}")
        if cfg["reflux_ratio"]:
            specs.append(f"Reflux ratio: {cfg['reflux_ratio']:.4f}")
        if cfg["rate"]:
            specs.append(f"Rate: {cfg['rate']:.3f} kmol/h")
        if cfg["duty"]:
            specs.append(f"Duty: {cfg['duty']:.3f} kW")
        return specs
