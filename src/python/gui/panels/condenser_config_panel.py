from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QGroupBox, QCheckBox
)
from PySide6.QtCore import Signal

from .unit_combo_box import UnitComboBox
from .sci_spin_box import SciDoubleSpinBox


class CondenserConfigPanel(QWidget):
    """Panel for configuring condenser settings."""

    configChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._setup_ui()
        self._setup_styles()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Condenser type
        type_group = QGroupBox("Condenser Type")
        type_layout = QHBoxLayout(type_group)

        self.type_combo = QComboBox(self)
        self.type_combo.addItems(["Total", "Partial", "None"])
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()

        main_layout.addWidget(type_group)

        # Subcooling (Total condenser) — opt-in via checkbox
        self.subcooling_group = QGroupBox("Subcooling")
        # Consumed by the rigorous energy balance (Initialization → Flow Model);
        # under constant molar overflow it has no enthalpy to act on and is
        # ignored — the tooltip says so rather than silently dropping it.
        self.subcooling_group.setToolTip(
            "Subcooled reflux/distillate ΔT below the bubble point. Consumed by "
            "the rigorous energy balance (Initialization → Flow Model); ignored "
            "under constant molar overflow. ΔT is a delta — °C and K coincide.")
        subcooling_layout = QHBoxLayout(self.subcooling_group)
        self.subcooling_check = QCheckBox("Subcool below bubble point by")
        self.subcooling_input = UnitComboBox("temperature")
        self.subcooling_input.setValue(0)
        self.subcooling_input.setEnabled(False)
        subcooling_layout.addWidget(self.subcooling_check)
        subcooling_layout.addWidget(self.subcooling_input)
        main_layout.addWidget(self.subcooling_group)

        # Reflux ratio — an operating spec, but editable here for convenience.
        # It writes to the same window_state.specs list as the Operating slots.
        self.reflux_group = QGroupBox("Reflux")
        reflux_layout = QHBoxLayout(self.reflux_group)
        reflux_layout.addWidget(QLabel("Reflux Ratio (L/D):"))
        self.reflux_spin = SciDoubleSpinBox(self)
        self.reflux_spin.setRange(0, 1000)
        self.reflux_spin.setDecimals(4)
        reflux_layout.addWidget(self.reflux_spin)
        reflux_layout.addStretch()
        main_layout.addWidget(self.reflux_group)

        # Partial condenser: extra vapour-distillate product spec
        self.partial_group = QGroupBox("Partial Condenser Specifications")
        partial_layout = QHBoxLayout(self.partial_group)
        partial_layout.addWidget(QLabel("Vapor Distillate Flow:"))
        self.vapor_flow_input = UnitComboBox("flow")
        partial_layout.addWidget(self.vapor_flow_input)
        main_layout.addWidget(self.partial_group)

        hint = QLabel("Reflux ratio and distillate rate also appear in "
                      "Operating Specifications — same value, two places to edit.")
        hint.setProperty("hint", True)
        hint.setWordWrap(True)
        main_layout.addWidget(hint)

        main_layout.addStretch()

    def _setup_styles(self):
        # Styling comes from the central theme (gui/theme/app.qss).
        pass

    def _connect_signals(self):
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self.subcooling_check.toggled.connect(self.subcooling_input.setEnabled)
        self.subcooling_check.toggled.connect(self._on_value_changed)
        self.subcooling_input.valueChanged.connect(self._on_value_changed)
        self.reflux_spin.valueChanged.connect(self._on_value_changed)
        self.vapor_flow_input.valueChanged.connect(self._on_value_changed)

    def _on_type_changed(self, condenser_type: str):
        """Handle condenser type change."""
        self.reflux_group.setVisible(condenser_type != "None")
        self.subcooling_group.setVisible(condenser_type == "Total")
        self.partial_group.setVisible(condenser_type == "Partial")
        self._on_value_changed()

    def _on_value_changed(self):
        self.configChanged.emit()

    def set_config(self, config: dict):
        """Set condenser configuration from a dictionary."""
        condenser_type = config.get("type", "Total")
        index = self.type_combo.findText(condenser_type)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)

        subcool = config.get("subcooling_temp", 0)
        self.subcooling_check.setChecked(bool(subcool))
        self.subcooling_input.setValue(subcool or 0)  # ΔT, not absolute
        self.reflux_spin.setValue(config.get("reflux_ratio", 0) or 0)
        self.vapor_flow_input.setValueInSI(config.get("vapor_distillate_flow", 0) or 0)

    def get_config(self) -> dict:
        """Type + subcooling (-> condenser_config) plus the operating values
        reflux_ratio / vapor_distillate_flow (-> the shared specs list)."""
        condenser_type = self.type_combo.currentText()
        config = {"type": condenser_type}
        if (condenser_type == "Total" and self.subcooling_check.isChecked()
                and self.subcooling_input.value() > 0):
            # ponytail: subcooling is a ΔT — store raw; °C/K deltas match, add °F-delta handling if needed
            config["subcooling_temp"] = self.subcooling_input.value()
        if condenser_type != "None" and self.reflux_spin.value() > 0:
            config["reflux_ratio"] = self.reflux_spin.value()
        if condenser_type == "Partial" and self.vapor_flow_input.value() > 0:
            config["vapor_distillate_flow"] = self.vapor_flow_input.valueInSI()
        return config
