from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QComboBox, QGroupBox
)
from PySide6.QtCore import Signal

from .unit_combo_box import UnitComboBox


class ReboilerConfigPanel(QWidget):
    """Reboiler type plus its operating specs (boilup ratio / bottoms rate).

    The operating values are written to the same window_state.specs list as the
    Operating Specifications slots — one source, two places to edit."""

    configChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_styles()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        type_group = QGroupBox("Reboiler Type")
        type_layout = QHBoxLayout(type_group)
        self.type_combo = QComboBox(self)
        self.type_combo.addItems(["Kettle", "Thermosiphon", "None"])
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()
        main_layout.addWidget(type_group)

        self.specs_group = QGroupBox("Reboiler Specifications")
        specs_layout = QVBoxLayout(self.specs_group)

        boilup_layout = QHBoxLayout()
        boilup_layout.addWidget(QLabel("Boilup Ratio (V/B):"))
        self.boilup_spin = QDoubleSpinBox(self)
        self.boilup_spin.setRange(0, 1000)
        self.boilup_spin.setDecimals(4)
        boilup_layout.addWidget(self.boilup_spin)
        boilup_layout.addStretch()
        specs_layout.addLayout(boilup_layout)

        bottoms_layout = QHBoxLayout()
        bottoms_layout.addWidget(QLabel("Bottoms Flow:"))
        self.bottoms_flow_input = UnitComboBox("flow")
        bottoms_layout.addWidget(self.bottoms_flow_input)
        specs_layout.addLayout(bottoms_layout)

        main_layout.addWidget(self.specs_group)

        hint = QLabel("These also appear in Operating Specifications — "
                      "same value, two places to edit.")
        hint.setStyleSheet("color: #666666; font-style: italic;")
        hint.setWordWrap(True)
        main_layout.addWidget(hint)

        main_layout.addStretch()

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
            QLabel {
                min-width: 120px;
            }
        """)

    def _connect_signals(self):
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self.boilup_spin.valueChanged.connect(self._on_value_changed)
        self.bottoms_flow_input.valueChanged.connect(self._on_value_changed)

    def _on_type_changed(self, reboiler_type: str):
        self.specs_group.setVisible(reboiler_type != "None")
        self._on_value_changed()

    def _on_value_changed(self):
        self.configChanged.emit()

    def set_config(self, config: dict):
        reboiler_type = config.get("type", "Kettle")
        index = self.type_combo.findText(reboiler_type)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)
        self.boilup_spin.setValue(config.get("boilup_ratio", 0) or 0)
        self.bottoms_flow_input.setValueInSI(config.get("bottoms_flow", 0) or 0)

    def get_config(self) -> dict:
        reboiler_type = self.type_combo.currentText()
        config = {"type": reboiler_type}
        if reboiler_type != "None":
            if self.boilup_spin.value() > 0:
                config["boilup_ratio"] = self.boilup_spin.value()
            if self.bottoms_flow_input.value() > 0:
                config["bottoms_flow"] = self.bottoms_flow_input.valueInSI()
        return config
