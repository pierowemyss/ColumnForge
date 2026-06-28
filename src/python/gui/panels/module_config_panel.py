from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QComboBox, QGroupBox, QGridLayout, QSpinBox
)
from PySide6.QtCore import Signal



class ModuleConfigPanel(QWidget):
    """Panel for configuring side module (Interreboiler, Side Stripper, Side Rectifier) settings."""

    configChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.module_types = ["Interreboiler", "Side Stripper", "Side Rectifier"]

        self._setup_ui()
        self._setup_styles()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Module type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Module Type:"))
        self.type_combo = QComboBox(self)
        self.type_combo.addItems(self.module_types)
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()
        main_layout.addLayout(type_layout)

        # Stage number
        stage_layout = QHBoxLayout()
        stage_layout.addWidget(QLabel("Stage Number:"))
        self.stage_spin = QSpinBox(self)
        self.stage_spin.setRange(1, 200)
        stage_layout.addWidget(self.stage_spin)
        stage_layout.addStretch()
        main_layout.addLayout(stage_layout)

        # Number of stages
        num_stages_layout = QHBoxLayout()
        num_stages_layout.addWidget(QLabel("Number of Stages:"))
        self.num_stages_spin = QSpinBox(self)
        self.num_stages_spin.setRange(1, 50)
        num_stages_layout.addWidget(self.num_stages_spin)
        num_stages_layout.addStretch()
        main_layout.addLayout(num_stages_layout)

        # Boilup/Reflux ratio
        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("Boilup Ratio (V/B):"))
        self.boilup_spin = QDoubleSpinBox(self)
        self.boilup_spin.setRange(0, 1000)
        self.boilup_spin.setDecimals(4)
        ratio_layout.addWidget(self.boilup_spin)
        ratio_layout.addStretch()
        main_layout.addLayout(ratio_layout)

        reflux_layout = QHBoxLayout()
        reflux_layout.addWidget(QLabel("Reflux Ratio (L/D):"))
        self.reflux_spin = QDoubleSpinBox(self)
        self.reflux_spin.setRange(0, 1000)
        self.reflux_spin.setDecimals(4)
        reflux_layout.addWidget(self.reflux_spin)
        reflux_layout.addStretch()
        main_layout.addLayout(reflux_layout)

        # Associated Streams
        streams_group = QGroupBox("Associated Streams")
        streams_layout = QGridLayout(streams_group)

        # Distillate
        streams_layout.addWidget(QLabel("Distillate:"), 0, 0)
        self.distillate_out_edit = QComboBox(self)
        self.distillate_out_edit.addItems(["out", "to tray"])
        streams_layout.addWidget(self.distillate_out_edit, 0, 1)

        streams_layout.addWidget(QLabel("To Tray #:"), 0, 2)
        self.distillate_tray_spin = QSpinBox(self)
        self.distillate_tray_spin.setRange(1, 200)
        streams_layout.addWidget(self.distillate_tray_spin, 0, 3)

        # Bottoms
        streams_layout.addWidget(QLabel("Bottoms:"), 1, 0)
        self.bottoms_out_edit = QComboBox(self)
        self.bottoms_out_edit.addItems(["out", "to tray"])
        streams_layout.addWidget(self.bottoms_out_edit, 1, 1)

        streams_layout.addWidget(QLabel("To Tray #:"), 1, 2)
        self.bottoms_tray_spin = QSpinBox(self)
        self.bottoms_tray_spin.setRange(1, 200)
        streams_layout.addWidget(self.bottoms_tray_spin, 1, 3)

        main_layout.addWidget(streams_group)

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
                min-width: 100px;
            }
        """)

    def _connect_signals(self):
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self.stage_spin.valueChanged.connect(self._on_value_changed)
        self.num_stages_spin.valueChanged.connect(self._on_value_changed)
        self.boilup_spin.valueChanged.connect(self._on_value_changed)
        self.reflux_spin.valueChanged.connect(self._on_value_changed)
        self.distillate_out_edit.currentTextChanged.connect(self._on_value_changed)
        self.distillate_tray_spin.valueChanged.connect(self._on_value_changed)
        self.bottoms_out_edit.currentTextChanged.connect(self._on_value_changed)
        self.bottoms_tray_spin.valueChanged.connect(self._on_value_changed)

    def _on_type_changed(self, module_type: str):
        """Handle module type change."""
        # Show/hide boilup/reflux based on module type
        if module_type == "Interreboiler":
            self.boilup_spin.setEnabled(True)
            self.reflux_spin.setEnabled(False)
        elif module_type in ["Side Stripper", "Side Rectifier"]:
            self.boilup_spin.setEnabled(False)
            self.reflux_spin.setEnabled(True)
        else:
            self.boilup_spin.setEnabled(True)
            self.reflux_spin.setEnabled(True)

        self._on_value_changed()

    def _on_value_changed(self):
        self.configChanged.emit()

    def set_config(self, config: dict):
        """Set module configuration from a dictionary."""
        module_type = config.get("type", "Interreboiler")
        index = self.type_combo.findText(module_type)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)

        self.stage_spin.setValue(config.get("stage", 1))
        self.num_stages_spin.setValue(config.get("num_stages", 1))
        self.boilup_spin.setValue(config.get("boilup_ratio", 0))
        self.reflux_spin.setValue(config.get("reflux_ratio", 0))

        # Associated streams
        streams = config.get("associated_streams", {})
        if "distillate" in streams:
            out_type, tray = streams["distillate"]
            self.distillate_out_edit.setCurrentText(out_type)
            self.distillate_tray_spin.setValue(tray)
        if "bottoms" in streams:
            out_type, tray = streams["bottoms"]
            self.bottoms_out_edit.setCurrentText(out_type)
            self.bottoms_tray_spin.setValue(tray)

    def get_config(self) -> dict:
        """Get module configuration as a dictionary."""
        module_type = self.type_combo.currentText()

        config = {
            "type": module_type,
            "stage": self.stage_spin.value(),
            "num_stages": self.num_stages_spin.value(),
            "boilup_ratio": self.boilup_spin.value() if self.boilup_spin.isEnabled() else None,
            "reflux_ratio": self.reflux_spin.value() if self.reflux_spin.isEnabled() else None,
            "associated_streams": {
                "distillate": (self.distillate_out_edit.currentText(), self.distillate_tray_spin.value()),
                "bottoms": (self.bottoms_out_edit.currentText(), self.bottoms_tray_spin.value())
            }
        }

        return config

    def get_specs(self) -> list:
        """Return list of active specifications for DoF tracking."""
        specs = []
        specs.append(f"Stage: {self.stage_spin.value()}")
        specs.append(f"Stages: {self.num_stages_spin.value()}")

        if self.boilup_spin.isEnabled() and self.boilup_spin.value() > 0:
            specs.append(f"Boilup ratio: {self.boilup_spin.value():.4f}")
        if self.reflux_spin.isEnabled() and self.reflux_spin.value() > 0:
            specs.append(f"Reflux ratio: {self.reflux_spin.value():.4f}")

        return specs
