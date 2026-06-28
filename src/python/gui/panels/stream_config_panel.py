from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QGroupBox, QComboBox, QHeaderView, QSpinBox
)
from PySide6.QtCore import Signal

from .unit_combo_box import UnitComboBox


class StreamConfigPanel(QWidget):
    """Panel for configuring stream properties."""

    streamChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_stream_id = None
        self.species_names = []
        self.stream_types = ["Feed", "Distillate", "Bottoms", "Sidestream"]
        self.window_state = None

        self._setup_ui()
        self._setup_styles()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Stream header
        self.header_label = QLabel("Select a stream to configure")
        self.header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(self.header_label)

        # Stream type dropdown
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox(self)
        self.type_combo.addItems(self.stream_types)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()
        main_layout.addLayout(type_layout)

        # Stage number (for all streams except those at condenser/reboiler)
        self.stage_container = QWidget()
        stage_layout = QHBoxLayout(self.stage_container)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.addWidget(QLabel("Stage #:"))
        self.stage_spin = QSpinBox(self)
        self.stage_spin.setRange(1, 200)
        self.stage_spin.valueChanged.connect(self._on_value_changed)
        stage_layout.addWidget(self.stage_spin)
        stage_layout.addStretch()
        main_layout.addWidget(self.stage_container)

        # Temperature
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temp_input = UnitComboBox("temperature")
        self.temp_input.valueChanged.connect(self._on_value_changed)
        temp_layout.addWidget(self.temp_input)
        main_layout.addLayout(temp_layout)

        # Flow
        flow_layout = QHBoxLayout()
        flow_layout.addWidget(QLabel("Flow:"))
        self.flow_input = UnitComboBox("flow")
        self.flow_input.valueChanged.connect(self._on_value_changed)
        flow_layout.addWidget(self.flow_input)
        main_layout.addLayout(flow_layout)

        # Saturation buttons (for feed streams)
        self.saturation_group = QWidget()
        saturation_layout = QHBoxLayout(self.saturation_group)
        saturation_layout.setContentsMargins(0, 0, 0, 0)

        self.sat_liquid_btn = QPushButton("Auto-saturate to Liquid")
        self.sat_liquid_btn.clicked.connect(lambda: self._on_saturate("liquid"))
        saturation_layout.addWidget(self.sat_liquid_btn)

        self.sat_vapor_btn = QPushButton("Auto-saturate to Vapor")
        self.sat_vapor_btn.clicked.connect(lambda: self._on_saturate("vapor"))
        saturation_layout.addWidget(self.sat_vapor_btn)

        main_layout.addWidget(self.saturation_group)

        # Composition table
        comp_group = QGroupBox("Composition")
        comp_layout = QVBoxLayout(comp_group)

        self.comp_table = QTableWidget(0, 2)
        self.comp_table.cellChanged.connect(self._on_comp_changed)
        self.comp_table.setHorizontalHeaderLabels(["Species", "Mole Fraction"])
        self.comp_table.horizontalHeader().setStretchLastSection(True)
        self.comp_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        comp_layout.addWidget(self.comp_table)

        # Composition validation
        self.comp_sum_label = QLabel("Sum: 0.0000")
        self.comp_sum_label.setStyleSheet("font-weight: bold;")
        comp_layout.addWidget(self.comp_sum_label)

        main_layout.addWidget(comp_group)

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
                min-width: 80px;
            }
            QPushButton {
                min-width: 120px;
            }
        """)

    def set_species_list(self, names: list):
        """Set the list of available species names."""
        self.species_names = names
        self._rebuild_comp_table()

    def set_window_state(self, window_state):
        """Set window state and update species list from it."""
        self.window_state = window_state
        species = list(window_state.species.keys()) if window_state and window_state.species else []
        self.set_species_list(species)

    def _rebuild_comp_table(self):
        """Rebuild the composition table with current species."""
        # Block signals: programmatic fills must not fire cellChanged -> save,
        # which would clobber the stored stream composition with these zeros.
        self.comp_table.blockSignals(True)
        self.comp_table.setRowCount(len(self.species_names))
        for i, name in enumerate(self.species_names):
            self.comp_table.setItem(i, 0, QTableWidgetItem(name))
            self.comp_table.setItem(i, 1, QTableWidgetItem("0.0000"))
        self.comp_table.blockSignals(False)

    def select_stream(self, stream_id: str, stream_data: dict = None):
        """Select a stream to configure."""
        self.current_stream_id = stream_id
        if stream_id:
            self.header_label.setText(f"Configure Stream: {stream_id}")
            if stream_data is not None:
                self._load_stream_data(stream_data)
        else:
            self.clear()

    def _on_saturate(self, phase: str):
        """Set the stream temperature to its bubble (liquid) or dew (vapour) point
        at the current composition, pressure and Antoine data. No-op with a header
        note if the setup is incomplete.
        # ponytail: T is in the Antoine fit's unit (bundled fits are degrees C);
        # set_value goes straight into the temperature box."""
        ws = self.window_state
        if not ws or not ws.species:
            self.header_label.setText("Define species first (Initialization).")
            return
        order = list(ws.species.keys())
        comp = self.get_stream_data().get("composition", {})
        z = [float(comp.get(nm, 0.0)) for nm in order]
        if abs(sum(z) - 1.0) > 1e-3:
            self.header_label.setText(f"Composition sums to {sum(z):.3f}, not 1.")
            return
        try:
            antoine = ws.thermodynamics_config.psat_params(order)  # (N,3) or (N,7) PLXANT
        except ValueError as exc:
            self.header_label.setText(str(exc))
            return

        import numpy as np
        from core.thermodynamics import bubble_T, dew_T
        gamma_fn = ws.build_gamma_fn(order)
        solver = bubble_T if phase == "liquid" else dew_T
        try:
            T = solver(np.array(z), float(ws.pressure), antoine,
                       gamma_fn=gamma_fn)
        except Exception as exc:
            self.header_label.setText(f"Saturation failed: {exc}")
            return
        self.temp_input.setValueInSI(float(T))
        self._on_value_changed()

    def _load_stream_data(self, data: dict):
        """Load stream data into UI."""
        # Stream type
        stream_type = data.get("type", "Feed")
        index = self.type_combo.findText(stream_type)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)

        # Stage
        self.stage_spin.setValue(data.get("stage", 1))

        # Temperature / flow are stored in SI; convert back to the displayed unit
        temp = data.get("temperature")
        self.temp_input.setValueInSI(float(temp)) if temp is not None else self.temp_input.setValue(0.0)

        flow = data.get("flow")
        self.flow_input.setValueInSI(float(flow)) if flow is not None else self.flow_input.setValue(0.0)

        # Composition (block signals: loading is not a user edit, must not save)
        composition = data.get("composition", {})
        self.comp_table.blockSignals(True)
        for row in range(self.comp_table.rowCount()):
            species = self.comp_table.item(row, 0).text()
            if species in composition:
                self.comp_table.item(row, 1).setText(f"{composition[species]:.4f}")
        self.comp_table.blockSignals(False)

        self._update_comp_sum()

    def clear(self):
        """Clear all stream configuration values."""
        self.current_stream_id = None
        self.header_label.setText("Select a stream to configure")
        self.type_combo.setCurrentIndex(0)
        self.stage_spin.setValue(1)
        self.temp_input.setValue(0)
        self.flow_input.setValue(0)
        self._rebuild_comp_table()
        self._update_comp_sum()

    def _on_type_changed(self, stream_type: str):
        """Handle stream type change."""
        # Show/hide stage spinner based on stream type
        is_fixed_stage = stream_type in ["Distillate", "Bottoms"]
        self.stage_container.setVisible(not is_fixed_stage)
        self.stage_spin.setEnabled(not is_fixed_stage)

        # Show/hide saturation buttons for feed streams
        is_feed_stream = stream_type == "Feed"
        self.saturation_group.setVisible(is_feed_stream)
        self.sat_liquid_btn.setEnabled(is_feed_stream)
        self.sat_vapor_btn.setEnabled(is_feed_stream)

        # Show/hide flow for distillate/bottoms
        is_product = stream_type in ["Distillate", "Bottoms"]
        self.flow_input.setEnabled(is_product or is_feed_stream)

        self._on_value_changed()

    def _on_value_changed(self):
        """Handle any value change."""
        if self.current_stream_id:
            self.streamChanged.emit()

    def _on_comp_changed(self, row, column):
        """Handle composition change."""
        # Validate and normalize compositions
        self._validate_compositions()
        self._update_comp_sum()
        self._on_value_changed()

    def _validate_compositions(self):
        """Validate that compositions are valid numbers and sum to 1.0."""
        total = 0.0
        values = []

        for row in range(self.comp_table.rowCount()):
            try:
                item = self.comp_table.item(row, 1)
                if item is None or not item.text():
                    value = 0.0
                else:
                    value = float(item.text())
                value = max(0, min(1, value))
                item = self.comp_table.item(row, 1)
                if item is not None:
                    item.setText(f"{value:.4f}")
                values.append(value)
                total += value
            except ValueError:
                item = self.comp_table.item(row, 1)
                if item is not None:
                    item.setText("0.0000")
                values.append(0)
        # ponytail: no auto-normalize; sum is shown by _update_comp_sum, enforced at solve time

    def _update_comp_sum(self):
        """Update the composition sum label."""
        total = 0.0
        for row in range(self.comp_table.rowCount()):
            try:
                item = self.comp_table.item(row, 1)
                if item and item.text():
                    total += float(item.text())
            except ValueError:
                pass

        self.comp_sum_label.setText(f"Sum: {total:.4f}")
        if abs(total - 1.0) > 0.0001:
            self.comp_sum_label.setStyleSheet("font-weight: bold; color: red;")
        else:
            self.comp_sum_label.setStyleSheet("font-weight: bold; color: green;")

    def get_stream_data(self) -> dict:
        """Get the current stream configuration data."""
        if not self.current_stream_id:
            return {}

        composition = {}
        for row in range(self.comp_table.rowCount()):
            species_item = self.comp_table.item(row, 0)
            value_item = self.comp_table.item(row, 1)
            if species_item and species_item.text():
                species = species_item.text()
                try:
                    composition[species] = float(value_item.text()) if value_item and value_item.text() else 0.0
                except ValueError:
                    composition[species] = 0.0

        return {
            "id": self.current_stream_id,
            "type": self.type_combo.currentText(),
            "stage": self.stage_spin.value(),
            "temperature": self.temp_input.valueInSI(),
            "pressure": None,  # Same as column
            "flow": self.flow_input.valueInSI() if self.flow_input.isEnabled() else None,
            "composition": composition
        }
