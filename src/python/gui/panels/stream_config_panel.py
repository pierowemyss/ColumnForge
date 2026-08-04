from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QGroupBox, QComboBox, QHeaderView, QSpinBox, QApplication
)
from PySide6.QtCore import Signal

from .unit_combo_box import UnitComboBox
from ..table_edit import parse_number, fmt_number
from ..theme import set_state


class StreamConfigPanel(QWidget):
    """Panel for configuring stream properties."""

    streamChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_stream_id = None
        self.species_names = []
        self.stream_types = ["Feed", "Distillate", "Bottoms", "Sidestream"]
        self.window_state = None
        # True while programmatically loading a stream: suppresses the save that
        # widget setters would otherwise fire (which writes the outgoing stream's
        # table into the newly selected one).
        self._loading = False

        self._setup_ui()
        self._setup_styles()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Stream header
        self.header_label = QLabel("Select a stream to configure")
        self.header_label.setObjectName("panelHeader")
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
        stage_layout.addWidget(QLabel("Stage # (0 = distillate):"))
        self.stage_spin = QSpinBox(self)
        self.stage_spin.setRange(0, 200)
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
        self.flow_input.set_mw_provider(self._avg_mw)
        self.flow_input.valueChanged.connect(self._on_value_changed)
        flow_layout.addWidget(self.flow_input)
        main_layout.addLayout(flow_layout)

        # Draw phase (sidestreams only): the solvers support vapor draws via
        # SolverInput.vapor_draw; this is the UI that finally exposes it.
        self.phase_container = QWidget()
        phase_layout = QHBoxLayout(self.phase_container)
        phase_layout.setContentsMargins(0, 0, 0, 0)
        phase_layout.addWidget(QLabel("Draw phase:"))
        self.phase_combo = QComboBox(self)
        self.phase_combo.addItems(["Liquid", "Vapor"])
        self.phase_combo.currentTextChanged.connect(self._on_value_changed)
        phase_layout.addWidget(self.phase_combo)
        phase_layout.addStretch()
        self.phase_container.setVisible(False)   # shown for Sidestream only
        main_layout.addWidget(self.phase_container)

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
        set_state(self.comp_sum_label, "neutral")
        comp_layout.addWidget(self.comp_sum_label)

        main_layout.addWidget(comp_group)

        main_layout.addStretch()

    def _setup_styles(self):
        # Styling comes from the central theme (gui/theme/app.qss).
        pass

    def set_species_list(self, names: list):
        """Set the list of available species names."""
        self.species_names = names
        self._rebuild_comp_table()

    def set_window_state(self, window_state):
        """Set window state and update species list from it."""
        self.window_state = window_state
        species = list(window_state.species.keys()) if window_state and window_state.species else []
        self.set_species_list(species)
        # Rebuild zeroed the table; refill comp from the selected stream so the
        # live table matches stored state. Else saturate/save read zeros over it.
        sid = self.current_stream_id
        if window_state and sid and sid in window_state.streams:
            self._fill_comp(window_state.streams[sid].composition)

    def _fill_comp(self, composition: dict):
        """Write a stored composition into the table without firing saves. Every
        row is set (missing species -> 0) so switching streams never leaves the
        previous stream's numbers behind."""
        self.comp_table.blockSignals(True)
        for row in range(self.comp_table.rowCount()):
            species = self.comp_table.item(row, 0).text()
            self.comp_table.item(row, 1).setText(f"{composition.get(species, 0.0):.4f}")
        self.comp_table.blockSignals(False)
        self._update_comp_sum()

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

    def flush_pending_edits(self):
        """Commit any in-progress edit (an open composition-cell editor or a
        typed-but-unconfirmed spinbox) into the CURRENT stream before another
        stream is loaded. Clicking a different stream never discards an edit.
        """
        for spin in (self.stage_spin, self.temp_input.spin_box,
                     self.flow_input.spin_box):
            spin.interpretText()        # parse typed text -> valueChanged -> save
        ed = QApplication.focusWidget()
        if ed is not None and self.comp_table.isAncestorOf(ed):
            # focus-out makes the item delegate commit -> cellChanged -> save
            ed.clearFocus()

    def select_stream(self, stream_id: str, stream_data: dict = None):
        """Select a stream to configure."""
        if stream_id != self.current_stream_id:
            self.flush_pending_edits()
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
        P = ws.thermodynamics_config.pressure_in_psat_unit(ws.pressure)  # bar -> Psat unit
        try:
            T = solver(np.array(z), P, antoine, gamma_fn=gamma_fn)
        except Exception as exc:
            self.header_label.setText(f"Saturation failed: {exc}")
            return
        # T is in the fit's unit (bundled fits: degC); SI storage is Kelvin.
        self.temp_input.setValueInSI(float(T) + 273.15)
        # Name the model used: "still on Antoine" is the classic surprise when a
        # user entered PLXANT params but never switched the active vle_model.
        vle = ws.thermodynamics_config.vle_model
        self.header_label.setText(
            f"Saturated ({phase}) to {float(T):.1f}°C using {vle}.")
        self._on_value_changed()

    def _load_stream_data(self, data: dict):
        """Load stream data into UI. _loading suppresses the save these setters
        would otherwise fire mid-load (corrupting the just-selected stream)."""
        self._loading = True
        try:
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

            phase = (data.get("phase") or "liquid").capitalize()
            idx = self.phase_combo.findText(phase)
            self.phase_combo.setCurrentIndex(idx if idx >= 0 else 0)

            # Composition (block signals: loading is not a user edit, must not save)
            self._fill_comp(data.get("composition", {}))
        finally:
            self._loading = False

    def clear(self):
        """Clear all stream configuration values."""
        self.current_stream_id = None
        self.header_label.setText("Select a stream to configure")
        self.type_combo.setCurrentIndex(0)
        self.stage_spin.setValue(1)
        self.temp_input.setValue(0)
        self.flow_input.setValue(0)
        self.phase_combo.setCurrentIndex(0)
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

        # Draw phase only applies to sidestreams
        self.phase_container.setVisible(stream_type == "Sidestream")

        self._on_value_changed()

    def _on_value_changed(self):
        """Handle any value change."""
        if self._loading:
            return
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
            item = self.comp_table.item(row, 1)
            # parse_number/fmt_number, not float()/"%.4f": a pasted unicode
            # minus used to zero the cell, and 4 decimals rounded a 5e-6 trace
            # component away to nothing.
            value = max(0.0, min(1.0, parse_number(item.text() if item else "", 0.0)))
            if item is not None:
                item.setText(fmt_number(value))
            values.append(value)
            total += value
        # ponytail: no auto-normalize; sum is shown by _update_comp_sum, enforced at solve time

    def _avg_mw(self):
        """Mixture molar mass (kg/kmol) of the current composition table, or
        None when the composition is empty or any present species lacks an MW
        — the kg/h flow unit greys out in that case."""
        if not self.window_state:
            return None
        total = mass = 0.0
        for row in range(self.comp_table.rowCount()):
            name_item = self.comp_table.item(row, 0)
            val_item = self.comp_table.item(row, 1)
            frac = parse_number(val_item.text() if val_item else "", 0.0)
            if frac <= 0.0:
                continue
            sp = self.window_state.species.get(name_item.text()) if name_item else None
            if sp is None or not sp.mw:
                return None
            total += frac
            mass += frac * float(sp.mw)
        return mass / total if total > 0 else None

    def _update_comp_sum(self):
        """Update the composition sum label."""
        total = 0.0
        for row in range(self.comp_table.rowCount()):
            item = self.comp_table.item(row, 1)
            total += parse_number(item.text() if item else "", 0.0)

        self.comp_sum_label.setText(f"Sum: {total:.4f}")
        set_state(self.comp_sum_label,
                  "error" if abs(total - 1.0) > 0.0001 else "ok")
        self.flow_input.refresh_units()      # composition drives kg/h validity

    def get_stream_data(self) -> dict:
        """Get the current stream configuration data."""
        if not self.current_stream_id:
            return {}

        composition = {}
        for row in range(self.comp_table.rowCount()):
            species_item = self.comp_table.item(row, 0)
            value_item = self.comp_table.item(row, 1)
            if species_item and species_item.text():
                composition[species_item.text()] = parse_number(
                    value_item.text() if value_item else "", 0.0)

        return {
            "id": self.current_stream_id,
            "type": self.type_combo.currentText(),
            "stage": self.stage_spin.value(),
            "temperature": self.temp_input.valueInSI(),
            "flow": self.flow_input.valueInSI() if self.flow_input.isEnabled() else None,
            "composition": composition,
            "phase": self.phase_combo.currentText().lower(),
        }
