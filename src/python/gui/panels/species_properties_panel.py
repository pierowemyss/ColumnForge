from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QPushButton, QGroupBox, QGridLayout
)
from PySide6.QtCore import Signal

from gui.state.window_state import Species


class SpeciesPropertiesPanel(QWidget):
    """Panel for displaying and editing physical properties of the selected species."""

    propertiesChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_species = None
        self.species_names = []
        self.window_state = None
        # Set while _load_species_from_state repopulates the widgets. The spinboxes
        # are *children*, so self.blockSignals() never stopped their valueChanged
        # from calling _update_species_from_ui() mid-load — which wrote the stale
        # (still-empty) group table back over the species' real unifac_groups.
        self._loading = False

        self._setup_ui()
        self._setup_styles()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        self.header_label = QLabel("Select a species to view properties")
        self.header_label.setObjectName("panelHeader")
        main_layout.addWidget(self.header_label)

        properties_group = QGroupBox("Physical Properties")
        properties_layout = QGridLayout(properties_group)

        properties_layout.addWidget(QLabel("Name:"), 0, 0)
        self.name_edit = QLineEdit(self)
        self.name_edit.editingFinished.connect(self._on_name_changed)
        properties_layout.addWidget(self.name_edit, 0, 1)

        properties_layout.addWidget(QLabel("MW (g/mol):"), 1, 0)
        self.mw_spin = QDoubleSpinBox(self)
        self.mw_spin.setRange(0, 10000)
        self.mw_spin.setDecimals(2)
        self.mw_spin.setValue(0)
        self.mw_spin.valueChanged.connect(self._on_property_changed)
        properties_layout.addWidget(self.mw_spin, 1, 1)

        # Honesty policy: density and Cp are stored but no solver consumes them
        # yet (enthalpy layer + column sizing will) — greyed, not ignored.
        _unused_tip = "Stored, but not consumed by any solver yet."
        properties_layout.addWidget(QLabel("ρ (kg/m³) (not used yet):"), 2, 0)
        self.density_spin = QDoubleSpinBox(self)
        self.density_spin.setRange(0, 10000)
        self.density_spin.setDecimals(2)
        self.density_spin.setValue(0)
        self.density_spin.setEnabled(False)
        self.density_spin.setToolTip(_unused_tip)
        self.density_spin.valueChanged.connect(self._on_property_changed)
        properties_layout.addWidget(self.density_spin, 2, 1)

        properties_layout.addWidget(QLabel("Cp (J/mol·K) (not used yet):"), 3, 0)
        self.cp_spin = QDoubleSpinBox(self)
        self.cp_spin.setRange(0, 10000)
        self.cp_spin.setDecimals(2)
        self.cp_spin.setValue(0)
        self.cp_spin.setEnabled(False)
        self.cp_spin.setToolTip(_unused_tip)
        self.cp_spin.valueChanged.connect(self._on_property_changed)
        properties_layout.addWidget(self.cp_spin, 3, 1)

        main_layout.addWidget(properties_group)

        unifac_group = QGroupBox("UNIFAC Groups")
        unifac_group.setToolTip(
            "Consumed by the UNIFAC activity model and by 'Estimate UNIQUAC "
            "r/q'. Group names must match core/data/unifac_groups.json "
            "(e.g. CH3, CH2, OH, H2O, ACH).")
        unifac_layout = QVBoxLayout(unifac_group)

        self.unifac_table = QTableWidget(0, 2)
        self.unifac_table.setHorizontalHeaderLabels(["Group", "Count"])
        self.unifac_table.horizontalHeader().setStretchLastSection(True)
        self.unifac_table.cellChanged.connect(self._on_unifac_changed)
        unifac_layout.addWidget(self.unifac_table)

        unifac_buttons = QHBoxLayout()
        self.add_group_btn = QPushButton("Add Group")
        self.add_group_btn.clicked.connect(self._add_unifac_group)
        self.remove_group_btn = QPushButton("Remove Group")
        self.remove_group_btn.clicked.connect(self._remove_unifac_group)
        unifac_buttons.addWidget(self.add_group_btn)
        unifac_buttons.addWidget(self.remove_group_btn)
        unifac_layout.addLayout(unifac_buttons)

        main_layout.addWidget(unifac_group)

        self.unifac_btn = QPushButton("Estimate UNIQUAC r/q from groups")
        self.unifac_btn.setToolTip(
            "Fills this species' UNIQUAC structural r and q by summing group "
            "R_k / Q_k (r = Σ ν·R, q = Σ ν·Q).")
        self.unifac_btn.setEnabled(False)
        self.unifac_btn.clicked.connect(self._estimate_uniquac)
        main_layout.addWidget(self.unifac_btn)

        main_layout.addStretch()

    def _setup_styles(self):
        # Styling comes from the central theme (gui/theme/app.qss).
        pass

    def set_window_state(self, window_state):
        """Set the window state reference."""
        self.window_state = window_state

    def _on_property_changed(self):
        if self.current_species and not self._loading:
            self._update_species_from_ui()
            self.propertiesChanged.emit()

    def _on_name_changed(self):
        if not self.current_species or not self.window_state:
            return
        
        new_name = self.name_edit.text().strip()
        if not new_name:
            self.name_edit.setText(self.current_species)
            return
        
        if new_name == self.current_species:
            return
        
        if new_name in self.species_names:
            self.name_edit.setText(self.current_species)
            return
        
        old_name = self.current_species
        success = self.window_state.rename_species(old_name, new_name)
        
        if success:
            self.current_species = new_name
            self.species_names = [new_name if n == old_name else n for n in self.species_names]
            self.header_label.setText(f"{new_name} Properties")
            self.propertiesChanged.emit()
        else:
            self.name_edit.setText(old_name)

    def _add_unifac_group(self):
        """Append a blank row and put the cursor in its Group cell.

        Signals stay blocked while both cells are seeded: an unblocked setItem
        writes through to state, which drops the still-unnamed row and rebuilds
        the table, deleting the row the user just asked for. Nothing reaches
        window_state until they commit a group name.
        """
        row = self.unifac_table.rowCount()
        self.unifac_table.blockSignals(True)
        self.unifac_table.insertRow(row)
        self.unifac_table.setItem(row, 0, QTableWidgetItem(""))
        self.unifac_table.setItem(row, 1, QTableWidgetItem("1"))
        self.unifac_table.blockSignals(False)
        self.unifac_table.setCurrentCell(row, 0)
        self.unifac_table.editItem(self.unifac_table.item(row, 0))

    def _remove_unifac_group(self):
        row = self.unifac_table.currentRow()
        if row >= 0:
            self.unifac_table.removeRow(row)
            self._on_unifac_changed(row, 0)

    def _on_unifac_changed(self, row, column):
        if self.current_species and not self._loading:
            self._update_species_from_ui()
            self._refresh_estimate_btn()
            self.propertiesChanged.emit()

    def _refresh_estimate_btn(self):
        """Enable the r/q estimate only when at least one group is entered."""
        has_groups = any(
            self.unifac_table.item(r, 0) and self.unifac_table.item(r, 0).text().strip()
            for r in range(self.unifac_table.rowCount()))
        self.unifac_btn.setEnabled(bool(self.current_species) and has_groups)

    def _estimate_uniquac(self):
        """r = Σ ν·R_k, q = Σ ν·Q_k from the entered groups -> UNIQUAC r/q."""
        from PySide6.QtWidgets import QMessageBox
        if not self.window_state or not self.current_species:
            return
        self._update_species_from_ui()
        groups = self.window_state.species[self.current_species].unifac_groups
        from core.thermodynamics import load_unifac_db
        sub = load_unifac_db()["subgroups"]
        unknown = [g for g in groups if g not in sub]
        if unknown:
            QMessageBox.warning(self, "Unknown UNIFAC groups",
                                "Not in the group DB: " + ", ".join(unknown))
            return
        r = sum(sub[g][2] * n for g, n in groups.items())
        q = sum(sub[g][3] * n for g, n in groups.items())
        p = self.window_state.thermodynamics_config.get_component_params(
            self.current_species)
        p.uniquac_r, p.uniquac_q = r, q
        self.window_state.is_modified = True
        self.propertiesChanged.emit()
        QMessageBox.information(
            self, "UNIQUAC r/q estimated",
            f"{self.current_species}: r = {r:.4f}, q = {q:.4f}\n"
            "Set as this species' UNIQUAC structural parameters "
            "(Initialization → Thermodynamics).")

    def set_species_list(self, names: list):
        """Set the list of available species names."""
        self.species_names = names
        if self.current_species and self.current_species not in names:
            self.clear()

    def select_species(self, name: str):
        """Select a species to display its properties."""
        if name in self.species_names:
            self.current_species = name
            self.name_edit.setText(name)
            self.header_label.setText(f"{name} Properties")
            self._load_species_from_state()
        else:
            self.clear()

    def _load_species_from_state(self):
        """Load species properties from window_state."""
        if not self.window_state or not self.current_species:
            return
        
        species = self.window_state.species.get(self.current_species)
        if not species:
            self.clear()
            return

        self._loading = True
        try:
            self.mw_spin.setValue(species.mw if species.mw else 0)
            self.density_spin.setValue(species.liquid_density if species.liquid_density else 0)
            self.cp_spin.setValue(species.cp if species.cp else 0)

            self.unifac_table.setRowCount(0)
            for group, count in species.unifac_groups.items():
                row = self.unifac_table.rowCount()
                self.unifac_table.insertRow(row)
                self.unifac_table.setItem(row, 0, QTableWidgetItem(group))
                self.unifac_table.setItem(row, 1, QTableWidgetItem(str(count)))
        finally:
            self._loading = False
        self._refresh_estimate_btn()

    def clear(self):
        """Clear all property values."""
        self.current_species = None
        self.header_label.setText("Select a species to view properties")
        self.name_edit.clear()
        self.mw_spin.setValue(0)
        self.density_spin.setValue(0)
        self.cp_spin.setValue(0)
        self.unifac_table.setRowCount(0)
        self.unifac_btn.setEnabled(False)

    def _update_species_from_ui(self):
        """Update the current species data in window_state."""
        if not self.window_state or not self.current_species:
            return

        if self.current_species not in self.window_state.species:
            self.window_state.species[self.current_species] = Species(name=self.current_species)

        species = self.window_state.species[self.current_species]
        species.mw = self.mw_spin.value() if self.mw_spin.value() > 0 else None
        species.liquid_density = self.density_spin.value() if self.density_spin.value() > 0 else None
        species.cp = self.cp_spin.value() if self.cp_spin.value() > 0 else None

        groups = {}
        for row in range(self.unifac_table.rowCount()):
            group_name = self.unifac_table.item(row, 0)
            count_item = self.unifac_table.item(row, 1)
            if group_name and count_item:
                try:
                    count = int(count_item.text())
                    if group_name.text():
                        groups[group_name.text()] = count
                except ValueError:
                    pass
        species.unifac_groups = groups

        self.window_state.is_modified = True

    def get_properties(self) -> dict:
        """Get the current property values."""
        if not self.current_species:
            return {}

        groups = {}
        for row in range(self.unifac_table.rowCount()):
            group_name = self.unifac_table.item(row, 0)
            count_item = self.unifac_table.item(row, 1)
            if group_name and count_item:
                try:
                    count = int(count_item.text())
                    if group_name.text():
                        groups[group_name.text()] = count
                except ValueError:
                    pass

        return {
            "name": self.current_species,
            "mw": self.mw_spin.value(),
            "liquid_density": self.density_spin.value(),
            "cp": self.cp_spin.value(),
            "unifac_groups": groups
        }
