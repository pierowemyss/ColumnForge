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

        self._setup_ui()
        self._setup_styles()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        self.header_label = QLabel("Select a species to view properties")
        self.header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
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

        properties_layout.addWidget(QLabel("ρ (kg/m³):"), 2, 0)
        self.density_spin = QDoubleSpinBox(self)
        self.density_spin.setRange(0, 10000)
        self.density_spin.setDecimals(2)
        self.density_spin.setValue(0)
        self.density_spin.valueChanged.connect(self._on_property_changed)
        properties_layout.addWidget(self.density_spin, 2, 1)

        properties_layout.addWidget(QLabel("Cp (J/mol·K):"), 3, 0)
        self.cp_spin = QDoubleSpinBox(self)
        self.cp_spin.setRange(0, 10000)
        self.cp_spin.setDecimals(2)
        self.cp_spin.setValue(0)
        self.cp_spin.valueChanged.connect(self._on_property_changed)
        properties_layout.addWidget(self.cp_spin, 3, 1)

        main_layout.addWidget(properties_group)

        unifac_group = QGroupBox("UNIFAC Groups")
        unifac_layout = QVBoxLayout(unifac_group)

        self.unifac_table = QTableWidget(0, 2)
        self.unifac_table.setHorizontalHeaderLabels(["Group", "Count"])
        self.unifac_table.horizontalHeader().setStretchLastSection(True)
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

        self.unifac_btn = QPushButton("Estimate with UNIFAC (#)")
        self.unifac_btn.setEnabled(False)
        main_layout.addWidget(self.unifac_btn)

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

    def set_window_state(self, window_state):
        """Set the window state reference."""
        self.window_state = window_state

    def _on_property_changed(self):
        if self.current_species:
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
        row = self.unifac_table.rowCount()
        self.unifac_table.insertRow(row)
        self.unifac_table.setItem(row, 0, QTableWidgetItem(""))
        self.unifac_table.setItem(row, 1, QTableWidgetItem("1"))
        self.unifac_table.cellChanged.connect(self._on_unifac_changed)

    def _remove_unifac_group(self):
        row = self.unifac_table.currentRow()
        if row >= 0:
            self.unifac_table.removeRow(row)
            self._on_unifac_changed(row, 0)

    def _on_unifac_changed(self, row, column):
        if self.current_species:
            self._update_species_from_ui()
            self.propertiesChanged.emit()

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
            self.unifac_btn.setEnabled(True)
            self.blockSignals(True)
            self._load_species_from_state()
            self.blockSignals(False)
        else:
            self.clear()

    def _load_species_from_state(self):
        """Load species properties from window_state."""
        if not self.window_state or not self.current_species:
            return
        
        species = self.window_state.species.get(self.current_species)
        if species:
            self.mw_spin.setValue(species.mw if species.mw else 0)
            self.density_spin.setValue(species.liquid_density if species.liquid_density else 0)
            self.cp_spin.setValue(species.cp if species.cp else 0)
            
            self.unifac_table.blockSignals(True)
            self.unifac_table.setRowCount(0)
            for group, count in species.unifac_groups.items():
                row = self.unifac_table.rowCount()
                self.unifac_table.insertRow(row)
                self.unifac_table.setItem(row, 0, QTableWidgetItem(group))
                self.unifac_table.setItem(row, 1, QTableWidgetItem(str(count)))
            self.unifac_table.blockSignals(False)
        else:
            self.clear()

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
