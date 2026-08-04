#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initialization Tab - ColumnForge Column Solver GUI
Contains Thermodynamics and Chemical Species sub-tabs

Author: Piero Wemyss
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QPushButton, QGroupBox, QStackedWidget,
    QCheckBox
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QToolTip

from gui.table_edit import parse_number, fmt_number
from gui.panels.reactions_panel import ReactionsPanel
from gui.panels.sub_tab_bar import SubTabBar
from gui.panels.species_properties_panel import SpeciesPropertiesPanel
from gui.state.window_state import Species, ThermodynamicsConfig

IMPLEMENTED_VLE = ThermodynamicsConfig.IMPLEMENTED_VLE
IMPLEMENTED_ACTIVITY = ThermodynamicsConfig.IMPLEMENTED_ACTIVITY
IMPLEMENTED_EOS = ThermodynamicsConfig.IMPLEMENTED_EOS

# Full option lists (implemented + greyed-out) — shared with the Simulation tab's
# mirror combos so the two stay in lock-step.
VLE_MODELS = ["Antoine", "Wagner", "PLXANT", "Ideal"]

# Per-component coefficient fields for each vapour-pressure model, in the
# column order the parameter grid shows them. Shared by save/load so a new
# model needs one entry, not two branches.
_PURE_PSAT_KEYS = {
    "Antoine": ("antoine_a", "antoine_b", "antoine_c"),
    "Wagner": ("wagner_a", "wagner_b", "wagner_c", "wagner_d"),
    "PLXANT": tuple(f"plxant_c{i + 1}" for i in range(7)),
}
ACTIVITY_MODELS = ["Ideal", "NRTL", "UNIQUAC", "Wilson", "Margules", "UNIFAC"]
EOS_MODELS = ["Ideal Gas", "SRK", "PR", "BWRS"]


def _grey_unimplemented(combo, implemented):
    """Disable (grey out) combo entries whose model isn't wired to a solver,
    so entered parameters are never silently ignored."""
    model = combo.model()
    for i in range(combo.count()):
        item = model.item(i)
        ok = combo.itemText(i) in implemented
        item.setEnabled(ok)
        if not ok:
            item.setToolTip("Not yet implemented")


class InitializationTab(QWidget):
    """Initialization tab with Thermodynamics and Chemical Species sub-tabs."""

    speciesChanged = Signal()
    thermoChanged = Signal()   # vle/activity/eos model changed; Simulation tab mirrors it

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.window_state = None
        
        # Main layout: Horizontal (LHS Sub-Tab Bar | Main Content)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create and add sub-tabs
        self.sub_tab_bar = SubTabBar(self)
        self.sub_tab_bar.addTab("Chemical Species")
        self.sub_tab_bar.addTab("Thermodynamics")
        self.sub_tab_bar.addTab("Reactions")
        self.sub_tab_bar.tabClicked.connect(self._on_sub_tab_changed)
        main_layout.addWidget(self.sub_tab_bar)

        # Right: Stacked widget for sub-tab content
        self.stack = QStackedWidget(self)
        # Chemical Species page (Index 0 now)
        self.species_page = self._create_species_page()
        self.stack.addWidget(self.species_page)

        # Thermodynamics page (Index 1 now)
        self.thermo_page = self._create_thermodynamics_page()
        self.stack.addWidget(self.thermo_page)

        # Reactions (Index 2). A reaction describes the CHEMISTRY, so it belongs
        # beside the species and the thermo models rather than inside one sizing
        # module's panel -- see `panels/reactions_panel.py`, which also records
        # that BVM is the only thing consuming it today.
        self.reactions_page = ReactionsPanel(self)
        self.reactions_page.changed.connect(self._on_reactions_changed)
        self.stack.addWidget(self.reactions_page)

        main_layout.addWidget(self.stack)

        # Start with Chemical Species selected (index 0)
        self.sub_tab_bar.setCurrentIndex(0)
        self.stack.setCurrentIndex(0)
        
        self._setup_styles()

    def _select_tab(self, index: int):
        """Internal helper to switch the stacked widget."""
        self.stack.setCurrentIndex(index)

    def _create_thermodynamics_page(self):
        """Create the Thermodynamics sub-tab content."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(15)

        # 1. Simulation Model Selection Row
        model_group = QGroupBox("Simulation Models")
        model_layout = QHBoxLayout(model_group)

        self.vle_combo = QComboBox(self)
        self.vle_combo.addItems(VLE_MODELS)
        _grey_unimplemented(self.vle_combo, IMPLEMENTED_VLE)
        self.vle_combo.currentTextChanged.connect(self._on_thermo_changed)
        model_layout.addWidget(QLabel("Vapor Pressure:"))
        model_layout.addWidget(self.vle_combo)
        model_layout.addSpacing(20)

        self.activity_combo = QComboBox(self)
        self.activity_combo.addItems(ACTIVITY_MODELS)
        _grey_unimplemented(self.activity_combo, IMPLEMENTED_ACTIVITY)
        self.activity_combo.currentTextChanged.connect(self._on_thermo_changed)
        model_layout.addWidget(QLabel("Activity Coefficient:"))
        model_layout.addWidget(self.activity_combo)
        model_layout.addSpacing(20)

        self.eos_combo = QComboBox(self)
        self.eos_combo.addItems(EOS_MODELS)
        _grey_unimplemented(self.eos_combo, IMPLEMENTED_EOS)
        self.eos_combo.currentTextChanged.connect(self._on_thermo_changed)
        model_layout.addWidget(QLabel("Equation of State:"))
        model_layout.addWidget(self.eos_combo)

        layout.addWidget(model_group)

        # Flow model: CMO vs a real stage energy balance (Inside-Out only).
        flow_group = QGroupBox("Flow Model")
        flow_layout = QHBoxLayout(flow_group)
        self.energy_balance_check = QCheckBox(
            "Rigorous energy balance (Inside-Out) — real duties, non-constant "
            "molar overflow", self)
        self.energy_balance_check.setToolTip(
            "Off = constant molar overflow (CMO). On = per-stage enthalpy "
            "balance; needs Cp, Tb, latent heat and Tc for every component "
            "(the component DB supplies them). The Wang-Henke Bubble-Point "
            "path always uses CMO.")
        self.energy_balance_check.toggled.connect(self._on_thermo_changed)
        flow_layout.addWidget(self.energy_balance_check)
        layout.addWidget(flow_group)

        # 2. Parameter Entry Context Selection
        context_group = QGroupBox("Parameter Entry Context")
        context_layout = QHBoxLayout(context_group)

        context_layout.addWidget(QLabel("Type:"))
        self.param_type_combo = QComboBox(self)
        self.param_type_combo.addItems(["Vapor Pressure", "Activity", "EOS"])
        # Honesty policy: if no EOS beyond Ideal Gas is implemented, Tc/Pc/ω
        # would be entered and silently ignored — grey the context out.
        # (SRK is implemented, so this gate is currently open.)
        if len(IMPLEMENTED_EOS) <= 1:
            eos_item = self.param_type_combo.model().item(2)
            eos_item.setEnabled(False)
            eos_item.setToolTip("No EOS implemented yet — Tc/Pc/ω would "
                                "not be consumed.")
        self.param_type_combo.currentTextChanged.connect(self._update_param_models)
        context_layout.addWidget(self.param_type_combo)

        context_layout.addWidget(QLabel("Model:"))
        self.param_model_combo = QComboBox(self)
        self.param_model_combo.currentTextChanged.connect(self._update_parameter_visibility)
        self.param_model_combo.currentTextChanged.connect(self._sync_active_model)
        context_layout.addWidget(self.param_model_combo)

        context_layout.addStretch()
        layout.addWidget(context_group)

        # 3. Binary Interaction Table Selection (Row 3)
        self.table_selection_group = QGroupBox("Parameter Table Selection")
        table_selection_layout = QHBoxLayout(self.table_selection_group)
        self.table_selection_combo = QComboBox(self)
        self.table_selection_combo.currentTextChanged.connect(self._on_interaction_table_type_changed)
        table_selection_layout.addWidget(QLabel("Table:"))
        table_selection_layout.addWidget(self.table_selection_combo)
        table_selection_layout.addStretch()
        layout.addWidget(self.table_selection_group)

        # 4. Parameter Table Display (Row 4)
        self.interaction_group = QGroupBox("Model Parameters")
        interaction_layout = QVBoxLayout(self.interaction_group)

        self.interaction_table = QTableWidget(3, 3)
        self.interaction_table.setHorizontalHeaderLabels(["", "Ethanol", "Water"])
        self.interaction_table.setVerticalHeaderLabels(["Ethanol", "Water", ""])
        self.interaction_table.horizontalHeader().setStretchLastSection(True)
        self.interaction_table.verticalHeader().setStretchLastSection(True)
        self.interaction_table.cellChanged.connect(self._on_table_cell_changed)

        interaction_layout.addWidget(self.interaction_table)
        layout.addWidget(self.interaction_group)

        # Initial update
        self._update_param_models()
        
        layout.addStretch()
        return page

    def _sync_active_model(self, model: str):
        """Picking a model in the Parameter-Entry context also activates it.

        Without this the user edits (say) PLXANT coefficients while the active
        vle_model is still the default Antoine, and every solver/auto-saturate
        silently runs the wrong (often placeholder) Antoine params. The top
        Simulation-Models combos stay the single source of truth; setting one
        here persists via _on_thermo_changed.
        """
        if not model:
            return
        ptype = self.param_type_combo.currentText()
        if ptype == "Vapor Pressure" and model in IMPLEMENTED_VLE:
            self.vle_combo.setCurrentText(model)
        elif ptype == "Activity" and model in IMPLEMENTED_ACTIVITY:
            self.activity_combo.setCurrentText(model)

    def _update_param_models(self):
        """Update the Model dropdown based on the selected Type."""
        param_type = self.param_type_combo.currentText()
        self.param_model_combo.blockSignals(True)
        self.param_model_combo.clear()
        
        if param_type == "Vapor Pressure":
            self.param_model_combo.setEnabled(True)
            self.param_model_combo.addItems(["Antoine", "Wagner", "PLXANT"])
            _grey_unimplemented(self.param_model_combo, IMPLEMENTED_VLE)
        elif param_type == "Activity":
            self.param_model_combo.setEnabled(True)
            self.param_model_combo.addItems(["NRTL", "UNIQUAC", "Wilson", "Margules"])
            _grey_unimplemented(self.param_model_combo, IMPLEMENTED_ACTIVITY)
        elif param_type == "EOS":
            self.param_model_combo.setEnabled(False)
            self.param_model_combo.addItem("(Uses Tc, Pc, ω)")
        
        self.param_model_combo.blockSignals(False)
        self._update_parameter_visibility()

    def _update_parameter_visibility(self):
        """Update visibility of parameter tables based on context."""
        param_type = self.param_type_combo.currentText()
        param_model = self.param_model_combo.currentText()
        
        is_eos = param_type == "EOS"
        
        if is_eos:
            self.interaction_group.setTitle("EOS Parameters (configured per-species)")
            self.interaction_group.setVisible(True)
            self._update_interaction_table_headers(False, ["Tc (K)", "Pc (bar)", "ω"])
        else:
            self.interaction_group.setTitle("Model Parameters")
            show_table = bool(param_model)
            self.interaction_group.setVisible(show_table)
        
        # Update Table Selection Dropdown
        self.table_selection_combo.blockSignals(True)
        self.table_selection_combo.clear()
        
        has_multiple_binary_tables = False
        is_binary = False
        pure_params = []
        
        if param_type == "Vapor Pressure":
            is_binary = False
            if param_model == "Antoine":
                pure_params = ["A", "B", "C"]
            elif param_model == "Wagner":
                # Reduced form: Tc/Pc come from the same per-component record
                # the EOS uses, so only a..d are entered here.
                pure_params = ["a", "b", "c", "d"]
            elif param_model == "PLXANT":
                pure_params = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        
        elif param_type == "Activity":
            is_binary = True
            if param_model == "NRTL":
                self.table_selection_combo.addItems(["aij", "bij", "cij"])
                has_multiple_binary_tables = True
            elif param_model == "UNIQUAC":
                self.table_selection_combo.addItems(["aij", "bij", "r/q"])
                has_multiple_binary_tables = True
            elif param_model == "Wilson":
                self.table_selection_combo.addItems(["aij", "bij"])
                has_multiple_binary_tables = True
            elif param_model == "Margules":
                # two-suffix: one symmetric, T-independent A_ij table
                self.table_selection_combo.addItems(["aij"])
            self._update_interaction_table_headers(is_binary, pure_params)
        elif is_eos:
            self._update_interaction_table_headers(False, ["Tc (K)", "Pc (bar)", "ω"])
        
        # Dropdown only needed if > 1 BINARY table set
        self.table_selection_group.setVisible(has_multiple_binary_tables)
        self.table_selection_combo.blockSignals(False)
        
        if not is_eos:
            self._update_interaction_table_headers(is_binary, pure_params)

        self.load_interaction_parameters()   # repopulate cells from stored values

    def _update_interaction_table_headers(self, is_binary=True, pure_params=None):
        """Update the table headers based on current species names and model type."""
        species = self.get_species_names()
        # If no species, show an empty table or a placeholder message instead of defaults
        if not species:
            self.interaction_table.setRowCount(0)
            self.interaction_table.setColumnCount(0)
            return
            
        if is_binary:
            # Species vs Species Matrix
            n = len(species)
            self.interaction_table.setRowCount(n)
            self.interaction_table.setColumnCount(n)
            self.interaction_table.setHorizontalHeaderLabels(species)
            self.interaction_table.setVerticalHeaderLabels(species)
        else:
            # Pure component parameters: Columns are Params (A, B, C...), Rows are Species
            params = pure_params if pure_params else ["Value"]
            self.interaction_table.setRowCount(len(species))
            self.interaction_table.setColumnCount(len(params))
            self.interaction_table.setHorizontalHeaderLabels(params)
            self.interaction_table.setVerticalHeaderLabels(species)


    def _on_interaction_table_type_changed(self, text):
        """Handle change in which interaction table is displayed (aij/bij/cij).

        Repopulate from stored values for the newly selected table — do NOT save
        first: the visible cells still hold the previous table's numbers (cell
        edits already persist via _on_table_cell_changed), so saving here would
        write them under the new table_type and clobber it.
        """
        if self.param_type_combo.currentText() == "Activity":
            # UNIQUAC's r/q is a pure-component table; everything else binary
            if text == "r/q":
                self._update_interaction_table_headers(False, ["r", "q"])
            else:
                self._update_interaction_table_headers(True, [])
        self.load_interaction_parameters()

    def _on_table_cell_changed(self, row, col):
        """Persist a cell edit — or bounce it if it isn't a number.

        Nothing silently ignored: text that won't parse used to be dropped on
        the floor and the cell blanked itself on the next reload, which reads
        as "my parameter got deleted".
        """
        item = self.interaction_table.item(row, col)
        text = item.text().strip() if item else ""
        if text and parse_number(text) is None:
            QToolTip.showText(QCursor.pos(),
                              f"'{text}' is not a number — entry discarded.")
            self.load_interaction_parameters()   # repaint from stored values
            return
        self.save_interaction_parameters()

    def _create_species_page(self):
        """Create Chemical Species sub-tab content."""
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left: Species list
        list_container = QWidget()
        list_layout = QVBoxLayout(list_container)
        list_group = QGroupBox("Species List")
        group_layout = QVBoxLayout(list_group)

        self.species_list = QTableWidget(0, 1)
        self.species_list.setHorizontalHeaderLabels(["Species"])
        self.species_list.horizontalHeader().setStretchLastSection(True)
        self.species_list.itemSelectionChanged.connect(self._on_species_selected)
        self.species_list.itemChanged.connect(self._on_species_name_changed)
        group_layout.addWidget(self.species_list)

        list_buttons = QHBoxLayout()
        self.add_species_btn = QPushButton("Add…")
        self.add_species_btn.setToolTip("Search the bundled component database")
        self.add_species_btn.clicked.connect(self._add_species_from_db)
        self.add_blank_species_btn = QPushButton("Add Blank")
        self.add_blank_species_btn.setToolTip(
            "Add an empty species and enter properties manually")
        self.add_blank_species_btn.clicked.connect(self._add_species)
        self.remove_species_btn = QPushButton("Delete")
        self.remove_species_btn.clicked.connect(self._remove_species)
        list_buttons.addWidget(self.add_species_btn)
        list_buttons.addWidget(self.add_blank_species_btn)
        list_buttons.addWidget(self.remove_species_btn)
        group_layout.addLayout(list_buttons)
        
        list_layout.addWidget(list_group)
        layout.addWidget(list_container, 1)

        # Right: Species properties
        self.species_props = SpeciesPropertiesPanel(self)
        self.species_props.propertiesChanged.connect(self._on_species_changed)
        layout.addWidget(self.species_props, 2)

        return page

    def _setup_styles(self):
        # Group-box styling now comes from the central theme (gui/theme/app.qss).
        pass

    def _on_sub_tab_changed(self, index: int):
        """Handle sub-tab change."""
        self.stack.setCurrentIndex(index)
        self.sub_tab_bar.setCurrentIndex(index)

    def _on_thermo_changed(self):
        """Handle thermodynamic model change and save to window_state."""
        if self.window_state:
            self.window_state.thermodynamics_config.vle_model = self.vle_combo.currentText()
            self.window_state.thermodynamics_config.activity_model = self.activity_combo.currentText()
            self.window_state.thermodynamics_config.eos_model = self.eos_combo.currentText()
            self.window_state.thermodynamics_config.energy_balance = \
                self.energy_balance_check.isChecked()
            self.window_state.is_modified = True
            self.thermoChanged.emit()

    def refresh_thermo(self):
        """Re-sync the model combos from window_state (called when the Simulation
        tab's mirror combos change them). Signals blocked to avoid a feedback loop."""
        if not self.window_state:
            return
        thermo = self.window_state.thermodynamics_config
        for combo, value in ((self.vle_combo, thermo.vle_model),
                             (self.activity_combo, thermo.activity_model),
                             (self.eos_combo, thermo.eos_model)):
            combo.blockSignals(True)
            combo.setCurrentText(value)
            combo.blockSignals(False)

    def _on_species_selected(self):
        """Handle species selection from list."""
        row = self.species_list.currentRow()
        if row >= 0:
            item = self.species_list.item(row, 0)
            if item:
                name = item.text()
                self.species_props.select_species(name)

    def _on_species_changed(self):
        """Handle species property change.

        The panel is the only emitter of propertiesChanged, so it is already in
        sync — re-selecting it here would reload the table out from under an
        in-progress edit (and delete a freshly added, not-yet-named UNIFAC row).
        """
        self._refresh_species_list()
        self.species_props.set_species_list(self.get_species_names())
        self.speciesChanged.emit()

    def _on_species_name_changed(self, item):
        """Handle species name change in the list."""
        if not item or not self.window_state:
            return
        
        row = item.row()
        new_name = item.text().strip()
        
        if not new_name:
            self._refresh_species_list()
            return
        
        all_names = list(self.window_state.species.keys())
        
        if row >= len(all_names):
            return
        
        old_name = all_names[row]
        
        if not old_name or old_name == new_name:
            return
        
        if new_name in all_names:
            item.setText(old_name)
            return
        
        success = self.window_state.rename_species(old_name, new_name)
        
        if success:
            self._refresh_species_list()
            self.species_props.set_species_list(self.get_species_names())
            self.species_props.select_species(new_name)
        else:
            item.setText(old_name)

    def _refresh_species_list(self):
        """Refresh the species list from window_state, keeping the highlighted row
        on whichever species the properties panel is showing."""
        current = self.species_props.current_species if self.species_props else None
        self.species_list.blockSignals(True)
        self.species_list.setRowCount(0)
        for name in self.get_species_names():
            row = self.species_list.rowCount()
            self.species_list.insertRow(row)
            self.species_list.setItem(row, 0, QTableWidgetItem(name))
            if name == current:
                self.species_list.setCurrentCell(row, 0)
        self.species_list.blockSignals(False)

    def _add_species_from_db(self):
        """Add a species from the bundled component database (search dialog)."""
        if not self.window_state:
            return
        from ..panels.species_search_dialog import SpeciesSearchDialog
        from core import component_db

        dlg = SpeciesSearchDialog(self, existing_names=self.get_species_names())
        if not dlg.exec() or not dlg.selected_name:
            return
        info = component_db.load_into(self.window_state, dlg.selected_name)
        self._refresh_species_list()
        self.species_props.set_species_list(self.get_species_names())
        self.species_props.select_species(info["record"]["name"])
        self._update_parameter_visibility()
        self.load_interaction_parameters()   # show any auto-filled NRTL pairs
        self.speciesChanged.emit()
        if info["missing_pairs"]:
            from PySide6.QtWidgets import QMessageBox
            pairs = ", ".join(f"{i}/{j}" for i, j in info["missing_pairs"])
            QMessageBox.information(
                self, "NRTL parameters missing",
                f"No curated NRTL binary parameters for: {pairs}.\n"
                "These pairs will be treated as ideal unless you enter "
                "parameters in Thermodynamics → Binary Interactions.")

    def _add_species(self):
        """Add a new species."""
        row = self.species_list.rowCount()
        name = f"Species {row + 1}"
        self.species_list.insertRow(row)
        self.species_list.setItem(row, 0, QTableWidgetItem(name))
        
        # Add species to window_state
        if self.window_state:
            self.window_state.add_species(Species(name=name))
        
        self.species_props.set_species_list(self.get_species_names())
        self._update_parameter_visibility() # Refresh thermo tables
        self.speciesChanged.emit()

    def _remove_species(self):
        """Remove selected species."""
        row = self.species_list.currentRow()
        if row >= 0:
            # Get name before removing
            item = self.species_list.item(row, 0)
            name = item.text() if item else None
            
            self.species_list.removeRow(row)
            
            # Remove from window_state
            if self.window_state and name:
                self.window_state.remove_species(name)
            
            names = self.get_species_names()
            self.species_props.set_species_list(names)
            self.species_props.clear()
            self._update_parameter_visibility() # Refresh thermo tables
            self.speciesChanged.emit()

    def get_species_names(self) -> list:
        """Get list of species names."""
        if self.window_state:
            return list(self.window_state.species.keys())
        return []

    def get_thermodynamics_config(self) -> dict:
        """Get current thermodynamics configuration."""
        return {
            "vle_model": self.vle_combo.currentText(),
            "activity_model": self.activity_combo.currentText(),
            "eos_model": self.eos_combo.currentText(),
            "param_model": self.param_model_combo.currentText(),
        }

    def get_interaction_parameters(self) -> list:
        """Get the parameter grid as a list of rows of float-or-None.

        None means "the cell is empty" and the caller clears the stored value —
        text that isn't a number never gets this far (_on_table_cell_changed
        bounces it), so a blank cell can only ever mean the user emptied it.
        """
        params = []
        for row in range(self.interaction_table.rowCount()):
            row_data = []
            for col in range(self.interaction_table.columnCount()):
                item = self.interaction_table.item(row, col)
                row_data.append(parse_number(item.text()) if item else None)
            params.append(row_data)
        return params

    def save_interaction_parameters(self):
        """Save parameters from table to window_state."""
        if not self.window_state:
            return
        
        param_type = self.param_type_combo.currentText()
        param_model = self.param_model_combo.currentText()
        species = list(self.window_state.species.keys())
        
        if param_type == "Vapor Pressure":
            params = self.get_interaction_parameters()
            # Same key map load_interaction_parameters reads back, so every
            # implemented Psat model round-trips without a per-model branch.
            keys = _PURE_PSAT_KEYS.get(param_model)
            # Shape guard: a grid that isn't this model's yet (mid-rebuild)
            # would null out the columns it doesn't have.
            if keys and params and all(len(r) == len(keys) for r in params):
                for i, name in enumerate(species):
                    if i >= len(params):
                        continue
                    cp = self.window_state.thermodynamics_config.get_component_params(name)
                    for j, k in enumerate(keys):
                        setattr(cp, k, params[i][j] if j < len(params[i]) else None)

        elif param_type == "EOS":
            params = self.get_interaction_parameters()
            if params and all(len(r) == 3 for r in params):   # Tc, Pc, ω
                for i, name in enumerate(species):
                    if i < len(params):
                        comp_params = self.window_state.thermodynamics_config.get_component_params(name)
                        comp_params.tc, comp_params.pc, comp_params.omega = params[i]
        
        elif param_type == "Activity":
            table_type = self.table_selection_combo.currentText()
            params = self.get_interaction_parameters()
            if not params:
                return
            
            binary = self.window_state.thermodynamics_config.binary

            if param_model == "UNIQUAC" and table_type == "r/q":
                for i, name in enumerate(species):
                    if i < len(params):
                        cp = self.window_state.thermodynamics_config.get_component_params(name)
                        cp.uniquac_r = params[i][0] if len(params[i]) > 0 else None
                        cp.uniquac_q = params[i][1] if len(params[i]) > 1 else None
                self.window_state.is_modified = True
                return

            # One attribute name per (model, table) — the same map
            # get_binary_param_dict reads back, so an unknown table_type writes
            # nowhere instead of into a throwaway dict.
            attr = {"NRTL": f"nrtl_{table_type}",
                    "UNIQUAC": f"uniquac_{table_type}",
                    "Wilson": f"wilson_{table_type}",
                    "Margules": "margules_aij"}.get(param_model)
            if not attr or not hasattr(binary, attr):
                return
            param_dict = getattr(binary, attr)
            # Only trust blanks-mean-delete when the grid really is the current
            # species matrix; a half-built table must never wipe stored pairs.
            grid_is_current = (len(params) == len(species)
                               and all(len(r) == len(species) for r in params))
            for i, name_i in enumerate(species):
                for j, name_j in enumerate(species):
                    if i < len(params) and j < len(params[i]):
                        value = params[i][j]
                        if value is not None:
                            param_dict[(name_i, name_j)] = value
                        elif grid_is_current:
                            param_dict.pop((name_i, name_j), None)

        self.window_state.is_modified = True

    def load_interaction_parameters(self):
        """Fill the parameter table from stored window_state values (inverse of
        save_interaction_parameters). Called after the grid headers rebuild so a
        loaded .colx actually shows its coefficients."""
        if not self.window_state:
            return
        tbl = self.interaction_table
        if tbl.rowCount() == 0 or tbl.columnCount() == 0:
            return

        ptype = self.param_type_combo.currentText()
        pmodel = self.param_model_combo.currentText()
        species = list(self.window_state.species.keys())
        thermo = self.window_state.thermodynamics_config

        def put(r, c, val):
            # fmt_number, not "%g": %g showed 1234.56789 as 1234.57 and the next
            # edit in the table wrote that rounded number back over the real one.
            tbl.setItem(r, c, QTableWidgetItem(fmt_number(val)))

        pure_keys = _PURE_PSAT_KEYS

        tbl.blockSignals(True)
        try:
            if ptype == "EOS":
                for r, name in enumerate(species):
                    p = thermo.component_params.get(name)
                    for c, k in enumerate(("tc", "pc", "omega")):
                        put(r, c, getattr(p, k, None) if p else None)
            elif ptype == "Vapor Pressure":
                keys = pure_keys.get(pmodel, ())
                for r, name in enumerate(species):
                    p = thermo.component_params.get(name)
                    for c, k in enumerate(keys):
                        put(r, c, getattr(p, k, None) if p else None)
            elif ptype == "Activity":
                ttype = self.table_selection_combo.currentText()
                if pmodel == "UNIQUAC" and ttype == "r/q":
                    for r, name in enumerate(species):
                        p = thermo.component_params.get(name)
                        put(r, 0, getattr(p, "uniquac_r", None) if p else None)
                        put(r, 1, getattr(p, "uniquac_q", None) if p else None)
                else:
                    d = thermo.get_binary_param_dict(pmodel, ttype)
                    for i, ni in enumerate(species):
                        for j, nj in enumerate(species):
                            put(i, j, d.get((ni, nj)))
        finally:
            tbl.blockSignals(False)

    def set_window_state(self, window_state):
        """Set the window state object and initialize UI from it."""
        self.window_state = window_state
        self.species_props.set_window_state(window_state)
        self.reactions_page.set_window_state(window_state)
        self._load_from_state()

    def _on_reactions_changed(self):
        """The panel has already mirrored itself onto `window_state.reactions`,
        and the modules read that at run time, so there is nothing to push --
        only the dirty flag. Deliberately NOT `speciesChanged`: that signal makes
        three other tabs rebuild their species lists, and a Keq coefficient is
        not a species edit."""
        if self.window_state:
            self.window_state.is_modified = True

    def _load_from_state(self):
        """Initialize UI components from the window state."""
        if not self.window_state:
            return
            
        # Load Species
        self.species_list.setRowCount(0)
        for species_name in self.window_state.species.keys():
            row = self.species_list.rowCount()
            self.species_list.insertRow(row)
            self.species_list.setItem(row, 0, QTableWidgetItem(species_name))
            
        self.species_props.set_species_list(list(self.window_state.species.keys()))
        
        # Load Thermodynamics Config. Signals must stay blocked while setting the
        # three combos: _on_thermo_changed reads all three at once, so setting them
        # one at a time unblocked clobbers the not-yet-set ones back to defaults.
        thermo = self.window_state.thermodynamics_config
        self.refresh_thermo()
        self.energy_balance_check.setChecked(bool(thermo.energy_balance))

        # Point the parameter inspector at the loaded vapour-pressure model so its
        # coefficients are visible on load instead of a blank Antoine grid.
        self.param_type_combo.setCurrentText("Vapor Pressure")
        self._update_param_models()
        if thermo.vle_model in ("Antoine", "Wagner", "PLXANT"):
            self.param_model_combo.setCurrentText(thermo.vle_model)

        # Update Thermo Table headers
        self._update_parameter_visibility()

    def clear(self):
        """Clear all settings."""
        if self.window_state:
            self._load_from_state()
        else:
            self.vle_combo.setCurrentIndex(0)
            self.activity_combo.setCurrentIndex(0)
            self.eos_combo.setCurrentIndex(0)
            self.param_type_combo.setCurrentIndex(0)
            self.species_list.setRowCount(0)
            self.species_props.clear()
