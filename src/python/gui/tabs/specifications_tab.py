from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QTableWidget,
    QTableWidgetItem, QPushButton, QGroupBox, QStackedWidget, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal

from gui.theme import set_state
from gui.panels.sub_tab_bar import SubTabBar
from gui.panels.stream_config_panel import StreamConfigPanel
from gui.panels.condenser_config_panel import CondenserConfigPanel
from gui.panels.reboiler_config_panel import ReboilerConfigPanel
from gui.panels.module_config_panel import ModuleConfigPanel
from gui.panels.column_overview_panel import ColumnOverviewCanvas
from gui.panels.connection_config_panel import ConnectionConfigPanel
from gui.panels.flowsheet_canvas import FlowsheetScene, FlowsheetView
from gui.panels.unit_combo_box import UnitComboBox
from gui.panels.operating_specs_panel import OperatingSpecsPanel
from gui.state.window_state import (
    Stream, StreamType, CondenserType, ReboilerType, ModuleConfig, ModuleType
)


def _module_to_config(m: ModuleConfig) -> dict:
    """ModuleConfig -> the dict ModuleConfigPanel.set_config takes."""
    return {"type": m.module_type.value, "stage": m.stage,
            "num_stages": m.num_stages, "boilup_ratio": m.boilup_ratio,
            "reflux_ratio": m.reflux_ratio, "duty": m.duty,
            "return_stage": m.return_stage, "rate": m.rate}


class SpecificationsTab(QWidget):
    """Specifications tab with static side labels and direct content display."""

    specsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.window_state = None
        self.current_stream_id = None
        self.current_module_id = None

        self._setup_ui()
        self._setup_styles()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left: Sub-tab bar
        self.sub_tab_bar = SubTabBar(self)
        self.sub_tab_bar.addTab("Column Overview")
        self.sub_tab_bar.addTab("Column Config")
        self.sub_tab_bar.addTab("Streams")
        self.sub_tab_bar.addTab("Advanced Modules")
        self.sub_tab_bar.tabClicked.connect(self._on_sub_tab_changed)
        main_layout.addWidget(self.sub_tab_bar)

        # Right: Stacked widget for main tab content
        self.stack = QStackedWidget(self)

        # Column Overview page (Now Index 0)
        self.overview_page = self._create_overview_page()
        self.stack.addWidget(self.overview_page)

        # Column Config page (Now Index 1)
        self.col_config_page = self._create_column_config_page()
        self.stack.addWidget(self.col_config_page)

        # Streams page (Now Index 2)
        self.streams_page = self._create_streams_page()
        self.stack.addWidget(self.streams_page)

        # Advanced Modules page (Now Index 3)
        self.modules_page = self._create_modules_page()
        self.stack.addWidget(self.modules_page)

        main_layout.addWidget(self.stack)

    def _create_column_config_page(self):
        """Create the Column Config page with nested sub-tabs."""
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Column 1: Nested sub-tabs (Operating, Condenser, Reboiler)
        self.config_sub_tabs = SubTabBar(page)
        self.config_sub_tabs.addTab("Operating")
        self.config_sub_tabs.addTab("Condenser")
        self.config_sub_tabs.addTab("Reboiler")
        self.config_sub_tabs.tabClicked.connect(self._on_config_sub_tab_changed)
        layout.addWidget(self.config_sub_tabs)

        # Column 2: Configuration panel (changes based on sub-tab selection)
        self.config_stack = QStackedWidget(page)

        # Operating page
        operating_page = QWidget()
        operating_layout = QVBoxLayout(operating_page)
        operating_layout.setSpacing(10)

        # Pressure
        self.pressure_input = UnitComboBox("pressure")
        pressure_group = QGroupBox("Operating Pressure")
        pressure_layout = QHBoxLayout(pressure_group)
        pressure_layout.addWidget(QLabel("Pressure:"))
        pressure_layout.addWidget(self.pressure_input)
        operating_layout.addWidget(pressure_group)

        # Pressure Drop
        self.pressure_drop_spin = QDoubleSpinBox(self)
        self.pressure_drop_spin.setRange(0, 10)
        self.pressure_drop_spin.setDecimals(4)
        pressure_drop_group = QGroupBox("Pressure Drop")
        pd_layout = QHBoxLayout(pressure_drop_group)
        pd_layout.addWidget(QLabel("bar/stage:"))
        pd_layout.addWidget(self.pressure_drop_spin)
        operating_layout.addWidget(pressure_drop_group)

        # Stage efficiency — Murphree vapour efficiency applied to every tray
        # (condenser & reboiler stay equilibrium). Column-wide for now; the
        # solvers already accept a per-stage vector, so a per-stage editor can
        # bind here later without a backend change.
        self.efficiency_spin = QDoubleSpinBox(self)
        self.efficiency_spin.setRange(0.1, 1.0)
        self.efficiency_spin.setDecimals(3)
        self.efficiency_spin.setSingleStep(0.05)
        self.efficiency_spin.setValue(1.0)
        eff_group = QGroupBox("Stage Efficiency")
        eff_layout = QHBoxLayout(eff_group)
        eff_layout.addWidget(QLabel("Murphree (0.1–1.0):"))
        eff_layout.addWidget(self.efficiency_spin)
        operating_layout.addWidget(eff_group)

        # Operating specifications (Aspen-style: pick any N variables). The
        # light/heavy keys live inline on the recovery-spec rows now (they still
        # write window_state.light_key_index / heavy_key_index, shared with the
        # BVM/FUG modules); there's no separate Key Components box.
        op_specs_group = QGroupBox("Operating Specifications")
        op_specs_layout = QVBoxLayout(op_specs_group)
        self.operating_specs_panel = OperatingSpecsPanel(self)
        self.operating_specs_panel.specsChanged.connect(self._on_operating_specs_changed)
        op_specs_layout.addWidget(self.operating_specs_panel)
        operating_layout.addWidget(op_specs_group)

        operating_layout.addStretch()
        self.config_stack.addWidget(operating_page)

        # Condenser page
        self.condenser_panel = CondenserConfigPanel(self)
        self.condenser_panel.configChanged.connect(
            lambda: self._on_condenser_changed(self.condenser_panel))
        self.config_stack.addWidget(self.condenser_panel)

        # Reboiler page
        self.reboiler_panel = ReboilerConfigPanel(self)
        self.reboiler_panel.configChanged.connect(
            lambda: self._on_reboiler_changed(self.reboiler_panel))
        self.config_stack.addWidget(self.reboiler_panel)

        layout.addWidget(self.config_stack)

        return page

    def _on_config_sub_tab_changed(self, index: int):
        """Handle nested config sub-tab change."""
        self.config_stack.setCurrentIndex(index)
        # Pull latest from state so edits made elsewhere (overview editor, the
        # other sub-tab) show here.
        if index == 0:
            self._reload_operating_panel()
        elif index == 1:
            self._load_condenser_into(self.condenser_panel)
        elif index == 2:
            self._load_reboiler_into(self.reboiler_panel)

    def _create_streams_page(self):
        """Create the Streams page."""
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setSpacing(10)

        # Left: Stream list
        list_group = QGroupBox("Stream List")
        list_layout = QVBoxLayout(list_group)

        self.stream_list = QTableWidget(0, 1)
        self.stream_list.setHorizontalHeaderLabels(["Stream"])
        self.stream_list.horizontalHeader().setStretchLastSection(True)
        self.stream_list.itemSelectionChanged.connect(self._on_stream_selected)
        # double-click rename: commit to window_state (or revert) — without
        # this the state keeps the old key and edits land on a ghost stream
        self.stream_list.itemChanged.connect(self._on_stream_renamed)
        list_layout.addWidget(self.stream_list)

        stream_buttons = QHBoxLayout()
        self.add_stream_btn = QPushButton("Add")
        self.add_stream_btn.clicked.connect(self._add_stream)
        self.remove_stream_btn = QPushButton("Delete")
        self.remove_stream_btn.clicked.connect(self._remove_stream)
        stream_buttons.addWidget(self.add_stream_btn)
        stream_buttons.addWidget(self.remove_stream_btn)
        list_layout.addLayout(stream_buttons)

        layout.addWidget(list_group, 1)

        # Right: Stream configuration
        self.stream_config = StreamConfigPanel(self)
        self.stream_config.streamChanged.connect(self._on_stream_changed)
        layout.addWidget(self.stream_config, 2)

        return page

    def _create_overview_page(self):
        """Create the Column Overview page with an inline element editor.

        The right-hand pane hosts the actual component config panels (same
        classes used by the other sub-tabs). Double-clicking an element in the
        diagram opens that element's panel here, reading/writing the shared
        window_state data structures.
        """
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setSpacing(10)

        # Left: the column diagram + DoF status.
        #
        # Two diagrams, one shown at a time. The single-column canvas is the
        # default experience and is what every non-beta user sees; the flowsheet
        # editor is behind Preferences -> Enable beta features. Both are built
        # (they are cheap) and `_update_column_canvas` feeds whichever is on
        # screen, so the switch needs no restart.
        self.single_canvas = ColumnOverviewCanvas(self)
        self.single_canvas.elementClicked.connect(self._on_single_element_clicked)
        self.single_canvas.streamClicked.connect(self._on_single_stream_clicked)

        self.flowsheet_scene = FlowsheetScene(self)
        self.flowsheet_view = FlowsheetView(self.flowsheet_scene, self)
        self.flowsheet_scene.elementClicked.connect(self._on_element_clicked)
        self.flowsheet_scene.streamClicked.connect(self._open_stream_by_id)
        self.flowsheet_scene.edgeClicked.connect(self._on_edge_clicked)
        self.flowsheet_scene.nodeClicked.connect(self._on_node_clicked)
        self.flowsheet_scene.nodeDoubleClicked.connect(self._on_node_clicked)
        self.flowsheet_scene.connectionRequested.connect(self._on_connection_requested)
        self.flowsheet_scene.deleteRequested.connect(self._on_delete_requested)
        self.flowsheet_scene.nodeMoved.connect(self._on_node_moved)
        self.flowsheet_scene.validationRejected.connect(self._on_validation_rejected)

        self.diagram_stack = QStackedWidget(self)
        self.diagram_stack.addWidget(self.single_canvas)      # 0: default
        self.diagram_stack.addWidget(self.flowsheet_view)     # 1: beta
        # every existing caller says `column_canvas`; keep the name pointing at
        # the diagram that is actually on screen
        self.column_canvas = self.single_canvas

        self.overview_group = QGroupBox("Column Diagram")
        overview_layout = QVBoxLayout(self.overview_group)

        # Column selector — beta only; a single-column case has nothing to pick.
        self.column_row = QWidget(self)
        col_row = QHBoxLayout(self.column_row)
        col_row.setContentsMargins(0, 0, 0, 0)
        col_row.addWidget(QLabel("Column:"))
        self.column_combo = QComboBox(self)
        self.column_combo.currentTextChanged.connect(self._on_column_selected)
        col_row.addWidget(self.column_combo, 1)
        for text, tip, slot in (
                ("+", "Add a column to the flowsheet", self._add_column),
                ("Copy", "Duplicate this column's configuration", self._duplicate_column),
                ("Rename", "Rename this column", self._rename_column),
                ("−", "Delete this column and its connections", self._remove_column)):
            b = QPushButton(text, self)
            b.setToolTip(tip)
            b.setMaximumWidth(64)
            b.clicked.connect(slot)
            col_row.addWidget(b)
        overview_layout.addWidget(self.column_row)
        overview_layout.addWidget(self.diagram_stack)
        overview_group = self.overview_group

        self.dof_status_label = QLabel("Under-specified: Need 2 more specs.")
        set_state(self.dof_status_label, "warn")
        overview_layout.addWidget(self.dof_status_label)

        layout.addWidget(overview_group, 1)

        # Right: inline element configuration
        self.ov_config_group = QGroupBox("Element Configuration")
        ov_config_layout = QVBoxLayout(self.ov_config_group)
        self.ov_editor_stack = QStackedWidget(self)

        # Page 0: hint placeholder
        self.ov_placeholder = QLabel(
            "Click an element in the diagram\n"
            "(condenser, reboiler, or any stream label)\n"
            "to configure it here."
        )
        self.ov_placeholder.setProperty("hint", True)
        self.ov_editor_stack.addWidget(self.ov_placeholder)

        # Page 1/2/3: reuse the real component panels
        self.ov_condenser_panel = CondenserConfigPanel(self)
        self.ov_condenser_panel.configChanged.connect(
            lambda: self._on_condenser_changed(self.ov_condenser_panel))
        self.ov_editor_stack.addWidget(self.ov_condenser_panel)

        self.ov_reboiler_panel = ReboilerConfigPanel(self)
        self.ov_reboiler_panel.configChanged.connect(
            lambda: self._on_reboiler_changed(self.ov_reboiler_panel))
        self.ov_editor_stack.addWidget(self.ov_reboiler_panel)

        self.ov_stream_panel = StreamConfigPanel(self)
        self.ov_stream_panel.streamChanged.connect(self._on_ov_stream_changed)
        self.ov_editor_stack.addWidget(self.ov_stream_panel)

        # Page 4: the same module editor the Advanced Modules subtab uses, so a
        # module clicked on the diagram is configured without leaving the canvas.
        self.ov_module_panel = ModuleConfigPanel(self)
        self.ov_module_panel.configChanged.connect(self._on_ov_module_changed)
        self.ov_editor_stack.addWidget(self.ov_module_panel)

        # Page 5: an inter-column connection — split fraction, destination
        # stage, optional quality override.
        self.ov_connection_panel = ConnectionConfigPanel(self)
        self.ov_connection_panel.configChanged.connect(self._on_ov_connection_changed)
        self.ov_editor_stack.addWidget(self.ov_connection_panel)

        ov_config_layout.addWidget(self.ov_editor_stack)
        layout.addWidget(self.ov_config_group, 1)

        return page

    def _create_modules_page(self):
        """Create the Advanced Modules page."""
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setSpacing(10)

        # Left: Module list
        list_group = QGroupBox("Module List")
        list_layout = QVBoxLayout(list_group)

        # Every type here is solved. Interreboiler/pumparound are heat terms and
        # need the energy balance; side strippers/rectifiers work under CMO too.
        # Each module also adds its own DoF specs — see the status above.
        note = QLabel("Interreboiler/intercooler and pumparound need the energy "
                      "balance (Initialization → Flow Model). Side strippers and "
                      "rectifiers are solved as torn side columns and export a "
                      "side product.")
        note.setWordWrap(True)
        note.setProperty("hint", True)
        list_layout.addWidget(note)

        self.module_list = QTableWidget(0, 1)
        self.module_list.setHorizontalHeaderLabels(["Module"])
        self.module_list.horizontalHeader().setStretchLastSection(True)
        self.module_list.itemSelectionChanged.connect(self._on_module_selected)
        list_layout.addWidget(self.module_list)

        module_buttons = QHBoxLayout()

        self.add_module_btn = QPushButton("Add")
        self.add_module_btn.clicked.connect(self._add_module)
        self.add_module_btn.setToolTip("Add a module of the type chosen alongside.")

        self.module_type_combo = QComboBox(self)
        self.module_type_combo.addItems([t.value for t in ModuleType])

        self.remove_module_btn = QPushButton("Delete")
        self.remove_module_btn.clicked.connect(self._remove_module)

        module_buttons.addWidget(self.add_module_btn)
        module_buttons.addWidget(self.module_type_combo)
        module_buttons.addWidget(self.remove_module_btn)
        list_layout.addLayout(module_buttons)

        layout.addWidget(list_group, 1)

        # Right: Module configuration
        self.module_config = ModuleConfigPanel(self)
        self.module_config.configChanged.connect(self._on_module_changed)
        layout.addWidget(self.module_config, 2)

        return page

    def _setup_styles(self):
        # Group-box styling now comes from the central theme (gui/theme/app.qss).
        pass

    def _connect_signals(self):
        self.pressure_input.valueChanged.connect(self._on_config_changed)
        self.pressure_drop_spin.valueChanged.connect(self._on_config_changed)
        self.efficiency_spin.valueChanged.connect(self._on_config_changed)
        self.flowsheet_scene.specsChanged.connect(self._on_config_changed)
        self.single_canvas.specsChanged.connect(self._on_config_changed)

    def _on_sub_tab_changed(self, index: int):
        """Handle main sub-tab change."""
        self.stack.setCurrentIndex(index)
        self.sub_tab_bar.setCurrentIndex(index)

    # --- beta gate --------------------------------------------------------

    @property
    def beta(self) -> bool:
        from gui.app_settings import beta_enabled
        return beta_enabled()

    def refresh_beta(self):
        """Swap between the single-column diagram and the flowsheet editor."""
        on = self.beta
        self.diagram_stack.setCurrentIndex(1 if on else 0)
        self.column_canvas = self.flowsheet_view if on else self.single_canvas
        self.column_row.setVisible(on)
        self.overview_group.setTitle("Flowsheet" if on else "Column Diagram")
        if self.window_state:
            self._update_column_canvas()
            self._update_dof_status()

    # The single-column canvas emits without a column id (it only ever draws
    # one). Adapt to the same handlers the scene uses, naming the active column.
    def _on_single_element_clicked(self, element_type: str):
        self._on_element_clicked(self.window_state.active_column_id
                                 if self.window_state else "C1", element_type)

    def _on_single_stream_clicked(self, sid: str):
        self._open_stream_by_id(self.window_state.active_column_id
                                if self.window_state else "C1", sid)

    # --- flowsheet: columns, connections, and the scene's requests ---------

    def _refresh_column_combo(self):
        """Keep the selector in step with the flowsheet without re-entering."""
        if not self.window_state:
            return
        ids = list(self.window_state.columns)
        self.column_combo.blockSignals(True)
        self.column_combo.clear()
        self.column_combo.addItems(ids)
        self.column_combo.setCurrentText(self.window_state.active_column_id)
        self.column_combo.blockSignals(False)
        # a one-column flowsheet is the old single-column app; hide the buttons
        # that only mean something once there are several
        self.column_combo.setEnabled(len(ids) > 1)

    def _set_active_column(self, unit_id):
        """Point every panel on this tab at another column. The flat
        window_state names delegate, so the panels themselves need no change."""
        if not self.window_state or unit_id == self.window_state.active_column_id:
            return
        if not self.window_state.set_active_column(unit_id):
            return
        self.ov_editor_stack.setCurrentWidget(self.ov_placeholder)
        self.ov_config_group.setTitle("Element Configuration")
        self._load_from_state()
        self.specsChanged.emit()

    def _on_column_selected(self, unit_id: str):
        if unit_id:
            self._set_active_column(unit_id)

    def _on_node_clicked(self, unit_id: str):
        self._set_active_column(unit_id)

    def _add_column(self):
        if not self.window_state:
            return
        self.window_state.add_column()
        self._load_from_state()
        self.specsChanged.emit()

    def _duplicate_column(self):
        if not self.window_state:
            return
        self.window_state.duplicate_column(self.window_state.active_column_id)
        self._load_from_state()
        self.specsChanged.emit()

    def _rename_column(self):
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        if not self.window_state:
            return
        old = self.window_state.active_column_id
        new, ok = QInputDialog.getText(self, "Rename Column", "New name:", text=old)
        if not ok or not new.strip() or new.strip() == old:
            return
        if not self.window_state.rename_column(old, new.strip()):
            QMessageBox.warning(self, "Rename Column",
                                f"'{new.strip()}' is already taken.")
            return
        self._load_from_state()
        self.specsChanged.emit()

    def _remove_column(self):
        from PySide6.QtWidgets import QMessageBox
        if not self.window_state:
            return
        cid = self.window_state.active_column_id
        if len(self.window_state.columns) <= 1:
            QMessageBox.information(
                self, "Delete Column",
                "A flowsheet needs at least one column.")
            return
        n_conn = sum(1 for c in self.window_state.connections
                     if c.src == cid or c.dst == cid)
        extra = f" and {n_conn} connection(s)" if n_conn else ""
        if QMessageBox.question(
                self, "Delete Column",
                f"Delete '{cid}'{extra}? This cannot be undone."
        ) != QMessageBox.Yes:
            return
        self.window_state.remove_column(cid)
        self._load_from_state()
        self.specsChanged.emit()

    def _on_connection_requested(self, conn):
        """The scene asked for a connection; the state decides."""
        if not self.window_state:
            return
        self.window_state.connections.append(conn)
        self.window_state.mark_modified()
        self._update_column_canvas()
        self._update_dof_status()
        self.specsChanged.emit()

    def _on_delete_requested(self, payload):
        from PySide6.QtWidgets import QMessageBox
        if not self.window_state:
            return
        ws = self.window_state
        ids = set(payload.get("edges", []))
        if ids:
            ws.connections = [c for c in ws.connections if c.id not in ids]
            ws.mark_modified()
        for cid in payload.get("nodes", []):
            if len(ws.columns) <= 1:
                break
            if QMessageBox.question(
                    self, "Delete Column",
                    f"Delete column '{cid}' and its connections?"
            ) == QMessageBox.Yes:
                ws.remove_column(cid)
        self._load_from_state()
        self.specsChanged.emit()

    def _on_node_moved(self, unit_id, x, y):
        if not self.window_state:
            return
        col = self.window_state.columns.get(unit_id)
        if col is not None:
            col.node_pos = (float(x), float(y))
            self.window_state.mark_modified()

    def _on_validation_rejected(self, reason: str):
        self.dof_status_label.setText(f"⚠️ {reason}")
        set_state(self.dof_status_label, "warn")

    def _on_edge_clicked(self, conn_id: str):
        """Open a connection in the inline editor."""
        if not self.window_state:
            return
        conn = next((c for c in self.window_state.connections if c.id == conn_id),
                    None)
        if conn is None:
            self.ov_editor_stack.setCurrentWidget(self.ov_placeholder)
            return
        dst = self.window_state.columns.get(conn.dst)
        natural = None
        fs = self._try_build_flowsheet()
        if fs is not None:
            try:
                from core.flowsheet import natural_q
                natural = natural_q(fs.units[conn.src], conn.port)
            except Exception:
                natural = None
        self.ov_connection_panel.set_config(
            conn, natural_q=natural,
            max_stage=dst.num_stages if dst else None)
        self.ov_editor_stack.setCurrentWidget(self.ov_connection_panel)
        self.ov_config_group.setTitle(f"{conn.src}.{conn.port} → {conn.dst}")

    def _on_ov_connection_changed(self):
        """Write an edited connection back. Stages are 0-based in the panel and
        1-based in core.flowsheet, converted here at the seam."""
        from dataclasses import replace
        if not self.window_state:
            return
        cfg = self.ov_connection_panel.get_config()
        ws = self.window_state
        ws.connections = [
            replace(c, stage=int(cfg["stage"]) + 1,
                    split_fraction=float(cfg["split_fraction"]), q=cfg["q"])
            if c.id == cfg["id"] else c
            for c in ws.connections]
        ws.mark_modified()
        self._update_column_canvas()
        self._update_dof_status()
        self.specsChanged.emit()

    def _on_element_clicked(self, unit_id: str, element_type: str):
        """Open the clicked element's config in the overview's inline editor."""
        self._set_active_column(unit_id)
        if element_type == "condenser":
            self._load_condenser_into(self.ov_condenser_panel)
            self.ov_editor_stack.setCurrentWidget(self.ov_condenser_panel)
            self.ov_config_group.setTitle("Condenser Configuration")
        elif element_type == "reboiler":
            self._load_reboiler_into(self.ov_reboiler_panel)
            self.ov_editor_stack.setCurrentWidget(self.ov_reboiler_panel)
            self.ov_config_group.setTitle("Reboiler Configuration")
        elif element_type.startswith("module_"):
            mid = element_type[len("module_"):]
            m = self.window_state.modules.get(mid) if self.window_state else None
            if m is None:
                self.ov_editor_stack.setCurrentWidget(self.ov_placeholder)
                return
            self.current_module_id = mid            # what the editor writes back to
            self.ov_module_panel.set_config(_module_to_config(m))
            self.ov_editor_stack.setCurrentWidget(self.ov_module_panel)
            self.ov_config_group.setTitle(f"{mid} Configuration")
        else:
            # column body / trays / modules: no dedicated editor yet
            self.ov_editor_stack.setCurrentWidget(self.ov_placeholder)
            self.ov_config_group.setTitle("Element Configuration")

    def _open_stream_by_id(self, unit_id: str, sid: str):
        """Load any clicked stream (feed/product/side draw) into the overview
        stream panel — the scene emits the column and the stream id directly."""
        self._set_active_column(unit_id)
        if not (self.window_state and sid in self.window_state.streams):
            self.ov_editor_stack.setCurrentWidget(self.ov_placeholder)
            self.ov_config_group.setTitle("Element Configuration")
            return
        self.ov_stream_panel.set_window_state(self.window_state)
        s = self.window_state.streams[sid]
        self.ov_stream_panel.select_stream(sid, {
            "type": s.stream_type.value,
            "stage": s.stage,
            "temperature": s.temperature,
            "flow": s.flow,
            "composition": s.composition,
            "phase": s.phase,
        })
        self.ov_editor_stack.setCurrentWidget(self.ov_stream_panel)
        self.ov_config_group.setTitle(f"{sid} Configuration")

    def _stream_item(self, stream_id):
        """List item carrying the canonical stream id in UserRole, so a rename
        edit still knows which stream it was."""
        item = QTableWidgetItem(stream_id)
        item.setData(Qt.UserRole, stream_id)
        return item

    def _add_inlet_rows(self):
        """Show what arrives from another column, as read-only rows.

        A connection feeds a *stage*, not one of this column's Stream objects —
        that is what keeps makeup working (an ordinary feed on the same stage
        just blends, see build_solver_input). But then nothing on the Streams
        page would mention the recycle at all, and a user would be looking at a
        column whose real inlet is invisible. These rows are that inlet: not
        editable here (the connection owns them), and greyed with a tooltip
        saying where to edit them, per the "nothing is silently ignored" rule.
        """
        ws = self.window_state
        if not ws or not self.beta:
            return                       # nothing can be connected without beta
        from PySide6.QtGui import QBrush, QColor
        from gui.theme import palette
        cid = ws.active_column_id
        for c in ws.connections:
            if c.dst != cid:
                continue
            label = f"← {c.src}.{c.port} @ stage {c.stage - 1}"
            item = QTableWidgetItem(label)
            item.setData(Qt.UserRole, None)          # not a stream: never renamed
            item.setFlags(Qt.ItemIsEnabled)          # selectable-looking, not editable
            item.setForeground(QBrush(QColor(palette.TEXT_MUTED)))
            split = (f", {c.split_fraction:.0%} of it" if c.split_fraction < 1.0
                     else "")
            item.setToolTip(
                f"Fed by the connection from {c.src}.{c.port}{split}. Its flow "
                "and composition come from that column's solution — click the "
                "stream on the Flowsheet diagram to change the stage or split.")
            row = self.stream_list.rowCount()
            self.stream_list.insertRow(row)
            self.stream_list.setItem(row, 0, item)

    def _on_stream_renamed(self, item):
        """Commit an in-place rename to window_state, or revert the cell."""
        old_id = item.data(Qt.UserRole)
        new_id = item.text().strip()
        if old_id is None or new_id == old_id:
            return
        if not (self.window_state and self.window_state.rename_stream(old_id, new_id)):
            self.stream_list.blockSignals(True)
            item.setText(old_id)                  # empty/duplicate/unknown -> revert
            self.stream_list.blockSignals(False)
            return
        self.stream_list.blockSignals(True)
        item.setData(Qt.UserRole, new_id)
        item.setText(new_id)                      # normalised (stripped) form
        self.stream_list.blockSignals(False)
        if self.current_stream_id == old_id:
            self.current_stream_id = new_id
        for panel in (self.stream_config, self.ov_stream_panel):
            if panel.current_stream_id == old_id:
                panel.current_stream_id = new_id
        self._update_column_canvas()
        self._update_dof_status()
        self.specsChanged.emit()

    def _on_stream_selected(self):
        """Handle stream selection."""
        row = self.stream_list.currentRow()
        if row >= 0:
            item = self.stream_list.item(row, 0)
            if item:
                if item.data(Qt.UserRole) is None:
                    return          # a read-only inlet row, not one of our streams
                self.current_stream_id = item.text()
                stream_data = {}
                if self.window_state and self.current_stream_id in self.window_state.streams:
                    stream = self.window_state.streams[self.current_stream_id]
                    stream_data = {
                        "type": stream.stream_type.value,
                        "stage": stream.stage,
                        "temperature": stream.temperature,
                        "flow": stream.flow,
                        "composition": stream.composition,
                        "phase": stream.phase,
                    }
                self.stream_config.select_stream(self.current_stream_id, stream_data)

    def _on_module_selected(self):
        """Handle module selection."""
        row = self.module_list.currentRow()
        if row >= 0:
            item = self.module_list.item(row, 0)
            if item:
                self.current_module_id = item.text()
                m = (self.window_state.modules.get(self.current_module_id)
                     if self.window_state else None)
                if m is not None:
                    # restore the stored config (previously reset to defaults,
                    # losing stage/duty/num_stages on every reselect)
                    self.module_config.set_config(_module_to_config(m))
                else:
                    self.module_config.set_config({
                        "type": self.current_module_id.split(" - ")[0]
                        if " - " in self.current_module_id else "Interreboiler"})

    def _on_config_changed(self):
        """Operating params (pressure / pressure drop) changed."""
        if self.window_state:
            self.window_state.pressure = self.pressure_input.valueInSI()
            self.window_state.pressure_drop = self.pressure_drop_spin.value()
            self.window_state.stage_efficiency = self.efficiency_spin.value()
            self.window_state.mark_modified()
        self._update_column_canvas()
        self._update_dof_status()
        self.specsChanged.emit()

    # --- Condenser / Reboiler: window_state is the single source of truth ---

    def _on_operating_specs_changed(self):
        """Operating-spec slots -> window_state.specs (one spec per kind, last
        row wins), then mirror back into the condenser/reboiler fields."""
        if self.window_state:
            by_kind = {}
            for s in self.operating_specs_panel.get_specs():
                by_kind[s.kind] = s            # last occurrence per kind wins
            self.window_state.specs = list(by_kind.values())
            self.window_state.mark_modified()
            self._load_condenser_into(self.condenser_panel)
            self._load_reboiler_into(self.reboiler_panel)
        self._update_column_canvas()
        self._update_dof_status()
        self.specsChanged.emit()

    def _on_condenser_changed(self, panel):
        """Persist a condenser edit (from either the config tab or overview)."""
        self._save_condenser_from(panel)
        self._update_column_canvas()
        self._update_dof_status()
        self.specsChanged.emit()

    def _on_reboiler_changed(self, panel):
        """Persist a reboiler edit (from either the config tab or overview)."""
        self._save_reboiler_from(panel)
        self._update_column_canvas()
        self._update_dof_status()
        self.specsChanged.emit()

    def _save_condenser_from(self, panel):
        if not self.window_state:
            return
        from core.dof import SpecKind
        cfg = panel.get_config()
        cc = self.window_state.condenser_config
        cc.condenser_type = CondenserType(cfg["type"])
        # Treat 0 as "unset" so DoF matches the panel's own >0 spec test
        cc.subcooling_temp = cfg.get("subcooling_temp") or None
        # Reflux / vapour distillate are operating specs -> the shared specs list,
        # keyed by kind so they never double-count with the Operating slots.
        self.window_state.upsert_operating_spec(
            SpecKind.REFLUX_RATIO, cfg.get("reflux_ratio"))
        self.window_state.upsert_operating_spec(
            SpecKind.DISTILLATE_RATE, cfg.get("vapor_distillate_flow"))
        self._reload_operating_panel()

    def _save_reboiler_from(self, panel):
        if not self.window_state:
            return
        from core.dof import SpecKind
        cfg = panel.get_config()
        rc = self.window_state.reboiler_config
        rc.reboiler_type = ReboilerType(cfg["type"])
        self.window_state.upsert_operating_spec(
            SpecKind.BOILUP_RATIO, cfg.get("boilup_ratio"))
        self.window_state.upsert_operating_spec(
            SpecKind.BOTTOMS_RATE, cfg.get("bottoms_flow"))
        self._reload_operating_panel()

    def _reload_operating_panel(self):
        """Keep the Operating slots in sync after a condenser/reboiler edit."""
        if hasattr(self, "operating_specs_panel"):
            self.operating_specs_panel.set_specs(self.window_state.specs)

    def _load_condenser_into(self, panel):
        if not self.window_state:
            return
        from core.dof import SpecKind
        cc = self.window_state.condenser_config
        reflux = self.window_state.get_operating_spec(SpecKind.REFLUX_RATIO)
        dist = self.window_state.get_operating_spec(SpecKind.DISTILLATE_RATE)
        panel.blockSignals(True)
        panel.set_config({
            "type": cc.condenser_type.value,
            "subcooling_temp": cc.subcooling_temp or 0,
            "reflux_ratio": reflux.value if reflux else 0,
            "vapor_distillate_flow": dist.value if dist else 0,
        })
        panel.blockSignals(False)

    def _load_reboiler_into(self, panel):
        if not self.window_state:
            return
        from core.dof import SpecKind
        rc = self.window_state.reboiler_config
        boilup = self.window_state.get_operating_spec(SpecKind.BOILUP_RATIO)
        bottoms = self.window_state.get_operating_spec(SpecKind.BOTTOMS_RATE)
        panel.blockSignals(True)
        panel.set_config({
            "type": rc.reboiler_type.value,
            "boilup_ratio": boilup.value if boilup else 0,
            "bottoms_flow": bottoms.value if bottoms else 0,
        })
        panel.blockSignals(False)

    # --- Streams: shared persist for both the Streams tab and overview editor ---
    # The PANEL's current_stream_id is the single source of truth for which
    # stream a save targets — the tab keeps no separate copy that can diverge
    # (it only mirrors the list-row selection).

    def _on_stream_changed(self):
        """Stream edited from the Streams sub-tab."""
        self._save_stream_from(self.stream_config)

    def _on_ov_stream_changed(self):
        """Stream edited from the overview inline editor."""
        self._save_stream_from(self.ov_stream_panel)

    def _save_stream_from(self, panel):
        stream_id = panel.current_stream_id
        if self.window_state and stream_id and stream_id in self.window_state.streams:
            data = panel.get_stream_data()
            stream = self.window_state.streams[stream_id]
            stream.stream_type = StreamType(data["type"])
            stream.stage = data["stage"]
            stream.temperature = data["temperature"]
            stream.flow = data["flow"]
            stream.composition = data["composition"]
            stream.phase = data.get("phase", "liquid")
            # Products the user edits by hand stop being auto_balance targets.
            if stream.stream_type in (StreamType.DISTILLATE, StreamType.BOTTOMS):
                stream.user_specified = True
            self.window_state.mark_modified()

        self._update_column_canvas()
        self._update_dof_status()
        self.specsChanged.emit()

    def _on_module_changed(self):
        """Handle module configuration change."""
        self._sync_module_to_state()
        self._update_column_canvas()
        self._update_dof_status()
        self.specsChanged.emit()

    def _on_ov_module_changed(self):
        """Same, for the module editor inline on the Column Overview page. Keeps
        the Advanced Modules panel in step so the two never show different
        numbers for one module."""
        self._sync_module_to_state(self.ov_module_panel)
        m = (self.window_state.modules.get(self.current_module_id)
             if self.window_state else None)
        if m is not None and self.module_list.currentRow() >= 0:
            item = self.module_list.item(self.module_list.currentRow(), 0)
            if item and item.text() == self.current_module_id:
                self.module_config.set_config(_module_to_config(m))
        self._update_column_canvas()
        self._update_dof_status()
        self.specsChanged.emit()

    def _sync_module_to_state(self, panel=None):
        """Persist a module panel's config into window_state.modules so it
        survives Save/Load and feeds DoF. No-op without a selected module."""
        if not self.window_state or not self.current_module_id:
            return
        cfg = (panel or self.module_config).get_config()
        try:
            mtype = ModuleType(cfg.get("type", "Interreboiler"))
        except ValueError:
            mtype = ModuleType.INTERREBOILER  # unknown label -> safe default
        rs = cfg.get("return_stage")
        self.window_state.modules[self.current_module_id] = ModuleConfig(
            module_type=mtype,
            stage=int(cfg.get("stage", 1)),
            num_stages=int(cfg.get("num_stages") or 1),
            boilup_ratio=cfg.get("boilup_ratio"),
            reflux_ratio=cfg.get("reflux_ratio"),
            duty=cfg.get("duty"),
            return_stage=int(rs) if rs is not None else None,
            rate=cfg.get("rate"),
        )
        self.window_state.mark_modified()

    def _add_stream(self):
        """Add a new stream."""
        row = self.stream_list.rowCount()
        stream_id = f"Stream {row + 1}"
        
        # Add to state first
        new_stream = Stream(id=stream_id, stream_type=StreamType.FEED, stage=10)
        if self.window_state:
            self.window_state.add_stream(new_stream)

        self.stream_list.insertRow(row)
        self.stream_list.setItem(row, 0, self._stream_item(stream_id))
        self.stream_config.set_window_state(self.window_state)
        self.current_stream_id = stream_id
        self.stream_list.setCurrentCell(row, 0)
        # Pass the new stream's data so the panel doesn't keep showing the
        # previously selected stream's numbers.
        self.stream_config.select_stream(stream_id, {
            "type": new_stream.stream_type.value,
            "stage": new_stream.stage,
            "temperature": new_stream.temperature,
            "flow": new_stream.flow,
            "composition": new_stream.composition,
        })
        self._update_column_canvas()
        self.specsChanged.emit()

    def _remove_stream(self):
        """Remove selected stream."""
        row = self.stream_list.currentRow()
        if row >= 0:
            item = self.stream_list.item(row, 0)
            if item and self.window_state:
                self.window_state.remove_stream(item.text())
                
            self.stream_list.removeRow(row)
            self.stream_config.clear()
            self.current_stream_id = None
            self._update_column_canvas()
            self.specsChanged.emit()

    def _add_module(self):
        """Add a new module."""
        row = self.module_list.rowCount()
        module_type = self.module_type_combo.currentText()
        module_id = f"{module_type} {row + 1}"
        self.module_list.insertRow(row)
        self.module_list.setItem(row, 0, QTableWidgetItem(module_id))
        self.current_module_id = module_id
        self.module_list.setCurrentCell(row, 0)
        # Sensible starting geometry: draw mid-column, return one tray away in the
        # direction that type requires (up for a stripper/pumparound, down for a
        # rectifier) — an invalid pair would refuse to solve.
        n = self.window_state.num_stages if self.window_state else 20
        draw = max(2, min(n - 2, n // 2))
        down = module_type == ModuleType.SIDE_RECTIFIER.value
        self.module_config.set_config({
            "type": module_type, "stage": draw,
            "return_stage": draw + 1 if down else draw - 1, "num_stages": 4})
        self._sync_module_to_state()
        self._update_column_canvas()
        self.specsChanged.emit()

    def _remove_module(self):
        """Remove selected module."""
        row = self.module_list.currentRow()
        if row >= 0:
            if self.window_state and self.current_module_id:
                self.window_state.modules.pop(self.current_module_id, None)
            self.module_list.removeRow(row)
            self.module_config.set_config({})
            self.current_module_id = None
            self._update_column_canvas()
            self.specsChanged.emit()

    def _column_view_model(self, col):
        """What the scene needs to draw one column. Same collection the single
        column canvas used, run per column instead of once."""
        num_stages = col.num_stages

        feeds, products, modules = [], [], []

        # Stages are 0-based from the top (0 = distillate/condenser, N-1 = reboiler).
        def _stage(s, default):
            st = s.stage if s.stage is not None else default
            return max(0, min(num_stages - 1, st))

        for stream_id, stream in col.streams.items():
            if stream.stream_type == StreamType.FEED:
                feeds.append((_stage(stream, 10), stream_id))
            elif stream.stream_type == StreamType.DISTILLATE:
                products.append((_stage(stream, 0), stream_id, "distillate"))
            elif stream.stream_type == StreamType.BOTTOMS:
                products.append((_stage(stream, num_stages - 1), stream_id, "bottoms"))
            elif stream.stream_type == StreamType.SIDESTREAM:
                products.append((_stage(stream, 10), stream_id, "sidestream"))

        for module_id, module in col.modules.items():
            modules.append({
                "id": module_id, "type": module.module_type.value,
                "stage": max(0, min(num_stages - 1, module.stage)),
                "return_stage": (None if module.return_stage is None else
                                 max(0, min(num_stages - 1, module.return_stage))),
                "rate": module.rate, "duty": module.duty,
            })

        return {
            "num_stages": num_stages,
            "condenser_type": col.condenser_config.condenser_type.value,
            "reboiler_type": col.reboiler_config.reboiler_type.value,
            "feeds": feeds, "products": products, "modules": modules,
            "node_pos": col.node_pos,
        }

    def _try_build_flowsheet(self):
        """A topology-only Flowsheet: ids, stages, ports and connections.

        Deliberately NOT the solver's gather. The editor has to say which edges
        are recycles and which are illegal *while the user is still wiring
        things up* — long before the columns have specs, feeds or thermo, which
        a real gather demands. Ports, tears and connection legality need none of
        that, so this builds exactly what those answers require and nothing
        else. `solve_flowsheet` refuses a Unit built this way.
        """
        if not self.window_state:
            return None
        from core.flowsheet import Flowsheet, Unit
        from core.side_sections import SideSection

        ws = self.window_state
        units = {}
        for cid, col in ws.columns.items():
            n = max(2, int(col.num_stages))
            draws = [(s.id, min(n, max(1, (s.stage or 0) + 1)),
                      0.0 if s.phase == "vapor" else 1.0,
                      1.0 if s.phase == "vapor" else 0.0)
                     for s in col.streams.values()
                     if s.stream_type == StreamType.SIDESTREAM]
            sections = []
            for mid, kind, ds, rs, rate, ratio, nst in col.side_sections():
                try:
                    sections.append(SideSection(
                        id=mid, kind=kind, draw_stage=ds + 1, return_stage=rs + 1,
                        rate=rate, ratio=ratio, n_stages=nst))
                except ValueError:
                    pass          # a half-configured module still draws
            units[cid] = Unit(
                id=cid, n_stages=n, draws=draws, sections=sections,
                condenser=col.condenser_config.condenser_type.value.lower())
        return Flowsheet(units=units, connections=list(ws.connections),
                         comps=ws.get_species_names())

    def _update_column_canvas(self):
        """Redraw whichever diagram is on screen from window_state."""
        if not self.window_state:
            return
        ws = self.window_state
        if not self.beta:
            # Default: the single-column diagram, fed exactly as it always was.
            vm = self._column_view_model(ws.active_column)
            feeds = vm["feeds"]
            feed_stage = feeds[0][0] if feeds else vm["num_stages"] // 2
            self.single_canvas.set_column_config(
                vm["num_stages"], feed_stage,
                vm["condenser_type"], vm["reboiler_type"])
            self.single_canvas.set_streams(feeds, vm["products"], vm["modules"])
            return
        vms = {cid: self._column_view_model(col) for cid, col in ws.columns.items()}
        self.flowsheet_scene.rebuild(vms, list(ws.connections),
                                     flowsheet=self._try_build_flowsheet(),
                                     active_unit=ws.active_column_id)
        self.flowsheet_scene.set_stream_results(getattr(ws, "flowsheet_result", None))
        self._refresh_column_combo()

    def _update_dof_status(self):
        """Update the DoF status from the unified analyzer; auto-balance when
        fully specified so Distillate/Bottoms reflect the current specs."""
        if not self.window_state:
            return
        icon, message, can_run = self.window_state.get_specification_status()
        self.dof_status_label.setText(f"{icon} {message}")
        set_state(self.dof_status_label, "ok" if can_run else "warn")
        if can_run:
            self.window_state.auto_balance()
            self._update_column_canvas()

    def set_window_state(self, window_state):
        """Set the window state object and initialize UI from it."""
        self.window_state = window_state
        # the single canvas keeps a state reference (it edits num_stages inline);
        # the flowsheet scene holds none and is fed by _update_column_canvas()
        self.single_canvas.set_window_state(window_state)
        self.refresh_beta()          # pick the right diagram before loading
        self._load_from_state()

    def _load_from_state(self):
        """Initialize UI components from the window state."""
        if not self.window_state:
            return
            
        # Load Operating Parameters
        self.pressure_input.setValue(self.window_state.pressure)
        self.pressure_drop_spin.setValue(self.window_state.pressure_drop)
        self.efficiency_spin.setValue(getattr(self.window_state, "stage_efficiency", 1.0))
        
        # Update stream config panels with window_state (to get species list)
        self.stream_config.set_window_state(self.window_state)
        self.ov_stream_panel.set_window_state(self.window_state)

        # Load condenser/reboiler panels from the shared state
        self._load_condenser_into(self.condenser_panel)
        self._load_reboiler_into(self.reboiler_panel)

        # Operating-spec slots (window_state first so recovery rows can read the
        # shared light/heavy key indices for their inline pickers).
        self.operating_specs_panel.set_window_state(self.window_state)
        self.operating_specs_panel.set_species(self.window_state.get_species_names())
        self.operating_specs_panel.set_specs(self.window_state.specs)

        # Load Streams
        self.stream_list.setRowCount(0)
        for stream_id, stream in self.window_state.streams.items():
            row = self.stream_list.rowCount()
            self.stream_list.insertRow(row)
            self.stream_list.setItem(row, 0, self._stream_item(stream_id))
        self._add_inlet_rows()

        # Always keep a stream selected (prefer the current one, else row 0) so
        # edits target a real stream instead of being silently dropped.
        if self.stream_list.rowCount():
            target = next((r for r in range(self.stream_list.rowCount())
                           if self.stream_list.item(r, 0).text() == self.current_stream_id), 0)
            self.stream_list.setCurrentCell(target, 0)

        # Load Modules — without this a .colx's side strippers/rectifiers/
        # pumparounds live in window_state (and solve) but never show in the list.
        self.module_list.setRowCount(0)
        for module_id in self.window_state.modules:
            row = self.module_list.rowCount()
            self.module_list.insertRow(row)
            self.module_list.setItem(row, 0, QTableWidgetItem(module_id))
        if self.module_list.rowCount():
            target = next((r for r in range(self.module_list.rowCount())
                           if self.module_list.item(r, 0).text() == self.current_module_id), 0)
            self.module_list.setCurrentCell(target, 0)   # fires _on_module_selected
        else:
            self.current_module_id = None

        # Update Canvas
        self._update_column_canvas()
        self._update_dof_status()

    def clear(self):
        """Clear all settings."""
        if self.window_state:
            self._load_from_state()
        else:
            self.pressure_input.setValue(1.0)
            self.pressure_drop_spin.setValue(0)
            self.stream_list.setRowCount(0)
            self.module_list.setRowCount(0)
            self.stream_config.clear()
            self.condenser_panel.set_config({})
            self.reboiler_panel.set_config({})

    def refresh(self):
        """Refresh the specs tab - call when species change."""
        if self.window_state:
            self._load_from_state()

    def get_column_config(self) -> dict:
        """Get current column configuration."""
        return {
            "pressure": self.pressure_input.valueInSI(),
            "pressure_drop": self.pressure_drop_spin.value(),
            "condenser": self.condenser_panel.get_config(),
            "reboiler": self.reboiler_panel.get_config()
        }

    def get_streams(self) -> dict:
        """Get stream configuration."""
        return self.stream_config.get_stream_data()

    def get_modules(self) -> dict:
        """Get module configuration."""
        return self.module_config.get_config()
