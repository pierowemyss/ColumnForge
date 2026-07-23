import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QTableWidget,
    QTableWidgetItem, QPushButton, QGroupBox, QSplitter, QStackedWidget
)
from PySide6.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.ticker import MaxNLocator

from gui.state.window_state import StreamType
from gui.panels.sub_tab_bar import SubTabBar
from gui.panels.sci_spin_box import fmt
from gui.plotting import (
    CompactNavigationToolbar, ternary_axes, composition_from_click,
    residue_curve, residue_curve_map, singular_points,
    distillation_boundaries, binary_equilibrium_curve, mccabe_thiele_steps,
    TEMP_C, DATA_C, BOUNDARY_C, RECT_C, STRIP_C,
)


# Data-type -> profile key. Stage 0 = distillate everywhere (app convention).
_SERIES_KEYS = {
    "Pressure": "pressure",
    "Liquid Flow": "liquid_flow",
    "Vapor Flow": "vapor_flow",
    "K-Values": "k_values",
    "Enthalpy": "enthalpy",
}
_SERIES_LABELS = {
    "Pressure": "Pressure (Psat unit)",
    "Liquid Flow": "Liquid flow L (kmol/h)",
    "Vapor Flow": "Vapor flow V (kmol/h)",
    "K-Values": "Geometric-mean K",
    "Enthalpy": "Vapour latent heat (J/mol)",
}


def stream_summary(profile: dict) -> dict:
    """Product streams (distillate, bottoms, side draws) with flows/T/comps,
    terminal duties in kW, and the per-component mass-balance closure. Pure (no
    Qt) so it is unit-testable.

    Closure = feed_in - (D*xD + B*xB + sum side draws); ~0 for a converged run.
    """
    comps = profile["comps"]
    T = np.asarray(profile["T"])
    C = len(comps)
    D, B = float(profile["D"]), float(profile["B"])
    xD, xB = np.asarray(profile["xD"]), np.asarray(profile["xB"])

    products = [
        {"name": "Distillate", "phase": profile.get("distillate_phase", "liquid"),
         "flow": D, "T": float(T[0]), "comp": xD},
        {"name": "Bottoms", "phase": "liquid",
         "flow": B, "T": float(T[-1]), "comp": xB},
    ]
    out = D * xD + B * xB
    for sd in profile.get("side_draws", []):
        j = sd["stage"]
        for phase, flow, comp in (("liquid", sd["liquid"], np.asarray(sd["x"])),
                                  ("vapor", sd["vapor"], np.asarray(sd["y"]))):
            if flow > 0.0:
                products.append({"name": f"Side draw @ stage {j}", "phase": phase,
                                 "flow": float(flow), "T": float(T[j]), "comp": comp})
                out = out + flow * comp

    # Side stripper/rectifier products. The section's draw and return are internal
    # (already netted out of side_draws/feed_totals by core.side_sections), so only
    # the product it exports shows up here.
    for ss in profile.get("side_sections", []):
        comp = np.asarray(ss["comp"])
        products.append({                     # stripper bottoms / condensed
            "name": f"{ss['id']} product",    # rectifier distillate: both liquid
            "phase": "liquid",
            "flow": float(ss["flow"]), "T": float(ss["T"]), "comp": comp})
        out = out + float(ss["flow"]) * comp

    feed = np.asarray(profile.get("feed_totals", np.full(C, np.nan)))
    closure = feed - out
    Qc, Qr = profile.get("condenser_duty"), profile.get("reboiler_duty")
    return {
        "comps": comps, "products": products,
        "condenser_duty": None if Qc is None else float(Qc),   # kJ/h (raw)
        "reboiler_duty": None if Qr is None else float(Qr),    # kJ/h (raw)
        "closure": closure,
        "closure_max": float(np.nanmax(np.abs(closure))) if C else 0.0,
    }


def profile_to_csv_rows(profile: dict, units=None, mws=None) -> list:
    """Header + data rows for CSV export. Pure (no Qt) so it is unit-testable.
    Stages are 0-based from the top (row for stage 0 = distillate first).

    units: optional DisplayUnits — T and flow columns convert to the chosen
    output units (labels carry the unit). mws: per-component molar masses,
    needed only for a kg/h flow unit."""
    from core.units import DisplayUnits
    u = units or DisplayUnits()
    comps = profile["comps"]
    x, T = profile["x"], np.asarray(profile["T"])
    mw = None
    if u.flow == "kg/h" and mws is not None and all(m is not None for m in mws):
        mw = np.asarray(x, float) @ np.asarray(mws, float)     # per-stage MW
    extra = []
    for label, key in _SERIES_KEYS.items():
        if key not in profile:
            continue
        vals = np.asarray(profile[key], float)
        if label in ("Liquid Flow", "Vapor Flow"):
            if mw is not None:
                vals = vals * mw
            extra.append((f"{label} ({u.f_label()})", vals))
        else:
            extra.append((label, vals))
    rows = [["Stage", f"T ({u.t_label()})"] + list(comps)
            + [label for label, _ in extra]]
    for i in range(profile["n_stages"]):
        rows.append([i, float(u.T(T[i]))] + [float(v) for v in x[i]]
                    + [float(vals[i]) for _, vals in extra])
    return rows


class ResultsTab(QWidget):
    """Results tab with visualization and data display."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.window_state = None
        self._tern_ctx = None            # (P, antoine, gamma_fn) for click-to-plot
        self._setup_ui()
        self._setup_styles()
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        self.data_combo.currentTextChanged.connect(lambda _: self._draw_plot())
        for combo in (self.temp_unit_combo, self.flow_unit_combo, self.duty_unit_combo):
            combo.currentTextChanged.connect(lambda _=None: self._on_units_changed())
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.set_view_type("Plot")

    def set_window_state(self, window_state):
        self.window_state = window_state

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left: Control Column (LHS Sub-Tab Column)
        self.sub_tab_bar = SubTabBar(self)
        self.sub_tab_bar.addTab("View")
        self.sub_tab_bar.addTab("Streams")
        self.sub_tab_bar.addTab("Performance")
        self.sub_tab_bar.tabClicked.connect(self._on_sub_tab_changed)
        main_layout.addWidget(self.sub_tab_bar)

        # Right: Results Display Pane
        self.stack = QStackedWidget(self)

        view_page = QWidget()
        view_layout = QVBoxLayout(view_page)
        view_layout.setSpacing(10)
        view_layout.setContentsMargins(10, 10, 10, 10)

        # Control Row (Dropdowns)
        control_layout = QHBoxLayout()

        # View type dropdown
        control_layout.addWidget(QLabel("Display:"))
        self.view_combo = QComboBox(self)
        self.view_combo.addItems(["Plot", "Table"])
        control_layout.addWidget(self.view_combo)

        # Data type dropdown. Types a run didn't produce are disabled (greyed)
        # rather than plotting placeholder text.
        control_layout.addWidget(QLabel("Data:"))
        self.data_combo = QComboBox(self)
        self.data_combo.addItems([
            "Compositions",
            "Temperature",
            "Pressure",
            "Liquid Flow",
            "Vapor Flow",
            "K-Values",
            "Enthalpy",
            "Ternary Map",
            "McCabe-Thiele",
        ])
        control_layout.addWidget(self.data_combo)
        control_layout.addStretch()

        # Output unit selectors (display/export only; internals stay in solver
        # units). Pressure is left in its Psat-fit unit — see _fill_table.
        from core.units import TEMPERATURE, FLOW, DUTY
        control_layout.addWidget(QLabel("T:"))
        self.temp_unit_combo = QComboBox(self)
        self.temp_unit_combo.addItems(list(TEMPERATURE))
        control_layout.addWidget(self.temp_unit_combo)
        control_layout.addWidget(QLabel("Flow:"))
        self.flow_unit_combo = QComboBox(self)
        self.flow_unit_combo.addItems(list(FLOW))
        control_layout.addWidget(self.flow_unit_combo)
        control_layout.addWidget(QLabel("Duty:"))
        self.duty_unit_combo = QComboBox(self)
        self.duty_unit_combo.addItems(list(DUTY))
        control_layout.addWidget(self.duty_unit_combo)

        view_layout.addLayout(control_layout)

        # Main Display: Plot or Table
        self.display_splitter = QSplitter(Qt.Vertical)

        # Matplotlib plot + navigation toolbar (shared app convention)
        plot_group = QGroupBox("Plot")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        # Data table
        table_group = QGroupBox("Data")
        table_layout = QVBoxLayout(table_group)
        self.data_table = QTableWidget(0, 5)
        self.data_table.setHorizontalHeaderLabels(
            ["Stage", "T (degC)", "x1", "x2", "x3"])
        self.data_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.data_table)

        # Export button
        self.export_btn = QPushButton("Export CSV")
        table_layout.addWidget(self.export_btn)

        # Add to splitter (one will be hidden based on view_combo)
        self.display_splitter.addWidget(plot_group)
        self.display_splitter.addWidget(table_group)

        view_layout.addWidget(self.display_splitter, 3)

        # Bottom Row: Simulation Summary
        summary_group = QGroupBox("Simulation Summary")
        summary_layout = QVBoxLayout(summary_group)

        self.summary_label = QLabel(
            "Status: Not Run\n"
            "Stages: --\n"
            "Iterations: --\n"
            "Runtime: --"
        )
        self.summary_label.setProperty("mono", True)
        summary_layout.addWidget(self.summary_label)

        view_layout.addWidget(summary_group, 1)

        self.stack.addWidget(view_page)
        self.stack.addWidget(self._build_streams_page())
        self.stack.addWidget(self._build_performance_page())
        main_layout.addWidget(self.stack)

    def _build_streams_page(self):
        """Stream Summary sub-view: product table, terminal duties (kW),
        mass-balance closure, and a spec-vs-achieved table (B8)."""
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        prod_group = QGroupBox("Product streams")
        pg = QVBoxLayout(prod_group)
        self.stream_table = QTableWidget(0, 4)
        pg.addWidget(self.stream_table)
        lay.addWidget(prod_group, 3)

        self.duty_label = QLabel("Run a simulation to see stream results.")
        self.duty_label.setProperty("mono", True)
        lay.addWidget(self.duty_label)

        spec_group = QGroupBox("Specifications: target vs achieved")
        sg = QVBoxLayout(spec_group)
        self.spec_table = QTableWidget(0, 3)
        self.spec_table.setHorizontalHeaderLabels(["Spec", "Target", "Achieved"])
        self.spec_table.horizontalHeader().setStretchLastSection(True)
        sg.addWidget(self.spec_table)
        lay.addWidget(spec_group, 2)
        return page

    def _build_performance_page(self):
        """Scalar column figures (reflux/boilup ratio, rates, duties) in one
        place, read from the raw profile."""
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(10, 10, 10, 10)
        grp = QGroupBox("Column performance")
        g = QVBoxLayout(grp)
        self.perf_table = QTableWidget(0, 2)
        self.perf_table.setHorizontalHeaderLabels(["Quantity", "Value"])
        self.perf_table.horizontalHeader().setStretchLastSection(True)
        self.perf_table.verticalHeader().setVisible(False)
        g.addWidget(self.perf_table)
        lay.addWidget(grp)
        return page

    def _fill_performance(self):
        """Populate the Performance sub-view from the raw profile."""
        prof = self._profile()
        if not prof or "D" not in prof:      # BVM-style profiles have no products
            self.perf_table.setRowCount(0)
            return
        u = self._units()
        D, B = float(prof["D"]), float(prof["B"])
        F = float(np.nansum(np.asarray(prof.get("feed_totals", [0.0]))))
        comps = prof["comps"]
        species = getattr(self.window_state, "species", {}) or {}
        mws = [getattr(species.get(c), "mw", None) for c in comps]

        def flow_str(val, comp):
            mw = (float(np.asarray(comp) @ np.asarray(mws, float))
                  if all(m is not None for m in mws) else None)
            v = u.F(val, mw) if u.flow == "kg/h" and mw else val
            return f"{fmt(v)} {u.f_label()}"

        Qc, Qr = prof.get("condenser_duty"), prof.get("reboiler_duty")
        rows = [
            ("Reflux ratio (R)", fmt(prof.get("R"))),
            ("Boilup ratio", fmt(prof.get("boilup_ratio"))),
            ("Distillate rate (D)", flow_str(D, prof["xD"])),
            ("Bottoms rate (B)", flow_str(B, prof["xB"])),
            ("D / F", fmt(D / F) if F > 0 else "—"),
            ("B / F", fmt(B / F) if F > 0 else "—"),
            ("Feed quality (q)", fmt(prof.get("feed_q"))),
            ("Condenser duty",
             f"{fmt(u.Q(Qc))} {u.q_label()}" if Qc is not None else "—"),
            ("Reboiler duty",
             f"{fmt(u.Q(Qr))} {u.q_label()}" if Qr is not None else "—"),
            ("Number of stages", fmt(prof.get("n_stages"))),
            ("Feed stage (from top)", fmt(prof.get("feed_stage", 0) + 1)),
        ]
        self.perf_table.setRowCount(len(rows))
        for r, (name, val) in enumerate(rows):
            self.perf_table.setItem(r, 0, QTableWidgetItem(name))
            self.perf_table.setItem(r, 1, QTableWidgetItem(str(val)))

    def _on_sub_tab_changed(self, index: int):
        """Handle sub-tab change."""
        self.stack.setCurrentIndex(index)
        self.sub_tab_bar.setCurrentIndex(index)

    def _on_view_changed(self, view_type: str):
        """Handle view type change."""
        plot_widget = self.display_splitter.widget(0)
        table_widget = self.display_splitter.widget(1)

        if view_type == "Plot":
            plot_widget.setVisible(True)
            table_widget.setVisible(False)
        else:
            plot_widget.setVisible(False)
            table_widget.setVisible(True)

    def _setup_styles(self):
        # Styling comes from the central theme (gui/theme/app.qss); this method
        # is kept as a no-op seam for any results-specific tweak.
        pass

    def set_view_type(self, view_type: str):
        """Set the current view type (Plot or Table)."""
        index = self.view_combo.findText(view_type)
        if index >= 0:
            self.view_combo.setCurrentIndex(index)
            self._on_view_changed(view_type)

    def set_data_type(self, data_type: str):
        """Set the current data type."""
        index = self.data_combo.findText(data_type)
        if index >= 0:
            self.data_combo.setCurrentIndex(index)

    def update_results(self, results: dict):
        """Update summary from the normalized dict; render plot + table from the
        raw profile in window_state.results."""
        self.summary_label.setText(
            f"Status: {results.get('status', 'Unknown')}\n"
            f"Stages: {results.get('stages', '--')}\n"
            f"Iterations: {results.get('iterations', '--')}\n"
            f"Runtime: {results.get('runtime', '--')}"
        )
        self._sync_unit_combos()
        self._update_data_choices()
        self._fill_table()
        self._fill_stream_summary()
        self._fill_performance()
        self._draw_plot()

    def _sync_unit_combos(self):
        """Reflect window_state.display_units in the combos without re-rendering."""
        u = self._units()
        for combo, val in ((self.temp_unit_combo, u.temperature),
                           (self.flow_unit_combo, u.flow),
                           (self.duty_unit_combo, u.duty)):
            combo.blockSignals(True)
            combo.setCurrentText(val)
            combo.blockSignals(False)

    def _fill_stream_summary(self):
        """Populate the Streams sub-view from the raw profile."""
        prof = self._profile()
        if not prof or "D" not in prof:      # BVM-style profiles have no products
            self.stream_table.setRowCount(0)
            self.spec_table.setRowCount(0)
            self.duty_label.setText(
                "No product summary for this run." if prof else
                "Run a simulation to see stream results.")
            return
        summ = stream_summary(prof)
        comps = summ["comps"]
        u = self._units()
        species = getattr(self.window_state, "species", {}) or {}
        mws = [getattr(species.get(c), "mw", None) for c in comps]
        headers = ["Stream", f"Flow ({u.f_label()})", f"T ({u.t_label()})"] \
            + [f"x {c}" for c in comps]
        self.stream_table.setColumnCount(len(headers))
        self.stream_table.setHorizontalHeaderLabels(headers)
        self.stream_table.horizontalHeader().setStretchLastSection(False)
        self.stream_table.setColumnWidth(0, 150)
        prods = summ["products"]
        self.stream_table.setRowCount(len(prods))
        for r, p in enumerate(prods):
            mw_mix = (float(np.asarray(p["comp"]) @ np.asarray(mws, float))
                      if all(m is not None for m in mws) else None)
            flow = u.F(p["flow"], mw_mix) if u.flow == "kg/h" and mw_mix \
                else p["flow"]
            cells = [f"{p['name']} ({p['phase']})",
                     f"{flow:.3f}", f"{u.T(p['T']):.2f}"] \
                + [fmt(v) for v in p["comp"]]
            for c, v in enumerate(cells):
                self.stream_table.setItem(r, c, QTableWidgetItem(v))

        lines = []
        Qc, Qr = summ["condenser_duty"], summ["reboiler_duty"]
        if Qc is not None:
            lines.append(f"Condenser duty: {u.Q(Qc):>12.3f} {u.q_label()}")
        if Qr is not None:
            lines.append(f"Reboiler duty:  {u.Q(Qr):>12.3f} {u.q_label()}")
        lines.append(f"Mass-balance closure (max |feed-out|): "
                     f"{summ['closure_max']:.2e} kmol/h")
        self.duty_label.setText("\n".join(lines))

        self._fill_spec_table(prof)

    def _fill_spec_table(self, prof):
        """Implicit specs (purity / key recovery) target vs what the run hit."""
        ws = self.window_state
        rows = []
        if ws is not None:
            from core.dof import SpecKind
            comps = prof["comps"]
            feed = np.asarray(prof.get("feed_totals"))
            for s in getattr(ws, "specs", []):
                i = s.component
                if s.kind == SpecKind.DIST_PURITY and 0 <= i < len(comps):
                    rows.append((f"Distillate purity ({comps[i]})", s.value,
                                 float(prof["xD"][i])))
                elif s.kind == SpecKind.BOTTOMS_PURITY and 0 <= i < len(comps):
                    rows.append((f"Bottoms purity ({comps[i]})", s.value,
                                 float(prof["xB"][i])))
                elif s.kind in (SpecKind.LK_RECOVERY, SpecKind.HK_RECOVERY):
                    idx = ws.light_key_index if s.kind == SpecKind.LK_RECOVERY \
                        else ws.heavy_key_index
                    if feed is not None and 0 <= idx < len(comps) and feed[idx] > 0:
                        rec = float(prof["D"] * prof["xD"][idx] / feed[idx])
                        rows.append((f"{s.kind.value} ({comps[idx]})", s.value, rec))
        self.spec_table.setRowCount(len(rows))
        for r, (name, target, got) in enumerate(rows):
            for c, v in enumerate((name, fmt(target), fmt(got))):
                self.spec_table.setItem(r, c, QTableWidgetItem(v))

    def _profile(self):
        return getattr(self.window_state, "results", None) if self.window_state else None

    def _units(self):
        """Current DisplayUnits (from window_state, defaults if absent)."""
        from core.units import DisplayUnits
        u = getattr(self.window_state, "display_units", None) if self.window_state \
            else None
        return u or DisplayUnits()

    def _stage_mw(self, prof):
        """Per-stage mixture molar mass [kg/kmol] from species MW and x, or None
        if any component's MW is missing (kg/h flow then falls back to kmol/h)."""
        species = getattr(self.window_state, "species", {}) or {}
        mws = [getattr(species.get(c), "mw", None) for c in prof["comps"]]
        if any(m is None for m in mws):
            return None
        return np.asarray(prof["x"]) @ np.asarray(mws, float)

    def _on_units_changed(self):
        """Persist the unit choices to window_state and re-render everything."""
        u = self._units()
        u.temperature = self.temp_unit_combo.currentText()
        u.flow = self.flow_unit_combo.currentText()
        u.duty = self.duty_unit_combo.currentText()
        if self.window_state is not None:
            self.window_state.display_units = u
        if self._profile():
            self._fill_table()
            self._fill_stream_summary()
            self._fill_performance()
            self._draw_plot()

    def _data_available(self, dtype, prof):
        if prof is None:
            return False
        if dtype in ("Compositions", "Temperature"):
            return True
        if dtype == "Ternary Map":
            return len(prof.get("comps", [])) == 3
        if dtype == "McCabe-Thiele":
            return len(prof.get("comps", [])) == 2 and "R" in prof
        return _SERIES_KEYS.get(dtype) in prof

    def _update_data_choices(self):
        """Enable only the data types the current profile carries; a solver
        that can't produce a series greys it out instead of a placeholder."""
        prof = self._profile()
        model = self.data_combo.model()
        first_ok = None
        for i in range(self.data_combo.count()):
            dtype = self.data_combo.itemText(i)
            ok = self._data_available(dtype, prof)
            item = model.item(i)
            item.setEnabled(ok)
            if ok and first_ok is None:
                first_ok = i
        current = self.data_combo.currentText()
        if prof and not self._data_available(current, prof) and first_ok is not None:
            self.data_combo.setCurrentIndex(first_ok)

    def _fill_table(self):
        """One row per stage, stage 0 = distillate on the top row. Columns are
        built from the profile: T, every component, plus any per-stage series
        the solver produced."""
        prof = self._profile()
        if not prof:
            self.data_table.setRowCount(0)
            return
        comps, x, T = prof["comps"], prof["x"], np.asarray(prof["T"])
        u = self._units()
        # Flow series (kmol/h) honour the flow unit; kg/h needs a per-stage MW.
        mw = self._stage_mw(prof) if u.flow == "kg/h" else None
        extra = []
        for label, key in _SERIES_KEYS.items():
            if key not in prof:
                continue
            vals = np.asarray(prof[key], float)
            if label in ("Liquid Flow", "Vapor Flow"):
                if mw is not None:
                    vals = vals * mw
                extra.append((f"{label} ({u.f_label()})", vals))
            else:
                extra.append((label, vals))
        # T is in the vapour-pressure fit's unit (bundled Antoine/PLXANT: degC)
        headers = (["Stage", f"T ({u.t_label()})"] + [f"x {c}" for c in comps]
                   + [label for label, _ in extra])
        self.data_table.setColumnCount(len(headers))
        self.data_table.setHorizontalHeaderLabels(headers)
        n = prof["n_stages"]
        self.data_table.setRowCount(n)
        for i in range(n):
            vals = ([i, round(float(u.T(T[i])), 2)]
                    + [round(float(v), 4) for v in x[i]]
                    + [round(float(vals_[i]), 4) for _, vals_ in extra])
            # stage 0 (distillate) on the top row
            for c, v in enumerate(vals):
                self.data_table.setItem(i, c, QTableWidgetItem(str(v)))

    # ----------------------------------------------------------- plotting
    def _draw_plot(self):
        self.figure.clear()
        self._tern_ctx = None
        ax = self.figure.add_subplot(111)
        prof = self._profile()
        if not prof:
            ax.text(0.5, 0.5, "Run a simulation to see results",
                    ha="center", va="center", color="#888888")
            ax.axis("off")
            self.canvas.draw()
            return

        dtype = self.data_combo.currentText()
        if not self._data_available(dtype, prof):
            dtype = "Compositions"

        if dtype == "Ternary Map":
            self._draw_ternary(ax, prof)
            self.canvas.draw()
            return

        if dtype == "McCabe-Thiele":
            self._draw_mccabe(ax, prof)
            self.canvas.draw()
            return

        x, T, comps = prof["x"], np.asarray(prof["T"]), prof["comps"]
        u = self._units()
        N = np.arange(prof["n_stages"])          # 0-based, 0 = distillate
        if dtype == "Temperature":
            ax.plot(N, u.T(T), "-o", color=TEMP_C)
            ax.set_ylabel(f"Temperature ({u.t_label()})")
        elif dtype == "Compositions":
            for j, name in enumerate(comps):
                ax.plot(N, x[:, j], "-o", label=name)
            ax.set_ylabel("Liquid mole fraction x")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
        else:
            series = np.asarray(prof[_SERIES_KEYS[dtype]], float)
            if dtype in ("Liquid Flow", "Vapor Flow") and u.flow == "kg/h":
                mw = self._stage_mw(prof)
                if mw is not None:
                    series = series * mw
                ax.set_ylabel(f"{dtype} ({u.f_label()})")
            else:
                ax.set_ylabel(_SERIES_LABELS.get(dtype, dtype))
            ax.plot(N, series, "-o", color=DATA_C)

        self._draw_feed_lines(ax, prof)
        ax.set_xlabel("Stage (0 = distillate)")
        ax.set_xlim(0, prof["n_stages"] - 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.figure.tight_layout()
        self.canvas.draw()

    def _feed_stages(self, prof):
        """{stage: [stream ids]} for every feed that has a flow (GUI stage
        convention, 0 = distillate). Falls back to the profile's single
        feed_stage when there's no window_state to read streams from."""
        stages = {}
        ws = self.window_state
        if ws:
            # getattr: the state may be a partial/duck-typed view without
            # streams, which is exactly the "fall back to prof[feed_stage]"
            # case below rather than an error.
            for s in getattr(ws, "streams", {}).values():
                if s.stream_type == StreamType.FEED and s.flow and s.composition:
                    stages.setdefault(int(s.stage), []).append(s.id)
        if not stages and "feed_stage" in prof:
            stages[int(prof["feed_stage"])] = ["Feed"]
        return stages

    def _draw_feed_lines(self, ax, prof):
        """One dashed marker per feed stage, labelled with the feed id(s)."""
        for stage, ids in self._feed_stages(prof).items():
            if 0 <= stage < prof["n_stages"]:
                ax.axvline(stage, color="grey", ls="--", lw=1)
                ax.text(stage, 0.98, ", ".join(ids),
                        transform=ax.get_xaxis_transform(),
                        ha="right", va="top", fontsize=7, color="grey",
                        rotation=90)

    def _thermo_ctx(self):
        """(P, antoine, gamma_fn) in the Psat-fit unit, or None if the shared
        thermo setup is incomplete."""
        ws = self.window_state
        if not ws:
            return None
        try:
            order = ws.get_species_names()
            antoine = ws.thermodynamics_config.psat_params(order)
            P = ws.thermodynamics_config.pressure_in_psat_unit(ws.pressure)
            gamma_fn = ws.build_gamma_fn(order)
            return P, antoine, gamma_fn
        except Exception:
            return None

    def _draw_ternary(self, ax, prof):
        """Ternary view: residue-curve map + boundaries + singular points as
        background, the column's liquid profile on top. Click inside the
        triangle to draw a residue curve through that composition."""
        comps = prof["comps"]
        ternary_axes(ax, comps)
        ctx = self._thermo_ctx()
        if ctx is not None:
            P, antoine, gamma_fn = ctx
            self._tern_ctx = ctx
            try:
                curves = residue_curve_map(P, antoine, comps, gamma_fn,
                                           lines=6, n_it=80)
                for c in curves:
                    ax.plot(c[:, 0], c[:, 1], lw=0.8, color="#bbbbbb")
                pts = singular_points(P, antoine, comps, gamma_fn, grid=3)
                for p in pts:
                    ax.plot(p["x"][0], p["x"][1], "k^" if p["kind"] == "saddle"
                            else "ko", ms=5, mfc="none" if p["pure"] else "k")
                for b in distillation_boundaries(P, antoine, comps, gamma_fn,
                                                 points=pts):
                    ax.plot(b[:, 0], b[:, 1], color=BOUNDARY_C, lw=1.6)
            except Exception:
                pass                     # background is best-effort decoration
        x = prof["x"]
        ax.plot(x[:, 0], x[:, 1], "-o", color=DATA_C, ms=4,
                label="Column profile")
        ax.plot(x[0, 0], x[0, 1], "s", color=DATA_C, ms=8)     # top stage
        ax.plot(x[-1, 0], x[-1, 1], "v", color=TEMP_C, ms=8)   # bottoms
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Ternary map (click to add a residue curve)", fontsize=9)

    def _draw_mccabe(self, ax, prof):
        """McCabe-Thiele diagram (binary): equilibrium curve, 45° line,
        rectifying/stripping operating lines, q-line and the stepped stages."""
        ctx = self._thermo_ctx()
        if ctx is None:
            ax.text(0.5, 0.5, "Thermo setup incomplete for McCabe-Thiele",
                    ha="center", va="center", color="#888888")
            ax.axis("off"); return
        P, antoine, gamma_fn = ctx
        xe, ye = binary_equilibrium_curve(P, antoine, gamma_fn)
        feed = np.asarray(prof.get("feed_totals", [1.0, 1.0]))
        zF = float(feed[0] / feed.sum()) if feed.sum() > 0 else 0.5
        mt = mccabe_thiele_steps(xe, ye, float(prof["xD"][0]),
                                 float(prof["xB"][0]), zF,
                                 float(prof["R"]), float(prof.get("feed_q", 1.0)))
        ax.plot([0, 1], [0, 1], color="#888888", lw=1)          # 45° line
        ax.plot(xe, ye, color=DATA_C, lw=1.5, label="equilibrium")
        rect = np.array(mt["rect"]); strip = np.array(mt["strip"])
        qln = np.array(mt["qline"]); steps = np.array(mt["steps"])
        ax.plot(rect[:, 0], rect[:, 1], color=RECT_C, lw=1.4, label="rectifying")
        ax.plot(strip[:, 0], strip[:, 1], color=STRIP_C, lw=1.4, label="stripping")
        ax.plot(qln[:, 0], qln[:, 1], color="#555555", lw=1, ls="--", label="q-line")
        ax.plot(steps[:, 0], steps[:, 1], color="#333333", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        c0 = prof["comps"][0]
        ax.set_xlabel(f"x {c0}"); ax.set_ylabel(f"y {c0}")
        ax.set_title(f"McCabe-Thiele — {mt['n_stages']} theoretical stages",
                     fontsize=9)
        ax.legend(fontsize=7, loc="lower right")

    def _on_plot_click(self, event):
        """Click-to-plot (freeRCM behaviour): a click inside the ternary
        triangle draws the residue curve through that composition."""
        if self._tern_ctx is None or not event.inaxes:
            return
        x0 = composition_from_click(event)
        if x0 is None:
            return
        P, antoine, gamma_fn = self._tern_ctx
        try:
            c, _ = residue_curve(x0, P, antoine, gamma_fn, n_it=120)
        except Exception:
            return
        if len(c) > 1:
            event.inaxes.plot(c[:, 0], c[:, 1], lw=1.2)
            self.canvas.draw()

    def clear_results(self):
        """Clear all results."""
        self.summary_label.setText(
            "Status: Not Run\n"
            "Stages: --\n"
            "Iterations: --\n"
            "Runtime: --"
        )
        self.data_table.setRowCount(0)
        self._fill_stream_summary()
        self._fill_performance()
        self._draw_plot()
