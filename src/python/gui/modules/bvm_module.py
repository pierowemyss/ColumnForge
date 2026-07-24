"""BVM side module -- boundary-value column sizing & feasibility.

A GUI over `src/side_features/bvm`: the difference-point-chain design
method (MatBVM_blueprint.md, v4). Feed, pressure, species, keys and the thermo
model come from the shared window_state (same as the RCM module); the
BVM levers -- reflux R (single or swept), key recoveries, optional
entrainer E/F, and the connection/marching tolerances -- are entered here.

Stage count is an OUTPUT, not an input: the method marches profiles from each
product end until they connect and reports the stages required, the feed/draw
locations, R_min, and a classified feasibility verdict. Three actions:

  * Size Column   -- size at the chosen R; plot the profile, mark the feed stage.
  * Design Map    -- sweep R; plot stage count vs R and click a point to load it.
  * Send to Solver -- hand the sized profiles to the rigorous MESH solver (warm
                     start) and report its convergence.
"""

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QSpinBox, QPushButton, QCheckBox, QTableWidget, QTableWidgetItem,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas

from ..panels.sci_spin_box import SciDoubleSpinBox
from ..state.window_state import StreamType
from ..plotting import CompactNavigationToolbar, active_comps, TEMP_C as _TEMP_C

from side_features.bvm import api as _mbvm_api
from side_features.bvm.problem import build_problem
from side_features.bvm.thermo_adapter import ColumnForgeThermo


class BVMModuleWidget(QWidget):
    """Parameter panel + profile / design-map plot for a BVM run."""

    def __init__(self, window_state=None, parent=None):
        super().__init__(parent)
        self.window_state = window_state
        self._design = None
        self._map = None
        self._order_warning = ""
        self._entrainer_prefilled = False
        self._restored = False
        self._thread = self._worker = None
        self._setup_ui()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        layout = QHBoxLayout(self)

        left = QWidget(); left.setMaximumWidth(340)
        left_col = QVBoxLayout(left)

        spec = QGroupBox("Separation")
        form = QFormLayout(spec)
        self.lk_combo = QComboBox()
        self.hk_combo = QComboBox()
        form.addRow("Light key:", self.lk_combo)
        form.addRow("Heavy key:", self.hk_combo)
        self.rec_lk = self._spin(0.5, 0.99999, 0.98, decimals=4, step=0.01)
        self.rec_hk = self._spin(1e-5, 0.5, 0.02, decimals=4, step=0.01)
        form.addRow("LK recovery to distillate:", self.rec_lk)
        form.addRow("HK recovery to distillate:", self.rec_hk)
        left_col.addWidget(spec)

        op = QGroupBox("Operating point")
        opf = QFormLayout(op)
        self.r_spin = self._spin(0.05, 1000.0, 3.0, decimals=3, step=0.25)
        opf.addRow("Reflux ratio R:", self.r_spin)
        self.q_spin = self._spin(-1.0, 2.0, 1.0, decimals=3, step=0.1)
        self.q_spin.setToolTip("Feed thermal quality q: 1 = saturated liquid, "
                               "0 = saturated vapour.")
        opf.addRow("Feed quality q:", self.q_spin)
        eff0 = float(getattr(self.window_state, "stage_efficiency", 1.0) or 1.0)
        self.eff_spin = self._spin(0.05, 1.0, eff0, decimals=3, step=0.05)
        self.eff_spin.setToolTip("Murphree vapour efficiency (column-wide); 1 = "
                                 "ideal stages. Defaults to the shared column value.")
        opf.addRow("Stage efficiency:", self.eff_spin)
        self.rmax_spin = self._spin(0.1, 1000.0, 10.0, decimals=2, step=1.0)
        opf.addRow("Design-map R max:", self.rmax_spin)
        self.map_pts = self._int_spin(4, 60, 16)
        opf.addRow("Design-map points:", self.map_pts)
        self.extractive = QCheckBox("Extractive distillation")
        opf.addRow(self.extractive)
        self.entrainer_combo = QComboBox()
        opf.addRow("Entrainer:", self.entrainer_combo)
        self.ef_spin = self._spin(0.0, 20.0, 0.5, decimals=3, step=0.1)
        opf.addRow("Entrainer ratio E/F:", self.ef_spin)
        left_col.addWidget(op)

        adv = QGroupBox("Advanced"); adv.setCheckable(True); adv.setChecked(False)
        advf = QFormLayout(adv)
        self.max_stages = self._int_spin(20, 1000, 200)
        advf.addRow("Max stages / section:", self.max_stages)
        self.eps_stage = self._spin(1e-4, 0.2, 1e-2, decimals=4, step=1e-3)
        advf.addRow("Connection tol (eps_stage):", self.eps_stage)
        left_col.addWidget(adv)

        self.size_btn = QPushButton("Size Column"); self.size_btn.clicked.connect(self._on_size)
        self.limits_btn = QPushButton("Compute R_min / min E/F")
        self.limits_btn.setToolTip("Runs the reflux (and E/F) minimum bisection "
                                   "(~dozens of sizings); left off the Size button "
                                   "so sizing stays snappy.")
        self.limits_btn.clicked.connect(self._on_limits)
        self.map_btn = QPushButton("Design Map"); self.map_btn.clicked.connect(self._on_map)
        self.send_btn = QPushButton("Send to Rigorous Solver")
        self.send_btn.clicked.connect(self._on_send)
        for b in (self.size_btn, self.limits_btn, self.map_btn, self.send_btn):
            left_col.addWidget(b)

        self.extractive.toggled.connect(self._sync_extractive_enabled)
        self._sync_extractive_enabled(self.extractive.isChecked())

        self.status = QLabel("Feed, pressure and thermo come from the shared "
                             "column setup. Stage count is computed, not entered.")
        self.status.setWordWrap(True)
        left_col.addWidget(self.status)
        left_col.addStretch()
        layout.addWidget(left)

        right = QWidget(); right_col = QVBoxLayout(right)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
        right_col.addWidget(self.toolbar)
        right_col.addWidget(self.canvas, stretch=3)

        view_row = QHBoxLayout()
        view_row.addStretch()
        view_row.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Ternary (LK vs HK)", "Full profile"])
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        view_row.addWidget(self.view_combo)
        right_col.addLayout(view_row)

        self.data_table = QTableWidget(0, 2)
        self.data_table.setHorizontalHeaderLabels(["Stage", "T (degC)"])
        self.data_table.horizontalHeader().setStretchLastSection(True)
        right_col.addWidget(self.data_table, stretch=1)
        layout.addWidget(right, stretch=1)

        self._refresh_species()

    def _refresh_species(self):
        """(Re)populate the key/entrainer combos from the shared species list,
        preserving each selection by name across species edits upstream."""
        names = self._species_order()
        for combo, default in ((self.lk_combo, 0),
                               (self.hk_combo, 1),
                               (self.entrainer_combo, len(names) - 1)):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            i = combo.findText(prev)
            combo.setCurrentIndex(i if i >= 0 else min(max(default, 0),
                                                       len(names) - 1))
            combo.blockSignals(False)

    def showEvent(self, event):
        # species may have changed on the Initialization tab since last shown
        self._refresh_species()
        if not self._restored and self.window_state and \
                getattr(self.window_state, "bvm_params", None):
            self.set_params(self.window_state.bvm_params)
            self._restored = self._entrainer_prefilled = True   # don't clobber saved
        self._prefill_entrainer()
        super().showEvent(event)

    def _prefill_entrainer(self):
        """Auto-detect a second FEED stream as the entrainer and prefill extractive
        mode once (E/F from the flow ratio, entrainer species from its dominant
        component). Runs a single time so it never clobbers later user edits."""
        if self._entrainer_prefilled or not self.window_state:
            return
        main, ent = self._feed_streams()
        if ent is None or not main or not main.flow:
            return
        dom = max(ent.composition, key=ent.composition.get)
        self.extractive.setChecked(True)
        i = self.entrainer_combo.findText(dom)
        if i >= 0:
            self.entrainer_combo.setCurrentIndex(i)
        self.ef_spin.setValue(float(ent.flow) / float(main.flow))
        self._entrainer_prefilled = True
        self.status.setText(f"Detected entrainer stream '{ent.id}' "
                            f"({dom}, E/F={ent.flow / main.flow:.3g}); main feed "
                            f"'{main.id}'. Extractive mode prefilled.")

    @staticmethod
    def _spin(lo, hi, val, decimals=3, step=0.1):
        s = SciDoubleSpinBox(); s.setDecimals(decimals); s.setRange(lo, hi)
        s.setSingleStep(step); s.setValue(val); return s

    @staticmethod
    def _int_spin(lo, hi, val):
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); return s

    # ------------------------------------------------------- state -> problem
    def _species_order(self):
        return self.window_state.get_species_names() if self.window_state else []

    def _feed_streams(self):
        """(main_feed, entrainer_feed|None) from the shared streams.

        Extractive columns carry a second FEED stream (a near-pure solvent fed
        high in the column). The entrainer is the near-pure single-component FEED
        (max mole fraction >= 0.95), or one whose id names it; the main feed is
        the multicomponent one. With <2 feeds the entrainer is None.
        """
        feeds = [s for s in self.window_state.streams.values()
                 if s.stream_type == StreamType.FEED and s.composition]
        if not feeds:
            return None, None
        if len(feeds) == 1:
            return feeds[0], None

        def is_entrainer(s):
            return ("entrain" in s.id.lower()
                    or (s.composition and max(s.composition.values()) >= 0.95))
        ent = next((s for s in feeds if "entrain" in s.id.lower()), None) \
            or next((s for s in feeds if is_entrainer(s)), None)
        if ent is None:
            return feeds[0], None
        main = next((s for s in feeds if s is not ent), feeds[0])
        return main, ent

    def _feed_stream(self):
        return self._feed_streams()[0]

    def _gather(self):
        """Build (problem, provider) from window_state + local levers.

        Raises ValueError with a user-facing message when setup is incomplete.
        """
        if not self.window_state:
            raise ValueError("No column state available.")
        order = self._species_order()
        if len(order) < 2:
            raise ValueError("Need at least 2 species (Initialization tab).")
        feed, ent_stream = self._feed_streams()
        if feed is None or not feed.flow or not feed.composition:
            raise ValueError("Feed stream needs a flow rate and composition.")
        z = np.array([feed.composition.get(n, 0.0) for n in order], float)
        if abs(z.sum() - 1.0) > 1e-3:
            raise ValueError(f"Feed composition sums to {z.sum():.4f}, not 1.")

        lk, hk = self.lk_combo.currentIndex(), self.hk_combo.currentIndex()
        if lk < 0 or hk < 0 or lk == hk:
            raise ValueError("Light and heavy keys must be two distinct species.")

        antoine = self.window_state.thermodynamics_config.psat_params(order)
        P = self.window_state.thermodynamics_config.pressure_in_psat_unit(
            self.window_state.pressure)
        gamma_fn = self.window_state.build_gamma_fn(order)
        phi_fn = self.window_state.build_phi_fn(order)   # SRK EOS or None (ideal gas)
        provider = ColumnForgeThermo(antoine, gamma_fn=gamma_fn, phi_fn=phi_fn)
        self._order_warning = self._volatility_warning(order, antoine, P, provider)

        extractive = self.extractive.isChecked()
        x_E = None
        if extractive:
            if ent_stream is not None:      # real entrainer stream: use its comp
                x_E = np.array([ent_stream.composition.get(n, 0.0) for n in order],
                               float)
                if x_E.sum() > 0:
                    x_E = x_E / x_E.sum()
            else:                            # no stream: pure entrainer from combo
                e = self.entrainer_combo.currentIndex()
                if e < 0:
                    raise ValueError("Select an entrainer species for extractive mode.")
                if e in (lk, hk):
                    raise ValueError("Entrainer must differ from the light/heavy keys.")
                x_E = np.zeros(len(order)); x_E[e] = 1.0

        prob = build_problem(
            comps=order, feeds=[(z, float(feed.flow), float(self.q_spin.value()))],
            pressure=P, lk=lk, hk=hk,
            rec_lk=self.rec_lk.value(), rec_hk=self.rec_hk.value(),
            x_E=x_E, extractive=extractive,
            max_stages=int(self.max_stages.value()),
            efficiency=float(self.eff_spin.value()))
        self.window_state.bvm_params = self.get_params()   # mirror for save
        return prob, provider

    @staticmethod
    def _volatility_warning(order, antoine, P, provider):
        """E10: species order is supposed to run light -> heavy (the non-key split
        rule keys off index position). We validate rather than reorder (reordering
        would remix the key/entrainer combos): return a warning string when the
        pure-component bubble points are not ascending, else ''."""
        try:
            Tb = [float(provider.bubble_T(np.eye(len(order))[i], P))
                  for i in range(len(order))]
        except Exception:
            return ""
        bad = [order[i] for i in range(1, len(Tb)) if Tb[i] < Tb[i - 1] - 1e-6]
        if bad:
            return ("species not ordered light->heavy by boiling point "
                    f"({', '.join(bad)} out of order); non-key split may be wrong")
        return ""

    # ------------------------------------------------------------- actions
    def _sync_extractive_enabled(self, on):
        """G7: entrainer combo + E/F spin are only consumed in extractive mode, so
        grey them out otherwise (the 'consumed or visibly disabled' honesty rule)."""
        for w in (self.entrainer_combo, self.ef_spin):
            w.setEnabled(bool(on))

    # ------------------------------------------------------- .colx persistence
    _PARAM_SPINS = ("rec_lk", "rec_hk", "r_spin", "q_spin", "eff_spin",
                    "rmax_spin", "map_pts", "ef_spin", "max_stages", "eps_stage")

    def get_params(self) -> dict:
        """G8: flat snapshot of the BVM knobs for .colx persistence (mirrored
        into window_state.bvm_params)."""
        d = {k: getattr(self, k).value() for k in self._PARAM_SPINS}
        d["lk"] = self.lk_combo.currentText()
        d["hk"] = self.hk_combo.currentText()
        d["extractive"] = self.extractive.isChecked()
        d["entrainer"] = self.entrainer_combo.currentText()
        return d

    def set_params(self, params: dict):
        """Apply a get_params() snapshot; ignores missing/unknown keys."""
        if not params:
            return
        for k in self._PARAM_SPINS:
            if k in params:
                getattr(self, k).setValue(type(getattr(self, k).value())(params[k]))
        for combo, key in ((self.lk_combo, "lk"), (self.hk_combo, "hk"),
                           (self.entrainer_combo, "entrainer")):
            if params.get(key):
                i = combo.findText(params[key])
                if i >= 0:
                    combo.setCurrentIndex(i)
        if "extractive" in params:
            self.extractive.setChecked(bool(params["extractive"]))

    # --------------------------------------------------------- threaded runs
    def _run_bg(self, label, job, on_done):
        """Run `job` on a QThread, same worker the main sim uses. Everything
        that touches widgets must be read on the GUI thread and closed over
        before the job is handed off."""
        from PySide6.QtCore import QThread
        from ..solver_worker import SolverWorker

        if getattr(self, "_thread", None) is not None:
            return                       # ponytail: one BVM run at a time
        self.status.setText(f"{label} ...")
        buttons = (self.size_btn, self.limits_btn, self.map_btn, self.send_btn)
        for b in buttons:
            b.setEnabled(False)

        self._worker = SolverWorker(lambda report, cancel: job())
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(on_done)
        self._worker.failed.connect(
            lambda msg, tb, _user: self.status.setText(f"{label} failed: {msg}"))
        for sig in (self._worker.finished, self._worker.failed):
            sig.connect(self._thread.quit)
        # No deleteLater here: dropping the Python refs in _run_done is enough,
        # and a pending deleteLater on an already-freed worker segfaults.
        self._thread.finished.connect(lambda: self._run_done(buttons))
        self._thread.start()

    def _run_done(self, buttons):
        self._thread = self._worker = None
        for b in buttons:
            b.setEnabled(True)

    def _on_size(self, with_limits=False):
        try:
            prob, provider = self._gather()
        except Exception as exc:
            self.status.setText(f"Sizing failed: {exc}")
            return
        R = self.r_spin.value()
        EF = self.ef_spin.value() if self.extractive.isChecked() else None
        self._run_bg("Sizing",
                     lambda: _mbvm_api.size_column(prob, provider, R=R, EF=EF,
                                                   with_limits=with_limits),
                     self._on_size_done)

    def _on_size_done(self, design):
        self._design = design
        if design["feasible"]:
            self.status.setText(self._feasible_status(design))
            self._fill_table(design["column"])
        else:
            reasons = "; ".join(f"{f.cls}"
                                + (f" [{f.section}]" if f.section else "")
                                for f in design["findings"])
            conn = design.get("connection")
            gap = (f" -- closest approach {conn['dmin']:.3f} (need <= {conn['tol']:.3f})"
                   if conn else "")
            self.status.setText(
                f"Infeasible at R={self.r_spin.value():g}: {reasons}{gap}")
            self.data_table.setRowCount(0)
        self._plot_current()

    def _feasible_status(self, design):
        """G6: always report the stage count / feed stage; append R_min & min-E/F
        only when they were computed (the old one-line conditional swallowed the
        stage count whenever R_min was absent)."""
        msg = f"Feasible: {design['N_total']} stages, feed@{design['feed_stages']}"
        rmin, efmin = design.get("R_min"), design.get("EF_min")
        if rmin is not None:
            msg += f". R_min={rmin:.2f}"
        if efmin is not None:
            msg += f", min E/F={efmin:.2f}"
        if self._order_warning:
            msg += f"  [warning: {self._order_warning}]"
        return msg

    def _on_limits(self):
        """G11: run the R_min / min-E/F bisection on demand, not on every size."""
        self._on_size(with_limits=True)

    def _on_view_changed(self, *_):
        if self._design is not None:
            self._plot_current()

    def _on_map(self):
        try:
            prob, provider = self._gather()
        except Exception as exc:
            self.status.setText(f"Design map failed: {exc}")
            return
        R_grid = np.linspace(0.2, self.rmax_spin.value(), int(self.map_pts.value()))
        self._run_bg("Design map",
                     lambda: _mbvm_api.feasibility_map(prob, provider, R_grid=R_grid),
                     self._on_map_done)

    def _on_map_done(self, fm):
        self._map = fm
        self._plot_map(fm)
        n_feas = int(np.count_nonzero(fm["feasible"]))
        self.status.setText(f"Design map: {n_feas}/{len(fm['feasible'])} R values feasible. "
                            "Click a point to size that column.")

    def _on_send(self):
        if self._design is None or not self._design.get("feasible"):
            self.status.setText("Size a feasible column first, then send it.")
            return
        try:
            init = _mbvm_api.to_solver(self._design)
            from core.column_solvers import solve_bubble_point
            comps = self._species_order()
            feed = self._feed_stream()
            z = np.array([feed.composition.get(n, 0.0) for n in comps])
            F = float(feed.flow)
            psat = self.window_state.thermodynamics_config.psat_params(comps)
            gamma_fn = self.window_state.build_gamma_fn(comps)
            eff = float(self.eff_spin.value())
        except Exception as exc:
            self.status.setText(f"Handoff failed: {exc}")
            return
        self._run_bg(
            "Warm start -> rigorous solver",
            lambda: solve_bubble_point(
                z, F, psat, comps, N=init["n_stages"],
                feed_stage=init["feed_stage"], R=init["R"], D=init["D"],
                P=init["pressure"], gamma_fn=gamma_fn, efficiency=eff,
                x0=init["x0"], T0=init["T0"]),
            self._on_send_done)

    def _on_send_done(self, sol):
        ok = sol.get("found")
        self._push_results(sol)              # G5: feed the Results tab, not just a label
        self.status.setText(
            f"Warm start -> rigorous solver: {'converged' if ok else 'did not converge'} "
            f"in {sol['iterations']} iterations ({sol['message']}). "
            "Profile sent to the Results tab.")

    def _push_results(self, profile):
        """G5: store the rigorous-solver profile in WindowState.results and render it
        in the Results tab (same handoff as the Simulation tab), so the sized column's
        profiles/duties are inspectable -- not stranded behind a status label."""
        if not self.window_state:
            return
        self.window_state.results = profile
        mw = self.window()
        try:
            summary = mw._normalize_results(profile)
            mw.results_tab.update_results(summary)
            mw.tab_widget.setCurrentIndex(3)
        except AttributeError:
            pass                              # headless / standalone: state is enough

    # -------------------------------------------------------------- plotting
    _SECTION_LS = {"rectifying": "-", "stripping": "--", "intermediate": ":",
                   "extractive": "-."}

    def _plot_current(self):
        if self._design is None:
            return
        if self.view_combo.currentIndex() == 0:
            self._plot_ternary(self._design)
        else:
            self._plot_full(self._design)

    def _plot_ternary(self, design):
        """LK/HK projection (Sec 13) on the composition right-triangle: each
        section's marched profile in the (x_LK, x_HK) plane, with products, feed,
        and the closest-approach gap. Drawn feasible or not, so a below-R_min run
        shows how close it came. Styled like the freeRCM ternary: no border, the
        three triangle legs (hypotenuse x_LK + x_HK = 1), vertices labelled."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        lk, hk = design["lk"], design["hk"]
        comps = design["comps"]

        # composition triangle: keep the bottom/left axes (ticks + labels) as the
        # two legs, drop the surrounding box, draw the hypotenuse x_LK + x_HK = 1.
        ax.plot([0, 1], [1, 0], "k-", lw=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal")

        for name, prof in design.get("profiles", {}).items():
            X = np.asarray(prof["X"])
            # per-stage markers (G9): a one/two-point section is invisible as a
            # bare line, so draw the stages as dots on the section linestyle.
            ax.plot(X[:, lk], X[:, hk], self._SECTION_LS.get(name, "-"),
                    marker="o", ms=3, lw=1.5, label=name)
        for pt, mark, lbl in ((design["xD"], "^", "x_D"),
                              (design["xB"], "v", "x_B"),
                              (design["feed_z"], "s", "feed")):
            ax.plot(pt[lk], pt[hk], mark, ms=8, mfc="none", mec="0.2", label=lbl)
        # entrainer stage marker (G9): the extractive column's upper feed stage
        col = design.get("column")
        fs = design.get("feed_stages", [])
        if col is not None and len(fs) > 1:
            xe = np.asarray(col["x"])[fs[0]]
            ax.plot(xe[lk], xe[hk], "*", ms=13, mfc="gold", mec="0.2",
                    label="entrainer stage")
        conn = design.get("connection")
        if conn is not None:
            pA, pB = conn.get("pointA"), conn.get("pointB")
            if pA is not None:
                col = "tab:green" if conn["connected"] else "tab:red"
                ax.plot([pA[lk], pB[lk]], [pA[hk], pB[hk]], "-", color=col, lw=2)
                ax.plot(conn["point"][lk], conn["point"][hk], "o", color=col, ms=7)

        ax.set_xlabel(f"{comps[lk]} mole fraction (LK)")
        ax.set_ylabel(f"{comps[hk]} mole fraction (HK)")
        ok = design["feasible"]
        # for C>3 the crossing lives in R^(C-1); this is only its LK/HK shadow (Sec 7)
        proj = "LK/HK projection" if len(comps) > 3 else "Ternary"
        title = (f"{proj} -- {design['N_total']} stages"
                 if ok else
                 f"{proj} -- gap {conn['dmin']:.3f} "
                 f"(need <= {conn['tol']:.3f})" if conn else proj)
        ax.set_title(title)
        ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8, loc="upper right")
        self.figure.tight_layout(); self.canvas.draw()

    def _plot_full(self, design):
        """Composition + temperature vs stage. When feasible, the assembled
        column; when not, the marched section profiles vs their own stage index
        (so an infeasible run still shows what was marched)."""
        self.figure.clear()
        comps = design["comps"]
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        col = design.get("column")
        if col is not None:
            x, T = col["x"], col["T"]
            stages = np.arange(x.shape[0])
            for j in active_comps(x, comps)[0]:   # drop all-zero (unfed) comps
                ax1.plot(stages, x[:, j], "-o", ms=3, label=comps[j])
            ax2.plot(stages, T, "-o", ms=3, color=_TEMP_C)
            for fs in design["feed_stages"]:
                ax1.axvline(fs, color="0.5", ls="--", lw=1)
                ax2.axvline(fs, color="0.5", ls="--", lw=1)
            ax1.set_title(f"Profile -- {design['N_total']} stages")
        else:
            profiles = design.get("profiles", {})
            # keep comps present in any section (one keep-set so color-by-index j
            # stays consistent across sections); drop all-zero (unfed) comps
            allX = np.vstack([np.asarray(p["X"]) for p in profiles.values()]) \
                if profiles else np.zeros((1, len(comps)))
            keep = active_comps(allX, comps)[0]
            for name, prof in profiles.items():
                X = np.asarray(prof["X"]); st = np.arange(len(X))
                ls = self._SECTION_LS.get(name, "-")
                for j in keep:
                    ax1.plot(st, X[:, j], ls, lw=1.2,
                             color=f"C{j}", label=comps[j] if name.startswith("rect") else None)
                ax2.plot(st, prof["T"], ls, lw=1.2, label=name)
            ax2.legend(fontsize=8)
            ax1.set_title("Marched sections (did not connect)")
        ax1.set_xlabel("Stage (0 = distillate)"); ax1.set_ylabel("Liquid x")
        ax1.set_ylim(0, 1); ax1.legend(fontsize=8)
        ax2.set_xlabel("Stage (0 = distillate)"); ax2.set_ylabel("T (degC)")
        ax2.set_title("Temperature")
        self.figure.tight_layout(); self.canvas.draw()

    def _plot_map(self, fm):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        R = np.atleast_1d(fm["R"]); N = np.atleast_1d(fm["stages"])
        feas = np.atleast_1d(fm["feasible"])
        good = feas & (N > 0)
        if good.any():
            ax.plot(R[good], N[good], "-o", ms=5, color="tab:blue",
                    picker=6, label="feasible")
        if (~feas).any():
            ax.plot(R[~feas], np.zeros(np.count_nonzero(~feas)), "x",
                    color="tab:red", label="infeasible")
        # R_min marker: first feasible R
        if good.any():
            ax.axvline(R[good][0], color="0.4", ls="--", lw=1,
                       label=f"~R_min={R[good][0]:.2f}")
        ax.set_xlabel("Reflux ratio R"); ax.set_ylabel("Total stages")
        ax.set_title("Design map (click a feasible point to load)")
        ax.legend(fontsize=8)
        self.figure.tight_layout(); self.canvas.draw()

    def _on_pick(self, event):
        if self._map is None:
            return
        xdata = event.artist.get_xdata()
        R = float(xdata[event.ind[0]])
        self.r_spin.setValue(R)
        self._on_size()

    def _fill_table(self, col):
        """Per-stage table: T, liquid x and vapour y for every species, plus the
        section liquid/vapour molar flows L, V (G10)."""
        comps = self._species_order()
        x, T = np.asarray(col["x"]), np.asarray(col["T"])
        y = np.asarray(col.get("y")) if col.get("y") is not None else None
        L = np.asarray(col.get("liquid_flow")) if col.get("liquid_flow") is not None else None
        V = np.asarray(col.get("vapor_flow")) if col.get("vapor_flow") is not None else None
        headers = ["Stage", "T (degC)"] + [f"x {c}" for c in comps]
        if y is not None:
            headers += [f"y {c}" for c in comps]
        if L is not None:
            headers.append("L (kmol/h)")
        if V is not None:
            headers.append("V (kmol/h)")
        self.data_table.setColumnCount(len(headers))
        self.data_table.setHorizontalHeaderLabels(headers)
        self.data_table.setRowCount(x.shape[0])
        for r in range(x.shape[0]):
            row = [r, round(float(T[r]), 2)] + [round(float(v), 4) for v in x[r]]
            if y is not None:
                row += [round(float(v), 4) for v in y[r]]
            if L is not None:
                row.append(round(float(L[r]), 2))
            if V is not None:
                row.append(round(float(V[r]), 2))
            for c, v in enumerate(row):
                self.data_table.setItem(r, c, QTableWidgetItem(str(v)))


def _demo():
    """Headless self-check: drive gather + size off a stub state, no event loop."""
    import sys, os
    _src_py = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # src/python
    sys.path.insert(0, _src_py)
    sys.path.insert(0, os.path.dirname(_src_py))                           # src (for side_features)
    from PySide6.QtWidgets import QApplication
    from gui.state.window_state import WindowState, Species, Stream, StreamType

    app = QApplication.instance() or QApplication([])

    def wait(w, timeout=120.0):
        """Sizing now runs on a QThread; pump the loop until it lands."""
        import time
        t0 = time.monotonic()
        while w._thread is not None and time.monotonic() - t0 < timeout:
            app.processEvents()
        assert w._thread is None, "BVM run did not finish"

    ws = WindowState()
    ws.pressure = 1.01325
    ws.light_key_index = 0
    abc = [(6.90565, 1211.033, 220.79), (6.95464, 1344.8, 219.48),
           (6.99052, 1453.43, 215.31)]
    for name, (a, b, c) in zip(["benzene", "toluene", "xylene"], abc):
        ws.add_species(Species(name=name))
        p = ws.thermodynamics_config.get_component_params(name)
        p.antoine_a, p.antoine_b, p.antoine_c = a, b, c
    ws.add_stream(Stream(id="Feed", stream_type=StreamType.FEED, stage=8,
                         flow=100.0, composition={"benzene": 0.4, "toluene": 0.35,
                                                  "xylene": 0.25}))
    w = BVMModuleWidget(window_state=ws)
    w._refresh_species()                       # showEvent won't fire headlessly
    assert [w.lk_combo.itemText(i) for i in range(w.lk_combo.count())] == \
        ["benzene", "toluene", "xylene"], "combos populated by species name"
    w.lk_combo.setCurrentText("benzene"); w.hk_combo.setCurrentText("toluene")
    w.r_spin.setValue(4.0)
    prob, provider = w._gather()
    assert prob.C == 3 and prob.lk == 0 and prob.hk == 1

    w._on_size(); wait(w)
    assert w._design is not None and w._design["feasible"], w.status.text()
    assert "Feasible:" in w.status.text() and "stages" in w.status.text(), w.status.text()
    assert w.data_table.rowCount() == w._design["N_total"]
    assert w.data_table.item(0, 0).text() == "0", "distillate on the top row"
    # G10: table carries x, y and L/V columns (2 + 3 x-cols + 3 y-cols + L + V)
    hdrs = [w.data_table.horizontalHeaderItem(c).text()
            for c in range(w.data_table.columnCount())]
    assert any(h.startswith("y ") for h in hdrs) and "L (kmol/h)" in hdrs, hdrs
    xD, xB = w._design["xD"], w._design["xB"]
    assert xD[0] > 0.4 > xB[0], (xD, xB)
    # G7: entrainer widgets are greyed while extractive is off, live while on
    assert not w.ef_spin.isEnabled(), "E/F greyed when not extractive"
    w.extractive.setChecked(True); assert w.ef_spin.isEnabled()
    w.extractive.setChecked(False)
    # default view is ternary; both views render without error
    assert w.view_combo.currentIndex() == 0, "ternary is the default view"
    w.view_combo.setCurrentIndex(1); w.view_combo.setCurrentIndex(0)

    # below R_min: still plots the marched profiles + the closest-approach gap
    w.r_spin.setValue(0.3)
    w._on_size(); wait(w)
    assert not w._design["feasible"], "R=0.3 should be below R_min"
    assert w._design["profiles"] and w._design["connection"] is not None
    assert "closest approach" in w.status.text(), w.status.text()
    w.view_combo.setCurrentIndex(1); w.view_combo.setCurrentIndex(0)  # both render
    w.r_spin.setValue(4.0); w._on_size(); wait(w)   # restore a feasible design

    w._on_map(); wait(w)
    assert w._map is not None and np.count_nonzero(w._map["feasible"]) > 0

    w._on_send(); wait(w)
    assert "rigorous solver" in w.status.text(), w.status.text()

    # second FEED stream -> auto-detected entrainer, extractive prefilled from flows
    ws.add_stream(Stream(id="Entrainer", stream_type=StreamType.FEED, stage=2,
                         flow=50.0, composition={"benzene": 0.0, "toluene": 0.0,
                                                 "xylene": 1.0}))
    main, ent = w._feed_streams()
    assert main.id == "Feed" and ent.id == "Entrainer", (main, ent)

    # G8: knobs round-trip through get_params/set_params (mirrored to window_state)
    w.r_spin.setValue(7.5); w.q_spin.setValue(0.6); w.hk_combo.setCurrentText("xylene")
    snap = w.get_params()
    assert snap["r_spin"] == 7.5 and snap["hk"] == "xylene", snap
    w2p = BVMModuleWidget(window_state=ws); w2p._refresh_species()
    w2p.set_params(snap)
    assert w2p.r_spin.value() == 7.5 and w2p.hk_combo.currentText() == "xylene"
    w2 = BVMModuleWidget(window_state=ws)
    w2._refresh_species(); w2._prefill_entrainer()
    assert w2.extractive.isChecked(), "extractive prefilled from second feed"
    assert abs(w2.ef_spin.value() - 0.5) < 1e-9, w2.ef_spin.value()  # 50/100
    assert w2.entrainer_combo.currentText() == "xylene", w2.entrainer_combo.currentText()
    print(f"bvm_module self-check OK  N={w._design['N_total']} "
          f"feed@{w._design['feed_stages']} xD={np.round(xD, 3)}  "
          f"entrainer-detect OK (E/F={w2.ef_spin.value():.2f})")


if __name__ == "__main__":
    _demo()
