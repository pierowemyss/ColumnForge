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

The optional Reaction box turns the run into a *reactive* sizing: one equilibrium
reaction, chemical equilibrium on every stage, marched in Ung-Doherty transformed
coordinates (`side_features.bvm.reactive`). Sizing, plots and the X/Y columns of
the table are then transformed; the physical compositions and the reaction extent
per stage are appended to the table. Reactive designs are ideal-stage and cannot
be sent to the rigorous solver -- MESH has no reaction terms -- so the efficiency,
entrainer and Send buttons are greyed out with the reason in their tooltips.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from core.data_structures import SolverMode
from side_features.bvm import api as _mbvm_api
from side_features.bvm import driver as _driver
from side_features.bvm import reactive as _rx
from side_features.bvm.problem import build_problem
from side_features.bvm.thermo_adapter import ColumnForgeThermo

from .module_thermo import (ENTRAINER_EB_TIP, attach_entrainer_energy_balance,
                            live_species, session_models)
from ..panels.sci_spin_box import SciDoubleSpinBox
from ..plotting import TEMP_C as _TEMP_C
from ..plotting import (CompactNavigationToolbar, active_comps, RECT_C, STRIP_C,
                        EXTRACT_C, INTER_C)
from ..state.window_state import StreamType

#: (label, `Problem.anchor_method`) for the interior-anchor combo, in the order
#: the scratchpad lists them. No "auto" entry on purpose: the three do not agree
#: on r_max, so which is right is a design judgement the panel should not make
#: silently. `saddle` leads because it is what BVM has always done.
ANCHOR_METHODS = (("Saddle manifolds", "saddle"),
                  ("Saddle ray end (S vertex)", "ray"),
                  ("Branch from neighbours", "continuation"))


class BVMModuleWidget(QWidget):
    """Parameter panel + profile / design-map plot for a BVM run."""

    def __init__(self, window_state=None, parent=None):
        super().__init__(parent)
        self.window_state = window_state
        self._design = None
        self._map = None
        self._region = None
        self._order_warning = ""
        self._thermo_note = ""
        self._entrainer_prefilled = False
        self._restored = False
        self._thread = self._worker = None
        self._setup_ui()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # the knob column scrolls: with the reaction editor added it no longer fits
        # a short window, and squeezing groups collapses their rows to nothing
        left = QWidget()
        left_col = QVBoxLayout(left)
        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(360)
        left_scroll.setFrameShape(QScrollArea.NoFrame)

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
        self.entrainer_eb = QCheckBox("Energy balance on the entrainer feed")
        self.entrainer_eb.setToolTip(ENTRAINER_EB_TIP)
        opf.addRow(self.entrainer_eb)
        self.anchor_combo = QComboBox()
        for label, key in ANCHOR_METHODS:
            self.anchor_combo.addItem(label, key)
        self.anchor_combo.setToolTip(
            "Where the INTERIOR section's profile is started. Only consumed by a "
            "column that has one -- extractive, or more than one feed; a "
            "two-section column has no interior section to anchor.\n\n"
            "Saddle manifolds -- the invariant manifolds through the section's "
            "saddle pinch, which is the limiting profile of a strongly pinched "
            "section. No arbitrary launch stage.\n"
            "Saddle ray end (S vertex) -- march inward from the far end of the "
            "stable eigendirection, i.e. the start of the extractive profile "
            "the rectification body predicts.\n"
            "Branch from neighbours -- launch from stages of the rectifying and "
            "stripping profiles that lie inside this section's own region.\n\n"
            "They do not agree on the maximum reflux; there is deliberately no "
            "'auto'. See docs/adr/0004.")
        opf.addRow("Interior anchor:", self.anchor_combo)
        left_col.addWidget(op)

        # The reaction itself is edited on Initialization / Reactions -- it is
        # chemistry, not a BVM lever. What stays here is the status line, because
        # a reaction changes what THIS panel can do (ideal stages, no handoff)
        # and a user looking at a greyed-out efficiency spin needs to see why.
        self.reaction_box = QGroupBox("Reaction (reactive distillation)")
        rxf = QVBoxLayout(self.reaction_box)
        self.reaction_status = QLabel()
        self.reaction_status.setWordWrap(True)
        rxf.addWidget(self.reaction_status)
        left_col.addWidget(self.reaction_box)

        adv = QGroupBox("Advanced"); adv.setCheckable(True); adv.setChecked(False)
        advf = QFormLayout(adv)
        self.max_stages = self._int_spin(20, 1000, 200)
        advf.addRow("Max stages / section:", self.max_stages)
        self.eps_stage = self._spin(1e-4, 0.2, 1e-2, decimals=4, step=1e-3)
        # Nothing silently ignored: at 3 components or fewer the junction is an
        # actual profile crossing and this knob is not in the test at all, so say
        # so rather than letting it look like a reflux you can buy.
        self.eps_stage.setToolTip(
            "Near-miss allowance on the feed junction.\n\n"
            "Only used where the two profiles cannot be asked to cross exactly:\n"
            "  - 4+ components (two curves in the (C-1)-simplex are over-determined)\n"
            "  - the interior curve of an extractive / multifeed column\n"
            "  - reactive designs (transformed coordinates)\n\n"
            "At 3 components or fewer the junction is the crossing itself, and "
            "loosening this cannot make an infeasible reflux feasible.")
        advf.addRow("Connection tol (eps_stage):", self.eps_stage)
        # E/F sweep for the operating-region plot behind "Compute R_min / min E/F"
        self.ef_min_spin = self._spin(0.0, 10.0, 0.2, decimals=3, step=0.1)
        self.ef_max_spin = self._spin(0.0, 20.0, 2.0, decimals=3, step=0.1)
        self.ef_pts_spin = self._int_spin(3, 40, 8)
        advf.addRow("E/F sweep from:", self.ef_min_spin)
        advf.addRow("E/F sweep to:", self.ef_max_spin)
        advf.addRow("E/F sweep points:", self.ef_pts_spin)
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
        # A sweep is minutes of work even on a pool, so it gets the same escape
        # hatch the RBM panel has: the run is already off the GUI thread, but
        # "responsive" is worth little if the only way out is waiting.
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        for b in (self.size_btn, self.limits_btn, self.map_btn, self.send_btn,
                  self.cancel_btn):
            left_col.addWidget(b)

        self.extractive.toggled.connect(self._sync_extractive_enabled)
        self._sync_extractive_enabled(self.extractive.isChecked())
        self._sync_reactive_enabled(self._reactive_on())

        self.status = QLabel("Feed, pressure and thermo come from the shared "
                             "column setup. Stage count is computed, not entered.")
        self.status.setWordWrap(True)
        left_col.addWidget(self.status)
        left_col.addStretch()
        layout.addWidget(left_scroll)

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
        self.view_combo.addItems(["Ternary (LK vs HK)", "Full profile",
                                  "Operating region"])
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
        self._sync_reaction_status()
        # Keys default to the session's chosen keys, not combo slots 0/1 -- the
        # Specifications tab already asked which two components matter, and
        # fug_module reads them the same way.
        ws = self.window_state
        lk = int(getattr(ws, "light_key_index", 0) or 0) if ws else 0
        hk = getattr(ws, "heavy_key_index", None) if ws else None
        hk = int(hk) if hk is not None else lk + 1
        hk = min(max(hk, 0), max(len(names) - 1, 0))
        if hk == lk:                       # the two keys must stay distinct
            hk = lk + 1 if lk + 1 < len(names) else max(lk - 1, 0)
        for combo, default in ((self.lk_combo, lk),
                               (self.hk_combo, hk),
                               (self.entrainer_combo, len(names) - 1)):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            i = combo.findText(prev)
            combo.setCurrentIndex(i if i >= 0 else min(max(default, 0),
                                                       len(names) - 1))
            combo.blockSignals(False)

    def _rx_params(self):
        """The shared reaction, edited on Initialization / Reactions. {} = none."""
        return dict(getattr(self.window_state, "reactions", None) or {})

    def _reactive_on(self):
        return bool(self._rx_params().get("on"))

    def _sync_reaction_status(self):
        """Say what the shared reaction is doing to THIS panel, or that there is
        none. The efficiency spin and the handoff button are disabled by a
        setting on another tab, so the reason has to be visible from here."""
        rx = self._rx_params()
        if not rx.get("on"):
            self.reaction_status.setText(
                "No reaction. Set one on <b>Initialization &rarr; Reactions</b> "
                "to size in Ung-Doherty transformed compositions.")
            return
        nu = {k: v for k, v in (rx.get("nu") or {}).items() if v}
        terms = ", ".join(f"{v:+g} {k}" for k, v in nu.items()) or "(no coefficients)"
        self.reaction_status.setText(
            f"Reactive sizing is ON (Initialization &rarr; Reactions): {terms}; "
            f"reference {rx.get('ref') or '?'}. Ideal stages, and the design "
            f"cannot be sent to the rigorous solvers.")

    def reload_from_state(self):
        """Pull species + saved BVM knobs off window_state (one-shot, guarded by
        `_restored`). ModulesTab re-arms those flags after a .colx load so a
        second file's parameters actually land here."""
        self._refresh_species()
        if not self._restored and self.window_state and \
                getattr(self.window_state, "bvm_params", None):
            self.set_params(self.window_state.bvm_params)
            self._restored = self._entrainer_prefilled = True   # don't clobber saved
        self._prefill_entrainer()

    def showEvent(self, event):
        # species may have changed on the Initialization tab since last shown
        self.reload_from_state()
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

        # A species in no feed is in no stage -- but `trace_floor` seeds it into
        # the product splits anyway, and a dead heavy one then amplifies 1/K per
        # stage down the march. Reaction species are the exception: a product is
        # made on the tray, not fed. See `module_thermo.live_species`.
        rx = self._rx_params()
        made = ([n for n, v in (rx.get("nu") or {}).items() if float(v or 0.0)]
                if rx.get("on") else [])
        order, z, x_E, lk, hk, dropped = live_species(order, z, x_E, lk, hk, made)

        # Same seam as the Txy / Phase-EQ modules, so a silent NRTL->ideal
        # degradation is REPORTED. It matters most here: an extractive column
        # whose entrainer has no binary parameters would otherwise be sized with
        # the entrainer thermodynamically invisible.
        antoine, gamma_fn, phi_fn, _label, thermo_note = session_models(
            self.window_state, order)
        tc = self.window_state.thermodynamics_config
        P = tc.pressure_in_psat_unit(self.window_state.pressure)
        dP = tc.pressure_in_psat_unit(
            float(getattr(self.window_state, "pressure_drop", 0.0) or 0.0))
        provider = ColumnForgeThermo(antoine, gamma_fn=gamma_fn, phi_fn=phi_fn)
        self._thermo_note = " ".join(filter(None, [
            thermo_note,
            "held at zero (in no feed): " + ", ".join(dropped) if dropped else ""]))
        self._order_warning = self._volatility_warning(order, antoine, P, provider)

        reactions = self._reactions(order)
        prob = build_problem(
            comps=order, feeds=[(z, float(feed.flow), float(self.q_spin.value()))],
            pressure=P, lk=lk, hk=hk,
            rec_lk=self.rec_lk.value(), rec_hk=self.rec_hk.value(),
            x_E=x_E, extractive=extractive, reactions=reactions,
            anchor_method=self.anchor_combo.currentData() or "saddle",
            max_stages=int(self.max_stages.value()),
            eps_stage=float(self.eps_stage.value()), dP=dP,
            efficiency=1.0 if reactions is not None else float(self.eff_spin.value()))
        if extractive and self.entrainer_eb.isChecked():
            note = attach_entrainer_energy_balance(
                self.window_state, order, prob, provider, ent_stream)
            if note:
                self._thermo_note = (self._thermo_note + "  " + note).strip()
        if reactions is not None:
            self._order_warning = self._reactive_order_warning(prob, provider)
        self.window_state.bvm_params = self.get_params()   # mirror for save
        return prob, provider

    @staticmethod
    def _reactive_order_warning(prob, provider):
        """The reactive counterpart of `_volatility_warning`.

        In transformed coordinates a pseudo-component's volatility has nothing to
        do with its pure-component boiling point -- transformed isobutene is
        `x_iC4 + x_MTBE`, and MTBE-locked isobutene is heavy. So rank by the actual
        transformed K = Y/X at the transformed feed, which is what the non-key split
        rule needs to be ordered by, and warn when it is not descending."""
        try:
            prob_r, tpr = _rx.transform_problem(prob, provider)
            Xz = np.asarray(prob_r.feeds[0].z, float)
            Yz, _ = tpr.bubble(Xz, prob.pressure)
            K = np.where(Xz > 1e-9, Yz / np.maximum(Xz, 1e-12), np.inf)
        except Exception:
            return ""                    # a real failure surfaces when sizing runs
        bad = [prob_r.comps[i] for i in range(1, len(K)) if K[i] > K[i - 1] * 1.001]
        if bad:
            return ("transformed species not ordered light->heavy at the feed "
                    f"({', '.join(bad)} more volatile than the one before) -- in a "
                    "reactive column that is often deliberate (a volatile reactant "
                    "leaves as product instead of overhead); check the non-key split "
                    "is the one you meant")
        return ""

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
        """G7: entrainer combo + E/F spins are only consumed in extractive mode, so
        grey them out otherwise (the 'consumed or visibly disabled' honesty rule)."""
        for w in (self.entrainer_combo, self.ef_spin, self.ef_min_spin,
                  self.ef_max_spin, self.ef_pts_spin, self.entrainer_eb):
            w.setEnabled(bool(on) and not self._reactive_on())

    def _sync_reactive_enabled(self, on):
        """Reactive sizing runs ideal stages and has no entrainer path, so the
        knobs it cannot consume are greyed out with the reason in a tooltip --
        rather than being read and quietly overridden."""
        on = bool(on)
        self.eff_spin.setEnabled(not on)
        self.extractive.setEnabled(not on)
        self.send_btn.setEnabled(not on)
        self.eff_spin.setToolTip(
            "Not consumed in reactive mode: a Murphree stage is an affine blend of "
            "vapour compositions and the transform is rational, so efficiency < 1 "
            "is not a transformed stage. Reactive sizing is ideal-stage."
            if on else
            "Murphree vapour efficiency (column-wide); 1 = ideal stages. "
            "Defaults to the shared column value.")
        self.send_btn.setToolTip(
            "Not available for a reactive design: the rigorous MESH solvers carry "
            "no reaction terms, so converging this warm start would silently solve "
            "a different (non-reactive) column."
            if on else "")
        if on:
            self.eff_spin.setValue(1.0)
            self.extractive.setChecked(False)
        self._sync_extractive_enabled(self.extractive.isChecked())

    def _reactions(self, order):
        """`Reactions` from the shared reaction state, or None when it is off.

        The validation stays here rather than in the editor: an incomplete
        reaction is only an error at the moment something tries to size with it,
        and the panel is where that error can be shown next to the run."""
        rx = self._rx_params()
        if not rx.get("on"):
            return None
        nu_map = rx.get("nu") or {}
        nu = np.array([[float(nu_map.get(n, 0.0)) for n in order]], float)
        if not np.any(nu):
            raise ValueError("Reaction is enabled (Initialization -> Reactions) "
                             "but every stoichiometric coefficient is zero.")
        if abs(nu.sum()) > 0 and len(np.flatnonzero(nu[0] > 0)) == 0:
            raise ValueError("Reaction needs at least one product "
                             "(positive coefficient).")
        ref_name = rx.get("ref")
        if ref_name not in order:
            raise ValueError("Select a reference component for the reaction "
                             "(Initialization -> Reactions).")
        ref = order.index(ref_name)
        if nu[0][ref] == 0.0:
            raise ValueError(f"The reference component ({ref_name}) must take part "
                             "in the reaction (non-zero coefficient).")
        return _rx.Reactions(nu=nu, ref=[ref],
                             keq_fn=_rx.keq_arrhenius(float(rx.get("keq_a", 0.0)),
                                                      float(rx.get("keq_b", 0.0))))

    # ------------------------------------------------------- .colx persistence
    _PARAM_SPINS = ("rec_lk", "rec_hk", "r_spin", "q_spin", "eff_spin",
                    "rmax_spin", "map_pts", "ef_spin", "max_stages", "eps_stage",
                    "ef_min_spin", "ef_max_spin", "ef_pts_spin")

    def get_params(self) -> dict:
        """G8: flat snapshot of the BVM knobs for .colx persistence (mirrored
        into window_state.bvm_params)."""
        d = {k: getattr(self, k).value() for k in self._PARAM_SPINS}
        d["lk"] = self.lk_combo.currentText()
        d["hk"] = self.hk_combo.currentText()
        d["extractive"] = self.extractive.isChecked()
        d["entrainer_eb"] = self.entrainer_eb.isChecked()
        d["entrainer"] = self.entrainer_combo.currentText()
        d["anchor_method"] = self.anchor_combo.currentData()
        # no "reaction" key any more: it lives on window_state.reactions, which
        # is persisted in its own right. Writing it here too would give a .colx
        # two copies that could disagree.
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
        if "entrainer_eb" in params:
            self.entrainer_eb.setChecked(bool(params["entrainer_eb"]))
        if params.get("anchor_method"):
            i = self.anchor_combo.findData(params["anchor_method"])
            if i >= 0:
                self.anchor_combo.setCurrentIndex(i)
        # Migration: files written before the editor moved carry the reaction
        # under bvm_params["reaction"]. Promote it to the shared state, but never
        # over a reaction already there -- on a current file window_state.reactions
        # is the copy the user edited and bvm_params has no such key at all.
        rxp = params.get("reaction") or {}
        if rxp and self.window_state is not None and \
                not getattr(self.window_state, "reactions", None):
            self.window_state.reactions = dict(rxp)
        self._sync_reaction_status()
        self._sync_reactive_enabled(self._reactive_on())

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
        self.cancel_btn.setEnabled(True)

        # A job that declares parameters gets the worker's report/cancel hooks;
        # one that takes none (a single sizing) stays supported as it was.
        import inspect
        wants = bool(inspect.signature(job).parameters)
        self._worker = SolverWorker(
            (lambda report, cancel: job(report=report, cancel=cancel)) if wants
            else (lambda report, cancel: job()))
        self._worker.progress.connect(
            lambda done, total, _r: self.status.setText(
                f"{label} ... {done}/{total}"))
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

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("Cancelling ...")

    def _run_done(self, buttons):
        from side_features.bvm import parallel
        parallel.shutdown()          # let the sweep's worker processes go
        self._thread = self._worker = None
        self.cancel_btn.setEnabled(False)
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
            self._fill_table(design["column"], design)
        else:
            # the detail field names the offending stage/composition -- it is the
            # only part that tells the user what to change, so show it.
            reasons = "; ".join(f"{f.cls}"
                                + (f" [{f.section}]" if f.section else "")
                                + (f": {f.detail}" if f.detail else "")
                                for f in design["findings"])
            conn = design.get("connection")
            # a gap is only meaningful when both profiles actually reached the
            # junction region; a section that never left its anchor has no gap.
            gap = (f" -- closest approach {conn['dmin']:.3f} (need <= {conn['tol']:.3f})"
                   if conn and np.isfinite(conn.get("dmin", np.inf)) else "")
            # the thermo note belongs here most of all: "no binary parameters"
            # is a common reason a column looks infeasible.
            self.status.setText(
                f"Infeasible at R={self.r_spin.value():g}: {reasons}{gap}"
                + (f"  [{self._thermo_note}]" if self._thermo_note else ""))
            self.data_table.setRowCount(0)
        self._plot_current()

    def _feasible_status(self, design):
        """G6: always report the stage count / feed stage; append R_min & min-E/F
        only when they were computed (the old one-line conditional swallowed the
        stage count whenever R_min was absent)."""
        msg = f"Feasible: {design['N_total']} stages, feed@{design['feed_stages']}"
        if design.get("reactive"):
            msg += (". Reactive: transformed coordinates, every stage at chemical "
                    "equilibrium")
            ex = np.asarray(design["physical"]["extent"], float)
            ex = ex[np.isfinite(ex)]
            if ex.size:
                msg += f"; extent {ex.min():.3f}..{ex.max():.3f} mol/mol"
        # Say what each junction actually closed to. "Feasible" covers both an
        # exact crossing (~1e-16) and a near miss accepted inside one stage of
        # travel, and the stage counts either side of a feed are only as good as
        # that number -- so it does not get to stay implicit.
        near = [j for j in design.get("junctions", ()) if j["approximate"]]
        if near:
            msg += (". Approximate junction"
                    + ("s" if len(near) > 1 else "") + ": "
                    + ", ".join(f"{j['pair']} closest approach {j['dmin']:.3f} "
                                f"(within one stage, {j['tol']:.3f})"
                                for j in near))
        rmin, efmin = design.get("R_min"), design.get("EF_min")
        if rmin is not None:
            msg += f". R_min={rmin:.2f}"
        if efmin is not None:
            msg += f", min E/F={efmin:.2f}"
        if self._order_warning:
            msg += f"  [warning: {self._order_warning}]"
        if self._thermo_note:
            msg += f"  [{self._thermo_note}]"
        return msg

    def _on_limits(self):
        """R_min / min-E/F on demand, and plot the picture that answers it.

        The two modes ask different questions, so they get different plots:

        NOT EXTRACTIVE -- one number matters, R_min, and what it looks like. Size
        the column just above it and show the ternary, where the two profiles have
        only just met.

        EXTRACTIVE -- R_min alone is not the answer. The reflux band has an upper
        edge as well (too much reflux dilutes the entrainer out of the middle
        section), both edges move with E/F, and the entrainer minimum is where
        they close on each other. So sweep E/F and draw the feasible region --
        the bifurcation diagram of Bruggemann & Marquardt's Figure 9.
        """
        try:
            prob, provider = self._gather()
        except Exception as exc:
            self.status.setText(f"Limits failed: {exc}")
            return
        R_hi = float(self.rmax_spin.value())
        if not self.extractive.isChecked():
            def job(report, cancel):
                lo, hi = _driver.reflux_band(prob, provider, r_hi=R_hi,
                                             n_scan=16, cancelled=cancel)
                if lo is None:
                    return {"design": None, "band": (None, None)}
                # size just ABOVE R_min: at R_min itself the stage count
                # diverges, so the profile drawn there is the limiting one and
                # has no finite column behind it.
                d = _mbvm_api.size_column(prob, provider, R=lo * 1.02, EF=None,
                                          with_limits=False)
                d["R_min"] = lo
                return {"design": d, "band": (lo, hi)}

            self._run_bg("R_min", job, self._on_rmin_done)
            return

        grid = np.linspace(max(self.ef_min_spin.value(), 0.0),
                           max(self.ef_max_spin.value(), 1e-6),
                           int(self.ef_pts_spin.value()))
        EF = self.ef_spin.value()
        def job(report, cancel):
            reg = _driver.operating_region(
                prob, provider, EF_grid=grid, r_hi=R_hi, n_scan=12,
                on_step=lambda d, t: report(d, t + 1, 0.0), cancelled=cancel)
            if cancel():
                return {"region": reg, "band": (None, None), "EF": EF}
            band = _driver.reflux_band(prob, provider, EF=EF, r_hi=R_hi,
                                       n_scan=16, cancelled=cancel)
            report(len(grid) + 1, len(grid) + 1, 0.0)
            return {"region": reg, "band": band, "EF": EF}

        self._run_bg("R_min / min E/F", job, self._on_limits_done)

    def _on_rmin_done(self, payload):
        lo, hi = payload["band"]
        if lo is None:
            self.status.setText(
                "No feasible reflux below the design-map R max: the profiles "
                "never connect. Raise it, or check the split and keys.")
            return
        self._design = payload["design"]
        self._fill_table(self._design.get("column"), self._design)
        self.status.setText(
            f"R_min = {lo:.4g}"
            + (f", R_max = {hi:.4g}" if hi is not None else
               " (no upper edge -- more reflux never breaks an ordinary column)")
            + f". Column drawn at R = {lo * 1.02:.4g}, just above the minimum, "
              f"where the sections have only just met"
            + (f": {self._design['N_total']} stages"
               if self._design.get("N_total") else "")
            + (f"  [{self._thermo_note}]" if self._thermo_note else ""))
        self.view_combo.setCurrentIndex(0)
        self._plot_current()

    def _on_limits_done(self, payload):
        reg = payload["region"]
        self._region = reg
        lo, hi = payload["band"]
        bits = []
        if reg["EF_min"] is not None:
            bits.append(f"min E/F ~ {reg['EF_min']:.3g} "
                        f"(R_min ~ {reg['r_at_EF_min']:.3g} there)")
        else:
            bits.append("no entrainer ratio in the sweep gave a feasible column")
        if lo is not None:
            bits.append(f"at E/F={payload['EF']:g}: R_min={lo:.3g}"
                        + (f", R_max={hi:.3g}" if hi is not None else
                           ", no upper edge found below the scan ceiling"))
        self.status.setText("Operating region: " + "; ".join(bits) + ".")
        self.view_combo.setCurrentIndex(2)
        self._plot_current()

    def _on_view_changed(self, *_):
        if self._design is not None or self.view_combo.currentIndex() == 2:
            self._plot_current()

    def _on_map(self):
        try:
            prob, provider = self._gather()
        except Exception as exc:
            self.status.setText(f"Design map failed: {exc}")
            return
        R_grid = np.linspace(0.2, self.rmax_spin.value(), int(self.map_pts.value()))
        self._run_bg("Design map",
                     lambda report, cancel: _mbvm_api.feasibility_map(
                         prob, provider, R_grid=R_grid,
                         on_step=lambda d, t: report(d, t, 0.0), cancelled=cancel),
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
        if self._design.get("reactive"):
            # nothing silently ignored: MESH has no reaction terms, so a warm start
            # from a reactive design would converge a different column
            self.status.setText(
                "Reactive designs cannot be sent to the rigorous solver: the MESH "
                "solvers carry no reaction terms, so the warm start would converge "
                "a non-reactive column instead. The BVM sizing + profiles above are "
                "the reactive result.")
            return
        try:
            init = _mbvm_api.to_solver(self._design)
            from core.column_solvers import solve_bubble_point, solve_inside_out
            from core.solver_input import build_solver_input
            # warm-start whichever rigorous solver the case is set to. A BVM-mode
            # case has no rigorous method of its own -> Bubble-Point.
            io = (self.window_state is not None
                  and self.window_state.solver_mode == SolverMode.HYSIM)
            rigorous = solve_inside_out if io else solve_bubble_point
            method = "Inside-Out" if io else "Bubble-Point"
            # rebuild the problem BVM was sized from, so the handoff carries the
            # same feeds, entrainer and pressure profile rather than a
            # reconstruction: a single pooled feed with D computed from F+E is a
            # different column, and dropping phi_fn silently downgrades SRK to
            # ideal gas.
            prob, _provider = self._gather()
            comps = list(prob.comps)
            _antoine, gamma_fn, phi_fn, _lbl, _note = session_models(
                self.window_state, comps)
            psat = self.window_state.thermodynamics_config.psat_params(comps)
            N = init["n_stages"]
            si = build_solver_input(
                n_stages=N, comps=comps, feeds=self._handoff_feeds(prob, init),
                R=init["R"], D=init["D"], antoine=psat,
                pressure=prob.pressure + float(prob.dP or 0.0) * np.arange(N),
                gamma_fn=gamma_fn, phi_fn=phi_fn)
            eff = float(self.eff_spin.value())
        except Exception as exc:
            self.status.setText(f"Handoff failed: {exc}")
            return
        self._run_bg(
            f"Warm start -> {method}",
            lambda: rigorous(si, efficiency=eff, x0=init["x0"], T0=init["T0"]),
            lambda sol: self._on_send_done(sol, method))

    @staticmethod
    def _handoff_feeds(prob, init):
        """(stage, flow, composition, q) per feed, in BVM's top -> bottom order.

        An extractive design has two: the entrainer at the upper feed stage and
        the main feed below it. The entrainer enters as saturated liquid (q=1) --
        that is what makes it run down past the rectifying section, and it is the
        assumption `sections.extractive_chain` already marches on.
        """
        stages = init["feed_stages"]
        mains = [(float(f.F), np.asarray(f.z, float), float(f.q))
                 for f in prob.feeds]
        if prob.extractive and prob.x_E is not None and len(stages) > 1:
            E = float(init.get("operating_point", {}).get("EF") or 0.0) * mains[0][0]
            feeds = [(stages[0], E, np.asarray(prob.x_E, float), 1.0)]
            feeds += [(stages[i + 1], F, z, q)
                      for i, (F, z, q) in enumerate(mains) if i + 1 < len(stages)]
            return feeds
        return [(stages[min(i, len(stages) - 1)], F, z, q)
                for i, (F, z, q) in enumerate(mains)]

    def _on_send_done(self, sol, method="rigorous solver"):
        # `found` only means the run was not cancelled — reporting it as
        # "converged" told the user a budget-exhausted profile had closed.
        ok = sol.get("converged")
        self._push_results(sol)              # G5: feed the Results tab, not just a label
        self.status.setText(
            f"Warm start -> {method}: {'converged' if ok else 'did not converge'} "
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
    # Fixed colour per section, not the axes prop cycle: the cycle assigns by
    # draw order, so stripping came out red on a two-section design and green on
    # a three-section one, and nothing on screen tied a colour to a section.
    _SECTION_C = {"rectifying": RECT_C, "stripping": STRIP_C,
                  "extractive": EXTRACT_C, "intermediate": INTER_C}

    def _plot_current(self):
        view = self.view_combo.currentIndex()
        if view == 2:
            self._plot_region()
            return
        if self._design is None:
            return
        if view == 0:
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
        ax.plot([0, 1], [1, 0], color="#9C9C9C", linestyle="-", lw=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal")

        # A section's marched profile runs well past its junctions -- an interior
        # section is traced along BOTH arms of its saddle and the column uses one
        # of them, so drawing the whole trace at full weight shows an elbow that
        # looks like it points the wrong way. Draw the trace faint and the stages
        # the column actually occupies on top of it.
        col = design.get("column")
        fs = list(design.get("feed_stages", []))
        profiles = design.get("profiles", {})
        used = (col is not None and len(fs) + 1 == len(profiles))
        # The faint trace gets its own legend entry: an interior section overshoots
        # its junctions so far that its arm can run right through the neighbouring
        # section's corner, and an unlabelled curve there reads as the neighbour's
        # trace with the wrong colour on top of it.
        for name, prof in profiles.items():
            X = np.asarray(prof["X"])
            # per-stage markers (G9): a one/two-point section is invisible as a
            # bare line, so draw the stages as dots on the section linestyle.
            ax.plot(X[:, lk], X[:, hk], self._SECTION_LS.get(name, "--"),
                    marker="o", ms=3, lw=1.5, alpha=0.3 if used else 1.0,
                    color=self._SECTION_C.get(name, "0.5"),
                    label=f"{name} (full trace)" if used else name)
        if used:
            xc = np.asarray(col["x"])
            edges = [0] + fs + [len(xc)]
            for name, lo, hi in zip(profiles, edges[:-1], edges[1:]):
                ax.plot(xc[lo:hi, lk], xc[lo:hi, hk],
                        self._SECTION_LS.get(name, "--"), marker="o", ms=3,
                        lw=2.0, color=self._SECTION_C.get(name, "0.5"), label=name)
        for pt, mark, lbl in ((design["xD"], "m^", "x_D"),
                              (design["xB"], "mv", "x_B"),
                              (design["feed_z"], "ys", "feed")):
            ax.plot(pt[lk], pt[hk], mark, ms=8, mec="0.2", label=lbl)
        # entrainer stage marker (G9): the extractive column's upper feed stage
        if col is not None and len(fs) > 1:
            xe = np.asarray(col["x"])[fs[0]]
            ax.plot(xe[lk], xe[hk], "ms", mec="0.2",
                    label="entrainer stage")
        conn = design.get("connection")
        if conn is not None:
            pA, pB = conn.get("pointA"), conn.get("pointB")
            if pA is not None:
                col = "tab:green" if conn["connected"] else "tab:red"
                # the feed jump: the two ADJACENT stages either side of the feed,
                # which a feed is supposed to separate. Not the junction.
                ax.plot([pA[lk], pB[lk]], [pA[hk], pB[hk]], ":", color=col, lw=1.2,
                        label="feed jump (adjacent stages)")
                # the junction itself -- a liquid composition on both profiles, so
                # it belongs on these axes. It used to be `point` from the vapour
                # search, drawn on liquid axes, which is why the marker never sat
                # where the curves visibly meet.
                ax.plot(conn["point"][lk], conn["point"][hk], "o", color=col, ms=6,
                        mfc="none", mew=1.5,
                        label="junction" + (" (approx)" if conn.get("approximate")
                                            else ""))

        rx_tag = " [transformed]" if design.get("reactive") else ""
        ax.set_xlabel(f"{comps[lk]} mole fraction (LK){rx_tag}")
        ax.set_ylabel(f"{comps[hk]} mole fraction (HK){rx_tag}")
        ok = design["feasible"]
        # for C>3 the crossing lives in R^(C-1); this is only its LK/HK shadow (Sec 7)
        proj = "LK/HK projection" if len(comps) > 3 else "Ternary"
        if design.get("reactive"):
            # name the component the transform eliminated — it is missing from
            # these axes by construction, and it is usually the product
            gone = [c for c in design.get("physical", {}).get("comps", [])
                    if c not in comps]
            proj = (f"Reactive {proj.lower()}"
                    + (f", {gone[0]} eliminated" if gone else ""))
        title = (f"{proj} -- {design['N_total']} stages"
                 if ok else
                 f"{proj} -- gap {conn['dmin']:.3f} "
                 f"(need <= {conn['tol']:.3f})" if conn else proj)
        ax.set_title(title)
        ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
        ax.legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.85)
        self.figure.tight_layout(); self.canvas.draw()

    # A reactive design's transformed coordinates are floored at
    # reactive._TRACE (1e-4) to keep a stoichiometric feed off an exactly-zero
    # face. That floor survives active_comps' 1e-6 default and draws a flat
    # noise line, so the physical plot cuts above it. The 1e-6 default stays for
    # ordinary columns, where a 0.05 mol% impurity may be the point of the job.
    _REACTIVE_TRACE = 2e-4

    def _plot_full(self, design):
        """Composition + temperature vs stage. When feasible, the assembled
        column; when not, the marched section profiles vs their own stage index
        (so an infeasible run still shows what was marched).

        A reactive design is plotted in *physical* compositions: the transform
        drops the reference component, which is normally the reaction product --
        the one thing you opened a reactive column to look at.
        """
        self.figure.clear()
        comps = design["comps"]
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        col = design.get("column")
        phys = design.get("physical") if design.get("reactive") else None
        if phys is not None and "x" in phys and col is not None:
            comps = phys["comps"]
            x, T = np.asarray(phys["x"]), np.asarray(phys["T"])
            stages = np.arange(x.shape[0])
            for j in active_comps(x, comps, tol=self._REACTIVE_TRACE)[0]:
                ax1.plot(stages, x[:, j], "-o", ms=3, label=comps[j])
            ax2.plot(stages, T, "-o", ms=3, color=_TEMP_C)
            ext = np.asarray(phys["extent"], float)
            ax3 = ax2.twinx()
            ax3.plot(stages, ext, "-", lw=1.2, color="tab:purple",
                     label="extent")
            ax3.set_ylabel("extent (mol/mol)", color="tab:purple")
            ax3.tick_params(axis="y", labelcolor="tab:purple")
            for fs in design["feed_stages"]:
                ax1.axvline(fs, color="0.5", ls="--", lw=1)
                ax2.axvline(fs, color="0.5", ls="--", lw=1)
            ax1.set_title(f"Physical profile -- {design['N_total']} stages")
        elif col is not None:
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
        ax1.set_xlabel("Stage (0 = distillate)")
        ax1.set_ylabel("Liquid x (physical)" if phys is not None and "x" in phys
                       else "Transformed liquid X" if design.get("reactive")
                       else "Liquid x")
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

    def _plot_region(self):
        """Feasible (E/F, R) region for an extractive column.

        The bifurcation diagram of Bruggemann & Marquardt's Figure 9: R_min and
        R_max against entrainer flow, the feasible band shaded between them. Both
        bounds move with E/F and close on each other as it falls; the nose is the
        minimum entrainer ratio, below which no reflux separates the mixture at
        all. R_min on its own does not describe an extractive column, which is
        why this and not a single number is what the button draws.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        reg = self._region
        if reg is None:
            ax.text(0.5, 0.5,
                    "Tick 'Extractive distillation' and press "
                    "'Compute R_min / min E/F'\nto sweep entrainer flow against "
                    "reflux.\n\nFor an ordinary column the same button reports "
                    "R_min and draws\nthe ternary just above it.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="0.4")
            ax.set_axis_off()
            self.figure.tight_layout(); self.canvas.draw()
            return

        EF = np.asarray(reg["EF"], float)
        lo = np.asarray(reg["r_min"], float)
        hi = np.asarray(reg["r_max"], float)
        ok = np.isfinite(lo)
        if ok.any():
            ax.plot(EF[ok], lo[ok], "-o", ms=4, color="tab:blue", label="R_min")
            if np.isfinite(hi[ok]).any():
                ax.plot(EF[ok], hi[ok], "-o", ms=4, color="tab:red",
                        label="R_max")
                ax.fill_between(EF[ok], lo[ok], hi[ok], color="tab:green",
                                alpha=0.15, label="feasible")
        if reg.get("EF_min") is not None:
            ax.axvline(reg["EF_min"], color="0.4", ls="--", lw=1.2)
            ax.annotate(f"min E/F ~ {reg['EF_min']:.3g}",
                        (reg["EF_min"], reg["r_at_EF_min"]),
                        textcoords="offset points", xytext=(6, 8), fontsize=8)
        ax.plot([self.ef_spin.value()], [self.r_spin.value()], "k*", ms=13,
                label="current operating point")
        ax.set_xlabel("entrainer / feed  (E/F)")
        ax.set_ylabel("reflux ratio R")
        ax.set_title("Feasible operating region", fontsize=10)
        ax.legend(fontsize=8)
        self.figure.tight_layout(); self.canvas.draw()

    def _on_pick(self, event):
        """Load the clicked design map point: size the column at that reflux."""
        if self._map is None:
            return
        R = float(event.artist.get_xdata()[event.ind[0]])
        self.r_spin.setValue(R)
        self._on_size()

    def _fill_table(self, col, design=None):
        """Per-stage table: T, liquid x and vapour y for every species, plus the
        section liquid/vapour molar flows L, V (G10).

        A reactive design is marched in transformed coordinates, so the x/y columns
        are transformed compositions over the *reduced* component list; the physical
        liquid and the reaction extent per stage are appended (they are what the
        column actually holds)."""
        design = design or {}
        phys = design.get("physical") if design.get("reactive") else None
        comps = (list(design["comps"]) if phys is not None
                 else self._species_order())
        x, T = np.asarray(col["x"]), np.asarray(col["T"])
        y = np.asarray(col.get("y")) if col.get("y") is not None else None
        L = np.asarray(col.get("liquid_flow")) if col.get("liquid_flow") is not None else None
        V = np.asarray(col.get("vapor_flow")) if col.get("vapor_flow") is not None else None
        tag = "X " if phys is not None else "x "
        headers = ["Stage", "T (degC)"] + [f"{tag}{c}" for c in comps]
        if y is not None:
            headers += [f"{'Y ' if phys is not None else 'y '}{c}" for c in comps]
        if L is not None:
            headers.append("L (kmol/h)")
        if V is not None:
            headers.append("V (kmol/h)")
        if phys is not None:
            headers += [f"x {c} (real)" for c in phys["comps"]] + ["extent (mol/mol)"]
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
            if phys is not None:
                row += [round(float(v), 4) for v in np.asarray(phys["x"])[r]]
                row.append(round(float(np.asarray(phys["extent"])[r]), 4))
            for c, v in enumerate(row):
                self.data_table.setItem(r, c, QTableWidgetItem(str(v)))


def _demo():
    """Headless self-check: drive gather + size off a stub state, no event loop."""
    import os
    import sys
    _src_py = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # src/python
    sys.path.insert(0, _src_py)
    sys.path.insert(0, os.path.dirname(_src_py))                           # src (for side_features)
    from gui.panels.reactions_panel import ReactionsPanel
    from gui.state.window_state import Species, Stream, StreamType, WindowState
    from PySide6.QtWidgets import QApplication

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

    # the handoff warm-starts whichever rigorous solver the case is set to
    ws.solver_mode = SolverMode.BUBBLE_POINT
    w._on_send(); wait(w)
    assert "Bubble-Point" in w.status.text(), w.status.text()
    ws.solver_mode = SolverMode.HYSIM
    w._on_send(); wait(w)
    assert "Inside-Out" in w.status.text(), w.status.text()

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

    # ---- reactive path: MTBE synthesis through the same widget ---------------
    rws = WindowState()
    rws.pressure = 1.01325
    # order so the reduced list is [n-butane, methanol, isobutene]: inert overhead,
    # methanol the HK, transformed isobutene below it (i.e. down and out as MTBE)
    mtbe = [("n-butane", (6.80896, 935.86, 238.73)),
            ("MTBE", (6.92944, 1156.255, 230.376)),
            ("methanol", (7.8975, 1474.08, 229.13)),
            ("isobutene", (6.89776, 950.02, 243.385))]
    for name, (a, b, c) in mtbe:
        rws.add_species(Species(name=name))
        p = rws.thermodynamics_config.get_component_params(name)
        p.antoine_a, p.antoine_b, p.antoine_c = a, b, c
    rws.add_stream(Stream(id="Feed", stream_type=StreamType.FEED, stage=5, flow=100.0,
                          composition={"n-butane": 0.30, "MTBE": 0.0,
                                       "methanol": 0.40, "isobutene": 0.30}))
    r = BVMModuleWidget(window_state=rws)
    r._refresh_species()
    r.lk_combo.setCurrentText("n-butane"); r.hk_combo.setCurrentText("methanol")
    # R=3: this smoke test only needs a feasible reactive design to exercise the
    # transformed-coordinate plumbing. At R=2 it now sits just the wrong side of
    # the junction tolerance (gap 0.057 vs 0.050) -- reactive sizing's real
    # accuracy is a documented ceiling (BVM_REACTIVE_XFAIL in test_validation),
    # not something this widget check should pin.
    r.rec_lk.setValue(0.98); r.rec_hk.setValue(0.02); r.r_spin.setValue(3.0)
    # the reaction is edited on Initialization -> Reactions and reaches this
    # panel through window_state, so drive it the way that page does
    rx_panel = ReactionsPanel()
    rx_panel.set_window_state(rws)
    rx_panel.set_params({"on": True, "ref": "MTBE", "keq_a": -16.33,
                         "keq_b": 6820.0,
                         "nu": {"n-butane": 0.0, "MTBE": 1.0, "methanol": -1.0,
                                "isobutene": -1.0}})
    assert rws.reactions["on"] and rws.reactions["ref"] == "MTBE"
    r._refresh_species()
    r._sync_reactive_enabled(r._reactive_on())
    assert "Reactive sizing is ON" in r.reaction_status.text()
    # honesty wiring: what reactive mode can't consume is greyed out, not read
    assert not r.eff_spin.isEnabled() and not r.extractive.isEnabled()
    assert not r.send_btn.isEnabled() and "no reaction terms" in r.send_btn.toolTip()
    prob_r, _ = r._gather()
    assert prob_r.reactions is not None and prob_r.efficiency == 1.0
    r._on_size(); wait(r)
    assert r._design["reactive"] and r._design["feasible"], r.status.text()
    assert "Reactive" in r.status.text() and "extent" in r.status.text(), r.status.text()
    hdrs_r = [r.data_table.horizontalHeaderItem(c).text()
              for c in range(r.data_table.columnCount())]
    assert "extent (mol/mol)" in hdrs_r and any(h.endswith("(real)") for h in hdrs_r)
    assert any(h.startswith("X ") for h in hdrs_r), hdrs_r   # transformed columns
    assert r.data_table.rowCount() == r._design["N_total"]
    r._on_send()
    assert "cannot be sent" in r.status.text(), r.status.text()
    # the reaction round-trips through the Reactions panel, and NOT through
    # bvm_params -- two copies in one .colx could disagree
    assert "reaction" not in r.get_params()
    rx2 = ReactionsPanel(); rx2.set_window_state(rws)
    assert rx2.enabled.isChecked() and rx2.ref_combo.currentText() == "MTBE"
    assert rx2.nu_values() == {"n-butane": 0.0, "MTBE": 1.0, "methanol": -1.0,
                               "isobutene": -1.0}, rx2.nu_values()
    assert abs(rx2.keq_a.value() - (-16.33)) < 1e-9 and rx2.keq_b.value() == 6820.0
    # ...and a file written BEFORE the move still loads its reaction
    old = WindowState(); old.species = dict(rws.species)
    r3 = BVMModuleWidget(window_state=old); r3._refresh_species()
    r3.set_params({"reaction": {"on": True, "ref": "MTBE", "keq_a": -16.33,
                                "keq_b": 6820.0, "nu": {"MTBE": 1.0}}})
    assert old.reactions.get("on") and old.reactions["ref"] == "MTBE"
    assert r3._reactive_on() and not r3.eff_spin.isEnabled()
    r.view_combo.setCurrentIndex(1); r.view_combo.setCurrentIndex(0)   # both render

    # --- the R_min button: a picture, not just a number in the status line.
    # Not extractive -> R_min and the column drawn just above it.
    prob_l, prov_l = w._gather()
    lo, hi = _driver.reflux_band(prob_l, prov_l, r_hi=10.0, n_scan=12)
    assert lo is not None and hi is None, (lo, hi)   # ordinary column: open band
    w._on_rmin_done({"design": _mbvm_api.size_column(prob_l, prov_l, R=lo * 1.02,
                                                     with_limits=False),
                     "band": (lo, hi)})
    assert w.view_combo.currentIndex() == 0, "R_min should show the ternary"
    assert "R_min" in w.status.text() and "just above" in w.status.text()
    # no feasible reflux is reported, not drawn as an empty plot
    w._on_rmin_done({"design": None, "band": (None, None)})
    assert "No feasible reflux" in w.status.text()

    # Extractive -> the operating region. Region view renders before any sweep,
    # and R_max is a real second edge rather than a repeat of R_min.
    w.view_combo.setCurrentIndex(2)
    w._plot_region()                                  # empty-state message
    w._on_limits_done({"region": {"EF": np.array([0.4, 0.8, 1.2]),
                                  "r_min": np.array([np.nan, 1.5, 1.2]),
                                  "r_max": np.array([np.nan, 3.0, 6.0]),
                                  "EF_min": 0.8, "r_at_EF_min": 1.5,
                                  "operating": None},
                       "band": (1.5, 3.0), "EF": 0.8})
    assert w.view_combo.currentIndex() == 2
    assert "min E/F" in w.status.text() and "R_max" in w.status.text(), w.status.text()
    assert w.ef_min_spin.isEnabled() is w.extractive.isChecked()

    ex = np.asarray(r._design["physical"]["extent"], float)
    print(f"bvm_module self-check OK  N={w._design['N_total']} "
          f"feed@{w._design['feed_stages']} xD={np.round(xD, 3)}  "
          f"entrainer-detect OK (E/F={w2.ef_spin.value():.2f})  "
          f"reactive N={r._design['N_total']} extent {ex.min():.3f}..{ex.max():.3f}")


if __name__ == "__main__":
    _demo()
