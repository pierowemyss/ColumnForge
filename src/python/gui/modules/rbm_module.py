"""RBM side module -- rectification bodies, minimum/maximum reflux, feasibility.

A GUI over `src/side_features/rbm`: Bausa/Marquardt's Rectification Body Method
as extended to extractive columns by Bruggemann & Marquardt
(`docs/papers/rbm_bruggemann_marquardt.md`). Feed, pressure, species, keys and
the thermo model come from the shared window_state, exactly as the BVM panel
takes them; the RBM levers -- reflux, key recoveries, optional entrainer E/F --
are entered here.

RBM answers "where is this separation feasible, and what does it cost", not "how
many stages". It solves each section's pinch equations, spans the pinches into
rectification bodies, and tests whether adjacent sections' bodies intersect. The
stage count genuinely does not exist in the method: a body approximates the set a
profile can reach, not the profile. Size the column with the BVM panel at the
operating point this one finds.

Three actions:

  * Analyze       -- pinches and bodies at the chosen (r, E/F); is it feasible?
  * r_min / r_max -- the feasible reflux band. For an extractive column this has
                    an upper edge too: too much reflux washes the entrainer out
                    of the middle section and the separation stops working.
  * Operating region -- that band against entrainer flow, the paper's Figure 9.
                    The two bounds meet at a nose, and the E/F there is the
                    minimum entrainer flow below which no reflux works at all.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from side_features.bvm.problem import build_problem
from side_features.bvm.thermo_adapter import ColumnForgeThermo
from side_features.rbm import api as _rbm_api
from side_features.rbm.bodies import lift_direction

from ..panels.sci_spin_box import SciDoubleSpinBox
from ..plotting import CompactNavigationToolbar
from ..state.window_state import StreamType
from .module_thermo import (ENTRAINER_EB_TIP, attach_entrainer_energy_balance,
                            live_species, session_models)

#: One colour per section, shared by every view so a body, its pinches and its
#: leg of the profile all read as the same object.
SECTION_COLOR = {
    "rectifying": "tab:blue",
    "extractive": "tab:purple",
    "stripping": "tab:red",
    "intermediate": "tab:green",
}

#: `view_combo` indices. Named because a saved .colx stores the raw integer, so
#: the numbering is persisted state and renumbering it silently reopens old files
#: on the wrong view.
_VIEW_TERNARY, _VIEW_PINCHES, _VIEW_REGION = 0, 1, 2


def _active_bodies(res):
    """{section name -> the body index taking part in a junction}.

    One index per section, including the middle one of an extractive column: it
    appears in two gaps but `driver._middle_gaps` makes both report the same body,
    because a middle body is the hull of a single profile polyline S -> x* -> E.
    """
    return {
        name: int(idx)
        for g in res["gaps"]
        for name, idx in zip(g["pair"], g["active"])
        if idx is not None
    }


def _region_job(prob, provider, grid, r_hi, n_scan, EF, report=None, cancel=None):
    """Sweep the (E/F, r) region and take the current point's band OUT of it.

    The current E/F is spliced into the grid rather than banded separately. Both
    buttons used to call `reflux_band` again alongside `operating_region`, which
    is the same ~45 pinch-map solves the sweep had just done at that ratio.
    """
    grid = np.unique(np.append(np.asarray(grid, float), float(EF)))
    reg = _rbm_api.operating_region(
        prob,
        provider,
        EF_grid=grid,
        r_hi=r_hi,
        n_scan=n_scan,
        on_step=(
            None if report is None else lambda done, total: report(done, total, 0.0)
        ),
        cancelled=cancel,
    )
    k = int(np.argmin(np.abs(np.asarray(reg["EF"], float) - float(EF))))
    lo, hi = float(reg["r_min"][k]), float(reg["r_max"][k])
    band = (
        (None, None)
        if not np.isfinite(lo)
        else (lo, None if not np.isfinite(hi) or hi >= r_hi else hi)
    )
    return {"region": reg, "band": band, "EF": EF}


def _hull_order(pts):
    """Projected vertices in convex-polygon order; interior points dropped.

    Falls back to sorting along the dominant direction when the points are
    collinear (a body lying in a face projects to a segment), so the caller gets
    a sensible 2-point line rather than a zero-area polygon.
    """
    pts = np.atleast_2d(np.asarray(pts, float))
    if len(pts) < 3:
        return pts
    try:
        from scipy.spatial import ConvexHull

        return pts[ConvexHull(pts).vertices]
    except Exception:
        d = pts.max(0) - pts.min(0)
        order = np.argsort(pts[:, int(np.argmax(d))])
        ends = pts[order][[0, -1]]
        return ends if not np.allclose(ends[0], ends[1]) else pts[:1]


#: Marker per pinch type. The distinction is the whole content of the diagram --
#: a ternary saddle is what makes an extractive separation possible, and a
#: ternary unstable node on the azeotrope branch is what kills it (paper p.84).
PINCH_MARKER = {
    "saddle": ("D", "saddle"),
    "stable_node": ("o", "stable node"),
    "unstable_node": ("s", "unstable node"),
    "?": ("x", "unclassified"),
}


class RBMModuleWidget(QWidget):
    """Parameter panel + pinch/body plots for a rectification-body analysis."""

    def __init__(self, window_state=None, parent=None):
        super().__init__(parent)
        self.window_state = window_state
        self._result = None  # last `analyze` dict
        self._region = None  # last `operating_region` dict
        self._band = None  # last (r_min, r_max)
        self._thermo_note = ""
        self._entrainer_prefilled = False
        self._restored = False
        self._thread = self._worker = None
        self._setup_ui()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        layout = QHBoxLayout(self)

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
        self.rec_lk = self._spin(0.0, 1.0, 0.98, decimals=4, step=0.01)
        self.rec_hk = self._spin(0.0, 1.0, 0.02, decimals=4, step=0.01)
        form.addRow("LK recovery to distillate:", self.rec_lk)
        form.addRow("HK recovery to distillate:", self.rec_hk)
        self.q_spin = self._spin(-1.0, 2.0, 1.0, decimals=3, step=0.1)
        form.addRow("Feed quality q:", self.q_spin)
        self.sharp = QCheckBox("Sharp split (zero out non-keys in the products)")
        self.sharp.setToolTip(
            "RBM takes the product specification verbatim -- exact zeros and all.\n"
            "They are what open the pinch branch structure the bodies are built\n"
            "from: with a trace of every component in x_D there is a single\n"
            "rectifying pinch and nothing to chain. This is the specification a\n"
            "marching method cannot use, because an exact zero traps its profile\n"
            "on a simplex face."
        )
        form.addRow(self.sharp)
        left_col.addWidget(spec)

        op = QGroupBox("Operating point")
        opf = QFormLayout(op)
        self.r_spin = self._spin(0.05, 100.0, 3.0, decimals=3, step=0.25)
        opf.addRow("Reflux ratio r:", self.r_spin)
        self.extractive = QCheckBox("Extractive distillation")
        opf.addRow(self.extractive)
        self.entrainer_combo = QComboBox()
        opf.addRow("Entrainer species:", self.entrainer_combo)
        self.ef_spin = self._spin(0.0, 10.0, 1.0, decimals=3, step=0.1)
        opf.addRow("Entrainer / feed (E/F):", self.ef_spin)
        self.entrainer_eb = QCheckBox("Energy balance on the entrainer feed")
        self.entrainer_eb.setToolTip(ENTRAINER_EB_TIP)
        opf.addRow(self.entrainer_eb)
        left_col.addWidget(op)

        adv = QGroupBox("Advanced")
        adv.setCheckable(True)
        adv.setChecked(False)
        advf = QFormLayout(adv)
        self.r_hi = self._spin(1.0, 200.0, 30.0, decimals=2, step=1.0)
        advf.addRow("Reflux scan ceiling:", self.r_hi)
        # 12, matching the BVM panel. The scan only has to BRACKET the band edge
        # -- bisection then finds it to `tol` either way -- so the extra points
        # cost a full pinch map each and buy resolution the refinement already
        # provides. On the operating region this is the difference between four
        # minutes and two, because it is paid once per E/F point.
        self.n_scan = self._int_spin(6, 80, 12)
        advf.addRow("Reflux scan points:", self.n_scan)
        self.ef_lo = self._spin(0.0, 10.0, 0.2, decimals=3, step=0.1)
        self.ef_hi = self._spin(0.0, 20.0, 2.0, decimals=3, step=0.1)
        self.ef_pts = self._int_spin(3, 40, 10)
        advf.addRow("E/F sweep from:", self.ef_lo)
        advf.addRow("E/F sweep to:", self.ef_hi)
        advf.addRow("E/F sweep points:", self.ef_pts)
        left_col.addWidget(adv)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setToolTip(
            "Solve every section's pinches at this (r, E/F), "
            "build the rectification bodies, and test "
            "whether adjacent sections' bodies intersect."
        )
        self.analyze_btn.clicked.connect(self._on_analyze)
        self.band_btn = QPushButton("Compute r_min / r_max")
        self.band_btn.setToolTip(
            "Bisect the reflux at which the bodies just touch.\n"
            "Not extractive: plots the ternary at r_min.\n"
            "Extractive: sweeps E/F and plots the feasible operating region, "
            "since an extractive column's reflux band closes as entrainer flow "
            "falls and the minimum entrainer ratio is where it pinches shut."
        )
        self.band_btn.clicked.connect(self._on_band)
        self.region_btn = QPushButton("Operating region (E/F vs r)")
        self.region_btn.clicked.connect(self._on_region)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip(
            "Stop the running sweep. An operating region is one full reflux band "
            "per entrainer ratio, so it is minutes of work; what has been swept "
            "so far is kept and plotted."
        )
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)
        for b in (self.analyze_btn, self.band_btn, self.region_btn, self.cancel_btn):
            left_col.addWidget(b)

        self.extractive.toggled.connect(self._sync_extractive_enabled)
        self._sync_extractive_enabled(self.extractive.isChecked())

        self.status = QLabel(
            "Feed, pressure and thermo come from the shared "
            "column setup. RBM reports feasibility and cost, "
            "not a stage count -- size with the BVM panel at "
            "the operating point found here."
        )
        self.status.setWordWrap(True)
        left_col.addWidget(self.status)
        left_col.addStretch()
        layout.addWidget(left_scroll)

        right = QWidget()
        right_col = QVBoxLayout(right)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
        right_col.addWidget(self.toolbar)
        right_col.addWidget(self.canvas, stretch=3)

        view_row = QHBoxLayout()
        view_row.addStretch()
        view_row.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(
            ["Ternary (LK vs HK)", "Pinches & bodies", "Operating region"]
        )
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        view_row.addWidget(self.view_combo)
        right_col.addLayout(view_row)

        self.data_table = QTableWidget(0, 5)
        self.data_table.setHorizontalHeaderLabels(
            ["Section", "Pinch composition", "Type", "|lambda|", "T (degC)"]
        )
        self.data_table.horizontalHeader().setStretchLastSection(True)
        right_col.addWidget(self.data_table, stretch=1)
        layout.addWidget(right, stretch=1)

        self._refresh_species()

    @staticmethod
    def _spin(lo, hi, val, decimals=3, step=0.1):
        s = SciDoubleSpinBox()
        s.setDecimals(decimals)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        return s

    @staticmethod
    def _int_spin(lo, hi, val):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    def _sync_extractive_enabled(self, on):
        """Entrainer widgets are only consumed in extractive mode -- grey them
        rather than let a value sit there doing nothing (repo rule: nothing
        silently ignored)."""
        for w in (
            self.entrainer_combo,
            self.ef_spin,
            self.ef_lo,
            self.ef_hi,
            self.ef_pts,
        ):
            w.setEnabled(bool(on))
            if not on:
                w.setToolTip("Extractive mode only -- not consumed as set.")
            else:
                w.setToolTip("")
        # its own tooltip explains what the balance does, so keep it either way
        self.entrainer_eb.setEnabled(bool(on))
        self.region_btn.setEnabled(bool(on))
        if not on:
            self.region_btn.setToolTip(
                "An operating region needs an entrainer flow to sweep. "
                "Use r_min / r_max for a simple column."
            )
        else:
            self.region_btn.setToolTip("")

    def _refresh_species(self):
        names = self._species_order()
        for combo, default in (
            (self.lk_combo, 0),
            (self.hk_combo, 1),
            (self.entrainer_combo, len(names) - 1),
        ):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if prev in names:
                combo.setCurrentIndex(names.index(prev))
            elif 0 <= default < len(names):
                combo.setCurrentIndex(default)
            combo.blockSignals(False)

    def reload_from_state(self):
        self._refresh_species()
        if not self._restored and self.window_state is not None:
            params = getattr(self.window_state, "rbm_params", None)
            if params:
                self.set_params(dict(params))
            self._restored = True
        self._prefill_entrainer()

    def showEvent(self, event):
        super().showEvent(event)
        self.reload_from_state()

    def _prefill_entrainer(self):
        """Detect a second, near-pure FEED stream and turn extractive mode on.

        Same rule the BVM panel uses, so a file that opens as extractive there
        opens as extractive here.
        """
        if self._entrainer_prefilled or not self.window_state:
            return
        main, ent = self._feed_streams()
        if ent is None or not ent.composition or not main or not main.flow:
            return
        dom = max(ent.composition, key=ent.composition.get)
        names = self._species_order()
        self.extractive.setChecked(True)
        if dom in names:
            self.entrainer_combo.setCurrentIndex(names.index(dom))
        self.ef_spin.setValue(float(ent.flow) / float(main.flow))
        self._entrainer_prefilled = True

    # ------------------------------------------------------- state -> problem
    def _species_order(self):
        return self.window_state.get_species_names() if self.window_state else []

    def _feed_streams(self):
        feeds = [
            s
            for s in self.window_state.streams.values()
            if s.stream_type == StreamType.FEED and s.composition
        ]
        if not feeds:
            return None, None
        if len(feeds) == 1:
            return feeds[0], None

        def is_entrainer(s):
            return "entrain" in s.id.lower() or (
                s.composition and max(s.composition.values()) >= 0.95
            )

        ent = next((s for s in feeds if "entrain" in s.id.lower()), None) or next(
            (s for s in feeds if is_entrainer(s)), None
        )
        if ent is None:
            return feeds[0], None
        main = next((s for s in feeds if s is not ent), feeds[0])
        return main, ent

    def _gather(self):
        """Build (problem, provider) from window_state + local levers."""
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
            if ent_stream is not None:
                x_E = np.array(
                    [ent_stream.composition.get(n, 0.0) for n in order], float
                )
                if x_E.sum() > 0:
                    x_E = x_E / x_E.sum()
            else:
                e = self.entrainer_combo.currentIndex()
                if e < 0:
                    raise ValueError("Select an entrainer species for extractive mode.")
                if e in (lk, hk):
                    raise ValueError("Entrainer must differ from the light/heavy keys.")
                x_E = np.zeros(len(order))
                x_E[e] = 1.0

        order, z, x_E, lk, hk, dropped = live_species(order, z, x_E, lk, hk)

        antoine, gamma_fn, phi_fn, _label, note = session_models(
            self.window_state, order
        )
        tc = self.window_state.thermodynamics_config
        P = tc.pressure_in_psat_unit(self.window_state.pressure)
        provider = ColumnForgeThermo(antoine, gamma_fn=gamma_fn, phi_fn=phi_fn)
        self._thermo_note = " ".join(
            filter(
                None,
                [
                    note,
                    "held at zero (in no feed): " + ", ".join(dropped)
                    if dropped
                    else "",
                ],
            )
        )

        kw = {}
        if self.sharp.isChecked():
            # exact product compositions, non-keys zeroed: the specification RBM
            # is for and a marcher cannot take.
            #
            # The bottoms is built from the POOLED feed, entrainer included --
            # the same total `overall_balance` closes against. Taking it from the
            # main feed alone leaves the entrainer nowhere to go: on
            # extractive_ipa_water_eg that gave x_B = pure water instead of
            # water + EG, a balance residual of 0.53 F, and a stripping section
            # whose difference point had a negative component.
            xD = np.zeros(len(order))
            xD[lk] = 1.0
            pooled = z * float(feed.flow)
            if extractive and x_E is not None:
                pooled = pooled + self.ef_spin.value() * float(feed.flow) * x_E
            xB = pooled.copy()
            xB[lk] = 0.0
            if xB.sum() <= 0:
                raise ValueError("Sharp split leaves no bottoms product.")
            kw = {"xD": xD, "xB": xB / xB.sum()}

        prob = build_problem(
            comps=order,
            feeds=[(z, float(feed.flow), float(self.q_spin.value()))],
            pressure=P,
            lk=lk,
            hk=hk,
            rec_lk=self.rec_lk.value(),
            rec_hk=self.rec_hk.value(),
            x_E=x_E,
            extractive=extractive,
            **kw,
        )
        if extractive and self.entrainer_eb.isChecked():
            note = attach_entrainer_energy_balance(
                self.window_state, order, prob, provider, ent_stream)
            if note:
                self._thermo_note = (self._thermo_note + "  " + note).strip()
        self.window_state.rbm_params = self.get_params()  # mirror for save
        return prob, provider

    # ------------------------------------------------------------- persistence
    def get_params(self) -> dict:
        return {
            "lk": self.lk_combo.currentText(),
            "hk": self.hk_combo.currentText(),
            "rec_lk": self.rec_lk.value(),
            "rec_hk": self.rec_hk.value(),
            "q_spin": self.q_spin.value(),
            "sharp": self.sharp.isChecked(),
            "r_spin": self.r_spin.value(),
            "extractive": self.extractive.isChecked(),
            "entrainer_eb": self.entrainer_eb.isChecked(),
            "entrainer": self.entrainer_combo.currentText(),
            "ef_spin": self.ef_spin.value(),
            "r_hi": self.r_hi.value(),
            "n_scan": self.n_scan.value(),
            "ef_lo": self.ef_lo.value(),
            "ef_hi": self.ef_hi.value(),
            "ef_pts": self.ef_pts.value(),
            "view": self.view_combo.currentIndex(),
        }

    def set_params(self, params: dict):
        names = self._species_order()
        for key, combo in (
            ("lk", self.lk_combo),
            ("hk", self.hk_combo),
            ("entrainer", self.entrainer_combo),
        ):
            v = params.get(key)
            if v in names:
                combo.setCurrentIndex(names.index(v))
        for key, spin in (
            ("rec_lk", self.rec_lk),
            ("rec_hk", self.rec_hk),
            ("q_spin", self.q_spin),
            ("r_spin", self.r_spin),
            ("ef_spin", self.ef_spin),
            ("r_hi", self.r_hi),
            ("ef_lo", self.ef_lo),
            ("ef_hi", self.ef_hi),
        ):
            if key in params:
                spin.setValue(float(params[key]))
        for key, spin in (("n_scan", self.n_scan), ("ef_pts", self.ef_pts)):
            if key in params:
                spin.setValue(int(params[key]))
        if "sharp" in params:
            self.sharp.setChecked(bool(params["sharp"]))
        if "entrainer_eb" in params:
            self.entrainer_eb.setChecked(bool(params["entrainer_eb"]))
        if "extractive" in params:
            self.extractive.setChecked(bool(params["extractive"]))
            self._entrainer_prefilled = True  # the file wins over detection
        if "view" in params:
            # clamped: files written before the body-path view was removed store
            # an index one past the end, which Qt would silently ignore, leaving
            # whatever view happened to be current
            self.view_combo.setCurrentIndex(
                min(int(params["view"]), self.view_combo.count() - 1)
            )
        self._sync_extractive_enabled(self.extractive.isChecked())

    # ------------------------------------------------------------------ runs
    def _run_bg(self, label, job, on_done):
        from PySide6.QtCore import QThread

        from ..solver_worker import SolverWorker

        if getattr(self, "_thread", None) is not None:
            return  # ponytail: one RBM run at a time
        self.status.setText(f"{label} ...")
        buttons = (self.analyze_btn, self.band_btn, self.region_btn)
        for b in buttons:
            b.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        # pass the worker's hooks on to jobs that want them; `job` taking no
        # arguments stays supported, which is every job but the region sweep
        import inspect

        wants = bool(inspect.signature(job).parameters)
        self._worker = SolverWorker(
            (lambda report, cancel: job(report=report, cancel=cancel))
            if wants
            else (lambda report, cancel: job())
        )
        self._worker.progress.connect(
            lambda done, total, _r: self.status.setText(
                f"{label} ... {done}/{total} entrainer ratios"
            )
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(on_done)
        self._worker.failed.connect(
            lambda msg, tb, _user: self.status.setText(f"{label} failed: {msg}")
        )
        for sig in (self._worker.finished, self._worker.failed):
            sig.connect(self._thread.quit)
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
            self._sync_extractive_enabled(self.extractive.isChecked())

    def _on_analyze(self):
        try:
            prob, provider = self._gather()
        except Exception as exc:
            self.status.setText(f"Analysis failed: {exc}")
            return
        r = self.r_spin.value()
        EF = self.ef_spin.value() if self.extractive.isChecked() else None
        self._run_bg(
            "Analyzing",
            lambda: _rbm_api.analyze(prob, provider, r=r, EF=EF),
            self._on_analyze_done,
        )

    def _on_analyze_done(self, result):
        self._result = result
        n_p = sum(len(s["pinches"]) for s in result["sections"])
        n_b = sum(len(s["bodies"]) for s in result["sections"])
        # a clipped pinch is a face point standing in for a branch that has left
        # the simplex, not a solve -- say so rather than let it read as exact
        n_c = sum(
            p.get("clipped", False) for s in result["sections"] for p in s["pinches"]
        )
        resid = result.get("balance_residual", 0.0)
        gaps = ", ".join(
            f"{a}/{b} {g['distance']:.4g}"
            for g, (a, b) in ((g, g["pair"]) for g in result["gaps"])
        )
        verdict = "FEASIBLE" if result["feasible"] else "infeasible"
        # a middle section with no body has no TERNARY saddle, which is a
        # different thing from bodies that are merely apart (paper p.84) and the
        # one diagnosis a gap of `inf` cannot convey on its own
        no_saddle = [
            s["name"]
            for s in result["sections"]
            if not s["bodies"] and s["name"] == "extractive"
        ]
        # rules 3-4 walk to the edge of COMPOSITION space, which is bigger than the
        # set the section can hold ({x : a x + bvec >= 0}). A body reaching outside
        # it is not clipped -- it is a symptom, and the eigenvector that points
        # there is a dK/dx quantity, i.e. the activity model (see bodies.py).
        n_out = sum(
            1
            for s in result["sections"]
            for b in s["bodies"]
            if b.get("outside_region")
        )
        self.status.setText(
            f"{verdict} at r={result['r']:g}"
            + (f", E/F={result['EF']:g}" if result["EF"] is not None else "")
            + f". {n_p} pinch points"
            + (f" ({n_c} clipped to a face)" if n_c else "")
            + f", {n_b} rectification bodies. Body gaps: {gaps}."
            + (
                "  No ternary saddle in the extractive section, so it spans no "
                "body: the paper's prerequisite for a feasible extractive split "
                "is not met at this point."
                if no_saddle
                else ""
            )
            + (
                f"  NOTE: {n_out} extractive body/bodies reach outside the "
                "section's own balance x_E >= E/L, so they cover compositions no "
                "stage there can hold; the eigenvector is rotated, which is an "
                "activity-model error rather than a geometric one."
                if n_out
                else ""
            )
            + (
                f"  WARNING: products leave {resid:.3g} of the feed unaccounted "
                f"for; D was fit by least squares."
                if resid > 1e-3
                else ""
            )
            + (f"  [{self._thermo_note}]" if self._thermo_note else "")
        )
        self._fill_table(result)
        if self.view_combo.currentIndex() == _VIEW_REGION:
            self.view_combo.setCurrentIndex(_VIEW_PINCHES)
        self._plot_current()

    def _on_band(self):
        """r_min / r_max, then plot whichever picture answers the question.

        Not extractive -> the ternary at r_min: one number, and the geometry that
        produced it. Extractive -> the operating region, because for an extractive
        column r_min alone is not the answer; the band has an upper edge and both
        edges move with E/F, and the entrainer minimum is where they meet.
        """
        try:
            prob, provider = self._gather()
        except Exception as exc:
            self.status.setText(f"Reflux band failed: {exc}")
            return
        r_hi, n_scan = self.r_hi.value(), int(self.n_scan.value())
        if self.extractive.isChecked():
            grid = np.linspace(
                self.ef_lo.value(), self.ef_hi.value(), int(self.ef_pts.value())
            )
            EF = self.ef_spin.value()
            self._run_bg(
                "Reflux band + operating region",
                lambda report=None, cancel=None: _region_job(
                    prob, provider, grid, r_hi, n_scan, EF, report, cancel
                ),
                self._on_region_done,
            )
            return

        def job(report, cancel):
            lo, hi = _rbm_api.reflux_band(prob, provider, r_hi=r_hi,
                                          n_scan=n_scan, cancelled=cancel)
            at = (
                None
                if lo is None
                else _rbm_api.analyze(prob, provider, r=lo * 1.02, EF=None)
            )
            return {"band": (lo, hi), "at": at}

        self._run_bg("Reflux band", job, self._on_band_done)

    def _on_band_done(self, payload):
        lo, hi = payload["band"]
        self._band = (lo, hi)
        if lo is None:
            self.status.setText(
                "No feasible reflux found below the scan ceiling: the bodies "
                "never intersect. Raise the ceiling, or check the split."
            )
            return
        self._result = payload["at"]
        self.status.setText(
            f"r_min = {lo:.4g}"
            + (
                f", r_max = {hi:.4g}"
                if hi is not None
                else " (no maximum -- an ordinary column never breaks from more reflux)"
            )
            + ". Ternary drawn just above r_min, where the bodies have just met."
            + (f"  [{self._thermo_note}]" if self._thermo_note else "")
        )
        if payload["at"] is not None:
            self._fill_table(payload["at"])
        self.view_combo.setCurrentIndex(0)
        self._plot_current()

    def _on_region(self):
        try:
            prob, provider = self._gather()
        except Exception as exc:
            self.status.setText(f"Operating region failed: {exc}")
            return
        grid = np.linspace(
            self.ef_lo.value(), self.ef_hi.value(), int(self.ef_pts.value())
        )
        r_hi, n_scan = self.r_hi.value(), int(self.n_scan.value())
        EF = self.ef_spin.value()
        self._run_bg(
            "Operating region",
            lambda report=None, cancel=None: _region_job(
                prob, provider, grid, r_hi, n_scan, EF, report, cancel
            ),
            self._on_region_done,
        )

    def _on_region_done(self, payload):
        reg = payload["region"]
        self._region = reg
        self._band = payload.get("band")
        bits = []
        if reg["EF_min"] is not None:
            bits.append(
                f"minimum E/F ~ {reg['EF_min']:.3g} "
                f"(r ~ {reg['r_at_EF_min']:.3g} there)"
            )
        else:
            bits.append("no entrainer ratio in the sweep admitted any reflux")
        if self._band and self._band[0] is not None:
            lo, hi = self._band
            bits.append(
                f"at E/F={payload['EF']:g}: r_min={lo:.3g}"
                + (f", r_max={hi:.3g}" if hi is not None else ", no upper edge")
            )
        if reg["operating"]:
            o = reg["operating"]
            bits.append(
                f"recommended E/F={o['EF']:.3g} "
                f"(r {o['r_min']:.3g}-{o['r_max']:.3g}), "
                f"the first point with the paper's operational headroom"
            )
        self.status.setText("Operating region: " + "; ".join(bits) + ".")
        self.view_combo.setCurrentIndex(_VIEW_REGION)
        self._plot_current()

    def _on_view_changed(self, *_):
        self._plot_current()

    # -------------------------------------------------------------- plotting
    def _plot_current(self):
        view = self.view_combo.currentIndex()
        if view == _VIEW_REGION:
            self._plot_region()
            return
        if self._result is None:
            self._blank("Run Analyze to solve the pinches and build the bodies.")
            return
        if view == _VIEW_TERNARY:
            self._plot_ternary(self._result)
        else:
            self._plot_pinches(self._result)

    def _blank(self, msg):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            wrap=True,
            transform=ax.transAxes,
            fontsize=9,
            color="0.4",
        )
        ax.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw()

    @staticmethod
    def _triangle(ax):
        # ponytail: not `gui.plotting.ternary_axes`. That one fixes comps[0]/[1]
        # to the axes and labels the origin comps[2], which is wrong for an LK/HK
        # pair that is not the first two species, and it turns the axes off --
        # these plots want the ticks, because reading a pinch composition off
        # them is the point.
        ax.plot([0, 1], [1, 0], color="#9C9C9C", linestyle="-", lw=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)

    def _draw_bodies(self, ax, res, base_alpha=0.30):
        """Every section's bodies, as filled polygons with their vertices shown.

        Three things this has to get right that the plain `Polygon(vertices)` it
        replaces did not:

        * ORDER. Body vertices are a hull's point set in no particular order, and
          a polygon through them in list order self-intersects and paints a
          bowtie. They are convex-hulled here first.
        * DEGENERACY. A body with two vertices is a legitimate rectification body
          (the paper's Figure 5 left is a line on a binary edge) and one with a
          single vertex is what a section with no usable pinch produces. Drawn as
          a bare `plot(..., "-")` the first reads as an axis artifact and the
          second paints NOTHING AT ALL -- which is why the rectifying section
          looked missing. Vertices are always marked, so a body is visible
          whatever its dimension.
        * CONTRAST. Fill, outline and markers are drawn separately so only the
          fill is transparent. Fading the whole patch to alpha 0.08, as before,
          left an inactive body invisible rather than secondary.
        """
        active = _active_bodies(res)
        lk, hk = res["lk"], res["hk"]
        for sec in res["sections"]:
            colour = SECTION_COLOR.get(sec["name"], "0.4")
            live = {active[sec["name"]]} if sec["name"] in active else set()
            for i, body in enumerate(sec["bodies"]):
                V = np.asarray(body["vertices"], float)
                pts = np.unique(
                    np.round(np.column_stack([V[:, lk], V[:, hk]]), 12), axis=0
                )
                on = i in live
                ring = _hull_order(pts)
                if len(ring) >= 3:
                    ax.add_patch(
                        Polygon(
                            ring,
                            closed=True,
                            facecolor=colour,
                            edgecolor="none",
                            alpha=base_alpha if on else base_alpha * 0.4,
                            zorder=1,
                        )
                    )
                    loop = np.vstack([ring, ring[:1]])
                    ax.plot(
                        loop[:, 0],
                        loop[:, 1],
                        "-",
                        color=colour,
                        lw=2.0 if on else 1.0,
                        alpha=0.95 if on else 0.55,
                        zorder=2,
                    )
                elif len(ring) == 2:
                    ax.plot(
                        ring[:, 0],
                        ring[:, 1],
                        "-",
                        color=colour,
                        lw=2.4 if on else 1.2,
                        alpha=0.95 if on else 0.55,
                        zorder=2,
                    )
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    "o",
                    color=colour,
                    ms=5.0 if on else 3.5,
                    mfc=colour if on else "none",
                    mew=1.2,
                    alpha=0.95 if on else 0.6,
                    zorder=3,
                )
            ax.plot([], [], "-", color=colour, lw=2, label=f"{sec['name']} bodies")

    def _plot_ternary(self, res):
        """LK/HK projection: bodies as filled polygons, pinches on top.

        For C > 3 this is a shadow of an object living in R^(C-1) -- two bodies
        can look crossed here and be far apart in the dropped directions. The
        title says so; the reported gap is always the full-dimensional one.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._triangle(ax)
        lk, hk = res["lk"], res["hk"]
        comps = res["comps"]
        self._draw_bodies(ax, res)

        seen = set()
        for sec in res["sections"]:
            for p in sec["pinches"]:
                if not p["in_simplex"]:
                    continue
                mark, lbl = PINCH_MARKER.get(p["kind"], PINCH_MARKER["?"])
                ax.plot(
                    p["x"][lk],
                    p["x"][hk],
                    mark,
                    color="k",
                    ms=6,
                    mfc="w",
                    mew=1.4,
                    label=lbl if lbl not in seen else None,
                )
                seen.add(lbl)

        for pt, mark, lbl in ((res["xD"], "m^", "x_D"), (res["xB"], "mv", "x_B")):
            ax.plot(pt[lk], pt[hk], mark, ms=9, mec="0.2", label=lbl)

        ax.set_xlabel(f"{comps[lk]} mole fraction (LK)")
        ax.set_ylabel(f"{comps[hk]} mole fraction (HK)")
        proj = "LK/HK projection" if len(comps) > 3 else "Ternary"
        verdict = (
            "bodies intersect"
            if res["feasible"]
            else f"bodies apart by {res['max_gap']:.3g}"
        )
        ax.set_title(f"{proj} at r={res['r']:g} -- {verdict}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_pinches(self, res):
        """The paper's diagram: pinch points by stability, their eigenvectors,
        and the rectification bodies (Figures 4-7).

        Eigenvector arrows are drawn from each pinch, scaled by |lambda| so the
        directions a profile is pushed along hardest are the longest. Stable
        directions (|lambda| < 1, the profile is drawn in) are solid; unstable
        ones are dashed. For an extractive section these arrows are not
        decoration -- the middle-section body is literally built by following the
        first saddle's most stable and the last saddle's most unstable
        eigenvector to the edge of the simplex (paper p.100, rules 3-4).
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._triangle(ax)
        lk, hk = res["lk"], res["hk"]
        comps = res["comps"]
        C = len(comps)

        self._draw_bodies(ax, res, base_alpha=0.18)

        # S and E of the active middle body (paper Figure 6). Named because they
        # are not pinches -- they are where the saddle's stable and unstable
        # eigendirections run off the edge of the composition space, i.e. where
        # the middle profile arrives from and leaves toward.
        active = _active_bodies(res)
        for sec in res["sections"]:
            body = sec["bodies"][active[sec["name"]]] if sec["name"] in active else None
            if body is None or "start" not in body:
                continue
            for pt, tag in ((body["start"], "S"), (body["end"], "E")):
                ax.annotate(
                    tag,
                    (pt[lk], pt[hk]),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=9,
                    fontweight="bold",
                    color=SECTION_COLOR.get(sec["name"], "0.4"),
                    zorder=6,
                )

        seen = set()
        for sec in res["sections"]:
            colour = SECTION_COLOR.get(sec["name"], "0.4")
            for p in sec["pinches"]:
                if not p["in_simplex"] or p["eigvals"] is None:
                    continue
                x = p["x"]
                mark, lbl = PINCH_MARKER.get(p["kind"], PINCH_MARKER["?"])
                ax.plot(
                    x[lk],
                    x[hk],
                    mark,
                    color=colour,
                    ms=8,
                    mfc="w",
                    mew=1.6,
                    label=lbl if lbl not in seen else None,
                    zorder=5,
                )
                seen.add(lbl)
                mags = np.abs(p["eigvals"])
                scale = 0.10 / max(mags.max(), 1e-9)
                for k in range(p["eigvecs"].shape[1]):
                    v = lift_direction(p["eigvecs"][:, k], C, p.get("drop"))
                    n = np.linalg.norm(v)
                    if n < 1e-12:
                        continue
                    v = v / n * mags[k] * scale
                    ax.annotate(
                        "",
                        xy=(x[lk] + v[lk], x[hk] + v[hk]),
                        xytext=(x[lk], x[hk]),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=colour,
                            lw=1.3,
                            linestyle="-" if mags[k] < 1.0 else "--",
                            alpha=0.9,
                        ),
                    )

        ax.plot([], [], "->", color="0.3", lw=1.3, label="eigenvector (solid |L|<1)")
        for pt, mark, lbl in ((res["xD"], "m^", "x_D"), (res["xB"], "mv", "x_B")):
            ax.plot(pt[lk], pt[hk], mark, ms=9, mec="0.2", label=lbl)
        ax.set_xlabel(f"{comps[lk]} mole fraction (LK)")
        ax.set_ylabel(f"{comps[hk]} mole fraction (HK)")
        ax.set_title(
            f"Pinches, eigendirections and rectification bodies " f"at r={res['r']:g}",
            fontsize=10,
        )
        ax.legend(fontsize=7, loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_region(self):
        """Feasible (E/F, r) region -- the paper's Figure 9.

        r_min and r_max against entrainer flow, the feasible band shaded between
        them. They close on each other as E/F falls; the nose is the minimum
        entrainer ratio, below which no reflux separates the mixture at all.
        """
        if self._region is None:
            self._blank(
                "Run 'Operating region' (or 'Compute r_min / r_max' in "
                "extractive mode) to sweep entrainer flow against reflux."
            )
            return
        reg = self._region
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        EF = np.asarray(reg["EF"], float)
        lo = np.asarray(reg["r_min"], float)
        hi = np.asarray(reg["r_max"], float)
        ok = np.isfinite(lo)

        ax.plot(EF[ok], lo[ok], "-o", ms=4, color="tab:blue", label="r_min")
        if np.any(np.isfinite(hi)):
            ax.plot(EF[ok], hi[ok], "-o", ms=4, color="tab:red", label="r_max")
            ax.fill_between(
                EF[ok], lo[ok], hi[ok], color="tab:green", alpha=0.15, label="feasible"
            )
        if reg["EF_min"] is not None:
            ax.axvline(reg["EF_min"], color="0.4", ls="--", lw=1.2)
            ax.annotate(
                f"(E/F)min ~ {reg['EF_min']:.3g}",
                (reg["EF_min"], reg["r_at_EF_min"]),
                textcoords="offset points",
                xytext=(6, 8),
                fontsize=8,
            )
        if reg["operating"]:
            o = reg["operating"]
            ax.plot(
                [o["EF"]],
                [0.5 * (o["r_min"] + o["r_max"])],
                "k*",
                ms=13,
                label="recommended operating point",
            )
        ax.plot(
            [self.ef_spin.value()],
            [self.r_spin.value()],
            "kP",
            ms=10,
            label="current operating point",
        )
        ax.set_xlabel("entrainer / feed  (E/F)")
        ax.set_ylabel("reflux ratio r")
        ax.set_title("Feasible operating region", fontsize=10)
        ax.legend(fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw()

    # ----------------------------------------------------------------- table
    def _fill_table(self, res):
        rows = [(sec["name"], p) for sec in res["sections"] for p in sec["pinches"]]
        self.data_table.setRowCount(len(rows))
        for i, (name, p) in enumerate(rows):
            mags = (
                "-"
                if p["eigvals"] is None
                else ", ".join(f"{m:.3g}" for m in np.abs(p["eigvals"]))
            )
            cells = [
                name,
                ", ".join(f"{v:.4f}" for v in p["x"]),
                p["kind"] + ("" if p["in_simplex"] else " (outside simplex)"),
                mags,
                "-" if not np.isfinite(p["T"]) else f"{p['T']:.2f}",
            ]
            for j, txt in enumerate(cells):
                self.data_table.setItem(i, j, QTableWidgetItem(txt))


def _demo():
    """Headless self-check: the panel builds, runs, and draws every view."""
    import sys

    from gui.state import persistence
    from gui.state.window_state import WindowState
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    ws = WindowState()
    ws.load_from_dict(persistence.load_colx("docs/examples/c2-c4.colx"))
    w = RBMModuleWidget(window_state=ws)
    w.reload_from_state()

    assert w.view_combo.count() == 3, w.view_combo.count()

    # --- species in no feed are held at zero and never reach the pinch solve
    z = np.array([0.5, 0.5, 0.0, 0.0])
    xE = np.array([0.0, 0.0, 0.0, 1.0])
    o, z2, xE2, lk, hk, dropped = live_species(["A", "B", "C", "E"], z, xE, 0, 1)
    assert o == ["A", "B", "E"] and dropped == ["C"], (o, dropped)
    assert (lk, hk) == (0, 1) and np.allclose(z2, [0.5, 0.5, 0.0])
    assert np.allclose(xE2, [0.0, 0.0, 1.0])
    # a key stays even with nothing of it in the feed, and a live list is untouched
    o, _, _, lk, hk, dropped = live_species(["A", "B", "C"], z[:3], None, 0, 2)
    assert o == ["A", "B", "C"] and dropped == [] and (lk, hk) == (0, 2), o

    # entrainer widgets are greyed while extractive is off, live while on
    assert not w.ef_spin.isEnabled()
    w.extractive.setChecked(True)
    assert w.ef_spin.isEnabled() and w.region_btn.isEnabled()
    w.extractive.setChecked(False)
    assert not w.region_btn.isEnabled()

    # every view renders before any run, rather than throwing on empty state
    for i in range(w.view_combo.count()):
        w.view_combo.setCurrentIndex(i)
        w._plot_current()

    # params survive a round trip, including the sharp-split flag
    w.sharp.setChecked(True)
    w.r_spin.setValue(2.75)
    p = w.get_params()
    w2 = RBMModuleWidget(window_state=ws)
    w2.reload_from_state()
    w2.set_params(p)
    assert w2.sharp.isChecked() and abs(w2.r_spin.value() - 2.75) < 1e-9
    assert w2.get_params()["lk"] == p["lk"]
    w2.set_params({**p, "view": 3})  # an index from before the body-path
    assert w2.view_combo.currentIndex() == w2.view_combo.count() - 1  # view went

    # A real analysis, run inline (no thread) so the self-check is deterministic.
    # Sharp split on purpose: it is the specification RBM is for, and the one
    # that opens the pinch ladder the bodies chain along. With a smeared recovery
    # spec each section has too few pinches for the bodies to reach each other,
    # and the panel correctly reports no feasible reflux -- exercised below.
    w.sharp.setChecked(True)
    prob, provider = w._gather()
    res = _rbm_api.analyze(prob, provider, r=float(w.r_spin.value()), EF=None)
    w._on_analyze_done(res)
    assert w._result is not None
    assert w.data_table.rowCount() == sum(len(s["pinches"]) for s in res["sections"])
    for i in range(w.view_combo.count()):
        w.view_combo.setCurrentIndex(i)
        w._plot_current()

    # --- the r_min button, non-extractive: a band plus the ternary that made it
    lo, hi = _rbm_api.reflux_band(prob, provider, r_hi=8.0, n_scan=8)
    w._on_band_done(
        {
            "band": (lo, hi),
            "at": (
                None
                if lo is None
                else _rbm_api.analyze(prob, provider, r=lo * 1.02, EF=None)
            ),
        }
    )
    if lo is not None:
        assert (
            w.view_combo.currentIndex() == _VIEW_TERNARY
        ), "r_min should show the ternary"
        assert "r_min" in w.status.text(), w.status.text()
    w._on_band_done({"band": (None, None), "at": None})
    assert "No feasible reflux" in w.status.text()

    # --- the extractive path: an operating region, and the empty state before it
    w.view_combo.setCurrentIndex(_VIEW_REGION)
    w._region = None
    w._plot_region()  # renders the hint, no throw
    w._on_region_done(
        {
            "region": {
                "EF": np.array([0.4, 0.8, 1.2]),
                "r_min": np.array([np.nan, 1.5, 1.2]),
                "r_max": np.array([np.nan, 3.0, 6.0]),
                "EF_min": 0.8,
                "r_at_EF_min": 1.5,
                "operating": {"EF": 1.2, "r_min": 1.2, "r_max": 6.0},
            },
            "band": (1.5, 3.0),
            "EF": 0.8,
        }
    )
    assert w.view_combo.currentIndex() == _VIEW_REGION
    txt = w.status.text()
    assert "minimum E/F" in txt and "r_max" in txt and "recommended" in txt, txt

    # --- a cancelled sweep returns what it swept, rather than nothing
    import side_features.rbm.driver as _drv

    seen = []
    stop = _drv.operating_region(
        prob,
        provider,
        EF_grid=np.linspace(0.4, 1.6, 4),
        r_hi=4.0,
        n_scan=6,
        on_step=lambda d, t: seen.append((d, t)),
        cancelled=lambda: len(seen) >= 2,
    )
    assert seen and seen[0][1] == 4, seen
    assert len(seen) == 2, seen  # stopped, did not run on
    assert np.isnan(stop["r_min"][-1]), stop["r_min"]  # unswept stays NaN
    assert w.cancel_btn is not None and not w.cancel_btn.isEnabled()

    print(
        f"rbm_module self-check OK  {w.data_table.rowCount()} pinches tabulated, "
        f"feasible={res['feasible']}, {w.view_combo.count()} views drawn, "
        f"r_min={'-' if lo is None else round(lo, 3)}"
    )


if __name__ == "__main__":
    _demo()
