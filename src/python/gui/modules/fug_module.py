"""Shortcut (FUG) design module — Fenske-Underwood-Gilliland screening.

FUG is a preliminary/shortcut method (like BVM), not a rigorous MESH solve, so it
lives here in the Modules tab rather than in the Simulation-tab method list. It
needs the two key components and their recoveries; keys are picked here (shared
with the other modules via window_state.light_key_index / heavy_key_index) and the
recoveries/reflux are entered here, seeded from the operating specs.

Relative volatilities are taken constant, at the feed bubble point — a screening
result. Hand the N/R/feed it produces to the rigorous solver for the real answer.
"""

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas

from core.shortcut import fug_design
from ..plotting import CompactNavigationToolbar
from ..panels.sci_spin_box import SciDoubleSpinBox
from ..state.window_state import StreamType


def gather_fug_inputs(ws, rec_lk, rec_hk, reflux_factor):
    """Build FUG (shortcut) inputs from window_state: constant relative
    volatilities at the feed bubble point, the mixed feed, the two keys. The
    recoveries and reflux factor come from the widget. Raises a user-facing
    ValueError when the setup can't support a shortcut design."""
    from core.thermodynamics import bubble_T, k_values
    from core.dof import SpecKind

    order = ws.get_species_names()
    if len(order) < 2:
        raise ValueError("Need at least 2 species (Initialization tab).")
    lk, hk = ws.light_key_index, ws.heavy_key_index
    if lk is None or hk is None:
        raise ValueError("Pick a light key and a heavy key.")
    if lk == hk:
        raise ValueError("Light and heavy keys must differ.")

    # Flow-weighted mixed feed and quality.
    zsum = np.zeros(len(order)); Ftot = 0.0; qF = 0.0
    for s in ws.streams.values():
        if s.stream_type != StreamType.FEED or not s.flow or not s.composition:
            continue
        z = np.array([s.composition.get(nm, 0.0) for nm in order], float)
        if z.sum() <= 0.0:
            continue
        z = z / z.sum()
        zsum += float(s.flow) * z; Ftot += float(s.flow)
        qF += ws.feed_quality(s, order) * float(s.flow)
    if Ftot <= 0.0:
        raise ValueError("At least one feed with a flow and composition is "
                         "required.")
    z_mixed = zsum / Ftot
    q = qF / Ftot

    antoine = ws.thermodynamics_config.psat_params(order)
    P = ws.thermodynamics_config.pressure_in_psat_unit(ws.pressure)
    gamma_fn = ws.build_gamma_fn(order)
    phi_fn = ws.build_phi_fn(order)
    Tb = bubble_T(z_mixed, P, antoine, gamma_fn=gamma_fn, phi_fn=phi_fn)
    K = k_values(Tb, P, antoine, gamma_fn, z_mixed, phi_fn)
    alpha = np.asarray(K, float) / float(K[hk])
    if alpha[lk] <= alpha[hk]:
        raise ValueError(
            f"'{order[lk]}' is not more volatile than '{order[hk]}' at the "
            "feed bubble point — pick keys so the light key boils lower.")

    # An explicit reflux-ratio spec overrides the R/Rmin factor.
    R_op = next((float(s.value) for s in ws.collect_specs()
                 if s.kind == SpecKind.REFLUX_RATIO), None)
    return dict(alpha=alpha, z=z_mixed, lk=lk, hk=hk,
                rec_lk=rec_lk, rec_hk=rec_hk, q=q,
                reflux_factor=reflux_factor, R_op=R_op), order


class FUGModuleWidget(QWidget):
    def __init__(self, window_state=None, parent=None):
        super().__init__(parent)
        self.window_state = window_state
        self._build_ui()
        if self.window_state:
            self._rebuild_key_combos()
            self._seed_specs()

    # ------------------------------------------------------------------ ui
    def _build_ui(self):
        layout = QHBoxLayout(self)

        left = QWidget(); left.setMaximumWidth(340)
        left_col = QVBoxLayout(left)

        sep = QGroupBox("Separation")
        sep_form = QFormLayout(sep)
        self.lk_combo = QComboBox(self)
        self.hk_combo = QComboBox(self)
        self.lk_combo.currentIndexChanged.connect(self._on_keys_changed)
        self.hk_combo.currentIndexChanged.connect(self._on_keys_changed)
        sep_form.addRow("Light key:", self.lk_combo)
        sep_form.addRow("Heavy key:", self.hk_combo)
        self.rec_lk = self._spin(0.5, 0.99999, 0.98, decimals=4, step=0.01)
        self.rec_hk = self._spin(1e-5, 0.5, 0.02, decimals=4, step=0.01)
        sep_form.addRow("LK recovery to distillate:", self.rec_lk)
        sep_form.addRow("HK recovery to distillate:", self.rec_hk)
        left_col.addWidget(sep)

        op = QGroupBox("Operating point")
        op_form = QFormLayout(op)
        self.reflux_factor = self._spin(1.01, 5.0, 1.3, decimals=2, step=0.05)
        op_form.addRow("Reflux factor R/Rmin:", self.reflux_factor)
        left_col.addWidget(op)

        self.compute_btn = QPushButton("Compute FUG design")
        self.compute_btn.clicked.connect(self._compute)
        left_col.addWidget(self.compute_btn)

        self.status = QLabel(
            "Constant relative volatilities at the feed bubble point — a "
            "screening result. An explicit reflux-ratio spec overrides the "
            "factor. Hand N/R/feed to the rigorous solver for the real answer.")
        self.status.setWordWrap(True)
        left_col.addWidget(self.status)
        left_col.addStretch()
        layout.addWidget(left)

        right = QWidget()
        right_col = QVBoxLayout(right)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        right_col.addWidget(self.summary)

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
        right_col.addWidget(self.toolbar)
        right_col.addWidget(self.canvas, stretch=3)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Component", "xD", "xB"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        right_col.addWidget(self.table, stretch=1)
        layout.addWidget(right, stretch=1)

    @staticmethod
    def _spin(lo, hi, val, decimals=3, step=0.1):
        s = SciDoubleSpinBox(); s.setDecimals(decimals); s.setRange(lo, hi)
        s.setSingleStep(step); s.setValue(val); return s

    # --------------------------------------------------------------- state
    def _rebuild_key_combos(self):
        order = self.window_state.get_species_names()
        lk = getattr(self.window_state, "light_key_index", 0) or 0
        hk = getattr(self.window_state, "heavy_key_index", None)
        if hk is None:
            hk = min(lk + 1, len(order) - 1)
        # commit the resolved keys so a fresh session that never touches the
        # combos still has a heavy key in state (else gather_fug_inputs blows up).
        self.window_state.light_key_index = lk
        self.window_state.heavy_key_index = hk
        for combo, idx in ((self.lk_combo, lk), (self.hk_combo, hk)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(order)
            if 0 <= idx < len(order):
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _seed_specs(self):
        """Prefill recoveries from the operating specs if the user set them."""
        from core.dof import SpecKind
        seed = {SpecKind.LK_RECOVERY: self.rec_lk, SpecKind.HK_RECOVERY: self.rec_hk}
        for s in self.window_state.collect_specs():
            if s.kind in seed:
                seed[s.kind].setValue(float(s.value))

    def showEvent(self, event):
        super().showEvent(event)
        if self.window_state:
            self._rebuild_key_combos()
            self._seed_specs()

    def _on_keys_changed(self, *_):
        if not self.window_state:
            return
        lk, hk = self.lk_combo.currentIndex(), self.hk_combo.currentIndex()
        if lk >= 0:
            self.window_state.light_key_index = lk
        if hk >= 0:
            self.window_state.heavy_key_index = hk
        if lk >= 0 and hk == lk:
            self.status.setText("Light and heavy keys must differ.")

    # ------------------------------------------------------------- compute
    def _compute(self):
        if not self.window_state:
            return
        try:
            kwargs, order = gather_fug_inputs(
                self.window_state, self.rec_lk.value(), self.rec_hk.value(),
                self.reflux_factor.value())
            report = fug_design(**kwargs)
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self._show_report(report, order)

    def _show_report(self, report, comps):
        R, Rmin = report["R"], report["Rmin"]
        self.summary.setText(
            f"<b>Minimum stages (Fenske):</b> {report['Nmin']:.1f} &nbsp; "
            f"<b>Minimum reflux (Underwood):</b> {Rmin:.3f}<br>"
            f"<b>At R = {R:.3f} (×{R / Rmin:.2f} Rmin):</b> "
            f"N ≈ {report['N']:.1f} stages, feed at stage {report['feed_stage']} "
            f"(from top) &nbsp; "
            f"<b>D/F = {report['D']:.3f}</b>, B/F = {report['B']:.3f}")

        self.table.setRowCount(len(comps))
        for i, nm in enumerate(comps):
            self.table.setItem(i, 0, QTableWidgetItem(nm))
            self.table.setItem(i, 1, QTableWidgetItem(f"{report['xD'][i]:.4f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{report['xB'][i]:.4f}"))

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(report["curve_R"], report["curve_N"], color="#218fa7")
        ax.axvline(Rmin, ls="--", color="#d00000", lw=1, label="Rmin")
        ax.axhline(report["Nmin"], ls=":", color="#606060", lw=1, label="Nmin")
        ax.plot([R], [report["N"]], "o", color="#fb8500", label="operating")
        # the curve diverges at Rmin; clamp the view so the knee is readable.
        ax.set_ylim(0, 3.0 * max(report["N"], report["Nmin"]))
        ax.set_xlabel("Reflux ratio R"); ax.set_ylabel("Stages N")
        ax.set_title("Gilliland: stages vs reflux"); ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()
