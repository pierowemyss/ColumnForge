"""Shortcut (FUG) design module — Fenske-Underwood-Gilliland screening.

FUG is a preliminary/shortcut method (like BVM), not a rigorous MESH solve, so it
lives here in the Modules tab rather than in the Simulation-tab method list. It
needs the two key components and their recoveries; keys are picked here (shared
with the other modules via window_state.light_key_index / heavy_key_index) and the
recoveries/reflux come from the operating specs.
"""

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QPushButton,
)

from core.shortcut import fug_design
from ..fug_report_dialog import FUGReportDialog
from ..state.window_state import StreamType


def gather_fug_inputs(ws):
    """Build FUG (shortcut) inputs from window_state: constant relative
    volatilities at the feed bubble point, the mixed feed, the two keys and their
    recoveries. Raises a user-facing ValueError when the setup can't support a
    shortcut design."""
    from core.thermodynamics import bubble_T, k_values
    from core.dof import SpecKind

    order = ws.get_species_names()
    if len(order) < 2:
        raise ValueError("Need at least 2 species (Initialization tab).")
    lk, hk = ws.light_key_index, ws.heavy_key_index
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

    # Recoveries from spec kinds if present, else a sharp-ish default.
    rec = {SpecKind.LK_RECOVERY: 0.98, SpecKind.HK_RECOVERY: 0.02}
    for s in ws.collect_specs():
        if s.kind in rec:
            rec[s.kind] = float(s.value)
    R_op = next((float(s.value) for s in ws.collect_specs()
                 if s.kind == SpecKind.REFLUX_RATIO), None)
    return dict(alpha=alpha, z=z_mixed, lk=lk, hk=hk,
                rec_lk=rec[SpecKind.LK_RECOVERY],
                rec_hk=rec[SpecKind.HK_RECOVERY], q=q, R_op=R_op), order


class FUGModuleWidget(QWidget):
    def __init__(self, window_state=None, parent=None):
        super().__init__(parent)
        self.window_state = window_state
        self._build_ui()
        if self.window_state:
            self._rebuild_key_combos()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        keys_group = QGroupBox("Key Components")
        keys_form = QFormLayout(keys_group)
        self.lk_combo = QComboBox(self)
        self.hk_combo = QComboBox(self)
        self.lk_combo.currentIndexChanged.connect(self._on_keys_changed)
        self.hk_combo.currentIndexChanged.connect(self._on_keys_changed)
        keys_form.addRow("Light key:", self.lk_combo)
        keys_form.addRow("Heavy key:", self.hk_combo)
        layout.addWidget(keys_group)

        layout.addWidget(QLabel(
            "Recoveries and reflux are taken from the operating specs "
            "(Specifications tab); defaults are 98% LK / 2% HK recovery."))

        row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute FUG design")
        self.compute_btn.clicked.connect(self._compute)
        row.addWidget(self.compute_btn)
        row.addStretch()
        layout.addLayout(row)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()

    def _rebuild_key_combos(self):
        order = self.window_state.get_species_names()
        lk = getattr(self.window_state, "light_key_index", 0) or 0
        hk = getattr(self.window_state, "heavy_key_index", None)
        if hk is None:
            hk = min(lk + 1, len(order) - 1)
        for combo, idx in ((self.lk_combo, lk), (self.hk_combo, hk)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(order)
            if 0 <= idx < len(order):
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

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

    def _compute(self):
        if not self.window_state:
            return
        try:
            kwargs, order = gather_fug_inputs(self.window_state)
            report = fug_design(**kwargs)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText(
            f"Nmin={report['Nmin']:.1f}, Rmin={report['Rmin']:.2f}, "
            f"N≈{report['N']:.1f} — see report.")
        FUGReportDialog(report, order, self).exec()
