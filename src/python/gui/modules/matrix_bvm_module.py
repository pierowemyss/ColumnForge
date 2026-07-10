"""Matrix BVM side module.

A thin GUI over `src/side_features/matrix_bvm` — the Naphtali-Sandholm
feasibility solver + MESH initializer. Feed, pressure, species, keys and the
Antoine/activity thermo come from the shared window_state (same as the BVM
module); only the Matrix-BVM knobs (stage count, feed stage, reflux ratio,
bottoms rate) are entered here.

Two actions:
  * Assess Feasibility — build the structured guess U0 and classify it
    (feasible / offending stages), no solve.
  * Converge — damped-Newton / continuation solve; plots the stage profile and
    reports condenser/reboiler duties + mass-balance closure.
"""

import os as _os
import sys as _sys

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QTableWidget,
    QTableWidgetItem,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas

from ..state.window_state import StreamType
from ..plotting import CompactNavigationToolbar, TEMP_C as _TEMP_C

# The matrix_bvm package uses bare intra-package imports (so each kernel runs
# standalone), so it is imported by putting its own directory on sys.path — same
# idea as the BVM file-path load. Refs are captured now, before any later path
# insert (e.g. freeRCM) could shadow the generic module names.
_MBVM_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(__file__), "..", "..", "..", "side_features", "matrix_bvm"))
if _MBVM_DIR not in _sys.path:
    _sys.path.insert(0, _MBVM_DIR)
import api as _mbvm_api                       # noqa: E402
from problem import build_problem, OpSpec     # noqa: E402
from thermo_adapter import FreeColumnThermo    # noqa: E402


class MatrixBVMModuleWidget(QWidget):
    """Parameter panel + profile plot for a Matrix BVM feasibility / solve run."""

    def __init__(self, window_state=None, parent=None):
        super().__init__(parent)
        self.window_state = window_state
        self._sol = None
        self._setup_ui()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        layout = QHBoxLayout(self)

        left = QWidget()
        left.setMaximumWidth(320)
        left_col = QVBoxLayout(left)

        params = QGroupBox("Matrix BVM Parameters")
        form = QFormLayout(params)

        self.n_spin = self._int_spin(3, 500, 16)
        form.addRow("Number of stages N:", self.n_spin)

        self.feed_spin = self._int_spin(0, 499, 8)
        form.addRow("Feed stage (0 = distillate):", self.feed_spin)

        self.r_spin = self._spin(0.01, 1000.0, 3.0, decimals=3, step=0.5)
        form.addRow("Reflux ratio:", self.r_spin)

        self.bottom_combo = QComboBox()
        self.bottom_combo.addItems(["Bottoms rate", "Distillate rate"])
        form.addRow("Bottom spec:", self.bottom_combo)

        self.rate_spin = self._spin(0.0, 1e9, 60.0, decimals=3, step=5.0)
        form.addRow("Rate value:", self.rate_spin)

        left_col.addWidget(params)
        left_col.addStretch()

        self.status = QLabel("Feed, pressure and thermo come from the shared "
                             "column setup.")
        self.status.setWordWrap(True)
        left_col.addWidget(self.status)

        self.assess_btn = QPushButton("Assess Feasibility")
        self.assess_btn.clicked.connect(self._on_assess)
        left_col.addWidget(self.assess_btn)

        self.converge_btn = QPushButton("Converge")
        self.converge_btn.clicked.connect(self._on_converge)
        left_col.addWidget(self.converge_btn)

        layout.addWidget(left)

        right = QWidget()
        right_col = QVBoxLayout(right)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
        right_col.addWidget(self.toolbar)
        right_col.addWidget(self.canvas, stretch=3)

        self.data_table = QTableWidget(0, 2)
        self.data_table.setHorizontalHeaderLabels(["Stage", "T (degC)"])
        self.data_table.horizontalHeader().setStretchLastSection(True)
        right_col.addWidget(self.data_table, stretch=1)
        layout.addWidget(right, stretch=1)

    @staticmethod
    def _spin(lo, hi, val, decimals=3, step=0.1):
        s = QDoubleSpinBox()
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

    # ------------------------------------------------------- state -> solver
    def _species_order(self):
        return self.window_state.get_species_names() if self.window_state else []

    def _feed_stream(self):
        for s in self.window_state.streams.values():
            if s.stream_type == StreamType.FEED:
                return s
        return None

    def _gather(self):
        """Build (problem, provider) from window_state + local knobs.

        Raises ValueError with a user-facing message when the shared setup is
        incomplete.
        """
        if not self.window_state:
            raise ValueError("No column state available.")
        order = self._species_order()
        if len(order) < 2:
            raise ValueError("Need at least 2 species (Initialization tab).")

        feed = self._feed_stream()
        if feed is None or not feed.flow or not feed.composition:
            raise ValueError("Feed stream needs a flow rate and composition.")
        zF = np.array([feed.composition.get(name, 0.0) for name in order], float)
        if abs(zF.sum() - 1.0) > 1e-3:
            raise ValueError(f"Feed composition sums to {zF.sum():.4f}, not 1.")

        N = int(self.n_spin.value())
        fstage = int(self.feed_spin.value())
        if not (0 <= fstage < N):
            raise ValueError(f"Feed stage {fstage} must be in 0..{N - 1} "
                             "(0 = distillate).")
        # App convention is 0 = distillate; Problem stages are 0-based top->bottom
        # — the same orientation, so no flip.
        fstage_internal = fstage

        antoine = self.window_state.thermodynamics_config.psat_params(order)
        P = self.window_state.thermodynamics_config.pressure_in_psat_unit(
            self.window_state.pressure)
        gamma_fn = self.window_state.build_gamma_fn(order)
        provider = FreeColumnThermo(antoine, gamma_fn=gamma_fn)

        rate = float(self.rate_spin.value())
        F = float(feed.flow)
        if self.bottom_combo.currentText() == "Bottoms rate":
            if not (0 < rate < F):
                raise ValueError(f"Bottoms rate must be in (0, feed={F:g}).")
            bottom_spec = OpSpec("bottoms_rate", rate)
        else:
            if not (0 < rate < F):
                raise ValueError(f"Distillate rate must be in (0, feed={F:g}).")
            bottom_spec = OpSpec("bottoms_rate", F - rate)

        prob = build_problem(
            n_stages=N, comps=order, feeds=[(fstage_internal, F, zF)], pressure=P,
            provider=provider, top_spec=OpSpec("reflux_ratio", self.r_spin.value()),
            bottom_spec=bottom_spec)
        return prob, provider

    # ------------------------------------------------------------- actions
    def _on_assess(self):
        try:
            prob, provider = self._gather()
            fa = _mbvm_api.assess_feasibility(prob, provider)
        except Exception as exc:
            self.status.setText(f"Assess failed: {exc}")
            return
        rep = fa["report"]
        if fa["feasible"]:
            self.status.setText("Feasible: structural, physical and "
                                "thermodynamic checks pass at the initial guess.")
        else:
            msg = "; ".join(f"{f.cls}" + (f" @ {f.stages}" if f.stages else "")
                            for f in rep.findings)
            self.status.setText(f"Not feasible: {msg}")
        # show the structured guess profile
        sol = _mbvm_api.extract_profiles(fa["U0"], prob, provider)
        self._plot_profile(sol, title="Initial guess (U0)")
        self._fill_table(sol)

    def _on_converge(self):
        try:
            prob, provider = self._gather()
            sol = _mbvm_api.converge(prob, provider)
        except Exception as exc:
            self.status.setText(f"Converge failed: {exc}")
            return
        self._sol = sol
        info = sol["info"]
        if sol["converged"]:
            from core.thermodynamics import KJH_TO_KW
            mb = np.max(np.abs(sol["mass_balance"]["per_component"]))
            self.status.setText(
                f"Converged in {info['iterations']} iters (|R|="
                f"{info['residual']:.1e}). "
                f"Qc={sol['condenser_duty'] * KJH_TO_KW:.1f} kW, "
                f"Qr={sol['reboiler_duty'] * KJH_TO_KW:.1f} kW; "
                f"mass-balance closure {mb:.1e}.")
        else:
            findings = sol.get("findings", [])
            msg = "; ".join(f.cls for f in findings) or info["message"]
            self.status.setText(f"Did not converge: {msg}")
        self._plot_profile(sol, title="Converged profile")
        self._fill_table(sol)

    # -------------------------------------------------------------- plotting
    def _plot_profile(self, sol, title=""):
        self.figure.clear()
        comps = list(sol["comps"])
        x, T = sol["x"], sol["T"]                       # top->bottom, 0 = distillate
        stages = np.arange(x.shape[0])

        ax1 = self.figure.add_subplot(121)
        for j, name in enumerate(comps):
            ax1.plot(stages, x[:, j], "-o", ms=3, label=name)
        ax1.set_xlabel("Stage (0 = distillate)")
        ax1.set_ylabel("Liquid mole fraction x")
        ax1.set_ylim(0, 1)
        ax1.set_title(title or "Column profile")
        ax1.legend(fontsize=8)

        ax2 = self.figure.add_subplot(122)
        ax2.plot(stages, T, "-o", ms=3, color=_TEMP_C)
        ax2.set_xlabel("Stage (0 = distillate)")
        ax2.set_ylabel("T (degC)")
        ax2.set_title("Temperature profile")

        self.figure.tight_layout()
        self.canvas.draw()

    def _fill_table(self, sol):
        comps = list(sol["comps"])
        headers = ["Stage", "T (degC)"] + [f"x {c}" for c in comps]
        x, T = sol["x"], sol["T"]                       # top->bottom, 0 = distillate
        n = x.shape[0]
        self.data_table.setColumnCount(len(headers))
        self.data_table.setHorizontalHeaderLabels(headers)
        self.data_table.setRowCount(n)
        for r in range(n):                              # distillate (0) on top row
            row = [r, round(float(T[r]), 2)] + [round(float(v), 4) for v in x[r]]
            for c, v in enumerate(row):
                self.data_table.setItem(r, c, QTableWidgetItem(str(v)))


def _demo():
    """Headless self-check: drive gather+converge off a stub state, no event loop."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from PySide6.QtWidgets import QApplication
    from gui.state.window_state import WindowState, Species, Stream, StreamType

    QApplication.instance() or QApplication([])  # Qt keeps the singleton alive
    ws = WindowState()
    ws.pressure = 1.01325                                # bar (= 760 mmHg)
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

    w = MatrixBVMModuleWidget(window_state=ws)
    w.n_spin.setValue(16); w.feed_spin.setValue(8); w.r_spin.setValue(3.0)
    prob, provider = w._gather()
    assert prob.n_stages == 16 and prob.C == 3

    w._on_assess()
    assert w.data_table.rowCount() == 16, "assess should fill the profile table"
    assert "easible" in w.status.text(), w.status.text()

    w._on_converge()
    assert w._sol is not None and w._sol["converged"], w.status.text()
    assert w.data_table.rowCount() == 16
    assert w.data_table.item(0, 0).text() == "0", \
        "distillate (stage 0) belongs on the top row"
    # separation happened
    xD, xB = w._sol["xD"], w._sol["xB"]
    assert xD[0] > 0.4 > xB[0], (xD, xB)
    print(f"matrix_bvm_module self-check OK "
          f"({w._sol['info']['iterations']} iters, xD={np.round(xD, 3)})")


if __name__ == "__main__":
    _demo()
