"""BVM (Boundary Value Method) side module.

Left column: BVM-specific parameters + run buttons. Right column: a matplotlib
plot of the rectifying/stripping section curves and, after building the profile,
their intersection and the assembled stage profile.

Feed, pressure, species, light key and Antoine coefficients come from the shared
window_state; only the BVM knobs (r, q, recovery specs, efficiency, optional
entrainer) are entered here.
"""

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QDoubleSpinBox, QCheckBox, QPushButton,
)
from PySide6.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas

from ..state.window_state import StreamType

# ponytail: load the BVM solver by file path. It now lives under src/side_features/bvm
# (alongside freeRCM), which isn't on sys.path by default, so a file-path load is the
# reliable way in; the solver's own `from core...` imports still resolve off src/python.
import importlib.util as _ilu
import os as _os
_solver_path = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..",
                  "side_features", "bvm", "solver.py"))
_spec = _ilu.spec_from_file_location("freecolumn_bvm_solver", _solver_path)
_solver = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_solver)
bound_val_method = _solver.bound_val_method
build_column_profile = _solver.build_column_profile

# Section curve colours (match the original .m: rectifying teal, stripping orange).
_RECT_C = "#218fa7"
_STRIP_C = "#fb8500"


class BVMModuleWidget(QWidget):
    """Parameter panel + plot for a BVM feasibility / profile run."""

    def __init__(self, window_state=None, parent=None):
        super().__init__(parent)
        self.window_state = window_state
        self._result = None          # stash from the last bound_val_method run
        self._xe_spins = {}          # species name -> entrainer composition spin
        self._setup_ui()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # --- left: parameters -------------------------------------------------
        left = QWidget()
        left.setMaximumWidth(320)
        left_col = QVBoxLayout(left)

        params = QGroupBox("BVM Parameters")
        form = QFormLayout(params)

        self.r_spin = self._spin(0.1, 1000.0, 4.0, decimals=3, step=0.5)
        form.addRow("Reflux ratio r:", self.r_spin)

        self.q_spin = self._spin(-1.0, 2.0, 1.0, decimals=3, step=0.1)
        form.addRow("Feed quality q:", self.q_spin)

        self.spec_combo = QComboBox()
        self.spec_combo.addItems(["Recovery", "Direct"])
        self.spec_combo.currentTextChanged.connect(self._on_spec_mode_changed)
        form.addRow("Spec mode:", self.spec_combo)

        # Light/heavy key pickers — any two distinct components (need not be
        # adjacent; components between them distribute into both products).
        self.lk_combo = QComboBox()
        self.lk_combo.currentIndexChanged.connect(self._on_keys_changed)
        form.addRow("Light key (LK):", self.lk_combo)

        self.hk_combo = QComboBox()
        self.hk_combo.currentIndexChanged.connect(self._on_keys_changed)
        form.addRow("Heavy key (HK):", self.hk_combo)

        self.fr_lk_spin = self._spin(0.0, 1.0, 0.98, decimals=4, step=0.01)
        form.addRow("LK recovery to distillate:", self.fr_lk_spin)

        self.fr_hk_spin = self._spin(0.0, 1.0, 0.98, decimals=4, step=0.01)
        form.addRow("HK recovery to bottoms:", self.fr_hk_spin)

        self.eff_spin = self._spin(0.1, 1.0, 1.0, decimals=3, step=0.05)
        form.addRow("Stage efficiency:", self.eff_spin)

        # Solver tuning — raise stages for tight/complex columns; tighten the
        # feasibility tolerance for a sharper section-crossing test.
        self.max_stages_spin = self._spin(2, 1000, 40, decimals=0, step=5)
        form.addRow("Max stages / section:", self.max_stages_spin)

        self.int_tol_spin = self._spin(1e-6, 1e-1, 1e-3, decimals=6, step=1e-4)
        form.addRow("Feasibility tol:", self.int_tol_spin)

        left_col.addWidget(params)

        # --- extractive (optional) -------------------------------------------
        self.extract_check = QCheckBox("Extractive column (entrainer)")
        self.extract_check.toggled.connect(self._on_extract_toggled)
        left_col.addWidget(self.extract_check)

        self.extract_group = QGroupBox("Entrainer")
        self._extract_form = QFormLayout(self.extract_group)
        self.e2f_spin = self._spin(0.0, 100.0, 0.5, decimals=3, step=0.1)
        self._extract_form.addRow("Entrainer/feed E2F:", self.e2f_spin)
        self.extract_group.setVisible(False)
        left_col.addWidget(self.extract_group)

        # --- direct spec (optional) ------------------------------------------
        # Full distillate/bottoms compositions, one mole-fraction spin per
        # species each, shown only in Direct spec mode.
        self.direct_group = QGroupBox("Direct product compositions")
        self._direct_form = QFormLayout(self.direct_group)
        self._xd_spins = {}      # species -> xD spin
        self._xb_spins = {}      # species -> xB spin
        self.direct_group.setVisible(False)
        left_col.addWidget(self.direct_group)

        left_col.addStretch()

        self.status = QLabel("")
        self.status.setWordWrap(True)
        left_col.addWidget(self.status)

        self.run_btn = QPushButton("Run BVM")
        self.run_btn.clicked.connect(self._on_run)
        left_col.addWidget(self.run_btn)

        self.profile_btn = QPushButton("Build Column Profile")
        self.profile_btn.clicked.connect(self._on_build_profile)
        self.profile_btn.setVisible(False)
        left_col.addWidget(self.profile_btn)

        layout.addWidget(left)

        # --- right: plot ------------------------------------------------------
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)

        # Restore knobs from a loaded .colx, if any.
        if self.window_state and getattr(self.window_state, "bvm_params", None):
            self.set_params(self.window_state.bvm_params)

    @staticmethod
    def _spin(lo, hi, val, decimals=3, step=0.1):
        s = QDoubleSpinBox()
        s.setDecimals(decimals)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        return s

    def _on_extract_toggled(self, on):
        self.extract_group.setVisible(on)
        if on:
            self._rebuild_xe_rows()

    def _rebuild_xe_rows(self):
        """(Re)build one entrainer mole-fraction spin per current species."""
        for name, spin in list(self._xe_spins.items()):
            self._extract_form.removeRow(spin)
        self._xe_spins = {}
        for name in self._species_order():
            spin = self._spin(0.0, 1.0, 0.0, decimals=4, step=0.05)
            self._extract_form.addRow(f"xE {name}:", spin)
            self._xe_spins[name] = spin

    def _on_spec_mode_changed(self, mode):
        is_direct = (mode == "Direct")
        self.direct_group.setVisible(is_direct)
        self.fr_lk_spin.setEnabled(not is_direct)
        self.fr_hk_spin.setEnabled(not is_direct)
        if is_direct:
            self._rebuild_direct_rows()

    def _rebuild_direct_rows(self):
        """(Re)build one xD and one xB mole-fraction spin per current species.
        # ponytail: rebuilt on entering Direct mode; reselect Direct if species
        change while it's already open."""
        for spins in (self._xd_spins, self._xb_spins):
            for spin in list(spins.values()):
                self._direct_form.removeRow(spin)
            spins.clear()
        for name in self._species_order():
            spin = self._spin(0.0, 1.0, 0.0, decimals=4, step=0.05)
            self._direct_form.addRow(f"xD {name}:", spin)
            self._xd_spins[name] = spin
        for name in self._species_order():
            spin = self._spin(0.0, 1.0, 0.0, decimals=4, step=0.05)
            self._direct_form.addRow(f"xB {name}:", spin)
            self._xb_spins[name] = spin

    # ------------------------------------------------------- state -> solver
    def _species_order(self):
        if not self.window_state:
            return []
        return self.window_state.get_species_names()

    def showEvent(self, event):
        """Refresh the key dropdowns from the current species list when shown."""
        super().showEvent(event)
        self._rebuild_key_combos()

    def _rebuild_key_combos(self):
        """Populate LK/HK dropdowns from species, preserving the stored indices."""
        order = self._species_order()
        ws = self.window_state
        lk = getattr(ws, "light_key_index", 0) or 0
        hk = getattr(ws, "heavy_key_index", None)
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
        """Persist LK/HK selections to window_state."""
        if not self.window_state:
            return
        lk, hk = self.lk_combo.currentIndex(), self.hk_combo.currentIndex()
        if lk >= 0:
            self.window_state.light_key_index = lk
        if hk >= 0:
            self.window_state.heavy_key_index = hk
        if lk >= 0 and hk == lk:
            self.status.setText("Light and heavy keys must differ.")

    def _feed_stream(self):
        for s in self.window_state.streams.values():
            if s.stream_type == StreamType.FEED:
                return s
        return None

    _PARAM_SPINS = ("r_spin", "q_spin", "fr_lk_spin", "fr_hk_spin", "eff_spin",
                    "max_stages_spin", "int_tol_spin", "e2f_spin")

    def get_params(self) -> dict:
        """Flat snapshot of the scalar BVM knobs, for .colx persistence.
        ponytail: per-species xD/xB/xE arrays aren't persisted (they rebuild per
        species list); add them when a use case needs saved compositions."""
        d = {k: getattr(self, k).value() for k in self._PARAM_SPINS}
        d["spec_mode"] = self.spec_combo.currentText()
        d["extract"] = self.extract_check.isChecked()
        return d

    def set_params(self, params: dict):
        """Apply a snapshot from get_params(); ignores missing/unknown keys."""
        if not params:
            return
        for k in self._PARAM_SPINS:
            if k in params:
                getattr(self, k).setValue(float(params[k]))
        if "spec_mode" in params:
            self.spec_combo.setCurrentText(params["spec_mode"])
        if "extract" in params:
            self.extract_check.setChecked(bool(params["extract"]))

    def _gather_inputs(self):
        """Assemble solver args from window_state. Raises ValueError with a
        user-facing message when the shared setup is incomplete."""
        if not self.window_state:
            raise ValueError("No column state available.")

        order = self._species_order()
        n = len(order)
        if n < 2:
            raise ValueError("Need at least 2 species defined (Initialization tab).")

        lk = self.window_state.light_key_index or 0
        hk = getattr(self.window_state, "heavy_key_index", None)
        if hk is None:
            hk = lk + 1
        if not (0 <= lk < n) or not (0 <= hk < n):
            raise ValueError(f"Key indices (LK={lk}, HK={hk}) out of range "
                             f"for {n} species.")
        if lk == hk:
            raise ValueError("Light and heavy keys must be different components.")

        feed = self._feed_stream()
        if feed is None or not feed.flow or not feed.composition:
            raise ValueError("Feed stream needs a flow rate and composition.")
        zF = np.array([feed.composition.get(name, 0.0) for name in order], float)
        if abs(zF.sum() - 1.0) > 1e-3:
            raise ValueError(f"Feed composition sums to {zF.sum():.4f}, not 1.")

        # Vapour-pressure coeffs per species, in species order: (N,3) Antoine or
        # (N,7) PLXANT depending on the selected vle_model.
        # ponytail: pressure unit must match the fit's unit (no conversion here) —
        # calibrate the coefficients to window_state.pressure's unit.
        antoine = self.window_state.thermodynamics_config.psat_params(order)

        kwargs = dict(
            zF=zF, F=float(feed.flow), r=self.r_spin.value(), q=self.q_spin.value(),
            antoine=antoine, comps=order, lk=lk, hk=hk,
            P=float(self.window_state.pressure), efficiency=self.eff_spin.value(),
            max_stages=int(self.max_stages_spin.value()),
        )

        if self.spec_combo.currentText() == "Direct":
            xD = np.array([self._xd_spins[n].value() if n in self._xd_spins else 0.0
                           for n in order], float)
            xB = np.array([self._xb_spins[n].value() if n in self._xb_spins else 0.0
                           for n in order], float)
            for label, arr in (("Distillate", xD), ("Bottoms", xB)):
                if abs(arr.sum() - 1.0) > 1e-3:
                    raise ValueError(
                        f"{label} composition sums to {arr.sum():.4f}, not 1.")
            kwargs.update(spec_mode="direct", xD=xD, xB=xB)
        else:
            kwargs.update(spec_mode="recovery", FR_LK=self.fr_lk_spin.value(),
                          FR_HK=self.fr_hk_spin.value())

        if self.extract_check.isChecked():
            xE = np.array([self._xe_spins[name].value() if name in self._xe_spins
                           else 0.0 for name in order], float)
            if abs(xE.sum() - 1.0) > 1e-3:
                raise ValueError(f"Entrainer composition sums to {xE.sum():.4f}, not 1.")
            kwargs.update(extract=True, E2F=self.e2f_spin.value(), xE=xE)

        self.window_state.bvm_params = self.get_params()   # mirror knobs for save
        return kwargs

    # ------------------------------------------------------------- actions
    def solve(self) -> dict:
        """Headless gather -> solve -> profile, for the main Simulation Run.

        Returns the build_column_profile() dict. Raises ValueError with a
        user-facing message when the shared setup is incomplete or the column
        is infeasible at these specs.
        """
        kwargs = self._gather_inputs()
        result = bound_val_method(**kwargs)
        profile = build_column_profile(result, int_tol=self.int_tol_spin.value())
        if not profile.get("found"):
            raise ValueError(profile.get(
                "message", "No feasible intersection at these specs."))
        self._result = result          # keep for the interactive plot path
        return profile

    def _on_run(self):
        try:
            kwargs = self._gather_inputs()
            self._result = bound_val_method(**kwargs)
        except Exception as exc:                       # surface, don't crash the GUI
            self._result = None
            self.profile_btn.setVisible(False)
            self.status.setText(f"Run failed: {exc}")
            return

        self.status.setText("Sections marched. Build the profile to find the "
                            "intersection.")
        self.profile_btn.setVisible(True)
        self._plot_sections(self._result)

    def _on_build_profile(self):
        if self._result is None:
            self.status.setText("Run BVM first.")
            return
        try:
            profile = build_column_profile(self._result, int_tol=self.int_tol_spin.value())
        except Exception as exc:
            self.status.setText(f"Profile build failed: {exc}")
            return

        if not profile.get("found"):
            self.status.setText(profile.get("message", "No intersection found."))
            return
        self.status.setText(
            f"Feasible: {profile['n_stages']} stages, feed at {profile['feed_stage']}.")
        self._plot_profile(profile, self._result)

    # -------------------------------------------------------------- plotting
    def _section_axes(self, ax, result):
        lk = result["lk"]
        xr = result["xRect"][:, [lk, lk + 1]]
        xs = result["xStrip"][:, [lk, lk + 1]]
        ax.plot(xr[:, 0], xr[:, 1], "-o", color=_RECT_C, label="Rectifying")
        ax.plot(xs[:, 0], xs[:, 1], "-o", color=_STRIP_C, label="Stripping")
        ax.plot([0, 1], [1, 0], "k-", lw=1)            # composition simplex edge
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("x  (light key)")
        ax.set_ylabel("x  (heavy key)")
        ax.legend(loc="upper right", fontsize=8)

    def _plot_sections(self, result):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._section_axes(ax, result)
        ax.set_title("BVM section profiles")
        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_profile(self, profile, result):
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        self._section_axes(ax1, result)
        pt = profile["intersection"]                   # [LK_rect, LK_strip, HK_rect, HK_strip]
        ax1.plot(pt[0], pt[2], "k*", markersize=14, label="Intersection")
        ax1.set_title("Sections + intersection")

        ax2 = self.figure.add_subplot(122)
        x = profile["x"]                               # (n_stages, n_comps)
        N = np.arange(1, x.shape[0] + 1)
        for j, name in enumerate(result["comps"]):
            ax2.plot(N, x[:, j], "-o", label=name)
        ax2.axvline(profile["feed_stage"], color="grey", ls="--", lw=1)
        ax2.set_xlabel("Stage N (bottom → top)")
        ax2.set_ylabel("Liquid mole fraction x")
        ax2.set_ylim(0, 1)
        ax2.set_title("Column profile")
        ax2.legend(fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw()


def _demo():
    """Headless self-check: drive the widget's gather+solve path off a stub state,
    no Qt event loop. Fails if the state->solver bridge or plotting data breaks."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from PySide6.QtWidgets import QApplication
    from gui.state.window_state import WindowState, Species, Stream, StreamType

    app = QApplication.instance() or QApplication([])
    ws = WindowState()
    ws.pressure = 760.0
    ws.light_key_index = 0
    abc = [(6.90565, 1211.033, 220.79), (6.95464, 1344.8, 219.48),
           (6.99052, 1453.43, 215.31)]
    for name, (a, b, c) in zip(["benzene", "toluene", "xylene"], abc):
        ws.add_species(Species(name=name))
        p = ws.thermodynamics_config.get_component_params(name)
        p.antoine_a, p.antoine_b, p.antoine_c = a, b, c
    ws.add_stream(Stream(id="Feed", stream_type=StreamType.FEED, stage=10,
                         flow=100.0, composition={"benzene": 0.4, "toluene": 0.35,
                                                  "xylene": 0.25}))

    w = BVMModuleWidget(window_state=ws)
    w.r_spin.setValue(12.0)
    kwargs = w._gather_inputs()
    assert abs(kwargs["zF"].sum() - 1.0) < 1e-9
    assert kwargs["antoine"].shape == (3, 3)
    res = bound_val_method(**kwargs)
    prof = build_column_profile(res)
    assert prof["found"], "r=12 ternary should be feasible"
    assert prof["x"].shape[1] == 3

    # headless solve() path used by the main Simulation Run
    prof2 = w.solve()
    assert prof2["found"] and prof2["n_stages"] >= 2
    assert prof2["x"].shape[1] == 3

    # direct spec mode: rows build and _gather_inputs dispatches spec_mode="direct"
    w.spec_combo.setCurrentText("Direct")
    for n, v in zip(["benzene", "toluene", "xylene"], [0.9, 0.08, 0.02]):
        w._xd_spins[n].setValue(v)
    for n, v in zip(["benzene", "toluene", "xylene"], [0.05, 0.40, 0.55]):
        w._xb_spins[n].setValue(v)
    dk = w._gather_inputs()
    assert dk["spec_mode"] == "direct"
    assert abs(dk["xD"].sum() - 1.0) < 1e-9 and abs(dk["xB"].sum() - 1.0) < 1e-9

    print(f"self-check OK: {prof['n_stages']} stages, feed at {prof['feed_stage']}")


if __name__ == "__main__":
    _demo()
