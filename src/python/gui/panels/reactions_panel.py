"""Chemical-reaction editor -- Initialization / Reactions.

One reaction, at chemical equilibrium on every stage, in Ung-Doherty transformed
compositions. It lived inside the BVM module panel; it sits here because a
reaction is a property of the CHEMISTRY, not of one sizing method, and the same
stoichiometry should describe the column whichever solver is pointed at it.

Consumed today by BVM alone, and the page says so. RBM has no reaction path (its
pinch equations are written on the physical compositions), and the rigorous MESH
solvers carry no reaction terms at all -- a reactive BVM design cannot be handed
to them, which `bvm_module` enforces by disabling its own handoff button. Moving
the editor here buys a shared HOME for the data, not shared consumption, and
saying otherwise on the page is exactly what the "nothing silently ignored" rule
exists to prevent.

State lives on `WindowState.reactions` as a plain dict -- {on, nu, ref, keq_a,
keq_b} -- which is the same shape it had under `bvm_params["reaction"]`, so an
older .colx still loads (see `set_params`).
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox,
                               QLabel, QVBoxLayout, QWidget)

from .sci_spin_box import SciDoubleSpinBox


def _spin(lo, hi, val, decimals=3, step=0.1):
    s = SciDoubleSpinBox()
    s.setDecimals(decimals)
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    return s


class ReactionsPanel(QWidget):
    """Stoichiometry, reference component and Keq for one equilibrium reaction."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_state = None
        self._nu_saved = {}
        self._nu_spins = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        self.enabled = QCheckBox("Reactive distillation (one equilibrium reaction)")
        self.enabled.setToolTip(
            "Chemical equilibrium on EVERY stage, in Ung-Doherty transformed "
            "compositions: one reaction, ideal stages, no rigorous handoff. BVM "
            "sizes in transformed coordinates and reports the physical "
            "compositions and the reaction extent per stage alongside them.")
        self.enabled.toggled.connect(self._emit)
        layout.addWidget(self.enabled)

        who = QLabel(
            "Consumed by the <b>BVM</b> module only. RBM solves the pinch "
            "equations on the physical compositions and has no reaction path; "
            "the rigorous Bubble-Point and Inside-Out solvers carry no reaction "
            "terms, so a reactive design cannot be sent to them.")
        who.setWordWrap(True)
        layout.addWidget(who)

        box = QGroupBox("Reaction")
        form = QFormLayout(box)
        # one spin box per species, rebuilt when the species list changes: an
        # ordinary form row per coefficient, no table geometry to fight
        self.nu_form = QFormLayout()
        self.nu_form.setContentsMargins(0, 0, 0, 0)
        form.addRow(QLabel("Stoichiometry (products +, reactants −, inerts 0):"))
        form.addRow(self.nu_form)

        self.ref_combo = QComboBox()
        self.ref_combo.setToolTip(
            "Reference component eliminated by the transform. Pick the reaction "
            "PRODUCT: then every transformed composition stays non-negative. A "
            "two-product reaction (an esterification, say) has no such choice and "
            "is reported as a geometry limit rather than mis-sized.")
        self.ref_combo.currentTextChanged.connect(self._emit)
        form.addRow("Reference component:", self.ref_combo)

        self.keq_a = _spin(-50.0, 50.0, 2.303, decimals=4, step=0.1)
        self.keq_b = _spin(-20000.0, 20000.0, 0.0, decimals=2, step=100.0)
        for w, tip in ((self.keq_a, "A in Keq = exp(A + B/T[K]); ln Keq at "
                                    "infinite T. Activity-based (gamma x)."),
                       (self.keq_b, "B in Keq = exp(A + B/T[K]) in K; 0 = "
                                    "temperature-independent Keq.")):
            w.setToolTip(tip)
            w.valueChanged.connect(self._emit)
        form.addRow("Keq: A =", self.keq_a)
        form.addRow("Keq: B (K) =", self.keq_b)
        layout.addWidget(box)
        layout.addStretch()

    # ------------------------------------------------------------------ state
    def set_window_state(self, ws):
        self.window_state = ws
        self.refresh_species()
        self.set_params(getattr(ws, "reactions", None) or {})

    def _species(self):
        ws = self.window_state
        return list(ws.get_species_names()) if ws else []

    def refresh_species(self):
        """Rebuild the stoichiometry rows and the reference combo for the current
        species, keeping any coefficients already set (or restored from a .colx)
        for species that survived. `_nu_saved` is what makes a restore
        order-proof: `set_params` can land before the species list exists, and
        rebuilding must not silently drop the reaction."""
        names = self._species()
        kept = self.nu_values()
        while self.nu_form.rowCount():
            self.nu_form.removeRow(0)
        self._nu_spins = {}
        for n in names:
            s = _spin(-9.0, 9.0, float(kept.get(n, 0.0)), decimals=2, step=1.0)
            s.valueChanged.connect(self._emit)
            self._nu_spins[n] = s
            self.nu_form.addRow(f"{n}:", s)

        prev = self.ref_combo.currentText()
        self.ref_combo.blockSignals(True)
        self.ref_combo.clear()
        self.ref_combo.addItems(names)
        i = self.ref_combo.findText(prev)
        self.ref_combo.setCurrentIndex(i if i >= 0 else max(len(names) - 1, 0))
        self.ref_combo.blockSignals(False)

    def nu_values(self):
        """{species: coefficient}, falling back to `_nu_saved` for species with
        no row yet."""
        return {**self._nu_saved,
                **{n: float(s.value()) for n, s in self._nu_spins.items()}}

    def get_params(self) -> dict:
        return {"on": self.enabled.isChecked(), "nu": self.nu_values(),
                "ref": self.ref_combo.currentText(),
                "keq_a": self.keq_a.value(), "keq_b": self.keq_b.value()}

    def set_params(self, params: dict):
        if not params:
            return
        self._nu_saved = {k: float(v) for k, v in (params.get("nu") or {}).items()}
        self._nu_spins = {}                  # drop stale rows so the file wins
        self.refresh_species()
        if params.get("ref"):
            i = self.ref_combo.findText(params["ref"])
            if i >= 0:
                self.ref_combo.setCurrentIndex(i)
        for key, spin in (("keq_a", self.keq_a), ("keq_b", self.keq_b)):
            if key in params:
                spin.setValue(float(params[key]))
        self.enabled.setChecked(bool(params.get("on")))

    def showEvent(self, event):
        # species may have been added or renamed on the Chemical Species sub-tab
        # since this page was last looked at. Refreshing on show rather than
        # hooking every emitter is what `bvm_module` does with the same problem.
        self.refresh_species()
        super().showEvent(event)

    def _emit(self, *_):
        if self.window_state is not None:
            self.window_state.reactions = self.get_params()
        self.changed.emit()
