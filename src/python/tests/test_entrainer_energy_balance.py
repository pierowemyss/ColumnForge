"""The extractive modules' "Energy balance on the entrainer feed" opt-in.

CMO feeds the entrainer as a saturated liquid AT THE TRAY, which for a heavy
entrainer is never true: pure glycol boils at 197 C and the extractive section
runs near 95 C, so it flashes and the section's vapour is smaller than the
rectifying section's. `sections.entrainer_q` has computed that for a while; this
is the wiring that lets the panels ask for it, and it has to reach both modules,
because they share `extractive_chain`.
"""
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

EXAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "docs", "examples",
                       "extractive_ipa_water_eg.colx")
R, EF = 2.042, 0.750          # the paper's operating point (Bruggemann Fig. 9)


def _panel(kind):
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow
    from gui.state.persistence import load_colx
    QApplication.instance() or QApplication([])
    win = MainWindow()
    win.window_state.load_from_dict(load_colx(EXAMPLE))
    for tab in (win.init_tab, win.specs_tab, win.sim_tab, win.modules_tab):
        tab.set_window_state(win.window_state)
    win.modules_tab.module_combo.setCurrentText(kind)
    p = win.modules_tab.bvm_widget if "BVM" in kind else win.modules_tab.rbm_widget
    p.extractive.setChecked(True)
    p.r_spin.setValue(R)
    p.ef_spin.setValue(EF)         # RBM's sharp bottoms is built from the POOLED
    if hasattr(p, "sharp"):        # feed, so E/F has to be set before _gather
        p.sharp.setChecked(True)   # RBM: its own limitation, and it fixes D, so
    return win, p                  # both panels compare at the same split


def _extractive_section(panel):
    from side_features.bvm.problem import overall_balance
    from side_features.bvm.sections import extractive_chain
    prob, provider = panel._gather()
    xD, xB, D, B = overall_balance(prob, EF)
    return prob, provider, extractive_chain(prob, R, EF, xD, xB, D, B)[1]


@pytest.mark.parametrize("kind", ["Boundary Value Method (BVM)",
                                  "Rectification Body Method (RBM)"])
def test_the_opt_in_flashes_the_entrainer_in_both_modules(kind):
    """Off = CMO exactly (V carried through the feed), on = the balance.

    The glycol level of every extractive pinch is a pure flow statement,
    x_EG ~ E/L, so the flash is what moves them: 0.372 -> 0.421 here, against the
    paper's 0.55. It closes ~40% of the gap; the rest is excess enthalpy and
    their Aspen data (`sections.entrainer_q`).
    """
    _win, p = _panel(kind)
    assert p.entrainer_eb.isEnabled(), "extractive mode consumes it"

    p.entrainer_eb.setChecked(False)
    prob, _prov, cmo = _extractive_section(p)
    E = EF * prob.feeds[0].F
    assert prob.q_E_fn is None
    assert abs(cmo.V - 188.5) < 1.0, cmo.V           # = the rectifying vapour
    assert abs(E / cmo.L - 0.372) < 5e-3, E / cmo.L

    p.entrainer_eb.setChecked(True)
    prob, _prov, eb = _extractive_section(p)
    assert prob.q_E_fn is not None, "checkbox has to reach the Problem"
    q = 1.0 - (cmo.V - eb.V) / E                     # what the flows imply
    assert 0.6 < q < 0.75, q
    assert abs(eb.V - 165.2) < 1.0, eb.V
    assert abs(E / eb.L - 0.421) < 5e-3, E / eb.L    # pinches move up in glycol


def test_off_by_default_and_persisted():
    """Every result before this shipped was CMO, so the default cannot change;
    and a knob that does not survive a save is a knob the user sets twice."""
    _win, p = _panel("Boundary Value Method (BVM)")
    assert p.entrainer_eb.isChecked() is False
    p.entrainer_eb.setChecked(True)
    params = p.get_params()
    assert params["entrainer_eb"] is True
    p.entrainer_eb.setChecked(False)
    p.set_params(params)
    assert p.entrainer_eb.isChecked() is True


def test_missing_enthalpy_data_says_so_instead_of_silently_using_cmo():
    from gui.modules.module_thermo import attach_entrainer_energy_balance
    _win, p = _panel("Boundary Value Method (BVM)")
    prob, provider = p._gather()
    ws, order = p.window_state, prob.comps
    for nm in order:                       # drop one property everyone needs
        ws.species[nm].cp = None
    note = attach_entrainer_energy_balance(ws, order, prob, provider)
    assert "CMO" in note and prob.q_E_fn is None, note
