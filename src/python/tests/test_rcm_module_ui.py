"""RCM panel: selection rules, thermo hand-off and .colx round-trip.

The parts worth pinning are the ones with no equivalent elsewhere in the app:
the hand-rolled ordered component list (Qt has no ordered multi-select), and
the fact that this module's settings now persist in the .colx rather than in
the predecessor's pickled side-file.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication                      # noqa: E402

from core import component_db, rcm                              # noqa: E402
from gui.modules.rcm_module import DEFAULT_P_BAR, RCMModuleWidget  # noqa: E402
from gui.state.window_state import WindowState                  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ws():
    state = WindowState()
    for name in ("ethanol", "water", "benzene", "acetone"):
        component_db.load_into(state, name)
    return state


@pytest.fixture
def panel(app, ws):
    w = RCMModuleWidget(window_state=ws)
    w.reload_from_state()
    return w


def test_plain_click_resets_shift_click_appends(panel):
    lst = panel.comp_list
    names = [lst.item(i).text().split(". ", 1)[-1] for i in range(lst.count())]
    assert names == ["ethanol", "water", "benzene", "acetone"]

    lst.set_picked(["benzene", "ethanol"])
    assert lst.picked() == ["benzene", "ethanol"]

    # Order is click order, not list order, and it drives the triangle corners.
    lst.set_picked(["water", "acetone", "ethanol"])
    assert lst.picked() == ["water", "acetone", "ethanol"]

    # Picked components are the only highlighted ones, and they are numbered.
    labels = [lst.item(i).text() for i in range(lst.count())]
    assert "1. water" in labels and "2. acetone" in labels and "3. ethanol" in labels
    assert "benzene" in labels                       # unpicked: bare name


def test_picked_survive_species_rename_by_name(panel):
    panel.comp_list.set_picked(["benzene", "water", "ethanol"])
    # A species disappearing upstream must drop out, not shift the others.
    panel.comp_list.set_components(["ethanol", "water", "acetone"])
    assert panel.comp_list.picked() == ["water", "ethanol"]


def test_generate_disabled_until_exactly_three(panel):
    panel.comp_list.set_picked(["ethanol", "water"])
    assert not panel.gen_btn.isEnabled()
    assert "3 components" in panel.gen_btn.toolTip()

    panel.comp_list.set_picked(["ethanol", "water", "benzene"])
    assert panel.gen_btn.isEnabled() == rcm.available()


def test_pressure_default_is_one_atm(panel):
    assert panel.p_spin.value() == pytest.approx(DEFAULT_P_BAR)


def test_params_round_trip_through_window_state(panel, ws):
    panel.comp_list.set_picked(["benzene", "ethanol", "water"])
    panel.p_spin.setValue(2.5)
    panel._opts.update(n_it=99, dxi=0.05)
    panel._plot_opts.update(lines=7, linewidth=2.0)
    panel._push_params()

    assert ws.rcm_params["components"] == ["benzene", "ethanol", "water"]

    # A .colx save/load hands the same dict back to a fresh panel.
    fresh = RCMModuleWidget(window_state=ws)
    fresh.reload_from_state()
    assert fresh.comp_list.picked() == ["benzene", "ethanol", "water"]
    assert fresh.p_spin.value() == pytest.approx(2.5)
    assert fresh._opts["n_it"] == 99 and fresh._opts["dxi"] == pytest.approx(0.05)
    assert fresh._plot_opts["lines"] == 7


def test_restore_does_not_clobber_state_with_defaults(app, ws):
    """Load a .colx, save it again without touching the panel: the pressure and
    solver options must survive. Restoring the component list emits
    orderChanged, which pushes a snapshot back to window_state before the rest
    of the restore has happened."""
    ws.rcm_params = {"components": ["benzene", "ethanol", "water"],
                     "pressure": 3.25, "n_it": 77, "dxi": 0.011,
                     "maxiter": 1000, "ftol": 1e-12, "xtol": 1e-12,
                     "linewidth": 1.2, "lines": 9}

    panel = RCMModuleWidget(window_state=ws)
    panel.reload_from_state()

    assert ws.rcm_params["pressure"] == pytest.approx(3.25)
    assert ws.rcm_params["n_it"] == 77
    assert ws.rcm_params["lines"] == 9
    assert ws.rcm_params["components"] == ["benzene", "ethanol", "water"]


def test_options_dialogs_write_back(panel):
    """Both dialogs were decorative in the predecessor -- every box in them was
    silently ignored. Accepting one must change what the solver is handed."""
    from gui.modules.rcm_module import _OptionsDialog

    dlg = _OptionsDialog("Solver Options", [
        ("n_it", "Points:", panel._int_spin(10, 20000, 321)),
        ("dxi", "Step:", panel._spin(1e-4, 1.0, 0.033, decimals=4)),
    ])
    vals = dlg.values()
    assert vals["n_it"] == 321 and vals["dxi"] == pytest.approx(0.033)

    panel._opts.update(n_it=int(vals["n_it"]), dxi=float(vals["dxi"]))
    panel._push_params()
    assert panel.get_params()["n_it"] == 321


def test_rcm_params_is_persisted_by_window_state(ws):
    """The _PERSIST checklist: a field that encodes but never reaches the UI is
    only half done, so pin the encode leg too."""
    ws.rcm_params = {"components": ["a", "b", "c"], "pressure": 1.5}
    assert "rcm_params" in WindowState._PERSIST
    assert ws.to_dict()["rcm_params"]["pressure"] == 1.5


def test_gather_uses_the_app_thermo(panel, ws):
    """The whole point of the rewrite: the map must be built from the app's
    vapour-pressure/activity/EOS selection, in the picked component order."""
    ws.thermodynamics_config.vle_model = "Antoine"
    panel.comp_list.set_picked(["benzene", "water", "ethanol"])

    comps, P, antoine, gamma_fn, phi_fn = panel._gather()
    assert comps == ["benzene", "water", "ethanol"]
    assert antoine.shape == (3, 3)                    # Antoine -> 3 columns
    assert P == pytest.approx(DEFAULT_P_BAR * 750.0617)   # bar -> mmHg
    assert gamma_fn is None and phi_fn is None        # Ideal / Ideal Gas

    # Switching the app's vapour-pressure model changes what RCM is handed.
    ws.thermodynamics_config.vle_model = "PLXANT"
    _c, P2, antoine2, _g, _p = panel._gather()
    assert antoine2.shape == (3, 7)
    assert P2 == pytest.approx(DEFAULT_P_BAR)


def test_gather_refuses_wrong_component_count(panel):
    panel.comp_list.set_picked(["ethanol", "water"])
    with pytest.raises(ValueError, match="exactly 3"):
        panel._gather()


def test_seed_points_cover_the_middle_and_spread(app):
    """Auto-generate seeds decide what the map looks like: every curve goes
    through its own seed, so a set clumped in one corner leaves the middle of
    the triangle (and its direction arrows) bare."""
    from gui.modules.rcm_module import seed_points

    for n in (1, 3, 8, 15, 30):
        s = np.array(seed_points(n))
        assert len(s) == n, f"asked for {n}, got {len(s)}"
        assert np.allclose(s.sum(axis=1), 1.0), "seeds must be compositions"
        assert s.min() > 0.0, "seeds must be strictly inside the simplex"

    s = np.array(seed_points(15))
    centroid = np.full(3, 1.0 / 3.0)
    # Something near the centre ...
    assert np.min(np.abs(s - centroid).max(axis=1)) < 0.12, "nothing near centre"
    # ... and real spread in every component, not a clump.
    assert (s.max(axis=0) - s.min(axis=0)).min() > 0.4, "seeds are clumped"
    # No two seeds effectively on top of each other (the transversals cross).
    d = np.abs(s[:, None, :] - s[None, :, :]).max(axis=2)
    np.fill_diagonal(d, 1.0)
    assert d.min() > 0.04, "duplicate seeds would draw the same curve twice"


def test_arrow_lands_where_the_curve_is_moving(app):
    """A curve converging into a node has coincident points at its tail, so an
    arrow pinned near the end renders as nothing. It must land on a step long
    enough to see."""
    from gui.plotting import _arrow_index

    # a curve that moves for the first half, then pins to a node
    moving = np.linspace([0.8, 0.1, 0.1], [0.2, 0.4, 0.4], 50)
    pinned = np.repeat([[0.2, 0.4, 0.4]], 50, axis=0)
    x = np.vstack([moving, pinned])

    k = _arrow_index(x)
    assert np.max(np.abs(x[k] - x[k - 1])) >= 1e-3, "arrow has zero length"
    assert k <= 50, "arrow should sit in the moving part of the curve"


def test_ternary_is_square_and_labels_fit(panel):
    """Equal aspect (the simplex is a right triangle with equal legs) and room
    outside [0,1] for the corner labels ternary_axes draws there."""
    panel.comp_list.set_picked(["ethanol", "water", "benzene"])
    panel._comps = ["ethanol", "water", "benzene"]
    panel._redraw()

    assert panel.ax.get_aspect() == 1.0
    lo_x, hi_x = panel.ax.get_xlim()
    lo_y, hi_y = panel.ax.get_ylim()
    assert lo_x < 0.0 and hi_x > 1.0, "corner labels would be cropped"
    assert lo_y < 0.0 and hi_y > 1.0


def test_singular_table_is_in_the_left_column(panel):
    """The table moved off the plot side so the triangle gets the full width."""
    left_scroll = panel.layout().itemAt(0).widget()
    assert left_scroll.isAncestorOf(panel.data_table), "table not in left column"
    assert not left_scroll.isAncestorOf(panel.canvas), "canvas must stay right"


def test_modules_tab_launches_and_relaunches_rcm(app, ws):
    """Go through ModulesTab._dispatch, not just the widget: the panel is built
    once and reused, and a File->Load re-arms its one-shot restore. (The first
    version of this rewrite left a dangling `self.rcm_placeholder.show()` in
    _launch_rcm that no widget-level test could have caught.)"""
    from gui.tabs.modules_tab import ModulesTab

    tab = ModulesTab()
    tab.set_window_state(ws)
    tab._dispatch("Residue Curve Map (RCM)")
    assert tab.rcm_widget is not None
    assert tab.content_stack.currentWidget() is tab.rcm_container
    first = tab.rcm_widget

    tab.rcm_widget.comp_list.set_picked(["ethanol", "water", "benzene"])
    tab.set_window_state(ws)                 # a second File->Load
    tab._dispatch("Residue Curve Map (RCM)")
    assert tab.rcm_widget is first, "panel must be reused, not rebuilt"
    assert tab.rcm_widget._restored is True


@pytest.mark.skipif(not rcm.available(), reason="RCM solver not built")
def test_curve_is_clipped_to_the_simplex(panel):
    """Euler can step outside the triangle near a vertex; those rows must not
    reach the plot."""
    panel.comp_list.set_picked(["benzene", "ethanol", "water"])
    comps, P, antoine, gamma_fn, phi_fn = panel._gather()
    x = panel._curve(np.array([0.4, 0.35, 0.25]), P, antoine, gamma_fn, phi_fn)
    assert len(x) > 10
    assert x.min() > -1e-9
    assert np.allclose(x.sum(axis=1), 1.0, atol=1e-6)
