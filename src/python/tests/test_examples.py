"""Every bundled example must still load and still run.

docs/examples/reactive_mtbe.colx shipped with "specs": [] — no column
specification at all — because nothing ever ran it. This is the gate that stops
that: each .colx is loaded through the real persistence layer and then solved
(or, for a reactive case, sized) exactly the way the app would.

Slow by nature — these are full column solves. Run with the rest of the suite:
    QT_QPA_PLATFORM=offscreen python -m pytest src/python/tests/test_examples.py
"""
import glob
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", "docs", "examples")

# Files the rigorous solver is not expected to run end-to-end:
#   *_bvm     — BVM sizing inputs; solver_mode "bvm" means Modules -> BVM, and
#               the distillate is a BVM prediction rather than a spec set
#   reactive_ — reactions live only in the BVM path (core/ has no reaction
#               terms), so "works" means "sizes", checked separately below
BVM_ONLY = {"multicomp_col", "extract_col", "matBVM_prediction_extract_col"}


def _cases():
    return sorted(glob.glob(os.path.join(EXAMPLES, "*.colx")))


def _name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _load(path):
    from gui.state.persistence import load_colx
    return load_colx(path)


@pytest.mark.parametrize("path", _cases(), ids=_name)
def test_example_loads(path):
    """Decodes cleanly and carries the pieces a case needs to mean anything."""
    state = _load(path)
    assert state["species"], _name(path)
    assert state["num_stages"] >= 3
    feeds = [s for s in state["streams"].values()
             if getattr(s.stream_type, "value", s.stream_type) == "Feed"]
    assert feeds, "no feed stream"
    assert all(s.flow for s in feeds), "a feed with no flow"


@pytest.mark.parametrize("path", _cases(), ids=_name)
def test_example_is_specified(path):
    """A case with no specs cannot be run — the exact rot that shipped once."""
    state = _load(path)
    name = _name(path)
    if name in BVM_ONLY or state.get("bvm_params", {}).get("reaction", {}).get("on"):
        pytest.skip("sized through the BVM module, not the spec ledger")
    assert state["specs"], f"{name} has no column specification"


def _solve(state):
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow
    from gui.state.window_state import SolverMode

    QApplication.instance() or QApplication([])
    win = MainWindow()
    win.window_state.load_from_dict(state)
    if win.window_state.solver_mode == SolverMode.HYSIM:
        return win._solve_inside_out()
    return win._solve_bubble_point()


@pytest.mark.parametrize(
    "path", [p for p in _cases() if _name(p) not in BVM_ONLY
             and not _name(p).startswith("reactive_")], ids=_name)
def test_example_solves(path):
    state = _load(path)
    prof = _solve(state)
    res = float(prof.get("residual", np.inf))
    assert prof.get("converged", res < 1e-3), (_name(path), res)
    T = np.asarray(prof["T"], float)
    assert np.all(np.isfinite(T))
    assert T[0] < T[-1] + 1e-6, "column runs cold at the bottom"


# Both reactive examples size at the reflux stored in their .colx. That is not
# free: under the honest junction test feasibility is a band in R, so an example
# whose R drifts out of it stops sizing — which is exactly what this test is for.
# (The band itself is BVM_REACTIVE_XFAIL in test_validation.py.)
@pytest.mark.parametrize(
    "path", [p for p in _cases() if _name(p).startswith("reactive_")], ids=_name)
def test_reactive_example_sizes_and_makes_product(path):
    """The point of a reactive example is that product appears where there was
    none in the feed. Checked on the *physical* profile: the transformed
    coordinates drop the reference component, which is the product."""
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow
    from gui.modules.bvm_module import BVMModuleWidget
    from side_features.bvm import driver

    state = _load(path)
    QApplication.instance() or QApplication([])
    win = MainWindow()
    win.window_state.load_from_dict(state)
    panel = BVMModuleWidget(window_state=win.window_state)
    panel.set_params(dict(state["bvm_params"]))
    prob, tp = panel._gather()
    design = driver.size_column(prob, tp,
                                R=float(state["bvm_params"]["r_spin"]))

    assert design["feasible"], [f.cls for f in design.get("findings", [])]
    phys = design["physical"]
    ref = state["bvm_params"]["reaction"]["ref"]
    assert ref in phys["comps"] and ref not in design["comps"]
    x = np.asarray(phys["x"])[:, phys["comps"].index(ref)]
    assert x[0] < 0.01 and x[-1] > 0.4, (ref, x[0], x[-1])
    assert np.asarray(phys["extent"])[-1] > 0.1


def test_the_example_set_covers_the_features_that_have_none():
    """Coverage ratchet: side sections, pumparounds, an energy balance, a side
    draw and a second feed each had zero example files. Keep it that way at your
    peril — this asserts the gap stays closed."""
    from gui.state.window_state import ModuleType, StreamType

    seen_modules, flags = set(), set()
    for path in _cases():
        state = _load(path)
        for m in state.get("modules", {}).values():
            seen_modules.add(getattr(m.module_type, "value", m.module_type))
        if state.get("energy_balance"):
            flags.add("energy_balance")
        types = [getattr(s.stream_type, "value", s.stream_type)
                 for s in state["streams"].values()]
        if types.count("Feed") > 1:
            flags.add("two_feeds")
        if StreamType.SIDESTREAM.value in types:
            flags.add("side_draw")
        if state["thermodynamics_config"].activity_model == "UNIFAC":
            flags.add("unifac")
        if state["thermodynamics_config"].vle_model == "PLXANT":
            flags.add("plxant")

    for want in (ModuleType.SIDE_STRIPPER, ModuleType.SIDE_RECTIFIER,
                 ModuleType.PUMPAROUND, ModuleType.INTERREBOILER):
        assert want.value in seen_modules, f"no example uses {want.value}"
    assert {"energy_balance", "two_feeds", "side_draw", "unifac",
            "plxant"} <= flags, flags
