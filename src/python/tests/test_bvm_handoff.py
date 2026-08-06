"""BVM -> rigorous solver handoff: the whole column, not most of it.

`_on_send` used to rebuild the MESH problem by hand from the widget's fields. It
dropped `phi_fn` (silently downgrading an SRK session to ideal gas) and dropped
the entrainer stream entirely, while still using a distillate rate computed from
F + E -- so the rigorous solver was asked to close a balance whose material was
missing. It now builds a `SolverInput` from the same `Problem` BVM was sized on.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import PySide6.QtWidgets as _qt  # noqa: E402  bind Qt to PySide6 before matplotlib
import matplotlib  # noqa: E402

assert _qt  # the import is the point; pyflakes (CI's linter) ignores noqa
import numpy as np  # noqa: E402
import pytest  # noqa: E402

matplotlib.use("Agg")

from gui.modules.bvm_module import BVMModuleWidget  # noqa: E402
from side_features.bvm.problem import build_problem


def _extractive_problem():
    return build_problem(["ethanol", "water", "ethylene glycol"],
                         [(np.array([0.85, 0.15, 0.0]), 100.0, 1.0)],
                         760.0, lk=0, hk=1,
                         x_E=np.array([0.0, 0.0, 1.0]), extractive=True)


def test_handoff_feeds_split_the_entrainer_from_the_feed():
    prob = _extractive_problem()
    init = {"feed_stages": [4, 20], "operating_point": {"EF": 0.6}}
    feeds = BVMModuleWidget._handoff_feeds(prob, init)

    assert len(feeds) == 2, feeds
    (e_stage, e_F, e_z, e_q), (f_stage, f_F, f_z, f_q) = feeds
    assert e_stage == 4 and f_stage == 20, "entrainer enters above the feed"
    assert e_F == pytest.approx(60.0), "E = (E/F) * F"
    assert e_z[2] == 1.0 and e_q == 1.0, "pure entrainer, saturated liquid"
    assert f_F == 100.0 and np.allclose(f_z, [0.85, 0.15, 0.0])


def test_a_single_feed_column_still_gets_exactly_one_feed():
    prob = build_problem(["benzene", "toluene", "xylene"],
                         [(np.array([0.4, 0.35, 0.25]), 100.0, 1.0)], 760.0)
    feeds = BVMModuleWidget._handoff_feeds(
        prob, {"feed_stages": [7], "operating_point": {}})
    assert len(feeds) == 1 and feeds[0][0] == 7 and feeds[0][1] == 100.0


def test_the_handed_off_column_carries_all_the_material():
    """The point of the entrainer feed: D + B out must equal F + E in.

    With the entrainer dropped, `_cmo_flows` computed B = F - D from a D that had
    already been sized for F + E -- a column short of 60 kmol/h of glycol.
    """
    from core.solver_input import build_solver_input

    prob = _extractive_problem()
    init = {"feed_stages": [4, 20], "operating_point": {"EF": 0.6}}
    si = build_solver_input(
        n_stages=30, comps=list(prob.comps),
        feeds=BVMModuleWidget._handoff_feeds(prob, init),
        R=3.0, D=40.0, pressure=760.0,
        antoine=np.zeros((3, 3)))
    assert si.total_feed == pytest.approx(160.0)
    assert si.feed_stages() == [4, 20]
    # the glycol arrives on the entrainer stage and nowhere else
    assert si.feed[3, 2] == pytest.approx(60.0)
    assert si.feed[:, 2].sum() == pytest.approx(60.0)
