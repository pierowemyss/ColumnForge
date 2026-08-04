"""Species that are in no feed must not shape a BVM or RBM run.

matBVM_prediction_extract_col.colx carries six species, three of which (2ME, EG,
EC) appear in no stream. Their difference-point component is exactly zero in every
section, so the pinch equation x_k (K_k - a) = bvec_k splits and the K_k = a
branch solves at any x_k -- inventing an extractive "ternary saddle" at
x_2ME = 0.53 that passes the k_gap test, and putting body vertices at x_EC = 0.97.
The real saddle then sits on three zero-faces at once, so `bodies._to_edge` stalls
at t = 0 and the S arm collapses onto it: paper p.100 rule 5's four bodies came
out as one sliver in the LK/HK projection, and r_min read 1.13 instead of 0.34.

The exception is a reaction product: absent from the feed on purpose, made on the
tray, and it has to survive the trim.
"""
import os
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", "docs", "examples")
EXAMPLE = os.path.join(EXAMPLES, "matBVM_prediction_extract_col.colx")


def _state(path=EXAMPLE):
    from PySide6.QtWidgets import QApplication
    from gui.state.persistence import load_colx
    from gui.state.window_state import WindowState
    QApplication.instance() or QApplication([])
    ws = WindowState()
    ws.load_from_dict(load_colx(path))
    return ws


def _panel(kind, ws=None):
    from gui.modules.bvm_module import BVMModuleWidget
    from gui.modules.rbm_module import RBMModuleWidget
    ws = ws or _state()
    cls = RBMModuleWidget if kind == "rbm" else BVMModuleWidget
    w = cls(window_state=ws)
    w.reload_from_state()
    return ws, w


def test_the_trim_is_the_analysis_list_not_the_case():
    for kind in ("rbm", "bvm"):
        ws, w = _panel(kind)
        prob, _ = w._gather()
        assert prob.comps == ["MEOH", "DMC", "AN"], (kind, prob.comps)
        assert ws.get_species_names() == ["MEOH", "DMC", "2ME", "EG", "EC", "AN"]
        for name in ("2ME", "EG", "EC"):
            assert name in w._thermo_note, (kind, w._thermo_note)


def test_the_extractive_section_spans_four_bodies_reaching_the_simplex_edges():
    from side_features.rbm import api as rbm_api

    _, w = _panel("rbm")
    prob, provider = w._gather()
    res = rbm_api.analyze(prob, provider, r=w.r_spin.value(), EF=w.ef_spin.value())
    mid = next(s for s in res["sections"] if s["name"] == "extractive")

    bodies = mid["bodies"]
    assert len(bodies) == 4, [b["id"] for b in bodies]      # rule 5
    assert len({b["id"] for b in bodies}) == 4
    for b in bodies:
        for arm in ("start", "end"):
            assert min(b[arm]) < 1e-9, (b["id"], arm, b[arm])   # on a face
            assert np.linalg.norm(b[arm] - b["saddle"]) > 1e-3  # not stalled


def test_a_reaction_product_survives_the_trim():
    """MTBE is 0 in the feed and made on the tray; trimming it would size a
    column for the reaction's reactants alone."""
    ws = _state(os.path.join(EXAMPLES, "reactive_mtbe.colx"))
    ws.reactions = {"on": True, "ref": "MTBE", "keq_a": 2.303, "keq_b": 0.0,
                    "nu": {"isobutene": -1.0, "methanol": -1.0, "MTBE": 1.0,
                           "n-butane": 0.0}}
    _, w = _panel("bvm", ws)
    prob, _ = w._gather()
    assert "MTBE" in prob.comps, prob.comps
    assert prob.reactions is not None
    assert prob.comps.index("MTBE") == int(prob.reactions.ref[0])

    ws.reactions = {"on": False}          # ...and is dead weight without it
    _, w2 = _panel("bvm", ws)
    assert "MTBE" not in w2._gather()[0].comps
