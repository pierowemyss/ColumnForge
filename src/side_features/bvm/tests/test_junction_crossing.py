"""The feed junction is where the LIQUID profiles cross (docs/papers/bvm_connect_criteria.md).

Pinned to `docs/examples/c2-c4.colx` -- ethane/propane/n-butane at R = 0.175 and
Murphree E = 0.5, which no other BVM test loads. It is the case where measuring
the junction on the vapour instead went wrong by the widest margin, because
ethane's K is large enough to compress vapour distances by ~20x:

  * the two VAPOUR curves cross exactly at (xi_R, xi_S) = (6.31, 12.04), giving
    feed stage 7, where the liquids are nowhere near each other;
  * the two LIQUID curves cross exactly at (10.17, 6.17), giving feed stage 10;
  * a rigorous MESH sweep over the feed stage at N = 17 peaks at feed 10-11
    (light-key recovery 0.986) against 0.967 at feed 7.

The same compression put R_min 22% low: at R = 0.1138 the accepted vapour gap was
0.019 while the liquids were 0.072 apart. Underwood gives 0.1466 for this split
and the RBM module 0.1342.
"""

import os

import numpy as np
import pytest

from gui.state.persistence import load_colx
from gui.state.window_state import WindowState
from side_features.bvm.connect import CROSS_TOL, connect
from side_features.bvm.driver import _size, r_min
from side_features.bvm.march import march_section
from side_features.bvm.problem import build_problem, overall_balance
from side_features.bvm.sections import single_feed_chain
from side_features.bvm.thermo_adapter import ColumnForgeThermo

_COLX = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "docs", "examples", "c2-c4.colx"))

R = 0.175


@pytest.fixture(scope="module")
def case():
    ws = WindowState()
    ws.load_from_dict(load_colx(_COLX))
    order = ws.get_species_names()
    assert order == ["ethane", "propane", "n-butane"], order
    P = ws.thermodynamics_config.pressure_in_psat_unit(ws.pressure)
    tp = ColumnForgeThermo(ws.thermodynamics_config.psat_params(order),
                           gamma_fn=ws.build_gamma_fn(order),
                           phi_fn=ws.build_phi_fn(order))
    prob = build_problem(comps=order, feeds=[(np.array([0.5, 0.25, 0.25]),
                                              100.0, 1.0)],
                         pressure=P, lk=0, hk=1, rec_lk=0.99, rec_hk=0.01)
    prob.efficiency = float(ws.stage_efficiency)          # 0.5 in the file
    return prob, tp, P


def test_the_feed_lands_on_the_profile_crossing(case):
    """The reported bug: the feed was placed 3-4 stages before the intersection."""
    prob, tp, _ = case
    d = _size(prob, tp, R=R)
    assert d["feasible"], [f.cls for f in d["findings"]]
    c = d["connection"]
    # a real crossing, not a near miss bought with a tolerance
    assert c["dmin"] <= CROSS_TOL, c["dmin"]
    assert not c["approximate"]
    # the junction the liquid geometry actually has, and the feed it implies
    assert c["nA"] == pytest.approx(9.17, abs=0.4), c["nA"]
    assert c["nB"] == pytest.approx(6.17, abs=0.4), c["nB"]
    assert d["feed_stages"] == [10], d["feed_stages"]
    assert d["N_total"] == 17, d["N_total"]


def test_the_crossing_is_on_both_marched_profiles(case):
    """`point` is a liquid composition that lies on each curve -- so it can be
    plotted on the liquid ternary, which the vapour-space answer could not."""
    prob, tp, P = case
    xD, xB, D, B = overall_balance(prob)
    rect, strip = single_feed_chain(prob, R, xD, xB, D, B)
    r = march_section(rect, xD, tp, P, prob.max_stages, efficiency=prob.efficiency)
    s = march_section(strip, xB, tp, P, prob.max_stages, efficiency=prob.efficiency)
    c = connect(r, s, rect, tp, P, eps_stage=prob.eps_stage,
                efficiency=prob.efficiency)
    pt = c["point"]
    for prof in (r["X"], s["X"]):
        seg = np.linalg.norm(prof - pt[None, :], axis=1).min()
        assert seg < 0.05, seg                    # within a fraction of one stage
    # and the feed stage is the crossing index on the UPPER profile, one past nA
    assert np.allclose(c["pointB"], _interp(r["X"], c["nA"] + 1.0), atol=1e-6)


def test_rmin_agrees_with_underwood_and_rbm(case):
    """0.1138 was 22% under Underwood. The references bracket 0.134-0.147."""
    prob, tp, _ = case
    rm = r_min(prob, tp)
    assert rm is not None, "feasibility is a band; bisect_min must still find it"
    assert 0.11 < rm < 0.16, rm


def test_a_loose_tolerance_cannot_buy_a_lower_rmin(case):
    """The knob that used to move R_min by 22% is out of the C <= 3 gate."""
    from dataclasses import replace
    prob, tp, _ = case
    below = replace(prob, eps_stage=0.2)
    assert not _size(below, tp, R=0.11)["feasible"]


def _interp(X, f):
    i = min(max(int(np.floor(f)), 0), len(X) - 2)
    return X[i] + (f - i) * (X[i + 1] - X[i])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
