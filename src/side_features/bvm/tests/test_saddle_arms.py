"""Which arm of the saddle's X the extractive section follows.

A saddle pinch has two eigenvectors and each is followed in two directions, so
there are four arms and four ways to pair them into a middle-section body
(Brueggemann & Marquardt rules 3-5, `docs/papers/rbm_bruggemann_marquardt.md`).
Only one pair is the column: the one whose ends reach the rectifying and the
stripping profile. `anchor.py` used to rank the arms by *marched length* instead,
which picked whichever arm happened to survive longest -- and that varies per
case, so the same code drew the extractive elbow pointing a different way in each
example file (`<`, `^`, `>`).

Measured here on 2-propanol / water with an ethylene-glycol entrainer, which is
the case where the ranking was most wrong: the correct departure arm dies after
3 stages against the section's own x_EG floor while the arm running off to the
entrainer vertex survives 10.
"""

import numpy as np
import pytest

from gui.state import persistence
from gui.state.window_state import WindowState
from side_features.bvm.driver import size_column
from side_features.bvm.problem import build_problem
from side_features.bvm.sections import extractive_chain, feasible_margin
from side_features.bvm.thermo_adapter import ColumnForgeThermo

_COLX = "docs/examples/extractive_ipa_water_eg.colx"
R, EF = 2.042, 1.0
IPA, WATER, EG = 0, 1, 2


@pytest.fixture(scope="module")
def case():
    ws = WindowState()
    ws.load_from_dict(persistence.load_colx(_COLX))
    order = ws.get_species_names()
    assert order == ["2-propanol", "water", "ethylene glycol"], order
    P = ws.thermodynamics_config.pressure_in_psat_unit(ws.pressure)
    tp = ColumnForgeThermo(ws.thermodynamics_config.psat_params(order),
                           gamma_fn=ws.build_gamma_fn(order),
                           phi_fn=ws.build_phi_fn(order))
    prob = build_problem(comps=order,
                         feeds=[(np.array([0.62, 0.38, 0.0]), 100.0, 1.0)],
                         pressure=P, lk=IPA, hk=WATER, rec_lk=0.98, rec_hk=0.02,
                         x_E=np.array([0.0, 0.0, 1.0]), extractive=True,
                         max_stages=300)
    return prob, tp, P


@pytest.fixture(scope="module")
def design(case):
    prob, tp, _ = case
    return size_column(prob, tp, R=R, EF=EF)


def test_the_extractive_elbow_opens_toward_its_neighbours(design):
    """The assembled band runs rectifying-end -> saddle, not off to the entrainer.

    This is the user-visible symptom: on the ternary plot the sharp bend at the
    saddle has to face the rectifying and stripping curves. Before the fix this
    case could not size at all -- the upper junction missed by 0.279 against a
    tolerance of 0.05, because every candidate ran to an interior node instead.
    """
    assert design["feasible"], [(f.cls, f.detail) for f in design["findings"]]
    x = design["column"]["x"]
    ent, feed = design["feed_stages"]
    band = x[ent:feed]
    top, bot = band[0], band[-1]

    # top of the band sits next to where the rectifying profile arrives
    rect_end = design["profiles"]["rectifying"]["X"][-1]
    assert np.linalg.norm(top - rect_end) < 0.1, (top, rect_end)
    # and it runs the right way: IPA falls, water rises, going down the section
    assert top[WATER] < 0.1 < bot[WATER], (top, bot)
    assert bot[IPA] < top[IPA], (top, bot)


def test_the_band_sits_on_the_sections_entrainer_floor(case, design):
    """x_EG is pinned at E/L_ext across the whole extractive section.

    The section can only hold liquids with a x + bvec >= 0, which for a heavy
    entrainer is the single constraint x_EG >= E/L_ext. A body that leaves it --
    the entrainer-corner arm the old ranking preferred -- is not this section's.
    """
    prob, tp, _ = case
    from side_features.bvm.problem import overall_balance
    xD, xB, D, B = overall_balance(prob, EF)
    _, ext, _ = extractive_chain(prob, R, EF, xD, xB, D, B)
    floor = (EF * prob.feeds[0].F) / ext.L

    x = design["column"]["x"]
    ent, feed = design["feed_stages"]
    band = x[ent:feed]
    assert np.allclose(band[:, EG], floor, atol=0.02), (band[:, EG].min(),
                                                        band[:, EG].max(), floor)
    assert all(feasible_margin(ext, xi) > -1e-6 for xi in band)


def test_no_arm_runs_to_the_entrainer_vertex(design):
    """The traced interior curve must not terminate in the pure-entrainer corner.

    That end is a real invariant manifold of the section's stage map, and it is
    exactly the one no column uses: nothing on the other side of either feed is
    anywhere near it. It used to win the length ranking and take the design with
    it (ethanol/EG ended at [0.010, 0.000, 0.990]).
    """
    ext = design["profiles"]["extractive"]
    assert ext["X"][-1][EG] < 0.9, ext["X"][-1]
    assert ext["X"][0][EG] < 0.9, ext["X"][0]
