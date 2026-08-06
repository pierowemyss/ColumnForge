"""Ternary extractive column: ethanol / water with an ethylene-glycol entrainer.

This is the case BVM could not size. Inside-Out converges it in 19 iterations
(xD = [0.928, 0.072, 0.000], xB = [0.099, 0.112, 0.790]) while BVM reported
`infeasible_entrainer -- closest approach 0.371 (need <= 0.013)` at every (R, E/F)
tried. None of that was thermodynamics: both solvers call the same
`core.thermodynamics.k_values` through the same `build_gamma_fn`, and matching
BVM's recoveries to Inside-Out's converged split agrees on the products to ~0.006
mole fraction.

Three things were wrong, and each has a test here:

  * the marcher CLIPPED a negative operating-line vapour to zero and renormalised,
    fabricating profiles from liquids that no stage of the section could hold;
  * the extractive section was anchored on an arbitrary stage of a neighbouring
    profile instead of on its own saddle pinch;
  * the junction test compared the wrong pair of liquids -- x_{f-1} against x_f,
    ADJACENT stages, which a feed genuinely jumps by 0.11 in entrainer content.

The fix for the third one was for a while to move the test into vapour space
entirely, and that turned out to be a worse bug: see `connect.py`. The junction
is the crossing of the two liquid profiles, at x_f, which lies on both.
"""

import numpy as np
import pytest

from side_features.bvm.driver import size_column
from side_features.bvm.splits import spectrum
from side_features.bvm.problem import build_problem
from side_features.bvm.sections import (extractive_chain, feasible,
                                        feasible_margin, region_center)
from side_features.bvm.march import march_section
from side_features.bvm.anchor import _classify
from side_features.bvm.pinch import pinch_solve
from side_features.bvm.problem import overall_balance
from side_features.bvm.thermo_adapter import ColumnForgeThermo
from gui.state.window_state import WindowState
from gui.state import persistence

_COLX = "docs/examples/extractive_ethanol_water_eg.colx"

# Inside-Out's converged split for this file, used so BVM is asked for the same
# separation rather than a different one.
REC_LK, REC_HK = 0.917, 0.405


@pytest.fixture(scope="module")
def case():
    ws = WindowState()
    ws.load_from_dict(persistence.load_colx(_COLX))
    order = ws.get_species_names()
    assert order == ["ethanol", "water", "ethylene glycol"], order
    P = ws.thermodynamics_config.pressure_in_psat_unit(ws.pressure)
    tp = ColumnForgeThermo(ws.thermodynamics_config.psat_params(order),
                           gamma_fn=ws.build_gamma_fn(order),
                           phi_fn=ws.build_phi_fn(order))
    prob = build_problem(comps=order,
                         feeds=[(np.array([0.85, 0.15, 0.0]), 100.0, 1.0)],
                         pressure=P, lk=0, hk=1, rec_lk=REC_LK, rec_hk=REC_HK,
                         x_E=np.array([0.0, 0.0, 1.0]), extractive=True,
                         max_stages=300)
    return prob, tp, P


def test_it_sizes_at_all(case):
    """The headline: a feasible ternary extractive design, entrainer above feed."""
    prob, tp, _ = case
    d = size_column(prob, tp, R=3.0, EF=1.0)
    assert d["feasible"], [(f.cls, f.detail) for f in d["findings"]]
    ent, feed = d["feed_stages"]
    assert 0 < ent < feed < d["N_total"], d["feed_stages"]
    # products agree with the Inside-Out reference this case was matched to
    assert d["xD"][0] > 0.9, d["xD"]          # ethanol past its water azeotrope
    assert d["xB"][2] > 0.75, d["xB"]         # entrainer leaves in the bottoms
    x = d["column"]["x"]
    assert np.allclose(x[0], d["xD"], atol=1e-6)
    assert np.allclose(x[-1], d["xB"], atol=1e-6)
    assert d["column"]["T"][-1] > d["column"]["T"][0]


def test_extractive_section_has_a_minimum_entrainer_content(case):
    """The feasible region is a hard constraint, not a numerical nicety.

    A section can only hold liquids whose operating-line vapour is non-negative.
    For this extractive section that is x_EG >= E/L_ext = 60/186 = 0.323, which
    excludes the entire rectifying profile and all but the reboiler end of the
    stripping one -- so the old code's launch stages were all infeasible, and it
    clipped its way through them anyway.
    """
    prob, tp, P = case
    xD, xB, D, B = overall_balance(prob, 0.6)
    rect, ext, strip = extractive_chain(prob, 1.5, 0.6, xD, xB, D, B)

    x_min = -ext.bvec[2] / ext.a
    assert 0.30 < x_min < 0.35, x_min
    assert np.isclose(x_min, 60.0 / ext.L, rtol=1e-3), (x_min, ext.L)

    assert not feasible(ext, xD), "the distillate is not an extractive liquid"
    assert feasible_margin(ext, np.array([0.3, 0.3, 0.4])) > 0
    assert region_center(ext) is not None and feasible(ext, region_center(ext))

    # and a march launched outside the region refuses to invent a profile
    bad = march_section(ext, xD, tp, P, prob.max_stages)
    assert bad["status"] == "operating_line" and bad["n"] == 1


def test_extractive_pinch_is_a_saddle_on_the_region_boundary(case):
    """Doherty & Malone's picture, measured: the extractive section is controlled
    by a saddle whose entrainer content sits at the section's balance limit. The
    profile approaches it slowly (|lambda| < 1) -- which is what makes an
    extractive section long -- and leaves fast."""
    prob, tp, P = case
    xD, xB, D, B = overall_balance(prob, 0.6)
    _, ext, _ = extractive_chain(prob, 1.5, 0.6, xD, xB, D, B)

    ps = pinch_solve(ext, tp, P)
    assert ps is not None and ps["converged"] and ps["in_region"]
    xstar = ps["xstar"]
    assert np.isclose(xstar[2], -ext.bvec[2] / ext.a, atol=5e-3), xstar

    # `down=True`: a middle section's profile runs top-to-bottom whatever
    # sign(Delta) says, and reading the up map instead reports the reciprocal
    # eigenvalues -- see `pinch.jacobian`.
    cl = _classify(ext, xstar, tp, P, down=True)
    assert cl["saddle"], (cl["kind"], np.abs(cl["eigvals"]))
    mag = np.sort(np.abs(cl["eigvals"]))
    assert mag[0] < 1.0 < mag[-1], mag


def test_the_feed_jump_is_real_but_it_is_not_the_junction(case):
    """Both sides of the main feed are pinned by their own section balances --
    x_EG = 60/186 = 0.323 above, 60/286 = 0.210 below -- so ADJACENT STAGES either
    side of the feed are ~0.11 apart in entrainer.

    That is a statement about x_{f-1} against x_f, and it was for a while read as
    "the liquid profiles cannot meet", which sent the junction test into vapour
    space. It does not follow. The feed-stage liquid x_f lies on BOTH curves --
    the vapour leaving stage f is what the section above puts its operating line
    on, so x_f = dew(y_f) whichever section computes it -- and the curves cross
    there. Measuring the junction on the vapour instead cost c2-c4 four stages of
    feed position and 22% of its R_min (see connect.py).
    """
    prob, tp, P = case
    xD, xB, D, B = overall_balance(prob, 0.6)
    _, ext, strip = extractive_chain(prob, 1.5, 0.6, xD, xB, D, B)
    x_ext = 60.0 / ext.L
    x_strip = 60.0 / strip.L
    assert x_ext - x_strip > 0.10, (x_ext, x_strip)

    # and the assembled column shows it: the extractive section sits at a constant
    # entrainer level set by its own balance, distinct from the stripping one.
    d = size_column(prob, tp, R=3.0, EF=1.0)
    assert d["feasible"], [f.cls for f in d["findings"]]
    x = d["column"]["x"]
    ent, feed = d["feed_stages"]
    band = x[ent:feed, 2]
    assert band.max() - band.min() < 0.01, band      # flat through the section
    assert band.mean() > x[0, 2] + 0.2               # and far above the distillate


def test_the_entrainer_split_is_the_design_freedom(case):
    """A ternary extractive column DOES have a one-parameter family, and the free
    parameter is the entrainer's distillate split.

    Both key recoveries are specified, so the only remaining component is the
    entrainer. Pinning it at exactly zero -- which is what this file used to
    assert -- does not make the design unique, it makes it impossible: the
    rectifying profile is then trapped on the entrainer-free face, where at R=3
    the operating line has a pinch pair straddling x_D (0.696 and 0.950) that no
    profile can cross, so the section never reaches the feed and 'connects' only
    to its own anchor.

    Solved instead, x_D,EG is determined by the junction and indexed by the feed
    position: one more rectifying stage costs a factor 1/K_EG (~100 here) in
    overhead glycol. That geometric ladder IS the family.
    """
    from side_features.bvm.problem import free_split_indices
    from side_features.bvm.splits import design_at_feed
    prob, tp, _ = case
    ent = 2
    assert free_split_indices(prob) == [ent], free_split_indices(prob)

    seen = []
    for feed_loc in (1, 2, 3):
        d, sol = design_at_feed(prob, tp, 3.0, float(feed_loc), EF=1.0)
        assert sol is not None and sol["converged"], (feed_loc, sol)
        assert sol["residual"] < 1e-6, (feed_loc, sol["residual"])
        assert d["feasible"] and d["exact"], (feed_loc, d["findings"])
        seen.append((d["feed_stages"][0], d["xD"][ent]))

    # deeper entrainer feed <=> fewer stages to strip the glycol <=> more of it
    # overhead, monotonically and by a large factor per stage
    stages = [s for s, _ in seen]
    xs = [x for _, x in seen]
    assert stages == sorted(stages) and stages[0] < stages[-1], seen
    assert all(0.0 < x < 1e-3 for x in xs), seen
    assert all(a > 10 * b for a, b in zip(xs, xs[1:])), seen


def test_the_distillate_carries_only_a_trace_of_entrainer(case):
    """x_D(entrainer) is a trace -- small enough to be a spec-grade product, and
    strictly positive so the rectifying profile can leave the entrainer-free face.

    The rectifying section that results is short and real: glycol falls by ~1/K
    per stage from the entrainer feed up to the condenser, which is what makes an
    extractive column's rectifying section a few trays rather than none.
    """
    prob, tp, _ = case
    d = size_column(prob, tp, R=3.0, EF=1.0)
    assert d["feasible"], [f.cls for f in d["findings"]]
    assert 0.0 < d["xD"][2] < 1e-3, d["xD"]
    assert d["xB"][2] > 0.75, d["xB"]

    # the rectifying section exists, and its glycol decays upward to the trace
    ent_stage = d["feed_stages"][0]
    assert ent_stage >= 1, d["feed_stages"]
    rect_eg = d["column"]["x"][:ent_stage + 1, 2]
    assert np.isclose(rect_eg[0], d["xD"][2]) and rect_eg[-1] > 0.2, rect_eg
    assert np.all(np.diff(rect_eg) > 0), rect_eg
    assert not any(w.cls == "vanished_section" for w in d["warnings"]), d["warnings"]
    # E/F = 1 doubles the material through the column and it is all accounted for
    assert abs((d["D"] + d["B"]) - 200.0) < 1e-6, (d["D"], d["B"])
    # and it agrees with the Inside-Out solution of the same file
    assert np.allclose(d["xD"], [0.9276, 0.0724, 0.0], atol=2e-3), d["xD"]


def test_no_entrainer_is_reported_not_fabricated(case):
    """E/F = 0 leaves the 'extractive' section identical to the rectifying one, so
    it connects trivially. That is a degenerate chain, not a design."""
    prob, tp, _ = case
    d = size_column(prob, tp, R=3.0, EF=0.0)
    assert not d["feasible"]
    assert any(f.cls == "infeasible_entrainer" for f in d["findings"]), d["findings"]
