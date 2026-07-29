"""RBM validation.

Three kinds of check, in increasing order of how much they can prove:

  1. internal -- the geometry does what it says (hull distances, chains, bodies);
  2. cross-method -- RBM's minimum reflux agrees with BVM's, which shares no code
     below `sections`, and with Underwood on a case where Underwood applies;
  3. against the paper -- Bruggemann & Marquardt's PWG case, structurally.

On (3), read `test_pwg_matches_the_paper_structurally` before trusting any
number: the paper's Table 1 values are NOT reproduced here and are not asserted.
See that test's docstring for exactly which claims do and do not carry over.
"""

import numpy as np
import pytest

from side_features.bvm.problem import build_problem
from side_features.bvm.sections import single_feed_chain
from side_features.bvm.thermo_adapter import ColumnForgeThermo
from side_features.rbm import bodies as B
from side_features.rbm import pinch as P
from side_features.rbm.driver import analyse, reflux_band
from side_features.rbm.pinch import pinch_points, solve_pinch

BTX = np.array([(6.90565, 1211.033, 220.79),
                (6.95464, 1344.8, 219.48),
                (6.99052, 1453.43, 215.31)])
Z = np.array([0.4, 0.35, 0.25])


def btx(**kw):
    return build_problem(["benzene", "toluene", "xylene"], [(Z, 100.0, 1.0)],
                         760.0, rec_lk=0.98, rec_hk=0.02, **kw)


@pytest.fixture(scope="module")
def tp():
    return ColumnForgeThermo(BTX)


# -- 1. the geometry ------------------------------------------------------

def test_hull_distance_is_a_real_distance():
    """Checked against answers that can be worked out by hand."""
    seg = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pt = np.array([[0.0, 0.0, 1.0]])
    assert abs(B.body_distance(seg, pt) - np.sqrt(1.5)) < 1e-6

    # a hull that reaches across the first one has zero distance
    across = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    assert B.body_distance(seg, across) < 1e-7

    # symmetric, and zero against itself
    assert abs(B.body_distance(seg, across) - B.body_distance(across, seg)) < 1e-9
    assert B.body_distance(seg, seg) < 1e-7


def test_chains_are_strictly_monotone_and_maximal():
    def pk(n, kind, x):
        return {"in_simplex": True, "n_stable": n, "kind": kind,
                "x": np.array(x, float), "eigvals": np.ones(2),
                "eigvecs": np.eye(2), "order": np.array([0, 1])}

    ps = [pk(0, "unstable_node", [1, 0, 0]), pk(1, "saddle", [0, 1, 0]),
          pk(1, "saddle", [0, 0.5, 0.5]), pk(2, "stable_node", [0, 0, 1])]
    chs = B.chains(ps)
    assert len(chs) == 2                       # one per n_stable==1 alternative
    for ch in chs:
        ns = [p["n_stable"] for p in ch]
        # strictly increasing and maximal -- but starting at 1, because a profile
        # arriving from the product can never touch an unstable node
        assert ns == [1, 2], ns
        assert all(p["kind"] != "unstable_node" for p in ch)

    # pinches outside the simplex are never used: a body has to live in
    # composition space to intersect another one there
    out = pk(1, "saddle", [1.4, -0.4, 0.0]); out["in_simplex"] = False
    assert all(out not in ch for ch in B.chains(ps + [out]))


def test_middle_bodies_follow_the_paper_eigenvector_rules():
    """Paper p.100 rules 3-5: start along the most stable eigenvector, end along
    the most unstable, both directions of each -> four bodies for one chain."""
    saddle = {"in_simplex": True, "n_stable": 1, "kind": "saddle",
              "x": np.array([0.3, 0.3, 0.4]), "eigvals": np.array([2.0, 0.5]),
              "eigvecs": np.eye(2), "order": np.array([0, 1])}
    bs = B.middle_bodies([saddle])
    assert len(bs) == 4
    for b in bs:
        v = b["vertices"]
        assert v.min() > -1e-9 and abs(v.sum(1) - 1.0).max() < 1e-6
        # the saddle itself is always a vertex, between its two extensions
        assert np.any(np.all(np.isclose(v, saddle["x"], atol=1e-9), axis=1))


# -- 2. cross-method ------------------------------------------------------

def test_pinch_points_really_solve_the_pinch_equation(tp):
    """Every pinch is a solve -- and the one exception says how far off it is.

    A `clipped` record is a face point standing in for a branch that has left the
    simplex (`pinch._relax_to_face`), so it does not solve the equation. What it
    must satisfy is the reason it was kept at all: the residual it leaves is
    exactly the difference point's own content of the off-face components, which
    is small only because the product is near-sharp there. Asserting that bound
    is what keeps `SHARP_TOL` honest -- a clipped point standing in for something
    the product is NOT sharp on would blow straight through it.
    """
    from side_features.bvm.pinch import pinch_residual
    from side_features.bvm.problem import overall_balance
    prob = btx()
    xD, xB, D, Bt = overall_balance(prob)
    rect, strip = single_feed_chain(prob, 4.0, xD, xB, D, Bt)
    for sec in (rect, strip):
        ps = pinch_points(sec, tp, 760.0)
        assert ps, sec.name
        for p in ps:
            r = np.linalg.norm(pinch_residual(sec, p["x"], tp, 760.0))
            if p["clipped"]:
                bound = P.SHARP_TOL * abs(sec.Delta / sec.V) * len(p["x"])
                assert r <= bound, (sec.name, p["x"], r, bound)
            else:
                assert r < 1e-6, (sec.name, p["x"], r)
            assert abs(p["x"].sum() - 1.0) < 1e-9


def test_a_smeared_spec_still_spans_a_two_dimensional_body(tp):
    """The regression this exists to prevent: a body collapsing to a segment.

    With a smeared product spec no pinch sits exactly on an edge -- every
    bvec_i != 0 -- so the edge families the sharp spec opens are all gone. What
    is left is the SAME pinches displaced a hair off the edges, and until
    `pinch._relax_to_face` they were simply not found: the four interior seeds
    never got near them and the face solve slid to the dominant node. Every
    stripping section in `docs/examples/` came back with one pinch, so
    `product_bodies` spanned {x_B, stable node} and the panel drew a line.

    Two straight lines are a poor stand-in for two curves, which is how a
    degenerate body turns into a wrong feasibility verdict.
    """
    from side_features.bvm.problem import overall_balance
    prob = btx()
    prob.trace_floor = prob.entrainer_trace = 0.0
    xD, xB, D, Bt = overall_balance(prob)
    # genuinely smeared: the bottoms carries a trace of every component, so its
    # bvec has no zeros and there is no edge left for the bracketing to find
    assert xB.min() > 0.0, xB

    for sec, x_prod in zip(single_feed_chain(prob, 4.0, xD, xB, D, Bt),
                           (xD, xB)):
        ps = pinch_points(sec, tp, 760.0)
        assert sum(p["in_simplex"] for p in ps) >= 2, \
            (sec.name, [np.round(p["x"], 4) for p in ps])
        bs = B.product_bodies(ps, x_prod)
        assert bs and max(len(b["vertices"]) for b in bs) >= 3, \
            (sec.name, [np.round(b["vertices"], 4) for b in bs])


def test_a_sharp_split_opens_the_pinch_ladder(tp):
    """The structural reason RBM needs the product spec verbatim.

    With a trace of every component in x_D there is ONE physical rectifying
    pinch: x_i (K_i - a) = (1-a) x_D,i forces every x_i > 0, which pins the
    temperature to a single root. Zero out a component and the edge families
    open, giving the unstable-node / saddle / stable-node ladder the body rules
    chain along. That is why `driver.analyse` switches BVM's trace floors off.
    """
    from side_features.bvm.problem import overall_balance
    smeared = btx()
    sharp = btx(xD=np.array([1.0, 0.0, 0.0]),
                xB=np.array([0.0, 0.5838, 0.4162]))

    got = {}
    for name, prob in (("smeared", smeared), ("sharp", sharp)):
        prob.trace_floor = prob.entrainer_trace = 0.0
        xD, xB, D, Bt = overall_balance(prob)
        rect, _ = single_feed_chain(prob, 4.0, xD, xB, D, Bt)
        got[name] = pinch_points(rect, tp, 760.0)

    # the sharp spec finds the extra pinch AT the product vertex -- the paper's
    # r0 -- which the smeared one cannot have, because x_D,i > 0 for every i
    # forces every pinch off the faces
    assert len(got["sharp"]) > len(got["smeared"]), \
        {k: [np.round(p["x"], 3) for p in v] for k, v in got.items()}
    assert any(p["kind"] == "unstable_node" for p in got["sharp"])
    assert not any(p["kind"] == "unstable_node" for p in got["smeared"])

    # and the ladder the body rules chain along: n_stable strictly increasing,
    # temperature-ordered, spanning unstable node -> saddle -> stable node
    ns = [p["n_stable"] for p in got["sharp"]]
    assert ns == sorted(ns) and len(set(ns)) == len(ns), ns
    assert ns[0] == 0 and ns[-1] == 2, ns


def test_rbm_minimum_reflux_matches_underwood(tp):
    """Against the exact answer, not against the other approximation.

    BTX is ideal and nearly sharply split, so Underwood's constant-alpha result
    is not a second opinion -- it is the answer, and RBM lands within 2% of it.

    This replaces a 15% RBM-vs-BVM cross-check. That check passed when RBM
    returned 1.736 and BVM 1.608, and what it was really measuring is that two
    methods can agree while both sit ~30% above the truth. BVM's minimum is the
    reflux at which two MARCHED PROFILES come within a stage of each other, a
    stricter and different condition than two reachable sets overlapping (see
    `bvm.driver.r_min`), so it reads high by construction and pinning RBM to it
    pinned RBM to that bias.
    """
    from core.shortcut import underwood_min_reflux, underwood_roots
    from side_features.bvm.problem import overall_balance

    prob = btx()
    sharp = btx()
    sharp.trace_floor = sharp.entrainer_trace = 0.0
    xD, _xB, _D, _B = overall_balance(sharp)
    _, T = tp.bubble(Z, 760.0)
    alpha = tp.Psat(T) / tp.Psat(T)[1]
    roots = underwood_roots(alpha, Z, 1.0, 0, 1)
    ref = max(underwood_min_reflux(alpha, xD, th) for th in roots)

    rbm = reflux_band(prob, tp)[0]
    assert rbm is not None
    assert abs(rbm - ref) / ref < 0.05, f"RBM {rbm:.4f} vs Underwood {ref:.4f}"


def test_rbm_sizes_a_sharp_split_that_bvm_cannot(tp):
    """The case that motivates having RBM at all.

    x_D = (1, 0, 0) traps a marched rectifying profile on a simplex face -- the
    non-distributing components can never appear, so the profile cannot reach the
    feed and BVM returns no minimum reflux. The pinch equations are algebraic and
    take the exact zeros in their stride.
    """
    from side_features.bvm.driver import r_min as bvm_r_min
    spec = dict(xD=np.array([1.0, 0.0, 0.0]),
                xB=np.array([0.0, 0.5838, 0.4162]))
    assert bvm_r_min(btx(**spec), tp) is None, "BVM unexpectedly solved this"
    rbm = reflux_band(btx(**spec), tp)[0]
    assert rbm is not None and 0.5 < rbm < 10.0, rbm


def test_the_body_gap_closes_as_reflux_rises(tp):
    """Below minimum reflux the bodies are apart and the gap shrinks toward it."""
    prob = btx()
    lo = reflux_band(prob, tp)[0]
    gaps = [analyse(prob, tp, r=f * lo)["max_gap"] for f in (0.4, 0.6, 0.85)]
    assert gaps == sorted(gaps, reverse=True), gaps
    assert analyse(prob, tp, r=lo * 1.5)["max_gap"] < B.TOUCH_TOL


def test_a_simple_column_has_no_maximum_reflux(tp):
    """More reflux never breaks an ordinary column -- the band is open above.

    Now true on the smeared recovery spec as well, and that is a fix rather than
    a loosened test. This used to pin the opposite: a spurious upper edge near
    R = 4.5, because above some reflux the sections stopped finding all their
    pinches, the bodies degenerated to short segments, and two segments stop
    crossing while the curved profiles they stand in for still do. The cause was
    the pinch solve, not the specification -- a pinch that migrates along an edge
    as reflux rises used to slide past the fixed seeds. `pinch._edge_pinches`
    brackets them instead, and both specs now report the open band an ordinary
    column has.
    """
    sharp = dict(xD=np.array([1.0, 0.0, 0.0]),
                 xB=np.array([0.0, 0.5838, 0.4162]))
    lo, hi = reflux_band(btx(**sharp), tp)
    assert lo is not None and hi is None, (lo, hi)

    lo_s, hi_s = reflux_band(btx(), tp)
    assert lo_s is not None and hi_s is None, (lo_s, hi_s)


def test_body_gap_is_reported_even_where_the_verdict_is_coarse(tp):
    """Whatever the verdict, the distance behind it is a real number the caller
    can inspect -- there is no hidden tolerance deciding feasibility on its own."""
    a = analyse(btx(), tp, r=0.5)
    assert not a["feasible"] and np.isfinite(a["max_gap"]) and a["max_gap"] > 0
    assert all(np.isfinite(g["distance"]) for g in a["gaps"])
    assert all(g["active"][0] is not None for g in a["gaps"])


# -- 3. against the paper -------------------------------------------------

_PWG = "docs/examples/extractive_ipa_water_eg.colx"


@pytest.fixture(scope="module")
def pwg():
    from gui.state import persistence
    from gui.state.window_state import WindowState
    ws = WindowState()
    ws.load_from_dict(persistence.load_colx(_PWG))
    order = ws.get_species_names()
    assert order == ["2-propanol", "water", "ethylene glycol"], order
    P = ws.thermodynamics_config.pressure_in_psat_unit(ws.pressure)
    provider = ColumnForgeThermo(ws.thermodynamics_config.psat_params(order),
                                 gamma_fn=ws.build_gamma_fn(order),
                                 phi_fn=ws.build_phi_fn(order))
    prob = build_problem(comps=order,
                         feeds=[(np.array([0.62, 0.38, 0.0]), 100.0, 1.0)],
                         pressure=P, lk=0, hk=1, rec_lk=0.999, rec_hk=1e-6,
                         x_E=np.array([0.0, 0.0, 1.0]), extractive=True,
                         max_stages=300)
    return prob, provider


def test_pwg_matches_the_paper_structurally(pwg):
    """Bruggemann & Marquardt's isopropanol / water / ethylene-glycol column,
    at their reported operating point E/F = 0.750, r = 2.042 (Table 1).

    WHAT REPRODUCES -- the structure the method is built on:
      * three sections, each with a solvable pinch map;
      * the extractive section is controlled by ternary SADDLE pinches, which the
        paper calls the prerequisite for a feasible extractive separation (p.84);
      * each saddle chain yields four middle-section bodies (rule 5);
      * the rectifying body collapses onto the alcohol/glycol face, as in the
        paper's Figure 5 (left), because x_D has exact zeros;
      * the design is feasible at the paper's operating point.

    WHAT DOES NOT -- and is deliberately not asserted: the paper's numbers,
    (E/F)_min = 0.649, r_min = 2.042, r_max = 4.084. Ours are well below these
    (r_min ~ 0.5 at E/F = 0.750), for two reasons stated in the module docs
    rather than papered over. First thermodynamics: the paper uses Wilson with
    parameters from Aspen, this repo has no Wilson binaries for the glycol pairs
    and falls back to UNIFAC, and a pinch map's bifurcation structure is exactly
    what is sensitive to that. Second the flow model: `sections` assumes constant
    molar overflow while the paper's pinch equations carry the energy balance
    (their eq. 6). The SHAPE of the operating region is what carries over, and
    `test_pwg_reflux_band_closes_as_entrainer_falls` is where that is checked.
    """
    prob, provider = pwg
    a = analyse(prob, provider, r=2.042, EF=0.750)

    assert [s["name"] for s in a["sections"]] == \
        ["rectifying", "extractive", "stripping"]
    # near-sharp distillate: the keys' recoveries put water at 1e-6, glycol is
    # exactly absent because it is fed below the rectifying section
    assert a["xD"][0] > 0.999 and a["xD"][1] < 1e-5, a["xD"]
    assert a["xD"][2] == 0.0, a["xD"]
    assert a["xB"][2] > 0.5, a["xB"]              # glycol leaves in the bottoms

    ext = next(s for s in a["sections"] if s["name"] == "extractive")
    saddles = [p for p in ext["pinches"] if p["kind"] == "saddle"]
    assert saddles, [p["kind"] for p in ext["pinches"]]
    assert all(p["in_simplex"] for p in saddles)
    assert len(ext["bodies"]) == 4 * len(saddles)   # rule 5

    rect = next(s for s in a["sections"] if s["name"] == "rectifying")
    face = rect["bodies"][0]["vertices"]
    assert np.allclose(face[:, 1] * face[:, 2], 0.0, atol=1e-6), face

    assert a["feasible"], [(g["pair"], g["distance"]) for g in a["gaps"]]


def test_the_rectifying_body_runs_to_the_glycol_vertex_not_the_water_one(pwg):
    """The paper's Figure 5 (left) at the panel's DEFAULT 98/2 recovery spec.

    The rectifying body is a thin region from pure isopropanol to pure glycol.
    What this repo drew instead was a line from pure isopropanol to pure water,
    for two compounding reasons, both fixed:

      * the near-glycol stable node was never found -- with a smeared spec it
        sits at (0.0077, 0.0002, 0.9921), just off the IPA/glycol edge, and only
        `pinch._relax_to_face` gets near it. Without that vertex there was no
        n_stable == 2 pinch at all and the chains were single saddles;
      * the surviving near-water saddle spans a body whose edge runs straight
        through the IPA/water azeotrope pinch, an unstable node -- unreachable,
        and dropped by `bodies.blocked_by_unstable_node`.

    Asserted on the smeared spec on purpose: the sharp one hides the bug, since
    exact zeros put the pinches ON the edges where the bracketing finds them.
    """
    prob, provider = pwg
    smeared = build_problem(comps=list(prob.comps),
                            feeds=[(np.array([0.62, 0.38, 0.0]), 100.0, 1.0)],
                            pressure=prob.pressure, lk=0, hk=1,
                            rec_lk=0.98, rec_hk=0.02,
                            x_E=np.array([0.0, 0.0, 1.0]), extractive=True,
                            max_stages=300)
    a = analyse(smeared, provider, r=3.0, EF=0.750)
    rect = next(s for s in a["sections"] if s["name"] == "rectifying")

    assert len(rect["bodies"]) == 1, \
        [np.round(b["vertices"], 4) for b in rect["bodies"]]
    V = rect["bodies"][0]["vertices"]
    assert V[:, 2].max() > 0.9, V        # reaches the glycol vertex
    assert V[:, 1].max() < 0.05, V       # and never the water one


@pytest.mark.xfail(reason="extractive middle section pinches on the water-free "
                          "face instead of at a ternary saddle; open, see the "
                          "docstring", strict=True)
def test_pwg_reflux_band_closes_as_entrainer_falls(pwg):
    """The shape of the paper's Figure 9, which is what does carry over.

    An extractive column has a MAXIMUM reflux as well as a minimum -- too much
    reflux dilutes the entrainer out of the middle section -- and the band
    between them narrows as entrainer flow falls, pinching shut at (E/F)_min.
    Here: no upper edge below the ceiling at E/F = 0.75, an upper edge at 0.40,
    and a much tighter one at 0.10.

    XFAIL, and deliberately not repaired by loosening the claim, because the
    claim is right and the method is not yet meeting it. What is known:

    * The only saddle the extractive section reports is on the WATER-FREE face,
      e.g. (0.756, 0, 0.244) at E/F = 0.40. The paper's is a ternary saddle
      (p.72, p.116) and its being ternary is the stated reason the middle section
      can join the other two. A face saddle exists mathematically here because
      x_D carries essentially no water, so the middle section's `bvec` water
      component is zero and the water-free face is invariant.
    * Rules 3-4 then walk that saddle's eigenvectors along its own face, so the
      bodies run from the pure-IPA vertex clear across to the far edge. Bodies
      that large intersect at every reflux, which is why no upper edge appears
      and why r_min stops responding to entrainer flow (0.1 and 0.4 give the
      same 0.628).
    * Filtering the middle section to ternary pinches is NOT the fix: there is
      then no saddle at all in this model, the middle section gets no body, and
      the paper's own operating point (E/F = 0.750, r = 2.042) reads infeasible.

    So the open question is whether the ternary saddle is genuinely absent under
    this repo's UNIFAC + constant-molar-overflow model -- the fixture already
    documents that the paper's Wilson + energy-balance numbers are not
    reproduced, r_min coming out near 0.5 against their 2.042 -- or whether it
    exists and is being missed. Denser interior seeding does not find one.

    This test DID pass before the `pinch.jacobian` boundary fix, and that is not
    evidence it was right: every PWG pinch sits on a face, the old derivative
    differenced across the simplex boundary there, and the upper edge it produced
    came out of stability labels computed from non-physical compositions.
    """
    prob, provider = pwg
    # coarse on purpose: each band is ~20 pinch-map solves through UNIFAC, and
    # this test is about the ORDERING of the upper edges, not their precision
    bands = {ef: reflux_band(prob, provider, EF=ef, r_lo=0.5, r_hi=10.0,
                             n_scan=6, tol=0.05)
             for ef in (0.10, 0.40, 0.75)}
    for ef, (lo, hi) in bands.items():
        assert lo is not None, (ef, bands)

    assert bands[0.75][1] is None, bands            # open at generous entrainer
    assert bands[0.40][1] is not None, bands        # closes as it falls
    assert bands[0.10][1] is not None, bands
    assert bands[0.10][1] < bands[0.40][1], bands   # and keeps closing

    for ef, (lo, hi) in bands.items():
        if hi is not None:
            assert hi > lo, (ef, lo, hi)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
