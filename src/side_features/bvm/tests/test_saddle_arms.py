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

Run at **efficiency 0.5**, deliberately. At ideal stages this operating point no
longer sizes: the rectifying section amplifies the entrainer ~45x per stage, so
it marches in 0.586- and 0.751-long segments, there is no stage anywhere near the
extractive section's entry, and the upper junction lands mid-chord 0.165 away.
That used to pass because the junction tolerance was the local step (0.63-0.75
here, i.e. vacuous); with `connect.STEP_CAP` applied to the interior path it is
correctly reported infeasible. Halving the efficiency doubles the stage count and
gives the rectifying march the resolution it needs -- the upper junction then
closes at an exact crossing -- so this is where the arm geometry can be pinned
without a tolerance argument underneath it. Fixing the ideal-stage case means
solving x_D,entrainer against the junction (`splits.solve_free_splits`); it is a free
split, not a specification, and it is still pinned at the
`Problem.entrainer_trace` seed.

**(R, E/F) moved from (2.042, 1.0) to (1.5, 2.0)** when the example file's
2-propanol record was corrected -- it had held ethanol's vapour pressure and
critical constants, boiling the light key 4 K low and putting the IPA/water
azeotrope at 0.79 instead of the measured 0.688 (see
`src/python/tests/test_ipa_water_eg_vle.py`). Real 2-propanol is the more
volatile of the two, so the same separation wants more entrainer, and E/F = 1.0
is now below the minimum: the old point reports `infeasible_entrainer` with the
sections 0.111 apart. Nothing about the arm geometry these tests pin changed --
only the operating point at which this case has a column at all.
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
R, EF, EFF = 1.5, 2.0, 0.5
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
                         max_stages=300, efficiency=EFF)
    return prob, tp, P


@pytest.fixture(scope="module")
def design(case):
    prob, tp, _ = case
    return size_column(prob, tp, R=R, EF=EF)


def test_every_saddle_is_found_not_the_one_a_seed_falls_into(case):
    """Rule 1: "calculate all saddle pinch points" (paper p.102).

    `pinch.pinch_solve` runs one Newton solve from `sections.region_center` and
    returns whichever root it lands in. On this section that root is not a saddle
    at all -- it is the unstable node -- so `anchor.interior_candidates` used to
    see `saddle=False`, return [], and the driver fell through to Sec 6.2
    continuation off a neighbouring profile. That is the "extractive section fails
    to anchor" verdict, on a section with two saddles sitting right there.

    Measured at the paper's own operating point (E/F = 0.750, r = 2.042, where
    Table 1 gives r_min = 2.042): four pinches, of which the ternary saddle at
    (0.050, 0.571, 0.380) and the face saddle at (0.621, 0.0005, 0.378). Which one
    a single seed lands on -- if either -- depends on the product spec, and at the
    file's own recoveries it lands on neither.

    `down=True` is not decoration. A middle section's Delta = D - E goes negative
    once the entrainer flow passes the distillate rate -- it does here, D = 62
    against E = 75 -- so `sec.dir` is the up map, whose eigenvalues are the
    reciprocals; reading it swaps stable for unstable and mislabels the arms.
    """
    prob, tp, P = case
    from side_features.bvm.anchor import interior_candidates
    from side_features.bvm.pinch import BRANCH_TOL, pinch_points, pinch_solve
    from side_features.bvm.problem import overall_balance

    xD, xB, D, B = overall_balance(prob, 0.75)
    _, ext, _ = extractive_chain(prob, 2.042, 0.75, xD, xB, D, B)
    assert ext.Delta < 0, ext.Delta          # so sec.dir would read the up map

    pts = pinch_points(ext, tp, P, down=True)
    saddles = [p for p in pts if p["kind"] == "saddle" and p["in_simplex"]]
    assert len(saddles) >= 2, [(np.round(p["x"], 4), p["kind"]) for p in pts]

    # one seed cannot be relied on to land on any of them. At these recoveries it
    # happens to find the ternary saddle; at the file's own bvm_params
    # (0.999/0.001) the same solve lands on the unstable node and the section
    # reported that it could not anchor. Both are the same code path -- which is
    # the argument for enumerating rather than seeding.
    seeded = pinch_solve(ext, tp, P)["xstar"]
    assert any(np.linalg.norm(seeded - p["x"]) < 1e-3 for p in pts), seeded
    sharp = build_problem(comps=prob.comps,
                          feeds=[(np.array([0.62, 0.38, 0.0]), 100.0, 1.0)],
                          pressure=P, lk=IPA, hk=WATER,
                          rec_lk=0.999, rec_hk=0.001,
                          x_E=np.array([0.0, 0.0, 1.0]), extractive=True)
    xDs, xBs, Ds, Bs = overall_balance(sharp, 0.75)
    _, ext_s, _ = extractive_chain(sharp, 2.042, 0.75, xDs, xBs, Ds, Bs)
    seeded_s = pinch_solve(ext_s, tp, P)["xstar"]
    pts_s = pinch_points(ext_s, tp, P, down=True)
    hit = next(p for p in pts_s
               if np.linalg.norm(seeded_s - p["x"]) < 1e-3)
    assert hit["kind"] != "saddle", (seeded_s, hit["kind"])
    assert sum(p["kind"] == "saddle" and p["in_simplex"] for p in pts_s) >= 2, \
        [(np.round(p["x"], 4), p["kind"]) for p in pts_s]

    # a ternary saddle exists and is the one the anchor uses (`BRANCH_TOL`); the
    # face saddle lies on a simplex face, hence on the neighbouring product
    # profile, and its junction closes on geometry that is not a connection
    tern = [p for p in saddles if p["k_gap"] <= BRANCH_TOL]
    assert len(tern) == 1, [(np.round(p["x"], 4), p["k_gap"]) for p in saddles]
    assert tern[0]["x"][WATER] > 0.5, tern[0]["x"]

    cands = interior_candidates(ext, tp, P, max_stages=prob.max_stages,
                                efficiency=prob.efficiency)
    assert cands, "two saddles in the region and no limiting profile"
    for c in cands:
        assert np.linalg.norm(c["classification"]["xstar"] - tern[0]["x"]) < 1e-6


def test_the_upper_junction_is_a_crossing_not_a_tolerance(design):
    """The rectifying and extractive profiles genuinely meet, at ~1e-16.

    The regression this guards is the one that made the whole exercise necessary:
    with the interior junction tolerance left at the local step it reached 0.686
    here, every arm of the saddle "connected" at the top, and the arm choice was
    decided by noise. A crossing is a different claim from a small distance, and
    `connect` records which one it made -- so assert on that, not on `feasible`.
    """
    assert design["feasible"], [(f.cls, f.detail) for f in design["findings"]]
    up = next(j for j in design["junctions"]
              if j["pair"] == "rectifying/extractive")
    assert not up["approximate"], up
    assert up["dmin"] < 1e-9, up


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

    # Top of the band sits where the rectifying profile arrives -- measured
    # against that profile's POLYLINE, not its last vertex. The rectifying march
    # is run with `stop_sec`, so it deliberately takes one step past the junction
    # and its final vertex is an overshoot into the entrainer corner (see
    # `march.march_section`); here it lands at (0.104, 0.004, 0.892) while the
    # stage before it is at (0.492, 0.005, 0.502) and the true hand-over is
    # halfway along that chord. Comparing to the vertex measures the overshoot,
    # which at efficiency 0.5 is 0.39 long. The polyline is what `connect`
    # intersects, so it is what the claim is about.
    rect = design["profiles"]["rectifying"]["X"]
    seg = rect[1:] - rect[:-1]
    t = np.clip(np.einsum("ij,ij->i", top - rect[:-1], seg)
                / np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-30), 0.0, 1.0)
    dist = np.linalg.norm(rect[:-1] + t[:, None] * seg - top, axis=1).min()
    assert dist < 0.02, (top, dist, rect[-2:])
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
    # One-sided, because the constraint is one-sided: x_EG >= E/L_ext is what the
    # section's balance demands, and "pinned at the floor" means the band rides
    # just above it, never below. `np.allclose(..., atol=0.02)` was a two-sided
    # test on a bounded-below quantity, and it failed on an excess of 0.020 --
    # a number that says the band is exactly where it should be.
    excess = band[:, EG] - floor
    assert excess.min() > -1e-6, (excess.min(), floor)      # never below
    assert excess.max() < 0.03, (excess.max(), floor)       # and pinned to it
    assert all(feasible_margin(ext, xi) > -1e-6 for xi in band)


def test_a_stalled_arm_is_still_walked_to_the_edge(case, design):
    """An arm that ran out of stage budget must be extended, not handed on short.

    `anchor._extend_along_ray` used to fire only for a branch that stalled on a
    `pinch`; a branch that exhausted `max_stages` came back `max` and truncated.
    Every step is scaled by the Murphree efficiency, so that is exactly what
    halving E does -- the arm ending at S1 needs more than 300 half-stages here.
    The band then started at (0.299, 0.210, 0.491) instead of (0.512, 0.000,
    0.488): 0.289 away from the rectifying hand-over, in a different wedge of the
    triangle and a different rectification body, which is the reported bug.

    Assert the PROPERTY -- both ends of the band reach a boundary -- rather than a
    named ray end. It used to compare the first extractive stage against
    `_ray_end` along the unstable eigenvector, and that is a proxy for "was it
    extended", not the thing itself. The proxy broke when the anchoring started
    obeying rule 1: the section's saddles are now all enumerated and the
    eigenvalues read off the DOWN map, so a different arm pair wins and the two
    branches are extended against different sections anyway -- the approach is
    traced backward, so its ray is cut by `sec._replace(dir=-sec.dir)`'s region,
    not `ext`'s. An arm walked to an edge is an arm with no slack left at its end,
    and that is checkable without knowing which arm it is.
    """
    prob, tp, P = case
    from side_features.bvm.problem import overall_balance

    xD, xB, D, B = overall_balance(prob, EF)
    _, ext, _ = extractive_chain(prob, R, EF, xD, xB, D, B)
    prof = design["profiles"]["extractive"]
    assert prof["status"] in ("manifold", "manifold+ray"), prof["status"]

    # a truncated arm stops in the interior with slack on every constraint; an
    # extended one is ON a face of the simplex or of the section's own region
    for name, x in (("start", prof["X"][0]), ("end", prof["X"][-1])):
        slack = min(float(np.min(x)), feasible_margin(ext, x))
        assert slack < 1e-3, f"{name} {np.round(x, 4)} stops short, slack {slack:.3g}"


def test_the_extractive_section_begins_on_the_stable_manifold(case, design):
    """It starts at S, i.e. on the arm the profile ARRIVES along (rules 3-4).

    The paper builds the middle-section body by walking the first saddle's stable
    eigenvector to the edge of composition space (S1/S2) and the last saddle's
    unstable one to the other edge (E1/E2). A marched profile is that polyline
    with curvature, so its first stage must lie on the stable eigendirection and
    its last on the unstable one -- and S is the end the rectifying section hands
    over at, which `test_the_extractive_elbow_opens_toward_its_neighbours`
    separately pins.

    Read straight off `anchor.stable_eigvec` / `unstable_eigvec`, which is the
    point: this test used to invert the eigenvalues by hand, because the old
    `jacobian_G` differentiated the one-stage march in `sec.dir`, and here
    Delta = D - E < 0 so `sec.dir` is the UP map and its labels were the
    reciprocals -- what it called stable was the direction this profile leaves
    along. `interior_candidates` now passes `down=True` explicitly
    (`pinch.jacobian`), so the labels mean what they say and a test that has to
    correct them is a test papering over the bug.
    """
    prob, tp, P = case
    from side_features.bvm.anchor import (_tangent, stable_eigvec,
                                          unstable_eigvec)
    from side_features.bvm.problem import overall_balance

    xD, xB, D, B = overall_balance(prob, EF)
    _, ext, _ = extractive_chain(prob, R, EF, xD, xB, D, B)
    assert ext.Delta < 0, ext.Delta            # the inverted case, on purpose

    cl = design["profiles"]["extractive"]["classification"]
    xstar = cl["xstar"]
    mag = np.abs(np.asarray(cl["eigvals"]))
    assert mag.min() < 1.0 < mag.max(), mag
    stable, unstable = stable_eigvec(cl), unstable_eigvec(cl)

    prof = design["profiles"]["extractive"]
    for x, vec in ((prof["X"][0], stable), (prof["X"][-1], unstable)):
        d = x - xstar
        n = np.linalg.norm(d)
        assert n > 1e-6, x
        assert abs(float(d / n @ _tangent(xstar, vec))) > 0.9, (x, xstar)


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
