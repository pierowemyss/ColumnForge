"""Choosing the interior section's rectification body before marching it.

Three things are pinned here, all on the paper's PWG example
(`docs/examples/extractive_ipa_water_eg.colx`, Bruggemann & Marquardt):

  * a junction that is a near miss AND leaves its section no stages is rejected,
    not warned about (`driver._degenerate`);
  * the bodies PRUNE the candidate curves rather than choosing among them, which
    is a claim about what a convex hull can and cannot tell you;
  * the three anchor methods are all reachable and all report which one ran.
"""

import numpy as np
import pytest

from gui.state import persistence
from gui.state.window_state import WindowState
from side_features.bvm import bodies as B
from side_features.bvm.anchor import (interior_candidates, ray_end_candidates,
                                      section_saddles)
from side_features.bvm.connect import connect
from side_features.bvm.driver import _degenerate, _launch_stages, size_column
from side_features.bvm.march import march_section
from side_features.bvm.pinch import pinch_points
from side_features.bvm.problem import build_problem, overall_balance
from side_features.bvm.sections import extractive_chain, feasible
from side_features.bvm.thermo_adapter import ColumnForgeThermo

_COLX = "docs/examples/extractive_ipa_water_eg.colx"
R, EF, EFF = 1.5, 2.0, 0.5
IPA, WATER, EG = 0, 1, 2


@pytest.fixture(scope="module")
def case():
    ws = WindowState()
    ws.load_from_dict(persistence.load_colx(_COLX))
    order = ws.get_species_names()
    P = ws.thermodynamics_config.pressure_in_psat_unit(ws.pressure)
    tp = ColumnForgeThermo(ws.thermodynamics_config.psat_params(order),
                           gamma_fn=ws.build_gamma_fn(order),
                           phi_fn=ws.build_phi_fn(order))

    def mk(**kw):
        return build_problem(**{
            "comps": order,
            "feeds": [(np.array([0.62, 0.38, 0.0]), 100.0, 1.0)],
            "pressure": P, "lk": IPA, "hk": WATER, "rec_lk": 0.98,
            "rec_hk": 0.02, "x_E": np.array([0.0, 0.0, 1.0]),
            "extractive": True, "max_stages": 300, "efficiency": EFF, **kw})
    return mk, tp, P


@pytest.fixture(scope="module")
def geometry(case):
    """The three sections, their marched product profiles and their bodies at
    (R, E/F) = (1.5, 2.0) -- the operating point where this case has a column."""
    mk, tp, P = case
    prob = mk()
    xD, xB, D, Bq = overall_balance(prob, EF)
    rect, ext, strip = extractive_chain(prob, R, EF, xD, xB, D, Bq)
    tprof = march_section(rect, xD, tp, P, 300, efficiency=EFF, stop_sec=ext)
    bprof = march_section(strip, xB, tp, P, 300, efficiency=EFF)
    ps = pinch_points(ext, tp, P, down=True)
    mids = B.middle_bodies(ps, ext)
    tops = B.product_bodies(pinch_points(rect, tp, P), xD)
    bots = B.product_bodies(pinch_points(strip, tp, P), xB)
    return dict(prob=prob, tp=tp, P=P, rect=rect, ext=ext, strip=strip,
                tprof=tprof, bprof=bprof, pinches=ps, mids=mids, tops=tops,
                bots=bots)


def test_a_near_miss_junction_bounding_an_empty_section_is_rejected():
    """`_degenerate`: approximate AND no stages, together, and only together.

    Each half on its own is a legitimate design. An exact crossing one stage
    above the reboiler is a real high-reflux column -- that is why `_at_anchor`
    was never simply widened -- and an approximate junction inside a long section
    is the ordinary 4-component case, where two curves in the (C-1)-simplex are
    over-determined. It is the pair that means the profile never travelled to the
    junction and the section either side needs infinitely many stages.
    """
    near = {"approximate": True, "dmin": 0.026, "tol": 0.05}
    exact = {"approximate": False, "dmin": 1e-16, "tol": 0.05}
    assert _degenerate(near, 1)
    assert _degenerate(near, 0)
    assert not _degenerate(near, 12)      # a near miss mid-section is ordinary
    assert not _degenerate(exact, 1)      # a real crossing above the reboiler
    assert not _degenerate(exact, 0)


def test_the_files_own_settings_no_longer_size_on_a_one_stage_stripping_section(case):
    """The design docs/adr/0004 recorded as "a design to look at, not one to
    trust": at R = 0.9, E/F = 0.95 the column came back with 127 stages, a
    stripping section of exactly one (the reboiler), and BOTH junctions
    approximate at 0.002 and 0.026. Two near misses holding up a section with no
    stages in it is the shape `_degenerate` exists to refuse.

    Either verdict is acceptable here -- another candidate may take it -- but not
    that one.
    """
    mk, tp, _ = case
    prob = mk(rec_lk=0.999, rec_hk=0.001, efficiency=0.999)
    d = size_column(prob, tp, R=0.9, EF=0.95)
    if not d["feasible"]:
        return
    secs = d["sections"]
    approx = [j for j in d["junctions"] if j["approximate"]]
    assert not (secs["stripping"]["n"] <= 1 and len(approx) == 2), \
        (secs, d["junctions"])


def test_the_stripping_side_rule_breaks_the_tie_the_gap_cannot(geometry):
    """The hull is an over-approximation, so the GAP stops discriminating at the
    top of the ranking: three of the four bodies here score 0.0000 and the width
    of the junction says nothing about which one a profile actually runs inside.

    WHERE the lower junction lands does say something. `rank_middle_bodies`'
    `anchor_w` is the closest point's barycentric weight on the stripping body's
    x_B vertex, so a small value means contact on the side running stripping
    saddle -> stripping stable node rather than back at the product anchor -- the
    stripping profile has already travelled its pinch chain by the time the
    extractive section meets it. Measured here, with the marched junction as the
    referee:

        S- E-   gap 0.0000  anchor_w 0.1883  marches to 0.0000   <- ranked first
        S- E+   gap 0.0000  anchor_w 0.6177  marches to 0.2965
        S+ E+   gap 0.0070  anchor_w 0.6614  marches to 0.1062
        S+ E-   gap 0.2983  anchor_w 0.3201  marches to 0.0001

    Before the tie-break the winner was S- E+ (whichever came first), the one
    that marches 0.29 wide. The last row is why this is a tie-break and not a
    verdict: away from the tie the gap still inverts against marching, so
    `viable_middle_bodies` keeps pruning and the marched test still decides.
    """
    g = geometry
    ranked = B.rank_middle_bodies(g["tops"], g["mids"], g["bots"])
    assert len(ranked) == 4, ranked
    assert sum(r[0] <= B.TOUCH_TOL for r in ranked) >= 2, \
        "this case is only interesting because bodies TIE at the top"

    def marched_gap(body):
        best = np.inf
        for m in interior_candidates(g["ext"], g["tp"], g["P"], max_stages=300,
                                     efficiency=EFF, pinches=g["pinches"],
                                     body=body):
            up = connect(g["tprof"], m, g["rect"], g["tp"], g["P"],
                         eps_stage=1e-2, efficiency=EFF, strict=False)
            lo = connect(m, g["bprof"], g["ext"], g["tp"], g["P"],
                         eps_stage=1e-2, efficiency=EFF, strict=False)
            best = min(best, max(up["dmin"], lo["dmin"]))
        return best

    keep = B.viable_middle_bodies(g["tops"], g["mids"], g["bots"])
    assert len(keep) >= 2, keep
    assert min(marched_gap(g["mids"][j]) for j in keep) < 1e-3, \
        "pruning dropped every body that marches to a junction"

    # the tie is broken toward the saddle -> stable-node side of the stripping
    # body, and that is the body whose profile actually closes both junctions
    tied = [r for r in ranked if r[0] <= B.TOUCH_TOL]
    assert tied[0][6] < tied[1][6], [(r[1], r[6]) for r in tied]
    won = B.winning_middle_body(g["tops"], g["mids"], g["bots"])[0]
    assert won == keep[0] == tied[0][1], (won, keep, tied[0][1])
    assert marched_gap(g["mids"][won]) < 1e-3, \
        "the preferred body no longer marches shut -- re-measure the rule"


def test_pruning_leaves_the_body_the_column_is_actually_on(geometry):
    """Whatever survives pruning has to include a body that closes both
    junctions -- otherwise the saving has cost a design."""
    g = geometry
    keep = B.viable_middle_bodies(g["tops"], g["mids"], g["bots"])
    kept = [g["mids"][j] for j in keep]
    assert kept
    d = size_column(g["prob"], g["tp"], R=R, EF=EF)
    assert d["feasible"], [(f.cls, f.detail) for f in d["findings"]]
    assert d["sections"]["extractive"]["body_id"] in {b["id"] for b in kept}


def test_a_body_names_the_same_thing_in_both_modules(geometry):
    """`body_id` is the debugging handle: the saddle and the two arm signs, which
    is exactly what fixes a middle body, formatted the same way wherever it is
    built. A design's reported body must be one of the bodies the geometry
    offered, or the name means nothing."""
    g = geometry
    ids = {b["id"] for b in g["mids"]}
    assert len(ids) == 4, ids
    for b in g["mids"]:
        assert b["id"] == B.body_id(b["saddle"], b["s_sign"], b["e_sign"])
    d = size_column(g["prob"], g["tp"], R=R, EF=EF)
    assert d["sections"]["extractive"]["body_id"] in ids


def test_continuation_from_the_stripping_side_only_lives_marched_backwards(geometry):
    """Sec 6.2 launches from BOTH neighbours, but only one march direction used to
    be tried -- `mid_sec.dir` = sign(Delta) -- and a curve that cannot take a step
    is dropped by `driver._interior_profiles`' `n >= 2` filter. Here Delta_ext =
    D - E < 0, so the stripping profile's one feasible launch stage died at n = 1
    and the stripping neighbour contributed nothing at all.

    Both halves are pinned, because the change rests on both: the forward march
    really is dead, and the backward one really does reach further than anything
    the rectifying side offers. Note which direction lives is NOT the one physical
    reasoning suggests -- the stripping-launched curve that survives is the
    down-map one -- which is why the driver tries both rather than choosing.
    """
    g = geometry
    ext = g["ext"]
    assert ext.Delta < 0, ext.Delta          # the case is only interesting for dir < 0

    def launches(src):
        seen = []
        for k in range(src["n"]):
            x0 = src["X"][k]
            if feasible(ext, x0) and not any(np.linalg.norm(x0 - q) < 1e-3
                                             for q in seen):
                seen.append(x0)
        return seen

    def gap(x0, d):
        m = march_section(ext._replace(dir=d), x0, g["tp"], g["P"], 300,
                          efficiency=EFF)
        if m["n"] < 2:
            return np.inf
        best = np.inf
        for mm in (m, {**m, "X": m["X"][::-1], "Y": m["Y"][::-1],
                       "T": m["T"][::-1]}):
            up = connect(g["tprof"], mm, g["rect"], g["tp"], g["P"],
                         eps_stage=1e-2, efficiency=EFF, strict=False)
            lo = connect(mm, g["bprof"], ext, g["tp"], g["P"],
                         eps_stage=1e-2, efficiency=EFF, strict=False)
            best = min(best, max(up["dmin"], lo["dmin"]))
        return best

    strip_pts, rect_pts = launches(g["bprof"]), launches(g["tprof"])
    assert strip_pts and rect_pts

    assert all(not np.isfinite(gap(x0, ext.dir)) for x0 in strip_pts), \
        "the stripping launch marches in sign(Delta) now -- re-measure the rule"
    back = min(gap(x0, -ext.dir) for x0 in strip_pts)
    fwd_rect = min(min(gap(x0, d) for d in (ext.dir, -ext.dir))
                   for x0 in rect_pts)
    assert back < fwd_rect, (back, fwd_rect)


def test_continuation_launches_from_the_body_not_the_feasible_region(geometry):
    """`driver._launch_stages`: which stages of a neighbour may anchor the interior
    section. The body is the test; the section's feasible region is not.

    They are nearly complementary on an extractive section, so this is not a
    refinement -- it changes which stage is chosen. Measured here, against the
    body `rank_middle_bodies` puts first: the region admits stripping stage 0 and
    only stage 0, the REBOILER, which sits 0.167 outside that body; the body
    admits stages 1-6, none of which is in the region. At r = 2.2, E/F = 0.750,
    rec 0.999/0.001 the same split is starker -- the reboiler is 0.374 outside and
    the first stage inside the body is the third one up.

    Anchoring at the reboiler is the design that puts the feed stage on top of it,
    `anchor_w ~ 1` in `rank_middle_bodies` and the thing that rule exists to
    disfavour, so the region test was selecting for exactly the wrong stage.

    Intersecting the two is not a safe middle course: it admits nothing at all.
    """
    g = geometry
    ext, bprof = g["ext"], g["bprof"]
    win = g["mids"][B.rank_middle_bodies(g["tops"], g["mids"], g["bots"])[0][1]]

    d_body = [B.body_distance(win["vertices"], bprof["X"][k][None, :])
              for k in range(bprof["n"])]
    in_body = [k for k, d in enumerate(d_body) if d <= B.TOUCH_TOL]
    in_reg = [k for k in range(bprof["n"]) if feasible(ext, bprof["X"][k])]

    assert in_reg == [0], in_reg           # the region admits the reboiler alone
    assert 0 not in in_body, d_body[0]     # and the reboiler is NOT in the body
    assert in_body, "the body admits no stripping stage at all"
    assert not set(in_body) & set(in_reg), "intersecting the two admits nothing"

    got = _launch_stages(ext, bprof, win)
    assert [tuple(np.round(x, 9)) for x in got] <= \
        [tuple(np.round(bprof["X"][k], 9)) for k in in_body]
    assert got, "the body test must leave the stripping side something to launch"
    # and without a body it is the old region test, which multifeed still needs
    assert len(_launch_stages(ext, bprof, None)) == len(in_reg)


@pytest.mark.parametrize("method", ["saddle", "ray", "continuation"])
def test_each_anchor_method_runs_and_says_so(case, method):
    """All three combo entries reach the driver, and a design reports which one
    built its interior curve. They are NOT required to agree on feasibility --
    they do not, which is why there is no "auto" -- only to be honest about
    which ran."""
    mk, tp, _ = case
    d = size_column(mk(anchor_method=method), tp, R=R, EF=EF)
    got = d["sections"]["extractive"].get("anchor_method")
    if d["feasible"]:
        assert got == method, (method, got, d["sections"])


def test_the_s_vertex_march_starts_at_the_ray_end_and_passes_the_saddle(geometry):
    """Method 2, and the direction it turns on.

    The profile must begin at the ray end -- a boundary of the section's own
    region, not an interior point -- and then approach the saddle. It does not
    converge onto it, and is not meant to: a finite-reflux profile passes near a
    pinch and leaves down the unstable side. What it must not do is never come
    near at all.

    That is the assertion the direction bug failed. `sec.dir` is sign(D - E) and
    goes negative here (D = 62 against E = 75), so marching in it from S walks
    along the x_IPA = 0 face to the entrainer corner without seeing the saddle --
    the trap `pinch.jacobian` documents. Marching `-sec.dir` instead closes from
    0.066 to 0.013 by stage 3 of 24.
    """
    g = geometry
    cands = ray_end_candidates(g["ext"], g["tp"], g["P"], max_stages=300,
                               efficiency=EFF, pinches=g["pinches"])
    assert cands, "the ternary saddle's stable ray has two ends; both were empty"
    sad = section_saddles(g["ext"], g["tp"], g["P"], g["pinches"])
    assert sad
    xstar = np.asarray(sad[0]["xstar"], float)
    approached = []
    for prof in cands:
        X = prof["X"]
        assert prof["anchor_method"] == "ray"
        # the launch point is ON a boundary: either a simplex face or the
        # section's own balance a x + bvec >= 0
        slack = min(float(X[0].min()),
                    float(np.min(g["ext"].a * X[0] + g["ext"].bvec)))
        assert slack < 1e-6, (X[0], slack)
        d = np.linalg.norm(X - xstar, axis=1)
        approached.append(d.min() < 0.5 * d[0])
    assert any(approached), [np.round(p["X"][[0, -1]], 4) for p in cands]
