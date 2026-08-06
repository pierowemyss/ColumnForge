"""Design driver -- size one column, sweep the design map (blueprint Sec 10).

`size_column` runs the whole method at one operating point: overall balance ->
difference-point chain -> march product-anchored sections -> anchor/march
interior sections -> connect (closest approach) -> place feeds/draws -> assemble
the full top->bottom column -> classify feasibility. `feasibility_map` sweeps
(R, S, E/F) and records feasibility + stage count on the grid so the UI can draw
a heatmap and let the user click a point to load that column.

Stage 0 = distillate (top), matching the ColumnForge GUI convention.
"""

from dataclasses import replace
from functools import partial

import numpy as np

from . import reactive
from .problem import overall_balance, SideDraw, free_split_indices
from .sections import (single_feed_chain, extractive_chain, multifeed_chain,
                       feasible)
from .march import march_section
from . import connect as CN
from .connect import connect
from .place import side_draw_stage
from .anchor import interior_candidates, ray_end_candidates
from .bodies import (TOUCH_TOL, body_distance, middle_bodies, product_bodies,
                     viable_middle_bodies)
from .diagnostics import classify, Finding
from .parallel import pmap
from .pinch import bisect_min, feasible_band, pinch_points


def _concat(top_prof, top_n, bot_prof_rev, bot_n, secs_top, secs_bot):
    """Join a top (marched down) and bottom (marched up, reversed) profile.

    Returns full-column x, y, T, L, V arrays top->bottom and the feed stage index.
    Section flows L,V are constant within a section.

    The junction stages are NOT shared: `top_n` is the last stage above the feed
    and `bot_n` the feed stage itself, which the feed-jump criterion (connect.py)
    makes two distinct compositions rather than one point counted twice.
    """
    xt = top_prof["X"][:top_n + 1]
    Tt = top_prof["T"][:top_n + 1]
    # The two marches store Y with different meanings: marching DOWN, Y[k] is
    # a*X[k]+bvec, the vapour leaving stage k+1; marching UP, Y[k] is the vapour
    # leaving stage k. Stacking them raw shifted the whole rectifying half one
    # stage down in the MESH warm start (`handoff.to_solver`'s y0). Shift it back
    # and give stage 0 the total condenser's vapour, which is x_D.
    yt = np.vstack([top_prof["X"][:1], top_prof["Y"][:top_n]])
    # bottom profile is anchored at the product and marched up; reverse to top->bottom
    xb = bot_prof_rev["X"][:bot_n + 1][::-1]
    Tb = bot_prof_rev["T"][:bot_n + 1][::-1]
    yb = bot_prof_rev["Y"][:bot_n + 1][::-1]
    x = np.vstack([xt, xb])
    T = np.concatenate([Tt, Tb])
    y = np.vstack([yt, yb])
    Lt, Vt = secs_top.L, secs_top.V
    Lb, Vb = secs_bot.L, secs_bot.V
    L = np.concatenate([np.full(len(xt), Lt), np.full(len(xb), Lb)])
    V = np.concatenate([np.full(len(xt), Vt), np.full(len(xb), Vb)])
    feed_stage = len(xt)          # first stage of the lower section = the feed stage
    return x, y, T, L, V, feed_stage


#: Largest temperature *fall* going down the column that is still read as physical.
#: An extractive section can genuinely dip a fraction of a degree while the
#: entrainer holds T nearly constant and the light key is displaced (extract_col
#: dips 0.07 K over eight stages), but a real column's temperature does not fall
#: by tens of degrees and come back.
T_INVERSION_TOL = 5.0

#: The same allowance for the one step that CROSSES A FEED, where a temperature
#: fall is not evidence of anything wrong. An extractive column's main feed is
#: 100 kmol/h of ~80 C liquid entering a 134 C stage -- some 40% of the liquid
#: traffic below it -- and it drops that tray 15 K. Judging that step by the
#: within-section number rejected valid designs (ipa/water/EG at R=0.9,
#: E/F=3.5 stalls at exactly 15.3 K), while the failure this check exists for --
#: an arm joining sections that are 40-90 K apart, as in `_reject_inverted`'s
#: 79 -> 187 -> 98 -> 140 C example -- clears 25 K several times over.
T_FEED_INVERSION_TOL = 25.0


def _temperature_inversion(T, skip=()):
    """Largest downward step in a top->bottom temperature profile (0 if monotone).

    `skip` names diff indices to leave out -- the steps that cross a feed, which
    `_inversion_verdict` judges against `T_FEED_INVERSION_TOL` instead.
    """
    if T is None or len(T) < 2:
        return 0.0
    d = np.diff(np.asarray(T, float))
    if skip:
        d = np.delete(d, [i for i in skip if 0 <= i < len(d)])
    return float(max(0.0, -np.min(d))) if len(d) else 0.0


def _inversion_verdict(T, feed_idx=()):
    """(ok, drop, where) for an assembled top->bottom temperature profile.

    Two allowances, because two different things are being tested. WITHIN a
    section the profile is one continuous march and nothing physical makes it
    fall by degrees, so `T_INVERSION_TOL` stays tight. ACROSS a feed the fall is
    ordinary mixing and needs the looser `T_FEED_INVERSION_TOL`. Reporting which
    of the two tripped matters as much as the number: "falls 15 K at the feed" and
    "falls 15 K mid-section" are different diagnoses with different fixes.
    """
    feed_idx = tuple(feed_idx)
    inner = _temperature_inversion(T, skip=feed_idx)
    if inner > T_INVERSION_TOL:
        return False, inner, "within a section"
    if T is None or len(T) < 2:
        return True, inner, ""
    d = np.diff(np.asarray(T, float))
    at_feed = max([0.0] + [-float(d[i]) for i in feed_idx if 0 <= i < len(d)])
    if at_feed > T_FEED_INVERSION_TOL:
        return False, at_feed, "across a feed"
    return True, max(inner, at_feed), ""


def _stage_budget(prob):
    """Largest assembled column this problem will accept (`Problem.max_column_stages`)."""
    cap = getattr(prob, "max_column_stages", None)
    return int(prob.max_stages if cap is None else cap)


def _reject_oversized(out, prob, section):
    """Refuse a design that only closes after an uneconomic number of stages.

    A pinched section reaches its neighbour eventually -- that is what a pinch
    is -- so "the profiles meet" on its own does not say a column exists. On
    ipa/water/EG at r=1.72, E/F=1 the extractive section met both neighbours after
    236 stages, of which the last fifty moved ~0. Nobody builds that; the operating
    point is simply sitting on top of R_min.

    Kept SEPARATE from the geometric verdict, and the finding says so: "the
    profiles cross but you need 354 trays" is a useful reading of where you are in
    the design space, not a failure to compute. `_reject_inverted` is the same
    shape of late veto for a different impossibility.
    """
    n = out.get("N_total")
    cap = _stage_budget(prob)
    if not n or n <= cap:
        return out
    out["feasible"] = False
    out["findings"] = [Finding(
        "too_many_stages", section,
        f"the sections do meet, but only after {int(n)} stages against a budget of "
        f"{cap}: at this operating point the column is on top of its minimum "
        f"reflux and the stage count is not economic. Raise R or E/F, or raise "
        f"Problem.max_column_stages if the count itself is the answer you want.")]
    out["N_total"] = None
    out["column"] = None
    out["feed_stages"] = []
    return out


def _reject_inverted(out, findings_section):
    """Turn a physically impossible assembled profile into an honest verdict.

    A column assembled across a junction that the tolerance accepted can still be
    nonsense: with an arbitrary trace of entrainer in x_D the rectifying profile
    amplifies down to the entrainer corner, and a junction found *there* produces
    a "feasible" design running 79 -> 187 -> 98 -> 140 C. Nothing else in the
    chain notices, because every individual section is internally consistent --
    only the assembled profile shows it.
    """
    col = out.get("column")
    if col is None:
        return out
    # the step INTO a feed stage is the one that crosses that feed
    feed_idx = [int(f) - 1 for f in out.get("feed_stages") or () if int(f) > 0]
    ok, drop, where = _inversion_verdict(col["T"], feed_idx)
    if ok:
        return out
    out["feasible"] = False
    out["findings"] = [Finding(
        "profile_inverted", findings_section,
        f"assembled temperature falls {drop:.1f} K {where} going down the "
        "column; the junction joins sections that do not belong to one column")]
    out["N_total"] = None
    out["column"] = None
    out["feed_stages"] = []
    return out


def _base(prob, xD, xB, D, B, R):
    f = prob.z_total
    return {"feasible": False, "findings": [], "warnings": [],
            "comps": list(prob.comps),
            "pressure": prob.pressure, "xD": xD, "xB": xB, "D": D, "B": B,
            "lk": prob.lk, "hk": prob.hk, "feed_z": f / f.sum(),
            "R": R, "S": None, "EF": None, "R_min": None, "EF_min": None,
            "N_total": None, "feed_stages": [], "column": None,
            "sections": {}, "profiles": {}, "connection": None, "junctions": []}


def _dP(prob):
    return float(getattr(prob, "dP", 0.0) or 0.0)


def _P_bot(prob):
    """Reboiler pressure. `size_column` resolves `Problem.P_bot` from the stage
    count it just measured; before that (and at dP = 0) the column is flat."""
    pb = getattr(prob, "P_bot", None)
    return prob.pressure if pb is None else float(pb)


def _P_mid(prob):
    """Anchor pressure for an INTERIOR section.

    ponytail: the midpoint of the ramp. An interior section is anchored at a
    saddle whose stage index is not known until the junction is placed, so there
    is no exact answer here; over a realistic drop (~0.007 bar/stage) this is
    within half the column's total drop of the truth. Upgrade path: re-march the
    interior curve once the feed stages are known, same refinement `size_column`
    already does for the reboiler end.
    """
    return 0.5 * (prob.pressure + _P_bot(prob))


def _vanished_warning(name, conn, which):
    """Finding describing a section the junction left with no stages.

    Carried in `design["warnings"]`, not `findings`: the design is still returned
    and still hands off, because a junction pinned at an anchor is sometimes the
    honest answer (a feed on the top tray) and sometimes a fiction (the profile
    never travelled and the match is the anchor itself). Which one it is depends
    on thermodynamics the connection test cannot see, so BVM reports the numbers
    and lets the reader judge. Vetoing outright would have failed columns that
    used to size.
    """
    return Finding(
        "vanished_section", name,
        f"junction pinned at this section's own anchor ({which}="
        f"{conn[which]:.2f}), so it contributes no stages: the profile recedes "
        f"from the junction rather than reaching it. Crossing gap "
        f"{conn['dmin']:.3g}, junction residual (E) "
        f"{conn.get('residual_vapour', float('nan')):.3g}.")


def _junction_report(pairs, warnings):
    """Record what each junction actually closed to, and say so when it is a miss.

    `connect` returns `approximate` whenever it accepted a near miss rather than a
    crossing, and until now nothing downstream carried that: the design said
    `connected` and the panel drew a column, whether the two sections met at 1e-16
    or at 0.289. Both numbers go into `design["junctions"]`, and an approximate one
    also raises a warning, because a reader deciding whether to trust a stage count
    needs to know which of those they are looking at.
    """
    rows = []
    for name, conn in pairs:
        if conn is None:
            continue
        rows.append({"pair": name, "dmin": conn["dmin"], "tol": conn["tol"],
                     "approximate": bool(conn.get("approximate"))})
        if conn.get("approximate"):
            warnings.append(Finding(
                "approximate_junction", name,
                f"the two profiles do not cross here: closest approach "
                f"{conn['dmin']:.3g} in liquid mole fraction, accepted because it "
                f"is inside one stage of travel ({conn['tol']:.3g}). The feed "
                f"location and the stage counts either side of it are only as good "
                f"as that gap."))
    return rows


def _at_anchor(n):
    """True when a junction index sits exactly on its profile's first vertex.

    `connect` minimises over segment pairs, so the answer is either INTERIOR to
    some segment or pinned at an endpoint. Pinned at index 0 means the closest
    approach is the anchor itself and the distance only grows from there: the
    profile is receding from the junction, not arriving at it. Whatever matched
    did so because the anchor happens to lie near the other curve, not because a
    profile connects the two.

    Note this is NOT the same as "the section came out short". A stripping section
    of one stage at high reflux is a real column with the feed just above the
    reboiler, and there the junction is interior (n ~ 0.3) -- that case must keep
    working. Only an exact endpoint hit is the pathology.
    """
    return float(n) <= 1e-9


def _degenerate(conn, n_used):
    """A junction that closed on an anchor rather than on a profile.

    Two things have to be true at once. The junction is a NEAR MISS -- `connect`
    returned `approximate`, so the two curves never crossed and the match was
    accepted only because it fits inside one stage of travel -- and the section
    it bounds contributes essentially nothing.

    Either alone is legitimate, which is why `_at_anchor` is not simply widened.
    An exact crossing one stage above the reboiler is a real high-reflux column,
    and an approximate junction in the middle of a long section is the ordinary
    4-component case, where two curves in the (C-1)-simplex are over-determined
    and cannot be asked to meet. Together they mean the match happened because an
    anchor happens to sit near the other curve: the profile never travelled to
    the junction, so the section either side of it needs infinitely many stages.

    This is what the extractive body containing the x_B anchor produces -- a feed
    stage right above the reboiler -- and it is the shape `docs/adr/0004` names as
    the reason the region clip in `anchor._ray_end` cannot be relaxed.
    """
    return bool(conn.get("approximate")) and n_used <= 1


def _section_report(profiles, assembled):
    """Per-section `n` = stages the assembled COLUMN uses; `n_marched` = how far
    the profile was traced.

    These differ, often wildly, and reporting the marched length as if it were the
    column hid a 33-stage rectifying profile contributing exactly one stage. The
    marched length still matters for debugging (it says where the profile ended and
    why), so keep both rather than swapping one lie for another.
    """
    out = {}
    for k, v in profiles.items():
        row = {"n": assembled.get(k, v["n"]), "n_marched": v["n"],
               "status": v["status"]}
        # only an interior section has these: which of the three anchor methods
        # built the curve, and which rectification body it turned at. Both are
        # choices the design depends on and neither was visible in the result --
        # `body_id` is the name RBM uses for the same body, so the two modules'
        # answers can be compared instead of guessed at.
        if v.get("anchor_method"):
            row["anchor_method"] = v["anchor_method"]
        if v.get("body_id"):
            row["body_id"] = v["body_id"]
        out[k] = row
    return out


def _size_two(prob, tp, rect, strip, xD, xB, P, out, forced=None):
    """Single-feed (two product-anchored sections) sizing + assembly."""
    dP = _dP(prob)
    # ponytail: no `stop_sec` here. A stripping section's region, {x : x_i >=
    # (B/L) x_B,i}, is entered well BEFORE the profiles actually cross, so
    # truncating there cuts off the junction itself -- BTX at R=8 lost its
    # nA=9.26 crossing and settled for a far worse one at 1.54 (N 13 -> 25). The
    # cut only marks the junction for an EXTRACTIVE middle section, whose region
    # is the sharp x_E >= E/L constraint; that is the one place it is wired.
    rprof = march_section(rect, xD, tp, P, prob.max_stages,
                          efficiency=prob.efficiency, dP=dP, P_lim=_P_bot(prob))
    sprof = march_section(strip, xB, tp, _P_bot(prob), prob.max_stages,
                          efficiency=prob.efficiency, dP=dP, P_lim=P)
    profiles = {"rectifying": rprof, "stripping": sprof}
    # A reactive column marches in TRANSFORMED coordinates, where the reduced
    # non-key split is pinned at its trace floor and holds the rectifying profile
    # on a face of the reduced simplex: MTBE's two profiles stay 0.22 apart at
    # every reflux, so there is no crossing to demand. Near-miss + `approximate`,
    # same as the interior-arm and C >= 4 paths (see connect.connect).
    strict = not isinstance(tp, reactive.ReactiveThermo)
    # step_cap only where the near miss is a resolution problem. A reactive
    # column's profiles are held apart by the transform itself, so capping the
    # tolerance there does not sharpen the junction, it deletes it (see `connect`).
    conn = connect(rprof, sprof, rect, tp, P, eps_stage=prob.eps_stage,
                   efficiency=prob.efficiency, strict=strict,
                   step_cap=None if not strict else CN.STEP_CAP)
    both_pinched = rprof["pinched"] and sprof["pinched"]
    side = side_draw_stage(sprof, prob.side_draws[0]) if prob.side_draws else None
    feasible, findings = classify(profiles, conn, both_pinched=both_pinched,
                                  side_draw=side)
    if forced is not None:
        # junction already solved for (solve_free_splits); don't let the tolerance-based
        # test overrule an exact solution.
        conn = {**conn, "nA": float(forced[0]), "nB": float(forced[1]),
                "connected": True}
        feasible, findings = True, []
    top_n, bot_n = int(np.floor(conn["nA"])), int(np.floor(conn["nB"]))
    if _at_anchor(conn["nA"]):
        out["warnings"].append(_vanished_warning("rectifying", conn, "nA"))
    if _at_anchor(conn["nB"]):
        out["warnings"].append(_vanished_warning("stripping", conn, "nB"))
    out["feasible"] = feasible; out["findings"] = findings
    if feasible:
        out["junctions"] = _junction_report([("rectifying/stripping", conn)],
                                            out["warnings"])
    out["sections"] = _section_report(profiles, {"rectifying": top_n + 1,
                                                 "stripping": bot_n + 1}
                                      if feasible else {})
    out["profiles"] = profiles; out["connection"] = conn
    if not feasible:
        return out
    # floor on BOTH sides: the junction sits at a fractional position between two
    # stages, so the last stage above the feed is the integer below nA and the
    # feed stage is the integer below nB. Rounding either one up steps past the
    # junction and re-covers it from the other side -- which showed up as a
    # non-monotone temperature at the feed (101.4 -> 92.3 -> 95.2 on BTX).
    x, y, T, L, V, feed_stage = _concat(rprof, top_n, sprof, bot_n, rect, strip)
    out["N_total"] = len(x); out["feed_stages"] = [feed_stage]
    if side is not None:
        out["side_draw_stage"] = feed_stage + 1 + side["stage"]
    out["column"] = {"x": x, "y": y, "T": T, "liquid_flow": L, "vapor_flow": V,
                     "feed_stage": feed_stage}
    return _reject_oversized(_reject_inverted(out, "connection"), prob,
                             "connection")


def _both_orientations(mprof):
    """An interior curve and its reverse -- offer both, let the junctions choose.

    Which end of an interior curve is 'up' cannot be read off sign(Delta) (the
    down- and up-maps are exact inverses, and Delta_ext = D - E flips sign as the
    entrainer flow crosses the distillate rate), and guessing it from proximity to
    the upper profile was worse than useless: that profile is marched with
    `stop_sec`, so its last vertex is a one-stage overshoot deep into the entrainer
    corner -- exactly where a wrong arm points. `_size_three` already enforces the
    real constraint, i_hi >= i_lo, so the honest move is to hand it both and pay
    one extra junction test. Reversal is free; nothing is re-marched.
    """
    rev = {**mprof, "X": mprof["X"][::-1], "Y": mprof["Y"][::-1],
           "T": mprof["T"][::-1]}
    if "P" in mprof:
        rev["P"] = mprof["P"][::-1]
    if "pinch_index" in mprof:
        rev["pinch_index"] = mprof["n"] - 1 - mprof["pinch_index"]
    return [mprof, rev]


def _viable_bodies(prob, tp, top_sec, mid_sec, bot_sec, tprof, bprof, P_mid):
    """Middle rectification bodies worth marching inside. Returns (bodies, pinches).

    Bodies are cheap -- eigenvectors, two rays, a convex-hull distance -- and they
    say which arm of which saddle can reach both neighbours without marching
    anything. Each surviving body fixes a saddle and both arm signs, so
    `interior_candidates` builds one curve per body instead of the eight it used
    to enumerate per saddle.

    They PRUNE rather than decide: the hull contains points no profile visits, so
    several bodies routinely tie at a gap of zero, and among those the ordering
    says nothing about which one MARCHES to a junction --
    `bodies.winning_middle_body` records the measurement. What survives goes to
    the junction test on the marched curve, which is the only thing that can tell
    them apart.

    An empty list means the section spans no body at all -- no ternary saddle, no
    feasible extractive separation (paper p.84). The caller falls back to
    enumerating saddles, which is what BVM did everywhere before this, so a
    face-saddle section still gets its chance.
    """
    ps = pinch_points(mid_sec, tp, P_mid, down=True)
    mids = middle_bodies(ps, mid_sec)
    if not mids:
        return [], ps
    # the product-anchored neighbours: their profiles start AT the product, so
    # the anchor is the composition already marched from.
    tops = product_bodies(pinch_points(top_sec, tp, prob.pressure), tprof["X"][0])
    bots = product_bodies(pinch_points(bot_sec, tp, _P_bot(prob)), bprof["X"][0])
    return [mids[j] for j in viable_middle_bodies(tops, mids, bots)], ps


def _launch_stages(mid_sec, src, body):
    """Stages of a neighbouring profile a continuation anchor may start from.

    With a middle rectification body given, the test is membership in THAT body:
    the body is the hull of the limiting profile S -> x* -> E, so a stage outside
    it is a stage the interior section does not run through, and continuing from
    it builds a curve inside a different body than the one the junctions were
    scored against. Which body is not a free choice -- `_viable_bodies` has
    already pruned to the ones that reach both neighbours, and `bodies.
    rank_middle_bodies` orders those by where the lower junction lands.

    Without a body -- an ordinary multifeed intermediate has no saddle and so no
    body at all -- it falls back to the section's own feasible region, the hard
    balance constraint `sections.feasible_margin` describes.

    The two tests are NOT interchangeable, and on an extractive section they are
    very nearly complementary. On ipa/water/EG at r = 2.2, E/F = 0.750 the region
    is x_EG >= 0.355 and the winning body is
    x* = (0.0445, 0.5939, 0.3616) S- E+:

        stripping stage 0 (reboiler)  x_EG 0.664   in region     0.374 from body
        stripping stage 1             x_EG 0.282   out           0.054 from body
        stripping stage 2             x_EG 0.246   out           INSIDE

    So the region test admits exactly one stage, the reboiler, and it is the one
    stage in the wrong body -- the anchor that puts the feed stage on top of the
    reboiler, which `bodies.rank_middle_bodies` calls `anchor_w ~ 1` and exists to
    disfavour. The body test admits stage 2 upward, the third stage up from the
    reboiler, and the curve it launches closes the lower junction exactly (0.000
    against the reboiler anchor's 0.353).

    That the two disagree is the over-reach `docs/adr/0004` records: `bodies.
    _to_edge` walks an arm to the edge of composition space, so a body legitimately
    extends past the section's own balance. Intersecting the two tests here is not
    a conservative middle course -- on this case it admits nothing at all.
    """
    # ponytail: dedupe by COMPOSITION, not by stage index. What over-counts is the
    # pinch tail, where consecutive stages differ by ~0 and march to the same
    # curve. 1e-3 in mole fraction; tighten only if a case wants two launches that
    # close together.
    seen = []
    for k in range(src["n"]):
        x0 = src["X"][k]
        inside = (feasible(mid_sec, x0) if body is None else
                  body_distance(body["vertices"], x0[None, :]) <= TOUCH_TOL)
        if not inside:
            continue
        if any(np.linalg.norm(x0 - q) < 1e-3 for q in seen):
            continue
        seen.append(x0)
    return seen


def _interior_profiles(mid_sec, tp, P, prob, tprof, bprof, bodies=None,
                       pinches=None):
    """Candidate interior-section curves, top -> bottom (Sec 6.3 then 6.2).

    Three ways to turn a section into a curve, chosen by `prob.anchor_method`:

    `saddle`        the limiting profile of a strongly pinched section, built
                    from the invariant manifolds through its saddle. Needs no
                    arbitrary launch stage. The default and what BVM has always
                    done.
    `ray`           march inward from the far end of the saddle's stable ray --
                    the body's S vertex. See `anchor.ray_end_candidates`.
    `continuation`  Sec 6.2: the liquid composition is continuous across a feed,
                    so launch from stages of the NEIGHBOURING profiles that lie
                    inside a candidate middle rectification body. See
                    `_launch_stages`.

    A section with no saddle at all (an ordinary multifeed intermediate) gets
    continuation whatever was asked for; the other two have nothing to anchor on.

    Each launch is marched BOTH ways. `mid_sec.dir` is sign(Delta), which for a
    middle section says nothing about the way its profile runs -- the trap
    `anchor.ray_end_candidates` and `pinch.jacobian` document -- and a march that
    cannot take a step in the direction offered is dropped by the `n >= 2` filter,
    so the neighbour it came from contributed nothing at all. On ipa/water/EG,
    where Delta_ext = D - E < 0, that silently reduced continuation to the
    rectifying profile at every E/F >= 0.75: the stripping launch died at n = 1
    forward and marched 15 stages backward, closing the worse junction to 0.124
    against the 0.191 the rectifying side manages (R = 1.5, E/F = 2.0). Nor is the
    live direction guessable from which end the stage came from -- here the
    stripping-launched curve that survives is the DOWN-map one. `_both_orientations`
    does not cover this: reversing the array re-reads the same curve, the inverse
    map traces a different one.
    """
    E = prob.efficiency
    method = getattr(prob, "anchor_method", "saddle") or "saddle"
    build = {"saddle": interior_candidates,
             "ray": ray_end_candidates}.get(method)
    cands = []
    if build is not None:
        for body in (bodies or [None]):
            cands += build(mid_sec, tp, P, max_stages=prob.max_stages,
                           efficiency=E, pinches=pinches, body=body)
    if cands:
        return [p for c in cands for p in _both_orientations(c)]
    if method not in ("continuation", "saddle", "ray"):
        raise ValueError(f"unknown anchor_method {method!r}")

    def _from(body):
        out = []
        for src in (tprof, bprof):
            for x0 in _launch_stages(mid_sec, src, body):
                for d in (mid_sec.dir, -mid_sec.dir):
                    # `P_lim` is the pressure at the column's OTHER end and clamps
                    # the ramp there (march.march_section), so it follows the
                    # march, not the section: an up-march anchored at _P_mid with
                    # the reboiler pressure would clamp to [P_mid, P_bot] and run
                    # flat.
                    m = march_section(mid_sec._replace(dir=d), x0, tp, P,
                                      prob.max_stages, efficiency=E, dP=_dP(prob),
                                      P_lim=_P_bot(prob) if d > 0 else prob.pressure)
                    if m["n"] >= 2:
                        m = {**m, "anchor_method": "continuation",
                             "swapped_arms": False,
                             "body_id": None if body is None else body.get("id")}
                        out.extend(_both_orientations(m))
        return out

    out = [p for body in (bodies or [None]) for p in _from(body)]
    # No neighbouring stage inside any body: fall back to the section's own
    # region, which is what this did before the bodies were consulted. A
    # continuation curve from the wrong set beats no curve at all -- the junction
    # test still has to pass -- and without this a section whose bodies all sit
    # off the neighbouring profiles reports `cannot_anchor` with nothing tried.
    return out or (_from(None) if bodies else out)


def _choose_interior(prob, tp, top_sec, mid_sec, bot_sec, tprof, bprof):
    """Pick the interior curve whose two junctions actually close (rule 5).

    The body chosen by `_winning_body` has already answered most of this: which
    saddle, and which way each arm points. What is left is the reading that the
    geometry cannot settle -- both eigendirection assignments and both ways up --
    and the checks a polytope cannot make, which is why the junction test still
    runs on the marched curve. Ranking the arms by marched length instead (what
    this used to inherit from `anchor`) picks whichever arm happens to survive
    longest, and that varies per case: the same code drew the extractive elbow
    rotated a different way in each of three files.

    Returns (best, rep, blocked) where best is (score, pieces) or None, and
    `blocked` is (rank, Finding) for the rejected candidate that got furthest --
    the reason `_fail_three` should report.
    """
    E = prob.efficiency
    P_mid, P_bot = _P_mid(prob), _P_bot(prob)
    best = None             # (score, pieces dict), minimised over candidate curves
    rep = (None, None)      # a representative (mprof, conn) for diagnostics
    # How far the most promising REJECTED candidate got, as (rank, Finding).
    # Every candidate that fails does so for its own reason, and reporting the
    # wrong one is its own bug: this used to keep only the out-of-order case, so
    # a design killed by the temperature check was reported as `junction_order`
    # -- a message describing a different arm entirely, printed next to a crossing
    # gap of 0.000. Ranks ascend with how much of the test the candidate passed,
    # so the surviving reason is the informative one.
    blocked = None
    def _block(rank, cls, detail):
        nonlocal blocked
        if blocked is None or rank > blocked[0]:
            blocked = (rank, Finding(cls, mid_sec.name, detail))

    bodies, pinches = _viable_bodies(prob, tp, top_sec, mid_sec, bot_sec,
                                     tprof, bprof, P_mid)
    for mprof in _interior_profiles(mid_sec, tp, P_mid, prob, tprof, bprof,
                                    bodies=bodies, pinches=pinches):
        # both junctions sit on the SAME interior curve; the interior stage count
        # is the arc length between them, so they must also be in top->bottom order.
        # strict=False: both junctions land on a saddle-launched manifold arm, not
        # on a product-anchored march, and no arm reaches both neighbours exactly
        # (see connect.connect). They are accepted within a stage and flagged.
        up = connect(tprof, mprof, top_sec, tp, P_mid,
                     eps_stage=prob.eps_stage, efficiency=E, strict=False)
        low = connect(mprof, bprof, mid_sec, tp, P_bot,
                      eps_stage=prob.eps_stage, efficiency=E, strict=False)
        # the representative is the candidate that came CLOSEST, not the first
        # one tried. `_fail_three` quotes its gap as "how far off is this
        # operating point", and the first candidate is an arbitrary answer to
        # that -- on the paper's example at E/F = 0.750 the order of the arms
        # decided whether the miss was reported as 0.054 or 0.248, for the same
        # infeasible verdict.
        miss = up if up["dmin"] >= low["dmin"] else low
        if rep[0] is None or miss["dmin"] < rep[1]["dmin"]:
            rep = (mprof, miss)
        if not (up["connected"] and low["connected"]):
            miss = up if not up["connected"] else low
            where = "upper (entrainer)" if not up["connected"] else "lower"
            _block(1 if (up["connected"] or low["connected"]) else 0,
                   "cannot_anchor",
                   f"{where} junction does not close: gap {miss['dmin']:.3g} "
                   f"> tol {miss['tol']:.3g}")
            continue
        # nB of the upper junction is the entrainer-feed stage (first interior
        # stage); nA of the lower junction is the last interior stage.
        #
        # Both are clamped to the curve's TRAVELLING part. A saddle-launched arm
        # spends its tail crawling into the pinch -- on ipa/water/EG the last ~50
        # of 236 stages move ~0 -- and `connect` already trims that before it
        # searches, so a junction index can legitimately come back pointing into
        # a stretch the search never considered. Slicing there counted a hundred
        # trays of standing still as column.
        travel = CN.travel_end(mprof["X"])
        i_lo = min(int(np.floor(up["nB"])), travel)
        i_hi = min(int(np.floor(low["nA"])), travel)
        if i_hi < i_lo:                # lower junction is above the upper one
            _block(2, "junction_order",
                   f"lower feed lands at interior stage {i_hi} which is not below "
                   f"the upper feed at {i_lo}; the interior section has no stages "
                   "to occupy")
            continue
        upper_n = int(np.floor(up["nA"]))
        bot_n = int(np.floor(low["nB"]))
        mid_n = i_hi - i_lo + 1
        # the arms that do NOT belong to one column announce themselves in the
        # assembled temperature: an arm running off to the entrainer vertex joins
        # sections 40 K apart. `_reject_inverted` already refuses such a column at
        # the end -- checking it here instead lets the next candidate be tried,
        # rather than the whole design dying on the first one that scored well.
        T = np.concatenate([tprof["T"][:upper_n + 1], mprof["T"][i_lo:i_hi + 1],
                            bprof["T"][:bot_n + 1][::-1]])
        # the two steps that cross a feed: into the first interior stage, and out
        # of the last one. Both are entitled to the looser allowance.
        ok, drop, where = _inversion_verdict(T, (upper_n, upper_n + mid_n))
        if not ok:
            _block(3, "profile_inverted",
                   f"both junctions close (gaps {up['dmin']:.3g} / "
                   f"{low['dmin']:.3g}) but the assembled temperature falls "
                   f"{drop:.1f} K {where} going down the column; the arm joins "
                   "sections that do not belong to one column")
            continue
        # A near-miss junction bounding a section with no stages in it is not a
        # design (`_degenerate`). Rejected HERE rather than at the end, so the
        # next candidate gets its turn -- that is the whole difference from the
        # blanket veto `_vanished_warning` argues against, which had nothing to
        # fall through to.
        deg = [(c, n, nm) for c, n, nm in ((up, upper_n, top_sec.name),
                                           (low, bot_n, bot_sec.name))
               if _degenerate(c, n)]
        if deg:
            conn, n_used, which = deg[0]
            _block(4, "vanished_section",
                   f"the {which} section gets {n_used} stage(s) and its junction "
                   f"is a near miss ({conn['dmin']:.3g} against tol "
                   f"{conn['tol']:.3g}), not a crossing: the profile does not "
                   f"travel to the junction, so this arm needs infinitely many "
                   f"stages to reach it")
            continue
        collapsed = _at_anchor(up["nA"]) or _at_anchor(low["nB"])
        # rank candidates by how exactly their junctions close, not by fewest
        # stages: the junction position is a free choice, and minimising N always
        # walks it to whichever profile end makes a section vanish (extract_col
        # came out with a single stripping stage spanning x_AN 0.57 -> 0.91).
        # A candidate that collapses a section sorts behind every one that does
        # not, whatever its residual -- it is only taken when it is all there is,
        # and then `warnings` says so rather than the design passing as ordinary.
        # `swapped_arms` leads: pairing the saddle's two eigendirections to the
        # two ends the other way round (anchor.interior_candidates, rule 5) is a
        # fallback for a section the physical reading cannot make a column out of,
        # so it must never outrank a candidate that reading produced -- otherwise
        # a swapped arm's tighter junction quietly replaces the real design.
        score = (mprof.get("swapped_arms", False), collapsed,
                 up["dmin"] + low["dmin"], upper_n + mid_n + bot_n)
        if best is None or score < best[0]:
            best = (score, {"upper_n": upper_n, "mprof": mprof, "i_lo": i_lo,
                            "i_hi": i_hi, "mid_n": mid_n, "bot_n": bot_n,
                            "conn": low, "conn_upper": up,
                            "collapsed": collapsed})
    return best, rep, blocked


def _size_three(prob, tp, top_sec, mid_sec, bot_sec, xD, xB, P, out, extractive,
                forced=None):
    """Three-section (multifeed / extractive) sizing routed THROUGH the interior.

    The two product-anchored profiles have no reason to approach each other in
    extractive mode -- the interior section is the bridge, and it must meet BOTH
    of them: once at the upper feed (entrainer) and once at the lower feed. Both
    junctions are located by arc length on the *same* interior curve, so the
    interior stage count is the distance between them and they have to come in
    top-to-bottom order.

    The interior curve itself comes from `_interior_profiles`: a saddle-anchored
    manifold where one exists (Sec 6.3), otherwise continuation from a switch
    stage that lies inside the interior section's feasible region (Sec 6.2).
    """
    E = prob.efficiency
    dP = _dP(prob)
    tprof = march_section(top_sec, xD, tp, P, prob.max_stages, efficiency=E,
                          dP=dP, P_lim=_P_bot(prob), stop_sec=mid_sec)
    bprof = march_section(bot_sec, xB, tp, _P_bot(prob), prob.max_stages,
                          efficiency=E, dP=dP, P_lim=P)

    if forced is not None:
        # junction indices already solved for (solve_free_splits): take them as given
        # rather than re-searching with the tolerance-based test, which would
        # otherwise be free to disagree with an exact solution.
        mprof, upper_n, i_lo, i_hi, bot_n = forced
        best = (0, {"upper_n": upper_n, "mprof": mprof, "i_lo": i_lo,
                    "i_hi": i_hi, "mid_n": i_hi - i_lo + 1, "bot_n": bot_n,
                    "conn": None, "conn_upper": None})
        return _assemble_three(prob, tp, P, top_sec, mid_sec, bot_sec,
                               tprof, mprof, bprof, best[1], out, extractive,
                               forced=True)

    best, rep, blocked = _choose_interior(prob, tp, top_sec, mid_sec, bot_sec,
                                          tprof, bprof)
    pieces = best[1] if best is not None else None
    if best is None:
        return _fail_three(out, top_sec, mid_sec, bot_sec, tprof, bprof, rep,
                           blocked, extractive)
    return _assemble_three(prob, tp, P, top_sec, mid_sec, bot_sec,
                           tprof, pieces["mprof"], bprof, pieces, out, extractive)


def _fail_three(out, top_sec, mid_sec, bot_sec, tprof, bprof, rep, blocked,
                extractive):
    """Classified verdict when no candidate interior curve produced a column.

    `blocked` is (rank, Finding) for the rejected candidate that got furthest
    (see `_choose_interior`). Which of the two accounts leads depends on how far
    that candidate got:

    rank >= 2  both junctions closed, and the candidate died on the ordering or
               the temperature check. `classify` cannot see either -- it reads
               the profiles and would say "pinched apart", which is simply untrue
               of a pair that met at 1e-17. The specific reason leads.
    rank <= 1  a junction genuinely failed to close, which is what `classify` is
               for: it can tell an entrainer shortage from a reflux shortage from
               a section that never reached its region. Its verdict leads and the
               junction gap follows as context.
    """
    mprof_rep, conn = rep
    if mprof_rep is None:
        mprof_rep = {"X": tprof["X"][:1], "Y": tprof["Y"][:1], "T": tprof["T"][:1],
                     "status": "no_region", "pinched": False, "n": 1}
    profiles = {top_sec.name: tprof, mid_sec.name: mprof_rep, bot_sec.name: bprof}
    both_pinched = mprof_rep.get("pinched", False) and bprof["pinched"]
    _, findings = classify(profiles, conn, both_pinched=both_pinched,
                           extractive=extractive)
    if blocked is not None:
        rank, finding = blocked
        findings = ([finding] + findings) if rank >= 2 else (findings + [finding])
    if not findings:
        findings = [Finding("cannot_anchor", mid_sec.name,
                            "no interior profile reaches both feeds at this "
                            "operating point")]
    out["feasible"] = False; out["findings"] = findings
    out["sections"] = _section_report(
        {top_sec.name: tprof,
         mid_sec.name: {**mprof_rep, "status": mprof_rep.get("status", "?")},
         bot_sec.name: bprof}, {})
    out["profiles"] = profiles
    out["connection"] = conn
    return out


def _assemble_three(prob, tp, P, top_sec, mid_sec, bot_sec, tprof, mprof, bprof,
                    pieces, out, extractive, forced=False):
    """Stack the three sections top -> bottom at the chosen junction indices.

    Each feed stage belongs to the section BELOW it (connect's nB), so the slices
    share no stage and none is dropped -- the liquid either side of a feed is
    genuinely two different trays.
    """
    upper_n, bot_n = pieces["upper_n"], pieces["bot_n"]
    i_lo, i_hi = pieces["i_lo"], pieces["i_hi"]
    profiles = {top_sec.name: tprof, mid_sec.name: mprof, bot_sec.name: bprof}
    up_conn = pieces.get("conn_upper")
    if up_conn is not None and _at_anchor(up_conn["nA"]):
        out["warnings"].append(_vanished_warning(top_sec.name, up_conn, "nA"))
    if pieces.get("conn") is not None and _at_anchor(pieces["conn"]["nB"]):
        out["warnings"].append(_vanished_warning(bot_sec.name, pieces["conn"], "nB"))
    out["feasible"] = True
    out["findings"] = []
    out["junctions"] = _junction_report(
        [(f"{top_sec.name}/{mid_sec.name}", up_conn),
         (f"{mid_sec.name}/{bot_sec.name}", pieces.get("conn"))], out["warnings"])
    out["sections"] = _section_report(
        {top_sec.name: tprof,
         mid_sec.name: {**mprof, "status": mprof.get("status", "assembled")},
         bot_sec.name: bprof},
        {top_sec.name: upper_n + 1, mid_sec.name: pieces["mid_n"],
         bot_sec.name: bot_n + 1})
    out["profiles"] = profiles
    out["connection"] = pieces.get("conn")

    xt = tprof["X"][:upper_n + 1]; Tt = tprof["T"][:upper_n + 1]
    yt = tprof["Y"][:upper_n + 1]
    xm = mprof["X"][i_lo:i_hi + 1]
    Tm = mprof["T"][i_lo:i_hi + 1]
    ym = mprof["Y"][i_lo:i_hi + 1]
    xb = bprof["X"][:bot_n + 1][::-1]; Tb = bprof["T"][:bot_n + 1][::-1]
    yb = bprof["Y"][:bot_n + 1][::-1]
    x = np.vstack([xt, xm, xb]); T = np.concatenate([Tt, Tm, Tb])
    y = np.vstack([yt, ym, yb])
    L = np.concatenate([np.full(len(xt), top_sec.L), np.full(len(xm), mid_sec.L),
                        np.full(len(xb), bot_sec.L)])
    Vv = np.concatenate([np.full(len(xt), top_sec.V), np.full(len(xm), mid_sec.V),
                         np.full(len(xb), bot_sec.V)])
    upper_feed = len(xt)                  # first extractive stage
    lower_feed = len(xt) + len(xm)        # first stripping stage
    out["N_total"] = len(x); out["feed_stages"] = [upper_feed, lower_feed]
    out["column"] = {"x": x, "y": y, "T": T, "liquid_flow": L, "vapor_flow": Vv,
                     "feed_stage": upper_feed, "feed_stages": [upper_feed, lower_feed]}
    return _reject_oversized(_reject_inverted(out, mid_sec.name), prob,
                             mid_sec.name)


def size_column(prob, tp, R, S=None, EF=None):
    """Size the column at (R, S, EF). Returns a `design` dict (see module doc).

    A `Problem.reactions` set switches the whole sizing into reduced transformed
    coordinates (reactive.py): same geometry, one fewer component, chemical
    equilibrium inside every stage. The design dict then also carries
    `reactive=True` and a `physical` block (real compositions + extent per stage);
    its `comps`/`x`/`y` stay transformed, so they match each other's shapes.
    """
    if getattr(prob, "reactions", None) is not None:
        prob_r, tp_r = reactive.transform_problem(prob, tp)
        out = _resolve_dP(prob_r, tp_r, R, S, EF)
        return _restore_reactive(out, prob, tp_r)
    return _resolve_dP(prob, tp, R, S, EF)


def _resolve_dP(prob, tp, R, S, EF):
    """Size, then re-size with the reboiler pressure the stage count implies.

    The column's bottom pressure is P_top + dP*(N-1), and N is exactly what this
    method computes -- so the pressure profile and the stage count are a small
    fixed point. One refinement closes it: dP is O(0.01 bar/stage), so a stage or
    two of error in N moves the reboiler pressure (and its bubble point) by less
    than the marching tolerance. Flat columns (dP = 0) skip it entirely.
    """
    out = _size(prob, tp, R, S=S, EF=EF)
    dP = _dP(prob)
    N = out.get("N_total")
    if not dP or not N:
        return out
    refined = replace(prob, P_bot=prob.pressure + dP * (int(N) - 1))
    out2 = _size(refined, tp, R, S=S, EF=EF)
    # keep the refinement only if it still produced a column: a pressure profile
    # is not worth losing a feasible design over.
    return out2 if out2.get("N_total") else out


def _restore_reactive(out, prob, tp_r):
    """Attach the physical (untransformed) reading of a transformed design."""
    rx = prob.reactions
    P = prob.pressure
    out["reactive"] = True
    out["transformed_comps"] = list(out["comps"])
    ok, why = reactive.simplex_safe(rx)
    if not ok:
        # the transform can leave the simplex for this stoichiometry: say so in the
        # verdict, so an infeasible result isn't mistaken for a numerical hiccup
        out["findings"] = list(out["findings"]) + [
            Finding(cls="leaves_simplex", section="transform", detail=why)]
    xD_p, _, xD_e = reactive.physical_profile([out["xD"]], rx, tp_r.tp, P)
    xB_p, _, xB_e = reactive.physical_profile([out["xB"]], rx, tp_r.tp, P)
    phys = {"comps": list(prob.comps), "xD": xD_p[0], "xB": xB_p[0],
            "extent_D": float(xD_e[0]), "extent_B": float(xB_e[0])}
    col = out.get("column")
    if col is not None:
        x, T, extent = reactive.physical_profile(col["x"], rx, tp_r.tp, P)
        # the physical vapour is the one leaving that stage's equilibrium liquid --
        # read it off the real VLE, not by inverting the transform (a transformed
        # vapour maps back to a *family* of physical compositions).
        y = np.array([tp_r.tp.bubble(xi, P)[0] if np.all(np.isfinite(xi))
                      else np.full_like(xi, np.nan) for xi in x])
        phys.update({"x": x, "y": y, "T": T, "extent": extent})
    out["physical"] = phys
    return out


#: Trace-floor values retried when the profiles fail to meet (`_size`). The
#: default seed is `Problem.trace_floor` = 1e-4, and for a heavy non-key that is
#: far too generous: its distillate content amplifies ~10x per stage marching
#: down, so 1e-4 of n-butane in c2-c4's distillate reaches 6% by stage 8 and
#: bends the rectifying profile clean past the stripping one. Ideal stages never
#: crossed at any reflux; at 1e-8 they cross exactly from R = 0.15 up.
_TRACE_LADDER = (1e-8,)


def _size(prob, tp, R, S=None, EF=None, split=None, forced=None):
    """The sizing loop proper, in whatever coordinates it is handed.

    When the junction fails and the problem HAS free non-key splits, the failure
    may be the seed rather than the reflux: those splits are design freedoms that
    the crossing condition determines (`problem.free_split_indices`), and
    `overall_balance` only seeds them at a trace floor whose value is admittedly
    not physical. So the sizing is retried down a ladder of floors before the
    design is called infeasible.

    How much the seed is worth, measured on c2-c4 (ethane/propane/n-butane, the
    near-ideal sharp split where Underwood is trustworthy at 0.1296) by holding
    the one free split fixed and bisecting r_min:

        split 1e-4 (floor)      r_min 2.751     +2020%
        split 9.3e-7 (Fenske)   r_min 0.1916      +48%
        split 1e-8 (ladder)     r_min 0.1364       +5%
        split 3.1e-10 (solved)  r_min 0.1271       -2%
        split -> 0              r_min 0.1225       -5%   (plateau)

    r_min is monotone in the seed and spans a factor of 21 across it, so the seed
    is not a detail; it converges as the seed shrinks, so there is no "too small";
    and BVM's junction criterion reproduces Underwood to 2% at the solved split,
    which says the method is right and the seed carries essentially all of the
    error. What it does NOT say is that any cheap closed-form seed is safe.
    `splits.fenske_split` is written and measured and is deliberately NOT wired
    in here: it is 100x better than the floor on c2-c4 and still 3000x above the
    solved value, it is worse than this ladder's own 1e-8 rung, and on the
    quaternary reference column it takes the junction from dmin 0.008 to 0.036.
    A seed that helps one case and moves three others without a measurement to
    arbitrate them is not an improvement. Use it as the starting point for a real
    solve, not as the answer.
    """
    given = split is not None          # a SOLVED split from the caller stands
    out = _size_once(prob, tp, R, S=S, EF=EF, split=split, forced=forced)
    if (out.get("feasible") or given or forced is not None
            or not free_split_indices(prob)):
        return out
    for floor in _TRACE_LADDER:
        if floor >= prob.trace_floor:
            continue
        retry = _size_once(replace(prob, trace_floor=floor), tp, R, S=S, EF=EF)
        if retry.get("feasible"):
            retry["trace_floor"] = floor
            retry["warnings"] = list(retry.get("warnings", [])) + [
                f"non-key distillate splits reseeded at {floor:.0e} (the default "
                f"{prob.trace_floor:.0e} left the profiles apart); they are free "
                "design variables, not a specification"]
            return retry
    return out


def _size_once(prob, tp, R, S=None, EF=None, split=None, forced=None):
    """One sizing pass at the seeds it is handed."""
    extractive = prob.extractive and prob.x_E is not None
    xD, xB, D, B = overall_balance(prob, EF if extractive else None, split=split)
    P = prob.pressure
    out = _base(prob, xD, xB, D, B, R)
    out["S"] = S; out["EF"] = EF

    if extractive:
        if not EF or EF <= 0.0:
            # No entrainer stream means no cut between the feeds, so the
            # "extractive" section is bit-identical to the rectifying one
            # (Delta_ext = D - 0, delta_ext = x_D) and trivially connects to it.
            # That is a degenerate chain, not a feasible extractive column: say so
            # rather than reporting a design that has no entrainer in it.
            out["findings"] = [Finding("infeasible_entrainer", "extractive",
                                       "E/F = 0: no entrainer stream, so there is "
                                       "no extractive section to bridge with")]
            return out
        rect, ext, strip = extractive_chain(prob, R, EF, xD, xB, D, B)
        return _size_three(prob, tp, rect, ext, strip, xD, xB, P, out, True,
                           forced=forced)
    if len(prob.feeds) > 1:
        secs = multifeed_chain(prob, R, xD, xB, D, B)
        if len(secs) == 3:
            return _size_three(prob, tp, secs[0], secs[1], secs[2], xD, xB, P,
                               out, False, forced=forced)
        # >3 sections not assembled yet -> size the enclosing two-section problem
        return _size_two(prob, tp, secs[0], secs[-1], xD, xB, P, out, forced=forced)
    rect, strip = single_feed_chain(prob, R, xD, xB, D, B)
    return _size_two(prob, tp, rect, strip, xD, xB, P, out, forced=forced)


def r_min(prob, tp, R_hi=30.0, S=None, EF=None, tol=1e-3):
    """Smallest reflux at which the marched profiles still form a column (Sec 8).

    This is a SIZING minimum, not the thermodynamic one: it is the smallest
    reflux at which the two marched liquid profiles still INTERSECT (connect.py),
    where minimum reflux proper asks when the two reachable SETS first overlap --
    a weaker condition, so this sits above it. Use `side_features.rbm` for the
    thermodynamic minimum and this for the reflux a real column can be built at.

    The two are close once the junction is asked for honestly. On c2-c4 this
    returns 0.1364 against RBM's 0.1342 and Underwood's 0.1466. It used to return
    0.1138 -- 22% under Underwood -- because the junction was measured in vapour
    space, where a large K compresses distances by ~20x and a fixed tolerance is
    correspondingly loose: at that reflux the two liquids were 0.072 apart.

    Feasibility in R is a BAND, not an upper set: at high reflux the crossing runs
    off the end of the stripping profile (c2-c4 closes around R ~ 10, its
    stripping section having shrunk below one stage). `bisect_min` brackets the
    lower edge of the last feasible run rather than assuming `R_hi` is feasible.

    `tol` is the bisection's own resolution in reflux units, tightened from 1e-2.
    """
    return bisect_min(partial(_feasible_at_R, prob, tp, S, EF), 0.05, R_hi,
                      tol=tol)


def _feasible_at_R(prob, tp, S, EF, R):
    """Is there a column at this reflux? A module-level function behind a
    `partial`, not the closure it used to be, so `pinch.bisect_min` can hand the
    whole pre-scan to a process pool (see `parallel.pmap`)."""
    return size_column(prob, tp, float(R), S=S, EF=EF)["feasible"]


def _feasible_at_EF(prob, tp, R, EF):
    return size_column(prob, tp, R, EF=float(EF))["feasible"]


def ef_min(prob, tp, R, EF_hi=5.0, tol=1e-2):
    """Minimum entrainer-to-feed ratio by bisection (extractive mode, Sec 8)."""
    return bisect_min(partial(_feasible_at_EF, prob, tp, R), 0.0, EF_hi, tol=tol)


def reflux_band(prob, tp, EF=None, S=None, r_lo=0.05, r_hi=30.0, n_scan=24,
                tol=1e-3, cancelled=None):
    """(R_min, R_max) at this operating point; R_max is None when the band is open.

    `r_min` bisects the lower edge only. That is the right shape for an ordinary
    column, where more reflux never hurts, and the wrong shape for an extractive
    one, where it does: past some reflux the entrainer is diluted out of the
    middle section and the separation stops working. This reports both edges so
    the GUI can draw the band rather than a half-line.
    """
    return feasible_band(partial(_feasible_at_R, prob, tp, S, EF), r_lo, r_hi,
                         n_scan=n_scan, tol=tol, cancelled=cancelled)


def _band_at_EF(prob, tp, S, r_lo, r_hi, n_scan, ef):
    return reflux_band(prob, tp, EF=float(ef), S=S, r_lo=r_lo, r_hi=r_hi,
                       n_scan=n_scan)


def operating_region(prob, tp, EF_grid=None, S=None, r_lo=0.05, r_hi=30.0,
                     n_scan=16, on_step=None, cancelled=None):
    """Feasible (E/F, R) region: the reflux band against entrainer flow.

    Same shape as `side_features.rbm.driver.operating_region`, deliberately, so
    the two panels can plot it with one piece of code. `EF_min` is the smallest
    sampled entrainer ratio that admits any reflux at all -- the nose where the
    two reflux bounds meet.

    One entrainer ratio is a whole reflux band -- dozens of columns -- so the
    ratios go to a process pool (`parallel.pmap`) and the bands inside them stay
    serial. `on_step(done, total)` and `cancelled()` are the GUI's hooks; a
    cancelled sweep keeps NaN for the ratios it never reached.
    """
    if EF_grid is None:
        EF_grid = np.linspace(0.2, 2.0, 10)
    EFs = np.atleast_1d(np.asarray(EF_grid, float))
    lo = np.full(len(EFs), np.nan)
    hi = np.full(len(EFs), np.nan)
    bands = pmap(partial(_band_at_EF, prob, tp, S, r_lo, r_hi, n_scan),
                 [float(ef) for ef in EFs], on_step=on_step, cancelled=cancelled)
    for i, band in enumerate(bands):
        if band is None:                     # cancelled before reaching this one
            continue
        a, b = band
        if a is not None:
            lo[i] = a
            hi[i] = r_hi if b is None else b
    idx = np.flatnonzero(np.isfinite(lo))
    return {"EF": EFs, "r_min": lo, "r_max": hi,
            "EF_min": float(EFs[idx[0]]) if len(idx) else None,
            "r_at_EF_min": float(lo[idx[0]]) if len(idx) else None,
            "operating": None}


def _map_point(prob, tp, rse):
    """One (R, S, E/F) grid point -> (feasible, stages). Returns the two numbers
    rather than the design, so a process pool ships back 16 bytes and not a full
    set of profiles."""
    R, S, E = rse
    d = size_column(prob, tp, R, S=S, EF=E)
    return bool(d["feasible"]), int(d["N_total"]) if d["feasible"] else -1


def feasibility_map(prob, tp, R_grid, S_grid=None, EF_grid=None,
                    on_step=None, cancelled=None):
    """Sweep (R, S, EF) -> feasibility + stage-count grids (Sec 10).

    Returns dict(R, S, EF, feasible (bool grid), stages (int grid, -1 if not)).
    S_grid/EF_grid default to [None] (single value) so a plain R sweep is 1-D.
    The grid points are independent, so they go to `parallel.pmap`.
    """
    Rs = np.atleast_1d(R_grid)
    Ss = np.atleast_1d(S_grid) if S_grid is not None else np.array([None])
    Es = np.atleast_1d(EF_grid) if EF_grid is not None else np.array([None])
    shape = (len(Rs), len(Ss), len(Es))
    feas = np.zeros(shape, bool)
    stages = np.full(shape, -1, int)
    points = [(float(R), None if S is None else float(S),
               None if E is None else float(E))
              for R in Rs for S in Ss for E in Es]
    for (i, j, k), got in zip(np.ndindex(shape),
                              pmap(partial(_map_point, prob, tp), points,
                                   on_step=on_step, cancelled=cancelled)):
        if got is None:                      # cancelled before reaching this one
            continue
        feas[i, j, k], stages[i, j, k] = got
    return {"R": Rs, "S": Ss, "EF": Es, "feasible": feas.squeeze(),
            "stages": stages.squeeze()}


def _demo():
    from .thermo_adapter import ColumnForgeThermo
    from .problem import build_problem

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = ColumnForgeThermo(abc)
    z = np.array([0.4, 0.35, 0.25])
    prob = build_problem(["b", "t", "x"], [(z, 100.0, 1.0)], 760.0,
                         rec_lk=0.98, rec_hk=0.02)

    d = size_column(prob, tp, R=4.0)
    assert d["feasible"], d["findings"]
    col = d["column"]
    assert col["x"].shape[0] == d["N_total"]

    # every feasible design says what its junctions actually closed to, and a
    # crossing is reported as a crossing rather than as a tolerance (R4).
    assert d["junctions"] and not d["junctions"][0]["approximate"]
    assert d["junctions"][0]["dmin"] < 1e-9, d["junctions"]

    # the economic gate: the same design, refused for being uneconomic, and the
    # finding must name that rather than blame the geometry (R3).
    tight = size_column(replace(prob, max_column_stages=d["N_total"] - 1), tp, R=4.0)
    assert not tight["feasible"] and tight["N_total"] is None
    assert tight["findings"][0].cls == "too_many_stages", tight["findings"]
    assert size_column(replace(prob, max_column_stages=d["N_total"]), tp,
                       R=4.0)["feasible"], "the budget is inclusive"
    assert np.allclose(col["x"][0], d["xD"], atol=1e-6), "stage 0 is the distillate"
    assert np.allclose(col["x"][-1], d["xB"], atol=1e-6), "last stage is the bottoms"
    assert col["T"][-1] > col["T"][0], "reboiler hotter than condenser"
    assert 0 < col["feed_stage"] < d["N_total"] - 1

    # R_min < design R, and stages grow as R -> R_min
    Rmin = r_min(prob, tp)
    assert Rmin is not None and Rmin < 4.0, Rmin
    N_lo = size_column(prob, tp, R=Rmin * 1.1)["N_total"]
    N_hi = size_column(prob, tp, R=Rmin * 3.0)["N_total"]
    assert N_lo >= N_hi, (N_lo, N_hi)

    # feasibility map: infeasible below R_min, feasible above
    fm = feasibility_map(prob, tp, R_grid=[Rmin * 0.5, Rmin * 1.5, Rmin * 3])
    assert not fm["feasible"][0] and fm["feasible"][1] and fm["feasible"][2]
    print(f"driver self-check OK  N={d['N_total']} feed@{col['feed_stage']} "
          f"R_min={Rmin:.2f}  map={fm['feasible'].tolist()}")


if __name__ == "__main__":
    _demo()
