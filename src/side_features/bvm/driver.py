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

import numpy as np

from scipy.optimize import least_squares

from . import reactive
from .problem import overall_balance, SideDraw, free_split_indices
from .sections import (single_feed_chain, extractive_chain, multifeed_chain,
                       feasible)
from .march import march_section
from .connect import connect
from .place import side_draw_stage
from .anchor import interior_candidates
from .diagnostics import classify, Finding
from .pinch import bisect_min, feasible_band


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


def _temperature_inversion(T):
    """Largest downward step in a top->bottom temperature profile (0 if monotone)."""
    if T is None or len(T) < 2:
        return 0.0
    return float(max(0.0, -np.min(np.diff(np.asarray(T, float)))))


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
    drop = _temperature_inversion(col["T"])
    if drop <= T_INVERSION_TOL:
        return out
    out["feasible"] = False
    out["findings"] = [Finding(
        "profile_inverted", findings_section,
        f"assembled temperature falls {drop:.1f} K going down the column; the "
        "junction joins sections that do not belong to one column")]
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
            "sections": {}, "profiles": {}, "connection": None}


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


def _section_report(profiles, assembled):
    """Per-section `n` = stages the assembled COLUMN uses; `n_marched` = how far
    the profile was traced.

    These differ, often wildly, and reporting the marched length as if it were the
    column hid a 33-stage rectifying profile contributing exactly one stage. The
    marched length still matters for debugging (it says where the profile ended and
    why), so keep both rather than swapping one lie for another.
    """
    return {k: {"n": assembled.get(k, v["n"]), "n_marched": v["n"],
                "status": v["status"]}
            for k, v in profiles.items()}


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
    conn = connect(rprof, sprof, rect, tp, P, eps_stage=prob.eps_stage,
                   efficiency=prob.efficiency, strict=strict)
    both_pinched = rprof["pinched"] and sprof["pinched"]
    side = side_draw_stage(sprof, prob.side_draws[0]) if prob.side_draws else None
    feasible, findings = classify(profiles, conn, both_pinched=both_pinched,
                                  side_draw=side)
    if forced is not None:
        # junction already solved for (solve_omega); don't let the tolerance-based
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
    return _reject_inverted(out, "connection")


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


def _interior_profiles(mid_sec, tp, P, prob, tprof, bprof):
    """Candidate interior-section curves, top -> bottom (Sec 6.3 then 6.2).

    Saddle-anchored manifolds first -- that is the limiting profile of a strongly
    pinched section and needs no arbitrary launch stage. If the section has no
    saddle (an ordinary multifeed intermediate), fall back to continuation off the
    neighbouring profiles, but only from switch stages that are actually *inside*
    the interior section's feasible region: a liquid outside it cannot sit on any
    stage of that section, so launching there was never meaningful.
    """
    E = prob.efficiency
    cands = interior_candidates(mid_sec, tp, P, max_stages=prob.max_stages,
                                efficiency=E)
    if cands:
        return [p for c in cands for p in _both_orientations(c)]

    out = []
    for src in (tprof, bprof):
        for k in range(src["n"]):
            x0 = src["X"][k]
            if not feasible(mid_sec, x0):
                continue
            m = march_section(mid_sec, x0, tp, P, prob.max_stages, efficiency=E,
                              dP=_dP(prob), P_lim=_P_bot(prob))
            if m["n"] >= 2:
                out.extend(_both_orientations(m))
    return out


def _choose_interior(prob, tp, top_sec, mid_sec, bot_sec, tprof, bprof):
    """Pick the interior curve whose two junctions actually close (rule 5).

    A saddle has four arm pairs and each can be read either way up, so there are
    eight candidate curves and nothing local to the section distinguishes them --
    Bruggemann & Marquardt keep the one whose body meets the rectifying AND the
    stripping body, which is exactly the double-junction test below. Ranking the
    arms by marched length instead (what this used to inherit from `anchor`) picks
    whichever arm happens to survive longest, and that varies per case: the same
    code drew the extractive elbow rotated a different way in each of three files.

    Returns (best, rep, out_of_order) where best is (score, pieces) or None.
    """
    E = prob.efficiency
    P_mid, P_bot = _P_mid(prob), _P_bot(prob)
    best = None             # (score, pieces dict), minimised over candidate curves
    rep = (None, None)      # a representative (mprof, conn) for diagnostics
    out_of_order = None     # a candidate that met both ends but in the wrong order
    for mprof in _interior_profiles(mid_sec, tp, P_mid, prob, tprof, bprof):
        # both junctions sit on the SAME interior curve; the interior stage count
        # is the arc length between them, so they must also be in top->bottom order.
        # strict=False: both junctions land on a saddle-launched manifold arm, not
        # on a product-anchored march, and no arm reaches both neighbours exactly
        # (see connect.connect). They are accepted within a stage and flagged.
        up = connect(tprof, mprof, top_sec, tp, P_mid,
                     eps_stage=prob.eps_stage, efficiency=E, strict=False)
        low = connect(mprof, bprof, mid_sec, tp, P_bot,
                      eps_stage=prob.eps_stage, efficiency=E, strict=False)
        if rep[0] is None:
            rep = (mprof, up if not up["connected"] else low)
        if not (up["connected"] and low["connected"]):
            continue
        # nB of the upper junction is the entrainer-feed stage (first interior
        # stage); nA of the lower junction is the last interior stage.
        i_lo = int(np.floor(up["nB"])); i_hi = int(np.floor(low["nA"]))
        if i_hi < i_lo:                # lower junction is above the upper one
            if out_of_order is None:
                out_of_order = (mprof, up, low, i_lo, i_hi)
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
        if _temperature_inversion(T) > T_INVERSION_TOL:
            continue
        collapsed = _at_anchor(up["nA"]) or _at_anchor(low["nB"])
        # rank candidates by how exactly their junctions close, not by fewest
        # stages: the junction position is a free choice, and minimising N always
        # walks it to whichever profile end makes a section vanish (extract_col
        # came out with a single stripping stage spanning x_AN 0.57 -> 0.91).
        # A candidate that collapses a section sorts behind every one that does
        # not, whatever its residual -- it is only taken when it is all there is,
        # and then `warnings` says so rather than the design passing as ordinary.
        score = (collapsed, up["dmin"] + low["dmin"], upper_n + mid_n + bot_n)
        if best is None or score < best[0]:
            best = (score, {"upper_n": upper_n, "mprof": mprof, "i_lo": i_lo,
                            "i_hi": i_hi, "mid_n": mid_n, "bot_n": bot_n,
                            "conn": low, "conn_upper": up,
                            "collapsed": collapsed})
    return best, rep, out_of_order


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
        # junction indices already solved for (solve_omega): take them as given
        # rather than re-searching with the tolerance-based test, which would
        # otherwise be free to disagree with an exact solution.
        mprof, upper_n, i_lo, i_hi, bot_n = forced
        best = (0, {"upper_n": upper_n, "mprof": mprof, "i_lo": i_lo,
                    "i_hi": i_hi, "mid_n": i_hi - i_lo + 1, "bot_n": bot_n,
                    "conn": None, "conn_upper": None})
        return _assemble_three(prob, tp, P, top_sec, mid_sec, bot_sec,
                               tprof, mprof, bprof, best[1], out, extractive,
                               forced=True)

    best, rep, out_of_order = _choose_interior(prob, tp, top_sec, mid_sec, bot_sec,
                                               tprof, bprof)
    pieces = best[1] if best is not None else None
    if best is None:
        return _fail_three(out, top_sec, mid_sec, bot_sec, tprof, bprof, rep,
                           out_of_order, extractive)
    return _assemble_three(prob, tp, P, top_sec, mid_sec, bot_sec,
                           tprof, pieces["mprof"], bprof, pieces, out, extractive)


def _fail_three(out, top_sec, mid_sec, bot_sec, tprof, bprof, rep, out_of_order,
                extractive):
    """Classified verdict when no candidate interior curve produced a column."""
    mprof_rep, conn = rep
    if mprof_rep is None:
        mprof_rep = {"X": tprof["X"][:1], "Y": tprof["Y"][:1], "T": tprof["T"][:1],
                     "status": "no_region", "pinched": False, "n": 1}
    profiles = {top_sec.name: tprof, mid_sec.name: mprof_rep, bot_sec.name: bprof}
    both_pinched = mprof_rep.get("pinched", False) and bprof["pinched"]
    _, findings = classify(profiles, conn, both_pinched=both_pinched,
                           extractive=extractive)
    if out_of_order is not None and not findings:
        # both junctions landed on the interior curve, but the lower one sits at or
        # above the upper one -- the interior section would have to run backwards,
        # so there is no column here even though every individual test "connected".
        _, _, _, i_lo, i_hi = out_of_order
        findings = [Finding("junction_order", mid_sec.name,
                            f"lower feed lands at interior stage {i_hi} which is "
                            f"not below the upper feed at {i_lo}; the interior "
                            "section has no stages to occupy")]
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
    return _reject_inverted(out, mid_sec.name)


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

    ponytail: a ladder, not `solve_omega`'s least squares over the splits. The
    solve is the principled version and it is already written -- but it costs 40 s
    on c2-c4 and 200 s on the C=6 reference column, and it does not converge above
    C=3 (it walks the splits into the simplex corners), so it cannot sit inside
    the R_min bisection. The ladder is three extra marches, only on failure, and
    it recovers the same crossings. Replace it when the split solve is fixed.
    """
    out = _size_once(prob, tp, R, S=S, EF=EF, split=split, forced=forced)
    if (out.get("feasible") or split is not None or forced is not None
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


def _at(prof, t):
    """Composition at fractional stage index t along a marched profile.

    Linearly EXTRAPOLATES past either end rather than clamping: a clamped index
    makes the residual flat in that direction, so a least-squares solver that
    steps off the end has no gradient to come back on and simply stalls there.
    `_range_penalty` is what keeps the answer on the curve.
    """
    X = prof["X"]
    last = len(X) - 1
    if last <= 0:
        return X[0]
    i = int(np.clip(np.floor(t), 0, last - 1))
    return X[i] + (float(t) - i) * (X[i + 1] - X[i])


def _range_penalty(prof, t, scale=1.0):
    """How far a fractional stage index falls outside its profile, 0 when inside."""
    last = len(prof["X"]) - 1
    return scale * (max(0.0, -float(t)) + max(0.0, float(t) - last))


def _profiles_for(prob, tp, R, EF, split, P, mid=None, bot=None):
    """March every section for a candidate free-split vector. Returns
    (chain, profiles, xD, xB, D, B) with profiles ordered top -> bottom.

    `mid` and `bot` supply pre-computed interior and stripping curves. Of the
    three, only the RECTIFYING profile depends strongly on the free split -- that
    is the whole point of solving for it, a trace of entrainer in the distillate
    amplifies downward -- while the other two shift by ~1e-3 across two decades of
    x_D,EG. Since every profile costs UNIFAC evaluations (the dominant expense by
    far), `solve_omega` freezes those two through each inner solve and refreshes
    them in an outer loop, re-measuring the final residual against fresh ones.
    """
    extractive = prob.extractive and prob.x_E is not None and EF
    xD, xB, D, B = overall_balance(prob, EF if extractive else None, split=split)
    E = prob.efficiency
    dP = _dP(prob)
    if extractive:
        rect, ext, strip = extractive_chain(prob, R, EF, xD, xB, D, B)
        tprof = march_section(rect, xD, tp, P, prob.max_stages, efficiency=E,
                              dP=dP, P_lim=_P_bot(prob), stop_sec=ext)
        bprof = bot if bot is not None else march_section(
            strip, xB, tp, _P_bot(prob), prob.max_stages, efficiency=E, dP=dP,
            P_lim=P)
        if mid is None:
            # the same double-junction choice `_size_three` makes -- an arbitrary
            # candidate would hand the least-squares a curve running the wrong way
            # up, which its top->bottom order penalty can then never satisfy.
            best, _, _ = _choose_interior(prob, tp, rect, ext, strip, tprof, bprof)
            if best is not None:
                mid = best[1]["mprof"]
            else:
                mids = _interior_profiles(ext, tp, _P_mid(prob), prob, tprof, bprof)
                mid = mids[0] if mids else None
        return (rect, ext, strip), (tprof, mid, bprof), xD, xB, D, B
    rect, strip = single_feed_chain(prob, R, xD, xB, D, B)
    tprof = march_section(rect, xD, tp, P, prob.max_stages, efficiency=E,
                          dP=dP, P_lim=_P_bot(prob))
    bprof = bot if bot is not None else march_section(
        strip, xB, tp, _P_bot(prob), prob.max_stages, efficiency=E, dP=dP,
        P_lim=P)
    return (rect, strip), (tprof, None, bprof), xD, xB, D, B


def solve_omega(prob, tp, R, omega, EF=None, split0=None):
    """Solve for the free distillate splits that make the sections meet at omega.

    omega is the junction location on the rectifying profile -- the feed-tray
    position, counted in stages from the distillate. Holding it fixed leaves

        C-2 free splits  +  the remaining arc lengths

    against C-1 junction equations per junction, which is square for a two-section
    column at any C, and square for a three-section (extractive) column at C=3.
    Above that an exact double junction is over-determined and genuinely may not
    exist; we solve in least squares and hand back `residual` so the caller can
    say so instead of hiding it in a tolerance.

    This is the answer to "which distillate composition lets the sections
    intersect": there is a one-parameter family of them, indexed by omega, and
    sweeping it (see `spectrum`) is the spectrum of feasible designs.

    Returns dict(split, residual, omega, converged) or None when there is nothing
    free to solve for (C == 2, or no interior curve to aim at).
    """
    free = free_split_indices(prob)
    if not free:
        return None
    P = prob.pressure
    C = prob.C
    extractive = bool(prob.extractive and prob.x_E is not None and EF)

    # unconstrained parameterisation: split = sigmoid(theta) keeps every free
    # split strictly inside (0, 1) without the solver ever having to be clipped.
    def to_split(theta):
        return 1.0 / (1.0 + np.exp(-np.clip(theta, -40.0, 40.0)))

    if split0 is None:
        _, _, D0, _ = overall_balance(prob, EF if extractive else None)
        base = overall_balance(prob, EF if extractive else None)[0]
        split0 = np.array([max(base[k], 1e-6) for k in free])
    s0 = np.clip(np.asarray(split0, float), 1e-9, 1 - 1e-9)
    theta0 = np.log(s0 / (1.0 - s0))

    n_t = 3 if extractive else 1        # free arc lengths besides omega

    def unpack(u):
        return to_split(u[:len(free)]), u[len(free):]

    def junction(sec_above, prof_above, t_above, prof_below, t_below):
        """Residual of (E): a x_above + b == K(x_below) x_below (connect.py)."""
        xa = _at(prof_above, t_above)
        xb = _at(prof_below, t_below)
        try:
            y_below, _ = tp.bubble(xb, P)
        except (ValueError, FloatingPointError):
            return np.ones(C)
        return (sec_above.a * xa + sec_above.bvec) - y_below

    def residual(u, mid=None, bot=None):
        split, ts = unpack(u)
        try:
            chain, (tprof, mprof, bprof), *_ = _profiles_for(
                prob, tp, R, EF, split, P, mid=mid, bot=bot)
        except (ValueError, FloatingPointError):
            return np.full(_n_res(), 1.0)
        if extractive:
            if mprof is None:
                return np.full(_n_res(), 1.0)
            rect, ext, _strip = chain
            r1 = junction(rect, tprof, omega, mprof, ts[0])
            r2 = junction(ext, mprof, ts[1], bprof, ts[2])
            pen = (_range_penalty(mprof, ts[0]) + _range_penalty(mprof, ts[1])
                   + _range_penalty(bprof, ts[2]))
            order = max(0.0, ts[0] - ts[1])   # junctions must run top -> bottom
            return np.concatenate([r1[:C - 1], r2[:C - 1], [pen, order]])
        rect = chain[0]
        r = junction(rect, tprof, omega, bprof, ts[0])[:C - 1]
        return np.concatenate([r, [_range_penalty(bprof, ts[0])]])

    def _n_res():
        return (2 * (C - 1) + 2) if extractive else C

    # Outer loop refreshes the interior curve; inner solve holds it fixed (see
    # `_profiles_for`). Two passes are enough -- the curve is nearly independent
    # of the free split -- and the final residual is always re-measured against a
    # freshly computed curve, so nothing is accepted on a stale one.
    best = None
    theta = theta0
    for _outer in range(2):
        try:
            _, (_t, _m, _b), *_ = _profiles_for(prob, tp, R, EF,
                                                to_split(theta), P)
        except (ValueError, FloatingPointError):
            return None
        if extractive and _m is None:
            return None
        span_m = max((_m["n"] - 1) if _m is not None else 1, 1)
        span_b = max(_b["n"] - 1, 1)
        # The junction system is multi-modal: with the interior curve tens of
        # stages long, one start lands in whichever local minimum it is nearest
        # and plateaus near 0.1 while an exact solution (1e-10) sits elsewhere on
        # the same curve. Spread the arc-length starts and keep the best.
        if extractive:
            starts = [np.array([a * span_m, b * span_m, c * span_b])
                      for a, b, c in ((0.02, 0.5, 0.1), (0.1, 0.9, 0.3),
                                      (0.3, 0.7, 0.05))]
        else:
            starts = [np.array([f * span_b]) for f in (0.2, 0.6, 0.05)]
        if best is not None:
            starts.insert(0, best[1][len(free):])      # continue from incumbent

        for t0 in starts:
            u0 = np.concatenate([theta, t0])
            try:
                sol = least_squares(residual, u0, args=(_m, _b), xtol=1e-10,
                                    ftol=1e-10, max_nfev=80)
            except (ValueError, FloatingPointError):
                continue
            res = float(np.linalg.norm(residual(sol.x)))   # fresh curves
            if best is None or res < best[0]:
                best = (res, sol.x, bool(sol.success))
            if res < 1e-8:
                break
        if best is None:
            return None
        if best[0] < 1e-8:
            break
        theta = best[1][:len(free)]

    res, u, ok = best
    split, ts = unpack(u)
    return {"split": split, "arc": ts, "residual": res, "omega": float(omega),
            "converged": ok and res < 1e-6, "free": free}


def design_at_omega(prob, tp, R, omega, EF=None, split0=None):
    """Solve for the free splits at this feed-tray position and build the column.

    The design is assembled at the junction indices the solve produced, not by
    re-running the tolerance-based search over them -- otherwise an exactly solved
    junction can still be rejected by `connect`'s stage-width test. Returns
    (design, solution) with `design["exact"]` recording whether the junction
    equations actually closed.
    """
    sol = solve_omega(prob, tp, R, omega, EF=EF, split0=split0)
    if sol is None:
        return _size(prob, tp, R, EF=EF), None
    extractive = bool(prob.extractive and prob.x_E is not None and EF)
    P = prob.pressure
    forced = None
    if sol["converged"]:
        arc = sol["arc"]
        try:
            _, (tprof, mprof, bprof), *_ = _profiles_for(prob, tp, R, EF,
                                                         sol["split"], P)
        except (ValueError, FloatingPointError):
            tprof = mprof = bprof = None
        if extractive and mprof is not None:
            i_lo, i_hi = int(np.floor(arc[0])), int(np.floor(arc[1]))
            if 0 <= i_lo <= i_hi < mprof["n"]:
                forced = (mprof, int(np.floor(omega)), i_lo, i_hi,
                          int(np.floor(arc[2])))
        elif not extractive:
            forced = (int(np.floor(omega)), int(np.floor(arc[0])))
    d = _size(prob, tp, R, EF=EF, split=sol["split"], forced=forced)
    d["omega"] = float(omega)
    d["split"] = sol["split"]
    d["junction_residual"] = sol["residual"]
    d["exact"] = bool(sol["converged"] and forced is not None)
    return d, sol


def spectrum(prob, tp, R, omega_grid, EF=None):
    """Sweep the feed-tray position: N_total and the solved x_D at each omega.

    This is the spectrum of designs. Fixing the two key recoveries leaves C-2 free
    distillate splits, and requiring the sections to meet is C-1 equations per
    junction -- one short of determining everything, so feasible designs come as a
    ONE-PARAMETER FAMILY indexed by where the feed tray sits. For each omega there
    is a unique distillate composition that makes the sections intersect; N_total
    against omega has an interior minimum at the best feed location.

    Each omega is warm-started from the previous solution, so the sweep is one
    continuation rather than N cold solves. Returns a list of dicts ordered by
    omega.
    """
    rows = []
    split0 = None
    for w in np.atleast_1d(omega_grid):
        d, sol = design_at_omega(prob, tp, R, float(w), EF=EF, split0=split0)
        if sol is None:
            continue
        split0 = sol["split"]              # continuation
        rows.append({"omega": float(w), "split": sol["split"],
                     "residual": sol["residual"], "exact": d["exact"],
                     "feasible": d["feasible"], "N_total": d["N_total"],
                     "feed_stages": d["feed_stages"], "xD": d["xD"],
                     "xB": d["xB"], "findings": d["findings"],
                     "free_indices": list(sol["free"])})
    return rows


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
    def feasible(R):
        return size_column(prob, tp, R, S=S, EF=EF)["feasible"]
    return bisect_min(feasible, 0.05, R_hi, tol=tol)


def ef_min(prob, tp, R, EF_hi=5.0, tol=1e-2):
    """Minimum entrainer-to-feed ratio by bisection (extractive mode, Sec 8)."""
    def feasible(EF):
        return size_column(prob, tp, R, EF=EF)["feasible"]
    return bisect_min(feasible, 0.0, EF_hi, tol=tol)


def reflux_band(prob, tp, EF=None, S=None, r_lo=0.05, r_hi=30.0, n_scan=24,
                tol=1e-3):
    """(R_min, R_max) at this operating point; R_max is None when the band is open.

    `r_min` bisects the lower edge only. That is the right shape for an ordinary
    column, where more reflux never hurts, and the wrong shape for an extractive
    one, where it does: past some reflux the entrainer is diluted out of the
    middle section and the separation stops working. This reports both edges so
    the GUI can draw the band rather than a half-line.
    """
    def feasible_at(R):
        return size_column(prob, tp, float(R), S=S, EF=EF)["feasible"]
    return feasible_band(feasible_at, r_lo, r_hi, n_scan=n_scan, tol=tol)


def operating_region(prob, tp, EF_grid=None, S=None, r_lo=0.05, r_hi=30.0,
                     n_scan=16):
    """Feasible (E/F, R) region: the reflux band against entrainer flow.

    Same shape as `side_features.rbm.driver.operating_region`, deliberately, so
    the two panels can plot it with one piece of code. `EF_min` is the smallest
    sampled entrainer ratio that admits any reflux at all -- the nose where the
    two reflux bounds meet.
    """
    if EF_grid is None:
        EF_grid = np.linspace(0.2, 2.0, 10)
    EFs = np.atleast_1d(np.asarray(EF_grid, float))
    lo = np.full(len(EFs), np.nan)
    hi = np.full(len(EFs), np.nan)
    for i, ef in enumerate(EFs):
        a, b = reflux_band(prob, tp, EF=float(ef), S=S, r_lo=r_lo, r_hi=r_hi,
                           n_scan=n_scan)
        if a is not None:
            lo[i] = a
            hi[i] = r_hi if b is None else b
    idx = np.flatnonzero(np.isfinite(lo))
    return {"EF": EFs, "r_min": lo, "r_max": hi,
            "EF_min": float(EFs[idx[0]]) if len(idx) else None,
            "r_at_EF_min": float(lo[idx[0]]) if len(idx) else None,
            "operating": None}


def feasibility_map(prob, tp, R_grid, S_grid=None, EF_grid=None):
    """Sweep (R, S, EF) -> feasibility + stage-count grids (Sec 10).

    Returns dict(R, S, EF, feasible (bool grid), stages (int grid, -1 if not)).
    S_grid/EF_grid default to [None] (single value) so a plain R sweep is 1-D.
    """
    Rs = np.atleast_1d(R_grid)
    Ss = np.atleast_1d(S_grid) if S_grid is not None else np.array([None])
    Es = np.atleast_1d(EF_grid) if EF_grid is not None else np.array([None])
    shape = (len(Rs), len(Ss), len(Es))
    feas = np.zeros(shape, bool)
    stages = np.full(shape, -1, int)
    for i, R in enumerate(Rs):
        for j, S in enumerate(Ss):
            for k, E in enumerate(Es):
                d = size_column(prob, tp, float(R),
                                S=(None if S is None else float(S)),
                                EF=(None if E is None else float(E)))
                feas[i, j, k] = d["feasible"]
                if d["feasible"]:
                    stages[i, j, k] = d["N_total"]
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
