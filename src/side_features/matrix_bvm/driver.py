"""Design driver -- size one column, sweep the design map (blueprint Sec 10).

`size_column` runs the whole method at one operating point: overall balance ->
difference-point chain -> march product-anchored sections -> anchor/march
interior sections -> connect (closest approach) -> place feeds/draws -> assemble
the full top->bottom column -> classify feasibility. `feasibility_map` sweeps
(R, S, E/F) and records feasibility + stage count on the grid so the UI can draw
a heatmap and let the user click a point to load that column.

Stage 0 = distillate (top), matching the FreeColumn GUI convention.
"""

import numpy as np

from problem import overall_balance, SideDraw
from sections import (single_feed_chain, extractive_chain, multifeed_chain)
from march import march_section
from connect import connect
from place import crossover_stage, side_draw_stage
from anchor import saddle_pinch, launch_from_saddle
from diagnostics import classify
from pinch import bisect_min


def _concat(top_prof, top_n, bot_prof_rev, bot_n, secs_top, secs_bot):
    """Join a top (marched down) and bottom (marched up, reversed) profile.

    Returns full-column x, y, T, L, V arrays top->bottom and the boundary stage
    index (the feed/junction). Section flows L,V are constant within a section.
    """
    xt = top_prof["X"][:top_n + 1]
    Tt = top_prof["T"][:top_n + 1]
    yt = top_prof["Y"][:top_n + 1]
    # bottom profile is anchored at the product and marched up; reverse to
    # top->bottom and drop its topmost point (the shared junction, otherwise the
    # feed stage is counted twice and N_total inflates by one).
    xb = bot_prof_rev["X"][:bot_n + 1][::-1][1:]
    Tb = bot_prof_rev["T"][:bot_n + 1][::-1][1:]
    yb = bot_prof_rev["Y"][:bot_n + 1][::-1][1:]
    x = np.vstack([xt, xb])
    T = np.concatenate([Tt, Tb])
    y = np.vstack([yt, yb])
    Lt, Vt = secs_top.L, secs_top.V
    Lb, Vb = secs_bot.L, secs_bot.V
    L = np.concatenate([np.full(len(xt), Lt), np.full(len(xb), Lb)])
    V = np.concatenate([np.full(len(xt), Vt), np.full(len(xb), Vb)])
    feed_stage = len(xt) - 1
    return x, y, T, L, V, feed_stage


def _base(prob, xD, xB, D, B, R):
    f = prob.z_total
    return {"feasible": False, "findings": [], "comps": list(prob.comps),
            "pressure": prob.pressure, "xD": xD, "xB": xB, "D": D, "B": B,
            "lk": prob.lk, "hk": prob.hk, "feed_z": f / f.sum(),
            "R": R, "S": None, "EF": None, "R_min": None, "EF_min": None,
            "N_total": None, "feed_stages": [], "column": None,
            "sections": {}, "profiles": {}, "connection": None}


def _size_two(prob, tp, rect, strip, xD, xB, P, out):
    """Single-feed (two product-anchored sections) sizing + assembly."""
    rprof = march_section(rect, xD, tp, P, prob.max_stages)
    sprof = march_section(strip, xB, tp, P, prob.max_stages)
    profiles = {"rectifying": rprof, "stripping": sprof}
    conn = connect(rprof, sprof)
    both_pinched = rprof["pinched"] and sprof["pinched"]
    side = side_draw_stage(sprof, prob.side_draws[0]) if prob.side_draws else None
    feasible, findings = classify(profiles, conn, both_pinched=both_pinched,
                                  side_draw=side)
    out["feasible"] = feasible; out["findings"] = findings
    out["sections"] = {k: {"n": v["n"], "status": v["status"]}
                       for k, v in profiles.items()}
    out["profiles"] = profiles; out["connection"] = conn
    if not feasible:
        return out
    x, y, T, L, V, feed_stage = _concat(rprof, int(np.ceil(conn["nA"])),
                                        sprof, int(np.ceil(conn["nB"])), rect, strip)
    out["N_total"] = len(x); out["feed_stages"] = [feed_stage]
    if side is not None:
        out["side_draw_stage"] = feed_stage + 1 + side["stage"]
    out["column"] = {"x": x, "y": y, "T": T, "liquid_flow": L, "vapor_flow": V,
                     "feed_stage": feed_stage}
    return out


def _mid_march(mid_sec, anchor, tp, P, max_stages):
    """March the interior section from `anchor`, escalating to a saddle launch
    if the ordinary continuation crawls to the cap without pinching (Sec 6.3)."""
    mprof = march_section(mid_sec, anchor, tp, P, max_stages)
    if mprof["status"] == "max" and not mprof["pinched"]:
        cl = saddle_pinch(mid_sec, anchor, tp, P)
        if cl["saddle"]:
            mprof = launch_from_saddle(mid_sec, cl["xstar"], cl["eigvecs"][:, 0],
                                       tp, P, n=max_stages)
    return mprof


def _size_three(prob, tp, top_sec, mid_sec, bot_sec, xD, xB, P, out, extractive):
    """Three-section (multifeed / extractive) sizing routed THROUGH the interior.

    The switch stage where the interior section begins is a *free parameter*
    (Knapp-Doherty): the two product-anchored profiles have no reason to approach
    each other in extractive mode -- the interior section is the bridge. So we
    scan every candidate switch stage on the anchoring profile, march the interior
    from there, and test whether it connects to the opposite product profile;
    feasible iff any launch bridges, and we keep the minimum-stage one (Sec 6.2).
    """
    tprof = march_section(top_sec, xD, tp, P, prob.max_stages)
    bprof = march_section(bot_sec, xB, tp, P, prob.max_stages)

    best = None                    # (N_total, pieces dict) minimised over switch stage
    rep = None                     # a representative (mprof, conn) for diagnostics
    if mid_sec.dir > 0:            # interior marches down: continue off the top profile
        for k in range(1, tprof["n"]):
            mprof = _mid_march(mid_sec, tprof["X"][k], tp, P, prob.max_stages)
            low = connect(mprof, bprof)
            if rep is None:
                rep = (mprof, low)
            if not low["connected"]:
                continue
            mid_n = int(np.ceil(low["nA"])); bot_n = int(np.ceil(low["nB"]))
            N = k + mid_n + bot_n
            if best is None or N < best[0]:
                best = (N, {"upper_n": k, "mprof": mprof, "mid_n": mid_n,
                            "bot_n": bot_n, "conn": low, "reverse_mid": False})
    else:                         # interior marches up: continue off the bottom profile
        for l in range(1, bprof["n"]):
            mprof = _mid_march(mid_sec, bprof["X"][l], tp, P, prob.max_stages)
            up = connect(tprof, mprof)
            if rep is None:
                rep = (mprof, up)
            if not up["connected"]:
                continue
            upper_n = int(np.ceil(up["nA"])); mid_n = int(np.ceil(up["nB"]))
            N = upper_n + mid_n + l
            if best is None or N < best[0]:
                best = (N, {"upper_n": upper_n, "mprof": mprof, "mid_n": mid_n,
                            "bot_n": l, "conn": up, "reverse_mid": True})

    if rep is None:                # top/bottom profiles too short to scan
        rep = (march_section(mid_sec, 0.5 * (xD + xB), tp, P, prob.max_stages),
               {"connected": False, "in_simplex": True, "dmin": np.inf, "tol": 0.0,
                "point": 0.5 * (xD + xB), "pointA": xD, "pointB": xB})

    pieces = best[1] if best is not None else None
    mprof_diag, conn = (pieces["mprof"], pieces["conn"]) if best else rep
    # orient the interior profile top->bottom for display/assembly
    if best and pieces["reverse_mid"]:
        m = pieces["mprof"]; mid_n = pieces["mid_n"]
        mprof_oriented = {**m, "X": m["X"][:mid_n + 1][::-1],
                          "Y": m["Y"][:mid_n + 1][::-1], "T": m["T"][:mid_n + 1][::-1]}
    else:
        mprof_oriented = mprof_diag

    profiles = {top_sec.name: tprof, mid_sec.name: mprof_oriented, bot_sec.name: bprof}
    both_pinched = mprof_diag["pinched"] and bprof["pinched"]
    feasible = best is not None
    _, findings = classify(profiles, conn, both_pinched=both_pinched,
                           extractive=extractive)
    if feasible:
        findings = []
    out["feasible"] = feasible; out["findings"] = findings
    out["sections"] = {top_sec.name: {"n": tprof["n"], "status": tprof["status"]},
                       mid_sec.name: {"n": mprof_oriented.get("n", len(mprof_oriented["X"])),
                                      "status": mprof_oriented.get("status", "assembled")},
                       bot_sec.name: {"n": bprof["n"], "status": bprof["status"]}}
    out["profiles"] = profiles; out["connection"] = conn
    if not feasible:
        return out

    # assemble top -> bottom, dropping the two shared junction stages (upper feed/
    # entrainer, lower feed) so each feed stage is counted exactly once.
    upper_n, mid_n, bot_n = pieces["upper_n"], pieces["mid_n"], pieces["bot_n"]
    xt = tprof["X"][:upper_n + 1]; Tt = tprof["T"][:upper_n + 1]
    yt = tprof["Y"][:upper_n + 1]
    xm = mprof_oriented["X"][1:mid_n + 1]; Tm = mprof_oriented["T"][1:mid_n + 1]
    ym = mprof_oriented["Y"][1:mid_n + 1]
    xb = bprof["X"][:bot_n + 1][::-1][1:]; Tb = bprof["T"][:bot_n + 1][::-1][1:]
    yb = bprof["Y"][:bot_n + 1][::-1][1:]
    x = np.vstack([xt, xm, xb]); T = np.concatenate([Tt, Tm, Tb])
    y = np.vstack([yt, ym, yb])
    L = np.concatenate([np.full(len(xt), top_sec.L), np.full(len(xm), mid_sec.L),
                        np.full(len(xb), bot_sec.L)])
    Vv = np.concatenate([np.full(len(xt), top_sec.V), np.full(len(xm), mid_sec.V),
                         np.full(len(xb), bot_sec.V)])
    upper_feed = len(xt) - 1
    lower_feed = len(xt) + len(xm) - 1
    out["N_total"] = len(x); out["feed_stages"] = [upper_feed, lower_feed]
    out["column"] = {"x": x, "y": y, "T": T, "liquid_flow": L, "vapor_flow": Vv,
                     "feed_stage": upper_feed, "feed_stages": [upper_feed, lower_feed]}
    return out


def size_column(prob, tp, R, S=None, EF=None):
    """Size the column at (R, S, EF). Returns a `design` dict (see module doc)."""
    extractive = prob.extractive and prob.x_E is not None
    xD, xB, D, B = overall_balance(prob, EF if extractive else None)
    P = prob.pressure
    out = _base(prob, xD, xB, D, B, R)
    out["S"] = S; out["EF"] = EF

    if extractive:
        rect, ext, strip = extractive_chain(prob, R, EF, xD, xB, D, B)
        return _size_three(prob, tp, rect, ext, strip, xD, xB, P, out, True)
    if len(prob.feeds) > 1:
        secs = multifeed_chain(prob, R, xD, xB, D, B)
        if len(secs) == 3:
            return _size_three(prob, tp, secs[0], secs[1], secs[2], xD, xB, P,
                               out, False)
        # >3 sections not assembled yet -> size the enclosing two-section problem
        return _size_two(prob, tp, secs[0], secs[-1], xD, xB, P, out)
    rect, strip = single_feed_chain(prob, R, xD, xB, D, B)
    return _size_two(prob, tp, rect, strip, xD, xB, P, out)


def r_min(prob, tp, R_hi=30.0, S=None, EF=None, tol=1e-2):
    """Minimum reflux by bisection on the connection test (Sec 8)."""
    def feasible(R):
        return size_column(prob, tp, R, S=S, EF=EF)["feasible"]
    return bisect_min(feasible, 0.05, R_hi, tol=tol)


def ef_min(prob, tp, R, EF_hi=5.0, tol=1e-2):
    """Minimum entrainer-to-feed ratio by bisection (extractive mode, Sec 8)."""
    def feasible(EF):
        return size_column(prob, tp, R, EF=EF)["feasible"]
    return bisect_min(feasible, 0.0, EF_hi, tol=tol)


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
    from thermo_adapter import FreeColumnThermo
    from problem import build_problem

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = FreeColumnThermo(abc)
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
