"""Closest-approach connection test (blueprint Sec 7).

Two 1-D profile curves in the (C-1)-simplex generically do NOT cross for C>=4
(dim of intersection = 3 - C). The classic "profiles intersect" test is a
ternary accident; projecting to an LK/HK plane forces a crossing that isn't
there. The honest, dimension-independent test is minimum distance in full space:

    min_{s,t} || xA(s) - xB(t) ||          (Sec 7.1)

Profiles are polylines here, so this is a segment-segment closest-approach scan.
The split *connects* when the minimum is within one stage step and the crossing
sits inside the simplex. Stages per section = steps from each product end to the
junction; the feed/draw stage is the junction location (Sec 7.1).
"""

import numpy as np


def _seg_seg(p1, p2, q1, q2):
    """Closest approach between segments p1->p2 and q1->q2 in R^n.

    Returns (dist, s, t, midpoint) with s,t in [0,1] the fractional positions.
    Standard clamped formulation (Ericson, Real-Time Collision Detection).
    """
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r
    if a < 1e-30 and e < 1e-30:
        s = t = 0.0
    elif a < 1e-30:
        s = 0.0; t = np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e < 1e-30:
            t = 0.0; s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-30 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0; s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0; s = np.clip((b - c) / a, 0.0, 1.0)
    cp = p1 + s * d1
    cq = q1 + t * d2
    return float(np.linalg.norm(cp - cq)), float(s), float(t), 0.5 * (cp + cq)


def connect(profA, profB, eps_stage=1e-2, efficiency=1.0):
    """Closest approach between two marched profiles (dicts from march_section).

    Returns dict(connected, dmin, nA, nB, point, in_simplex, tol). nA/nB are the
    fractional stage counts from each anchor to the junction. `connected` holds
    when dmin is within one stage step -- the *local* step at the winning segment
    pair, not the whole-profile median. Both profiles spend most of their length
    pinched, so a median step collapses to ~0 and the eps_stage floor wrongly
    rules for C>=4 (Sec 7.1); the meaningful question is whether the gap is within
    a stage of travel *at the junction*.

    With Murphree efficiency E<1 the per-stage composition step is ~E times the
    equilibrium step, so the raw local step understates the gap a feed stage can
    bridge. The junction is a feed mixing jump set by the thermodynamics, not by
    E; dividing the tolerance by E recovers the equilibrium-scale bridge width
    (E=1 leaves the ideal-stage behaviour unchanged).
    """
    E = max(float(efficiency), 1e-6)
    XA, XB = profA["X"], profB["X"]
    best = (np.inf, 0.0, 0.0, XA[0])
    bi = bj = 0
    for i in range(len(XA) - 1):
        for j in range(len(XB) - 1):
            d, s, t, mid = _seg_seg(XA[i], XA[i + 1], XB[j], XB[j + 1])
            if d < best[0]:
                best = (d, s, t, mid); bi, bj = i, j
    dmin, s, t, mid = best
    nA = bi + s
    nB = bj + t
    pA = XA[bi] + s * (XA[bi + 1] - XA[bi])       # closest point on each curve
    pB = XB[bj] + t * (XB[bj + 1] - XB[bj])
    locA = float(np.linalg.norm(XA[bi + 1] - XA[bi]))   # local step at the junction
    locB = float(np.linalg.norm(XB[bj + 1] - XB[bj]))
    tol = max(eps_stage, 0.5 * (locA + locB) / E)
    in_simplex = bool(mid.min() > -1e-6 and mid.max() < 1.0 + 1e-6)
    return {"connected": bool(dmin <= tol and in_simplex), "dmin": float(dmin),
            "nA": float(nA), "nB": float(nB), "point": mid,
            "pointA": pA, "pointB": pB,
            "in_simplex": in_simplex, "tol": float(tol)}


def _demo():
    from thermo_adapter import FreeColumnThermo
    from problem import build_problem, overall_balance
    from sections import single_feed_chain
    from march import march_section

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = FreeColumnThermo(abc)
    z = np.array([0.4, 0.35, 0.25])
    prob = build_problem(["b", "t", "x"], [(z, 100.0, 1.0)], 760.0,
                         rec_lk=0.98, rec_hk=0.02)
    xD, xB, D, B = overall_balance(prob)

    # feasible reflux: profiles connect, finite stage counts
    rect, strip = single_feed_chain(prob, 5.0, xD, xB, D, B)
    r = march_section(rect, xD, tp, 760.0, 80)
    s = march_section(strip, xB, tp, 760.0, 80)
    c = connect(r, s)
    assert c["connected"], f"R=5 should connect, dmin={c['dmin']:.3g} tol={c['tol']:.3g}"
    assert c["in_simplex"] and c["nA"] > 0 and c["nB"] > 0
    N = int(np.ceil(c["nA"]) + np.ceil(c["nB"]))
    assert 5 < N < 80, f"total stages {N} implausible"

    # segment-segment sanity: two crossing unit segments meet at the origin
    d, ss, tt, mid = _seg_seg(np.array([-1., 0, 0]), np.array([1., 0, 0]),
                              np.array([0, -1., 0]), np.array([0, 1., 0]))
    assert d < 1e-9 and np.allclose(mid, 0.0)
    print(f"connect self-check OK  dmin={c['dmin']:.4f} tol={c['tol']:.4f}  N~{N}")


if __name__ == "__main__":
    _demo()
