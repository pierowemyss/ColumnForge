"""Feed-stage junction test (blueprint Sec 7, docs/papers/bvm_connect_criteria.md).

Two sections meet at a feed, and the condition is that their *liquid* profiles
INTERSECT: there exist continuous stage coordinates with

    x_R(xi_R) == x_S(xi_S)                                               (X)

Liquid equality is the primary criterion (criteria doc Sec 5); vapour agreement
follows from it, because y = K(x) x is a function of x. The converse is not
true, and that is the whole point of this module's history: this test used to be
run on the vapour, and a vapour match does NOT imply a liquid match wherever
dy/dx is small. On c2-c4 (ethane/propane/n-butane, R=0.175, E=0.5) the two
vapour curves cross EXACTLY at (xi_R, xi_S) = (6.31, 12.04) while the liquids
there are far apart; the real liquid crossing is at (10.17, 6.17), four stages
down the column. A rigorous MESH sweep over the feed stage puts the optimum at
10-11, and the old answer of 7 lost 2 points of light-key recovery.

The same compression made R_min ~22% low: at R=0.1138 the vapour gap was 0.019
against a tolerance that reached 0.10, while the liquids at that junction were
0.072 apart. Measured on the liquid, c2-c4's R_min is ~0.139, against 0.1342
from the RBM module and 0.1466 from Underwood.

An earlier docstring here claimed the liquid profiles cannot meet, because a
feed jumps the liquid composition. That conflated two different distances. The
jump is between ADJACENT STAGES either side of the feed -- x_{f-1} on the upper
section against x_f on the lower one, 0.11 in entrainer across the main feed of
ethanol/water/EG -- and it is real; `gap_liquid` reports it. But x_f itself lies
on BOTH curves: the vapour leaving the feed stage is what the section above puts
its operating line on, so x_f = dew(y_f) whichever section computes it, and the
curves cross there. Verified on BTX at every reflux tested: gap ~1e-16.

Stages per section are read off the crossing (criteria doc Sec 4, step 5): the
crossing index on the upper profile IS the feed stage, so `nA` (the last stage
above the feed) is one less, and `nB` is the crossing index on the lower one.
Both stay fractional -- the caller rounds.
"""

import numpy as np

#: Geometric tolerance on the crossing itself, in mole-fraction L2. A real
#: transversal intersection lands at ~1e-16 (segment/segment solve); the nearest
#: near-miss measured across BTX, c2-c4 and the extractive case is 5e-3. Three
#: orders of margin on both sides, so this is not a tuning knob.
CROSS_TOL = 1e-6

#: Largest one-stage step that still counts as a junction at C >= 4, where an
#: exact crossing provably does not exist (see `connect`). Caps a stiff section's
#: first step -- the C=6 reference column moves 0.46 in one stage near the
#: distillate -- so the near-miss allowance cannot swallow half the simplex.
STEP_CAP = 0.05


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
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e < 1e-30:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-30 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    cp = p1 + s * d1
    cq = q1 + t * d2
    return float(np.linalg.norm(cp - cq)), float(s), float(t), 0.5 * (cp + cq)


def _travel_end(X, tol=CROSS_TOL):
    """Number of leading segments that still travel -- the pinch-tail onset.

    Criteria doc Sec 6: a profile that has stopped moving needs infinite stages
    to get anywhere, so the crawl at the end of a pinched march is not somewhere
    a feed can be placed. Trimming it also keeps the scan from matching against a
    pile-up of near-coincident points.

    Cut where the step falls below the crossing tolerance and stays there -- a
    step smaller than that cannot make a crossing this test could tell apart. A
    fraction-of-the-largest-step rule was tried first and is wrong: 2% of the max
    step cut c2-c4's genuine crossing off the stripping profile.
    """
    if len(X) < 3:
        return len(X) - 1
    step = np.linalg.norm(np.diff(X, axis=0), axis=1)
    k = len(step)
    while k > 1 and step[k - 1] < tol:
        k -= 1
    return k


def _at(X, f):
    """Profile composition at fractional stage coordinate f."""
    i = min(max(int(np.floor(f)), 0), len(X) - 2)
    return X[i] + (f - i) * (X[i + 1] - X[i])


def _no_connection(XA, XB, eps_stage):
    """Result for a pair that has no segment to compare (a 1-point march)."""
    return {"connected": False, "dmin": float(np.inf), "nA": 0.0, "nB": 0.0,
            "point": XA[0], "pointA": XA[0], "pointB": XB[0],
            "gap_liquid": float(np.linalg.norm(XA[0] - XB[0])),
            "residual_vapour": float(np.inf), "approximate": False,
            "in_simplex": False, "tol": float(eps_stage)}


def connect(profA, profB, secA, tp, P, eps_stage=1e-2, efficiency=1.0,
            strict=True):
    """Feed-stage junction test between the section above a feed and the one below.

    Locates the intersection (X) of the two marched liquid profiles by an
    all-pairs segment/segment scan over their travelling parts, and returns

        nA  last stage ABOVE the feed  (crossing index on profA, minus one)
        nB  the feed stage itself      (crossing index on profB)

    both fractional. `connected` is a statement about an actual crossing, not
    about proximity: the criteria doc's Sec 1 point that a small distance is not
    sufficient is exactly the bug this replaced.

    `strict=False` drops to the same near-miss rule for a junction that involves a
    saddle-launched INTERIOR curve (`driver._choose_interior`). That curve is a
    manifold arm, not a product-anchored march, and on 2-propanol/water/EG no arm
    reaches both its neighbours exactly: the best pair leaves 0.082 at the lower
    junction and the gap only creeps to 0.068 by E/F = 2. Demanding a crossing
    there would reject every extractive design the arm construction can produce,
    so those junctions are accepted within a stage and flagged `approximate`
    until the arms themselves are fixed.

    C >= 4 is the other place a near miss is still accepted. Two 1-D curves in the
    (C-1)-simplex carry 2 free coordinates against C-1 equations, so at C >= 4
    the system is over-determined and an exact crossing generically does not
    exist: the quaternary reference case sits at 8.4e-3 and the C=6 one at 0.035,
    at every reflux. The missing degrees of freedom are the non-key distillate
    splits (`problem.free_split_indices`), held at a trace-floor guess -- and
    `driver.solve_omega`, which is supposed to solve them, does not converge at
    C >= 4 (it drives the splits to the simplex corners: residual 0.139 at C=4,
    0.064 at C=6 after 204 s). So until that is replaced, C >= 4 is accepted
    within one stage of travel and flagged `approximate`; the flag is the
    honest part, and callers must not read it as a crossing.

    `efficiency` no longer widens the tolerance at C <= 3 -- it used to divide it,
    doubling the accepted gap at E=0.5, which is half of why R_min came out low.
    It still does on the C >= 4 near-miss path, where there is no crossing to
    tighten onto.

    ponytail: O(N*M) scan. Profiles are a few hundred points and the marching,
    not this, is what costs; the criteria doc's two-pointer walk is the upgrade
    path if that ever stops being true.
    """
    XA, XB = profA["X"], profB["X"]
    if len(XA) < 2 or len(XB) < 2:
        return _no_connection(XA, XB, eps_stage)

    kA, kB = _travel_end(XA), _travel_end(XB)
    best = (np.inf, 0.0, 0.0, XA[0])
    bi = bj = 0
    for i in range(kA):
        for j in range(kB):
            d, s, t, mid = _seg_seg(XA[i], XA[i + 1], XB[j], XB[j + 1])
            if d < best[0]:
                best = (d, s, t, mid)
                bi, bj = i, j
    dmin, s, t, mid = best

    # criteria doc Sec 4/5: the crossing index on the UPPER profile is the feed
    # stage, because that is the liquid both sections share. The last stage above
    # it is one less, which is what `driver._concat` slices on.
    cross = bi + s
    nA = max(cross - 1.0, 0.0)
    nB = bj + t

    if strict and XA.shape[1] <= 3:
        tol = CROSS_TOL
    else:
        # one stage of travel, from the larger side: at the junction one profile
        # is often pinched and its own step has collapsed to ~0. Divided by E
        # because a marched step is already efficiency-scaled and the bridge is an
        # EQUILIBRIUM stage's reach. This is the shipped rule, moved into liquid
        # space -- deliberately unchanged, because neither of these paths has a
        # crossing to tighten onto.
        locA = float(np.linalg.norm(XA[bi + 1] - XA[bi]))
        locB = float(np.linalg.norm(XB[bj + 1] - XB[bj]))
        loc = max(locA, locB)
        if strict:
            # C >= 4: cap a stiff section's first step (the C=6 reference column
            # moves 1.36 in one stage) so the test cannot go vacuous.
            loc = min(loc, STEP_CAP)
        tol = max(float(eps_stage), loc / max(float(efficiency), 1e-6))
    in_simplex = bool(mid.min() > -1e-6 and mid.max() < 1.0 + 1e-6)
    connected = bool(dmin <= tol and in_simplex)

    pA = _at(XA, nA)                       # liquid on the last stage above the feed
    pB = _at(XB, nB)                       # liquid on the feed stage
    # The junction equation (E), a x_{f-1} + b == y_f: REPORTED, never gated on.
    # At E = 1 liquid equality implies it and it comes out ~0 (1e-3 on BTX, all
    # interpolation error). Below E = 1 it does not, and the residual measures a
    # real inconsistency in the marching model rather than a bad junction: the
    # rectifying march computes x_{f-1} assuming stage f carries rectifying flows,
    # when it carries stripping ones. 0.066 on c2-c4 at E = 0.5. Gating on it is
    # what put the feed four stages too high.
    #
    # Uses the lower march's OWN vapour: recomputing K(x)x here would silently
    # drop the Murphree efficiency the march applied (up to 0.25 in y at E = 0.5).
    yA = secA.a * pA + secA.bvec
    yB = _at(profB["Y"], nB) if len(profB.get("Y", ())) > 1 else yA
    return {
        "connected": connected,
        "dmin": float(dmin),
        "nA": float(nA),
        "nB": float(nB),
        "point": mid,                      # the crossing, a LIQUID composition
        "pointA": pA,
        "pointB": pB,
        # the feed jump: adjacent stages either side of the feed, pinned by their
        # own section balances (0.11 in entrainer across ethanol/water/EG's main
        # feed). Supposed to be non-zero; nothing is gated on it.
        "gap_liquid": float(np.linalg.norm(pA - pB)),
        "residual_vapour": float(np.linalg.norm(yA - yB)),
        "approximate": bool(connected and dmin > CROSS_TOL),
        "in_simplex": in_simplex,
        "tol": float(tol),
    }


def _demo():
    from .march import march_section
    from .problem import build_problem, overall_balance
    from .sections import single_feed_chain
    from .thermo_adapter import ColumnForgeThermo

    abc = np.array(
        [
            (6.90565, 1211.033, 220.79),
            (6.95464, 1344.8, 219.48),
            (6.99052, 1453.43, 215.31),
        ]
    )
    tp = ColumnForgeThermo(abc)
    z = np.array([0.4, 0.35, 0.25])
    prob = build_problem(
        ["b", "t", "x"], [(z, 100.0, 1.0)], 760.0, rec_lk=0.98, rec_hk=0.02
    )
    xD, xB, D, B = overall_balance(prob)

    # feasible reflux: the profiles genuinely cross, finite stage counts
    rect, strip = single_feed_chain(prob, 5.0, xD, xB, D, B)
    r = march_section(rect, xD, tp, 760.0, 80)
    s = march_section(strip, xB, tp, 760.0, 80)
    c = connect(r, s, rect, tp, 760.0)
    assert c["connected"], f"R=5 should connect, dmin={c['dmin']:.3g}"
    assert c["dmin"] < 1e-9, f"a real crossing, not a near miss: {c['dmin']:.3g}"
    assert not c["approximate"] and c["in_simplex"]
    assert c["nA"] > 0 and c["nB"] > 0
    N = int(np.ceil(c["nA"]) + np.ceil(c["nB"]))
    assert 5 < N < 80, f"total stages {N} implausible"

    # the index convention, which is the easy thing to get backwards: the feed
    # stage is the crossing on the UPPER profile, so nA is one stage above it and
    # the liquid there is a stage of travel away from the feed-stage liquid.
    x_feed = _at(r["X"], c["nA"] + 1.0)
    assert np.allclose(x_feed, c["pointB"], atol=1e-6), (x_feed, c["pointB"])
    assert c["gap_liquid"] > 1e-3, "pointA/pointB are adjacent stages, not the crossing"

    # liquid equality implies vapour equality (criteria doc Sec 5) -- the old
    # primary test, demoted to a consequence
    assert c["residual_vapour"] <= 1e-2, c["residual_vapour"]

    # below R_min the profiles pinch apart and no tolerance may buy a connection
    rect2, strip2 = single_feed_chain(prob, 0.4, xD, xB, D, B)
    c2 = connect(
        march_section(rect2, xD, tp, 760.0, 80),
        march_section(strip2, xB, tp, 760.0, 80),
        rect2,
        tp,
        760.0,
    )
    assert not c2["connected"], f"R=0.4 should not connect, dmin={c2['dmin']:.3g}"

    # pinch tails are trimmed off the search (criteria doc Step 0)
    crawl = np.vstack([np.linspace([0.9, 0.1, 0.0], [0.4, 0.5, 0.1], 6),
                       np.full((20, 3), [0.4, 0.5, 0.1])])
    assert _travel_end(crawl) == 5, _travel_end(crawl)

    # segment-segment sanity: two crossing unit segments meet at the origin
    d, ss, tt, mid = _seg_seg(
        np.array([-1.0, 0, 0]),
        np.array([1.0, 0, 0]),
        np.array([0, -1.0, 0]),
        np.array([0, 1.0, 0]),
    )
    assert d < 1e-9 and np.allclose(mid, 0.0)
    print(f"connect self-check OK  dmin={c['dmin']:.3g} nA={c['nA']:.2f} "
          f"nB={c['nB']:.2f}  N~{N}")


if __name__ == "__main__":
    _demo()
