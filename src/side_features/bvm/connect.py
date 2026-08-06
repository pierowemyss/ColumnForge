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

#: Largest one-stage step that still counts as a junction on either near-miss
#: path -- C >= 4, where an exact crossing provably does not exist, and a
#: saddle-launched interior arm (see `connect`). Caps a stiff section's step --
#: the C=6 reference column moves 0.46 in one stage near the distillate, and an
#: extractive column's rectifying section moves 0.75 -- so the near-miss
#: allowance cannot swallow half the simplex.
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


def _closest_pair(XA, kA, XB, kB):
    """(i, j, dist, s, t, midpoint) for the closest of all kA*kB segment pairs.

    The same clamped formulation as `_seg_seg`, run over every pair at once.
    Ties break to the lowest (i, j) exactly as the scalar loop did, because
    `argmin` on a C-ordered (kA, kB) grid scans in that order.

    ponytail: materialises a (kA, kB, C) difference array. Profiles are bounded
    by `Problem.max_column_stages` (~100s of stages, C <= ~6), so that is a few
    MB; chunk over i if a column ever gets big enough to matter.
    """
    P1, D1 = XA[:kA], XA[1:kA + 1] - XA[:kA]
    Q1, D2 = XB[:kB], XB[1:kB + 1] - XB[:kB]
    a = np.einsum("ik,ik->i", D1, D1)                    # (kA,)
    e = np.einsum("jk,jk->j", D2, D2)                    # (kB,)
    r = P1[:, None, :] - Q1[None, :, :]                  # (kA, kB, C)
    f = np.einsum("jk,ijk->ij", D2, r)
    c = np.einsum("ik,ijk->ij", D1, r)
    b = D1 @ D2.T

    tiny = 1e-30
    a_ok, e_ok = a[:, None] > tiny, e[None, :] > tiny
    a_safe = np.where(a_ok, a[:, None], 1.0)
    e_safe = np.where(e_ok, e[None, :], 1.0)
    denom = a[:, None] * e[None, :] - b * b
    denom_ok = denom > tiny

    # general case, both segments non-degenerate
    s = np.where(denom_ok,
                 np.clip((b * f - c * e[None, :]) / np.where(denom_ok, denom, 1.0),
                         0.0, 1.0), 0.0)
    t = (b * s + f) / e_safe
    # t off either end pins that segment and re-solves for s
    s = np.where(t < 0.0, np.clip(-c / a_safe, 0.0, 1.0),
                 np.where(t > 1.0, np.clip((b - c) / a_safe, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)

    # degenerate segments (a marched profile can repeat a point once it pinches)
    s = np.where(a_ok, s, 0.0)
    t = np.where(a_ok | ~e_ok, t, np.clip(f / e_safe, 0.0, 1.0))   # point vs segment
    s = np.where(e_ok | ~a_ok, s, np.clip(-c / a_safe, 0.0, 1.0))  # segment vs point
    t = np.where(e_ok, t, 0.0)

    cp = P1[:, None, :] + s[..., None] * D1[:, None, :]
    cq = Q1[None, :, :] + t[..., None] * D2[None, :, :]
    diff = cp - cq
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    flat = int(np.argmin(dist))
    i, j = divmod(flat, kB)
    return (i, j, float(dist[i, j]), float(s[i, j]), float(t[i, j]),
            0.5 * (cp[i, j] + cq[i, j]))


def travel_end(X, tol=CROSS_TOL):
    """Number of leading segments that still travel -- the pinch-tail onset.

    Criteria doc Sec 6: a profile that has stopped moving needs infinite stages
    to get anywhere, so the crawl at the end of a pinched march is not somewhere
    a feed can be placed. Trimming it also keeps the scan from matching against a
    pile-up of near-coincident points.

    Cut where the step falls below the crossing tolerance and stays there -- a
    step smaller than that cannot make a crossing this test could tell apart. A
    fraction-of-the-largest-step rule was tried first and is wrong: 2% of the max
    step cut c2-c4's genuine crossing off the stripping profile.

    CROSS_TOL is also the right depth, not merely a safe one, and the temptation
    to trim harder should be resisted. Measured where the junctions of the shipped
    cases actually land relative to this cut:

        multicomp eff=1.0    stripping  n=46  junction@44.00  step 9.5e-7  trim@44
        multicomp eff=0.75   stripping  n=61  junction@59.00  step 9.5e-7  trim@59
        extract   eff=0.5    extractive n=87  junction@ 3.81  step 8.2e-4  trim@86

    Every one of them sits at or just inside the cut, on a step of ~1e-6 -- i.e.
    right at the pinch onset, which is exactly where a boundary-value junction
    belongs. And the first of those is the design that returns N=46 against the
    file's Inside-Out reference of 45 real stages. Any stricter rule deletes that
    junction and with it the case's agreement with MESH. The 104-stage answer this
    module was suspected of producing through a pinch tail came from the
    efficiency-scaled tolerance in `connect`, not from an untrimmed crawl.
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
            strict=True, step_cap=STEP_CAP):
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
    `splits.solve_free_splits`, which is supposed to solve them, does not converge at
    C >= 4 (it drives the splits to the simplex corners: residual 0.139 at C=4,
    0.064 at C=6 after 204 s). So until that is replaced, C >= 4 is accepted
    within one stage of travel and flagged `approximate`; the flag is the
    honest part, and callers must not read it as a crossing.

    `efficiency` no longer widens the tolerance anywhere -- it used to divide it,
    doubling the accepted gap at E=0.5, which is half of why R_min came out low.
    It was removed from the C <= 3 crossing branch first and from the near-miss
    branches second; the argument is the same in both places and is written out
    at the `tol` assignment below. The parameter is kept in the signature because
    callers pass it positionally, and ignored.

    `step_cap` bounds that local step so the near-miss allowance cannot swallow
    half the simplex, and it now applies to the INTERIOR path as well as C >= 4.
    An extractive column's rectifying section amplifies the entrainer ~45x per
    stage, so on ipa/water/EG its marched segments are 0.586 and 0.751 long, and
    the uncapped rule handed the interior junction a tolerance of 0.130-1.209 --
    larger than most distances in the simplex. Every arm of the saddle then
    "connected" at the upper junction at every reflux tested, so that junction
    stopped discriminating between them and a 0.289 miss read as a connection. A
    junction located on a 0.6-long chord is uncertain to +/-0.3 and is not a
    junction.

    Pass `step_cap=None` where the near miss is known to be structural rather than
    a resolution problem. `driver._size_two` does exactly that for a REACTIVE
    column: in transformed coordinates the reduced non-key split pins the
    rectifying profile on a face of the reduced simplex, so MTBE's two profiles
    stay 0.22 apart at every reflux and there is no crossing to find at any
    tolerance. Capping there just deletes the column.

    ponytail: still an O(N*M) scan, but vectorised (`_closest_pair`) -- it was
    44% of an extractive `size_column` as a Python double loop. The criteria
    doc's two-pointer walk is the upgrade path if N*M ever stops fitting in
    memory.
    """
    XA, XB = profA["X"], profB["X"]
    if len(XA) < 2 or len(XB) < 2:
        return _no_connection(XA, XB, eps_stage)

    kA, kB = travel_end(XA), travel_end(XB)
    bi, bj, dmin, s, t, mid = _closest_pair(XA, kA, XB, kB)

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
        # is often pinched and its own step has collapsed to ~0.
        #
        # NOT divided by the efficiency. `loc` is the step the marched profile
        # actually takes, so it is already the resolution at which this test can
        # tell a crossing from a miss; dividing by E inflates it to the reach of
        # an EQUILIBRIUM stage, which is larger than any step either profile
        # takes, and the allowance then exceeds the discretisation it is supposed
        # to describe. Efficiency belongs on the stage COUNT, after the geometry
        # has closed -- it is not a licence to accept a wider gap. Concretely, on
        # the C=6 reference column at R=1 this rule at E=0.5 doubled the
        # tolerance to 0.065 and admitted a 0.054 miss as a 104-stage column;
        # the same geometry at E=1.0 was correctly below_min_reflux. A verdict
        # that flips on tray efficiency alone, with the profiles unchanged, is
        # reporting the tolerance and not the column.
        locA = float(np.linalg.norm(XA[bi + 1] - XA[bi]))
        locB = float(np.linalg.norm(XB[bj + 1] - XB[bj]))
        loc = max(locA, locB)
        if step_cap is not None:
            loc = min(loc, float(step_cap))
        tol = max(float(eps_stage), loc)
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
    assert travel_end(crawl) == 5, travel_end(crawl)

    # segment-segment sanity: two crossing unit segments meet at the origin
    d, ss, tt, mid = _seg_seg(
        np.array([-1.0, 0, 0]),
        np.array([1.0, 0, 0]),
        np.array([0, -1.0, 0]),
        np.array([0, 1.0, 0]),
    )
    assert d < 1e-9 and np.allclose(mid, 0.0)

    # the vectorised all-pairs scan must agree with the scalar kernel it replaced,
    # degenerate (repeated-point) segments included -- that is the whole contract.
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.normal(size=(7, 3))
        B = rng.normal(size=(5, 3))
        A[3] = A[2]                       # a pinched stage on each side
        B[1] = B[0]
        kA, kB = len(A) - 1, len(B) - 1
        best, bij = (np.inf, 0.0, 0.0, None), (0, 0)
        for i in range(kA):
            for j in range(kB):
                got = _seg_seg(A[i], A[i + 1], B[j], B[j + 1])
                if got[0] < best[0]:
                    best, bij = got, (i, j)
        vi, vj, vd, vs, vt, vmid = _closest_pair(A, kA, B, kB)
        assert (vi, vj) == bij and abs(vd - best[0]) < 1e-12, ((vi, vj), bij, vd)
        assert abs(vs - best[1]) < 1e-9 and abs(vt - best[2]) < 1e-9
        assert np.allclose(vmid, best[3], atol=1e-12)

    print(f"connect self-check OK  dmin={c['dmin']:.3g} nA={c['nA']:.2f} "
          f"nB={c['nB']:.2f}  N~{N}")


if __name__ == "__main__":
    _demo()
