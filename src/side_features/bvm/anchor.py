"""Section anchoring -- where each profile starts (blueprint Sec 6).

For S=2 the problem is two-point: both product ends are known. For S>2 interior
sections have no product anchor. The escalation ladder:

  6.1 product-terminated  -> start at x_D / x_B and march inward (trivial here).
  6.2 ordinary interior   -> *continuation*: the liquid composition is continuous
      across a feed (only the difference point jumps), so anchor the interior
      section at the upstream profile's composition where the sections switch,
      and keep marching in the stable direction.
  6.3 strongly pinched    -> *saddle launch*: extractive sections and pure side
      draws crawl for many stages past a bottleneck, so anchor at the section's
      saddle pinches and build the profile from their 1-D invariant manifolds.

Only a saddle (mixed |lambda|, Sec 8) has the one-dimensional manifolds that ARE
the limiting interior profile. The unstable manifold is traced by marching the
forward map off x* + eps*v_u; the stable side is reached by continuation from the
adjacent product-anchored profile.

Saddle pinches, plural: a section generally has several, and which one carries the
column is decided by the junctions, not locally (Bruggemann & Marquardt rules 1
and 5). `pinch.pinch_points` enumerates them; asking `pinch.pinch_solve` for one
root instead is what made this module report that it could not anchor an
extractive section that had two saddles sitting in its region -- see
`docs/adr/0004-extractive-anchoring-and-the-r-max-gap.md`.
"""

import numpy as np
from scipy.optimize import fsolve

from .bodies import body_id
from .march import march_section
from .pinch import (BRANCH_TOL, eigenstructure, jacobian, lift_direction,
                    pinch_points, pinch_solve)
from .sections import feasible, feasible_margin, region_center


def product_anchor(xprod):
    """Section terminated by an actual product: anchor = that composition."""
    return np.asarray(xprod, float)


def continuation_anchor(upstream_profile, switch_index):
    """Ordinary interior section (Sec 6.2): liquid comp is continuous across the
    feed, so the anchor is the upstream profile's liquid at the switch stage."""
    return upstream_profile["X"][int(switch_index)]


def saddle_pinch(sec, x_guess, tp, P):
    """Locate the section's pinch and classify it.

    Returns dict(xstar, kind, eigvals, eigvecs, saddle, in_region). The fixed
    point comes from `pinch.pinch_solve`, i.e. from the direction-free pinch
    equation K(x) x = a x + bvec seeded inside the section's feasible region --
    not from a one-step march seeded at `0.5*(x_D + x_B)`, which for every
    extractive section lies outside the region and made the residual identically
    zero (a march that cannot start returns its own anchor, so *any* infeasible
    point looked like a fixed point).
    """
    ps = pinch_solve(sec, tp, P, x_guess)
    if ps is None:                       # empty feasible region: nothing to find
        return {"kind": "no_region", "eigvals": np.zeros(sec.delta.shape[0] - 1),
                "eigvecs": np.eye(sec.delta.shape[0] - 1), "saddle": False,
                "xstar": np.asarray(x_guess, float), "in_region": False}
    xstar = ps["xstar"]
    cl = _classify(sec, xstar, tp, P)
    cl["in_region"] = ps["in_region"]
    return cl


def _classify(sec, xstar, tp, P, *, down=None):
    """Eigenstructure of the stage map at `xstar`, with `xstar` carried along.

    `down` says which way the section's profile is TRACED, and it matters: the two
    directions are exact inverses, so their eigenvalues are reciprocals and
    "stable" and "unstable" swap. It defaults to sign(Delta), correct for a
    product-anchored section, and a middle section has to pass it -- see
    `pinch.jacobian`.
    """
    got = jacobian(sec, np.asarray(xstar, float), tp, P, down=down)
    C = sec.delta.shape[0]
    if got is None:
        return {"kind": "no_region", "eigvals": np.zeros(C - 1),
                "eigvecs": np.eye(C - 1), "saddle": False, "drop": C - 1,
                "order": np.arange(C - 1), "n_stable": 0, "xstar": xstar}
    J, drop = got
    cl = eigenstructure(J)
    cl["drop"] = drop
    cl["xstar"] = xstar
    return cl


def _eigvec(cl, which):
    """One eigendirection of a classified pinch, as a full composition-space vector.

    `np.linalg.eig` column order is arbitrary, so `eigvecs[:, 0]` can be either
    direction; `eigenstructure` supplies `order`, sorted by decreasing |lambda|,
    which is the order rules 3-4 are written in (paper p.100). Lifted with the
    pinch's own `drop`: `pinch.jacobian` differentiates along the edges leaving the
    pinch's LARGEST component, not the first C-1 coordinates, so lifting as if the
    last component were dependent silently rotates the direction.
    """
    V = np.asarray(cl["eigvecs"])
    order = cl.get("order")
    if order is None:
        order = np.argsort(-np.abs(np.asarray(cl["eigvals"])))
    k = int(order[0] if which == "unstable" else order[-1])
    C = np.asarray(cl["xstar"]).shape[0]
    return lift_direction(V[:, k], C, cl.get("drop"))


def unstable_eigvec(cl):
    """The eigenvector of the largest |lambda| (E8) -- the unstable manifold, the
    one the interior profile follows AWAY from the saddle."""
    return _eigvec(cl, "unstable")


def stable_eigvec(cl):
    """The eigenvector of the smallest |lambda| -- the direction the interior
    profile *arrives* along, and the reason an extractive section is long."""
    return _eigvec(cl, "stable")


def _tangent(xstar, eigvec):
    """Normalise an eigendirection into a sum-zero (simplex-tangent) unit vector.

    Accepts either a full composition-space direction (what `_eigvec` returns) or a
    reduced (C-1) one, which is lifted assuming the last component is the dependent
    coordinate. Handing a full vector to the reduced path would drop its last entry
    and recompute it, i.e. quietly return a different direction.
    """
    C = xstar.shape[0]
    v = np.asarray(eigvec, float).real.ravel()
    if v.shape[0] == C:
        vf = v - v.mean()                    # project onto the sum-zero tangent
    else:
        vf = np.zeros(C)
        vf[:C - 1] = v[:C - 1]
        vf[C - 1] = -vf[:C - 1].sum()
    n = np.linalg.norm(vf)
    return vf / n if n > 0 else vf


def _ray_end(sec, xstar, v):
    """Farthest x* + t*v (t > 0) still in the simplex AND in the section's region.

    Both constraints are linear in t -- x >= 0 and a x + bvec >= 0 -- so the
    largest admissible t is a min over 2C ratios, no solver. Returns None if the
    ray is unbounded (v = 0).

    ponytail: the region constraint makes this NARROWER than the paper's rules
    3-4, which walk the eigenvector to the edge of composition space
    (`rbm.bodies._to_edge`). That is deliberate and it is why BVM's r_max is short
    -- on ipa/water/EG at E/F = 0.750 the arm stops 29% of the way to the edge and
    never reaches (0.732, 0, 0.268), the vertex carrying the upper junction in the
    paper's Figure 6. Swapping in the wider walk was measured: it makes every
    reflux from 1.5 to 8 report feasible, because a body that leaves the section's
    own balance intersects everything. RBM can afford the over-reach (it discards
    separately, `bodies.blocked_by_unstable_node`); a proximity test between
    marched curves cannot. Upgrade path is not a wider ray -- it is rejecting a
    junction that sits where both profiles have pinched, which needs infinitely
    many stages and which `driver._at_anchor` currently only warns about. See
    `docs/adr/0004-extractive-anchoring-and-the-r-max-gap.md`.
    """
    t = np.inf
    y0 = sec.a * xstar + sec.bvec
    for lo, slope in ((xstar, v), (y0, sec.a * v)):
        neg = slope < -1e-12
        if np.any(neg):
            t = min(t, float(np.min(-lo[neg] / slope[neg])))
    if not np.isfinite(t):
        return None
    x = np.clip(xstar + max(t, 0.0) * v, 0.0, None)
    s = x.sum()
    return x / s if s > 0 else None


#: Stages the ray extension may add to a branch that used its whole stage budget
#: (`status == "max"`), which by definition has none left over. The ray is a
#: straight body, not a marched profile, so this is a resolution for it -- the
#: caller reads the stage count off the junctions, not off this.
RAY_STAGES = 60


def _extend_along_ray(sec, xstar, v, prof, tp, P, max_extra):
    """Continue a branch that did not reach a boundary straight out to the edge.

    Bruggemann & Marquardt rules 3-4: the middle-section profile "matches the
    eigenvectors of the saddle nearly exactly and shows only small curvature", so
    the body is built by extending the eigenvector LINEARLY to the edge of
    composition space. Marching the stage map instead converges onto whatever
    node lies in the way -- on ipa/water/EG the +v_unstable arm parks at
    (0.406, 0.217, 0.378) and never reaches the rectifying profile at
    (0.607, 0.006, 0.387), so the upper junction cannot close at all.

    The gate is "did the branch reach a boundary". `simplex`, `operating_line`
    and `crossed` did, and have nothing to continue to. `pinch` and `max` did
    not: one stalled on an interior node, the other simply ran out of stage
    budget, and both are still short of the edge rule 3 asks for.

    `max` used to be missed, and the Murphree efficiency is what exposes it: every
    step is scaled by E, so halving E doubles the stages an arm needs. On
    ipa/water/EG at r=1.72, E/F=1 the arm that ends at S1 = (0.512, 0, 0.488)
    stalls on a `pinch` at E=1 and is extended, but at E=0.5 it exhausts 300
    stages at (0.299, 0.210, 0.491) and used to be handed on truncated -- 0.289
    from the rectifying section's hand-over, which is a different wedge of the
    triangle and a different rectification body.
    """
    if prof["status"] not in ("pinch", "max"):
        return prof
    if prof["status"] == "max":
        max_extra = max(max_extra, RAY_STAGES)
    if max_extra < 1:
        return prof
    end = _ray_end(sec, xstar, v)
    if end is None:
        return prof
    x0 = prof["X"][-1]
    gap = float(np.linalg.norm(end - x0))
    # ponytail: sample the ray at the LARGEST step the branch actually took, so
    # the extension carries a stage count instead of a resolution. The mean step
    # is the wrong scale -- a branch that stalls on a node spends its last dozen
    # stages moving ~0, which would turn a 0.2 gap into a hundred fictitious
    # trays. This is a lower bound on the stages the section needs (the fastest
    # motion it was measured doing on this arm); the ray is a straight body, not a
    # marched profile, so `status` says 'manifold+ray' and the section report
    # carries it. Upgrade path: take the count from the rbm body instead.
    steps = (np.linalg.norm(np.diff(prof["X"], axis=0), axis=1)
             if prof["n"] > 1 else np.array([gap]))
    step = float(steps.max()) if steps.size else gap
    if gap <= max(step, 1e-6):
        return prof                     # already at the edge; nothing to add
    k = int(min(max(np.ceil(gap / max(step, 1e-9)), 1.0), max_extra))
    Xn = x0[None, :] + (np.arange(1, k + 1) / k)[:, None] * (end - x0)[None, :]
    Yn, Tn = [], []
    for x in Xn:
        try:
            y_eq, T = tp.bubble(x, P)
        except (ValueError, FloatingPointError):
            break
        Yn.append(sec.a * x + sec.bvec if sec.dir > 0 else y_eq)
        Tn.append(T)
    if not Yn:
        return prof
    m = len(Yn)
    return {**prof, "X": np.vstack([prof["X"], Xn[:m]]),
            "Y": np.vstack([prof["Y"], Yn]),
            "T": np.concatenate([prof["T"], Tn]),
            "P": np.concatenate([prof["P"], np.full(m, prof["P"][-1])]),
            "n": prof["n"] + m, "status": "manifold+ray",
            "travel": float(np.linalg.norm(Xn[m - 1] - xstar))}


def manifold_branches(sec, xstar, eigvec, tp, P, *, backward=False, eps=1e-4,
                      n=200, efficiency=1.0, phase=0.0):
    """Both half-traces of one invariant manifold through the saddle x*.

    `backward=True` traces with the inverse stage map, which is just the same
    march with `dir` flipped -- down = dew(a x + b) and up = (bubble(x) - b)/a are
    exact inverses, so no separate integrator is needed. Use it for the *stable*
    manifold: forward-marching along a stable direction only walks into the pinch
    and stalls, while backward-marching unrolls the approach that a real profile
    spends its stages on.

    Returns BOTH signed branches, in the fixed order (+v, -v). They are
    deliberately not ranked: the arm a real column uses is the one pointing at the
    neighbouring sections, which nothing here can see, and the correct arm is
    routinely the *short* one -- on ipa/water/EG the saddle sits on the section's
    own x_E floor, so the stripping-side arm dies after 3 stages while the arm
    running off to the entrainer corner survives 10 and used to win.

    `phase` in [0, 1) shifts WHERE ON THE CURVE the stages fall, without changing
    the curve. The manifold is invariant, so launching further out along the same
    eigenvector traces the same arm sampled part of a stage later; growing the
    launch offset by one stage's worth returns the original sampling exactly.
    That matters because the arm crawls at the pinch for tens of stages and then
    escapes in a single step -- 0.40 of the simplex in one stage on ipa/water/EG
    -- so which points the stage grid lands on decides whether `connect` sees the
    junction on a short chord it can resolve or a long one it cannot. `eps` is an
    arbitrary launch offset, so the phase it implies is arbitrary too; offering
    several lets the junction test pick, exactly as it already picks the arm.

    The growth per stage is measured off the trace rather than taken from the
    eigenvalue: `classify_pinch` has it, but the eigenvalue is the linearisation
    at x*, and the trace's own first step is what the sampling actually follows.
    """
    march_sec = sec._replace(dir=-sec.dir) if backward else sec
    vf = _tangent(xstar, eigvec)
    out = []
    for sign in (+1.0, -1.0):
        v = sign * vf

        def _march(offset):
            x0 = np.clip(xstar + offset * v, 1e-12, None)
            return march_section(march_sec, x0 / x0.sum(), tp, P, max_stages=n,
                                 efficiency=efficiency)

        prof = _march(eps)
        if phase:
            d = np.linalg.norm(prof["X"][:2] - xstar, axis=1)
            # ratio > 1 is the outward growth this branch showed on its first
            # stage. A branch that is not moving has no phase to shift.
            if prof["n"] > 1 and d[0] > 0 and d[1] > d[0]:
                shifted = _march(eps * (d[1] / d[0]) ** float(phase))
                if shifted["n"] > 1:
                    prof = shifted
        prof = {**prof, "travel": float(np.linalg.norm(prof["X"][-1] - xstar))}
        out.append(_extend_along_ray(march_sec, xstar, v, prof, tp, P,
                                     max_extra=max(n - prof["n"], 0)))
    return out


def _join(approach, departure, xstar):
    """Splice a reversed stable branch, the saddle, and an unstable branch into
    one profile running in the section's march direction."""
    def part(p, rev):
        if p is None:
            return (np.empty((0, xstar.shape[0])),) * 2 + (np.empty(0),)
        X, Y, T = p["X"], p["Y"], p["T"]
        return (X[::-1], Y[::-1], T[::-1]) if rev else (X, Y, T)

    aX, aY, aT = part(approach, True)      # runs *toward* x*
    dX, dY, dT = part(departure, False)    # runs *away* from x*
    X = np.vstack([aX, xstar[None, :], dX])
    # the saddle's own conjugate vapour: take it from whichever neighbour exists
    yj = aY[-1] if len(aY) else (dY[0] if len(dY) else xstar)
    Tj = aT[-1] if len(aT) else (dT[0] if len(dT) else 0.0)
    Y = np.vstack([aY, yj[None, :], dY])
    T = np.concatenate([aT, [Tj], dT])
    ray = any(p is not None and p["status"] == "manifold+ray"
              for p in (approach, departure))
    return {"X": X, "Y": Y, "T": T, "n": len(X), "pinch_index": len(aX),
            "status": "manifold+ray" if ray else "manifold", "pinched": True}


#: Sampling phases offered per saddle arm (see `manifold_branches`). 4 brings the
#: worst one-stage chord on ipa/water/EG from 0.40 down under `connect.STEP_CAP`,
#: which is what a junction has to be resolved to before it means anything. It
#: costs 4x the marches and 4x the junction tests; the marches are cheap next to
#: the O(N*M) segment scan, so raise it only against a measured need.
#:
#: ponytail: phase-shifting a fixed-step trace is a sampling trick, not a finer
#: integrator -- it moves where the stages land, it does not make the arm itself
#: better resolved between them. True adaptive arc-length continuation of the
#: manifold is the upgrade path if 4 phases ever stops being enough.
ARM_PHASES = 4


def _cl_of(p):
    """A pinch record from `pinch_points(want_eigen=True)` IS the classification.

    `_classify` would redo `jacobian` + `eigenstructure` on a point that already
    carries both. Beyond the wasted solve, recomputing is how this module and
    `bodies.middle_bodies` came to hold different eigenvectors for the same
    saddle: `np.linalg.eig` fixes neither the column order nor the sign, so two
    calls can disagree about which way an arm points. Reading the one record is
    what makes a body's S/E signs mean the same thing here as they did there.
    """
    return {**p, "xstar": p["x"], "saddle": p.get("kind") == "saddle"}


def section_saddles(sec, tp, P, pinches=None):
    """Every saddle of a section that lies inside its own feasible region.

    Rule 1 -- "calculate all saddle pinch points" (paper p.102). Ternary saddles
    (`pinch.BRANCH_TOL` on `k_gap`) are returned alone when there are any; a FACE
    saddle lies on a simplex face and therefore essentially on the neighbouring
    product profile, so its upper junction closes at 1e-4 while the lower one
    misses by 0.14 -- and above r ~ 5, where the paper's criterion says
    infeasible, that near-zero gap reads as a connection. It stays as a fallback
    for a section with no ternary saddle, never as a preference.

    `pinches` reuses an enumeration the caller already paid for -- `bvm.driver`
    needs the same list to build the section's bodies.
    """
    ps = pinch_points(sec, tp, P, down=True) if pinches is None else pinches
    sad = [_cl_of(p) for p in ps
           if p["in_simplex"] and p.get("kind") == "saddle"
           and feasible(sec, p["x"], tol=1e-6)]
    ternary = [s for s in sad if s.get("k_gap", np.inf) <= BRANCH_TOL]
    return ternary or sad


def _pick(branches, sign):
    """(sign, branch) pairs. `manifold_branches` returns (+v, -v) in that fixed
    order, so a body's +/- selects an index; `sign=None` keeps both."""
    pairs = [(+1.0, branches[0] if branches else None),
             (-1.0, branches[1] if len(branches) > 1 else None)]
    return pairs if sign is None else [p for p in pairs if p[0] == float(sign)]


def interior_candidates(sec, tp, P, *, max_stages=200, efficiency=1.0,
                        x_hint=None, eps=1e-4, phases=ARM_PHASES,
                        pinches=None, body=None):
    """Limiting profiles for an interior section, from its controlling saddle.

    A strongly-pinched interior section -- an extractive section, or one feeding a
    near-pure side draw -- is governed by a saddle pinch of its own stage map. The
    profile arrives along the saddle's *stable* manifold (slowly: that approach is
    what makes the section many stages long), lingers at the pinch, and leaves
    along the *unstable* one (fast). The limiting profile is therefore the union
    of the two manifolds through x*, and that union is the curve the neighbouring
    sections must connect to.

    Measured on ethanol/water/EG at R=1.5, E/F=0.6: the extractive saddle sits at
    x = (0.565, 0.109, 0.327) with |lambda| = (45.1, 0.72); its backward-traced
    stable branch meets the rectifying profile *exactly* (dmin ~ 1e-16), while the
    forward unstable branch leaves the feasible region in two stages. Anchoring on
    an arbitrary stage of a neighbouring profile -- what this code used to do --
    starts outside the section's feasible region entirely.

    EVERY saddle of the section is used, not the one root a seed happens to fall
    into. That is rule 1 -- "calculate all saddle pinch points" (paper p.102) --
    and skipping it is what made this return [] on the paper's own PWG example:
    at E/F = 0.750, r = 2.042 the extractive section has four pinches, and
    `pinch_solve` seeded at `region_center` lands on the unstable node at
    (0.421, 0.203, 0.377), so both saddles -- the ternary one at
    (0.050, 0.571, 0.380) and the face one at (0.621, 0.0005, 0.378) -- went
    unseen and the section reported that it could not anchor. Anchored on the
    ternary saddle the same machinery closes both junctions (0.048 / 0.040).

    Ternary saddles are tried first (`pinch.BRANCH_TOL` on `k_gap`). A FACE saddle
    lies on a simplex face and therefore essentially on the neighbouring product
    profile, so its upper junction closes at 1e-4 while the lower one misses by
    0.14 -- and above r ~ 5, where the paper's criterion says infeasible, that
    near-zero gap reads as a connection. It stays as a fallback for a section with
    no ternary saddle, never as a preference.

    All four arm pairs of each saddle are returned, unranked -- which pair is the
    real column's is decided by whether its ends meet the neighbouring sections,
    which only the driver's junction test can see (rule 5). Returns [] when the
    section has no feasible region or no saddle.

    BOTH ways of assigning the two eigendirections to the two ends are offered.
    The physics reads one way -- the profile arrives along the down map's stable
    direction, so that arm is traced backward, and leaves along the unstable one,
    traced forward -- and that is the pairing listed first. But it is a local
    choice about a section with no product anchor, and rule 5 is explicit that
    those are not made locally: "consider both directions of the eigenvectors in
    steps 3 and 4". Measured, it is not academic. `extract_col.colx` (MEOH/DMC +
    acetonitrile) sizes at R = 3, E/F = 1 only on the OTHER assignment, and it did
    so for a long time by accident, because the old `jacobian_G` read the up map
    for a middle section and handed these two helpers each other's labels. Fixing
    the direction without offering both pairings turns that reference case
    infeasible. `driver._both_orientations` already hedges the same kind of choice
    for the profile's overall direction; this is the same hedge one level down. It
    doubles the candidate count, and one sizing of ipa/water/EG goes from 1.4 s to
    2.6 s -- the marches are cheap, the O(N*M) junction scan in `connect` is not.
    """
    saddles = section_saddles(sec, tp, P, pinches)
    s_want = e_want = None
    if body is not None:
        # The body already decided which saddle and which way each arm points
        # (rule 5's four combinations ARE its four bodies), so there is nothing
        # left to enumerate. That is the whole saving: eight candidate curves
        # marched and ranked collapse to one, chosen from geometry that costs no
        # marches at all.
        x_body = np.asarray(body["saddle"], float)
        saddles = [s for s in saddles
                   if np.linalg.norm(np.asarray(s["xstar"], float) - x_body) < 1e-6]
        s_want, e_want = float(body["s_sign"]), float(body["e_sign"])

    cands = []
    for cl in saddles:
        xstar = cl["xstar"]
        v_stable, v_unstable = stable_eigvec(cl), unstable_eigvec(cl)
        for i in range(max(int(phases), 1)):
            ph = i / max(int(phases), 1)

            def _arms(v, backward):
                return manifold_branches(sec, xstar, v, tp, P, backward=backward,
                                         eps=eps, n=max_stages,
                                         efficiency=efficiency, phase=ph)

            # rule 5, both assignments: the physical one (arrive along stable,
            # leave along unstable) and the swap. `swapped` is carried so the
            # driver can rank it behind -- the swap is a fallback for a section
            # the physical reading cannot make a column out of, not a peer. Kept
            # even when a body is given: the body fixes which SIGN each end
            # takes, and `extract_col.colx` sizes only on the swapped assignment,
            # so dropping it here would cost a reference case to save one march.
            for swapped, (v_in, v_out) in enumerate(((v_stable, v_unstable),
                                                     (v_unstable, v_stable))):
                approaches = _pick(_arms(v_in, True), s_want)
                departures = _pick(_arms(v_out, False), e_want)
                for s_sgn, a in approaches:
                    for e_sgn, d in departures:
                        if a is None and d is None:
                            continue
                        prof = _join(a, d, xstar)
                        prof["classification"] = cl
                        prof["phase"] = ph
                        prof["swapped_arms"] = bool(swapped)
                        prof["anchor_method"] = "saddle"
                        prof["body_id"] = body_id(xstar, s_sgn, e_sgn)
                        cands.append(prof)
    return cands


def ray_end_candidates(sec, tp, P, *, max_stages=200, efficiency=1.0,
                       pinches=None, body=None):
    """March the interior section inward from the far end of the saddle's ray.

    The scratchpad's method 2: start at the body's S vertex -- the approximate
    start of the extractive profile, fixed by the saddle's eigendirections -- and
    march forward from there.

    `_extend_along_ray` says why marching OUT from the pinch does not reach the
    edge: the map converges onto whatever node lies in the way, so that extension
    is a straight segment standing in for an arm it could not trace. Starting at
    the far end instead gives a real march for the same stretch -- one profile
    that comes in along the stable side, passes the saddle and leaves down the
    unstable one, with a stage count of its own rather than a ray resampled at
    the largest step the arm managed.

    It does not converge ONTO the saddle and is not meant to: a finite-reflux
    profile passes near a pinch and leaves. Measured here it closes to 0.013 of
    a saddle it started 0.066 from, at stage 3 of 24.

    Both signs are returned when no body is given; a body picks one. Note the
    departure arm is not traced separately -- the profile IS the walk from S, and
    where it ends is where the section ends.

    ponytail: S is `_ray_end`, the ray clipped to the section's own balance, not
    `bodies._to_edge`'s composition-space edge. Narrower on purpose, for the
    reason `_ray_end` gives: the wide walk makes every reflux from 1.5 to 8
    report feasible. That also means this method cannot reach a junction the wide
    body would have carried, which is the r_max gap of docs/adr/0004 and is why
    it is offered rather than made the default.
    """
    saddles = section_saddles(sec, tp, P, pinches)
    s_want = None
    if body is not None:
        x_body = np.asarray(body["saddle"], float)
        saddles = [s for s in saddles
                   if np.linalg.norm(np.asarray(s["xstar"], float) - x_body) < 1e-6]
        s_want = float(body["s_sign"])

    # `-sec.dir`, the direction `manifold_branches` calls backward -- which for a
    # section with no product anchor is the one that traces the stable manifold,
    # and therefore the one that runs S -> saddle. Taking `sec.dir` instead is
    # the trap `pinch.jacobian` documents: sign(Delta) is unrelated to the way a
    # middle section's profile runs, and marching that way from S walks along the
    # x_LK = 0 face to the entrainer corner without ever seeing the saddle.
    #
    # Measured on ipa/water/EG at R = 1.5, E/F = 2.0: from S = (0, 0.290, 0.710),
    # 0.066 from the saddle, this marches 24 stages, closes to 0.013 at stage 3
    # and then leaves down the unstable side. That is the extractive profile.
    march_sec = sec._replace(dir=-sec.dir)
    out = []
    for cl in saddles:
        xstar = np.asarray(cl["xstar"], float)
        vf = _tangent(xstar, stable_eigvec(cl))
        for sgn in ((+1.0, -1.0) if s_want is None else (s_want,)):
            xs = _ray_end(sec, xstar, sgn * vf)
            if xs is None:
                continue
            # S sits ON the section's own balance boundary by construction, so
            # `feasible` at any positive tolerance would reject its own ray end.
            xs = np.clip(xs, 1e-12, None)
            xs = xs / xs.sum()
            prof = march_section(march_sec, xs, tp, P, max_stages=max_stages,
                                 efficiency=efficiency)
            if prof["n"] < 2:
                continue
            out.append({**prof, "classification": cl, "phase": 0.0,
                        "swapped_arms": False, "anchor_method": "ray",
                        "pinched": True,
                        "body_id": body_id(xstar, sgn, sgn)})
    return out


def launch_from_saddle(sec, xstar, eigvec, tp, P, eps=1e-3, n=200):
    """Trace one manifold and return its longest branch (kept for callers that
    want a single half-trace; `interior_candidates` is the section-level API)."""
    br = [b for b in manifold_branches(sec, xstar, eigvec, tp, P, eps=eps, n=n)
          if b["X"].shape[0] > 1]
    if br:
        return max(br, key=lambda p: p["travel"])
    return march_section(sec, xstar, tp, P, max_stages=n)


def _demo():
    from .thermo_adapter import ColumnForgeThermo
    from .problem import build_problem, overall_balance
    from .sections import extractive_chain

    abc = np.array([(7.11714, 1210.595, 229.664),   # acetone
                    (7.20211, 1582.271, 239.726),   # methanol
                    (8.07131, 1730.63, 233.426)])   # water (entrainer)
    tp = ColumnForgeThermo(abc)
    z = np.array([0.5, 0.5, 0.0])
    prob = build_problem(["acetone", "methanol", "water"], [(z, 100.0, 1.0)], 760.0,
                         lk=0, hk=1, x_E=np.array([0.0, 0.0, 1.0]), extractive=True)
    xD, xB, D, B = overall_balance(prob)
    rect, ext, strip = extractive_chain(prob, 3.0, 0.8, xD, xB, D, B)

    # the extractive pinch is found INSIDE the section's feasible region -- the
    # old 0.5*(xD+xB) seed is not even in it, which is the bug this fixes.
    guess = 0.5 * (xD + xB)
    assert not feasible(ext, guess), "the old seed should be outside the region"
    cl = saddle_pinch(ext, guess, tp, 760.0)
    assert cl["kind"] in ("saddle", "stable_node", "unstable_node")
    assert np.isfinite(cl["eigvals"]).all()
    assert cl["in_region"] and feasible(ext, cl["xstar"], tol=1e-6), cl["xstar"]
    # E8: the |lambda|>1 direction, not an arbitrary eig column
    assert abs(cl["eigvals"][int(np.argmax(np.abs(cl["eigvals"])))]) >= \
        abs(cl["eigvals"][int(np.argmin(np.abs(cl["eigvals"])))])

    # the eigenvector ray stops ON the boundary of the section's own region --
    # either a mole fraction or an operating-line vapour component hits zero --
    # and never outside it. That endpoint is what a stalled branch is extended to.
    v = _tangent(cl["xstar"], unstable_eigvec(cl))
    for s in (+1.0, -1.0):
        end = _ray_end(ext, cl["xstar"], s * v)
        assert end is not None and end.min() >= -1e-9
        assert feasible(ext, end, tol=1e-9), (end, feasible_margin(ext, end))
        touching = min(end.min(), feasible_margin(ext, end))
        assert touching < 1e-6, f"ray must reach an edge, slack {touching:.3g}"

    # Rule 1: EVERY saddle, not the one root a seed falls into. `pinch_solve` and
    # `pinch_points` are two different questions, and the gap between them is what
    # made `interior_candidates` return [] on the paper's own PWG example while
    # two saddles sat there unused. The seeded root need not even be a saddle.
    all_p = pinch_points(ext, tp, 760.0, down=True)
    assert len(all_p) > 1, all_p
    saddles = [p for p in all_p
               if p["kind"] == "saddle" and p["in_simplex"]
               and feasible(ext, p["x"], tol=1e-6)]
    if saddles:
        cands = interior_candidates(ext, tp, 760.0, max_stages=80)
        assert cands, "a saddle in the region must yield a limiting profile"
        # every candidate is launched from one of THOSE saddles, and its
        # `pinch_index` is where that saddle sits on the joined curve
        for c in cands:
            xs = c["classification"]["xstar"]
            assert any(np.linalg.norm(xs - p["x"]) < 1e-6 for p in saddles), xs
            assert np.allclose(c["X"][c["pinch_index"]], xs, atol=1e-8)
        # ternary saddles are preferred; a face saddle only stands in when there
        # is no ternary one (its arms lie on the neighbouring product profile)
        tern = [p for p in saddles if p["k_gap"] <= BRANCH_TOL]
        if tern:
            used = {tuple(np.round(c["classification"]["xstar"], 9)) for c in cands}
            assert used <= {tuple(np.round(p["x"], 9)) for p in tern}, used

    if cl["saddle"]:
        # both arms come back, unranked, in (+v, -v) order: the short one is
        # routinely the arm the column actually uses.
        br = manifold_branches(ext, cl["xstar"], unstable_eigvec(cl), tp, 760.0,
                               n=40)
        assert len(br) == 2, len(br)
        cands = interior_candidates(ext, tp, 760.0, max_stages=80)
        assert cands, "a saddle must yield at least one limiting profile"
        best = cands[0]
        assert best["n"] >= 2
        assert best["X"].shape == best["Y"].shape
        assert best["T"].shape[0] == best["n"]
        # the approach is traced BACKWARD, so it must arrive at the saddle rather
        # than sit on it: a forward stable-direction march would not move at all.
        if best["pinch_index"] > 0:
            assert np.linalg.norm(best["X"][0] - cl["xstar"]) > 1e-4

        # a branch that ran out of stage budget is EXTENDED to the edge; one that
        # already reached a boundary is left alone. Getting this wrong is what
        # started an extractive section 0.289 from its own hand-over point.
        xs, vu = cl["xstar"], unstable_eigvec(cl)
        short = manifold_branches(ext, xs, vu, tp, 760.0, n=2)
        for br, ray in zip(short, (_ray_end(ext, xs, s * _tangent(xs, vu))
                                   for s in (+1.0, -1.0))):
            assert br["status"] != "max", "n=2 must terminate, not extend blindly"
            if br["status"] == "manifold+ray" and ray is not None:
                assert np.linalg.norm(br["X"][-1] - ray) < 1e-6, (br["X"][-1], ray)
        stalled = {"status": "max", "X": np.array([xs, xs + 1e-3 * _tangent(xs, vu)]),
                   "Y": np.zeros((2, xs.size)), "T": np.zeros(2),
                   "P": np.full(2, 760.0), "n": 2}
        grown = _extend_along_ray(ext, xs, _tangent(xs, vu), stalled, tp, 760.0,
                                  max_extra=0)
        assert grown["n"] > stalled["n"], "a `max` branch must still be extended"
        assert grown["status"] == "manifold+ray", grown["status"]
        for name in ("simplex", "operating_line", "crossed"):
            at_edge = {**stalled, "status": name}
            assert _extend_along_ray(ext, xs, _tangent(xs, vu), at_edge, tp,
                                     760.0, max_extra=RAY_STAGES) is at_edge, name

    # continuation anchor just reads a composition off an upstream profile
    up = march_section(rect, xD, tp, 760.0, 30)
    a = continuation_anchor(up, 5)
    assert np.allclose(a, up["X"][5])
    print(f"anchor self-check OK  extractive pinch={cl['kind']} "
          f"at {np.round(cl['xstar'], 3)}")


if __name__ == "__main__":
    _demo()
