"""Section anchoring -- where each profile starts (blueprint Sec 6).

For S=2 the problem is two-point: both product ends are known. For S>2 interior
sections have no product anchor. The escalation ladder:

  6.1 product-terminated  -> start at x_D / x_B and march inward (trivial here).
  6.2 ordinary interior   -> *continuation*: the liquid composition is continuous
      across a feed (only the difference point jumps), so anchor the interior
      section at the upstream profile's composition where the sections switch,
      and keep marching in the stable direction.
  6.3 strongly pinched    -> *saddle launch*: extractive sections and pure side
      draws crawl for many stages past a bottleneck, so anchor at the controlling
      saddle pinch and build the profile from its 1-D invariant manifolds.

Only a saddle (mixed |lambda|, Sec 8) has the one-dimensional manifolds that ARE
the limiting interior profile. The unstable manifold is traced by marching the
forward map off x* + eps*v_u; the stable side is reached by continuation from the
adjacent product-anchored profile.
"""

import numpy as np
from scipy.optimize import fsolve

from .march import march_section
from .pinch import jacobian_G, classify_pinch, pinch_solve
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
    cl = classify_pinch(jacobian_G(sec, xstar, tp, P))
    cl["xstar"] = xstar
    cl["in_region"] = ps["in_region"]
    return cl


def unstable_eigvec(cl):
    """The eigenvector of the largest |lambda| (E8): np.linalg.eig column order is
    arbitrary, so `eigvecs[:, 0]` can be the *stable* direction and trace the wrong
    manifold. The unstable manifold -- the one the interior profile follows away
    from the saddle -- is spanned by the eigenvector whose |lambda| > 1 is largest."""
    w, V = np.asarray(cl["eigvals"]), np.asarray(cl["eigvecs"])
    return V[:, int(np.argmax(np.abs(w)))]


def stable_eigvec(cl):
    """The eigenvector of the smallest |lambda| -- the direction the interior
    profile *arrives* along, and the reason an extractive section is long."""
    w, V = np.asarray(cl["eigvals"]), np.asarray(cl["eigvecs"])
    return V[:, int(np.argmin(np.abs(w)))]


def _tangent(xstar, eigvec):
    """Lift a reduced eigenvector to a sum-zero (simplex-tangent) full vector."""
    C = xstar.shape[0]
    v = np.asarray(eigvec, float).real
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


def _extend_along_ray(sec, xstar, v, prof, tp, P, max_extra):
    """Continue a branch that stalled on an interior node straight out to the edge.

    Bruggemann & Marquardt rules 3-4: the middle-section profile "matches the
    eigenvectors of the saddle nearly exactly and shows only small curvature", so
    the body is built by extending the eigenvector LINEARLY to the edge of
    composition space. Marching the stage map instead converges onto whatever
    node lies in the way -- on ipa/water/EG the +v_unstable arm parks at
    (0.406, 0.217, 0.378) and never reaches the rectifying profile at
    (0.607, 0.006, 0.387), so the upper junction cannot close at all.

    Only a branch that ended on a `pinch` is extended; one that stopped on the
    simplex, on the operating line, or on the next section's region already
    reached a boundary and has nothing to continue to.
    """
    if prof["status"] != "pinch" or max_extra < 1:
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
                      n=200, efficiency=1.0):
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
    """
    march_sec = sec._replace(dir=-sec.dir) if backward else sec
    vf = _tangent(xstar, eigvec)
    out = []
    for sign in (+1.0, -1.0):
        v = sign * vf
        x0 = np.clip(xstar + eps * v, 1e-12, None)
        x0 = x0 / x0.sum()
        prof = march_section(march_sec, x0, tp, P, max_stages=n,
                             efficiency=efficiency)
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


def interior_candidates(sec, tp, P, *, max_stages=200, efficiency=1.0,
                        x_hint=None, eps=1e-4):
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

    All four arm pairs are returned, unranked -- which pair is the real column's
    is decided by whether its ends meet the neighbouring sections, which only the
    driver's junction test can see (Bruggemann & Marquardt rule 5). Returns [] when
    the section has no feasible region or no saddle.
    """
    ps = pinch_solve(sec, tp, P, x_hint)
    if ps is None or not ps["in_region"]:
        return []
    xstar = ps["xstar"]
    cl = classify_pinch(jacobian_G(sec, xstar, tp, P))
    cl["xstar"] = xstar
    if not cl["saddle"]:
        return []

    approaches = manifold_branches(sec, xstar, stable_eigvec(cl), tp, P,
                                   backward=True, eps=eps, n=max_stages,
                                   efficiency=efficiency)
    departures = manifold_branches(sec, xstar, unstable_eigvec(cl), tp, P,
                                   backward=False, eps=eps, n=max_stages,
                                   efficiency=efficiency)
    cands = []
    for a in (approaches or [None]):
        for d in (departures or [None]):
            if a is None and d is None:
                continue
            prof = _join(a, d, xstar)
            prof["classification"] = cl
            cands.append(prof)
    return cands


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

    if cl["saddle"]:
        # both arms come back, unranked, in (+v, -v) order: the short one is
        # routinely the arm the column actually uses.
        br = manifold_branches(ext, cl["xstar"], unstable_eigvec(cl), tp, 760.0,
                               n=40)
        assert len(br) == 2, len(br)
        cands = interior_candidates(ext, tp, 760.0, max_stages=80)
        assert cands, "a saddle must yield at least one limiting profile"
        best = cands[0]
        assert best["n"] >= 2 and np.allclose(best["X"][best["pinch_index"]],
                                              cl["xstar"], atol=1e-8)
        assert best["X"].shape == best["Y"].shape
        assert best["T"].shape[0] == best["n"]
        # the approach is traced BACKWARD, so it must arrive at the saddle rather
        # than sit on it: a forward stable-direction march would not move at all.
        if best["pinch_index"] > 0:
            assert np.linalg.norm(best["X"][0] - cl["xstar"]) > 1e-4

    # continuation anchor just reads a composition off an upstream profile
    up = march_section(rect, xD, tp, 760.0, 30)
    a = continuation_anchor(up, 5)
    assert np.allclose(a, up["X"][5])
    print(f"anchor self-check OK  extractive pinch={cl['kind']} "
          f"at {np.round(cl['xstar'], 3)}")


if __name__ == "__main__":
    _demo()
