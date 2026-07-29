"""Pinch points, minimum reflux, minimum entrainer (blueprint Sec 8).

A pinch is a fixed point of the stage map, x* = G(x*): the composition where the
operating line meets the equilibrium surface and marching stalls. The marcher
already lands on it (status='pinch'), so `pinch_point` just reads that endpoint.

Feasibility hinges on the *eigenstructure* of J = dG/dx at the pinch, not the
operating-line coefficients -- this is the sense in which the method is "matrix"
based (Sec 8). A saddle (mixed |lambda|) has the 1-D invariant manifolds that
anchor strongly-pinched interior sections (Sec 6.3).

R_min (and minimum E/F) is where the rectifying and stripping reachable regions
only just touch. Operationally that is the smallest R at which the profiles
still connect, found by bisection on the connection test (Sec 7) -- robust, and
equivalent to the pinch-tangency condition. ponytail: bisection here; a direct
pinch-tangency solve would be faster but no more correct for sizing.
"""

import numpy as np
from scipy.optimize import root

from .march import march_section
from .sections import feasible, region_center


def pinch_point(sec, x0, tp, P, max_stages=400):
    """March the section to its pinch; return x* (or the last comp if uncapped)."""
    prof = march_section(sec, x0, tp, P, max_stages=max_stages)
    return prof["X"][-1], prof["pinched"]


def pinch_residual(sec, x, tp, P):
    """Pinch equation residual: K(x) x - (a x + bvec).

    A pinch is where the passing streams are simultaneously in equilibrium and on
    the operating line -- y* = K(x*) x* and y* = a x* + bvec -- so the two
    statements coincide. Written this way the condition is *direction-free*: it
    does not go through dew() or bubble() inversions, so it neither inherits the
    marcher's root-jumping nor cares whether the section is marched up or down.
    (Summing the components gives 0 = 0 identically, so only C-1 are independent.)
    """
    x = np.clip(np.asarray(x, float), 0.0, None)
    x = x / x.sum()
    y_eq, _ = tp.bubble(x, P)
    return y_eq - (sec.a * x + sec.bvec)


def pinch_solve(sec, tp, P, x_guess=None):
    """Solve K(x) x = a x + bvec inside the section's feasible region.

    Returns dict(xstar, residual, converged, in_region) or None when the section
    has no feasible region at all. Seeded at `region_center` unless a guess that
    is itself in the region is supplied.
    """
    seed = None
    if x_guess is not None and feasible(sec, x_guess):
        seed = np.asarray(x_guess, float)
    if seed is None:
        seed = region_center(sec)
    if seed is None:
        return None

    C = sec.delta.shape[0]

    def lift(u):
        x = np.empty(C); x[:C - 1] = u; x[C - 1] = 1.0 - u.sum()
        return np.clip(x, 1e-12, None)

    def resid(u):
        return pinch_residual(sec, lift(u), tp, P)[:C - 1]

    try:
        sol = root(resid, seed[:C - 1], method="hybr")
        xstar = lift(sol.x)
        xstar = xstar / xstar.sum()
        ok = bool(sol.success)
    except (ValueError, FloatingPointError):
        xstar, ok = seed, False
    r = float(np.linalg.norm(pinch_residual(sec, xstar, tp, P)))
    return {"xstar": xstar, "residual": r, "converged": ok and r < 1e-6,
            "in_region": feasible(sec, xstar, tol=1e-6)}


def jacobian_G(sec, xstar, tp, P, h=1e-6):
    """Finite-difference Jacobian of the stage map G in reduced (C-1) coords.

    Composition lives on the simplex (sum=1), so we differentiate the first C-1
    components with x_C = 1 - sum. Returns the (C-1)x(C-1) matrix and its eigen-
    decomposition-ready form. Central differences (the thermo closure is real).
    """
    C = xstar.shape[0]

    def G(x):
        prof = march_section(sec, x, tp, P, max_stages=1)
        return prof["X"][1] if prof["X"].shape[0] > 1 else prof["X"][0]

    u0 = xstar[:C - 1].copy()

    def lift(u):
        x = np.empty(C); x[:C - 1] = u; x[C - 1] = 1.0 - u.sum()
        return np.clip(x, 0, None)

    J = np.empty((C - 1, C - 1))
    for k in range(C - 1):
        up = u0.copy(); up[k] += h
        um = u0.copy(); um[k] -= h
        gp = G(lift(up))[:C - 1]
        gm = G(lift(um))[:C - 1]
        J[:, k] = (gp - gm) / (2 * h)
    return J


def classify_pinch(J):
    """Classify a pinch from J's eigenvalues (Sec 6.3 table).

    Returns dict(kind, eigvals, eigvecs, saddle). kind in
    {stable_node, unstable_node, saddle}. Only a saddle (mixed |lambda|) carries
    the 1-D manifolds usable to anchor an interior section.
    """
    w, V = np.linalg.eig(J)
    mag = np.abs(w)
    inside = mag < 1.0
    if np.all(inside):
        kind = "stable_node"
    elif np.all(~inside):
        kind = "unstable_node"
    else:
        kind = "saddle"
    return {"kind": kind, "eigvals": w, "eigvecs": V, "saddle": kind == "saddle"}


def bisect_min(feasible_fn, lo, hi, tol=1e-3, max_iter=40, prescan=12):
    """Smallest x in [lo,hi] with feasible_fn(x) True, by bisection.

    Feasibility in R (and E/F) is an upper set in the ideal case, but E2's local
    connection tolerance can open a spurious feasible *island* at low x. So we
    coarse pre-scan first (E14) and bracket the lower boundary of the LAST
    feasible run, then bisect inside that bracket. A lone low-x island is skipped,
    not mistaken for R_min.

    The run need not reach `hi`: feasibility is a band whenever the junction can
    run off the end of a profile at high reflux, which is ordinary once the
    junction test asks for a real crossing (c2-c4 closes at R ~ 10, where its
    stripping section has shrunk below one stage). Requiring `hi` itself to be
    feasible reported no minimum at all for those columns. Returns None only when
    nothing in [lo, hi] is feasible.
    """
    if feasible_fn(lo):
        return lo
    if prescan and prescan >= 2:
        grid = [lo + (hi - lo) * k / (prescan - 1) for k in range(prescan)]
        feas = [feasible_fn(x) for x in grid]
        if not any(feas):
            return None
        start = max(i for i, f in enumerate(feas) if f)   # last feasible sample
        while start > 0 and feas[start - 1]:              # back to its run's start
            start -= 1
        lo, hi = grid[start - 1], grid[start]     # boundary lies in this cell
    elif not feasible_fn(hi):
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if feasible_fn(mid):
            hi = mid
        else:
            lo = mid
    return hi


def feasible_band(feasible_fn, lo, hi, n_scan=24, tol=1e-3, log=True):
    """Both edges of the feasible interval of a boolean predicate.

    Returns (first, last): the smallest and largest arguments that pass, or
    (None, None) if nothing does. `last` is None when the top of the range still
    passes -- an open band, which is what an ordinary column's reflux does.

    `bisect_min` finds the lower edge only, on the assumption that feasibility is
    an upper set. That assumption fails for an extractive column: too much reflux
    washes the entrainer out of the middle section and the separation stops
    working, so the feasible set is a bounded interval with no monotone predicate
    to bisect on. Scanning first finds the bracket; each edge is then refined the
    same way `bisect_min` refines its one.
    """
    grid = (np.geomspace(max(lo, 1e-6), hi, int(n_scan)) if log
            else np.linspace(lo, hi, int(n_scan)))
    ok = [bool(feasible_fn(float(x))) for x in grid]
    if not any(ok):
        return None, None
    i0 = int(np.argmax(ok))
    i1 = len(ok) - 1 - int(np.argmax(ok[::-1]))

    def refine(bad, good):
        for _ in range(40):
            if abs(good - bad) <= tol:
                break
            mid = 0.5 * (bad + good)
            if feasible_fn(mid):
                good = mid
            else:
                bad = mid
        return good

    first = float(grid[0]) if i0 == 0 else refine(float(grid[i0 - 1]),
                                                  float(grid[i0]))
    last = None if i1 == len(grid) - 1 else refine(float(grid[i1 + 1]),
                                                   float(grid[i1]))
    return first, last


def _demo():
    from .thermo_adapter import ColumnForgeThermo
    from .problem import build_problem, overall_balance
    from .sections import single_feed_chain
    from .connect import connect

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = ColumnForgeThermo(abc)
    z = np.array([0.4, 0.35, 0.25])
    prob = build_problem(["b", "t", "x"], [(z, 100.0, 1.0)], 760.0,
                         rec_lk=0.98, rec_hk=0.02)
    xD, xB, D, B = overall_balance(prob)

    # rectifying pinch exists and is classified; a saddle carries manifolds
    rect, strip = single_feed_chain(prob, 3.0, xD, xB, D, B)
    xstar, pinched = pinch_point(rect, xD, tp, 760.0)
    J = jacobian_G(rect, xstar, tp, 760.0)
    cl = classify_pinch(J)
    assert cl["kind"] in ("stable_node", "unstable_node", "saddle")
    assert cl["eigvals"].shape == (2,)

    # R_min by bisection: below it the split can't connect, above it can
    def feasible(R):
        rc, st = single_feed_chain(prob, R, xD, xB, D, B)
        r = march_section(rc, xD, tp, 760.0, prob.max_stages)
        s = march_section(st, xB, tp, 760.0, prob.max_stages)
        return connect(r, s, rc, tp, 760.0)["connected"]

    Rmin = bisect_min(feasible, 0.2, 20.0, tol=1e-2)
    assert Rmin is not None and 0.2 < Rmin < 20.0, Rmin
    assert not feasible(Rmin * 0.7), "below R_min should be infeasible"
    assert feasible(Rmin * 1.5), "above R_min should be feasible"

    # E14: a spurious low-x feasible island must not be reported as the minimum --
    # the pre-scan brackets the boundary of the final feasible run instead.
    def islanded(x):
        return (0.10 <= x <= 0.15) or x >= 5.0     # island at ~0.1, real onset at 5
    got = bisect_min(islanded, 0.0, 10.0, tol=1e-3, prescan=21)
    assert 4.5 < got < 5.5, f"island skipped -> boundary near 5, got {got}"

    # feasible_band finds BOTH edges of a bounded interval, which bisect_min
    # cannot: an extractive column's reflux band closes at the top.
    first, last = feasible_band(lambda x: 2.0 <= x <= 6.0, 0.5, 20.0, n_scan=40)
    assert first is not None and abs(first - 2.0) < 0.05, first
    assert last is not None and abs(last - 6.0) < 0.05, last
    # an open band reports no upper edge rather than the scan ceiling
    first, last = feasible_band(lambda x: x >= 3.0, 0.5, 20.0, n_scan=40)
    assert abs(first - 3.0) < 0.05 and last is None, (first, last)
    assert feasible_band(lambda x: False, 0.5, 20.0) == (None, None)
    print(f"pinch self-check OK  pinch={cl['kind']}  R_min~{Rmin:.2f}  island->{got:.2f}")


if __name__ == "__main__":
    _demo()
