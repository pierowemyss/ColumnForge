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

from march import march_section


def pinch_point(sec, x0, tp, P, max_stages=400):
    """March the section to its pinch; return x* (or the last comp if uncapped)."""
    prof = march_section(sec, x0, tp, P, max_stages=max_stages)
    return prof["X"][-1], prof["pinched"]


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
    coarse pre-scan first (E14) and bracket the boundary of the final feasible
    run -- the smallest grid point whose whole tail stays feasible -- then bisect
    inside that bracket. A lone low-x island is skipped, not mistaken for R_min.
    Returns None if even `hi` is infeasible.
    """
    if not feasible_fn(hi):
        return None
    if feasible_fn(lo):
        return lo
    if prescan and prescan >= 2:
        grid = [lo + (hi - lo) * k / (prescan - 1) for k in range(prescan)]
        feas = [feasible_fn(x) for x in grid]
        # first index from which every sample is feasible (start of the final run)
        start = prescan - 1
        while start > 0 and feas[start - 1]:
            start -= 1
        lo, hi = grid[start - 1], grid[start]     # boundary lies in this cell
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if feasible_fn(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _demo():
    from thermo_adapter import FreeColumnThermo
    from problem import build_problem, overall_balance
    from sections import single_feed_chain
    from connect import connect

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = FreeColumnThermo(abc)
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
        return connect(r, s)["connected"]

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
    print(f"pinch self-check OK  pinch={cl['kind']}  R_min~{Rmin:.2f}  island->{got:.2f}")


if __name__ == "__main__":
    _demo()
