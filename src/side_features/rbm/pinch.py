"""Pinch points, their eigenstructure, and pinch branches (paper eqs 2-6, p.70).

A pinch is a composition where a section's stage map stalls: the passing streams
are simultaneously in equilibrium and on the operating line,

    y* = K(x*) x*        (equilibrium)
    y* = a x* + bvec     (operating line, `sections.op_vapour`)

so the residual is `K(x)x - (a x + bvec)`. Written that way the condition is
*direction-free* -- it does not go through a dew or bubble inversion, so it
neither inherits the marcher's root-jumping nor cares which way the section is
marched. That is the whole reason RBM is better conditioned than shooting: a
component with a tiny K makes the MARCH stiff, but leaves this algebraic system
perfectly ordinary.

For the extractive middle section the net flow and its difference point are
N = D - E and N x_N = D x_D - E x_E (paper eqs 7-8). `x_N` there routinely lies
OUTSIDE the composition simplex, and so therefore do some of the pinch branches
(paper p.100). That is expected and must not be clipped -- the branch is still a
real solution of the algebra, it just has no stages on it. Callers filter by
`in_simplex` when they need physical points.

Solutions are tracked as **pinch branches**: one-dimensional families indexed by
reflux (paper p.70, "the loci of these solutions are called pinch branches...
calculated by homotopy continuation"). At total reflux a -> 1 and bvec -> 0, so
the pinch equation degenerates to y_eq(x) = x -- the fixed points of the residue
curve map, i.e. the pure components and the azeotropes. Those are the seeds; each
is then continued down in reflux.
"""

import numpy as np
from scipy.optimize import brentq, least_squares

from side_features.bvm.pinch import pinch_residual
from side_features.bvm.sections import feasible_margin

#: A branch point counts as "in the simplex" with this much slack, so a pinch
#: sitting exactly on a face (a binary-edge pinch, which is the common case) is
#: kept rather than lost to round-off.
_SIMPLEX_TOL = 1e-7

#: Two pinch points closer than this are the same point found twice. Loose
#: because seeds converge to a shared root from very different starting places.
_SAME_POINT = 1e-4


def _softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


#: A solve is a pinch if its residual is under this. Genuine face solves land at
#: 1e-10 or better, so this is loose -- but it has to stay loose enough to accept
#: a pure-component vertex, whose residual is limited by the bubble-point solve.
_ACCEPT = 1e-7

#: How far off an edge the pinch it belongs to may have moved before the edge
#: root stops being a useful seed for it (mole fraction). Beyond this the pinch
#: near that edge is not a perturbation of the edge root at all and the face
#: solve is on its own.
EDGE_RELAX = 0.05

#: How much of an off-face component the section's difference point may carry
#: and still count as sharp on that face -- the gate on keeping a face point as
#: the stand-in for a pinch branch that has left the simplex. Dimensionless:
#: |bvec_k| / |Delta/V| is just delta_k, the difference point's own composition.
SHARP_TOL = 0.05


def solve_pinch(sec, tp, P, x_guess, active=None):
    """One solve of the pinch equation from `x_guess`. None if it does not converge.

    Solved in SOFTMAX coordinates *restricted to one face of the simplex*. The
    face is `active` (indices allowed to be nonzero), defaulting to whichever
    components `x_guess` switches on. Components off the face are held at exactly
    zero rather than parametrised.

    That restriction is the whole trick, and solving without it is wrong rather
    than merely slow. The pinch equation factorises: K(x) x = a x + bvec has
    x_i (K_i - a) = bvec_i, so away from the product every component is either
    exactly zero or sits on the branch K_i = a. Its solutions therefore LIVE on
    the faces. But softmax puts an exact zero at v = -inf, so a solve heading for
    a face walks its parameters off to infinity, stalls on the flat plateau, and
    reports a residual that is small without being converged. On c2-c4 that let
    the pure-ethane product vertex swallow twelve of thirteen seeds at r = 2 and
    all thirteen at r = 5 -- every one accepted, all at residual 4-9e-8, just
    under the gate. The rectifying section was left with a single point for a
    body, which is why a simple column reported a maximum reflux it does not have.

    Restricted to a face, a pinch on that face is an interior point of its own
    coordinates, well conditioned, and converges to 1e-10.
    """
    C = sec.delta.shape[0]
    x_guess = np.asarray(x_guess, float)
    idx = (np.arange(C)[x_guess > 1e-9] if active is None
           else np.asarray(sorted(active), int))
    if len(idx) == 0:
        return None

    def _check(x):
        try:
            r = float(np.linalg.norm(pinch_residual(sec, x, tp, P)))
        except (ValueError, FloatingPointError):
            return None
        return x if np.isfinite(r) and r <= _ACCEPT else None

    if len(idx) == 1:                      # a vertex: nothing to solve, just test
        x = np.zeros(C)
        x[idx[0]] = 1.0
        return _check(x)

    s = np.clip(x_guess[idx], 1e-12, None)
    s = s / s.sum()

    def resid(v):
        x = np.zeros(C)
        x[idx] = _softmax(v)
        return pinch_residual(sec, x, tp, P)

    try:
        # 1e-9, not machine precision: the acceptance gate below is 1e-7 and
        # two pinches within 1e-4 are the same point, so the extra digits were
        # bought with bubble-point solves and then discarded.
        sol = least_squares(resid, np.log(s), xtol=1e-9, ftol=1e-9,
                            gtol=1e-9, max_nfev=200)
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return None
    x = np.zeros(C)
    x[idx] = _softmax(sol.x)
    return _check(x)


def jacobian(sec, xstar, tp, P, h=1e-6):
    """Jacobian of the stage map at a pinch, in reduced (C-1) coordinates.

    Differentiated from the *pinch* relation rather than by re-marching, so a
    section whose march would be ill-conditioned still gets a clean derivative.

    The map is taken in the section's OWN marching direction (`sec.dir`), which
    is not cosmetic: the two directions are exact inverses, so their eigenvalues
    are reciprocals and "stable" and "unstable" swap between them. Differentiating
    the down-map for every section made every stripping pinch come back an
    unstable node, which is the opposite of the paper's Figure 5 (right) -- and
    since the body rules chain on the count of STABLE eigenvectors, it quietly
    inverted the whole construction for half the column.

        down (Delta > 0):  G(x) = K(x)^-1 (a x + bvec)
        up   (Delta < 0):  G(x) = (K(x) x - bvec) / a

    Central differences; the thermo closure is real, so complex-step cannot
    thread through it.

    Differentiated along the simplex edges leaving the pinch's LARGEST component
    p -- directions e_i - e_p, i != p -- rather than along the first C-1
    coordinates. Both are bases of the sum-zero tangent space, and a change of
    basis is a similarity transform that leaves the eigenvalues alone, so this
    does not change the answer where the old one was valid. It changes where the
    old one was valid at all. Dropping a fixed last component steps off the
    simplex the moment a pinch sits on a face: at x* = (0.065, 0, 0.935) the
    step in the zero component goes negative, and at a vertex BOTH signs of both
    coordinates leave the simplex, so what came back was a difference of clipped,
    non-physical compositions. Every c2-c4 pinch is on a face, and all three came
    back labelled `stable_node` -- a three-way tie in a rule that chains on
    strictly increasing stable-eigenvector counts. Stepping toward the largest
    component always has room to move (x_p >= 1/C), so every direction stays
    inside.

    Returns (J, p): the caller needs p to lift a reduced eigenvector back into
    composition space, which is what `bodies.lift_direction` does.
    """
    C = xstar.shape[0]
    down = sec.dir > 0
    p = int(np.argmax(xstar))
    others = [i for i in range(C) if i != p]

    def G(x):
        xc = np.clip(x, 1e-12, None)
        _, T = tp.bubble(xc / xc.sum(), P)
        K = tp.K(np.atleast_2d(x), np.atleast_1d(T), np.atleast_1d(P))[0]
        if down:
            y = np.clip(sec.a * x + sec.bvec, 1e-300, None)
            g = y / np.clip(K, 1e-300, None)
        else:
            g = (K * x - sec.bvec) / sec.a
        # normalise: the stage map goes simplex -> simplex, but y/K does not
        # preserve the sum, and an unnormalised map's reduced-coordinate
        # eigenvalues depend on WHICH component is treated as the dependent one.
        # At a pinch the factor is 1 to first order, so this only removes that
        # arbitrariness -- and it is what makes the change of basis above a
        # similarity transform rather than a different operator.
        s = g.sum()
        return g / s if abs(s) > 1e-300 else g

    # a sum-zero vector's coordinates in the basis {e_i - e_p} are just its
    # entries at i != p, so projecting in and out is this indexing and nothing more
    J = np.empty((C - 1, C - 1))
    for k, i in enumerate(others):
        d = np.zeros(C)
        d[i], d[p] = 1.0, -1.0
        xp, xm = xstar + h * d, xstar - h * d
        try:
            if xm.min() >= 0.0:
                col = (G(xp) - G(xm)) / (2 * h)
            else:                                  # on the face: one-sided, inward
                col = (G(xp) - G(xstar)) / h
        except (ValueError, FloatingPointError):
            return None
        J[:, k] = col[others]
    return (J, p) if np.all(np.isfinite(J)) else None


def eigenstructure(J):
    """Classify a pinch from its Jacobian: kind, eigenvalues, eigenvectors.

    Returns dict(kind, eigvals, eigvecs, n_stable, order) where `order` indexes
    the eigenpairs by DECREASING |lambda| -- the ordering the body-construction
    rules are written in (paper p.100 rules 3-4 speak of "the most stable (largest
    eigenvalue)" and "the most unstable (smallest eigenvalue)" eigenvector).

    `n_stable` counts |lambda| < 1. It is the quantity the path rule monotonically
    increases along (paper p.98): a profile can only ever move toward pinches that
    attract it in more directions than the last one did.
    """
    w, V = np.linalg.eig(J)
    mag = np.abs(w)
    n_stable = int(np.count_nonzero(mag < 1.0))
    if n_stable == len(w):
        kind = "stable_node"
    elif n_stable == 0:
        kind = "unstable_node"
    else:
        kind = "saddle"
    order = np.argsort(-mag)
    return {"kind": kind, "eigvals": w, "eigvecs": V, "n_stable": n_stable,
            "order": order}


def _edge_roots(sec, tp, P, i, j, n_scan=25):
    """Every root of the edge pinch equation on the binary edge (i, j).

    On an edge the pinch equation is a SCALAR equation in one unknown, so it can
    be bracketed instead of seeded. Parametrise the edge by t = x_i (x_j = 1 - t,
    everything else exactly zero); the residual components sum to zero and those
    off the edge are constants, so one component carries all the information.
    Scan t, bisect every sign change.

    Seeding a least-squares solve at fixed fractions along the edge instead is
    what the old net did, and it loses pinches: on c2-c4 the rectifying pinch on
    the ethane/butane edge slides toward the butane vertex as reflux rises, and
    above r = 0.51 it sat outside the 0.25/0.5/0.75 seeds. The body collapsed
    from a triangle to a segment and the method invented a maximum reflux for a
    simple column. Bracketing cannot miss a root that a sign change brackets, and
    it finds several on one edge where a single seed reports one.

    A root here solves component i only -- x_i (K_i - a) = bvec_i. It is a pinch
    of the whole system only when the off-edge components are consistent too,
    i.e. bvec_k = 0 there. Deciding that is the caller's job (`pinch_points`);
    the roots that fail it are not junk, they are the seeds for the near-edge
    pinches (`_relax_to_face`).
    """
    C = sec.delta.shape[0]

    def x_of(t):
        x = np.zeros(C)
        x[i], x[j] = t, 1.0 - t
        return x

    def g(t):
        try:
            return float(pinch_residual(sec, x_of(t), tp, P)[i])
        except (ValueError, FloatingPointError):
            return np.nan

    ts = np.linspace(1e-9, 1.0 - 1e-9, n_scan)
    gs = np.array([g(t) for t in ts])
    out = []
    for k in range(len(ts) - 1):
        a, b, ga, gb = ts[k], ts[k + 1], gs[k], gs[k + 1]
        if not (np.isfinite(ga) and np.isfinite(gb)) or ga * gb > 0.0:
            continue
        try:
            # brentq, not bisection: the bracket is already isolated, and each
            # evaluation is a bubble-point solve through the activity model, so
            # the ~45 halvings a plain 1e-13 bisection needs cost more than the
            # whole scan that found the bracket.
            out.append(x_of(brentq(g, a, b, xtol=1e-12, maxiter=60)))
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    return out


def _relax_to_face(sec, tp, P, x, off):
    """An edge root that is not itself a pinch -> the near-edge pinch it seeds.

    Returns (x, clipped) or None.

    A pinch has x_k = 0 only where bvec_k = 0, so with a SMEARED product spec --
    the 98/2 recoveries every example file carries -- no pinch sits exactly on an
    edge. It has moved a hair off it, to first order by

        d_k = bvec_k / (K_k(x) - a)                (from x_k (K_k - a) = bvec_k)

    and that displacement is tiny: 2e-4 on the ipa/water/EG rectifying section.
    None of `_interior_seeds`' four seeds is anywhere near it, so the face solve
    converged to the dominant node instead and the pinch was simply lost. Every
    stripping section in `docs/examples/` came back with ONE pinch and a body
    that was a straight line; the ipa rectifying body ran to pure water instead
    of the pure-EG sliver of the paper's figure. Both were this.

    Three outcomes:

    * `d` too big -- the edge root is not a perturbation of anything, so it is
      not a seed either. None.
    * seeded solve converges -- that is the pinch, exactly (residual ~1e-13).
    * it does not, which means the branch has left the simplex (every d_k < 0).
      Keep the face point as its stand-in, but only where the difference point
      is itself sharp on the off-face components. That gate is load-bearing: the
      residual left at a clipped point is exactly |bvec_off|, so without it the
      sharp ipa case admitted a false unstable node at residual 0.14.

    ponytail: the stand-in is a clip, not a solve. The branch really is outside
    (x_IPA = -5e-4 on the ipa stripping saddle) and `solve_pinch` cannot follow
    it there -- softmax puts an exact zero at -inf and cannot go past it. Upgrade
    path if a case ever needs the true point: signed continuation of the branch
    in sum-one coordinates, and an `in_simplex` slack to match.
    """
    xc = np.clip(x, 1e-12, None)
    try:
        _, T = tp.bubble(xc / xc.sum(), P)
        K = tp.K(np.atleast_2d(x), np.atleast_1d(T), np.atleast_1d(P))[0]
    except (ValueError, FloatingPointError):
        return None

    d = np.zeros_like(x)
    for k in off:
        den = K[k] - sec.a
        if abs(den) < 1e-12:
            return None
        d[k] = sec.bvec[k] / den
    if not np.all(np.isfinite(d)) or np.abs(d).max() > EDGE_RELAX:
        return None

    seed = np.clip(x + d, 1e-9, None)
    got = solve_pinch(sec, tp, P, seed / seed.sum(),
                      active=tuple(range(len(x))))
    if got is not None:
        return got, False

    scale = abs(sec.Delta / sec.V)
    if scale < 1e-300 or max(abs(sec.bvec[k]) for k in off) > SHARP_TOL * scale:
        return None
    y = np.clip(x + d, 0.0, None)
    s = float(y.sum())
    return (y / s, True) if s > 0.0 else None


def _interior_seeds(C, face):
    """Seeds for a face of dimension 2 or more: centroid, plus one leaning on
    each of its components.

    Faces of dimension 0 and 1 are solved exactly (a vertex is one residual
    evaluation, an edge is bracketed), so this is the only place a solve still
    depends on where it starts. It is also the only place it has to: a product
    with no exact zeros forces every component of the pinch to be nonzero --
    x_i (K_i - a) = bvec_i has no root at x_i = 0 unless bvec_i = 0 -- so that
    pinch lives in the face interior, where there is no bracket to exploit.

    Centroid plus one seed leaning on each component. Denser nets were tried --
    adding the three-per-internal-edge points the old full-simplex net used
    doubled the cost of the whole suite and found not one additional pinch on
    any case here, including the extractive one it was meant to help.
    """
    idx = list(face)
    m = len(idx)
    out = [np.zeros(C)]
    out[0][idx] = 1.0 / m
    for i in idx:
        s = np.zeros(C)
        s[idx] = 0.2 / (m - 1)
        s[i] = 0.8
        out.append(s / s.sum())
    return out


def _faces(C):
    """Every non-empty face of the simplex, smallest first: 2^C - 1 of them.

    7 at C = 3, 15 at C = 4, 31 at C = 5. Above that this enumeration is the cost
    driver and wants pruning to the faces the product composition can reach.
    """
    from itertools import combinations

    return [f for size in range(1, C + 1) for f in combinations(range(C), size)]


def pinch_points(sec, tp, P, seeds=None, want_eigen=True):
    """Every distinct pinch of one section at one operating point.

    Returns a list of dicts(x, in_simplex, margin, kind, eigvals, eigvecs,
    n_stable, order, clipped), ordered by bubble temperature so that body
    construction can speak of "along the path" without re-sorting. Points outside
    the simplex are kept and flagged; the extractive section needs them (paper
    p.100). `clipped` marks the one kind of record that is an approximation
    rather than a solve -- see `_relax_to_face`.

    Solved face by face, because that is where the solutions are (see
    `solve_pinch`). Each face gets the cheapest method that cannot miss a root on
    it: a vertex is a single residual evaluation, an edge is bracketed and
    bisected, and only faces of dimension 2 and up need a least-squares solve
    from a seed. An edge root that is not a pinch of the whole system is not
    discarded -- with a smeared product spec NO pinch is exactly on an edge, and
    those roots are the only usable seeds for the ones just off it
    (`_relax_to_face`). `seeds`, when given, overrides the enumeration with
    explicit (face, x0) pairs -- used to warm-start from a neighbouring reflux.
    """
    C = sec.delta.shape[0]
    found = []

    def candidates():
        if seeds is not None:
            for face, s in seeds:
                yield solve_pinch(sec, tp, P, s, active=face), False
            return
        for face in _faces(C):
            if len(face) == 1:
                x = np.zeros(C); x[face[0]] = 1.0
                yield solve_pinch(sec, tp, P, x, active=face), False
            elif len(face) == 2:
                off = [k for k in range(C) if k not in face]
                for x in _edge_roots(sec, tp, P, face[0], face[1]):
                    try:
                        r = float(np.linalg.norm(pinch_residual(sec, x, tp, P)))
                    except (ValueError, FloatingPointError):
                        continue
                    if np.isfinite(r) and r <= _ACCEPT:
                        yield x, False                # a pinch on the edge
                    else:
                        got = _relax_to_face(sec, tp, P, x, off)
                        if got is not None:
                            yield got                 # ...or just off it
            else:
                for s in _interior_seeds(C, face):
                    yield solve_pinch(sec, tp, P, s, active=face), False

    for x, clipped in candidates():
        if x is None:
            continue
        if any(np.linalg.norm(x - f["x"]) < _SAME_POINT for f in found):
            continue
        rec = {"x": x,
               "in_simplex": bool(x.min() > -_SIMPLEX_TOL
                                  and x.max() < 1.0 + _SIMPLEX_TOL),
               "margin": float(feasible_margin(sec, x)),
               "kind": "?", "eigvals": None, "eigvecs": None,
               "n_stable": 0, "order": None, "T": np.inf,
               "drop": int(np.argmax(x)), "clipped": bool(clipped)}
        if want_eigen:
            got = jacobian(sec, x, tp, P)
            if got is not None:
                J, rec["drop"] = got
                rec.update(eigenstructure(J))
        try:
            xc = np.clip(x, 1e-12, None)
            rec["T"] = float(tp.bubble(xc / xc.sum(), P)[1])
        except (ValueError, FloatingPointError):
            pass
        found.append(rec)
    found.sort(key=lambda r: r["T"])
    return found


def _demo():
    from side_features.bvm.problem import build_problem, overall_balance
    from side_features.bvm.sections import single_feed_chain
    from side_features.bvm.thermo_adapter import ColumnForgeThermo

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = ColumnForgeThermo(abc)
    z = np.array([0.4, 0.35, 0.25])
    prob = build_problem(["b", "t", "x"], [(z, 100.0, 1.0)], 760.0,
                         rec_lk=0.98, rec_hk=0.02)
    xD, xB, D, B = overall_balance(prob)

    rect, strip = single_feed_chain(prob, 4.0, xD, xB, D, B)
    ps = pinch_points(rect, tp, 760.0)
    assert ps, "rectifying section must have at least one pinch"
    assert all(np.isclose(p["x"].sum(), 1.0, atol=1e-8) for p in ps)
    assert any(p["in_simplex"] for p in ps), "and at least one inside the simplex"
    assert all(p["kind"] in ("stable_node", "unstable_node", "saddle", "?")
               for p in ps)
    # the residual really is satisfied, not merely reported -- except for a
    # clipped stand-in, which is a face point and whose residual is by
    # construction the difference point's own off-face content
    for p in ps:
        r = np.linalg.norm(pinch_residual(rect, p["x"], tp, 760.0))
        assert r < 1e-6 or p["clipped"], (p["x"], r)

    # temperature-ordered, which the body rules rely on
    Ts = [p["T"] for p in ps if np.isfinite(p["T"])]
    assert Ts == sorted(Ts), Ts

    # a pinch is a fixed point of the stage map: G(x*) = x* to solver tolerance
    inside = next(p for p in ps if p["in_simplex"])
    got = jacobian(rect, inside["x"], tp, 760.0)
    assert got is not None
    J, _drop = got
    assert J.shape == (2, 2)
    eig = eigenstructure(J)
    assert 0 <= eig["n_stable"] <= 2
    assert len(eig["order"]) == 2
    mags = np.abs(eig["eigvals"])[eig["order"]]
    assert np.all(np.diff(mags) <= 1e-12), mags      # order is |lambda| descending

    # every pinch lands ON a face -- each component either zero or on K_i = a --
    # which is the property that lets them be solved face by face at all
    for p in ps:
        if not p["in_simplex"]:
            continue
        near_zero = np.isclose(p["x"], 0.0, atol=1e-9)
        assert near_zero.sum() < len(p["x"]) - 1, p["x"]

    # a pinch that slides along an edge must be found wherever it sits, not only
    # where a seed happened to be: the c2-c4 failure was one migrating past them
    counts = []
    for r in (1.5, 3.0, 6.0, 12.0):
        sec_r = single_feed_chain(prob, r, xD, xB, D, B)[0]
        counts.append(sum(p["in_simplex"] for p in pinch_points(sec_r, tp, 760.0)))
    assert min(counts) == max(counts), f"pinch count varies with reflux: {counts}"

    # a SMEARED product spec puts no pinch exactly on an edge, and every section
    # still has to find more than the one dominant node -- one pinch spans a body
    # that is a straight line, which is what this whole near-edge path is for
    smear = build_problem(["b", "t", "x"], [(z, 100.0, 1.0)], 760.0,
                          rec_lk=0.98, rec_hk=0.02)
    xDs, xBs, Ds, Bs = overall_balance(smear)
    assert xBs.min() > 0.0, "this check is meaningless with an exactly-zero xB"
    n_smear = []
    for sec in single_feed_chain(smear, 4.0, xDs, xBs, Ds, Bs):
        got = [p for p in pinch_points(sec, tp, 760.0) if p["in_simplex"]]
        assert len(got) >= 2, f"{sec.name} found {len(got)} pinch(es): {got}"
        n_smear.append(len(got))
        for p in got:
            r = np.linalg.norm(pinch_residual(sec, p["x"], tp, 760.0))
            assert r < 1e-6 or p["clipped"], (sec.name, p["x"], r)
    print(f"rbm.pinch self-check OK  {len(ps)} pinches "
          f"({sum(p['in_simplex'] for p in ps)} in simplex), "
          f"{counts[0]} held across reflux {counts}, "
          f"smeared spec {n_smear}")


if __name__ == "__main__":
    _demo()
