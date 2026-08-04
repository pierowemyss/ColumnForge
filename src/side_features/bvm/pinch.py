"""Pinch points, minimum reflux, minimum entrainer (blueprint Sec 8).

A pinch is a fixed point of the stage map, x* = G(x*): the composition where the
operating line meets the equilibrium surface and marching stalls. The marcher
already lands on it (status='pinch'), so `pinch_point` just reads that endpoint.

Feasibility hinges on the *eigenstructure* of J = dG/dx at the pinch, not the
operating-line coefficients -- this is the sense in which the method is "matrix"
based (Sec 8). A saddle (mixed |lambda|) has the 1-D invariant manifolds that
anchor strongly-pinched interior sections (Sec 6.3).

`pinch_solve` returns ONE root, from one seed. A section generally has several
pinches and the anchoring rules need all of them (Bruggemann & Marquardt rule 1,
`docs/papers/rbm_bruggemann_marquardt.md` p.102), so `pinch_points` below
enumerates the lot, face by face. It lived in `side_features.rbm.pinch` until
BVM's extractive anchoring turned out to need the same thing; RBM re-exports it
from here, which keeps the one dependency arrow (rbm -> bvm) pointing the way it
already did.

R_min (and minimum E/F) is where the rectifying and stripping reachable regions
only just touch. Operationally that is the smallest R at which the profiles
still connect, found by bisection on the connection test (Sec 7) -- robust, and
equivalent to the pinch-tangency condition. ponytail: bisection here; a direct
pinch-tangency solve would be faster but no more correct for sizing.
"""

import numpy as np
from scipy.optimize import brentq, least_squares, root

from .march import march_section
from .parallel import pmap, pnarrow
from .sections import feasible, feasible_margin, region_center

#: A branch point counts as "in the simplex" with this much slack, so a pinch
#: sitting exactly on a face (a binary-edge pinch, which is the common case) is
#: kept rather than lost to round-off.
_SIMPLEX_TOL = 1e-7

#: Two pinch points closer than this are the same point found twice. Loose
#: because seeds converge to a shared root from very different starting places.
_SAME_POINT = 1e-4

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

#: A saddle seeds a middle-section anchor only if its `k_gap` -- min_k |K_k - a|/a
#: at the pinch -- is under this, i.e. some component sits on the K_k = a pinch
#: branch.
#:
#: The pinch equation factorises as x_k (K_k - a) = bvec_k, so wherever the
#: difference point is sharp (bvec_k ~ 0) it splits into an x_k = 0 branch and a
#: K_k = a branch. The first is the binary/face saddle, the second is the paper's
#: TERNARY saddle -- the one whose existence is the prerequisite for a feasible
#: extractive separation (p.84) and the one Figure 6 builds its four bodies from.
#: Both are real solutions and both are reported; only the ternary one is
#: preferred (`rbm.bodies.middle_bodies`, `anchor.interior_candidates`).
#:
#: Measured on ipa/water/EG over product specs 0.999/1e-6, 0.99/0.01 and
#: 0.98/0.02, E/F in {0.40, 0.75, 1.50}, r in {2.2, 3, 4}: ternary saddles land
#: at 0.0016-0.0067, face saddles at 0.073-0.585. A threshold on the smallest
#: mole fraction cannot separate the same set -- at 0.98/0.02, r = 4 the ternary
#: saddle's own minimum component is 0.033, larger than nothing in particular.
BRANCH_TOL = 0.02


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


def _softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


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


def jacobian(sec, xstar, tp, P, h=1e-6, down=None):
    """Jacobian of the stage map at a pinch, in reduced (C-1) coordinates.

    Differentiated from the *pinch* relation rather than by re-marching, so a
    section whose march would be ill-conditioned still gets a clean derivative.

    The map is taken in the direction the section's profile is TRACED, which is
    not cosmetic: the two directions are exact inverses, so their eigenvalues are
    reciprocals and "stable" and "unstable" swap between them. Differentiating
    the down-map for every section made every stripping pinch come back an
    unstable node, which is the opposite of the paper's Figure 5 (right) -- and
    since the body rules chain on the count of STABLE eigenvectors, it quietly
    inverted the whole construction for half the column.

        down (Delta > 0):  G(x) = K(x)^-1 (a x + bvec)
        up   (Delta < 0):  G(x) = (K(x) x - bvec) / a

    `down` defaults to `sec.dir > 0`, i.e. sign(Delta), which is right for a
    PRODUCT-ANCHORED section: rectifying profiles are traced down from x_D and
    stripping profiles up from x_B, and that is exactly what sign(Delta) says.
    A MIDDLE section has no product anchor. Its Delta = D - E changes sign as the
    entrainer flow crosses the distillate rate while its profile still runs
    top-to-bottom, so sign(Delta) is unrelated to the tracing direction and the
    caller has to say. On ipa/water/EG at r = 2.2 the default labels the
    near-glycol pinch an unstable node at E/F = 0.750 (Delta = -13.2) and a
    stable node at E/F = 0.400 (Delta = +21.8) -- same topology, opposite
    reading, and a discontinuity in r_min/r_max at E = D. Forced down, it is the
    stable node the paper reports (p.72) at both.

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
    composition space, which is what `lift_direction` does.
    """
    C = xstar.shape[0]
    down = (sec.dir > 0) if down is None else bool(down)
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

    Returns dict(kind, eigvals, eigvecs, saddle, n_stable, order) where `order`
    indexes the eigenpairs by DECREASING |lambda| -- the ordering the
    body-construction rules are written in (paper p.100 rules 3-4 speak of "the
    most stable (largest eigenvalue)" and "the most unstable (smallest
    eigenvalue)" eigenvector).

    `n_stable` counts |lambda| < 1. It is the quantity the path rule monotonically
    increases along (paper p.98): a profile can only ever move toward pinches that
    attract it in more directions than the last one did. Only a saddle (mixed
    |lambda|) carries the 1-D invariant manifolds that anchor a strongly-pinched
    interior section (Sec 6.3).
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
            "order": order, "saddle": kind == "saddle"}


def lift_direction(v, C, drop=None):
    """Reduced (C-1) eigenvector -> full composition-space direction.

    `jacobian` differentiates along the simplex edges leaving the pinch's largest
    component `drop`, so a reduced step u moves that component by -sum(u) and the
    others by u. `drop` defaults to the last component, which is what the
    fixed-coordinate version always used.

    Eigenvectors of a real non-symmetric Jacobian can be complex; only the real
    part is a direction in composition space, and a complex pair means the
    profile spirals, which a straight-line body cannot represent anyway.
    """
    u = np.asarray(v, float if np.isrealobj(v) else complex).real.ravel()
    if u.shape[0] == C:
        return u
    d = np.empty(C)
    p = C - 1 if drop is None else int(drop)
    d[[i for i in range(C) if i != p]] = u
    d[p] = -u.sum()
    return d


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


def pinch_points(sec, tp, P, seeds=None, want_eigen=True, down=None):
    """Every distinct pinch of one section at one operating point.

    Returns a list of dicts(x, in_simplex, margin, kind, eigvals, eigvecs,
    n_stable, order, k_gap, clipped), ordered by bubble temperature so that body
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
    `down` forces the direction the stage map is differentiated in -- see
    `jacobian`; a middle section has to pass it.

    `k_gap` is min_k |K_k(x*) - a| / a, which says whether the pinch sits on a
    K_k = a branch. `rbm.bodies.middle_bodies` and `anchor.interior_candidates`
    use it to tell a ternary saddle from a face one; see `BRANCH_TOL`.
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
               "n_stable": 0, "order": None, "T": np.inf, "k_gap": np.inf,
               "drop": int(np.argmax(x)), "clipped": bool(clipped)}
        if want_eigen:
            got = jacobian(sec, x, tp, P, down=down)
            if got is not None:
                J, rec["drop"] = got
                rec.update(eigenstructure(J))
        try:
            xc = np.clip(x, 1e-12, None)
            T = float(tp.bubble(xc / xc.sum(), P)[1])
            K = tp.K(np.atleast_2d(x), np.atleast_1d(T), np.atleast_1d(P))[0]
            rec["T"] = T
            rec["k_gap"] = float(np.min(np.abs(K - sec.a)) / abs(sec.a))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            pass
        found.append(rec)
    found.sort(key=lambda r: r["T"])
    return found


def bisect_min(feasible_fn, lo, hi, tol=1e-3, max_iter=40, prescan=12,
               cancelled=None):
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
        # the pre-scan is the expensive half and its points are independent
        feas = pmap(feasible_fn, grid, cancelled=cancelled)
        if any(f is None for f in feas):
            return None                      # cancelled part-way through
        if not any(feas):
            return None
        start = max(i for i, f in enumerate(feas) if f)   # last feasible sample
        while start > 0 and feas[start - 1]:              # back to its run's start
            start -= 1
        lo, hi = grid[start - 1], grid[start]     # boundary lies in this cell
    elif not feasible_fn(hi):
        return None
    return pnarrow(feasible_fn, lo, hi, tol, cancelled=cancelled,
                   max_rounds=max_iter)


def feasible_band(feasible_fn, lo, hi, n_scan=24, tol=1e-3, log=True,
                  cancelled=None):
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

    The scan is the expensive half and its points are independent, so it goes to
    `parallel.pmap`; the two refinements are bisections and stay serial.
    `cancelled()` is checked between evaluations -- a band is minutes of work on
    a non-ideal ternary and the GUI's Cancel has to reach in here to mean
    anything. A cancelled scan returns (None, None).
    """
    grid = (np.geomspace(max(lo, 1e-6), hi, int(n_scan)) if log
            else np.linspace(lo, hi, int(n_scan)))
    scanned = pmap(feasible_fn, [float(x) for x in grid], cancelled=cancelled)
    if any(v is None for v in scanned):
        return None, None                  # cancelled part-way through the scan
    ok = [bool(v) for v in scanned]
    if not any(ok):
        return None, None
    i0 = int(np.argmax(ok))
    i1 = len(ok) - 1 - int(np.argmax(ok[::-1]))

    def refine(bad, good):
        return pnarrow(feasible_fn, bad, good, tol, cancelled=cancelled)

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
    jac = jacobian(rect, xstar, tp, 760.0)
    assert jac is not None
    cl = eigenstructure(jac[0])
    assert cl["kind"] in ("stable_node", "unstable_node", "saddle")
    assert cl["eigvals"].shape == (2,)
    # `order` is by decreasing |lambda| -- rules 3-4 are written in that order
    mags = np.abs(cl["eigvals"])[cl["order"]]
    assert np.all(np.diff(mags) <= 1e-12), mags

    # the whole reason `pinch_points` lives here: one seed finds ONE root, and a
    # section generally has several. On the extractive middle section of
    # ipa/water/EG that difference is the two saddles `pinch_solve` misses
    # entirely, which is what left the extractive anchor with nothing to launch
    # from -- see `anchor.interior_candidates`.
    one = pinch_solve(rect, tp, 760.0)
    many = [p for p in pinch_points(rect, tp, 760.0) if p["in_simplex"]]
    assert len(many) > 1, many
    assert any(np.linalg.norm(p["x"] - one["xstar"]) < 1e-3 for p in many), \
        (one["xstar"], [p["x"] for p in many])
    for p in many:
        r = np.linalg.norm(pinch_residual(rect, p["x"], tp, 760.0))
        assert r < 1e-6 or p["clipped"], (p["x"], r)
    # a lifted eigenvector is a direction along the simplex: it sums to zero
    v = lift_direction(many[0]["eigvecs"][:, 0], 3, many[0]["drop"])
    assert v.shape == (3,) and abs(v.sum()) < 1e-12, v

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
    print(f"pinch self-check OK  pinch={cl['kind']}  {len(many)} pinches vs 1 seeded"
          f"  R_min~{Rmin:.2f}  island->{got:.2f}")


if __name__ == "__main__":
    _demo()
