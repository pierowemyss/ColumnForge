"""Rectification bodies and their intersection (paper p.98 and p.100).

A rectification body is a linearised stand-in for everything a section's profiles
can do at one operating point. Instead of marching a curve, RBM spans the
section's pinch points into a simplex and treats that polytope as the reachable
set. Two adjacent sections can be joined by a real column profile exactly when
their bodies intersect (paper p.120):

    bodies apart      -> infeasible, below minimum reflux
    bodies touching   -> minimum reflux
    bodies overlapping-> feasible, above minimum reflux

The payoff over comparing marched curves is dimensional. Two 1-D curves in the
(C-1)-simplex generically miss for C >= 4 -- which is why BVM has to solve for
non-key product compositions to force an intersection. Bodies are up to
(C-1)-dimensional, so they meet generically at any C, and the quaternary example
in the paper needs no such fixing.

Two constructions, because the two kinds of section have different pinch maps.

PRODUCT-ANCHORED sections (rectifying, stripping), paper p.98:
  1. take the section's pinch points and their stability;
  2. profiles start at the product composition;
  3. a profile may only touch pinches whose count of stable eigenvectors
     increases STRICTLY MONOTONOUSLY along it;
  4. the body is the simplex spanned by the product composition and that chain.

EXTRACTIVE MIDDLE sections, paper p.100 rules 1-5 -- different, because the
middle section's net product x_N usually lies outside the composition space and
its pinch map is therefore incomplete:
  1. take only the SADDLE pinches;
  2. chain them under the same strict-monotone stable-eigenvector rule;
  3. start the body by following the chain's first saddle's MOST STABLE
     (largest |lambda|) eigenvector to the edge of the simplex;
  4. end it by following the last saddle's MOST UNSTABLE (smallest |lambda|)
     eigenvector to the edge;
  5. take both directions of each, so one chain gives (typically) four bodies.
"""

import numpy as np
from scipy.optimize import minimize

#: Bodies within this distance are treated as touching. Composition-space units,
#: so it is a mole-fraction gap and the same number means the same thing at every
#: C -- unlike a tolerance derived from a stage step, which is what made BVM's
#: connection test scale-dependent.
TOUCH_TOL = 1e-4


def _dedupe(points, tol=1e-6):
    """Drop repeated vertices. The tolerance is loose on purpose: `solve_pinch`
    works in softmax coordinates and so cannot return an exact zero, which puts a
    pure-component pinch a few times 1e-9 away from the product composition that
    IS exactly the vertex. At 1e-9 the two survived as separate vertices of the
    same hull."""
    out = []
    for p in points:
        if not any(np.linalg.norm(p - q) < tol for q in out):
            out.append(np.asarray(p, float))
    return out


def chains(pinches, saddles_only=False):
    """Maximal pinch chains under the strict stable-eigenvector rule.

    Only MAXIMAL chains are enumerated. A body is a convex hull, so the hull of
    {x_prod, p0, p1, p2} already contains the hull of {x_prod, p2} -- every
    shorter chain's body is a face of a longer one's, and testing it separately
    would find nothing new. Where two pinches share a stable-eigenvector count
    (the paper's r1a and r1b) they cannot both be on one chain, so each
    combination across the levels is its own maximal chain.

    UNSTABLE NODES are dropped from product-section chains. Rule 2 says the
    profile starts at the product composition and is thereafter drawn toward the
    pinches it touches; an unstable node repels in every direction, so no profile
    arriving from elsewhere can reach one. This is the paper's PWG walkthrough
    discarding r0 (p.98). It matters: without it the chain reaches an extra,
    far-flung vertex, the body it spans is much larger, and intersection becomes
    nearly automatic -- which showed up as a simple column reporting feasibility
    at every reflux and an extractive one at every entrainer ratio.

    The paper's other discard in that walkthrough -- r1a and r1b, on grounds it
    does not spell out -- is `blocked_by_unstable_node`, applied by
    `product_bodies` before the pinches get here.
    """
    usable = [p for p in pinches
              if p["in_simplex"] and p["eigvals"] is not None
              and (p["kind"] != "unstable_node" if not saddles_only else
                   p["kind"] == "saddle")]
    if not usable:
        return []
    levels = {}
    for p in usable:
        levels.setdefault(p["n_stable"], []).append(p)

    out = [[]]
    for k in sorted(levels):
        out = [c + [p] for c in out for p in levels[k]]
    return out


def lift_direction(v, C, drop=None):
    """Reduced (C-1) eigenvector -> full composition-space direction.

    `pinch.jacobian` differentiates along the simplex edges leaving the pinch's
    largest component `drop`, so a reduced step u moves that component by
    -sum(u) and the others by u. `drop` defaults to the last component, which is
    what the fixed-coordinate version always used.

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


def _to_edge(x, direction, max_step=2.0):
    """Follow `direction` from `x` until the simplex boundary (paper p.100, 3-4).

    Returns the boundary point. The simplex is {x >= 0, sum x = 1}; the direction
    is projected to be sum-preserving first, so the walk stays on the plane and
    only ever runs into a `x_i = 0` face.
    """
    d = np.asarray(direction, float).real
    d = d - d.mean()                       # sum-preserving
    n = np.linalg.norm(d)
    if n < 1e-300:
        return np.asarray(x, float)
    d = d / n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d < 0, -x / d, np.inf)     # first face hit
    # t >= 0, not t > 0. A pinch sitting ON the face it is being walked toward
    # hits it at zero distance; excluding that left no candidate, and the
    # `max_step` fallback below then invented a point two units away along an
    # unbounded direction. That is how an extractive body picked up the vertex
    # (2.345, 0, 0.07) -- outside the composition space entirely. Bodies that
    # large intersect everything, which is why no extractive column showed a
    # maximum reflux and why r_min stopped varying with entrainer flow.
    t = float(np.min(np.where(np.isfinite(t) & (t >= 0), t, np.inf)))
    if not np.isfinite(t):
        return np.asarray(x, float)            # no face this way: invent nothing
    return np.clip(x + min(t, max_step) * d, 0.0, None)


def blocked_by_unstable_node(x_prod, x_pinch, unstable, tol=1e-3):
    """Does an unstable node sit ON the segment x_prod -> x_pinch?

    If it does, the body edge between those two vertices runs straight through a
    repellor, and no profile sweeps it: a profile leaving the product along that
    line is pushed away in every direction the moment it arrives. The body is
    therefore not a reachable set and the chain that spans it is discarded.

    This is the paper's second, unstated discard in the PWG walkthrough (p.98,
    r1a and r1b). On ipa/water/EG the rectifying pinches are the near-IPA saddle,
    the IPA/water azeotrope pinch (unstable node) and the near-water saddle, all
    three on the same simplex edge with the azeotrope between the product and the
    near-water one. Without this the near-water chain spans a body running pure
    IPA -> pure water, it is the one that takes part in the junction, and the
    thin pure-IPA -> pure-EG body of the paper's figure is drawn as an also-ran.

    ponytail: collinear-only -- projection onto the segment, then perpendicular
    distance. An unstable node that sits NEAR the segment rather than on it does
    not block. Upgrade if a case needs it: integrate the unstable manifold and
    test reachability properly, which is a whole marcher and the thing RBM exists
    to avoid.
    """
    d = np.asarray(x_pinch, float) - np.asarray(x_prod, float)
    n2 = float(d @ d)
    if n2 < 1e-18:
        return False
    for u in unstable:
        lam = float((np.asarray(u, float) - x_prod) @ d / n2)
        # strictly between: an unstable node AT either end is the product itself
        # or the pinch itself, neither of which blocks anything
        if not (0.02 < lam < 0.98):
            continue
        if np.linalg.norm(x_prod + lam * d - u) < tol:
            return True
    return False


def product_bodies(pinches, x_prod):
    """Rectification bodies of a product-anchored section (paper p.98).

    One body per chain: the simplex spanned by the product composition and the
    pinches along it. For c2-c4's rectifying section that is the single chain
    saddle -> stable node, so the body is the triangle x_D-p1-p2 of the textbook
    figure, and its bodies touch the stripping section's at r = 0.134 against
    Underwood's 0.147 for the same split.

    Getting that triangle depends entirely on the pinches carrying honest
    stability, which is worth knowing because for a while they did not. Every
    c2-c4 pinch sits on a face of the simplex, `pinch.jacobian` used to difference
    across the simplex boundary there, and all three came back `stable_node` --
    one level, so the chain rule made them three mutually exclusive alternatives
    and the body degenerated to a segment lying in a face. The stripping body is
    interior and cannot touch a segment in a face at any reflux, so the split
    read as infeasible everywhere. With the derivative taken along the inward
    simplex edges the ladder is unstable node -> saddle -> stable node, one chain,
    one triangle. A section whose pinches all report the same `n_stable` is the
    symptom to look for if this comes back.
    """
    x_prod = np.asarray(x_prod, float)
    unstable = [p["x"] for p in pinches
                if p["in_simplex"] and p["kind"] == "unstable_node"]
    reachable = [p for p in pinches
                 if p["kind"] == "unstable_node" or not p["in_simplex"]
                 or not blocked_by_unstable_node(x_prod, p["x"], unstable)]
    out = []
    for ch in chains(reachable):
        verts = _dedupe([x_prod] + [p["x"] for p in ch])
        if len(verts) >= 2:
            out.append({"vertices": np.array(verts),
                        "pinches": [p["x"] for p in ch],
                        "kinds": [p["kind"] for p in ch]})
    if not out:                              # no usable pinch: the product alone
        out.append({"vertices": np.array([x_prod]), "pinches": [], "kinds": []})
    return out


def middle_bodies(pinches):
    """Rectification bodies of an extractive middle section (paper p.100, 1-5)."""
    out = []
    for ch in chains(pinches, saddles_only=True):
        if not ch:
            continue
        first, last = ch[0], ch[-1]
        C = first["x"].shape[0]
        # rule 3: most stable = LARGEST |lambda| -> first entry of `order`
        v_start = lift_direction(first["eigvecs"][:, first["order"][0]], C,
                                 first.get("drop"))
        # rule 4: most unstable = SMALLEST |lambda| -> last entry of `order`
        v_end = lift_direction(last["eigvecs"][:, last["order"][-1]], C,
                               last.get("drop"))
        # rule 5: both directions of each -> four bodies per chain
        for s_sign in (+1.0, -1.0):
            for e_sign in (+1.0, -1.0):
                start = _to_edge(first["x"], s_sign * v_start)
                end = _to_edge(last["x"], e_sign * v_end)
                verts = _dedupe([start] + [p["x"] for p in ch] + [end])
                if len(verts) >= 2:
                    out.append({"vertices": np.array(verts),
                                "pinches": [p["x"] for p in ch],
                                "kinds": [p["kind"] for p in ch],
                                "start": start, "end": end})
    return out


def body_distance(A, B):
    """Distance between two convex hulls: min ||A.lam - B.mu||, lam, mu simplices.

    A small convex QP -- the objective is a quadratic in (lam, mu) and the
    feasible set is a product of two simplices -- so a local solve is the global
    one and SLSQP is enough. Zero means the bodies intersect.
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    na, nb = len(A), len(B)
    if na == 0 or nb == 0:
        return float("inf")
    if na == 1 and nb == 1:
        return float(np.linalg.norm(A[0] - B[0]))

    def unpack(w):
        return w[:na], w[na:]

    def f(w):
        lam, mu = unpack(w)
        d = lam @ A - mu @ B
        return float(d @ d)

    def jac(w):
        lam, mu = unpack(w)
        d = lam @ A - mu @ B
        return np.concatenate([2.0 * A @ d, -2.0 * B @ d])

    w0 = np.concatenate([np.full(na, 1.0 / na), np.full(nb, 1.0 / nb)])
    cons = [{"type": "eq", "fun": lambda w: w[:na].sum() - 1.0,
             "jac": lambda w: np.concatenate([np.ones(na), np.zeros(nb)])},
            {"type": "eq", "fun": lambda w: w[na:].sum() - 1.0,
             "jac": lambda w: np.concatenate([np.zeros(na), np.ones(nb)])}]
    res = minimize(f, w0, jac=jac, bounds=[(0.0, 1.0)] * (na + nb),
                   constraints=cons, method="SLSQP",
                   options={"maxiter": 200, "ftol": 1e-16})
    return float(np.sqrt(max(res.fun, 0.0)))


def sets_distance(bodies_a, bodies_b):
    """Closest approach over every pairing of two sets of bodies.

    Returns (distance, index_a, index_b). A section offers several alternative
    bodies (the paper's four per middle-section chain); the section is joinable
    if ANY of them meets any of the neighbour's, and which pair it was is what
    the diagram marks as the ACTIVE body.
    """
    best, ia, ib = float("inf"), None, None
    for i, A in enumerate(bodies_a):
        for j, B in enumerate(bodies_b):
            d = body_distance(A["vertices"], B["vertices"])
            if d < best:
                best, ia, ib = d, i, j
    return best, ia, ib


def _demo():
    # --- body_distance is a real hull distance, checked against known answers
    seg_a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    seg_b = np.array([[0.0, 0.0, 1.0]])
    d = body_distance(seg_a, seg_b)
    assert abs(d - np.sqrt(1.5)) < 1e-6, d          # vertex to segment midpoint

    crossing = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    assert body_distance(seg_a, crossing) < 1e-7, "touching hulls -> zero"

    apart = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])
    near = np.array([[0.1, 0.1, 0.8], [0.2, 0.2, 0.6]])
    assert body_distance(apart, near) > 0.5

    # --- the walk to the simplex edge lands ON a face and stays in the simplex
    x = np.array([0.4, 0.35, 0.25])
    e = _to_edge(x, np.array([1.0, -1.0, 0.0]))
    assert abs(e.sum() - 1.0) < 1e-9 and e.min() > -1e-12, e
    assert min(e) < 1e-9, e                          # reached a face

    # a walk that starts ON a face and pushes further into it must stay put, not
    # run off to `max_step`: that is what put the vertex (2.345, 0, 0.07) into an
    # extractive body and made every extractive column look feasible at any reflux
    on_face = np.array([0.6, 0.0, 0.4])
    for d in (np.array([1.0, -1.0, 0.0]), np.array([-1.0, -1.0, 2.0])):
        w = _to_edge(on_face, d)
        assert w.max() <= 1.0 + 1e-9 and w.min() > -1e-12, (d, w)

    # --- chains respect strict monotonicity and are maximal
    def pk(n, kind, x):
        return {"in_simplex": True, "n_stable": n, "kind": kind,
                "x": np.array(x, float), "eigvals": np.ones(2),
                "eigvecs": np.eye(2), "order": np.array([0, 1])}

    ps = [pk(0, "unstable_node", [1, 0, 0]), pk(1, "saddle", [0, 1, 0]),
          pk(1, "saddle", [0, 0.5, 0.5]), pk(2, "stable_node", [0, 0, 1])]
    chs = chains(ps)
    assert len(chs) == 2, chs                        # the two n_stable==1 options
    for ch in chs:
        ns = [p["n_stable"] for p in ch]
        assert ns == [1, 2], ns                      # unstable node dropped
        assert all(p["kind"] != "unstable_node" for p in ch)
    assert len(chains(ps, saddles_only=True)) == 2   # one saddle each

    # --- product bodies include the product composition itself
    xD = np.array([0.9, 0.1, 0.0])
    bs = product_bodies(ps, xD)
    assert bs and all(np.any(np.all(np.isclose(b["vertices"], xD), axis=1))
                      for b in bs)
    assert all(b["vertices"].shape[1] == 3 for b in bs)

    # --- a chain whose body edge runs through an unstable node is discarded
    prod = np.array([1.0, 0.0, 0.0])
    far = np.array([0.0, 1.0, 0.0])
    mid = np.array([0.5, 0.5, 0.0])                  # on the segment
    assert blocked_by_unstable_node(prod, far, [mid])
    assert not blocked_by_unstable_node(prod, far, [np.array([0.4, 0.3, 0.3])])
    assert not blocked_by_unstable_node(prod, far, [prod, far])   # the ends
    edge = [pk(0, "unstable_node", mid), pk(1, "saddle", far),
            pk(1, "saddle", [0.85, 0.15, 0.0]), pk(2, "stable_node", [0, 0, 1])]
    kept = product_bodies(edge, prod)
    assert len(kept) == 1, [b["pinches"] for b in kept]
    assert not any(np.allclose(v, far) for b in kept for v in b["vertices"])

    # --- a reduced eigenvector lifts to a sum-preserving composition direction
    assert np.allclose(lift_direction(np.array([1.0, 0.0]), 3), [1.0, 0.0, -1.0])
    assert abs(lift_direction(np.array([0.3, -0.2]), 3).sum()) < 1e-12

    # --- a middle chain yields four bodies (rule 5: both directions of each end)
    mb = middle_bodies([pk(1, "saddle", [0.3, 0.3, 0.4])])
    assert len(mb) == 4, len(mb)
    for b in mb:
        assert b["vertices"].min() > -1e-9 and abs(b["vertices"].sum(1) - 1).max() < 1e-6

    # --- sets_distance reports which pair was active
    d, i, j = sets_distance([{"vertices": seg_a}], [{"vertices": seg_b},
                                                    {"vertices": crossing}])
    assert j == 1 and d < 1e-7, (d, i, j)
    print("rbm.bodies self-check OK  hull distance, edge walk, chains, "
          f"{len(mb)} middle bodies per chain")


if __name__ == "__main__":
    _demo()
