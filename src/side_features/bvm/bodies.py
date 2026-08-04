"""Rectification bodies and their intersection (paper p.98 and p.100).

A rectification body is a linearised stand-in for everything a section's profiles
can do at one operating point. Instead of marching a curve, span the section's
pinch points into a simplex and treat that polytope as the reachable set. Two
adjacent sections can be joined by a real column profile exactly when their
bodies intersect (paper p.120):

    bodies apart      -> infeasible, below minimum reflux
    bodies touching   -> minimum reflux
    bodies overlapping-> feasible, above minimum reflux

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
  1. take only the SADDLE pinches -- and only the TERNARY ones, see `BRANCH_TOL`;
  2. chain them under the same strict-monotone stable-eigenvector rule;
  3. start the body by following the chain's first saddle's STABLE eigenvector
     to the edge of the simplex (the profile ARRIVES along it);
  4. end it by following the last saddle's UNSTABLE eigenvector to the edge
     (the profile LEAVES along it);
  5. take both directions of each, so one chain gives (typically) four bodies.

The body a chain spans is the convex hull of ONE profile polyline
S -> x*_1 -> ... -> x*_n -> E. A real middle section runs down one arm of it or
turns the elbow at a saddle; it cannot be two bodies at once, which is what
`winning_middle_body` enforces for both callers.

THIS MODULE LIVES IN BVM, and RBM re-exports it. It was RBM's, and RBM is still
the only caller that treats a body as the whole answer -- but BVM needs the same
geometry for a different job: choosing WHICH body its interior section is going
to march inside, before it marches anything. Deciding that from the bodies is
far cheaper than marching every candidate curve and ranking the results, and
having one construction rather than two is what lets the two modules be compared
at all (`body_id`). The dependency arrow stays rbm -> bvm, as it already did for
`pinch`.

ponytail: `chains` drops pinches outside the simplex, which contradicts p.100's
premise that some middle-section branches leave it (and `pinch`'s own docstring).
Left alone because every pinch measured on the paper's own PWG case is inside;
upgrade when a case produces an outside one that matters.
"""

import numpy as np
from scipy.optimize import minimize

from .pinch import BRANCH_TOL, lift_direction  # noqa: F401  (re-exported)

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
    usable = [
        p
        for p in pinches
        if p["in_simplex"]
        and p["eigvals"] is not None
        and (
            p["kind"] != "unstable_node" if not saddles_only else p["kind"] == "saddle"
        )
    ]
    if not usable:
        return []
    levels = {}
    for p in usable:
        levels.setdefault(p["n_stable"], []).append(p)

    out = [[]]
    for k in sorted(levels):
        out = [c + [p] for c in out for p in levels[k]]
    return out


def _to_edge(x, direction, max_step=2.0):
    """Follow `direction` from `x` until the simplex boundary (paper p.100, 3-4).

    Returns the boundary point. The simplex is {x >= 0, sum x = 1}; the direction
    is projected to be sum-preserving first, so the walk stays on the plane and
    only ever runs into a `x_i = 0` face.

    This is the edge of COMPOSITION space, which for a middle section is wider
    than the set of compositions the section can hold. `bvm.anchor._ray_end` is
    the same walk clipped to the section's own balance, and the difference is
    load-bearing: swapping this in there makes every reflux from 1.5 to 8 report
    feasible (docs/adr/0004). The wide walk is safe for SELECTING among bodies,
    which is all `winning_middle_body` uses it for, and unsafe for deciding
    whether a junction closes, which stays with the marched profile.
    """
    d = np.asarray(direction, float).real
    d = d - d.mean()  # sum-preserving
    n = np.linalg.norm(d)
    if n < 1e-300:
        return np.asarray(x, float)
    d = d / n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d < 0, -x / d, np.inf)  # first face hit
    # t >= 0, not t > 0. A pinch sitting ON the face it is being walked toward
    # hits it at zero distance; excluding that left no candidate, and the
    # `max_step` fallback below then invented a point two units away along an
    # unbounded direction. That is how an extractive body picked up the vertex
    # (2.345, 0, 0.07) -- outside the composition space entirely. Bodies that
    # large intersect everything, which is why no extractive column showed a
    # maximum reflux and why r_min stopped varying with entrainer flow.
    t = float(np.min(np.where(np.isfinite(t) & (t >= 0), t, np.inf)))
    if not np.isfinite(t):
        return np.asarray(x, float)  # no face this way: invent nothing
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
    test reachability properly, which is a whole marcher and the thing the body
    method exists to avoid.
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
    unstable = [
        p["x"] for p in pinches if p["in_simplex"] and p["kind"] == "unstable_node"
    ]
    reachable = [
        p
        for p in pinches
        if p["kind"] == "unstable_node"
        or not p["in_simplex"]
        or not blocked_by_unstable_node(x_prod, p["x"], unstable)
    ]
    out = []
    for ch in chains(reachable):
        verts = _dedupe([x_prod] + [p["x"] for p in ch])
        if len(verts) >= 2:
            out.append(
                {
                    "vertices": np.array(verts),
                    "pinches": [p["x"] for p in ch],
                    "kinds": [p["kind"] for p in ch],
                    # x_prod is put first and `_dedupe` keeps order, so the anchor
                    # is vertex 0 -- recorded rather than re-derived because
                    # `rank_middle_bodies` reads it to tell the product side of the
                    # body from the pinch side.
                    "anchor": 0,
                }
            )
    if not out:  # no usable pinch: the product alone
        out.append(
            {"vertices": np.array([x_prod]), "pinches": [], "kinds": [], "anchor": 0}
        )
    return out


def body_id(xstar, s_sign, e_sign):
    """Stable, printable name for one middle body.

    A middle body is fixed by three things and only three: which saddle it turns
    at, and which way each of the two eigendirections was walked (rule 5). Both
    modules already have that triple -- `middle_bodies` as (chain head, s_sign,
    e_sign), `bvm.anchor` as the saddle plus which of `manifold_branches`' fixed
    (+v, -v) pair each end came from -- and neither recorded it, so the same body
    had no name in the two panels and the two answers could not be compared. Four
    decimals is a stage-map coordinate, well inside the tolerance any two solves
    of the same pinch agree to.
    """
    x = np.round(np.asarray(xstar, float), 4)
    return "x*=({}) S{} E{}".format(
        ",".join(f"{v:.4f}" for v in x),
        "+" if float(s_sign) >= 0 else "-",
        "+" if float(e_sign) >= 0 else "-",
    )


def middle_bodies(pinches, sec=None):
    """Rectification bodies of an extractive middle section (paper p.100, 1-5).

    Rule 1 is read as TERNARY saddles only -- `BRANCH_TOL`. No ternary saddle
    means NO BODY, and the caller reads that as infeasible. That is not a
    degenerate case to paper over, it is the paper's own criterion: "the existence
    of a ternary saddle originating from a pure component is a prerequisite for a
    feasible process" (p.84). Emitting the face saddle's bodies instead is what
    made r_min insensitive to entrainer flow -- 0.628 at both E/F = 0.10 and 0.40,
    against 1.47 and 2.00 once the face bodies stop standing in (paper: 2.042).
    Concretely, on this system the face saddle contributes four more bodies, two of
    them degenerate slivers (its arm hits the water = 0 face 0.005 away at r = 2.2),
    and the huge remaining two carried the upper junction.

    Given `sec`, each body also reports `outside_region`: whether any vertex fails
    that section's own balance `a x + bvec >= 0`. Rules 3-4 walk to the edge of
    COMPOSITION space, which for a middle section is a larger set than the
    compositions the section can hold -- for a heavy entrainer the balance reduces
    to x_E >= E/L, and on ipa/water/EG at r=1.72, E/F=1 that floor is 0.485 while
    two of the four bodies have an E-vertex at x_EG = 0.

    Reported, deliberately NOT clipped. Clipping is defensible -- those really are
    compositions no stage of this section can have, and `bvm.anchor._ray_end`
    already walks its rays that way -- but here it would hide the cause rather
    than fix it. The paper's own E1 for this system sits at roughly (0, 0.15,
    0.85), well inside the floor; ours points out of the strip because the saddle's
    eigenvector is rotated, and an eigenvector is a dK/dx quantity. This repo's
    UNIFAC fallback puts the 2-propanol/water azeotrope at x_IPA = 0.777 / 77.5 C
    against a literature 0.68 / 80.4 C -- the topology is right and the derivatives
    are not. The flag says so and becomes a no-op once Wilson binaries exist for
    the glycol pairs; a clip would keep firing forever and say nothing.

    Each body carries `saddle` / `s_sign` / `e_sign` / `id`, which is what
    `bvm.driver` anchors its march on once `winning_middle_body` has picked one.
    """
    ternary = [
        p
        for p in pinches
        if p["kind"] == "saddle" and p.get("k_gap", np.inf) <= BRANCH_TOL
    ]

    out = []
    for ch in chains(ternary, saddles_only=True):
        if not ch:
            continue
        first, last = ch[0], ch[-1]
        C = first["x"].shape[0]
        # Rules 3-4. The paper's parentheticals -- "most stable (largest
        # eigenvalue)", "most unstable (smallest eigenvalue)" -- are written in
        # the opposite sign convention to this module's, where stable is
        # |lambda| < 1 (`pinch.eigenstructure`, and the GUI draws it that way).
        # The physics is unambiguous and is what Figure 6 shows: the profile
        # ARRIVES along the stable eigendirection, so that end is S, and LEAVES
        # along the unstable one, so that end is E. `order` is |lambda|
        # descending, hence order[-1] for S and order[0] for E.
        v_start = lift_direction(
            first["eigvecs"][:, first["order"][-1]], C, first.get("drop")
        )
        v_end = lift_direction(
            last["eigvecs"][:, last["order"][0]], C, last.get("drop")
        )
        # rule 5: both directions of each -> four bodies per chain
        for s_sign in (+1.0, -1.0):
            for e_sign in (+1.0, -1.0):
                start = _to_edge(first["x"], s_sign * v_start)
                end = _to_edge(last["x"], e_sign * v_end)
                verts = _dedupe([start] + [p["x"] for p in ch] + [end])
                if len(verts) >= 2:
                    out.append(
                        {
                            "vertices": np.array(verts),
                            "pinches": [p["x"] for p in ch],
                            "kinds": [p["kind"] for p in ch],
                            "start": start,
                            "end": end,
                            "saddle": first["x"],
                            "s_sign": s_sign,
                            "e_sign": e_sign,
                            "id": body_id(first["x"], s_sign, e_sign),
                            "outside_region": _leaves_region(verts, sec),
                        }
                    )
    return out


def _leaves_region(verts, sec):
    """Does any vertex fail the section's own balance `a x + bvec >= 0`?"""
    if sec is None:
        return False
    return bool(
        any(np.min(sec.a * np.asarray(v, float) + sec.bvec) < -1e-9 for v in verts)
    )


def body_distance(A, B, witness=False):
    """Distance between two convex hulls: min ||A.lam - B.mu||, lam, mu simplices.

    A small convex QP -- the objective is a quadratic in (lam, mu) and the
    feasible set is a product of two simplices -- so a local solve is the global
    one and SLSQP is enough. Zero means the bodies intersect.

    With `witness`, also returns the argmin (lam, mu): the barycentric weights of
    the closest points on each hull. That is WHERE the two bodies meet, expressed
    in the only coordinates that survive the solve, and it is what lets a caller
    ask which facet of a body the contact lies on -- see `rank_middle_bodies`.
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    na, nb = len(A), len(B)
    if na == 0 or nb == 0:
        inf = float("inf")
        return (inf, np.zeros(na), np.zeros(nb)) if witness else inf
    if na == 1 and nb == 1:
        d = float(np.linalg.norm(A[0] - B[0]))
        return (d, np.ones(1), np.ones(1)) if witness else d

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
    cons = [
        {
            "type": "eq",
            "fun": lambda w: w[:na].sum() - 1.0,
            "jac": lambda w: np.concatenate([np.ones(na), np.zeros(nb)]),
        },
        {
            "type": "eq",
            "fun": lambda w: w[na:].sum() - 1.0,
            "jac": lambda w: np.concatenate([np.zeros(na), np.ones(nb)]),
        },
    ]
    res = minimize(
        f,
        w0,
        jac=jac,
        bounds=[(0.0, 1.0)] * (na + nb),
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-16},
    )
    d = float(np.sqrt(max(res.fun, 0.0)))
    return (d, res.x[:na], res.x[na:]) if witness else d


def sets_distance(bodies_a, bodies_b, witness=False):
    """Closest approach over every pairing of two sets of bodies.

    Returns (distance, index_a, index_b), plus the winning pair's (lam, mu) when
    `witness`. A section offers several alternative bodies (the paper's four per
    middle-section chain); the section is joinable if ANY of them meets any of
    the neighbour's, and which pair it was is what the diagram marks as the
    ACTIVE body.
    """
    best, ia, ib, wit = float("inf"), None, None, (None, None)
    for i, A in enumerate(bodies_a):
        for j, B in enumerate(bodies_b):
            got = body_distance(A["vertices"], B["vertices"], witness=witness)
            d, w = (got[0], got[1:]) if witness else (got, (None, None))
            if d < best:
                best, ia, ib, wit = d, i, j, w
    return (best, ia, ib, *wit) if witness else (best, ia, ib)


def rank_middle_bodies(top_bodies, mid_bodies, bot_bodies):
    """Every middle body, scored by how well it joins BOTH neighbours.

    Returns a list of (score, j, d_up, d_lo, i_top, i_bot, anchor_w) sorted best
    first, where score is `max(d_up, d_lo)`.

    ONE body has to do both. A middle body is the convex hull of a single profile
    polyline S -> x* -> E; a real column runs down one arm of it or turns the
    elbow at the saddle, so it cannot be two bodies at once. Scoring the two
    junctions independently let them pick different ones, which they did: at
    r = 2.2 on ipa/water/EG the upper junction ran through the body spanned by the
    face saddle and the lower through the ternary one. That is two profiles, not
    one, and it passes splits no column can do.

    `max` rather than a sum is the point: a sum lets a near-exact upper junction
    buy tolerance for a bad lower one, which is exactly the trade that produces a
    feed stage one tray above the reboiler.

    WHERE the lower junction lands is the tie-break, because the distance alone
    stops discriminating well before the ranking does: on ipa/water/EG at
    R = 1.5, E/F = 2.0 three of the four bodies score exactly 0.0000 and the
    winner was whichever came first. `anchor_w` is the closest point's
    barycentric weight on the stripping body's PRODUCT vertex, so it says which
    side of that body the middle section arrives on:

        anchor_w ~ 0  -- contact on the facet opposite x_B, i.e. the side running
                         stripping saddle -> stripping stable node. This is the
                         one to prefer: the stripping profile has already left
                         the reboiler and travelled its pinch chain by the time
                         the extractive section meets it.
        anchor_w ~ 1  -- contact back at x_B itself, which puts the feed stage on
                         top of the reboiler.

    Continuous, so it needs no tolerance of its own, and it is secondary to the
    gap: a body that does not reach both neighbours cannot buy its way up.
    """
    out = []
    for j, body in enumerate(mid_bodies):
        d_up, i_top, _ = sets_distance(top_bodies, [body])
        d_lo, _, i_bot, _, mu = sets_distance([body], bot_bodies, witness=True)
        anchor = bot_bodies[i_bot].get("anchor", 0) if i_bot is not None else None
        anchor_w = 1.0 if anchor is None else float(mu[anchor])
        out.append((max(d_up, d_lo), j, d_up, d_lo, i_top, i_bot, anchor_w))
    # Everything at or below TOUCH_TOL is TOUCHING, so clamp the primary key
    # there: without it the tie-break never fires, because two bodies that both
    # penetrate come back at 1.8e-11 and 8.4e-11 and float noise decides the
    # order before `anchor_w` is ever compared.
    out.sort(key=lambda r: (max(r[0], TOUCH_TOL), r[6], r[1]))
    return out


def winning_middle_body(top_bodies, mid_bodies, bot_bodies):
    """The single best-joining middle body: (j, d_up, d_lo, i_top, i_bot), or
    None when the section spans no body at all.

    This is RBM's answer -- the gaps ARE its verdict, and it reports which body
    they were measured on. The gap alone is still not a safe way to pick a body
    to march inside: the hull contains points no profile visits, so several
    bodies tie at zero. On ipa/water/EG at R = 1.5, E/F = 2.0 three of the four
    score 0.0000, and the one this used to return marches to an upper junction
    0.2965 wide while another zero-scoring one closes exactly. The ordering does
    not even correlate away from the tie: the body scoring WORST there (0.2983,
    its E vertex outside the section's own balance) marches to 0.0001.

    `rank_middle_bodies`' `anchor_w` breaks the tie by WHERE the lower junction
    lands rather than how wide it is, which does discriminate -- at that same
    point it moves the winner off the body that marches to 0.2965. It is still a
    hull test, so `bvm.driver` keeps using `viable_middle_bodies` to prune and
    lets the marched junction choose among what survives; the tie-break only
    decides which candidate is tried first.
    """
    ranked = rank_middle_bodies(top_bodies, mid_bodies, bot_bodies)
    if not ranked:
        return None
    _, j, d_up, d_lo, i_top, i_bot, _ = ranked[0]
    return j, d_up, d_lo, i_top, i_bot


def viable_middle_bodies(top_bodies, mid_bodies, bot_bodies, slack=TOUCH_TOL):
    """Indices of every middle body that could plausibly carry both junctions.

    The bodies PRUNE, they do not decide -- see `winning_middle_body` for the
    measurement that says why. Everything within `slack` of the best score is
    kept and handed to the caller's own (marched, expensive) test; everything
    else is a body whose hull does not even reach both neighbours, and no profile
    inside a hull reaches further than the hull does.

    That is still most of the saving. On the ipa case it takes four bodies to
    two, and each surviving body fixes a saddle and both arm signs, so the eight
    curves `bvm.anchor` used to enumerate per saddle become one per body.
    """
    ranked = rank_middle_bodies(top_bodies, mid_bodies, bot_bodies)
    if not ranked:
        return []
    cut = ranked[0][0] + float(slack)
    return [r[1] for r in ranked if r[0] <= cut]


def _demo():
    # --- body_distance is a real hull distance, checked against known answers
    seg_a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    seg_b = np.array([[0.0, 0.0, 1.0]])
    d = body_distance(seg_a, seg_b)
    assert abs(d - np.sqrt(1.5)) < 1e-6, d  # vertex to segment midpoint

    crossing = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    assert body_distance(seg_a, crossing) < 1e-7, "touching hulls -> zero"

    apart = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])
    near = np.array([[0.1, 0.1, 0.8], [0.2, 0.2, 0.6]])
    assert body_distance(apart, near) > 0.5

    # --- the walk to the simplex edge lands ON a face and stays in the simplex
    x = np.array([0.4, 0.35, 0.25])
    e = _to_edge(x, np.array([1.0, -1.0, 0.0]))
    assert abs(e.sum() - 1.0) < 1e-9 and e.min() > -1e-12, e
    assert min(e) < 1e-9, e  # reached a face

    # a walk that starts ON a face and pushes further into it must stay put, not
    # run off to `max_step`: that is what put the vertex (2.345, 0, 0.07) into an
    # extractive body and made every extractive column look feasible at any reflux
    on_face = np.array([0.6, 0.0, 0.4])
    for d in (np.array([1.0, -1.0, 0.0]), np.array([-1.0, -1.0, 2.0])):
        w = _to_edge(on_face, d)
        assert w.max() <= 1.0 + 1e-9 and w.min() > -1e-12, (d, w)

    # --- chains respect strict monotonicity and are maximal
    def pk(n, kind, x, k_gap=0.0):
        return {
            "in_simplex": True,
            "n_stable": n,
            "kind": kind,
            "x": np.array(x, float),
            "eigvals": np.array([2.0, 0.5]),
            "eigvecs": np.eye(2),
            "order": np.array([0, 1]),
            "k_gap": k_gap,
        }

    ps = [
        pk(0, "unstable_node", [1, 0, 0]),
        pk(1, "saddle", [0, 1, 0]),
        pk(1, "saddle", [0, 0.5, 0.5]),
        pk(2, "stable_node", [0, 0, 1]),
    ]
    chs = chains(ps)
    assert len(chs) == 2, chs  # the two n_stable==1 options
    for ch in chs:
        ns = [p["n_stable"] for p in ch]
        assert ns == [1, 2], ns  # unstable node dropped
        assert all(p["kind"] != "unstable_node" for p in ch)
    assert len(chains(ps, saddles_only=True)) == 2  # one saddle each

    # --- product bodies include the product composition itself
    xD = np.array([0.9, 0.1, 0.0])
    bs = product_bodies(ps, xD)
    assert bs and all(np.any(np.all(np.isclose(b["vertices"], xD), axis=1)) for b in bs)
    assert all(b["vertices"].shape[1] == 3 for b in bs)

    # --- a chain whose body edge runs through an unstable node is discarded
    prod = np.array([1.0, 0.0, 0.0])
    far = np.array([0.0, 1.0, 0.0])
    mid = np.array([0.5, 0.5, 0.0])  # on the segment
    assert blocked_by_unstable_node(prod, far, [mid])
    assert not blocked_by_unstable_node(prod, far, [np.array([0.4, 0.3, 0.3])])
    assert not blocked_by_unstable_node(prod, far, [prod, far])  # the ends
    edge = [
        pk(0, "unstable_node", mid),
        pk(1, "saddle", far),
        pk(1, "saddle", [0.85, 0.15, 0.0]),
        pk(2, "stable_node", [0, 0, 1]),
    ]
    kept = product_bodies(edge, prod)
    assert len(kept) == 1, [b["pinches"] for b in kept]
    assert not any(np.allclose(v, far) for b in kept for v in b["vertices"])

    # --- a reduced eigenvector lifts to a sum-preserving composition direction
    assert np.allclose(lift_direction(np.array([1.0, 0.0]), 3), [1.0, 0.0, -1.0])
    assert abs(lift_direction(np.array([0.3, -0.2]), 3).sum()) < 1e-12

    # --- a middle chain yields four bodies (rule 5: both directions of each end)
    sad = pk(1, "saddle", [0.3, 0.3, 0.4])
    mb = middle_bodies([sad])
    assert len(mb) == 4, len(mb)
    assert not any(b["outside_region"] for b in mb), "no section given, no verdict"
    # every body is named, and the four names are distinct: the id has to
    # identify a body, or comparing BVM's choice with RBM's says nothing
    assert len({b["id"] for b in mb}) == 4, [b["id"] for b in mb]
    assert all(b["id"] == body_id(b["saddle"], b["s_sign"], b["e_sign"]) for b in mb)

    # with the section supplied, a body reaching below its own entrainer floor is
    # FLAGGED, not clipped -- see the docstring. Floor here is x_2 >= 0.4/1.0.
    from collections import namedtuple

    S = namedtuple("S", "a bvec")
    flagged = middle_bodies([sad], S(a=1.0, bvec=np.array([0.2, 0.2, -0.4])))
    assert any(b["outside_region"] for b in flagged), [b["end"] for b in flagged]
    assert all(
        len(b["vertices"]) == len(o["vertices"]) for b, o in zip(flagged, mb)
    ), "flagging must not clip"
    for b in mb:
        assert (
            b["vertices"].min() > -1e-9 and abs(b["vertices"].sum(1) - 1).max() < 1e-6
        )

    # rules 3-4: S runs along the SMALLEST |lambda| eigendirection (the profile
    # arrives along it), E along the largest. eigvals are (2.0, 0.5) on the
    # identity basis, so S moves in (0, 1, -1) and E in (1, 0, -1).
    x0 = sad["x"]
    for b in mb:
        for pt, want in ((b["start"], [0.0, 1.0, -1.0]), (b["end"], [1.0, 0.0, -1.0])):
            d = pt - x0
            d = d / np.linalg.norm(d)
            want = np.asarray(want) / np.linalg.norm(want)
            assert abs(abs(float(d @ want)) - 1.0) < 1e-9, (pt, want)

    # rule 1 as ternary-only: a face saddle spans no body, and a section with
    # nothing but face saddles gets none at all -- no ternary saddle, no feasible
    # extractive separation (paper p.84)
    face = pk(1, "saddle", [0.6, 0.005, 0.395], k_gap=0.4)
    assert len(middle_bodies([sad, face])) == 4, "face saddle must not span a body"
    assert middle_bodies([face]) == []

    # --- sets_distance reports which pair was active
    d, i, j = sets_distance(
        [{"vertices": seg_a}], [{"vertices": seg_b}, {"vertices": crossing}]
    )
    assert j == 1 and d < 1e-7, (d, i, j)

    # --- the winner takes the WORSE of its two gaps, not their sum. Body 0 is
    # perfect above and hopeless below; body 1 is mediocre at both ends. A sum
    # picks 0 (0.00 + 0.60 < 0.20 + 0.20) and hands back a column whose lower
    # junction never closes; `max` picks 1, which is the one that can join both.
    top = [{"vertices": np.array([[1.0, 0.0, 0.0]])}]
    bot = [{"vertices": np.array([[0.0, 0.0, 1.0]])}]
    b_split = {"vertices": np.array([[1.0, 0.0, 0.0], [0.6, 0.4, 0.0]])}
    b_even = {"vertices": np.array([[0.8, 0.2, 0.0], [0.0, 0.2, 0.8]])}
    got = winning_middle_body(top, [b_split, b_even], bot)
    assert got is not None
    j, d_up, d_lo, _, _ = got
    assert j == 1, (j, d_up, d_lo)
    assert max(d_up, d_lo) < 0.5, (d_up, d_lo)
    assert winning_middle_body(top, [], bot) is None  # no ternary saddle, no body

    # --- the tie-break: WHERE the lower junction lands, when how wide it is has
    # stopped saying anything. Stripping triangle x_B - s1 - s2 with x_B recorded
    # as vertex 0; both middle bodies sit inside it, so both gaps are 0.0000 at
    # both ends and the old ranking would have taken whichever came first.
    strip = {
        "vertices": np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]  # x_B, saddle, node
        ),
        "anchor": 0,
    }
    on_pinch_side = {"vertices": np.array([[0.5, 0.5, 0.0]])}  # midpoint s1-s2
    on_anchor_side = {"vertices": np.array([[0.5, 0.0, 0.5]])}  # midpoint x_B-s2
    for mids, want in (
        ([on_anchor_side, on_pinch_side], 1),
        ([on_pinch_side, on_anchor_side], 0),  # order must not decide it
    ):
        ranked = rank_middle_bodies([strip], mids, [strip])
        assert all(r[0] < 1e-6 for r in ranked), ranked  # genuinely tied on gap
        assert ranked[0][1] == want, ranked
        assert ranked[0][6] < 1e-6 < ranked[1][6], ranked  # anchor weight 0 vs 0.5

    # --- pruning keeps every body the winner cannot be distinguished from, and
    # drops the one that plainly cannot reach both ends. Bodies tie at the top of
    # this ranking all the time (three of four do on ipa/water/EG), which is why
    # `bvm.driver` prunes here and lets the marched test decide.
    keep = viable_middle_bodies(top, [b_split, b_even], bot)
    assert keep == [1], keep
    tie = viable_middle_bodies(top, [b_even, b_even, b_split], bot)
    assert sorted(tie) == [0, 1], tie
    assert viable_middle_bodies(top, [], bot) == []

    print(
        "bvm.bodies self-check OK  hull distance, edge walk, chains, "
        f"{len(mb)} middle bodies per chain, winner by worst gap"
    )


if __name__ == "__main__":
    _demo()
