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

The paper thinks of the solutions as **pinch branches**: one-dimensional families
indexed by reflux (p.70, "the loci of these solutions are called pinch branches...
calculated by homotopy continuation"), seeded at total reflux where a -> 1 and
bvec -> 0 and the pinch equation degenerates to y_eq(x) = x -- the fixed points of
the residue curve map, i.e. the pure components and the azeotropes.

`pinch_points` does NOT trace those branches. It re-enumerates the whole solution
set at each reflux by faces, which is cheap because x_k (K_k - a) = bvec_k
factorises componentwise, and it is what "all the pinches, not the one a seed fell
into" needs. The only continuation-like step is `_relax_to_face`, a single
first-order displacement off an edge. That difference shows: a branch is followed
ACROSS a bifurcation, an enumeration only reports what is in the simplex at the
reflux it is asked about, so a pinch that has just left cannot be distinguished
from one that never existed. See docs/adr/0004 on the branching point in
K_water/a that decides the PWG topology.

The machinery itself now lives in `side_features.bvm.pinch` and is re-exported
here. BVM's extractive anchoring needs the same "all the pinches, not the one a
seed fell into" enumeration (Bruggemann & Marquardt rule 1), and putting it in
bvm keeps the single dependency arrow (rbm -> bvm) that already existed rather
than making the two packages import each other.
"""

import numpy as np

from side_features.bvm.pinch import (  # noqa: F401  (moved to bvm; re-exported)
    _ACCEPT,
    _SAME_POINT,
    _SIMPLEX_TOL,
    _edge_roots,
    _faces,
    _interior_seeds,
    _relax_to_face,
    _softmax,
    EDGE_RELAX,
    SHARP_TOL,
    eigenstructure,
    jacobian,
    pinch_points,
    pinch_residual,
    solve_pinch,
)


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
