"""Rectification bodies and their intersection, as RBM uses them.

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

The hull is bigger than the profile polyline it spans, so intersection stays a
necessary-not-sufficient test -- that over-approximation is what RBM is, and it
is why RBM answers feasibility and reflux limits but never a stage count.

The construction itself now lives in `side_features.bvm.bodies` and is
re-exported here. BVM needs the same geometry to choose WHICH body its interior
section marches inside before it marches anything, and putting it in bvm keeps
the single dependency arrow (rbm -> bvm) that `pinch` already established rather
than making the two packages import each other. `winning_middle_body` in
particular is shared on purpose: `driver.analyze` and `bvm.driver` must not
disagree about which body is active.
"""

import numpy as np

from side_features.bvm.bodies import (  # noqa: F401  (moved to bvm; re-exported)
    BRANCH_TOL,
    TOUCH_TOL,
    _dedupe,
    _leaves_region,
    _to_edge,
    blocked_by_unstable_node,
    body_distance,
    body_id,
    chains,
    lift_direction,
    middle_bodies,
    product_bodies,
    sets_distance,
    winning_middle_body,
)


def _demo():
    """The construction is checked in `bvm.bodies`; this checks the re-export and
    the one property RBM depends on that BVM does not -- that a body's hull is a
    superset of the polyline, so a gap of zero is necessary, not sufficient."""
    seg = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert abs(body_distance(seg, np.array([[0.0, 0.0, 1.0]])) - np.sqrt(1.5)) < 1e-6

    sad = {
        "in_simplex": True, "n_stable": 1, "kind": "saddle",
        "x": np.array([0.3, 0.3, 0.4]), "eigvals": np.array([2.0, 0.5]),
        "eigvecs": np.eye(2), "order": np.array([0, 1]), "k_gap": 0.0,
    }
    mb = middle_bodies([sad])
    assert len(mb) == 4 and len({b["id"] for b in mb}) == 4

    # the hull contains points the polyline does not: two bodies can "touch" at a
    # composition neither profile ever visits, which is the over-approximation
    # RBM trades a stage count for
    body = mb[0]
    interior = body["vertices"].mean(axis=0)
    on_line = np.linalg.norm(interior - body["vertices"], axis=1).min()
    assert on_line > 1e-3, interior

    # and one body has to carry both junctions -- not one each
    top = [{"vertices": np.array([[1.0, 0.0, 0.0]])}]
    bot = [{"vertices": np.array([[0.0, 0.0, 1.0]])}]
    got = winning_middle_body(top, mb, bot)
    assert got is not None and 0 <= got[0] < len(mb), got

    print(f"rbm.bodies self-check OK  re-exports bvm.bodies, {len(mb)} middle "
          f"bodies, winner {mb[got[0]]['id']}")


if __name__ == "__main__":
    _demo()
