"""RBM -- the Rectification Body Method (side module).

Bausa, von Watzdorf & Marquardt, "Shortcut Methods for Nonideal Multicomponent
Distillation: 1. Simple Columns", AIChE J. 44(10) 2181-2198 (1998), extended to
extractive columns by Bruggemann & Marquardt, "Shortcut Design of Extractive
Distillation Columns" (2002) -- the paper this package follows step for step, in
`docs/papers/rbm_bruggemann_marquardt.md`. Section references below ("paper
p.98") are to that markdown.

RBM answers a different question from BVM, which is why it is a separate module
rather than a mode of one. BVM *marches* profiles and asks whether two curves
meet; RBM never marches at all. It solves the pinch equations directly, spans the
resulting pinch points into linearised **rectification bodies**, and asks whether
those bodies intersect. The distinction matters most exactly where BVM is
weakest:

  * Two marched profiles are 1-D curves in the (C-1)-dimensional simplex, so for
    C >= 4 they generically MISS and specific product compositions have to be
    solved for. Bodies are (up to) (C-1)-dimensional, so they intersect
    generically at any C. The quaternary example in the paper is the case in
    point.
  * Marching a section through a component with a tiny K amplifies it ~1/K per
    stage, so single shooting from a product composition is exponentially
    ill-conditioned in that direction. The pinch equations are algebraic and do
    not care.
  * An extractive column has a maximum reflux as well as a minimum, and the
    feasible band closes as the entrainer flow falls (paper Fig. 9). RBM gets
    both bounds from the same body-intersection test.

What RBM does NOT give is a stage count -- bodies approximate profiles, they are
not profiles. Use it to find where a design is feasible and what it costs, and
BVM to size the column there.

KNOWN LIMIT, worth reading before trusting a verdict: RBM needs a rich pinch
structure. It gets the cleanest one from a SHARP product specification, whose
exact zeros in x_D and x_B put pinches exactly ON the simplex edges, where they
can be bracketed rather than seeded.

A SMEARED spec (the 98/2 recoveries the example files carry) has no pinch on any
edge -- every bvec_i is nonzero, so no x_i can be zero -- and every pinch is
instead displaced a hair off the edge it belongs to. Those are found by
continuing each bracketed edge root onto the parent face (`pinch._relax_to_face`)
rather than hoping a seed lands near them, which is what makes a smeared spec
usable at all: before it, every stripping section in `docs/examples/` reported
ONE pinch and spanned a body that was a straight line. Where a branch has left
the simplex entirely the face point is kept as its stand-in, flagged `clipped`,
and the panel says how many there are -- those are approximations, and a verdict
resting on one deserves a look.

The symptom to watch for either way is a section that ends up with one or two
pinches: two straight segments are a poor stand-in for two curves, and the
result is a body gap that shrinks smoothly without reaching zero. When a verdict
looks wrong, plot the pinches and count them before believing it.

Module map:
    pinch      pinch equations, branch continuation in reflux, eigenstructure
    bodies     rectification-body construction + convex-hull distance
    driver     r_min / r_max / (E/F)_min, the feasible operating region
    api        the public entry points the GUI calls

Sections, the difference-point algebra and the thermo adapter are shared with
`side_features.bvm` rather than duplicated: the operating line y = a x + bvec is
the same object in both methods, and RBM's contribution is the geometry built on
top of it.
"""

__all__ = ["pinch", "bodies", "driver", "api"]
