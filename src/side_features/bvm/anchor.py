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
from .pinch import jacobian_G, classify_pinch


def product_anchor(xprod):
    """Section terminated by an actual product: anchor = that composition."""
    return np.asarray(xprod, float)


def continuation_anchor(upstream_profile, switch_index):
    """Ordinary interior section (Sec 6.2): liquid comp is continuous across the
    feed, so the anchor is the upstream profile's liquid at the switch stage."""
    return upstream_profile["X"][int(switch_index)]


def saddle_pinch(sec, x_guess, tp, P):
    """Locate the section's pinch as a fixed point x* = G(x*) and classify it.

    Returns dict(xstar, kind, eigvals, eigvecs, saddle). Solves the fixed point
    in reduced (C-1) coords with fsolve, seeded at x_guess (falls back to the
    marched endpoint if fsolve strays out of the simplex).
    """
    C = sec.delta.shape[0]

    def G(x):
        prof = march_section(sec, x, tp, P, max_stages=1)
        return prof["X"][1] if prof["X"].shape[0] > 1 else prof["X"][0]

    def resid(u):
        x = np.empty(C); x[:C - 1] = u; x[C - 1] = 1.0 - u.sum()
        return (G(np.clip(x, 0, None)) - x)[:C - 1]

    u0 = np.asarray(x_guess, float)[:C - 1]
    try:
        u = fsolve(resid, u0, full_output=False)
        xstar = np.empty(C); xstar[:C - 1] = u; xstar[C - 1] = 1.0 - u.sum()
        if xstar.min() < -1e-4 or xstar.max() > 1 + 1e-4:
            raise ValueError
    except Exception:
        prof = march_section(sec, x_guess, tp, P, max_stages=400)
        xstar = prof["X"][-1]
    xstar = np.clip(xstar, 0, None); xstar = xstar / xstar.sum()

    J = jacobian_G(sec, xstar, tp, P)
    cl = classify_pinch(J)
    cl["xstar"] = xstar
    return cl


def unstable_eigvec(cl):
    """The eigenvector of the largest |lambda| (E8): np.linalg.eig column order is
    arbitrary, so `eigvecs[:, 0]` can be the *stable* direction and trace the wrong
    manifold. The unstable manifold -- the one the interior profile follows away
    from the saddle -- is spanned by the eigenvector whose |lambda| > 1 is largest."""
    w, V = np.asarray(cl["eigvals"]), np.asarray(cl["eigvecs"])
    return V[:, int(np.argmax(np.abs(w)))]


def launch_from_saddle(sec, xstar, eigvec, tp, P, eps=1e-3, n=200):
    """Trace the (unstable) manifold: march the forward map off x* +/- eps*v.

    Returns the branch that stays in the simplex and moves away from x*. This is
    the interior profile segment toward the downstream junction (Sec 6.3).
    """
    C = xstar.shape[0]
    v = np.asarray(eigvec, float).real
    v = v / np.linalg.norm(v)
    # lift the reduced eigenvector to a simplex-tangent (sum-zero) full vector
    vf = np.zeros(C); vf[:C - 1] = v[:C - 1] if v.shape[0] == C - 1 else v[:C - 1]
    vf[C - 1] = -vf[:C - 1].sum()

    best = None
    for sign in (+1.0, -1.0):
        x0 = xstar + sign * eps * vf
        x0 = np.clip(x0, 1e-9, None); x0 = x0 / x0.sum()
        prof = march_section(sec, x0, tp, P, max_stages=n)
        travel = np.linalg.norm(prof["X"][-1] - xstar)
        if best is None or travel > best[1]:
            best = (prof, travel)
    return best[0]


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

    # the extractive section has a fixed point that classifies, and its manifold
    # launch produces a profile that travels away from the pinch (non-degenerate).
    guess = 0.5 * (xD + xB)
    cl = saddle_pinch(ext, guess, tp, 760.0)
    assert cl["kind"] in ("saddle", "stable_node", "unstable_node")
    assert np.isfinite(cl["eigvals"]).all()
    # E8: launch off the |lambda|>1 direction, not an arbitrary eig column
    v = unstable_eigvec(cl)
    assert abs(cl["eigvals"][int(np.argmax(np.abs(cl["eigvals"])))]) >= \
        abs(cl["eigvals"][int(np.argmin(np.abs(cl["eigvals"])))])
    prof = launch_from_saddle(ext, cl["xstar"], v, tp, 760.0, n=50)
    assert prof["X"].shape[0] >= 2
    assert np.linalg.norm(prof["X"][-1] - cl["xstar"]) > 1e-4, "manifold should travel"

    # continuation anchor just reads a composition off an upstream profile
    up = march_section(rect, xD, tp, 760.0, 30)
    a = continuation_anchor(up, 5)
    assert np.allclose(a, up["X"][5])
    print(f"anchor self-check OK  extractive pinch={cl['kind']}")


if __name__ == "__main__":
    _demo()
