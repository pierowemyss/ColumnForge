"""Shared plotting helpers: one set of conventions for the whole app.

Ported from freeRCM (`plot_widget.py:RCMplot`, `RCM_module_window.py`):
right-triangle ternary axes with all three simplex edges and vertex labels,
a compact matplotlib navigation toolbar, and residue-curve machinery. The
residue curves here are pure Python (dx/dxi = x - y(x) with the app's own
bubble-point VLE) — freeRCM's compiled .so core is not required.

Conventions shared by the Results tab and the BVM module:
  - stage axis: 0-based from the top (stage 0 = distillate/condenser)
  - section colours: rectifying teal, stripping orange (the original .m),
    extractive green
  - ternary projection: plot x-axis = comps[0], y-axis = comps[1];
    vertex labels comps[2] at the origin, comps[1] top, comps[0] right.
"""

import numpy as np
from core.thermodynamics import bubble_T, k_values
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar

# Section / series colours (rectifying teal + stripping orange match the .m).
RECT_C = "#218fa7"
STRIP_C = "#fb8500"
EXTRACT_C = "#2f9e44"
INTER_C = "#9c36b5"  # intermediate (multifeed) section
TEMP_C = "#fb8500"
DATA_C = "#219ebc"
BOUNDARY_C = "#d00000"


class CompactNavigationToolbar(NavigationToolbar):
    """freeRCM's toolbar: standard matplotlib navigation, no coordinate text."""

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent, coordinates=False)


# --------------------------------------------------------------------------
# Ternary (right-triangle) axes — RCMplot conventions
# --------------------------------------------------------------------------


def active_comps(x, comps, tol=1e-6):
    """Indices + names of components present anywhere in a per-stage profile. A
    species that is 0 in every feed stays ~0 (normalises to ~1e-14) on every
    stage, so on-screen it is just a flat zero line / column of noise.
    # ponytail: fixed tol cleanly splits 'not in the column' from a real trace."""
    x = np.asarray(x, float)
    keep = [j for j in range(len(comps)) if x[:, j].max() >= tol]
    return keep, [comps[j] for j in keep]


def ternary_axes(ax, comps):
    """Draw the composition simplex as a right triangle on `ax`.

    All three edges + vertex labels, axes off; x-axis carries comps[0],
    y-axis comps[1], the origin is pure comps[2] (freeRCM's RCMplot layout).
    """
    ax.plot([0, 1], [1, 0], color="#9C9C9C", linestyle="-", linewidth=1.5)
    ax.plot([0, 0], [0, 1], color="#9C9C9C", linestyle="-", linewidth=1.5)
    ax.plot([0, 1], [0, 0], color="#9C9C9C", linestyle="-", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.annotate(comps[2], [0, 0], [-0.025, -0.04], fontsize=11)
    ax.annotate(comps[1], [0, 1], [-0.02, 1.02], fontsize=11)
    ax.annotate(comps[0], [1, 0], [1.0, -0.04], fontsize=11)
    ax.annotate(
        "Generated using ColumnForge by Piero Wemyss",
        [0.55, 1.0],
        [0.55, 1.0],
        fontsize=7,
        color="tab:gray",
    )
    return ax


def composition_from_click(event):
    """Map a matplotlib click on ternary axes to a 3-component composition.

    Returns [x0, x1, x2] (comps order) or None when the click is outside the
    simplex. Use with `canvas.mpl_connect("button_press_event", ...)`.
    """
    if event.xdata is None or event.ydata is None:
        return None
    a, b = float(event.xdata), float(event.ydata)
    if a < 0.0 or b < 0.0 or a + b > 1.0:
        return None
    return np.array([a, b, 1.0 - a - b])


# --------------------------------------------------------------------------
# Residue curves: dx/dxi = x - y(x)
# --------------------------------------------------------------------------


def equilibrium_y(x, P, antoine, gamma_fn=None):
    """Bubble-point vapour in equilibrium with liquid x. Returns (y, T)."""
    x = np.asarray(x, float)
    T = bubble_T(x, P, antoine, gamma_fn=gamma_fn)
    y = k_values(T, P, antoine, gamma_fn, x) * x
    return y / y.sum(), T


def binary_equilibrium_curve(P, antoine, gamma_fn=None, n=51):
    """y-x equilibrium curve for the light component of a binary at pressure P.
    Returns (xs, ys) with xs ascending in [0, 1]."""
    xs = np.linspace(0.0, 1.0, n)
    ys = np.empty(n)
    for i, x1 in enumerate(xs):
        y, _ = equilibrium_y([x1, 1.0 - x1], P, antoine, gamma_fn)
        ys[i] = y[0]
    return xs, ys


def mccabe_thiele_steps(xeq, yeq, xD, xB, zF, R, q, max_steps=200):
    """McCabe-Thiele construction for a binary column (light component).

    xeq/yeq  ascending equilibrium curve samples (from binary_equilibrium_curve)
    xD/xB    distillate/bottoms light-component mole fractions
    zF       feed light-component mole fraction
    R        reflux ratio; q feed thermal quality

    Returns the operating-line/q-line segments, the intersection (feed pinch),
    the stage-step polyline and the stage count (steps to reach xB, the last is
    the reboiler). Pure geometry — the equilibrium curve is the only thermo in.
    """
    xeq = np.asarray(xeq, float)
    yeq = np.asarray(yeq, float)

    def y_eq(x):
        return float(np.interp(x, xeq, yeq))

    def x_eq(y):  # invert (yeq ascending in x)
        return float(np.interp(y, yeq, xeq))

    def y_rect(x):
        return R / (R + 1.0) * x + xD / (R + 1.0)

    # feed pinch = rectifying ∩ q-line. q=1 is a vertical q-line at x=zF.
    if abs(q - 1.0) < 1e-9:
        xf = zF
    else:
        m = q / (q - 1.0)  # q-line slope
        # y_rect(xf) = m*xf - zF/(q-1)  ->  solve for xf
        xf = (xD / (R + 1.0) + zF / (q - 1.0)) / (m - R / (R + 1.0))
    yf = y_rect(xf)

    def y_strip(x):  # (xB,xB) -> (xf,yf)
        return xB + (yf - xB) / (xf - xB) * (x - xB)

    def y_op(x):
        return y_rect(x) if x >= xf else y_strip(x)

    # step off stages from (xD, xD) on the 45° line
    steps = [(xD, xD)]
    y = xD
    n = 0
    for _ in range(max_steps):
        xn = x_eq(y)  # horizontal to equilibrium curve
        steps.append((xn, y))
        n += 1
        if xn <= xB:
            break
        yn = y_op(xn)  # vertical to the operating line
        steps.append((xn, yn))
        y = yn
    return {
        "xf": xf,
        "yf": yf,
        "n_stages": n,
        "rect": [(xf, yf), (xD, xD)],
        "strip": [(xB, xB), (xf, yf)],
        "qline": [(zF, zF), (xf, yf)],
        "steps": steps,
    }


def _march_residue(x0, P, antoine, gamma_fn, n_it, h, sign):
    """One direction of the residue ODE from x0 (sign=+1 toward the heavy
    node, -1 toward the light node). Midpoint (RK2) steps, renormalised."""
    xs, Ts = [], []
    x = np.asarray(x0, float).copy()
    for _ in range(n_it):
        try:
            y, T = equilibrium_y(x, P, antoine, gamma_fn)
            k1 = sign * (x - y)
            xm = np.clip(x + 0.5 * h * k1, 0.0, 1.0)
            xm /= xm.sum()
            ym, _ = equilibrium_y(xm, P, antoine, gamma_fn)
            step = sign * h * (xm - ym)
        except (ValueError, ZeroDivisionError):
            break
        xs.append(x.copy())
        Ts.append(T)
        x = np.clip(x + step, 0.0, 1.0)
        s = x.sum()
        if s <= 0.0:
            break
        x /= s
        if np.max(np.abs(step)) < 1e-7 or x.max() > 1.0 - 1e-4:
            xs.append(x.copy())
            Ts.append(Ts[-1])
            break  # pinned to a node
    return xs, Ts


def residue_curve(x0, P, antoine, gamma_fn=None, n_it=100, h=0.1):
    """Residue curve through x0, integrated both ways (light -> heavy node).

    Returns (x, T): x is (n, C) light-to-heavy along the curve, T the
    bubble-point temperature per point (monotonically increasing for a
    boundary-free region).
    """
    fwd_x, fwd_T = _march_residue(x0, P, antoine, gamma_fn, n_it, h, +1)
    bwd_x, bwd_T = _march_residue(x0, P, antoine, gamma_fn, n_it, h, -1)
    xs = bwd_x[::-1] + fwd_x[1:]
    Ts = bwd_T[::-1] + fwd_T[1:]
    if not xs:
        return np.empty((0, len(x0))), np.empty(0)
    return np.array(xs), np.array(Ts)


def residue_curve_map(P, antoine, comps, gamma_fn=None, lines=8, n_it=100, h=0.1):
    """Auto-seeded residue curves across the ternary simplex.

    Returns a list of (n, 3) composition arrays. Seeds sit on an interior
    grid so the curves fan across the whole triangle.
    """
    seeds = []
    for a in np.linspace(0.1, 0.8, 6):
        for b in np.linspace(0.1, 0.8, 6):
            if a + b <= 0.9:
                seeds.append(np.array([a, b, 1.0 - a - b]))
    step = max(1, len(seeds) // lines)
    curves = []
    for x0 in seeds[::step][:lines]:
        x, _ = residue_curve(x0, P, antoine, gamma_fn, n_it=n_it, h=h)
        if len(x) > 1:
            curves.append(x)
    return curves


def _arrow_index(x, min_step=1e-3):
    """Index k whose step x[k-1] -> x[k] is long enough to draw, nearest the
    middle of the curve.

    The middle is both where the seed sits and where consecutive points are
    furthest apart. freeRCM's RCMplot put the arrow at a flat 90% along, which
    is inside the node the curve is converging into: there the Euler steps have
    collapsed to nothing and the arrow renders as an invisible zero-length
    annotation on every well-behaved system.
    """
    mid = max(1, len(x) // 2)
    for off in range(len(x)):
        for k in (mid - off, mid + off):
            if 1 <= k < len(x) and np.max(np.abs(x[k] - x[k - 1])) >= min_step:
                return k
    return mid


def plot_residue_curves(ax, curves, linewidth=1.2, color=None, arrows=True):
    """Draw residue curves in the ternary projection with direction arrows,
    pointing along the march (light -> heavy, increasing xi)."""
    for x in curves:
        if len(x) < 2:
            continue
        (line,) = ax.plot(x[:, 0], x[:, 1], linewidth=linewidth, color=color)
        c = line.get_color()
        if arrows and len(x) > 3:
            k = _arrow_index(x)
            ax.annotate(
                "",
                xy=(x[k, 0], x[k, 1]),
                xytext=(x[k - 1, 0], x[k - 1, 1]),
                arrowprops=dict(
                    facecolor=c, edgecolor=c, shrink=0.0, headwidth=5, headlength=5
                ),
            )


# --------------------------------------------------------------------------
# Singular points (pure components + azeotropes) and boundaries
# --------------------------------------------------------------------------


def _residue_jacobian(x, P, antoine, gamma_fn, eps=1e-6):
    """Numeric Jacobian of f(x) = x - y(x) in the reduced (C-1) space."""
    C = len(x)

    def f(u):
        v = np.append(u, 1.0 - u.sum())
        y, _ = equilibrium_y(np.clip(v, 1e-12, 1.0), P, antoine, gamma_fn)
        return (v - y)[: C - 1]

    u0 = np.asarray(x[: C - 1], float)
    J = np.zeros((C - 1, C - 1))
    f0 = f(u0)
    for j in range(C - 1):
        du = np.zeros(C - 1)
        du[j] = eps
        J[:, j] = (f(u0 + du) - f0) / eps
    return J


def singular_points(P, antoine, comps, gamma_fn=None, grid=5):
    """Fixed points of the residue ODE: pure components and azeotropes.

    Solves x = y(x) from a coarse simplex grid, dedupes, and classifies each
    point by the eigenvalues of the residue Jacobian: 'stable node' (heavy
    end), 'unstable node' (light end), or 'saddle'. Returns a list of dicts
    {x, T, kind, pure}.
    """
    from scipy.optimize import fsolve

    C = len(comps)
    found = []

    def record(x):
        x = np.asarray(x, float)
        if x.min() < -1e-6 or abs(x.sum() - 1.0) > 1e-6:
            return
        x = np.clip(x, 0.0, 1.0)
        x /= x.sum()
        for p in found:
            if np.max(np.abs(p - x)) < 1e-4:
                return
        found.append(x)

    for i in range(C):  # pure components are always fixed
        e = np.zeros(C)
        e[i] = 1.0
        record(e)

    def resid(u):
        v = np.append(u, 1.0 - u.sum())
        if v.min() < -0.2 or v.max() > 1.2:
            return u * 1e3  # push fsolve back toward the simplex
        y, _ = equilibrium_y(np.clip(v, 1e-12, 1.0), P, antoine, gamma_fn)
        return (v - y)[: C - 1]

    for seed in _simplex_grid(C, grid):
        try:
            u, info, ier, _ = fsolve(resid, seed[: C - 1], full_output=True)
        except Exception:
            continue
        if ier == 1:
            record(np.append(u, 1.0 - u.sum()))

    out = []
    for x in found:
        try:
            _, T = equilibrium_y(np.clip(x, 1e-12, 1.0), P, antoine, gamma_fn)
            eig = np.linalg.eigvals(
                _residue_jacobian(np.clip(x, 1e-9, 1.0), P, antoine, gamma_fn)
            )
            re = np.real(eig)
            kind = (
                "saddle"
                if (re.max() > 1e-9 and re.min() < -1e-9)
                else "stable node" if re.max() < 0 else "unstable node"
            )
        except Exception:
            T, kind = float("nan"), "unknown"
        out.append({"x": x, "T": T, "kind": kind, "pure": bool(np.max(x) > 1.0 - 1e-4)})
    return out


def _simplex_grid(C, grid):
    """Interior composition grid seeds for the azeotrope search."""
    seeds = []
    if C == 3:
        for a in np.linspace(0.1, 0.8, grid):
            for b in np.linspace(0.1, 0.8, grid):
                if a + b <= 0.9:
                    seeds.append(np.array([a, b, 1.0 - a - b]))
    else:
        # ponytail: binary midpoint seeds for C != 3; densify if quaternary
        # azeotrope hunting ever matters.
        for i in range(C):
            for j in range(i + 1, C):
                s = np.full(C, 0.01)
                s[i] = s[j] = (1.0 - 0.01 * (C - 2)) / 2
                seeds.append(s / s.sum())
    return seeds


def distillation_boundaries(
    P, antoine, comps, gamma_fn=None, points=None, n_it=200, h=0.05
):
    """Separatrices: residue curves launched from interior saddle azeotropes.

    Ideal systems have no interior saddles, hence no boundaries — an empty
    list. Returns a list of (n, 3) composition arrays.
    """
    if points is None:
        points = singular_points(P, antoine, comps, gamma_fn)
    curves = []
    for p in points:
        if p["kind"] != "saddle" or p["pure"]:
            continue  # edge saddles: the boundary is the edge
        J = _residue_jacobian(np.clip(p["x"], 1e-9, 1.0), P, antoine, gamma_fn)
        vals, vecs = np.linalg.eig(J)
        for k in np.argsort(np.real(vals)):
            v2 = np.real(vecs[:, k])
            v = np.append(v2, -v2.sum())  # back to full simplex tangent
            v /= max(1e-12, np.linalg.norm(v))
            for sgn in (+1.0, -1.0):
                x0 = np.clip(p["x"] + 1e-3 * sgn * v, 1e-9, 1.0)
                x0 /= x0.sum()
                x, _ = residue_curve(x0, P, antoine, gamma_fn, n_it=n_it, h=h)
                if len(x) > 2:
                    curves.append(x)
    return curves


# --------------------------------------------------------------------------
# Self-check (Qt-free math paths): `PYTHONPATH=src/python python -m gui.plotting`
# --------------------------------------------------------------------------


def _demo():
    antoine = np.array(
        [  # benzene / toluene / p-xylene, mmHg/degC
            [6.90565, 1211.033, 220.79],
            [6.95464, 1344.8, 219.48],
            [6.99052, 1453.43, 215.31],
        ]
    )
    comps = ["benzene", "toluene", "xylene"]
    P = 760.0

    x, T = residue_curve([0.3, 0.3, 0.4], P, antoine)
    assert len(x) > 10
    assert np.allclose(x.sum(axis=1), 1.0, atol=1e-6)
    assert x.min() > -1e-9 and x.max() < 1 + 1e-9
    assert T[-1] > T[0], "T must increase toward the heavy node"
    assert x[0, 0] > 0.9, "backward end should approach benzene (light node)"
    assert x[-1, 2] > 0.9, "forward end should approach xylene (heavy node)"

    curves = residue_curve_map(P, antoine, comps, lines=4, n_it=60)
    assert len(curves) >= 3
    for c in curves:
        assert c.min() > -1e-9 and c.max() < 1 + 1e-9

    pts = singular_points(P, antoine, comps, grid=3)
    pure = [p for p in pts if p["pure"]]
    azeo = [p for p in pts if not p["pure"]]
    assert len(pure) == 3 and not azeo, "ideal BTX has exactly 3 pure nodes"
    kinds = {tuple(np.round(p["x"])): p["kind"] for p in pure}
    assert kinds[(1.0, 0.0, 0.0)] == "unstable node"  # benzene: light
    assert kinds[(0.0, 0.0, 1.0)] == "stable node"  # xylene: heavy
    assert kinds[(0.0, 1.0, 0.0)] == "saddle"  # toluene: middle

    assert distillation_boundaries(P, antoine, comps, points=pts) == []

    # McCabe-Thiele on a constant-relative-volatility binary (alpha=2.5)
    a = 2.5
    xe = np.linspace(0, 1, 101)
    ye = a * xe / (1 + (a - 1) * xe)
    mt = mccabe_thiele_steps(xe, ye, xD=0.95, xB=0.05, zF=0.5, R=2.0, q=1.0)
    assert 0.05 < mt["xf"] < 0.95, mt["xf"]
    assert 4 < mt["n_stages"] < 15, mt["n_stages"]  # sane stage count
    assert mt["steps"][-1][0] <= 0.05 + 1e-9  # stepped down to xB
    # q=1 puts the feed pinch on the vertical x=zF
    assert abs(mt["xf"] - 0.5) < 1e-9
    # more reflux -> fewer stages
    mt2 = mccabe_thiele_steps(xe, ye, 0.95, 0.05, 0.5, R=5.0, q=1.0)
    assert mt2["n_stages"] <= mt["n_stages"]

    class _Ev:
        xdata, ydata = 0.2, 0.3

    x0 = composition_from_click(_Ev())
    assert np.allclose(x0, [0.2, 0.3, 0.5])
    _Ev.xdata = 0.9  # outside the simplex
    assert composition_from_click(_Ev()) is None

    print(
        "plotting self-check OK "
        f"(curve {len(x)} pts, {len(curves)} map lines, {len(pts)} nodes)"
    )


if __name__ == "__main__":
    _demo()
