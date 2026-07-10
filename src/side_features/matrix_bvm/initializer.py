"""U0: a structured initial guess (blueprint Section 9).

Pipeline (each step feeds the next):

    pressure profile (from prob)
      -> operating point R, D, B resolved from the two terminal specs
      -> FUG shortcut split xD, xB   (Geddes-Hengstebeck via core.material_balance)
      -> composition profile         (BVM trajectory stepping, reused from
                                      side_features/bvm; linear-interp fallback)
      -> per-stage temperatures       (bubble points of the profile)
      -> constant-molal-overflow flows L_i, V_i
      -> component flows l = L x, v = V y  (+ zero reaction extents)

The guess only needs to sit in Newton's basin (blueprint 9.3) — it satisfies
neither equilibrium nor the energy balance exactly. LK/HK enter only as a
projection to shape monotone profiles (blueprint Section 10), never as an
equation.

Convention: stage 0 is the (partial-condenser) top stage, so v_0 is the vapour
distillate of rate D and l_0 = R D is the reflux. That differs from
core.column_solvers, so the CMO flows are built directly here.
"""

import os as _os
import sys as _sys
_HERE = _os.path.dirname(__file__)
for _p in ("../../python", ".."):        # src/python (core), src/side_features (bvm)
    _ap = _os.path.normpath(_os.path.join(_HERE, _p))
    if _ap not in _sys.path:
        _sys.path.insert(0, _ap)

import numpy as np

from core.material_balance import geddes_distribution
from residual import pack

try:
    from bvm.solver import march_rectifying, march_stripping
    _HAVE_BVM = True
except Exception:                         # pragma: no cover - fallback path
    _HAVE_BVM = False


def resolve_operating(prob):
    """Resolve (R, D, B) from the two terminal specs + overall feed.

    Ratios (reflux/boilup) don't pin flows by themselves; rate specs do. When a
    rate is absent a sensible default is used (this is only a starting point).
    Returns (R, D, B).
    """
    F = float(prob.feed.sum())
    top, bot = prob.top_spec, prob.bottom_spec

    D = None
    if top.kind == "distillate_rate":
        D = top.value
    if bot.kind == "bottoms_rate":
        B = bot.value
        D = F - B if D is None else D
    if D is None:
        D = 0.5 * F                       # default split when only ratios/purities set
    D = float(np.clip(D, 1e-6, F - 1e-6))
    B = F - D

    if top.kind == "reflux_ratio":
        R = top.value
    elif top.kind == "reflux_rate":
        R = top.value / D
    else:
        R = 2.0                           # default reflux when top spec is a rate/purity
    return float(R), float(D), float(B)


def fug_split(prob, provider, lk, hk, FR_LK=0.9, FR_HK=0.9):
    """FUG product-composition shapes via Geddes-Hengstebeck at the feed pinch."""
    F = float(prob.feed.sum())
    z = prob.feed.sum(axis=0) / F
    Pmean = float(np.mean(prob.pressure))
    Tb = provider.bubble_T(z, Pmean)
    K = provider.K(z[None, :], np.array([Tb]), np.array([Pmean]))[0]
    alpha = K / K[hk]
    xD, _, xB, _ = geddes_distribution(z, F, lk, hk, FR_LK, FR_HK, alpha)
    return xD, xB


def _bvm_profile(prob, provider, xD, xB, R, N, feed_stage):
    """Composition profile from BVM marches, resampled onto N stages.

    Rectifying march runs down from xD; stripping march up from xB; the two are
    resampled to fill the stages above/below the feed. Returns x (N, C) top->bot
    or None if the marches pinch too short to use.
    """
    if not _HAVE_BVM:
        return None
    s = R + 1.0                            # rough boilup for the stripping march
    ant = provider.antoine
    P = float(np.mean(prob.pressure))
    xr, _, _, _ = march_rectifying(xD, R, P, ant, gamma_fn=provider.gamma_fn)
    xs, _, _, _ = march_stripping(xB, s, P, ant, gamma_fn=provider.gamma_fn)
    if len(xr) < 2 or len(xs) < 2:
        return None
    n_rect = feed_stage                    # stages 0..feed_stage-1 rectifying
    n_strip = N - feed_stage
    # resample by index: rectifying top(0)->feed, stripping feed->bottom
    xi_r = np.linspace(0, len(xr) - 1, max(n_rect, 1))
    xi_s = np.linspace(len(xs) - 1, 0, max(n_strip, 1))   # xs[0] is bottoms
    prof = np.empty((N, prob.C))
    for c in range(prob.C):
        top = np.interp(xi_r, np.arange(len(xr)), xr[:, c]) if n_rect > 0 else np.empty(0)
        bottom = np.interp(xi_s, np.arange(len(xs)), xs[:, c])
        prof[:, c] = np.concatenate([top, bottom])[:N]
    return _floor_norm(prof)


def _interp_profile(xD, xB, N, feed_stage, C):
    """Linear composition interpolation xD (top) -> xB (bottom), kinked at feed."""
    prof = np.empty((N, C))
    for i in range(N):
        t = i / max(N - 1, 1)
        prof[i] = (1 - t) * xD + t * xB
    return _floor_norm(prof)


# Trace components are floored well above zero: a component-flow Newton throttles
# its whole step to fraction-to-boundary if any l_ij / v_ij starts near zero, so
# a ~1e-3 floor keeps every trace species movable.
_COMP_FLOOR = 1e-3


def _floor_norm(prof):
    prof = np.clip(np.asarray(prof, float), _COMP_FLOOR, None)
    return prof / prof.sum(axis=1, keepdims=True)


def cmo_flows(prob, R, D, feed_stage):
    """Direct CMO flows in this module's convention (v_0 = D distillate).

    L_i = R D (+ q F below feed);  V_0 = D, V_i = (R+1)D (- (1-q)F below feed).
    A starting point only — positivity-clipped, exact balance left to Newton.
    """
    N = prob.n_stages
    F = float(prob.feed.sum())
    q = 1.0                                # saturated-liquid feed assumption for init
    L = np.full(N, R * D)
    V = np.full(N, (R + 1.0) * D)
    V[0] = D
    below = np.arange(N) > feed_stage
    L[below] += q * F
    V[below] -= (1.0 - q) * F
    return np.clip(L, 1e-6, None), np.clip(V, 1e-6, None)


def initialize(prob, provider, *, lk=0, hk=1, R=None, D=None, use_bvm=True):
    """Build U0 for a Problem. Returns the packed state vector.

    lk/hk: projection keys for profile shaping (default 0,1). R/D override the
    spec-resolved operating point. use_bvm toggles BVM trajectory stepping.
    """
    N, C = prob.n_stages, prob.C
    Rr, Dd, Bb = resolve_operating(prob)
    if R is not None:
        Rr = float(R)
    if D is not None:
        Dd = float(D)
    feed_stage = int(np.argmax(prob.feed.sum(axis=1)))

    xD, xB = fug_split(prob, provider, lk, hk)
    x = None
    if use_bvm:
        x = _bvm_profile(prob, provider, xD, xB, Rr, N, feed_stage)
    if x is None:
        x = _interp_profile(xD, xB, N, feed_stage, C)

    T = np.array([provider.bubble_T(x[i], prob.pressure[i]) for i in range(N)])
    L, V = cmo_flows(prob, Rr, Dd, feed_stage)

    K = provider.K(x, T, prob.pressure)
    y = _floor_norm(K * x)
    y[0] = _floor_norm((xD / xD.sum())[None, :])[0]    # stage-0 vapour is the distillate
    l = L[:, None] * x
    v = V[:, None] * y

    Rn = prob.reactions.n_rxn if prob.reactions is not None else 0
    xi = np.zeros((N, Rn))
    return pack(l, v, T, xi if Rn else None)


def _demo():
    from thermo_adapter import FreeColumnThermo
    from problem import build_problem, OpSpec
    from residual import unpack, flows, mass_balance_residual, residual

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = FreeColumnThermo(abc)
    comps = ["benzene", "toluene", "xylene"]
    N, C = 14, 3
    prob = build_problem(
        n_stages=N, comps=comps, feeds=[(7, 100.0, [0.4, 0.35, 0.25])],
        pressure=760.0, provider=tp, top_spec=OpSpec("reflux_ratio", 4.0),
        bottom_spec=OpSpec("bottoms_rate", 60.0))

    U0 = initialize(prob, tp)
    l, v, T, xi = unpack(U0, N, C, 0)
    L, V, x, y = flows(l, v)

    # every stage is a valid composition and a hotter reboiler than condenser
    assert np.allclose(x.sum(1), 1.0, atol=1e-8)
    assert np.all(L > 0) and np.all(V > 0)
    assert T[-1] > T[0], f"reboiler {T[-1]:.1f} should exceed condenser {T[0]:.1f}"
    # light key concentrates toward the top
    assert x[0, 0] > x[-1, 0], "benzene should be richer at the top"
    # distillate rate roughly honoured by V_0
    assert abs(V[0] - 40.0) < 1e-6, V[0]     # D = F - B = 100 - 60
    # reflux ratio in the flows
    assert abs(L[0] / V[0] - 4.0) < 1e-6

    # the residual is finite and not absurd at U0 (a basin guess, not a solution)
    Res = residual(U0, prob, tp)
    assert np.all(np.isfinite(Res))

    # interp fallback path also produces a valid profile
    U0b = initialize(prob, tp, use_bvm=False)
    lb, vb, Tb, _ = unpack(U0b, N, C, 0)
    assert np.allclose((lb / lb.sum(1, keepdims=True)).sum(1), 1.0)

    print(f"initializer self-check OK (BVM march used={_HAVE_BVM}; "
          f"condenser T={T[0]:.1f}, reboiler T={T[-1]:.1f})")


if __name__ == "__main__":
    _demo()
