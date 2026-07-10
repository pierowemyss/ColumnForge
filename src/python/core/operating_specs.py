"""Resolve a column's operating specifications to its two free knobs.

Aspen's RadFrac lets you specify *any* two operating variables for a simple
column (reflux ratio, reboil ratio, distillate rate, a product purity, a key
recovery, ...) and solves for the rest. Under the constant-molar-overflow (CMO)
assumption the column has exactly two degrees of freedom, which we carry as
(R, D) = (reflux ratio L/D, distillate rate). Every operating spec is then one
of two flavours:

  * algebraic  - a closed-form constraint on (R, D) alone (all the ratio/rate
                 specs). Inverted without touching the column.
  * implicit   - depends on the *solved* product compositions (purity, key
                 recovery). Hit by re-solving the column inside a root-find.

`resolve_operating_point` builds one residual per spec and solves the 2x2 system
with a bounded least-squares (R>0, 0<D<F), re-solving the column only as often
as an implicit spec demands. The common all-algebraic pair (e.g. reflux +
distillate) converges without a single column solve.

Side draws: a SIDEDRAW_RATE spec *is* its own answer (the draw rate is given
directly), so such specs don't enter the root-find — they reduce the bottoms
by their total W, which every B-dependent residual accounts for. A column with
no condenser has no reflux: pass fixed_R=0.0 and supply exactly one remaining
operating spec (D is then the only free knob).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from scipy.optimize import least_squares

from core.dof import Spec, SpecKind, OPERATING_KINDS


# Specs whose residual needs a solved column (compositions, or — for the duty
# specs — the energy-balance duties the solve returns).
_IMPLICIT = frozenset({
    SpecKind.LK_RECOVERY, SpecKind.HK_RECOVERY,
    SpecKind.DIST_PURITY, SpecKind.BOTTOMS_PURITY,
    SpecKind.CONDENSER_DUTY, SpecKind.REBOILER_DUTY,
})


def _residual(spec: Spec, R, D, F, zF, lk, hk, profile, W=0.0):
    """Residual (=0 at the spec) for one spec at the trial (R, D).

    `profile` is the solved column dict (xD/xB arrays) or None for algebraic
    specs that don't need it. W = total side-draw rate, so B = F - D - W.
    """
    v = spec.value
    k = spec.kind
    B = F - D - W
    if k == SpecKind.REFLUX_RATIO:
        return R - v
    if k == SpecKind.REFLUX_RATE:
        return R * D - v
    if k == SpecKind.BOILUP_RATE:
        return (R + 1.0) * D - v               # CMO: vapour = (R+1)D everywhere
    if k == SpecKind.BOILUP_RATIO:
        return (R + 1.0) * D - v * B            # boilup ratio = V/B
    if k == SpecKind.DISTILLATE_RATE:
        return D - v
    if k == SpecKind.BOTTOMS_RATE:
        return B - v
    if k == SpecKind.DF_RATIO:
        return D - v * F
    if k == SpecKind.BF_RATIO:
        return B - v * F
    if k == SpecKind.LK_RECOVERY:
        return D * profile["xD"][lk] - v * F * zF[lk]   # rec = D xD_lk / (F z_lk)
    if k == SpecKind.HK_RECOVERY:
        return D * profile["xD"][hk] - v * F * zF[hk]
    if k == SpecKind.DIST_PURITY:
        return profile["xD"][spec.component] - v
    if k == SpecKind.BOTTOMS_PURITY:
        return profile["xB"][spec.component] - v
    # Duty specs: hit the energy balance's real duties (same unit as spec.value;
    # scaled by 1/|v| so a MJ/h-scale residual sits near the purity residuals'
    # magnitude for the shared least-squares tolerance).
    if k == SpecKind.CONDENSER_DUTY:
        return (profile["condenser_duty"] - v) / (abs(v) or 1.0)
    if k == SpecKind.REBOILER_DUTY:
        return (profile["reboiler_duty"] - v) / (abs(v) or 1.0)
    raise ValueError(f"{k} is not an operating spec the resolver handles")


def resolve_operating_point(
    specs: List[Spec], F: float, zF, *,
    solve_fn: Optional[Callable[[float, float], dict]] = None,
    lk: int = 0, hk: Optional[int] = None,
    R0: float = 2.0, D0: Optional[float] = None,
    side_draw_total: float = 0.0, fixed_R: Optional[float] = None,
) -> tuple:
    """Return (R, D) satisfying the operating specs.

    solve_fn(R, D) -> profile dict with "xD"/"xB" — required only when an
    implicit (purity/recovery) spec is present.

    side_draw_total: sum of all side-draw rates (SIDEDRAW_RATE specs are
    given values, not free knobs — they arrive here as this total and shift
    B = F - D - W in every B-dependent residual). SIDEDRAW_RATE specs in
    `specs` are ignored for the root-find.
    fixed_R: pin the reflux ratio (0.0 for a condenser-less column); then
    exactly one remaining spec is required and D is the only free knob.
    """
    W = float(side_draw_total)
    ops = [s for s in specs
           if s.kind in OPERATING_KINDS and s.kind != SpecKind.SIDEDRAW_RATE]
    n_free = 1 if fixed_R is not None else 2
    if len(ops) != n_free:
        raise ValueError(
            f"This column needs exactly {n_free} operating spec(s) besides "
            f"side-draw rates; got {len(ops)}.")
    zF = np.asarray(zF, float)
    if hk is None:
        hk = lk + 1
    needs_solve = any(s.kind in _IMPLICIT for s in ops)
    if needs_solve and solve_fn is None:
        raise ValueError("Purity/recovery specs require a solve_fn.")

    # Cache the column solve so two implicit specs share one solve per (R, D).
    cache: dict = {}

    def profile_at(R, D):
        key = (round(R, 12), round(D, 12))
        if key not in cache:
            cache[key] = solve_fn(R, D)
        return cache[key]

    def residuals(theta):
        R, D = (fixed_R, theta[0]) if fixed_R is not None else theta
        prof = profile_at(R, D) if needs_solve else None
        return [_residual(s, R, D, F, zF, lk, hk, prof, W) for s in ops]

    if D0 is None:
        # A light-key-ish guess keeps the implicit root-find in a sane basin.
        D0 = float(np.clip(F * zF[:max(lk + 1, 1)].sum(), 1e-3 * F,
                           0.999 * (F - W)))
    Dmax = (F - W) * (1 - 1e-9)
    if fixed_R is not None:
        sol = least_squares(residuals, x0=[D0], bounds=([1e-9], [Dmax]),
                            xtol=1e-12, ftol=1e-12)
        R, D = float(fixed_R), float(sol.x[0])
    else:
        sol = least_squares(residuals, x0=[R0, D0],
                            bounds=([1e-9, 1e-9], [np.inf, Dmax]),
                            xtol=1e-12, ftol=1e-12)
        R, D = float(sol.x[0]), float(sol.x[1])
    if np.max(np.abs(sol.fun)) > 1e-6:
        raise ValueError(
            "Could not satisfy the operating specs — they may be infeasible for "
            f"this column (residual {np.max(np.abs(sol.fun)):.3g}).")
    return R, D


def _demo():
    # ---- algebraic pairs invert exactly, no column solve ----
    F = 100.0
    zF = np.array([0.5, 0.3, 0.2])

    R, D = resolve_operating_point(
        [Spec(SpecKind.REFLUX_RATIO, 3.0), Spec(SpecKind.DISTILLATE_RATE, 40.0)],
        F, zF)
    assert abs(R - 3.0) < 1e-6 and abs(D - 40.0) < 1e-6, (R, D)

    # reflux + bottoms rate  -> D = F - B
    R, D = resolve_operating_point(
        [Spec(SpecKind.REFLUX_RATIO, 2.0), Spec(SpecKind.BOTTOMS_RATE, 70.0)],
        F, zF)
    assert abs(D - 30.0) < 1e-6, D

    # boilup ratio + distillate: (R+1)D = BR*(F-D)
    BR, Dv = 1.5, 40.0
    R, D = resolve_operating_point(
        [Spec(SpecKind.BOILUP_RATIO, BR), Spec(SpecKind.DISTILLATE_RATE, Dv)],
        F, zF)
    assert abs(D - Dv) < 1e-6 and abs((R + 1) * D - BR * (F - D)) < 1e-6, (R, D)

    # D:F + reflux rate
    R, D = resolve_operating_point(
        [Spec(SpecKind.DF_RATIO, 0.45), Spec(SpecKind.REFLUX_RATE, 90.0)],
        F, zF)
    assert abs(D - 45.0) < 1e-6 and abs(R * D - 90.0) < 1e-6, (R, D)

    # ---- implicit spec: a purity target hit by re-solving the column ----
    from core.column_solvers import solve_bubble_point
    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    comps = ["benzene", "toluene", "xylene"]

    def solve_fn(R, D):
        return solve_bubble_point(zF, F, abc, comps, N=20, feed_stage=10,
                                  R=R, D=D, P=760.0)

    # Fix reflux, vary distillate to hit 95% benzene in the distillate.
    target = 0.95
    R, D = resolve_operating_point(
        [Spec(SpecKind.REFLUX_RATIO, 3.0),
         Spec(SpecKind.DIST_PURITY, target, component=0)],
        F, zF, solve_fn=solve_fn)
    prof = solve_fn(R, D)
    assert abs(prof["xD"][0] - target) < 2e-3, prof["xD"][0]
    assert abs(R - 3.0) < 1e-6

    # Light-key recovery target, reflux fixed.
    R, D = resolve_operating_point(
        [Spec(SpecKind.REFLUX_RATIO, 3.0),
         Spec(SpecKind.LK_RECOVERY, 0.9)],
        F, zF, solve_fn=solve_fn, lk=0)
    prof = solve_fn(R, D)
    rec = D * prof["xD"][0] / (F * zF[0])
    assert abs(rec - 0.9) < 2e-3, rec

    # Duty spec: fix reflux, solve for the D that hits a target reboiler duty.
    # Feasible-by-construction — read the duty at a known D, then recover it.
    from core.column_solvers import solve_inside_out, make_energy_balance
    cp_l = np.array([136.0, 157.0, 186.0]); hv_tb = np.array([30.8, 33.2, 36.2])
    tb_k = np.array([353.2, 383.8, 417.6]); tc_k = np.array([562.0, 591.8, 630.3])

    def solve_eb(R, D):
        return solve_inside_out(zF, F, abc, comps, N=20, feed_stage=10, R=R, D=D,
                                P=760.0, max_iter=80,
                                flows_hook=make_energy_balance(cp_l, hv_tb, tb_k, tc_k))

    Qr_target = solve_eb(3.0, 42.0)["reboiler_duty"]
    R, D = resolve_operating_point(
        [Spec(SpecKind.REFLUX_RATIO, 3.0),
         Spec(SpecKind.REBOILER_DUTY, Qr_target)],
        F, zF, solve_fn=solve_eb)
    assert abs(D - 42.0) < 0.2, D
    assert abs(solve_eb(R, D)["reboiler_duty"] - Qr_target) < 1e-3 * abs(Qr_target)

    # Side draws shift B: reflux + bottoms rate with W=20 -> D = F - W - B.
    R, D = resolve_operating_point(
        [Spec(SpecKind.REFLUX_RATIO, 2.0), Spec(SpecKind.BOTTOMS_RATE, 50.0),
         Spec(SpecKind.SIDEDRAW_RATE, 20.0, "S1")],
        F, zF, side_draw_total=20.0)
    assert abs(D - 30.0) < 1e-6, D

    # Condenser-less column: R pinned to 0, one spec resolves D alone.
    R, D = resolve_operating_point(
        [Spec(SpecKind.DISTILLATE_RATE, 25.0)], F, zF, fixed_R=0.0)
    assert R == 0.0 and abs(D - 25.0) < 1e-6, (R, D)

    # Infeasible spec is reported, not silently mangled.
    try:
        resolve_operating_point(
            [Spec(SpecKind.DISTILLATE_RATE, 40.0),
             Spec(SpecKind.BOTTOMS_RATE, 40.0)],   # implies F=80, not 100
            F, zF)
    except ValueError:
        pass
    else:
        raise AssertionError("infeasible D+B pair should raise")

    print("operating_specs self-check OK")


if __name__ == "__main__":
    _demo()
