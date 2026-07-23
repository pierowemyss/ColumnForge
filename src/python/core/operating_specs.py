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
    # Recoveries are compared as *fractions*, not flows: a kmol/h residual would
    # sit orders of magnitude above the purity residuals and never clear the
    # shared 1e-6 gate, sending the root-find on a long hunt for precision the
    # column solve doesn't have.
    if k == SpecKind.LK_RECOVERY:
        return profile["xD"][lk] * D / (F * zF[lk]) - v   # rec = D xD_lk /(F z_lk)
    if k == SpecKind.HK_RECOVERY:
        return profile["xD"][hk] * D / (F * zF[hk]) - v
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
    # A reflux-ratio spec IS R — no need to hunt for it. Pin it and drop into the
    # 1-D branch below: halves the finite-difference cost of an implicit resolve
    # and stops the exact R residual from being mixed into a least-squares whose
    # other residual is only accurate to the column solve's noise floor.
    if fixed_R is None:
        for s in ops:
            if s.kind == SpecKind.REFLUX_RATIO:
                fixed_R = float(s.value)
                ops = [t for t in ops if t is not s]
                break
    for s in ops:
        i = lk if s.kind == SpecKind.LK_RECOVERY else hk
        if s.kind in (SpecKind.LK_RECOVERY, SpecKind.HK_RECOVERY):
            if not 0 <= i < len(zF):
                raise ValueError(
                    f"{s.kind.value}: key component index {i} is not in the feed.")
            if zF[i] <= 0:
                raise ValueError(
                    f"{s.kind.value}: the key component has no feed — its "
                    "recovery is undefined. Pick a different key.")

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
        D0 = F * zF[:max(lk + 1, 1)].sum()
        # A recovery spec says how much of its key goes overhead — fold that in
        # so the first trials already sit near the answer (a cold start 10% off
        # can wander through operating points that cost a full solver budget).
        for s in ops:
            if s.kind == SpecKind.LK_RECOVERY:
                D0 = F * (zF[:lk].sum() + float(s.value) * zF[lk])
                break
            if s.kind == SpecKind.HK_RECOVERY:
                D0 = F * (zF[:max(lk + 1, 1)].sum() + float(s.value) * zF[hk])
                break
        D0 = float(np.clip(D0, 1e-3 * F, 0.999 * (F - W)))
    Dmax = (F - W) * (1 - 1e-9)
    # Implicit specs cost a full column solve per residual evaluation, so give
    # the root-find a hard budget and stop chasing precision the solve can't
    # deliver: an unreachable target used to walk R off to infinity, one slow
    # solve at a time, hanging the caller for minutes.
    # ponytail: R_MAX=100 is well past any real column; a target that needs more
    # reflux than that is reported infeasible rather than chased. Raise it if a
    # legitimate design ever bumps into it.
    R_MAX = 100.0
    tol = 1e-6 if needs_solve else 1e-12
    kw = dict(xtol=tol, ftol=tol, max_nfev=60 if needs_solve else None)
    if needs_solve:
        # The implicit residuals are only as smooth as the column solve under
        # them: a converged solve's duty still jitters ~1% between neighbouring
        # D's near a pinch. least_squares' default relative FD step (~1e-8) then
        # differentiates pure noise — the Jacobian comes back with the wrong
        # *sign* and the trust region walks away from the answer until its budget
        # runs out (that is the "recovery spec never solves" bug). 5e-2 relative
        # (~2.5 kmol/h on a 50 kmol/h distillate) sits well above the jitter and
        # still resolves the near-linear trend. Below ~3e-2 the duty spec fails.
        kw["diff_step"] = 5e-2
    if fixed_R is not None:
        sol = least_squares(residuals, x0=[D0], bounds=([1e-9], [Dmax]), **kw)
        R, D = float(fixed_R), float(sol.x[0])
    else:
        sol = least_squares(residuals, x0=[min(R0, R_MAX), D0],
                            bounds=([1e-9, 1e-9], [R_MAX, Dmax]), **kw)
        R, D = float(sol.x[0]), float(sol.x[1])
    # Implicit residuals are fractions (purity/recovery/relative duty) and are
    # only as accurate as the column solve underneath — 1e-4 is the honest gate.
    if np.max(np.abs(sol.fun)) > (1e-4 if needs_solve else 1e-6):
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

    # Light-key recovery target, reflux fixed. A reflux spec pins R, so this is a
    # 1-D root-find: it must land on the target in a handful of column solves, not
    # burn the whole max_nfev budget (a too-small FD step on a noisy residual did
    # exactly that, and never converged).
    n_solves = [0]

    def counted(R, D):
        n_solves[0] += 1
        return solve_fn(R, D)

    R, D = resolve_operating_point(
        [Spec(SpecKind.REFLUX_RATIO, 3.0),
         Spec(SpecKind.LK_RECOVERY, 0.9)],
        F, zF, solve_fn=counted, lk=0)
    prof = solve_fn(R, D)
    rec = D * prof["xD"][0] / (F * zF[0])
    assert abs(rec - 0.9) < 2e-3, rec
    assert abs(R - 3.0) < 1e-12, R          # the reflux spec is R, exactly
    assert n_solves[0] < 40, n_solves[0]

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
