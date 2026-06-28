#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary Value Method (BVM) solver — pure-Python reference implementation.

Translated from boundValMethod.m (Piero Wemyss). Ideal vapour-liquid
equilibrium (Raoult + Antoine) for now; the VLE residuals are isolated so a
compiled activity/EOS model (nifco NRTL/SRK) can drop in later.

Scope of this slice: simple columns (rectifying + stripping) and extractive
columns (rectifying + extractive + stripping), n-component, with the two key
components projected onto a 2-D plot.

Index convention: everything here is 0-based. `lk` is the 0-based light-key
index, so the heavy key is `lk + 1`. The original MATLAB was 1-based; spots
where that translation matters are marked `# MATLAB:` for review.
"""

import numpy as np
from scipy.optimize import fsolve, brentq

# Single source of truth for the overall balance (shared with the sandbox).
from core.material_balance import matbal_recovery, matbal_direct


# --------------------------------------------------------------------------
# Thermodynamics (ideal: gamma = phi = 1). Swap these two for nifco later.
# --------------------------------------------------------------------------

def antoine_psat(T, antoine):
    """Saturation pressure for every component at temperature T.

    antoine: (N, 3) [A, B, C] with log10(Psat) = A - B/(T + C), or (N, 7) Aspen
    PLXANT (extended Antoine, ln form, T in K) — dispatched on the column count.
    Returns Psat in the same pressure unit the coefficients are fitted to.
    """
    if antoine.shape[1] == 7:
        c, Tk = antoine, T + 273.15            # PLXANT is defined in Kelvin
        lnP = (c[:, 0] + c[:, 1] / (c[:, 2] + Tk) + c[:, 3] * Tk
               + c[:, 4] * np.log(Tk) + c[:, 5] * Tk ** c[:, 6])
        return np.exp(lnP)
    A, B, C = antoine[:, 0], antoine[:, 1], antoine[:, 2]
    return 10.0 ** (A - B / (T + C))


def antoine_Tsat(P, abc):
    """Invert vapour pressure for one component's boiling point at P — initial-T
    guess only. 3-term: closed form. 7-term PLXANT: bracketed numeric solve.
    """
    if len(abc) == 7:
        from scipy.optimize import brentq
        row = np.asarray(abc, float)[None, :]
        return brentq(lambda T: antoine_psat(T, row)[0] - P, -100.0, 500.0)
    A, B, C = abc
    return B / (A - np.log10(P)) - C


def _feed_bubble_T(zF, P, antoine):
    """Ideal bubble-point temperature of the feed — reference T for relative
    volatilities (the Geddes material-balance input)."""
    f = lambda T: float(np.sum(zF * antoine_psat(T, antoine) / P) - 1.0)
    return brentq(f, -100.0, 500.0)


def _rect_residual(X, y, P, antoine):
    """Given vapour y leaving a stage, solve for liquid x and T on it.

    Unknowns X = [x_0..x_{N-1}, T]. Equations: Raoult per component + sum(x)=1.
    """
    N = len(y)
    x, T = X[:N], X[N]
    Psat = antoine_psat(T, antoine)
    raoult = x * Psat - y * P            # ideal: gamma = phi = 1
    return np.append(raoult, np.sum(x) - 1.0)


def _strip_residual(Y, x, P, antoine):
    """Given liquid x on a stage, solve for vapour y and T in equilibrium.

    Unknowns Y = [y_0..y_{N-1}, T]. Equations: Raoult per component + sum(y)=1.
    """
    N = len(x)
    y, T = Y[:N], Y[N]
    Psat = antoine_psat(T, antoine)
    raoult = x * Psat - y * P
    return np.append(raoult, np.sum(y) - 1.0)


# Material balance lives in core.material_balance (matbal_recovery /
# matbal_direct), imported above and reused here so the BVM solver and the
# sandbox auto-balance share one implementation.


# --------------------------------------------------------------------------
# Section marches
# --------------------------------------------------------------------------

def _solve_count(n_comps, efficiency):
    """Stage count per section (the .m: 20, or 40 for >=4 comps, /efficiency)."""
    n_extra = 20 if n_comps >= 4 else 0
    return max(1, int(round((20 + n_extra) / efficiency)))


def _safe(comp):
    """A march has gone unphysical if comps leave [0,1] or go non-finite."""
    return np.all(np.isfinite(comp)) and comp.min() > -1e-6 and comp.max() < 1.0 + 1e-6


def march_rectifying(xD, r, P, antoine, efficiency=1.0, nstages=None):
    """March down the rectifying section from the distillate.

    Operating line: y_{i+1} = r/(r+1) x_i + xD/(r+1).
    Returns (xRect, yRect, Trect, flags) as arrays.
    """
    xD = np.asarray(xD, float)
    N = len(xD)
    nstages = nstages or _solve_count(N, efficiency)

    y_i = xD.copy()
    T0 = antoine_Tsat(P, antoine[int(np.argmax(xD))])
    X0 = np.append(xD, T0)

    xRect, yRect, Trect, flags = [], [], [], []
    for _ in range(nstages):
        X, info, ier, _ = fsolve(_rect_residual, X0, args=(y_i, P, antoine),
                                 full_output=True)
        x_i, T_i = X[:N], X[N]
        if not _safe(x_i):
            break
        xRect.append(x_i)
        yRect.append(y_i.copy())
        Trect.append(T_i)
        flags.append(1 if ier == 1 else 0)

        y_next = (r / (r + 1)) * x_i + xD / (r + 1)
        y_next = efficiency * (y_next - y_i) + y_i          # efficiency damping
        y_i = y_next
        X0 = np.append(x_i, T_i)

    return np.array(xRect), np.array(yRect), np.array(Trect), np.array(flags)


def march_stripping(xB, s, P, antoine, efficiency=1.0, nstages=None,
                    extract=False, xE=None, E=0.0, B=0.0):
    """March up the stripping section from the bottoms.

    Operating line (simple): x_{i+1} = (y_i + xB/s) / ((s+1)/s).
    Extractive variant uses the entrainer terms, per the .m.
    Returns (xStrip, yStrip, Tstrip, flags); xStrip has one extra leading row.
    """
    xB = np.asarray(xB, float)
    N = len(xB)
    nstages = nstages or _solve_count(N, efficiency)

    x_i = xB.copy()
    # MATLAB seeds T from comps(LK_ind+1); here the heavy key is index 1 in the
    # 2-key ordering, but we use the bottoms-dominant component generally.
    T0 = antoine_Tsat(P, antoine[int(np.argmax(xB))])
    Y0 = np.append(xB, T0)

    xStrip, yStrip, Tstrip, flags = [x_i.copy()], [], [], []
    for _ in range(nstages):
        Y, info, ier, _ = fsolve(_strip_residual, Y0, args=(x_i, P, antoine),
                                 full_output=True)
        y_i, T_i = Y[:N], Y[N]
        if not _safe(y_i):
            break
        yStrip.append(y_i)
        Tstrip.append(T_i)
        flags.append(1 if ier == 1 else 0)

        if not extract:
            x_next = (y_i + xB / s) / ((s + 1) / s)
        else:
            xE = np.asarray(xE, float)
            x_next = ((y_i + ((1 + xE * E / B) / s) * xB)
                      / (((s + 1) + xE * E / B) / s))
        x_next = efficiency * (x_next - x_i) + x_i
        if not _safe(x_next):
            break
        xStrip.append(x_next)
        x_i = x_next
        Y0 = np.append(y_i, T_i)

    return np.array(xStrip), np.array(yStrip), np.array(Tstrip), np.array(flags)


# --------------------------------------------------------------------------
# Section intersection (the BVM feasibility test) + profile assembly
# --------------------------------------------------------------------------

def _interp_curve(x_lk, x_hk, mesh):
    """Interpolate HK over a uniform LK mesh. Requires LK sorted ascending."""
    order = np.argsort(x_lk)
    xs, ys = x_lk[order], x_hk[order]
    # drop duplicate LK values that would break interpolation
    keep = np.concatenate(([True], np.diff(xs) > 1e-12))
    xs, ys = xs[keep], ys[keep]
    lk_mesh = np.linspace(xs.min(), xs.max(), mesh)
    return lk_mesh, np.interp(lk_mesh, xs, ys)


def find_intersection(xRectProj, xStripProj, int_tol=1e-3, mesh=4000):
    """Find where two section profiles cross in the 2-key (LK, HK) projection.

    Each argument is an (n, 2) array of [LK, HK] along that section.
    Mirrors the .m: interpolate both onto a dense LK mesh, collect candidates
    where the HK values match within int_tol, then pick the true crossing as
    the candidate minimising |LK_rect - LK_strip|.

    Returns dict {found, point, candidates} where point is
    [LK_rect, LK_strip, HK_rect, HK_strip] or None when the sections don't
    cross (infeasible: raise reflux or change interpolation).
    """
    rL, rH = _interp_curve(xRectProj[:, 0], xRectProj[:, 1], mesh)
    sL, sH = _interp_curve(xStripProj[:, 0], xStripProj[:, 1], mesh)

    candidates = []
    for i in range(len(sL)):
        diff = np.abs(rH - sH[i])
        hits = np.where(diff < int_tol)[0]
        if hits.size:
            j = hits[-1]                       # .m uses the last match
            candidates.append([rL[j], sL[i], rH[j], sH[i]])

    if not candidates:
        return {"found": False, "point": None, "candidates": np.empty((0, 4))}

    candidates = np.array(candidates)
    best = int(np.argmin(np.abs(candidates[:, 0] - candidates[:, 1])))
    return {"found": True, "point": candidates[best], "candidates": candidates}


def _feed_indices(xRect, xStrip, point, lk, hk):
    """Locate feed stages: stage on each section nearest the crossing HK."""
    feed_rect = int(np.argmin(np.abs(xRect[:, hk] - point[2])))
    feed_strip = int(np.argmin(np.abs(xStrip[1:, hk] - point[3]))) + 1
    return feed_rect, feed_strip


def assemble_profile(xRect, yRect, Trect, xStrip, yStrip, Tstrip,
                     feed_rect, feed_strip):
    """Combine the two trimmed sections into a single bottom->top profile."""
    xR = np.flip(xRect[:feed_rect + 1], axis=0)
    yR = np.flip(yRect[:feed_rect + 1], axis=0)
    TR = np.flip(Trect[:feed_rect + 1])
    xS = xStrip[:feed_strip + 1]
    yS = yStrip[:feed_strip + 1]
    TS = Tstrip[:feed_strip + 1]

    x = np.vstack([xS, xR])
    y = np.vstack([yS, yR])
    T = np.concatenate([TS, TR])
    feed_stage = len(xS)                 # 1-based feed stage from the bottom
    return {"x": x, "y": y, "T": T, "feed_stage": feed_stage,
            "n_stages": len(x), "n_rect": feed_rect, "n_strip": feed_strip}


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def bound_val_method(zF, F, r, q, antoine, comps, *, lk=0, hk=None, P=1.0,
                     spec_mode="recovery", FR_LK=0.9, FR_HK=None, NK_spec=1e-3,
                     xD=None, xB=None, efficiency=1.0, max_stages=None,
                     extract=False, E2F=0.0, xE=None):
    """Run a BVM feasibility pass: section marches only (no intersection yet).

    lk/hk: light/heavy-key indices, any two distinct components (hk defaults to
    lk+1). FR_LK/FR_HK: key recoveries to distillate/bottoms (FR_HK defaults to
    FR_LK). Returns a dict of section profiles, balance, and boilup ratio. Call
    build_column_profile() afterwards to find the intersection and assemble the
    stage profile (kept separate so the UI can have two buttons).
    """
    zF = np.asarray(zF, float)
    antoine = np.asarray(antoine, float)
    if abs(np.sum(zF) - 1.0) > 1e-3:
        raise ValueError(f"Feed composition sums to {np.sum(zF):.4f}, not 1")
    if hk is None:
        hk = lk + 1
    if hk == lk:
        raise ValueError("light and heavy keys must be different components")

    if spec_mode == "recovery":
        # Relative volatilities at the feed bubble point feed the Geddes split.
        T_ref = _feed_bubble_T(zF, P, antoine)
        alpha = antoine_psat(T_ref, antoine)
        alpha = alpha / alpha[hk]
        xD, D, xB, B = matbal_recovery(zF, F, lk, FR_LK, NK_spec,
                                       extract=extract, E2F=E2F, xE=xE,
                                       hk=hk, FR_HK=FR_HK, alpha=alpha)
    elif spec_mode == "direct":
        if xD is None or xB is None:
            raise ValueError("direct spec_mode needs xD and xB")
        xD, D, xB, B = matbal_direct(zF, F, lk, xD, xB)
    else:
        raise ValueError(f"unknown spec_mode {spec_mode!r}")

    s = (D / B) * (r + q) - (1 - q)          # boilup ratio

    xRect, yRect, Trect, rflags = march_rectifying(
        xD, r, P, antoine, efficiency, nstages=max_stages)
    E = E2F * F
    xStrip, yStrip, Tstrip, sflags = march_stripping(
        xB, s, P, antoine, efficiency, nstages=max_stages,
        extract=extract, xE=xE, E=E, B=B)

    return {
        "xD": xD, "D": D, "xB": xB, "B": B, "s": s, "r": r, "q": q,
        "comps": comps, "lk": lk, "hk": hk,
        "xRect": xRect, "yRect": yRect, "Trect": Trect, "rect_flags": rflags,
        "xStrip": xStrip, "yStrip": yStrip, "Tstrip": Tstrip, "strip_flags": sflags,
    }


def build_column_profile(result, int_tol=1e-3, mesh=4000):
    """Find the rect/strip intersection and assemble the column profile.

    Returns dict {found, message, ...}. When sections don't cross, found is
    False with a message (infeasible: raise reflux ratio / adjust tolerance).
    """
    lk = result["lk"]
    hk = result.get("hk", lk + 1)
    xRect, xStrip = result["xRect"], result["xStrip"]
    if len(xRect) < 2 or len(xStrip) < 2:
        return {"found": False, "message": "A section failed to march; "
                "check inputs / thermo.", "candidates": np.empty((0, 4))}

    proj_r = xRect[:, [lk, hk]]
    proj_s = xStrip[:, [lk, hk]]
    inter = find_intersection(proj_r, proj_s, int_tol=int_tol, mesh=mesh)
    if not inter["found"]:
        return {"found": False, "candidates": inter["candidates"],
                "message": "Rectifying and stripping sections do not intersect "
                "— increase reflux ratio or adjust interpolation/tolerance."}

    feed_rect, feed_strip = _feed_indices(xRect, xStrip, inter["point"], lk, hk)
    profile = assemble_profile(
        result["xRect"], result["yRect"], result["Trect"],
        result["xStrip"], result["yStrip"], result["Tstrip"],
        feed_rect, feed_strip)
    profile.update({"found": True, "message": "Feasible.",
                    "intersection": inter["point"],
                    "candidates": inter["candidates"],
                    "comps": result["comps"], "lk": lk, "hk": hk})
    return profile


# --------------------------------------------------------------------------
# Self-check: ideal ternary, recovery spec. Runs with `python solver.py`.
# --------------------------------------------------------------------------

def _demo():
    # Real Antoine [A, B, C], log10(P[mmHg]) vs T[degC]: benzene / toluene / p-xylene.
    antoine = np.array([
        [6.90565, 1211.033, 220.79],
        [6.95464, 1344.8,   219.48],
        [6.99052, 1453.43,  215.31],
    ])
    zF = [0.4, 0.35, 0.25]

    def run(r):
        return bound_val_method(zF, F=100.0, r=r, q=1.0, antoine=antoine,
                                comps=["benzene", "toluene", "xylene"], lk=0,
                                P=760.0, spec_mode="recovery",
                                FR_LK=0.98, NK_spec=1e-3)

    # Balance + physicality on a representative run.
    res = run(6.0)
    assert abs(res["xD"].sum() - 1.0) < 1e-6, res["xD"].sum()
    assert abs(res["xB"].sum() - 1.0) < 1e-6, res["xB"].sum()
    assert abs(res["D"] + res["B"] - 100.0) < 1e-6
    assert res["xRect"].min() > -1e-6 and res["xRect"].max() < 1 + 1e-6
    assert res["xStrip"].min() > -1e-6 and res["xStrip"].max() < 1 + 1e-6

    # Core BVM property: low reflux is infeasible (no crossing); raising reflux
    # makes it feasible. (Feed-stage detection is discrete, so the stage count
    # is compared only between well-separated reflux values, not adjacent ones.)
    assert build_column_profile(run(0.5))["found"] is False, "r=0.5 should be infeasible"
    low_r = build_column_profile(run(4.0))
    high_r = build_column_profile(run(12.0))
    assert low_r["found"] and high_r["found"]
    assert high_r["x"].shape[1] == 3
    assert 1 <= high_r["feed_stage"] <= high_r["n_stages"]
    assert high_r["n_stages"] <= low_r["n_stages"], "more reflux should not need more stages"

    # Extractive variant: the entrainer terms fold into the stripping march
    # (single-march approximation). ponytail: the proper model is a 3-section
    # column (rectifying + a dedicated extractive section + stripping); upgrade
    # here when an extractive case needs the middle-section profile rather than
    # the folded operating line. Smoke-check: runs, stays physical, ~balanced.
    ext = bound_val_method(zF, F=100.0, r=8.0, q=1.0, antoine=antoine,
                           comps=["benzene", "toluene", "xylene"], lk=0, P=760.0,
                           spec_mode="recovery", FR_LK=0.98, NK_spec=1e-3,
                           extract=True, E2F=0.3, xE=[0.0, 0.0, 1.0])
    assert abs(ext["xB"].sum() - 1.0) < 1e-2, ext["xB"].sum()
    assert ext["xStrip"].min() > -1e-6 and ext["xStrip"].max() < 1 + 1e-6

    # Non-adjacent keys (lk=0, hk=2): feasible, balance closes, and the middle
    # component (toluene) distributes into both products — the sloppy split.
    nak = bound_val_method(zF, F=100.0, r=8.0, q=1.0, antoine=antoine,
                           comps=["benzene", "toluene", "xylene"], lk=0, hk=2,
                           P=760.0, spec_mode="recovery", FR_LK=0.98, FR_HK=0.98)
    assert nak["hk"] == 2 and abs(nak["D"] + nak["B"] - 100.0) < 1e-6
    assert nak["xD"][1] > 1e-3 and nak["xB"][1] > 1e-3, "middle key should distribute"
    assert build_column_profile(nak)["found"]

    print(f"feasible @ r=12: {high_r['n_stages']} stages, feed at "
          f"{high_r['feed_stage']} ({high_r['n_rect']} rect / {high_r['n_strip']} strip)")
    print("extractive smoke OK (approximate single-march)")
    print("self-check OK")


if __name__ == "__main__":
    _demo()
