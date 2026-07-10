#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary Value Method (BVM) solver — pure-Python reference implementation.

Translated from boundValMethod.m (Piero Wemyss). Ideal vapour-liquid
equilibrium (Raoult + Antoine) for now; the VLE residuals are isolated so a
compiled activity/EOS model (nifco NRTL/SRK) can drop in later.

A column is an ordered list of `Section`s, top -> bottom. Each section is
marched from a product composition (rectifying: xD down; stripping: xB up) or
from a linking pinch (the extractive middle section starts from the rectifying
liquid mixed with the entrainer feed). Adjacent sections are joined where
their liquid profiles cross in the [lk, hk] projection — the BVM feasibility
test. Simple columns are [rectifying, stripping]; extractive columns are
[rectifying, extractive, stripping]; the same linking machinery generalises
to N sections.

Index convention: everything here is 0-based. `lk`/`hk` are the 0-based
light/heavy-key indices (hk defaults to lk + 1). Assembled profiles run
top -> bottom with row 0 = distillate, matching the rest of the app.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

# Single sources of truth, shared with the rigorous solvers and the sandbox.
from core.material_balance import matbal_recovery, matbal_direct
from core.thermodynamics import antoine_psat, antoine_Tsat, bubble_T


# --------------------------------------------------------------------------
# VLE residuals (ideal: gamma = phi = 1). Swap for nifco later.
# --------------------------------------------------------------------------

def _gamma(gamma_fn, x, T, N):
    """Activity coefficients gamma(x, T); ideal (all-ones) when no model given.
    This is the one VLE non-ideality seam — the same gamma_fn closure the
    rigorous solvers use (NRTL today; a future nifco fugacity/activity model
    drops in here without touching the marches)."""
    if gamma_fn is None:
        return np.ones(N)
    return np.asarray(gamma_fn(np.asarray(x, float), float(T)), float)


def _rect_residual(X, y, P, antoine, gamma_fn=None):
    """Given vapour y leaving a stage, solve for liquid x and T on it.

    Unknowns X = [x_0..x_{N-1}, T]. Modified Raoult's law per component
    (x_i gamma_i Psat_i = y_i P) + sum(x)=1. gamma_fn=None => ideal (gamma=1).
    """
    N = len(y)
    x, T = X[:N], X[N]
    Psat = antoine_psat(T, antoine)
    raoult = x * _gamma(gamma_fn, x, T, N) * Psat - y * P
    return np.append(raoult, np.sum(x) - 1.0)


def _strip_residual(Y, x, P, antoine, gamma_fn=None):
    """Given liquid x on a stage, solve for vapour y and T in equilibrium.

    Unknowns Y = [y_0..y_{N-1}, T]. Modified Raoult's law + sum(y)=1.
    """
    N = len(x)
    y, T = Y[:N], Y[N]
    Psat = antoine_psat(T, antoine)
    raoult = x * _gamma(gamma_fn, x, T, N) * Psat - y * P
    return np.append(raoult, np.sum(y) - 1.0)


# --------------------------------------------------------------------------
# Section marches
# --------------------------------------------------------------------------

# A march stops at a pinch (composition step below _PINCH_TOL) or after
# _MAX_STAGES stages, whichever comes first — no more hardcoded 20/40 caps.
_PINCH_TOL = 1e-8
_MAX_STAGES = 200


@dataclass
class Section:
    """One marched column section, in march order (rectifying/extractive:
    top -> down from their start; stripping: bottom -> up from xB)."""
    name: str
    x: np.ndarray       # (n, C) liquid composition per stage
    y: np.ndarray       # (n, C) or (n-1, C) vapour composition
    T: np.ndarray
    flags: np.ndarray   # fsolve convergence flag per stage (1 = converged)

    def proj(self, lk, hk):
        """This section's liquid profile in the 2-key [lk, hk] projection."""
        return self.x[:, [lk, hk]]


def _n_stages(nstages, efficiency):
    return nstages if nstages else max(1, int(round(_MAX_STAGES / efficiency)))


def _safe(comp):
    """A march has gone unphysical if comps leave [0,1] or go non-finite."""
    return np.all(np.isfinite(comp)) and comp.min() > -1e-6 and comp.max() < 1.0 + 1e-6


def march_rectifying(xD, r, P, antoine, efficiency=1.0, nstages=None,
                     gamma_fn=None):
    """March down the rectifying section from the distillate.

    Operating line: y_{i+1} = r/(r+1) x_i + xD/(r+1).
    Returns (xRect, yRect, Trect, flags) as arrays.
    """
    xD = np.asarray(xD, float)
    N = len(xD)
    nstages = _n_stages(nstages, efficiency)

    y_i = xD.copy()
    T0 = antoine_Tsat(P, antoine[int(np.argmax(xD))])
    X0 = np.append(xD, T0)

    xRect, yRect, Trect, flags = [], [], [], []
    for _ in range(nstages):
        X, info, ier, _ = fsolve(_rect_residual, X0, args=(y_i, P, antoine, gamma_fn),
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
        if np.max(np.abs(y_next - y_i)) < _PINCH_TOL:
            break                                           # pinched
        y_i = y_next
        X0 = np.append(x_i, T_i)

    return np.array(xRect), np.array(yRect), np.array(Trect), np.array(flags)


def march_stripping(xB, s, P, antoine, efficiency=1.0, nstages=None,
                    gamma_fn=None):
    """March up the stripping section from the bottoms.

    Operating line: x_{i+1} = (y_i + xB/s) / ((s+1)/s).
    Returns (xStrip, yStrip, Tstrip, flags); xStrip has one extra leading row
    (the bottoms itself).
    """
    xB = np.asarray(xB, float)
    N = len(xB)
    nstages = _n_stages(nstages, efficiency)

    x_i = xB.copy()
    T0 = antoine_Tsat(P, antoine[int(np.argmax(xB))])
    Y0 = np.append(xB, T0)

    xStrip, yStrip, Tstrip, flags = [x_i.copy()], [], [], []
    for _ in range(nstages):
        Y, info, ier, _ = fsolve(_strip_residual, Y0, args=(x_i, P, antoine, gamma_fn),
                                 full_output=True)
        y_i, T_i = Y[:N], Y[N]
        if not _safe(y_i):
            break
        yStrip.append(y_i)
        Tstrip.append(T_i)
        flags.append(1 if ier == 1 else 0)

        x_next = (y_i + xB / s) / ((s + 1) / s)
        x_next = efficiency * (x_next - x_i) + x_i
        if not _safe(x_next):
            break
        xStrip.append(x_next)
        if np.max(np.abs(x_next - x_i)) < _PINCH_TOL:
            break                                           # pinched
        x_i = x_next
        Y0 = np.append(y_i, T_i)

    return np.array(xStrip), np.array(yStrip), np.array(Tstrip), np.array(flags)


def march_extractive(x_start, r, D, E, xE, xD, P, antoine,
                     efficiency=1.0, nstages=None, gamma_fn=None):
    """March the extractive middle section down from below the entrainer feed.

    CMO balances (Levy / Van Dongen / Doherty), entrainer fed as saturated
    liquid above the main feed:

        L_m = r*D + E        V_m = (r + 1)*D
        V_m * y_{n+1} = L_m * x_n + D*xD - E*xE

    x_start is the liquid leaving the entrainer stage (rectifying liquid mixed
    with the entrainer). Returns (xExtract, yExtract, Textract, flags).
    """
    x_start = np.asarray(x_start, float)
    xD = np.asarray(xD, float)
    xE = np.asarray(xE, float)
    N = len(x_start)
    nstages = _n_stages(nstages, efficiency)

    L_m = r * D + E
    V_m = (r + 1.0) * D

    x_i = x_start.copy()
    T0 = antoine_Tsat(P, antoine[int(np.argmax(x_start))])

    xM, yM, TM, flags = [], [], [], []
    for _ in range(nstages):
        y_next = (L_m * x_i + D * xD - E * xE) / V_m
        if not _safe(y_next):
            break
        X, info, ier, _ = fsolve(_rect_residual, np.append(x_i, T0),
                                 args=(y_next, P, antoine, gamma_fn),
                                 full_output=True)
        x_next, T_i = X[:N], X[N]
        if not _safe(x_next):
            break
        xM.append(x_next)
        yM.append(y_next)
        TM.append(T_i)
        flags.append(1 if ier == 1 else 0)
        if np.max(np.abs(x_next - x_i)) < _PINCH_TOL:
            break                                           # pinched
        x_i = x_next
        T0 = T_i

    return np.array(xM), np.array(yM), np.array(TM), np.array(flags)


# --------------------------------------------------------------------------
# Section linking (the BVM feasibility test) + profile assembly
# --------------------------------------------------------------------------

def _cross_all(A, B):
    """All polyline crossings between A (n,2) and B (m,2), vectorised.

    Returns (idx, tA) where idx is a (k, 2) array of (segment_in_A,
    segment_in_B) pairs and tA the (n-1, m-1) parameter matrix along A.
    """
    p0, p1 = A[:-1], A[1:]
    q0, q1 = B[:-1], B[1:]
    d1, d2 = p1 - p0, q1 - q0
    den = d1[:, None, 0] * d2[None, :, 1] - d1[:, None, 1] * d2[None, :, 0]
    r0 = q0[None, :, :] - p0[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (r0[..., 0] * d2[None, :, 1] - r0[..., 1] * d2[None, :, 0]) / den
        u = (r0[..., 0] * d1[:, None, 1] - r0[..., 1] * d1[:, None, 0]) / den
    ok = (np.isfinite(t) & np.isfinite(u)
          & (t >= -1e-9) & (t <= 1 + 1e-9)
          & (u >= -1e-9) & (u <= 1 + 1e-9))
    return np.argwhere(ok), t


def find_intersection(projA, projB, int_tol=1e-3, mesh=None):
    """Where two section profiles cross in a 2-key [LK, HK] projection.

    Exact polyline segment intersection (replaces the old dense-mesh
    `hits[-1]` heuristic, which was unsafe for non-monotonic profiles). When
    no strict crossing exists, profiles that pinch onto each other within
    `int_tol` still count. `mesh` is accepted for backward compatibility and
    ignored.

    Returns dict {found, point, i_a, i_b, candidates}: point is
    [LK_a, LK_b, HK_a, HK_b] (a == b at an exact crossing); i_a/i_b index the
    stage on each profile at the link. Among several crossings the one using
    the fewest total stages wins.
    """
    A = np.asarray(projA, float)
    B = np.asarray(projB, float)
    if len(A) < 2 or len(B) < 2:
        return {"found": False, "point": None, "i_a": None, "i_b": None,
                "candidates": np.empty((0, 4))}

    idx, t = _cross_all(A, B)
    cands = []
    for i, j in idx:
        p = A[i] + t[i, j] * (A[i + 1] - A[i])
        # stage index = the bracketing stage nearer the crossing
        ia = i if t[i, j] <= 0.5 else i + 1
        ib = j if np.linalg.norm(A[i] + t[i, j] * (A[i + 1] - A[i]) - B[j]) <= \
            np.linalg.norm(p - B[j + 1]) else j + 1
        cands.append((int(ia), int(ib), p[0], p[0], p[1], p[1]))

    if not cands:
        # near-miss: closest approach between the two point sets
        d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        if d[i, j] < int_tol:
            cands.append((int(i), int(j), A[i, 0], B[j, 0], A[i, 1], B[j, 1]))

    if not cands:
        return {"found": False, "point": None, "i_a": None, "i_b": None,
                "candidates": np.empty((0, 4))}

    cands.sort(key=lambda c: c[0] + c[1])
    ia, ib, *pt = cands[0]
    return {"found": True, "point": np.array(pt), "i_a": ia, "i_b": ib,
            "candidates": np.array([c[2:] for c in cands])}


def link_sections(upper, lower, lk, hk, int_tol=1e-3):
    """Join two adjacent sections where their profiles cross in [lk, hk].

    `upper` is marched downward (rectifying/extractive), `lower` upward
    (stripping) or downward (another middle section) — the crossing test is
    direction-agnostic. Returns find_intersection's dict; i_a indexes
    upper.x, i_b indexes lower.x.
    """
    return find_intersection(upper.proj(lk, hk), lower.proj(lk, hk),
                             int_tol=int_tol)


def assemble_profile(sections, links):
    """Stack linked sections into one top -> bottom profile.

    sections: ordered top -> bottom, e.g. [rect, strip] or
    [rect, extract, strip]. links: for each adjacent pair (top-down order),
    the trim index into each section's march ((i_upper, i_lower)); the last
    section pair's lower index trims the stripping march. Row 0 of the result
    is the distillate.
    """
    # trim each section: first by the link below it (its own march), then by
    # the link above it (the upper neighbour's march index)
    trims = []
    for k, sec in enumerate(sections):
        start = 0
        stop = len(sec.x)
        if k < len(links):                 # link to the section below
            stop = links[k][0] + 1
        if k > 0:                          # link to the section above
            stop = min(stop, len(sec.x))
        trims.append((start, stop))
    # the bottom section is trimmed by the last link's lower index
    if links:
        trims[-1] = (0, links[-1][1] + 1)

    xs, ys, Ts, breaks = [], [], [], []
    count = 0
    for sec, (a, b) in zip(reversed(sections), reversed(trims)):
        n = max(0, b - a)
        n_y = min(n, len(sec.y))
        if sec.name == "stripping":
            xs.append(sec.x[a:a + n_y])    # keep x/y/T row counts equal
            ys.append(sec.y[a:a + n_y])
            Ts.append(sec.T[a:a + n_y])
            count += n_y
        else:                              # marched top-down: flip
            xs.append(np.flip(sec.x[a:b], axis=0))
            ys.append(np.flip(sec.y[a:b], axis=0))
            Ts.append(np.flip(sec.T[a:b]))
            count += n
        breaks.append(count)

    # assembled bottoms-first; flip to the app convention (row 0 = distillate).
    x = np.vstack(xs)[::-1]
    y = np.vstack(ys)[::-1]
    T = np.concatenate(Ts)[::-1]
    M = len(x)
    feed_stage = M - breaks[0]             # top of the stripping section, from top
    out = {"x": x, "y": y, "T": T, "feed_stage": feed_stage,
           "n_stages": M, "section_breaks": [M - b for b in breaks[:-1]]}
    if len(sections) == 3:                 # rect / extract / strip
        out["entrainer_stage"] = M - breaks[1]
        out["n_extract"] = breaks[1] - breaks[0]
    out["n_strip"] = breaks[0]
    out["n_rect"] = M - (breaks[-2] if len(breaks) > 1 else 0)
    return out


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def bound_val_method(zF, F, r, q, antoine, comps, *, lk=0, hk=None, P=1.0,
                     spec_mode="recovery", FR_LK=0.9, FR_HK=None, NK_spec=1e-3,
                     xD=None, xB=None, efficiency=1.0, max_stages=None,
                     extract=False, E2F=0.0, xE=None, gamma_fn=None):
    """Run a BVM feasibility pass: the section marches.

    lk/hk: light/heavy-key indices, any two distinct components (hk defaults
    to lk+1). FR_LK/FR_HK: key recoveries to distillate/bottoms (FR_HK
    defaults to FR_LK). With extract=True a true 3-section column is marched:
    rectifying, a dedicated extractive middle section (entrainer balance,
    seeded from the rectifying profile), and stripping. Returns a dict of
    section profiles (including the ordered `sections` list), balance, and
    boilup ratio. Call build_column_profile() afterwards to link the sections
    and assemble the stage profile (kept separate so the UI can have two
    buttons).
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
        # With an activity model, K_i = gamma_i Psat_i / P, so gamma enters alpha.
        T_ref = bubble_T(zF, P, antoine, gamma_fn=gamma_fn)
        alpha = antoine_psat(T_ref, antoine)
        if gamma_fn is not None:
            alpha = alpha * _gamma(gamma_fn, zF, T_ref, len(zF))
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
        xD, r, P, antoine, efficiency, nstages=max_stages, gamma_fn=gamma_fn)
    xStrip, yStrip, Tstrip, sflags = march_stripping(
        xB, s, P, antoine, efficiency, nstages=max_stages, gamma_fn=gamma_fn)

    rect = Section("rectifying", xRect, yRect, Trect, rflags)
    strip = Section("stripping", xStrip, yStrip, Tstrip, sflags)
    sections = [rect, strip]

    result = {
        "xD": xD, "D": D, "xB": xB, "B": B, "s": s, "r": r, "q": q,
        "comps": comps, "lk": lk, "hk": hk, "extract": bool(extract),
        "xRect": xRect, "yRect": yRect, "Trect": Trect, "rect_flags": rflags,
        "xStrip": xStrip, "yStrip": yStrip, "Tstrip": Tstrip,
        "strip_flags": sflags,
    }

    if extract:
        if xE is None:
            raise ValueError("extract=True needs entrainer composition xE")
        xE = np.asarray(xE, float)
        E = E2F * F
        LD = r * D
        strip_proj = strip.proj(lk, hk)
        best = None
        # Seed the middle section from a rectifying stage, top-down; the first
        # seed whose extractive march crosses the stripping profile wins.
        # ponytail: cap the seed search at 30 rectifying stages; scan all of
        # them if deep entrainer feeds ever matter.
        for k in range(min(len(xRect), 30)):
            x_start = (LD * xRect[k] + E * xE) / (LD + E)
            xM, yM, TM, mflags = march_extractive(
                x_start, r, D, E, xE, xD, P, antoine, efficiency,
                nstages=max_stages, gamma_fn=gamma_fn)
            if len(xM) < 2:
                continue
            if best is None:
                best = (k, xM, yM, TM, mflags)   # fallback: first marchable
            if find_intersection(xM[:, [lk, hk]], strip_proj)["found"]:
                best = (k, xM, yM, TM, mflags)
                break
        if best is None:
            raise ValueError("extractive middle section failed to march from "
                             "any rectifying seed; check entrainer spec")
        k, xM, yM, TM, mflags = best
        extract_sec = Section("extractive", xM, yM, TM, mflags)
        sections = [rect, extract_sec, strip]
        result.update({
            "xExtract": xM, "yExtract": yM, "Textract": TM,
            "extract_flags": mflags, "entrainer_stage_rect": k,
            "E": E, "xE": xE,
        })

    result["sections"] = sections
    return result


def build_column_profile(result, int_tol=1e-3, mesh=4000):
    """Link the marched sections and assemble the column profile.

    Returns dict {found, message, ...}. When adjacent sections don't cross,
    found is False with a message (infeasible: raise reflux ratio / adjust
    tolerance). For extractive runs the rectifying/extractive link is the
    entrainer feed stage (a linking pinch fixed by the seed search); the
    extractive/stripping link — like rectifying/stripping in a simple column —
    is the profile intersection.
    """
    lk = result["lk"]
    hk = result.get("hk", lk + 1)
    sections = result.get("sections")
    if sections is None:                       # legacy dicts without sections
        sections = [
            Section("rectifying", result["xRect"], result["yRect"],
                    result["Trect"], result.get("rect_flags", np.array([]))),
            Section("stripping", result["xStrip"], result["yStrip"],
                    result["Tstrip"], result.get("strip_flags", np.array([]))),
        ]
    if any(len(sec.x) < 2 for sec in sections):
        return {"found": False, "message": "A section failed to march; "
                "check inputs / thermo.", "candidates": np.empty((0, 4))}

    # The bottom link (into the stripping section) is the feasibility test.
    upper = sections[-2]
    inter = link_sections(upper, sections[-1], lk, hk, int_tol=int_tol)
    if not inter["found"]:
        what = ("Extractive and stripping" if upper.name == "extractive"
                else "Rectifying and stripping")
        return {"found": False, "candidates": inter["candidates"],
                "message": f"{what} sections do not intersect — increase "
                "reflux ratio or adjust interpolation/tolerance."}
    # stripping profile row 0 is the bottoms itself; never link below stage 1
    i_lower = max(1, min(inter["i_b"], len(sections[-1].y) - 1))

    if len(sections) == 3:
        # rect/extract link = entrainer stage (the seed the search picked)
        links = [(result["entrainer_stage_rect"], 0),
                 (inter["i_a"], i_lower)]
    else:
        links = [(inter["i_a"], i_lower)]

    profile = assemble_profile(sections, links)
    profile.update({"found": True, "message": "Feasible.",
                    "intersection": inter["point"],
                    "candidates": inter["candidates"],
                    "comps": result["comps"], "lk": lk, "hk": hk,
                    "extract": result.get("extract", False)})
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
    assert [s.name for s in res["sections"]] == ["rectifying", "stripping"]

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
    # assembled profile is internally consistent (x/y/T same length)
    assert len(high_r["x"]) == len(high_r["y"]) == len(high_r["T"])

    # True 3-section extractive column: dedicated middle-section march.
    ext = bound_val_method(zF, F=100.0, r=8.0, q=1.0, antoine=antoine,
                           comps=["benzene", "toluene", "xylene"], lk=0, P=760.0,
                           spec_mode="recovery", FR_LK=0.98, NK_spec=1e-3,
                           extract=True, E2F=0.3, xE=[0.0, 0.0, 1.0])
    assert abs(ext["xB"].sum() - 1.0) < 1e-2, ext["xB"].sum()
    assert [s.name for s in ext["sections"]] == \
        ["rectifying", "extractive", "stripping"]
    assert len(ext["xExtract"]) >= 1 and len(ext["Textract"]) == len(ext["xExtract"])
    assert ext["xExtract"].min() > -1e-6 and ext["xExtract"].max() < 1 + 1e-6
    prof_e = build_column_profile(ext)
    assert prof_e["found"], prof_e["message"]
    assert prof_e["entrainer_stage"] <= prof_e["feed_stage"], \
        "entrainer must enter above the main feed (smaller stage index from top)"
    assert len(prof_e["x"]) == len(prof_e["y"]) == len(prof_e["T"])

    # Non-adjacent keys (lk=0, hk=2): feasible, balance closes, and the middle
    # component (toluene) distributes into both products — the sloppy split.
    nak = bound_val_method(zF, F=100.0, r=8.0, q=1.0, antoine=antoine,
                           comps=["benzene", "toluene", "xylene"], lk=0, hk=2,
                           P=760.0, spec_mode="recovery", FR_LK=0.98, FR_HK=0.98)
    assert nak["hk"] == 2 and abs(nak["D"] + nak["B"] - 100.0) < 1e-6
    assert nak["xD"][1] > 1e-3 and nak["xB"][1] > 1e-3, "middle key should distribute"
    assert build_column_profile(nak)["found"]

    # NRTL activity model threads through the marches (modified Raoult's law):
    # the run stays feasible and every stage is physical. Same gamma_fn closure
    # the rigorous solvers consume, so BVM and HYSIM share one thermo path.
    from core.thermodynamics import nrtl_gamma_fn
    a = 0.1
    gfn = nrtl_gamma_fn([[0.0, a, a], [a, 0.0, a], [a, a, 0.0]],
                        [[0.0] * 3] * 3,
                        [[0.0, 0.3, 0.3], [0.3, 0.0, 0.3], [0.3, 0.3, 0.0]])
    ni = bound_val_method(zF, F=100.0, r=12.0, q=1.0, antoine=antoine,
                          comps=["benzene", "toluene", "xylene"], lk=0, P=760.0,
                          spec_mode="recovery", FR_LK=0.98, NK_spec=1e-3,
                          gamma_fn=gfn)
    idl = bound_val_method(zF, F=100.0, r=12.0, q=1.0, antoine=antoine,
                           comps=["benzene", "toluene", "xylene"], lk=0, P=760.0,
                           spec_mode="recovery", FR_LK=0.98, NK_spec=1e-3)
    assert ni["xRect"].min() > -1e-6 and ni["xRect"].max() < 1 + 1e-6
    m = min(len(ni["xRect"]), len(idl["xRect"]))
    assert np.abs(ni["xRect"][:m] - idl["xRect"][:m]).max() > 1e-2, \
        "activity model must shift the rectifying profile off the ideal one"
    assert build_column_profile(ni)["found"], "mild-NRTL ternary should stay feasible"
    # (PLXANT / 7-coeff extended Antoine also marches unchanged — antoine_psat
    # dispatches on the column count; that path is covered in thermodynamics._demo,
    # and BVM calls the same antoine_psat, so no separate check is needed here.)

    print(f"feasible @ r=12: {high_r['n_stages']} stages, feed at "
          f"{high_r['feed_stage']} ({high_r['n_rect']} rect / {high_r['n_strip']} strip)")
    print(f"extractive 3-section: {prof_e['n_stages']} stages, feed at "
          f"{prof_e['feed_stage']}, entrainer at {prof_e['entrainer_stage']} "
          f"({prof_e['n_extract']} extractive stages)")
    print("self-check OK")


if __name__ == "__main__":
    _demo()
