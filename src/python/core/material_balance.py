#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overall (shortcut) material balance — the foundation for complex columns.

A component balance over arbitrary feeds and products: sum of feeds equals sum
of products for every component. The simple-column recovery and direct-spec
balances are special cases, shared by the BVM feasibility solver and the
sandbox auto-balance so there is a single implementation.

Index convention: 0-based. ``lk``/``hk`` are the light/heavy-key indices and may
be any two distinct components (not necessarily adjacent) — components whose
volatility falls between them simply *distribute* into both products (a sloppy
split). When ``alpha`` (relative volatilities) is supplied the split is set by
the Geddes-Hengstebeck distribution; without it, a legacy positional sharp split
(heavy key = ``lk + 1``) is used for back-compatibility.
"""

import numpy as np


def geddes_distribution(zF, F, lk, hk, FR_LK, FR_HK, alpha):
    """Component split from the Geddes-Hengstebeck (Hengstebeck-Geddes) line.

    The defining relation (same family Aspen's DSTWU uses) is linear in logs:

        log10(d_i / b_i) = m * log10(alpha_i) + c

    anchored by the two keys' specified recoveries. With f_i = F z_i = d_i + b_i
    every component's distillate/bottoms split follows — including components
    between the keys, which distribute. Collapses to a sharp split when the keys
    are far apart in relative volatility.

    FR_LK : fractional recovery of the light key to the distillate.
    FR_HK : fractional recovery of the heavy key to the bottoms.
    alpha : (N,) relative volatilities (any consistent reference; ratios matter).
    Returns (xD, D, xB, B).
    """
    zF = np.asarray(zF, float)
    alpha = np.asarray(alpha, float)
    if not (0.0 < FR_LK < 1.0 and 0.0 < FR_HK < 1.0):
        raise ValueError("key recoveries FR_LK, FR_HK must be in (0, 1)")
    if np.any(alpha <= 0):
        raise ValueError("relative volatilities must be positive")
    a_lk, a_hk = alpha[lk], alpha[hk]
    if abs(np.log10(a_lk) - np.log10(a_hk)) < 1e-12:
        raise ValueError("light and heavy keys have equal volatility; "
                         "pick keys with distinct relative volatility")

    # Anchor distribution ratios phi = d/b for the two keys.
    phi_lk = FR_LK / (1.0 - FR_LK)            # LK: FR_LK recovered overhead
    phi_hk = (1.0 - FR_HK) / FR_HK            # HK: FR_HK recovered to bottoms
    x_lk, x_hk = np.log10(a_lk), np.log10(a_hk)
    m = (np.log10(phi_lk) - np.log10(phi_hk)) / (x_lk - x_hk)
    c = np.log10(phi_lk) - m * x_lk

    phi = 10.0 ** (m * np.log10(alpha) + c)   # d_i / b_i for every component
    f = F * zF
    d = f * phi / (1.0 + phi)
    b = f - d
    D, B = d.sum(), b.sum()
    return d / D, D, b / B, B


def matbal_recovery(zF, F, lk, FR_LK, NK_spec=None, extract=False, E2F=0.0,
                    xE=None, *, hk=None, FR_HK=None, alpha=None):
    """Recovery-based overall balance.

    General path (``alpha`` given): Geddes-Hengstebeck distribution with
    independent keys ``lk``/``hk`` and recoveries ``FR_LK``/``FR_HK`` (``FR_HK``
    defaults to ``FR_LK`` for a symmetric split). Legacy path (no ``alpha``):
    positional sharp split with heavy key ``lk + 1`` and ``NK_spec``.
    Returns (xD, D, xB, B).
    """
    zF = np.asarray(zF, float)

    if alpha is not None:
        if hk is None:
            hk = lk + 1
        xD, D, xB, B = geddes_distribution(
            zF, F, lk, hk, FR_LK, FR_LK if FR_HK is None else FR_HK, alpha)
    else:
        xD, D, xB, B = _matbal_recovery_positional(zF, F, lk, FR_LK, NK_spec, extract)

    if extract:
        if xE is None:
            raise ValueError("Extractive balance needs entrainer composition xE")
        xE = np.asarray(xE, float)
        E = E2F * F
        xB = (B * xB + E * xE) / (B + E)
        B = E + B

    return xD, D, xB, B


def _matbal_recovery_positional(zF, F, lk, FR_LK, NK_spec, extract):
    """Legacy positional sharp split (heavy key = lk+1), the original .m logic."""
    if NK_spec is None:
        raise ValueError("positional recovery balance needs NK_spec (or pass alpha)")
    N = len(zF)
    xD = np.zeros(N)
    xB = np.zeros(N)

    if lk == 0:
        if extract:
            fD = FR_LK * zF[0] + (1 - FR_LK) * zF[1] + (N - lk - 3) * NK_spec
        else:
            fD = FR_LK * zF[0] + (1 - FR_LK) * zF[1] + np.sum(zF[2:]) * NK_spec
    else:
        fD = ((1 - NK_spec) * np.sum(zF[:lk])
              + FR_LK * zF[lk]
              + (1 - FR_LK) * zF[lk + 1]
              + np.sum(zF[lk + 2:]) * NK_spec)

    xD[:lk] = (1 - NK_spec) * zF[:lk] / fD
    xD[lk] = FR_LK * zF[lk] / fD
    xD[lk + 1] = (1 - FR_LK) * zF[lk + 1] / fD
    if extract:
        xD[lk + 2:] = NK_spec
        if lk + 2 > 0:
            xD[:lk + 2] -= np.sum(xD[lk + 2:]) / (lk + 2)
    else:
        xD[lk + 2:] = NK_spec * zF[lk + 2:] / fD

    D = fD * F
    fB = 1.0 - fD

    xB[:lk] = NK_spec * zF[:lk] / fB
    xB[lk] = (1 - FR_LK) * zF[lk] / fB
    xB[lk + 1] = FR_LK * zF[lk + 1] / fB
    xB[lk + 2:] = (1 - NK_spec) * zF[lk + 2:] / fB
    B = fB * F
    return xD, D, xB, B


def matbal_direct(zF, F, lk, xD, xB):
    """Direct-spec balance: distillate/bottoms compositions given.

    D from the light-key lever rule (the .m's commented direct block):
        D = F (z_LK - xB_LK) / (xD_LK - xB_LK),  B = F - D.
    """
    zF = np.asarray(zF, float)
    xD = np.asarray(xD, float)
    xB = np.asarray(xB, float)
    denom = xD[lk] - xB[lk]
    if abs(denom) < 1e-12:
        raise ValueError("xD and xB have equal light-key fraction; D undefined")
    D = F * (zF[lk] - xB[lk]) / denom
    B = F - D
    return xD, D, xB, B


def mix_feeds(feeds):
    """Combine feed streams into a single (F, z) pair.

    feeds: iterable of (F_i, z_i). Returns (F_total, z_mixed).
    """
    feeds = [(float(F), np.asarray(z, float)) for F, z in feeds]
    if not feeds:
        raise ValueError("at least one feed is required")
    F = sum(F for F, _ in feeds)
    if F <= 0:
        raise ValueError("total feed flow must be positive")
    z = sum(F_i * z_i for F_i, z_i in feeds) / F
    return F, z


def overall_balance(feeds, *, lk, spec_mode="recovery", FR_LK=None,
                    NK_spec=None, xD=None, xB=None, extract=False,
                    E2F=0.0, xE=None, hk=None, FR_HK=None, alpha=None,
                    side_draws=()):
    """General overall balance entry point.

    Combines arbitrary feeds, removes any side draws from the pool, then
    dispatches to the spec-mode balance for the remaining D + B split.

    side_draws: iterable of (flow, composition-or-None). A None composition
    draws at the mixed-feed composition — the shortcut assumption when the
    stage composition isn't known yet (a rigorous solve refines it).
    # ponytail: draw-at-feed-composition is the shortcut ceiling; the rigorous
    # solvers close the true multi-product balance from stage compositions.

    Returns (xD, D, xB, B).
    """
    F, z = mix_feeds(feeds)
    for wflow, wcomp in side_draws:
        w = np.asarray(wcomp, float) if wcomp is not None else z
        Fz = F * z - float(wflow) * w
        F = F - float(wflow)
        if F <= 0 or np.any(Fz < -1e-9):
            raise ValueError("side draws exceed the feed pool (overall balance)")
        z = np.clip(Fz, 0.0, None) / F
    if spec_mode == "recovery":
        if FR_LK is None or (NK_spec is None and alpha is None):
            raise ValueError("recovery balance needs FR_LK and either alpha "
                             "(Geddes) or NK_spec (positional)")
        return matbal_recovery(z, F, lk, FR_LK, NK_spec, extract, E2F, xE,
                               hk=hk, FR_HK=FR_HK, alpha=alpha)
    if spec_mode == "direct":
        if xD is None or xB is None:
            raise ValueError("direct balance needs xD and xB")
        return matbal_direct(z, F, lk, xD, xB)
    raise ValueError(f"unknown spec_mode {spec_mode!r}")


def balance_closes(feeds, products, tol=1e-6):
    """True if the component balance closes: sum(F z) == sum(P x) per component.

    feeds/products: iterables of (flow, composition_array).
    """
    inflow = sum(float(F) * np.asarray(z, float) for F, z in feeds)
    outflow = sum(float(P) * np.asarray(x, float) for P, x in products)
    return bool(np.max(np.abs(inflow - outflow)) < tol)


def _demo():
    zF = np.array([0.4, 0.35, 0.25])
    F = 100.0

    # Recovery balance closes overall and per-component.
    xD, D, xB, B = overall_balance([(F, zF)], lk=0, spec_mode="recovery",
                                   FR_LK=0.98, NK_spec=1e-3)
    assert abs(xD.sum() - 1) < 1e-9 and abs(xB.sum() - 1) < 1e-9
    assert abs(D + B - F) < 1e-9
    assert balance_closes([(F, zF)], [(D, xD), (B, xB)], tol=1e-6)

    # Direct mode reproduces flows consistent with the lever rule and closes.
    xD2, D2, xB2, B2 = overall_balance([(F, zF)], lk=0, spec_mode="direct",
                                       xD=xD, xB=xB)
    assert abs(D2 - D) < 1e-6 and abs(B2 - B) < 1e-6
    assert balance_closes([(F, zF)], [(D2, xD2), (B2, xB2)], tol=1e-6)

    # Two feeds mix by flow-weighted composition.
    Fm, zm = mix_feeds([(60.0, [0.5, 0.3, 0.2]), (40.0, [0.25, 0.4, 0.35])])
    assert abs(Fm - 100.0) < 1e-9
    assert abs(zm.sum() - 1) < 1e-9 and abs(zm[0] - 0.4) < 1e-9

    # Multi-product: a side draw at feed composition closes the full balance.
    xDs, Ds, xBs, Bs = overall_balance([(F, zF)], lk=0, spec_mode="recovery",
                                       FR_LK=0.98, NK_spec=1e-3,
                                       side_draws=[(20.0, None)])
    assert balance_closes([(F, zF)], [(Ds, xDs), (Bs, xBs), (20.0, zF)], tol=1e-6)
    assert abs(Ds + Bs + 20.0 - F) < 1e-9

    # Geddes distribution: balance closes, keys hit their specified recoveries,
    # and a distributing middle component (non-adjacent keys) appears in BOTH
    # products. 4 comps, alpha descending; keys lk=0, hk=3.
    z4 = np.array([0.3, 0.25, 0.25, 0.2])
    alpha = np.array([5.0, 2.5, 1.6, 1.0])
    xD, D, xB, B = matbal_recovery(z4, 100.0, lk=0, FR_LK=0.95,
                                   hk=3, FR_HK=0.95, alpha=alpha)
    assert abs(xD.sum() - 1) < 1e-9 and abs(xB.sum() - 1) < 1e-9
    assert abs(D + B - 100.0) < 1e-9
    assert balance_closes([(100.0, z4)], [(D, xD), (B, xB)], tol=1e-6)
    d_lk = D * xD[0]
    assert abs(d_lk / (100.0 * z4[0]) - 0.95) < 1e-6   # LK recovery to D
    assert xD[1] > 1e-3 and xB[1] > 1e-3               # comp 1 distributes
    print("material-balance self-check OK")


if __name__ == "__main__":
    _demo()
