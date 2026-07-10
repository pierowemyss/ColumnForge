"""Fenske-Underwood-Gilliland (FUG) shortcut column design.

The classic "where do I start" tool: given a feed, the two key components and
their split, and constant relative volatilities, estimate minimum stages
(Fenske), minimum reflux (Underwood), the actual stage count at an operating
reflux (Gilliland/Molokanov) and the feed-stage location (Kirkbride). No column
solve — closed-form correlations, so it's the preliminary design that seeds a
rigorous run or pairs with BVM feasibility.

Relative volatilities are taken *constant* (geometric mean over the column is
the usual practice) and referenced to the heavy key. That is the whole model's
ceiling: FUG is a screening tool, not a rating method — hand its N/R/feed to the
rigorous solver for the real answer.

Everything is mole-fraction / recovery based and unit-free; α is dimensionless.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def _key_split(z, lk, hk, rec_lk, rec_hk):
    """Distillate/bottoms flows per component (basis F=1) from the two key
    recoveries, distributing non-keys by relative volatility (Fenske). Returns
    (d, b, xD, xB) with d+b=z."""
    z = np.asarray(z, float)
    # keys: recovery = fraction of that component leaving in the distillate.
    d = np.empty_like(z)
    d[lk] = rec_lk * z[lk]
    d[hk] = rec_hk * z[hk]
    return d, z - d


def fenske_min_stages(alpha, xD, xB, lk, hk):
    """Minimum equilibrium stages at total reflux (Fenske), keys only.

        Nmin = ln[(xLK/xHK)_D * (xHK/xLK)_B] / ln(alpha_LK,HK)

    alpha referenced to any common basis (only the LK/HK ratio enters). Nmin
    counts equilibrium stages including the reboiler."""
    a = alpha[lk] / alpha[hk]
    sep = (xD[lk] / xD[hk]) * (xB[hk] / xB[lk])
    return np.log(sep) / np.log(a)


def fenske_distribution(alpha, d_keys, b_keys, z, lk, hk):
    """Non-key component split from Fenske at total reflux: for each component i,

        (d_i/b_i) = alpha_i^Nmin * (d_HK/b_HK)      [Fenske, HK basis]

    Returns (d, b) per component (basis F=1), with the keys kept as given."""
    z = np.asarray(z, float)
    alpha = np.asarray(alpha, float)
    Nmin = np.log((d_keys[lk] / b_keys[lk]) * (b_keys[hk] / d_keys[hk])) \
        / np.log(alpha[lk] / alpha[hk])
    ahk = alpha / alpha[hk]
    ratio_hk = d_keys[hk] / b_keys[hk]
    db = ahk ** Nmin * ratio_hk           # d_i / b_i for every component
    d = z * db / (1.0 + db)
    d[lk], d[hk] = d_keys[lk], d_keys[hk]
    return d, z - d


def underwood_theta(alpha, z, q, lk, hk):
    """Underwood root θ in (alpha_HK, alpha_LK) of

        sum_i  alpha_i * z_i / (alpha_i - θ)  =  1 - q

    (α referenced to HK so α_HK = 1). The relevant root for a sharp LK/HK split
    lies strictly between the two keys' volatilities."""
    alpha = np.asarray(alpha, float)
    z = np.asarray(z, float)

    def f(theta):
        return float(np.sum(alpha * z / (alpha - theta)) - (1.0 - q))

    lo, hi = alpha[hk], alpha[lk]
    eps = 1e-6 * (hi - lo)
    return brentq(f, lo + eps, hi - eps)


def underwood_min_reflux(alpha, xD, theta):
    """Minimum reflux ratio from the Underwood distillate relation

        Rmin + 1 = sum_i  alpha_i * xD_i / (alpha_i - θ)."""
    alpha = np.asarray(alpha, float)
    xD = np.asarray(xD, float)
    return float(np.sum(alpha * xD / (alpha - theta))) - 1.0


def gilliland_stages(Nmin, Rmin, R):
    """Actual stage count at operating reflux R via the Molokanov fit of the
    Gilliland correlation:

        X = (R - Rmin)/(R + 1)
        Y = 1 - exp[ (1 + 54.4X)/(11 + 117.2X) * (X - 1)/sqrt(X) ]
        N = (Y + Nmin)/(1 - Y)

    At R->Rmin, X->0, Y->1, N->inf; at R->inf, X->1, Y->0, N->Nmin."""
    X = (R - Rmin) / (R + 1.0)
    if X <= 0.0:
        return float("inf")
    Y = 1.0 - np.exp((1.0 + 54.4 * X) / (11.0 + 117.2 * X)
                     * (X - 1.0) / np.sqrt(X))
    return (Y + Nmin) / (1.0 - Y)


def kirkbride_feed_stage(N, z, xD, xB, D, B, lk, hk):
    """Kirkbride feed-stage location — ratio of stripping (Ns) to rectifying
    (Nr) stages:

        Ns/Nr = [ (z_HK/z_LK) * (xLK_B/xHK_D)^2 * (B/D) ] ^ 0.206

    Returns the 1-based feed stage from the top (Nr rectifying stages sit above
    it), rounded to the nearest stage."""
    ratio = ((z[hk] / z[lk]) * (xB[lk] / xD[hk]) ** 2 * (B / D)) ** 0.206
    Nr = N / (1.0 + ratio)                # Ns = ratio * Nr, Nr + Ns = N
    return int(round(Nr)) + 1


def fug_design(alpha, z, lk, hk, rec_lk, rec_hk, q=1.0, reflux_factor=1.3,
               R_op=None):
    """Full FUG screening design.

    alpha        (C,) constant relative volatilities (any common basis)
    z            (C,) feed mole fractions (sum 1)
    lk, hk       light/heavy key component indices
    rec_lk       recovery of LK in the distillate (e.g. 0.98)
    rec_hk       recovery of HK in the distillate (e.g. 0.02)
    q            feed thermal quality (1 = sat. liquid)
    reflux_factor operating reflux as a multiple of Rmin (1.2-1.5 typical)
    R_op         explicit operating reflux ratio; overrides reflux_factor when
                 given (must exceed Rmin or the stage count is infinite)

    Returns a report dict: Nmin, Rmin, R, N (float, incl. reboiler), feed_stage
    (1-based from top), D, B, xD, xB, theta, and an (R, N) curve for plotting.
    """
    z = np.asarray(z, float)
    alpha = np.asarray(alpha, float) / np.asarray(alpha, float)[hk]   # HK basis

    d, b = _key_split(z, lk, hk, rec_lk, rec_hk)
    d, b = fenske_distribution(alpha, d, b, z, lk, hk)
    D = float(d.sum()); B = float(b.sum())
    xD = d / D; xB = b / B

    Nmin = fenske_min_stages(alpha, xD, xB, lk, hk)
    theta = underwood_theta(alpha, z, q, lk, hk)
    Rmin = underwood_min_reflux(alpha, xD, theta)
    R = float(R_op) if R_op is not None else reflux_factor * Rmin
    N = gilliland_stages(Nmin, Rmin, R)
    feed_stage = kirkbride_feed_stage(N, z, xD, xB, D, B, lk, hk)

    # N-vs-R curve for the design plot: from just above Rmin out to 3*Rmin.
    Rs = np.linspace(1.02 * Rmin, 3.0 * Rmin, 30)
    Ns = np.array([gilliland_stages(Nmin, Rmin, r) for r in Rs])

    return dict(Nmin=Nmin, Rmin=Rmin, R=R, N=N, feed_stage=feed_stage,
                D=D, B=B, xD=xD, xB=xB, theta=theta,
                curve_R=Rs, curve_N=Ns)


def _demo():
    # --- Fenske binary hand-check: alpha=2.4, sharp-ish split ---
    alpha2 = np.array([2.4, 1.0])
    xD = np.array([0.95, 0.05]); xB = np.array([0.05, 0.95])
    Nmin = fenske_min_stages(alpha2, xD, xB, 0, 1)
    # ln((0.95/0.05)*(0.95/0.05)) / ln(2.4) = ln(361)/ln(2.4)
    assert abs(Nmin - np.log(361.0) / np.log(2.4)) < 1e-9, Nmin
    assert abs(Nmin - 6.727) < 1e-2, Nmin

    # --- Underwood root sits between the keys and solves its equation ---
    z2 = np.array([0.5, 0.5])
    th = underwood_theta(alpha2, z2, q=1.0, lk=0, hk=1)
    assert 1.0 < th < 2.4, th
    assert abs(np.sum(alpha2 * z2 / (alpha2 - th)) - 0.0) < 1e-6   # 1-q = 0

    # --- Gilliland limits: total reflux -> Nmin, min reflux -> infinite ---
    assert gilliland_stages(10.0, 1.5, 1.5) == float("inf")
    assert abs(gilliland_stages(10.0, 1.5, 1e6) - 10.0) < 0.1      # R huge -> Nmin
    # monotonic: more reflux, fewer stages
    n_lo = gilliland_stages(10.0, 1.5, 2.0)
    n_hi = gilliland_stages(10.0, 1.5, 4.0)
    assert 10.0 < n_hi < n_lo, (n_lo, n_hi)

    # --- Full design on a ternary (LK=1, between a light and a heavy) ---
    # depropanizer-flavoured: C2 (light), C3=LK, C4=HK, referenced to C4.
    alpha = np.array([5.0, 2.0, 1.0])
    z = np.array([0.3, 0.4, 0.3])
    rep = fug_design(alpha, z, lk=1, hk=2, rec_lk=0.98, rec_hk=0.02,
                     q=1.0, reflux_factor=1.3)
    # material balance closes and keys land on the right side
    assert abs(rep["D"] + rep["B"] - 1.0) < 1e-9
    assert rep["xD"][1] > rep["xB"][1]           # LK richer in distillate
    assert rep["xB"][2] > rep["xD"][2]           # HK richer in bottoms
    # the very light non-key (alpha=5) goes essentially all overhead
    d0 = rep["xD"][0] * rep["D"]
    assert d0 > 0.99 * z[0], (d0, z[0])
    # physically ordered result: Nmin < N (finite), Rmin < R, sane feed stage
    assert 0 < rep["Nmin"] < rep["N"] < np.inf, rep
    assert 0 < rep["Rmin"] < rep["R"], rep
    assert 1 < rep["feed_stage"] < rep["N"] + 2, rep["feed_stage"]
    assert rep["curve_N"][0] > rep["curve_N"][-1]   # curve falls with reflux

    # explicit operating reflux overrides the factor
    rep2 = fug_design(alpha, z, lk=1, hk=2, rec_lk=0.98, rec_hk=0.02, q=1.0,
                      R_op=2.0 * rep["Rmin"])
    assert abs(rep2["R"] - 2.0 * rep["Rmin"]) < 1e-9
    assert rep2["N"] < rep["N"]                      # more reflux -> fewer stages

    print(f"shortcut FUG self-check OK: ternary Nmin={rep['Nmin']:.1f} "
          f"Rmin={rep['Rmin']:.2f} N={rep['N']:.1f} feed@{rep['feed_stage']}")


if __name__ == "__main__":
    _demo()
