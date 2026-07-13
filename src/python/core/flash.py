"""Isothermal / vapour-fraction flash on the existing gamma-phi K-value seam.

Built for the Phase-EQ module (a live test-bench for every thermo model) and
reused later by the roadmap's Month-5 "feed q flash". All the thermo comes from
core.thermodynamics.k_values — this file is only the Rachford-Rice root find and
the successive-substitution outer loop, so any activity/EOS model wired into
k_values flashes for free.

Conventions match the rest of core: P is in the vle_model's Psat unit, T in the
Antoine fit's temperature unit, antoine is the (N,3)/(N,7) coefficient matrix.
"""
from collections import namedtuple

import numpy as np
from scipy.optimize import brentq

from core.thermodynamics import k_values, _solve_T

# beta = vapour fraction (V/F); x/y liquid/vapour comps; K the equilibrium ratios
# used; T the (solved or given) temperature; phase a plain tag.
FlashResult = namedtuple("FlashResult", "beta x y K T phase")


def rachford_rice(z, K):
    """Vapour fraction beta solving sum_i z_i (K_i-1)/(1+beta(K_i-1)) = 0.

    Returns beta clamped to [0, 1] with an all-liquid (beta=0) / all-vapour
    (beta=1) short-circuit when the feed is single-phase at these K-values
    (detected from the RR function's signs at the bracket ends). Within the
    two-phase window the denominator stays positive on [0,1], so brentq needs
    no pole guarding.
    """
    z = np.asarray(z, float)
    K = np.asarray(K, float)
    km1 = K - 1.0
    f0 = float(np.sum(z * km1))            # f(0) = sum z_i K_i - 1
    f1 = float(np.sum(z * km1 / K))        # f(1) = 1 - sum z_i / K_i
    if f0 <= 0.0:                          # below bubble -> all liquid
        return 0.0
    if f1 >= 0.0:                          # above dew -> all vapour
        return 1.0

    def f(beta):
        return float(np.sum(z * km1 / (1.0 + beta * km1)))

    return float(brentq(f, 0.0, 1.0, xtol=1e-12))


def flash_TP(z, T, P, antoine, gamma_fn=None, phi_fn=None,
             tol=1e-10, maxit=200):
    """Isothermal flash at fixed (T, P) by successive substitution.

    Returns a FlashResult. Single-phase feeds come back honestly: beta clamped
    to 0/1 with phase 'liquid'/'vapour' and the *incipient* other phase in
    x/y (the composition that would first appear on crossing the phase
    boundary). K is re-evaluated at the current liquid each sweep, so
    composition-dependent gamma/phi models converge instead of freezing at the
    feed estimate.
    """
    z = np.asarray(z, float)
    z = z / z.sum()
    x = z.copy()
    beta, phase = 0.5, "two-phase"
    for _ in range(maxit):
        K = np.asarray(k_values(T, P, antoine, gamma_fn, x, phi_fn), float)
        km1 = K - 1.0
        f0 = float(np.sum(z * km1))
        f1 = float(np.sum(z * km1 / K))
        if f0 <= 0.0:
            beta, phase = 0.0, "liquid"
            xn = z.copy()
            yn = K * z
        elif f1 >= 0.0:
            beta, phase = 1.0, "vapour"
            xn = z / K
            yn = z.copy()
        else:
            beta, phase = rachford_rice(z, K), "two-phase"
            xn = z / (1.0 + beta * km1)
            yn = K * xn
        xn = xn / xn.sum()
        yn = yn / yn.sum()
        if np.max(np.abs(xn - x)) < tol:
            x = xn
            break
        x = xn
    return FlashResult(beta=beta, x=x, y=yn, K=K, T=float(T), phase=phase)


def flash_PbetaT(z, P, beta, antoine, gamma_fn=None, phi_fn=None,
                 lo=-100.0, hi=500.0):
    """Solve T at fixed pressure for a target vapour fraction `beta`.

    Uses the standard result that specifying beta turns the flash into a 1-D
    T-solve of the Rachford-Rice function at that beta (monotone in T because
    K rises with T). beta=0 recovers bubble_T, beta=1 recovers dew_T. Returns
    a full FlashResult from flash_TP at the found T.

    ponytail: K in the T-solve is evaluated with the activity/EOS argument at
    the feed z (exact for ideal VLE; the same proxy dew_T already uses). The
    reported x/y/K come from flash_TP, which does refine against the phase
    composition.
    """
    z = np.asarray(z, float)
    z = z / z.sum()
    if not 0.0 <= beta <= 1.0:
        raise ValueError("target vapour fraction beta must be in [0, 1]")

    def h(T):
        K = np.asarray(k_values(T, P, antoine, gamma_fn, z, phi_fn), float)
        return float(np.sum(z * (K - 1.0) / (1.0 + beta * (K - 1.0))))

    T = _solve_T(h, lo, hi)
    return flash_TP(z, T, P, antoine, gamma_fn, phi_fn)


def _demo():
    # benzene / toluene, log10(Psat[mmHg]) vs degC; P = 1 atm.
    antoine = np.array([[6.90565, 1211.033, 220.79],
                        [6.95464, 1344.8, 219.48]])
    P = 760.0
    z = np.array([0.5, 0.5])
    from core.thermodynamics import bubble_T, dew_T

    # equimolar at 95 C is between bubble (~92 C) and dew (~98 C) -> two-phase,
    # vapour enriched in the lighter benzene.
    r = flash_TP(z, 95.0, P, antoine)
    assert 0.0 < r.beta < 1.0, r.beta
    assert r.y[0] > z[0] > r.x[0], (r.x, r.y)
    assert abs(r.x.sum() - 1.0) < 1e-9 and abs(r.y.sum() - 1.0) < 1e-9

    # RR edge cases: K all > 1 -> all vapour; K all < 1 -> all liquid.
    assert rachford_rice(z, np.array([3.0, 2.0])) == 1.0
    assert rachford_rice(z, np.array([0.3, 0.2])) == 0.0

    # beta continuity: sweep T across the two-phase region, beta monotone up.
    betas = [flash_TP(z, T, P, antoine).beta for T in np.linspace(85, 105, 40)]
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(betas, betas[1:])), betas
    assert betas[0] == 0.0 and betas[-1] == 1.0

    # beta=0 / beta=1 T-solves reproduce bubble / dew T.
    Tb = bubble_T(z, P, antoine)
    Td = dew_T(z, P, antoine)
    assert abs(flash_PbetaT(z, P, 0.0, antoine).T - Tb) < 0.05
    assert abs(flash_PbetaT(z, P, 1.0, antoine).T - Td) < 0.05

    # trace component doesn't break the split.
    zt = np.array([0.999, 0.001])
    rt = flash_TP(zt, 95.0, P, antoine)
    assert abs(rt.x.sum() - 1.0) < 1e-9 and abs(rt.y.sum() - 1.0) < 1e-9
    print("flash OK: two-phase beta=%.3f, Tb=%.2f, Td=%.2f" % (r.beta, Tb, Td))


if __name__ == "__main__":
    _demo()
