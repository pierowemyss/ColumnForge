"""Ideal vapour-liquid equilibrium: Antoine vapour pressure + Raoult's law.

gamma = phi = 1 (ideal). Antoine convention matches the BVM solver:

    log10(Psat) = A - B / (T + C)

so Psat comes out in the unit the coefficients were fitted to, and the pressure
P passed in must be in that same unit (the bundled benzene/toluene/xylene fits
are mmHg with T in degrees C). Swap these for an activity/EOS model later without
touching callers — the signatures are model-agnostic.
"""

import numpy as np
from scipy.optimize import brentq


def antoine_psat(T, antoine):
    """Saturation pressure of every component at temperature T.

    antoine: (N, 3) [A, B, C] regular Antoine, or (N, 7) [C1..C7] Aspen PLXANT
    (extended Antoine) — dispatched on the column count so every caller/solver
    gets PLXANT for free just by passing a 7-wide matrix. Returns an (N,) array.
    """
    antoine = np.asarray(antoine, float)
    if antoine.shape[1] == 7:
        return plxant_psat(T, antoine)
    A, B, C = antoine[:, 0], antoine[:, 1], antoine[:, 2]
    return 10.0 ** (A - B / (T + C))


def plxant_psat(T, c, t_to_K=lambda T: T + 273.15):
    """Aspen extended Antoine (PLXANT). T in the Antoine fit's unit; PLXANT is
    defined in Kelvin, so t_to_K converts (default degC->K, matching the bundled
    fits and nrtl_gamma_fn). c: (N, 7) of [C1..C7].

        ln(Psat) = C1 + C2/(C3+T) + C4*T + C5*ln(T) + C6*T**C7   (T in K)

    Psat comes out in the unit the coefficients were fitted to (Aspen default Pa).
    """
    c = np.asarray(c, float)
    Tk = t_to_K(T)
    lnP = (c[:, 0] + c[:, 1] / (c[:, 2] + Tk) + c[:, 3] * Tk
           + c[:, 4] * np.log(Tk) + c[:, 5] * Tk ** c[:, 6])
    return np.exp(lnP)


def k_values(T, P, antoine, gamma_fn=None, x=None):
    """K-values. Ideal (Raoult) K_i = Psat_i(T)/P unless an activity model is
    supplied: given gamma_fn and the liquid composition x, K_i = gamma_i(x,T)
    Psat_i(T)/P. gamma_fn(x, T) -> (n,) activity coefficients (see nrtl_gamma_fn).
    """
    psat = antoine_psat(T, antoine)
    if gamma_fn is None or x is None:
        return psat / P
    return np.asarray(gamma_fn(x, T), float) * psat / P


def bubble_T(x, P, antoine, lo=-100.0, hi=500.0, gamma_fn=None):
    """Bubble-point temperature: T such that sum_i K_i(T) x_i = 1.

    lo/hi bracket the root (in the Antoine fit's temperature unit). Raises if the
    bracket doesn't straddle the root — widen it for very high/low boilers.
    gamma_fn (optional) makes the K-values non-ideal (activity model).
    """
    x = np.asarray(x, float)

    def f(T):
        return float(np.sum(k_values(T, P, antoine, gamma_fn, x) * x) - 1.0)

    return brentq(f, lo, hi)


def dew_T(y, P, antoine, lo=-100.0, hi=500.0, gamma_fn=None):
    """Dew-point temperature: T such that sum_i y_i / K_i(T) = 1.

    ponytail: with an activity model, gamma is evaluated at the vapour composition
    as a proxy for the (unknown) liquid in equilibrium — exact for ideal VLE,
    approximate otherwise (a rigorous dew point needs an inner liquid-comp solve).
    """
    y = np.asarray(y, float)

    def f(T):
        return float(np.sum(y / k_values(T, P, antoine, gamma_fn, y)) - 1.0)

    return brentq(f, lo, hi)


def nrtl_gamma(x, tau, alpha):
    """NRTL activity coefficients for a multicomponent liquid.

    x      (n,) liquid mole fractions
    tau    (n,n) dimensionless interaction energies, evaluated at T (tau_ii = 0)
    alpha  (n,n) non-randomness factors (symmetric, alpha_ii = 0)
    Returns gamma (n,). Pure-component limit gives gamma_i = 1.

    Vectorised form of the standard NRTL: with S_j = sum_k x_k G_kj, r = x/S and
    C = (x @ (tau*G))/S,  ln gamma = C + (tau*G) @ r - G @ (r*C).
    """
    x = np.asarray(x, float)
    tau = np.asarray(tau, float)
    G = np.exp(-np.asarray(alpha, float) * tau)
    S = x @ G
    r = x / S
    tG = tau * G
    C = (x @ tG) / S
    ln_gamma = C + (tG @ r) - (G @ (r * C))
    return np.exp(ln_gamma)


def nrtl_gamma_fn(tau_a, tau_b, alpha, t_to_K=lambda T: T + 273.15):
    """Build a gamma_fn(x, T) closure for k_values/bubble_T, with the common
    temperature-dependent form tau_ij = a_ij + b_ij / T_K.

    tau_a, tau_b, alpha are (n,n). t_to_K converts the temperature that the
    solver passes (the Antoine fit's unit) to Kelvin for the tau correlation.
    ponytail: default assumes the bundled fits' degrees-C unit; pass
    t_to_K=lambda T: T if your Antoine coefficients are already in Kelvin.
    """
    tau_a = np.asarray(tau_a, float)
    tau_b = np.asarray(tau_b, float)
    alpha = np.asarray(alpha, float)

    def gamma_fn(x, T):
        tau = tau_a + tau_b / t_to_K(T)
        return nrtl_gamma(x, tau, alpha)

    return gamma_fn


def _demo():
    # benzene / toluene / xylene, classic mmHg + degC Antoine fits
    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    P = 760.0

    # pure-component boiling points at 1 atm (lit: ~80.1 / 110.6 / ~138 degC)
    for i, bp in enumerate((80.1, 110.6, 138.0)):
        x = np.zeros(3); x[i] = 1.0
        T = bubble_T(x, P, abc)
        assert abs(T - bp) < 2.0, f"comp {i} bubble T {T:.1f} != ~{bp}"

    # equimolar mix: bubble T between light and heavy bp; K ordering light>heavy
    x = np.array([1, 1, 1]) / 3
    T = bubble_T(x, P, abc)
    assert 80.0 < T < 138.0
    K = k_values(T, P, abc)
    assert K[0] > K[1] > K[2], "light key must be most volatile"
    assert abs(np.sum(K * x) - 1.0) < 1e-9, "bubble point: sum(K x) = 1"

    # dew point sits above the bubble point for a multicomponent mix
    Td = dew_T(x, P, abc)
    assert Td > T, f"dew T {Td:.1f} should exceed bubble T {T:.1f}"

    # --- NRTL activity model ---------------------------------------------
    tau = np.array([[0.0, 1.0], [1.2, 0.0]])
    alpha = np.array([[0.0, 0.3], [0.3, 0.0]])
    assert abs(nrtl_gamma([1.0, 0.0], tau, alpha)[0] - 1.0) < 1e-12, "pure gamma=1"
    g_dilute = nrtl_gamma([1e-6, 1 - 1e-6], tau, alpha)
    assert g_dilute[0] > 1.0, "positive tau => gamma at infinite dilution > 1"

    # Non-ideal K of the dilute component exceeds its ideal (Raoult) K.
    gfn = nrtl_gamma_fn([[0.0, 1.0], [1.2, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0]], alpha)
    ab2, x2 = abc[:2], np.array([0.05, 0.95])
    Tb = bubble_T(x2, P, ab2, gamma_fn=gfn)
    K_ni = k_values(Tb, P, ab2, gfn, x2)
    K_id = k_values(Tb, P, ab2)
    assert K_ni[0] > K_id[0], "gamma>1 must raise the dilute component's K"
    print(f"thermodynamics self-check OK (mix bubble T = {T:.1f} degC; "
          f"NRTL gamma_inf = {g_dilute[0]:.2f})")


if __name__ == "__main__":
    _demo()
