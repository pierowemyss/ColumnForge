"""Ideal vapour-liquid equilibrium: Antoine vapour pressure + Raoult's law.

gamma = phi = 1 (ideal). Antoine convention matches the BVM solver:

    log10(Psat) = A - B / (T + C)

so Psat comes out in the unit the coefficients were fitted to, and the pressure
P passed in must be in that same unit (the bundled benzene/toluene/xylene fits
are mmHg with T in degrees C). Swap these for an activity/EOS model later without
touching callers — the signatures are model-agnostic.
"""

import math
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq


def antoine_psat(T, antoine):
    """Saturation pressure of every component at temperature T.

    Dispatched on the column count, so every caller/solver gets a different fit
    for free just by passing a differently-shaped matrix. Returns an (N,) array.

        (N, 3)  [A, B, C]            regular Antoine, log10(mmHg)
        (N, 6)  [a, b, c, d, Tc, Pc] Wagner (reduced form, bar)
        (N, 7)  [C1..C7]             Aspen PLXANT / extended Antoine
    """
    antoine = np.asarray(antoine, float)
    if antoine.shape[1] == 7:
        return plxant_psat(T, antoine)
    if antoine.shape[1] == 6:
        return wagner_psat(T, antoine)
    A, B, C = antoine[:, 0], antoine[:, 1], antoine[:, 2]
    return 10.0 ** (A - B / (T + C))


def plxant_psat(T, c, t_to_K=lambda T: T + 273.15):
    """Aspen extended Antoine (PLXANT). T in the Antoine fit's unit; PLXANT is
    defined in Kelvin, so t_to_K converts (default degC->K, matching the bundled
    fits and nrtl_gamma_fn). c: (N, 7) of [C1..C7].

        ln(Psat) = C1 + C2/(C3+T) + C4*T + C5*ln(T) + C6*T**C7   (T in K)

    Psat comes out in the unit the coefficients were fitted to. The bundled
    database emits **bar** (see ThermodynamicsConfig._BAR_TO_PSAT_UNIT); an
    Aspen export in Pa differs only in C1, by ln(1e5).
    """
    c = np.asarray(c, float)
    Tk = t_to_K(T)
    lnP = (c[:, 0] + c[:, 1] / (c[:, 2] + Tk) + c[:, 3] * Tk
           + c[:, 4] * np.log(Tk) + c[:, 5] * Tk ** c[:, 6])
    return np.exp(lnP)


def wagner_psat(T, c, t_to_K=lambda T: T + 273.15):
    """Wagner 25-form vapour pressure. c: (N, 6) of [a, b, c, d, Tc[K], Pc[bar]].

        ln(Psat/Pc) = (a*tau + b*tau^1.5 + c*tau^3 + d*tau^6) / Tr
        Tr = T_K/Tc,   tau = 1 - Tr

    Critically anchored by construction: Psat(Tc) == Pc exactly, which is what
    PLXANT and Antoine only approximate. Returns Psat in **bar** (Pc's unit).

    ponytail: tau^1.5 is complex above Tc, so Tr is clamped to 1 and the result
    saturates at Pc there. A solver that walks a light component past its
    critical temperature gets Pc rather than a NaN — but the value is a floor,
    not physics. PLXANT has no such limit, which is why it stays the default.
    """
    c = np.asarray(c, float)
    Tc, Pc = c[:, 4], c[:, 5]
    Tr = np.minimum(t_to_K(T) / Tc, 1.0)
    tau = 1.0 - Tr
    f = c[:, 0] * tau + c[:, 1] * tau ** 1.5 + c[:, 2] * tau ** 3 + c[:, 3] * tau ** 6
    return Pc * np.exp(f / Tr)


def antoine_Tsat(P, abc):
    """Invert vapour pressure for one component's boiling point at P — initial-T
    guess only. 3-term Antoine: closed form. Wagner/PLXANT: bracketed solve.
    """
    abc = np.asarray(abc, float)
    if abc.shape[-1] == 7:
        row = abc.reshape(1, 7)
        return _solve_T(lambda T: float(plxant_psat(T, row)[0]) - P, -100.0, 500.0)
    if abc.shape[-1] == 6:
        row = abc.reshape(1, 6)
        return _solve_T(lambda T: float(wagner_psat(T, row)[0]) - P, -100.0, 500.0)
    A, B, C = abc
    return B / (A - np.log10(P)) - C


# Gas constant (J/mol/K) for Clausius-Clapeyron latent heats.
R_GAS = 8.314462618

# Duties come out of the solvers as flow (kmol/h) x molar enthalpy (J/mol),
# which is numerically kJ/h. Multiply by this to display in kW.
KJH_TO_KW = 1.0 / 3600.0


def latent_heat(T, antoine, t_to_K=lambda T: T + 273.15):
    """Molar heat of vaporisation (J/mol) per component at temperature T, from
    the Clausius-Clapeyron slope of the vapour-pressure fit:

        lambda_i = R * T_K^2 * d ln(Psat_i)/dT

    T is in the fit's temperature unit (bundled fits: degC); t_to_K converts to
    Kelvin for the R*T^2 factor. antoine: (N,3) Antoine, (N,6) Wagner or
    (N,7) PLXANT.
    """
    antoine = np.asarray(antoine, float)
    Tk = t_to_K(T)
    if antoine.shape[1] == 7:
        c = antoine
        dlnP = (-c[:, 1] / (c[:, 2] + Tk) ** 2 + c[:, 3] + c[:, 4] / Tk
                + c[:, 5] * c[:, 6] * Tk ** (c[:, 6] - 1.0))
    elif antoine.shape[1] == 6:
        # d/dT [ f(tau)/Tr ] with tau = 1 - Tr, Tr = Tk/Tc:
        #   dlnP/dT = -(f'(tau)*Tr + f(tau)) / (Tc * Tr^2)
        c = antoine
        Tc = c[:, 4]
        Tr = np.minimum(Tk / Tc, 1.0 - 1e-9)
        tau = 1.0 - Tr
        f = (c[:, 0] * tau + c[:, 1] * tau ** 1.5
             + c[:, 2] * tau ** 3 + c[:, 3] * tau ** 6)
        fp = (c[:, 0] + 1.5 * c[:, 1] * tau ** 0.5
              + 3.0 * c[:, 2] * tau ** 2 + 6.0 * c[:, 3] * tau ** 5)
        dlnP = -(fp * Tr + f) / (Tc * Tr ** 2)
    else:
        # log10 form: ln Psat = ln10 * (A - B/(T+C)); dT in the fit unit == dT in K
        B, C = antoine[:, 1], antoine[:, 2]
        dlnP = np.log(10.0) * B / (T + C) ** 2
    return R_GAS * Tk ** 2 * dlnP


def k_values(T, P, antoine, gamma_fn=None, x=None, phi_fn=None):
    """K-values. Ideal (Raoult) K_i = Psat_i(T)/P; with an activity model,
    K_i = gamma_i(x,T) Psat_i/P; with a vapour-phase EOS on top (gamma-phi):

        K_i = gamma_i * Psat_i * phi_i^sat / (phi_i^V * P)

    phi_fn(y, T, P) -> (n,) vapour-mixture fugacity coefficients (see
    srk_phi_fn). phi^sat_i is pure-i vapour at (T, Psat_i); phi^V is evaluated
    at the vapour in equilibrium with x, estimated as y ~ K_raoult*x normalised
    (ponytail: one-shot estimate, no inner y-iteration — the outer solver loop
    refines x/T anyway; Poynting factor neglected, fine below ~10 bar).
    """
    psat = antoine_psat(T, antoine)
    K = psat / P
    if gamma_fn is not None and x is not None:
        K = np.asarray(gamma_fn(x, T), float) * K
    if phi_fn is not None and x is not None:
        y = K * np.asarray(x, float)
        s = y.sum()
        y = y / s if s > 0.0 else np.full_like(K, 1.0 / len(K))
        n = len(K)
        if hasattr(phi_fn, "pure"):          # vectorised fast path (srk_phi_fn)
            phi_sat = phi_fn.pure(T, psat)
        else:                                # generic phi_fn seam: one-hot loop
            phi_sat = np.array([phi_fn(np.eye(n)[i], T, psat[i])[i]
                                for i in range(n)])
        K = K * phi_sat / phi_fn(y, T, P)
    return K


def _solve_T(f, lo, hi, guess=None, walk_lo=False):
    """brentq, but widen the upper bound until the bracket straddles a root.
    High boilers (e.g. glycols/carbonates) saturate above the nominal 500 unit
    range, which otherwise trips 'f(a) and f(b) must have different signs'.
    ponytail: only hi widens; lo=-100 keeps T+273 > 0 so PLXANT stays finite.

    `guess`: a nearby temperature (the column solvers hold last iteration's T,
    which is a stage away from this one). Secant from there lands in 2-4 f-evals;
    brentq on the raw 600-degree bracket needs ~17, and it was the single biggest
    cost in a column solve. Falls back to the bracket if secant misbehaves, so
    the answer is brentq's either way.
    """
    if guess is not None and lo < guess < hi:
        T0 = float(guess)
        f0 = f(T0)
        if abs(f0) < _T_FTOL:
            return T0
        T1 = T0 + 0.5
        f1 = f(T1)
        for _ in range(12):
            if f1 == f0:
                break
            T2 = T1 - f1 * (T1 - T0) / (f1 - f0)
            if not (lo < T2 < hi):
                break
            T0, f0, T1 = T1, f1, T2
            f1 = f(T1)
            if abs(f1) < _T_FTOL:
                return T1
    flo = f(lo)
    while walk_lo and flo > 0.0 and lo + 50.0 < hi:
        # Spurious positive at the floor: activity models extrapolated far below
        # their fit range return gamma ~ 1e12, faking a boiling point at -100.
        # Walk the floor up onto the physical branch (f < 0 below the real root).
        # Only valid for bubble-type f (negative below the root); dew_T's f has
        # the opposite sign at the floor, so it opts out.
        lo += 50.0
        flo = f(lo)
    for _ in range(10):
        if flo * f(hi) <= 0.0:
            return brentq(f, lo, hi)
        hi += 200.0
    raise ValueError(f"no saturation temperature found below {hi:.0f}; check "
                     "vapour-pressure coefficients and that P is in the Psat unit")


# |sum K x - 1| below this is a converged saturation temperature: K is O(1), so
# this is ~1e-10 degC of temperature error — far tighter than any solver tol.
_T_FTOL = 1e-11


def bubble_T(x, P, antoine, lo=-100.0, hi=500.0, gamma_fn=None, phi_fn=None,
             T_guess=None):
    """Bubble-point temperature: T such that sum_i K_i(T) x_i = 1.

    lo/hi bracket the root (in the Antoine fit's temperature unit); hi auto-widens
    via _solve_T for very high boilers. gamma_fn/phi_fn (optional) make the
    K-values non-ideal (activity model / vapour-phase EOS, see k_values).
    T_guess (optional): a nearby temperature to secant from — same root, far fewer
    K-value evaluations (see _solve_T).
    """
    x = np.asarray(x, float)

    def f(T):
        v = float(np.sum(k_values(T, P, antoine, gamma_fn, x, phi_fn) * x)
                  - 1.0)
        # psat overflows to inf far above saturation (PLXANT exp), turning f
        # into nan and silently breaking _solve_T's bracket test; sum(K x) - 1
        # is hugely positive out there, so say so instead.
        return v if math.isfinite(v) else 1e12

    return _solve_T(f, lo, hi, guess=T_guess, walk_lo=True)


def dew_T(y, P, antoine, lo=-100.0, hi=500.0, gamma_fn=None, phi_fn=None):
    """Dew-point temperature: T such that sum_i y_i / K_i(T) = 1.

    ponytail: with an activity model, gamma is evaluated at the vapour composition
    as a proxy for the (unknown) liquid in equilibrium — exact for ideal VLE,
    approximate otherwise (a rigorous dew point needs an inner liquid-comp solve).
    """
    y = np.asarray(y, float)

    def f(T):
        return float(np.sum(y / k_values(T, P, antoine, gamma_fn, y, phi_fn))
                     - 1.0)

    return _solve_T(f, lo, hi)


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

    This is the HYSYS-modified NRTL form, matching src/native/nifco.f90's NRTL
    subroutine (tau = a + b/T with T in K, G = exp(-alpha*tau), alpha the
    non-randomness). A future compiled nifco NRTL/fugacity model is a drop-in
    replacement for this closure — same (x, T) -> gamma signature.

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


def wilson_gamma(x, Lam):
    """Wilson activity coefficients for a multicomponent liquid.

    x    (n,) liquid mole fractions
    Lam  (n,n) Wilson Lambda_ij (Lam_ii = 1)
    ln gamma_i = 1 - ln(S_i) - sum_k x_k Lam_ki / S_k,  S_i = sum_j x_j Lam_ij.
    """
    x = np.asarray(x, float)
    Lam = np.asarray(Lam, float)
    S = Lam @ x
    ln_gamma = 1.0 - np.log(S) - (x / S) @ Lam
    return np.exp(ln_gamma)


def wilson_gamma_fn(a, b, t_to_K=lambda T: T + 273.15):
    """gamma_fn(x, T) closure for Wilson with ln Lambda_ij = a_ij + b_ij/T_K
    (the Aspen WILSON form; a_ii = b_ii = 0 gives Lambda_ii = 1)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    def gamma_fn(x, T):
        return wilson_gamma(x, np.exp(a + b / t_to_K(T)))

    return gamma_fn


def uniquac_gamma(x, r, q, tau, z=10.0):
    """UNIQUAC activity coefficients (combinatorial + residual).

    x (n,) mole fractions; r, q (n,) structural volume/area parameters;
    tau (n,n) interaction terms (tau_ii = 1); z the coordination number (10).
    Ratio forms (Phi_i/x_i = r_i / sum x r) keep the pure/zero-x limits exact.
    """
    x = np.asarray(x, float)
    r = np.asarray(r, float)
    q = np.asarray(q, float)
    tau = np.asarray(tau, float)
    xr = x @ r
    xq = x @ q
    phi_x = r / xr                       # Phi_i / x_i
    theta_phi = (q / xq) / phi_x         # theta_i / Phi_i
    l = 0.5 * z * (r - q) - (r - 1.0)
    ln_gC = np.log(phi_x) + 0.5 * z * q * np.log(theta_phi) + l - phi_x * (x @ l)
    theta = q * x / xq
    S = theta @ tau                      # S_i = sum_j theta_j tau_ji
    ln_gR = q * (1.0 - np.log(S) - tau @ (theta / S))
    return np.exp(ln_gC + ln_gR)


def uniquac_gamma_fn(r, q, a, b, t_to_K=lambda T: T + 273.15):
    """gamma_fn(x, T) closure for UNIQUAC with tau_ij = exp(a_ij + b_ij/T_K)
    (the Aspen UNIQ form; a_ii = b_ii = 0 gives tau_ii = 1)."""
    r = np.asarray(r, float)
    q = np.asarray(q, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    def gamma_fn(x, T):
        return uniquac_gamma(x, r, q, np.exp(a + b / t_to_K(T)))

    return gamma_fn


def margules_gamma(x, A):
    """Two-suffix (one-constant) Margules, multicomponent regular-solution form:
    G^E/RT = 1/2 sum_ij A_ij x_i x_j (A symmetric, A_ii = 0), so
    ln gamma_i = sum_j A_ij x_j - G^E/RT. Binary limit: ln gamma_1 = A x_2**2."""
    x = np.asarray(x, float)
    A = np.asarray(A, float)
    Ax = A @ x
    return np.exp(Ax - 0.5 * (x @ Ax))


def margules_gamma_fn(A):
    """gamma_fn(x, T) closure for two-suffix Margules (A dimensionless,
    temperature-independent — the teaching model)."""
    A = np.asarray(A, float)
    return lambda x, T: margules_gamma(x, A)


# --- UNIFAC (group-contribution activity coefficients) --------------------
# Classic UNIFAC-VLE. Combinatorial term is identical to UNIQUAC (with r/q
# built from group counts); residual term is the group-activity sum. See
# core/data/unifac_groups.json for the curated parameter DB.

def _unifac_group_ln_gamma(nu_vec, Q, Psi):
    """ln Gamma_k for every subgroup, given a group-count/weight vector.
    nu_vec: (m,) group weights (mixture: sum_i x_i nu_i^k; pure: nu_i^k).
    Q: (m,) subgroup areas. Psi: (m,m) interaction matrix exp(-a_mn/T)."""
    Xg = nu_vec / nu_vec.sum()
    QX = Q * Xg
    theta = QX / QX.sum()
    S = theta @ Psi                       # S_k = sum_m theta_m Psi_mk (also denom_m)
    term2 = Psi @ (theta / S)             # sum_m Psi_km theta_m / S_m
    return Q * (1.0 - np.log(S) - term2)


def unifac_gamma(x, nu, R, Q, main_idx, a_mn, T_K):
    """UNIFAC activity coefficients.
    x: (n,) liquid mole fractions. nu: (n,m) subgroup counts per species.
    R,Q: (m,) subgroup volume/area. main_idx: (m,) main-group index per subgroup.
    a_mn: (g,g) main-group interaction energies [K]. Returns (n,) gamma."""
    x = np.asarray(x, float)
    x = np.clip(x, 1e-12, None)
    x = x / x.sum()
    nu = np.asarray(nu, float)

    # Combinatorial (Staverman-Guggenheim), z = 10.
    r = nu @ R
    q = nu @ Q
    phi = r * x / (r @ x)
    theta = q * x / (q @ x)
    l = 5.0 * (r - q) - (r - 1.0)
    ln_c = np.log(phi / x) + 5.0 * q * np.log(theta / phi) + l - (phi / x) * (x @ l)

    # Residual. Psi over subgroups via their main groups.
    Psi = np.exp(-a_mn[np.ix_(main_idx, main_idx)] / T_K)
    lnG_mix = _unifac_group_ln_gamma(x @ nu, Q, Psi)
    ln_r = np.empty(len(x))
    for i in range(len(x)):
        lnG_pure = _unifac_group_ln_gamma(nu[i], Q, Psi)
        ln_r[i] = float(nu[i] @ (lnG_mix - lnG_pure))

    return np.exp(ln_c + ln_r)


@lru_cache(maxsize=1)
def load_unifac_db():
    """Parsed core/data/unifac_groups.json (cached)."""
    import json, os
    with open(os.path.join(os.path.dirname(__file__), "data",
                           "unifac_groups.json")) as fh:
        return json.load(fh)


def unifac_gamma_fn(species_groups, db, t_to_K=lambda T: T + 273.15, names=None):
    """gamma_fn(x, T) closure for UNIFAC. `species_groups` is a list (one per
    component, same order as x) of {subgroup_name: count}. `db` is the parsed
    unifac_groups.json. `names` (optional) labels the components in error
    messages. Raises if a species has no groups, names an unknown subgroup, or
    needs a main-group interaction pair the published table does not have —
    nothing is silently treated as ideal."""
    sub = db["subgroups"]
    inter = db["interactions"]
    label = list(names) if names else None

    # Union of subgroups actually used, in a stable order.
    used = []
    for g in species_groups:
        if not g:
            raise ValueError("UNIFAC needs at least one group per component "
                             "(Initialization → Species → UNIFAC Groups).")
        for name in g:
            if name not in sub:
                raise ValueError(f"Unknown UNIFAC subgroup '{name}' — not in the "
                                 "group DB (core/data/unifac_groups.json).")
            if name not in used:
                used.append(name)

    R = np.array([sub[s][2] for s in used], float)
    Q = np.array([sub[s][3] for s in used], float)
    mains = [sub[s][1] for s in used]
    main_names = sorted(set(mains))
    gidx = {m: i for i, m in enumerate(main_names)}
    main_idx = np.array([gidx[m] for m in mains], int)

    # Only 1270 of the 2862 possible main-group pairs are published. A missing
    # pair used to default to a = 0 (Psi = 1), which is an ideal residual for
    # that pair — invisible, and wrong more often than not. Say so instead.
    g = len(main_names)
    a_mn = np.zeros((g, g))
    missing = []
    for mi in main_names:
        row = inter.get(mi, {})
        for mj in main_names:
            if mi == mj:
                continue                      # a_mm = 0 by definition
            if mj in row:
                a_mn[gidx[mi], gidx[mj]] = row[mj]
            else:
                missing.append((mi, mj))
    if missing:
        mi, mj = missing[0]
        who = [n for n, grp in zip(label or [f"component {i}" for i in
                                             range(len(species_groups))],
                                   species_groups)
               if any(sub[s][1] in (mi, mj) for s in grp)]
        raise ValueError(
            f"UNIFAC has no published interaction parameter for main groups "
            f"{mi}/{mj} (needed by {', '.join(who)}). The classic UNIFAC-VLE "
            f"table covers 1270 of 2862 pairs; this chemistry is outside it, "
            f"so choose another activity model rather than run it as ideal.")

    nu = np.array([[grp.get(s, 0) for s in used] for grp in species_groups], float)

    def gamma_fn(x, T):
        return unifac_gamma(x, nu, R, Q, main_idx, a_mn, t_to_K(T))

    return gamma_fn


def _srk_z(A, B):
    """Vapour compressibility from Z^3 - Z^2 + (A-B-B^2) Z - A B = 0,
    elementwise over arrays.

    Analytic (Cardano/trigonometric) with a two-step Newton polish — np.roots'
    companion-matrix *eigensolve* here was 70%% of a whole column solve.
    Returns (Z, ok); ok False marks "no vapour-like root", reproducing the
    np.roots-based selection exactly: real roots only, filtered to > B, the
    single-root-below-0.5 case rejected (see srk_phi for the physics).
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    c1 = A - B - B * B
    c0 = -A * B
    p = c1 - 1.0 / 3.0                       # depressed cubic: Z = t + 1/3
    q = c1 / 3.0 + c0 - 2.0 / 27.0
    disc = 0.25 * q * q + p ** 3 / 27.0
    one = disc > 0.0                         # one real root vs three
    s = np.sqrt(np.where(one, disc, 0.0))
    t1 = np.cbrt(-0.5 * q + s) + np.cbrt(-0.5 * q - s)
    pm = np.where(one, -1.0, np.minimum(p, -1e-30))      # p <= 0 when disc <= 0
    m = 2.0 * np.sqrt(-pm / 3.0)
    th = np.arccos(np.clip(3.0 * q / (pm * m), -1.0, 1.0))
    t3 = m[..., None] * np.cos((th[..., None]
                                - 2.0 * np.pi * np.array([0.0, 1.0, 2.0])) / 3.0)
    pad = np.full(np.shape(t1) + (2,), -np.inf)
    cand = np.where(one[..., None],
                    np.concatenate([t1[..., None], pad], axis=-1),
                    t3) + 1.0 / 3.0
    with np.errstate(invalid="ignore"):      # the -inf pad rows produce nan steps,
        for _ in range(2):                   # which the where() discards
            f = ((cand - 1.0) * cand + c1[..., None]) * cand + c0[..., None]
            fp = (3.0 * cand - 2.0) * cand + c1[..., None]
            step = np.where(np.isfinite(cand) & (fp != 0.0),
                            f / np.where(fp != 0.0, fp, 1.0), 0.0)
            cand = cand - step
    valid = cand > B[..., None]
    nsel = valid.sum(axis=-1)
    Z = np.where(valid, cand, -np.inf).max(axis=-1)
    ok = ~((nsel == 0) | ((nsel == 1) & (Z < 0.5)))
    return Z, ok


def srk_phi(y, T_K, P_Pa, tc, pc, omega):
    """Soave-Redlich-Kwong vapour-mixture fugacity coefficients.

    y     (n,) vapour mole fractions
    T_K   temperature [K];  P_Pa pressure [Pa]
    tc    (n,) critical temperatures [K]; pc (n,) critical pressures [Pa]
    omega (n,) acentric factors

    kij = 0 mixing (a_mix = (sum y_i sqrt(a_i))^2, b_mix = sum y_i b_i);
    Z is the largest real root of Z^3 - Z^2 + (A-B-B^2)Z - AB = 0 (vapour).
    ponytail: kij=0 is fine for hydrocarbon/hydrocarbon; add a kij matrix
    argument if polar/light-gas pairs ever need it.
    """
    y = np.asarray(y, float)
    tc = np.asarray(tc, float)
    pc = np.asarray(pc, float)
    w = np.asarray(omega, float)
    m = 0.480 + 1.574 * w - 0.176 * w * w
    alpha = (1.0 + m * (1.0 - np.sqrt(T_K / tc))) ** 2
    ai = 0.42748 * (R_GAS * tc) ** 2 / pc * alpha
    bi = 0.08664 * R_GAS * tc / pc
    sa = np.sqrt(ai)
    ysa = float(y @ sa)
    bmix = float(y @ bi)
    A = ysa ** 2 * P_Pa / (R_GAS * T_K) ** 2
    B = bmix * P_Pa / (R_GAS * T_K)
    # No vapour-like root means no vapour exists at (T, P) — e.g. a bubble-T
    # root-search probing far below the true bubble point, where the lone real
    # root is liquid-like (Z ~ B). Using it would return phi ~ 1e-3 and forge a
    # spurious low-T bubble root, so fall back to ideal gas: the correction is
    # meaningless where there is no vapour. ponytail: Z<0.5 vapour test is fine
    # for the sub-10-bar gamma-phi scope; Mathias pseudo-root extrapolation is
    # the upgrade if high-pressure columns ever land.
    Zr, ok = _srk_z(A, B)
    if not ok:
        return np.ones_like(y)
    Z = float(Zr)
    lnphi = ((bi / bmix) * (Z - 1.0) - math.log(Z - B)
             - (A / B) * (2.0 * sa / ysa - bi / bmix) * math.log(1.0 + B / Z))
    return np.exp(lnphi)


def srk_phi_fn(tc, pc, omega, t_to_K=lambda T: T + 273.15, p_to_Pa=133.322):
    """phi_fn(y, T, P) closure for SRK (see k_values). tc in K, pc in bar,
    omega dimensionless; T/P arrive in the Antoine fit's units and are
    converted via t_to_K / p_to_Pa (default: degC and mmHg)."""
    tc = np.asarray(tc, float)
    pc_Pa = np.asarray(pc, float) * 1.0e5
    omega = np.asarray(omega, float)

    def phi_fn(y, T, P):
        return srk_phi(y, t_to_K(T), P * p_to_Pa, tc, pc_Pa, omega)

    def pure(T, psat):
        """All pure-component phi^sat in one vectorised call — the k_values
        fast path (it otherwise loops n one-hot srk_phi calls per K-value).
        Pure i: a_mix = a_i, b_mix = b_i, each at its own pressure psat[i]."""
        T_K = t_to_K(T)
        P = np.asarray(psat, float) * p_to_Pa
        m = 0.480 + 1.574 * omega - 0.176 * omega * omega
        alpha = (1.0 + m * (1.0 - np.sqrt(T_K / tc))) ** 2
        ai = 0.42748 * (R_GAS * tc) ** 2 / pc_Pa * alpha
        bi = 0.08664 * R_GAS * tc / pc_Pa
        A = ai * P / (R_GAS * T_K) ** 2
        B = bi * P / (R_GAS * T_K)
        Z, ok = _srk_z(A, B)
        Zs = np.where(ok, Z, 2.0)            # dummies where the ideal-gas
        Bs = np.where(ok, B, 1.0)            # fallback applies (discarded)
        lnphi = (Zs - 1.0) - np.log(Zs - Bs) - (A / Bs) * np.log(1.0 + Bs / Zs)
        return np.where(ok, np.exp(lnphi), 1.0)

    phi_fn.pure = pure
    return phi_fn


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

    # High boiler whose bubble T exceeds the nominal hi=500 -> _solve_T must widen
    # the bracket instead of raising brentq's "different signs". Antoine (mmHg):
    high = np.array([(7.0, 3000.0, 200.0)])
    Thi = bubble_T(np.array([1.0]), 760.0, high)
    assert Thi > 500.0, f"expected widening past 500, got {Thi:.0f}"

    # antoine_Tsat inverts the fit: benzene boils at ~80.1 degC at 760 mmHg
    assert abs(antoine_Tsat(760.0, abc[0]) - 80.1) < 0.5

    # Clausius-Clapeyron latent heat: benzene ~30.8 kJ/mol at its boiling point
    lam = latent_heat(80.1, abc)
    assert 28e3 < lam[0] < 33e3, f"benzene latent heat {lam[0]:.0f} J/mol"
    assert lam[2] > lam[0], "heavier aromatic should have larger latent heat"

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

    # --- Wilson ------------------------------------------------------------
    Lam = np.array([[1.0, 0.5], [0.3, 1.0]])
    assert abs(wilson_gamma([1.0, 0.0], Lam)[0] - 1.0) < 1e-12, "Wilson pure"
    # binary infinite dilution: ln gamma1_inf = 1 - ln(Lam12) - Lam21
    g1inf = wilson_gamma([0.0, 1.0], Lam)[0]
    assert abs(np.log(g1inf) - (1.0 - np.log(0.5) - 0.3)) < 1e-12

    # SVA 7e Table 12.1 ethanol(1)/water(2): a12=382.30, a21=955.45 cal/mol,
    # V1=58.68, V2=18.07 cm3/mol; Lam_ij = (Vj/Vi) exp(-a_ij/(R T_K)) maps to
    # the a + b/T_K closure as a = ln(Vj/Vi), b = -a_ij/R (R in cal/mol-K).
    Rcal = 1.98721
    aW = np.array([[0.0, np.log(18.07 / 58.68)], [np.log(58.68 / 18.07), 0.0]])
    bW = np.array([[0.0, -382.30 / Rcal], [-955.45 / Rcal, 0.0]])
    gW = wilson_gamma_fn(aW, bW)
    etoh_h2o = np.array([(8.20417, 1642.89, 230.300),   # ethanol, mmHg/degC
                         (8.07131, 1730.63, 233.426)])  # water
    xs = np.linspace(0.02, 0.998, 245)
    Ts = np.array([bubble_T(np.array([x, 1 - x]), 760.0, etoh_h2o, gamma_fn=gW)
                   for x in xs])
    k = int(np.argmin(Ts))
    assert 0 < k < len(xs) - 1, "azeotrope must be interior"
    assert abs(xs[k] - 0.894) < 0.03 and abs(Ts[k] - 78.15) < 0.5, (xs[k], Ts[k])

    # --- UNIQUAC -----------------------------------------------------------
    r_u = np.array([2.1055, 0.9200])   # ethanol, water
    q_u = np.array([1.9720, 1.4000])
    tau = np.array([[1.0, 0.8], [0.6, 1.0]])
    assert abs(uniquac_gamma([1.0, 0.0], r_u, q_u, tau)[0] - 1.0) < 1e-12
    assert abs(uniquac_gamma([0.0, 1.0], r_u, q_u, tau)[1] - 1.0) < 1e-12
    # with r = q = 1 the combinatorial part vanishes and the residual reduces
    # to Wilson with Lam_ij = tau_ji — a hand-checkable algebraic identity
    ones = np.ones(2)
    xu = np.array([0.3, 0.7])
    assert np.allclose(uniquac_gamma(xu, ones, ones, tau),
                       wilson_gamma(xu, tau.T), atol=1e-12)

    # --- two-suffix Margules -------------------------------------------------
    A_m = np.array([[0.0, 1.5], [1.5, 0.0]])
    xm = np.array([0.25, 0.75])
    gm = margules_gamma(xm, A_m)
    assert abs(np.log(gm[0]) - 1.5 * 0.75 ** 2) < 1e-12, "ln g1 = A x2^2"
    assert abs(np.log(gm[1]) - 1.5 * 0.25 ** 2) < 1e-12
    assert abs(margules_gamma([0.0, 1.0], A_m)[1] - 1.0) < 1e-12

    # --- SRK vapour-phase phi ------------------------------------------------
    # propane / n-butane: Tc [K], Pc [bar], omega; Antoine mmHg/degC
    tc_s = np.array([369.83, 425.12])
    pc_s = np.array([42.48, 37.96])
    om_s = np.array([0.152, 0.200])
    c3c4 = np.array([(6.80398, 803.810, 246.99),    # propane
                     (6.80896, 935.860, 238.73)])   # n-butane
    pfn = srk_phi_fn(tc_s, pc_s, om_s)
    yv = np.array([0.5, 0.5])
    # low-pressure limit: phi -> 1
    assert np.allclose(pfn(yv, 20.0, 7.6), 1.0, atol=2e-3), pfn(yv, 20.0, 7.6)
    # subcritical vapour at 4 atm is denser than ideal: phi < 1, Z < 1
    phi4 = pfn(yv, 20.0, 4.0 * 760.0)
    assert np.all(phi4 < 1.0) and np.all(phi4 > 0.7), phi4
    # heavier component departs more from ideality
    assert phi4[1] < phi4[0], phi4
    # the vectorised pure-component fast path equals the one-hot loop
    psat_s = antoine_psat(20.0, c3c4)
    loop_s = np.array([pfn(np.eye(2)[i], 20.0, psat_s[i])[i] for i in range(2)])
    assert np.allclose(pfn.pure(20.0, psat_s), loop_s, atol=1e-9), \
        (pfn.pure(20.0, psat_s), loop_s)
    # pure-component saturation: phi_sat and phi_V cancel exactly in k_values,
    # so the pure bubble point is unchanged by the phi correction
    Tp_id = bubble_T(np.array([1.0, 0.0]), 760.0, c3c4)
    Tp_srk = bubble_T(np.array([1.0, 0.0]), 760.0, c3c4, phi_fn=pfn)
    assert abs(Tp_id - Tp_srk) < 1e-6, (Tp_id, Tp_srk)
    # gamma-phi at 4 atm: the vapour-phase correction compresses the C3/C4
    # relative volatility vs Raoult (classic direction of EOS non-ideality)
    P4 = 4.0 * 760.0
    x4 = np.array([0.5, 0.5])
    T4 = bubble_T(x4, P4, c3c4, phi_fn=pfn)
    K_id4 = k_values(T4, P4, c3c4)
    K_sr4 = k_values(T4, P4, c3c4, None, x4, pfn)
    a_id = K_id4[0] / K_id4[1]
    a_sr = K_sr4[0] / K_sr4[1]
    assert 1.0 < a_sr < a_id, (a_sr, a_id)

    # --- UNIFAC --------------------------------------------------------------
    import json, os
    with open(os.path.join(os.path.dirname(__file__), "data",
                           "unifac_groups.json")) as fh:
        udb = json.load(fh)
    # n-hexane / n-heptane: same groups -> residual 0; only the small (real)
    # combinatorial size term remains, so gamma sits just below 1
    ufn_hh = unifac_gamma_fn([{"CH3": 2, "CH2": 4}, {"CH3": 2, "CH2": 5}], udb)
    assert np.allclose(ufn_hh([0.4, 0.6], 60.0), 1.0, atol=1e-2), ufn_hh([0.4, 0.6], 60.0)
    # pure species: gamma = 1 exactly
    assert abs(ufn_hh([1.0, 0.0], 60.0)[0] - 1.0) < 1e-9
    # ethanol / water: strong positive deviation; gamma_EtOH^inf in the 3-8 band
    ufn_ew = unifac_gamma_fn([{"CH3": 1, "CH2": 1, "OH": 1}, {"H2O": 1}], udb)
    g_inf = ufn_ew([1e-6, 1 - 1e-6], 70.0)[0]
    assert 3.0 < g_inf < 8.0, g_inf
    # UNIFAC predicts the EtOH/water azeotrope near x_EtOH ~ 0.9
    xg = np.linspace(0.02, 0.998, 245)
    Tg = np.array([bubble_T(np.array([x, 1 - x]), 760.0, etoh_h2o, gamma_fn=ufn_ew)
                   for x in xg])
    kg = int(np.argmin(Tg))
    assert 0 < kg < len(xg) - 1 and abs(xg[kg] - 0.90) < 0.06, (xg[kg], Tg[kg])

    print(f"thermodynamics self-check OK (mix bubble T = {T:.1f} degC; "
          f"NRTL gamma_inf = {g_dilute[0]:.2f}; Wilson EtOH/H2O azeotrope "
          f"x={xs[k]:.3f} @ {Ts[k]:.2f} degC; SRK C3/C4 alpha "
          f"{a_id:.3f} -> {a_sr:.3f} @ 4 atm; UNIFAC EtOH/H2O gamma_inf "
          f"{g_inf:.2f}, azeotrope x={xg[kg]:.3f})")


if __name__ == "__main__":
    _demo()
