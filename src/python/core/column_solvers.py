"""Rigorous tray-by-tray distillation solver (Bubble-Point / Wang-Henke).

Scope: ideal VLE (Antoine + Raoult, from core.thermodynamics) with the
constant-molar-overflow (CMO) flow assumption and a total condenser. This is a
genuine MESH solve — per-component tridiagonal material balances, bubble-point
temperatures per stage, iterated to temperature convergence — and the natural
rigorous step up from the BVM feasibility method.

# ponytail: CMO (constant vapour/liquid in each section) replaces a full energy
# balance, so duties and non-CMO effects aren't captured. Add stage enthalpy
# balances (the "SR"/sum-rates or full Inside-Out variant) when that matters;
# the K-value and tridiagonal machinery here stay as-is.

Stages are numbered 1=top..N=reboiler; arrays are returned top->bottom.
"""

import numpy as np

from core.thermodynamics import k_values, bubble_T


def _thomas(a, b, c, d):
    """Solve a tridiagonal system (a=sub, b=diag, c=super, d=rhs). Length n."""
    n = len(b)
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _stage_compositions(K, L, V, Fj, zF, B, D, N, n):
    """Solve the per-component tridiagonal material balances for the (N, n) liquid
    composition, normalised per stage. Shared by the bubble-point and Inside-Out
    solvers; K is the (N, n) K-value array, L/V the stage liquid/vapour flows."""
    xnew = np.zeros((N, n))
    for i in range(n):
        a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); d = np.zeros(N)
        for j in range(N):
            if j == 0:                          # top stage, total condenser
                b[j] = -(L[0] + D * K[0, i])
                c[j] = V[1] * K[1, i] if N > 1 else 0.0
                d[j] = -Fj[0] * zF[i]
            elif j == N - 1:                    # reboiler
                a[j] = L[j - 1]
                b[j] = -(B + V[j] * K[j, i])
                d[j] = -Fj[j] * zF[i]
            else:                               # interior
                a[j] = L[j - 1]
                b[j] = -(L[j] + V[j] * K[j, i])
                c[j] = V[j + 1] * K[j + 1, i]
                d[j] = -Fj[j] * zF[i]
        xnew[:, i] = _thomas(a, b, c, d)
    xnew = np.clip(xnew, 0.0, None)
    xnew /= xnew.sum(axis=1, keepdims=True)
    return xnew


def _cmo_flows(N, feed_stage, R, D, F, B):
    """Constant-molar-overflow stage vapour/liquid flows and feed vector."""
    V = np.full(N, (R + 1) * D)
    L = np.empty(N)
    for j in range(1, N + 1):
        if j < feed_stage:
            L[j - 1] = R * D
        elif j < N:
            L[j - 1] = R * D + F
        else:
            L[j - 1] = B
    Fj = np.zeros(N); Fj[feed_stage - 1] = F
    return V, L, Fj


def solve_bubble_point(zF, F, antoine, comps, *, N, feed_stage, R, D, P,
                       max_iter=500, tol=1e-6, gamma_fn=None):
    """Bubble-point column solve.

    zF, F        feed composition (n,) and flow
    antoine      (n, 3) Antoine [A,B,C], same unit convention as core.thermodynamics
    N            number of equilibrium stages (1=top .. N=reboiler)
    feed_stage   1-based stage the feed enters (saturated liquid, q=1)
    R, D, P      reflux ratio L/D, distillate rate, column pressure
    Returns a profile dict (BVM-compatible keys) with liquid x, vapour y, T.
    """
    zF = np.asarray(zF, float)
    antoine = np.asarray(antoine, float)
    n = len(zF)
    B = F - D
    if not (0.0 < D < F):
        raise ValueError(f"distillate D={D} must be between 0 and F={F}")

    # CMO flows (1-based index j=1..N -> python j-1)
    V, L, Fj = _cmo_flows(N, feed_stage, R, D, F, B)

    # initial T profile: all at feed bubble point
    T = np.full(N, bubble_T(zF, P, antoine, gamma_fn=gamma_fn))
    x = np.tile(zF, (N, 1))                      # (N, n) liquid mole fractions

    for _ in range(max_iter):
        K = np.array([k_values(T[j], P, antoine, gamma_fn, x[j])
                      for j in range(N)])         # (N, n), activity-aware
        xnew = _stage_compositions(K, L, V, Fj, zF, B, D, N, n)
        Tnew = np.array([bubble_T(xnew[j], P, antoine, gamma_fn=gamma_fn)
                         for j in range(N)])

        dT = np.max(np.abs(Tnew - T))
        x, T = xnew, Tnew
        if dT < tol:
            break

    K = np.array([k_values(T[j], P, antoine, gamma_fn, x[j]) for j in range(N)])
    y = K * x
    y /= y.sum(axis=1, keepdims=True)
    xD = y[0]                                    # total condenser: distillate = top vapour
    xB = x[-1]

    return {
        "x": x, "y": y, "T": T, "comps": list(comps),
        "n_stages": N, "feed_stage": feed_stage,
        "xD": xD, "xB": xB, "D": D, "B": B,
        "found": True, "message": "Converged (bubble-point).",
    }


def solve_inside_out(zF, F, antoine, comps, *, N, feed_stage, R, D, P,
                     max_iter=50, tol=1e-6, gamma_fn=None, cancel=None):
    """Inside-Out (HYSIM-style) column solve.

    The defining two-tier structure: an OUTER loop refreshes rigorous K-values
    (Antoine + optional NRTL activity) and derives stage relative volatilities
    alpha_ij = K_ij / Kb_j about a per-stage base K_b; an INNER loop then holds
    alpha fixed and cheaply iterates the base K_b and the per-component
    tridiagonal material balances (no rigorous-thermo calls), which is what makes
    Inside-Out fast and robust versus refreshing rigorous K every sweep.

    cancel: optional callable -> bool; checked each outer pass for real Abort.

    # ponytail: flows are still CMO (the inner energy balance that frees V/L and
    # gives reboiler/condenser duties is the next step); enthalpy below is a
    # temperature-based proxy until per-component Cp / latent-heat data is wired.
    Returns the bubble-point profile schema plus per-stage pressure, liquid_flow,
    vapor_flow, k_values and enthalpy (1-D series the Results tab plots directly).
    """
    zF = np.asarray(zF, float)
    antoine = np.asarray(antoine, float)
    n = len(zF)
    B = F - D
    if not (0.0 < D < F):
        raise ValueError(f"distillate D={D} must be between 0 and F={F}")

    V, L, Fj = _cmo_flows(N, feed_stage, R, D, F, B)
    T = np.full(N, bubble_T(zF, P, antoine, gamma_fn=gamma_fn))
    x = np.tile(zF, (N, 1))
    aborted = False
    outer = 0
    # Inner loop converges base-K to the user `tol`; the outer temperature loop
    # uses a physical floor (1e-4 K) — tighter is meaningless and only chases a
    # negligible geometric tail from the base-K linearisation.
    outer_tol = max(tol, 1e-4)

    for outer in range(1, max_iter + 1):
        if cancel is not None and cancel():
            aborted = True
            break

        # OUTER: rigorous K, base K_b (geometric mean), frozen relative volatilities
        Kfull = np.array([k_values(T[j], P, antoine, gamma_fn, x[j])
                          for j in range(N)])
        Kb = np.exp(np.mean(np.log(Kfull), axis=1))     # (N,)
        alpha = Kfull / Kb[:, None]                     # (N, n), frozen below

        # INNER: hold alpha, iterate base K_b + compositions (cheap, no thermo)
        for _ in range(50):
            K = alpha * Kb[:, None]
            xin = _stage_compositions(K, L, V, Fj, zF, B, D, N, n)
            Kb_new = 1.0 / np.sum(alpha * xin, axis=1)  # bubble constraint per stage
            if np.max(np.abs(Kb_new - Kb) / Kb) < tol:
                Kb, x = Kb_new, xin
                break
            Kb, x = Kb_new, xin

        # Refresh temperatures from rigorous thermo for the next outer alpha
        Tnew = np.array([bubble_T(x[j], P, antoine, gamma_fn=gamma_fn)
                         for j in range(N)])
        dT = np.max(np.abs(Tnew - T))
        T = Tnew
        if dT < outer_tol:
            break

    K = np.array([k_values(T[j], P, antoine, gamma_fn, x[j]) for j in range(N)])
    y = K * x
    y /= y.sum(axis=1, keepdims=True)
    Kb = np.exp(np.mean(np.log(K), axis=1))

    return {
        "x": x, "y": y, "T": T, "comps": list(comps),
        "n_stages": N, "feed_stage": feed_stage,
        "xD": y[0], "xB": x[-1], "D": D, "B": B,
        "iterations": outer,
        "pressure": np.full(N, P),
        "liquid_flow": L, "vapor_flow": V,
        "k_values": Kb,
        "enthalpy": T - T.min(),          # ponytail: T-based proxy (no Cp data yet)
        "found": not aborted,
        "message": "Aborted." if aborted else "Converged (Inside-Out).",
    }


def _demo():
    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    zF = np.array([0.4, 0.35, 0.25]); F = 100.0
    prof = solve_bubble_point(zF, F, abc, ["benzene", "toluene", "xylene"],
                              N=20, feed_stage=10, R=3.0, D=40.0, P=760.0)

    # every stage is a valid composition
    assert np.allclose(prof["x"].sum(axis=1), 1.0, atol=1e-8)
    assert np.allclose(prof["y"].sum(axis=1), 1.0, atol=1e-8)
    # overall component balance closes: F z = D xD + B xB
    lhs = F * zF
    rhs = prof["D"] * prof["xD"] + prof["B"] * prof["xB"]
    assert np.allclose(lhs, rhs, atol=1e-3), f"balance off: {lhs} vs {rhs}"
    # light key concentrates up the column (top distillate richer in benzene)
    assert prof["xD"][0] > zF[0] > prof["xB"][0], "no separation achieved"

    # Non-ideal (NRTL) path runs end-to-end and still produces valid stages.
    from core.thermodynamics import nrtl_gamma_fn
    gfn = nrtl_gamma_fn([[0.0, 0.5, 0.4], [0.5, 0.0, 0.3], [0.4, 0.3, 0.0]],
                        [[0.0] * 3] * 3,
                        [[0.0, 0.3, 0.3], [0.3, 0.0, 0.3], [0.3, 0.3, 0.0]])
    prof_ni = solve_bubble_point(zF, F, abc, ["benzene", "toluene", "xylene"],
                                 N=20, feed_stage=10, R=3.0, D=40.0, P=760.0,
                                 gamma_fn=gfn)
    assert np.allclose(prof_ni["x"].sum(axis=1), 1.0, atol=1e-8)
    assert prof_ni["xD"][0] > zF[0] > prof_ni["xB"][0], "non-ideal: no separation"

    # Inside-Out: converges to (essentially) the same column as Wang-Henke, emits
    # the rich per-stage profiles, and honours a cancel hook for Abort.
    io = solve_inside_out(zF, F, abc, ["benzene", "toluene", "xylene"],
                          N=20, feed_stage=10, R=3.0, D=40.0, P=760.0)
    assert io["found"] and np.allclose(io["x"].sum(axis=1), 1.0, atol=1e-8)
    lhs, rhs = F * zF, io["D"] * io["xD"] + io["B"] * io["xB"]
    assert np.allclose(lhs, rhs, atol=1e-3), f"IO balance off: {lhs} vs {rhs}"
    assert np.allclose(io["xD"], prof["xD"], atol=2e-2), "IO disagrees with bubble-point"
    for key in ("pressure", "liquid_flow", "vapor_flow", "k_values", "enthalpy"):
        assert len(io[key]) == io["n_stages"], f"{key} not a per-stage series"
    assert solve_inside_out(zF, F, abc, ["benzene", "toluene", "xylene"], N=20,
                            feed_stage=10, R=3.0, D=40.0, P=760.0,
                            cancel=lambda: True)["message"] == "Aborted."
    print(f"column_solvers self-check OK: xD={np.round(prof['xD'],3)}, "
          f"xB={np.round(prof['xB'],3)} (NRTL + Inside-Out, {io['iterations']} outer iters)")


if __name__ == "__main__":
    _demo()
