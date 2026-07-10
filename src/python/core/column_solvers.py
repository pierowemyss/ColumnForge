"""Rigorous tray-by-tray distillation solvers (Bubble-Point / Inside-Out).

Both solvers consume a `core.solver_input.SolverInput` — the one canonical
column description (per-stage feed/draw/duty/pressure arrays), so multiple
feeds, side draws, feed quality and a pressure profile are just non-zero
entries, not special cases. The historical scalar signature
(zF, F, antoine, comps, N=, feed_stage=, R=, D=, P=) still works as a thin
shim that builds a single-feed SolverInput.

Scope: Antoine/PLXANT + Raoult VLE with optional activity model (gamma_fn) and
the constant-molar-overflow (CMO) flow assumption. This is a genuine MESH solve
— per-component tridiagonal material balances, bubble-point temperatures per
stage, iterated to temperature convergence.

# ponytail: CMO replaces a full stage-enthalpy balance, so inter-stage duties
# are not captured. `flows_hook` on solve_inside_out is the structured seam
# where a stage energy balance plugs in (it receives the current K/x/T state
# and returns updated L/V); until one is wired, CMO flows are used throughout.

Stage conventions:
  * internally stages run 1=top(condenser side)..N=reboiler, matching the
    tridiagonal assembly;
  * RETURNED profiles are ordered top -> bottom: index 0 = condenser/distillate
    ("stage 0 = distillate" everywhere in the UI), and `feed_stage`/`feed_stages`
    are 0-based indices from the top.
"""

import numpy as np

from core.thermodynamics import k_values, bubble_T, latent_heat
from core.solver_input import SolverInput, build_solver_input


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


def _stage_compositions(K, L, V, Fz, W, U, B, D, N, n):
    """Per-component tridiagonal material balances -> (N, n) liquid composition,
    normalised per stage. Shared by both solvers.

    K (N,n) K-values; L/V (N,) inter-stage liquid/vapour leaving each stage;
    Fz (N,n) component molar feed per stage; W/U (N,) liquid/vapour side draws.
    Stage 0 is the top (condenser boundary: net vapour product D at y[0], reflux
    already netted out), stage N-1 the reboiler (liquid product B).
    """
    xnew = np.zeros((N, n))
    for i in range(n):
        a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); d = np.zeros(N)
        for j in range(N):
            if j == 0:                          # top stage / condenser boundary
                b[j] = -(L[0] + W[0] + (D + U[0]) * K[0, i])
                c[j] = V[1] * K[1, i] if N > 1 else 0.0
                d[j] = -Fz[0, i]
            elif j == N - 1:                    # reboiler
                a[j] = L[j - 1]
                b[j] = -(B + W[j] + (V[j] + U[j]) * K[j, i])
                d[j] = -Fz[j, i]
            else:                               # interior
                a[j] = L[j - 1]
                b[j] = -(L[j] + W[j] + (V[j] + U[j]) * K[j, i])
                c[j] = V[j + 1] * K[j + 1, i]
                d[j] = -Fz[j, i]
        xnew[:, i] = _thomas(a, b, c, d)
    xnew = np.clip(xnew, 0.0, None)
    xnew /= xnew.sum(axis=1, keepdims=True)
    return xnew


def _cmo_flows(si: SolverInput):
    """Constant-molar-overflow inter-stage flows from the per-stage arrays.

    Every feed splits by its thermal quality q: q*F joins the liquid below the
    feed stage, (1-q)*F joins the vapour above it. Side draws deplete their own
    phase. Returns (V, L, B): V[j]/L[j] = vapour/liquid leaving stage j
    (0-based from the top), B = bottoms rate.
    """
    N = si.n_stages
    Fj = si.feed.sum(axis=1)                     # total molar feed per stage
    W, U, q = si.liquid_draw, si.vapor_draw, si.q
    R, D = si.R, si.D
    B = float(Fj.sum() - D - W.sum() - U.sum())
    if B <= 0.0:
        raise ValueError(
            f"bottoms rate B={B:.4g} must be positive: distillate + side draws "
            "exceed the total feed")

    V = np.empty(N); L = np.empty(N)
    V[0] = (R + 1.0) * D
    for j in range(1, N):
        # stage balance: V[j] enters stage j-1 from below
        V[j] = V[j - 1] + U[j - 1] - (1.0 - q[j - 1]) * Fj[j - 1]
    L[0] = R * D + q[0] * Fj[0] - W[0]
    for j in range(1, N - 1):
        L[j] = L[j - 1] + q[j] * Fj[j] - W[j]
    if N > 1:
        L[N - 1] = B
    if np.any(V[1:] <= 0.0) or np.any(L[:-1] < 0.0):
        raise ValueError(
            "CMO flows went non-physical (a vapour or liquid flow <= 0): "
            "check reflux, feed quality and side-draw rates")
    return V, L, B


def make_energy_balance(cp_liq, hvap_Tb, tb, tc, relax=0.5,
                        t_to_K=lambda T: T + 273.15):
    """Build a `flows_hook` for solve_inside_out that replaces CMO with a real
    stage-enthalpy balance.

    cp_liq/hvap_Tb/tb/tc are per-component arrays (DB units: J/mol-K, kJ/mol, K,
    K) aligned to si.comps. The returned callable (si, K, x, T, L, V) -> (L, V)
    runs a top-down mass+energy envelope sweep — each stage's two balances solve
    for (L[j], V[j+1]) — then under-relaxes against the incoming flows. Real
    condenser/reboiler duties [J/time on the kmol/h * J/mol basis] land on the
    hook's `.Qc` / `.Qr` after each call, and interheater si.duty[] is consumed.

    ponytail: feed enthalpy uses the local stage T as the feed T (we don't carry
    a separate feed temperature); relax=0.5 damps the L/V update — lower it if a
    steep column oscillates. Upgrade path: a full Naphtali-Sandholm energy row is
    already in Matrix BVM (roadmap Path B).
    """
    from core.enthalpy import enthalpy_fns
    hL_c, hV_c = enthalpy_fns(cp_liq, hvap_Tb, tb, tc)

    class _EnergyBalance:
        Qc = 0.0
        Qr = 0.0

        def __call__(self, si, K, x, T, L, V):
            N = si.n_stages
            TK = t_to_K(np.asarray(T, float))
            hLj = np.empty(N); hVj = np.empty(N); hFj = np.zeros(N)
            for j in range(N):
                hLp = hL_c(TK[j]); hVp = hV_c(TK[j])
                y = K[j] * x[j]; y = y / y.sum()
                hLj[j] = float(x[j] @ hLp)
                hVj[j] = float(y @ hVp)
                Fsum = si.feed[j].sum()
                if Fsum > 0.0:
                    z = si.feed[j] / Fsum
                    q = float(si.q[j])
                    hFj[j] = q * float(z @ hLp) + (1.0 - q) * float(z @ hVp)

            F = si.feed.sum(axis=1)
            W, U, Qd = si.liquid_draw, si.vapor_draw, si.duty
            R, D = si.R, si.D

            # Top boundary: reflux comes down as saturated liquid at stage 0;
            # V0 leaves stage 0 upward to the condenser.
            reflux_in = R * D
            V0 = (R + 1.0) * D if si.condenser != "none" else D
            if si.condenser == "total":
                # Subcooled reflux/distillate: hL(T0-dT) = hL(T0) - cp*dT exactly
                # in this model. Colder reflux removes more heat and condenses
                # extra vapour on stage 0 (raises the effective internal reflux).
                dT = float(getattr(si, "subcooling", 0.0) or 0.0)
                h_reflux = hLj[0] if dT <= 0.0 else float(x[0] @ hL_c(TK[0] - dT))
                hD = h_reflux
                Qc = (R + 1.0) * D * (h_reflux - hVj[0])    # = -(R+1)D * lambda_0
            elif si.condenser == "partial":
                hD = hVj[0]
                Qc = R * D * (hLj[0] - hVj[0])              # = -R*D * lambda_0
            else:                                          # no condenser
                hD = hVj[0]
                Qc = 0.0

            Ln = np.array(L, float); Vn = np.array(V, float)
            Vn[0] = V0
            for j in range(N - 1):
                if j == 0:
                    h_ref0 = hD if si.condenser == "total" else hLj[0]
                    liq_in, h_liq_in, Vj_up = reflux_in, h_ref0, V0
                else:
                    liq_in, h_liq_in, Vj_up = Ln[j - 1], hLj[j - 1], Vn[j]
                A = liq_in + F[j] - Vj_up - W[j] - U[j]     # L[j] - V[j+1]
                Ben = (liq_in * h_liq_in + F[j] * hFj[j] + Qd[j]
                       - Vj_up * hVj[j] - W[j] * hLj[j] - U[j] * hVj[j])
                denom = hLj[j] - hVj[j + 1]                 # ~ -lambda, safely != 0
                Vn[j + 1] = (Ben - A * hLj[j]) / denom
                Ln[j] = Vn[j + 1] + A
            Ln[N - 1] = float(F.sum() - D - W.sum() - U.sum())   # bottoms B

            # Reboiler duty closes the overall energy balance.
            self.Qc = Qc
            self.Qr = (D * hD + Ln[N - 1] * hLj[N - 1]
                       + float(W @ hLj + U @ hVj) - float(F @ hFj) - Qc - float(Qd.sum()))

            # Under-relax against the flows from the previous outer pass.
            Lnew = relax * Ln + (1.0 - relax) * np.asarray(L, float)
            Vnew = relax * Vn + (1.0 - relax) * np.asarray(V, float)
            return Lnew, Vnew

    return _EnergyBalance()


def _murphree_keff(K, x, efficiency):
    """Effective K-values that fold in Murphree vapour efficiency E.

    y_j = E_j K_j x_j + (1-E_j) y_{j+1}, so with y_j = K_eff_j x_j the tridiagonal
    material balance is untouched — only K changes. y_{j+1} (stage below) is taken
    as its equilibrium value K_{j+1} x_{j+1} on the *current* iterate (lagged), which
    keeps the solve local/tridiagonal instead of coupling every stage downward.

    efficiency: scalar or (N,) in (0,1]. Condenser (stage 0) and reboiler (last
    stage) are kept as equilibrium stages (E=1), the standard convention. With
    E=1 everywhere this returns K unchanged, so default runs are bit-identical.
    """
    N = K.shape[0]
    eff = np.broadcast_to(np.asarray(efficiency, float), (N,)).astype(float).copy()
    eff[0] = 1.0
    eff[-1] = 1.0
    if np.all(eff == 1.0):
        return K
    y_eq = K * x
    y_below = np.vstack([y_eq[1:], y_eq[-1:]])         # stage below; last row unused
    xs = np.where(x > 1e-30, x, 1e-30)
    return eff[:, None] * K + (1.0 - eff[:, None]) * (y_below / xs)


def _coerce_input(si_or_zF, F, antoine, comps, *, N, feed_stage, R, D, P,
                  gamma_fn, phi_fn=None) -> SolverInput:
    """Accept either a SolverInput (canonical) or the legacy scalar args."""
    if isinstance(si_or_zF, SolverInput):
        return si_or_zF
    zF = np.asarray(si_or_zF, float)
    return build_solver_input(
        n_stages=int(N), comps=list(comps), feeds=[(int(feed_stage), float(F), zF)],
        R=float(R), D=float(D), pressure=P, antoine=antoine, gamma_fn=gamma_fn,
        phi_fn=phi_fn)


def _finish_profile(si, x, T, L, V, B, extra=None, efficiency=1.0):
    """Common post-processing: vapour compositions, products and side-draw report.

    Solver arrays are internally 1=top..N=bottom (index j=0 is the condenser
    stage); the profile keeps that ordering, so profile index 0 = distillate/top
    and index N-1 = reboiler/bottoms (the app-wide "stage 0 = distillate"
    convention). `efficiency` (scalar or (N,)) makes the reported vapour the
    actual (Murphree, non-equilibrium) vapour leaving each stage."""
    N = si.n_stages
    K = np.array([k_values(T[j], si.pressure[j], si.antoine, si.gamma_fn, x[j],
                           si.phi_fn) for j in range(N)])
    y = _murphree_keff(K, x, efficiency) * x
    y /= y.sum(axis=1, keepdims=True)
    xD, xB = y[0].copy(), x[-1].copy()

    side_draws = [
        {"stage": j, "liquid": float(si.liquid_draw[j]),
         "vapor": float(si.vapor_draw[j]),
         "x": x[j].copy(), "y": y[j].copy()}
        for j in range(N)
        if si.liquid_draw[j] > 0.0 or si.vapor_draw[j] > 0.0
    ]

    feed_stages = [j for j in range(N) if si.feed[j].sum() > 0.0]

    prof = {
        # per-stage series, top -> bottom (index 0 = condenser/distillate)
        "x": np.asarray(x), "y": np.asarray(y), "T": np.asarray(T),
        "pressure": np.asarray(si.pressure),
        "liquid_flow": np.asarray(L), "vapor_flow": np.asarray(V),
        "k_values": np.exp(np.mean(np.log(K), axis=1)),
        # stage vapour molar latent heat via Clausius-Clapeyron (J/mol) — a real,
        # if approximate, stage enthalpy series (no Cp data: sensible heat omitted)
        "enthalpy": np.asarray([float(np.dot(y[j], latent_heat(T[j], si.antoine)))
                                for j in range(N)]),
        "comps": list(si.comps),
        "n_stages": N,
        "feed_stage": feed_stages[0] if feed_stages else 0,   # 0-based from top
        "feed_stages": feed_stages,
        "xD": xD, "xB": xB, "D": si.D, "B": B,
        "R": float(si.R),                                 # resolved reflux ratio
        "feed_totals": np.asarray(si.feed.sum(axis=0)),   # per-component molar feed
        "feed_q": float(si.q[feed_stages[0]]) if feed_stages else 1.0,
        "side_draws": side_draws,
        "condenser": si.condenser,
        "distillate_phase": "vapor" if si.condenser in ("partial", "none") else "liquid",
    }
    if extra:
        prof.update(extra)
    return prof


def solve_bubble_point(si_or_zF, F=None, antoine=None, comps=None, *, N=None,
                       feed_stage=None, R=None, D=None, P=None,
                       max_iter=500, tol=1e-6, gamma_fn=None, efficiency=1.0,
                       cancel=None, report=None):
    """Bubble-point (Wang-Henke) column solve.

    Canonical call: solve_bubble_point(solver_input, max_iter=..., tol=...).
    Legacy call: solve_bubble_point(zF, F, antoine, comps, N=, feed_stage=,
    R=, D=, P=) — builds a single-feed SolverInput internally.
    `efficiency`: Murphree vapour efficiency, scalar or (N,) in (0,1]; 1 = ideal
    equilibrium stages. Returns a profile dict (see _finish_profile).
    cancel: optional callable -> bool, checked each iteration for real Abort.
    report: optional callable (iteration, dT_residual) for progress display.
    """
    si = _coerce_input(si_or_zF, F, antoine, comps, N=N, feed_stage=feed_stage,
                       R=R, D=D, P=P, gamma_fn=gamma_fn)
    Nst, n = si.n_stages, si.n_comps
    V, L, B = _cmo_flows(si)

    zmix = si.feed.sum(axis=0) / si.feed.sum()   # flow-mixed overall feed
    T = np.full(Nst, bubble_T(zmix, float(np.mean(si.pressure)), si.antoine,
                              gamma_fn=si.gamma_fn, phi_fn=si.phi_fn))
    x = np.tile(zmix, (Nst, 1))

    iterations = 0
    aborted = False
    dT = float("inf")
    for iterations in range(1, max_iter + 1):
        if cancel is not None and cancel():
            aborted = True
            break
        K = np.array([k_values(T[j], si.pressure[j], si.antoine, si.gamma_fn, x[j],
                               si.phi_fn) for j in range(Nst)])
        Keff = _murphree_keff(K, x, efficiency)
        xnew = _stage_compositions(Keff, L, V, si.feed, si.liquid_draw,
                                   si.vapor_draw, B, si.D, Nst, n)
        Tnew = np.array([bubble_T(xnew[j], si.pressure[j], si.antoine,
                                  gamma_fn=si.gamma_fn, phi_fn=si.phi_fn)
                         for j in range(Nst)])
        dT = np.max(np.abs(Tnew - T))
        x, T = xnew, Tnew
        if report is not None:
            report(iterations, float(dT))
        if dT < tol:
            break

    if aborted:
        message = "Aborted."
    elif dT < tol:
        message = "Converged (bubble-point)."
    else:
        # direct substitution's slow geometric tail: the profile is usable,
        # but say so instead of claiming tol was met
        message = (f"Bubble-point: max iterations ({max_iter}) reached, "
                   f"dT residual {dT:.1e} degC.")
    return _finish_profile(si, x, T, L, V, B, efficiency=efficiency, extra={
        "iterations": iterations, "residual": float(dT),
        "found": not aborted,
        "message": message,
    })


def solve_inside_out(si_or_zF, F=None, antoine=None, comps=None, *, N=None,
                     feed_stage=None, R=None, D=None, P=None,
                     max_iter=50, tol=1e-6, gamma_fn=None, efficiency=1.0,
                     cancel=None, flows_hook=None, report=None):
    """Inside-Out (HYSIM-style) column solve.

    The defining two-tier structure: an OUTER loop refreshes rigorous K-values
    (Antoine/PLXANT + optional activity model) and derives stage relative
    volatilities alpha_ij = K_ij / Kb_j about a per-stage base K_b; an INNER
    loop then holds alpha fixed and cheaply iterates the base K_b and the
    per-component tridiagonal material balances (no rigorous-thermo calls).

    cancel: optional callable -> bool; checked each outer pass for real Abort.
    report: optional callable (outer_iteration, dT_residual) for progress display.
    flows_hook: optional callable (si, K, x, T, L, V) -> (L, V) — the energy-
    balance seam. Called once per outer pass after the rigorous K refresh; a
    stage-enthalpy balance plugs in here to free L/V from CMO and yield
    condenser/reboiler duties. None (default) keeps CMO flows.

    Returns the bubble-point profile schema plus iterations and approximate
    condenser/reboiler duties (latent-heat basis, kJ/h — flows are kmol/h and
    latent heats J/mol; multiply by thermodynamics.KJH_TO_KW for kW).
    """
    si = _coerce_input(si_or_zF, F, antoine, comps, N=N, feed_stage=feed_stage,
                       R=R, D=D, P=P, gamma_fn=gamma_fn)
    Nst, n = si.n_stages, si.n_comps
    V, L, B = _cmo_flows(si)

    zmix = si.feed.sum(axis=0) / si.feed.sum()
    T = np.full(Nst, bubble_T(zmix, float(np.mean(si.pressure)), si.antoine,
                              gamma_fn=si.gamma_fn, phi_fn=si.phi_fn))
    x = np.tile(zmix, (Nst, 1))
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
        Kfull = np.array([k_values(T[j], si.pressure[j], si.antoine, si.gamma_fn,
                                   x[j], si.phi_fn) for j in range(Nst)])
        Kfull = _murphree_keff(Kfull, x, efficiency)    # fold in stage efficiency
        Kb = np.exp(np.mean(np.log(Kfull), axis=1))     # (N,)
        alpha = Kfull / Kb[:, None]                     # (N, n), frozen below

        if flows_hook is not None:                      # energy-balance seam
            L, V = flows_hook(si, Kfull, x, T, L, V)

        # INNER: hold alpha, iterate base K_b + compositions (cheap, no thermo)
        for _ in range(50):
            K = alpha * Kb[:, None]
            xin = _stage_compositions(K, L, V, si.feed, si.liquid_draw,
                                      si.vapor_draw, B, si.D, Nst, n)
            Kb_new = 1.0 / np.sum(alpha * xin, axis=1)  # bubble constraint per stage
            done = np.max(np.abs(Kb_new - Kb) / Kb) < tol
            Kb, x = Kb_new, xin
            if done:
                break

        # Refresh temperatures from rigorous thermo for the next outer alpha
        Tnew = np.array([bubble_T(x[j], si.pressure[j], si.antoine,
                                  gamma_fn=si.gamma_fn, phi_fn=si.phi_fn)
                         for j in range(Nst)])
        dT = np.max(np.abs(Tnew - T))
        T = Tnew
        if report is not None:
            report(outer, float(dT))
        if dT < outer_tol:
            break

    # Terminal duties. With an energy-balance hook they are real outputs of the
    # stage-enthalpy balance; otherwise fall back to the latent-heat estimate
    # (CMO carries no stage enthalpy balance — condense-all / boil-all).
    if flows_hook is not None and hasattr(flows_hook, "Qc"):
        Qc, Qr = float(flows_hook.Qc), float(flows_hook.Qr)
    else:
        lam_top = float(np.dot(x[0] if Nst else zmix, latent_heat(T[0], si.antoine)))
        lam_bot = float(np.dot(x[-1], latent_heat(T[-1], si.antoine)))
        Qc = -V[0] * lam_top if si.condenser == "total" else -si.R * si.D * lam_top
        Qr = V[-1] * lam_bot

    return _finish_profile(si, x, T, L, V, B, efficiency=efficiency, extra={
        "iterations": outer,
        "condenser_duty": Qc, "reboiler_duty": Qr,
        "found": not aborted,
        "message": "Aborted." if aborted else "Converged (Inside-Out).",
    })


def _demo():
    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    comps = ["benzene", "toluene", "xylene"]
    zF = np.array([0.4, 0.35, 0.25]); F = 100.0
    prof = solve_bubble_point(zF, F, abc, comps, N=20, feed_stage=10,
                              R=3.0, D=40.0, P=760.0)

    # every stage is a valid composition; profiles are top -> bottom
    assert np.allclose(prof["x"].sum(axis=1), 1.0, atol=1e-8)
    assert np.allclose(prof["y"].sum(axis=1), 1.0, atol=1e-8)
    assert np.allclose(prof["x"][-1], prof["xB"]), "index -1 must be the bottoms"
    assert np.allclose(prof["y"][0], prof["xD"]), "index 0 must be the top"
    assert prof["T"][-1] > prof["T"][0], "reboiler must be hotter than the top"
    assert prof["feed_stage"] == 10 - 1, "feed_stage is 0-based from the top"
    # overall component balance closes: F z = D xD + B xB
    lhs = F * zF
    rhs = prof["D"] * prof["xD"] + prof["B"] * prof["xB"]
    assert np.allclose(lhs, rhs, atol=1e-3), f"balance off: {lhs} vs {rhs}"
    # light key concentrates up the column (top distillate richer in benzene)
    assert prof["xD"][0] > zF[0] > prof["xB"][0], "no separation achieved"
    assert prof["iterations"] >= 1

    # SolverInput path gives the identical column (Phase 0: one config seam)
    si = build_solver_input(n_stages=20, comps=comps, feeds=[(10, F, zF)],
                            R=3.0, D=40.0, pressure=760.0, antoine=abc)
    prof_si = solve_bubble_point(si)
    assert np.allclose(prof_si["x"], prof["x"], atol=1e-12)

    # Feed quality matters: a vapour feed (q=0) shifts the flow map and product
    si_q0 = build_solver_input(n_stages=20, comps=comps, feeds=[(10, F, zF, 0.0)],
                               R=3.0, D=40.0, pressure=760.0, antoine=abc)
    prof_q0 = solve_bubble_point(si_q0)
    assert not np.allclose(prof_q0["xB"], prof["xB"], atol=1e-4), \
        "q=0 vs q=1 must change the column"

    # A pressure profile (top->bottom rise) changes stage temperatures
    si_dp = build_solver_input(n_stages=20, comps=comps, feeds=[(10, F, zF)],
                               R=3.0, D=40.0,
                               pressure=np.linspace(760.0, 950.0, 20), antoine=abc)
    prof_dp = solve_bubble_point(si_dp)
    assert prof_dp["T"][-1] > prof["T"][-1] + 1.0, "extra bottom pressure must heat the reboiler"

    # Two feeds: balance closes over both
    si2 = build_solver_input(
        n_stages=20, comps=comps,
        feeds=[(8, 60.0, zF), (14, 40.0, [0.2, 0.3, 0.5])],
        R=3.0, D=40.0, pressure=760.0, antoine=abc)
    p2 = solve_bubble_point(si2)
    lhs = 60.0 * zF + 40.0 * np.array([0.2, 0.3, 0.5])
    rhs = p2["D"] * p2["xD"] + p2["B"] * p2["xB"]
    assert np.allclose(lhs, rhs, atol=1e-3), f"2-feed balance off: {lhs} vs {rhs}"
    assert p2["feed_stages"] == [8 - 1, 14 - 1]

    # Side draw: component balance closes including the draw
    si_sd = build_solver_input(
        n_stages=20, comps=comps, feeds=[(10, F, zF)],
        draws=[(5, 10.0, 0.0)], R=3.0, D=35.0, pressure=760.0, antoine=abc)
    psd = solve_bubble_point(si_sd)
    sd = psd["side_draws"][0]
    rhs = (psd["D"] * psd["xD"] + psd["B"] * psd["xB"]
           + sd["liquid"] * sd["x"] + sd["vapor"] * sd["y"])
    assert np.allclose(F * zF, rhs, atol=1e-3), f"side-draw balance off: {rhs}"
    assert psd["B"] == F - 35.0 - 10.0

    # Non-ideal (NRTL) path runs end-to-end and still produces valid stages.
    from core.thermodynamics import nrtl_gamma_fn
    gfn = nrtl_gamma_fn([[0.0, 0.5, 0.4], [0.5, 0.0, 0.3], [0.4, 0.3, 0.0]],
                        [[0.0] * 3] * 3,
                        [[0.0, 0.3, 0.3], [0.3, 0.0, 0.3], [0.3, 0.3, 0.0]])
    prof_ni = solve_bubble_point(zF, F, abc, comps, N=20, feed_stage=10,
                                 R=3.0, D=40.0, P=760.0, gamma_fn=gfn)
    assert np.allclose(prof_ni["x"].sum(axis=1), 1.0, atol=1e-8)
    assert prof_ni["xD"][0] > zF[0] > prof_ni["xB"][0], "non-ideal: no separation"

    # Inside-Out: converges to (essentially) the same column as Wang-Henke, emits
    # the rich per-stage profiles + duties, and honours a cancel hook for Abort.
    io = solve_inside_out(zF, F, abc, comps, N=20, feed_stage=10,
                          R=3.0, D=40.0, P=760.0)
    assert io["found"] and np.allclose(io["x"].sum(axis=1), 1.0, atol=1e-8)
    lhs, rhs = F * zF, io["D"] * io["xD"] + io["B"] * io["xB"]
    assert np.allclose(lhs, rhs, atol=1e-3), f"IO balance off: {lhs} vs {rhs}"
    assert np.allclose(io["xD"], prof["xD"], atol=2e-2), "IO disagrees with bubble-point"
    for key in ("pressure", "liquid_flow", "vapor_flow", "k_values", "enthalpy"):
        assert len(io[key]) == io["n_stages"], f"{key} not a per-stage series"
    # enthalpy is a real latent-heat series now (tens of kJ/mol), not a T proxy
    assert 20e3 < np.mean(io["enthalpy"]) < 60e3, np.mean(io["enthalpy"])
    assert io["condenser_duty"] < 0.0 < io["reboiler_duty"]
    assert io["iterations"] >= 1
    assert solve_inside_out(zF, F, abc, comps, N=20, feed_stage=10, R=3.0,
                            D=40.0, P=760.0, cancel=lambda: True)["message"] == "Aborted."

    # Both solvers honour cancel and emit per-iteration progress reports.
    assert solve_bubble_point(zF, F, abc, comps, N=20, feed_stage=10, R=3.0,
                              D=40.0, P=760.0,
                              cancel=lambda: True)["message"] == "Aborted."
    ticks = []
    pr = solve_bubble_point(si, report=lambda i, r: ticks.append((i, r)))
    assert ticks and ticks[0][0] == 1 and ticks[-1][1] == pr["residual"]
    assert ticks[-1][1] < 1e-3          # tight-tol runs end in the slow tail...
    assert "max iterations" in pr["message"], pr["message"]   # ...and say so
    assert "Converged" in solve_bubble_point(si, tol=1e-3)["message"]
    ticks = []
    solve_inside_out(si, report=lambda i, r: ticks.append((i, r)))
    assert ticks and ticks[-1][1] < 1e-3

    # flows_hook is the energy-balance seam: it is called and its L/V are used.
    calls = []
    def hook(si_, K, x_, T_, L_, V_):
        calls.append(1)
        return L_, V_
    io_h = solve_inside_out(si, flows_hook=hook)
    assert calls and io_h["found"]

    # Real energy balance (make_energy_balance): retires CMO, produces stage-wise
    # varying L/V and duties that close the overall energy balance. BTX props:
    # cp_liq [J/mol-K], hvap_Tb [kJ/mol], Tb, Tc [K].
    cp_l = np.array([136.0, 157.0, 186.0])
    hv_tb = np.array([30.8, 33.2, 36.2])
    tb_k = np.array([353.2, 383.8, 417.6])
    tc_k = np.array([562.0, 591.8, 630.3])
    eb = make_energy_balance(cp_l, hv_tb, tb_k, tc_k)
    io_eb = solve_inside_out(si, flows_hook=eb, max_iter=80)
    assert io_eb["found"], io_eb["message"]
    assert np.allclose(io_eb["x"].sum(axis=1), 1.0, atol=1e-8)
    lhs, rhs = F * zF, io_eb["D"] * io_eb["xD"] + io_eb["B"] * io_eb["xB"]
    assert np.allclose(lhs, rhs, atol=1e-2), f"EB balance off: {lhs} vs {rhs}"
    assert io_eb["condenser_duty"] < 0.0 < io_eb["reboiler_duty"], \
        (io_eb["condenser_duty"], io_eb["reboiler_duty"])
    # duties are the same order as the latent-heat estimate but not identical
    assert 0.3 < abs(io_eb["reboiler_duty"] / io["reboiler_duty"]) < 3.0
    # energy balance frees the vapour flow from CMO's near-constant profile:
    # the top/bottom internal vapour rates differ by more than round-off
    Vprof = np.asarray(io_eb["vapor_flow"])
    assert abs(Vprof[1] - Vprof[-1]) > 1e-3 * abs(Vprof[1]), "V still looks like CMO"
    # overall energy balance closes: feeds + Qr = products + |Qc|  (within 1%)
    Qc, Qr = io_eb["condenser_duty"], io_eb["reboiler_duty"]
    assert abs(Qc + Qr) < 0.5 * max(abs(Qc), abs(Qr)), (Qc, Qr)   # near-balanced column

    # Subcooled reflux: 10 K below the bubble point removes more condenser heat
    # and, by condensing extra vapour on stage 0, raises the internal reflux.
    si_sub = build_solver_input(
        n_stages=20, comps=comps, feeds=[(10, F, zF)],
        R=3.0, D=40.0, pressure=760.0, antoine=abc, subcooling=10.0)
    io_sub = solve_inside_out(si_sub, flows_hook=make_energy_balance(
        cp_l, hv_tb, tb_k, tc_k), max_iter=80)
    assert io_sub["found"], io_sub["message"]
    assert io_sub["condenser_duty"] < Qc, \
        (io_sub["condenser_duty"], Qc)                 # more heat removed (more negative)
    L_base = np.asarray(io_eb["liquid_flow"]); L_sub = np.asarray(io_sub["liquid_flow"])
    assert L_sub[0] > L_base[0], (L_sub[0], L_base[0])  # colder reflux -> more internal L

    # Inside-Out with a side draw agrees with bubble-point's closure too
    io_sd = solve_inside_out(si_sd)
    sd = io_sd["side_draws"][0]
    rhs = (io_sd["D"] * io_sd["xD"] + io_sd["B"] * io_sd["xB"]
           + sd["liquid"] * sd["x"] + sd["vapor"] * sd["y"])
    assert np.allclose(F * zF, rhs, atol=1e-3)

    # Stage efficiency: E=1 is identical to the default; E<1 degrades the split
    # (a real tray under-performs equilibrium) but keeps every stage physical.
    prof_e1 = solve_bubble_point(zF, F, abc, comps, N=20, feed_stage=10,
                                 R=3.0, D=40.0, P=760.0, efficiency=1.0)
    assert np.allclose(prof_e1["x"], prof["x"], atol=1e-12), "E=1 must be a no-op"
    prof_eff = solve_bubble_point(zF, F, abc, comps, N=20, feed_stage=10,
                                  R=3.0, D=40.0, P=760.0, efficiency=0.6)
    assert np.allclose(prof_eff["x"].sum(axis=1), 1.0, atol=1e-8)
    assert prof_eff["xD"][0] < prof["xD"][0], "lower efficiency must worsen the split"
    io_eff = solve_inside_out(zF, F, abc, comps, N=20, feed_stage=10, R=3.0,
                              D=40.0, P=760.0, efficiency=0.6)
    assert io_eff["found"] and np.allclose(io_eff["x"].sum(axis=1), 1.0, atol=1e-8)
    assert io_eff["xD"][0] < io["xD"][0], "Inside-Out efficiency must worsen the split"

    print(f"column_solvers self-check OK: xD={np.round(prof['xD'],3)}, "
          f"xB={np.round(prof['xB'],3)} (multi-feed, side-draw, q, dP, NRTL, "
          f"Inside-Out {io['iterations']} outer iters)")


if __name__ == "__main__":
    _demo()
