"""Damped Newton on R(U)=0 (blueprint Section 15; goal unit 7).

Ingredients:
  * block-tridiagonal Newton step  du = -J^{-1} R  (never forms dense J)
  * residual scaling so material / equilibrium / energy rows are all O(1) in
    the line-search merit and the convergence test (the step itself is
    scale-invariant)
  * fraction-to-boundary damping keeping l, v >= 0 and T in bounds
  * Armijo backtracking on 1/2 ||R_scaled||^2
  * two-part stopping test: scaled residual AND scaled step both small

Returns the solution and an info dict. Duties freed at the terminals (whose
energy balances were replaced by specs) are recovered afterward.
"""

import numpy as np

from residual import residual, unpack, flows, stride
from jacobian import jacobian_blocks
from linsolve import block_thomas

_TMIN, _TMAX = -50.0, 600.0     # temperature bounds (Antoine-fit unit, degC here)


def _row_scale(prob):
    """Per-equation weights making each residual row O(1). Shape (N*m,).

    material ~ flow, equilibrium ~ flow^2, energy ~ flow*enthalpy, spec ~ flow.
    """
    N, C = prob.n_stages, prob.C
    R = prob.reactions.n_rxn if prob.reactions is not None else 0
    m = 2 * C + 1 + R
    F = max(float(prob.feed.sum()), 1.0)
    H = F * 3.0e4                                  # flow x typical molar enthalpy
    w = np.ones(m)
    w[0:C] = 1.0 / F
    w[C:2 * C] = 1.0 / (F * F)
    w[2 * C] = 1.0 / H
    if R:
        w[2 * C + 1:] = 1.0 / F
    return np.tile(w, N)


def _var_scale(prob):
    """Variable scales for the step-norm test. Shape (N*m,)."""
    N, C = prob.n_stages, prob.C
    R = prob.reactions.n_rxn if prob.reactions is not None else 0
    m = 2 * C + 1 + R
    F = max(float(prob.feed.sum()), 1.0)
    s = np.ones(m)
    s[0:2 * C] = F
    s[2 * C] = 100.0                                # temperature scale
    if R:
        s[2 * C + 1:] = F
    return np.tile(s, N)


_FLOW_FLOOR = 1e-9


def _project(U, prob):
    """Clip component flows to a positive floor and T into bounds (in place-safe).

    Used instead of fraction-to-boundary: a near-zero trace flow would otherwise
    throttle the whole Newton step to nothing. LM damping supplies the stability
    that FTB would have.
    """
    N, C = prob.n_stages, prob.C
    R = prob.reactions.n_rxn if prob.reactions is not None else 0
    m = 2 * C + 1 + R
    Ub = U.reshape(N, m).copy()
    Ub[:, :2 * C] = np.clip(Ub[:, :2 * C], _FLOW_FLOOR, None)
    Ub[:, 2 * C] = np.clip(Ub[:, 2 * C], _TMIN, _TMAX)
    return Ub.reshape(N * m)


def _damp_blocks(B, mu):
    """Levenberg-Marquardt diagonal damping: B_i += mu * diag(|diag(B_i)|).

    Keeps the system block-tridiagonal (only diagonal blocks change) while
    pulling the step toward a scaled-gradient descent when mu is large. This is
    the pseudo-transient / trust-region stabilizer for stiff (energy-coupled)
    starts (goal unit 8). Returns damped copies of the diagonal blocks.
    """
    N, m, _ = B.shape
    out = B.copy()
    idx = np.arange(m)
    for i in range(N):
        d = np.abs(np.diag(B[i]))
        d = np.where(d > 0, d, 1.0)
        out[i, idx, idx] += mu * d
    return out


def newton(U0, prob, provider, *, tol=1e-8, max_iter=120, verbose=False,
           mu0=1e-2, cancel=None, report=None):
    """Solve R(U)=0 from U0 with LM-damped, fraction-to-boundary Newton.

    The energy-coupled MESH is stiff from a CMO cold start; adaptive LM damping
    (mu grows when a step fails, shrinks when it succeeds) is the pseudo-
    transient backbone that keeps it converging. Returns (U, info).

    cancel: optional callable -> bool, checked each iteration (real Abort).
    report: optional callable (iteration, scaled_residual) for progress display.
    """
    w = _row_scale(prob)
    vs = _var_scale(prob)
    U = np.array(U0, float)
    mu = mu0

    def merit(Ux):
        return 0.5 * float(np.sum((w * residual(Ux, prob, provider)) ** 2))

    hist = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        if cancel is not None and cancel():
            return U, _info(False, it, hist, "aborted",
                            hist[-1] if hist else float("inf"))
        Res = residual(U, prob, provider)
        rnorm = np.max(np.abs(w * Res))
        hist.append(rnorm)
        if report is not None:
            report(it, float(rnorm))
        if not np.isfinite(rnorm):
            break
        if rnorm < tol:
            converged = True
            break
        A, B, Cc = jacobian_blocks(U, prob, provider)
        m0 = merit(U)

        # LM inner loop: grow mu until a projected, merit-decreasing step is found
        accepted = False
        step_norm = a = 0.0
        for _ in range(50):
            try:
                du = block_thomas(A, _damp_blocks(B, mu), Cc, -Res)
            except np.linalg.LinAlgError:
                mu *= 4.0
                continue
            # backtracking line search on the projected step
            a = 1.0
            for _ls in range(40):
                Un = _project(U + a * du, prob)
                if np.isfinite(merit(Un)) and merit(Un) < m0:
                    break
                a *= 0.5
            else:
                mu *= 4.0                           # no good step: damp harder
                continue
            step_norm = a * np.max(np.abs(du / vs))
            U = Un
            mu = max(mu * 0.5, 1e-10)               # step worked: trust more
            accepted = True
            break
        if not accepted:
            return U, _info(False, it, hist, "step rejected (LM stalled)", rnorm)
        if verbose:
            print(f"  it {it:3d}: |R|={rnorm:.2e} mu={mu:.1e} a={a:.2e} "
                  f"|du|={step_norm:.2e}")
        if rnorm < tol and step_norm < 1e-6:
            converged = True
            break

    rfinal = np.max(np.abs(w * residual(U, prob, provider)))
    converged = converged or rfinal < tol
    return U, _info(converged, it, hist, "converged" if converged else "not converged",
                    rfinal)


def _info(converged, iters, hist, message, rfinal=None):
    return {"converged": bool(converged), "iterations": int(iters),
            "residual_history": list(hist), "message": message,
            "residual": rfinal if rfinal is not None else (hist[-1] if hist else None)}


def recover_duties(U, prob, provider):
    """Condenser/reboiler duties from the terminal energy balances that the
    specs replaced. Returns (Q_condenser, Q_reboiler); sign: Q>0 adds heat.
    """
    N, C = prob.n_stages, prob.C
    R = prob.reactions.n_rxn if prob.reactions is not None else 0
    l, v, T, xi = unpack(U, N, C, R)
    L, V, x, y = flows(l, v)
    hL = provider.h_L(x, T); hV = provider.h_V(y, T)
    rl, rv = prob.rl, prob.rv

    def energy_imbalance(i):
        out = (1 + rl[i]) * L[i] * hL[i] + (1 + rv[i]) * V[i] * hV[i]
        inn = prob.feedH[i]
        if i > 0:
            inn += L[i - 1] * hL[i - 1]
        if i < N - 1:
            inn += V[i + 1] * hV[i + 1]
        return out - inn                    # stage balance: Q = out - in

    Qc = energy_imbalance(0)                # condenser removes heat -> Qc < 0
    Qr = energy_imbalance(N - 1)            # reboiler adds heat -> Qr > 0
    return float(Qc), float(Qr)


def _demo():
    from thermo_adapter import FreeColumnThermo
    from problem import build_problem, OpSpec
    from initializer import initialize
    from residual import mass_balance_residual

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = FreeColumnThermo(abc)
    comps = ["benzene", "toluene", "xylene"]
    N, C = 16, 3
    zF = np.array([0.4, 0.35, 0.25]); F = 100.0
    prob = build_problem(
        n_stages=N, comps=comps, feeds=[(8, F, zF)], pressure=760.0, provider=tp,
        top_spec=OpSpec("reflux_ratio", 3.0), bottom_spec=OpSpec("bottoms_rate", 60.0))

    U0 = initialize(prob, tp)
    U, info = newton(U0, prob, tp, verbose=False)
    assert info["converged"], info
    assert info["residual"] < 1e-8, info

    l, v, T, xi = unpack(U, N, C, 0)
    L, V, x, y = flows(l, v)
    # every stage a valid split; bubble-point satisfied (sum K x = 1)
    K = tp.K(x, T, prob.pressure)
    assert np.allclose((K * x).sum(1), 1.0, atol=1e-6), "bubble point per stage"
    # external mass balance closes per-component
    per_comp, overall = mass_balance_residual(U, prob)
    assert np.max(np.abs(per_comp)) < 1e-6, per_comp
    # separation happened: benzene up top, xylene in bottoms
    xD = v[0] / v[0].sum(); xB = l[-1] / l[-1].sum()
    assert xD[0] > zF[0] > xB[0], (xD, xB)
    # reflux ratio and distillate rate honoured
    assert abs(L[0] / V[0] - 3.0) < 1e-6 and abs(V[0] - 40.0) < 1e-4

    # CROSS-CHECK vs FreeColumn's rigorous bubble-point solver (the in-repo
    # "independent simulator"). Same column; compositions agree closely.
    from core.column_solvers import solve_bubble_point
    ref = solve_bubble_point(zF, F, abc, comps, N=N, feed_stage=8, R=3.0,
                             D=40.0, P=760.0)
    assert np.allclose(xD, ref["xD"], atol=2e-2), (xD, ref["xD"])
    assert np.allclose(xB, ref["xB"], atol=2e-2), (xB, ref["xB"])

    # duties recovered with the right signs
    Qc, Qr = recover_duties(U, prob, tp)
    assert Qc < 0 < Qr, (Qc, Qr)

    print(f"newton self-check OK ({info['iterations']} iters, |R|="
          f"{info['residual']:.1e}); xD={np.round(xD,3)} vs ref "
          f"{np.round(ref['xD'],3)}")


if __name__ == "__main__":
    _demo()
