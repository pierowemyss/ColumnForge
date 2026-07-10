"""Continuation / homotopy wrappers (blueprint Section 2.3; goal unit 8).

Newton (`newton.newton`) already carries an adaptive Levenberg-Marquardt /
pseudo-transient stabilizer, so it handles most stiff starts alone. When it
still stalls, these wrappers walk the solution in from an easier problem:

  * thermodynamic homotopy  gamma -> gamma^lambda, lambda: 0 (ideal) -> 1 (real)
  * parameter homotopy       march a spec/parameter through a value schedule,
                             warm-starting each step from the last solution

Each step is an ordinary damped-Newton solve; the predictor is the previous
converged state (a zeroth-order predictor with an adaptive step count is enough
here, and keeps the kernel simple).
"""

import numpy as np

from newton import newton
from initializer import initialize
from thermo_adapter import FreeColumnThermo


def ideal_twin(provider):
    """An ideal (gamma=1) copy of a FreeColumnThermo provider, else the provider
    itself (already ideal / unknown model)."""
    if isinstance(provider, FreeColumnThermo) and provider.gamma_fn is not None:
        return FreeColumnThermo(provider.antoine, gamma_fn=None,
                                Cp=provider.Cp, Tref=provider.Tref)
    return provider


def _gamma_power(gamma_fn, lam):
    """gamma(x,T) ** lam — the thermodynamic homotopy path (lam=0 -> ideal)."""
    def g(x, T):
        return np.asarray(gamma_fn(x, T), float) ** lam
    return g


def thermodynamic_homotopy(prob, provider, U0=None, *, n_steps=5, tol=1e-8,
                           verbose=False):
    """Solve by ramping the activity model from ideal to real.

    Only meaningful for a FreeColumnThermo with an activity model; otherwise it
    is a single plain Newton solve. Returns (U, info) with info['path'] the per-
    step residuals.
    """
    if U0 is None:
        U0 = initialize(prob, provider)
    if not (isinstance(provider, FreeColumnThermo) and provider.gamma_fn is not None):
        U, info = newton(U0, prob, provider, tol=tol)
        info["path"] = [info["residual"]]
        return U, info

    base_gamma = provider.gamma_fn
    U = np.array(U0, float)
    path = []
    for lam in np.linspace(0.0, 1.0, n_steps + 1)[1:]:
        p = FreeColumnThermo(provider.antoine, gamma_fn=_gamma_power(base_gamma, lam),
                             Cp=provider.Cp, Tref=provider.Tref)
        U, info = newton(U, prob, p, tol=tol)
        path.append(info["residual"])
        if verbose:
            print(f"  lambda={lam:.2f}: |R|={info['residual']:.2e} "
                  f"({info['iterations']} it, {info['message']})")
        if not info["converged"]:
            info["path"] = path
            return U, info
    info["path"] = path
    return U, info


def parameter_homotopy(build_prob, provider, values, *, U0=None, tol=1e-8,
                       verbose=False):
    """March a problem parameter through `values`, warm-starting each solve.

    build_prob(value) -> Problem. Use e.g. to ramp a demanding purity/reflux
    spec in from an easy one. Returns (U, prob_final, info).
    """
    prob = build_prob(values[0])
    if U0 is None:
        U0 = initialize(prob, provider)
    U = np.array(U0, float)
    info = None
    for val in values:
        prob = build_prob(val)
        U, info = newton(U, prob, provider, tol=tol)
        if not info["converged"]:
            # predictor (warm start) landed in a merit stagnation point: re-predict
            # from a fresh structured guess for this value and retry.
            U, info = newton(initialize(prob, provider), prob, provider, tol=tol)
        if verbose:
            print(f"  value={val}: |R|={info['residual']:.2e} ({info['message']})")
        if not info["converged"]:
            break
    return U, prob, info


def _demo():
    from problem import build_problem, OpSpec
    from residual import unpack, flows, mass_balance_residual
    from core.thermodynamics import nrtl_gamma_fn

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    comps = ["benzene", "toluene", "xylene"]
    N = 16
    zF = np.array([0.4, 0.35, 0.25]); F = 100.0

    # Strong NRTL non-ideality that can trip a cold-start Newton; the ideal->real
    # homotopy walks it in.
    a = 1.2
    gfn = nrtl_gamma_fn([[0, a, a], [a, 0, a], [a, a, 0]], [[0.0] * 3] * 3,
                        [[0, .3, .3], [.3, 0, .3], [.3, .3, 0]])
    tp = FreeColumnThermo(abc, gamma_fn=gfn)
    prob = build_problem(n_stages=N, comps=comps, feeds=[(8, F, zF)], pressure=760.0,
                         provider=tp, top_spec=OpSpec("reflux_ratio", 4.0),
                         bottom_spec=OpSpec("bottoms_rate", 60.0))

    U, info = thermodynamic_homotopy(prob, tp, n_steps=5, verbose=False)
    assert info["converged"], info
    per_comp, _ = mass_balance_residual(U, prob)
    assert np.max(np.abs(per_comp)) < 1e-5, per_comp
    l, v, T, xi = unpack(U, N, 3, 0)
    L, V, x, y = flows(l, v)
    K = tp.K(x, T, prob.pressure)
    assert np.allclose((K * x).sum(1), 1.0, atol=1e-5), "bubble point per stage"

    # parameter homotopy: ramp reflux in from an easy 2.0 to a tighter 6.0
    def build(r):
        return build_problem(n_stages=N, comps=comps, feeds=[(8, F, zF)],
                             pressure=760.0, provider=tp,
                             top_spec=OpSpec("reflux_ratio", r),
                             bottom_spec=OpSpec("bottoms_rate", 60.0))
    Up, probp, infop = parameter_homotopy(build, tp, [2.0, 4.0, 6.0])
    assert infop["converged"], infop

    print(f"continuation self-check OK (ideal->real path "
          f"{['%.0e' % r for r in info['path']]})")


if __name__ == "__main__":
    _demo()
