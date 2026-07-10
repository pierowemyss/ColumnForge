"""Public API (blueprint Section 15; goal unit 10).

Three entry points, feasibility-first:

    assess_feasibility(prob, provider) -> report (+ the structured guess U0)
    initialize(prob, provider)         -> U0
    converge(prob, provider)           -> solution dict

Matrix BVM's primary product is the feasibility report and U0; convergence is
offered but not the point. So the module also hands the raw pieces to any
external nonlinear solver:

    residual_fn(prob, provider)  -> f(U) -> R
    jacobian_fn(prob, provider)  -> f(U) -> (A, B, C) block diagonals
    dense_jacobian_fn(...)       -> f(U) -> dense J   (small problems / SciPy)

`extract_profiles` turns a solved U into the app's usual top->bottom profile
dict (index 0 = condenser/distillate), so results drop into FreeColumn's plots.
"""

import numpy as np

from problem import Problem
from residual import residual, unpack, flows, mass_balance_residual, stride
from jacobian import jacobian_blocks, dense_from_blocks
from initializer import initialize as _initialize
from newton import newton, recover_duties
from continuation import thermodynamic_homotopy
from diagnostics import assess as _assess, classify


def initialize(prob, provider, **kw):
    """Structured initial guess U0 for a Problem (see initializer.initialize)."""
    prob.require_square()
    return _initialize(prob, provider, **kw)


def assess_feasibility(prob, provider, U0=None):
    """Primary product: a feasibility report + the structured guess.

    Returns {report, findings, U0, feasible}. Does not converge anything.
    """
    prob.require_square()
    if U0 is None:
        U0 = _initialize(prob, provider)
    report = _assess(prob, provider, U0)
    return {"report": report, "findings": report.findings, "U0": U0,
            "feasible": report.feasible}


def converge(prob, provider, U0=None, *, tol=1e-8, use_continuation=True,
             verbose=False):
    """Solve R(U)=0. Returns a solution dict (profiles, duties, balance, info).

    Plain damped Newton first; if it stalls and the model is non-ideal, retry via
    the ideal->real thermodynamic homotopy. On failure the diagnostics findings
    are attached so the caller learns *why*, not just that it failed.
    """
    prob.require_square()
    if U0 is None:
        U0 = _initialize(prob, provider)

    U, info = newton(U0, prob, provider, tol=tol, verbose=verbose)
    if not info["converged"] and use_continuation:
        U, info = thermodynamic_homotopy(prob, provider, U0, tol=tol, verbose=verbose)

    sol = extract_profiles(U, prob, provider)
    sol["info"] = info
    sol["converged"] = info["converged"]
    sol["U"] = U
    if not info["converged"]:
        sol["findings"] = classify(prob, provider, U)
    return sol


def extract_profiles(U, prob, provider):
    """Solved state -> profile dict, top->bottom (index 0 = condenser/distillate).

    Matches core.column_solvers._finish_profile's orientation so results are
    interchangeable in the app. Duties are kJ/h (flows kmol/h x molar
    enthalpies J/mol), same basis as the core solvers.
    """
    N, C = prob.n_stages, prob.C
    R = prob.reactions.n_rxn if prob.reactions is not None else 0
    l, v, T, xi = unpack(U, N, C, R)
    L, V, x, y = flows(l, v)
    xD = v[0] / max(v[0].sum(), 1e-300)          # partial-condenser vapour product
    xB = l[N - 1] / max(l[N - 1].sum(), 1e-300)
    D = float(V[0]); Bp = float(L[N - 1])
    Qc, Qr = recover_duties(U, prob, provider)
    per_comp, overall = mass_balance_residual(U, prob)

    # Problem stages are already 0-based top->bottom (0 = condenser) — no flip.
    return {
        "x": np.asarray(x), "y": np.asarray(y), "T": np.asarray(T),
        "liquid_flow": np.asarray(L), "vapor_flow": np.asarray(V),
        "pressure": np.asarray(prob.pressure),
        "comps": list(prob.comps), "n_stages": N,
        "xD": xD, "xB": xB, "D": D, "B": Bp,
        "condenser_duty": Qc, "reboiler_duty": Qr,
        "reaction_extent": np.asarray(xi) if R else None,
        "mass_balance": {"per_component": per_comp, "overall": overall},
    }


def residual_fn(prob, provider):
    """f(U) -> R for an external solver (e.g. scipy.optimize.root)."""
    prob.require_square()
    return lambda U: residual(np.asarray(U, float), prob, provider)


def jacobian_fn(prob, provider):
    """f(U) -> (A, B, C) block-tridiagonal Jacobian for an external solver."""
    prob.require_square()
    return lambda U: jacobian_blocks(np.asarray(U, float), prob, provider)


def dense_jacobian_fn(prob, provider):
    """f(U) -> dense J (for scipy.optimize.root's jac=, small problems)."""
    prob.require_square()

    def J(U):
        A, B, Cc = jacobian_blocks(np.asarray(U, float), prob, provider)
        return dense_from_blocks(A, B, Cc)
    return J


def _demo():
    from thermo_adapter import FreeColumnThermo
    from problem import build_problem, OpSpec

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = FreeColumnThermo(abc)
    comps = ["benzene", "toluene", "xylene"]
    N = 16
    zF = np.array([0.4, 0.35, 0.25]); F = 100.0
    prob = build_problem(n_stages=N, comps=comps, feeds=[(8, F, zF)], pressure=760.0,
                         provider=tp, top_spec=OpSpec("reflux_ratio", 3.0),
                         bottom_spec=OpSpec("bottoms_rate", 60.0))

    # feasibility-first: report + U0, no solve
    fa = assess_feasibility(prob, tp)
    assert fa["feasible"], fa["findings"]
    assert fa["U0"].shape == (N * stride(prob),)

    # converge and check the profile dict is app-shaped and balanced
    sol = converge(prob, tp)
    assert sol["converged"], sol["info"]
    assert sol["x"].shape == (N, 3)
    assert np.allclose(sol["x"][-1], sol["xB"]), "index -1 is the bottoms"
    assert np.allclose(sol["y"][0], sol["xD"]), "index 0 is the distillate"
    assert sol["T"][-1] > sol["T"][0], "reboiler hotter than condenser"
    assert np.max(np.abs(sol["mass_balance"]["per_component"])) < 1e-6
    assert sol["condenser_duty"] < 0 < sol["reboiler_duty"]

    # external-solver contract: R and J are consumable by SciPy. Global cold-start
    # convergence is our LM's job, so we hand SciPy a near-solution start and it
    # refines to the same root (proving the functions are wired correctly). The
    # residual is row-scaled here so the energy row (raw ~1e6) doesn't dominate.
    from scipy.optimize import root
    from newton import _row_scale
    w = _row_scale(prob)
    f = residual_fn(prob, tp)
    Jf = dense_jacobian_fn(prob, tp)
    fs = lambda U: w * f(U)
    Js = lambda U: w[:, None] * Jf(U)
    assert np.max(np.abs(fs(sol["U"]))) < 1e-6, "scaled R(U*) not ~0 at our solution"
    warm = sol["U"] * (1 + 1e-3 * np.random.default_rng(0).standard_normal(sol["U"].shape))
    r = root(fs, warm, jac=Js, method="hybr", tol=1e-12)
    l, v, T, xi = unpack(r.x, N, 3, 0)
    xD_scipy = v[0] / v[0].sum()
    assert np.allclose(xD_scipy, sol["xD"], atol=1e-3), (xD_scipy, sol["xD"])

    print(f"api self-check OK (feasible; converged {sol['info']['iterations']} "
          f"iters; SciPy external solve agrees; Qc={sol['condenser_duty']:.2e} "
          f"Qr={sol['reboiler_duty']:.2e})")


if __name__ == "__main__":
    _demo()
