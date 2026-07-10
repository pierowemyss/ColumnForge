"""Section-15 cross-cutting validation for Matrix BVM.

Covers, on top of each kernel's own _demo self-check:
  * per-component + overall mass-balance closure on every returned state
  * analytic Jacobian blocks vs a full finite-difference across configs
  * ternary ideal cross-check + a min-reflux (sharper split with more reflux) trend
  * rigorous cross-check vs FreeColumn's own solve_bubble_point (the in-repo
    "independent simulator"), ideal and NRTL
  * continuation stress: a Newton-stalling NRTL case converges under homotopy;
    an infeasible spec is classified, not diverged
  * reactive column: converges and conserves atoms

Runs under pytest, or standalone via `python tests/test_validation.py`.
"""

import numpy as np

from thermo_adapter import FreeColumnThermo
from problem import build_problem, OpSpec, Reactions
from initializer import initialize
from residual import unpack, flows, mass_balance_residual, pack
from jacobian import jacobian_blocks, dense_from_blocks, fd_jacobian
from newton import newton
from continuation import thermodynamic_homotopy
from diagnostics import classify, assess
from api import converge, assess_feasibility

_ABC = np.array([(6.90565, 1211.033, 220.79),
                 (6.95464, 1344.8, 219.48),
                 (6.99052, 1453.43, 215.31)])
_COMPS = ["benzene", "toluene", "xylene"]
_ZF = np.array([0.4, 0.35, 0.25])


def _nrtl(a=0.6):
    from core.thermodynamics import nrtl_gamma_fn
    return nrtl_gamma_fn([[0, a, a], [a, 0, a], [a, a, 0]], [[0.0] * 3] * 3,
                         [[0, .3, .3], [.3, 0, .3], [.3, .3, 0]])


def _column(provider, N=16, feed_stage=8, R=3.0, B=60.0):
    return build_problem(n_stages=N, comps=_COMPS, feeds=[(feed_stage, 100.0, _ZF)],
                         pressure=760.0, provider=provider,
                         top_spec=OpSpec("reflux_ratio", R),
                         bottom_spec=OpSpec("bottoms_rate", B))


def test_mass_balance_closure():
    for prov in (FreeColumnThermo(_ABC), FreeColumnThermo(_ABC, gamma_fn=_nrtl())):
        for R, B in ((2.0, 55.0), (3.0, 60.0), (5.0, 65.0)):
            prob = _column(prov, R=R, B=B)
            sol = converge(prob, prov)
            assert sol["converged"], sol["info"]
            per, overall = sol["mass_balance"]["per_component"], sol["mass_balance"]["overall"]
            assert np.max(np.abs(per)) < 1e-6, (R, B, per)
            assert abs(overall) < 1e-6


def test_newton_report_and_cancel_hooks():
    from initializer import initialize

    prov = FreeColumnThermo(_ABC)
    prob = _column(prov)
    U0 = initialize(prob, prov)

    ticks = []
    U, info = newton(U0, prob, prov,
                     report=lambda i, r: ticks.append((i, r)))
    assert info["converged"]
    assert ticks and ticks[0][0] == 1 and ticks[-1][1] <= ticks[0][1]

    _, info = newton(U0, prob, prov, cancel=lambda: True)
    assert not info["converged"] and info["message"] == "aborted"


def test_analytic_jacobian_matches_fd():
    rng = np.random.default_rng(1)
    for prov, tol in ((FreeColumnThermo(_ABC), 1e-5),
                      (FreeColumnThermo(_ABC, gamma_fn=_nrtl()), 3e-4)):
        prob = _column(prov, N=8, feed_stage=4)
        N, C = prob.n_stages, prob.C
        L = np.linspace(180, 220, N); V = np.linspace(230, 250, N)
        x = np.clip(_ZF + 0.05 * rng.standard_normal((N, C)), 0.05, None)
        x /= x.sum(1, keepdims=True)
        y = np.clip(_ZF + 0.05 * rng.standard_normal((N, C)), 0.05, None)
        y /= y.sum(1, keepdims=True)
        U = pack(L[:, None] * x, V[:, None] * y, np.linspace(85, 125, N))
        Jan = dense_from_blocks(*jacobian_blocks(U, prob, prov))
        Jfd = fd_jacobian(U, prob, prov)
        rel = np.abs(Jan - Jfd).max() / max(1.0, np.abs(Jfd).max())
        assert rel < tol, (prov.gamma_fn is not None, rel)


def test_cross_check_vs_column_solvers():
    from core.column_solvers import solve_bubble_point
    for gfn in (None, _nrtl(0.3)):
        prov = FreeColumnThermo(_ABC, gamma_fn=gfn)
        prob = _column(prov, N=18, feed_stage=9, R=4.0, B=60.0)
        sol = converge(prob, prov)
        assert sol["converged"]
        ref = solve_bubble_point(_ZF, 100.0, _ABC, _COMPS, N=18, feed_stage=9,
                                 R=4.0, D=40.0, P=760.0, gamma_fn=gfn)
        # both are genuine MESH solves; compositions agree closely
        assert np.allclose(sol["xD"], ref["xD"], atol=2e-2), (sol["xD"], ref["xD"])
        assert np.allclose(sol["xB"], ref["xB"], atol=2e-2), (sol["xB"], ref["xB"])


def test_min_reflux_trend():
    # More reflux -> sharper split (light key richer overhead). A monotone proxy
    # for approaching, then clearing, the minimum-reflux pinch.
    prov = FreeColumnThermo(_ABC)
    xD_lk = []
    for R in (1.5, 3.0, 6.0):
        sol = converge(_column(prov, R=R, B=60.0), prov)
        assert sol["converged"], R
        xD_lk.append(sol["xD"][0])
    assert xD_lk[0] < xD_lk[1] < xD_lk[2], xD_lk


def test_continuation_stress_and_classification():
    # Strong NRTL that stalls a cold-start Newton, rescued by ideal->real homotopy.
    prov = FreeColumnThermo(_ABC, gamma_fn=_nrtl(1.4))
    prob = _column(prov, R=4.0, B=60.0)
    U, info = thermodynamic_homotopy(prob, prov, n_steps=6)
    assert info["converged"], info
    per, _ = mass_balance_residual(U, prob)
    assert np.max(np.abs(per)) < 1e-5

    # An infeasible spec is classified (not diverged): distillate > feed.
    bad = build_problem(n_stages=16, comps=_COMPS, feeds=[(8, 100.0, _ZF)],
                        pressure=760.0, provider=prov,
                        top_spec=OpSpec("distillate_rate", 140.0),
                        bottom_spec=OpSpec("bottoms_rate", 60.0))
    U0 = initialize(bad, prov)
    findings = classify(bad, prov, U0)
    assert any(f.cls == "infeasible_feed_coupling" for f in findings), findings


def test_reactive_atom_balance():
    prov = FreeColumnThermo(_ABC)
    # kinetic A -> B on the mid stages; A+B atoms conserved.
    rx = Reactions(nu=np.array([[-1.0, 1.0, 0.0]]), stages=np.array([5, 6, 7, 8]),
                   kind="kinetic", k_fwd=np.array([0.3]), holdup=1.0)
    prob = build_problem(n_stages=14, comps=["A", "B", "C"],
                         feeds=[(7, 100.0, [0.7, 0.15, 0.15])], pressure=760.0,
                         provider=prov, top_spec=OpSpec("reflux_ratio", 3.0),
                         bottom_spec=OpSpec("bottoms_rate", 55.0), reactions=rx)
    sol = converge(prob, prov)
    assert sol["converged"], sol["info"]
    per, overall = sol["mass_balance"]["per_component"], sol["mass_balance"]["overall"]
    assert np.max(np.abs(per)) < 1e-5, per          # closure incl. generation
    # the reaction actually ran
    assert float(np.nansum(sol["reaction_extent"])) > 1e-3


def test_feasibility_report_shape():
    prov = FreeColumnThermo(_ABC)
    fa = assess_feasibility(_column(prov), prov)
    assert fa["feasible"] and fa["U0"] is not None
    rep = assess(_column(prov), prov, fa["U0"])
    assert rep.structural and rep.physical and rep.thermodynamic


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} validation tests passed.")


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python")))
    _main()
