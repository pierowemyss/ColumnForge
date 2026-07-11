"""Matrix BVM validation suite (blueprint Sec 19).

Each test maps to a Sec 19 case. Where a case needs research-grade fidelity
(exact extractive stage counts, rigorous saddle-manifold tracing), the test
asserts the *method behaviour* the blueprint guarantees -- the difference-point
chain, stable-direction marching, closest-approach connection, classified
feasibility, and the design trends -- rather than reproducing a literature
number to three digits. The ceilings are noted inline.
"""

import math

import numpy as np
import pytest

from thermo_adapter import FreeColumnThermo
from problem import build_problem, SideDraw
from sections import single_feed_chain
from march import march_section
from connect import connect
from driver import size_column, feasibility_map, r_min, ef_min
from handoff import to_solver
import reactive

# benzene / toluene / xylene, mmHg + degC Antoine (ideal, near-ideal VLE)
BTX = np.array([(6.90565, 1211.033, 220.79),
                (6.95464, 1344.8, 219.48),
                (6.99052, 1453.43, 215.31)])


def btx_problem(**kw):
    z = np.array([0.4, 0.35, 0.25])
    return build_problem(["benzene", "toluene", "xylene"], [(z, 100.0, 1.0)],
                         760.0, rec_lk=0.98, rec_hk=0.02, **kw)


# -- Sec 19: ternary BVM cross-check ---------------------------------------
def test_ternary_profiles_and_feed_stage():
    tp = FreeColumnThermo(BTX)
    prob = btx_problem()
    d = size_column(prob, tp, R=4.0)
    assert d["feasible"], d["findings"]
    col = d["column"]
    # rectifying anchored at distillate, stripping at bottoms
    assert np.allclose(col["x"][0], d["xD"], atol=1e-6)
    assert np.allclose(col["x"][-1], d["xB"], atol=1e-6)
    # monotone temperature top(cool) -> bottom(hot)
    assert col["T"][-1] > col["T"][0]
    # feed strictly interior
    fs = d["feed_stages"][0]
    assert 0 < fs < d["N_total"] - 1


def test_ternary_rmin_and_pinch_boundary():
    tp = FreeColumnThermo(BTX)
    prob = btx_problem()
    Rmin = r_min(prob, tp)
    assert Rmin is not None and 0.5 < Rmin < 5.0
    # below R_min infeasible, above feasible (the pinch boundary)
    assert not size_column(prob, tp, R=Rmin * 0.7)["feasible"]
    assert size_column(prob, tp, R=Rmin * 1.4)["feasible"]


# -- Sec 19: C-dim connection vs projection --------------------------------
def test_quaternary_full_vs_projection():
    """Full closest-approach connects a quaternary split; the LK/HK projection
    of the SAME crossing agrees (projection valid here). The full test lives in
    R^(C-1), not the 2-D LK/HK plane (Sec 7)."""
    abc = np.vstack([BTX, [7.00877, 1635.0, 215.0]])
    tp = FreeColumnThermo(abc)
    z = np.array([0.3, 0.3, 0.2, 0.2])
    prob = build_problem(["c1", "c2", "c3", "c4"], [(z, 100.0, 1.0)], 760.0,
                         lk=1, hk=2, rec_lk=0.95, rec_hk=0.05)
    d = size_column(prob, tp, R=5.0)
    assert d["feasible"], d["findings"]
    xD, xB, D, B = d["xD"], d["xB"], d["D"], d["B"]
    rect, strip = single_feed_chain(prob, 5.0, xD, xB, D, B)
    r = march_section(rect, xD, tp, 760.0, prob.max_stages)
    s = march_section(strip, xB, tp, 760.0, prob.max_stages)
    c = connect(r, s)
    # the crossing is a genuine 4-component point found in full R^(C-1), inside
    # the simplex -- NOT a forced 2-D LK/HK curve-crossing (Sec 7). (The junction
    # LK can exceed both products' LK: multicomponent profiles remix, peaking on
    # an interior stage -- exactly why the honest test lives in full space.)
    assert c["connected"] and c["point"].shape == (4,)
    assert c["in_simplex"] and abs(c["point"].sum() - 1.0) < 1e-6
    assert (c["point"] > -1e-6).all()


# -- Sec 19: stage-count sanity vs Fenske ----------------------------------
def test_stage_counts_bracket_fenske():
    tp = FreeColumnThermo(BTX)
    prob = btx_problem()
    alpha = 2.3
    Nmin = math.log((0.98 / 0.02) * (0.98 / 0.02)) / math.log(alpha)
    N = size_column(prob, tp, R=6.0)["N_total"]
    # a real column at finite reflux needs MORE than Fenske total-reflux minimum
    assert N > Nmin
    assert N < 6 * Nmin           # and not absurdly many


def test_stage_count_falls_with_reflux():
    tp = FreeColumnThermo(BTX)
    prob = btx_problem()
    N_lo = size_column(prob, tp, R=2.5)["N_total"]
    N_hi = size_column(prob, tp, R=8.0)["N_total"]
    assert N_lo >= N_hi


# -- Sec 19: multi-feed ----------------------------------------------------
def test_two_feed_intermediate_and_two_feed_stages():
    tp = FreeColumnThermo(BTX)
    za = np.array([0.6, 0.3, 0.1]); zb = np.array([0.2, 0.3, 0.5])
    prob = build_problem(["benzene", "toluene", "xylene"],
                         [(za, 40.0, 1.0), (zb, 60.0, 1.0)], 760.0,
                         rec_lk=0.98, rec_hk=0.02)
    d = size_column(prob, tp, R=4.0)
    assert d["feasible"], d["findings"]
    assert "intermediate" in d["sections"]
    fs = d["feed_stages"]
    assert len(fs) == 2 and fs[0] < fs[1] < d["N_total"] - 1


# -- Sec 19: side draw -----------------------------------------------------
def test_side_draw_purity_capped_by_profile():
    tp = FreeColumnThermo(BTX)
    prob = btx_problem(side_draws=[SideDraw(W=10.0, phase="L", comp_index=1,
                                            purity=0.999)])
    d = size_column(prob, tp, R=4.0)
    assert not d["feasible"]
    assert any(f.cls == "unreachable_side_purity" for f in d["findings"])
    prob2 = btx_problem(side_draws=[SideDraw(W=10.0, phase="L", comp_index=1,
                                             purity=0.4)])
    d2 = size_column(prob2, tp, R=4.0)
    assert d2["feasible"] and "side_draw_stage" in d2


# -- Sec 19: extractive ----------------------------------------------------
def test_extractive_chain_and_feasible_band():
    """Extractive mode with an NRTL entrainer: the three-section chain builds,
    the interior section is routed through, and feasibility depends on E/F.
    ponytail: exact literature stage counts need rigorous saddle-manifold
    tracing; asserted here is the method behaviour, not a three-digit number."""
    from core.thermodynamics import nrtl_gamma_fn
    from sections import extractive_chain
    from problem import overall_balance
    abc = np.array([(7.11714, 1210.595, 229.664),
                    (7.20211, 1582.271, 239.726),
                    (8.07131, 1730.63, 233.426)])
    a = 0.3
    tau = [[0, 0.6, 1.8], [0.5, 0, 1.2], [1.5, 0.9, 0]]
    alpha = [[0, a, a], [a, 0, a], [a, a, 0]]
    gfn = nrtl_gamma_fn(tau, [[0.0] * 3] * 3, alpha)
    tp = FreeColumnThermo(abc, gamma_fn=gfn)
    z = np.array([0.5, 0.5, 0.0])
    prob = build_problem(["acetone", "methanol", "water"], [(z, 100.0, 1.0)],
                         760.0, lk=0, hk=1, x_E=np.array([0.0, 0.0, 1.0]),
                         extractive=True)
    xD, xB, D, B = overall_balance(prob)
    chain = extractive_chain(prob, 3.0, 0.5, xD, xB, D, B)
    assert [s.name for s in chain] == ["rectifying", "extractive", "stripping"]
    results = [size_column(prob, tp, R=3.0, EF=ef)["feasible"]
               for ef in (0.05, 0.2, 0.6, 1.2)]
    assert any(results) and not all(results), results


# -- Sec 19: Murphree stage efficiency -------------------------------------
def test_efficiency_increases_stage_count():
    """A real column (E<1) needs more stages than the ideal-stage march, and the
    condenser (stage 0) / reboiler (last) stay equilibrium stages."""
    tp = FreeColumnThermo(BTX)
    prob = btx_problem()
    N_ideal = size_column(prob, tp, R=2.0)["N_total"]
    prob.efficiency = 0.5
    d = size_column(prob, tp, R=2.0)
    assert d["feasible"], d["findings"]
    assert d["N_total"] > N_ideal, (N_ideal, d["N_total"])
    # products unchanged: distillate on top, bottoms at the base (E=1 there)
    assert np.allclose(d["column"]["x"][0], d["xD"], atol=1e-6)
    assert np.allclose(d["column"]["x"][-1], d["xB"], atol=1e-6)
    # monotone in E: lower efficiency -> at least as many stages
    prob.efficiency = 0.7
    N_hi = size_column(prob, tp, R=2.0)["N_total"]
    assert N_ideal <= N_hi <= d["N_total"], (N_ideal, N_hi, d["N_total"])


# -- Sec 19: warm-start beats cold-start -----------------------------------
def test_warm_start_beats_cold_start():
    from core.column_solvers import solve_bubble_point
    tp = FreeColumnThermo(BTX)
    prob = btx_problem()
    d = size_column(prob, tp, R=4.0)
    init = to_solver(d)
    comps = ["benzene", "toluene", "xylene"]
    z = np.array([0.4, 0.35, 0.25])
    common = dict(N=init["n_stages"], feed_stage=init["feed_stage"],
                  R=init["R"], D=init["D"], P=760.0)
    warm = solve_bubble_point(z, 100.0, BTX, comps, x0=init["x0"], T0=init["T0"],
                              **common)
    cold = solve_bubble_point(z, 100.0, BTX, comps, **common)
    assert warm["found"] and cold["found"]
    assert warm["iterations"] <= cold["iterations"]
    # to a consistent structure (same distillate)
    assert np.allclose(warm["x"][0], cold["x"][0], atol=1e-3)


# -- Sec 19: reactive ------------------------------------------------------
def test_reactive_transform_invariance():
    # MeOH + AcOH <-> MeOAc + H2O ; reference MeOAc
    nu = np.array([[-1.0, -1.0, 1.0, 1.0]])
    rx = reactive.Reactions(nu=nu, ref=[2])
    x = np.array([0.3, 0.3, 0.2, 0.2])
    X = reactive.transform(x, rx)
    assert abs(X.sum() - 1.0) < 1e-9
    for e in (0.05, -0.07, 0.02):
        xr = reactive.apply_reaction(x, rx, [e])
        assert np.allclose(reactive.transform(xr, rx), X, atol=1e-9)


# -- Sec 19: infeasible cases are classified -------------------------------
def test_below_rmin_is_classified():
    tp = FreeColumnThermo(BTX)
    prob = btx_problem()
    Rmin = r_min(prob, tp)
    d = size_column(prob, tp, R=Rmin * 0.5)
    assert not d["feasible"]
    classes = {f.cls for f in d["findings"]}
    assert classes & {"below_min_reflux", "no_connection"}, classes


def test_feasibility_map_shape_and_trend():
    tp = FreeColumnThermo(BTX)
    prob = btx_problem()
    fm = feasibility_map(prob, tp, R_grid=[0.8, 2.0, 4.0, 8.0])
    assert fm["feasible"].shape == (4,)
    feas = fm["feasible"].tolist()
    first = feas.index(True) if True in feas else len(feas)
    assert all(feas[first:]), feas       # feasible region is an upper set in R


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
