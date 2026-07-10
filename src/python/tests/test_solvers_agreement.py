"""Phase 8 checks the plan calls for explicitly:
  - solve_bubble_point vs solve_inside_out agreement on one column
  - material-balance closure: single feed, two feeds, and a side draw,
    both in core.material_balance and through the rigorous solver
  - BVM 3-section extractive feasibility
Qt-free."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))          # for side_features.bvm

ANTOINE = np.array([
    [6.90565, 1211.033, 220.79],           # benzene
    [6.95464, 1344.8,   219.48],           # toluene
    [6.99052, 1453.43,  215.31],           # p-xylene
])
COMPS = ["benzene", "toluene", "xylene"]
Z = [0.4, 0.35, 0.25]


def _si(**over):
    from core.solver_input import build_solver_input
    kw = dict(n_stages=20, comps=COMPS, antoine=ANTOINE,
              feeds=[(10, 100.0, Z)], R=3.0, D=40.0, pressure=760.0)
    kw.update(over)
    return build_solver_input(**kw)


def test_bubble_point_inside_out_agreement():
    from core.column_solvers import solve_bubble_point, solve_inside_out

    si = _si()
    bp = solve_bubble_point(si)
    io = solve_inside_out(si)
    assert bp["found"] and io["found"]
    # same column, same thermo -> products agree closely
    assert np.allclose(bp["xD"], io["xD"], atol=2e-2), (bp["xD"], io["xD"])
    assert np.allclose(bp["xB"], io["xB"], atol=2e-2), (bp["xB"], io["xB"])
    assert np.allclose(bp["T"], io["T"], atol=2.0)
    # both top -> bottom: reboiler hotter than condenser stage
    for prof in (bp, io):
        assert prof["T"][-1] > prof["T"][0]


def test_solver_component_closure_multifeed_sidedraw():
    from core.column_solvers import solve_bubble_point

    si = _si(feeds=[(14, 60.0, [0.6, 0.3, 0.1]), (6, 40.0, [0.1, 0.4, 0.5])],
             draws=[(10, 15.0, 0.0)], D=35.0)
    prof = solve_bubble_point(si)
    assert prof["found"]

    F_in = 60.0 * np.array([0.6, 0.3, 0.1]) + 40.0 * np.array([0.1, 0.4, 0.5])
    D, B = prof["D"], prof["B"]
    out = D * prof["xD"] + B * prof["xB"]
    for sd in prof["side_draws"]:
        out += sd["liquid"] * sd["x"] + sd["vapor"] * sd["y"]
    assert abs((D + B + 15.0) - 100.0) < 1e-8          # total balance
    assert np.allclose(out, F_in, atol=1e-3), (out, F_in)  # per-component
    assert prof["feed_stages"] == [5, 13]              # 0-based from top


def test_material_balance_closure():
    from core.material_balance import overall_balance

    feeds = [(60.0, [0.6, 0.3, 0.1]), (40.0, [0.1, 0.4, 0.5])]
    F_in = 60.0 * np.array([0.6, 0.3, 0.1]) + 40.0 * np.array([0.1, 0.4, 0.5])

    # single + multi-feed, with and without a side draw
    for draws, w in (((), 0.0), (((20.0, None),), 20.0)):
        xD, D, xB, B = overall_balance(
            feeds, lk=0, spec_mode="recovery", FR_LK=0.98, NK_spec=1e-3,
            alpha=np.array([2.4, 1.0, 0.4]), hk=1, side_draws=draws)
        pool = F_in * (1.0 - w / 100.0)                # draw at mixed-feed comp
        assert abs(D + B + w - 100.0) < 1e-9
        assert np.allclose(D * xD + B * xB, pool, atol=1e-9)


def test_stage_efficiency_degrades_split():
    from core.column_solvers import solve_bubble_point, solve_inside_out

    si = _si()
    ideal = solve_bubble_point(si)
    eff = solve_bubble_point(si, efficiency=0.6)
    assert eff["found"] and np.allclose(eff["x"].sum(axis=1), 1.0, atol=1e-8)
    # E=1 is a no-op; a lower Murphree efficiency must worsen the top purity
    assert np.allclose(solve_bubble_point(si, efficiency=1.0)["x"], ideal["x"],
                       atol=1e-12)
    assert eff["xD"][0] < ideal["xD"][0]
    io_eff = solve_inside_out(si, efficiency=0.6)
    assert io_eff["found"] and io_eff["xD"][0] < solve_inside_out(si)["xD"][0]


def test_bvm_activity_model_shifts_profile():
    from side_features.bvm.solver import bound_val_method, build_column_profile
    from core.thermodynamics import nrtl_gamma_fn

    a = 0.1
    gfn = nrtl_gamma_fn([[0.0, a, a], [a, 0.0, a], [a, a, 0.0]],
                        [[0.0] * 3] * 3,
                        [[0.0, 0.3, 0.3], [0.3, 0.0, 0.3], [0.3, 0.3, 0.0]])
    kw = dict(zF=Z, F=100.0, r=12.0, q=1.0, antoine=ANTOINE, comps=COMPS, lk=0,
              P=760.0, spec_mode="recovery", FR_LK=0.98, NK_spec=1e-3)
    ni = bound_val_method(gamma_fn=gfn, **kw)
    idl = bound_val_method(**kw)
    m = min(len(ni["xRect"]), len(idl["xRect"]))
    assert np.abs(ni["xRect"][:m] - idl["xRect"][:m]).max() > 1e-2
    assert build_column_profile(ni)["found"]


def test_bvm_three_section_extractive_feasible():
    from side_features.bvm.solver import bound_val_method, build_column_profile

    res = bound_val_method(Z, F=100.0, r=8.0, q=1.0, antoine=ANTOINE,
                           comps=COMPS, lk=0, P=760.0, spec_mode="recovery",
                           FR_LK=0.98, NK_spec=1e-3,
                           extract=True, E2F=0.3, xE=[0.0, 0.0, 1.0])
    assert [s.name for s in res["sections"]] == \
        ["rectifying", "extractive", "stripping"]
    prof = build_column_profile(res)
    assert prof["found"], prof["message"]
    assert prof["entrainer_stage"] <= prof["feed_stage"]   # entrainer above feed
    assert len(prof["x"]) == len(prof["T"]) == prof["n_stages"]


if __name__ == "__main__":
    test_bubble_point_inside_out_agreement()
    test_solver_component_closure_multifeed_sidedraw()
    test_material_balance_closure()
    test_stage_efficiency_degrades_split()
    test_bvm_activity_model_shifts_profile()
    test_bvm_three_section_extractive_feasible()
    print("solver-agreement checks OK")
