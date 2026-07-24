"""Handoff to the rigorous MESH solver (blueprint Sec 12).

Matrix BVM sizes the column; it does NOT converge MESH. `to_solver` packages the
design as a plain data structure -- total stages, feed/draw assignments, the
operating point, and the initial stage profiles x, y, T, L, V (stage 0 = top) --
with no coupling to the solver's internals. ColumnForge's `solve_bubble_point`
consumes it directly through its `x0`/`T0` warm-start hook.
"""

import numpy as np


def to_solver(design):
    """design (from driver.size_column) -> init_state dict for the MESH solver.

    Keys: n_stages, feed_stage, R, D, pressure, comps, and the warm-start arrays
    x0 (N,C), T0 (N,), L0 (N,), V0 (N,), all stage 0 = distillate/top. Raises if
    the design is infeasible (nothing to hand off).
    """
    if not design.get("feasible"):
        raise ValueError("cannot hand off an infeasible design: "
                         + "; ".join(f.cls for f in design.get("findings", [])))
    col = design["column"]
    return {
        "n_stages": int(design["N_total"]),
        "feed_stage": int(col["feed_stage"]),
        "draw_stages": ([int(design["side_draw_stage"])]
                        if "side_draw_stage" in design else []),
        "R": float(design["R"]),
        "D": float(design["D"]),
        "B": float(design["B"]),
        "pressure": float(design["pressure"]),
        "comps": list(design["comps"]),
        "x0": np.asarray(col["x"], float),
        "y0": np.asarray(col["y"], float),
        "T0": np.asarray(col["T"], float),
        "L0": np.asarray(col["liquid_flow"], float),
        "V0": np.asarray(col["vapor_flow"], float),
        "operating_point": {"R": design["R"], "S": design["S"], "EF": design["EF"]},
    }


def _demo():
    from .thermo_adapter import ColumnForgeThermo
    from .problem import build_problem
    from .driver import size_column

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = ColumnForgeThermo(abc)
    z = np.array([0.4, 0.35, 0.25])
    prob = build_problem(["b", "t", "x"], [(z, 100.0, 1.0)], 760.0,
                         rec_lk=0.98, rec_hk=0.02)
    d = size_column(prob, tp, R=4.0)
    init = to_solver(d)

    N = init["n_stages"]
    assert init["x0"].shape == (N, 3) and init["T0"].shape == (N,)
    assert np.allclose(init["x0"].sum(axis=1), 1.0, atol=1e-6)
    assert 0 < init["feed_stage"] < N - 1
    assert init["T0"][-1] > init["T0"][0]

    # the structure feeds the rigorous solver's warm-start hook directly
    from core.column_solvers import solve_bubble_point
    sol = solve_bubble_point(
        z, 100.0, abc, ["b", "t", "x"], N=N, feed_stage=init["feed_stage"],
        R=init["R"], D=init["D"], P=760.0, x0=init["x0"], T0=init["T0"])
    assert sol["found"], sol.get("message")
    print(f"handoff self-check OK  N={N} feed@{init['feed_stage']}  "
          f"MESH converged in {sol['iterations']} iters")


if __name__ == "__main__":
    _demo()
