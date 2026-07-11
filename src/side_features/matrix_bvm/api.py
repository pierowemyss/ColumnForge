"""Public API (blueprint Sec 13, Sec 18.13).

Three entry points over the difference-point-chain core:

    size_column(problem, provider, R, S, EF) -> design
    feasibility_map(problem, provider, grid) -> map
    to_solver(design)                        -> init_state

`design` carries the feasibility verdict (classified on failure), stages per
section, feed/draw locations, R_min / min-E/F, and the full top->bottom profiles.
`to_solver` repackages a feasible design as the rigorous MESH solver's warm start
(Sec 12). Matrix BVM sizes; it never converges MESH itself.
"""

import numpy as np

from driver import (size_column as _size, feasibility_map as _fmap,
                    r_min as _rmin, ef_min as _efmin)
from handoff import to_solver


def size_column(prob, provider, R, S=None, EF=None, with_limits=True):
    """Size the column at (R, S, EF); attach R_min (and min E/F) when asked."""
    design = _size(prob, provider, R, S=S, EF=EF)
    if with_limits:
        design["R_min"] = _rmin(prob, provider, S=S, EF=EF)
        if prob.extractive and prob.x_E is not None:
            design["EF_min"] = _efmin(prob, provider, R)
    return design


def feasibility_map(prob, provider, R_grid, S_grid=None, EF_grid=None):
    """Feasibility + stage-count grids over the swept operating parameters."""
    return _fmap(prob, provider, R_grid, S_grid=S_grid, EF_grid=EF_grid)


def r_min(prob, provider, **kw):
    return _rmin(prob, provider, **kw)


def ef_min(prob, provider, R, **kw):
    return _efmin(prob, provider, R, **kw)


def _demo():
    from thermo_adapter import FreeColumnThermo
    from problem import build_problem

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = FreeColumnThermo(abc)
    z = np.array([0.4, 0.35, 0.25])
    prob = build_problem(["benzene", "toluene", "xylene"], [(z, 100.0, 1.0)],
                         760.0, rec_lk=0.98, rec_hk=0.02)

    design = size_column(prob, tp, R=4.0)
    assert design["feasible"], design["findings"]
    assert design["R_min"] is not None and design["R_min"] < 4.0
    assert design["N_total"] > design["feed_stages"][0] > 0

    init = to_solver(design)
    assert init["x0"].shape[0] == design["N_total"]

    fm = feasibility_map(prob, tp, R_grid=[1.0, 2.5, 5.0])
    assert fm["feasible"].shape == (3,)
    print(f"api self-check OK  N={design['N_total']}  R_min={design['R_min']:.2f}  "
          f"map={fm['feasible'].tolist()} stages={fm['stages'].tolist()}")


if __name__ == "__main__":
    _demo()
