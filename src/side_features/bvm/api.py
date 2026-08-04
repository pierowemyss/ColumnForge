"""Public API (blueprint Sec 13, Sec 18.13).

Entry points over the difference-point-chain core:

    size_column(problem, provider, R, S, EF) -> design
    spectrum(problem, provider, R, omegas)   -> [design row per feed position]
    feasibility_map(problem, provider, grid) -> map
    to_solver(design)                        -> init_state

`design` carries the feasibility verdict (classified on failure), stages per
section, feed/draw locations, R_min / min-E/F, and the full top->bottom profiles.
`to_solver` repackages a feasible design as the rigorous MESH solver's warm start
(Sec 12). BVM sizes; it never converges MESH itself.

`spectrum` is the design family: two key recoveries leave C-2 free distillate
splits, and requiring the sections to meet is one equation short of pinning
them, so feasible designs form a one-parameter family indexed by the feed-tray
position omega. Sweeping omega gives N_total against feed location, and the
unique distillate composition that closes the junctions at each one.
"""

import numpy as np

from .driver import (size_column as _size, feasibility_map as _fmap,
                    r_min as _rmin, ef_min as _efmin, spectrum as _spectrum,
                    design_at_omega as _at_omega)
from .handoff import to_solver


def size_column(prob, provider, R, S=None, EF=None, with_limits=True,
                omega=None):
    """Size the column at (R, S, EF); attach R_min (and min E/F) when asked.

    With `omega` the free distillate splits are SOLVED for that feed-tray
    position instead of being left at their trace-floor starting guess, and the
    design reports `exact` / `junction_residual`.
    """
    if omega is not None:
        design, _ = _at_omega(prob, provider, R, float(omega), EF=EF)
    else:
        design = _size(prob, provider, R, S=S, EF=EF)
    if with_limits:
        design["R_min"] = _rmin(prob, provider, S=S, EF=EF)
        if prob.extractive and prob.x_E is not None:
            design["EF_min"] = _efmin(prob, provider, R)
    return design


def spectrum(prob, provider, R, omega_grid, EF=None):
    """The one-parameter family of designs indexed by feed-tray position."""
    return _spectrum(prob, provider, R, omega_grid, EF=EF)


def feasibility_map(prob, provider, R_grid, S_grid=None, EF_grid=None, **kw):
    """Feasibility + stage-count grids over the swept operating parameters.

    `on_step` / `cancelled` pass through to the driver's parallel sweep.
    """
    return _fmap(prob, provider, R_grid, S_grid=S_grid, EF_grid=EF_grid, **kw)


def r_min(prob, provider, **kw):
    return _rmin(prob, provider, **kw)


def ef_min(prob, provider, R, **kw):
    return _efmin(prob, provider, R, **kw)


def _demo():
    from .thermo_adapter import ColumnForgeThermo
    from .problem import build_problem

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = ColumnForgeThermo(abc)
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

    # the design spectrum: each feed position has its OWN distillate composition,
    # solved so the sections meet exactly, and N_total varies over the family.
    rows = spectrum(prob, tp, 4.0, [3, 4, 5, 6, 7])
    exact = [r for r in rows if r["exact"]]
    assert len(exact) >= 3, [r["residual"] for r in rows]
    assert all(r["residual"] < 1e-6 for r in exact)
    xd = [r["xD"][2] for r in exact]            # free (non-key) xylene in x_D
    assert max(xd) > 3 * min(xd), f"x_D should vary across the family: {xd}"
    assert len({r["N_total"] for r in exact}) > 1, "N should vary with feed position"
    print(f"api self-check OK  N={design['N_total']}  R_min={design['R_min']:.2f}  "
          f"map={fm['feasible'].tolist()}  spectrum N="
          f"{[r['N_total'] for r in exact]} xD_nonkey={[f'{v:.1e}' for v in xd]}")


if __name__ == "__main__":
    _demo()
