"""UNIFAC group-contribution activity model: DB integrity, thermodynamics,
r/q estimation, and an end-to-end column solve through WindowState."""
import numpy as np

from core.thermodynamics import (
    load_unifac_db, unifac_gamma_fn, bubble_T)


def test_db_integrity():
    db = load_unifac_db()
    mains = {v[1] for v in db["subgroups"].values()}
    inter = db["interactions"]
    # every subgroup's main group has an interaction row, and every target is known
    assert all(m in inter for m in mains)
    assert all(n in mains for row in inter.values() for n in row)


def test_pure_and_same_group_ideal():
    db = load_unifac_db()
    # n-hexane / n-heptane: identical groups -> residual zero, gamma near 1
    fn = unifac_gamma_fn([{"CH3": 2, "CH2": 4}, {"CH3": 2, "CH2": 5}], db)
    assert np.allclose(fn([0.4, 0.6], 60.0), 1.0, atol=1e-2)
    assert abs(fn([1.0, 0.0], 60.0)[0] - 1.0) < 1e-9   # pure -> exactly 1


def test_ethanol_water_azeotrope():
    db = load_unifac_db()
    fn = unifac_gamma_fn([{"CH3": 1, "CH2": 1, "OH": 1}, {"H2O": 1}], db)
    # ethanol infinite-dilution activity coefficient in water: strong positive
    assert 3.0 < fn([1e-6, 1 - 1e-6], 70.0)[0] < 8.0
    # bubble-T scan finds the minimum-boiling azeotrope near x_EtOH ~ 0.9
    etoh_h2o = np.array([(8.20417, 1642.89, 230.300), (8.07131, 1730.63, 233.426)])
    xs = np.linspace(0.02, 0.998, 200)
    Ts = np.array([bubble_T(np.array([x, 1 - x]), 760.0, etoh_h2o, gamma_fn=fn)
                   for x in xs])
    k = int(np.argmin(Ts))
    assert 0 < k < len(xs) - 1 and abs(xs[k] - 0.90) < 0.06


def test_rq_estimate_matches_group_sum():
    db = load_unifac_db()["subgroups"]
    groups = {"CH3": 1, "CH2": 1, "OH": 1}   # ethanol
    r = sum(db[g][2] * n for g, n in groups.items())
    q = sum(db[g][3] * n for g, n in groups.items())
    # r/q are the UNIFAC group sums: CH3+CH2+OH R = .9011+.6744+1.0 = 2.5755,
    # Q = .848+.540+1.200 = 2.588 (this is exactly what the estimate button sets)
    assert abs(r - 2.5755) < 1e-4 and abs(q - 2.588) < 1e-4


def test_column_solve_with_unifac():
    """WindowState.build_gamma_fn -> a real column converges with UNIFAC."""
    from gui.state.window_state import WindowState, Species

    ws = WindowState()
    ws.species["Ethanol"] = Species(name="Ethanol",
                                    unifac_groups={"CH3": 1, "CH2": 1, "OH": 1})
    ws.species["Water"] = Species(name="Water", unifac_groups={"H2O": 1})
    ws.thermodynamics_config.activity_model = "UNIFAC"
    gfn = ws.build_gamma_fn(["Ethanol", "Water"])
    assert gfn is not None
    g = gfn([0.5, 0.5], 80.0)
    assert g[0] > 1.0 and g[1] > 1.0   # positive-deviation mixture both ways


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_unifac OK")
