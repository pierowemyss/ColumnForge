"""Month 7 — complex topology, slice 1: interheater/intercooler duty modules.

An Interreboiler/-cooler is just a signed per-stage heat term the energy balance
already consumes (si.duty[]). These checks pin the wiring end to end: state ->
duty list, .colx roundtrip, and that the solver actually offloads the reboiler
when heat is added mid-column. Qt-free."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_interheater_duties_from_state():
    from gui.state.window_state import WindowState, ModuleConfig, ModuleType
    ws = WindowState()
    ws.add_module("Interreboiler 1",
                  ModuleConfig(module_type=ModuleType.INTERREBOILER, stage=12,
                               duty=250.0))
    ws.add_module("Side Stripper 1",           # no duty -> not wired
                  ModuleConfig(module_type=ModuleType.SIDE_STRIPPER, stage=6))
    assert ws.interheater_duties() == [(12, 250.0)], ws.interheater_duties()


def test_duty_survives_colx_roundtrip():
    from gui.state.window_state import WindowState, ModuleConfig, ModuleType
    ws = WindowState()
    ws.add_module("Intercooler 1",
                  ModuleConfig(module_type=ModuleType.INTERREBOILER, stage=4,
                               duty=-125.5))
    ws2 = WindowState()
    ws2.load_from_dict(ws.to_dict())
    assert ws2.modules["Intercooler 1"].duty == -125.5


def test_interheater_offloads_reboiler():
    """Adding heat mid-column (a real si.duty entry) must cut the reboiler duty
    under the energy balance — otherwise the term is being ignored."""
    from core.solver_input import build_solver_input
    from core.column_solvers import solve_inside_out, make_energy_balance

    antoine = np.array([[6.90565, 1211.033, 220.79],
                        [6.95464, 1344.8, 219.48],
                        [6.99052, 1453.43, 215.31]])
    eb_args = (np.array([136.0, 157.0, 186.0]), np.array([30.8, 33.2, 36.2]),
               np.array([353.2, 383.8, 417.6]), np.array([562.0, 591.8, 630.3]))

    def run(duties):
        si = build_solver_input(
            n_stages=16, comps=["benzene", "toluene", "xylene"], antoine=antoine,
            feeds=[(8, 100.0, [0.4, 0.35, 0.25])], duties=duties,
            R=3.0, D=40.0, pressure=760.0)
        return solve_inside_out(si, flows_hook=make_energy_balance(*eb_args),
                                max_iter=120)

    base = run(())
    Q = 3.0e5                                  # kJ/h of interreboiler heat, stage 12
    heated = run([(12, Q)])
    # heat added below the feed offloads the reboiler
    assert heated["reboiler_duty"] < base["reboiler_duty"], \
        (heated["reboiler_duty"], base["reboiler_duty"])
    drop = base["reboiler_duty"] - heated["reboiler_duty"]
    assert 0.2 * Q < drop < 1.2 * Q, (drop, Q)   # a real, bounded fraction of Q


def test_kw_to_kjh_conversion():
    # the main_window wiring converts entered kW -> kJ/h via /KJH_TO_KW (= x3600)
    from core.thermodynamics import KJH_TO_KW
    assert abs((250.0 / KJH_TO_KW) - 250.0 * 3600.0) < 1e-6


# --- Pumparounds -----------------------------------------------------------
# A pumparound draws liquid rate P at stage i, cools it (removing duty Q), and
# returns it to a higher stage j (j < i). It is an internal recycle: products
# (D, B) are unchanged; only the internal liquid traffic between j and i and the
# heat load shift. These pin the material + energy bookkeeping end to end.

_BTX_ANTOINE = np.array([[6.90565, 1211.033, 220.79],
                         [6.95464, 1344.8, 219.48],
                         [6.99052, 1453.43, 215.31]])
_BTX_EB = (np.array([136.0, 157.0, 186.0]), np.array([30.8, 33.2, 36.2]),
           np.array([353.2, 383.8, 417.6]), np.array([562.0, 591.8, 630.3]))
_BTX_Z = np.array([0.4, 0.35, 0.25])


def _btx(pumparounds=(), energy=True, max_iter=300):
    from core.solver_input import build_solver_input
    from core.column_solvers import solve_inside_out, make_energy_balance
    si = build_solver_input(
        n_stages=16, comps=["benzene", "toluene", "xylene"], antoine=_BTX_ANTOINE,
        feeds=[(8, 100.0, list(_BTX_Z))], pumparounds=pumparounds,
        R=3.0, D=40.0, pressure=760.0)
    hook = make_energy_balance(*_BTX_EB) if energy else None
    return si, solve_inside_out(si, flows_hook=hook, max_iter=max_iter)


def test_pumparound_degenerate_is_noop():
    """Draw and return on the SAME stage (Q=0) is an exact no-op: the empty
    [j, i) interval leaves flows untouched and the injected source cancels the
    draw. (Built white-box: a real pumparound requires j < i.)"""
    from core.solver_input import build_solver_input
    from core.column_solvers import solve_inside_out, make_energy_balance
    _, base = _btx()
    si = build_solver_input(
        n_stages=16, comps=["benzene", "toluene", "xylene"], antoine=_BTX_ANTOINE,
        feeds=[(8, 100.0, list(_BTX_Z))], R=3.0, D=40.0, pressure=760.0)
    si.pumparounds = np.array([[6., 6., 15., 0.]])
    noop = solve_inside_out(si, flows_hook=make_energy_balance(*_BTX_EB), max_iter=300)
    assert np.max(np.abs(noop["xD"] - base["xD"])) < 1e-5
    assert np.max(np.abs(noop["xB"] - base["xB"])) < 1e-5


def test_pumparound_preserves_products():
    """P recycles internally, so distillate/bottoms rates are unchanged and the
    overall component mass balance still closes (draw is not a product)."""
    _, base = _btx()
    _, pa = _btx(pumparounds=[(10, 5, 20.0, 0.0)])
    assert abs(pa["D"] - base["D"]) < 1e-9 and abs(pa["B"] - base["B"]) < 1e-9
    closure = np.max(np.abs(100.0 * _BTX_Z - (pa["D"] * pa["xD"] + pa["B"] * pa["xB"])))
    assert closure < 1e-3, closure


def test_pumparound_raises_internal_liquid():
    """Under CMO the recycled P raises the liquid flow by exactly P on the
    stages it circulates over, [j-1, i-1) in top-down index terms, and nowhere
    else."""
    _, base = _btx(energy=False)
    _, pa = _btx(pumparounds=[(10, 5, 20.0, 0.0)], energy=False)
    diff = pa["liquid_flow"] - base["liquid_flow"]
    expect = np.zeros(16); expect[4:9] = 20.0          # i=10, j=5 -> indices 4..8
    assert np.allclose(diff, expect, atol=1e-6), np.round(diff, 3)


def test_pumparound_heat_loads_reboiler():
    """Removing Q mid-column must be made up by the reboiler under the energy
    balance (the mirror of the interheater test), and the balance stays closed."""
    Q = 3.0e5
    _, base = _btx()
    _, pa = _btx(pumparounds=[(10, 5, 20.0, Q)])
    rise = pa["reboiler_duty"] - base["reboiler_duty"]
    assert 0.5 * Q < rise < 1.5 * Q, (rise, Q)


def test_pumparound_from_state():
    from gui.state.window_state import WindowState, ModuleConfig, ModuleType
    ws = WindowState()
    ws.add_module("Pumparound 1",
                  ModuleConfig(module_type=ModuleType.PUMPAROUND, stage=12,
                               return_stage=6, rate=50.0, duty=200.0))
    ws.add_module("Interreboiler 1",           # not a pumparound -> excluded
                  ModuleConfig(module_type=ModuleType.INTERREBOILER, stage=4,
                               duty=100.0))
    assert ws.pumparounds() == [(12, 6, 50.0, 200.0)], ws.pumparounds()
    # the pumparound's duty is its cooler; interheater_duties() must not also
    # claim it, or si.duty gets +Q at the draw as well as -Q at the return and
    # the column solves almost adiabatic.
    assert ws.interheater_duties() == [(4, 100.0)], ws.interheater_duties()
    # a pumparound missing its rate/return is not yet solvable -> excluded
    ws.add_module("Pumparound 2",
                  ModuleConfig(module_type=ModuleType.PUMPAROUND, stage=10))
    assert ws.pumparounds() == [(12, 6, 50.0, 200.0)]


def test_pumparound_survives_colx_roundtrip():
    from gui.state.window_state import WindowState, ModuleConfig, ModuleType
    ws = WindowState()
    ws.add_module("Pumparound 1",
                  ModuleConfig(module_type=ModuleType.PUMPAROUND, stage=14,
                               return_stage=5, rate=42.5, duty=175.0))
    ws2 = WindowState()
    ws2.load_from_dict(ws.to_dict())
    m = ws2.modules["Pumparound 1"]
    assert (m.return_stage, m.rate, m.duty) == (5, 42.5, 175.0)


def _side_section_column(kind, draw, ret, ratio, rate=30.0):
    """Solve the 3-component reference column with one side section attached,
    through the same tear the GUI uses. Returns (profile, section)."""
    from core.solver_input import build_solver_input
    from core.column_solvers import solve_bubble_point
    from core.side_sections import SideSection, make_side_solver

    antoine = np.array([[6.90565, 1211.033, 220.79],
                        [6.95464, 1344.8, 219.48],
                        [6.99052, 1453.43, 215.31]])
    z = np.array([0.4, 0.35, 0.25])
    sec = SideSection(id="S1", kind=kind, draw_stage=draw, return_stage=ret,
                      rate=rate, ratio=ratio, n_stages=4)

    def build(_si=None):
        feeds = [(8, 100.0, z)]
        if sec.return_comp is not None:
            feeds.append((sec.return_stage, sec.return_flow, sec.return_comp,
                          sec.return_q))
        return build_solver_input(
            n_stages=16, comps=["benzene", "toluene", "xylene"], feeds=feeds,
            draws=[(sec.draw_stage, *sec.draw_rates())],
            R=3.0, D=35.0, pressure=760.0, antoine=antoine)

    prof = make_side_solver(solve_bubble_point, [sec], build)(build(),
                                                             max_iter=300)
    return prof, sec


def test_side_stripper_closes_the_mass_balance():
    """The draw is internal and the return is a recycle: what leaves the system
    is D + B + the stripper's bottoms, and it must equal the external feed."""
    prof, sec = _side_section_column("stripper", 11, 10, 1.5)
    ss = prof["side_sections"][0]
    assert abs(ss["flow"] - 30.0 / 2.5) < 1e-9
    out = prof["D"] * prof["xD"] + prof["B"] * prof["xB"] + ss["flow"] * ss["comp"]
    assert np.allclose(np.array([40.0, 35.0, 25.0]), out, atol=1e-3), out
    # the recycle is not external feed, and the internal draw is not a product
    assert np.allclose(prof["feed_totals"], [40.0, 35.0, 25.0], atol=1e-3)
    assert prof["side_draws"] == []
    # stripping strips: the product is leaner in the light key than the draw
    assert ss["comp"][0] < prof["x"][ss["stage"]][0]


def test_side_rectifier_closes_the_mass_balance():
    prof, sec = _side_section_column("rectifier", 6, 7, 2.0, rate=25.0)
    sr = prof["side_sections"][0]
    assert abs(sr["flow"] - 25.0 / 3.0) < 1e-9
    out = prof["D"] * prof["xD"] + prof["B"] * prof["xB"] + sr["flow"] * sr["comp"]
    assert np.allclose(np.array([40.0, 35.0, 25.0]), out, atol=1e-3), out
    # rectifying enriches: the product beats the vapour drawn
    assert sr["comp"][0] > prof["y"][sr["stage"]][0]


def test_side_section_return_direction_is_enforced():
    from core.side_sections import SideSection
    import pytest
    with pytest.raises(ValueError):
        SideSection(id="S", kind="stripper", draw_stage=8, return_stage=9,
                    rate=10.0, ratio=1.0)
    with pytest.raises(ValueError):
        SideSection(id="S", kind="rectifier", draw_stage=8, return_stage=7,
                    rate=10.0, ratio=1.0)


def test_side_sections_from_state_and_dof():
    """Each module type adds its own specs, and the ledger balances only once
    they are set — a module used to leave the column permanently 'under'."""
    from gui.state.window_state import (WindowState, ModuleConfig, ModuleType,
                                        Species)
    from core.dof import SpecKind
    ws = WindowState()
    for nm in ("benzene", "toluene"):
        ws.add_species(Species(name=nm))
    ws.upsert_operating_spec(SpecKind.REFLUX_RATIO, 3.0)
    ws.upsert_operating_spec(SpecKind.DISTILLATE_RATE, 40.0)
    assert ws.analyze_dof().status == "exact"

    ws.add_module("Side Stripper 1",
                  ModuleConfig(module_type=ModuleType.SIDE_STRIPPER, stage=11,
                               return_stage=10, rate=30.0, num_stages=4))
    assert ws.side_sections() == []                    # no ratio yet
    assert ws.analyze_dof().status == "under"          # and the ledger says so

    ws.modules["Side Stripper 1"].boilup_ratio = 1.5
    assert ws.side_sections() == [("Side Stripper 1", "stripper", 11, 10,
                                   30.0, 1.5, 4)]
    assert ws.analyze_dof().status == "exact"

    # a stripper carries no heat term, so it stays valid under CMO
    assert not ws.energy_balance
    assert ws.analyze_dof().can_run


def test_side_stripper_survives_colx_roundtrip():
    from gui.state.window_state import WindowState, ModuleConfig, ModuleType
    ws = WindowState()
    ws.add_module("Side Stripper 1",
                  ModuleConfig(module_type=ModuleType.SIDE_STRIPPER, stage=11,
                               return_stage=10, rate=30.0, boilup_ratio=1.5,
                               num_stages=5))
    ws2 = WindowState()
    ws2.load_from_dict(ws.to_dict())
    m = ws2.modules["Side Stripper 1"]
    assert (m.return_stage, m.rate, m.boilup_ratio, m.num_stages) == (
        10, 30.0, 1.5, 5)


if __name__ == "__main__":
    test_interheater_duties_from_state()
    test_duty_survives_colx_roundtrip()
    test_interheater_offloads_reboiler()
    test_kw_to_kjh_conversion()
    test_pumparound_degenerate_is_noop()
    test_pumparound_preserves_products()
    test_pumparound_raises_internal_liquid()
    test_pumparound_heat_loads_reboiler()
    test_pumparound_from_state()
    test_pumparound_survives_colx_roundtrip()
    test_side_stripper_closes_the_mass_balance()
    test_side_rectifier_closes_the_mass_balance()
    test_side_section_return_direction_is_enforced()
    test_side_sections_from_state_and_dof()
    test_side_stripper_survives_colx_roundtrip()
    print("month7 topology (interheater duty + pumparound + side sections) OK")
