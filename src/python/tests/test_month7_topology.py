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


if __name__ == "__main__":
    test_interheater_duties_from_state()
    test_duty_survives_colx_roundtrip()
    test_interheater_offloads_reboiler()
    test_kw_to_kjh_conversion()
    print("month7 topology (interheater duty) checks OK")
