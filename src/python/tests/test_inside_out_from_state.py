"""Phase F integration check: a WindowState (configured like the GUI, with NRTL
selected) feeds the Inside-Out solver, which converges, closes its balance, emits
the per-stage profiles the Results tab plots, and honours the cancel hook. Qt-free.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _state():
    from gui.state.window_state import (
        WindowState, Species, Stream, StreamType, CondenserType,
    )
    ws = WindowState()
    ws.pressure = 760.0
    ws.num_stages = 20
    ws.feed_stage = 10
    ws.condenser_config.condenser_type = CondenserType.PARTIAL
    ws.condenser_config.reflux_ratio = 3.0
    ws.condenser_config.vapor_distillate_flow = 40.0
    abc = [(6.90565, 1211.033, 220.79), (6.95464, 1344.8, 219.48),
           (6.99052, 1453.43, 215.31)]
    for nm, (a, b, c) in zip(["benzene", "toluene", "xylene"], abc):
        ws.add_species(Species(name=nm))
        p = ws.thermodynamics_config.get_component_params(nm)
        p.antoine_a, p.antoine_b, p.antoine_c = a, b, c
    ws.add_stream(Stream(id="Feed", stream_type=StreamType.FEED, stage=10,
                         flow=100.0, composition={"benzene": 0.4, "toluene": 0.35,
                                                  "xylene": 0.25}))
    return ws


def test_inside_out_from_state():
    from core.column_solvers import solve_inside_out

    ws = _state()
    # turn on NRTL with a couple of binary params so build_gamma_fn returns a fn
    ws.thermodynamics_config.activity_model = "NRTL"
    ws.thermodynamics_config.binary.nrtl_aij[("benzene", "toluene")] = 0.3
    ws.thermodynamics_config.binary.nrtl_aij[("toluene", "benzene")] = 0.2

    order = ws.get_species_names()
    feed = ws.streams["Feed"]
    zF = np.array([feed.composition[nm] for nm in order])
    cp = ws.thermodynamics_config.component_params
    antoine = np.array([[cp[nm].antoine_a, cp[nm].antoine_b, cp[nm].antoine_c]
                        for nm in order])

    prof = solve_inside_out(
        zF, feed.flow, antoine, order, N=ws.num_stages, feed_stage=ws.feed_stage,
        R=ws.condenser_config.reflux_ratio, D=ws.condenser_config.vapor_distillate_flow,
        P=ws.pressure, gamma_fn=ws.build_gamma_fn(order))

    assert prof["found"]
    assert np.allclose(zF * feed.flow,
                       prof["D"] * prof["xD"] + prof["B"] * prof["xB"], atol=1e-3)
    assert prof["xD"][0] > zF[0] > prof["xB"][0]      # benzene concentrates overhead
    # rich per-stage series the Results tab plots
    for key in ("pressure", "liquid_flow", "vapor_flow", "k_values", "enthalpy"):
        assert len(prof[key]) == prof["n_stages"]
    # cancel hook -> aborted result
    ab = solve_inside_out(zF, feed.flow, antoine, order, N=ws.num_stages,
                          feed_stage=ws.feed_stage, R=3.0, D=40.0, P=ws.pressure,
                          cancel=lambda: True)
    assert ab["found"] is False and ab["message"] == "Aborted."
    print(f"inside-out-from-state OK: {prof['iterations']} outer iters")


if __name__ == "__main__":
    test_inside_out_from_state()
