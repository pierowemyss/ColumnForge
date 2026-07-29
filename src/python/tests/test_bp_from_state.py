"""Phase 5 integration check: a WindowState configured like the GUI feeds the
bubble-point solver, which converges and closes its mass balance. Qt-free
(window_state and core import no Qt), so it runs in any environment."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_bp_from_state():
    from gui.state.window_state import (
        WindowState, Species, Stream, StreamType, CondenserType,
    )
    from core.column_solvers import solve_bubble_point

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

    # mirror main_window._solve_bubble_point's gather from window_state
    order = ws.get_species_names()
    feed = next(s for s in ws.streams.values()
                if s.stream_type == StreamType.FEED)
    zF = np.array([feed.composition[nm] for nm in order])
    cp = ws.thermodynamics_config.component_params
    antoine = np.array([[cp[nm].antoine_a, cp[nm].antoine_b, cp[nm].antoine_c]
                        for nm in order])

    prof = solve_bubble_point(
        zF, feed.flow, antoine, order, N=ws.num_stages,
        feed_stage=ws.feed_stage, R=ws.condenser_config.reflux_ratio,
        D=ws.condenser_config.vapor_distillate_flow, P=ws.pressure)

    assert prof["found"]
    assert np.allclose(zF * feed.flow,
                       prof["D"] * prof["xD"] + prof["B"] * prof["xB"], atol=1e-3)
    assert prof["xD"][0] > zF[0] > prof["xB"][0]   # benzene concentrates overhead
    print("bp-from-state self-check OK")


if __name__ == "__main__":
    test_bp_from_state()
