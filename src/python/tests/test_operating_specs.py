"""Operating-point resolver: a WindowState carrying any 2 operating specs
resolves to (R, D) and the solved column honours them — algebraic specs exactly
and implicit (purity/recovery) targets to tolerance. Qt-free."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _benzene_column():
    from gui.state.window_state import (
        WindowState, Species, Stream, StreamType, ComponentThermoParams,
    )
    ws = WindowState()
    ws.pressure = 760.0
    ws.num_stages = 20
    ws.feed_stage = 10
    ws.light_key_index = 0
    abc = [(6.90565, 1211.033, 220.79), (6.95464, 1344.8, 219.48),
           (6.99052, 1453.43, 215.31)]
    for nm, (a, b, c) in zip(["benzene", "toluene", "xylene"], abc):
        ws.add_species(Species(name=nm))
        ws.thermodynamics_config.component_params[nm] = ComponentThermoParams(
            antoine_a=a, antoine_b=b, antoine_c=c)
    ws.add_stream(Stream(id="Feed", stream_type=StreamType.FEED, stage=10,
                         flow=100.0,
                         composition={"benzene": 0.5, "toluene": 0.3, "xylene": 0.2}))
    return ws


def _resolve(ws):
    from core.dof import OPERATING_KINDS
    from core.operating_specs import resolve_operating_point
    from core.column_solvers import solve_bubble_point
    order = ws.get_species_names()
    feed = next(s for s in ws.streams.values()
                if s.stream_type.name == "FEED")
    zF = np.array([feed.composition[nm] for nm in order])
    ant = ws.thermodynamics_config.psat_params(order)
    ops = [s for s in ws.collect_specs() if s.kind in OPERATING_KINDS]

    def sfn(R, D):
        return solve_bubble_point(zF, feed.flow, ant, order, N=ws.num_stages,
                                  feed_stage=ws.feed_stage, R=R, D=D, P=ws.pressure)

    R, D = resolve_operating_point(ops, feed.flow, zF, solve_fn=sfn,
                                   lk=ws.light_key_index, hk=ws.heavy_key_index)
    return R, D, sfn(R, D)


def test_algebraic_reflux_distillate_is_exact():
    from core.dof import Spec, SpecKind
    ws = _benzene_column()
    ws.specs = [Spec(SpecKind.REFLUX_RATIO, 3.0, "op0"),
                Spec(SpecKind.DISTILLATE_RATE, 40.0, "op1")]
    assert ws.get_specification_status()[2]   # perfectly specified, can run
    R, D, _ = _resolve(ws)
    assert abs(R - 3.0) < 1e-6 and abs(D - 40.0) < 1e-6


def test_distillate_purity_target_is_hit():
    from core.dof import Spec, SpecKind
    ws = _benzene_column()
    ws.specs = [Spec(SpecKind.REFLUX_RATIO, 3.0, "op0"),
                Spec(SpecKind.DIST_PURITY, 0.95, "op1", component=0)]
    R, D, prof = _resolve(ws)
    assert abs(R - 3.0) < 1e-6
    assert abs(prof["xD"][0] - 0.95) < 2e-3


def test_light_key_recovery_target_is_hit():
    from core.dof import Spec, SpecKind
    ws = _benzene_column()
    ws.specs = [Spec(SpecKind.REFLUX_RATIO, 3.0, "op0"),
                Spec(SpecKind.LK_RECOVERY, 0.9, "op1")]
    _, D, prof = _resolve(ws)
    rec = D * prof["xD"][0] / (100.0 * 0.5)
    assert abs(rec - 0.9) < 2e-3


def test_overspecification_blocks_run():
    from core.dof import Spec, SpecKind
    ws = _benzene_column()
    ws.specs = [Spec(SpecKind.REFLUX_RATIO, 3.0, "op0"),
                Spec(SpecKind.BOILUP_RATIO, 1.5, "op1"),
                Spec(SpecKind.DISTILLATE_RATE, 40.0, "op2")]
    icon, message, can_run = ws.get_specification_status()
    assert not can_run and "Over-specified" in message


if __name__ == "__main__":
    test_algebraic_reflux_distillate_is_exact()
    test_distillate_purity_target_is_hit()
    test_light_key_recovery_target_is_hit()
    test_overspecification_blocks_run()
    print("operating-specs integration self-check OK")
