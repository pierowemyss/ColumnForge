"""Phase 3 check: window state survives a to_dict -> pickle -> load_from_dict
round-trip with all column data intact. Qt-free (window_state imports no Qt)."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_roundtrip():
    from gui.state.window_state import (
        WindowState, Species, Stream, StreamType, CondenserType,
        ModuleConfig, ModuleType,
    )
    from core.dof import Spec, SpecKind

    ws = WindowState()
    ws.pressure = 760.0
    ws.light_key_index = 1
    ws.condenser_config.condenser_type = CondenserType.PARTIAL
    ws.condenser_config.reflux_ratio = 3.5
    ws.reboiler_config.boilup_ratio = 1.8
    ws.specs = [Spec(SpecKind.LK_RECOVERY, 0.98, "column")]
    ws.modules["Side Stripper 1"] = ModuleConfig(
        module_type=ModuleType.SIDE_STRIPPER, stage=7, num_stages=4)
    ws.bvm_params = {"r_spin": 12.0, "spec_mode": "Direct", "extract": True}
    for name in ("benzene", "toluene", "xylene"):
        ws.add_species(Species(name=name))
        p = ws.thermodynamics_config.get_component_params(name)
        p.antoine_a = 6.9
    ws.add_stream(Stream(id="Feed", stream_type=StreamType.FEED, stage=10,
                         flow=100.0, composition={"benzene": 0.4, "toluene": 0.35,
                                                  "xylene": 0.25}))

    # to_dict -> pickle (what .colx does) -> load into a fresh state
    blob = pickle.dumps(ws.to_dict())
    ws2 = WindowState()
    ws2.load_from_dict(pickle.loads(blob))

    assert list(ws2.species.keys()) == ["benzene", "toluene", "xylene"]
    assert ws2.streams["Feed"].flow == 100.0
    assert ws2.streams["Feed"].composition["benzene"] == 0.4
    assert ws2.thermodynamics_config.get_component_params("benzene").antoine_a == 6.9
    assert ws2.condenser_config.reflux_ratio == 3.5
    assert ws2.reboiler_config.boilup_ratio == 1.8
    assert ws2.pressure == 760.0
    assert ws2.light_key_index == 1
    assert ws2.specs[0].kind == SpecKind.LK_RECOVERY and ws2.specs[0].value == 0.98
    assert ws2.modules["Side Stripper 1"].module_type == ModuleType.SIDE_STRIPPER
    assert ws2.modules["Side Stripper 1"].stage == 7
    assert ws2.bvm_params["spec_mode"] == "Direct" and ws2.bvm_params["extract"] is True
    print("state-roundtrip self-check OK")


if __name__ == "__main__":
    test_roundtrip()
