"""`.colx` is JSON now (ADR-0001): a WindowState survives save_colx -> file ->
load_colx with all column data intact, and the file is human-readable JSON, not
pickle. Qt-free."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_colx_json_roundtrip():
    from gui.state.window_state import (
        WindowState, Species, Stream, StreamType, CondenserType,
        ModuleConfig, ModuleType,
    )
    from gui.state.persistence import save_colx, load_colx, SCHEMA_VERSION
    from core.dof import Spec, SpecKind

    ws = WindowState()
    ws.pressure = 760.0
    ws.light_key_index = 1
    ws.condenser_config.condenser_type = CondenserType.PARTIAL
    ws.condenser_config.reflux_ratio = 3.5
    ws.reboiler_config.boilup_ratio = 1.8
    ws.specs = [Spec(SpecKind.LK_RECOVERY, 0.98, "column")]
    ws.modules["Side Stripper 1"] = ModuleConfig(
        module_type=ModuleType.SIDE_STRIPPER, stage=7, num_stages=4,
        return_stage=6, rate=30.0, boilup_ratio=1.5)
    ws.bvm_params = {"r_spin": 12.0, "spec_mode": "Direct", "extract": True,
                     # reactive-distillation spec: nested dict, species-keyed
                     "reaction": {"on": True, "ref": "xylene", "keq_a": 2.3,
                                  "keq_b": -150.0,
                                  "nu": {"benzene": -1.0, "toluene": -1.0,
                                         "xylene": 1.0}}}
    for name in ("benzene", "toluene", "xylene"):
        ws.add_species(Species(name=name))
        ws.thermodynamics_config.get_component_params(name).antoine_a = 6.9
    # tuple-keyed binary params — the JSON-hostile path
    ws.thermodynamics_config.binary.nrtl_aij[("benzene", "toluene")] = 0.123
    ws.add_stream(Stream(id="Feed", stream_type=StreamType.FEED, stage=10,
                         flow=100.0, composition={"benzene": 0.4, "toluene": 0.35,
                                                  "xylene": 0.25}))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "case.colx")
        save_colx(path, ws.to_dict(), name="t")

        # it is real, inspectable JSON — not a pickle blob
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        doc = json.loads(raw)
        assert doc["schema_version"] == SCHEMA_VERSION
        assert doc["cases"][0]["name"] == "t"
        assert "benzene" in raw   # text, greppable

        ws2 = WindowState()
        ws2.load_from_dict(load_colx(path))

    assert list(ws2.species.keys()) == ["benzene", "toluene", "xylene"]
    assert ws2.streams["Feed"].stream_type == StreamType.FEED
    assert ws2.streams["Feed"].flow == 100.0
    assert ws2.streams["Feed"].composition["benzene"] == 0.4
    assert ws2.thermodynamics_config.get_component_params("benzene").antoine_a == 6.9
    assert ws2.thermodynamics_config.binary.nrtl_aij[("benzene", "toluene")] == 0.123
    assert ws2.condenser_config.condenser_type == CondenserType.PARTIAL
    assert ws2.condenser_config.reflux_ratio == 3.5
    assert ws2.reboiler_config.boilup_ratio == 1.8
    assert ws2.light_key_index == 1
    assert ws2.specs[0].kind == SpecKind.LK_RECOVERY and ws2.specs[0].value == 0.98
    assert ws2.modules["Side Stripper 1"].module_type == ModuleType.SIDE_STRIPPER
    m2 = ws2.modules["Side Stripper 1"]
    assert (m2.return_stage, m2.rate, m2.boilup_ratio) == (6, 30.0, 1.5)
    assert ws2.bvm_params["extract"] is True
    rxp = ws2.bvm_params["reaction"]
    assert rxp["on"] is True and rxp["ref"] == "xylene" and rxp["keq_b"] == -150.0
    assert rxp["nu"] == {"benzene": -1.0, "toluene": -1.0, "xylene": 1.0}
    print("colx-json self-check OK")


if __name__ == "__main__":
    test_colx_json_roundtrip()
