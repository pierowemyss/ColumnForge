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


def test_a_flowsheet_round_trips_with_its_connections():
    """v3: several columns and the streams between them survive the file."""
    from gui.state.window_state import WindowState, Species, StreamType
    from gui.state.persistence import save_colx, load_colx
    from core.flowsheet import Connection

    ws = WindowState()
    for name in ("benzene", "toluene", "xylene"):
        ws.add_species(Species(name=name))
    ws.num_stages = 16
    ws.node_pos = (-170.0, 0.0)
    ws.active_column.method = "Bubble-Point"
    ws.add_column("C2")
    ws.num_stages = 22
    ws.node_pos = (170.0, 0.0)
    ws.streams["Feed"].flow = 55.0
    ws.set_active_column("C1")
    ws.connections = [
        Connection("C1.B->C2@8", "C1", "B", "C2", 8),
        Connection("C2.D->C1@8", "C2", "D", "C1", 8, split_fraction=0.9, q=0.5),
    ]
    ws.default_method = "Bubble-Point"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fs.colx")
        save_colx(path, ws.to_dict())
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        assert set(doc["cases"][0]["state"]["columns"]) == {"C1", "C2"}
        ws2 = WindowState()
        ws2.load_from_dict(load_colx(path))

    assert set(ws2.columns) == {"C1", "C2"}
    assert ws2.active_column_id == "C1" and ws2.num_stages == 16
    assert ws2.columns["C2"].num_stages == 22
    assert ws2.columns["C2"].streams["Feed"].flow == 55.0
    assert ws2.columns["C1"].node_pos == (-170.0, 0.0)
    assert ws2.columns["C1"].method == "Bubble-Point"
    assert ws2.default_method == "Bubble-Point"
    # species are flowsheet-global, stored once, shared by both columns
    assert list(ws2.species) == ["benzene", "toluene", "xylene"]
    assert "species" not in doc["cases"][0]["state"]["columns"]["C1"]

    recycle = {c.id: c for c in ws2.connections}["C2.D->C1@8"]
    assert (recycle.src, recycle.port, recycle.dst, recycle.stage) == ("C2", "D", "C1", 8)
    assert recycle.split_fraction == 0.9 and recycle.q == 0.5
    assert {c.id for c in ws2.connections} == {"C1.B->C2@8", "C2.D->C1@8"}
    assert ws2.streams["Feed"].stream_type == StreamType.FEED


def test_a_v2_file_migrates_to_a_one_column_flowsheet():
    """Every file saved by an earlier build is v2. Losing a field here loses a
    user's column, so this walks the whole per-column set."""
    from gui.state.window_state import WindowState, CondenserType, ModuleType
    from gui.state.persistence import load_colx
    from core.dof import SpecKind

    v2 = {
        "schema_version": 2, "app": "ColumnForge",
        "cases": [{"name": "old", "state": {
            "num_stages": 24, "pressure": 760.0, "pressure_drop": 0.01,
            "stage_efficiency": 0.8,
            "species": {"benzene": {"name": "benzene", "mw": 78.11}},
            "streams": {"Feed": {"id": "Feed", "stream_type": "Feed", "stage": 12,
                                 "flow": 100.0, "composition": {"benzene": 1.0}},
                        "S1": {"id": "S1", "stream_type": "Sidestream", "stage": 5,
                               "flow": 9.0, "phase": "vapor"}},
            "condenser_config": {"condenser_type": "Partial", "reflux_ratio": 2.5},
            "reboiler_config": {"reboiler_type": "Kettle", "boilup_ratio": 1.4},
            "thermodynamics_config": {"vle_model": "PLXANT", "energy_balance": True},
            "modules": {"Pump 1": {"module_type": "Pumparound", "stage": 8,
                                   "return_stage": 4, "rate": 20.0, "duty": 5.0}},
            "specs": [{"kind": SpecKind.LK_RECOVERY.value, "value": 0.97,
                       "unit_ref": "column", "component": -1}],
            "light_key_index": 1, "heavy_key_index": 2,
            "bvm_params": {"r_spin": 4.0}, "solver_mode": "bubble_point",
        }}],
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.colx")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(v2, f)
        ws = WindowState()
        ws.load_from_dict(load_colx(path))

    # one column, nothing connected, and it is the active one
    assert list(ws.columns) == ["C1"] and ws.active_column_id == "C1"
    assert ws.connections == []
    # every per-column field arrived, reachable under its old flat name
    assert ws.num_stages == 24 and ws.pressure == 760.0
    assert ws.pressure_drop == 0.01 and ws.stage_efficiency == 0.8
    assert ws.light_key_index == 1 and ws.heavy_key_index == 2
    assert ws.streams["Feed"].flow == 100.0
    assert ws.streams["S1"].phase == "vapor"
    assert ws.condenser_config.condenser_type == CondenserType.PARTIAL
    assert ws.condenser_config.reflux_ratio == 2.5
    assert ws.reboiler_config.boilup_ratio == 1.4
    assert ws.modules["Pump 1"].module_type == ModuleType.PUMPAROUND
    assert ws.modules["Pump 1"].return_stage == 4
    assert ws.specs[0].kind == SpecKind.LK_RECOVERY
    assert ws.active_column.method is None and ws.node_pos is None
    # and the globals stayed global
    assert list(ws.species) == ["benzene"]
    assert ws.thermodynamics_config.vle_model == "PLXANT"
    assert ws.energy_balance is True
    assert ws.bvm_params["r_spin"] == 4.0


def test_an_unknown_schema_version_is_still_refused():
    """The migration chain must not turn into 'load anything and hope'."""
    import pytest
    from gui.state.persistence import load_colx

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "future.colx")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 99, "app": "ColumnForge",
                       "cases": [{"name": "x", "state": {}}]}, f)
        with pytest.raises(ValueError, match="schema_version 99"):
            load_colx(path)


if __name__ == "__main__":
    test_colx_json_roundtrip()
