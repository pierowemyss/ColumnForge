"""Month-1 acceptance regressions (roadmap §Month 1): B2, B4, B13.

B2  — kg/h flow unit converts through the real mixture MW (and stays greyed
      out when MWs are missing) instead of silently equating kg/h to kmol/h.
B4  — operating-point resolution solves with the SAME stage efficiency as the
      final run, so purity specs are met at E < 1.
B13 — a vapor-phase sidestream lands in SolverInput.vapor_draw, not
      liquid_draw.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])


def _configured_window():
    from gui.main_window import MainWindow
    from gui.state.window_state import Species, Stream, StreamType
    from core.dof import SpecKind

    w = MainWindow()
    ws = w.window_state
    ws.pressure = 1.01325
    ws.num_stages = 20
    ws.light_key_index = 0
    ws.heavy_key_index = 1
    abc = [(6.90565, 1211.033, 220.79), (6.95464, 1344.8, 219.48),
           (6.99052, 1453.43, 215.31)]
    for nm, (a, b, c) in zip(["benzene", "toluene", "xylene"], abc):
        ws.add_species(Species(name=nm))
        p = ws.thermodynamics_config.get_component_params(nm)
        p.antoine_a, p.antoine_b, p.antoine_c = a, b, c
    ws.streams.clear()
    ws.add_stream(Stream(id="Feed", stream_type=StreamType.FEED, stage=10,
                         flow=100.0,
                         composition={"benzene": 0.4, "toluene": 0.35,
                                      "xylene": 0.25}))
    ws.specs = []
    ws.upsert_operating_spec(SpecKind.REFLUX_RATIO, 3.0)
    ws.upsert_operating_spec(SpecKind.DISTILLATE_RATE, 40.0)
    return w, ws


# --- B2: MW-aware kg/h flow conversion --------------------------------------

def test_b2_kgh_converts_through_mixture_mw():
    from gui.panels.unit_combo_box import UnitComboBox

    box = UnitComboBox("flow")
    kgh_idx = box.unit_combo.findText("kg/h")

    # No MW provider: kg/h greyed out, selecting it snaps back to kmol/h.
    assert not box.unit_combo.model().item(kgh_idx).isEnabled()
    box.unit_combo.setCurrentIndex(kgh_idx)
    box.refresh_units()
    assert box.unit() == "kmol/h"

    # With a provider: real conversion both ways (benzene, 78.11 kg/kmol).
    box.set_mw_provider(lambda: 78.11)
    assert box.unit_combo.model().item(kgh_idx).isEnabled()
    assert abs(box.convertToSI(781.1, "kg/h") - 10.0) < 1e-9      # -> kmol/h
    assert abs(box.convertFromSI(10.0, "kg/h") - 781.1) < 1e-9    # <- kmol/h
    # and it is NOT the old identity conversion
    assert abs(box.convertToSI(100.0, "kg/h") - 100.0) > 1.0

    # Provider losing its MW data greys kg/h back out and falls back.
    box.setUnit("kg/h")
    box.set_mw_provider(lambda: None)
    assert not box.unit_combo.model().item(kgh_idx).isEnabled()
    assert box.unit() == "kmol/h"


def test_b2_stream_panel_average_mw():
    from gui.panels.stream_config_panel import StreamConfigPanel
    from gui.state.window_state import WindowState, Species

    ws = WindowState()
    ws.add_species(Species(name="benzene", mw=78.11))
    ws.add_species(Species(name="toluene", mw=92.14))
    panel = StreamConfigPanel()
    panel.set_window_state(ws)
    panel._fill_comp({"benzene": 0.5, "toluene": 0.5})
    mw = panel._avg_mw()
    assert mw is not None and abs(mw - (0.5 * 78.11 + 0.5 * 92.14)) < 1e-6

    # any present species without an MW -> no conversion offered
    ws.species["toluene"].mw = None
    assert panel._avg_mw() is None


# --- B4: efficiency reaches the operating-point resolution ------------------

def test_b4_resolution_solves_at_run_efficiency(monkeypatch):
    import core.column_solvers as cs
    from core.dof import SpecKind

    w, ws = _configured_window()
    ws.stage_efficiency = 0.7
    ws.specs = []
    ws.upsert_operating_spec(SpecKind.REFLUX_RATIO, 4.0)
    ws.upsert_operating_spec(SpecKind.DIST_PURITY, 0.55, component=0)

    seen = []
    real = cs.solve_bubble_point

    def recording(si, **knobs):
        seen.append(knobs.get("efficiency"))
        return real(si, **knobs)

    monkeypatch.setattr(cs, "solve_bubble_point", recording)
    # Pin the solver: the Simulation tab defaults to Inside-Out, so leaving this
    # implicit made the patch above watch a function resolution never calls, and
    # `seen` came back empty no matter how efficiency behaved. The efficiency
    # knob is built once for both solvers, so Bubble-Point covers the behaviour.
    si, knobs = w._gather_rigorous_inputs(method="Bubble-Point")

    assert knobs["efficiency"] == 0.7
    # the purity spec forces iterative solves during resolution — every one
    # of them must run at the real column's efficiency, not E=1
    assert seen and all(e == 0.7 for e in seen)

    # and the resolved point actually meets the purity target at E=0.7
    prof = real(si, **knobs)
    assert prof["found"]
    assert abs(prof["xD"][0] - 0.55) < 1e-3


# --- B13: sidestream phase -> SolverInput draws ------------------------------

def test_b13_vapor_sidestream_reaches_vapor_draw():
    from gui.state.window_state import Stream, StreamType

    w, ws = _configured_window()
    ws.add_stream(Stream(id="S1", stream_type=StreamType.SIDESTREAM, stage=5,
                         flow=8.0, phase="vapor"))
    ws.add_stream(Stream(id="S2", stream_type=StreamType.SIDESTREAM, stage=14,
                         flow=6.0))                     # default: liquid

    si, _ = w._gather_rigorous_inputs()
    # GUI stage s (0 = distillate) -> internal index s in the (N,) arrays
    assert si.vapor_draw[5] == 8.0
    assert si.liquid_draw[5] == 0.0
    assert si.liquid_draw[14] == 6.0
    assert si.vapor_draw[14] == 0.0
    assert si.vapor_draw.sum() == 8.0 and si.liquid_draw.sum() == 6.0


def test_b13_phase_survives_colx_roundtrip():
    from gui.state.persistence import encode_state, decode_state
    from gui.state.window_state import WindowState, Stream, StreamType

    ws = WindowState()
    ws.add_stream(Stream(id="S1", stream_type=StreamType.SIDESTREAM, stage=5,
                         flow=8.0, phase="vapor"))
    back = decode_state(encode_state(ws.to_dict()))
    assert back["streams"]["S1"].phase == "vapor"
    # old files without the field default to liquid
    enc = encode_state(ws.to_dict())
    del enc["streams"]["S1"]["phase"]
    assert decode_state(enc)["streams"]["S1"].phase == "liquid"


if __name__ == "__main__":
    test_b2_kgh_converts_through_mixture_mw()
    test_b13_vapor_sidestream_reaches_vapor_draw()
    test_b13_phase_survives_colx_roundtrip()
    print("month-1 regressions OK (run under pytest for the monkeypatch case)")
