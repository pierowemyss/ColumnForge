"""Phase 8 end-to-end smoke: the real MainWindow, configured through
window_state exactly as the GUI stores it, solves via both rigorous paths and
renders the Results tab. Offscreen Qt."""
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
    ws.pressure = 1.01325                    # bar = 760 mmHg
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


def test_end_to_end_both_solvers_and_results_tab():
    w, ws = _configured_window()

    for solver in (w._solve_bubble_point, w._solve_inside_out):
        profile = solver()
        assert profile["found"], profile.get("message")
        assert profile["n_stages"] == 20
        assert profile["feed_stage"] == 10           # 0-based from the top
        assert abs(profile["D"] - 40.0) < 1e-6
        assert profile["xD"][0] > profile["xB"][0]   # benzene up top

        ws.results = profile
        w.results_tab.update_results(w._normalize_results(profile))
        assert w.results_tab.data_table.rowCount() == 20
        # stage 0 (distillate) on the top row of the table
        assert w.results_tab.data_table.item(0, 0).text() == "0"
        # ternary + every rigorous series available (McCabe-Thiele is binary-only,
        # so it stays greyed for this 3-component column)
        model = w.results_tab.data_combo.model()
        for i in range(w.results_tab.data_combo.count()):
            name = w.results_tab.data_combo.itemText(i)
            if name == "McCabe-Thiele":
                continue
            assert model.item(i).isEnabled(), name

    # Inside-Out extras: real (approximate) duties with the right signs
    assert profile["condenser_duty"] < 0 < profile["reboiler_duty"]


def test_gather_normalizes_near_one_composition():
    w, ws = _configured_window()
    ws.streams["Feed"].composition = {"benzene": 0.4, "toluene": 0.35,
                                      "xylene": 0.251}     # sums to 1.001
    si, _ = w._gather_rigorous_inputs()
    stage_feed = si.feed[si.feed.sum(axis=1) > 0][0]
    assert abs(stage_feed.sum() - 100.0) < 1e-9            # normalized flow*z
    import numpy as np
    assert np.allclose(stage_feed / stage_feed.sum(),
                       np.array([0.4, 0.35, 0.251]) / 1.001)


def test_every_input_matters():
    """Plan acceptance #2: feed stage, feed T (via q), pressure drop and
    condenser type each visibly change the solved result."""
    import numpy as np
    from gui.state.window_state import CondenserType
    from core.dof import SpecKind

    w, ws = _configured_window()
    base = w._solve_bubble_point()

    ws.streams["Feed"].stage = 4                      # feed stage
    moved = w._solve_bubble_point()
    assert moved["feed_stage"] == 4
    assert not np.allclose(base["T"], moved["T"], atol=1e-6)
    ws.streams["Feed"].stage = 10

    ws.streams["Feed"].temperature = 273.15 + 200.0   # superheated -> q < 1
    hot = w._solve_bubble_point()
    assert not np.allclose(base["T"], hot["T"], atol=1e-6)
    ws.streams["Feed"].temperature = None

    ws.pressure_drop = 0.02                           # bar per stage
    dp = w._solve_bubble_point()
    assert dp["pressure"][-1] > dp["pressure"][0]     # bottoms runs hotter/higher P
    assert not np.allclose(base["T"], dp["T"], atol=1e-6)
    ws.pressure_drop = 0.0

    ws.condenser_config.condenser_type = CondenserType.PARTIAL
    part = w._solve_bubble_point()
    assert part["distillate_phase"] == "vapor" and base["distillate_phase"] == "liquid"

    ws.condenser_config.condenser_type = CondenserType.NONE
    ws.specs = []                                     # no condenser -> 1 free knob
    ws.upsert_operating_spec(SpecKind.DISTILLATE_RATE, 40.0)
    none_ = w._solve_bubble_point()
    assert none_["condenser"] == "none"


def test_gather_rejects_bad_composition_and_stage():
    import pytest
    w, ws = _configured_window()
    ws.streams["Feed"].composition = {"benzene": 0.4, "toluene": 0.2,
                                      "xylene": 0.1}       # sums to 0.7
    with pytest.raises(ValueError):
        w._gather_rigorous_inputs()

    w2, ws2 = _configured_window()
    ws2.streams["Feed"].stage = 25                          # > num_stages - 1
    with pytest.raises(ValueError):
        w2._gather_rigorous_inputs()


if __name__ == "__main__":
    test_end_to_end_both_solvers_and_results_tab()
    test_gather_normalizes_near_one_composition()
    print("gui smoke OK (run under pytest for the rejection cases)")
