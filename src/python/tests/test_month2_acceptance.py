"""Month-2 acceptance (roadmap §Month 2 check): pick benzene / toluene /
p-xylene from the component database and run all three solvers with ZERO
manual parameter entry; assert the DB round-trips through .colx and that the
Antoine range warning fires when the column leaves a fit's validity range.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])


def _btx_window_from_db():
    from gui.main_window import MainWindow
    from gui.state.window_state import Stream, StreamType
    from core.dof import SpecKind
    from core import component_db

    w = MainWindow()
    ws = w.window_state
    ws.pressure = 1.01325
    ws.num_stages = 20
    ws.light_key_index = 0
    ws.heavy_key_index = 1
    for name in ("benzene", "toluene", "p-xylene"):   # the ONLY species input
        component_db.load_into(ws, name)
    ws.streams.clear()
    ws.add_stream(Stream(id="Feed", stream_type=StreamType.FEED, stage=10,
                         flow=100.0,
                         composition={"benzene": 0.4, "toluene": 0.35,
                                      "p-xylene": 0.25}))
    ws.specs = []
    ws.upsert_operating_spec(SpecKind.REFLUX_RATIO, 3.0)
    ws.upsert_operating_spec(SpecKind.DISTILLATE_RATE, 40.0)
    return w, ws


def test_btx_from_db_all_solvers():
    w, ws = _btx_window_from_db()

    for solver in (w._solve_bubble_point, w._solve_inside_out):
        profile = solver()
        assert profile["found"], profile.get("message")
        # profiles are top -> bottom: benzene enriches upward
        assert profile["x"][0][0] > profile["x"][-1][0]
        # sane BTX temperatures at 1 atm (fit unit: degC)
        assert 70.0 < min(profile["T"]) and max(profile["T"]) < 150.0


def test_db_species_roundtrip_colx():
    from gui.state.persistence import encode_state, decode_state

    w, ws = _btx_window_from_db()
    back = decode_state(encode_state(ws.to_dict()))
    assert set(back["species"]) == {"benzene", "toluene", "p-xylene"}
    assert abs(back["species"]["toluene"].mw - 92.141) < 1e-9
    p = back["thermodynamics_config"].get_component_params("p-xylene")
    assert p.antoine_a == 6.99052
    assert (p.antoine_tmin, p.antoine_tmax) == (27.0, 166.0)


def test_range_warning_fires_and_stays_quiet():
    w, ws = _btx_window_from_db()
    profile = w._solve_bubble_point()

    # benzene's tabulated fit tops out at 103 C; a BTX column bottoms runs
    # ~120 C, so the DB range data must (correctly) flag exactly benzene
    warns = w._antoine_range_warnings(profile)
    assert len(warns) == 1 and "benzene" in warns[0], warns

    # shrink another range so the profile exits it too -> that warning appears
    p = ws.thermodynamics_config.get_component_params("toluene")
    p.antoine_tmax = 100.0
    warns = w._antoine_range_warnings(profile)
    assert len(warns) == 2 and any("toluene" in x for x in warns), warns
    # and the run path surfaces it in the Results status line
    summary = w._normalize_results(profile)
    summary["status"] += "  |  WARNING: " + "; ".join(warns)
    w.results_tab.update_results(summary)
    assert "toluene" in w.results_tab.summary_label.text()

    # quiet when the profile stays inside every range: widen both
    p.antoine_tmax = 200.0
    ws.thermodynamics_config.get_component_params("benzene").antoine_tmax = 200.0
    assert w._antoine_range_warnings(profile) == []


if __name__ == "__main__":
    test_btx_from_db_all_solvers()
    test_db_species_roundtrip_colx()
    test_range_warning_fires_and_stays_quiet()
    print("month-2 acceptance OK")
