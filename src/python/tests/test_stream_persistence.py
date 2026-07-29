"""Phase 1 regression: stream edits persist to the right stream.

Covers the three historical bugs:
  1. pending spin-box text lost when switching streams (flush_pending_edits)
  2. a save landing on the wrong stream after a switch (panel.current_stream_id
     is the single save target)
  3. a newly added stream showing the previous stream's numbers
Plus: hand-edited products set user_specified so auto_balance leaves them alone.
Needs Qt offscreen: QT_QPA_PLATFORM=offscreen pytest this file.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])


def _fresh_tab():
    from gui.tabs.specifications_tab import SpecificationsTab
    from gui.state.window_state import WindowState, Species

    ws = WindowState()
    for name in ("benzene", "toluene"):
        ws.add_species(Species(name=name))
    tab = SpecificationsTab()
    tab.set_window_state(ws)
    return tab, ws


def _flow(s):
    return s.flow if s.flow is not None else 0.0


def _select(tab, stream_id):
    for r in range(tab.stream_list.rowCount()):
        if tab.stream_list.item(r, 0).text() == stream_id:
            tab.stream_list.setCurrentCell(r, 0)
            return
    raise AssertionError(f"{stream_id} not in list")


def test_pending_edit_flushes_to_old_stream_on_switch():
    tab, ws = _fresh_tab()
    _select(tab, "Feed")
    panel = tab.stream_config
    # Simulate a typed-but-not-committed edit: setValue fires valueChanged,
    # so instead poke the line edit text like a user mid-keystroke.
    panel.flow_input.spin_box.lineEdit().setText("123.0")
    _select(tab, "Distillate")  # switch away -> flush must commit to Feed
    assert abs(_flow(ws.streams["Feed"]) - 123.0) < 1e-9, ws.streams["Feed"].flow
    # and it must NOT have leaked onto the newly selected stream
    assert abs(_flow(ws.streams["Distillate"]) - 123.0) > 1e-9


def test_save_targets_panel_stream_not_tab_stream():
    tab, ws = _fresh_tab()
    _select(tab, "Feed")
    panel = tab.stream_config
    # Divergence scenario: tab thinks another stream is current
    tab.current_stream_id = "Distillate"
    panel.flow_input.spin_box.setValue(77.0)  # fires streamChanged -> save
    assert abs(_flow(ws.streams["Feed"]) - 77.0) < 1e-9
    assert abs(_flow(ws.streams["Distillate"]) - 77.0) > 1e-9


def test_new_stream_shows_its_own_defaults():
    tab, ws = _fresh_tab()
    _select(tab, "Feed")
    tab.stream_config.flow_input.spin_box.setValue(999.0)
    tab._add_stream()
    new_id = tab.current_stream_id
    assert new_id in ws.streams
    shown = tab.stream_config.get_stream_data()
    assert abs((shown["flow"] or 0.0) - 999.0) > 1e-9, \
        "new stream shows old stream's flow"
    assert abs(_flow(ws.streams[new_id]) - 999.0) > 1e-9


def test_product_edit_sets_user_specified():
    tab, ws = _fresh_tab()
    assert not ws.streams["Distillate"].user_specified
    _select(tab, "Distillate")
    tab.stream_config.flow_input.spin_box.setValue(42.0)
    assert ws.streams["Distillate"].user_specified
    # feed edits never set it
    _select(tab, "Feed")
    tab.stream_config.flow_input.spin_box.setValue(50.0)
    assert not ws.streams["Feed"].user_specified


def test_composition_survives_stream_switch_roundtrip():
    """Plan acceptance #1: set a composition on feed A, click stream B, click
    back — A's value survives."""
    tab, ws = _fresh_tab()
    _select(tab, "Feed")
    table = tab.stream_config.comp_table
    for row in range(table.rowCount()):
        if table.item(row, 0).text() == "benzene":
            table.item(row, 1).setText("0.7")     # fires cellChanged -> save
            break
    assert abs(ws.streams["Feed"].composition.get("benzene", 0) - 0.7) < 1e-9
    _select(tab, "Distillate")
    _select(tab, "Feed")
    assert abs(ws.streams["Feed"].composition.get("benzene", 0) - 0.7) < 1e-9
    shown = tab.stream_config.get_stream_data()["composition"]
    assert abs(shown.get("benzene", 0) - 0.7) < 1e-9


def test_rename_keeps_stream_data():
    """Regression: renaming a stream in the list must keep its data (values
    used to fall back to 0 because the state still keyed the old name)."""
    tab, ws = _fresh_tab()
    _select(tab, "Feed")
    tab.stream_config.flow_input.spin_box.setValue(123.0)
    row = tab.stream_list.currentRow()
    tab.stream_list.item(row, 0).setText("Crude In")   # fires itemChanged
    assert "Crude In" in ws.streams and "Feed" not in ws.streams
    assert abs(_flow(ws.streams["Crude In"]) - 123.0) < 1e-9
    assert ws.streams["Crude In"].id == "Crude In"
    assert tab.current_stream_id == "Crude In"
    # further edits land on the renamed stream, not a ghost
    tab.stream_config.flow_input.spin_box.setValue(77.0)
    assert abs(_flow(ws.streams["Crude In"]) - 77.0) < 1e-9
    # duplicate / empty names revert instead of clobbering another stream
    item = tab.stream_list.item(row, 0)
    item.setText("Distillate")
    assert item.text() == "Crude In" and "Crude In" in ws.streams
    item.setText("   ")
    assert item.text() == "Crude In"
    # renamed stream survives a save/load roundtrip in list order
    from gui.state.persistence import encode_state, decode_state
    from gui.state.window_state import WindowState
    ws2 = WindowState()
    ws2.load_from_dict(decode_state(encode_state(ws.to_dict())))
    assert list(ws2.streams)[0] == "Crude In"
    assert abs(_flow(ws2.streams["Crude In"]) - 77.0) < 1e-9


def test_user_specified_survives_persistence():
    from gui.state.persistence import encode_state, decode_state
    from gui.state.window_state import WindowState

    tab, ws = _fresh_tab()
    ws.streams["Bottoms"].user_specified = True
    ws2 = WindowState()
    ws2.load_from_dict(decode_state(encode_state(ws.to_dict())))
    assert ws2.streams["Bottoms"].user_specified
    assert not ws2.streams["Feed"].user_specified


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
            print(f"{fn.__name__} OK")
    print("all stream-persistence checks passed")
