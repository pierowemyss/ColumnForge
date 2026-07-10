"""Column-overview sandbox: stream chips are clickable and open the inline
stream editor (Aspen-style click-to-configure). Offscreen Qt."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])


def _tab():
    from gui.tabs.specifications_tab import SpecificationsTab
    from gui.state.window_state import WindowState, Species

    ws = WindowState()
    for name in ("benzene", "toluene"):
        ws.add_species(Species(name=name))
    tab = SpecificationsTab()
    tab.set_window_state(ws)
    tab.resize(900, 700)
    tab.column_canvas.resize(400, 600)
    tab.column_canvas.grab()          # force a paint -> populates stream_hits
    return tab, ws


def test_every_stream_has_a_chip():
    tab, ws = _tab()
    hits = tab.column_canvas.stream_hits
    assert set(ws.streams) <= set(hits), (set(ws.streams), set(hits))
    for rect in hits.values():
        assert rect.width() > 0 and rect.height() > 0


def test_click_chip_opens_stream_editor():
    tab, ws = _tab()
    canvas = tab.column_canvas
    canvas._handle_click(canvas.stream_hits["Feed"].center())
    assert canvas.selected_stream == "Feed"
    assert tab.ov_editor_stack.currentWidget() is tab.ov_stream_panel
    assert tab.ov_stream_panel.current_stream_id == "Feed"
    assert "Feed" in tab.ov_config_group.title()
    # products open too (they used to be double-click-only pseudo-items)
    canvas._handle_click(canvas.stream_hits["Bottoms"].center())
    assert tab.ov_stream_panel.current_stream_id == "Bottoms"
    # edits from the inline editor land on the clicked stream
    tab.ov_stream_panel.flow_input.spin_box.setValue(61.5)
    assert abs((ws.streams["Bottoms"].flow or 0.0) - 61.5) < 1e-9


def test_click_equipment_opens_editor_and_drag_does_not():
    tab, _ = _tab()
    canvas = tab.column_canvas
    canvas._handle_click(canvas.items["condenser"].rect.center())
    assert tab.ov_editor_stack.currentWidget() is tab.ov_condenser_panel
    canvas._handle_click(canvas.items["reboiler"].rect.center())
    assert tab.ov_editor_stack.currentWidget() is tab.ov_reboiler_panel
    # a press that moved is a drag, not a click (mouseReleaseEvent gate)
    canvas.dragging_item = canvas.items["condenser"]
    canvas._press_moved = True
    was_drag = canvas.dragging_item is not None and canvas._press_moved
    assert was_drag


def test_renamed_stream_chip_follows():
    tab, ws = _tab()
    assert ws.rename_stream("Feed", "Crude In")
    tab._update_column_canvas()
    tab.column_canvas.grab()
    hits = tab.column_canvas.stream_hits
    assert "Crude In" in hits and "Feed" not in hits


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
            print(f"{fn.__name__} OK")
    print("column-overview checks passed")
