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


def test_module_is_anchored_to_its_stage_and_opens_its_editor():
    """A module is drawn beside the tray it hangs off (not parked at a fixed
    spot) and clicking it opens the same editor the Advanced Modules subtab uses."""
    from gui.state.window_state import ModuleConfig, ModuleType
    tab, ws = _tab()
    ws.add_module("Side Stripper 1",
                  ModuleConfig(module_type=ModuleType.SIDE_STRIPPER, stage=14,
                               return_stage=13, rate=25.0, boilup_ratio=1.5,
                               num_stages=4))
    ws.add_module("Interreboiler 1",
                  ModuleConfig(module_type=ModuleType.INTERREBOILER, stage=4,
                               duty=250.0))
    tab._update_column_canvas()
    canvas = tab.column_canvas
    canvas.grab()

    col = canvas.items["column"].rect
    low = canvas.items["module_Side Stripper 1"].rect
    high = canvas.items["module_Interreboiler 1"].rect
    assert low.center().y() > high.center().y()          # stage 14 below stage 4
    assert high.right() < col.left()                     # left corridor, clear of it
    assert abs(low.center().y() - canvas._stage_to_y(col, 14)) < 1.0

    canvas._handle_click(low.center())
    assert tab.ov_editor_stack.currentWidget() is tab.ov_module_panel
    assert tab.current_module_id == "Side Stripper 1"
    assert "Side Stripper 1" in tab.ov_config_group.title()
    # the inline editor writes back to the clicked module
    tab.ov_module_panel.rate_spin.setValue(31.0)
    assert ws.modules["Side Stripper 1"].rate == 31.0
    assert ws.modules["Side Stripper 1"].boilup_ratio == 1.5   # nothing else lost

    # the side product is a clickable chip (like distillate/bottoms) that opens
    # the module editor — the module is what defines the product
    prod = "Side Stripper 1 product"
    assert prod in canvas.stream_hits
    tab._on_element_clicked("condenser")                 # move focus away first
    canvas._handle_click(canvas.stream_hits[prod].center())
    assert tab.ov_editor_stack.currentWidget() is tab.ov_module_panel
    assert tab.current_module_id == "Side Stripper 1"

    # deleting a module clears its canvas item
    ws.remove_module("Interreboiler 1")
    tab._update_column_canvas()
    assert "module_Interreboiler 1" not in canvas.items


def test_module_form_only_shows_fields_its_type_uses():
    """Pumparounds have no reflux ratio and no stage count — those rows must be
    gone, and get_config must not report values for them."""
    from gui.panels.module_config_panel import ModuleConfigPanel
    panel = ModuleConfigPanel()
    panel.show()
    panel.set_config({"type": "Pumparound", "stage": 12, "return_stage": 6,
                      "rate": 50.0, "duty": 200.0})
    assert not panel.rows["ratio"][1].isVisible()
    assert not panel.rows["num_stages"][1].isVisible()
    assert panel.rows["return_stage"][1].isVisible()
    cfg = panel.get_config()
    assert cfg["boilup_ratio"] is None and cfg["reflux_ratio"] is None
    assert cfg["num_stages"] is None
    assert (cfg["rate"], cfg["duty"], cfg["return_stage"]) == (50.0, 200.0, 6)

    panel.set_config({"type": "Side Rectifier", "stage": 6, "return_stage": 7,
                      "rate": 25.0, "reflux_ratio": 2.0, "num_stages": 4})
    assert panel.rows["ratio"][1].isVisible()
    assert not panel.rows["duty"][1].isVisible()
    cfg = panel.get_config()
    assert cfg["reflux_ratio"] == 2.0 and cfg["boilup_ratio"] is None
    assert cfg["duty"] is None
    panel.hide()


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
