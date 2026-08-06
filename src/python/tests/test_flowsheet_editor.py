"""Beta: the multi-column flowsheet editor (Preferences → Enable beta features).

The default experience is the single-column diagram, guarded by
test_column_overview.py. Everything here is behind the beta flag, which the
`beta` fixture turns on for the duration of a test and restores afterwards — so
running this file cannot leave a developer's own preference flipped.

Every behavioural claim the single-column canvas made is re-made here against
the scene: `scene.items()` and `sceneBoundingRect()` instead of hand-computed
rects, and real Qt events instead of calling the click handler directly. Plus
the things only a flowsheet can get wrong — recycle marking, illegal
connections, stale connections, node positions.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def beta():
    """Turn beta features on for this test only, then put the user's setting
    back — these tests must not change what the developer sees in the app."""
    from gui.app_settings import beta_enabled, set_beta_enabled
    was = beta_enabled()
    set_beta_enabled(True)
    yield
    set_beta_enabled(was)


def _tab():
    from gui.tabs.specifications_tab import SpecificationsTab
    from gui.state.window_state import WindowState, Species

    ws = WindowState()
    for name in ("benzene", "toluene"):
        ws.add_species(Species(name=name))
    tab = SpecificationsTab()
    tab.set_window_state(ws)
    tab.resize(900, 700)
    return tab, ws


def _chips(tab):
    """{stream_id: StreamChip} currently in the scene."""
    from gui.panels.flowsheet_items import StreamChip
    return {i.stream_id: i for i in tab.flowsheet_scene.items()
            if isinstance(i, StreamChip)}


def _node(tab, unit_id="C1"):
    return tab.flowsheet_scene.nodes[unit_id]


def test_every_stream_has_a_chip():
    tab, ws = _tab()
    chips = _chips(tab)
    assert set(ws.streams) <= set(chips), (set(ws.streams), set(chips))
    for chip in chips.values():
        r = chip.sceneBoundingRect()
        assert r.width() > 0 and r.height() > 0


def test_click_chip_opens_stream_editor():
    tab, ws = _tab()
    scene = tab.flowsheet_scene
    chips = _chips(tab)
    scene.activate(chips["Feed"])
    assert scene.selected_stream == "Feed"
    assert tab.ov_editor_stack.currentWidget() is tab.ov_stream_panel
    assert tab.ov_stream_panel.current_stream_id == "Feed"
    assert "Feed" in tab.ov_config_group.title()
    # products open too (they used to be double-click-only pseudo-items)
    scene.activate(_chips(tab)["Bottoms"])
    assert tab.ov_stream_panel.current_stream_id == "Bottoms"
    # edits from the inline editor land on the clicked stream
    tab.ov_stream_panel.flow_input.spin_box.setValue(61.5)
    assert abs((ws.streams["Bottoms"].flow or 0.0) - 61.5) < 1e-9


def test_a_real_mouse_click_on_a_chip_reaches_the_editor():
    """`activate()` is the scene's own entry point, so every test above would
    still pass with the mouse handlers unwired. This one goes through Qt."""
    tab, _ = _tab()
    tab.show()
    view = tab.flowsheet_view
    chip = _chips(tab)["Feed"]
    view.fit()
    pos = view.mapFromScene(chip.sceneBoundingRect().center())
    QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
    assert tab.ov_editor_stack.currentWidget() is tab.ov_stream_panel
    assert tab.ov_stream_panel.current_stream_id == "Feed"
    tab.hide()


def test_click_equipment_opens_editor_and_drag_moves_the_node():
    tab, ws = _tab()
    scene = tab.flowsheet_scene
    scene.click_equipment("C1", "condenser")
    assert tab.ov_editor_stack.currentWidget() is tab.ov_condenser_panel
    scene.click_equipment("C1", "reboiler")
    assert tab.ov_editor_stack.currentWidget() is tab.ov_reboiler_panel

    # A drag moves the node and is not a click: QGraphicsView owns that
    # distinction now, so this exercises the real thing rather than a flag.
    tab.show()
    view = tab.flowsheet_view
    view.fit()
    node = _node(tab)
    tab.ov_editor_stack.setCurrentWidget(tab.ov_placeholder)
    p0 = node.pos()
    start = view.mapFromScene(node.shell_rect_scene().center())
    QTest.mousePress(view.viewport(), Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(view.viewport(), start + QPoint(60, 40))
    QTest.mouseMove(view.viewport(), start + QPoint(90, 60))
    QTest.mouseRelease(view.viewport(), Qt.LeftButton, Qt.NoModifier,
                       start + QPoint(90, 60))
    assert node.pos() != p0, "dragging the node body did not move it"
    assert tab.ov_editor_stack.currentWidget() is tab.ov_placeholder
    # ...and the move was written back, so it survives a save
    assert ws.columns["C1"].node_pos == (node.pos().x(), node.pos().y())
    tab.hide()


def test_module_is_anchored_to_its_stage_and_opens_its_editor():
    """A module is drawn beside the tray it hangs off (not parked at a fixed
    spot) and clicking it opens the same editor the Advanced Modules subtab uses."""
    from gui.panels.flowsheet_items import ModuleBadge
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

    node = _node(tab)
    low = node.module_badge("Side Stripper 1")
    high = node.module_badge("Interreboiler 1")
    lo_r, hi_r = low.sceneBoundingRect(), high.sceneBoundingRect()
    assert lo_r.center().y() > hi_r.center().y()         # stage 14 below stage 4
    assert hi_r.right() < node.shell_rect_scene().left()  # left corridor, clear
    assert abs(lo_r.center().y() - node.stage_y_scene(14)) < 1.0

    tab.flowsheet_scene.activate(low)
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
    chips = _chips(tab)
    assert prod in chips
    tab._on_element_clicked("C1", "condenser")           # move focus away first
    tab.flowsheet_scene.activate(chips[prod])
    assert tab.ov_editor_stack.currentWidget() is tab.ov_module_panel
    assert tab.current_module_id == "Side Stripper 1"

    # deleting a module clears its badge — from the node AND from the scene
    ws.remove_module("Interreboiler 1")
    tab._update_column_canvas()
    node = _node(tab)
    assert node.module_badge("Interreboiler 1") is None
    assert not [i for i in tab.flowsheet_scene.items()
                if isinstance(i, ModuleBadge) and i.module_id == "Interreboiler 1"]


def test_renamed_stream_chip_follows():
    tab, ws = _tab()
    assert ws.rename_stream("Feed", "Crude In")
    tab._update_column_canvas()
    chips = _chips(tab)
    assert "Crude In" in chips and "Feed" not in chips


def test_a_connection_survives_renaming_the_draw_it_comes_from():
    """A side draw's stream id doubles as its flowsheet port key, so the rename
    has to carry the connection with it. This is the case that made the old
    label-keyed chip map a correctness bug rather than a shortcut."""
    from gui.state.window_state import Stream, StreamType
    from core.flowsheet import Connection
    tab, ws = _tab()
    ws.add_stream(Stream(id="Side1", stream_type=StreamType.SIDESTREAM,
                         stage=5, flow=10.0))
    ws.add_column("C2")
    ws.set_active_column("C1")
    ws.connections.append(Connection("k", "C1", "Side1", "C2", 8))
    tab._update_column_canvas()
    assert "k" in tab.flowsheet_scene.edges

    assert ws.rename_stream("Side1", "Kerosene")
    tab._update_column_canvas()
    assert ws.connections[0].port == "Kerosene"
    assert "k" in tab.flowsheet_scene.edges, "the edge died with the rename"
    assert "Kerosene" in _chips(tab)


def test_a_second_column_gets_its_own_node_and_becomes_active_on_click():
    tab, ws = _tab()
    ws.add_column("C2")
    tab._update_column_canvas()
    scene = tab.flowsheet_scene
    assert set(scene.nodes) == {"C1", "C2"}
    # the two nodes do not sit on top of each other
    assert (scene.nodes["C1"].shell_rect_scene().center()
            != scene.nodes["C2"].shell_rect_scene().center())

    scene.activate(scene.nodes["C1"])
    assert ws.active_column_id == "C1"
    assert scene.nodes["C1"].active and not scene.nodes["C2"].active
    # and the flat names follow the active column
    ws.num_stages = 33
    scene.activate(scene.nodes["C2"])
    assert ws.active_column_id == "C2" and ws.num_stages == 20
    assert ws.columns["C1"].num_stages == 33


def test_a_recycle_edge_is_drawn_differently_from_a_forward_one():
    """`is_recycle` comes from the same tear_set the solver uses, so the picture
    and the streams the solver guesses cannot disagree."""
    from core.flowsheet import Connection
    from gui.theme.palette import canvas as palette
    tab, ws = _tab()
    ws.add_column("C2")
    ws.set_active_column("C1")
    ws.connections = [Connection("fwd", "C1", "B", "C2", 8),
                      Connection("back", "C2", "D", "C1", 8, split_fraction=0.9)]
    tab._update_column_canvas()
    edges = tab.flowsheet_scene.edges
    assert not edges["fwd"].recycle
    assert edges["back"].recycle
    assert edges["back"].color() == palette.RECYCLE
    assert edges["fwd"].color() == palette.INTERNAL


def test_an_illegal_connection_is_refused_and_adds_no_edge():
    tab, ws = _tab()
    ws.add_column("C2")
    tab._update_column_canvas()
    scene = tab.flowsheet_scene
    before = len(scene.edges)

    node = scene.nodes["C1"]                    # a column cannot feed itself
    port = node.ports["B"]
    conn, why = scene._candidate(port, node, port.scenePos())
    assert why is not None and "cannot feed itself" in why
    assert len(scene.edges) == before


def test_a_stale_connection_is_marked_when_the_column_shrinks():
    """The silent case: the stage was legal when drawn, and the user later cut
    num_stages on a different page entirely."""
    from core.flowsheet import Connection
    tab, ws = _tab()
    ws.add_column("C2")
    ws.set_active_column("C1")
    ws.connections = [Connection("c", "C1", "B", "C2", 12)]
    tab._update_column_canvas()
    assert tab.flowsheet_scene.edges["c"].invalid is None

    ws.columns["C2"].num_stages = 5             # stage 12 no longer exists
    tab._update_column_canvas()
    edge = tab.flowsheet_scene.edges["c"]
    assert edge.invalid and "interior tray" in edge.invalid


def _render(tab):
    """Actually paint the scene into an image, so every item's paint() runs.

    Nothing else here calls paint(): items exist as soon as `rebuild()` runs, so
    hit-testing and geometry pass on a scene that has never been drawn. A
    `QPainter.drawPolygon(a, b, c)` in the arrowhead — the varargs form C++
    allows and PySide6 does not bind — shipped exactly that way and only turned
    up when a human launched the app. This is the check that would have caught
    it, and it costs a millisecond.
    """
    from PySide6.QtGui import QImage, QPainter
    scene = tab.flowsheet_scene
    rect = scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
    img = QImage(600, 600, QImage.Format_ARGB32)
    img.fill(0)
    painter = QPainter(img)
    try:
        scene.render(painter, target=img.rect(), source=rect)
    finally:
        painter.end()          # never leave an active painter on the image
    return img


def test_the_scene_paints_without_raising():
    """Every glyph on a column: shell, condenser, reboiler, chips, arrows."""
    tab, _ = _tab()
    img = _render(tab)
    assert not img.isNull()


def test_a_flowsheet_with_modules_and_a_recycle_paints():
    """The paths a bare column never reaches: module badges and their draw /
    return arrows, side-product chips, forward edges and recycle edges."""
    from core.flowsheet import Connection
    from gui.state.window_state import (ModuleConfig, ModuleType, Stream,
                                        StreamType)
    tab, ws = _tab()
    ws.add_module("Side Stripper 1",
                  ModuleConfig(module_type=ModuleType.SIDE_STRIPPER, stage=14,
                               return_stage=13, rate=25.0, boilup_ratio=1.5,
                               num_stages=4))
    ws.add_module("Interreboiler 1",
                  ModuleConfig(module_type=ModuleType.INTERREBOILER, stage=4,
                               duty=250.0))
    ws.add_stream(Stream(id="Side1", stream_type=StreamType.SIDESTREAM,
                         stage=8, flow=5.0))
    ws.add_column("C2")
    ws.set_active_column("C1")
    ws.connections = [Connection("fwd", "C1", "B", "C2", 8),
                      Connection("back", "C2", "D", "C1", 8, split_fraction=0.9)]
    tab._update_column_canvas()
    assert not _render(tab).isNull()


@pytest.mark.parametrize("condenser,reboiler", [
    ("Partial", "Kettle"), ("None", "Kettle"), ("Total", "None"),
])
def test_terminal_variants_paint(condenser, reboiler):
    """A column with no condenser has no reflux loop to draw, and one with no
    reboiler no boilup — both are early returns in paint()."""
    from gui.state.window_state import CondenserType, ReboilerType
    tab, ws = _tab()
    ws.condenser_config.condenser_type = CondenserType(condenser)
    ws.reboiler_config.reboiler_type = ReboilerType(reboiler)
    tab._update_column_canvas()
    assert not _render(tab).isNull()


@pytest.mark.parametrize("zoom", [0.3, 1.0, 2.5])
def test_every_level_of_detail_paints(zoom):
    """paint() takes three different branches by zoom (silhouette / normal /
    tray ticks and stage numbers). Each one has to survive being drawn."""
    from PySide6.QtGui import QImage, QPainter
    tab, _ = _tab()
    scene = tab.flowsheet_scene
    img = QImage(400, 400, QImage.Format_ARGB32)
    img.fill(0)
    painter = QPainter(img)
    painter.scale(zoom, zoom)
    try:
        scene.render(painter)
    finally:
        painter.end()
    assert not img.isNull()


def test_node_positions_round_trip_through_the_scene():
    tab, ws = _tab()
    ws.columns["C1"].node_pos = (120.0, -60.0)
    tab._update_column_canvas()
    node = _node(tab)
    assert (node.pos().x(), node.pos().y()) == (120.0, -60.0)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
            print(f"{fn.__name__} OK")
    print("flowsheet-editor checks passed")
