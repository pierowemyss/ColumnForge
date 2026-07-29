"""Species properties panel — the UNIFAC group table must survive being looked at.

Regression net for a wipe that made every DB component silently fall back to
ideal: _load_species_from_state sets mw_spin, whose valueChanged reached
_update_species_from_ui() while the group table was still empty, so simply
clicking a species erased its groups. self.blockSignals() did not stop it — the
spinboxes are children, not the panel.

Headless: QT_QPA_PLATFORM=offscreen python -m pytest src/python/tests/test_species_panel.py
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication          # noqa: E402

from core import component_db                       # noqa: E402
from gui.panels.species_properties_panel import SpeciesPropertiesPanel  # noqa: E402
from gui.state.window_state import WindowState      # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app):
    ws = WindowState()
    for name in ("methyl acetate", "methanol"):
        component_db.load_into(ws, name)
    p = SpeciesPropertiesPanel()
    p.set_window_state(ws)
    p.set_species_list(list(ws.species))
    return p, ws


def _groups(ws):
    return {k: dict(v.unifac_groups) for k, v in ws.species.items()}


def test_selecting_species_does_not_wipe_unifac_groups(panel):
    p, ws = panel
    before = _groups(ws)
    assert before["methyl acetate"] and before["methanol"], "DB load must fill groups"

    for name in ("methyl acetate", "methanol", "methyl acetate"):
        p.select_species(name)
        assert _groups(ws) == before, f"groups changed after selecting {name!r}"


def test_add_group_leaves_an_editable_row(panel):
    p, _ws = panel
    p.select_species("methanol")
    rows = p.unifac_table.rowCount()

    p.add_group_btn.click()

    assert p.unifac_table.rowCount() == rows + 1
    item = p.unifac_table.item(rows, 0)
    assert item is not None and item.text() == ""
    assert p.unifac_table.item(rows, 1).text() == "1"


def test_naming_an_added_group_writes_through(panel):
    p, ws = panel
    p.select_species("methanol")
    p.add_group_btn.click()

    row = p.unifac_table.rowCount() - 1
    p.unifac_table.item(row, 0).setText("CH2")       # fires cellChanged

    assert ws.species["methanol"].unifac_groups == {"CH3OH": 1, "CH2": 1}


def test_editing_one_species_leaves_the_other_alone(panel):
    p, ws = panel
    p.select_species("methanol")
    p.mw_spin.setValue(99.0)

    assert ws.species["methanol"].mw == pytest.approx(99.0)
    assert ws.species["methyl acetate"].unifac_groups == {"CH3": 1, "CH3COO": 1}
