"""Thermodynamics parameter tables — entered coefficients must not evaporate.

Three ways they used to:
  * switching the Table: combo (aij -> bij) while a cell editor was still open
    threw the half-typed value away — clicking a combo does not move focus out
    of the editor, so nothing committed it before the grid reloaded;
  * the grid displayed "%g" (6 sig figs), and the next edit anywhere in the
    table wrote the rounded number back over the stored one;
  * text float() couldn't read (a pasted unicode minus, a decimal comma) was
    dropped silently and the cell blanked itself on the next reload.

Headless: QT_QPA_PLATFORM=offscreen python -m pytest src/python/tests/test_thermo_table_entry.py
"""
import os
import warnings

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (                       # noqa: E402
    QApplication, QLineEdit, QTableWidgetItem)
from PySide6.QtTest import QTest                      # noqa: E402

from gui import table_edit                            # noqa: E402
from gui.tabs.initialization_tab import InitializationTab   # noqa: E402
from gui.state.window_state import WindowState, Species     # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    table_edit.install(a)
    return a


@pytest.fixture
def tab(app):
    ws = WindowState()
    ws.add_species(Species(name="Ethanol"))
    ws.add_species(Species(name="Water"))
    t = InitializationTab()
    t.set_window_state(ws)
    t.show()
    # Offscreen windows never activate on their own, and until one is active
    # focusWidget() stays None — which is exactly what commit_open_editor reads.
    with warnings.catch_warnings():          # setActiveWindow is deprecated, and
        warnings.simplefilter("ignore")      # is still the only thing that works
        app.setActiveWindow(t)
    t.param_type_combo.setCurrentText("Activity")
    t.param_model_combo.setCurrentText("NRTL")
    return t, ws


def _type_into(tab, row, col, text):
    """Open the cell editor and type, without committing (no Enter)."""
    tbl = tab.interaction_table
    tbl.setCurrentCell(row, col)
    tbl.editItem(tbl.item(row, col))
    editor = tbl.viewport().findChild(QLineEdit)
    editor.setFocus()
    QTest.keyClicks(editor, text)
    return editor


def test_open_edit_survives_a_table_switch(tab):
    t, ws = tab
    editor = _type_into(t, 0, 1, "1.5")
    table_edit.commit_open_editor(t.table_selection_combo)   # what the click does
    t.table_selection_combo.setCurrentText("bij")

    b = ws.thermodynamics_config.binary
    assert b.nrtl_aij == {("Ethanol", "Water"): 1.5}
    assert b.nrtl_bij == {}, "the aij value must not leak into bij"
    assert editor.isVisible() is False


def test_values_round_trip_between_tables(tab):
    t, ws = tab
    tbl = t.interaction_table
    tbl.setItem(0, 1, QTableWidgetItem("1234.56789"))
    t.table_selection_combo.setCurrentText("bij")
    tbl.setItem(0, 1, QTableWidgetItem("-42.5"))
    t.table_selection_combo.setCurrentText("aij")

    b = ws.thermodynamics_config.binary
    assert tbl.item(0, 1).text() == "1234.56789"     # no %g rounding on display
    tbl.setItem(1, 0, QTableWidgetItem("2"))         # edit elsewhere in the grid
    assert b.nrtl_aij[("Ethanol", "Water")] == 1234.56789
    assert b.nrtl_bij[("Ethanol", "Water")] == -42.5


def test_pasted_text_is_parsed_or_refused(tab):
    t, ws = tab
    tbl = t.interaction_table
    b = ws.thermodynamics_config.binary

    tbl.setItem(0, 1, QTableWidgetItem("−1,5"))      # unicode minus, decimal comma
    assert b.nrtl_aij[("Ethanol", "Water")] == -1.5

    tbl.setItem(1, 0, QTableWidgetItem("not a number"))
    assert ("Water", "Ethanol") not in b.nrtl_aij
    assert tbl.item(1, 0).text() == "", "refused text is reverted, not left to rot"


def test_clearing_a_cell_deletes_the_parameter(tab):
    t, ws = tab
    tbl = t.interaction_table
    b = ws.thermodynamics_config.binary
    tbl.setItem(0, 1, QTableWidgetItem("3.3"))
    assert b.nrtl_aij[("Ethanol", "Water")] == 3.3
    tbl.item(0, 1).setText("")
    assert ("Ethanol", "Water") not in b.nrtl_aij


def test_copy_and_paste_cover_the_whole_selection(tab):
    t, ws = tab
    tbl = t.interaction_table
    QApplication.clipboard().setText("0\t1.25\n-3.5\t0")
    tbl.setCurrentCell(0, 0)
    assert table_edit.paste_selection(tbl)
    assert ws.thermodynamics_config.binary.nrtl_aij[("Ethanol", "Water")] == 1.25
    assert ws.thermodynamics_config.binary.nrtl_aij[("Water", "Ethanol")] == -3.5

    tbl.selectAll()
    assert table_edit.copy_selection(tbl)
    assert QApplication.clipboard().text() == "0\t1.25\n-3.5\t0"
