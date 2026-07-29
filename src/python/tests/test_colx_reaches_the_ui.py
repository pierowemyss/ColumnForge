"""A .colx that round-trips through persistence is only half the story: the
tabs have to *show* it. These are the gaps that shipped —

  * side modules (pumparound, side rectifier, ...) restored into
    window_state.modules but never listed in Specifications -> Advanced Modules,
  * `solver_mode` persisted in every file and read by nothing, so a case saved
    on Bubble-Point came back on Inside-Out,
  * the BVM panel's one-shot restore latching, so a second .colx kept the first
    file's BVM knobs.
"""
import glob
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", "docs", "examples")
WITH_MODULES = sorted(glob.glob(os.path.join(EXAMPLES, "*.colx")))


def _win():
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow
    QApplication.instance() or QApplication([])
    return MainWindow()


def _load_into(win, path):
    from gui.state.persistence import load_colx
    win.window_state.load_from_dict(load_colx(path))
    for tab in (win.init_tab, win.specs_tab, win.sim_tab, win.modules_tab):
        tab.set_window_state(win.window_state)


@pytest.mark.parametrize("path", WITH_MODULES,
                         ids=lambda p: os.path.splitext(os.path.basename(p))[0])
def test_modules_reach_the_module_list(path):
    win = _win()
    _load_into(win, path)
    listed = {win.specs_tab.module_list.item(r, 0).text()
              for r in range(win.specs_tab.module_list.rowCount())}
    assert listed == set(win.window_state.modules), path


def test_module_list_shows_the_stored_config_not_defaults():
    from gui.state.window_state import ModuleConfig, ModuleType
    win = _win()
    win.window_state.modules["Upper pumparound"] = ModuleConfig(
        module_type=ModuleType.PUMPAROUND, stage=9, return_stage=5,
        rate=250.0, duty=-1500.0)
    win.specs_tab.set_window_state(win.window_state)
    cfg = win.specs_tab.module_config.get_config()
    assert cfg["type"] == "Pumparound"
    assert cfg["stage"] == 9 and cfg["return_stage"] == 5
    assert cfg["rate"] == pytest.approx(250.0)
    assert cfg["duty"] == pytest.approx(-1500.0)


def test_solver_mode_round_trips_through_the_method_combo():
    from core.data_structures import SolverMode
    win = _win()

    win.window_state.solver_mode = SolverMode.BUBBLE_POINT
    win.sim_tab.set_window_state(win.window_state)
    assert win.sim_tab.solver_combo.currentText() == "Bubble-Point"

    win.sim_tab.solver_combo.setCurrentText("Inside-Out")
    assert win.window_state.solver_mode == SolverMode.HYSIM

    # BVM has no rigorous entry in this combo; loading a BVM case must not
    # silently rewrite its mode to whatever the combo happens to show.
    win.window_state.solver_mode = SolverMode.BVM
    win.sim_tab.set_window_state(win.window_state)
    assert win.window_state.solver_mode == SolverMode.BVM


def test_energy_balance_is_one_flag_not_two():
    """The Flow Model checkbox wrote thermodynamics_config.energy_balance while
    the DoF ledger read WindowState.energy_balance, which nothing ever set. A
    column could run the energy balance and still have its duty specs rejected."""
    win = _win()
    win.init_tab.energy_balance_check.setChecked(True)
    assert win.window_state.energy_balance is True
    assert win.window_state.build_dof_analyzer().energy_balance is True

    win.init_tab.energy_balance_check.setChecked(False)
    assert win.window_state.energy_balance is False
    assert win.window_state.build_dof_analyzer().energy_balance is False


def test_energy_balance_survives_a_save_load_round_trip(tmp_path):
    from gui.state.persistence import save_colx, load_colx
    win = _win()
    win.init_tab.energy_balance_check.setChecked(True)
    path = str(tmp_path / "eb.colx")
    save_colx(path, win.window_state.to_dict())

    win2 = _win()
    win2.window_state.load_from_dict(load_colx(path))
    assert win2.window_state.energy_balance is True
    assert win2.window_state.build_dof_analyzer().energy_balance is True


def test_bvm_panel_picks_up_a_second_file():
    """The panel restores window_state.bvm_params once. Loading another case has
    to re-arm it, or the second file runs with the first file's parameters."""
    win = _win()
    first = os.path.join(EXAMPLES, "reactive_mtbe.colx")
    second = os.path.join(EXAMPLES, "reactive_tame.colx")
    _load_into(win, first)
    win.modules_tab.module_combo.setCurrentText("BVM")
    panel = win.modules_tab.bvm_widget
    assert panel is not None
    assert panel.get_params()["lk"] == "n-butane"

    _load_into(win, second)
    assert panel.get_params()["lk"] == "n-pentane", panel.get_params()["lk"]
