"""Phase 6 check: Results tab renders every data type it claims, greys out the
rest, and tables/plots follow the stage-0-=-distillate convention."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])


def _rigorous_profile():
    """Solve the bundled BTX column through the real solver so the profile has
    every series key (flows, K, enthalpy, pressure)."""
    from core.solver_input import build_solver_input
    from core.column_solvers import solve_bubble_point

    antoine = np.array([
        [6.90565, 1211.033, 220.79],
        [6.95464, 1344.8,   219.48],
        [6.99052, 1453.43,  215.31],
    ])
    si = build_solver_input(
        n_stages=12, comps=["benzene", "toluene", "xylene"], antoine=antoine,
        feeds=[(6, 100.0, [0.4, 0.35, 0.25])], R=3.0, D=40.0, pressure=760.0)
    return solve_bubble_point(si)


class _WS:                       # minimal stand-in for window_state
    def __init__(self, results):
        self.results = results


def test_results_tab_renders_all_available():
    from gui.tabs.results_tab import ResultsTab, _SERIES_KEYS

    prof = _rigorous_profile()
    for key in _SERIES_KEYS.values():
        assert key in prof, f"rigorous profile should carry {key}"

    tab = ResultsTab()
    tab.set_window_state(_WS(prof))
    tab.update_results({"status": "Converged", "stages": prof["n_stages"]})

    model = tab.data_combo.model()
    for i in range(tab.data_combo.count()):
        name = tab.data_combo.itemText(i)
        if name == "McCabe-Thiele":
            continue                    # binary-only; this profile is ternary
        assert model.item(i).isEnabled(), name

    # every data type draws without error
    for i in range(tab.data_combo.count()):
        tab.data_combo.setCurrentIndex(i)   # triggers _draw_plot

    # table: stage 0 (distillate) on the top row, comps-driven headers
    n = prof["n_stages"]
    assert tab.data_table.rowCount() == n
    assert tab.data_table.item(0, 0).text() == "0"
    assert tab.data_table.item(n - 1, 0).text() == str(n - 1)
    headers = [tab.data_table.horizontalHeaderItem(c).text()
               for c in range(tab.data_table.columnCount())]
    assert "x benzene" in headers
    assert any(h.startswith("Liquid Flow") for h in headers)


def test_results_tab_disables_missing_series():
    from gui.tabs.results_tab import ResultsTab

    # BVM-style minimal profile: x/y/T only
    prof = _rigorous_profile()
    slim = {k: prof[k] for k in ("x", "y", "T", "comps", "n_stages", "feed_stage")}
    tab = ResultsTab()
    tab.set_window_state(_WS(slim))
    tab.update_results({"status": "Feasible"})

    model = tab.data_combo.model()
    states = {tab.data_combo.itemText(i): model.item(i).isEnabled()
              for i in range(tab.data_combo.count())}
    assert states["Compositions"] and states["Temperature"]
    assert states["Ternary Map"]                  # 3 components
    for name in ("Pressure", "Liquid Flow", "Vapor Flow", "K-Values", "Enthalpy"):
        assert not states[name], f"{name} should be greyed out"
    # a disabled selection falls back to something available
    assert tab._data_available(tab.data_combo.currentText(), slim)


def test_csv_rows_zero_based_with_series():
    from gui.tabs.results_tab import profile_to_csv_rows

    prof = _rigorous_profile()
    rows = profile_to_csv_rows(prof)
    assert rows[1][0] == 0 and rows[-1][0] == prof["n_stages"] - 1
    assert any(str(h).startswith("Liquid Flow") for h in rows[0])


def test_stream_summary_products_duties_closure():
    from gui.tabs.results_tab import stream_summary
    from core.solver_input import build_solver_input
    from core.column_solvers import solve_inside_out, make_energy_balance

    antoine = np.array([[6.90565, 1211.033, 220.79],
                        [6.95464, 1344.8, 219.48],
                        [6.99052, 1453.43, 215.31]])
    # side draw + energy balance so duties and a third product appear
    si = build_solver_input(
        n_stages=16, comps=["benzene", "toluene", "xylene"], antoine=antoine,
        feeds=[(8, 100.0, [0.4, 0.35, 0.25])], draws=[(5, 8.0, 0.0)],
        R=3.0, D=36.0, pressure=760.0)
    eb = make_energy_balance(np.array([136.0, 157.0, 186.0]),
                             np.array([30.8, 33.2, 36.2]),
                             np.array([353.2, 383.8, 417.6]),
                             np.array([562.0, 591.8, 630.3]))
    prof = solve_inside_out(si, flows_hook=eb, max_iter=80)

    summ = stream_summary(prof)
    names = [p["name"] for p in summ["products"]]
    assert names[0] == "Distillate" and names[1] == "Bottoms"
    assert any("Side draw" in n for n in names)          # the draw shows up
    # every product composition sums to ~1
    for p in summ["products"]:
        assert abs(float(np.sum(p["comp"])) - 1.0) < 1e-6
    # duties present (raw kJ/h) with the right signs, closure ~ 0
    assert summ["condenser_duty"] < 0.0 < summ["reboiler_duty"]
    assert summ["closure_max"] < 1e-3, summ["closure_max"]

    # render path populates the Streams sub-view without error
    from gui.tabs.results_tab import ResultsTab
    tab = ResultsTab()
    tab.set_window_state(_WS(prof))
    tab.update_results({"status": "Solved"})
    assert tab.stream_table.rowCount() == len(summ["products"])
    assert "Reboiler duty" in tab.duty_label.text()


def test_mccabe_thiele_binary_only():
    from gui.tabs.results_tab import ResultsTab
    from core.solver_input import build_solver_input
    from core.column_solvers import solve_bubble_point

    antoine = np.array([[6.90565, 1211.033, 220.79],
                        [6.95464, 1344.8, 219.48]])
    si = build_solver_input(
        n_stages=14, comps=["benzene", "toluene"], antoine=antoine,
        feeds=[(7, 100.0, [0.5, 0.5])], R=2.5, D=50.0, pressure=760.0)
    prof = solve_bubble_point(si)

    # a WindowState-like stub carrying the thermo the McCabe plot needs
    class _WSthermo:
        def __init__(self, prof):
            self.results = prof
            self.pressure = 760.0
            from gui.state.window_state import WindowState, Species
            ws = WindowState()
            for nm, abc in zip(("benzene", "toluene"), antoine):
                ws.add_species(Species(name=nm))
                p = ws.thermodynamics_config.get_component_params(nm)
                p.antoine_a, p.antoine_b, p.antoine_c = [float(v) for v in abc]
            self._ws = ws
        def get_species_names(self): return self._ws.get_species_names()
        @property
        def thermodynamics_config(self): return self._ws.thermodynamics_config
        def build_gamma_fn(self, order): return self._ws.build_gamma_fn(order)

    tab = ResultsTab()
    tab.set_window_state(_WSthermo(prof))
    tab.update_results({"status": "Solved"})
    states = {tab.data_combo.itemText(i): tab.data_combo.model().item(i).isEnabled()
              for i in range(tab.data_combo.count())}
    assert states["McCabe-Thiele"]                # enabled for a binary
    tab.set_data_type("McCabe-Thiele")
    tab._draw_plot()                               # must not raise
    assert tab.figure.axes and tab.figure.axes[0].get_title()


def test_display_units_convert_table_and_summary():
    import core.units as units
    units._demo()                                  # pure conversions self-check

    from gui.tabs.results_tab import ResultsTab, profile_to_csv_rows
    from core.units import DisplayUnits

    prof = _rigorous_profile()

    class _WSu:                                    # window_state stub with units
        def __init__(self, prof):
            self.results = prof
            self.display_units = DisplayUnits(temperature="K", duty="MW")
            self.species = {}
    tab = ResultsTab()
    tab.set_window_state(_WSu(prof))
    tab.update_results({"status": "Solved"})

    # header carries the chosen unit and the value is converted (degC -> K)
    hdr = [tab.data_table.horizontalHeaderItem(c).text()
           for c in range(tab.data_table.columnCount())]
    assert "T (K)" in hdr
    tcol = hdr.index("T (K)")
    got = float(tab.data_table.item(0, tcol).text())
    assert abs(got - (float(prof["T"][0]) + 273.15)) < 0.05

    # CSV export honours the same units
    rows = profile_to_csv_rows(prof, units=DisplayUnits(temperature="K"))
    assert rows[0][1] == "T (K)"
    assert abs(rows[1][1] - (float(prof["T"][0]) + 273.15)) < 0.05


if __name__ == "__main__":
    test_results_tab_renders_all_available()
    test_results_tab_disables_missing_series()
    test_csv_rows_zero_based_with_series()
    test_stream_summary_products_duties_closure()
    test_mccabe_thiele_binary_only()
    test_display_units_convert_table_and_summary()
    print("results-tab checks OK")
