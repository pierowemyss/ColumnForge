"""Phase 2 check: CSV export rows match the solved profile. Qt-free."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_profile_to_csv_rows():
    # import here so the path insert above is in effect
    from gui.tabs.results_tab import profile_to_csv_rows
    from core.solver_input import build_solver_input
    from core.column_solvers import solve_bubble_point
    import numpy as np

    # The BVM entry points this used to call (bound_val_method /
    # build_column_profile) no longer exist. CSV export is for the rigorous
    # solver's output anyway, so build the profile the way the other results
    # tests do.
    antoine = np.array([[6.90565, 1211.033, 220.79],
                        [6.95464, 1344.8, 219.48],
                        [6.99052, 1453.43, 215.31]])
    si = build_solver_input(
        n_stages=12, comps=["benzene", "toluene", "xylene"], antoine=antoine,
        feeds=[(6, 100.0, [0.4, 0.35, 0.25])], R=3.0, D=40.0, pressure=760.0)
    prof = solve_bubble_point(si)
    assert prof["found"]

    rows = profile_to_csv_rows(prof)
    assert rows[0][:5] == ["Stage", "T (degC)", "benzene", "toluene", "xylene"]
    assert len(rows) == prof["n_stages"] + 1            # header + one row per stage
    assert rows[1][0] == 0, "stages are 0-based from the top"
    assert rows[-1][0] == prof["n_stages"] - 1
    print(f"results-export self-check OK: {len(rows) - 1} rows, {len(rows[0])} cols")


if __name__ == "__main__":
    test_profile_to_csv_rows()
