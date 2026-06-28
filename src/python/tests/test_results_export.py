"""Phase 2 check: CSV export rows match the solved profile. Qt-free."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# the BVM solver lives under src/ (sibling of src/python), so add src/ too
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def test_profile_to_csv_rows():
    # import here so the path insert above is in effect
    from gui.tabs.results_tab import profile_to_csv_rows
    from side_features.bvm.solver import bound_val_method, build_column_profile
    import numpy as np

    abc = [(6.90565, 1211.033, 220.79), (6.95464, 1344.8, 219.48),
           (6.99052, 1453.43, 215.31)]
    prof = build_column_profile(bound_val_method(
        zF=np.array([0.4, 0.35, 0.25]), F=100.0, r=12.0, q=1.0,
        antoine=np.array(abc), comps=["benzene", "toluene", "xylene"],
        lk=0, P=760.0, spec_mode="recovery", FR_LK=0.98, NK_spec=0.001))
    assert prof["found"]

    rows = profile_to_csv_rows(prof)
    assert rows[0] == ["Stage", "T", "benzene", "toluene", "xylene"]
    assert len(rows) == prof["n_stages"] + 1            # header + one row per stage
    assert len(rows[0]) == len(prof["comps"]) + 2       # Stage, T, + one per comp
    print(f"results-export self-check OK: {len(rows) - 1} rows, {len(rows[0])} cols")


if __name__ == "__main__":
    test_profile_to_csv_rows()
