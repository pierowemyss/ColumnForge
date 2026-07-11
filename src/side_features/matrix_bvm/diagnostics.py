"""Infeasibility classification (blueprint Sec 11).

Matrix BVM returns a *classified* verdict, not a boolean. Each finding names the
offending section and the controlling composition/pinch so the user learns why a
split failed, not merely that it did. Classes mirror the Sec 11 table:

    below_min_reflux        regions pinch before overlapping
    no_connection           closest approach never within a stage step
    leaves_simplex          a marched profile went negative / exited
    boundary_block          required connection lies across a distillation boundary
    infeasible_entrainer    extractive manifolds cannot bridge at this E/F
    cannot_anchor           no controlling saddle on the interior-section pathway
    unreachable_side_purity side-product target never attained on the profile
    thermo_invalid          thermo returned no valid K / single phase

`classify` inspects marched profiles + a connection result; it does not re-march.
"""

from collections import namedtuple

import numpy as np

Finding = namedtuple("Finding", "cls section detail")


def _left_simplex(prof):
    return prof["status"] == "simplex" or float(prof["X"].min()) < -1e-3


def classify(profiles, conn, *, both_pinched=None, extractive=False,
             side_draw=None):
    """Return (feasible, [Finding]). `profiles` maps section name -> march dict.

    conn is the connect() result for the controlling section pair. Pass
    both_pinched=True when both connecting sections reached a pinch without
    meeting (the R_min / min-E/F signature). side_draw is a place.side_draw_stage
    result when a side spec is present.
    """
    findings = []

    for name, prof in profiles.items():
        if _left_simplex(prof):
            bad = int(np.argmin(prof["X"].min(axis=1)))
            findings.append(Finding("leaves_simplex", name,
                                    f"stage {bad} comp {np.round(prof['X'][bad], 3)}"))

    if side_draw is not None and side_draw.get("capped"):
        findings.append(Finding("unreachable_side_purity", "side_draw",
                                f"best achievable {side_draw['achieved']:.3f}"))

    if conn is not None and not conn["connected"] and not findings:
        if not conn["in_simplex"]:
            findings.append(Finding("boundary_block", "connection",
                                    f"junction outside simplex at {np.round(conn['point'], 3)}"))
        elif both_pinched:
            cls = "infeasible_entrainer" if extractive else "below_min_reflux"
            findings.append(Finding(cls, "connection",
                                    f"pinched apart, gap {conn['dmin']:.3g} > tol {conn['tol']:.3g}"))
        else:
            findings.append(Finding("no_connection", "connection",
                                    f"closest approach {conn['dmin']:.3g} > tol {conn['tol']:.3g}"))

    feasible = len(findings) == 0 and (conn is None or conn["connected"])
    return feasible, findings


def _demo():
    # a connected result with in-simplex profiles is feasible, no findings
    good_prof = {"X": np.array([[0.9, 0.1], [0.5, 0.5]]), "status": "pinch"}
    conn_ok = {"connected": True, "in_simplex": True, "dmin": 0.0, "tol": 0.01,
               "point": np.array([0.5, 0.5])}
    feas, f = classify({"rectifying": good_prof}, conn_ok)
    assert feas and not f

    # pinched-apart -> below_min_reflux
    conn_bad = {"connected": False, "in_simplex": True, "dmin": 0.2, "tol": 0.01,
                "point": np.array([0.5, 0.5])}
    feas, f = classify({"rectifying": good_prof}, conn_bad, both_pinched=True)
    assert not feas and f[0].cls == "below_min_reflux", f

    # extractive variant names the entrainer
    feas, f = classify({"rectifying": good_prof}, conn_bad, both_pinched=True,
                       extractive=True)
    assert f[0].cls == "infeasible_entrainer", f

    # a profile that left the simplex is caught first
    bad_prof = {"X": np.array([[0.9, 0.1], [1.2, -0.2]]), "status": "simplex"}
    feas, f = classify({"stripping": bad_prof}, conn_ok)
    assert not feas and f[0].cls == "leaves_simplex", f

    # capped side draw
    feas, f = classify({"rectifying": good_prof}, conn_ok,
                       side_draw={"capped": True, "achieved": 0.7})
    assert not feas and f[0].cls == "unreachable_side_purity", f
    print("diagnostics self-check OK")


if __name__ == "__main__":
    _demo()
