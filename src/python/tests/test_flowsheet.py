"""Multi-column flowsheets: topology, the recycle tear, and honest reporting.

`core.flowsheet._demo()` is the real self-check and covers the whole surface
(acyclic order, a torn recycle with a purge, an implicit spec nested inside a
tear, a starved tear, cancellation, the cold start). It is invoked here because
CI runs pytest and never `python -m core.flowsheet`, so without this the demo's
coverage stops at the developer's terminal — the same arrangement
test_month5_energy.py uses for the energy-balance modules.

The cases below the demo are the ones that want pytest rather than an assert
chain: parametrized topology, and the two failure *messages*, which are what a
user actually sees and which a single "not converged" would silently merge.
"""

import numpy as np
import pytest

from core.dof import Spec, SpecKind
from core.flowsheet import (
    Connection, Flowsheet, FlowsheetError, auto_layout, is_recycle, natural_q,
    sccs, solve_flowsheet, tear_set, validate, validate_connection,
)
import core.flowsheet as fsmod


KNOBS = dict(max_iter=300, tol=1e-8)


def test_core_self_check():
    """The module's own _demo(), so CI runs it too."""
    fsmod._demo()


# --- topology --------------------------------------------------------------

def test_a_sequence_is_solved_in_dependency_order_and_tears_nothing():
    fs = fsmod._series()
    assert sccs(fs) == [["C1"], ["C2"]]
    assert tear_set(fs) == []
    assert not any(is_recycle(fs, c) for c in fs.connections)


def test_a_recycle_puts_both_columns_in_one_component_and_tears_one_edge():
    fs = fsmod._recycle()
    assert sccs(fs) == [["C1", "C2"]]
    assert tear_set(fs) == ["C2.D->C1@8"]
    assert [is_recycle(fs, c) for c in fs.connections] == [False, True]


def test_the_tear_set_does_not_depend_on_connection_insertion_order():
    """A tear set that differs between a saved file and the session that saved
    it would make a .colx non-reproducible."""
    fs = fsmod._recycle()
    shuffled = Flowsheet(units=dict(fs.units), connections=list(reversed(fs.connections)),
                         comps=fs.comps, default_method=fs.default_method)
    assert tear_set(shuffled) == tear_set(fs)


def test_port_keys_are_stable_when_a_draw_is_renamed_or_moved():
    """The whole reason a Connection references an opaque key and not a label:
    renaming a side draw must not rewrite the connections that source from it."""
    fs = fsmod._series()
    fs.units["C1"].draws = [("S1", 6, 5.0, 0.0)]
    conn = Connection("c", "C1", "S1", "C2", 8)
    assert validate_connection(fs, conn) is None
    ports = fs.units["C1"].ports()
    assert ports["S1"].stage == 6 and ports["S1"].phase == "liquid"

    fs.units["C1"].draws = [("S1", 11, 0.0, 5.0)]      # moved, and now vapor
    assert validate_connection(fs, conn) is None       # the connection survives
    assert fs.units["C1"].ports()["S1"].stage == 11
    assert natural_q(fs.units["C1"], "S1") == 0.0      # vapor draw


@pytest.mark.parametrize("condenser,expected", [("total", 1.0), ("partial", 0.0)])
def test_distillate_quality_follows_the_condenser(condenser, expected):
    fs = fsmod._series()
    fs.units["C1"].condenser = condenser
    assert natural_q(fs.units["C1"], "D") == expected
    assert natural_q(fs.units["C1"], "B") == 1.0


def test_auto_layout_is_deterministic_and_left_to_right():
    fs = fsmod._series()
    pos = auto_layout(fs)
    assert pos["C1"][0] < pos["C2"][0]
    assert auto_layout(fs) == pos


# --- validation ------------------------------------------------------------

@pytest.mark.parametrize("conn,fragment", [
    (Connection("x", "C1", "B", "C2", 1), "interior tray"),
    (Connection("x", "C1", "B", "C2", 99), "interior tray"),
    (Connection("x", "C1", "B", "C1", 8), "cannot feed itself"),
    (Connection("x", "C1", "nope", "C2", 8), "no outlet"),
    (Connection("x", "C9", "B", "C2", 8), "no column"),
    (Connection("x", "C1", "B", "C2", 8, split_fraction=1.5), "split fraction"),
    # port "D" here, not "B": _series() already commits all of C1.B, and the
    # over-subscription check would (correctly) fire first.
    (Connection("x", "C1", "D", "C2", 8, q=7.0), "thermal quality"),
])
def test_an_illegal_connection_is_refused_with_a_reason(conn, fragment):
    why = validate_connection(fsmod._series(), conn)
    assert why is not None and fragment in why, why


def test_a_port_cannot_send_more_than_all_of_itself():
    fs = fsmod._series()          # already sends C1.B fully to C2
    why = validate_connection(fs, Connection("y", "C1", "B", "C2", 9, split_fraction=0.5))
    assert why is not None and "of one stream" in why


def test_a_purge_split_is_legal_and_leaves_as_an_external_product():
    fs = fsmod._recycle(split=0.9)
    assert validate(fs) == []
    res = solve_flowsheet(fs, knobs=KNOBS)
    purge = [p for p in res.products if p["unit"] == "C2" and p["port"] == "D"]
    assert len(purge) == 1 and purge[0]["purge"]
    assert purge[0]["flow"] == pytest.approx(0.1 * 30.0, abs=1e-6)


def test_a_recycle_loop_with_no_external_feed_is_refused():
    fs = fsmod._recycle()
    fs.units["C1"] = fsmod._col("C1", D=50.0)          # feed removed
    problems = validate(fs)
    assert any("no external feed" in p for p in problems), problems
    with pytest.raises(FlowsheetError, match="no external feed"):
        solve_flowsheet(fs, knobs=KNOBS)


def test_a_stale_connection_is_caught_when_the_column_shrinks():
    """The silent case: the destination stage was legal when it was drawn, and
    the user later cut num_stages on a different page entirely."""
    fs = fsmod._series()
    assert validate(fs) == []
    fs.units["C2"].n_stages = 5                        # connection targets stage 8
    assert any("interior tray" in p for p in validate(fs)), validate(fs)
    with pytest.raises(FlowsheetError, match="interior tray"):
        solve_flowsheet(fs, knobs=KNOBS)


# --- solving ---------------------------------------------------------------

def test_a_sequence_closes_on_external_streams_only():
    res = solve_flowsheet(fsmod._series(), knobs=KNOBS)
    assert res.converged, res.message
    assert res.closure < 1e-3
    # the inter-unit stream is neither an external feed nor an external product
    assert {(p["unit"], p["port"]) for p in res.products} == {
        ("C1", "D"), ("C2", "D"), ("C2", "B")}
    assert np.allclose(res.feed_totals, 100.0 * np.array([0.5, 0.3, 0.2]))


def test_a_recycle_closes_on_external_streams_and_the_recycle_is_not_feed():
    res = solve_flowsheet(fsmod._recycle(), knobs=KNOBS)
    assert res.converged, res.message
    assert res.tear_converged and res.tear_residual < 1e-5
    assert np.allclose(res.feed_totals, 100.0 * np.array([0.5, 0.3, 0.2]))
    out = sum(p["flow"] * p["comp"] for p in res.products)
    assert np.allclose(res.feed_totals, out, atol=1e-3), (res.feed_totals, out)


def test_the_direct_sequence_puts_each_component_in_its_own_product():
    """BTX down a direct sequence: benzene overhead of C1, toluene overhead of
    C2, xylene out the bottom. If the connection carried the wrong port this is
    the assertion that notices."""
    res = solve_flowsheet(fsmod._series(), knobs=KNOBS)
    by = {(p["unit"], p["port"]): p for p in res.products}
    assert by[("C1", "D")]["comp"][0] > 0.99      # benzene
    assert by[("C2", "D")]["comp"][1] > 0.97      # toluene
    assert by[("C2", "B")]["comp"][2] > 0.97      # xylene


def test_a_recycle_larger_than_the_external_feed_still_starts():
    """The general form of side_sections' 'bottoms rate B=-5 must be positive':
    without a mass-balanced cold start, pass 1 is solved short by the recycle."""
    res = solve_flowsheet(fsmod._recycle(split=0.98, D2=140.0), knobs=KNOBS,
                          max_passes=60)
    assert res.converged, res.message
    assert res.streams["C2.D->C1@8"].flow > 100.0


def test_an_implicit_spec_inside_a_recycle_converges_and_relaxes_the_tolerance():
    fs = fsmod._recycle()
    fs.units["C1"].specs = [Spec(SpecKind.REFLUX_RATIO, 3.0, "condenser"),
                            Spec(SpecKind.DIST_PURITY, 0.99, "column", component=0)]
    res = solve_flowsheet(fs, tol=1e-5, knobs=KNOBS)
    assert res.converged, res.message
    # the resolver's own gate is 1e-4, so a tighter outer tolerance would only
    # chase its noise -- the relaxation is reported rather than hidden
    assert res.tol_used == 1e-4
    assert res.units["C1"].profile["xD"][0] >= 0.99 - 1e-3


# --- honesty ---------------------------------------------------------------

def test_a_starved_tear_does_not_claim_convergence():
    res = solve_flowsheet(fsmod._recycle(), tol=1e-14, max_passes=2, knobs=KNOBS)
    assert all(u.converged for u in res.units.values())    # the columns are fine
    assert not res.tear_converged and not res.converged     # the tear is not
    assert "tear" in res.message and "NOT converged" in res.message


def test_a_failed_tear_and_a_failed_column_are_different_messages():
    """One 'not converged' covering both sends the user to the wrong knob."""
    tear = solve_flowsheet(fsmod._recycle(), tol=1e-14, max_passes=2, knobs=KNOBS)
    unit = solve_flowsheet(fsmod._recycle(), knobs=dict(max_iter=2, tol=1e-8))
    assert "did not converge" not in tear.message, tear.message
    assert "did not converge" in unit.message, unit.message
    assert unit.message != tear.message


def test_a_failed_tear_is_visible_on_each_unit_profile_too():
    """The Results tab reads a single profile's message; it must not read
    'Converged' while the flowsheet knows better."""
    res = solve_flowsheet(fsmod._recycle(), tol=1e-14, max_passes=2, knobs=KNOBS)
    for u in res.units.values():
        assert u.profile["converged"] is False
        assert "NOT converged" in u.profile["message"]


def test_cancellation_is_aborted_not_converged():
    res = solve_flowsheet(fsmod._recycle(), knobs=KNOBS, cancel=lambda: True)
    assert res.aborted and not res.converged
    assert res.message == "Aborted."


def test_progress_is_monotonic_and_never_overruns():
    ticks = []
    solve_flowsheet(fsmod._recycle(), knobs=KNOBS,
                    report=lambda d, t, r: ticks.append((d, t)))
    assert ticks
    assert all(a[0] <= b[0] for a, b in zip(ticks, ticks[1:]))
    assert all(d <= t for d, t in ticks)


def test_an_infeasible_spec_names_the_column_that_could_not_meet_it():
    """Unwrapped, the resolver's ValueError names no unit, so inside a tear it
    reads as an unattributable whole-flowsheet failure."""
    fs = fsmod._series()
    fs.units["C2"].specs = [Spec(SpecKind.REFLUX_RATIO, 3.0, "condenser"),
                            Spec(SpecKind.DIST_PURITY, 0.999999, "column", component=2)]
    with pytest.raises(FlowsheetError, match="'C2' could not meet"):
        solve_flowsheet(fs, knobs=KNOBS)


def test_a_duty_spec_on_the_bubble_point_solver_is_refused_not_ignored():
    """Only Inside-Out reports duties; the residual would otherwise KeyError."""
    fs = fsmod._series()
    fs.units["C2"].method = "Bubble-Point"
    fs.units["C2"].specs = [Spec(SpecKind.REFLUX_RATIO, 3.0, "condenser"),
                            Spec(SpecKind.REBOILER_DUTY, 1e6, "reboiler")]
    with pytest.raises(FlowsheetError, match="duty spec"):
        solve_flowsheet(fs, knobs=KNOBS)


def test_per_column_method_override_is_honored():
    fs = fsmod._series()
    fs.default_method = "Bubble-Point"
    fs.units["C2"].method = "Inside-Out"
    res = solve_flowsheet(fs, knobs=KNOBS)
    assert res.converged, res.message
    # Inside-Out reports duties; the bubble-point path does not
    assert "reboiler_duty" in res.units["C2"].profile
    assert "reboiler_duty" not in res.units["C1"].profile
