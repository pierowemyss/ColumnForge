"""Levers that were entered, persisted, and then dropped on the floor.

Each of these was a "nothing silently ignored" violation: a value the user could
type (or a model they had selected) that never reached the engine. The tests are
deliberately about the WIRING -- that the number moves -- not about the physics,
which the reference cases already gate.
"""

import numpy as np
import pytest

from side_features.bvm.driver import size_column
from side_features.bvm.handoff import to_solver
from side_features.bvm.problem import build_problem
from side_features.bvm.thermo_adapter import ColumnForgeThermo

# benzene / toluene / xylene, mmHg + degC -- the package's stock ternary
ABC = np.array([(6.90565, 1211.033, 220.79),
                (6.95464, 1344.8, 219.48),
                (6.99052, 1453.43, 215.31)])
Z = np.array([0.4, 0.35, 0.25])


def btx(**kw):
    return build_problem(["benzene", "toluene", "xylene"], [(Z, 100.0, 1.0)],
                         760.0, rec_lk=0.98, rec_hk=0.02, **kw)


@pytest.fixture(scope="module")
def tp():
    return ColumnForgeThermo(ABC)


# ------------------------------------------------------------ eps_stage

def quaternary(**kw):
    """A C=4 split, where no exact profile crossing exists (connect.connect)."""
    return build_problem(["c1", "c2", "c3", "c4"], [(np.array([0.3, 0.3, 0.2, 0.2]),
                                                    100.0, 1.0)],
                         760.0, lk=1, hk=2, rec_lk=0.95, rec_hk=0.05, **kw)


@pytest.fixture(scope="module")
def tp4():
    return ColumnForgeThermo(np.vstack([ABC, [7.00877, 1635.0, 215.0]]))


def test_eps_stage_reaches_the_near_miss_paths(tp4):
    """The Advanced box's connection tolerance is a FLOOR on connect's tol.

    It was persisted to .colx and read back into the widget, but `_gather` never
    put it on the Problem, so connect's 1e-2 default won every time. It now only
    applies where a near miss is all there is -- C >= 4 here.
    """
    loose = size_column(quaternary(eps_stage=0.2), tp4, R=5.0)
    tight = size_column(quaternary(eps_stage=1e-4), tp4, R=5.0)
    assert loose["connection"]["tol"] >= 0.2, loose["connection"]
    assert tight["connection"]["tol"] < 0.2, tight["connection"]
    # ...and it is a floor, not an override: the step-size term still applies
    assert tight["connection"]["tol"] >= 1e-4


def test_eps_stage_cannot_buy_a_connection_at_C3(tp):
    """The knob must NOT widen a junction that can be asked for exactly.

    At C <= 3 the two profiles genuinely intersect when the split is feasible, so
    the gate is the crossing itself and `eps_stage` is not in it. It used to be,
    and that is half of why c2-c4's R_min came out 22% low: at E=0.5 the accepted
    gap reached 0.10 in vapour space, which is 0.07 in liquid.
    """
    R = 1.15                      # just under this column's R_min (~1.25)
    assert not size_column(btx(eps_stage=1e-4), tp, R=R)["feasible"]
    assert not size_column(btx(eps_stage=0.2), tp, R=R)["feasible"]
    # and a feasible reflux is accepted on the crossing, not on the tolerance
    ok = size_column(btx(eps_stage=1e-4), tp, R=4.0)
    assert ok["feasible"] and ok["connection"]["dmin"] < 1e-9
    assert not ok["connection"]["approximate"]


# ------------------------------------------------------------ pressure drop

def test_pressure_drop_ramps_the_column(tp):
    """`Problem.dP` existed and was never set; the GUI's bar/stage went nowhere.

    A real drop makes every stage below the condenser hotter, most of all the
    reboiler, and the marched profiles must record the pressure they were
    evaluated at.
    """
    flat = size_column(btx(), tp, R=4.0)
    ramp = size_column(btx(dP=8.0), tp, R=4.0)          # 8 mmHg/stage
    assert flat["feasible"] and ramp["feasible"]
    assert ramp["column"]["T"][-1] > flat["column"]["T"][-1] + 1.0

    N = ramp["N_total"]
    rect = ramp["profiles"]["rectifying"]
    strip = ramp["profiles"]["stripping"]
    P_bot = 760.0 + 8.0 * (N - 1)
    # the rectifying march ramps down from the condenser...
    k = min(rect["n"], N)
    assert np.allclose(rect["P"][:k], 760.0 + 8.0 * np.arange(k))
    # ...and stops at the reboiler pressure rather than running the ramp out to
    # max_stages, which would walk a 200-stage march clean off the column
    assert rect["P"].max() <= P_bot + 1e-9
    # the stripping section is anchored at the REBOILER and marches up, so its
    # pressure falls, and never below the condenser
    assert strip["P"][0] == pytest.approx(P_bot)
    assert strip["P"][-1] < strip["P"][0] and strip["P"].min() >= 760.0 - 1e-9


def test_the_reboiler_pressure_is_resolved_from_the_stage_count(tp):
    """P_bot = P_top + dP*(N-1) and N is the method's output, so the two are a
    fixed point. `size_column` closes it; a flat column skips the extra pass."""
    ramp = size_column(btx(dP=8.0), tp, R=4.0)
    N = ramp["N_total"]
    strip = ramp["profiles"]["stripping"]
    assert abs(strip["P"][0] - (760.0 + 8.0 * (N - 1))) < 1e-9, (strip["P"][0], N)
    # zero drop leaves every stage at the top pressure, as before
    flat = size_column(btx(), tp, R=4.0)
    assert np.allclose(flat["profiles"]["stripping"]["P"], 760.0)


# ------------------------------------------------------------ handoff

def test_handoff_reports_every_feed(tp):
    """`to_solver` used to emit one feed stage. An extractive design has two, and
    pooling them hands the rigorous solver a different column."""
    init = to_solver(size_column(btx(), tp, R=4.0))
    assert init["feed_stages"] == [init["feed_stage"]]
