"""The free non-key split is BVM's error budget, not a detail of its seeding.

Two key recoveries fix two of the C component splits; the remaining C-2 are free
design variables that the crossing condition determines
(`problem.free_split_indices`). `overall_balance` only seeds them, and how good
that seed is turns out to set r_min almost entirely.

Measured here on c2-c4 (ethane/propane/n-butane, one free split: n-butane).
These are alkanes, UNIFAC gammas within a percent of unity, sharply split -- the
constant-alpha assumption Underwood needs actually holds, which is why this is
the case worth comparing against. Underwood for the recovery-based products is
0.1296. Holding the one free split FIXED and bisecting BVM's r_min:

    split 1e-4   (the default trace floor)   r_min 2.751     +2020%
    split 9.3e-7 (Fenske, splits.py)         r_min 0.1916      +48%
    split 1e-8   (driver._TRACE_LADDER)      r_min 0.1364       +5%
    split 3.1e-10 (junction-solved)          r_min 0.1271       -2%
    split -> 0                               r_min 0.1225       -5%

Three claims, one test each below.

  1. r_min is MONOTONE decreasing in the seed and spans a factor of 21 across the
     range a reasonable person might pick. A seed is not a detail.
  2. It CONVERGES as the seed shrinks -- there is no "too small" to be afraid of,
     which is what makes a downward ladder a safe repair and an upward one not.
  3. At the junction-solved split, BVM's r_min reproduces Underwood to within the
     10% the cross-method gates use. The junction criterion is sound; what was
     wrong was never the geometry.

Underwood: Underwood, A.J.V., "Fractional Distillation of Multicomponent
Mixtures", Chem. Eng. Prog. 44 (1948) 603; as implemented in `core/shortcut.py`
and used the same way by `rbm/tests/test_rmin_agreement.py`.
"""

import os
from functools import partial

import numpy as np
import pytest

from core.shortcut import underwood_min_reflux, underwood_roots
from gui.state.persistence import load_colx
from gui.state.window_state import WindowState
from side_features.bvm.driver import _size_once
from side_features.bvm.pinch import bisect_min
from side_features.bvm.problem import build_problem, overall_balance
from side_features.bvm.thermo_adapter import ColumnForgeThermo

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                      "..", "..", "..", "..", "docs", "examples"))

#: How far BVM's SIZING minimum may sit from Underwood's thermodynamic one. The
#: same 10% the RBM cross-method gate uses -- these answer slightly different
#: questions (profiles intersecting vs reachable sets touching), so they are not
#: expected to agree exactly, only to agree.
REL_TOL = 0.10

#: The one free split, at the values the table above pins.
_FLOOR = 1e-4
_LADDER = 1e-8
_SOLVED = 3.066e-10          # splits.solve_free_splits at feed_loc=6, R=0.14-2.0


@pytest.fixture(scope="module")
def c2c4():
    """c2-c4 with RECOVERY-based products, so the non-key split is actually free.

    The sharp (explicit xD/xB) variant `rbm/tests` uses would pin n-butane at
    exactly zero and there would be nothing to measure.
    """
    ws = WindowState()
    ws.load_from_dict(load_colx(os.path.join(_ROOT, "c2-c4.colx")))
    order = ws.get_species_names()
    assert order == ["ethane", "propane", "n-butane"], order
    tc = ws.thermodynamics_config
    tp = ColumnForgeThermo(tc.psat_params(order),
                           gamma_fn=ws.build_gamma_fn(order),
                           phi_fn=ws.build_phi_fn(order))
    z = np.array([0.5, 0.25, 0.25])
    prob = build_problem(comps=order, feeds=[(z, 100.0, 1.0)],
                         pressure=tc.pressure_in_psat_unit(ws.pressure),
                         lk=0, hk=1, rec_lk=0.99, rec_hk=0.01)
    assert len(_free(prob)) == 1, "expected exactly one free split (n-butane)"
    return prob, tp, z


def _free(prob):
    from side_features.bvm.problem import free_split_indices
    return free_split_indices(prob)


def _rmin_at(prob, tp, split):
    """BVM's r_min with the free split HELD at `split` -- no ladder, no reseeding."""
    s = np.array([float(split)])

    def feasible(R):
        return _size_once(prob, tp, float(R), split=s)["feasible"]

    return bisect_min(feasible, 0.05, 30.0, tol=1e-3)


def _underwood(prob, tp, z, split):
    """Underwood's R_min for the products that same split produces."""
    _, T = tp.bubble(z, prob.pressure)
    alpha = tp.Psat(T)
    alpha = alpha / alpha[prob.hk]
    roots = underwood_roots(alpha, z, 1.0, prob.lk, prob.hk)
    assert roots, "no Underwood root between the keys"
    xD, _xB, _D, _B = overall_balance(prob, None, split=np.array([float(split)]))
    return max(underwood_min_reflux(alpha, xD, th) for th in roots)


def test_rmin_is_monotone_in_the_free_split(c2c4):
    """Claim 1: the seed sets r_min, over a factor of 21, in one direction."""
    prob, tp, _z = c2c4
    got = [(s, _rmin_at(prob, tp, s))
           for s in (_FLOOR, _LADDER, _SOLVED)]
    for s, r in got:
        assert r is not None, f"no feasible reflux at all with split {s:.1e}"
    vals = [r for _s, r in got]
    assert vals == sorted(vals, reverse=True), got   # smaller seed -> smaller r_min
    assert vals[0] / vals[-1] > 10, got              # measured 21x
    # and the coarse floor is not merely imprecise, it is a different answer
    assert vals[0] > 1.0, f"trace floor should be wildly high, got {vals[0]:.4f}"


def test_rmin_converges_as_the_free_split_shrinks(c2c4):
    """Claim 2: there is a plateau, so a DOWNWARD reseeding ladder is safe.

    This is what licenses `driver._TRACE_LADDER` to retry at smaller floors: the
    answer stops moving rather than running away, so the ladder cannot overshoot
    into a different regime the way raising the floor does.
    """
    prob, tp, _z = c2c4
    lo = _rmin_at(prob, tp, 1e-12)
    lower = _rmin_at(prob, tp, 1e-14)
    assert lo is not None and lower is not None
    assert abs(lo - lower) / lo < 0.02, (lo, lower)


def test_bvm_rmin_matches_underwood_at_the_solved_split(c2c4):
    """Claim 3: the junction criterion is right; the seed carried the error.

    The literature anchor for the whole module. If this drifts, either the
    junction test or the thermo moved -- not the seeding, which is pinned above.
    """
    prob, tp, z = c2c4
    ref = _underwood(prob, tp, z, _SOLVED)
    got = _rmin_at(prob, tp, _SOLVED)
    assert got is not None, "no feasible reflux at the solved split"
    assert abs(got - ref) / ref < REL_TOL, f"BVM {got:.4f} vs Underwood {ref:.4f}"


def test_design_solved_closes_the_junction_or_says_why_not(c2c4):
    """`splits.design_solved`: one corrector step, and it never lies about itself.

    Seeded, the junction is a tolerance-accepted near miss. Solved at the feed
    position the first sizing found, it closes to ~1e-11 and the design says
    `exact`. What matters as much is the other branch: above C=3 the underlying
    least squares often does not converge, and the design must then come back
    carrying the SEEDED splits with `split_solved` False and a reason -- not
    looking solved. Costs one marched-profile least squares (~7 s at C=3), which
    is why it is opt-in and stays out of the r_min bisection.
    """
    from side_features.bvm.splits import design_solved

    prob, tp, _z = c2c4
    seeded = _size_once(prob, tp, 0.5)
    solved = design_solved(prob, tp, 0.5)

    assert "split_solved" in solved and solved.get("split_solve_note")
    assert solved["split_solved"], solved["split_solve_note"]
    assert solved["exact"] and solved["junction_residual"] < 1e-6, solved
    # it actually moved the split, and toward the solved value, not away
    free = _free(prob)[0]
    assert solved["xD"][free] != seeded["xD"][free]
    assert abs(np.log10(solved["xD"][free]) - np.log10(_SOLVED)) < 2.0, (
        solved["xD"][free], _SOLVED)


def test_underwood_barely_notices_the_split_that_moves_bvm_21x(c2c4):
    """The disagreement is BVM's alone, which is what makes it diagnosable.

    A trace heavy contributes almost nothing to Underwood's distillate sum, so
    the reference stays put across four decades of the same seed that moves BVM
    from 2.751 to 0.127. Without this, "BVM disagrees with Underwood" could be
    read as the two methods pricing the non-key differently; they do not.
    """
    prob, tp, z = c2c4
    refs = [_underwood(prob, tp, z, s) for s in (_FLOOR, _LADDER, _SOLVED)]
    assert max(refs) / min(refs) < 1.01, refs
