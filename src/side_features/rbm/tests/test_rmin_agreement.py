"""Does RBM's minimum reflux agree with anything that can be checked?

`test_rbm.py` cross-checks RBM against BVM at 15% on BTX and otherwise asserts
wide sanity bands. That was enough to catch a broken method; it is not enough to
catch a biased one, and c2-c4 came out under both BVM and Underwood with the
rectification bodies visibly apart at the reported r_min.

So this file pins the three claims that actually decide whether r_min is right:

  1. against Underwood -- c2-c4 is ethane/propane/n-butane, near-ideal and
     sharply split, which is exactly where Underwood is trustworthy;
  2. against BVM -- an independent method that shares no code below `sections`;
  3. against the geometry -- at r_min the bodies must TOUCH. This is the check
     the other two cannot make: a number can agree with Underwood by luck, but
     bodies that do not meet at their own claimed touching point are wrong on
     their own terms. `test_rbm.py:182` checks the gap shrinks; it never checks
     that it reaches zero at the value r_min returns.

Run this file directly for a timed table of every number instead of a verdict:

    PYTHONPATH=src:src/python python -m side_features.rbm.tests.test_rmin_agreement
"""

import time

import numpy as np
import pytest

from side_features.bvm.driver import r_min as bvm_r_min
from side_features.bvm.problem import build_problem
from side_features.bvm.thermo_adapter import ColumnForgeThermo
from side_features.rbm import bodies as B
from side_features.rbm.driver import analyse, reflux_band

_C2C4 = "docs/examples/c2-c4.colx"

#: How far RBM may sit from a trusted reference. Underwood and RBM answer the
#: same question by unrelated routes, so this is a real gate, not a formality --
#: but they are not the same approximation, so it is not 1% either.
REL_TOL = 0.10


def _c2c4_problem(sharp=True):
    """(prob, provider) for c2-c4.colx, built the way `rbm_module._gather` does.

    Sharp by default: RBM needs the exact zeros to open the pinch ladder (see
    `rbm/__init__.py`), and the .colx distillate is 0.995 ethane anyway.
    """
    from gui.state import persistence
    from gui.state.window_state import WindowState

    ws = WindowState()
    ws.load_from_dict(persistence.load_colx(_C2C4))
    order = ws.get_species_names()
    assert order == ["ethane", "propane", "n-butane"], order

    P = ws.thermodynamics_config.pressure_in_psat_unit(ws.pressure)
    provider = ColumnForgeThermo(ws.thermodynamics_config.psat_params(order),
                                 gamma_fn=ws.build_gamma_fn(order),
                                 phi_fn=ws.build_phi_fn(order))
    z = np.array([0.5, 0.25, 0.25])
    lk, hk = 0, 1

    kw = {}
    if sharp:
        xD = np.zeros(3); xD[lk] = 1.0
        xB = z.copy(); xB[lk] = 0.0
        kw = {"xD": xD, "xB": xB / xB.sum()}

    prob = build_problem(comps=order, feeds=[(z, 100.0, 1.0)], pressure=P,
                         lk=lk, hk=hk, rec_lk=0.99, rec_hk=0.01, **kw)
    return prob, provider, z


def _underwood_rmin(prob, tp, z, q=1.0):
    """Underwood's R_min for the same split, at feed-bubble relative volatility.

    alpha comes from the pure-component vapour pressures at the feed bubble
    point, HK basis. These are alkanes -- UNIFAC gammas are within a percent of
    unity -- so the constant-alpha assumption Underwood needs actually holds
    here, which is why c2-c4 is the case worth comparing against.
    """
    from core.shortcut import underwood_min_reflux, underwood_roots

    _, T = tp.bubble(z, prob.pressure)
    alpha = tp.Psat(T)
    alpha = alpha / alpha[prob.hk]
    roots = underwood_roots(alpha, z, q, prob.lk, prob.hk)
    assert roots, "no Underwood root between the keys"
    return max(underwood_min_reflux(alpha, prob.xD, th) for th in roots)


@pytest.fixture(scope="module")
def c2c4():
    return _c2c4_problem()


def test_rbm_rmin_agrees_with_underwood_on_c2c4(c2c4):
    """The trusted reference for a near-ideal sharp split."""
    prob, tp, z = c2c4
    rbm = reflux_band(prob, tp)[0]
    ref = _underwood_rmin(prob, tp, z)
    assert rbm is not None, "RBM found no feasible reflux at all"
    assert abs(rbm - ref) / ref < REL_TOL, f"RBM {rbm:.4f} vs Underwood {ref:.4f}"


def test_rbm_rmin_agrees_with_bvm_on_c2c4(c2c4):
    """Two methods, no shared code below `sections`, same answer.

    BVM is asked the SMEARED version of the split: it marches profiles, and an
    exactly-zero product component traps the march on a simplex face, so it
    returns nothing at all for the sharp spec (pinned in
    `test_rbm_sizes_a_sharp_split_that_bvm_cannot`). RBM is asked the sharp one,
    which is the specification it exists to handle. 99%/1% recoveries put the two
    questions close enough that the answers are comparable, and where they are
    not, that is the finding.
    """
    prob, tp, _ = c2c4
    rbm = reflux_band(prob, tp)[0]
    bvm = bvm_r_min(_c2c4_problem(sharp=False)[0], tp)
    assert rbm is not None and bvm is not None, (rbm, bvm)
    assert abs(rbm - bvm) / bvm < REL_TOL, f"RBM {rbm:.4f} vs BVM {bvm:.4f}"


def test_the_bodies_actually_touch_at_the_reported_rmin(c2c4):
    """r_min is defined as the reflux where the bodies just touch, so they must.

    The failure this catches is a feasibility predicate that flipped true for a
    reason other than the geometry -- a spurious low-reflux island, or a body
    that should not have been constructed. Either way the returned number is not
    a touching point and the plot shows two shapes with daylight between them.
    """
    prob, tp, _ = c2c4
    lo = reflux_band(prob, tp)[0]
    assert lo is not None
    assert analyse(prob, tp, r=lo)["max_gap"] <= B.TOUCH_TOL, "bodies apart at r_min"
    below = analyse(prob, tp, r=0.95 * lo)["max_gap"]
    assert below > B.TOUCH_TOL, f"still touching 5% below r_min ({below:.3g})"


def test_the_stripping_stable_node_lands_on_the_rectifying_body(c2c4):
    """The textbook picture of minimum reflux, stated geometrically.

    At R_min a simple column pinches, and the pinch the stripping section runs
    into is a point the rectifying profile also reaches. In body terms: the
    stripping section's stable node lies ON the rectifying body. It is a
    stricter statement than `max_gap == 0` -- the sections could touch anywhere
    along their hulls and still satisfy that -- and it is the one an eye reads
    off the ternary plot.
    """
    prob, tp, _ = c2c4
    lo = reflux_band(prob, tp)[0]
    res = analyse(prob, tp, r=lo)
    secs = {s["name"]: s for s in res["sections"]}
    rect = np.vstack([b["vertices"] for b in secs["rectifying"]["bodies"]])
    nodes = [p["x"] for p in secs["stripping"]["pinches"]
             if p["in_simplex"] and p["kind"] == "stable_node"]
    assert nodes, "the stripping section has no stable node in the simplex"
    d = min(B.body_distance(np.atleast_2d(n), rect) for n in nodes)
    assert d <= B.TOUCH_TOL, f"stripping stable node sits {d:.3g} off the rectifying body"


def _report():
    """Every number in one timed table -- the baseline to judge a change against."""
    prob, tp, z = _c2c4_problem()
    ref = _underwood_rmin(prob, tp, z)

    t0 = time.perf_counter()
    lo, hi = reflux_band(prob, tp)
    t_rbm = time.perf_counter() - t0

    t0 = time.perf_counter()
    bvm = bvm_r_min(_c2c4_problem()[0], tp)
    t_bvm = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = analyse(prob, tp, r=lo) if lo else None
    t_one = time.perf_counter() - t0

    print(f"\nc2-c4  (sharp split, ethane/propane over n-butane)")
    print(f"  Underwood r_min   {ref:.4f}")
    print(f"  BVM       r_min   {bvm if bvm is None else f'{bvm:.4f}'}"
          f"   [{t_bvm:.1f} s]")
    print(f"  RBM       r_min   {lo if lo is None else f'{lo:.4f}'}"
          f"   r_max {hi}   [{t_rbm:.1f} s]")
    if res is not None:
        print(f"  body gap at RBM r_min   {res['max_gap']:.3g}"
              f"   (touching if <= {B.TOUCH_TOL:g})")
        for s in res["sections"]:
            kinds = [p["kind"] for p in s["pinches"] if p["in_simplex"]]
            print(f"    {s['name']:<11} {len(s['bodies'])} bodies, "
                  f"{len(kinds)} pinches in simplex: {kinds}")
        print(f"  one analyse()   [{t_one:.2f} s]")


if __name__ == "__main__":
    _report()
