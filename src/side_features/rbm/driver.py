"""RBM design driver: feasibility, minimum/maximum reflux, minimum entrainer.

`analyse(prob, tp, r, EF)` is the single operating-point call. It builds the
section chain, solves each section's pinch points, spans them into rectification
bodies, and reports the closest approach between adjacent sections' bodies. Zero
closest approach means the sections can be joined by a real column profile, so
the split is feasible at that operating point (paper p.120).

Everything else is a search over that one test:

  * `r_min` / `r_max` -- the reflux ratios at which the bodies just touch,
    by bisection between a reflux known to be infeasible and one known to be
    feasible, exactly as the paper prescribes (p.126). Extractive columns have
    BOTH bounds: raising the reflux washes the entrainer out of the extractive
    section and the separation stops working, so the feasible band is an interval
    rather than a ray (paper p.13, "there is a maximum reflux above which
    separation cannot be achieved").
  * `operating_region` -- that band as a function of entrainer flow. The two
    curves meet at a nose, and the entrainer ratio there is (E/F)_min: below it
    no reflux works at all. That is the paper's Figure 9, and reading (E/F)_min
    off the nose avoids needing the saddle-node bifurcation tracking the paper
    uses to get it directly (p.84).

RBM gives no stage count -- a body approximates a profile, it is not one. Feed
the operating point it finds to BVM to size the column there.
"""

import numpy as np

from side_features.bvm.problem import overall_balance
from side_features.bvm.pinch import feasible_band
from side_features.bvm.sections import single_feed_chain, extractive_chain

from .bodies import TOUCH_TOL, middle_bodies, product_bodies, sets_distance
from .pinch import pinch_points

#: Operational constraints from the paper (p.146): a design wants headroom, not
#: the feasibility edge. E/F is pushed up until the reflux band is at least
#: `PI_R_MIN` wide, and E/F itself at least `PI_EF_MIN` above its own minimum.
PI_EF_MIN = 1.1
PI_R_MIN = 2.0


def _chain(prob, r, EF, xD, xB, D, B):
    """Section chain + which of them is the extractive middle (None if simple)."""
    if prob.extractive and prob.x_E is not None and EF:
        rect, ext, strip = extractive_chain(prob, r, EF, xD, xB, D, B)
        return [rect, ext, strip], 1
    rect, strip = single_feed_chain(prob, r, xD, xB, D, B)
    return [rect, strip], None


def analyse(prob, tp, r, EF=None):
    """Pinches, bodies and body gaps for one operating point.

    Returns dict(feasible, gaps, sections, r, EF, ...) where `sections` carries,
    per section, its pinch records and its rectification bodies -- everything the
    diagram needs -- and `gaps` is one closest-approach per adjacent pair.

    Product compositions come from `overall_balance` with the trace floors turned
    OFF. That is deliberate and is the substantive difference from how BVM builds
    the same numbers: BVM needs a trace of every component to keep a marched
    profile from being trapped on a simplex face, but the pinch equations are
    algebraic and handle an exact zero without complaint. More than that, the
    exact zeros are what CREATE the branch structure RBM lives on -- with
    x_D,i > 0 for every i there is a single physical rectifying pinch, while
    x_D = (1, 0, 0) opens the edge families and gives the unstable-node /
    saddle / stable-node ladder the body rules need.
    """
    trace, ent = prob.trace_floor, prob.entrainer_trace
    try:
        prob.trace_floor, prob.entrainer_trace = 0.0, 0.0
        xD, xB, D, B = overall_balance(prob, EF)
    finally:
        prob.trace_floor, prob.entrainer_trace = trace, ent

    P = prob.pressure
    secs, mid = _chain(prob, r, EF, xD, xB, D, B)
    prods = [xD, None, xB] if mid is not None else [xD, xB]

    info = []
    for k, sec in enumerate(secs):
        ps = pinch_points(sec, tp, P)
        if k == mid:
            bs = middle_bodies(ps)
        else:
            bs = product_bodies(ps, prods[k])
        info.append({"name": sec.name, "section": sec, "pinches": ps,
                     "bodies": bs})

    gaps = []
    for k in range(len(info) - 1):
        d, ia, ib = sets_distance(info[k]["bodies"], info[k + 1]["bodies"])
        gaps.append({"pair": (info[k]["name"], info[k + 1]["name"]),
                     "distance": d, "active": (ia, ib)})

    feasible = bool(gaps) and all(g["distance"] <= TOUCH_TOL for g in gaps)
    return {"feasible": feasible, "gaps": gaps, "sections": info,
            "r": float(r), "EF": None if EF is None else float(EF),
            # carried out so a caller can SAY that an explicit xD/xB does not
            # close the balance. `overall_balance` only warns, and a warning
            # raised on a solver thread reaches nobody.
            "balance_residual": float(prob.balance_residual),
            "xD": xD, "xB": xB, "D": D, "B": B,
            "comps": list(prob.comps), "lk": prob.lk, "hk": prob.hk,
            "max_gap": max((g["distance"] for g in gaps), default=float("inf"))}


def _feasible_at(prob, tp, r, EF):
    try:
        return analyse(prob, tp, r, EF)["feasible"]
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return False


def reflux_band(prob, tp, EF=None, r_lo=0.05, r_hi=30.0, n_scan=24, tol=1e-3):
    """(r_min, r_max) at this entrainer ratio, or (None, None) if nothing works.

    Scanned first, then bisected at each end -- see `bvm.pinch.feasible_band`,
    which both methods share so the two panels' operating-region plots are
    reading the same kind of number. A simple column's band runs to the scan
    ceiling and `r_max` comes back None.
    """
    return feasible_band(lambda r: _feasible_at(prob, tp, float(r), EF),
                         r_lo, r_hi, n_scan=n_scan, tol=tol)


def r_min(prob, tp, EF=None, r_hi=30.0):
    """Minimum reflux by body intersection (paper p.126)."""
    return reflux_band(prob, tp, EF=EF, r_hi=r_hi)[0]


def r_max(prob, tp, EF=None, r_hi=30.0):
    """Maximum reflux; None when the band is open at the top (a simple column)."""
    return reflux_band(prob, tp, EF=EF, r_hi=r_hi)[1]


def operating_region(prob, tp, EF_grid=None, r_hi=30.0, n_scan=24,
                     on_step=None, cancelled=None):
    """The feasible (E/F, r) region -- the paper's Figure 9.

    Returns dict(EF, r_min, r_max, EF_min, r_at_EF_min, operating). Entries are
    NaN where no reflux is feasible at that entrainer ratio. `EF_min` is the
    smallest sampled ratio that admits any reflux at all -- the nose where the
    two bounds meet -- and `operating` is the paper's recommended point, the
    first ratio with enough headroom in both directions (PI_EF_MIN, PI_R_MIN).

    `on_step(done, total)` and `cancelled()` are optional hooks for a caller that
    is driving this from a GUI thread. One entrainer ratio is a whole reflux
    band, i.e. tens of pinch maps, so this is minutes of work on a non-ideal
    ternary and it has to be both visible and abortable. A cancelled sweep
    returns what it has: the unswept ratios keep their NaN and the plot already
    draws around them.
    """
    if EF_grid is None:
        EF_grid = np.linspace(0.2, 2.0, 12)
    EFs = np.atleast_1d(np.asarray(EF_grid, float))
    lo = np.full(len(EFs), np.nan)
    hi = np.full(len(EFs), np.nan)
    for i, ef in enumerate(EFs):
        if cancelled is not None and cancelled():
            break
        a, b = reflux_band(prob, tp, EF=float(ef), r_hi=r_hi, n_scan=n_scan)
        if a is not None:
            lo[i] = a
            hi[i] = r_hi if b is None else b
        if on_step is not None:
            on_step(i + 1, len(EFs))

    feasible_idx = np.flatnonzero(np.isfinite(lo))
    ef_min = float(EFs[feasible_idx[0]]) if len(feasible_idx) else None
    r_at = float(lo[feasible_idx[0]]) if len(feasible_idx) else None

    operating = None
    for i in feasible_idx:
        if ef_min and EFs[i] < PI_EF_MIN * ef_min:
            continue
        if np.isfinite(hi[i]) and hi[i] < PI_R_MIN * lo[i]:
            continue
        operating = {"EF": float(EFs[i]), "r_min": float(lo[i]),
                     "r_max": float(hi[i])}
        break

    return {"EF": EFs, "r_min": lo, "r_max": hi, "EF_min": ef_min,
            "r_at_EF_min": r_at, "operating": operating}


def _demo():
    from side_features.bvm.problem import build_problem
    from side_features.bvm.thermo_adapter import ColumnForgeThermo

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = ColumnForgeThermo(abc)
    z = np.array([0.4, 0.35, 0.25])
    prob = build_problem(["b", "t", "x"], [(z, 100.0, 1.0)], 760.0,
                         rec_lk=0.98, rec_hk=0.02,
                         xD=np.array([1.0, 0.0, 0.0]),
                         xB=np.array([0.0, 0.5838, 0.4162]))

    # one operating point: every section gets pinches and at least one body
    a = analyse(prob, tp, r=4.0)
    assert [s["name"] for s in a["sections"]] == ["rectifying", "stripping"]
    assert all(s["pinches"] for s in a["sections"]), "a section found no pinch"
    assert all(s["bodies"] for s in a["sections"])
    assert len(a["gaps"]) == 1
    # the rectifying ladder the body rules need: strictly increasing n_stable
    ns = [p["n_stable"] for p in a["sections"][0]["pinches"]]
    assert ns == sorted(ns) and len(set(ns)) > 1, ns
    # RBM must NOT apply BVM's trace floors -- exact zeros are the whole point
    assert a["xD"][1] == 0.0 and a["xD"][2] == 0.0, a["xD"]

    # feasibility is monotone in reflux for a simple column, and r_min is inside
    lo, hi = reflux_band(prob, tp)
    assert lo is not None and 0.1 < lo < 30.0, lo
    assert hi is None, "a simple column's reflux band has no upper edge"
    assert not a["feasible"] or lo <= 4.0
    assert _feasible_at(prob, tp, lo * 1.5, None), "above the edge must pass"
    assert not _feasible_at(prob, tp, lo * 0.5, None), "below it must fail"

    # the body gap is a real distance: it shrinks as reflux approaches r_min
    far = analyse(prob, tp, r=lo * 0.5)["max_gap"]
    near = analyse(prob, tp, r=lo * 0.95)["max_gap"]
    assert near < far, (near, far)
    print(f"rbm.driver self-check OK  r_min={lo:.3f} (r_max={hi})  "
          f"gap {far:.3g} -> {near:.3g} approaching it")


if __name__ == "__main__":
    _demo()
