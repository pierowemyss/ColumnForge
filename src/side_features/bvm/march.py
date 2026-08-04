"""Profile marching (blueprint Sec 5).

One section, one anchor composition, marched in its *stable* direction (Sec 5.3,
set by sign(Delta)): the map is contracting that way and blows up the other way.

    Delta > 0  (rectifying-like): march DOWN from the top anchor.
        x_n --op line--> y_{n+1} = a x_n + bvec ,  x_{n+1} = dew_liquid(y_{n+1})
    Delta < 0  (stripping-like):  march UP from the bottom anchor.
        x_n --bubble--> y_n ,  x_{n-1} = (y_n - bvec)/a   (operating line inverted)

The two maps are exact inverses (down = dew o op, up = op^-1 o bubble), so the
direction does not change which curve you trace -- only which end of it pinches
and truncates. `sec.dir` is therefore a numerical-conditioning choice, and for a
product-anchored section the anchor already fixes it.

Each step calls the shared thermo for K/T (Sec 17). Marching stops at a pinch
(step below `eps_pinch`), on leaving the simplex, on leaving the section's
feasible region (`sections.feasible_margin`), or at `max_stages`. Profiles are
full C-vectors; the LK/HK projection is only for reading direction (Sec 13).
"""

import numpy as np
from scipy.optimize import fsolve

from .sections import feasible

# Roundoff slack on the operating-line non-negativity test. Real violations are
# O(0.1-1) in mole fraction (a heavy-entrainer section demands x_E >= 0.3 or the
# balance goes negative), so this only has to absorb float noise.
_OP_TOL = 1e-9


def _clean(x):
    """Clip tiny numerical negatives and renormalise; report gross excursions."""
    xc = np.where(x < 0, 0.0, x)
    s = xc.sum()
    return (xc / s if s > 0 else x), float(x.min())


def _K_stage(tp, x, T, P):
    """K-vector at one stage through the provider interface (stage-major API)."""
    return tp.K(np.atleast_2d(x), np.atleast_1d(T), np.atleast_1d(P))[0]


def _dew_eff(sec, y, tp, P, E, x_seed=None):
    """Marching-down conjugate liquid with Murphree vapour efficiency E.

    The actual vapour leaving the stage is y = E*K(x)*x + (1-E)*op(x) with
    op(x)=a*x+bvec the vapour entering from below (blueprint E4). Given y, solve
    for the stage liquid x; E=1 collapses to the equilibrium dew point (condenser
    convention). Stage T is the bubble point of that liquid.

    ponytail: fsolve per stage (the direct fixed-point iteration diverges for
    E<~0.6). Only taken on the efficiency<1 path; the E=1 dew step stays fast.
    """
    # the stage above is the continuation seed: it is the nearest point on the
    # physical branch, which is what keeps the gamma(x) dew off the spurious root.
    x0, T0 = tp.dew(y, P, x_seed)       # equilibrium guess (also the E=1 answer)
    if E >= 1.0:
        return x0, T0
    C = y.shape[0]

    def _bt(xx):                        # bubble_T can diverge near a pure heavy
        try:
            return tp.bubble_T(xx, P)
        except Exception:
            return T0

    def resid(u):
        xx = np.empty(C); xx[:C - 1] = u; xx[C - 1] = 1.0 - u.sum()
        xx = np.clip(xx, 1e-12, None); xx = xx / xx.sum()
        K = _K_stage(tp, xx, _bt(xx), P)
        yy = E * K * xx + (1.0 - E) * (sec.a * xx + sec.bvec)
        yy = np.clip(yy, 1e-12, None); yy = yy / yy.sum()
        return (yy - y)[:C - 1]

    try:
        u = fsolve(resid, x0[:C - 1], xtol=1e-10)
    except Exception:
        return x0, T0                   # degrade to an equilibrium step
    x = np.empty(C); x[:C - 1] = u; x[C - 1] = 1.0 - u.sum()
    x = np.clip(x, 0.0, None); x = x / x.sum()
    return x, _bt(x)


def march_section(sec, x0, tp, P, max_stages=200, eps_pinch=1e-6, efficiency=1.0,
                  dP=0.0, P_lim=None, stop_sec=None):
    """March one section from anchor x0. Returns dict(X, Y, T, P, status, pinched).

    X[k] is the liquid on the k-th stage from the anchor (X[0] = x0). Y and T are
    the conjugate vapour and stage temperature. `status` in
    {pinch, max, simplex, operating_line, crossed}; `operating_line` on the anchor
    stage itself (n == 1) means x0 was never a composition this section could hold.

    `stop_sec` is the section on the far side of the next feed. Marching stops the
    first stage that lands inside ITS feasible region (`sections.feasible`), with
    status 'crossed'. Everything past that point is fiction: the profile has run
    through the junction, and for a heavy entrainer it then blows up (x_E grows
    ~800x per stage), which `connect`'s all-pairs segment search would happily
    match against. Stopping there is the same hard balance constraint the region
    itself expresses, not a tolerance.

    It only ARMS when the anchor starts outside that region -- otherwise the test
    carries no information and would stop the march instantly. A stripping section
    anchored at an entrainer-rich reboiler is already inside the extractive
    section's region (x_E >= E/L) on stage 0, so for that end there is nothing to
    cross and the profile runs to its own pinch as before.

    `P` is the pressure ON THE ANCHOR STAGE and `dP >= 0` the per-stage pressure
    drop down the column, so the k-th stage sits at `P + k*dP` marching down and
    `P - k*dP` marching up (an up-march is anchored at the reboiler, the
    high-pressure end). `P` is per-stage from here on: `prof["P"][k]` is what the
    stage was evaluated at, which is what `connect` needs to boil the lower
    profile at its own pressure rather than the top one.

    `P_lim` is the pressure at the column's OTHER end and clamps the ramp there.
    It is not a nicety: a march runs to `max_stages`, which is several times the
    column, so an unclamped ramp walks a stripping section 200 stages up and down
    to 4 mmHg -- the pinch disappears and the profile is fiction. The column is
    only as long as it is.

    `efficiency` is the Murphree vapour efficiency E in (0,1]; the anchor stage
    (condenser marching down / reboiler marching up) stays an equilibrium stage,
    so N_total is directly comparable to the MESH solvers' `_murphree_keff`. E=1
    reproduces the ideal-stage march bit-for-bit.
    """
    E = float(efficiency)
    X = [np.asarray(x0, float) / np.sum(x0)]
    if stop_sec is not None and feasible(stop_sec, X[0]):
        stop_sec = None                # anchor already inside: nothing to cross
    Y = []
    T = []
    status = "max"
    pinched = False
    down = sec.dir > 0
    y_below = None                     # actual vapour from the stage below (up-march)
    dPs = float(dP) * (1.0 if down else -1.0)
    if P_lim is None:                  # no far end given: let the ramp run
        P_lo, P_hi = 1e-9, np.inf
    else:
        P_lo, P_hi = sorted((float(P), float(P_lim)))
    Ps = [float(P)]

    # Marching DOWN, `_dew_eff` returns the next stage's liquid *and its* bubble
    # point, so the temperature it hands back describes X[k+1], not X[k]. Stage
    # k's own temperature is the one the previous pass produced, so carry it
    # forward and seed it from the anchor. Appending Tn directly put the whole
    # profile one stage out -- invisible in X and Y, and it survived because both
    # consumers read T without ever comparing it to the liquid beside it:
    # `driver._temperature_inversion`, which judges a design on the size of the
    # steps in this array, and `handoff`'s T0, which hands it to MESH as the
    # warm-start temperature for stages it does not describe. The tell was the
    # last two entries coming out bit-identical, because the padding block below
    # wrote the correct final temperature into an otherwise shifted array.
    # The UP branch computes T from x itself and was always aligned.
    T_cur = None
    if down:
        try:
            T_cur = float(tp.bubble_T(X[0], float(P)))
        except (ValueError, FloatingPointError):
            T_cur = None                   # filled from the first step below

    for _ in range(max_stages):
        x = X[-1]
        Pk = Ps[-1]
        try:
            if down:
                # NO clipping here. y = a x + bvec sums to 1 exactly (V = L+Delta),
                # so a negative component means x is outside the section's feasible
                # region -- the balance cannot close, and clipping+renormalising
                # would invent a stage that cannot exist. Stop and say so.
                y = sec.a * x + sec.bvec
                if y.min() < -_OP_TOL:
                    status = "operating_line"; break
                y = np.clip(y, 0.0, None)
                xn, Tn = _dew_eff(sec, y, tp, Pk, E, x_seed=x)
            else:
                y_eq, Tn = tp.bubble(x, Pk)
                if E < 1.0 and y_below is not None:
                    y = E * y_eq + (1.0 - E) * y_below
                    y = np.clip(y, 0.0, None); y = y / y.sum()
                else:
                    y = y_eq           # reboiler / anchor stage is equilibrium
                # inverted operating line; a > 0, so a negative liquid here is the
                # same statement as above read the other way round.
                xn = (y - sec.bvec) / sec.a
                if xn.min() < -_OP_TOL:
                    status = "operating_line"; break
                y_below = y
        except (ValueError, FloatingPointError):
            # thermo has no saturation root here -- the march ran off the
            # physical region (heavy-corner blow-up); stop as if it left.
            status = "simplex"; break
        Y.append(y)
        if down:
            T.append(Tn if T_cur is None else T_cur)
            T_cur = Tn
        else:
            T.append(Tn)

        xn, xmin = _clean(xn)
        # one step further down (or up) the ramp, held inside the column's ends
        Ps.append(min(max(Pk + dPs, P_lo), P_hi))
        if xmin < -1e-3:                       # left the physical region
            X.append(xn); status = "simplex"; break
        step = float(np.linalg.norm(xn - x))
        X.append(xn)
        if step < eps_pinch:                   # fixed point reached
            status = "pinch"; pinched = True; break
        if stop_sec is not None and feasible(stop_sec, xn):
            # entered the next section's region: this stage is past the junction
            status = "crossed"; break

    # pad Y/T so lengths line up with X (last stage has no forward step recorded)
    if len(Y) < len(X):
        try:
            if status == "operating_line":
                # the last stage has no physical conjugate vapour -- that is why we
                # stopped. Don't manufacture one.
                raise ValueError
            if down:
                yl = np.clip(sec.a * X[-1] + sec.bvec, 0, None); yl /= yl.sum()
                # X[-1]'s OWN temperature is already in hand -- it is what the
                # last step computed. Taking the dew point of yl here would name
                # the stage after this one, which is the same off-by-one again.
                Tl = T_cur if T_cur is not None else tp.dew(yl, Ps[len(X) - 1], X[-1])[1]
            else:
                yl, Tl = tp.bubble(X[-1], Ps[len(X) - 1])
        except (ValueError, FloatingPointError):
            yl = Y[-1] if Y else X[-1]
            Tl = (T_cur if down and T_cur is not None
                  else (T[-1] if T else 0.0))
        Y.append(yl); T.append(Tl)

    return {"X": np.array(X), "Y": np.array(Y), "T": np.array(T),
            "P": np.array(Ps[:len(X)]),
            "status": status, "pinched": pinched, "n": len(X)}


def _demo():
    from .thermo_adapter import ColumnForgeThermo
    from .problem import build_problem, overall_balance
    from .sections import single_feed_chain

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = ColumnForgeThermo(abc)
    z = np.array([0.4, 0.35, 0.25])
    prob = build_problem(["b", "t", "x"], [(z, 100.0, 1.0)], 760.0,
                         rec_lk=0.98, rec_hk=0.02)
    xD, xB, D, B = overall_balance(prob)
    rect, strip = single_feed_chain(prob, 4.0, xD, xB, D, B)

    r = march_section(rect, xD, tp, 760.0, max_stages=60)
    s = march_section(strip, xB, tp, 760.0, max_stages=60)

    # rectifying starts at distillate and drops the light key toward the feed
    assert np.allclose(r["X"][0], xD)
    assert r["X"][-1][0] < r["X"][0][0], "light key falls marching down"
    # stripping starts at bottoms and gains the light key marching up
    assert np.allclose(s["X"][0], xB)
    assert s["X"][-1][0] > s["X"][0][0], "light key rises marching up"
    # both profiles stay in the simplex
    for prof in (r, s):
        assert prof["X"].min() > -1e-6 and prof["X"].max() < 1.0 + 1e-6
        assert np.allclose(prof["X"].sum(axis=1), 1.0, atol=1e-6)
    # temperature rises from condenser (rect top) toward the reboiler (strip top)
    assert r["T"][0] < s["T"][0], "distillate cooler than bottoms"

    # T[k] is stage k's temperature -- the bubble point of X[k], not of the stage
    # after it. The down-march used to be one out here (it recorded the dew-step
    # product's temperature), which nothing in X or Y could show.
    for prof, name in ((r, "down"), (s, "up")):
        Tb = np.array([tp.bubble_T(x, P) for x, P in zip(prof["X"], prof["P"])])
        worst = float(np.abs(Tb - prof["T"]).max())
        assert worst < 0.5, f"{name}-march T misaligned with X by up to {worst:.2f} K"

    # dP ramps the pressure the right way from each anchor, and a hotter column
    # is what a real pressure drop buys you
    rp = march_section(rect, xD, tp, 760.0, max_stages=60, dP=5.0)
    sp = march_section(strip, xB, tp, 800.0, max_stages=60, dP=5.0)
    assert rp["P"].shape == (rp["n"],) and sp["P"].shape == (sp["n"],)
    assert np.allclose(rp["P"], 760.0 + 5.0 * np.arange(rp["n"]))   # down: rises
    assert np.allclose(sp["P"], 800.0 - 5.0 * np.arange(sp["n"]))   # up: falls
    assert rp["T"][3] > r["T"][3], "higher pressure -> higher stage temperature"
    assert np.allclose(r["P"], 760.0), "dP=0 leaves the column flat"
    print(f"march self-check OK  rect {r['status']}/{r['n']}  strip {s['status']}/{s['n']}")


if __name__ == "__main__":
    _demo()
