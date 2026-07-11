"""Profile marching (blueprint Sec 5).

One section, one anchor composition, marched in its *stable* direction (Sec 5.3,
set by sign(Delta)): the map is contracting that way and blows up the other way.

    Delta > 0  (rectifying-like): march DOWN from the top anchor.
        x_n --op line--> y_{n+1} = a x_n + bvec ,  x_{n+1} = dew_liquid(y_{n+1})
    Delta < 0  (stripping-like):  march UP from the bottom anchor.
        x_n --bubble--> y_n ,  x_{n-1} = (y_n - bvec)/a   (operating line inverted)

Each step calls the shared thermo for K/T (Sec 17). Marching stops at a pinch
(step below `eps_pinch`), on leaving the simplex, or at `max_stages`. Profiles
are full C-vectors; the LK/HK projection is only for reading direction (Sec 13).
"""

import numpy as np
from scipy.optimize import fsolve


def _clean(x):
    """Clip tiny numerical negatives and renormalise; report gross excursions."""
    xc = np.where(x < 0, 0.0, x)
    s = xc.sum()
    return (xc / s if s > 0 else x), float(x.min())


def _K_stage(tp, x, T, P):
    """K-vector at one stage through the provider interface (stage-major API)."""
    return tp.K(np.atleast_2d(x), np.atleast_1d(T), np.atleast_1d(P))[0]


def _dew_eff(sec, y, tp, P, E):
    """Marching-down conjugate liquid with Murphree vapour efficiency E.

    The actual vapour leaving the stage is y = E*K(x)*x + (1-E)*op(x) with
    op(x)=a*x+bvec the vapour entering from below (blueprint E4). Given y, solve
    for the stage liquid x; E=1 collapses to the equilibrium dew point (condenser
    convention). Stage T is the bubble point of that liquid.

    ponytail: fsolve per stage (the direct fixed-point iteration diverges for
    E<~0.6). Only taken on the efficiency<1 path; the E=1 dew step stays fast.
    """
    x0, T0 = tp.dew(y, P)               # equilibrium guess (also the E=1 answer)
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


def march_section(sec, x0, tp, P, max_stages=200, eps_pinch=1e-6, efficiency=1.0):
    """March one section from anchor x0. Returns dict(X, Y, T, status, pinched).

    X[k] is the liquid on the k-th stage from the anchor (X[0] = x0). Y and T are
    the conjugate vapour and stage temperature. `status` in {pinch, max, simplex}.

    `efficiency` is the Murphree vapour efficiency E in (0,1]; the anchor stage
    (condenser marching down / reboiler marching up) stays an equilibrium stage,
    so N_total is directly comparable to the MESH solvers' `_murphree_keff`. E=1
    reproduces the ideal-stage march bit-for-bit.
    """
    E = float(efficiency)
    X = [np.asarray(x0, float) / np.sum(x0)]
    Y = []
    T = []
    status = "max"
    pinched = False
    down = sec.dir > 0
    y_below = None                     # actual vapour from the stage below (up-march)

    for _ in range(max_stages):
        x = X[-1]
        try:
            if down:
                y = sec.a * x + sec.bvec
                y = np.clip(y, 0.0, None)
                s = y.sum()
                if s <= 0:
                    status = "simplex"; break
                y = y / s
                xn, Tn = _dew_eff(sec, y, tp, P, E)
            else:
                y_eq, Tn = tp.bubble(x, P)
                if E < 1.0 and y_below is not None:
                    y = E * y_eq + (1.0 - E) * y_below
                    y = np.clip(y, 0.0, None); y = y / y.sum()
                else:
                    y = y_eq           # reboiler / anchor stage is equilibrium
                xn = (y - sec.bvec) / sec.a
                y_below = y
        except (ValueError, FloatingPointError):
            # thermo has no saturation root here -- the march ran off the
            # physical region (heavy-corner blow-up); stop as if it left.
            status = "simplex"; break
        Y.append(y); T.append(Tn)

        xn, xmin = _clean(xn)
        if xmin < -1e-3:                       # left the physical region
            X.append(xn); status = "simplex"; break
        step = float(np.linalg.norm(xn - x))
        X.append(xn)
        if step < eps_pinch:                   # fixed point reached
            status = "pinch"; pinched = True; break

    # pad Y/T so lengths line up with X (last stage has no forward step recorded)
    if len(Y) < len(X):
        try:
            if down:
                yl = sec.a * X[-1] + sec.bvec; yl = np.clip(yl, 0, None); yl /= yl.sum()
                _, Tl = tp.dew(yl, P)
            else:
                yl, Tl = tp.bubble(X[-1], P)
        except (ValueError, FloatingPointError):
            yl = Y[-1] if Y else X[-1]
            Tl = T[-1] if T else 0.0
        Y.append(yl); T.append(Tl)

    return {"X": np.array(X), "Y": np.array(Y), "T": np.array(T),
            "status": status, "pinched": pinched, "n": len(X)}


def _demo():
    from thermo_adapter import FreeColumnThermo
    from problem import build_problem, overall_balance
    from sections import single_feed_chain

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = FreeColumnThermo(abc)
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
    print(f"march self-check OK  rect {r['status']}/{r['n']}  strip {s['status']}/{s['n']}")


if __name__ == "__main__":
    _demo()
