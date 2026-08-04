"""Difference-point chain (blueprint Sec 4) -- the single structural primitive.

The column is cut at every net-flow discontinuity (each feed, draw, entrainer).
Between cuts is one section with a constant net molar flow Delta_k and a
difference point delta_k. Every operating line is the one statement that the two
passing streams and delta_k are collinear:

    y_{n+1} = (L/V) x_n + (Delta/V) delta            (march down, Delta>0)

Crossing a stream f of *signed* flow F (feed F>0, draw F<0) jumps the point by a
lever rule:

    Delta' = Delta - F ,   Delta' delta' = Delta delta - F f .

A `Section` bundles the marching coefficients so the hot loop touches no dicts:
`a = L/V`, `bvec = (Delta/V) delta`, and `dir = sign(Delta)` (the *stable* march
direction, Sec 5.3 -- which end pinches, not which curve you get: the down-map
dew(a x + b) and the up-map (bubble(x) - b)/a are exact inverses). Flows L, V
come from CMO with per-feed thermal quality q.

Every section also carries a **feasible region** `{x : a x + bvec >= 0}` -- see
`feasible_margin`. Compositions outside it cannot occur on any stage of that
section, and marching from one is meaningless.
"""

from collections import namedtuple

import numpy as np
from scipy.optimize import linprog

# name : label; Delta,delta : net flow + difference point; L,V : section flows;
# a,bvec : op-line y = a x + bvec (down); dir : +1 march down / -1 march up.
Section = namedtuple("Section", "name Delta delta L V a bvec dir")


def _mk(name, Delta, delta, L, V):
    a = L / V
    bvec = (Delta / V) * delta
    return Section(name, float(Delta), np.asarray(delta, float), float(L),
                   float(V), float(a), np.asarray(bvec, float),
                   1 if Delta > 0 else -1)


def op_vapour(sec, x):
    """The vapour the operating line puts above liquid x: y = a x + bvec.

    Exact, unclipped. Since V = L + Delta the row sums to 1 whenever x does, so
    the only way this leaves the simplex is a *negative* component.
    """
    return sec.a * np.asarray(x, float) + sec.bvec


def feasible_margin(sec, x):
    """How far liquid x is inside the section's feasible region (min y component).

    A section can only carry liquid compositions whose operating-line vapour is
    non-negative:

        Reg(sec) = { x : a x + bvec >= 0 }

    Outside it the section's own material balance cannot close -- the net flow
    Delta*delta it must export exceeds what the passing liquid supplies -- so no
    stage of that section can hold such a liquid. This is a hard constraint, not
    a numerical nicety: a heavy entrainer section has Reg = {x_E-rich corner},
    e.g. x_EG >= 0.32 for ethanol/water/EG at R=1.5, E/F=0.6, which excludes the
    whole rectifying profile and all but the reboiler end of the stripping one.

    Negative return = infeasible by that margin.
    """
    return float(np.min(op_vapour(sec, x)))


def feasible(sec, x, tol=1e-9):
    """True when liquid x lies in the section's feasible region (Sec 4)."""
    return feasible_margin(sec, x) >= -tol


def region_center(sec):
    """Deepest point of the section's feasible region, or None if it is empty.

    Reg(sec) = {x : sum x = 1, x >= 0, a x + bvec >= 0} is a polytope, so the
    Chebyshev-style centre -- maximise the slack r with x >= r and a x + bvec >= r
    -- is a small LP. This is the one seed for pinch solves and manifold launches
    that does not presuppose where the answer is; the old `0.5*(x_D + x_B)` guess
    lies outside the region for every extractive section, which is why the saddle
    machinery never took hold.

    A non-positive optimum means the section cannot exist at this operating point
    at all (no composition satisfies its balance) -- returns None so the caller
    reports that rather than solving in an empty set.
    """
    C = sec.delta.shape[0]
    I = np.eye(C)
    # variables [x_0..x_{C-1}, r]; maximise r
    A_ub = np.vstack([np.hstack([-I, np.ones((C, 1))]),          # r - x_i <= 0
                      np.hstack([-sec.a * I, np.ones((C, 1))])])  # r - a x_i <= b_i
    b_ub = np.concatenate([np.zeros(C), sec.bvec])
    res = linprog(c=np.concatenate([np.zeros(C), [-1.0]]),
                  A_ub=A_ub, b_ub=b_ub,
                  A_eq=np.concatenate([np.ones(C), [0.0]])[None, :], b_eq=[1.0],
                  bounds=[(0.0, 1.0)] * C + [(None, None)])
    if not res.success or res.x[-1] <= 1e-12:
        return None
    x = np.clip(res.x[:C], 0.0, None)
    return x / x.sum()


def rectifying(R, D, xD):
    """Top section: Delta = D, delta = x_D, L = R D, V = (R+1) D."""
    return _mk("rectifying", D, xD, R * D, (R + 1.0) * D)


def cross(sec, F, f, q=1.0, phase="feed"):
    """Section just below a crossed stream. F signed (feed +, draw -).

    Flows update under CMO: a feed adds q F to the downflowing liquid and
    (1-q) F to the upflowing vapour (so V below = V above - (1-q) F). A liquid
    draw removes W from L; a vapour draw removes W from V.
    """
    Delta = sec.Delta - F
    Dd = sec.Delta * sec.delta - F * np.asarray(f, float)
    delta = Dd / Delta if abs(Delta) > 1e-30 else Dd  # Delta~0 -> point at infinity
    if phase == "feed":                 # F>0
        L = sec.L + q * F
        V = sec.V - (1.0 - q) * F
    elif phase == "L":                  # liquid side draw, F = -W
        L = sec.L + F
        V = sec.V
    elif phase == "V":                  # vapour side draw, F = -W
        L = sec.L
        V = sec.V + F
    else:
        raise ValueError(f"unknown cut phase {phase!r}")
    return _mk(phase if phase != "feed" else "section", Delta, delta, L, V)


def single_feed_chain(prob, R, xD, xB, D, B):
    """Standard two-section column: [rectifying, stripping] around one feed."""
    fd = prob.feeds[0]
    rect = rectifying(R, D, xD)
    strip = cross(rect, F=fd.F, f=fd.z, q=fd.q, phase="feed")
    return [rect._replace(name="rectifying"), strip._replace(name="stripping")]


def extractive_chain(prob, R, EF, xD, xB, D, B):
    """Entrainer above the main feed: [rectifying, extractive, stripping].

    Heavy entrainer E = EF * F_main enters at x_E with thermal quality `prob.q_E`,
    creating the strongly-pinched extractive section between it and the main feed
    (Sec 14). q_E defaults to 1.0 (saturated liquid at the tray = CMO); see
    `entrainer_q` for the energy-balanced value, and `prob.q_E_fn` for the opt-in
    that computes it here, at THIS (R, EF), rather than freezing one number.
    """
    fd = prob.feeds[0]
    E = EF * fd.F
    rect = rectifying(R, D, xD)
    q_E = prob.q_E if prob.q_E_fn is None else prob.q_E_fn(R, EF, xD, D)
    ext = cross(rect, F=E, f=prob.x_E, q=q_E, phase="feed")
    strip = cross(ext, F=fd.F, f=fd.z, q=fd.q, phase="feed")
    return [rect._replace(name="rectifying"), ext._replace(name="extractive"),
            strip._replace(name="stripping")]


def entrainer_q(prob, tp, P, R, EF, xD, D, hL, hV, T_E, x_tray=None, sweeps=2):
    """Thermal quality of the entrainer feed from the section ENERGY balance.

    CMO carries V straight through a saturated-liquid feed, so `extractive_chain`
    with q_E = 1 gives the extractive section the rectifying section's vapour
    flow. That is wrong whenever the entrainer arrives at a temperature the tray
    is nowhere near, which for a heavy entrainer is always: pure glycol at its own
    bubble point is 197 C entering a section running at 95 C. The extra enthalpy
    boils liquid at the feed tray, so the vapour arriving from BELOW is smaller
    than the vapour leaving above -- an effective q < 1, which is what this
    returns. Brueggemann & Marquardt carry this as the enthalpy balance in their
    pinch equations (their eqs. 6 and 9); we carry it as one number, because a
    section here has one L and one V by construction.

    The envelope is condenser -> entrainer feed -> a tray in the extractive
    section:

        V [h_V(y,T) - h_L(x,T)] = D h_D + Q_c - E h_E - (D - E) h_L(x,T)

    and q = 1 - (V_rect - V)/E. `hL`/`hV` are the per-component closures
    `core.enthalpy.enthalpy_fns` returns (T in K -> J/mol vectors); `T_E` is the
    entrainer's feed temperature in K.

    `x_tray` is the extractive liquid the balance is written at, and it matters:
    the deepest point of the section's feasible region (`region_center`, the
    default -- `pinch` imports this module, so this module cannot ask for a pinch)
    gives V = 164.8 on ipa/water/EG at r = 2.042, E/F = 0.750, where the section's
    own pinch gives 158.8. Pass the pinch if you have it. Either way it is ~40% of
    the way from CMO's 188.4 to the ~126 the paper's saddle position implies; the
    rest is excess enthalpy (`core.enthalpy` mixes ideally) and their Aspen data.

    ponytail: two fixed-point sweeps, not a stage-by-stage non-CMO section. The
    second sweep moves V by 1% on ipa/water/EG and the third by 0.06%; upgrade if
    a case makes it move more.
    """
    E = float(EF) * prob.feeds[0].F
    if E <= 0.0:
        return 1.0
    xE = np.asarray(prob.x_E, float)
    T_D = tp.bubble(xD, P)[1] + 273.15
    V_rect = (R + 1.0) * D
    Qc = V_rect * float(xD @ (hV(T_D) - hL(T_D)))       # total condenser
    rhs0 = D * float(xD @ hL(T_D)) + Qc - E * float(xE @ hL(float(T_E)))
    N = D - E
    q = 1.0
    for _ in range(max(1, int(sweeps))):
        ext = cross(rectifying(R, D, xD), F=E, f=xE, q=q, phase="feed")
        x = region_center(ext) if x_tray is None else np.asarray(x_tray, float)
        if x is None:                                   # section cannot exist here
            return q
        y, T_C = tp.bubble(x, P)
        T = T_C + 273.15
        h_l = float(x @ hL(T))
        lam = float(y @ hV(T)) - h_l
        if not np.isfinite(lam) or lam <= 0.0:
            return q
        q = 1.0 - (V_rect - (rhs0 - N * h_l) / lam) / E
    return float(q)


def entrainer_q_fn(prob, tp, P, hL, hV, T_E, x_tray=None, sweeps=2):
    """`Problem.q_E_fn` closure: `entrainer_q` re-evaluated at each (R, EF).

    The opt-in seam. A reflux band or an operating region walks dozens of
    operating points, and the entrainer's flash is not the same at all of them
    (it scales with the vapour it has to boil), so freezing one q_E at the panel's
    nominal point would be a different kind of wrong from CMO. Attach this and
    every caller of `extractive_chain` -- BVM's marching, RBM's pinch equations,
    both drivers' scans -- gets the balance for the point it is actually asking
    about, with no signature change.

    ponytail: memoised on (R, EF) alone. xD/D are functions of EF through
    `overall_balance`, and a sweep re-asks the same point many times.
    """
    cache = {}

    def q_of(R, EF, xD, D):
        key = (round(float(R), 9), round(float(EF), 9))
        if key not in cache:
            cache[key] = entrainer_q(prob, tp, P, R, EF, xD, D, hL, hV, T_E,
                                     x_tray=x_tray, sweeps=sweeps)
        return cache[key]

    return q_of


def multifeed_chain(prob, R, xD, xB, D, B):
    """One rectifying, one intermediate per interior feed gap, one stripping.

    Feeds are taken top -> bottom in `prob.feeds` order. With two feeds this is
    [rectifying, intermediate, stripping]; the intermediate section is marched by
    continuation (Sec 6.2).
    """
    secs = [rectifying(R, D, xD)._replace(name="rectifying")]
    sec = secs[0]
    for i, fd in enumerate(prob.feeds):
        sec = cross(sec, F=fd.F, f=fd.z, q=fd.q, phase="feed")
        name = "stripping" if i == len(prob.feeds) - 1 else "intermediate"
        sec = sec._replace(name=name)
        secs.append(sec)
    return secs


def _demo():
    from .problem import build_problem, overall_balance
    z = np.array([0.4, 0.35, 0.25])
    prob = build_problem(["b", "t", "x"], [(z, 100.0, 1.0)], 760.0,
                         rec_lk=0.98, rec_hk=0.02)
    xD, xB, D, B = overall_balance(prob)
    rect, strip = single_feed_chain(prob, 3.0, xD, xB, D, B)

    # rectifying difference point IS the distillate; net flow is D
    assert np.allclose(rect.delta, xD) and abs(rect.Delta - D) < 1e-9
    assert rect.dir == 1, "rectifying marches down"
    # lever rule reproduces the stripping row of the catalogue: Delta=-B, delta=xB
    assert abs(strip.Delta + B) < 1e-9, strip.Delta
    assert np.allclose(strip.delta, xB, atol=1e-9), (strip.delta, xB)
    assert strip.dir == -1, "stripping marches up"
    # collinearity of (delta_rect, feed z, delta_strip): the two spanning
    # vectors are parallel, so the 2xC matrix is rank 1.
    v1 = z - rect.delta; v2 = strip.delta - z
    assert np.linalg.matrix_rank(np.array([v1, v2]), tol=1e-9) == 1, "not collinear"
    # rectifying reduction: y = R/(R+1) x + 1/(R+1) xD
    R = 3.0
    x = np.array([0.6, 0.3, 0.1])
    y = rect.a * x + rect.bvec
    assert np.allclose(y, R / (R + 1) * x + xD / (R + 1)), "op-line reduction"

    # extractive chain: three sections, entrainer point may sit outside simplex
    pe = build_problem(["b", "t", "x"], [(z, 100.0)], 760.0,
                       x_E=np.array([0.0, 0.0, 1.0]), extractive=True)
    xDe, xBe, De, Be = overall_balance(pe)
    ch = extractive_chain(pe, 3.0, 0.5, xDe, xBe, De, Be)
    assert [s.name for s in ch] == ["rectifying", "extractive", "stripping"]

    # q_E is the entrainer's thermal quality, exactly as `Feed.q` is a feed's:
    # the default 1.0 is CMO (V carried straight through), and only a value away
    # from 1 moves V. Checked against the balance V_below = V_above - (1-q) E,
    # E = 0.5 * 100.
    assert abs(ch[1].V - ch[0].V) < 1e-9, "q_E = 1 must not touch V"
    pe.q_E = 0.6
    flashed = extractive_chain(pe, 3.0, 0.5, xDe, xBe, De, Be)[1]
    assert abs(flashed.V - (ch[0].V - 0.4 * 50.0)) < 1e-9, flashed.V
    assert flashed.V < ch[1].V and flashed.L < ch[1].L
    assert abs(flashed.Delta - ch[1].Delta) < 1e-12, "q moves flows, never Delta"
    pe.q_E = 1.0

    # q_E_fn overrides the constant and is asked at THIS (R, EF); memoised, so
    # one call per operating point no matter how often the chain is rebuilt.
    calls = []
    pe.q_E_fn = lambda R_, EF_, xD_, D_: (calls.append((R_, EF_)) or 0.6)
    assert abs(extractive_chain(pe, 3.0, 0.5, xDe, xBe, De, Be)[1].V
               - flashed.V) < 1e-9, "q_E_fn must feed the same flow balance"
    assert calls == [(3.0, 0.5)], calls
    pe.q_E_fn = None

    # operating-line vapour sums to 1 for any liquid that does (V = L + Delta)
    for sec in (rect, strip, ch[1]):
        assert abs(op_vapour(sec, x).sum() - 1.0) < 1e-12, sec.name
    # a product-anchored section always contains its own product composition
    assert feasible(rect, xD) and feasible(strip, xB)
    # the extractive section does NOT contain the distillate: a pure heavy
    # entrainer forces a minimum entrainer content on every one of its stages
    ext = ch[1]
    assert not feasible(ext, xDe), "extractive region should exclude x_D"
    # the margin is affine in x, so it changes sign exactly once along any segment
    # that leaves the region: walk x_D -> x_E and find the entrainer content where
    # the extractive section becomes able to hold the liquid at all.
    ts = np.linspace(0.0, 1.0, 501)
    marg = [feasible_margin(ext, (1 - t) * xDe + t * pe.x_E) for t in ts]
    assert marg[0] < 0 <= marg[-1], (marg[0], marg[-1])
    assert sum(np.diff(np.sign(marg)) != 0) == 1, "margin should cross zero once"
    t_min = ts[int(np.argmax(np.array(marg) >= 0))]
    print(f"sections self-check OK  Delta: "
          f"{[round(s.Delta, 2) for s in [rect, strip]]}  "
          f"extractive feasible from {t_min:.2f} of the way x_D -> x_E")


if __name__ == "__main__":
    _demo()
