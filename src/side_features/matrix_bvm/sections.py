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
`a = L/V`, `bvec = (Delta/V) delta`, and `dir = sign(Delta)` (the stable march
direction, Sec 5.3). Flows L, V come from CMO with per-feed thermal quality q.
"""

from collections import namedtuple

import numpy as np

# name : label; Delta,delta : net flow + difference point; L,V : section flows;
# a,bvec : op-line y = a x + bvec (down); dir : +1 march down / -1 march up.
Section = namedtuple("Section", "name Delta delta L V a bvec dir")


def _mk(name, Delta, delta, L, V):
    a = L / V
    bvec = (Delta / V) * delta
    return Section(name, float(Delta), np.asarray(delta, float), float(L),
                   float(V), float(a), np.asarray(bvec, float),
                   1 if Delta > 0 else -1)


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

    Heavy entrainer E = EF * F_main enters as saturated liquid at x_E, creating
    the strongly-pinched extractive section between it and the main feed (Sec 14).
    """
    fd = prob.feeds[0]
    E = EF * fd.F
    rect = rectifying(R, D, xD)
    ext = cross(rect, F=E, f=prob.x_E, q=1.0, phase="feed")
    strip = cross(ext, F=fd.F, f=fd.z, q=fd.q, phase="feed")
    return [rect._replace(name="rectifying"), ext._replace(name="extractive"),
            strip._replace(name="stripping")]


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
    from problem import build_problem, overall_balance
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
    print("sections self-check OK  Delta:", [round(s.Delta, 2) for s in [rect, strip]])


if __name__ == "__main__":
    _demo()
