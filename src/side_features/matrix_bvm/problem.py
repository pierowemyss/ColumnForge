"""Problem definition + overall balance (blueprint Sec 2, Sec 18.1).

A `Problem` is plain data describing the separation: components, feed(s), the
optional entrainer, side draws, pressure, and the product specification (two
keys + non-key distribution, or explicit x_D/x_B). `overall_balance` turns the
spec into the four numbers every difference point needs -- x_D, x_B, D, B --
from a straight component material balance. Nothing here marches or solves; it
is the boundary data for the difference-point chain (Sec 4).

Convention: components are listed light -> heavy (decreasing volatility). The
light key `lk` and heavy key `hk` index into that list with lk < hk. Non-keys
default light-of-LK -> distillate, heavy-of-HK -> bottoms; override per
component with `nonkey_to_dist`.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Feed:
    z: np.ndarray          # composition (C,), sums to 1
    F: float               # molar flow
    q: float = 1.0         # thermal quality: 1 sat-liquid, 0 sat-vapour


@dataclass
class SideDraw:
    W: float               # draw molar flow
    phase: str = "L"       # 'L' liquid draw (x_n) or 'V' vapour draw (y_n)
    comp_index: int = 0    # component the purity target refers to
    purity: float = 0.9    # target mole fraction of comp_index in the draw


@dataclass
class Problem:
    comps: list
    feeds: list                       # list[Feed]
    pressure: float                   # top pressure, in the Antoine/Psat unit
    lk: int = 0                       # light-key index
    hk: int = 1                       # heavy-key index
    rec_lk: float = 0.98              # fraction of LK recovered to distillate
    rec_hk: float = 0.02              # fraction of HK recovered to distillate
    nonkey_to_dist: np.ndarray = None  # per-comp fraction to distillate (non-keys)
    xD: np.ndarray = None             # explicit distillate comp (overrides recoveries)
    xB: np.ndarray = None             # explicit bottoms comp
    x_E: np.ndarray = None            # entrainer composition (extractive mode)
    extractive: bool = False
    side_draws: list = field(default_factory=list)   # list[SideDraw]
    dP: float = 0.0                   # per-stage pressure drop (Psat unit)
    reactions: object = None          # Reactions (reactive.py) or None
    max_stages: int = 200             # per-section marching cap
    efficiency: float = 1.0           # Murphree vapour efficiency (1 = ideal stages)

    @property
    def C(self):
        return len(self.comps)

    @property
    def z_total(self):
        """Overall feed composition (flow-weighted, unnormalised) x total flow."""
        f = np.zeros(self.C)
        for fd in self.feeds:
            f = f + fd.F * np.asarray(fd.z, float)
        return f

    @property
    def F_total(self):
        return sum(fd.F for fd in self.feeds)


def overall_balance(prob, EF=None):
    """Spec + feed(s) -> (xD, xB, D, B). Straight component balance f = D xD + B xB.

    The balance runs over *every* feed: the main feed(s) in `prob.feeds` and, in
    extractive mode, the entrainer stream E = EF*F_main at x_E. Excluding the
    entrainer traps the rectifying march on the entrainer=0 face (it carries zero
    entrainer to the distillate) and gives the bottoms the wrong anchor, so both
    the total flow F and the pooled feed f include it (blueprint Sec 14, 18.1).

    Explicit xD & xB win (D solved by least squares on f = D xD + B xB, D+B=F).
    Otherwise a per-component split-to-distillate fraction is built from the key
    recoveries and the non-key distribution, and d = frac*f, b = f - d. The trace
    floor on `frac` then keeps every component present anywhere in the column at
    >=1e-4 of its feed amount in each product, so the profiles can leave a face.
    """
    C = prob.C
    f = prob.z_total.copy()
    F = prob.F_total
    if prob.extractive and prob.x_E is not None and EF:
        E = float(EF) * prob.feeds[0].F
        f = f + E * np.asarray(prob.x_E, float)
        F = F + E

    if prob.xD is not None and prob.xB is not None:
        xD = np.asarray(prob.xD, float); xD = xD / xD.sum()
        xB = np.asarray(prob.xB, float); xB = xB / xB.sum()
        # f = D xD + (F-D) xB  ->  (xD-xB) D = f - F xB ; least-squares D
        a = xD - xB
        D = float(a @ (f - F * xB) / (a @ a))
        D = min(max(D, 1e-9), F - 1e-9)
        return xD, xB, D, F - D

    # recovery-based split
    if prob.nonkey_to_dist is not None:
        frac = np.asarray(prob.nonkey_to_dist, float).copy()
    else:
        # light-of-LK -> distillate, heavy-of-HK -> bottoms, in-between split evenly
        frac = np.where(np.arange(C) < prob.lk, 1.0,
                        np.where(np.arange(C) > prob.hk, 0.0, 0.5))
    frac[prob.lk] = prob.rec_lk
    frac[prob.hk] = prob.rec_hk

    # A strictly non-distributing component (frac 0 or 1) traps the profile on a
    # simplex face: heavies amplify downward in the rectifying section, so with
    # exactly zero distillate they never appear and can't reach the feed. Keep a
    # trace in each product (1e-4) so the profile can bend off the face -- the
    # physical reality (nothing is *perfectly* non-distributing) and standard BVM
    # practice. ponytail: fixed floor; expose per-comp if a case needs it.
    frac = np.clip(frac, 1e-4, 1.0 - 1e-4)

    d = frac * f
    b = f - d
    D = float(d.sum()); B = float(b.sum())
    if D <= 0 or B <= 0:
        raise ValueError(f"degenerate split: D={D:.3g}, B={B:.3g} -- check keys/recoveries")
    return d / D, b / B, D, B


def build_problem(comps, feeds, pressure, lk=0, hk=1, **kw):
    """Convenience constructor. feeds: list of Feed or (z, F[, q]) tuples."""
    fs = []
    for fd in feeds:
        if isinstance(fd, Feed):
            fs.append(fd)
        else:
            z, Fv = fd[0], fd[1]
            q = fd[2] if len(fd) > 2 else 1.0
            fs.append(Feed(z=np.asarray(z, float), F=float(Fv), q=float(q)))
    return Problem(comps=list(comps), feeds=fs, pressure=float(pressure),
                   lk=lk, hk=hk, **kw)


def _demo():
    comps = ["benzene", "toluene", "xylene"]
    z = np.array([0.4, 0.35, 0.25])
    prob = build_problem(comps, [(z, 100.0, 1.0)], pressure=760.0, lk=0, hk=1,
                         rec_lk=0.98, rec_hk=0.02)
    xD, xB, D, B = overall_balance(prob)
    assert abs(D + B - 100.0) < 1e-9, "D+B must equal total feed"
    # LK concentrates in distillate, HK in bottoms
    assert xD[0] > z[0] > xB[0], (xD, xB)
    assert xB[1] > z[1] > xD[1], (xD, xB)
    # component balance closes: D xD + B xB == f
    assert np.allclose(D * xD + B * xB, 100.0 * z, atol=1e-9)
    # xylene (heavy non-key) went essentially all to bottoms (trace floor in xD)
    assert xD[2] < 1e-3 and xB[2] > z[2]

    # explicit xD/xB path
    p2 = build_problem(comps, [(z, 100.0)], pressure=760.0,
                       xD=np.array([0.8, 0.2, 0.0]), xB=np.array([0.05, 0.45, 0.5]))
    xD2, xB2, D2, B2 = overall_balance(p2)
    assert abs(D2 + B2 - 100.0) < 1e-9
    print("problem self-check OK", np.round(xD, 3), np.round(xB, 3), round(D, 2))


if __name__ == "__main__":
    _demo()
