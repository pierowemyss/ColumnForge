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
    q_E: float = 1.0                  # entrainer feed thermal quality, same meaning
                                      # as `Feed.q`. 1.0 = saturated liquid AT THE
                                      # TRAY, which is what CMO assumes and what
                                      # every result before this field was computed
                                      # with. A heavy entrainer fed at its own
                                      # bubble point is far hotter than the tray it
                                      # lands on and flashes: on ipa/water/EG at
                                      # 197 C into a 95 C section the energy balance
                                      # puts it at q = 0.61, which drops V in the
                                      # extractive section from 188 to 159 and moves
                                      # every extractive pinch from x_EG = 0.373 to
                                      # 0.442 (the paper's is 0.55). Set it from
                                      # `sections.entrainer_q`; left at 1.0 nothing
                                      # in the module changes. See docs/adr/0004.
    q_E_fn: object = None             # callable(R, EF, xD, D) -> q_E, overriding
                                      # the constant above. The energy balance is
                                      # not a constant: q_E depends on the reflux
                                      # and entrainer ratio, which every band /
                                      # region scan varies. `sections.entrainer_q_fn`
                                      # builds one; None = use `q_E` (CMO default).
    extractive: bool = False
    side_draws: list = field(default_factory=list)   # list[SideDraw]
    dP: float = 0.0                   # per-stage pressure drop (Psat unit)
    P_bot: float = None               # reboiler pressure; None = flat column.
                                      # Resolved by `driver.size_column` once the
                                      # stage count is known (it is dP*(N-1) above
                                      # the top, and N is the method's OUTPUT).
    eps_stage: float = 1e-2           # junction tolerance floor (connect.connect)
    reactions: object = None          # Reactions (reactive.py) or None
    max_stages: int = 200             # per-section marching cap
    max_column_stages: int = None     # economic cap on the ASSEMBLED column; None
                                      # = `max_stages`. A design needing more than
                                      # this is reported infeasible with a
                                      # `too_many_stages` finding rather than
                                      # returned: profiles that only meet after
                                      # hundreds of trays are sitting on top of
                                      # R_min, and no such column gets built. This
                                      # is a separate verdict from the geometric
                                      # one, and the finding says which it is.
    efficiency: float = 1.0           # Murphree vapour efficiency (1 = ideal stages)
    anchor_method: str = "saddle"     # how an INTERIOR section is started:
                                      #   saddle       invariant manifolds through
                                      #                the section's saddle pinch
                                      #   ray          march inward from the far
                                      #                end of the stable ray (the
                                      #                body's S vertex)
                                      #   continuation launch from stages of the
                                      #                neighbouring profiles that
                                      #                lie inside this section
                                      # No "auto": which one is right is a design
                                      # judgement, and the three do not agree on
                                      # r_max (docs/adr/0004). Two-section columns
                                      # ignore this -- there is no interior section
                                      # to anchor. See `driver._interior_profiles`.
    balance_residual: float = 0.0     # E13: unclosed |f - (D xD + B xB)|/F, explicit path
    trace_floor: float = 1e-4         # starting-guess floor on every product split
    entrainer_trace: float = 1e-6     # ...and a smaller one for the entrainer
    # Both are SEEDS for the free splits, not answers -- `driver.solve_omega`
    # solves them against the junction condition. They still need sane values,
    # because a seed that overshoots sends the march somewhere the solver cannot
    # come back from, and the right value is component-specific: a heavy entrainer
    # amplifies ~1/K = 100-800 per stage marching down, so 1e-4 of glycol overhead
    # is already 0.3 within two stages -- past the extractive section's own level,
    # so the rectifying section comes out one stage long. Ordinary non-keys are
    # the other way round: reactive_mtbe leaves the simplex at 1e-5 and inverts its
    # temperature profile at 1e-6, and wants the coarser floor.

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


def free_split_indices(prob):
    """Components whose distillate split is NOT fixed by the specification.

    Two key recoveries fix two of the C component splits; the balance
    d_i + b_i = f_i fixes nothing else. The remaining splits are genuine free
    parameters of the design -- the classic multicomponent BVM statement that the
    non-key distillate compositions are determined by *requiring the section
    profiles to intersect*, not by a rule of thumb. `overall_balance` defaults
    them to a light/heavy heuristic with a trace floor, which is only a starting
    guess; `driver.solve_omega` solves for them.

    The ENTRAINER is among them, and has to be. It enters below the rectifying
    section so its distillate content is *small*, but pinning it at exactly zero
    traps the whole rectifying profile on the entrainer-free face -- where, for
    ethanol/water/EG at R=3, the operating line has a pinch pair straddling x_D
    (0.696 and 0.950) that no profile can cross. The section then cannot reach
    the feed at all, and the only thing it can "connect" to is its own anchor.
    Its K really is small enough that a down-march multiplies a trace by ~800 per
    stage, but that is an argument for stopping the march at the junction
    (`march_section(stop_sec=...)`), not for deleting the degree of freedom:
    x_D,entrainer is determined by the junction condition like every other free
    split, and `driver.solve_omega` solves for it.
    """
    return [i for i in range(prob.C) if i not in (prob.lk, prob.hk)]


def overall_balance(prob, EF=None, split=None):
    """Spec + feed(s) -> (xD, xB, D, B). Straight component balance f = D xD + B xB.

    The balance runs over *every* feed: the main feed(s) in `prob.feeds` and, in
    extractive mode, the entrainer stream E = EF*F_main at x_E. Excluding the
    entrainer traps the rectifying march on the entrainer=0 face (it carries zero
    entrainer to the distillate) and gives the bottoms the wrong anchor, so both
    the total flow F and the pooled feed f include it (blueprint Sec 14, 18.1).

    Explicit xD & xB win (D solved by least squares on f = D xD + B xB, D+B=F).
    Otherwise a per-component split-to-distillate fraction is built from the key
    recoveries and the non-key distribution, and d = frac*f, b = f - d. The trace
    floor on `frac` (`prob.trace_floor`) then keeps every component present
    anywhere in the column at >= that fraction of its feed amount in each
    product, so the profiles can leave a face. It is a seed, not an answer.
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
        B = F - D
        # E13: least-squares D can absorb products that don't actually close the
        # component balance. Report the leftover so an inconsistent xD/xB isn't
        # silently "solved" -- the residual is the mass the products can't account
        # for, per unit feed.
        resid = float(np.linalg.norm(f - (D * xD + B * xB)) / F)
        prob.balance_residual = resid
        if resid > 1e-3:
            import warnings
            warnings.warn(f"explicit xD/xB do not close the feed balance "
                          f"(residual {resid:.3g} of F); D fit by least squares",
                          stacklevel=2)
        return xD, xB, D, B

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
    # trace in each product so the profile can bend off the face.
    #
    # That floor is only the STARTING GUESS for the free splits -- its value is
    # not physical, and the design IS sensitive to it, which is exactly why it
    # must not be the final answer. `split` carries the solved values (in
    # `free_split_indices` order); see `driver.solve_omega`.
    eps = float(prob.trace_floor)
    frac = np.clip(frac, eps, 1.0 - eps)

    # The entrainer gets its own, smaller seed rather than the common floor: it is
    # fed BELOW the rectifying section, so its distillate content is genuinely
    # tiny, and its down-march amplification is large enough that the common floor
    # overshoots the extractive section in a single stage. Not zero, though --
    # exactly zero traps the profile on the entrainer-free face, where the
    # operating line's pinch pair straddles x_D and no profile reaches the feed.
    # Only where the floor is what put it there: an explicit `nonkey_to_dist`
    # asking for real entrainer overhead is a spec, not a seed, and stands.
    if prob.extractive and prob.x_E is not None:
        ent = (np.asarray(prob.x_E, float) > 0.5) & (frac <= eps)
        frac[ent] = float(prob.entrainer_trace)

    if split is not None:
        for i, k in enumerate(free_split_indices(prob)):
            frac[k] = float(np.clip(split[i], 0.0, 1.0))

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

    # E13: products inconsistent with the feed -> residual reported, warning raised
    import warnings
    # both products omit xylene, which the feed has at 0.25 -> cannot close
    p3 = build_problem(comps, [(z, 100.0)], pressure=760.0,
                       xD=np.array([0.9, 0.1, 0.0]), xB=np.array([0.7, 0.3, 0.0]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        overall_balance(p3)
    assert p3.balance_residual > 1e-3 and any("balance" in str(w.message)
                                              for w in caught), p3.balance_residual
    print("problem self-check OK", np.round(xD, 3), np.round(xB, 3), round(D, 2))


if __name__ == "__main__":
    _demo()
