"""Problem: column topology + specs + a square DOF ledger (blueprint Section 5).

A `Problem` is pure data — per-stage NumPy arrays plus the two terminal specs.
Everything that varies down the column (feeds, side draws, duties, pressure) is
a length-N array, so multifeed / draws / pumparounds / inter-heaters are
parameter changes, never new equation types (blueprint 7.x).

Squareness. The base Naphtali-Sandholm system has exactly 2C+1 equations and
2C+1 unknowns per stage (C material + C equilibrium + 1 energy). The two
terminal energy balances are *replaced* by the operating specs (reflux/boilup
family); the freed condenser/reboiler duties are recovered afterwards. So the
system stays square and block-tridiagonal, and "how many specs are required"
is exactly the MESH design-DoF count — which we borrow from `core.dof`.

Reactive stages carry n_rxn extra extent unknowns and n_rxn extra reaction
closures, so those blocks are 2C+1+n_rxn and the count stays balanced.
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(__file__), "..", "..", "python")))

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core.dof import DoFAnalyzer, Spec, SpecKind


# Spec kinds this solver replaces a terminal energy balance with. Recovery-type
# specs are deliberately absent: LK/HK recovery is projection-only (blueprint
# Section 10) and lives in the initializer, never as a governing equation.
_TOP_KINDS = {"reflux_ratio", "reflux_rate", "distillate_rate", "dist_purity"}
_BOT_KINDS = {"boilup_ratio", "boilup_rate", "bottoms_rate", "bottoms_purity"}


@dataclass(frozen=True)
class OpSpec:
    """One terminal spec that replaces a terminal energy balance.

    kind : one of _TOP_KINDS / _BOT_KINDS
    value: the specified number
    comp : component index for purity specs (ignored otherwise)
    """
    kind: str
    value: float
    comp: int = -1


@dataclass
class Reactions:
    """Reaction set on a subset of stages (blueprint Section 11).

    nu     : (n_rxn, C) stoichiometric coefficients (products +, reactants -)
    stages : 0-based stage indices that are reactive
    kind   : "kinetic" or "equilibrium"
    Keq    : (n_rxn,) equilibrium constants          (equilibrium kind)
    k_fwd  : (n_rxn,) forward rate coefficients       (kinetic kind)
    holdup : (n_rxn,) or scalar liquid holdup factor multiplying the rate
    """
    nu: np.ndarray
    stages: np.ndarray
    kind: str = "kinetic"
    Keq: Optional[np.ndarray] = None
    k_fwd: Optional[np.ndarray] = None
    holdup: float = 1.0

    @property
    def n_rxn(self) -> int:
        return int(np.asarray(self.nu).shape[0])


@dataclass
class Problem:
    """Full topology + specs for one column. Stage index is 0-based, top->bottom.

    n_stages N, comps (length C). feed (N,C) component molar feed; feedH (N,)
    feed enthalpy rate; rl/rv (N,) liquid/vapour side-draw ratios W_i/L_i,
    U_i/V_i; duty (N,) inter-stage heat (terminals recovered, so duty[0] and
    duty[-1] are ignored); pressure (N,). top_spec / bottom_spec are OpSpec.
    reactions is optional.
    """
    n_stages: int
    comps: List[str]
    feed: np.ndarray
    feedH: np.ndarray
    pressure: np.ndarray
    top_spec: OpSpec
    bottom_spec: OpSpec
    rl: np.ndarray = None
    rv: np.ndarray = None
    duty: np.ndarray = None
    reactions: Optional[Reactions] = None
    condenser: str = "partial"          # top stage is an equilibrium drum (v_1 = distillate)

    def __post_init__(self):
        N = self.n_stages
        self.feed = np.asarray(self.feed, float).reshape(N, -1)
        self.feedH = np.asarray(self.feedH, float).reshape(N)
        self.pressure = np.asarray(self.pressure, float).reshape(N)
        if self.rl is None:
            self.rl = np.zeros(N)
        if self.rv is None:
            self.rv = np.zeros(N)
        if self.duty is None:
            self.duty = np.zeros(N)
        self.rl = np.asarray(self.rl, float).reshape(N)
        self.rv = np.asarray(self.rv, float).reshape(N)
        self.duty = np.asarray(self.duty, float).reshape(N)
        if self.top_spec.kind not in _TOP_KINDS:
            raise ValueError(f"top spec {self.top_spec.kind!r} not in {_TOP_KINDS}")
        if self.bottom_spec.kind not in _BOT_KINDS:
            raise ValueError(f"bottom spec {self.bottom_spec.kind!r} not in {_BOT_KINDS}")

    @property
    def C(self) -> int:
        return len(self.comps)

    @property
    def reactive_mask(self) -> np.ndarray:
        """(N,) bool: True on stages carrying reactions."""
        m = np.zeros(self.n_stages, bool)
        if self.reactions is not None:
            m[np.asarray(self.reactions.stages, int)] = True
        return m

    def block_sizes(self) -> np.ndarray:
        """(N,) unknowns/equations per stage: 2C+1, plus n_rxn on reactive stages."""
        base = np.full(self.n_stages, 2 * self.C + 1)
        if self.reactions is not None:
            base[np.asarray(self.reactions.stages, int)] += self.reactions.n_rxn
        return base

    def n_side_draws(self) -> int:
        return int(np.count_nonzero(self.rl) + np.count_nonzero(self.rv))

    def dof_report(self):
        """DoF ledger via core.dof. required = terminal duties + side draws;
        provided = the two terminal specs (+ any extra you declare). Returns a
        core.dof.DoFResult. Square <=> result.status == 'exact'."""
        analyzer = DoFAnalyzer(n_components=self.C, condenser=True, reboiler=True,
                               partial_condenser=False,
                               n_side_draws=self.n_side_draws())
        # Map our OpSpecs onto core.dof Spec kinds for the count (kinds differ in
        # spelling but the ledger only counts distinct (kind, unit) pairs).
        provided = [Spec(SpecKind.REFLUX_RATIO, self.top_spec.value, "condenser"),
                    Spec(SpecKind.BOILUP_RATIO, self.bottom_spec.value, "reboiler")]
        # one spec per side draw (drawn rate pins that extra product)
        for j in range(self.n_stages):
            if self.rl[j] > 0 or self.rv[j] > 0:
                provided.append(Spec(SpecKind.SIDEDRAW_RATE, 0.0, f"stage{j}"))
        return analyzer.analyze(provided)

    def is_square(self) -> bool:
        return self.dof_report().status == "exact"

    def require_square(self):
        """Refuse to run unless the DOF ledger balances (blueprint Section 5)."""
        rep = self.dof_report()
        if rep.status != "exact":
            raise ValueError(f"problem is not square ({rep.status}): {rep.message}")
        return rep


def build_problem(*, n_stages, comps, feeds, pressure, provider,
                  top_spec, bottom_spec, draws=(), duties=(), reactions=None):
    """Scatter feeds/draws/duties onto per-stage arrays and build a Problem.

    feeds : (stage, flow, composition[C][, T_feed]) — T_feed sets the feed
            enthalpy (defaults to the feed bubble point). stage is 0-based.
    draws : (stage, liquid_ratio, vapor_ratio)  side-draw ratios W/L, U/V.
    duties: (stage, heat).
    provider: a ThermoProvider, used only to size feed enthalpies.
    """
    N, C = int(n_stages), len(comps)
    feed = np.zeros((N, C))
    feedH = np.zeros(N)
    rl = np.zeros(N)
    rv = np.zeros(N)
    duty = np.zeros(N)

    for f in feeds:
        stage, flow, comp = int(f[0]), float(f[1]), np.asarray(f[2], float)
        if not (0 <= stage < N):
            raise ValueError(f"feed stage {stage} out of range 0..{N-1}")
        if comp.shape != (C,) or abs(comp.sum() - 1.0) > 1e-3:
            raise ValueError(f"feed at stage {stage}: bad composition {comp}")
        Tf = float(f[3]) if len(f) > 3 else provider.bubble_T(comp, np.atleast_1d(pressure).ravel()[0]
                                                              if np.ndim(pressure) else pressure)
        feed[stage] += flow * comp
        feedH[stage] += flow * provider.h_L(comp[None, :], np.array([Tf]))[0]

    for stage, lr, vr in draws:
        rl[int(stage)] += float(lr)
        rv[int(stage)] += float(vr)
    for stage, q in duties:
        duty[int(stage)] += float(q)

    P = np.asarray(pressure, float)
    P = np.full(N, float(P)) if P.ndim == 0 else P.reshape(N)

    return Problem(n_stages=N, comps=list(comps), feed=feed, feedH=feedH,
                   pressure=P, top_spec=top_spec, bottom_spec=bottom_spec,
                   rl=rl, rv=rv, duty=duty, reactions=reactions)


def _demo():
    from thermo_adapter import FreeColumnThermo
    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = FreeColumnThermo(abc)
    comps = ["benzene", "toluene", "xylene"]

    prob = build_problem(
        n_stages=12, comps=comps, feeds=[(6, 100.0, [0.4, 0.35, 0.25])],
        pressure=760.0, provider=tp,
        top_spec=OpSpec("reflux_ratio", 3.0),
        bottom_spec=OpSpec("distillate_rate", 40.0) if False else OpSpec("bottoms_rate", 60.0))

    # square: total condenser + reboiler => 2 specs, we provide 2
    rep = prob.require_square()
    assert rep.status == "exact", rep
    assert prob.block_sizes().tolist() == [7] * 12   # 2C+1 = 7
    assert abs(prob.feed.sum() - 100.0) < 1e-9
    assert prob.feedH[6] != 0.0, "feed enthalpy sized"

    # a ratio-pinned side draw stays square: the draw ratio IS its spec (extra
    # product, extra provided spec), so the ledger still balances.
    prob_sd = build_problem(
        n_stages=12, comps=comps, feeds=[(6, 100.0, [0.4, 0.35, 0.25])],
        pressure=760.0, provider=tp, draws=[(4, 0.1, 0.0)],
        top_spec=OpSpec("reflux_ratio", 3.0), bottom_spec=OpSpec("bottoms_rate", 60.0))
    assert prob_sd.n_side_draws() == 1 and prob_sd.is_square()

    # reactive stage grows the block by n_rxn
    rx = Reactions(nu=np.array([[-1.0, 1.0, 0.0]]), stages=np.array([6]),
                   kind="kinetic", k_fwd=np.array([0.5]))
    prob_rx = build_problem(
        n_stages=12, comps=comps, feeds=[(6, 100.0, [0.4, 0.35, 0.25])],
        pressure=760.0, provider=tp, top_spec=OpSpec("reflux_ratio", 3.0),
        bottom_spec=OpSpec("bottoms_rate", 60.0), reactions=rx)
    bs = prob_rx.block_sizes()
    assert bs[6] == 8 and bs[5] == 7, bs   # 2C+1+1 on the reactive stage

    # a bad spec kind is rejected
    try:
        Problem(n_stages=2, comps=comps, feed=np.zeros((2, 3)), feedH=np.zeros(2),
                pressure=np.full(2, 760.0), top_spec=OpSpec("boilup_ratio", 1.0),
                bottom_spec=OpSpec("bottoms_rate", 1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("top spec must be a top kind")
    print("problem self-check OK")


if __name__ == "__main__":
    _demo()
