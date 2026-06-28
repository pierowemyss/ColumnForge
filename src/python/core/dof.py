#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Degrees-of-freedom analysis for column specification.

A MESH-derived *design* degrees-of-freedom count, parameterized by whether the
energy (enthalpy) balance is active:

- Under the constant-molar-overflow (CMO) assumption used by shortcut /
  feasibility tools such as BVM, the energy balance drops out and heat-duty
  specs are unavailable. The required operating-spec count then reduces to the
  familiar result: one spec per terminal duty unit (condenser, reboiler) plus
  one per extra product -> "two specs for a simple column".
- The same ledger extends to the full MESH count when ``energy_balance=True``
  for a future rigorous solver: duty spec kinds become valid. (That path is a
  marked extension point; the column count structure is already in place.)

Specs are structured objects (kind + value + which unit), never display
strings, so the count is exact and free of string collisions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class SpecKind(Enum):
    REFLUX_RATIO = "Reflux ratio"
    REFLUX_RATE = "Reflux rate"
    BOILUP_RATIO = "Boilup ratio"
    BOILUP_RATE = "Boilup rate"
    DISTILLATE_RATE = "Distillate rate"
    BOTTOMS_RATE = "Bottoms rate"
    DF_RATIO = "Distillate-to-feed ratio"
    BF_RATIO = "Bottoms-to-feed ratio"
    LK_RECOVERY = "Light-key recovery"
    HK_RECOVERY = "Heavy-key recovery"
    DIST_PURITY = "Distillate purity"
    BOTTOMS_PURITY = "Bottoms purity"
    SIDEDRAW_RATE = "Side-draw rate"
    CONDENSER_DUTY = "Condenser duty"   # requires energy balance
    REBOILER_DUTY = "Reboiler duty"     # requires energy balance


# Operating specs that pin a simple column's two free knobs (reflux R, distillate
# D) under CMO — the set the operating-point resolver knows how to invert.
OPERATING_KINDS = frozenset({
    SpecKind.REFLUX_RATIO, SpecKind.REFLUX_RATE,
    SpecKind.BOILUP_RATIO, SpecKind.BOILUP_RATE,
    SpecKind.DISTILLATE_RATE, SpecKind.BOTTOMS_RATE,
    SpecKind.DF_RATIO, SpecKind.BF_RATIO,
    SpecKind.LK_RECOVERY, SpecKind.HK_RECOVERY,
    SpecKind.DIST_PURITY, SpecKind.BOTTOMS_PURITY,
})


# Spec kinds that are only meaningful once the energy balance is active.
ENERGY_ONLY = frozenset({SpecKind.CONDENSER_DUTY, SpecKind.REBOILER_DUTY})


@dataclass(frozen=True)
class Spec:
    """A single operating specification.

    unit_ref disambiguates specs of the same kind on different units
    (e.g. a per-side-draw rate), so two distinct specs never collide.
    """
    kind: SpecKind
    value: float
    unit_ref: str = "column"
    component: int = -1   # 0-based species index for purity specs; -1 = n/a


@dataclass
class DoFResult:
    required: int
    provided: int
    status: str            # "under" | "exact" | "over" | "invalid"
    icon: str
    message: str
    can_run: bool
    invalid: List[Spec] = field(default_factory=list)


class DoFAnalyzer:
    """Counts required vs provided design specs for a column configuration."""

    def __init__(self, *, n_components, condenser=True, reboiler=True,
                 partial_condenser=False, n_side_draws=0,
                 module_spec_counts=None, energy_balance=False):
        self.C = n_components
        self.condenser = condenser
        self.reboiler = reboiler
        self.partial_condenser = partial_condenser
        self.n_side_draws = n_side_draws
        self.module_spec_counts = list(module_spec_counts or [])
        self.energy_balance = energy_balance

    def required_specs(self) -> int:
        """MESH design-DoF: one spec per terminal duty unit + one per extra
        product. Energy balance does not change this *count* (it only enables
        duty spec kinds); it changes which kinds are valid in ``analyze``.
        """
        req = 0
        if self.condenser:
            req += 1
        if self.reboiler:
            req += 1
        if self.partial_condenser:        # vapour + liquid distillate -> extra product
            req += 1
        req += self.n_side_draws
        req += sum(self.module_spec_counts)
        return req

    def analyze(self, provided: List[Spec]) -> DoFResult:
        invalid = [s for s in provided
                   if s.kind in ENERGY_ONLY and not self.energy_balance]
        valid = [s for s in provided if s not in invalid]
        # de-duplicate by (kind, unit_ref) so the same spec set twice counts once
        provided_count = len({(s.kind, s.unit_ref) for s in valid})
        required = self.required_specs()

        if invalid:
            kinds = ", ".join(s.kind.value for s in invalid)
            return DoFResult(required, provided_count, "invalid", "❌",
                             f"Duty specs require the energy balance: {kinds}",
                             False, invalid)
        if provided_count < required:
            need = required - provided_count
            return DoFResult(required, provided_count, "under", "⚠️",
                             f"Under-specified: need {need} more spec(s).", False)
        if provided_count > required:
            extra = provided_count - required
            names = ", ".join(sorted({s.kind.value for s in valid}))
            return DoFResult(required, provided_count, "over", "❌",
                             f"Over-specified by {extra}: you set [{names}], but this "
                             f"column takes {required}. Remove {extra} — the solver holds "
                             f"specs exactly and won't reconcile a conflict.", False)
        return DoFResult(required, provided_count, "exact", "✅",
                         "Perfectly specified.", True)


def _demo():
    simple = DoFAnalyzer(n_components=3)                 # total cond + reboiler
    assert simple.required_specs() == 2

    exact = simple.analyze([Spec(SpecKind.REFLUX_RATIO, 2.0, "condenser"),
                            Spec(SpecKind.LK_RECOVERY, 0.98, "column")])
    assert exact.status == "exact" and exact.can_run, exact

    assert DoFAnalyzer(n_components=3, n_side_draws=1).required_specs() == 3
    assert DoFAnalyzer(n_components=2, condenser=False).required_specs() == 1
    assert DoFAnalyzer(n_components=3, partial_condenser=True).required_specs() == 3
    assert DoFAnalyzer(n_components=3,
                       module_spec_counts=[1, 1]).required_specs() == 4

    # duty spec rejected under CMO, accepted with energy balance on
    duty = [Spec(SpecKind.CONDENSER_DUTY, 1e6, "condenser"),
            Spec(SpecKind.BOILUP_RATIO, 1.5, "reboiler")]
    assert simple.analyze(duty).status == "invalid"
    assert DoFAnalyzer(n_components=3, energy_balance=True).analyze(duty).status == "exact"

    assert simple.analyze([Spec(SpecKind.REFLUX_RATIO, 2.0, "condenser")]).status == "under"
    assert simple.analyze([Spec(SpecKind.REFLUX_RATIO, 2.0, "condenser"),
                           Spec(SpecKind.BOILUP_RATIO, 1.5, "reboiler"),
                           Spec(SpecKind.LK_RECOVERY, 0.9, "column")]).status == "over"
    print("dof self-check OK")


if __name__ == "__main__":
    _demo()
