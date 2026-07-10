"""Feasibility classification (blueprint Section 12-13; goal unit 9).

Given a Problem, a ThermoProvider and a state U (usually U0), return a list of
`Finding`s — each naming a failure *class* and the offending stage(s), never a
bare boolean. The classes mirror the blueprint's failure modes:

    pinch_singularity        Jacobian degenerates (a near-singular diagonal block)
    phase_disappearance      a liquid or vapour flow collapses toward zero
    thermodynamic_invalidity K undefined / non-positive, or no bubble point
    flow_reversal            a component flow goes negative
    infeasible_feed_coupling DOF not square, or a non-positive product rate

`assess` bundles them into a structural / physical / thermodynamic feasibility
report (Section 12).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from residual import unpack, flows
from jacobian import jacobian_blocks, dense_from_blocks

PINCH = "pinch_singularity"
PHASE = "phase_disappearance"
THERMO = "thermodynamic_invalidity"
REVERSAL = "flow_reversal"
FEED = "infeasible_feed_coupling"


@dataclass
class Finding:
    cls: str
    stages: List[int]
    message: str

    def __repr__(self):
        return f"Finding({self.cls}, stages={self.stages}, {self.message!r})"


def _flow_findings(prob, l, v, L, V, floor=1e-6):
    out = []
    rev = sorted(set(np.where(l < -1e-9)[0].tolist())
                 | set(np.where(v < -1e-9)[0].tolist()))
    if rev:
        out.append(Finding(REVERSAL, rev, "negative component flow(s)"))
    gone = sorted(set(np.where(L < floor)[0].tolist())
                  | set(np.where(V < floor)[0].tolist()))
    if gone:
        out.append(Finding(PHASE, gone,
                           "liquid or vapour flow collapsed toward zero"))
    return out


def _thermo_findings(prob, provider, x, T):
    bad = []
    try:
        K = provider.K(x, T, prob.pressure)
        rows = np.where(~np.all(np.isfinite(K) & (K > 0), axis=1))[0]
        bad.extend(rows.tolist())
    except Exception:
        bad.append(-1)
    for i in range(prob.n_stages):
        try:
            provider.bubble_T(x[i], prob.pressure[i])
        except Exception:
            bad.append(i)
    bad = sorted(set(bad))
    return [Finding(THERMO, bad, "K non-positive/non-finite or no bubble point")] if bad else []


def _feed_findings(prob):
    out = []
    rep = prob.dof_report()
    if rep.status != "exact":
        out.append(Finding(FEED, [], f"DOF not square ({rep.status}): {rep.message}"))
    F = float(prob.feed.sum())
    # raw product rates implied by the specs, before any clipping
    D = prob.top_spec.value if prob.top_spec.kind == "distillate_rate" else None
    B = prob.bottom_spec.value if prob.bottom_spec.kind == "bottoms_rate" else None
    if D is not None and B is not None and abs(D + B - F) > 1e-6 * max(F, 1.0):
        out.append(Finding(FEED, [], f"distillate + bottoms ({D:.3g}+{B:.3g}) "
                                     f"conflict with feed {F:.3g}"))
    if (D is not None and not (0 < D < F)) or (B is not None and not (0 < B < F)):
        out.append(Finding(FEED, [], f"product rate outside (0, F={F:.3g})"))
    return out


def _pinch_findings(prob, provider, U, cond_tol=1e12):
    """Near-singular Jacobian => pinch. Reports the worst-conditioned diagonal
    block's stage(s). Dense SVD is fine at feasibility-assessment scale."""
    try:
        A, B, Cc = jacobian_blocks(U, prob, provider)
    except Exception:
        return [Finding(PINCH, [], "Jacobian assembly failed")]
    # smallest singular value of each diagonal block -> most degenerate stage
    smin = np.array([np.linalg.svd(B[i], compute_uv=False)[-1]
                     for i in range(prob.n_stages)])
    smax = np.array([np.linalg.svd(B[i], compute_uv=False)[0]
                     for i in range(prob.n_stages)])
    cond = np.where(smin > 0, smax / smin, np.inf)
    worst = np.where(cond > cond_tol)[0].tolist()
    if worst:
        return [Finding(PINCH, worst,
                        f"near-singular Jacobian block (cond>{cond_tol:.0e})")]
    return []


def classify(prob, provider, U):
    """All findings for a state U. Empty list => no failure mode detected."""
    N, C = prob.n_stages, prob.C
    R = prob.reactions.n_rxn if prob.reactions is not None else 0
    l, v, T, xi = unpack(U, N, C, R)
    L, V, x, y = flows(l, v)
    findings = []
    findings += _feed_findings(prob)
    findings += _flow_findings(prob, l, v, L, V)
    findings += _thermo_findings(prob, provider, x, T)
    findings += _pinch_findings(prob, provider, U)
    return findings


@dataclass
class FeasibilityReport:
    structural: bool          # Jacobian non-singular at U
    physical: bool            # mass conserved-ish, no negative/collapsed flow
    thermodynamic: bool       # all stages admit valid K
    findings: List[Finding] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.structural and self.physical and self.thermodynamic


def assess(prob, provider, U):
    """Structural / physical / thermodynamic feasibility report for state U."""
    findings = classify(prob, provider, U)
    cls = {f.cls for f in findings}
    return FeasibilityReport(
        structural=PINCH not in cls,
        physical=not (cls & {PHASE, REVERSAL, FEED}),
        thermodynamic=THERMO not in cls,
        findings=findings)


def _demo():
    import numpy as np
    from thermo_adapter import FreeColumnThermo
    from problem import build_problem, OpSpec
    from initializer import initialize
    from residual import pack

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = FreeColumnThermo(abc)
    comps = ["benzene", "toluene", "xylene"]
    N, C = 14, 3
    prob = build_problem(n_stages=N, comps=comps, feeds=[(7, 100.0, [0.4, 0.35, 0.25])],
                         pressure=760.0, provider=tp,
                         top_spec=OpSpec("reflux_ratio", 4.0),
                         bottom_spec=OpSpec("bottoms_rate", 60.0))

    # A healthy U0 is feasible on all three axes.
    U0 = initialize(prob, tp)
    rep = assess(prob, tp, U0)
    assert rep.feasible, rep.findings

    # Flow reversal is caught with the offending stages named.
    Ubad = U0.copy()
    l, v, T, xi = unpack(Ubad, N, C, 0)
    l[5, 1] = -1.0
    Ubad = pack(l, v, T)
    f = classify(prob, tp, Ubad)
    assert any(x.cls == "flow_reversal" and 5 in x.stages for x in f), f

    # Phase disappearance: collapse a stage's vapour.
    l2, v2, T2, _ = unpack(U0.copy(), N, C, 0)
    v2[9, :] = 1e-9
    f2 = classify(prob, tp, pack(l2, v2, T2))
    assert any(x.cls == "phase_disappearance" and 9 in x.stages for x in f2), f2

    # Infeasible feed coupling: distillate exceeds the feed (D>F via a rate spec).
    prob_bad = build_problem(n_stages=N, comps=comps,
                             feeds=[(7, 100.0, [0.4, 0.35, 0.25])], pressure=760.0,
                             provider=tp, top_spec=OpSpec("distillate_rate", 150.0),
                             bottom_spec=OpSpec("bottoms_rate", 60.0))
    f3 = _feed_findings(prob_bad)
    assert any(x.cls == "infeasible_feed_coupling" for x in f3), f3

    print(f"diagnostics self-check OK (healthy U0 feasible; reversal/phase/feed "
          f"classified with stages)")


if __name__ == "__main__":
    _demo()
