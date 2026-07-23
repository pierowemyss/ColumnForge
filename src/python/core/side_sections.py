"""Side strippers and side rectifiers, coupled to the main column by tearing.

A side section is a small column hung off a main-column stage:

  * **stripper**  — liquid S drawn at stage i is fed to the top of a reboiled
    sub-column; its bottoms is a side product, its overhead vapour returns to the
    main column at stage j (above the draw).
  * **rectifier** — vapour S drawn at stage i is fed to the bottom of a
    condensed sub-column; its distillate is a side product, its bottom liquid
    returns to the main column at stage j (below the draw).

Nothing new is needed inside the solvers: the draw is an ordinary `liquid_draw`/
`vapor_draw` and the return is an ordinary feed (saturated vapour for a stripper,
saturated liquid for a rectifier), so the main column's mass balance closes on its
own. Only the *composition* of the return is unknown, and that is torn: solve the
main column, solve each sub-column on the drawn composition, feed the returns back,
repeat. Same trick the pumparound uses, one level up.

Under CMO the section's own split is exact from its ratio spec:

    stripper : product = S/(1+VB),  return = S*VB/(1+VB)
    rectifier: product = S/(1+R),   return = S*R/(1+R)

# ponytail: the sub-column runs at the main column's draw-stage pressure (no
# sub-column dP) and is always solved with the bubble-point solver — a 3-6 stage
# section is small and Wang-Henke is the robust choice. Upgrade path: a fully
# coupled Naphtali-Sandholm block if the tear ever fails to converge.

Stage numbers here are solver-internal (1 = top .. N = reboiler), like SolverInput.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.solver_input import build_solver_input


@dataclass
class SideSection:
    id: str
    kind: str                 # "stripper" | "rectifier"
    draw_stage: int           # main-column internal stage the section hangs off
    return_stage: int         # where the section's return re-enters
    rate: float               # kmol/h drawn (liquid for a stripper, vapour for a rectifier)
    ratio: float              # boilup V/B (stripper) | reflux L/D (rectifier)
    n_stages: int = 4
    # torn state: composition of the stream returned to the main column
    return_comp: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        if self.kind not in ("stripper", "rectifier"):
            raise ValueError(f"{self.id}: kind must be 'stripper' or 'rectifier'")
        if self.rate <= 0.0:
            raise ValueError(f"{self.id}: draw rate must be positive")
        if self.ratio <= 0.0:
            raise ValueError(f"{self.id}: boilup/reflux ratio must be positive")
        if self.n_stages < 1:
            raise ValueError(f"{self.id}: needs at least 1 stage")
        if self.kind == "stripper" and self.return_stage >= self.draw_stage:
            raise ValueError(
                f"{self.id}: a side stripper returns vapour ABOVE its draw "
                f"(return {self.return_stage} must be < draw {self.draw_stage})")
        if self.kind == "rectifier" and self.return_stage <= self.draw_stage:
            raise ValueError(
                f"{self.id}: a side rectifier returns liquid BELOW its draw "
                f"(return {self.return_stage} must be > draw {self.draw_stage})")

    @property
    def product_flow(self) -> float:
        """kmol/h leaving the system as this section's side product."""
        return self.rate / (1.0 + self.ratio)

    @property
    def return_flow(self) -> float:
        """kmol/h returned to the main column (the rest of the draw)."""
        return self.rate - self.product_flow

    @property
    def return_q(self) -> float:
        """Thermal quality of the return: vapour off a stripper, liquid off a
        rectifier."""
        return 0.0 if self.kind == "stripper" else 1.0

    def draw_rates(self) -> tuple:
        """(liquid, vapour) drawn from the main column at draw_stage."""
        return (self.rate, 0.0) if self.kind == "stripper" else (0.0, self.rate)


def solve_section(sec: SideSection, main_prof: dict, si_main, **knobs) -> dict:
    """Solve one side section against the current main-column profile.

    Returns the sub-column profile (see column_solvers._finish_profile); its
    "xD"/"y[0]" is a stripper's returning vapour, its "xB"/"x[-1]" a rectifier's
    returning liquid.
    """
    from core.column_solvers import solve_bubble_point

    j = sec.draw_stage - 1                          # 0-based index into the profile
    P = float(np.asarray(si_main.pressure)[j])
    if sec.kind == "stripper":
        z = np.asarray(main_prof["x"])[j]           # liquid drawn
        feed_stage, q, condenser = 1, 1.0, "none"
        R, D = 0.0, sec.return_flow                 # overhead vapour = the return
    else:
        z = np.asarray(main_prof["y"])[j]           # vapour drawn
        feed_stage, q, condenser = sec.n_stages, 0.0, "total"
        R, D = sec.ratio, sec.product_flow          # distillate = the side product

    si = build_solver_input(
        n_stages=sec.n_stages, comps=si_main.comps,
        feeds=[(feed_stage, sec.rate, z / z.sum(), q)],
        R=R, D=D, pressure=P, antoine=si_main.antoine,
        gamma_fn=si_main.gamma_fn, phi_fn=si_main.phi_fn, condenser=condenser)
    return solve_bubble_point(si, **knobs)


def _returned(sec: SideSection, sub_prof: dict) -> np.ndarray:
    """Composition of the stream the section sends back to the main column."""
    return (np.asarray(sub_prof["xD"]) if sec.kind == "stripper"
            else np.asarray(sub_prof["xB"])).copy()


def _product(sec: SideSection, sub_prof: dict) -> tuple:
    """(composition, temperature) of the section's side product."""
    if sec.kind == "stripper":
        return np.asarray(sub_prof["xB"]).copy(), float(sub_prof["T"][-1])
    return np.asarray(sub_prof["xD"]).copy(), float(sub_prof["T"][0])


def make_side_solver(solver, sections: List[SideSection], rebuild,
                     tol: float = 1e-5, max_passes: int = 25):
    """Wrap `solver` so every call converges the side-section tear.

    solver:  solve_bubble_point / solve_inside_out (called as solver(si, **knobs))
    rebuild: (si) -> SolverInput, re-scattering the *current* sec.return_comp of
             every section onto the main column's per-stage arrays (si carries the
             operating point to rebuild at). The caller owns that closure (it knows
             the feeds/draws); we only mutate sec.return_comp between passes.

    The returned callable has the same signature and adds "side_sections" (plus
    "side_tear_residual") to the profile, with the report bookkeeping
    (feed_totals / side_draws) already netted so the recycle never shows up as an
    external feed or a product.

    # ponytail: plain successive substitution. It converges linearly (~0.4 per
    # pass on a refinery-ish stripper) but each section keeps its return_comp
    # between calls, so only the first solve of a run pays the full tear and the
    # operating-point root-find's later trials need 2-3 passes. Add Aitken
    # extrapolation here if a stiff section ever runs out of passes.
    """
    def solve(si, **knobs):
        sub_knobs = {k: v for k, v in knobs.items()
                     if k in ("max_iter", "tol", "efficiency", "cancel")}
        prof = solver(si, **knobs)
        subs = []
        moved = 0.0
        for _ in range(max_passes):
            if prof.get("message") == "Aborted.":
                break
            subs = [solve_section(s, prof, si, **sub_knobs) for s in sections]
            moved = 0.0
            for sec, sub in zip(sections, subs):
                new = _returned(sec, sub)
                if sec.return_comp is not None:
                    moved = max(moved, float(np.max(np.abs(new - sec.return_comp))))
                else:
                    moved = float("inf")
                sec.return_comp = new
            if moved < tol:
                break
            si = rebuild(si)
            prof = solver(si, x0=prof["x"], T0=prof["T"], **knobs)
        prof["side_tear_residual"] = moved if subs else 0.0
        _net_report(prof, si, sections, subs)
        return prof

    return solve


def _net_report(prof: dict, si, sections: List[SideSection], subs: List[dict]):
    """Turn the tear's internal streams back into what a user sees: the returns
    are not external feed, the draws are not products — the section product is."""
    if not subs:
        return
    feed_totals = np.asarray(prof.get("feed_totals"), float).copy()
    entries = []
    for sec, sub in zip(sections, subs):
        comp, T = _product(sec, sub)
        feed_totals -= sec.return_flow * (sec.return_comp
                                          if sec.return_comp is not None else comp)
        # the draw itself is internal; drop (or shrink) its side-draw report row
        liq, vap = sec.draw_rates()
        for sd in prof.get("side_draws", []):
            if sd["stage"] == sec.draw_stage - 1:
                sd["liquid"] = max(0.0, sd["liquid"] - liq)
                sd["vapor"] = max(0.0, sd["vapor"] - vap)
        entries.append({
            "id": sec.id, "kind": sec.kind, "stage": sec.draw_stage - 1,
            "return_stage": sec.return_stage - 1, "rate": sec.rate,
            "ratio": sec.ratio, "flow": sec.product_flow, "comp": comp, "T": T,
            "profile": sub,
        })
    prof["feed_totals"] = feed_totals
    prof["side_draws"] = [sd for sd in prof.get("side_draws", [])
                          if sd["liquid"] > 1e-9 or sd["vapor"] > 1e-9]
    prof["side_sections"] = entries


def _demo():
    from core.column_solvers import solve_bubble_point

    comps = ["benzene", "toluene", "xylene"]
    antoine = np.array([(6.90565, 1211.033, 220.79),
                        (6.95464, 1344.8, 219.48),
                        (6.99052, 1453.43, 215.31)])
    z = np.array([0.4, 0.35, 0.25])
    F, D, R, N = 100.0, 35.0, 3.0, 16

    sec = SideSection(id="SS1", kind="stripper", draw_stage=11, return_stage=10,
                      rate=30.0, ratio=1.5, n_stages=4)

    def build(_si=None):
        feeds = [(8, F, z)]
        if sec.return_comp is not None:
            feeds.append((sec.return_stage, sec.return_flow, sec.return_comp, 0.0))
        liq, vap = sec.draw_rates()
        return build_solver_input(
            n_stages=N, comps=comps, feeds=feeds,
            draws=[(sec.draw_stage, liq, vap)],
            R=R, D=D, pressure=760.0, antoine=antoine)

    solve = make_side_solver(solve_bubble_point, [sec], build)
    prof = solve(build(), max_iter=300)

    ss = prof["side_sections"][0]
    assert abs(ss["flow"] - 30.0 / 2.5) < 1e-9, ss["flow"]

    # per-component closure: what comes in as feed leaves as D + B + side product
    out = (prof["D"] * prof["xD"] + prof["B"] * prof["xB"]
           + ss["flow"] * ss["comp"])
    assert np.allclose(F * z, out, atol=1e-3), (F * z, out)
    # the recycle must not show up as external feed
    assert np.allclose(prof["feed_totals"], F * z, atol=1e-3), prof["feed_totals"]
    # a stripped side product is heavier than the liquid drawn from that stage
    assert ss["comp"][0] < prof["x"][ss["stage"]][0], (ss["comp"], prof["x"][ss["stage"]])

    # --- side rectifier: same closure, product lighter than the vapour drawn ---
    rec = SideSection(id="SR1", kind="rectifier", draw_stage=6, return_stage=7,
                      rate=25.0, ratio=2.0, n_stages=4)

    def build_r(_si=None):
        feeds = [(8, F, z)]
        if rec.return_comp is not None:
            feeds.append((rec.return_stage, rec.return_flow, rec.return_comp, 1.0))
        liq, vap = rec.draw_rates()
        return build_solver_input(
            n_stages=N, comps=comps, feeds=feeds,
            draws=[(rec.draw_stage, liq, vap)],
            R=R, D=D, pressure=760.0, antoine=antoine)

    prof_r = make_side_solver(solve_bubble_point, [rec], build_r)(build_r(),
                                                                 max_iter=300)
    sr = prof_r["side_sections"][0]
    out_r = (prof_r["D"] * prof_r["xD"] + prof_r["B"] * prof_r["xB"]
             + sr["flow"] * sr["comp"])
    assert np.allclose(F * z, out_r, atol=1e-3), (F * z, out_r)
    assert sr["comp"][0] > prof_r["y"][sr["stage"]][0], (sr["comp"],
                                                         prof_r["y"][sr["stage"]])
    print("side_sections self-check OK")


if __name__ == "__main__":
    _demo()
