"""Side strippers and side rectifiers, coupled to the main column by tearing.

A side section is a small column hung off a main-column stage:

  * **stripper**  — liquid S drawn at stage i is fed to the top of a reboiled
    sub-column; its bottoms is a side product, its overhead vapour returns to the
    main column at stage j (above the draw).
  * **rectifier** — vapour S drawn at stage i is fed to the bottom of a
    condensed sub-column; its distillate is a side product, its bottom liquid
    returns to the main column at stage j (below the draw).

Which side of the FEED the draw sits on is what makes the arrangement a coupled
column rather than an expensive way to make a product twice: a stripper draws
liquid ABOVE the feed (below it the liquid is already bottoms) and a rectifier
draws vapour BELOW it (above it the vapour is already distillate). Nothing here
enforces that — it is a design statement, not a mass balance — but both bundled
examples once had it backwards, so `test_examples` now pins the side product
being a genuine intermediate.

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


def _extrapolate(new: np.ndarray, step: np.ndarray, prev_step) -> np.ndarray:
    """Aitken jump along a geometrically converging recycle step.

    The tear crawls: successive substitution on a side column contracts by the
    fraction of the draw that comes back, which on a real arrangement is 0.75-0.96
    per pass, i.e. hundreds of passes to 1e-5. Two consecutive steps give the
    ratio r, and a geometric series sums the whole remaining tail at once:

        x* = x_k + r/(1-r) * (x_k - x_{k-1})

    Only the *next iterate* is extrapolated -- the pass after it re-solves from
    scratch, so an overshoot costs one pass rather than a wrong answer. Guarded:
    a ratio that is not a contraction (r >= 0.98, or growing) means the sequence
    is not geometric here and the plain substitution stands.
    """
    if prev_step is None:
        return new
    n0 = float(np.linalg.norm(prev_step))
    n1 = float(np.linalg.norm(step))
    if n0 <= 0.0:
        return new
    r = n1 / n0
    if not (0.0 < r < 0.98):
        return new
    x = np.clip(new + r / (1.0 - r) * step, 0.0, None)
    s = float(x.sum())
    return x / s if s > 0.0 else new


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

    # ponytail: successive substitution with an Aitken jump (`_extrapolate`) --
    # plain substitution contracts by the returned fraction of the draw, which on
    # a real arrangement left both bundled examples still moving after 25 passes.
    # Each section keeps its return_comp between calls, so only the first solve of
    # a run pays the full tear and the operating-point root-find's later trials
    # need 2-3 passes. Upgrade path if a section ever still runs out: a fully
    # coupled Naphtali-Sandholm block instead of a tear.
    """
    def solve(si, **knobs):
        sub_knobs = {k: v for k, v in knobs.items()
                     if k in ("max_iter", "tol", "efficiency", "cancel")}
        # Every pass after the first warm-starts from the pass before, so the
        # caller's own warm start is consumed once and then dropped -- keeping it
        # in `knobs` made the re-solve pass x0 twice ("got multiple values for
        # keyword argument 'x0'"), which only ever fired on the GUI's threaded
        # path, the one place that warm-starts the final solve.
        pass_knobs = {k: v for k, v in knobs.items() if k not in ("x0", "T0")}
        # Seed the tear before the first solve. `rebuild` only puts a return feed
        # on the column once return_comp exists, so an unseeded first pass carries
        # the DRAW without the RETURN: the main column is solved short by the
        # recycle, which at best costs the tear a dozen passes crawling back and at
        # worst fails outright with "bottoms rate B=-5 must be positive" -- an
        # error that blames the user's distillate rate for a missing internal
        # stream. The external feed is the honest seed; one pass replaces it.
        if any(s.return_comp is None for s in sections):
            z = np.asarray(si.feed, float).sum(axis=0)
            z = z / z.sum() if z.sum() > 0 else z
            for s in sections:
                if s.return_comp is None:
                    s.return_comp = z.copy()
            si = rebuild(si)
        prof = solver(si, **knobs)
        subs = []
        moved = 0.0
        prev_step = {}                     # per-section step, for _extrapolate
        for _ in range(max_passes):
            if prof.get("message") == "Aborted.":
                break
            subs = [solve_section(s, prof, si, **sub_knobs) for s in sections]
            moved = 0.0
            for sec, sub in zip(sections, subs):
                new = _returned(sec, sub)
                # `moved` stays the raw fixed-point residual |G(x) - x|, not the
                # accelerated jump, so the convergence test means what it says.
                step = new - sec.return_comp
                moved = max(moved, float(np.max(np.abs(step))))
                sec.return_comp = _extrapolate(new, step, prev_step.get(sec.id))
                prev_step[sec.id] = step
            if moved < tol:
                break
            si = rebuild(si)
            prof = solver(si, x0=prof["x"], T0=prof["T"], **pass_knobs)
        prof["side_tear_residual"] = moved if subs else 0.0
        # A tear that ran out of passes used to return mid-crawl compositions
        # wearing the main solver's "Converged" message. Say it in the one place
        # the GUI already shows (Simulation status), because the side product and
        # everything above the return stage are wrong by `moved`, not by `tol`.
        if subs and moved >= tol:
            prof["message"] = (
                f"{prof.get('message', 'Solved')}  [side-section recycle NOT "
                f"converged: composition still moving {moved:.2e} after "
                f"{max_passes} passes]")
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

    # --- the cold start has to be mass-balanced -------------------------------
    # A draw big enough that D + draw > F is perfectly legal (most of it comes
    # straight back as the return), but only if the FIRST solve already carries
    # the return. Unseeded it died with "bottoms rate B=-5 must be positive".
    big = SideSection(id="SR2", kind="rectifier", draw_stage=6, return_stage=7,
                      rate=70.0, ratio=1.0, n_stages=4)

    def build_b(_si=None):
        feeds = [(8, F, z)]
        if big.return_comp is not None:
            feeds.append((big.return_stage, big.return_flow, big.return_comp, 1.0))
        liq, vap = big.draw_rates()
        return build_solver_input(
            n_stages=N, comps=comps, feeds=feeds,
            draws=[(big.draw_stage, liq, vap)],
            R=R, D=D, pressure=760.0, antoine=antoine)

    prof_b = make_side_solver(solve_bubble_point, [big], build_b)(build_b(),
                                                                 max_iter=300)
    assert abs(prof_b["B"] - (F - D - big.product_flow)) < 1e-6, prof_b["B"]
    assert prof_b["side_tear_residual"] < 1e-5, prof_b["side_tear_residual"]
    assert "NOT converged" not in prof_b["message"], prof_b["message"]
    print("side_sections self-check OK")


if __name__ == "__main__":
    _demo()
