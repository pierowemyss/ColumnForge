"""Draw stage placement (blueprint Sec 9.3).

Side-draw stage = purity target on the inherited composition: a liquid draw
carries x_n, a vapour draw y_n, so a side spec is a constraint the profile must
meet. The product is capped by the purest the profile ever gets; if the target
is never reached, no stage works (raise R / add a side unit).

The feed-stage crossover that used to live here is gone: `connect` locates the
junction from the feed-stage vapour balance and `splits.solve_free_splits` solves for
it exactly, so a second, geometry-only placement rule had no caller left.
"""

import numpy as np


def side_draw_stage(prof, draw):
    """Locate a side draw by its purity target on the inherited composition.

    `prof` is a marched section dict; the draw carries x_n (liquid) or y_n
    (vapour). Returns dict(stage, achieved, capped): the stage where comp_index
    first reaches `purity`, or the profile's best stage with capped=True if the
    target is never met.
    """
    field = prof["X"] if draw.phase == "L" else prof["Y"]
    col = field[:, draw.comp_index]
    best = int(np.argmax(col))
    hit = np.where(col >= draw.purity)[0]
    if len(hit):
        s = int(hit[0])
        return {"stage": s, "achieved": float(col[s]), "capped": False}
    return {"stage": best, "achieved": float(col[best]), "capped": True}


def _demo():
    from .problem import SideDraw

    # a profile whose middle stages are richest in component 1
    prof = {"X": np.array([[0.8, 0.15, 0.05],
                           [0.5, 0.45, 0.05],
                           [0.2, 0.60, 0.20],
                           [0.05, 0.35, 0.60]]),
            "Y": np.array([[0.9, 0.08, 0.02],
                           [0.6, 0.35, 0.05],
                           [0.3, 0.60, 0.10],
                           [0.1, 0.45, 0.45]])}

    # target reached -> the FIRST stage that meets it, not the best one
    res = side_draw_stage(prof, SideDraw(W=10.0, phase="L", comp_index=1,
                                         purity=0.4))
    assert res == {"stage": 1, "achieved": 0.45, "capped": False}, res
    # target never reached -> the profile's best stage, flagged capped
    hi = side_draw_stage(prof, SideDraw(W=10.0, phase="L", comp_index=1,
                                        purity=0.99))
    assert hi["capped"] and hi["stage"] == 2 and hi["achieved"] == 0.60, hi
    # a vapour draw reads Y, so it can hit a target the liquid never does
    vap = side_draw_stage(prof, SideDraw(W=10.0, phase="V", comp_index=1,
                                         purity=0.55))
    assert not vap["capped"] and vap["stage"] == 2, vap
    print("place self-check OK  side draw @2 (capped), @1 (met), vapour @2")


if __name__ == "__main__":
    _demo()
