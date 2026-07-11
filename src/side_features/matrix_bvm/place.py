"""Feed and draw stage placement (blueprint Sec 9).

Feed stage = operating-line crossover (Sec 9.1), the multicomponent
McCabe-Thiele "feed at the operating-line intersection." The upstream
product-anchored profile is fixed; at each stage the next liquid is stepped by
both the upstream and downstream operating lines, and we switch at the first
stage where the downstream line advances further toward the far product.
Switching early or late steps closer to equilibrium -> more trays / a pinch.

Side-draw stage = purity target on the inherited composition (Sec 9.3): a liquid
draw carries x_n, a vapour draw y_n, so a side spec is a constraint the profile
must meet. The product is capped by the purest the profile ever gets; if the
target is never reached, no stage works (raise R / add a side unit).
"""

import numpy as np


def crossover_stage(prof_up, prof_far):
    """Optimal feed/switch stage on the fixed upstream profile (Sec 9.1).

    The operating-line crossover is the composition where the upstream profile is
    closest to the downstream section's reachable profile -- the multicomponent
    generalisation of the McCabe-Thiele operating-line intersection. That is
    exactly the closest-approach junction (Sec 7). Returns the upstream stage
    index at the junction.
    """
    from connect import connect
    c = connect(prof_up, prof_far)
    return int(round(c["nA"]))


def total_stages_for_switch(prof_up, k, sec_dn, prof_far, tp, P, max_stages=200):
    """Total stages if the section switches at stage k (for comparing placements).

    Marches the downstream section by continuation from the upstream liquid at k
    and finds its closest approach to the far fixed profile. Returns
    (N_total, connected, dmin).
    """
    from march import march_section
    from connect import connect
    anchor = prof_up["X"][int(k)]
    dn = march_section(sec_dn, anchor, tp, P, max_stages=max_stages)
    c = connect(dn, prof_far)
    N = int(k + np.ceil(c["nA"]) + np.ceil(c["nB"]))
    return N, c["connected"], c["dmin"]


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
    from thermo_adapter import FreeColumnThermo
    from problem import build_problem, overall_balance
    from sections import single_feed_chain
    from march import march_section

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = FreeColumnThermo(abc)
    z = np.array([0.4, 0.35, 0.25])
    prob = build_problem(["b", "t", "x"], [(z, 100.0, 1.0)], 760.0,
                         rec_lk=0.98, rec_hk=0.02)
    xD, xB, D, B = overall_balance(prob)
    rect, strip = single_feed_chain(prob, 3.0, xD, xB, D, B)
    r = march_section(rect, xD, tp, 760.0, 60)
    s = march_section(strip, xB, tp, 760.0, 60)

    # optimal feed stage is interior to the rectifying profile (the crossover)
    k = crossover_stage(r, s)
    assert 0 < k < r["n"], k
    # it coincides with the connection junction (operating-line intersection)
    from connect import connect
    assert abs(k - connect(r, s)["nA"]) < 1.0

    # side draw: purity target capped by the profile
    from problem import SideDraw
    draw_hi = SideDraw(W=10.0, phase="L", comp_index=1, purity=0.99)  # unreachable
    res = side_draw_stage(s, draw_hi)
    assert res["capped"] and res["achieved"] < 0.99, res
    draw_lo = SideDraw(W=10.0, phase="L", comp_index=1, purity=0.4)
    res2 = side_draw_stage(s, draw_lo)
    assert not res2["capped"] and res2["achieved"] >= 0.4, res2
    print(f"place self-check OK  feed_stage(crossover)={k}")


if __name__ == "__main__":
    _demo()
