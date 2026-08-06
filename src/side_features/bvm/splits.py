"""Free non-key distillate splits -- seeding, solving, and the design family.

Two key recoveries fix two of the C component splits. The remaining C-2 are
genuine free parameters (`problem.free_split_indices`), and the classical BVM
statement is that they are determined by *requiring the section profiles to
meet*, not by a rule of thumb. `problem.overall_balance` only seeds them; this
module is where they get an actual value.

Three routes here, plus the trace ladder in `driver._size` underneath them all:

    fenske_split       total-reflux estimate, alpha at the product ends, with a
                       guard that refuses any component whose alpha crosses 1
                       between the ends. Cheap (one bubble point per end), no
                       marching, no iteration. A SEED, and measured against the
                       others in `driver._size` -- not wired in.
    solve_free_splits  least squares on the MARCHED profiles, matching the
                       sections at a fixed feed location. General but a shooting
                       method: its Jacobian is d(marched profile)/d(x_D), which
                       IS the 1/K amplification. Kept because it is the only
                       route that handles the extractive double junction at C=3,
                       and because it is what `spectrum` continues along.
    design_solved      the above with the feed location taken from the geometry
                       instead of the caller: size, read the junction, solve
                       there, re-size. ONE corrector step -- iterating it to a
                       fixed point diverges. This is what
                       `api.size_column(solve_splits=True)` calls.

`feed_loc` is the junction location on the rectifying profile -- the feed-tray
position, counted in stages from the distillate, fractional. It is an INPUT, held
fixed while the splits are solved. It used to be called `omega`, which read as
though it were being solved for; it never was.

Fixing the two key recoveries leaves C-2 free splits, and requiring the sections
to meet is C-1 equations per junction. Counting the unknowns THIS MODULE varies
-- the splits and the arc lengths, with every marched curve held fixed:

    simple column, any C   (C-2) splits + 1 arc  =  C-1  vs  C-1   square
    extractive, C = 3      (C-2) splits + 3 arcs =  4     vs  4     square
    extractive, C > 3      (C-2) splits + 3 arcs =  C+1   vs  2C-2  over by C-3

So `solve_free_splits` has a solution to find on a simple column at any C, and
above C=3 on an extractive one it does not: that deficit is why it stalls there
at a residual of 0.1-0.3 and walks the splits into the simplex corners. It
reports `residual` so the caller can tell which situation it is in rather than
hiding it in a tolerance.

That count is about THIS formulation, not about extractive columns. It is not
what `driver._size_three` does, and the difference matters: the sizing path does
not hold the interior curve fixed. It anchors the middle section on a saddle
pinch and chooses among the manifold's arms and launch points (`anchor`,
`driver._choose_interior`, pruned by `bodies`), and that choice carries its own
freedom -- enough that extract_col at C=6 closes BOTH its junctions exactly
(dmin < 1e-6, not `approximate`), where this module's least squares on the same
column cannot. A pinch-anchored interior section is the better-posed
construction, and the sizing path already uses it.

One equation short of pinning everything down means feasible designs form a
ONE-PARAMETER FAMILY indexed by `feed_loc`; `spectrum` sweeps it.
"""

import numpy as np
from scipy.optimize import least_squares

from .march import march_section
from .problem import overall_balance, free_split_indices
from .sections import single_feed_chain, extractive_chain

#: Smallest distillate split a seed may propose. Below this the marched profile
#: underflows before it travels anywhere, and it is the lowest rung of
#: `driver._TRACE_LADDER`, past which r_min has stopped moving (see `_size`).
SPLIT_FLOOR = 1e-12

#: How close to 1.0 a relative volatility may come before Fenske is refused for
#: that component. alpha^N_min is the whole content of the estimate, so an alpha
#: that crosses unity between the product ends makes it meaningless -- that is
#: an azeotrope or a tangent pinch in that pair, and the seed falls back.
ALPHA_GUARD = 1.05


def _alpha_at(x, tp, P, ref):
    """Relative volatilities K_i/K_ref at composition x. None if the flash fails."""
    try:
        y, T = tp.bubble(np.asarray(x, float), P)
    except (ValueError, FloatingPointError):
        return None
    x = np.clip(np.asarray(x, float), 1e-30, None)
    K = np.asarray(y, float) / x
    if not np.all(np.isfinite(K)) or K[ref] <= 0:
        return None
    return K / K[ref]


def fenske_split(prob, tp, P, xD, xB, ref=None):
    """Total-reflux estimate of every free non-key's distillate split.

    Fenske at total reflux gives each component its own d_i/b_i from
    alpha_i^N_min, where N_min comes from the two keys. Applied to the NON-KEYS
    it is a seed for the free splits, and a far better one than a flat trace
    floor: on the C=6 reference column the floor puts x_D,EC at 1.5e-6 where the
    converged answer is 3.9e-12, and marching a rectifying profile downward
    multiplies a heavy by ~1/K per stage (380x for EC), so that seed error is
    what sends the profile into the heavy corner within three stages.

    Two deliberate choices:

    * alpha is evaluated at the PRODUCT ENDS and combined geometrically, not at
      the feed. The down-march amplification is governed by K at the top, which
      is where the seed error compounds, so a feed-average alpha seeds the wrong
      number for the direction that matters.
    * a component whose alpha crosses 1 between the ends gets NO Fenske value.
      alpha^N_min is the entire estimate; if alpha passes through unity the
      estimate is not merely inaccurate, it is meaningless, and that crossing is
      exactly the signature of an azeotrope or tangent pinch in that pair. Those
      components fall back to whatever seed they already had.

    Returns (split, used) with `split` in `free_split_indices` order and `used` a
    bool mask saying which entries Fenske actually produced. Returns (None, None)
    when the keys themselves are unusable, which is the caller's cue to keep the
    trace floor for everything.
    """
    free = free_split_indices(prob)
    if not free:
        return None, None
    ref = prob.hk if ref is None else ref
    a_top = _alpha_at(xD, tp, P, ref)
    a_bot = _alpha_at(xB, tp, P, ref)
    if a_top is None or a_bot is None:
        return None, None

    # geometric mean of the two ends -- the standard Fenske average, and the
    # right one for a quantity that enters as a product over stages.
    prod = a_top * a_bot
    if np.any(prod <= 0):
        return None, None
    alpha = np.sqrt(prod)

    # N_min from the keys: alpha_lk^N_min = (d_lk/b_lk)(b_hk/d_hk)
    r_lk = float(np.clip(prob.rec_lk, 1e-9, 1 - 1e-9))
    r_hk = float(np.clip(prob.rec_hk, 1e-9, 1 - 1e-9))
    sep = (r_lk / (1 - r_lk)) * ((1 - r_hk) / r_hk)
    a_lk = float(alpha[prob.lk])
    if not np.isfinite(sep) or sep <= 0 or a_lk <= 0 or abs(np.log(a_lk)) < 1e-9:
        return None, None
    n_min = np.log(sep) / np.log(a_lk)
    if not np.isfinite(n_min) or n_min <= 0:
        return None, None

    # the guard: alpha must stay on one side of unity across the column
    crossed = ((a_top - 1.0) * (a_bot - 1.0) < 0.0) | (
        (np.abs(a_top - 1.0) < ALPHA_GUARD - 1.0)
        & (np.abs(a_bot - 1.0) < ALPHA_GUARD - 1.0))

    # ...and the second guard: HEAVIES ONLY. The whole argument for a sharper
    # seed is that marching the rectifying profile DOWN multiplies component i by
    # ~1/K_i per stage, so an over-generous trace of a heavy is amplified into the
    # heavy corner. For a light non-key the march decays it instead, so there is
    # no amplification to correct for, and the mirror-image risk is real: driving
    # a light's BOTTOMS content to zero would trap the stripping profile (marched
    # UP, which amplifies lights by K per stage) on the light-free face.
    #
    # Scoping only, not a fix: on the quaternary reference column restricting to
    # heavies moved the junction's vapour residual from 0.1256 to 0.1259, i.e.
    # not at all. What moves that case is the HEAVY sharpening itself (0.0151 at
    # the trace floor), which is exactly why this estimate is not wired into
    # `driver._size`. See the table in that docstring.
    crossed = crossed | (alpha >= 1.0)

    split = np.empty(len(free))
    used = np.zeros(len(free), bool)
    for i, k in enumerate(free):
        if crossed[k] or alpha[k] <= 0:
            split[i] = np.nan
            continue
        # d_i/b_i = alpha_i^N_min * (d_hk/b_hk); split = (d/b)/(1 + d/b)
        with np.errstate(over="ignore", divide="ignore"):
            ratio = alpha[k] ** n_min * (r_hk / (1 - r_hk))
        if not np.isfinite(ratio):
            split[i] = 1.0 - SPLIT_FLOOR      # overflow = goes overhead entirely
        else:
            split[i] = float(ratio / (1.0 + ratio))
        used[i] = True
    return np.clip(split, SPLIT_FLOOR, 1.0 - SPLIT_FLOOR), used


def current_split(prob, xD, D):
    """The free-split vector implied by a distillate: split_k = D x_D,k / f_k.

    The inverse of what `overall_balance` does with `split`, so a seed can be
    read back off whatever products the problem currently produces.
    """
    free = free_split_indices(prob)
    f = prob.z_total
    return np.array([SPLIT_FLOOR if f[k] <= 0 else
                     float(np.clip(D * xD[k] / f[k], SPLIT_FLOOR,
                                   1.0 - SPLIT_FLOOR))
                     for k in free])


def seed_split(prob, tp, P, xD, xB, D):
    """The starting free-split vector: Fenske where it applies, trace floor else.

    Extractive columns get the trace floor untouched. Fenske needs two product
    compositions at total reflux and constant alpha; an extractive section has
    no products, its difference point can sit outside the simplex, and the whole
    mechanism of the separation is that alpha is composition-dependent. Applying
    it there would be denying the premise. See `solve_free_splits` for what the
    extractive path uses instead.
    """
    free = free_split_indices(prob)
    if not free:
        return None
    base = current_split(prob, xD, D)
    if prob.extractive and prob.x_E is not None:
        return base
    fen, used = fenske_split(prob, tp, P, xD, xB)
    if fen is None:
        return base
    return np.where(used, fen, base)


def fenske_seed(prob, tp, EF=None):
    """The Fenske seed `driver._size` starts from, or None when it does not apply.

    None means "keep the trace floor": no free splits, an extractive column, a
    flash that would not converge, or every free component refused by the alpha
    guard. The caller then falls back exactly as it did before this existed.
    """
    if prob.extractive and prob.x_E is not None:
        return None
    from . import reactive
    if prob.reactions is not None or isinstance(tp, reactive.ReactiveThermo):
        # REACTIVE columns march in transformed coordinates (Ung-Doherty), where
        # a "component" is a reaction invariant and alpha_i is not a relative
        # volatility of anything. Fenske has nothing to say there, and the one
        # case that exists is the one that suffers most from a fine seed:
        # reactive MTBE leaves the simplex at 1e-5 and inverts its temperature
        # profile at 1e-6 (see `problem.overall_balance`). It wants the coarse
        # floor, so leave it alone.
        return None
    if not free_split_indices(prob):
        return None
    try:
        xD, xB, D, _B = overall_balance(prob, None)
    except (ValueError, FloatingPointError):
        return None
    fen, used = fenske_split(prob, tp, prob.pressure, xD, xB)
    if fen is None or not used.any():
        return None
    return np.where(used, fen, current_split(prob, xD, D))


def _at(prof, t):
    """Composition at fractional stage index t along a marched profile.

    Linearly EXTRAPOLATES past either end rather than clamping: a clamped index
    makes the residual flat in that direction, so a least-squares solver that
    steps off the end has no gradient to come back on and simply stalls there.
    `_range_penalty` is what keeps the answer on the curve.
    """
    X = prof["X"]
    last = len(X) - 1
    if last <= 0:
        return X[0]
    i = int(np.clip(np.floor(t), 0, last - 1))
    return X[i] + (float(t) - i) * (X[i + 1] - X[i])


def _range_penalty(prof, t, scale=1.0):
    """How far a fractional stage index falls outside its profile, 0 when inside."""
    last = len(prof["X"]) - 1
    return scale * (max(0.0, -float(t)) + max(0.0, float(t) - last))


def _profiles_for(prob, tp, R, EF, split, P, mid=None, bot=None):
    """March every section for a candidate free-split vector. Returns
    (chain, profiles, xD, xB, D, B) with profiles ordered top -> bottom.

    `mid` and `bot` supply pre-computed interior and stripping curves. Of the
    three, only the RECTIFYING profile depends strongly on the free split -- that
    is the whole point of solving for it, a trace of entrainer in the distillate
    amplifies downward -- while the other two shift by ~1e-3 across two decades of
    x_D,EG. Since every profile costs UNIFAC evaluations (the dominant expense by
    far), `solve_free_splits` freezes those two through each inner solve and
    refreshes them in an outer loop, re-measuring the final residual against
    fresh ones.
    """
    from .driver import (_choose_interior, _dP, _interior_profiles, _P_bot,
                         _P_mid)
    extractive = prob.extractive and prob.x_E is not None and EF
    xD, xB, D, B = overall_balance(prob, EF if extractive else None, split=split)
    E = prob.efficiency
    dP = _dP(prob)
    if extractive:
        rect, ext, strip = extractive_chain(prob, R, EF, xD, xB, D, B)
        tprof = march_section(rect, xD, tp, P, prob.max_stages, efficiency=E,
                              dP=dP, P_lim=_P_bot(prob), stop_sec=ext)
        bprof = bot if bot is not None else march_section(
            strip, xB, tp, _P_bot(prob), prob.max_stages, efficiency=E, dP=dP,
            P_lim=P)
        if mid is None:
            # the same double-junction choice `_size_three` makes -- an arbitrary
            # candidate would hand the least-squares a curve running the wrong way
            # up, which its top->bottom order penalty can then never satisfy.
            best, _, _ = _choose_interior(prob, tp, rect, ext, strip, tprof, bprof)
            if best is not None:
                mid = best[1]["mprof"]
            else:
                mids = _interior_profiles(ext, tp, _P_mid(prob), prob, tprof, bprof)
                mid = mids[0] if mids else None
        return (rect, ext, strip), (tprof, mid, bprof), xD, xB, D, B
    rect, strip = single_feed_chain(prob, R, xD, xB, D, B)
    tprof = march_section(rect, xD, tp, P, prob.max_stages, efficiency=E,
                          dP=dP, P_lim=_P_bot(prob))
    bprof = bot if bot is not None else march_section(
        strip, xB, tp, _P_bot(prob), prob.max_stages, efficiency=E, dP=dP,
        P_lim=P)
    return (rect, strip), (tprof, None, bprof), xD, xB, D, B


def solve_free_splits(prob, tp, R, feed_loc, EF=None, split0=None):
    """Solve for the free distillate splits that make the sections meet at feed_loc.

    `feed_loc` is the junction location on the rectifying profile -- the feed-tray
    position, counted in stages from the distillate. Holding it fixed leaves

        C-2 free splits  +  the remaining arc lengths

    against C-1 junction equations per junction, which is square for a two-section
    column at any C, and square for a three-section (extractive) column at C=3.
    Above that an exact double junction is over-determined and genuinely may not
    exist; we solve in least squares and hand back `residual` so the caller can
    say so instead of hiding it in a tolerance.

    This is the answer to "which distillate composition lets the sections
    intersect": there is a one-parameter family of them, indexed by feed_loc, and
    sweeping it (see `spectrum`) is the spectrum of feasible designs.

    ponytail: a shooting method, and it shows -- the Jacobian here is
    d(marched profile)/d(x_D), which IS the 1/K down-march amplification, so the
    solve is inverting the very operator that makes a bad seed catastrophic.
    `solve_split_pinch` is the well-conditioned route for a simple column at any
    C and is what `driver._size` uses. This one stays for the extractive double
    junction, which has no pinch-only formulation, and for `spectrum`.

    Returns dict(split, residual, feed_loc, converged) or None when there is
    nothing free to solve for (C == 2, or no interior curve to aim at).
    """
    free = free_split_indices(prob)
    if not free:
        return None
    P = prob.pressure
    C = prob.C
    extractive = bool(prob.extractive and prob.x_E is not None and EF)

    # unconstrained parameterisation: split = sigmoid(theta) keeps every free
    # split strictly inside (0, 1) without the solver ever having to be clipped.
    def to_split(theta):
        return 1.0 / (1.0 + np.exp(-np.clip(theta, -40.0, 40.0)))

    if split0 is None:
        base = overall_balance(prob, EF if extractive else None)[0]
        split0 = np.array([max(base[k], 1e-6) for k in free])
    s0 = np.clip(np.asarray(split0, float), 1e-9, 1 - 1e-9)
    theta0 = np.log(s0 / (1.0 - s0))

    def unpack(u):
        return to_split(u[:len(free)]), u[len(free):]

    def junction(sec_above, prof_above, t_above, prof_below, t_below):
        """Residual of (E): a x_above + b == K(x_below) x_below (connect.py)."""
        xa = _at(prof_above, t_above)
        xb = _at(prof_below, t_below)
        try:
            y_below, _ = tp.bubble(xb, P)
        except (ValueError, FloatingPointError):
            return np.ones(C)
        return (sec_above.a * xa + sec_above.bvec) - y_below

    def residual(u, mid=None, bot=None):
        split, ts = unpack(u)
        try:
            chain, (tprof, mprof, bprof), *_ = _profiles_for(
                prob, tp, R, EF, split, P, mid=mid, bot=bot)
        except (ValueError, FloatingPointError):
            return np.full(_n_res(), 1.0)
        if extractive:
            if mprof is None:
                return np.full(_n_res(), 1.0)
            rect, ext, _strip = chain
            r1 = junction(rect, tprof, feed_loc, mprof, ts[0])
            r2 = junction(ext, mprof, ts[1], bprof, ts[2])
            pen = (_range_penalty(mprof, ts[0]) + _range_penalty(mprof, ts[1])
                   + _range_penalty(bprof, ts[2]))
            order = max(0.0, ts[0] - ts[1])   # junctions must run top -> bottom
            return np.concatenate([r1[:C - 1], r2[:C - 1], [pen, order]])
        rect = chain[0]
        r = junction(rect, tprof, feed_loc, bprof, ts[0])[:C - 1]
        return np.concatenate([r, [_range_penalty(bprof, ts[0])]])

    def _n_res():
        return (2 * (C - 1) + 2) if extractive else C

    # Outer loop refreshes the interior curve; inner solve holds it fixed (see
    # `_profiles_for`). Two passes are enough -- the curve is nearly independent
    # of the free split -- and the final residual is always re-measured against a
    # freshly computed curve, so nothing is accepted on a stale one.
    best = None
    theta = theta0
    for _outer in range(2):
        try:
            _, (_t, _m, _b), *_ = _profiles_for(prob, tp, R, EF,
                                                to_split(theta), P)
        except (ValueError, FloatingPointError):
            return None
        if extractive and _m is None:
            return None
        span_m = max((_m["n"] - 1) if _m is not None else 1, 1)
        span_b = max(_b["n"] - 1, 1)
        # The junction system is multi-modal: with the interior curve tens of
        # stages long, one start lands in whichever local minimum it is nearest
        # and plateaus near 0.1 while an exact solution (1e-10) sits elsewhere on
        # the same curve. Spread the arc-length starts and keep the best.
        if extractive:
            starts = [np.array([a * span_m, b * span_m, c * span_b])
                      for a, b, c in ((0.02, 0.5, 0.1), (0.1, 0.9, 0.3),
                                      (0.3, 0.7, 0.05))]
        else:
            starts = [np.array([f * span_b]) for f in (0.2, 0.6, 0.05)]
        if best is not None:
            starts.insert(0, best[1][len(free):])      # continue from incumbent

        for t0 in starts:
            u0 = np.concatenate([theta, t0])
            try:
                sol = least_squares(residual, u0, args=(_m, _b), xtol=1e-10,
                                    ftol=1e-10, max_nfev=80)
            except (ValueError, FloatingPointError):
                continue
            res = float(np.linalg.norm(residual(sol.x)))   # fresh curves
            if best is None or res < best[0]:
                best = (res, sol.x, bool(sol.success))
            if res < 1e-8:
                break
        if best is None:
            return None
        if best[0] < 1e-8:
            break
        theta = best[1][:len(free)]

    res, u, ok = best
    split, ts = unpack(u)
    return {"split": split, "arc": ts, "residual": res,
            "feed_loc": float(feed_loc),
            "converged": ok and res < 1e-6, "free": free}


def design_at_feed(prob, tp, R, feed_loc, EF=None, split0=None):
    """Solve for the free splits at this feed-tray position and build the column.

    The design is assembled at the junction indices the solve produced, not by
    re-running the tolerance-based search over them -- otherwise an exactly solved
    junction can still be rejected by `connect`'s stage-width test. Returns
    (design, solution) with `design["exact"]` recording whether the junction
    equations actually closed.
    """
    from .driver import _size
    sol = solve_free_splits(prob, tp, R, feed_loc, EF=EF, split0=split0)
    if sol is None:
        return _size(prob, tp, R, EF=EF), None
    extractive = bool(prob.extractive and prob.x_E is not None and EF)
    P = prob.pressure
    forced = None
    if sol["converged"]:
        arc = sol["arc"]
        try:
            _, (tprof, mprof, bprof), *_ = _profiles_for(prob, tp, R, EF,
                                                         sol["split"], P)
        except (ValueError, FloatingPointError):
            tprof = mprof = bprof = None
        if extractive and mprof is not None:
            i_lo, i_hi = int(np.floor(arc[0])), int(np.floor(arc[1]))
            if 0 <= i_lo <= i_hi < mprof["n"]:
                forced = (mprof, int(np.floor(feed_loc)), i_lo, i_hi,
                          int(np.floor(arc[2])))
        elif not extractive:
            forced = (int(np.floor(feed_loc)), int(np.floor(arc[0])))
    d = _size(prob, tp, R, EF=EF, split=sol["split"], forced=forced)
    d["feed_loc"] = float(feed_loc)
    d["split"] = sol["split"]
    d["junction_residual"] = sol["residual"]
    d["exact"] = bool(sol["converged"] and forced is not None)
    return d, sol


def design_solved(prob, tp, R, S=None, EF=None):
    """Size, then solve the free splits AT the feed location that sizing found.

    The free splits and the feed location are not independent -- they are the two
    coordinates of the one-parameter family (`spectrum`), so "the" solved split
    only means something once a feed location is named. `design_at_feed` needs the
    caller to name one. This names it from the geometry instead: size at the
    seeded splits, read where the junction landed, solve the splits there, and
    re-size on the result.

    ONE corrector step, deliberately, not a fixed-point iteration. Iterating the
    equivalent amplification relation to convergence DIVERGES: a smaller x_D makes
    the rectifying profile take more stages to reach the junction, which pushes
    x_D smaller again. Measured on c2-c4 it runs to 6e-21 at R=0.15 and 7e-2 at
    R=2.0 -- the same 1/K positive feedback that makes the seed matter in the
    first place, now in the outer loop. A single pass has no such loop to run
    away in, and if the solve fails or makes things worse the first design stands.

    Cost is one `solve_free_splits` (6-8 s on c2-c4, C=3). That is affordable once
    per user-facing size and NOT affordable inside `driver.r_min`'s bisection,
    which is why this is opt-in and the reflux limits stay on the cheap path.
    Above C=3 the solve frequently does not converge (it drives the splits to the
    simplex corners); the design then keeps its seeded splits and says so through
    `exact` / `junction_residual` rather than pretending.
    """
    from .driver import _size
    first = _size(prob, tp, R, S=S, EF=EF)

    def _kept(why):
        """The seeded design, saying why it is still the seeded one."""
        first["split_solved"] = False
        first["split_solve_note"] = why
        return first

    if not free_split_indices(prob):
        return _kept("no free splits: the spec fixes every component")
    conn = first.get("connection") or {}
    loc = conn.get("nA")
    if loc is None or not np.isfinite(loc):
        return _kept("no junction to solve at: the sections never connected")
    if loc < 1.0:
        # the rectifying section came out shorter than a stage, so there is no
        # interior feed position to hold fixed. Common at C >= 4, where a heavy
        # non-key seeded too high collapses the rectifying march onto its anchor.
        return _kept(f"rectifying section is {loc:.2f} stages: no interior feed "
                     f"position to solve at")
    try:
        solved, sol = design_at_feed(prob, tp, R, float(loc), EF=EF)
    except (ValueError, FloatingPointError):
        return _kept("the split solve raised on this geometry")
    if sol is None:
        return _kept("nothing free to solve for at this feed position")
    if not sol.get("converged"):
        first["junction_residual"] = sol["residual"]
        return _kept(f"split solve did not converge (residual {sol['residual']:.3g}); "
                     f"above C=3 it tends to drive the splits to the simplex corners")
    if not solved.get("feasible"):
        first["junction_residual"] = sol["residual"]
        return _kept("the solved splits closed the junction but the design they "
                     "give is infeasible; keeping the seeded one")
    solved["split_solved"] = True
    solved["split_solve_note"] = f"solved at feed position {loc:.2f}"
    return solved


def spectrum(prob, tp, R, feed_locs, EF=None):
    """Sweep the feed-tray position: N_total and the solved x_D at each location.

    This is the spectrum of designs. Fixing the two key recoveries leaves C-2 free
    distillate splits, and requiring the sections to meet is C-1 equations per
    junction -- one short of determining everything, so feasible designs come as a
    ONE-PARAMETER FAMILY indexed by where the feed tray sits. For each feed_loc
    there is a unique distillate composition that makes the sections intersect;
    N_total against feed_loc has an interior minimum at the best feed location.

    Each position is warm-started from the previous solution, so the sweep is one
    continuation rather than N cold solves. Returns a list of dicts ordered by
    feed_loc.
    """
    rows = []
    split0 = None
    for w in np.atleast_1d(feed_locs):
        d, sol = design_at_feed(prob, tp, R, float(w), EF=EF, split0=split0)
        if sol is None:
            continue
        split0 = sol["split"]              # continuation
        rows.append({"feed_loc": float(w), "split": sol["split"],
                     "residual": sol["residual"], "exact": d["exact"],
                     "feasible": d["feasible"], "N_total": d["N_total"],
                     "feed_stages": d["feed_stages"], "xD": d["xD"],
                     "xB": d["xB"], "findings": d["findings"],
                     "free_indices": list(sol["free"])})
    return rows


def _demo():
    from .problem import build_problem
    from .thermo_adapter import ColumnForgeThermo

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = ColumnForgeThermo(abc)
    z = np.array([0.4, 0.35, 0.25])
    prob = build_problem(["benzene", "toluene", "xylene"], [(z, 100.0, 1.0)],
                         760.0, rec_lk=0.98, rec_hk=0.02)
    xD, xB, D, B = overall_balance(prob)

    # Fenske puts the heavy non-key far below the trace floor, which is the whole
    # point: 1e-4 of xylene overhead is a seed, not a distillate composition.
    fen, used = fenske_split(prob, tp, 760.0, xD, xB)
    assert fen is not None and used.all(), (fen, used)
    assert fen[0] < 1e-4, f"xylene split should be well under the trace floor: {fen}"

    # the guard: a component whose alpha straddles 1 gets no Fenske value. Fake it
    # by making the reference the component itself -- alpha == 1 identically.
    _, used_self = fenske_split(prob, tp, 760.0, xD, xB, ref=2)
    assert not used_self[0], "alpha == 1 must be refused, not extrapolated"

    # the corrector step: it must either solve, or say why it did not -- never
    # come back looking solved when it is still sitting on the seed.
    got = design_solved(prob, tp, 4.0)
    assert "split_solved" in got and "split_solve_note" in got, sorted(got)
    if got["split_solved"]:
        assert got["exact"] and got["junction_residual"] < 1e-6, got["junction_residual"]
    else:
        assert got["split_solve_note"], "a refusal must carry its reason"

    # the family: each feed position has its OWN distillate composition, solved so
    # the sections meet, and N_total varies over it.
    rows = spectrum(prob, tp, 4.0, [3, 4, 5, 6, 7])
    exact = [r for r in rows if r["exact"]]
    assert len(exact) >= 3, [r["residual"] for r in rows]
    assert all(r["residual"] < 1e-6 for r in exact)
    xd = [r["xD"][2] for r in exact]
    assert max(xd) > 3 * min(xd), f"x_D should vary across the family: {xd}"
    assert len({r["N_total"] for r in exact}) > 1, "N should vary with feed position"
    print(f"splits self-check OK  fenske={fen[0]:.2e}  "
          f"family N={[r['N_total'] for r in exact]} "
          f"xD_nonkey={[f'{v:.1e}' for v in xd]}")


if __name__ == "__main__":
    _demo()
