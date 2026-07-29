"""Reactive distillation via transformed compositions (blueprint Sec 15).

Reaction is folded into *reaction-invariant* transformed composition variables
(Ung-Doherty transform, a linear-rational map from the stoichiometry). In the
transformed space the ordinary geometry -- difference points, operating lines,
marching, pinch and connection tests -- applies unchanged, because the transform
absorbs the reaction extent: two states differing only by reaction map to the
same transformed composition.

For R reactions over C components with stoichiometric matrix nu (R x C) and R
reference components, the transform is

    X_i = (x_i - nu_i^T Nref^{-1} x_ref) / (1 - nuT^T Nref^{-1} x_ref)

where Nref is nu restricted to the reference columns (R x R), nu_i the i-th
column, and nuT = sum_i nu_i (net mole change). The X_i sum to 1 and the R
reference components are redundant (X_ref == 0 identically), so a one-reaction
system over C components is a (C-1)-component problem in *reduced* transformed
coordinates -- and that is what makes this cheap: `ReactiveThermo` presents
bubble/dew/bubble_T in those coordinates, and march/sections/connect/pinch/place
are then the same code as for a non-reactive column.

Two pieces are needed beyond the coordinate map:

  * the stagewise closure. A transformed X does not fix the physical liquid; the
    reaction-equilibrium condition does. Along the reaction line through X the
    physical compositions are, with lam = x_ref / nu_ref (one reaction),

        x_i(lam) = X_i (1 - nuT lam) + nu_i lam         (sums to 1 for any lam)

    and `equilibrium_state` roots  prod_i (gamma_i x_i)^nu_i = Keq(T)  in lam at
    the mixture's own bubble temperature. The reported extent is
    xi = lam / (1 - nuT lam), moles reacted per mole of transformed stream.
  * transformed flow rates. A stream (x, F) carries F_bar = F * denom(x); with
    that scaling the transformed component balance closes *exactly* even though
    the physical one does not (the reaction term cancels), which is why the
    ordinary overall_balance can be reused. For a total condenser the reflux and
    the distillate share a composition, so R_bar == R with no conversion.

ponytail: one equilibrium reaction, ideal stages, chemical equilibrium on every
stage (no reactive-holdup / kinetic closure, no partially-reactive stage range --
a reaction-free zone breaks the invariance that lets the geometry be reused).
Kinetic closures slot in at `equilibrium_state`; multiple reactions need the
R-dimensional extent solve in place of the 1-D bracket.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, fsolve


@dataclass
class Reactions:
    nu: np.ndarray        # (R, C) stoichiometric coefficients (products +, reactants -)
    ref: np.ndarray       # (R,) indices of the reference components
    keq_fn: object = None  # callable T -> Keq (T in the Antoine unit); None = geometry only

    def __post_init__(self):
        self.nu = np.atleast_2d(np.asarray(self.nu, float))
        self.ref = np.asarray(self.ref, int)
        assert self.nu.shape[0] == len(self.ref), "one reference per reaction"

    @property
    def free(self):
        """Non-reference component indices -- the reduced transformed coordinates."""
        ref = set(int(i) for i in self.ref)
        return np.array([i for i in range(self.nu.shape[1]) if i not in ref], int)


def keq_arrhenius(A, B=0.0):
    """Keq(T) = exp(A + B/T[K]) from two numbers; B = 0 is a constant Keq."""
    def keq(T):
        return math.exp(A + B / (float(T) + 273.15))
    return keq


def _Nref_inv(rx):
    Nref = rx.nu[:, rx.ref]                     # (R, R)
    return np.linalg.inv(Nref)


def transform(x, rx):
    """Physical composition x (C,) -> reaction-invariant X (C,), sums to 1."""
    x = np.asarray(x, float)
    Ninv = _Nref_inv(rx)
    xref = x[rx.ref]
    nuT = rx.nu.sum(axis=1)                      # (R,) net mole change per reaction
    denom = 1.0 - nuT @ (Ninv @ xref)
    X = (x - rx.nu.T @ (Ninv @ xref)) / denom
    return X


def inverse_transform(X, rx):
    """Reaction-invariant X (C,) -> physical composition x (C,) at zero extent.

    Recovers the representative physical composition on the reaction surface with
    the reference components consistent with X (the extent-free representative).
    """
    X = np.asarray(X, float)
    # With reference extents zero, x_i = X_i * denom + nu_i^T Ninv x_ref, and the
    # reference rows fix x_ref self-consistently. For zero reference extent the
    # representative is x_ref = X_ref renormalised; then propagate.
    x = X.copy()
    x = np.clip(x, 0, None)
    s = x.sum()
    return x / s if s > 0 else x


def apply_reaction(x, rx, extent):
    """Advance composition by a reaction extent vector (R,); renormalise.

    Used only to *demonstrate* invariance: transform(x) is unchanged by this.
    """
    x = np.asarray(x, float) + rx.nu.T @ np.asarray(extent, float)
    x = np.clip(x, 0, None)
    return x / x.sum()


def denom(x, rx):
    """Transform denominator at x -- also the transformed/physical flow ratio."""
    x = np.asarray(x, float)
    nuT = rx.nu.sum(axis=1)
    return float(1.0 - nuT @ (_Nref_inv(rx) @ x[rx.ref]))


def reduce_X(X, rx):
    """Full transformed X (C,) -> reduced coordinates (C-R,), renormalised."""
    Xr = np.asarray(X, float)[rx.free]
    s = Xr.sum()
    return Xr / s if s > 0 else Xr


def expand_X(Xr, rx):
    """Reduced coordinates (C-R,) -> full transformed X (C,) with X_ref = 0."""
    X = np.zeros(rx.nu.shape[1])
    X[rx.free] = np.asarray(Xr, float)
    return X


# ---------------------------------------------------------------- the closure

def _one_reaction(rx):
    if rx.nu.shape[0] != 1:
        raise NotImplementedError(
            "reactive sizing handles one reaction; multiple reactions need the "
            "R-dimensional extent solve in equilibrium_state.")
    return int(rx.ref[0])


def _lam_bracket(X, rx):
    """Feasible range of lam = x_ref/nu_ref keeping every x_i(lam) >= 0.

    x_i(lam) = X_i + lam (nu_i - nuT X_i), so each component gives one linear
    bound; the reference component (X_ref = 0) fixes the sign of lam.
    """
    nu = rx.nu[0]
    nuT = float(nu.sum())
    slope = nu - nuT * X
    lo, hi = -np.inf, np.inf
    for Xi, si in zip(X, slope):
        if abs(si) < 1e-14:
            if Xi < -1e-12:
                return None
            continue
        bound = -Xi / si
        if si > 0:
            lo = max(lo, bound)
        else:
            hi = min(hi, bound)
    return None if not lo < hi else (lo, hi)


def physical_from_lam(X, rx, lam):
    """x_i(lam) = X_i (1 - nuT lam) + nu_i lam, clipped and renormalised."""
    nu = rx.nu[0]
    x = np.asarray(X, float) * (1.0 - float(nu.sum()) * lam) + nu * lam
    x = np.clip(x, 0.0, None)
    s = x.sum()
    return x / s if s > 0 else x


def equilibrium_state(X, rx, tp, P):
    """Transformed X (C,) -> (x, T, extent) at chemical + phase equilibrium.

    Roots  ln prod_i (gamma_i x_i)^nu_i = ln Keq(T)  in lam along the reaction
    line through X, with T the bubble temperature of the trial liquid. Raises
    ValueError when no bracket exists -- the march then treats the step the same
    way it treats a missing saturation root (it left the physical region).
    """
    _one_reaction(rx)
    if rx.keq_fn is None:
        raise ValueError("reactive sizing needs an equilibrium constant: "
                         "set Reactions.keq_fn (see keq_arrhenius).")
    X = np.asarray(X, float)
    nu = rx.nu[0]
    nuT = float(nu.sum())
    bracket = _lam_bracket(X, rx)
    if bracket is None:
        raise ValueError("no physical reaction line through this transformed point")
    lo, hi = bracket
    span = hi - lo
    # step just inside the ends: at a bound some x_i hits 0, so ln Ka -> -+inf
    lo_i, hi_i = lo + 1e-9 * span, hi - 1e-9 * span

    def g(lam):
        x = physical_from_lam(X, rx, lam)
        T = tp.bubble_T(x, P)
        gam = np.ones_like(x) if tp.gamma_fn is None else np.asarray(tp.gamma_fn(x, T), float)
        a = np.clip(gam * x, 1e-300, None)
        return float(nu @ np.log(a)) - math.log(float(rx.keq_fn(T)))

    g_lo, g_hi = g(lo_i), g(hi_i)
    if g_lo * g_hi > 0:
        # no sign change: equilibrium lies outside the physical line -> the
        # closest feasible end is the answer (fully converted / fully unreacted)
        lam = lo_i if abs(g_lo) < abs(g_hi) else hi_i
    else:
        lam = brentq(g, lo_i, hi_i, xtol=1e-12, rtol=1e-10, maxiter=200)
    x = physical_from_lam(X, rx, lam)
    T = tp.bubble_T(x, P)
    extent = lam / (1.0 - nuT * lam) if abs(1.0 - nuT * lam) > 1e-12 else float("inf")
    return x, float(T), float(extent)


class ReactiveThermo:
    """A ThermoProvider in *reduced transformed* coordinates (C-1 of them).

    Wraps a physical provider (`ColumnForgeThermo`): every stage liquid is put at
    reaction equilibrium by `equilibrium_state`, flashed with the real VLE, and
    the resulting vapour is transformed back. The vapour is NOT assumed to be at
    reaction equilibrium -- only the liquid is -- which is the Ung-Doherty
    statement of a reactive equilibrium stage.
    """

    def __init__(self, tp, rx):
        _one_reaction(rx)
        self.tp = tp
        self.rx = rx
        self.n_comps = len(rx.free)
        self.gamma_fn = getattr(tp, "gamma_fn", None)
        self._cache = {}
        self._last_dew = None          # continuation warm start for dew()

    # -- physical state behind a reduced transformed liquid ----------------
    def physical(self, Xr, P):
        """(x, T, extent) of the physical liquid behind reduced transformed Xr."""
        Xr = np.asarray(Xr, float)
        key = (Xr.tobytes(), float(P))
        hit = self._cache.get(key)
        if hit is None:
            if len(self._cache) > 4096:            # ponytail: plain bounded dict
                self._cache.clear()
            hit = equilibrium_state(expand_X(Xr, self.rx), self.rx, self.tp, P)
            self._cache[key] = hit
        return hit

    # -- ThermoProvider surface the march actually calls -------------------
    def bubble(self, Xr, P):
        """Reduced transformed liquid -> (transformed vapour, T)."""
        Xr = np.asarray(Xr, float)
        s = Xr.sum()
        Xr = Xr / s if s > 0 else Xr
        x, T, _ = self.physical(Xr, P)
        y, Ty = self.tp.bubble(x, P)
        return reduce_X(transform(y, self.rx), self.rx), float(Ty)

    def dew(self, Yr, P, x_seed=None):
        """Reduced transformed vapour -> (conjugate transformed liquid, T).

        Inverts `bubble`: no closed form once the reaction closure sits inside, so
        solve the (C-2)-dimensional system as march._dew_eff does for a Murphree
        stage -- but in log coordinates, Xr = softmax(u). The conjugate liquid of a
        transformed vapour routinely sits *on* a face (the top of a reactive column
        wants X_excess-reactant ~ 0 while its vapour carries plenty), and a solve
        parameterised in Xr directly stalls on the clipping plateau there. In u the
        face is at -inf and the iteration stays smooth. Warm-started from the
        previous step, because a march is a continuation.
        """
        Yr = np.asarray(Yr, float)
        s = Yr.sum()
        Yr = Yr / s if s > 0 else Yr
        n = self.n_comps

        def to_X(u):
            e = np.exp(np.clip(np.concatenate([u, [0.0]]), -60.0, 60.0))
            return e / e.sum()

        def to_u(X):
            lg = np.log(np.clip(np.asarray(X, float), 1e-24, None))
            return (lg - lg[-1])[:n - 1]

        def resid(u):
            return (self.bubble(to_X(u), P)[0] - Yr)[:n - 1]

        best = None
        starts = [to_u(Yr)]
        # `x_seed` is the caller's actual stage above; `_last_dew` is only
        # whatever this provider happened to solve last, which after a section
        # switch belongs to a different profile. Prefer the explicit one.
        for prev in (self._last_dew, x_seed):
            if prev is not None:
                starts.insert(0, to_u(prev))
        starts.append(np.zeros(n - 1))                 # equimolar fallback
        for u0 in starts:
            u, _, _, msg = fsolve(resid, u0, xtol=1e-12, full_output=True)
            err = float(np.linalg.norm(resid(u)))
            if best is None or err < best[0]:
                best = (err, u)
            if err < 1e-9:
                break
        err, u = best
        if err > 1e-6:
            raise ValueError(f"reactive dew step did not converge (residual "
                             f"{err:.2e}): {msg}")
        Xr = to_X(u)
        self._last_dew = Xr
        return Xr, self.bubble_T(Xr, P)

    def bubble_T(self, Xr, P):
        Xr = np.asarray(Xr, float)
        s = Xr.sum()
        Xr = Xr / s if s > 0 else Xr
        return self.physical(Xr, P)[1]

    def K(self, x, T, P):
        raise NotImplementedError(
            "K in transformed coordinates is not defined for a Murphree stage: "
            "efficiency < 1 blends vapour compositions affinely, and the "
            "transform is rational. Reactive sizing runs ideal stages only.")


# ------------------------------------------------------- problem translation

_TRACE = 1e-4


def simplex_safe(rx):
    """Can every physical composition map inside the transformed simplex?

    X_i = x_i - (nu_i/nu_ref) x_ref, so X_i >= 0 for *all* physical x exactly when
    nu_i and nu_ref have opposite signs (or nu_i = 0) for every other component.
    That holds for a one-product reaction with the product as reference --
    etherification (MTBE/ETBE/TAME), hydration (EO + H2O -> EG), hydrogenation --
    and fails for a two-product reaction such as an esterification, where ester
    and water are both products: whichever one is the reference, the other's
    coordinate goes negative once it gets rich, which is exactly what a reactive
    column's two ends do.

    Returns (ok, message). Marching, pinch and connection tests all assume the
    simplex, so a False here means this chemistry is outside what the geometry can
    represent -- not a numerical difficulty to be tuned away.
    """
    nu = rx.nu[0] if rx.nu.shape[0] == 1 else None
    if nu is None:
        return True, ""                      # multi-reaction: guarded elsewhere
    ref = int(rx.ref[0])
    bad = [i for i in range(len(nu)) if i != ref and nu[i] * nu[ref] > 0]
    if not bad:
        return True, ""
    return False, (
        f"stoichiometry {nu.tolist()} with component {ref} as reference: "
        f"components {bad} share the reference's sign, so their transformed "
        f"coordinates go negative for reactant/product-rich streams. The "
        f"difference-point geometry needs 0 <= X <= 1, so such a design will be "
        f"reported infeasible (or rejected) rather than mis-sized. A one-product "
        f"reaction with the product as reference has no such limit.")


def _trace_floor(Zr, prob, ref):
    """Keep the transformed feed off an exactly-zero coordinate.

    A stoichiometric feed lands a transformed coordinate at exactly 0 (feeding
    MeOH and AcOH 1:1 gives transformed-MeOH = 0), and then both product anchors
    sit on that face with nothing able to leave it: the operating line prescribes
    Y = a X + b, so X_i = 0 forces Y_i = 0, and no real equilibrium vapour has
    X_i = 0 when the two components it differences have different volatilities.
    The march dies on its anchor stage. Same 1e-4 trace convention (and the same
    reason) as the non-distributing floor in problem.overall_balance: physically,
    a hair of excess reactant.

    A genuinely *negative* coordinate is not a trace -- it means this reference
    component puts the feed outside the transformed simplex, which the geometry
    cannot represent, so say so.
    """
    Zr = np.asarray(Zr, float)
    if Zr.min() < -1e-9:
        raise ValueError(
            f"the transform puts this feed outside the transformed simplex "
            f"(X = {np.round(Zr, 4).tolist()}) with {prob.comps[ref]!r} as the "
            f"reaction reference; BVM's geometry needs 0 <= X <= 1, so choose a "
            f"different reference component.")
    Zr = np.clip(Zr, _TRACE, None)
    return Zr / Zr.sum()


def transform_problem(prob, tp):
    """Physical reactive `Problem` -> (reduced transformed Problem, ReactiveThermo).

    Compositions map through the transform, flows scale by `denom`, and the
    reference component drops out. The reflux ratio is unchanged (see module doc).
    Anything the transform cannot carry honestly raises instead of being ignored.
    """
    from .problem import Feed, Problem          # local: problem.py must not need us

    rx = prob.reactions
    ref = _one_reaction(rx)
    if rx.keq_fn is None:
        raise ValueError("reactive sizing needs an equilibrium constant: "
                         "set Reactions.keq_fn (see keq_arrhenius).")
    if float(prob.efficiency) != 1.0:
        raise NotImplementedError(
            "reactive sizing runs ideal stages; a Murphree efficiency < 1 is not "
            "defined in transformed coordinates (see ReactiveThermo.K).")
    if prob.extractive and prob.x_E is not None:
        raise NotImplementedError("reactive + extractive (entrainer) is not wired yet")
    if prob.side_draws:
        raise NotImplementedError("side draws on a reactive column are not wired yet")
    ok, why = simplex_safe(rx)
    if not ok:
        warnings.warn(f"reactive geometry limit -- {why}", stacklevel=2)

    free = rx.free
    if len(free) < 3:
        raise NotImplementedError(
            f"one reaction over {prob.C} components leaves a {len(free)}-component "
            "transformed problem, and connection by closest approach is degenerate "
            "there: every profile lies on the same line, so the junction (and with "
            "it the stage count and feed location) is arbitrary. A reactive column "
            "needs at least 4 components for one reaction; a genuinely binary "
            "column belongs to McCabe-Thiele.")
    comps_r = [prob.comps[i] for i in free]
    feeds_r = []
    for fd in prob.feeds:
        z = np.asarray(fd.z, float)
        dn = denom(z, rx)
        if dn <= 1e-9:
            raise ValueError(
                f"feed composition sits at the transform singularity (denominator "
                f"{dn:.3g}); pick a different reference component than "
                f"{prob.comps[ref]!r}.")
        feeds_r.append(Feed(z=_trace_floor(reduce_X(transform(z, rx), rx), prob, ref),
                            F=fd.F * dn, q=fd.q))

    kw = {}
    if prob.xD is not None and prob.xB is not None:
        kw["xD"] = reduce_X(transform(np.asarray(prob.xD, float), rx), rx)
        kw["xB"] = reduce_X(transform(np.asarray(prob.xB, float), rx), rx)
        lk_r = hk_r = 0                      # unused on the explicit-product path
    else:
        if prob.lk == ref or prob.hk == ref:
            raise ValueError(
                f"{prob.comps[ref]!r} is the reaction reference component, so it "
                "has no transformed recovery; choose other keys or give explicit "
                "xD/xB.")
        idx = {int(i): k for k, i in enumerate(free)}
        lk_r, hk_r = idx[prob.lk], idx[prob.hk]
        kw["rec_lk"], kw["rec_hk"] = prob.rec_lk, prob.rec_hk
        if prob.nonkey_to_dist is not None:
            kw["nonkey_to_dist"] = np.asarray(prob.nonkey_to_dist, float)[free]

    prob_r = Problem(comps=comps_r, feeds=feeds_r, pressure=prob.pressure,
                     lk=lk_r, hk=hk_r, dP=prob.dP, P_bot=prob.P_bot,
                     eps_stage=prob.eps_stage, max_stages=prob.max_stages,
                     trace_floor=prob.trace_floor,
                     entrainer_trace=prob.entrainer_trace, **kw)
    return prob_r, ReactiveThermo(tp, rx)


def physical_profile(Xr_rows, rx, tp, P):
    """Reduced transformed liquid profile (N, C-1) -> physical (x (N,C), T, extent)."""
    xs, Ts, ex = [], [], []
    for Xr in np.atleast_2d(np.asarray(Xr_rows, float)):
        try:
            x, T, e = equilibrium_state(expand_X(Xr, rx), rx, tp, P)
        except ValueError:                     # off the physical line: report the gap
            x, T, e = np.full(rx.nu.shape[1], np.nan), np.nan, np.nan
        xs.append(x); Ts.append(T); ex.append(e)
    return np.array(xs), np.array(Ts), np.array(ex)


def _demo():
    # Methyl acetate esterification: MeOH + AcOH <-> MeOAc + H2O
    # order: [MeOH, AcOH, MeOAc, H2O], one reaction, reference = MeOAc (idx 2)
    nu = np.array([[-1.0, -1.0, 1.0, 1.0]])     # (1, 4)
    rx = Reactions(nu=nu, ref=[2])

    x = np.array([0.3, 0.3, 0.2, 0.2])
    X = transform(x, rx)
    assert abs(X.sum() - 1.0) < 1e-9, X.sum()

    # reaction-invariance: run the reaction forward, X is unchanged
    x2 = apply_reaction(x, rx, extent=[0.05])
    X2 = transform(x2, rx)
    assert np.allclose(X, X2, atol=1e-9), (X, X2)

    # a different extent, still the same transformed point
    x3 = apply_reaction(x, rx, extent=[-0.08])
    assert np.allclose(transform(x3, rx), X, atol=1e-9)

    # transformed composition stays a valid (summing) coordinate for a sweep of x
    rng = np.random.default_rng(0)
    for _ in range(20):
        xr = rng.dirichlet(np.ones(4))
        Xr = transform(xr, rx)
        assert abs(Xr.sum() - 1.0) < 1e-9

    # -- the stagewise closure ------------------------------------------------
    from .thermo_adapter import ColumnForgeThermo

    # MeOH / AcOH / MeOAc / H2O, bundled Antoine fits (log10 mmHg, degC)
    antoine = np.array([(8.08097, 1582.271, 239.726),     # methanol
                        (7.38782, 1533.313, 222.309),     # acetic acid
                        (7.06524, 1157.63, 219.726),      # methyl acetate
                        (8.07131, 1730.63, 233.426)])     # water
    rxk = Reactions(nu=nu, ref=[2], keq_fn=keq_arrhenius(math.log(5.2)))
    tp = ColumnForgeThermo(antoine)                        # ideal gamma for the check
    P = 760.0

    # equilibrium closure: Ka == Keq at the solved lam, and the transformed
    # composition of the solved liquid is the X we asked for (invariance holds)
    X0 = transform(np.array([0.4, 0.4, 0.1, 0.1]), rx)
    x_eq, T_eq, xi = equilibrium_state(X0, rxk, tp, P)
    Ka = np.prod(np.clip(x_eq, 1e-300, None) ** nu[0])
    assert abs(Ka - 5.2) / 5.2 < 1e-6, (Ka, x_eq)
    assert np.allclose(transform(x_eq, rxk), X0, atol=1e-9), (transform(x_eq, rxk), X0)
    assert 50.0 < T_eq < 130.0, T_eq
    assert np.isfinite(xi)

    # reduce/expand are inverses on the free coordinates, and X_ref == 0
    assert abs(X0[2]) < 1e-12
    assert np.allclose(expand_X(reduce_X(X0, rxk), rxk), X0, atol=1e-12)

    # the reduced-coordinate provider: dew inverts bubble
    tpr = ReactiveThermo(tp, rxk)
    assert tpr.n_comps == 3
    Xr = reduce_X(X0, rxk)
    Yr, Tb = tpr.bubble(Xr, P)
    assert abs(Yr.sum() - 1.0) < 1e-9 and abs(Tb - T_eq) < 1e-6
    Xback, _ = tpr.dew(Yr, P)
    assert np.allclose(Xback, Xr, atol=1e-6), (Xback, Xr)
    # MeOAc is the light product: it enriches the vapour relative to the liquid
    y_phys, _ = tp.bubble(x_eq, P)
    assert y_phys[2] > x_eq[2], (y_phys, x_eq)

    print(f"reactive self-check OK  X(MeOAc-invariant)={np.round(X, 3)}  "
          f"x_eq={np.round(x_eq, 3)} T={T_eq:.1f}C xi={xi:.3f}")


if __name__ == "__main__":
    _demo()
