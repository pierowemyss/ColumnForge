"""R(U): the Naphtali-Sandholm residual system (blueprint Section 4).

Variables are component flows + temperature (+ reaction extents), packed
stage-major with a *constant* per-stage stride so the tridiagonal kernels have
a fixed block size (portable to C):

    block i = [ l_i0..l_i,C-1 | v_i0..v_i,C-1 | T_i | xi_i0..xi_i,R-1 ]   (len m)
    m = 2C + 1 + R          (R = reactions.n_rxn, 0 if none)

Derived per stage: L_i = sum_j l_ij, V_i = sum_j v_ij, x_ij = l_ij/L_i,
y_ij = v_ij/V_i.

Equation rows in each block (same order as the variables they close):
    0 .. C-1     material balance   M_ij
    C .. 2C-1    equilibrium        E_ij = K_ij l_ij V_i - v_ij L_i   (cleared)
    2C           energy balance H_i  (REPLACED by the terminal spec on i=0, i=N-1)
    2C+1 .. 2C+R reaction closure    (xi_ir = 0 on non-reactive stages)

Boundaries: liquid entering the top (l_{-1}) and vapour entering the bottom
(v_{N}) are zero. Stage 1's vapour v_0 is therefore the (partial-condenser)
distillate; stage N's liquid l_{N-1} is the bottoms.

Summation is implicit: summing E_ij over j gives V_i(sum_j K_ij x_ij - 1) = 0,
i.e. the bubble-point constraint, so temperature is pinned without a separate
sum equation.

Everything here is a pure function of NumPy arrays.
"""

import numpy as np

_TINY = 1e-300


def stride(prob):
    """Per-stage block size m = 2C+1+R."""
    R = prob.reactions.n_rxn if prob.reactions is not None else 0
    return 2 * prob.C + 1 + R


def pack(l, v, T, xi=None):
    """Flatten (l (N,C), v (N,C), T (N,), xi (N,R)) into U (N*m,)."""
    N, C = l.shape
    R = 0 if xi is None else xi.shape[1]
    m = 2 * C + 1 + R
    U = np.empty(N * m)
    U.reshape(N, m)[:, :C] = l
    U.reshape(N, m)[:, C:2 * C] = v
    U.reshape(N, m)[:, 2 * C] = T
    if R:
        U.reshape(N, m)[:, 2 * C + 1:] = xi
    return U


def unpack(U, N, C, R=0):
    """Inverse of pack: returns (l (N,C), v (N,C), T (N,), xi (N,R))."""
    m = 2 * C + 1 + R
    B = U.reshape(N, m)
    l = B[:, :C]
    v = B[:, C:2 * C]
    T = B[:, 2 * C]
    xi = B[:, 2 * C + 1:] if R else np.zeros((N, 0))
    return l, v, T, xi


def flows(l, v):
    """Total liquid/vapour flows and phase compositions (guarded division)."""
    L = l.sum(axis=1)
    V = v.sum(axis=1)
    x = l / np.maximum(L[:, None], _TINY)
    y = v / np.maximum(V[:, None], _TINY)
    return L, V, x, y


def _reaction_order(nu):
    """Forward mass-action orders: a_j = max(-nu_j, 0) (reactants only)."""
    return np.maximum(-nu, 0.0)


def _rate_kinetic(rx, x_stage):
    """Forward mass-action rates on one stage: (R,) = holdup k_r prod_j x_j^order."""
    orders = _reaction_order(rx.nu)                     # (R, C)
    xp = np.maximum(x_stage, _TINY)
    g = np.prod(xp[None, :] ** orders, axis=1)          # (R,)
    return rx.holdup * np.asarray(rx.k_fwd, float) * g


def _closure_equilibrium(rx, x_stage):
    """Equilibrium quotient residual on one stage: (R,) = prod_j x_j^nu_rj - Keq_r."""
    xp = np.maximum(x_stage, _TINY)
    quot = np.prod(xp[None, :] ** rx.nu, axis=1)        # (R,)
    return quot - np.asarray(rx.Keq, float)


def residual(U, prob, provider):
    """Assemble R(U) (shape (N*m,)) for the packed state U.

    prob: a Problem. provider: a ThermoProvider. Pure function.
    """
    N, C = prob.n_stages, prob.C
    R = prob.reactions.n_rxn if prob.reactions is not None else 0
    m = 2 * C + 1 + R
    l, v, T, xi = unpack(U, N, C, R)
    L, V, x, y = flows(l, v)

    K = provider.K(x, T, prob.pressure)                 # (N,C)
    hL = provider.h_L(x, T)                              # (N,)
    hV = provider.h_V(y, T)                              # (N,)

    Res = np.zeros((N, m))
    rl, rv = prob.rl, prob.rv
    rmask = prob.reactive_mask

    # neighbour flows with zero boundaries
    l_above = np.vstack([np.zeros((1, C)), l[:-1]])     # liquid from stage i-1
    v_below = np.vstack([v[1:], np.zeros((1, C))])      # vapour from stage i+1

    # --- material balances (rows 0..C-1) ---
    reac_mat = np.zeros((N, C))
    if R:
        for i in np.where(rmask)[0]:
            reac_mat[i] = prob.reactions.nu.T @ xi[i]   # sum_r nu_rj xi_ir
    Res[:, :C] = ((1 + rl)[:, None] * l + (1 + rv)[:, None] * v
                  - l_above - v_below - prob.feed - reac_mat)

    # --- equilibrium (rows C..2C-1), cleared form ---
    Res[:, C:2 * C] = K * l * V[:, None] - v * L[:, None]

    # --- energy (row 2C) for interior stages ---
    Lh_above = np.concatenate([[0.0], (L * hL)[:-1]])
    Vh_below = np.concatenate([(V * hV)[1:], [0.0]])
    energy = ((1 + rl) * L * hL + (1 + rv) * V * hV
              - Lh_above - Vh_below - prob.feedH - prob.duty)
    Res[:, 2 * C] = energy
    # terminals: replace energy with the operating specs
    Res[0, 2 * C] = _top_spec_residual(prob.top_spec, l, v, L, V)
    Res[N - 1, 2 * C] = _bottom_spec_residual(prob.bottom_spec, l, v, L, V)

    # --- reaction closures (rows 2C+1..) ---
    if R:
        for i in range(N):
            if rmask[i]:
                if prob.reactions.kind == "kinetic":
                    Res[i, 2 * C + 1:] = xi[i] - _rate_kinetic(prob.reactions, x[i])
                else:
                    Res[i, 2 * C + 1:] = _closure_equilibrium(prob.reactions, x[i])
            else:
                Res[i, 2 * C + 1:] = xi[i]              # pinned to zero

    return Res.reshape(N * m)


def _top_spec_residual(spec, l, v, L, V):
    k, val = spec.kind, spec.value
    if k == "reflux_ratio":
        return L[0] - val * V[0]
    if k == "reflux_rate":
        return L[0] - val
    if k == "distillate_rate":
        return V[0] - val
    if k == "dist_purity":
        return v[0, spec.comp] - val * V[0]
    raise ValueError(f"unknown top spec {k!r}")


def _bottom_spec_residual(spec, l, v, L, V):
    k, val = spec.kind, spec.value
    n = L.shape[0] - 1
    if k == "boilup_ratio":
        return V[n] - val * L[n]
    if k == "boilup_rate":
        return V[n] - val
    if k == "bottoms_rate":
        return L[n] - val
    if k == "bottoms_purity":
        return l[n, spec.comp] - val * L[n]
    raise ValueError(f"unknown bottom spec {k!r}")


def mass_balance_residual(U, prob):
    """Overall + per-component external mass-balance closure of a state.

    Returns (per_component (C,), overall scalar): total in (feeds) minus total
    out (distillate v_0, bottoms l_{N-1}, side draws). Independent of thermo —
    a direct audit of the returned flows (blueprint Section 12.2 / 15).
    """
    N, C = prob.n_stages, prob.C
    R = prob.reactions.n_rxn if prob.reactions is not None else 0
    l, v, T, xi = unpack(U, N, C, R)
    L, V, x, y = flows(l, v)
    feed_in = prob.feed.sum(axis=0)
    dist = v[0]                                    # partial-condenser vapour product
    bot = l[N - 1]                                 # bottoms liquid product
    side = (prob.rl[:, None] * l + prob.rv[:, None] * v).sum(axis=0)
    generation = np.zeros(C)
    if R:
        for i in np.where(prob.reactive_mask)[0]:
            generation += prob.reactions.nu.T @ xi[i]
    per_comp = feed_in + generation - dist - bot - side
    return per_comp, float(per_comp.sum())


def _demo():
    import numpy as np
    from thermo_adapter import FreeColumnThermo
    from problem import build_problem, OpSpec

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = FreeColumnThermo(abc)
    comps = ["benzene", "toluene", "xylene"]
    N, C = 8, 3
    prob = build_problem(
        n_stages=N, comps=comps, feeds=[(4, 100.0, [0.4, 0.35, 0.25])],
        pressure=760.0, provider=tp,
        top_spec=OpSpec("reflux_ratio", 3.0),
        bottom_spec=OpSpec("distillate_rate", 40.0) if False else OpSpec("bottoms_rate", 60.0))

    # pack/unpack round-trips exactly
    l = np.abs(np.random.default_rng(0).random((N, C))) + 0.1
    v = np.abs(np.random.default_rng(1).random((N, C))) + 0.1
    T = np.linspace(80, 130, N)
    U = pack(l, v, T)
    l2, v2, T2, xi2 = unpack(U, N, C, 0)
    assert np.allclose(l, l2) and np.allclose(v, v2) and np.allclose(T, T2)
    assert stride(prob) == 2 * C + 1

    # residual has the right length and is finite
    Res = residual(U, prob, tp)
    assert Res.shape == (N * (2 * C + 1),) and np.all(np.isfinite(Res))

    # A state that satisfies material balance exactly makes the material rows
    # vanish: build l,v with CMO so M=0, and check the C material rows are ~0.
    L = np.full(N, 200.0); V = np.full(N, 240.0)
    # simple consistent flows: constant molal overflow, uniform composition
    z = np.array([0.4, 0.35, 0.25])
    l = L[:, None] * z; v = V[:, None] * z
    # inject feed so top/bottom net out: put nothing, just check interior M rows
    U = pack(l, v, np.full(N, 100.0))
    Res = residual(U, prob, tp).reshape(N, -1)
    # interior stage 2 (no feed, no draw): (1)*l + (1)*v - l_above - v_below == 0
    assert np.allclose(Res[2, :C], 0.0, atol=1e-9), Res[2, :C]

    # equilibrium row vanishes only at a self-consistent phase split: bubble-T
    # temperatures with v = K*l give V=L and E = K l V - v L = 0.
    x = l / L[:, None]
    Tb = np.array([tp.bubble_T(x[i], prob.pressure[i]) for i in range(N)])
    K = tp.K(x, Tb, prob.pressure)
    v_eq = K * l                                     # V = sum(K l) = L
    U2 = pack(l, v_eq, Tb)
    Res2 = residual(U2, prob, tp).reshape(N, -1)
    assert np.allclose(Res2[3, C:2 * C], 0.0, atol=1e-6), Res2[3, C:2 * C]

    # terminal spec rows: reflux ratio residual = L0 - r V0
    Res = residual(pack(l, v, np.full(N, 100.0)), prob, tp).reshape(N, -1)
    assert abs(Res[0, 2 * C] - (L[0] - 3.0 * V[0])) < 1e-6
    assert abs(Res[N - 1, 2 * C] - (L[N - 1] - 60.0)) < 1e-6   # bottoms_rate

    print("residual self-check OK")


if __name__ == "__main__":
    _demo()
