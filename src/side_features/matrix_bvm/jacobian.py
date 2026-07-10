"""Block-tridiagonal Jacobian J = dR/dU (blueprint Section 8).

Returns the three per-stage block diagonals, never the dense matrix:

    A[i] = dR_i/dU_{i-1}   (m, m)   sub-diagonal  (A[0] unused)
    B[i] = dR_i/dU_i       (m, m)   diagonal
    C[i] = dR_i/dU_{i+1}   (m, m)   super-diagonal (C[N-1] unused)

with m = 2C+1+R the constant block size from `residual.stride`. The equilibrium
and reaction rows are purely local, so A/C carry only material (-I) and energy
(enthalpy) couplings — the source of the band's sparsity.

All entries are analytic. Derivatives of K and the enthalpies with respect to
the flow variables come from the chain rule through x=l/L, y=v/V:

    dx_m/dl_k = (delta_mk - x_m)/L,     dK_j/dl_k = (dKdx[j,k] - (dKdx@x)[j])/L

The provider supplies dK/dx, dK/dT and the enthalpy derivatives; where those are
finite-differenced (FreeColumnThermo) the assembled J is validated against a
full finite-difference of R in the self-check (blueprint Section 15).
"""

import numpy as np

from residual import unpack, flows, residual, stride, _reaction_order

_TINY = 1e-300


def jacobian_blocks(U, prob, provider):
    """Assemble (A, B, C), each (N, m, m). Pure function of arrays."""
    N, C = prob.n_stages, prob.C
    R = prob.reactions.n_rxn if prob.reactions is not None else 0
    m = 2 * C + 1 + R
    l, v, T, xi = unpack(U, N, C, R)
    L, V, x, y = flows(l, v)

    K = provider.K(x, T, prob.pressure)
    dKdx = provider.dK_dx(x, T, prob.pressure)          # (N,C,C) dK_j/dx_k
    dKdT = provider.dK_dT(x, T, prob.pressure)          # (N,C)
    hL = provider.h_L(x, T); hV = provider.h_V(y, T)
    dhLdx = provider.dhL_dx(x, T); dhLdT = provider.dhL_dT(x, T)
    dhVdy = provider.dhV_dy(y, T); dhVdT = provider.dhV_dT(y, T)

    # chain-rule derived quantities (per stage)
    Kl = np.empty((N, C, C))     # dK_j/dl_k
    hLx = np.empty((N, C))       # dhL/dl_k
    hVy = np.empty((N, C))       # dhV/dv_k
    for i in range(N):
        Kl[i] = (dKdx[i] - (dKdx[i] @ x[i])[:, None]) / max(L[i], _TINY)
        hLx[i] = (dhLdx[i] - dhLdx[i] @ x[i]) / max(L[i], _TINY)
        hVy[i] = (dhVdy[i] - dhVdy[i] @ y[i]) / max(V[i], _TINY)

    A = np.zeros((N, m, m)); B = np.zeros((N, m, m)); Cc = np.zeros((N, m, m))
    I = np.eye(C)
    rl, rv = prob.rl, prob.rv
    rmask = prob.reactive_mask

    for i in range(N):
        # ---- material rows [0:C] ----
        B[i, 0:C, 0:C] = (1 + rl[i]) * I
        B[i, 0:C, C:2 * C] = (1 + rv[i]) * I
        if R and rmask[i]:
            B[i, 0:C, 2 * C + 1:] = -prob.reactions.nu.T        # dM_j/dxi_r = -nu_rj
        if i > 0:
            A[i, 0:C, 0:C] = -I                                 # dM_j/dl_{i-1,k}
        if i < N - 1:
            Cc[i, 0:C, C:2 * C] = -I                            # dM_j/dv_{i+1,k}

        # ---- equilibrium rows [C:2C] (local) ----
        # dE_j/dl_k = V(Kl[j,k] l_j + K_j d_jk) - v_j
        B[i, C:2 * C, 0:C] = V[i] * (Kl[i] * l[i][:, None] + np.diag(K[i])) - v[i][:, None]
        # dE_j/dv_k = K_j l_j - d_jk L
        B[i, C:2 * C, C:2 * C] = np.outer(K[i] * l[i], np.ones(C)) - L[i] * I
        # dE_j/dT = l_j V dKdT_j
        B[i, C:2 * C, 2 * C] = l[i] * V[i] * dKdT[i]

        # ---- energy row [2C] (interior form; terminals overwritten below) ----
        B[i, 2 * C, 0:C] = (1 + rl[i]) * (hL[i] + L[i] * hLx[i])
        B[i, 2 * C, C:2 * C] = (1 + rv[i]) * (hV[i] + V[i] * hVy[i])
        B[i, 2 * C, 2 * C] = (1 + rl[i]) * L[i] * dhLdT[i] + (1 + rv[i]) * V[i] * dhVdT[i]
        if i > 0:
            A[i, 2 * C, 0:C] = -(hL[i - 1] + L[i - 1] * hLx[i - 1])
            A[i, 2 * C, 2 * C] = -L[i - 1] * dhLdT[i - 1]
        if i < N - 1:
            Cc[i, 2 * C, C:2 * C] = -(hV[i + 1] + V[i + 1] * hVy[i + 1])
            Cc[i, 2 * C, 2 * C] = -V[i + 1] * dhVdT[i + 1]

        # ---- reaction rows [2C+1:] ----
        for r in range(R):
            row = 2 * C + 1 + r
            if not rmask[i]:
                B[i, row, row] = 1.0                            # xi_r = 0 pinned
                continue
            if prob.reactions.kind == "kinetic":
                B[i, row, row] = 1.0                            # dclos/dxi_r
                orders = _reaction_order(prob.reactions.nu)[r]  # (C,)
                g = np.prod(np.maximum(x[i], _TINY) ** orders)
                kf = prob.reactions.holdup * float(prob.reactions.k_fwd[r])
                dg_dl = (g / max(L[i], _TINY)) * (
                    orders / np.maximum(x[i], _TINY) - orders.sum())
                B[i, row, 0:C] = -kf * dg_dl                    # dclos/dl_k
            else:  # equilibrium: quotient constraint, no xi term
                quot = np.prod(np.maximum(x[i], _TINY) ** prob.reactions.nu[r])
                nu_r = prob.reactions.nu[r]
                dq_dl = (quot / max(L[i], _TINY)) * (
                    nu_r / np.maximum(x[i], _TINY) - nu_r.sum())
                B[i, row, 0:C] = dq_dl

    # ---- terminal spec rows overwrite the energy row (local, so zero the
    #      energy couplings in the neighbour blocks that touch it) ----
    _set_top_spec(B[0], prob.top_spec, C)
    Cc[0, 2 * C, :] = 0.0                                       # top spec is local
    _set_bottom_spec(B[N - 1], prob.bottom_spec, C, L[N - 1], l[N - 1])
    A[N - 1, 2 * C, :] = 0.0                                    # bottom spec is local

    return A, B, Cc


def _set_top_spec(Brow_block, spec, C):
    """Overwrite row 2C of the top block with the spec derivative."""
    Brow_block[2 * C, :] = 0.0
    k, val = spec.kind, spec.value
    if k == "reflux_ratio":
        Brow_block[2 * C, 0:C] = 1.0
        Brow_block[2 * C, C:2 * C] = -val
    elif k == "reflux_rate":
        Brow_block[2 * C, 0:C] = 1.0
    elif k == "distillate_rate":
        Brow_block[2 * C, C:2 * C] = 1.0
    elif k == "dist_purity":
        Brow_block[2 * C, C:2 * C] = -val
        Brow_block[2 * C, C + spec.comp] += 1.0
    else:
        raise ValueError(f"unknown top spec {k!r}")


def _set_bottom_spec(Brow_block, spec, C, L_n, l_n):
    Brow_block[2 * C, :] = 0.0
    k, val = spec.kind, spec.value
    if k == "boilup_ratio":
        Brow_block[2 * C, C:2 * C] = 1.0
        Brow_block[2 * C, 0:C] = -val
    elif k == "boilup_rate":
        Brow_block[2 * C, C:2 * C] = 1.0
    elif k == "bottoms_rate":
        Brow_block[2 * C, 0:C] = 1.0
    elif k == "bottoms_purity":
        Brow_block[2 * C, 0:C] = -val
        Brow_block[2 * C, spec.comp] += 1.0
    else:
        raise ValueError(f"unknown bottom spec {k!r}")


def dense_from_blocks(A, B, Cc):
    """Assemble the dense J from blocks — for testing / small problems only."""
    N, m, _ = B.shape
    J = np.zeros((N * m, N * m))
    for i in range(N):
        J[i * m:(i + 1) * m, i * m:(i + 1) * m] = B[i]
        if i > 0:
            J[i * m:(i + 1) * m, (i - 1) * m:i * m] = A[i]
        if i < N - 1:
            J[i * m:(i + 1) * m, (i + 1) * m:(i + 2) * m] = Cc[i]
    return J


def fd_jacobian(U, prob, provider, h=1e-6):
    """Full central finite-difference Jacobian of R — the analytic-vs-FD oracle."""
    n = U.shape[0]
    J = np.zeros((n, n))
    for k in range(n):
        du = np.zeros(n); du[k] = h * max(1.0, abs(U[k]))
        Rp = residual(U + du, prob, provider)
        Rm = residual(U - du, prob, provider)
        J[:, k] = (Rp - Rm) / (2 * du[k])
    return J


def _demo():
    import numpy as np
    from thermo_adapter import FreeColumnThermo
    from problem import build_problem, OpSpec
    from residual import pack

    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    comps = ["benzene", "toluene", "xylene"]
    N, C = 6, 3

    def check(prob, tp, U, tol):
        A, B, Cc = jacobian_blocks(U, prob, tp)
        Jan = dense_from_blocks(A, B, Cc)
        Jfd = fd_jacobian(U, prob, tp)
        err = np.abs(Jan - Jfd).max()
        rel = err / max(1.0, np.abs(Jfd).max())
        assert rel < tol, f"analytic vs FD off: abs {err:.2e} rel {rel:.2e}"
        return rel

    # A physically-scaled, non-trivial state (not a solution)
    rng = np.random.default_rng(42)
    z = np.array([0.4, 0.35, 0.25])
    L = np.linspace(180, 220, N); V = np.linspace(230, 250, N)
    x = np.clip(z + 0.05 * rng.standard_normal((N, C)), 0.05, None)
    x /= x.sum(1, keepdims=True)
    yv = np.clip(z + 0.05 * rng.standard_normal((N, C)), 0.05, None)
    yv /= yv.sum(1, keepdims=True)
    T = np.linspace(85, 125, N)
    U = pack(L[:, None] * x, V[:, None] * yv, T)

    # ideal
    tp = FreeColumnThermo(abc)
    prob = build_problem(n_stages=N, comps=comps, feeds=[(3, 100.0, z)],
                         pressure=760.0, provider=tp,
                         top_spec=OpSpec("reflux_ratio", 3.0),
                         bottom_spec=OpSpec("bottoms_rate", 60.0))
    rel_ideal = check(prob, tp, U, 1e-5)

    # NRTL (nonzero dK/dx exercises the equilibrium-block chain rule)
    from core.thermodynamics import nrtl_gamma_fn
    a = 0.4
    gfn = nrtl_gamma_fn([[0, a, a], [a, 0, a], [a, a, 0]], [[0.0] * 3] * 3,
                        [[0, .3, .3], [.3, 0, .3], [.3, .3, 0]])
    tp2 = FreeColumnThermo(abc, gamma_fn=gfn)
    prob2 = build_problem(n_stages=N, comps=comps, feeds=[(3, 100.0, z)],
                          pressure=760.0, provider=tp2,
                          top_spec=OpSpec("reflux_ratio", 3.0),
                          bottom_spec=OpSpec("boilup_ratio", 1.5))
    rel_nrtl = check(prob2, tp2, U, 2e-4)

    # purity specs at both ends
    prob3 = build_problem(n_stages=N, comps=comps, feeds=[(3, 100.0, z)],
                          pressure=760.0, provider=tp,
                          top_spec=OpSpec("dist_purity", 0.95, 0),
                          bottom_spec=OpSpec("bottoms_purity", 0.9, 2))
    check(prob3, tp, U, 1e-5)

    # reactive stage: block grows, analytic reaction rows vs FD
    from problem import Reactions
    rx = Reactions(nu=np.array([[-1.0, 1.0, 0.0]]), stages=np.array([3]),
                   kind="kinetic", k_fwd=np.array([2.0]), holdup=1.0)
    prob4 = build_problem(n_stages=N, comps=comps, feeds=[(3, 100.0, z)],
                          pressure=760.0, provider=tp,
                          top_spec=OpSpec("reflux_ratio", 3.0),
                          bottom_spec=OpSpec("bottoms_rate", 60.0), reactions=rx)
    m = stride(prob4)
    xi = np.zeros((N, 1)); xi[3, 0] = 1.0
    Ur = pack(L[:, None] * x, V[:, None] * yv, T, xi)
    check(prob4, tp, Ur, 1e-5)

    print(f"jacobian self-check OK (analytic vs FD: ideal {rel_ideal:.1e}, "
          f"NRTL {rel_nrtl:.1e}, reactive/purity within tol)")


if __name__ == "__main__":
    _demo()
