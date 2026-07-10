"""Block-tridiagonal solve: block Thomas with per-block pivoting.

Solves J du = -R where J is stored as the three block diagonals (A, B, C) from
`jacobian.jacobian_blocks`. Cost is O(N m^3): one m x m LU (via np.linalg.solve,
which partial-pivots) per stage. The dense J is never formed.

Block Thomas (LU sweep):
    forward:  E_0 = B_0^{-1} C_0,  g_0 = B_0^{-1} d_0
              D_i = B_i - A_i E_{i-1},  E_i = D_i^{-1} C_i,  g_i = D_i^{-1}(d_i - A_i g_{i-1})
    back:     u_{N-1} = g_{N-1},  u_i = g_i - E_i u_{i+1}
"""

import numpy as np


def block_thomas(A, B, C, d):
    """Solve the block-tridiagonal system. d is (N*m,) or (N, m).

    A, B, C: (N, m, m); A[0] and C[N-1] are ignored. Returns u (N*m,).
    Raises np.linalg.LinAlgError on a singular pivot block (pinch/degeneracy).
    """
    N, m, _ = B.shape
    d = np.asarray(d, float).reshape(N, m)
    E = np.empty((N, m, m))
    g = np.empty((N, m))

    E[0] = np.linalg.solve(B[0], C[0])
    g[0] = np.linalg.solve(B[0], d[0])
    for i in range(1, N):
        D = B[i] - A[i] @ E[i - 1]
        rhs = d[i] - A[i] @ g[i - 1]
        if i < N - 1:
            E[i] = np.linalg.solve(D, C[i])
        g[i] = np.linalg.solve(D, rhs)

    u = np.empty((N, m))
    u[N - 1] = g[N - 1]
    for i in range(N - 2, -1, -1):
        u[i] = g[i] - E[i] @ u[i + 1]
    return u.reshape(N * m)


def _demo():
    rng = np.random.default_rng(0)
    N, m = 7, 5

    # Build a diagonally-dominant block-tridiagonal system so it's well-posed.
    A = rng.standard_normal((N, m, m)) * 0.1
    C = rng.standard_normal((N, m, m)) * 0.1
    B = rng.standard_normal((N, m, m)) * 0.1 + np.eye(m) * (m + 2)
    A[0] = 0.0; C[N - 1] = 0.0

    # dense reference
    J = np.zeros((N * m, N * m))
    for i in range(N):
        J[i*m:(i+1)*m, i*m:(i+1)*m] = B[i]
        if i > 0:
            J[i*m:(i+1)*m, (i-1)*m:i*m] = A[i]
        if i < N - 1:
            J[i*m:(i+1)*m, (i+1)*m:(i+2)*m] = C[i]
    d = rng.standard_normal(N * m)

    u = block_thomas(A, B, C, d)
    u_ref = np.linalg.solve(J, d)
    assert np.allclose(u, u_ref, atol=1e-10), np.abs(u - u_ref).max()
    assert np.allclose(J @ u, d, atol=1e-9)

    # singular pivot raises (a degenerate leading diagonal block)
    Bsing = B.copy(); Bsing[0] = np.zeros((m, m))
    try:
        block_thomas(A, Bsing, C, d)
    except np.linalg.LinAlgError:
        pass
    else:
        raise AssertionError("singular block should raise LinAlgError")

    print("linsolve self-check OK (block Thomas == dense solve to 1e-10)")


if __name__ == "__main__":
    _demo()
