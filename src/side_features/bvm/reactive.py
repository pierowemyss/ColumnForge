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
reference components are redundant. Profiles are marched in X and mapped back for
reporting. ponytail: equilibrium reactions with the physical VLE as the stagewise
closure; kinetic/rate closures slot into the same transform.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Reactions:
    nu: np.ndarray        # (R, C) stoichiometric coefficients (products +, reactants -)
    ref: np.ndarray       # (R,) indices of the reference components

    def __post_init__(self):
        self.nu = np.atleast_2d(np.asarray(self.nu, float))
        self.ref = np.asarray(self.ref, int)
        assert self.nu.shape[0] == len(self.ref), "one reference per reaction"


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
    print("reactive self-check OK  X(MeOAc-invariant) =", np.round(X, 3))


if __name__ == "__main__":
    _demo()
