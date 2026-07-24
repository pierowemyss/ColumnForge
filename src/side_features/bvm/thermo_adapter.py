"""ThermoProvider interface (blueprint Section 6) + a ColumnForge wrapper.

The residual/Jacobian kernels see thermodynamics only through this interface:

    K(x, T, P)        -> (N, C)   equilibrium ratios y = K x
    dK_dx(x, T, P)    -> (N, C, C)  dK_ij / dx_ik   (per stage j-block)
    dK_dT(x, T, P)    -> (N, C)
    dK_dP(x, T, P)    -> (N, C)     optional (None if pressure-independent path)
    h_L(x, T)         -> (N,)     molar liquid enthalpy
    h_V(y, T)         -> (N,)     molar vapour enthalpy
    dhL_dx, dhV_dy    -> (N, C)
    dhL_dT, dhV_dT    -> (N,)
    bubble_T, dew_T, Psat         scalar helpers for the initializer

The solver never assumes where a derivative came from: `ColumnForgeThermo`
returns them by central finite differences of the underlying `core` maps (the
NRTL closure casts its argument to real, so complex-step can't thread through
it — FD is the robust choice, and the goal sanctions analytic/complex-step/FD
alike). A model that hands back analytic derivatives is equally valid.

Arrays are stage-major: x is (N, C), T and P are (N,). Everything is a pure
function of those arrays — no per-stage Python objects.
"""

import numpy as np

from core.thermodynamics import (
    antoine_psat, bubble_T as _bubble_T, dew_T as _dew_T, k_values, latent_heat,
)

# Central-difference steps. Small enough for O(h^2) accuracy, large enough to
# clear round-off in the (well-scaled) K and enthalpy maps.
_HX = 1e-7      # composition step
_HT = 1e-5      # temperature step


def _cs_grad_scalar(f, x, T):
    """Gradient of a scalar map f(x, T) at one stage: returns (df/dx (C,), df/dT).

    Central finite differences. x is (C,), T scalar.
    """
    C = x.shape[0]
    dfdx = np.empty(C)
    for k in range(C):
        xp = x.copy(); xp[k] += _HX
        xm = x.copy(); xm[k] -= _HX
        dfdx[k] = (f(xp, T) - f(xm, T)) / (2 * _HX)
    dfdT = (f(x, T + _HT) - f(x, T - _HT)) / (2 * _HT)
    return dfdx, dfdT


def _cs_jac_vector(f, x, T):
    """Jacobian of a vector map f(x, T)->(C,) at one stage.

    Returns (dfdx (C,C) with dfdx[j,k]=df_j/dx_k, dfdT (C,)). Central FD.
    """
    C = x.shape[0]
    dfdx = np.empty((C, C))
    for k in range(C):
        xp = x.copy(); xp[k] += _HX
        xm = x.copy(); xm[k] -= _HX
        dfdx[:, k] = (f(xp, T) - f(xm, T)) / (2 * _HX)
    dfdT = (f(x, T + _HT) - f(x, T - _HT)) / (2 * _HT)
    return dfdx, dfdT


class ThermoProvider:
    """Interface contract. Subclass and implement; the solver only calls these.

    Every method takes stage-major arrays (x/y: (N,C); T/P: (N,)) and returns
    the shapes documented in the module docstring. Derivative methods may be
    analytic, complex-step, or finite-difference — the solver does not care.
    """

    n_comps: int

    def K(self, x, T, P): raise NotImplementedError
    def dK_dx(self, x, T, P): raise NotImplementedError
    def dK_dT(self, x, T, P): raise NotImplementedError
    def dK_dP(self, x, T, P): return None            # optional
    def h_L(self, x, T): raise NotImplementedError
    def h_V(self, y, T): raise NotImplementedError
    def dhL_dx(self, x, T): raise NotImplementedError
    def dhL_dT(self, x, T): raise NotImplementedError
    def dhV_dy(self, y, T): raise NotImplementedError
    def dhV_dT(self, y, T): raise NotImplementedError
    def bubble_T(self, x, P): raise NotImplementedError
    def dew_T(self, y, P): raise NotImplementedError
    def Psat(self, T): raise NotImplementedError


class ColumnForgeThermo(ThermoProvider):
    """Wrap ColumnForge `core.thermodynamics` as a ThermoProvider.

    K_ij = gamma_i(x,T) Psat_i(T) / P_i (ideal gamma=1). Enthalpies use a
    constant-Cp sensible term plus a Clausius-Clapeyron latent heat for the
    vapour:

        h_L(x,T) = sum_i x_i Cp_i (T - Tref)
        h_V(y,T) = sum_i y_i [Cp_i (T - Tref) + lambda_i(T)]

    Cp and Tref are the calibration knobs a minimal model still needs (real
    heat-capacity data slots straight in here). Derivative methods use central
    finite differences (the NRTL closure casts to real, so complex-step can't
    thread through it); an SRK vapour EOS enters through `phi_fn`.
    """

    def __init__(self, antoine, gamma_fn=None, phi_fn=None, Cp=None, Tref=0.0):
        self.antoine = np.asarray(antoine, float)
        self.gamma_fn = gamma_fn
        self.phi_fn = phi_fn            # vapour-phase EOS (SRK); None = ideal gas
        self.n_comps = self.antoine.shape[0]
        # Default Cp: a single representative liquid molar heat capacity. The
        # energy balance only needs a smooth, physically-scaled enthalpy; swap
        # in per-component Cp when data exists.
        self.Cp = (np.full(self.n_comps, 150.0) if Cp is None
                   else np.asarray(Cp, float))
        self.Tref = float(Tref)

    # ---- K-values -------------------------------------------------------
    def _K_stage(self, x, T, P):
        """K on one stage: complex-safe. x (C,), T scalar (maybe complex)."""
        return k_values(T, P, self.antoine, self.gamma_fn, x, self.phi_fn)

    def K(self, x, T, P):
        x, T, P = np.asarray(x, float), np.asarray(T, float), np.asarray(P, float)
        return np.array([self._K_stage(x[i], T[i], P[i]) for i in range(len(T))])

    def dK_dx(self, x, T, P):
        x, T, P = np.asarray(x, float), np.asarray(T, float), np.asarray(P, float)
        out = np.empty((len(T), self.n_comps, self.n_comps))
        for i in range(len(T)):
            Pi = P[i]
            dfdx, _ = _cs_jac_vector(lambda xx, TT: self._K_stage(xx, TT, Pi),
                                     x[i], T[i])
            out[i] = dfdx
        return out

    def dK_dT(self, x, T, P):
        x, T, P = np.asarray(x, float), np.asarray(T, float), np.asarray(P, float)
        out = np.empty((len(T), self.n_comps))
        for i in range(len(T)):
            Pi = P[i]
            _, dfdT = _cs_jac_vector(lambda xx, TT: self._K_stage(xx, TT, Pi),
                                     x[i], T[i])
            out[i] = dfdT
        return out

    def dK_dP(self, x, T, P):
        # K = gamma Psat / P  => dK/dP = -K / P analytically. Cheap and exact.
        return -self.K(x, T, P) / np.asarray(P, float)[:, None]

    # ---- enthalpies -----------------------------------------------------
    def _hL_stage(self, x, T):
        return np.sum(x * self.Cp * (T - self.Tref))

    def _hV_stage(self, y, T):
        lam = latent_heat(T, self.antoine)
        return np.sum(y * (self.Cp * (T - self.Tref) + lam))

    def h_L(self, x, T):
        x, T = np.asarray(x, float), np.asarray(T, float)
        return np.array([self._hL_stage(x[i], T[i]) for i in range(len(T))])

    def h_V(self, y, T):
        y, T = np.asarray(y, float), np.asarray(T, float)
        return np.array([self._hV_stage(y[i], T[i]) for i in range(len(T))])

    def dhL_dx(self, x, T):
        x, T = np.asarray(x, float), np.asarray(T, float)
        out = np.empty((len(T), self.n_comps))
        for i in range(len(T)):
            g, _ = _cs_grad_scalar(self._hL_stage, x[i], T[i])
            out[i] = g
        return out

    def dhL_dT(self, x, T):
        x, T = np.asarray(x, float), np.asarray(T, float)
        out = np.empty(len(T))
        for i in range(len(T)):
            _, g = _cs_grad_scalar(self._hL_stage, x[i], T[i])
            out[i] = g
        return out

    def dhV_dy(self, y, T):
        y, T = np.asarray(y, float), np.asarray(T, float)
        out = np.empty((len(T), self.n_comps))
        for i in range(len(T)):
            g, _ = _cs_grad_scalar(self._hV_stage, y[i], T[i])
            out[i] = g
        return out

    def dhV_dT(self, y, T):
        y, T = np.asarray(y, float), np.asarray(T, float)
        out = np.empty(len(T))
        for i in range(len(T)):
            _, g = _cs_grad_scalar(self._hV_stage, y[i], T[i])
            out[i] = g
        return out

    # ---- scalar helpers for the initializer -----------------------------
    def bubble_T(self, x, P):
        return _bubble_T(np.asarray(x, float), P, self.antoine,
                         gamma_fn=self.gamma_fn, phi_fn=self.phi_fn)

    def dew_T(self, y, P):
        return _dew_T(np.asarray(y, float), P, self.antoine,
                      gamma_fn=self.gamma_fn, phi_fn=self.phi_fn)

    def Psat(self, T):
        return antoine_psat(T, self.antoine)

    # ---- conjugate-composition steps (the BVM marching equilibrium, Sec 17) ---
    def bubble(self, x, P):
        """Bubble point of liquid x at P -> (y, T): the vapour it boils into.

        y_i = K_i(x, T) x_i, normalised. Used by the stripping-direction march.
        """
        x = np.asarray(x, float)
        x = x / x.sum()
        T = _bubble_T(x, P, self.antoine, gamma_fn=self.gamma_fn, phi_fn=self.phi_fn)
        y = k_values(T, P, self.antoine, self.gamma_fn, x, self.phi_fn) * x
        return y / y.sum(), T

    def dew(self, y, P):
        """Dew point of vapour y at P -> (x, T): the liquid it condenses to.

        x_i = y_i / K_i, normalised. Used by the rectifying march.

        ponytail: gamma is evaluated at y as a PROXY, not at the true liquid x.
        A self-consistent gamma(x) fixed point (audit E6) was tried and reverted:
        for the stiff MEOH/DMC/EG multicomp reference the global gamma(x) dew has
        a second, EG-heavy root that the rectifying march jumps to at stage ~2
        (T -> 1700 K, blows up), breaking the multicomp .colx contract. The proxy
        stays on the physical MEOH-rich branch. Upgrade path: branch-continuation
        dew seeded from the previous stage's liquid, tracked in the audit.
        """
        y = np.asarray(y, float)
        y = y / y.sum()
        T = _dew_T(y, P, self.antoine, gamma_fn=self.gamma_fn, phi_fn=self.phi_fn)
        x = y / k_values(T, P, self.antoine, self.gamma_fn, y, self.phi_fn)
        return x / x.sum(), T


def _demo():
    # benzene / toluene / xylene, mmHg + degC Antoine
    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    tp = ColumnForgeThermo(abc)
    N, C = 2, 3
    x = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    T = np.array([90.0, 110.0])
    P = np.full(N, 760.0)

    K = tp.K(x, T, P)
    assert K.shape == (N, C) and np.all(K > 0)
    assert K[0, 0] > K[0, 1] > K[0, 2], "benzene most volatile"

    # complex-step derivatives match a central finite difference
    dKdx = tp.dK_dx(x, T, P)
    dKdT = tp.dK_dT(x, T, P)
    assert dKdx.shape == (N, C, C) and dKdT.shape == (N, C)
    h = 1e-5
    for i in range(N):
        # dK/dT central FD
        Kp = tp.K(x[i:i+1], T[i:i+1] + h, P[i:i+1])[0]
        Km = tp.K(x[i:i+1], T[i:i+1] - h, P[i:i+1])[0]
        assert np.allclose((Kp - Km) / (2 * h), dKdT[i], rtol=1e-4, atol=1e-6)
    # dK/dP analytic == -K/P
    assert np.allclose(tp.dK_dP(x, T, P), -K / P[:, None])

    # ideal K has zero dK/dx (gamma=1); NRTL makes it nonzero
    assert np.allclose(dKdx, 0.0), "ideal K independent of composition"
    from core.thermodynamics import nrtl_gamma_fn
    a = 0.4
    gfn = nrtl_gamma_fn([[0, a, a], [a, 0, a], [a, a, 0]],
                        [[0.0] * 3] * 3, [[0, .3, .3], [.3, 0, .3], [.3, .3, 0]])
    tp2 = ColumnForgeThermo(abc, gamma_fn=gfn)
    assert np.abs(tp2.dK_dx(x, T, P)).max() > 1e-4, "NRTL K depends on x"

    # enthalpy: vapour above liquid by ~latent heat; derivatives finite
    hL, hV = tp.h_L(x, T), tp.h_V(x, T)
    assert np.all(hV > hL), "vapour enthalpy exceeds liquid (latent heat)"
    assert tp.dhL_dT(x, T).shape == (N,) and tp.dhV_dy(x, T).shape == (N, C)
    hh = 1e-4
    fd = (tp.h_L(x, T + hh) - tp.h_L(x, T - hh)) / (2 * hh)
    assert np.allclose(fd, tp.dhL_dT(x, T), rtol=1e-5, atol=1e-6)

    # scalar helpers
    Tb = tp.bubble_T(x[0], 760.0)
    assert 80.0 < Tb < 138.0
    print("thermo_adapter self-check OK")


if __name__ == "__main__":
    _demo()
