"""Residue curve maps through the compiled solver — ctypes bridge to
`side_features/rcm/RCM_solv.c`.

The C solver integrates dx/dxi = x - y(x) with a full bubble-point VLE solve at
every Euler step, and computes no thermodynamics itself: it forwards model
parameters straight into `src/native/nifco2.f90`. This module's whole job is to
take the *same* objects the rest of the app already builds — the `antoine`
matrix from `ThermodynamicsConfig.psat_params`, the `gamma_fn` from
`WindowState.build_gamma_fn`, the `phi_fn` from `build_phi_fn` — and unpack
them into that struct. There is deliberately no second parameter-gathering
path, because the one that used to exist (the predecessor app's `orgProps`) is
how the RCM module ended up ignoring the app's thermo selection entirely.

Nothing here is required. The `.so` only exists if somebody ran
`make -C src/side_features/rcm`, and it needs GSL, so `available()`
returning False is a normal outcome — see `core/nifco.py`, which this mirrors.
`gui/plotting.py` has a pure-NumPy `residue_curve` that computes the same thing
~50x slower; it is what a future engine switch would select.
"""

import ctypes
import os

import numpy as np

from .thermodynamics import degC_to_K

#: src/side_features/rcm/lib — where that Makefile puts the built solver.
LIB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "side_features", "rcm", "lib")

BUILD_HINT = "make -C src/side_features/rcm"

# Model ids — must match the enums in RCM_solv.c. Keyed by exact type, not
# isinstance: _WilsonGamma and _UNIQUACGamma both subclass _NRTLGamma, so an
# isinstance ladder would silently send Wilson parameters to the NRTL routine.
_GAMMA_MODELS = {
    "_NRTLGamma": (1, ("tau_a", "tau_b", "alpha")),
    "_WilsonGamma": (2, ("a", "b")),
    "_UNIQUACGamma": (3, ("r", "q", "a", "b")),
    "_MargulesGamma": (4, ("A",)),
    "_UNIFACGamma": (5, ("nu", "R", "Q", "a_sub")),
}
_EOS_MODELS = {"_SRKPhi": 1}

_DOUBLE_P = ctypes.POINTER(ctypes.c_double)


class Params(ctypes.Structure):
    """Mirrors `params_t` in RCM_solv.c field for field."""
    _fields_ = [
        ("x0", _DOUBLE_P),
        ("P", ctypes.c_double),
        ("psat", _DOUBLE_P),
        ("npsat", ctypes.c_int),
        ("gp", _DOUBLE_P * 4),
        ("gammaModel", ctypes.c_int),
        ("ngroups", ctypes.c_int),
        ("TcCel", _DOUBLE_P),
        ("Pc_Pa", _DOUBLE_P),
        ("omega", _DOUBLE_P),
        ("eosModel", ctypes.c_int),
        ("pToPa", ctypes.c_double),
        ("Ncomps", ctypes.c_int),
        ("dxi", ctypes.c_double),
        ("n_it", ctypes.c_int),
        ("maxiter", ctypes.c_int),
        ("ftol", ctypes.c_double),
        ("xtol", ctypes.c_double),
    ]


class Curves(ctypes.Structure):
    _fields_ = [("x", _DOUBLE_P), ("y", _DOUBLE_P), ("T", _DOUBLE_P),
                ("nfail", ctypes.c_int)]


_lib = None
_load_error = None
_loaded = False


def load():
    """The compiled RCM solver, or None if it will not load."""
    global _lib, _load_error, _loaded
    if _loaded:
        return _lib
    _loaded = True
    try:
        # libminpack first: RCM_solver.so needs it and does not carry a path.
        ctypes.CDLL(os.path.join(LIB_DIR, "libminpack.so"))
        lib = ctypes.CDLL(os.path.join(LIB_DIR, "RCM_solver.so"))
    except OSError as exc:
        _load_error = str(exc)
        return None
    lib.RCM.argtypes = [ctypes.POINTER(Params)]
    lib.RCM.restype = ctypes.POINTER(Curves)
    lib.freeCurveMem.argtypes = [ctypes.POINTER(Curves)]
    lib.freeCurveMem.restype = None
    _lib = lib
    return lib


def available():
    return load() is not None


def load_error():
    """Why the library did not load, or None. For a UI to show verbatim."""
    load()
    return _load_error


def _fortran(a, name):
    """A contiguous column-major float64 copy, and a reference to keep it
    alive. C never indexes these — it forwards the pointer to Fortran — so
    column-major is not a detail, it is the contract."""
    arr = np.asfortranarray(np.asarray(a, dtype=np.float64))
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _unpack_gamma(gamma_fn, keep):
    """(model_id, [4 pointers], ngroups) for an activity closure from
    `core.thermodynamics`. `keep` collects the backing arrays."""
    gp = (_DOUBLE_P * 4)()
    if gamma_fn is None:
        return 0, gp, 0

    entry = _GAMMA_MODELS.get(type(gamma_fn).__name__)
    if entry is None:
        raise ValueError(
            f"{type(gamma_fn).__name__} has no compiled RCM equivalent. Add it "
            "to nifco2.f90 and _GAMMA_MODELS, or use an implemented activity "
            "model.")
    model, slots = entry

    # The Fortran hard-codes Tk = Tcel + 273.15, so a closure built with any
    # other temperature convention would be silently wrong. Same gate as
    # thermodynamics._native_lib. (Margules carries no t_to_K; it is
    # temperature-independent.)
    t_to_K = getattr(gamma_fn, "t_to_K", degC_to_K)
    if t_to_K is not degC_to_K:
        raise ValueError("the compiled RCM solver needs a degC gamma_fn; this "
                         "one carries a different temperature convention.")

    for i, slot in enumerate(slots):
        arr = _fortran(getattr(gamma_fn, slot), f"gamma_fn.{slot}")
        keep.append(arr)
        gp[i] = arr.ctypes.data_as(_DOUBLE_P)
    ngroups = int(gamma_fn.nu.shape[1]) if model == 5 else 0
    return model, gp, ngroups


def _unpack_phi(phi_fn, keep):
    """(model_id, TcCel, Pc_Pa, omega, pToPa) for an EOS closure."""
    if phi_fn is None:
        return 0, None, None, None, 1.0

    model = _EOS_MODELS.get(type(phi_fn).__name__)
    if model is None:
        raise ValueError(
            f"{type(phi_fn).__name__} has no compiled RCM equivalent. Add it "
            "to nifco2.f90 and _EOS_MODELS, or use Ideal Gas / SRK.")
    if getattr(phi_fn, "t_to_K", degC_to_K) is not degC_to_K:
        raise ValueError("the compiled RCM solver needs a degC phi_fn; this "
                         "one carries a different temperature convention.")

    # nifco2's SRK_phi takes Tc in Celsius and — despite its header comment
    # saying bar — P and Pc in Pa, since it carries R in J/mol/K. _SRKPhi
    # stores tc in K and pc already in Pa.
    arrs = [_fortran(phi_fn.tc - 273.15, "phi_fn.tc"),
            _fortran(phi_fn.pc_Pa, "phi_fn.pc"),
            _fortran(phi_fn.omega, "phi_fn.omega")]
    keep.extend(arrs)
    return (model, *(a.ctypes.data_as(_DOUBLE_P) for a in arrs),
            float(phi_fn.p_to_Pa))


def curves(x0, P, antoine, gamma_fn=None, phi_fn=None, n_it=250, dxi=0.02,
           maxiter=1000, ftol=1e-12, xtol=1e-12):
    """One residue curve through `x0`, integrated both ways.

    `P` is in the same unit the `antoine` matrix's Psat model emits — i.e. what
    `ThermodynamicsConfig.pressure_in_psat_unit()` returns — exactly as for
    `core.thermodynamics.k_values`. The vapour-pressure model is dispatched on
    the column count (3 Antoine / 6 Wagner / 7 PLXANT), the same rule
    `thermodynamics.antoine_psat` uses.

    Returns (x, y, T): x and y are (2*n_it, C) ordered light -> heavy with the
    seed at row n_it-1, T the bubble-point temperature in degC per row. Raises
    RuntimeError if the library is not built, ValueError on an unsupported
    model.
    """
    lib = load()
    if lib is None:
        raise RuntimeError(f"RCM solver not built ({BUILD_HINT}): {_load_error}")

    keep = []                                   # keeps every buffer alive
    x0a = np.ascontiguousarray(np.asarray(x0, dtype=np.float64))
    psat = _fortran(antoine, "antoine")
    if psat.ndim != 2 or psat.shape[1] not in (3, 6, 7):
        raise ValueError(f"antoine must be (N,3), (N,6) or (N,7), got "
                         f"{psat.shape}")
    ncomps = psat.shape[0]
    if x0a.shape != (ncomps,):
        raise ValueError(f"x0 has {x0a.shape[0]} components, antoine has "
                         f"{ncomps}")
    keep.extend((x0a, psat))

    gmodel, gp, ngroups = _unpack_gamma(gamma_fn, keep)
    emodel, tc, pc, om, p_to_Pa = _unpack_phi(phi_fn, keep)

    params = Params(
        x0=x0a.ctypes.data_as(_DOUBLE_P),
        P=float(P),
        psat=psat.ctypes.data_as(_DOUBLE_P),
        npsat=psat.shape[1],
        gp=gp, gammaModel=gmodel, ngroups=ngroups,
        TcCel=tc, Pc_Pa=pc, omega=om,
        eosModel=emodel, pToPa=p_to_Pa,
        Ncomps=ncomps, dxi=float(dxi), n_it=int(n_it),
        maxiter=int(maxiter), ftol=float(ftol), xtol=float(xtol))

    res = lib.RCM(ctypes.byref(params))
    if not res:
        raise RuntimeError("RCM solver returned no curve (allocation failed)")
    try:
        npts = 2 * int(n_it)
        c = res.contents
        # Copy: these views alias the C heap, which freeCurveMem is about to
        # release.
        x = np.ctypeslib.as_array(c.x, shape=(npts, ncomps)).copy()
        y = np.ctypeslib.as_array(c.y, shape=(npts, ncomps)).copy()
        T = np.ctypeslib.as_array(c.T, shape=(npts,)).copy()
    finally:
        lib.freeCurveMem(res)
    return x, y, T


def _demo():
    if not available():
        print(f"rcm self-check SKIPPED (no library in {LIB_DIR}): {_load_error}")
        return

    from . import thermodynamics as th

    # benzene / toluene / xylene, mmHg + degC Antoine fits (thermodynamics._demo)
    abc = np.array([(6.90565, 1211.033, 220.79),
                    (6.95464, 1344.8, 219.48),
                    (6.99052, 1453.43, 215.31)])
    P = 760.0
    x0 = np.array([0.4, 0.35, 0.25])

    # 1. Ideal: every T on the curve must be the bubble point of its own x, and
    #    the curve must run light (benzene) -> heavy (xylene).
    x, y, T = curves(x0, P, abc, n_it=120, dxi=0.05)
    ok = np.isfinite(T) & (x.min(axis=1) > -1e-6)
    assert ok.sum() > 100, f"only {ok.sum()} usable points"
    err = max(abs(T[i] - th.bubble_T(x[i], P, abc)) for i in np.flatnonzero(ok))
    assert err < 1e-4, f"T is not the bubble point of x (max err {err:g} degC)"
    assert T[ok][0] < T[ok][-1], "curve must run light -> heavy"
    assert x[ok][-1].argmax() == 2, "heavy end should approach pure xylene"
    assert abs(x[119] - x0).max() < 1e-12, "seed must sit at row n_it-1"

    # 2. NRTL: the same check with an activity model, which is what catches a
    #    transposed parameter matrix — an asymmetric b makes it detectable.
    a = np.zeros((3, 3))
    b = np.array([[0.0, 220.0, -60.0], [-95.0, 0.0, 310.0], [140.0, 40.0, 0.0]])
    alpha = np.full((3, 3), 0.3); np.fill_diagonal(alpha, 0.0)
    g = th.nrtl_gamma_fn(a, b, alpha)
    x, y, T = curves(x0, P, abc, gamma_fn=g, n_it=120, dxi=0.05)
    ok = np.isfinite(T) & (x.min(axis=1) > -1e-6)
    err = max(abs(T[i] - th.bubble_T(x[i], P, abc, gamma_fn=g))
              for i in np.flatnonzero(ok))
    assert err < 1e-4, f"NRTL T is not the bubble point (max err {err:g} degC)"

    # 3. y must be the equilibrium vapour of x, not just any root.
    i = int(np.flatnonzero(ok)[len(np.flatnonzero(ok)) // 2])
    K = th.k_values(T[i], P, abc, gamma_fn=g, x=x[i])
    assert np.allclose(y[i], K * x[i], atol=1e-6), "y != K x"

    print("rcm self-check OK")


if __name__ == "__main__":
    _demo()
