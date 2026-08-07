"""Loader for the optional compiled thermo kernels — f2py'd `src/native/nifco2.f90`.

Same equations as the NumPy paths in `thermodynamics.py`; they agree to the last
bit on NRTL, Wilson, UNIQUAC, Margules and UNIFAC. So this module buys speed,
never a different answer, and `thermodynamics.set_native()` is the switch.

Nothing here is required: the `.so` only exists if somebody ran
`make -C src/native`, and it is compiler- and arch-specific, so an install with
no library must behave exactly as it always did. That makes `ImportError` a
normal outcome, not a failure — hence `load()` returning None rather than
raising. A wrong-arch build fails here at `dlopen`, not at compile time (see
CLAUDE.md on python3.13 vs the x86_64 Homebrew), and lands in the same bucket.
"""

import importlib
import os
import sys
from functools import lru_cache

#: src/native/lib — where the Makefile puts the built extension.
LIB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "native", "lib")


@lru_cache(maxsize=1)
def load():
    """The compiled `nifco` extension module, or None if it will not load."""
    if LIB_DIR not in sys.path:
        sys.path.append(LIB_DIR)
    try:
        return importlib.import_module("nifco")
    except ImportError:
        return None


def available():
    return load() is not None


def _demo():
    lib = load()
    if lib is None:
        print(f"nifco self-check SKIPPED (no library in {LIB_DIR})")
        return

    import numpy as np
    from . import thermodynamics as th

    rng = np.random.default_rng(0)
    n = 3
    x = rng.random(n); x /= x.sum()
    T = 75.0
    a = rng.normal(0, 0.5, (n, n)); np.fill_diagonal(a, 0.0)
    b = rng.normal(0, 200.0, (n, n)); np.fill_diagonal(b, 0.0)
    alpha = np.full((n, n), 0.3); np.fill_diagonal(alpha, 0.0)
    r = np.array([1.4, 2.1, 3.2]); q = np.array([1.4, 1.9, 2.7])
    A = (a + a.T) / 2.0; np.fill_diagonal(A, 0.0)

    pairs = [
        ("NRTL", lib.nrtl_gamma(x, T, a, b, alpha), th.nrtl_gamma_fn(a, b, alpha)(x, T)),
        ("Wilson", lib.wilson_gamma(x, T, a, b), th.wilson_gamma_fn(a, b)(x, T)),
        ("UNIQUAC", lib.uniquac_gamma(x, T, r, q, a, b),
         th.uniquac_gamma_fn(r, q, a, b)(x, T)),
        ("Margules", lib.margules_gamma(x, T, A), th.margules_gamma_fn(A)(x, T)),
    ]
    g = th.unifac_gamma_fn([{"CH3": 1, "CH2": 1, "OH": 1}, {"H2O": 1}],
                           th.load_unifac_db())
    xb = np.array([0.4, 0.6])
    pairs.append(("UNIFAC", lib.unifac_gamma(xb, T, g.nu, g.R, g.Q, g.a_sub),
                  g(xb, T)))

    for name, native, numpy_ in pairs:
        err = float(np.max(np.abs(np.asarray(native) - np.asarray(numpy_))))
        assert err < 1e-12, f"{name} native/numpy disagree by {err:g}"

    # set_native() must be a no-op on the numbers, and must refuse a closure
    # carrying a non-degC temperature convention (the Fortran hard-codes it).
    ref = th.nrtl_gamma_fn(a, b, alpha)(x, T)
    kelvin = th.nrtl_gamma_fn(a, b, alpha, t_to_K=lambda T: T)
    ref_k = kelvin(x, T)
    assert th.set_native(True) is True
    try:
        assert np.allclose(th.nrtl_gamma_fn(a, b, alpha)(x, T), ref, rtol=0, atol=1e-12)
        assert np.allclose(kelvin(x, T), ref_k, rtol=0, atol=1e-12)
    finally:
        th.set_native(False)
    print("nifco self-check OK")


if __name__ == "__main__":
    _demo()
