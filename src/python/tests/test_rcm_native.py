"""The compiled RCM solver must call the same thermodynamics Python does.

`RCM_solv.c` computes no properties of its own: it forwards opaque, column-major
parameter blobs into `src/native/nifco2.f90`. That forwarding is exactly where a
row/column transposition or a wrong argument order hides — the previous version
of the file read the Antoine table transposed for years, which is why the GUI
had quietly defaulted to Extended Antoine.

So the gate is per-model and per-point rather than per-curve: take the (x, y, T)
the C solver returned and re-evaluate the VLE residual in Python with the same
closures. If C used the model Python thinks it used, the residual is zero. This
deliberately does not compare against `gui.plotting.residue_curve` point for
point — that would fold in Euler-step differences and prove much less.

All 3 Psat models x 6 activity models x 2 EOS are covered. Binary parameters are
synthetic and asymmetric on purpose: symmetric matrices cannot catch a
transposition, and the property under test is the plumbing, not a literature
value.
"""

import json
import os

import numpy as np
import pytest

from core import rcm
from core import thermodynamics as th

pytestmark = pytest.mark.skipif(
    not rcm.available(),
    reason=f"RCM solver not built ({rcm.BUILD_HINT}): {rcm.load_error()}")

COMPS = ("ethanol", "water", "benzene")
BAR_TO_PSAT_UNIT = {"Antoine": 750.0617, "PLXANT": 1.0, "Wagner": 1.0}
P_BAR = 1.01325


@pytest.fixture(scope="module")
def db():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "core", "data", "components.json")
    with open(path) as fh:
        return {c["name"]: c for c in json.load(fh)["components"]}


def psat_matrix(db, model):
    """The same matrices ThermodynamicsConfig.psat_params builds: (N,3)
    Antoine, (N,7) PLXANT, (N,6) Wagner (a..d plus Tc/Pc)."""
    rows = []
    for name in COMPS:
        c = db[name]
        if model == "Antoine":
            rows.append(c["antoine"])
        elif model == "PLXANT":
            rows.append(c["plxant"])
        else:
            rows.append(list(c["wagner"]) + [c["tc"], c["pc"]])
    return np.array(rows, float)


def gamma_closure(db, model):
    n = len(COMPS)
    # Asymmetric on purpose: a transposed matrix must change the answer.
    a = np.zeros((n, n))
    b = np.array([[0.0, 220.0, -60.0],
                  [-95.0, 0.0, 310.0],
                  [140.0, 40.0, 0.0]])
    alpha = np.full((n, n), 0.3)
    np.fill_diagonal(alpha, 0.0)

    if model == "Ideal":
        return None
    if model == "NRTL":
        return th.nrtl_gamma_fn(a, b, alpha)
    if model == "Wilson":
        # Lambda = exp(a + b/T); keep b small so Lambda stays well-conditioned.
        return th.wilson_gamma_fn(a, b / 10.0)
    if model == "UNIQUAC":
        r = np.array([db[c]["uniquac_rq"][0] for c in COMPS])
        q = np.array([db[c]["uniquac_rq"][1] for c in COMPS])
        return th.uniquac_gamma_fn(r, q, a, b / 10.0)
    if model == "Margules":
        A = np.array([[0.0, 0.9, 0.5], [0.9, 0.0, 1.3], [0.5, 1.3, 0.0]])
        return th.margules_gamma_fn(A)
    if model == "UNIFAC":
        return th.unifac_gamma_fn([db[c]["unifac_groups"] for c in COMPS],
                                  th.load_unifac_db(), names=list(COMPS))
    raise AssertionError(model)


def phi_closure(db, eos, vle_model):
    if eos == "Ideal Gas":
        return None
    return th.srk_phi_fn(np.array([db[c]["tc"] for c in COMPS]),
                         np.array([db[c]["pc"] for c in COMPS]),
                         np.array([db[c]["omega"] for c in COMPS]),
                         p_to_Pa=1.0e5 / BAR_TO_PSAT_UNIT[vle_model])


@pytest.mark.parametrize("vle_model", ["Antoine", "PLXANT", "Wagner"])
@pytest.mark.parametrize(
    "activity", ["Ideal", "NRTL", "Wilson", "UNIQUAC", "Margules", "UNIFAC"])
@pytest.mark.parametrize("eos", ["Ideal Gas", "SRK"])
def test_native_matches_python_thermo(db, vle_model, activity, eos):
    antoine = psat_matrix(db, vle_model)
    gamma_fn = gamma_closure(db, activity)
    phi_fn = phi_closure(db, eos, vle_model)
    P = P_BAR * BAR_TO_PSAT_UNIT[vle_model]

    x, y, T = rcm.curves(np.array([0.4, 0.35, 0.25]), P, antoine,
                         gamma_fn=gamma_fn, phi_fn=phi_fn, n_it=40, dxi=0.05)

    ok = np.flatnonzero(np.isfinite(T) & (x.min(axis=1) > -1e-9)
                        & (np.abs(x.sum(axis=1) - 1.0) < 1e-6))
    assert len(ok) >= 40, f"only {len(ok)}/{len(T)} usable points"

    # Re-evaluate what C solved, in Python, with the same closures:
    #   x_i gamma_i Psat_i phi^sat_i - y_i phi^V_i P = 0
    for i in ok[:: max(1, len(ok) // 12)]:
        xi, yi, Ti = x[i], y[i], T[i]
        psat = th.antoine_psat(Ti, antoine)
        g = np.ones(3) if gamma_fn is None else np.asarray(gamma_fn(xi, Ti))
        if phi_fn is None:
            phi_v = phi_sat = np.ones(3)
        else:
            phi_v = np.asarray(phi_fn(yi, Ti, P))
            phi_sat = np.asarray(phi_fn.pure(Ti, psat))
        resid = xi * g * psat * phi_sat - yi * phi_v * P
        scale = max(P, float(np.max(np.abs(xi * g * psat * phi_sat))))
        assert np.max(np.abs(resid)) / scale < 1e-7, (
            f"{vle_model}/{activity}/{eos} row {i}: VLE residual "
            f"{np.max(np.abs(resid)):.3g} (scale {scale:.3g})")
        assert abs(yi.sum() - 1.0) < 1e-8


@pytest.mark.parametrize("vle_model", ["Antoine", "PLXANT", "Wagner"])
@pytest.mark.parametrize(
    "activity", ["Ideal", "NRTL", "Wilson", "UNIQUAC", "Margules", "UNIFAC"])
def test_native_T_is_the_bubble_point(db, vle_model, activity):
    """With an ideal vapour the C residual and `bubble_T` are the same equation,
    so they must agree to solver tolerance. (Under SRK they differ in the
    second order: k_values estimates the vapour as normalised K_raoult*x, while
    the C solver uses its converged y — hence the residual check above instead.)
    """
    antoine = psat_matrix(db, vle_model)
    gamma_fn = gamma_closure(db, activity)
    P = P_BAR * BAR_TO_PSAT_UNIT[vle_model]

    x, y, T = rcm.curves(np.array([0.4, 0.35, 0.25]), P, antoine,
                         gamma_fn=gamma_fn, n_it=40, dxi=0.05)
    ok = np.flatnonzero(np.isfinite(T) & (x.min(axis=1) > -1e-9))
    for i in ok[:: max(1, len(ok) // 8)]:
        want = th.bubble_T(x[i], P, antoine, gamma_fn=gamma_fn)
        assert abs(T[i] - want) < 1e-4, (
            f"{vle_model}/{activity} row {i}: T {T[i]:.6f} != bubble_T "
            f"{want:.6f}")


def test_curve_runs_light_to_heavy(db):
    """Ordering contract: row n_it-1 is the seed, rows run light -> heavy, so
    the array drops straight into gui.plotting.plot_residue_curves (whose arrow
    points along the march) exactly like the pure-Python engine's output."""
    antoine = psat_matrix(db, "Antoine")
    P = P_BAR * BAR_TO_PSAT_UNIT["Antoine"]
    x0 = np.array([0.4, 0.35, 0.25])
    x, y, T = rcm.curves(x0, P, antoine, n_it=60, dxi=0.05)

    assert np.allclose(x[59], x0), "seed must sit at row n_it-1"
    ok = np.flatnonzero(np.isfinite(T) & (x.min(axis=1) > -1e-9))
    assert T[ok[0]] < T[ok[-1]], "temperature must increase along the curve"
    # ethanol/water/benzene: benzene is the light end, water the heavy one.
    assert x[ok[-1]].argmax() == 1, "heavy end should approach pure water"


def test_unsupported_model_is_refused(db):
    """An activity closure with no compiled equivalent must raise, not silently
    fall back to ideal — 'nothing is silently ignored'."""
    class _Bogus:
        def __call__(self, x, T):
            return np.ones(len(x))

    with pytest.raises(ValueError, match="no compiled RCM equivalent"):
        rcm.curves(np.array([0.4, 0.35, 0.25]), 760.0,
                   psat_matrix(db, "Antoine"), gamma_fn=_Bogus(), n_it=5)


def test_non_celsius_closure_is_refused(db):
    """The Fortran hard-codes Tk = Tcel + 273.15, so a closure carrying another
    temperature convention must be rejected rather than quietly mis-scaled."""
    n = len(COMPS)
    a = np.zeros((n, n))
    b = np.zeros((n, n))
    alpha = np.full((n, n), 0.3)
    np.fill_diagonal(alpha, 0.0)
    kelvin = th.nrtl_gamma_fn(a, b, alpha, t_to_K=lambda T: T)

    with pytest.raises(ValueError, match="degC gamma_fn"):
        rcm.curves(np.array([0.4, 0.35, 0.25]), 760.0,
                   psat_matrix(db, "Antoine"), gamma_fn=kelvin, n_it=5)
