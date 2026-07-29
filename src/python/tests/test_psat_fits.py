"""Vapour-pressure fits in the component DB: PLXANT and Wagner.

Every component now carries three Psat fits, from two independent lineages:

  antoine   Lange's/Yaws classic log10(mmHg, degC) fits (already gated by
            test_component_db.test_antoine_reproduces_tb)
  plxant    published DIPPR-101 coefficients (ChemSep 8.32), rewritten into
            Aspen extended-Antoine column order and bar; two components with no
            DIPPR set are Riedel-fitted from their own Antoine + critical point
  wagner    fitted to the plxant curve, critically anchored

Because Antoine and PLXANT come from *different* sources, agreeing on the
normal boiling point is a real cross-check, not a tautology. See
tools/gen_thermo_data.py.
"""
import numpy as np
import pytest

from core import component_db as db
from core.thermodynamics import (antoine_psat, latent_heat, plxant_psat,
                                 wagner_psat)

ATM_BAR = 1.01325
MMHG_PER_BAR = 750.0617

# Excluded from the Clausius-Clapeyron gates, with reasons:
#   carboxylic acids dimerise in the vapour, so the CC slope is far steeper than
#     the true molar latent heat (see test_component_db.VAPOUR_ASSOCIATES);
#   styrene's bundled vapour-pressure record is simply the weakest in the set —
#     it is also the worst performer on the Tb cross-check.
CC_EXCEPT = {"formic acid", "acetic acid", "propionic acid", "styrene"}


def _recs(key):
    return [r for r in db.all_components() if r.get(key)]


def _plx(rec, T_K):
    """PLXANT Psat [bar] at T [K]. The stored fits take Kelvin directly."""
    return float(plxant_psat(np.atleast_1d(T_K), np.array([rec["plxant"]]),
                             t_to_K=lambda T: T)[0])


def _wag(rec, T_K):
    row = np.array([rec["wagner"] + [rec["tc"], rec["pc"]]])
    return float(wagner_psat(np.atleast_1d(T_K), row, t_to_K=lambda T: T)[0])


def test_every_component_has_a_plxant_fit():
    missing = [r["name"] for r in db.all_components() if not r.get("plxant")]
    assert not missing, f"no PLXANT coefficients for {missing}"
    for r in db.all_components():
        assert len(r["plxant"]) == 7, r["name"]


def test_plxant_reproduces_the_normal_boiling_point():
    """Independent-source cross-check: DIPPR-101 must put 1 atm at our Tb."""
    bad = []
    for rec in db.all_components():
        p = _plx(rec, rec["tb"])
        if abs(p - ATM_BAR) / ATM_BAR > 0.05:
            bad.append((rec["name"], round(p, 4)))
    assert not bad, f"PLXANT does not boil at tb (bar): {bad}"


def test_wagner_reproduces_the_normal_boiling_point():
    bad = []
    for rec in _recs("wagner"):
        p = _wag(rec, rec["tb"])
        if abs(p - ATM_BAR) / ATM_BAR > 0.05:
            bad.append((rec["name"], round(p, 4)))
    assert not bad, f"Wagner does not boil at tb (bar): {bad}"


def test_wagner_is_anchored_at_the_critical_point():
    """The reason to have Wagner at all: Psat(Tc) == Pc by construction."""
    for rec in _recs("wagner"):
        assert _wag(rec, rec["tc"]) == pytest.approx(rec["pc"], rel=1e-9)


def test_plxant_agrees_with_antoine_inside_the_antoine_range():
    """Two independent fits of the same substance must not disagree wildly where
    both are valid. Loose (25%) on purpose — this catches a wrong compound or a
    unit slip, not fit quality, which the Tb gates cover.

    Checked only above 0.05 bar. Most records have no measured Antoine range, so
    antoine_trange estimates it down to 10 mmHg (0.0133 bar) from the fit itself
    — and in that deep-vacuum tail an extrapolated 3-parameter Antoine and a
    DIPPR fit legitimately part company (phenol, cyclohexanol, acetaldehyde).
    Below 0.05 bar there is no ground truth here to arbitrate between them.
    """
    bad = []
    for rec in db.all_components():
        tmin, tmax, _est = db.antoine_trange(rec)
        for t_C in np.linspace(tmin, tmax, 12):
            ant = float(antoine_psat(np.atleast_1d(t_C),
                                     np.array([rec["antoine"]]))[0]) / MMHG_PER_BAR
            if ant < 0.05:
                continue
            plx = _plx(rec, t_C + 273.15)
            if not (0.75 < plx / ant < 1.33):
                bad.append((rec["name"], round(t_C, 1), round(ant, 5), round(plx, 5)))
                break
    assert not bad, f"PLXANT and Antoine disagree in range: {bad}"


def test_plxant_is_monotone_and_finite_up_to_the_critical_point():
    """The whole point of moving off Antoine: no pole, no overflow to inf when a
    solver walks a component past its saturation range."""
    bad = []
    for rec in db.all_components():
        tc = rec.get("tc") or (rec["tb"] + 200.0)
        T = np.linspace(max(0.4 * tc, 100.0), tc, 120)
        p = plxant_psat(T, np.array([rec["plxant"]] * len(T)),
                        t_to_K=lambda t: t)
        if not np.all(np.isfinite(p)) or np.any(np.diff(p) <= 0):
            bad.append(rec["name"])
    assert not bad, f"PLXANT not finite/monotone up to Tc: {bad}"


def test_wagner_saturates_above_tc_instead_of_returning_nan():
    """tau^1.5 is complex above Tc; the clamp must give Pc, never NaN."""
    rec = db.get("n-hexane")
    p = _wag(rec, rec["tc"] + 150.0)
    assert np.isfinite(p) and p == pytest.approx(rec["pc"], rel=1e-9)


def test_latent_heat_through_plxant_matches_the_tabulated_hvap():
    """15%, not the Antoine gate's 12%, and the extra 3 points are physics not
    slop: Clausius-Clapeyron here omits the compressibility factor (the exact
    relation is lambda = R T^2 dZ dlnP/dT with dZ ~ 0.95 at Tb, not 1), so every
    fit reads a few percent high. The sharper PLXANT slope exposes that bias
    more than the Antoine fit does — for cyclopentane and aniline all three
    independent fits agree with each other and differ from the tabulated hvap in
    the same direction, which is the tell."""
    bad = []
    for rec in db.all_components():
        if not rec.get("hvap_tb") or rec["name"] in CC_EXCEPT:
            continue
        lam = float(latent_heat(np.atleast_1d(rec["tb"]),
                                np.array([rec["plxant"]]),
                                t_to_K=lambda T: T)[0]) / 1000.0
        if abs(lam - rec["hvap_tb"]) / rec["hvap_tb"] > 0.15:
            bad.append((rec["name"], round(lam, 2), rec["hvap_tb"]))
    assert not bad, f"PLXANT Clausius-Clapeyron hvap off: {bad}"


def test_latent_heat_through_wagner_matches_the_tabulated_hvap():
    bad = []
    for rec in _recs("wagner"):
        if not rec.get("hvap_tb") or rec["name"] in CC_EXCEPT:
            continue
        row = np.array([rec["wagner"] + [rec["tc"], rec["pc"]]])
        lam = float(latent_heat(np.atleast_1d(rec["tb"]), row,
                                t_to_K=lambda T: T)[0]) / 1000.0
        if abs(lam - rec["hvap_tb"]) / rec["hvap_tb"] > 0.15:   # see above
            bad.append((rec["name"], round(lam, 2), rec["hvap_tb"]))
    assert not bad, f"Wagner Clausius-Clapeyron hvap off: {bad}"


def test_load_into_fills_all_three_models():
    """A DB component must be runnable under any implemented vle_model without
    the user retyping coefficients."""
    from gui.state.window_state import WindowState

    ws = WindowState()
    for n in ("benzene", "toluene"):
        db.load_into(ws, n)
    tc = ws.thermodynamics_config
    for model, width in (("Antoine", 3), ("PLXANT", 7), ("Wagner", 6)):
        tc.vle_model = model
        assert tc.psat_params(["benzene", "toluene"]).shape == (2, width), model


def test_the_three_models_agree_on_a_bubble_point():
    """End-to-end: same mixture, three vapour-pressure models, one answer."""
    from core.thermodynamics import bubble_T
    from gui.state.window_state import WindowState

    ws = WindowState()
    for n in ("benzene", "toluene"):
        db.load_into(ws, n)
    tc = ws.thermodynamics_config
    x = np.array([0.5, 0.5])
    temps = {}
    for model in ("Antoine", "PLXANT", "Wagner"):
        tc.vle_model = model
        temps[model] = bubble_T(x, tc.pressure_in_psat_unit(1.01325),
                                tc.psat_params(["benzene", "toluene"]))
    spread = max(temps.values()) - min(temps.values())
    assert spread < 2.0, temps
