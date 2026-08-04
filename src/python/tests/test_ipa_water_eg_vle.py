"""The extractive IPA/water/EG example, gated against its own measured VLE.

`docs/examples/extractive_ipa_water_eg.colx` shipped with **ethanol's** record in
the `2-propanol` slot -- Antoine 8.20417/1642.89/230.30, Tc 513.92, Pc 61.48,
omega 0.649, uniquac r/q 2.1055/1.972 are all ethanol's values exactly. It boiled
2-propanol at 78.3 C instead of 82.3, dragged the IPA/water azeotrope from 0.688
to 0.79, and made the BVM module report a phantom junction on the extractive
saddle arm. The file contradicted itself the whole time: `species["2-propanol"]`
carries the right mw and tb, so a Tb cross-check would have caught it on day one.

That is what this file is: the cross-check, tied to the measurements rather than
to whatever we believed when we typed the coefficients in. Both sources are
NIST TRC ThermoML, sitting in `docs/papers/ipa_water_eg Data/`:

    Lin & Tu, Fluid Phase Equilib. 368 (2014) 104-111   10.1016/j.fluid.2014.02.006
        IPA/water binary T-x-y at 101.3 kPa (17 pts) + the measured azeotrope
    Zhang et al., J. Chem. Eng. Data 61 (2016) 2596-2604  10.1021/acs.jced.6b00264
        water/EG binary (6 pts) and IPA/water/EG ternary (16 pts)

The NRTL binaries in the .colx are NOT under suspicion and are not fitted here --
with the vapour pressure corrected they reproduce all three sets to well inside
the tolerances below, which is the point of gating on all three.
"""
import json
import os

import numpy as np
import pytest

from gui.state import persistence
from gui.state.window_state import WindowState
from gui.modules.module_thermo import session_models
from side_features.bvm.thermo_adapter import ColumnForgeThermo

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
DATA_DIR = os.path.join(_ROOT, "docs", "papers", "ipa_water_eg Data")
EXAMPLE = os.path.join(_ROOT, "docs", "examples", "extractive_ipa_water_eg.colx")

LIN_TU = os.path.join(DATA_DIR, "j.fluid.2014.02.006.json")
ZHANG = os.path.join(DATA_DIR, "acs.jced.6b00264.json")

MW = {"2-propanol": 60.096, "water": 18.015, "ethylene glycol": 62.068}

#: ThermoML names -> the example's species names.
ALIAS = {"propan-2-ol": "2-propanol", "water": "water",
         "1,2-ethanediol": "ethylene glycol"}


# --------------------------------------------------------------- ThermoML

def _doc(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compounds(doc):
    out = {}
    for c in doc["Compound"]:
        nm = c["sCommonName"]
        out[c["RegNum"]["nOrgNum"]] = nm[0] if isinstance(nm, list) else nm
    return out


def _block(doc, n):
    return next(p for p in doc["PureOrMixtureData"]
                if p["nPureOrMixtureDataNumber"] == n)


def _series(doc, n):
    """(vars, props) for one PureOrMixtureData block.

    Each is a list of per-point dicts keyed by the *component* the column
    describes (via its RegNum), or by None for a whole-mixture column such as
    the boiling temperature. Reading the RegNum rather than assuming a column
    order is what keeps this honest: Zhang's two ternary vapour columns are
    water then 2-propanol, which is not the Component order.
    """
    comps = _compounds(doc)
    pm = _block(doc, n)
    vkey = {v["nVarNumber"]: comps.get(v["VariableID"].get("RegNum", {}).get("nOrgNum"))
            for v in pm["Variable"]}
    pkey = {p["nPropNumber"]:
            comps.get(p.get("Property-MethodID", {}).get("RegNum", {}).get("nOrgNum"))
            for p in pm["Property"]}
    vs, ps = [], []
    for nv in pm.get("NumValues", []):
        vs.append({vkey[v["nVarNumber"]]: float(v["nVarValue"])
                   for v in nv.get("VariableValue", [])})
        ps.append({pkey[p["nPropNumber"]]: float(p["nPropValue"])
                   for p in nv.get("PropertyValue", []) if "nPropValue" in p})
    return vs, ps


# ------------------------------------------------------------ the example

def _thermo(vle_model="Antoine"):
    ws = WindowState()
    ws.load_from_dict(persistence.load_colx(EXAMPLE))
    ws.thermodynamics_config.vle_model = vle_model
    order = ws.get_species_names()
    antoine, gamma_fn, phi_fn, label, note = session_models(ws, order)
    assert not note, f"thermo silently degraded: {note}"
    P = ws.thermodynamics_config.pressure_in_psat_unit(ws.pressure)
    return order, P, ColumnForgeThermo(antoine, gamma_fn=gamma_fn, phi_fn=phi_fn)


def _x(order, **frac):
    x = np.array([frac.get(n, 0.0) for n in order], float)
    return x / x.sum()


# ------------------------------------------------------------------ tests

@pytest.mark.parametrize("vle_model", ["Antoine", "PLXANT", "Wagner"])
def test_pure_boiling_points(vle_model):
    """Every Psat fit must put the three pure components on their measured Tb.

    Lin & Tu measure pure 2-propanol at 355.46 K (82.31 C) and water at 373.15 K
    in the same apparatus; EG's 470.45 K is the .colx's own `species.tb`. The
    ethanol mix-up showed up here as a 4 K miss on 2-propanol in all three fits
    at once, which is the tell that the *record* was wrong rather than one fit.
    """
    order, P, tp = _thermo(vle_model)
    measured = {"2-propanol": 82.31, "water": 100.00, "ethylene glycol": 197.30}
    for i, name in enumerate(order):
        Tb = float(tp.bubble_T(np.eye(len(order))[i], P))
        assert abs(Tb - measured[name]) < 0.5, (
            f"{vle_model}: {name} boils at {Tb:.2f} C, measured {measured[name]}")


def test_psat_fits_agree_on_tb():
    """The three fits are independent representations of one substance.

    0.6 K, not 0.2: ethylene glycol's shipped Antoine and PLXANT come from
    different lineages and sit 0.54 K apart on Tb (197.49 vs 196.96, either side
    of the measured 197.3). That spread is ordinary and predates this file --
    `test_pure_boiling_points` is the gate that actually matters, since it
    measures each fit against reality rather than against its siblings. What
    this catches is one fit describing a *different substance*, which is a 4 K
    effect, not a 0.5 K one.
    """
    tbs = {}
    for vle_model in ("Antoine", "PLXANT", "Wagner"):
        order, P, tp = _thermo(vle_model)
        tbs[vle_model] = [float(tp.bubble_T(np.eye(len(order))[i], P))
                          for i in range(len(order))]
    ref = tbs["Antoine"]
    for model, got in tbs.items():
        assert np.allclose(got, ref, atol=0.6), f"{model} disagrees with Antoine: {got} vs {ref}"


def test_ipa_water_azeotrope():
    """The measured azeotrope, Lin & Tu sets 13/14 -- x = 0.688, T = 353.21 K.

    This is the single number the ethanol record got most wrong (0.79 / 77.6 C),
    and it is the one the extractive design is most sensitive to, since the whole
    point of the EG is to move past it.
    """
    doc = _doc(LIN_TU)
    _, aT = _series(doc, 13)
    _, ax = _series(doc, 14)
    T_meas = list(aT[0].values())[0] - 273.15
    x_meas = list(ax[0].values())[0]

    order, P, tp = _thermo()
    scan = np.arange(0.30, 0.999, 0.001)
    err = []
    for xi in scan:
        y, T = tp.bubble(_x(order, **{"2-propanol": xi, "water": 1 - xi}), P)
        err.append(abs(y[order.index("2-propanol")] - xi))
    x_model = scan[int(np.argmin(err))]
    _, T_model = tp.bubble(
        _x(order, **{"2-propanol": x_model, "water": 1 - x_model}), P)

    assert abs(x_model - x_meas) < 0.02, f"azeotrope at x={x_model:.3f}, measured {x_meas}"
    assert abs(T_model - T_meas) < 0.5, f"azeotrope at T={T_model:.2f} C, measured {T_meas:.2f}"


def test_ipa_water_binary():
    """Lin & Tu sets 11/12: 17-point T-x-y at 101.3 kPa."""
    doc = _doc(LIN_TU)
    xs, Ts = _series(doc, 11)
    _, ys = _series(doc, 12)
    order, P, tp = _thermo()
    i_ipa = order.index("2-propanol")

    dT, dy = [], []
    for xv, Tp, yp in zip(xs, Ts, ys):
        x1 = xv["propan-2-ol"]
        if x1 in (0.0, 1.0):
            continue
        y, T = tp.bubble(_x(order, **{"2-propanol": x1, "water": 1 - x1}), P)
        dT.append(T + 273.15 - Tp[None])
        dy.append(y[i_ipa] - yp["propan-2-ol"])

    assert len(dT) == 15
    assert np.sqrt(np.mean(np.square(dT))) < 0.4, f"RMS dT = {np.sqrt(np.mean(np.square(dT))):.3f} K"
    assert np.sqrt(np.mean(np.square(dy))) < 0.015, f"RMS dy = {np.sqrt(np.mean(np.square(dy))):.4f}"


def test_water_eg_binary():
    """Zhang sets 10/11: 6-point water/EG T-x-y. Gates the water/EG NRTL pair."""
    doc = _doc(ZHANG)
    xs, Ts = _series(doc, 10)
    _, ys = _series(doc, 11)
    order, P, tp = _thermo()
    i_w = order.index("water")

    dT, dy = [], []
    for xv, Tp, yp in zip(xs, Ts, ys):
        xw = xv["water"]
        y, T = tp.bubble(_x(order, water=xw, **{"ethylene glycol": 1 - xw}), P)
        dT.append(T + 273.15 - Tp[None])
        dy.append(y[i_w] - yp["water"])

    assert len(dT) == 6
    assert np.sqrt(np.mean(np.square(dT))) < 1.0, f"RMS dT = {np.sqrt(np.mean(np.square(dT))):.3f} K"
    assert np.sqrt(np.mean(np.square(dy))) < 0.015, f"RMS dy = {np.sqrt(np.mean(np.square(dy))):.4f}"


def _ternary_points(order):
    """Zhang sets 4/5/6 as (x, T_exp[K], y_IPA_exp, y_water_exp).

    Composition arrives as (mole fraction 2-propanol on an EG-free basis, MASS
    fraction EG), so it has to be converted before it means anything -- and the
    two vapour columns are water then 2-propanol, which `_series` reads off the
    RegNum rather than assuming.
    """
    doc = _doc(ZHANG)
    xs, Ts = _series(doc, 4)
    _, yw = _series(doc, 5)
    _, yi = _series(doc, 6)
    out = []
    for xv, Tp, aw, ai in zip(xs, Ts, yw, yi):
        x1f = xv["propan-2-ol"]                    # EG-free mole fraction IPA
        wEG = xv["1,2-ethanediol"]                 # MASS fraction EG
        n_eg = wEG / MW["ethylene glycol"]
        mbar = x1f * MW["2-propanol"] + (1 - x1f) * MW["water"]
        n_rest = (1 - wEG) / mbar
        x = _x(order, **{"2-propanol": x1f * n_rest,
                         "water": (1 - x1f) * n_rest,
                         "ethylene glycol": n_eg})
        out.append((x, Tp[None], ai["propan-2-ol"], aw["water"]))
    return out


def test_ternary():
    """Zhang sets 4/5/6: 16-point IPA/water/EG. Gates the IPA/EG pair.

    ABSOLUTE dy, which gates the water-rich half and almost nothing on the
    water-trace half -- see `test_ternary_k_water_where_the_extractive_effect_is`.
    """
    order, P, tp = _thermo()
    i_ipa, i_w = order.index("2-propanol"), order.index("water")

    dT, dyi, dyw = [], [], []
    for x, T_exp, yi_exp, yw_exp in _ternary_points(order):
        y, T = tp.bubble(x, P)
        dT.append(T + 273.15 - T_exp)
        dyi.append(y[i_ipa] - yi_exp)
        dyw.append(y[i_w] - yw_exp)

    assert len(dT) == 16
    assert np.sqrt(np.mean(np.square(dT))) < 0.4, f"RMS dT = {np.sqrt(np.mean(np.square(dT))):.3f} K"
    assert np.sqrt(np.mean(np.square(dyi))) < 0.015, f"RMS dy_IPA = {np.sqrt(np.mean(np.square(dyi))):.4f}"
    assert np.sqrt(np.mean(np.square(dyw))) < 0.015, f"RMS dy_water = {np.sqrt(np.mean(np.square(dyw))):.4f}"


def test_ternary_k_water_where_the_extractive_effect_is():
    """RELATIVE K_water on Zhang's water-trace slice -- the corner the extractive
    section is decided in, and the one `test_ternary` cannot see.

    Half of Zhang's ternary points sit at x_water = 0.011 - 0.045 with x_EG up to
    0.80. `test_ternary`'s gate is RMS |dy_water| < 0.015 ABSOLUTE; at x_water
    ~ 0.02, y_water ~ 0.02, so that tolerance is ~100% relative error on K_water,
    and the glycol pairs shipped with a systematic 12-17% bias it could not see:

        x_IPA  x_w    x_EG  |  Kw_exp   was    refit
       0.2996 0.0157 0.6848 |   1.321   1.476   1.34
       0.4898 0.0256 0.4846 |   1.206   1.379   1.24
       0.6772 0.0354 0.2874 |   1.217   1.424   1.25

    (the water-RICH slice was always fine, |ratio - 1| <= 0.03 and on the other
    side, which is how the bias survived.) The water/EG and IPA/EG NRTL pairs in
    the example were refitted against this slice to close it; the water/EG binary
    improved too (RMS dT 0.610 -> 0.121 K).

    Why it matters here rather than in the third decimal of a duty. The extractive
    section's pinch topology turns over at a transcritical branching point where
    K_water(x) = L/V on the water-free isopropanol/glycol edge: below it the edge
    pinch is an UNSTABLE NODE and the ternary pinch that detaches from it is a
    SADDLE, which is exactly the structure Brueggemann & Marquardt report at
    r_min (their Figure 4, left). Above it they exchange, and that is their
    r = 5.0 picture -- infeasible.

    This gate does NOT get us there, and was not expected to. K_water/a on that
    edge went from 1.30-1.40 to 1.12-1.25 and the ternary unstable node moved
    most of the way toward the edge (x_water 0.203 -> 0.110 at r = 2.042), but it
    did not cross: the paper's window still never opens. Cutting gamma_water a
    further 25% does reproduce their three pinches exactly, and no fit to these
    data will do that, so what is left is their Wilson/Aspen model rather than our
    arithmetic. See docs/adr/0004.
    """
    order, P, tp = _thermo()
    i_w = order.index("water")

    err = []
    for x, _T, _yi, yw_exp in _ternary_points(order):
        if x[i_w] >= 0.05:                      # water-rich slice: test_ternary
            continue
        y, _ = tp.bubble(x, P)
        err.append(abs((y[i_w] / yw_exp) - 1.0))   # K ratio: x_water cancels

    assert len(err) == 8, len(err)
    assert max(err) < 0.05, f"max |K_water error| = {max(err):.3f}"


def test_species_tb_matches_psat():
    """The self-consistency check the file already had the data for.

    `species[n].tb` is entered independently of the Psat coefficients, so the two
    disagreeing means one of them is another substance. This is the cheap gate
    that would have caught the ethanol record immediately.
    """
    ws = WindowState()
    ws.load_from_dict(persistence.load_colx(EXAMPLE))
    order, P, tp = _thermo()
    for i, name in enumerate(order):
        tb = ws.species[name].tb
        if tb is None:
            continue
        Tb = float(tp.bubble_T(np.eye(len(order))[i], P)) + 273.15
        assert abs(Tb - tb) < 1.0, (
            f"{name}: Psat boils it at {Tb:.2f} K but species.tb says {tb:.2f} K")
