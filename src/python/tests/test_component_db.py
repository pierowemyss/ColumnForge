"""Data-quality gates for the bundled component database (roadmap Month 2).

Every record must be physically self-consistent — this is the anti-hallucination
net for the curated JSON:
  * Antoine fit reproduces the listed normal boiling point within 1 K.
  * hvap_tb matches the Clausius-Clapeyron slope of the fit within 12 %
    (the slope systematically overestimates by a few %).
  * NRTL binaries reproduce their known azeotropes.
"""
import math
import os
import re
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core import component_db as db
from core.thermodynamics import (antoine_psat, bubble_T, latent_heat,
                                 nrtl_gamma_fn, uniquac_gamma_fn,
                                 wilson_gamma_fn)


def test_names_and_cas_unique():
    names, cas = [], []
    for rec in db.all_components():
        names.extend(n.lower() for n in [rec["name"]] + rec.get("aliases", []))
        cas.append(rec["cas"])
    assert len(names) == len(set(names)), "duplicate name/alias"
    assert len(cas) == len(set(cas)), "duplicate CAS"


def test_antoine_reproduces_tb():
    bad = []
    for rec in db.all_components():
        tb_C = rec["tb"] - 273.15
        psat = float(antoine_psat(tb_C, [rec["antoine"]])[0])
        # invert: T at which the fit hits 760 mmHg
        a, b, c = rec["antoine"]
        t760 = b / (a - math.log10(760.0)) - c
        if abs(t760 - tb_C) > 1.0 or not 300.0 < psat < 2000.0:
            bad.append((rec["name"], t760, tb_C))
    assert not bad, f"Antoine/tb mismatch >1K: {bad}"


# Clausius-Clapeyron assumes an ideal, monomeric vapour. Carboxylic acids
# dimerise in the vapour phase, so the slope of their vapour-pressure curve is
# much steeper than their true molar latent heat -- acetic acid reads ~39 kJ/mol
# against a real 23.5. Not a bad fit; the wrong equation for that vapour.
VAPOUR_ASSOCIATES = {"formic acid", "acetic acid", "propionic acid"}


def test_hvap_matches_clausius_clapeyron():
    bad = []
    for rec in db.all_components():
        if rec.get("hvap_tb") is None or rec["name"] in VAPOUR_ASSOCIATES:
            continue
        lam = float(latent_heat(rec["tb"] - 273.15, [rec["antoine"]])[0]) / 1000.0
        if abs(lam - rec["hvap_tb"]) / rec["hvap_tb"] > 0.12:
            bad.append((rec["name"], round(lam, 2), rec["hvap_tb"]))
    assert not bad, f"hvap vs Antoine slope off >12%: {bad}"


def test_scalar_sanity():
    for rec in db.all_components():
        assert 2.0 < rec["mw"] < 300.0, rec["name"]
        assert rec["tb"] > 0.0, rec["name"]
        if rec.get("tc") is not None:                     # null = not curated
            assert rec["tc"] > rec["tb"], rec["name"]
        if rec.get("pc") is not None:
            assert 1.0 < rec["pc"] < 250.0, rec["name"]
        if rec.get("liquid_density") is not None:
            assert 300.0 < rec["liquid_density"] < 2000.0, rec["name"]
        tmin, tmax, _ = db.antoine_trange(rec)
        # inclusive: water's tabulated range ends exactly at Tb = 100 C
        assert tmin <= rec["tb"] - 273.15 <= tmax + 1e-9, rec["name"]


def test_binaries_reference_real_components():
    for b in db.all_binaries():
        assert db.get(b["i"]) is not None, b["i"]
        assert db.get(b["j"]) is not None, b["j"]
        assert 0.0 < b["cij"] <= 0.6


def _azeotrope(pair, P_mmHg=760.0, model="nrtl"):
    """(x1_azeo, T_azeo) minimising/maximising bubble T over x1 for a DB pair."""
    b = db._find_binary(*pair, section=f"{model}_binaries")[0]
    recs = [db.get(n) for n in pair]
    antoine = [r["antoine"] for r in recs]
    idx = {n: k for k, n in enumerate(pair)}
    a = np.zeros((2, 2)); bb = np.zeros((2, 2))
    a[idx[b["i"]], idx[b["j"]]] = b["aij"]; a[idx[b["j"]], idx[b["i"]]] = b["aji"]
    bb[idx[b["i"]], idx[b["j"]]] = b["bij"]; bb[idx[b["j"]], idx[b["i"]]] = b["bji"]
    if model == "nrtl":
        alpha = np.full((2, 2), b["cij"]); np.fill_diagonal(alpha, 0.0)
        gamma = nrtl_gamma_fn(a, bb, alpha)
    elif model == "wilson":
        gamma = wilson_gamma_fn(a, bb)
    else:
        rq = np.array([r["uniquac_rq"] for r in recs])
        gamma = uniquac_gamma_fn(rq[:, 0], rq[:, 1], a, bb)
    xs = np.linspace(0.01, 0.99, 197)
    Ts = [bubble_T(np.array([x, 1 - x]), P_mmHg, antoine, gamma_fn=gamma)
          for x in xs]
    ext = min if Ts[len(Ts)//2] < (Ts[0] + Ts[-1]) / 2 else max
    k = int(np.argmin(Ts) if ext is min else np.argmax(Ts))
    return float(xs[k]), float(Ts[k])


def test_nrtl_azeotropes():
    x, T = _azeotrope(("ethanol", "water"))
    assert abs(x - 0.894) < 0.03 and abs(T - 78.15) < 0.5, (x, T)
    x, T = _azeotrope(("2-propanol", "water"))
    assert 0.62 < x < 0.72 and abs(T - 80.3) < 0.7, (x, T)
    x, T = _azeotrope(("acetone", "chloroform"))          # max-boiling
    assert 0.28 < x < 0.45 and abs(T - 64.5) < 1.0, (x, T)
    x, T = _azeotrope(("acetone", "methanol"))            # min-boiling
    assert 0.72 < x < 0.86 and abs(T - 55.5) < 1.0, (x, T)


def _unifac_azeotrope(pair, P_mmHg=760.0):
    """(x1, T) at the bubble-T extremum for a pair run on UNIFAC alone.

    UNIFAC takes no binary parameters, so this is the end-to-end check that the
    group assignments AND the a_mn table are right — a wrong group or a wrong
    interaction moves the azeotrope, and nothing else in the suite would notice.
    """
    from core.thermodynamics import load_unifac_db, unifac_gamma_fn

    recs = [db.get(n) for n in pair]
    gamma = unifac_gamma_fn([r["unifac_groups"] for r in recs],
                            load_unifac_db(), names=list(pair))
    antoine = [r["antoine"] for r in recs]
    xs = np.linspace(0.01, 0.99, 197)
    Ts = [bubble_T(np.array([x, 1 - x]), P_mmHg, antoine, gamma_fn=gamma)
          for x in xs]
    ext = min if Ts[len(Ts) // 2] < (Ts[0] + Ts[-1]) / 2 else max
    k = int(np.argmin(Ts) if ext is min else np.argmax(Ts))
    return float(xs[k]), float(Ts[k])


def test_unifac_azeotropes():
    """Predicted (not fitted) azeotropes — the gate on the whole group pipeline."""
    x, T = _unifac_azeotrope(("methyl acetate", "methanol"))   # ~0.66 at 53.6 C
    assert 0.58 < x < 0.74 and abs(T - 53.6) < 2.5, (x, T)
    x, T = _unifac_azeotrope(("ethanol", "water"))             # ~0.894 at 78.15 C
    assert 0.80 < x < 0.95 and abs(T - 78.2) < 2.5, (x, T)
    x, T = _unifac_azeotrope(("benzene", "2-propanol"))        # ~0.61 at 71.9 C
    assert 0.50 < x < 0.72 and abs(T - 71.9) < 3.0, (x, T)


def test_unifac_covers_the_curated_database():
    """Every DB component that carries groups must resolve against the group
    table, and the common chemistries must have their interaction pairs — this
    is what stops a silent-ideal regression from creeping back in."""
    from core.thermodynamics import load_unifac_db, unifac_gamma_fn

    udb = load_unifac_db()
    recs = _unifac_records()
    assert len(recs) >= 74
    # solvents that any distillation example is likely to reach for
    common = ["water", "ethanol", "methanol", "benzene", "toluene", "acetone",
              "n-hexane", "n-heptane", "methyl acetate", "ethyl acetate",
              "2-propanol", "acetic acid", "cyclohexane", "chloroform"]
    for i, a in enumerate(common):
        for b in common[i + 1:]:
            ra, rb = db.get(a), db.get(b)
            unifac_gamma_fn([ra["unifac_groups"], rb["unifac_groups"]],
                            udb, names=[a, b])       # raises if a pair is missing


def test_wilson_uniquac_azeotropes():
    """Wilson (SVA 7e) and UNIQUAC (fitted to the gated NRTL) ethanol/water
    pairs must land the azeotrope inside the same gate as NRTL."""
    for model in ("wilson", "uniquac"):
        x, T = _azeotrope(("ethanol", "water"), model=model)
        assert abs(x - 0.894) < 0.03 and abs(T - 78.15) < 0.5, (model, x, T)


def test_uniquac_rq_sanity():
    n = 0
    for rec in db.all_components():
        rq = rec.get("uniquac_rq")
        if rq is None:
            continue
        n += 1
        r, q = rq
        assert 0.5 <= r <= 8.0 and 0.5 <= q <= 8.0, rec["name"]
    assert n >= 13
    # group-additive consistency: r and q grow with each added CH2/CH3
    r_b, q_b = db.get("benzene")["uniquac_rq"]
    r_t, q_t = db.get("toluene")["uniquac_rq"]
    r_x, q_x = db.get("p-xylene")["uniquac_rq"]
    assert r_b < r_t < r_x and q_b < q_t < q_x
    # every uniquac_binaries end carries r/q — the model is unusable otherwise
    for b in db.all_binaries("uniquac_binaries"):
        assert db.get(b["i"]).get("uniquac_rq"), b["i"]
        assert db.get(b["j"]).get("uniquac_rq"), b["j"]


def _group_atoms():
    """Atoms each UNIFAC subgroup contributes — the objective check on a group
    assignment: the groups must add up to the molecular formula, no tolerance to
    argue about. Ships in the group DB (tools/gen_thermo_data.py copies it from
    the DDBST tables) so it is not a second hand-maintained list to drift."""
    from core.thermodynamics import load_unifac_db

    return {g: {el: n for el, n in at.items() if n}
            for g, at in load_unifac_db()["atoms"].items()}


def _formula_atoms(formula):
    atoms = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if el:
            atoms[el] = atoms.get(el, 0) + (int(n) if n else 1)
    return atoms


def _unifac_records():
    return [r for r in db.all_components() if r.get("unifac_groups")]


def test_unifac_groups_add_up_to_the_formula():
    """A wrong group count is a wrong molecule — catch it on atoms, not vibes."""
    group_atoms = _group_atoms()
    recs = _unifac_records()
    assert len(recs) >= 74, f"only {len(recs)} components carry UNIFAC groups"
    bad = []
    for rec in recs:
        atoms = {}
        for g, count in rec["unifac_groups"].items():
            for el, n in group_atoms[g].items():
                atoms[el] = atoms.get(el, 0) + n * count
        if atoms != _formula_atoms(rec["formula"]):
            bad.append((rec["name"], rec["formula"], atoms))
    assert not bad, f"UNIFAC groups do not match the formula: {bad}"


def test_unifac_groups_exist_in_the_group_db():
    from core.thermodynamics import load_unifac_db

    subgroups = load_unifac_db()["subgroups"]
    assert set(_group_atoms()) <= set(subgroups)
    for rec in _unifac_records():
        for g, count in rec["unifac_groups"].items():
            assert g in subgroups, (rec["name"], g)
            assert isinstance(count, int) and count > 0, (rec["name"], g, count)


def test_unifac_group_sums_agree_with_tabulated_rq():
    """Where a record carries both, the group-sum r/q must reproduce the
    tabulated UNIQUAC r/q — except for the alcohols, whose UNIQUAC r/q come from
    the UNIQUAC table itself and legitimately differ from UNIFAC group sums
    (ethanol: 2.1055 vs 2.5755). The two models are not the same tabulation, so
    the exception list is documented, not tolerance-fudged."""
    from core.thermodynamics import load_unifac_db

    sub = load_unifac_db()["subgroups"]
    tabulated_from_uniquac_table = {"ethanol", "2-propanol"}
    checked = 0
    for rec in _unifac_records():
        rq = rec.get("uniquac_rq")
        if not rq or rec["name"] in tabulated_from_uniquac_table:
            continue
        r = sum(sub[g][2] * n for g, n in rec["unifac_groups"].items())
        q = sum(sub[g][3] * n for g, n in rec["unifac_groups"].items())
        assert abs(r - rq[0]) < 1e-4 and abs(q - rq[1]) < 1e-4, (rec["name"], r, q, rq)
        checked += 1
    assert checked >= 8


def test_load_into_fills_unifac_groups():
    """UNIFAC needs no binary parameters, so DB load is the whole setup path:
    if groups don't ride along, every user retypes them by hand."""
    from gui.state.window_state import WindowState

    ws = WindowState()
    for n in ("methanol", "acetic acid", "methyl acetate", "water"):
        db.load_into(ws, n)
    assert ws.species["methyl acetate"].unifac_groups == {"CH3": 1, "CH3COO": 1}
    assert ws.species["water"].unifac_groups == {"H2O": 1}
    # a species with no classic UNIFAC-VLE subgroup at all (ammonia) loads
    # without groups, and UNIFAC then refuses honestly rather than running ideal
    db.load_into(ws, "ammonia")
    assert ws.species["ammonia"].unifac_groups == {}
    ws.thermodynamics_config.activity_model = "UNIFAC"
    try:
        ws.build_gamma_fn(["methanol", "ammonia"])
    except Exception as exc:
        assert "group" in str(exc).lower(), exc
    else:
        raise AssertionError("UNIFAC accepted a species with no groups")


def test_unifac_refuses_a_missing_interaction_pair():
    """Only 1270 of 2862 main-group pairs are published. An unpublished pair used
    to default to a = 0 — an ideal residual for that pair, with nothing on screen
    to say so. It must raise instead."""
    from gui.state.window_state import WindowState

    ws = WindowState()
    for n in ("carbon disulfide", "formic acid"):
        db.load_into(ws, n)
    ws.thermodynamics_config.activity_model = "UNIFAC"
    try:
        ws.build_gamma_fn(["carbon disulfide", "formic acid"])
    except ValueError as exc:
        assert "interaction parameter" in str(exc), exc
    else:
        raise AssertionError("UNIFAC silently ran an unparameterised pair")


def test_load_into_windowstate_roundtrip():
    from gui.state.persistence import encode_state, decode_state
    from gui.state.window_state import WindowState

    ws = WindowState()
    for n in ("Benzene", "toluene", "para-xylene"):       # alias + case lookup
        db.load_into(ws, n)
    assert set(ws.species) == {"benzene", "toluene", "p-xylene"}
    assert abs(ws.species["benzene"].mw - 78.114) < 1e-9
    p = ws.thermodynamics_config.get_component_params("toluene")
    assert p.antoine_a == 6.95464 and p.tc == 591.75
    assert p.antoine_tmin == 6.0 and p.antoine_tmax == 136.0
    # estimated range lands too
    pb = ws.thermodynamics_config.get_component_params("p-xylene")
    assert pb.antoine_tmin == 27.0                        # explicit for p-xylene
    # BTX has no curated NRTL pairs -> flagged, not silently absent
    info = db.load_into(WindowState(), "benzene")
    assert info["missing_pairs"] == []

    back = decode_state(encode_state(ws.to_dict()))
    tp = back["thermodynamics_config"].get_component_params("toluene")
    assert tp.antoine_tmin == 6.0 and tp.antoine_tmax == 136.0


def test_load_into_fills_nrtl_both_directions():
    from gui.state.window_state import WindowState

    ws = WindowState()
    db.load_into(ws, "water")
    info = db.load_into(ws, "ethanol")
    assert ("ethanol", "water") in info["nrtl_pairs"]
    b = ws.thermodynamics_config.binary
    assert b.nrtl_aij[("ethanol", "water")] == -0.8009
    assert b.nrtl_aij[("water", "ethanol")] == 3.4578
    assert b.nrtl_bij[("ethanol", "water")] == 246.18
    assert b.nrtl_cij[("water", "ethanol")] == 0.3
    # Wilson/UNIQUAC binaries and structural r/q ride along
    assert b.wilson_bij[("ethanol", "water")] == -192.3803
    assert b.wilson_bij[("water", "ethanol")] == -480.7997
    assert b.uniquac_bij[("ethanol", "water")] == -30.5121
    p = ws.thermodynamics_config.get_component_params("ethanol")
    assert (p.uniquac_r, p.uniquac_q) == (2.1055, 1.972)

    # a DB pair with no curated params is reported missing
    info = db.load_into(ws, "benzene")
    assert ("benzene", "water") in info["missing_pairs"]
    assert ("benzene", "ethanol") in info["missing_pairs"]


def test_coverage_flags_match_the_record():
    cov = db.coverage(db.get("benzene"), existing_names=["water"])
    assert cov["antoine"] and cov["plxant"] and cov["wagner"]
    assert cov["unifac"] and cov["uniquac_rq"] and cov["srk"]
    assert cov["nrtl_pairs"] == 0 and cov["nrtl_missing"] == 1
    # ammonia has no classic UNIFAC-VLE subgroup — the flag must say so
    assert not db.coverage(db.get("ammonia"))["unifac"]
    # every flag in COVERAGE_LABELS is actually produced
    assert set(k for k, _ in db.COVERAGE_LABELS) <= set(cov)


def test_search_dialog_shows_parameter_coverage():
    """The dialog must say what a component can be run with *before* it is added
    — otherwise the first sign of a gap is a solver refusing."""
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from PySide6.QtCore import Qt
    from gui.panels.species_search_dialog import SpeciesSearchDialog

    dlg = SpeciesSearchDialog(existing_names=[])
    rows = {dlg.results.item(i).data(Qt.UserRole): dlg.results.item(i).text()
            for i in range(dlg.results.count())}
    assert "UNIFAC" in rows["benzene"] and "SRK" in rows["benzene"]
    assert "PLX" in rows["benzene"] and "WAG" in rows["benzene"]
    assert "UNIFAC" not in rows["ammonia"]

    for i in range(dlg.results.count()):
        if dlg.results.item(i).data(Qt.UserRole) == "ammonia":
            dlg._show_details(dlg.results.item(i))
    text = dlg.details.text()
    assert "Not available:" in text and "UNIFAC" in text.split("Not available:")[1]


def test_search_dialog_and_tab_wiring(monkeypatch):
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from PySide6.QtCore import Qt
    from gui.panels.species_search_dialog import SpeciesSearchDialog
    import gui.panels.species_search_dialog as ssd_mod
    from gui.main_window import MainWindow

    dlg = SpeciesSearchDialog(existing_names=["benzene"])
    dlg.search_edit.setText("benz")
    labels = [dlg.results.item(i).data(Qt.UserRole)
              for i in range(dlg.results.count())]
    assert "benzene" in labels and "ethylbenzene" in labels
    bz = labels.index("benzene")
    assert not (dlg.results.item(bz).flags() & Qt.ItemIsEnabled)  # already added
    # first enabled hit is preselected; accept() picks it
    assert dlg.results.currentItem().flags() & Qt.ItemIsEnabled
    dlg.accept()
    assert dlg.selected_name and dlg.selected_name != "benzene"

    # tab wiring: DB add lands in window_state with thermo params filled
    w = MainWindow()
    tab = w.init_tab

    class FakeDialog:
        def __init__(self, *a, **k):
            self.selected_name = "toluene"

        def exec(self):
            return True

    monkeypatch.setattr(ssd_mod, "SpeciesSearchDialog", FakeDialog)
    tab._add_species_from_db()
    ws = w.window_state
    assert "toluene" in ws.species and ws.species["toluene"].mw is not None
    assert ws.thermodynamics_config.get_component_params("toluene").antoine_a


if __name__ == "__main__":
    for f in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        f()
    print("component-db gates OK")
