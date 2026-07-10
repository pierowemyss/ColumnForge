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


def test_hvap_matches_clausius_clapeyron():
    bad = []
    for rec in db.all_components():
        if rec.get("hvap_tb") is None:
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
