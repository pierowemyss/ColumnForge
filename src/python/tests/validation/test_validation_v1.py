"""Validation suite v1 (roadmap Month 3) — the acceptance gate for every later
solver/thermo change. Four literature-grade cases, each run through every
applicable solver:

  1. BTX ideal            — Wang-Henke's classic system; anchored to pure-
                            component boiling points (NIST), overall balances,
                            near-total light-key recovery, and cross-solver
                            agreement (BVM's own cross-check vs
                            solve_bubble_point lives in matrix_bvm/tests).
  2. Depropanizer, PLXANT — light hydrocarbons at 4 atm through the PLXANT
                            (extended Antoine) path. The PLXANT set is the
                            EXACT ln-Pa transform of the Tb-validated Antoine
                            fits, so the two paths must agree to solver
                            precision — validates the PLXANT plumbing with
                            zero transcription risk.
  3. Ethanol/water, NRTL  — Horsley azeotrope (x1 = 0.894, 78.15 C at 1 atm)
                            caps the distillate; ideal Raoult overshoots it.
  4. Methanol/water, NRTL — zeotropic: VLE point at x1 = 0.5 vs Perry's
                            (T ~= 73 C, y1 ~= 0.78), plus a clean column split.

Values asserted here come from pure-component data already gated by
test_component_db.py, thermodynamic limits (azeotropes), and exact algebraic
equivalences — not from transcribed stage tables.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for _record

from core import component_db as db
from core.column_solvers import build_solver_input, solve_bubble_point, \
    solve_inside_out
from core.thermodynamics import (antoine_Tsat, bubble_T, k_values,
                                 nrtl_gamma_fn, uniquac_gamma_fn,
                                 wilson_gamma_fn)

from _record import check  # noqa: E402  (sys.path is set just above)

SOLVERS = (solve_bubble_point, solve_inside_out)

#: Inside-Out cannot close the 4-atm depropanizer's outer temperature loop: the
#: composition front translates one stage per ~4 outer passes, a period-4 limit
#: cycle whose ratio never holds steady, so the Aitken acceleration never fires
#: and the residual only creeps (1.1e-2 at 50 passes, 3.1e-3 at 400). The
#: PRODUCTS are unaffected — they match the bubble-point solve, which closes the
#: same column in 140 iterations — so the cases below still assert every physical
#: claim for both solvers and only excuse Inside-Out from `converged`.
#: `test_inside_out_depropanizer_limit_cycle` is the canary: fix the outer loop
#: and it xpasses, which is the signal to delete this.
IO_CANNOT_CLOSE = (solve_inside_out,)


def _gate(solver, prof):
    """Assert the run converged, unless it is the one documented exception."""
    if solver not in IO_CANNOT_CLOSE:
        assert prof["converged"], prof["message"]


def _antoine(*names):
    return np.array([db.get(n)["antoine"] for n in names])


def _gamma_from_db(n1, n2):
    """2x2 NRTL gamma_fn for (n1, n2) from the bundled binary DB."""
    b, flipped = db._find_binary(n1, n2)
    ij = (0, 1) if not flipped else (1, 0)
    a = np.zeros((2, 2)); bb = np.zeros((2, 2))
    a[ij] = b["aij"]; a[ij[::-1]] = b["aji"]
    bb[ij] = b["bij"]; bb[ij[::-1]] = b["bji"]
    alpha = np.array([[0.0, b["cij"]], [b["cij"], 0.0]])
    return nrtl_gamma_fn(a, bb, alpha)


def _closes_balance(prof, feeds, atol=1e-3):
    lhs = sum(F * np.asarray(z) for _, F, z in feeds)
    rhs = prof["D"] * prof["xD"] + prof["B"] * prof["xB"]
    assert np.allclose(lhs, rhs, atol=atol), f"balance: {lhs} vs {rhs}"


# --- Case 1: BTX, ideal ------------------------------------------------------

def test_case1_btx_ideal():
    comps = ["benzene", "toluene", "p-xylene"]
    antoine = _antoine(*comps)
    zF = np.array([0.4, 0.35, 0.25])
    si = build_solver_input(n_stages=20, comps=comps, feeds=[(10, 100.0, zF)],
                            R=3.0, D=40.0, pressure=760.0, antoine=antoine)
    profs = []
    for solver in SOLVERS:
        p = solver(si)
        assert p["converged"]
        _closes_balance(p, [(10, 100.0, zF)])
        # top boils like (nearly pure) benzene, bottoms between toluene/xylene
        check("distillate temperature", p["T"][0], 80.1, 1.5, unit=" degC",
              layer="MESH", case=f"BTX, 1 atm ({solver.__name__})",
              source="NIST WebBook: normal boiling point of benzene, 80.1 degC "
                     "— a near-pure benzene distillate must boil there",
              url="https://webbook.nist.gov/cgi/cbook.cgi?ID=C71432&Mask=4")
        assert 110.6 < p["T"][-1] < 138.4            # between Tb toluene/p-xylene
        # 40 kmol/h benzene fed, D = 40: near-total light-key recovery
        assert p["xD"][0] > 0.99
        assert p["xB"][0] < 0.005
        profs.append(p)
    # two independent MESH implementations agree on the products
    assert np.allclose(profs[0]["xD"], profs[1]["xD"], atol=0.02)
    assert np.allclose(profs[0]["T"], profs[1]["T"], atol=1.0)


# --- Case 2: depropanizer through the PLXANT path ----------------------------

def _plxant_from_antoine(abc):
    """Exact transform: log10(P[mmHg]) = A - B/(C + T[degC])  ->
    ln(P[Pa]) = (ln133.322 + A ln10) - B ln10 / ((C - 273.15) + T[K]).
    Identical function, different encoding — zero data risk."""
    A, B, C = abc
    ln10 = np.log(10.0)
    return [np.log(133.322) + A * ln10, -B * ln10, C - 273.15,
            0.0, 0.0, 0.0, 1.0]


def test_case2_depropanizer_plxant():
    comps = ["ethane", "propane", "n-butane", "n-pentane"]
    antoine = _antoine(*comps)
    plxant = np.array([_plxant_from_antoine(abc) for abc in antoine])
    zF = np.array([0.05, 0.30, 0.40, 0.25])
    P = 4.0 * 760.0                                   # mmHg -> Pa for PLXANT
    feeds = [(12, 100.0, zF)]
    kw = dict(n_stages=25, comps=comps, feeds=feeds, R=2.5, D=35.0)

    si_a = build_solver_input(pressure=P, antoine=antoine, **kw)
    si_p = build_solver_input(pressure=P * 133.322, antoine=plxant, **kw)
    for solver in SOLVERS:
        pa, pp = solver(si_a), solver(si_p)
        _gate(solver, pa); _gate(solver, pp)
        # The PLXANT path IS the Antoine path in another encoding, so every
        # difference here is solver noise, not thermodynamics. How much noise is
        # a property of direct substitution: `converged` means the last STEP fell
        # under tol, and for a geometric tail of ratio r the distance still left
        # to the fixed point is ~ tol * r/(1-r). This column runs r ~ 0.99994,
        # an amplification of ~1e4, and the measured spread between the two
        # encodings tracks it exactly: max|dT| = 9.3e-3 at tol=1e-6, 7.4e-4 at
        # 1e-8, 1.1e-6 at 1e-10. So a 1e-3 gate on T was never a statement about
        # the encodings — it was a statement about how far the solver had run.
        #
        # The PRODUCTS carry none of that amplification (max|dxD| = 2e-9 at every
        # tolerance above), so they take the tight gate and T takes the honest one.
        assert np.allclose(pa["xD"], pp["xD"], atol=1e-6)
        assert np.allclose(pa["xB"], pp["xB"], atol=1e-6)
        # Inside-Out never converges here (IO_CANNOT_CLOSE); the two encodings
        # then sit at different phases of the same limit cycle, ~its residual apart.
        assert np.allclose(pa["T"], pp["T"], atol=(
            0.05 if pa["converged"]
            else 10.0 * max(pa["residual"], pp["residual"])))
        _closes_balance(pa, feeds)
        # C3/C4 split: propane up, butane down (D = 35 ~= C2 + C3 fed)
        d_c3 = pa["D"] * pa["xD"][1] / (100.0 * zF[1])
        b_c4 = pa["B"] * pa["xB"][2] / (100.0 * zF[2])
        assert d_c3 > 0.95, f"propane recovery {d_c3:.3f}"
        assert b_c4 > 0.95, f"butane recovery {b_c4:.3f}"
        # column runs between the C2 and C5 boiling points at 4 atm
        t_lo = antoine_Tsat(P, antoine[0]); t_hi = antoine_Tsat(P, antoine[3])
        assert t_lo - 1.0 < min(pa["T"]) and max(pa["T"]) < t_hi + 1.0


@pytest.mark.xfail(strict=True, reason="IO outer loop period-4 limit cycle; "
                                       "see IO_CANNOT_CLOSE")
def test_inside_out_depropanizer_limit_cycle():
    """Canary for the one convergence failure the suite tolerates.

    Two claims, and only the second is allowed to fail: Inside-Out's products on
    the depropanizer agree with the bubble-point solve (asserted outright, so a
    regression here is a hard failure), and its outer loop closes (xfail). When
    someone gives that loop a Newton step this xpasses — delete the marker and
    IO_CANNOT_CLOSE together.
    """
    comps = ["ethane", "propane", "n-butane", "n-pentane"]
    zF = np.array([0.05, 0.30, 0.40, 0.25])
    si = build_solver_input(n_stages=25, comps=comps, feeds=[(12, 100.0, zF)],
                            R=2.5, D=35.0, pressure=4.0 * 760.0,
                            antoine=_antoine(*comps))
    bp, io = solve_bubble_point(si), solve_inside_out(si)
    assert bp["converged"], bp["message"]
    # the answer is right even though the residual will not settle
    assert np.allclose(bp["xD"], io["xD"], atol=2e-3), (bp["xD"], io["xD"])
    assert np.allclose(bp["xB"], io["xB"], atol=2e-3), (bp["xB"], io["xB"])
    assert io["converged"], io["message"]        # <- the xfail


# --- Case 3: ethanol/water, NRTL vs the azeotrope ----------------------------

def test_case3_ethanol_water_nrtl():
    comps = ["ethanol", "water"]
    antoine = _antoine(*comps)
    gamma = _gamma_from_db(*comps)

    # Horsley: minimum-boiling azeotrope at x_EtOH ~= 0.894, 78.15 C, 1 atm
    xs = np.linspace(0.5, 0.999, 300)
    Ts = [bubble_T(np.array([x, 1 - x]), 760.0, antoine, gamma_fn=gamma)
          for x in xs]
    k = int(np.argmin(Ts))
    src = dict(layer="Thermodynamics", case="ethanol/water, 1 atm",
               source="Horsley, Azeotropic Data III (ACS 1973); the same "
                      "azeotrope is tabulated by DDBST",
               url="http://www.ddbst.com/en/EED/VLE/VLE%20Ethanol%3BWater.php")
    check("azeotrope composition x_EtOH", xs[k], 0.894, 0.03, **src)
    check("azeotrope temperature", Ts[k], 78.15, 0.5, unit=" degC", **src)

    zF = np.array([0.10, 0.90])
    feeds = [(15, 100.0, zF)]
    kw = dict(n_stages=30, comps=comps, feeds=feeds, R=5.0, D=12.0,
              pressure=760.0, antoine=antoine)
    for solver in SOLVERS:
        p = solver(build_solver_input(gamma_fn=gamma, **kw))
        assert p["converged"]
        # NRTL runs converge with a slow substitution tail: 0.1% of feed
        _closes_balance(p, feeds, atol=0.05)
        # distillate approaches but cannot cross the azeotrope
        assert 0.70 < p["xD"][0] <= 0.894 + 0.02, p["xD"]
        assert p["xB"][0] < 0.01                     # bottoms ~pure water
        assert abs(p["T"][-1] - 100.0) < 1.0         # Tb water (bottoms)
        assert 78.0 < p["T"][0] < 82.0               # near-azeotrope top
    # Activity coefficients matter: dilute ethanol has gamma_inf ~= 5, so NRTL
    # strips it far more easily than Raoult — the ideal run recovers LESS.
    p_ideal = solve_bubble_point(build_solver_input(**kw))
    assert p["xD"][0] > p_ideal["xD"][0] + 0.05, (p["xD"], p_ideal["xD"])


# --- Case 4: methanol/water, NRTL (zeotropic) --------------------------------

def test_case4_methanol_water_nrtl():
    comps = ["methanol", "water"]
    antoine = _antoine(*comps)
    gamma = _gamma_from_db(*comps)

    # Perry's VLE at 1 atm, x_MeOH = 0.5: T ~= 73 C, y_MeOH ~= 0.78
    x = np.array([0.5, 0.5])
    T = bubble_T(x, 760.0, antoine, gamma_fn=gamma)
    y = k_values(T, 760.0, antoine, gamma, x) * x
    src = dict(layer="Thermodynamics", case="methanol/water, 1 atm, x1 = 0.5",
               source="Perry's Chemical Engineers' Handbook, methanol/water "
                      "VLE; same binary tabulated by DDBST",
               url="http://www.ddbst.com/en/EED/VLE/VLE%20Methanol%3BWater.php")
    check("bubble temperature", T, 73.1, 1.5, unit=" degC", **src)
    check("vapour composition y_MeOH", y[0], 0.78, 0.05, **src)

    zF = np.array([0.5, 0.5])
    feeds = [(12, 100.0, zF)]
    kw = dict(n_stages=25, comps=comps, feeds=feeds, R=2.5, D=50.0,
              pressure=760.0, antoine=antoine, gamma_fn=gamma)
    profs = []
    for solver in SOLVERS:
        p = solver(build_solver_input(**kw))
        assert p["converged"]
        _closes_balance(p, feeds, atol=0.05)         # NRTL substitution tail
        assert p["xD"][0] > 0.90 and p["xB"][0] < 0.10
        assert abs(p["T"][0] - 64.7) < 2.0           # Tb methanol (distillate)
        assert abs(p["T"][-1] - 100.0) < 2.5         # Tb water (bottoms)
        profs.append(p)
    assert np.allclose(profs[0]["xD"], profs[1]["xD"], atol=0.02)


# --- Case 5: ethanol/water across NRTL / Wilson / UNIQUAC --------------------

def _gamma_model_from_db(n1, n2, model):
    """2x2 Wilson/UNIQUAC gamma_fn from the bundled binary DB sections."""
    b, flipped = db._find_binary(n1, n2, section=f"{model}_binaries")
    ij = (0, 1) if not flipped else (1, 0)
    a = np.zeros((2, 2)); bb = np.zeros((2, 2))
    a[ij] = b["aij"]; a[ij[::-1]] = b["aji"]
    bb[ij] = b["bij"]; bb[ij[::-1]] = b["bji"]
    if model == "wilson":
        return wilson_gamma_fn(a, bb)
    rq = np.array([db.get(n)["uniquac_rq"] for n in (n1, n2)])
    return uniquac_gamma_fn(rq[:, 0], rq[:, 1], a, bb)


def test_case5_ethanol_water_cross_model():
    """Month-4 gate: NRTL (Aspen databank), Wilson (SVA 7e) and UNIQUAC —
    independently parameterised activity models — must agree on the same
    ethanol/water column within model-difference tolerance, and all must
    respect the azeotrope cap."""
    comps = ["ethanol", "water"]
    antoine = _antoine(*comps)
    zF = np.array([0.10, 0.90])
    feeds = [(15, 100.0, zF)]
    kw = dict(n_stages=30, comps=comps, feeds=feeds, R=5.0, D=12.0,
              pressure=760.0, antoine=antoine)
    gammas = {"nrtl": _gamma_from_db(*comps),
              "wilson": _gamma_model_from_db(*comps, "wilson"),
              "uniquac": _gamma_model_from_db(*comps, "uniquac")}
    profs = {}
    for name, g in gammas.items():
        p = solve_bubble_point(build_solver_input(gamma_fn=g, **kw))
        assert p["converged"], name
        _closes_balance(p, feeds, atol=0.05)
        assert 0.70 < p["xD"][0] <= 0.894 + 0.02, (name, p["xD"])
        profs[name] = p
    # products and terminal temperatures agree; mid-column T is NOT compared
    # pointwise — the composition front shifts a fraction of a stage between
    # models, moving the local T by a few degC without changing the split
    for m in ("wilson", "uniquac"):
        assert abs(profs[m]["xD"][0] - profs["nrtl"]["xD"][0]) < 0.03, \
            (m, profs[m]["xD"][0], profs["nrtl"]["xD"][0])
        assert abs(profs[m]["xB"][0] - profs["nrtl"]["xB"][0]) < 0.005, m
        assert abs(profs[m]["T"][0] - profs["nrtl"]["T"][0]) < 0.5, m
        assert abs(profs[m]["T"][-1] - profs["nrtl"]["T"][-1]) < 0.5, m


def test_case6_depropanizer_srk():
    """Month-4 gate: SRK vapour-phase phi on the 4-atm depropanizer. The
    correction compresses relative volatilities (phi_heavy < phi_light < 1),
    so the split must survive but move in the less-sharp direction, and the
    temperature profile shifts only modestly."""
    from core.thermodynamics import srk_phi_fn

    comps = ["ethane", "propane", "n-butane", "n-pentane"]
    antoine = _antoine(*comps)
    crit = np.array([(db.get(n)["tc"], db.get(n)["pc"], db.get(n)["omega"])
                     for n in comps])
    phi = srk_phi_fn(crit[:, 0], crit[:, 1], crit[:, 2])   # degC/mmHg fits
    zF = np.array([0.05, 0.30, 0.40, 0.25])
    feeds = [(12, 100.0, zF)]
    kw = dict(n_stages=25, comps=comps, feeds=feeds, R=2.5, D=35.0,
              pressure=4.0 * 760.0, antoine=antoine)

    for solver in SOLVERS:
        p_id = solver(build_solver_input(**kw))
        p_sr = solver(build_solver_input(phi_fn=phi, **kw))
        _gate(solver, p_id); _gate(solver, p_sr)
        # phi couples K to x, so the substitution tail is a touch longer than
        # ideal-K runs; 0.01 kmol on a 100 kmol feed is converged for a gate
        _closes_balance(p_sr, feeds, atol=0.01)
        d_c3 = p_sr["D"] * p_sr["xD"][1] / (100.0 * zF[1])
        b_c4 = p_sr["B"] * p_sr["xB"][2] / (100.0 * zF[2])
        assert d_c3 > 0.90 and b_c4 > 0.90, (d_c3, b_c4)
        # phi < 1 lowers K, so bubble points rise on average — but the split
        # itself shifts too, so a few stages may come out marginally cooler;
        # assert the direction in the mean and a modest magnitude, not per stage.
        # The pointwise bound must clear a whole stage-step: SRK moves the C3/C4
        # front ~1 stage lower, and adjacent stages at the front differ ~15 degC
        # (converged truth is max dT ~ 12; the old <10 gate only passed because
        # both runs used to park their fronts at the same unconverged spot).
        dT = np.asarray(p_sr["T"]) - np.asarray(p_id["T"])
        assert np.mean(dT) > 0.0 and np.min(dT) > -1.0 and np.max(dT) < 20.0, \
            (dT.min(), dT.mean(), dT.max())
        # and the C3/C4 split is (slightly) less sharp than Raoult claims
        d_c3_id = p_id["D"] * p_id["xD"][1] / (100.0 * zF[1])
        assert d_c3 <= d_c3_id + 1e-9, (d_c3, d_c3_id)


if __name__ == "__main__":
    test_case1_btx_ideal()
    test_case2_depropanizer_plxant()
    test_case3_ethanol_water_nrtl()
    test_case4_methanol_water_nrtl()
    test_case5_ethanol_water_cross_model()
    test_case6_depropanizer_srk()
    print("validation suite v1 OK")
