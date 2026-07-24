"""Binary VLE diagram math: Txy, Pxy and xy loci for the diagram module.

All thermo comes from the existing seams (bubble_T / k_values / antoine_psat),
so any activity or EOS model wired into k_values plots for free. Pure numpy;
the GUI module (gui/modules/txy_module.py) only draws what these return.

The first two species of `antoine` are treated as the binary pair (component 1
= the x-axis light component). Compositions are [x1, 1-x1].
"""
import numpy as np
from scipy.optimize import brentq

from core.thermodynamics import antoine_psat, bubble_T, k_values


def txy_curve(P, antoine, gamma_fn=None, phi_fn=None, n=101):
    """Bubble and dew loci at fixed P. Returns (x1, y1, T) each length n.

    x1 is the liquid light-component fraction swept over [0,1]; T the
    bubble-point temperature there; y1 the equilibrium vapour — so (x1, T) is
    the bubble curve and (y1, T) the dew curve.
    """
    x1 = np.linspace(0.0, 1.0, n)
    T = np.empty(n)
    y1 = np.empty(n)
    for i, a in enumerate(x1):
        x = np.array([a, 1.0 - a])
        T[i] = bubble_T(x, P, antoine[:2], gamma_fn=gamma_fn, phi_fn=phi_fn)
        y = k_values(T[i], P, antoine[:2], gamma_fn, x, phi_fn) * x
        y1[i] = y[0] / y.sum()
    return x1, y1, T


def _p_bubble(x, T, antoine, gamma_fn, phi_fn):
    """Bubble pressure at fixed T for liquid x. Closed form for ideal vapour;
    a monotone 1-D solve when an EOS phi is active (sum K_i x_i = 1)."""
    psat = antoine_psat(T, antoine)
    gamma = np.asarray(gamma_fn(x, T), float) if gamma_fn is not None \
        else np.ones_like(psat)
    if phi_fn is None:
        return float(np.sum(x * gamma * psat))

    def g(P):
        return float(np.sum(k_values(T, P, antoine, gamma_fn, x, phi_fn) * x)
                     - 1.0)

    lo, hi = psat.min() * 1e-3, psat.max() * 1e3
    return float(brentq(g, lo, hi))


def pxy_curve(T, antoine, gamma_fn=None, phi_fn=None, n=101):
    """Bubble and dew loci at fixed T. Returns (x1, y1, P) each length n."""
    x1 = np.linspace(0.0, 1.0, n)
    P = np.empty(n)
    y1 = np.empty(n)
    for i, a in enumerate(x1):
        x = np.array([a, 1.0 - a])
        P[i] = _p_bubble(x, T, antoine[:2], gamma_fn, phi_fn)
        y = k_values(T, P[i], antoine[:2], gamma_fn, x, phi_fn) * x
        y1[i] = y[0] / y.sum()
    return x1, y1, P


def binary_azeotropes(x1, y1, T, P, antoine, gamma_fn=None, phi_fn=None):
    """Azeotropes on a swept binary: sign changes of (y1 - x1), refined.

    Finds each interior crossing of the y=x diagonal by linear interpolation
    for x_az, then recomputes the exact bubble temperature there. Returns a
    list of {x1, T}. Pure-component ends (x1 in {0,1}) are excluded.
    """
    d = np.asarray(y1) - np.asarray(x1)
    out = []
    for i in range(len(d) - 1):
        if d[i] == 0.0 and 0.0 < x1[i] < 1.0:
            xa = float(x1[i])
        elif d[i] * d[i + 1] < 0.0:
            # linear root of d between the two grid points
            t = d[i] / (d[i] - d[i + 1])
            xa = float(x1[i] + t * (x1[i + 1] - x1[i]))
        else:
            continue
        if not 1e-4 < xa < 1.0 - 1e-4:
            continue
        x = np.array([xa, 1.0 - xa])
        Ta = float(bubble_T(x, P, antoine[:2], gamma_fn=gamma_fn,
                            phi_fn=phi_fn))
        out.append({"x1": xa, "T": Ta})
    return out


def diagram(mode, antoine, gamma_fn=None, phi_fn=None, P=None, T=None,
            n=101):
    """One binary diagram as a dict for the GUI.

    mode: 'Txy' (needs P), 'Pxy' (needs T), 'xy' (needs P).
    Returns {mode, x1, y1, z, zlabel, azeotropes} where z is T (Txy/xy) or P
    (Pxy); azeotropes is a list of {x1, T} (empty for Pxy — the diagonal test
    is a T-diagram concept here).
    """
    if mode == "Pxy":
        if T is None:
            raise ValueError("Pxy needs a fixed temperature")
        x1, y1, z = pxy_curve(T, antoine, gamma_fn, phi_fn, n)
        return {"mode": mode, "x1": x1, "y1": y1, "z": z, "zlabel": "P",
                "azeotropes": []}
    if P is None:
        raise ValueError(f"{mode} needs a fixed pressure")
    x1, y1, T_arr = txy_curve(P, antoine, gamma_fn, phi_fn, n)
    az = binary_azeotropes(x1, y1, T_arr, P, antoine, gamma_fn, phi_fn)
    return {"mode": mode, "x1": x1, "y1": y1, "z": T_arr, "zlabel": "T",
            "azeotropes": az}


def to_csv_rows(data):
    """Diagram dict -> list of rows (header first) for csv.writer."""
    zlabel = data["zlabel"]
    rows = [["x1", "y1", zlabel]]
    for a, b, c in zip(data["x1"], data["y1"], data["z"]):
        rows.append([f"{a:.6g}", f"{b:.6g}", f"{c:.6g}"])
    return rows


def _demo():
    # ethanol / water NRTL (from the bundled fit) should show the minimum-
    # boiling azeotrope near x_EtOH ~ 0.89, ~78.2 C at 1 atm.
    from core import component_db
    from core.thermodynamics import nrtl_gamma

    e, w = component_db.get("ethanol"), component_db.get("water")
    antoine = np.array([e["antoine"], w["antoine"]])          # log10 mmHg / degC
    P = 760.0
    # DB stores i=water,j=ethanol or flipped; build tau at ~350 K in our order
    # [ethanol, water] using the curated NRTL binary directly.
    import numpy as _np
    pair = component_db._find_binary("ethanol", "water")
    rec, flip = pair
    # rec.i/j are canonical names; map into [ethanol, water] order.
    def tau_alpha(Tk):
        # aij + bij/T form; two off-diagonal entries.
        A = _np.zeros((2, 2)); B = _np.zeros((2, 2)); AL = _np.zeros((2, 2))
        # order index: 0=ethanol,1=water
        ij = (0, 1) if rec["i"] == "ethanol" else (1, 0)
        A[ij] = rec["aij"]; A[ij[::-1]] = rec["aji"]
        B[ij] = rec["bij"]; B[ij[::-1]] = rec["bji"]
        AL[ij] = AL[ij[::-1]] = rec["cij"]
        tau = A + B / Tk
        return tau, AL

    def gamma_fn(x, T):
        Tk = T + 273.15
        tau, al = tau_alpha(Tk)
        return nrtl_gamma(_np.asarray(x, float), tau, al)

    x1, y1, Tt = txy_curve(P, antoine, gamma_fn=gamma_fn, n=201)
    az = binary_azeotropes(x1, y1, Tt, P, antoine, gamma_fn=gamma_fn)
    assert az, "no azeotrope found for ethanol/water"
    a = min(az, key=lambda p: abs(p["x1"] - 0.89))
    assert abs(a["T"] - 78.2) < 0.5, a
    assert 0.85 < a["x1"] < 0.92, a

    # benzene / toluene: ideal-ish, no azeotrope.
    bz, tol = component_db.get("benzene"), component_db.get("toluene")
    ant2 = np.array([bz["antoine"], tol["antoine"]])
    x2, y2, T2 = txy_curve(P, ant2, n=101)
    assert not binary_azeotropes(x2, y2, T2, P, ant2)

    # CSV round-trips shape.
    d = diagram("Txy", antoine, gamma_fn=gamma_fn, P=P, n=51)
    rows = to_csv_rows(d)
    assert rows[0] == ["x1", "y1", "T"] and len(rows) == 52
    print("vle_diagrams OK: EtOH/H2O azeotrope x=%.3f, T=%.2f C" %
          (a["x1"], a["T"]))


if __name__ == "__main__":
    _demo()
