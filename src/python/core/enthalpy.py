"""Shortcut enthalpy layer shared by both solver stacks.

Data actually in the component DB is a *constant* liquid Cp (J/mol·K) and the
latent heat at the normal boiling point (kJ/mol) plus Tb/Tc. That fixes the
model:

    reference state: saturated liquid at T_ref (enthalpies are relative, so the
                     absolute level is arbitrary — only differences drive the
                     energy balance)
    h_liq(T)  = cp_liq * (T - T_ref)                         [J/mol]
    h_vap(T)  = h_liq(T) + dHvap(T)                          [J/mol]
    dHvap(T)  = hvap_Tb * ((Tc - T)/(Tc - Tb)) ** 0.38       (Watson)

All temperatures are in KELVIN here (the DB's Tb/Tc are), unlike the °C that
the Antoine/thermo layer carries — callers convert at the seam.

ponytail: constant liquid Cp + Watson latent is the shortcut model; upgrade to
ideal-gas Cp polynomials + a residual-enthalpy departure when the DB grows
those fields (roadmap Month 5 aspirational Cp polynomials).
"""
import numpy as np

T_REF = 298.15          # K — reference for the (relative) enthalpy scale
WATSON_N = 0.38


def watson_hvap(T_K, hvap_Tb, tb, tc, n=WATSON_N):
    """Latent heat [J/mol] at T from the value at the normal boiling point,
    Watson-corrected. hvap_Tb in kJ/mol (DB units). Above Tc, dHvap -> 0."""
    T_K = np.asarray(T_K, float)
    hvap_Tb = np.asarray(hvap_Tb, float) * 1.0e3       # kJ/mol -> J/mol
    tb = np.asarray(tb, float)
    tc = np.asarray(tc, float)
    frac = np.clip((tc - T_K) / (tc - tb), 0.0, None)
    return hvap_Tb * frac ** n


def liquid_enthalpy(T_K, cp_liq, t_ref=T_REF):
    """Molar liquid enthalpy [J/mol] on the saturated-liquid-at-T_ref scale."""
    return np.asarray(cp_liq, float) * (np.asarray(T_K, float) - t_ref)


def vapor_enthalpy(T_K, cp_liq, hvap_Tb, tb, tc, t_ref=T_REF):
    """Molar vapor enthalpy [J/mol] = liquid enthalpy + latent heat at T."""
    return liquid_enthalpy(T_K, cp_liq, t_ref) + watson_hvap(T_K, hvap_Tb, tb, tc)


def mix_enthalpy(z, h_pure):
    """Ideal-mixing molar enthalpy of a stream: sum_i z_i h_i [J/mol]
    (no excess enthalpy — consistent with the shortcut model)."""
    return float(np.asarray(z, float) @ np.asarray(h_pure, float))


def enthalpy_fns(cp_liq, hvap_Tb, tb, tc, t_ref=T_REF):
    """Build (hL, hV) closures over component arrays: hL(T_K)/hV(T_K) return
    the per-component molar enthalpy vectors [J/mol]. Handy for the solvers,
    which evaluate the same components at many stage temperatures."""
    cp_liq = np.asarray(cp_liq, float)
    hvap_Tb = np.asarray(hvap_Tb, float)
    tb = np.asarray(tb, float)
    tc = np.asarray(tc, float)

    def hL(T_K):
        return liquid_enthalpy(T_K, cp_liq, t_ref)

    def hV(T_K):
        return vapor_enthalpy(T_K, cp_liq, hvap_Tb, tb, tc, t_ref)

    return hL, hV


def _demo():
    # ethanol / water: cp_liq J/mol-K, hvap_Tb kJ/mol, Tb/Tc K (DB values)
    cp = np.array([112.0, 75.3])
    hv = np.array([38.56, 40.66])
    tb = np.array([351.44, 373.15])
    tc = np.array([513.92, 647.10])

    # Watson recovers the tabulated latent heat exactly at Tb
    lat_tb = watson_hvap(tb, hv, tb, tc)
    assert np.allclose(lat_tb, hv * 1e3, rtol=1e-12), lat_tb

    # latent heat falls with rising T, and vanishes at (or above) Tc
    assert watson_hvap(400.0, hv[0], tb[0], tc[0]) < hv[0] * 1e3
    assert watson_hvap(tc[0], hv[0], tb[0], tc[0]) == 0.0
    assert watson_hvap(tc[0] + 50, hv[0], tb[0], tc[0]) == 0.0

    hL, hV = enthalpy_fns(cp, hv, tb, tc)

    # at T_ref the liquid enthalpy is exactly zero (reference state)
    assert np.allclose(hL(T_REF), 0.0)
    # vapor sits above liquid by the latent heat everywhere below Tc
    for T in (330.0, 351.44, 373.15):
        assert np.all(hV(T) > hL(T))
        assert np.allclose(hV(T) - hL(T), watson_hvap(T, hv, tb, tc))
    # both are monotonically increasing in T
    assert np.all(hL(360.0) > hL(340.0))
    assert np.all(hV(360.0) > hV(340.0))

    # heat to vaporize an equimolar liquid at 350 K = mix latent heat
    z = np.array([0.5, 0.5])
    dh_vap = mix_enthalpy(z, hV(350.0)) - mix_enthalpy(z, hL(350.0))
    assert abs(dh_vap - mix_enthalpy(z, watson_hvap(350.0, hv, tb, tc))) < 1e-9
    assert 30e3 < dh_vap < 45e3, dh_vap    # ~ tens of kJ/mol, right ballpark

    print(f"enthalpy self-check OK (equimolar EtOH/H2O dHvap @ 350 K "
          f"= {dh_vap/1e3:.1f} kJ/mol)")


if __name__ == "__main__":
    _demo()
