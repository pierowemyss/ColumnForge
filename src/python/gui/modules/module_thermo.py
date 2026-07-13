"""Shared bridge from WindowState's selected models to the core thermo seams.

The Txy and Phase-EQ modules both need the session's (antoine, gamma_fn,
phi_fn) for a chosen species order, plus an honest label and any "no binary
parameters for this pair" note. One helper so the honesty convention lives in
exactly one place.
"""


def session_models(ws, order):
    """(antoine, gamma_fn, phi_fn, label, note) for `order` from the session.

    Mirrors how the Simulation tab pulls the selected VLE/activity/EOS models.
    Raises ValueError (with a user-facing message) if the vapour-pressure or
    EOS parameters are missing — the caller shows it in the status label. When
    a non-ideal activity model is selected but the pair has no binary
    parameters, gamma_fn comes back None (ideal); `note` says so rather than
    letting the fallback pass silently.
    """
    tc = ws.thermodynamics_config
    antoine = tc.psat_params(order)          # raises if Psat coeffs missing
    gamma_fn = ws.build_gamma_fn(order)      # None => ideal (Raoult)
    phi_fn = ws.build_phi_fn(order)          # None => ideal gas
    label = f"{tc.vle_model} / {tc.activity_model} / {tc.eos_model}"
    note = ""
    if tc.activity_model != "Ideal" and gamma_fn is None:
        note = (f"No {tc.activity_model} binary parameters for this pair — "
                "showing ideal (Raoult) VLE.")
    return antoine, gamma_fn, phi_fn, label, note
