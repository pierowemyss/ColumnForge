"""Shared bridge from WindowState's selected models to the core thermo seams.

The Txy and Phase-EQ modules both need the session's (antoine, gamma_fn,
phi_fn) for a chosen species order, plus an honest label and any "no binary
parameters for this pair" note. One helper so the honesty convention lives in
exactly one place. `live_species` is here for the same reason: BVM and RBM both
have to trim the same dead components off the same window_state before either
one builds a problem.
"""

import numpy as np


def live_species(order, z, x_E, lk, hk, keep_names=()):
    """Drop species that enter the column nowhere. Returns the reduced
    (order, z, x_E, lk, hk) plus the names held at zero.

    A component absent from every feed is absent from every stage, but its pinch
    branches are not: the difference point carries bvec_k = 0 EXACTLY for it, so
    the pinch equation x_k (K_k - a) = bvec_k splits and the K_k = a branch
    solves at any x_k it likes. On matBVM_prediction_extract_col (six species,
    three of them in no stream) that invents an extractive "ternary saddle" at
    x_2ME = 0.53 -- k_gap 1e-10, so it sails through `BRANCH_TOL` -- and pins
    body vertices at x_EG = 0.084 and x_EC = 0.975. Those dead directions also
    hijack the geometry: the saddle sits on the x_2ME = x_EG = x_EC = 0 faces, so
    `bodies._to_edge` stalls at t = 0 on any arm pointing into one, and the S
    ray collapses onto the saddle. RBM's four bodies (paper p.100 rule 5) came
    out as one flat sliver in the LK/HK projection, and r_min read 1.13 against
    0.345 for the same case without them.

    BVM does not escape it either, by a second route: `Problem.trace_floor` seeds
    every product split at 1e-4, so a dead HEAVY component starts in the
    distillate and amplifies ~1/K per stage marching down -- the same mechanism
    the trace_floor docstring describes for a heavy entrainer.

    `keep_names` holds back components that are absent from the feed on purpose:
    a reaction PRODUCT is made on the tray, so BVM passes the reacting species.
    The keys are always kept. Species stay in `window_state` and in the .colx --
    this is the analysis's own component list, not an edit to the column.
    """
    keep = [
        i
        for i, n in enumerate(order)
        if z[i] > 0
        or (x_E is not None and x_E[i] > 0)
        or i in (lk, hk)
        or n in keep_names
    ]
    if len(keep) == len(order):
        return order, z, x_E, lk, hk, []
    dropped = [n for i, n in enumerate(order) if i not in keep]
    z = np.asarray(z, float)[keep]
    z = z / z.sum()
    if x_E is not None:
        x_E = np.asarray(x_E, float)[keep]
        x_E = x_E / x_E.sum()
    return [order[i] for i in keep], z, x_E, keep.index(lk), keep.index(hk), dropped


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


ENTRAINER_EB_TIP = (
    "Off (default): the entrainer is a saturated liquid AT THE TRAY, so constant\n"
    "molar overflow carries the rectifying vapour straight through it.\n\n"
    "On: the section energy balance sets the entrainer's thermal quality instead.\n"
    "A heavy entrainer arrives far hotter than the section it lands on (pure\n"
    "glycol boils at 197 C, the extractive section runs near 95 C) and flashes;\n"
    "on ipa/water/EG that is q = 0.69, which cuts the extractive vapour from 188\n"
    "to 165 and raises the entrainer level of every extractive pinch.\n\n"
    "Needs Cp, latent heat, Tb and Tc for every species; without them the run\n"
    "stays on CMO and says so. Entrainer feed temperature comes from the\n"
    "entrainer stream if it has one, else its own bubble point."
)


def attach_entrainer_energy_balance(ws, order, prob, provider, ent_stream=None):
    """Put the extractive section's energy balance on `prob`; return a note.

    Both extractive modules (BVM, RBM) build their sections through
    `bvm.sections.extractive_chain`, so both get the flash for free once
    `prob.q_E_fn` is set — see `bvm.sections.entrainer_q`. The entrainer feed
    temperature is the entrainer stream's if the user gave it one, otherwise the
    entrainer's own bubble point at column pressure (a heavy entrainer arriving
    straight from the recovery column's bottoms).

    Returns "" on success, or a note naming the reason it stayed on CMO — the
    caller shows it, because a silently-ignored checkbox is exactly what the
    repo's honesty rule forbids.
    """
    if prob.x_E is None:
        return "no entrainer composition — entrainer energy balance not applied."
    props = ws._enthalpy_props(order)
    if props is None:
        return ("no Cp / latent heat / Tb / Tc for every species — entrainer fed "
                "as saturated liquid at the tray (CMO).")
    from core.enthalpy import enthalpy_fns
    from side_features.bvm.sections import entrainer_q_fn
    hL, hV = enthalpy_fns(*props)
    T = getattr(ent_stream, "temperature", None)
    T_E = float(T) if T else provider.bubble(prob.x_E, prob.pressure)[1] + 273.15
    prob.q_E_fn = entrainer_q_fn(prob, provider, prob.pressure, hL, hV, T_E)
    return ""
