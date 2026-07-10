"""Bundled pure-component / NRTL-binary database (roadmap Month 2).

Data lives in core/data/components.json — Antoine as log10(Psat[mmHg]),
T in degC (the app's bundled-fit convention), tb/tc in K, pc in bar,
hvap_tb in kJ/mol. Every record is gated by tests/test_component_db.py.

gui-free at module level; load_into() imports gui.state lazily so `core`
never depends on `gui` at import time.
"""
import json
import math
import os
from functools import lru_cache

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "components.json")


@lru_cache(maxsize=1)
def _db():
    with open(_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def all_components():
    """The raw component record list (do not mutate)."""
    return _db()["components"]


def all_binaries(section="nrtl_binaries"):
    """The raw binary record list for a model section (do not mutate)."""
    return _db().get(section, [])


def _names(rec):
    return [rec["name"]] + list(rec.get("aliases", ()))


def get(name):
    """Exact (case-insensitive) lookup by name, alias or CAS. None if absent."""
    q = name.strip().lower()
    for rec in all_components():
        if q == rec.get("cas") or any(q == n.lower() for n in _names(rec)):
            return rec
    return None


def search(query, limit=20):
    """Ranked substring search over name/aliases/CAS/formula.

    Rank: exact > prefix > substring, name matches before alias/CAS/formula.
    Empty query returns everything (alphabetical), for a browse-all list.
    """
    q = query.strip().lower()
    if not q:
        return sorted(all_components(), key=lambda r: r["name"])[:limit]
    scored = []
    for rec in all_components():
        best = None
        for field_rank, text in ([(0, rec["name"])]
                                 + [(1, a) for a in rec.get("aliases", ())]
                                 + [(1, rec.get("cas", "")),
                                    (1, rec.get("formula", ""))]):
            t = text.lower()
            if q == t:
                m = 0
            elif t.startswith(q):
                m = 1
            elif q in t:
                m = 2
            else:
                continue
            cand = (m, field_rank)
            if best is None or cand < best:
                best = cand
        if best is not None:
            scored.append((best, rec["name"], rec))
    scored.sort()
    return [rec for _, _, rec in scored[:limit]]


def antoine_trange(rec):
    """(tmin, tmax) in degC for the record's Antoine fit.

    Explicit range from the source table when present; otherwise estimated as
    [Tsat(10 mmHg), Tsat(1520 mmHg)] from the fit itself — roughly the span
    such classic fits were regressed over. Third element says which.
    Returns (tmin, tmax, estimated: bool).
    """
    if rec.get("antoine_trange"):
        lo, hi = rec["antoine_trange"]
        return float(lo), float(hi), False
    a, b, c = rec["antoine"]

    def tsat(p_mmhg):
        return b / (a - math.log10(p_mmhg)) - c

    return tsat(10.0), tsat(1520.0), True


def load_into(ws, name):
    """Fill a WindowState with one DB component: Species + ComponentThermoParams
    + directional NRTL params for every DB pair whose partner already exists.

    Returns {"record": rec, "nrtl_pairs": [(i, j), ...], "missing_pairs":
    [(i, j), ...]} — missing_pairs are existing-species pairs the DB has no
    NRTL data for (the GUI flags them). Raises KeyError if name is unknown.
    """
    rec = get(name)
    if rec is None:
        raise KeyError(f"component not in database: {name!r}")
    # ponytail: core->gui import kept local; move Species into core/ if the
    # layering ever tightens.
    from gui.state.window_state import Species

    ws.add_species(Species(name=rec["name"], mw=rec.get("mw"),
                           liquid_density=rec.get("liquid_density"),
                           cp=rec.get("cp_liq"), tb=rec.get("tb"),
                           hvap_tb=rec.get("hvap_tb")))
    p = ws.thermodynamics_config.get_component_params(rec["name"])
    p.tc, p.pc, p.omega = rec.get("tc"), rec.get("pc"), rec.get("omega")
    p.antoine_a, p.antoine_b, p.antoine_c = rec["antoine"]
    p.antoine_tmin, p.antoine_tmax, _ = antoine_trange(rec)
    if rec.get("uniquac_rq"):
        p.uniquac_r, p.uniquac_q = rec["uniquac_rq"]

    binary = ws.thermodynamics_config.binary
    others = [s for s in ws.species if s != rec["name"]]
    filled, missing = [], []
    for other in others:
        orec = get(other)
        # Wilson/UNIQUAC pairs ride along when the DB has them (missing_pairs
        # stays NRTL-based: NRTL is the default model the GUI flags for)
        for section, prefix in (("wilson_binaries", "wilson"),
                                ("uniquac_binaries", "uniquac")):
            pw = _find_binary(rec["name"], other, section)
            if pw is None:
                continue
            bw, wflip = pw
            wi, wj = (rec["name"], other) if not wflip else (other, rec["name"])
            getattr(binary, f"{prefix}_aij")[(wi, wj)] = bw["aij"]
            getattr(binary, f"{prefix}_aij")[(wj, wi)] = bw["aji"]
            getattr(binary, f"{prefix}_bij")[(wi, wj)] = bw["bij"]
            getattr(binary, f"{prefix}_bij")[(wj, wi)] = bw["bji"]
        pair = _find_binary(rec["name"], other)
        if pair is None:
            # only flag pairs where both ends are DB components — hand-entered
            # species can't be expected to have curated parameters
            if orec is not None:
                missing.append((rec["name"], other))
            continue
        b, flip = pair
        i, j = (rec["name"], other) if not flip else (other, rec["name"])
        # b uses canonical DB names; key the ws dicts by ws species names
        binary.nrtl_aij[(i, j)] = b["aij"]
        binary.nrtl_aij[(j, i)] = b["aji"]
        binary.nrtl_bij[(i, j)] = b["bij"]
        binary.nrtl_bij[(j, i)] = b["bji"]
        binary.nrtl_cij[(i, j)] = b["cij"]
        binary.nrtl_cij[(j, i)] = b["cij"]
        filled.append((i, j))
    return {"record": rec, "nrtl_pairs": filled, "missing_pairs": missing}


def _find_binary(name_a, name_b, section="nrtl_binaries"):
    """DB binary record for the unordered pair, resolving aliases.

    Returns (record, flipped) where flipped means record's i corresponds to
    name_b; None if the pair isn't in the DB.
    """
    ra, rb = get(name_a), get(name_b)
    if ra is None or rb is None:
        return None
    for b in all_binaries(section):
        if b["i"] == ra["name"] and b["j"] == rb["name"]:
            return b, False
        if b["i"] == rb["name"] and b["j"] == ra["name"]:
            return b, True
    return None


def _demo():
    assert get("Benzene")["cas"] == "71-43-2"
    assert get("71-43-2")["name"] == "benzene"
    assert get("methylbenzene")["name"] == "toluene"
    assert get("no-such-thing") is None
    assert search("xyl")[0]["name"] in ("m-xylene", "o-xylene", "p-xylene")
    assert search("benzene")[0]["name"] == "benzene"          # exact beats substring
    lo, hi, est = antoine_trange(get("benzene"))
    assert (lo, hi, est) == (8.0, 103.0, False)
    lo, hi, est = antoine_trange(get("n-hexane"))
    assert est and lo < 68.7 - 273.15 + 342 and hi > 68.7      # spans Tb ~68.7 C
    assert _find_binary("water", "ethanol")[1] is True         # flipped lookup
    print(f"component_db OK: {len(all_components())} components, "
          f"{len(all_binaries())} binaries")


if __name__ == "__main__":
    _demo()
