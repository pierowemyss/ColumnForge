"""JSON persistence for ColumnForge Cases (ADR-0001).

`.colx` is JSON: ``{schema_version, app, cases: [{name, state}]}``. ``state`` is
the *primitive projection* of ``WindowState.to_dict()`` — dataclasses flattened
to dicts, enums to their ``.value``, the tuple-keyed binary-interaction dicts to
``[i, j, value]`` triples. Language-neutral, versioned, inspectable — the format
a future native build reads. Pickle is gone.

Since v3 a ``state`` holds a *flowsheet*: ``columns`` (id -> the per-column dict
a v2 state used to be) plus ``connections`` between them, with species,
thermodynamics and display units left at the top because every column shares
them. ``_migrate_v2_flowsheet`` wraps an old flat state as ``columns["C1"]``, so
every file saved by an earlier build keeps loading.

Note ``cases`` is unrelated: a Case is an *alternative scenario* of the whole
flowsheet, not one of its units.

``to_dict()`` / ``load_from_dict()`` still produce/consume *live* objects;
``encode_state`` / ``decode_state`` are the primitive boundary used only at file
I/O. Decoders take fields by ``.get(...)`` with defaults, so a schema that gains
a field stays readable by older builds and vice-versa (the whole point of JSON +
schema_version over pickle).
"""

from __future__ import annotations

import json
from dataclasses import fields

from core.data_structures import SolverMode
from core.dof import Spec, SpecKind
from core.flowsheet import Connection

from .window_state import (
    ColumnState, Species, Stream, StreamType, CondenserConfig, CondenserType,
    ReboilerConfig, ReboilerType, ModuleConfig, ModuleType,
    ThermodynamicsConfig, ComponentThermoParams, BinaryInteractionParams,
)

SCHEMA_VERSION = 3   # v3: a flowsheet of columns + the connections between them
                     # v2: stages are 0-based from the top (0 = distillate)

_BINARY_DICTS = ("nrtl_aij", "nrtl_bij", "nrtl_cij", "uniquac_aij",
                 "uniquac_bij", "wilson_aij", "wilson_bij", "margules_aij")


def _only(cls, d: dict) -> dict:
    """Keep only keys that are fields of `cls` — tolerates extra/missing keys."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


# --- encode: live objects -> JSON primitives -------------------------------

def _enc_species(s: Species) -> dict:
    return {"name": s.name, "mw": s.mw, "liquid_density": s.liquid_density,
            "cp": s.cp, "tb": s.tb, "hvap_tb": s.hvap_tb,
            "unifac_groups": dict(s.unifac_groups)}


def _enc_thermo_params(p: ComponentThermoParams) -> dict:
    return {f.name: getattr(p, f.name) for f in fields(p)}   # all Optional[float]


def _enc_binary(b: BinaryInteractionParams) -> dict:
    return {name: [[i, j, v] for (i, j), v in getattr(b, name).items()]
            for name in _BINARY_DICTS}


def _enc_thermo(tc: ThermodynamicsConfig) -> dict:
    return {"vle_model": tc.vle_model, "activity_model": tc.activity_model,
            "eos_model": tc.eos_model, "energy_balance": tc.energy_balance,
            "component_params": {n: _enc_thermo_params(p)
                                 for n, p in tc.component_params.items()},
            "binary": _enc_binary(tc.binary)}


def _enc_stream(s: Stream) -> dict:
    return {"id": s.id, "stream_type": s.stream_type.value, "stage": s.stage,
            "temperature": s.temperature, "flow": s.flow,
            "composition": dict(s.composition),
            "user_specified": s.user_specified,
            "phase": s.phase}


def _enc_condenser(c: CondenserConfig) -> dict:
    return {"condenser_type": c.condenser_type.value,
            "subcooling_temp": c.subcooling_temp, "reflux_ratio": c.reflux_ratio,
            "vapor_distillate_flow": c.vapor_distillate_flow}


def _enc_reboiler(r: ReboilerConfig) -> dict:
    return {"reboiler_type": r.reboiler_type.value,
            "boilup_ratio": r.boilup_ratio, "bottoms_flow": r.bottoms_flow}


def _enc_module(m: ModuleConfig) -> dict:
    return {"module_type": m.module_type.value, "stage": m.stage,
            "num_stages": m.num_stages, "boilup_ratio": m.boilup_ratio,
            "reflux_ratio": m.reflux_ratio, "duty": m.duty,
            "return_stage": m.return_stage, "rate": m.rate}


def _enc_spec(sp: Spec) -> dict:
    return {"kind": sp.kind.value, "value": sp.value,
            "unit_ref": sp.unit_ref, "component": sp.component}


def _enc_connection(c: Connection) -> dict:
    return {"id": c.id, "src": c.src, "port": c.port, "dst": c.dst,
            "stage": c.stage, "split_fraction": c.split_fraction, "q": c.q}


def _enc_column(col: ColumnState) -> dict:
    """One Column's whole configuration — the shape a v2 `state` used to have."""
    return {
        "num_stages": col.num_stages, "pressure": col.pressure,
        "pressure_drop": col.pressure_drop,
        "stage_efficiency": col.stage_efficiency,
        "streams": {n: _enc_stream(s) for n, s in col.streams.items()},
        "condenser_config": _enc_condenser(col.condenser_config),
        "reboiler_config": _enc_reboiler(col.reboiler_config),
        "modules": {n: _enc_module(m) for n, m in col.modules.items()},
        "specs": [_enc_spec(sp) for sp in col.specs],
        "light_key_index": col.light_key_index,
        "heavy_key_index": col.heavy_key_index,
        "method": col.method,
        "node_pos": list(col.node_pos) if col.node_pos is not None else None,
    }


def encode_state(state: dict) -> dict:
    """Primitive projection of a live WindowState.to_dict() snapshot.

    v3 shape: the flowsheet-global things (species, thermodynamics, display
    units, the BVM/RBM panel knobs) sit at the top; every per-column field lives
    under `columns[id]`, which is exactly the dict a v2 `state` was.
    """
    e = dict(state)   # bvm_params / rbm_params / reactions / active_column_id /
                      # default_method are already primitives and pass through.
                      # The energy-balance flag rides on thermodynamics_config.
    e["species"] = {n: _enc_species(s) for n, s in state["species"].items()}
    e["thermodynamics_config"] = _enc_thermo(state["thermodynamics_config"])
    e["columns"] = {cid: _enc_column(c) for cid, c in state["columns"].items()}
    e["connections"] = [_enc_connection(c) for c in state.get("connections", [])]
    du = state.get("display_units")
    if du is not None:
        e["display_units"] = {f.name: getattr(du, f.name) for f in fields(du)}
    sm = state.get("solver_mode")
    e["solver_mode"] = getattr(sm, "value", sm)
    return e


# --- decode: JSON primitives -> live objects -------------------------------

def _dec_species(d: dict) -> Species:
    return Species(**_only(Species, {**d, "unifac_groups": dict(d.get("unifac_groups", {}))}))


def _dec_thermo_params(d: dict) -> ComponentThermoParams:
    return ComponentThermoParams(**_only(ComponentThermoParams, d))


def _dec_binary(d: dict) -> BinaryInteractionParams:
    b = BinaryInteractionParams()
    for name in _BINARY_DICTS:
        setattr(b, name, {(i, j): v for i, j, v in d.get(name, [])})
    return b


def _dec_thermo(d: dict) -> ThermodynamicsConfig:
    tc = ThermodynamicsConfig(
        vle_model=d.get("vle_model", "Antoine"),
        activity_model=d.get("activity_model", "Ideal"),
        eos_model=d.get("eos_model", "Ideal Gas"),
        energy_balance=bool(d.get("energy_balance", False)))
    tc.component_params = {n: _dec_thermo_params(p)
                           for n, p in d.get("component_params", {}).items()}
    tc.binary = _dec_binary(d.get("binary", {}))
    return tc


def _dec_stream(d: dict) -> Stream:
    return Stream(id=d["id"], stream_type=StreamType(d["stream_type"]),
                  stage=d.get("stage"), temperature=d.get("temperature"),
                  flow=d.get("flow"),
                  composition=dict(d.get("composition", {})),
                  user_specified=bool(d.get("user_specified", False)),
                  phase=d.get("phase", "liquid"))


def _dec_condenser(d: dict) -> CondenserConfig:
    return CondenserConfig(
        condenser_type=CondenserType(d.get("condenser_type", "Total")),
        subcooling_temp=d.get("subcooling_temp"),
        reflux_ratio=d.get("reflux_ratio"),
        vapor_distillate_flow=d.get("vapor_distillate_flow"))


def _dec_reboiler(d: dict) -> ReboilerConfig:
    return ReboilerConfig(
        reboiler_type=ReboilerType(d.get("reboiler_type", "Kettle")),
        boilup_ratio=d.get("boilup_ratio"), bottoms_flow=d.get("bottoms_flow"))


def _dec_module(d: dict) -> ModuleConfig:
    return ModuleConfig(
        module_type=ModuleType(d["module_type"]),
        stage=d.get("stage", 1), num_stages=d.get("num_stages", 1),
        boilup_ratio=d.get("boilup_ratio"), reflux_ratio=d.get("reflux_ratio"),
        duty=d.get("duty"),
        # older files may carry "associated_streams"; it was never consumed by a
        # solver and the panel no longer offers it, so it is dropped on load
        return_stage=d.get("return_stage"), rate=d.get("rate"))


def _dec_spec(d: dict) -> Spec:
    return Spec(kind=SpecKind(d["kind"]), value=d["value"],
                unit_ref=d.get("unit_ref", "column"),
                component=d.get("component", -1))


def _dec_connection(d: dict) -> Connection:
    return Connection(
        id=d["id"], src=d["src"], port=d["port"], dst=d["dst"],
        stage=int(d["stage"]),
        split_fraction=float(d.get("split_fraction", 1.0)),
        q=d.get("q"))


def _dec_column(d: dict) -> ColumnState:
    pos = d.get("node_pos")
    return ColumnState(
        num_stages=d.get("num_stages", 20),
        pressure=d.get("pressure", 1.0),
        pressure_drop=d.get("pressure_drop", 0.0),
        stage_efficiency=d.get("stage_efficiency", 1.0),
        streams={n: _dec_stream(x) for n, x in d.get("streams", {}).items()},
        condenser_config=_dec_condenser(d.get("condenser_config", {})),
        reboiler_config=_dec_reboiler(d.get("reboiler_config", {})),
        modules={n: _dec_module(x) for n, x in d.get("modules", {}).items()},
        specs=[_dec_spec(x) for x in d.get("specs", [])],
        light_key_index=d.get("light_key_index", 0),
        heavy_key_index=d.get("heavy_key_index"),
        method=d.get("method"),
        node_pos=tuple(pos) if pos is not None else None)


def decode_state(e: dict) -> dict:
    """Live-object state dict, ready for WindowState.load_from_dict()."""
    s = dict(e)
    s["species"] = {n: _dec_species(d) for n, d in e.get("species", {}).items()}
    s["thermodynamics_config"] = _dec_thermo(e.get("thermodynamics_config", {}))
    s["columns"] = {cid: _dec_column(d) for cid, d in e.get("columns", {}).items()}
    s["connections"] = [_dec_connection(d) for d in e.get("connections", [])]
    if e.get("display_units") is not None:
        from core.units import DisplayUnits
        s["display_units"] = DisplayUnits(**_only(DisplayUnits, e["display_units"]))
    if e.get("solver_mode") is not None:
        s["solver_mode"] = SolverMode(e["solver_mode"])
    return s


# --- file I/O --------------------------------------------------------------

def save_colx(path: str, state: dict, *, name: str = "Case 1") -> None:
    """Write a one-Case `.colx` (JSON). The cases list leaves room for more."""
    doc = {"schema_version": SCHEMA_VERSION, "app": "ColumnForge",
           "cases": [{"name": name, "state": encode_state(state)}]}
    if not path.endswith(".colx"):
        path += ".colx"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def load_colx(path: str) -> dict:
    """Return the first Case's live state dict (for WindowState.load_from_dict)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("This .colx isn't JSON — it may be from an older "
                         "pickle-based build, which this version can't read.")
    if not isinstance(doc, dict) or not doc.get("cases"):
        raise ValueError("Not a ColumnForge .colx file (no cases).")
    ver = doc.get("schema_version")
    state = doc["cases"][0]["state"]
    if ver == 1:
        state = _migrate_v1_stages(state)   # v1 counted stages 0 = bottoms
        ver = 2
    if ver == 2:
        state = _migrate_v2_flowsheet(state)   # v2 was one flat column
        ver = 3
    if ver != SCHEMA_VERSION:
        raise ValueError(f"Unsupported .colx schema_version {ver} "
                         f"(this build reads {SCHEMA_VERSION}).")
    return decode_state(state)


def _migrate_v1_stages(state: dict) -> dict:
    """v1 -> v2: flip every stored stage from 0=bottoms to 0=distillate
    (stage -> num_stages - 1 - stage)."""
    N = int(state.get("num_stages", 1))
    for coll in ("streams", "modules"):
        for d in state.get(coll, {}).values():
            if d.get("stage") is not None:
                d["stage"] = max(0, N - 1 - int(d["stage"]))
    return state


#: Per-column keys of a v2 state. Everything else there was already global.
_V2_COLUMN_KEYS = ("num_stages", "pressure", "pressure_drop", "stage_efficiency",
                   "streams", "condenser_config", "reboiler_config", "modules",
                   "specs", "light_key_index", "heavy_key_index")


def _migrate_v2_flowsheet(state: dict) -> dict:
    """v2 -> v3: a flat single column becomes a one-column flowsheet.

    Every shipped example and every file a user has saved is v2, so this is the
    load path for essentially all existing data — it has to be exactly
    lossless. The per-column keys move down into columns["C1"] unchanged;
    species/thermodynamics/display units stay at the top, where they now belong
    to the flowsheet rather than to the column.
    """
    out = {k: v for k, v in state.items() if k not in _V2_COLUMN_KEYS}
    column = {k: state[k] for k in _V2_COLUMN_KEYS if k in state}
    column.setdefault("method", None)
    column.setdefault("node_pos", None)
    out["columns"] = {"C1": column}
    out["active_column_id"] = "C1"
    out["connections"] = []                  # a v2 file had nothing to connect
    out.setdefault("default_method", "Inside-Out")
    return out
