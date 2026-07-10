"""JSON persistence for FreeColumn Cases (ADR-0001).

`.colx` is JSON: ``{schema_version, app, cases: [{name, state}]}``. ``state`` is
the *primitive projection* of ``WindowState.to_dict()`` — dataclasses flattened
to dicts, enums to their ``.value``, the tuple-keyed binary-interaction dicts to
``[i, j, value]`` triples. Language-neutral, versioned, inspectable — the format
a future native build reads. Pickle is gone.

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

from .window_state import (
    Species, Stream, StreamType, CondenserConfig, CondenserType,
    ReboilerConfig, ReboilerType, ModuleConfig, ModuleType,
    ThermodynamicsConfig, ComponentThermoParams, BinaryInteractionParams,
)

SCHEMA_VERSION = 2   # v2: stages are 0-based from the top (0 = distillate)

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
            "associated_streams": {k: list(v)
                                   for k, v in m.associated_streams.items()}}


def _enc_spec(sp: Spec) -> dict:
    return {"kind": sp.kind.value, "value": sp.value,
            "unit_ref": sp.unit_ref, "component": sp.component}


def encode_state(state: dict) -> dict:
    """Primitive projection of a live WindowState.to_dict() snapshot."""
    e = dict(state)   # scalars (num_stages, feed_stage, pressure, pressure_drop,
                      # bvm_params, energy_balance, light/heavy_key_index) pass through
    e["species"] = {n: _enc_species(s) for n, s in state["species"].items()}
    e["streams"] = {n: _enc_stream(s) for n, s in state["streams"].items()}
    e["condenser_config"] = _enc_condenser(state["condenser_config"])
    e["reboiler_config"] = _enc_reboiler(state["reboiler_config"])
    e["thermodynamics_config"] = _enc_thermo(state["thermodynamics_config"])
    e["modules"] = {n: _enc_module(m) for n, m in state["modules"].items()}
    e["specs"] = [_enc_spec(sp) for sp in state["specs"]]
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
        associated_streams={k: tuple(v)
                            for k, v in d.get("associated_streams", {}).items()})


def _dec_spec(d: dict) -> Spec:
    return Spec(kind=SpecKind(d["kind"]), value=d["value"],
                unit_ref=d.get("unit_ref", "column"),
                component=d.get("component", -1))


def decode_state(e: dict) -> dict:
    """Live-object state dict, ready for WindowState.load_from_dict()."""
    s = dict(e)
    s["species"] = {n: _dec_species(d) for n, d in e.get("species", {}).items()}
    s["streams"] = {n: _dec_stream(d) for n, d in e.get("streams", {}).items()}
    s["condenser_config"] = _dec_condenser(e.get("condenser_config", {}))
    s["reboiler_config"] = _dec_reboiler(e.get("reboiler_config", {}))
    s["thermodynamics_config"] = _dec_thermo(e.get("thermodynamics_config", {}))
    s["modules"] = {n: _dec_module(d) for n, d in e.get("modules", {}).items()}
    s["specs"] = [_dec_spec(d) for d in e.get("specs", [])]
    if e.get("display_units") is not None:
        from core.units import DisplayUnits
        s["display_units"] = DisplayUnits(**_only(DisplayUnits, e["display_units"]))
    if e.get("solver_mode") is not None:
        s["solver_mode"] = SolverMode(e["solver_mode"])
    return s


# --- file I/O --------------------------------------------------------------

def save_colx(path: str, state: dict, *, name: str = "Case 1") -> None:
    """Write a one-Case `.colx` (JSON). The cases list leaves room for more."""
    doc = {"schema_version": SCHEMA_VERSION, "app": "FreeColumn",
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
        raise ValueError("Not a FreeColumn .colx file (no cases).")
    ver = doc.get("schema_version")
    state = doc["cases"][0]["state"]
    if ver == 1:
        state = _migrate_v1_stages(state)   # v1 counted stages 0 = bottoms
        ver = 2
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
