from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.data_structures import SolverMode  # canonical, single definition
from core.dof import DoFAnalyzer, Spec, SpecKind
from core.material_balance import overall_balance


class StreamType(Enum):
    FEED = "Feed"
    DISTILLATE = "Distillate"
    BOTTOMS = "Bottoms"
    SIDESTREAM = "Sidestream"


class CondenserType(Enum):
    TOTAL = "Total"
    PARTIAL = "Partial"
    NONE = "None"


class ReboilerType(Enum):
    KETTLE = "Kettle"
    THERMOSIPHON = "Thermosiphon"
    NONE = "None"


class ModuleType(Enum):
    INTERREBOILER = "Interreboiler"
    SIDE_STRIPPER = "Side Stripper"
    SIDE_RECTIFIER = "Side Rectifier"


@dataclass
class Species:
    """Physical properties intrinsic to each chemical component."""
    name: str
    mw: Optional[float] = None           # Molecular weight (g/mol)
    liquid_density: Optional[float] = None  # Liquid density at ref conditions (kg/m³)
    cp: Optional[float] = None            # Heat capacity (J/mol·K)
    unifac_groups: Dict[str, int] = field(default_factory=dict)  # group -> count


@dataclass
class BinaryInteractionParams:
    """Binary interaction parameters - stored independently of model selection.
    
    Keys are (component_i, component_j) tuples - explicit in both directions.
    """
    nrtl_aij: Dict[Tuple[str, str], float] = field(default_factory=dict)
    nrtl_bij: Dict[Tuple[str, str], float] = field(default_factory=dict)
    nrtl_cij: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    uniquac_aij: Dict[Tuple[str, str], float] = field(default_factory=dict)
    uniquac_bij: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    wilson_aij: Dict[Tuple[str, str], float] = field(default_factory=dict)
    wilson_bij: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def remove_component(self, component_name: str):
        """Remove all entries involving a component."""
        keys_to_remove = []
        for key in self.nrtl_aij:
            if component_name in key:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.nrtl_aij.pop(key, None)
            self.nrtl_bij.pop(key, None)
            self.nrtl_cij.pop(key, None)
            self.uniquac_aij.pop(key, None)
            self.uniquac_bij.pop(key, None)
            self.wilson_aij.pop(key, None)
            self.wilson_bij.pop(key, None)


@dataclass
class ComponentThermoParams:
    """Thermodynamic parameters for a single component."""
    tc: Optional[float] = None  # Critical temperature (K)
    pc: Optional[float] = None  # Critical pressure (bar)
    omega: Optional[float] = None  # Acentric factor
    antoine_a: Optional[float] = None
    antoine_b: Optional[float] = None
    antoine_c: Optional[float] = None
    wagner_a: Optional[float] = None
    wagner_b: Optional[float] = None
    wagner_c: Optional[float] = None
    wagner_d: Optional[float] = None
    # PLXANT (Aspen extended Antoine), C1..C7. T in K:
    #   ln(Psat) = C1 + C2/(C3+T) + C4*T + C5*ln(T) + C6*T**C7
    plxant_c1: Optional[float] = None
    plxant_c2: Optional[float] = None
    plxant_c3: Optional[float] = None
    plxant_c4: Optional[float] = None
    plxant_c5: Optional[float] = None
    plxant_c6: Optional[float] = None
    plxant_c7: Optional[float] = None


@dataclass
class ThermodynamicsConfig:
    """Thermodynamic model configuration - separate from parameter storage.
    
    Allows user to switch between models without losing parameter data.
    """
    vle_model: str = "Antoine"        # Vapor pressure model
    activity_model: str = "Ideal"     # Activity coefficient model
    eos_model: str = "Ideal Gas"      # Equation of state
    
    # Per-component thermodynamic parameters (component name -> params)
    component_params: Dict[str, ComponentThermoParams] = field(default_factory=dict)
    
    # Binary interaction parameters
    binary: BinaryInteractionParams = field(default_factory=BinaryInteractionParams)
    
    def get_component_params(self, component_name: str) -> ComponentThermoParams:
        """Get or create thermo params for a component."""
        if component_name not in self.component_params:
            self.component_params[component_name] = ComponentThermoParams()
        return self.component_params[component_name]
    
    def remove_component(self, component_name: str):
        """Remove component params when a component is deleted."""
        self.component_params.pop(component_name, None)
        self.binary.remove_component(component_name)
    
    # Coefficient field names per vapour-pressure model, in matrix-column order.
    _PSAT_KEYS = {
        "PLXANT": ("plxant_c1", "plxant_c2", "plxant_c3", "plxant_c4",
                   "plxant_c5", "plxant_c6", "plxant_c7"),
        "Antoine": ("antoine_a", "antoine_b", "antoine_c"),
    }

    def psat_params(self, order):
        """Vapour-pressure coefficient matrix for `order`, shaped per vle_model.

        (N,7) for PLXANT, else (N,3) Antoine. core.thermodynamics.antoine_psat
        dispatches on the column count, so callers just pass this straight through.
        Raises ValueError naming the first component missing required coefficients.
        """
        import numpy as np
        keys = self._PSAT_KEYS.get(self.vle_model, self._PSAT_KEYS["Antoine"])
        rows = []
        for nm in order:
            p = self.component_params.get(nm)
            vals = [getattr(p, k, None) for k in keys] if p else None
            if not vals or any(v is None for v in vals):
                raise ValueError(f"Missing {self.vle_model} coefficients for '{nm}' "
                                 "(Initialization → Thermodynamics).")
            rows.append(vals)
        return np.array(rows, float)

    def get_binary_param_dict(self, model: str, param: str) -> Dict[Tuple[str, str], float]:
        """Get binary params for a specific model and parameter."""
        model_lower = model.lower()
        param_lower = param.lower()
        
        if model_lower == "nrtl":
            return getattr(self.binary, f"nrtl_{param_lower}", {})
        elif model_lower == "uniquac":
            return getattr(self.binary, f"uniquac_{param_lower}", {})
        elif model_lower == "wilson":
            return getattr(self.binary, f"wilson_{param_lower}", {})
        return {}


@dataclass
class Stream:
    """Represents a stream in the column."""
    id: str
    stream_type: StreamType
    stage: Optional[int] = None
    temperature: Optional[float] = None  # SI units (K)
    pressure: Optional[float] = None  # SI units (bar)
    flow: Optional[float] = None  # SI units (kmol/h)
    composition: Dict[str, float] = field(default_factory=dict)  # species -> mole fraction

    def get_species_list(self) -> List[str]:
        return list(self.composition.keys())


@dataclass
class CondenserConfig:
    """Configuration for condenser."""
    condenser_type: CondenserType = CondenserType.TOTAL
    subcooling_temp: Optional[float] = None  # K
    reflux_ratio: Optional[float] = None  # L/D
    vapor_distillate_flow: Optional[float] = None  # kmol/h

    def get_specs(self) -> List[str]:
        specs = []
        if self.condenser_type == CondenserType.TOTAL:
            if self.subcooling_temp is not None:
                specs.append(f"Subcooling: {self.subcooling_temp:.2f} K")
        elif self.condenser_type == CondenserType.PARTIAL:
            if self.reflux_ratio is not None:
                specs.append(f"Reflux ratio: {self.reflux_ratio:.4f}")
            if self.vapor_distillate_flow is not None:
                specs.append(f"Vapor distillate: {self.vapor_distillate_flow:.2f} kmol/h")
        return specs


@dataclass
class ReboilerConfig:
    """Configuration for reboiler."""
    reboiler_type: ReboilerType = ReboilerType.KETTLE
    boilup_ratio: Optional[float] = None  # V/B
    bottoms_flow: Optional[float] = None  # kmol/h

    def get_specs(self) -> List[str]:
        specs = []
        if self.boilup_ratio is not None:
            specs.append(f"Boilup ratio: {self.boilup_ratio:.4f}")
        if self.bottoms_flow is not None:
            specs.append(f"Bottoms flow: {self.bottoms_flow:.2f} kmol/h")
        return specs


@dataclass
class ModuleConfig:
    """Configuration for side modules."""
    module_type: ModuleType
    stage: int = 1
    num_stages: int = 1
    boilup_ratio: Optional[float] = None
    reflux_ratio: Optional[float] = None
    associated_streams: Dict[str, tuple] = field(default_factory=dict)  # name -> (out, to_tray)


class WindowState:
    """Manages the overall window state including column configuration."""

    def __init__(self):
        self.current_tab = 0
        self.solver_mode = SolverMode.HYSIM
        self.is_modified = False
        self.file_path = None
        self._tab_states = {}
        self.column_config = None

        # Column data
        self.num_stages = 20
        self.feed_stage = 10
        self.pressure = 1.0  # bar
        self.pressure_drop = 0.0  # bar/stage

        # Species management
        self.species: Dict[str, Species] = {}

        # Streams management
        self.streams: Dict[str, Stream] = {}

        # Configuration
        self.condenser_config = CondenserConfig()
        self.reboiler_config = ReboilerConfig()
        self.thermodynamics_config = ThermodynamicsConfig()

        # Modules
        self.modules: Dict[str, ModuleConfig] = {}

        # BVM solver knobs live in the Modules/BVM widget; mirrored here so they
        # round-trip through to_dict()/load_from_dict() (.colx). {} = use defaults.
        self.bvm_params: dict = {}

        # Spec/DoF: structured extra specs (e.g. key-recovery) + CMO flag.
        # Condenser/reboiler/side-draw specs are derived from config in
        # collect_specs(); self.specs holds anything not on those panels yet.
        self.specs: List[Spec] = []
        self.energy_balance = False          # CMO; True enables duty specs later
        self.light_key_index = 0             # 0-based light-key index
        self.heavy_key_index = None          # 0-based; None => defaults to lk+1
        self.results = None                  # last solver profile (Results tab reads this)

        # Add default streams
        self._add_default_streams()

    def create_new_column(self):
        """Reset to a new empty column configuration."""
        self.column_config = None
        self.is_modified = False
        self.file_path = None
        self.num_stages = 20
        self.feed_stage = 10
        self.pressure = 1.0
        self.pressure_drop = 0.0
        self.species = {}
        self.streams = {}
        self.condenser_config = CondenserConfig()
        self.reboiler_config = ReboilerConfig()
        self.thermodynamics_config = ThermodynamicsConfig()
        self.modules = {}
        self.bvm_params = {}
        self.specs = []
        self.energy_balance = False
        self.light_key_index = 0
        self.heavy_key_index = None
        self.results = None

        # Add default streams
        self._add_default_streams()

    def _add_default_streams(self):
        """Add the standard Feed, Distillate, and Bottoms streams."""
        feed = Stream(id="Feed", stream_type=StreamType.FEED, stage=10)
        distillate = Stream(id="Distillate", stream_type=StreamType.DISTILLATE, stage=1)
        bottoms = Stream(id="Bottoms", stream_type=StreamType.BOTTOMS, stage=20)
        
        self.add_stream(feed)
        self.add_stream(distillate)
        self.add_stream(bottoms)
        self.is_modified = False # Reset modified flag after defaults

    def load_column_config(self, config):
        """Load a column configuration."""
        self.column_config = config
        self.is_modified = False

    # Persisted column state. Stored as live objects (dataclasses + enums) — the
    # .colx is pickled, so no field-by-field (de)serialization is needed.
    # ponytail: BVM knobs (r, q, FR_LK…) live in the Modules/BVM widget, not here,
    # so they aren't persisted yet — re-enter them after load, or lift them into
    # window_state when the BVM panel should round-trip too.
    _PERSIST = ("num_stages", "feed_stage", "pressure", "pressure_drop",
                "species", "streams", "condenser_config", "reboiler_config",
                "thermodynamics_config", "modules", "bvm_params", "specs",
                "energy_balance", "light_key_index", "heavy_key_index", "solver_mode")

    def to_dict(self) -> dict:
        """Snapshot the persistable column state (goes into the .colx)."""
        return {k: getattr(self, k) for k in self._PERSIST}

    def load_from_dict(self, state: dict):
        """Restore a snapshot produced by to_dict()."""
        for k in self._PERSIST:
            if k in state:
                setattr(self, k, state[k])
        self.results = None
        self.is_modified = False

    def build_gamma_fn(self, order):
        """Activity-coefficient closure for the thermo layer, or None (ideal).

        Reads the active activity model + binary parameters and returns a
        gamma_fn(x, T) for core.thermodynamics. Only NRTL is implemented; any
        other selection (or NRTL with no parameters entered) falls back to ideal.
        ponytail: alpha defaults to the standard 0.3; add a per-pair field if a
        system needs a different non-randomness factor.
        """
        tc = self.thermodynamics_config
        if tc.activity_model != "NRTL":
            return None
        n = len(order)
        aij, bij = tc.binary.nrtl_aij, tc.binary.nrtl_bij
        a = [[0.0] * n for _ in range(n)]
        b = [[0.0] * n for _ in range(n)]
        alpha = [[0.0] * n for _ in range(n)]
        any_param = False
        for i, ci in enumerate(order):
            for j, cj in enumerate(order):
                if i == j:
                    continue
                alpha[i][j] = 0.3
                if (ci, cj) in aij:
                    a[i][j] = aij[(ci, cj)]; any_param = True
                if (ci, cj) in bij:
                    b[i][j] = bij[(ci, cj)]; any_param = True
        if not any_param:
            return None
        from core.thermodynamics import nrtl_gamma_fn
        return nrtl_gamma_fn(a, b, alpha)

    def get_tab_state(self, tab_name):
        return self._tab_states.get(tab_name, None)

    def set_tab_state(self, tab_name, state):
        self._tab_states[tab_name] = state
        self.is_modified = True

    def mark_modified(self):
        self.is_modified = True

    def reset_modified(self):
        self.is_modified = False

    def set_file_path(self, path):
        self.file_path = path
        self.is_modified = False

    def add_species(self, species: Species):
        """Add a species to the column."""
        self.species[species.name] = species
        self.thermodynamics_config.get_component_params(species.name)
        self.is_modified = True

    def remove_species(self, name: str):
        """Remove a species from the column."""
        if name in self.species:
            del self.species[name]
            self.thermodynamics_config.remove_component(name)
            self.is_modified = True

    def rename_species(self, old_name: str, new_name: str) -> bool:
        """Rename a species throughout all data structures.
        
        Returns True if successful, False if new_name already exists.
        """
        if old_name == new_name:
            return True
        if new_name in self.species:
            return False
        
        if old_name not in self.species:
            return False
        
        species_obj = self.species.pop(old_name)
        species_obj.name = new_name
        self.species[new_name] = species_obj
        
        if old_name in self.thermodynamics_config.component_params:
            params = self.thermodynamics_config.component_params.pop(old_name)
            self.thermodynamics_config.component_params[new_name] = params
        
        self._rename_binary_keys(old_name, new_name)
        
        for stream in self.streams.values():
            if old_name in stream.composition:
                stream.composition[new_name] = stream.composition.pop(old_name)
        
        self.is_modified = True
        return True

    def _rename_binary_keys(self, old_name: str, new_name: str):
        """Rename component keys in binary interaction parameters."""
        binary = self.thermodynamics_config.binary
        for attr_name in dir(binary):
            if attr_name.startswith('_') or not isinstance(getattr(binary, attr_name, None), dict):
                continue
            d = getattr(binary, attr_name)
            keys_to_rename = [(k, v) for k, v in d.items() if old_name in k]
            for key, value in keys_to_rename:
                new_key = tuple(new_name if x == old_name else x for x in key)
                d.pop(key, None)
                d[new_key] = value

    def add_stream(self, stream: Stream):
        """Add a stream to the column. (Feeds/side-draws are counted from
        self.streams by the DoF analyzer, so no separate registration.)"""
        self.streams[stream.id] = stream
        self.is_modified = True

    def remove_stream(self, stream_id: str):
        """Remove a stream from the column."""
        if stream_id in self.streams:
            del self.streams[stream_id]
            self.is_modified = True

    def add_module(self, module_id: str, module: ModuleConfig):
        """Add a side module to the column."""
        self.modules[module_id] = module
        self.is_modified = True

    def remove_module(self, module_id: str):
        """Remove a side module from the column."""
        if module_id in self.modules:
            del self.modules[module_id]
            self.is_modified = True

    # --- Unified DoF + auto material balance (single source of truth) ---

    def _ordered_species(self) -> List[str]:
        return list(self.species.keys())

    def get_species_names(self) -> List[str]:
        """Species in insertion order. Used by the BVM/BP solvers' input gather."""
        return list(self.species.keys())

    def upsert_operating_spec(self, kind, value, component: int = -1):
        """Set one operating spec by kind in self.specs (the single source).

        Operating specs are keyed by kind — a column can't have two reflux
        ratios — so this replaces any existing spec of the same kind. A falsy
        value drops it. unit_ref is the kind name, so collect_specs' dedup keeps
        exactly one entry per kind no matter which panel wrote it.
        """
        from core.dof import Spec, SpecKind, OPERATING_KINDS
        if kind not in OPERATING_KINDS:
            raise ValueError(f"{kind} is not an operating spec")
        self.specs = [s for s in self.specs if s.kind != kind]
        if value:
            self.specs.append(Spec(kind, float(value), kind.name, component))
        self.mark_modified()

    def get_operating_spec(self, kind):
        """Current value for an operating-spec kind, or None."""
        return next((s for s in self.specs if s.kind == kind), None)

    def collect_specs(self) -> List[Spec]:
        """Structured operating specs for DoF + the operating-point resolver.

        self.specs is the single source — written by the Operating-specs slots
        and by the condenser/reboiler panels alike (both go through
        upsert_operating_spec, keyed by kind, so nothing double-counts). For
        configs saved before that existed, fall back to the reflux/boilup/rate
        fields on condenser_config/reboiler_config. Side-draw rates always count.
        """
        from core.dof import OPERATING_KINDS
        specs: List[Spec] = list(self.specs)
        has_slots = any(s.kind in OPERATING_KINDS for s in self.specs)
        if not has_slots:        # legacy fallback: pull from condenser/reboiler
            cc = self.condenser_config
            if cc.condenser_type != CondenserType.NONE:
                if cc.reflux_ratio:
                    specs.append(Spec(SpecKind.REFLUX_RATIO, cc.reflux_ratio, "condenser"))
                if cc.vapor_distillate_flow:
                    specs.append(Spec(SpecKind.DISTILLATE_RATE, cc.vapor_distillate_flow, "condenser"))
            rc = self.reboiler_config
            if rc.reboiler_type != ReboilerType.NONE:
                if rc.boilup_ratio:
                    specs.append(Spec(SpecKind.BOILUP_RATIO, rc.boilup_ratio, "reboiler"))
                if rc.bottoms_flow:
                    specs.append(Spec(SpecKind.BOTTOMS_RATE, rc.bottoms_flow, "reboiler"))
        for s in self.streams.values():
            if s.stream_type == StreamType.SIDESTREAM and s.flow:
                specs.append(Spec(SpecKind.SIDEDRAW_RATE, s.flow, s.id))
        return specs

    def build_dof_analyzer(self) -> DoFAnalyzer:
        cc, rc = self.condenser_config, self.reboiler_config
        n_side = sum(1 for s in self.streams.values()
                     if s.stream_type == StreamType.SIDESTREAM)
        return DoFAnalyzer(
            n_components=len(self.species),
            condenser=cc.condenser_type != CondenserType.NONE,
            reboiler=rc.reboiler_type != ReboilerType.NONE,
            partial_condenser=cc.condenser_type == CondenserType.PARTIAL,
            n_side_draws=n_side,
            # ponytail: 1 required spec per module; refine when module DoF matters
            module_spec_counts=[1] * len(self.modules),
            energy_balance=self.energy_balance,
        )

    def analyze_dof(self):
        """DoFResult for the current config + collected specs."""
        return self.build_dof_analyzer().analyze(self.collect_specs())

    def get_specification_status(self) -> tuple:
        """(icon, message, can_run) from the unified DoF analyzer."""
        r = self.analyze_dof()
        return r.icon, r.message, r.can_run

    def auto_balance(self):
        """Run the overall component balance and write D/B flows + comps.

        Fires only when the column is fully specified AND a key-recovery spec is
        present (reflux/boilup alone don't pin the component split). Returns
        (xD, D, xB, B) on success, else None.
        ponytail: recovery path only (what BVM uses); add direct-xD/xB when a
        key-spec panel exists to supply it. NK_spec defaults to 0 — exact for the
        2-key case, the upgrade is a non-key distribution spec.
        """
        if not self.analyze_dof().can_run:
            return None
        order = self._ordered_species()
        if len(order) < 2:
            return None
        lk_recovery = next((sp.value for sp in self.collect_specs()
                            if sp.kind == SpecKind.LK_RECOVERY), None)
        if lk_recovery is None:
            return None
        feeds = []
        for s in self.streams.values():
            if s.stream_type == StreamType.FEED and s.flow and s.composition:
                z = [s.composition.get(name, 0.0) for name in order]
                if sum(z) > 0:
                    feeds.append((s.flow, z))
        if not feeds:
            return None
        xD, D, xB, B = overall_balance(
            feeds, lk=self.light_key_index, spec_mode="recovery",
            FR_LK=lk_recovery, NK_spec=0.0)
        self._write_product(StreamType.DISTILLATE, D, xD, order)
        self._write_product(StreamType.BOTTOMS, B, xB, order)
        return xD, D, xB, B

    def _write_product(self, stream_type, flow, x, order):
        s = next((st for st in self.streams.values()
                  if st.stream_type == stream_type), None)
        if s is None:
            return
        s.flow = float(flow)
        s.composition = {name: float(xi) for name, xi in zip(order, x)}

    def clear(self):
        self.current_tab = 0
        self.solver_mode = SolverMode.HYSIM
        self.is_modified = False
        self.file_path = None
        self._tab_states = {}
        self.column_config = None
        self.num_stages = 20
        self.feed_stage = 10
        self.pressure = 1.0
        self.pressure_drop = 0.0
        self.species = {}
        self.streams = {}
        self.condenser_config = CondenserConfig()
        self.reboiler_config = ReboilerConfig()
        self.thermodynamics_config = ThermodynamicsConfig()
        self.modules = {}
        self.bvm_params = {}
        self.specs = []
        self.energy_balance = False
        self.light_key_index = 0
        self.heavy_key_index = None
        self.results = None


def _demo():
    """Self-check: unified DoF status + auto-balance end to end (no Qt)."""
    ws = WindowState()                      # default Feed/Distillate/Bottoms streams
    ws.add_species(Species(name="A"))
    ws.add_species(Species(name="B"))

    # Simple column: total condenser + kettle reboiler -> required == 2.
    assert ws.build_dof_analyzer().required_specs() == 2

    # One operating spec + one key-recovery spec -> exact, and solvable.
    ws.reboiler_config.boilup_ratio = 1.5
    ws.specs.append(Spec(SpecKind.LK_RECOVERY, 0.98, "column"))
    icon, msg, can_run = ws.get_specification_status()
    assert can_run, (icon, msg)

    feed = ws.streams["Feed"]
    feed.flow = 100.0
    feed.composition = {"A": 0.6, "B": 0.4}

    result = ws.auto_balance()
    assert result is not None
    xD, D, xB, B = result
    # Overall balance closes: F == D + B and component balance holds.
    assert abs(D + B - 100.0) < 1e-6, (D, B)
    for i, name in enumerate(["A", "B"]):
        assert abs(100.0 * feed.composition[name]
                   - (D * xD[i] + B * xB[i])) < 1e-6, name
    # Products were written back into the streams.
    assert ws.streams["Distillate"].flow == D
    assert abs(sum(ws.streams["Bottoms"].composition.values()) - 1.0) < 1e-6

    # Under-spec when both specs are removed -> no balance.
    ws.specs.clear()
    ws.reboiler_config.boilup_ratio = None
    assert not ws.get_specification_status()[2]
    assert ws.auto_balance() is None
    print("window_state self-check OK")


if __name__ == "__main__":
    _demo()
