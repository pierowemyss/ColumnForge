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
    PUMPAROUND = "Pumparound"


@dataclass
class Species:
    """Physical properties intrinsic to each chemical component."""
    name: str
    mw: Optional[float] = None           # Molecular weight (g/mol)
    liquid_density: Optional[float] = None  # Liquid density at ref conditions (kg/m³)
    cp: Optional[float] = None            # Liquid heat capacity (J/mol·K)
    tb: Optional[float] = None            # Normal boiling point (K) — energy balance
    hvap_tb: Optional[float] = None       # Latent heat at Tb (kJ/mol) — energy balance
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

    # two-suffix Margules A_ij (dimensionless, symmetric; one direction suffices)
    margules_aij: Dict[Tuple[str, str], float] = field(default_factory=dict)

    _ALL_DICTS = ("nrtl_aij", "nrtl_bij", "nrtl_cij", "uniquac_aij",
                  "uniquac_bij", "wilson_aij", "wilson_bij", "margules_aij")

    def remove_component(self, component_name: str):
        """Remove all entries involving a component."""
        for name in self._ALL_DICTS:
            d = getattr(self, name)
            for key in [k for k in d if component_name in k]:
                d.pop(key, None)


@dataclass
class ComponentThermoParams:
    """Thermodynamic parameters for a single component."""
    tc: Optional[float] = None  # Critical temperature (K)
    pc: Optional[float] = None  # Critical pressure (bar)
    omega: Optional[float] = None  # Acentric factor
    antoine_a: Optional[float] = None
    antoine_b: Optional[float] = None
    antoine_c: Optional[float] = None
    # Validity range of the Antoine fit (degC). None = unknown; the component
    # DB fills an estimate when the source table gives no explicit range.
    antoine_tmin: Optional[float] = None
    antoine_tmax: Optional[float] = None
    # UNIQUAC structural (van der Waals) volume/area parameters
    uniquac_r: Optional[float] = None
    uniquac_q: Optional[float] = None
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
    energy_balance: bool = False      # False => constant molar overflow (CMO);
                                      # True => real stage enthalpy balance in
                                      # Inside-Out (needs cp/tb/hvap/tc per comp)
    
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
        # Wagner is a *reduced* form: the user enters a..d, and Tc/Pc come from
        # the same per-component record the EOS uses. psat_params appends them so
        # core.thermodynamics keeps its plain-matrix contract (6 columns).
        "Wagner": ("wagner_a", "wagner_b", "wagner_c", "wagner_d"),
    }
    _PSAT_EXTRA = {"Wagner": ("tc", "pc")}

    # Multiplier from bar to the pressure unit each Psat model emits, so P and
    # Psat compare correctly in Raoult's law. Aspen-exported PLXANT coefficients
    # emit Psat in bar (C1 is the Pa-basis C1 minus ln(1e5)); Antoine fits are
    # mmHg. ponytail: both assumed a fixed unit; add a per-fit unit field if a
    # coefficient set in another unit (e.g. Pa-basis PLXANT) is ever entered.
    _BAR_TO_PSAT_UNIT = {"PLXANT": 1.0, "Wagner": 1.0, "Antoine": 750.0617}

    # Which activity/EOS models have a working implementation. The GUI greys
    # out everything else so entered parameters are never silently ignored.
    IMPLEMENTED_VLE = ("Antoine", "PLXANT", "Wagner")
    IMPLEMENTED_ACTIVITY = ("Ideal", "NRTL", "Wilson", "UNIQUAC", "Margules",
                            "UNIFAC")
    IMPLEMENTED_EOS = ("Ideal Gas", "SRK")

    def pressure_in_psat_unit(self, p_bar: float) -> float:
        """Pressure given in bar, converted to this vle_model's Psat unit."""
        return float(p_bar) * self._BAR_TO_PSAT_UNIT.get(self.vle_model, 750.0617)

    def psat_params(self, order):
        """Vapour-pressure coefficient matrix for `order`, shaped per vle_model.

        (N,7) for PLXANT, (N,6) for Wagner (a..d plus Tc/Pc), else (N,3) Antoine.
        core.thermodynamics.antoine_psat dispatches on the column count, so
        callers just pass this straight through. Raises ValueError naming the
        first component missing required coefficients, or an unimplemented
        vle_model — nothing entered is silently ignored.
        """
        import numpy as np
        if self.vle_model not in self._PSAT_KEYS:
            raise ValueError(
                f"Vapour-pressure model '{self.vle_model}' is not implemented — "
                f"choose one of {', '.join(self.IMPLEMENTED_VLE)} "
                "(Initialization → Thermodynamics).")
        keys = self._PSAT_KEYS[self.vle_model] + self._PSAT_EXTRA.get(
            self.vle_model, ())
        rows = []
        for nm in order:
            p = self.component_params.get(nm)
            vals = [getattr(p, k, None) for k in keys] if p else None
            if not vals or any(v is None for v in vals):
                missing = ", ".join(k for k, v in zip(keys, vals or [])
                                    if v is None) or "all"
                raise ValueError(f"Missing {self.vle_model} coefficients for '{nm}' "
                                 f"({missing}) — Initialization → Thermodynamics.")
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
        elif model_lower == "margules":
            return getattr(self.binary, f"margules_{param_lower}", {})
        return {}


@dataclass
class Stream:
    """Represents a stream in the column.

    user_specified: True once the user has edited this stream's flow or
    composition directly. auto_balance only overwrites *derived* products
    (user_specified False), so a hand-entered Distillate/Bottoms survives.
    (The dead per-stream pressure field was removed: column pressure + the
    per-stage profile cover it.)
    """
    id: str
    stream_type: StreamType
    stage: Optional[int] = None
    temperature: Optional[float] = None  # SI units (K)
    flow: Optional[float] = None  # SI units (kmol/h)
    composition: Dict[str, float] = field(default_factory=dict)  # species -> mole fraction
    user_specified: bool = False
    phase: str = "liquid"                # sidestream draw phase: liquid | vapor

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
    duty: Optional[float] = None          # internal heat duty (kW); + heat in
                                          # (interreboiler), - heat out (intercooler).
                                          # For a pumparound: heat removed by the
                                          # cooler (enter as a positive kW).
    return_stage: Optional[int] = None    # pumparound: stage the cooled liquid is
                                          # returned to (0-based from top; above the
                                          # draw stage)
    rate: Optional[float] = None          # circulating (pumparound) or drawn
                                          # (side stripper/rectifier) rate, kmol/h


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
        self.pressure = 1.0  # bar
        self.pressure_drop = 0.0  # bar/stage
        self.stage_efficiency = 1.0  # Murphree vapour efficiency (column-wide)

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

        # Same for the RBM (rectification-body) panel. Separate key: the two
        # modules answer different questions and share no levers beyond the
        # keys, so one dict would have them overwriting each other's saved
        # settings on every File->Save.
        self.rbm_params: dict = {}

        # Spec/DoF: structured extra specs (e.g. key-recovery) + CMO flag.
        # Condenser/reboiler/side-draw specs are derived from config in
        # collect_specs(); self.specs holds anything not on those panels yet.
        self.specs: List[Spec] = []
        self.light_key_index = 0             # 0-based light-key index
        self.heavy_key_index = None          # 0-based; None => defaults to lk+1
        self.results = None                  # last solver profile (Results tab reads this)
        from core.units import DisplayUnits
        self.display_units = DisplayUnits()  # output-only unit choices (Results/export)

        # Add default streams
        self._add_default_streams()

    def create_new_column(self):
        """Reset to a new empty column configuration (the single reset path;
        a separate clear() used to drift out of sync with this)."""
        self.current_tab = 0
        self.solver_mode = SolverMode.HYSIM
        self._tab_states = {}
        self.column_config = None
        self.is_modified = False
        self.file_path = None
        self.num_stages = 20
        self.pressure = 1.0
        self.pressure_drop = 0.0
        self.stage_efficiency = 1.0
        self.species = {}
        self.streams = {}
        self.condenser_config = CondenserConfig()
        self.reboiler_config = ReboilerConfig()
        self.thermodynamics_config = ThermodynamicsConfig()
        self.modules = {}
        self.bvm_params = {}
        self.rbm_params = {}
        self.specs = []
        self.light_key_index = 0
        self.heavy_key_index = None
        self.results = None
        from core.units import DisplayUnits
        self.display_units = DisplayUnits()

        # Add default streams
        self._add_default_streams()

    def _add_default_streams(self):
        """Add the standard Feed, Distillate, and Bottoms streams.
        Stages are 0-based from the top (0 = condenser/distillate, N-1 = reboiler)."""
        feed = Stream(id="Feed", stream_type=StreamType.FEED, stage=10)
        distillate = Stream(id="Distillate", stream_type=StreamType.DISTILLATE,
                            stage=0)
        bottoms = Stream(id="Bottoms", stream_type=StreamType.BOTTOMS,
                         stage=self.num_stages - 1)
        
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
    # `energy_balance` is deliberately absent: it lives on thermodynamics_config
    # (which is persisted) and is exposed here as a property. It used to be a
    # second, independent field — the Flow Model checkbox wrote the config one
    # while the DoF ledger read this one, so it stayed False forever and duty
    # specs were rejected on a column whose solver was running the energy
    # balance. Old .colx files still carry the stale key; it is ignored on load.
    _PERSIST = ("num_stages", "pressure", "pressure_drop", "stage_efficiency",
                "species", "streams", "condenser_config", "reboiler_config",
                "thermodynamics_config", "modules", "bvm_params", "rbm_params",
                "specs",
                "light_key_index", "heavy_key_index",
                "display_units", "solver_mode")

    @property
    def energy_balance(self) -> bool:
        """CMO off / energy balance on — the single flag, kept on the thermo
        config so the DoF ledger and the solver hook cannot disagree."""
        return bool(self.thermodynamics_config.energy_balance)

    @energy_balance.setter
    def energy_balance(self, value):
        self.thermodynamics_config.energy_balance = bool(value)

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

    # --- Activity-model registry (Phase 7) --------------------------------
    # model name -> builder(self, order) -> gamma_fn or None (ideal).
    # UNIQUAC / Wilson / SRK slot in here without touching any call site;
    # anything absent raises loudly instead of silently running ideal.

    def _gamma_ideal(self, order):
        return None

    def _gamma_nrtl(self, order):
        """HYSYS-modified NRTL closure from the entered binary parameters (None if
        none set). Matches src/native/nifco.f90's NRTL: tau_ij = a_ij + b_ij/T (K),
        G_ij = exp(-c_ij tau_ij), with c_ij the per-pair non-randomness (alpha).
        c_ij defaults to the usual 0.3 when the pair isn't entered."""
        tc = self.thermodynamics_config
        n = len(order)
        aij, bij, cij = tc.binary.nrtl_aij, tc.binary.nrtl_bij, tc.binary.nrtl_cij
        a = [[0.0] * n for _ in range(n)]
        b = [[0.0] * n for _ in range(n)]
        alpha = [[0.0] * n for _ in range(n)]
        any_param = False
        for i, ci in enumerate(order):
            for j, cj in enumerate(order):
                if i == j:
                    continue
                alpha[i][j] = cij.get((ci, cj), 0.3)   # nifco `c` = non-randomness
                if (ci, cj) in aij:
                    a[i][j] = aij[(ci, cj)]; any_param = True
                if (ci, cj) in bij:
                    b[i][j] = bij[(ci, cj)]; any_param = True
        if not any_param:
            return None
        from core.thermodynamics import nrtl_gamma_fn
        return nrtl_gamma_fn(a, b, alpha)

    def _pair_matrices(self, order, dict_names):
        """(n,n) matrices from directional binary dicts + any-entry flag."""
        b = self.thermodynamics_config.binary
        n = len(order)
        mats = [[[0.0] * n for _ in range(n)] for _ in dict_names]
        any_param = False
        for m, name in zip(mats, dict_names):
            d = getattr(b, name)
            for i, ci in enumerate(order):
                for j, cj in enumerate(order):
                    if i != j and (ci, cj) in d:
                        m[i][j] = d[(ci, cj)]
                        any_param = True
        return mats, any_param

    def _gamma_wilson(self, order):
        """Wilson closure, ln Lambda_ij = a_ij + b_ij/T_K (Aspen WILSON form).
        None when no parameters are entered (ideal)."""
        (a, b), any_param = self._pair_matrices(order, ("wilson_aij", "wilson_bij"))
        if not any_param:
            return None
        from core.thermodynamics import wilson_gamma_fn
        return wilson_gamma_fn(a, b)

    def _gamma_uniquac(self, order):
        """UNIQUAC closure, tau_ij = exp(a_ij + b_ij/T_K) (Aspen UNIQ form).
        Needs r/q for every component once any binary parameter is entered —
        raises instead of silently dropping the combinatorial term."""
        (a, b), any_param = self._pair_matrices(order, ("uniquac_aij", "uniquac_bij"))
        if not any_param:
            return None
        tc = self.thermodynamics_config
        r, q = [], []
        for nm in order:
            p = tc.component_params.get(nm)
            if p is None or p.uniquac_r is None or p.uniquac_q is None:
                raise ValueError(f"UNIQUAC needs structural r and q for '{nm}' "
                                 "(Initialization → Thermodynamics).")
            r.append(p.uniquac_r)
            q.append(p.uniquac_q)
        from core.thermodynamics import uniquac_gamma_fn
        return uniquac_gamma_fn(r, q, a, b)

    def _gamma_margules(self, order):
        """Two-suffix Margules closure. A_ij is symmetric, so a pair entered in
        either direction fills both."""
        (A,), any_param = self._pair_matrices(order, ("margules_aij",))
        if not any_param:
            return None
        import numpy as np
        A = np.asarray(A, float)
        A = np.where(A != 0.0, A, A.T)
        from core.thermodynamics import margules_gamma_fn
        return margules_gamma_fn(A)

    def _gamma_unifac(self, order):
        """UNIFAC closure built from each species' group counts (group-
        contribution — no binary parameters to enter). Raises naming the first
        species with no groups rather than silently running ideal."""
        groups = []
        for nm in order:
            sp = self.species.get(nm)
            g = dict(sp.unifac_groups) if sp and sp.unifac_groups else {}
            if not g:
                raise ValueError(f"UNIFAC needs group counts for '{nm}' "
                                 "(Initialization → Species → UNIFAC Groups).")
            groups.append(g)
        from core.thermodynamics import unifac_gamma_fn, load_unifac_db
        return unifac_gamma_fn(groups, load_unifac_db(), names=list(order))

    GAMMA_BUILDERS = {"Ideal": _gamma_ideal, "NRTL": _gamma_nrtl,
                      "Wilson": _gamma_wilson, "UNIQUAC": _gamma_uniquac,
                      "Margules": _gamma_margules, "UNIFAC": _gamma_unifac}

    def build_gamma_fn(self, order):
        """Activity-coefficient closure for the thermo layer, or None (ideal).

        Dispatches on the registry above. An unregistered model (UNIQUAC,
        Wilson, ...) raises with a user-facing message rather than silently
        ignoring the entered parameters.
        """
        model = self.thermodynamics_config.activity_model
        builder = self.GAMMA_BUILDERS.get(model)
        if builder is None:
            raise ValueError(
                f"Activity model '{model}' is not implemented yet — choose one "
                f"of {', '.join(self.thermodynamics_config.IMPLEMENTED_ACTIVITY)} "
                "(Initialization → Thermodynamics).")
        return builder(self, order)

    def build_phi_fn(self, order):
        """Vapour-phase fugacity closure for the thermo layer, or None (ideal gas).

        SRK consumes each component's Tc [K] / Pc [bar] / omega and raises
        naming the first component missing them — nothing silently ignored.
        """
        tc = self.thermodynamics_config
        model = tc.eos_model
        if model == "Ideal Gas":
            return None
        if model != "SRK":
            raise ValueError(
                f"Equation of state '{model}' is not implemented yet — choose "
                f"one of {', '.join(tc.IMPLEMENTED_EOS)} "
                "(Initialization → Thermodynamics).")
        crits = []
        for nm in order:
            p = tc.component_params.get(nm)
            vals = (getattr(p, "tc", None), getattr(p, "pc", None),
                    getattr(p, "omega", None)) if p else (None,) * 3
            if any(v is None for v in vals):
                raise ValueError(f"SRK needs Tc, Pc and omega for '{nm}' "
                                 "(Initialization → Thermodynamics).")
            crits.append(vals)
        import numpy as np
        from core.thermodynamics import srk_phi_fn
        c = np.array(crits, float)
        # P arrives in the vle_model's Psat unit; convert to Pa for the EOS
        p_to_Pa = 1.0e5 / tc._BAR_TO_PSAT_UNIT.get(tc.vle_model, 750.0617)
        return srk_phi_fn(c[:, 0], c[:, 1], c[:, 2], p_to_Pa=p_to_Pa)

    def _enthalpy_props(self, order):
        """(cp, hvap_tb, tb, tc) numpy arrays for `order`, or None if any
        component is missing any of the four — the non-raising sibling of
        build_energy_hook's collector (feed_quality falls back when None)."""
        import numpy as np
        tc = self.thermodynamics_config
        cp, hv, tb, tcr = [], [], [], []
        for nm in order:
            sp = self.species.get(nm)
            p = tc.component_params.get(nm)
            vals = (getattr(sp, "cp", None), getattr(sp, "hvap_tb", None),
                    getattr(sp, "tb", None), getattr(p, "tc", None) if p else None)
            if any(v is None for v in vals):
                return None
            cp.append(vals[0]); hv.append(vals[1]); tb.append(vals[2]); tcr.append(vals[3])
        return (np.array(cp, float), np.array(hv, float),
                np.array(tb, float), np.array(tcr, float))

    def build_energy_hook(self, order):
        """Energy-balance `flows_hook` for solve_inside_out, or None when CMO is
        selected. Needs each component's liquid Cp, Tb, latent heat at Tb, and
        Tc — raises naming the first component missing any, rather than silently
        falling back to CMO.
        """
        tc = self.thermodynamics_config
        if not tc.energy_balance:
            return None
        cp, hv, tb, tcr = [], [], [], []
        for nm in order:
            sp = self.species.get(nm)
            p = tc.component_params.get(nm)
            vals = (getattr(sp, "cp", None), getattr(sp, "hvap_tb", None),
                    getattr(sp, "tb", None), getattr(p, "tc", None) if p else None)
            if any(v is None for v in vals):
                raise ValueError(
                    f"Energy balance needs Cp, Tb, latent heat (Hvap) and Tc for "
                    f"'{nm}' — load it from the component DB or enter them "
                    "(Initialization → Species / Thermodynamics), or turn the "
                    "energy balance off to use constant molar overflow.")
            cp.append(vals[0]); hv.append(vals[1]); tb.append(vals[2]); tcr.append(vals[3])
        import numpy as np
        from core.column_solvers import make_energy_balance
        return make_energy_balance(np.array(cp, float), np.array(hv, float),
                                   np.array(tb, float), np.array(tcr, float))

    def feed_quality(self, stream, order) -> float:
        """Thermal quality q of a feed stream from its temperature (SI, K).

        q is the feed's liquid fraction *by enthalpy*:
            q = (Hv_sat - Hf) / (Hv_sat - Hl_sat)
        (Hl_sat at bubble T, Hv_sat at dew T, Hf at the actual feed T). This is
        the definition the energy balance's hFj = q*hL + (1-q)*hV expects, and
        it extends past [0,1] on purpose: a subcooled feed gives q>1 (its
        sensible-heat deficit → extra reboiler duty), a superheated feed q<0.

        Needs Cp/latent/Tb/Tc for every component; without them it falls back to
        the linear bubble/dew T-interpolation clamped to [0,1].
        """
        if not stream.temperature:
            return 1.0
        import numpy as np
        from core.thermodynamics import bubble_T, dew_T
        z = np.array([stream.composition.get(nm, 0.0) for nm in order], float)
        if abs(z.sum() - 1.0) > 1e-3:
            return 1.0
        tc = self.thermodynamics_config
        antoine = tc.psat_params(order)
        P = tc.pressure_in_psat_unit(self.pressure)
        gamma_fn = self.build_gamma_fn(order)
        phi_fn = self.build_phi_fn(order)
        T = float(stream.temperature) - 273.15   # SI K -> fit unit (degC)
        Tb = bubble_T(z, P, antoine, gamma_fn=gamma_fn, phi_fn=phi_fn)
        Td = dew_T(z, P, antoine, gamma_fn=gamma_fn, phi_fn=phi_fn)

        props = self._enthalpy_props(order)
        if props is not None:
            from core.enthalpy import enthalpy_fns
            hL, hV = enthalpy_fns(*props)
            Tf_K, Tb_K, Td_K = T + 273.15, Tb + 273.15, Td + 273.15
            Hl_sat = float(z @ hL(Tb_K))
            Hv_sat = float(z @ hV(Td_K))
            if T <= Tb:                       # subcooled/sat liquid
                Hf = float(z @ hL(Tf_K))
            elif T >= Td:                     # superheated/sat vapor
                Hf = float(z @ hV(Tf_K))
            else:                             # two-phase: interpolate in T
                Hf = Hl_sat + (T - Tb) / (Td - Tb) * (Hv_sat - Hl_sat)
            return (Hv_sat - Hf) / (Hv_sat - Hl_sat)

        # ponytail: linear bubble/dew T-interp fallback when Cp/latent absent;
        # loses subcooled/superheat sensible heat (clamped to [0,1]).
        if T <= Tb:
            return 1.0
        if T >= Td:
            return 0.0
        return float((Td - T) / (Td - Tb))

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

    def rename_stream(self, old_id: str, new_id: str) -> bool:
        """Rename a stream, keeping its data and every reference to it.

        Returns False (no change) if old_id is unknown, new_id is empty, or
        new_id is already taken — the caller reverts its display text.
        """
        new_id = new_id.strip()
        if old_id not in self.streams or not new_id or new_id in self.streams:
            return False
        # rebuild the dict so the stream keeps its position in the list
        self.streams = {(new_id if k == old_id else k): v
                        for k, v in self.streams.items()}
        self.streams[new_id].id = new_id
        self.is_modified = True
        return True

    def add_module(self, module_id: str, module: ModuleConfig):
        """Add a side module to the column."""
        self.modules[module_id] = module
        self.is_modified = True

    def remove_module(self, module_id: str):
        """Remove a side module from the column."""
        if module_id in self.modules:
            del self.modules[module_id]
            self.is_modified = True

    def interheater_duties(self):
        """[(gui_stage, duty_kW)] for modules carrying an internal heat duty
        (interreboiler +kW / intercooler -kW). Consumed by the energy balance as
        si.duty[]; ignored under CMO. gui_stage is 0-based from the top (0 =
        distillate), like feeds/draws.
        """
        # A pumparound's duty is its cooler — claimed by pumparounds() below and
        # folded into si.duty at build; counting it here too would double it.
        return [(m.stage, float(m.duty))
                for m in self.modules.values()
                if m.duty and m.module_type == ModuleType.INTERREBOILER]

    def pumparounds(self):
        """[(draw_stage, return_stage, rate, duty_kW)] for pumparound modules.

        Draw liquid `rate` at draw_stage, cool it (removing `duty` kW), return it
        to return_stage (above the draw). gui_stage is 0-based from the top, like
        feeds/draws; the cooling Q is consumed by the energy balance (same guard
        as interheater duties). Only fully-specified pumparounds are returned.
        """
        out = []
        for m in self.modules.values():
            if (m.module_type == ModuleType.PUMPAROUND and m.rate
                    and m.return_stage is not None):
                out.append((m.stage, m.return_stage, float(m.rate),
                            float(m.duty or 0.0)))
        return out

    # Side stripper / rectifier: the section's ratio spec lives on the type's own
    # field (boilup for a stripper, reflux for a rectifier).
    _SECTION_KIND = {ModuleType.SIDE_STRIPPER: "stripper",
                     ModuleType.SIDE_RECTIFIER: "rectifier"}

    @staticmethod
    def _section_ratio(m) -> Optional[float]:
        return (m.boilup_ratio if m.module_type == ModuleType.SIDE_STRIPPER
                else m.reflux_ratio)

    def side_sections(self):
        """[(id, kind, draw_stage, return_stage, rate, ratio, n_stages)] for
        fully-specified side strippers/rectifiers (see core.side_sections).

        A stripper draws liquid and returns vapour above the draw; a rectifier
        draws vapour and returns liquid below it. Stages are 0-based from the top
        like feeds/draws. Unlike duties these work under CMO — the ratio spec, not
        a heat term, sets the split.
        """
        out = []
        for mid, m in self.modules.items():
            kind = self._SECTION_KIND.get(m.module_type)
            ratio = self._section_ratio(m)
            if kind and m.rate and ratio and m.return_stage is not None:
                out.append((mid, kind, m.stage, m.return_stage, float(m.rate),
                            float(ratio), int(m.num_stages or 1)))
        return out

    def module_spec_counts(self) -> List[int]:
        """Design specs each module adds, in modules order.

        MESH ledger: one spec per duty unit the module adds plus one per extra
        product. Interheater = its duty (1). Pumparound = rate + cooler duty (2).
        Side stripper/rectifier = draw rate + its boilup/reflux ratio (2: a duty
        unit and an extra product).
        """
        return [1 if m.module_type == ModuleType.INTERREBOILER else 2
                for m in self.modules.values()]

    def module_specs(self) -> List[Spec]:
        """One Spec per module value the user has actually set, keyed by module id
        so the DoF ledger balances against module_spec_counts()."""
        specs: List[Spec] = []
        for mid, m in self.modules.items():
            if m.module_type == ModuleType.INTERREBOILER:
                if m.duty:
                    specs.append(Spec(SpecKind.MODULE_DUTY, float(m.duty), mid))
            elif m.module_type == ModuleType.PUMPAROUND:
                if m.rate:
                    specs.append(Spec(SpecKind.MODULE_RATE, float(m.rate), mid))
                if m.duty:
                    specs.append(Spec(SpecKind.MODULE_DUTY, float(m.duty), mid))
            else:
                ratio = self._section_ratio(m)
                if m.rate:
                    specs.append(Spec(SpecKind.MODULE_RATE, float(m.rate), mid))
                if ratio:
                    specs.append(Spec(SpecKind.MODULE_RATIO, float(ratio), mid))
        return specs

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
        from core.dof import Spec, OPERATING_KINDS
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
        specs: List[Spec] = list(self.specs)
        present = {s.kind for s in self.specs}

        def _legacy(kind, value, ref):
            # Legacy fallback per kind: only fill kinds the slots don't already
            # carry, so a config mixing old condenser/reboiler fields with new
            # operating slots never drops or double-counts a spec.
            if value and kind not in present:
                specs.append(Spec(kind, value, ref))

        cc = self.condenser_config
        if cc.condenser_type != CondenserType.NONE:
            _legacy(SpecKind.REFLUX_RATIO, cc.reflux_ratio, "condenser")
            _legacy(SpecKind.DISTILLATE_RATE, cc.vapor_distillate_flow, "condenser")
        rc = self.reboiler_config
        if rc.reboiler_type != ReboilerType.NONE:
            _legacy(SpecKind.BOILUP_RATIO, rc.boilup_ratio, "reboiler")
            _legacy(SpecKind.BOTTOMS_RATE, rc.bottoms_flow, "reboiler")
        for s in self.streams.values():
            if s.stream_type == StreamType.SIDESTREAM and s.flow:
                specs.append(Spec(SpecKind.SIDEDRAW_RATE, s.flow, s.id))
        specs.extend(self.module_specs())      # module knobs, keyed by module id
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
            module_spec_counts=self.module_spec_counts(),
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
        if s is None or s.user_specified:
            return                      # never clobber a user-entered product
        s.flow = float(flow)
        s.composition = {name: float(xi) for name, xi in zip(order, x)}


def _demo():
    """Self-check: unified DoF status + auto-balance end to end (no Qt)."""
    ws = WindowState()                      # default Feed/Distillate/Bottoms streams
    ws.add_species(Species(name="A"))
    ws.add_species(Species(name="B"))

    # Simple column: total condenser + kettle reboiler -> required == 2.
    assert ws.build_dof_analyzer().required_specs() == 2

    # One legacy reboiler field + one structured spec: the per-kind fallback
    # must count BOTH (the old all-or-nothing fallback dropped the boilup).
    ws.reboiler_config.boilup_ratio = 1.5
    ws.specs.append(Spec(SpecKind.LK_RECOVERY, 0.98, "column"))
    kinds = {s.kind for s in ws.collect_specs()}
    assert SpecKind.BOILUP_RATIO in kinds and SpecKind.LK_RECOVERY in kinds
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

    # A user-entered product is never clobbered by auto_balance.
    dist = ws.streams["Distillate"]
    dist.user_specified = True
    dist.flow = 123.0
    ws.auto_balance()
    assert dist.flow == 123.0, "auto_balance overwrote a user-specified product"
    dist.user_specified = False

    # Under-spec when both specs are removed -> no balance.
    ws.specs.clear()
    ws.reboiler_config.boilup_ratio = None
    assert not ws.get_specification_status()[2]
    assert ws.auto_balance() is None

    # Activity-model registry: registered models resolve (None until params
    # are entered); anything unregistered raises loudly.
    tc_ = ws.thermodynamics_config
    for m in tc_.IMPLEMENTED_ACTIVITY:
        if m == "UNIFAC":
            continue   # group-contribution: no "ideal until params" state
        tc_.activity_model = m
        assert ws.build_gamma_fn(["A", "B"]) is None, m   # no params yet
    tc_.activity_model = "Scatchard-Hildebrand"
    try:
        ws.build_gamma_fn(["A", "B"])
    except ValueError as exc:
        assert "not implemented" in str(exc)
    else:
        raise AssertionError("unimplemented activity model must raise")

    # Wilson closure from entered params reproduces the core equation.
    tc_.activity_model = "Wilson"
    tc_.binary.wilson_aij[("A", "B")] = -0.5
    tc_.binary.wilson_bij[("B", "A")] = -120.0
    gfn = ws.build_gamma_fn(["A", "B"])
    g = gfn([0.4, 0.6], 25.0)
    assert g[0] > 1.0 and g[1] > 1.0            # Lam < 1 both ways -> gamma > 1

    # UNIQUAC with binaries but missing r/q raises instead of dropping terms.
    tc_.activity_model = "UNIQUAC"
    tc_.binary.uniquac_aij[("A", "B")] = -0.3
    try:
        ws.build_gamma_fn(["A", "B"])
    except ValueError as exc:
        assert "r and q" in str(exc)
    else:
        raise AssertionError("UNIQUAC without r/q must raise")
    for nm, (r_, q_) in {"A": (2.1055, 1.972), "B": (0.92, 1.4)}.items():
        p_ = tc_.get_component_params(nm)
        p_.uniquac_r, p_.uniquac_q = r_, q_
    assert ws.build_gamma_fn(["A", "B"]) is not None

    # Margules: one direction entered symmetrises; ln g1 = A x2^2 at the limit.
    import math as _math
    tc_.activity_model = "Margules"
    tc_.binary.margules_aij[("A", "B")] = 1.2
    gfn = ws.build_gamma_fn(["A", "B"])
    assert abs(_math.log(gfn([0.0, 1.0], 25.0)[0]) - 1.2) < 1e-12

    # UNIFAC: needs per-species groups (raises without), then builds a closure
    # straight from group counts — no binary params. A=ethanol, B=water.
    tc_.activity_model = "UNIFAC"
    try:
        ws.build_gamma_fn(["A", "B"])
    except ValueError as exc:
        assert "group counts" in str(exc)
    else:
        raise AssertionError("UNIFAC without groups must raise")
    ws.species["A"] = Species(name="A", unifac_groups={"CH3": 1, "CH2": 1, "OH": 1})
    ws.species["B"] = Species(name="B", unifac_groups={"H2O": 1})
    gfn = ws.build_gamma_fn(["A", "B"])
    assert gfn([1e-6, 1 - 1e-6], 70.0)[0] > 2.0   # ethanol strongly non-ideal in water
    ws.species.pop("A"); ws.species.pop("B")
    tc_.activity_model = "Ideal"

    # Energy balance: off => CMO (None); on but missing data => raises; on with
    # cp/tb/hvap/tc => a usable flows_hook.
    assert ws.build_energy_hook(["A", "B"]) is None      # energy_balance False
    tc_.energy_balance = True
    ws.species["A"] = Species(name="A", cp=136.0, tb=353.2, hvap_tb=30.8)
    tc_.get_component_params("A").tc = 562.0
    try:
        ws.build_energy_hook(["A", "B"])                 # B has no enthalpy data
    except ValueError as exc:
        assert "Energy balance needs" in str(exc)
    else:
        raise AssertionError("energy balance without data must raise")
    ws.species["B"] = Species(name="B", cp=157.0, tb=383.8, hvap_tb=33.2)
    tc_.get_component_params("B").tc = 591.8
    hook = ws.build_energy_hook(["A", "B"])
    assert callable(hook) and hasattr(hook, "Qc")
    tc_.energy_balance = False
    ws.species.pop("A"); ws.species.pop("B")

    # remove_component clears every model's dicts (incl. margules)
    tc_.binary.remove_component("A")
    assert not tc_.binary.wilson_aij and not tc_.binary.margules_aij
    tc_.activity_model = "Ideal"

    # EOS registry: Ideal Gas -> None; SRK without critical constants raises;
    # with Tc/Pc/omega it yields phi < 1 for a subcritical vapour; PR raises.
    assert ws.build_phi_fn(["A", "B"]) is None
    tc_.eos_model = "SRK"
    try:
        ws.build_phi_fn(["A", "B"])
    except ValueError as exc:
        assert "Tc, Pc and omega" in str(exc)
    else:
        raise AssertionError("SRK without Tc/Pc/omega must raise")
    for nm, (tcK, pcbar, om) in {"A": (369.83, 42.48, 0.152),
                                 "B": (425.12, 37.96, 0.200)}.items():
        p_ = tc_.get_component_params(nm)
        p_.tc, p_.pc, p_.omega = tcK, pcbar, om
    pfn = ws.build_phi_fn(["A", "B"])
    phi = pfn([0.5, 0.5], 20.0, 4.0 * 760.0)    # degC, mmHg (Antoine unit)
    assert all(0.7 < v < 1.0 for v in phi), phi
    tc_.eos_model = "PR"
    try:
        ws.build_phi_fn(["A", "B"])
    except ValueError as exc:
        assert "not implemented" in str(exc)
    else:
        raise AssertionError("unimplemented EOS must raise")
    tc_.eos_model = "Ideal Gas"

    # Unsupported vapour-pressure model raises instead of silently using Antoine.
    ws.thermodynamics_config.vle_model = "Rackett"
    try:
        ws.thermodynamics_config.psat_params(["A", "B"])
    except ValueError as exc:
        assert "not implemented" in str(exc)
    else:
        raise AssertionError("unimplemented vle model must raise")

    # Wagner IS implemented, but still refuses when a component has no
    # coefficients — naming which ones are missing.
    ws.thermodynamics_config.vle_model = "Wagner"
    try:
        ws.thermodynamics_config.psat_params(["A", "B"])
    except ValueError as exc:
        assert "wagner_a" in str(exc), exc
    else:
        raise AssertionError("Wagner must refuse a component with no constants")
    ws.thermodynamics_config.vle_model = "Antoine"

    # Feed quality from stream temperature: benzene/toluene feed at 1 atm.
    ws2 = WindowState()
    ws2.pressure = 1.01325
    abc = [(6.90565, 1211.033, 220.79), (6.95464, 1344.8, 219.48)]
    for nm, (a, b, c) in zip(["benzene", "toluene"], abc):
        ws2.add_species(Species(name=nm))
        p = ws2.thermodynamics_config.get_component_params(nm)
        p.antoine_a, p.antoine_b, p.antoine_c = a, b, c
    f = Stream(id="F", stream_type=StreamType.FEED, stage=10, flow=100.0,
               composition={"benzene": 0.5, "toluene": 0.5})
    assert ws2.feed_quality(f, ["benzene", "toluene"]) == 1.0  # no T -> sat liquid
    f.temperature = 273.15 + 60.0           # subcooled (bubble ~92 degC)
    assert ws2.feed_quality(f, ["benzene", "toluene"]) == 1.0
    f.temperature = 273.15 + 130.0          # superheated (dew ~98 degC)
    assert ws2.feed_quality(f, ["benzene", "toluene"]) == 0.0
    f.temperature = 273.15 + 95.0           # two-phase -> 0 < q < 1
    q = ws2.feed_quality(f, ["benzene", "toluene"])
    assert 0.0 < q < 1.0, q

    # With Cp/latent/Tb/Tc, feed_quality uses the enthalpy definition: subcooled
    # feed gives q>1, superheated q<0 (sensible heat, not clamped to [0,1]).
    props = {"benzene": (136.0, 30.8, 353.2, 562.0),
             "toluene": (157.0, 33.2, 383.8, 591.8)}
    for nm, (cp, hv, tb, tcc) in props.items():
        sp = ws2.species[nm]; sp.cp, sp.hvap_tb, sp.tb = cp, hv, tb
        ws2.thermodynamics_config.get_component_params(nm).tc = tcc
    f.temperature = 273.15 + 60.0           # subcooled -> q > 1
    assert ws2.feed_quality(f, ["benzene", "toluene"]) > 1.0
    f.temperature = 273.15 + 130.0          # superheated -> q < 0
    assert ws2.feed_quality(f, ["benzene", "toluene"]) < 0.0
    f.temperature = 273.15 + 95.0           # two-phase still in (0, 1)
    assert 0.0 < ws2.feed_quality(f, ["benzene", "toluene"]) < 1.0

    # Pressure unit bridge: 1 atm (bar) -> 760 mmHg (Antoine) / 1.01325 bar (PLXANT).
    tc = ws.thermodynamics_config
    tc.vle_model = "Antoine"
    assert abs(tc.pressure_in_psat_unit(1.01325) - 760.0) < 0.1
    tc.vle_model = "PLXANT"
    assert abs(tc.pressure_in_psat_unit(1.0) - 1.0) < 1e-9
    print("window_state self-check OK")


if __name__ == "__main__":
    _demo()
