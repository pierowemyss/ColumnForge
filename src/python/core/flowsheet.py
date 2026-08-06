#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A flowsheet of Columns coupled by streams, solved sequential-modular.

This is Tier 2 (CONTEXT.md): a directed graph of Columns, each of which is still
exactly the Column the rigorous solvers already rate. **Nothing in
core.column_solvers changes.** A downstream column's feed is an upstream
column's product, and every field needed to build it -- flow, composition,
thermal quality -- is already in the profile dict `_finish_profile` returns.

The model is `core.side_sections` one level up. That module couples a
sub-column to a main column by *tearing*: guess the returning stream, solve,
re-read it, repeat, with an Aitken jump summing the geometric tail. The same
scheme generalizes to a graph:

  * an **acyclic** flowsheet needs no tear at all -- solve the units in
    topological order, one pass, each one's inlet already final;
  * a **cyclic** one tears the back edges of each strongly connected component
    and iterates Gauss-Seidel until the torn streams stop moving.

Seams and ceilings:

  * Species, thermo and the gamma/phi closures are **flowsheet-global**: they
    live on the Flowsheet, never on a Unit. A Unit carries only what varies
    per column (stages, pressure, specs, its own feeds and draws).
  * Ports are keyed by a **stable opaque string** minted once by whoever builds
    the Unit. Renaming a side draw, or moving it to another stage, changes the
    Port's `label`/`stage` and never the key -- so a Connection is never
    rewritten by an edit somewhere else in the UI.
  * A **side stripper/rectifier stays a `SideSection`, not a Unit.** It is
    profile-coupled (it reads the liquid on stage j, not a product port), it has
    no operating point to resolve (its split is closed-form from its ratio
    spec), and it couples two-way into the same shell. Its tear nests inside
    this one and keeps its own `return_comp` between passes, so from pass 2 it
    costs 2-3 inner passes. See `side_sections.make_side_solver`.
  * A purge is a `Connection.split_fraction` below 1: the unsent remainder
    leaves the flowsheet as an external product. There are no splitter or mixer
    block types. Makeup is just another external feed on the destination stage --
    `build_solver_input` already blends co-fed stages with a flow-weighted q.

Two things this module is emphatic about, both learned the hard way elsewhere in
this codebase:

  * **`closure` is not a convergence test.** Under CMO every unit's flows close
    by construction, so a flowsheet whose recycle *compositions* are still badly
    wrong reports closure ~1e-12. Only `tear_residual` measures the tear, and
    only `converged` gates a result (`found` is a cancellation flag).
  * **"did not converge" must name which thing didn't.** A unit that ran out of
    MESH iterations and a tear that ran out of passes are different failures
    with different fixes, and a message that says only "not converged" sends the
    user to the wrong knob.

Stage numbers here are solver-internal (1 = top .. N = reboiler), like
SolverInput and side_sections. The GUI's 0-based-from-the-top numbering is
converted at the boundary, by `_stage_internal` in main_window, as always.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.dof import Spec, SpecKind, OPERATING_KINDS
from core.side_sections import SideSection, aitken_step, make_side_solver
from core.solver_input import build_solver_input


class FlowsheetError(ValueError):
    """A flowsheet that cannot be solved as posed.

    A ValueError subclass on purpose: the GUI's solver worker already routes
    ValueError to a user-facing dialog and anything else to a traceback
    (gui/solver_worker.py), and every one of these is the user's to fix.
    """


# --- the model -------------------------------------------------------------

#: Nominal tear passes used only to size the progress bar's denominator.
NOMINAL_PASSES = 12


@dataclass(frozen=True)
class Port:
    """An outlet of a Unit. `key` is stable; everything else is display or
    physics that an edit elsewhere may legitimately change."""
    key: str
    kind: str                  # distillate | bottoms | draw | section
    stage: Optional[int] = None
    phase: str = "liquid"
    label: str = ""


@dataclass(frozen=True)
class Connection:
    """`split_fraction` of `src`'s `port` goes to `dst` at `stage`.

    The remainder (1 - the sum of split fractions leaving that port) leaves the
    flowsheet as an external product -- that is what a purge is. `q=None` means
    the port's natural thermal quality; anything else is an explicit inter-unit
    heater/cooler and the UI must show it as the override it is.
    """
    id: str
    src: str
    port: str
    dst: str
    stage: int
    split_fraction: float = 1.0
    q: Optional[float] = None


@dataclass
class Unit:
    """One column. Everything global (species, thermo) lives on the Flowsheet.

    `pressure` and `antoine` are None on a **topology-only** Unit — one built
    just to ask about ports, tears and connection legality. The editor needs
    those answers while a column is still half-specified, and none of
    `ports`, `tear_set`, `validate_connection` or `auto_layout` touches thermo.
    `solve_flowsheet` does, and fails loudly if they are missing.
    """
    id: str
    n_stages: int
    pressure: object = None               # scalar or (N,), in the Psat fit's unit
    antoine: Optional[np.ndarray] = None
    specs: List[Spec] = field(default_factory=list)
    # external sources/sinks; inter-unit streams arrive via Connections
    feeds: List[Tuple[int, float, Sequence[float], float]] = field(default_factory=list)
    draws: List[Tuple[str, int, float, float]] = field(default_factory=list)
    duties: List[Tuple[int, float]] = field(default_factory=list)
    pumparounds: List[Tuple[int, int, float, float]] = field(default_factory=list)
    sections: List[SideSection] = field(default_factory=list)
    condenser: str = "total"
    subcooling: float = 0.0
    efficiency: float = 1.0
    method: Optional[str] = None          # None => Flowsheet.default_method
    flows_hook: Optional[Callable] = None  # energy balance; Inside-Out only
    lk: int = 0
    hk: Optional[int] = None
    node_pos: Optional[Tuple[float, float]] = None

    def ports(self) -> Dict[str, Port]:
        """Derived, never stored: a stored port table drifts from the draws it
        describes the first time someone deletes a side draw."""
        vapor_top = self.condenser in ("partial", "none")
        out = {
            "D": Port("D", "distillate", 1,
                      "vapor" if vapor_top else "liquid", "Distillate"),
            "B": Port("B", "bottoms", self.n_stages, "liquid", "Bottoms"),
        }
        for key, stage, liq, vap in self.draws:
            out[key] = Port(key, "draw", stage,
                            "vapor" if vap > liq else "liquid", key)
        for sec in self.sections:
            out[sec.id] = Port(sec.id, "section", sec.draw_stage, "liquid",
                               f"{sec.id} product")
        return out

    def draw_total(self) -> float:
        """Everything leaving as a side product: draws plus section products.
        This is `side_draw_total` for the operating-point resolver."""
        return (sum(liq + vap for _, _, liq, vap in self.draws)
                + sum(s.product_flow for s in self.sections))


@dataclass
class Flowsheet:
    units: Dict[str, Unit]
    connections: List[Connection] = field(default_factory=list)
    comps: List[str] = field(default_factory=list)
    default_method: str = "Inside-Out"
    gamma_fn: Optional[Callable] = None
    phi_fn: Optional[Callable] = None

    @property
    def n_comps(self) -> int:
        return len(self.comps)

    def out_edges(self, uid: str) -> List[Connection]:
        return [c for c in self._sorted() if c.src == uid]

    def in_edges(self, uid: str) -> List[Connection]:
        return [c for c in self._sorted() if c.dst == uid]

    def _sorted(self) -> List[Connection]:
        """Deterministic edge order -- the tear set is derived from a DFS, and a
        tear set that depends on dict insertion order is a tear set that differs
        between a saved file and the session that saved it."""
        return sorted(self.connections, key=lambda c: (c.src, c.port, c.dst, c.stage))


def natural_q(unit: Unit, port_key: str) -> float:
    """Thermal quality of what leaves this port: 1 = saturated liquid."""
    port = unit.ports().get(port_key)
    if port is None:
        raise FlowsheetError(f"'{unit.id}' has no port '{port_key}'")
    if port.kind == "distillate":
        return 0.0 if unit.condenser in ("partial", "none") else 1.0
    if port.kind == "bottoms":
        return 1.0
    if port.kind == "draw":
        return 0.0 if port.phase == "vapor" else 1.0
    return 1.0          # a section product is liquid at both ends (see side_sections)


def port_stream(unit: Unit, port_key: str, prof: dict) -> Tuple[float, np.ndarray]:
    """(flow, composition) leaving `port_key`, read off a solved profile."""
    port = unit.ports()[port_key]
    if port.kind == "distillate":
        return float(prof["D"]), np.asarray(prof["xD"], float)
    if port.kind == "bottoms":
        return float(prof["B"]), np.asarray(prof["xB"], float)
    if port.kind == "draw":
        for sd in prof.get("side_draws", []):
            if sd["stage"] == port.stage - 1:
                if port.phase == "vapor":
                    return float(sd["vapor"]), np.asarray(sd["y"], float)
                return float(sd["liquid"]), np.asarray(sd["x"], float)
        return 0.0, np.zeros(len(prof["comps"]))
    for ss in prof.get("side_sections", []):
        if ss["id"] == port_key:
            return float(ss["flow"]), np.asarray(ss["comp"], float)
    return 0.0, np.zeros(len(prof["comps"]))


# --- topology --------------------------------------------------------------

def sccs(fs: Flowsheet) -> List[List[str]]:
    """Strongly connected components, each sorted, in dependency order.

    Tarjan for the components, then Kahn over the condensation for the order --
    rather than relying on Tarjan's emission order, which is right but is the
    kind of "right" that a reader has to go and look up.
    """
    adj = {u: sorted({c.dst for c in fs.connections if c.src == u and c.dst in fs.units})
           for u in fs.units}
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {}
    stack: List[str] = []
    comps: List[List[str]] = []
    counter = [0]

    def strong(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in adj[v]:
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif on_stack.get(w):
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            comps.append(sorted(comp))

    for u in sorted(fs.units):
        if u not in index:
            strong(u)

    # Kahn over the condensation, ties broken by the component's first member
    where = {u: i for i, comp in enumerate(comps) for u in comp}
    n = len(comps)
    succ = {i: set() for i in range(n)}
    indeg = [0] * n
    for c in fs.connections:
        if c.src not in where or c.dst not in where:
            continue
        a, b = where[c.src], where[c.dst]
        if a != b and b not in succ[a]:
            succ[a].add(b)
            indeg[b] += 1
    ready = sorted([i for i in range(n) if indeg[i] == 0], key=lambda i: comps[i][0])
    order: List[List[str]] = []
    while ready:
        i = ready.pop(0)
        order.append(comps[i])
        for j in sorted(succ[i], key=lambda j: comps[j][0]):
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
        ready.sort(key=lambda i: comps[i][0])
    return order


def tear_set(fs: Flowsheet) -> List[str]:
    """Connection ids to tear: the DFS back edges inside each cyclic component.

    Not a minimum feedback arc set. MFAS is NP-hard, this program's realistic
    ceiling is under a dozen columns, and -- the part that actually decides it --
    the pass count of successive substitution is set by the contraction factor
    of the slowest loop, not by how many streams are torn. Tearing a superset
    costs carried state, not iterations. A DFS also finds no back edge exactly
    when the graph is acyclic, so the answer is a guaranteed-valid tear set
    rather than a heuristic that needs checking.

    # ponytail: DFS back edges. Upgrade path if a 20-unit sheet ever crawls:
    # Barkley-Motard weighted tearing, tearing the smallest nominal flow first.
    """
    torn: List[str] = []
    edges = fs._sorted()
    for comp in sccs(fs):
        members = set(comp)
        inside = [c for c in edges if c.src in members and c.dst in members]
        if not inside:
            continue
        color = {u: 0 for u in comp}          # 0 white, 1 gray, 2 black

        def visit(v: str) -> None:
            color[v] = 1
            for c in inside:
                if c.src != v:
                    continue
                if color[c.dst] == 1:          # back edge (a self-loop included)
                    torn.append(c.id)
                elif color[c.dst] == 0:
                    visit(c.dst)
            color[v] = 2

        for root in comp:
            if color[root] == 0:
                visit(root)
    return torn


def is_recycle(fs: Flowsheet, conn: Connection) -> bool:
    """True if this edge is torn — the same answer the driver uses, so the
    picture the editor draws and the streams the solver guesses cannot disagree."""
    return conn.id in set(tear_set(fs))


def _unit_order(fs: Flowsheet, comp: Sequence[str], torn: set) -> List[str]:
    """Units of one component, ordered so Gauss-Seidel sees fresh upstream
    values: topological over the component with the torn edges removed."""
    members = set(comp)
    inside = [c for c in fs._sorted()
              if c.src in members and c.dst in members and c.id not in torn]
    indeg = {u: 0 for u in comp}
    for c in inside:
        if c.src != c.dst:
            indeg[c.dst] += 1
    ready = sorted([u for u in comp if indeg[u] == 0])
    order: List[str] = []
    while ready:
        u = ready.pop(0)
        order.append(u)
        for c in inside:
            if c.src == u and c.dst != u:
                indeg[c.dst] -= 1
                if indeg[c.dst] == 0:
                    ready.append(c.dst)
        ready.sort()
    # a residual cycle can only mean the tear set missed one; fail loudly
    return order + sorted(set(comp) - set(order))


def validate_connection(fs: Flowsheet, conn: Connection) -> Optional[str]:
    """None if legal, else a sentence the UI can show verbatim."""
    if conn.src not in fs.units:
        return f"no column '{conn.src}'"
    if conn.dst not in fs.units:
        return f"no column '{conn.dst}'"
    src, dst = fs.units[conn.src], fs.units[conn.dst]
    if conn.port not in src.ports():
        return f"'{src.id}' has no outlet '{conn.port}'"
    if conn.src == conn.dst:
        return ("a column cannot feed itself — a stream drawn and returned to "
                "the same shell is a pumparound or a side section, not a "
                "connection")
    if not (2 <= conn.stage <= dst.n_stages - 1):
        return (f"stage {conn.stage} is not an interior tray of '{dst.id}' "
                f"(2..{dst.n_stages - 1}); the condenser and reboiler stages "
                "cannot take a feed")
    if not (0.0 < conn.split_fraction <= 1.0):
        return f"split fraction {conn.split_fraction:g} must be in (0, 1]"
    committed = sum(c.split_fraction for c in fs.connections
                    if c.src == conn.src and c.port == conn.port and c.id != conn.id)
    if committed + conn.split_fraction > 1.0 + 1e-9:
        return (f"'{src.id}.{conn.port}' already sends {committed:.3g} of itself "
                f"elsewhere; {conn.split_fraction:.3g} more would be "
                f"{committed + conn.split_fraction:.3g} of one stream")
    if any(c.id != conn.id and c.src == conn.src and c.port == conn.port
           and c.dst == conn.dst and c.stage == conn.stage for c in fs.connections):
        return f"'{src.id}.{conn.port}' already feeds '{dst.id}' stage {conn.stage}"
    if conn.q is not None and not (0.0 <= conn.q <= 1.0):
        return f"thermal quality {conn.q:g} must be between 0 and 1"
    return None


def validate(fs: Flowsheet) -> List[str]:
    """Every problem with the flowsheet as posed, for the status line."""
    problems = [f"{c.id}: {why}" for c in fs._sorted()
                if (why := validate_connection(fs, c)) is not None]

    for uid in sorted(fs.units):
        u = fs.units[uid]
        n_free = 1 if u.condenser == "none" else 2
        ops = [s for s in u.specs if s.kind in OPERATING_KINDS]
        if len(ops) != n_free:
            problems.append(
                f"{uid}: needs exactly {n_free} operating spec(s) besides "
                f"side-draw rates, has {len(ops)}")

    # A loop nobody feeds can only carry zero flow. Converging to zeros and
    # calling it solved is worse than saying so.
    fed = {uid for uid, u in fs.units.items() if any(f[1] > 0 for f in u.feeds)}
    for comp in sccs(fs):
        members = set(comp)
        cyclic = len(comp) > 1 or any(
            c.src == c.dst for c in fs.connections if c.src in members)
        if cyclic and not (members & fed):
            problems.append(
                f"recycle loop {{{', '.join(comp)}}} has no external feed — "
                "it can only carry zero flow")
    return problems


def auto_layout(fs: Flowsheet, dx: float = 340.0, dy: float = 420.0) -> Dict[str, Tuple[float, float]]:
    """Layered left-to-right placement, deterministic so a test can assert it.
    Members of one strongly connected component share a layer."""
    order = sccs(fs)
    layer_of: Dict[str, int] = {}
    for depth, comp in enumerate(order):
        layer = 0
        for c in fs.connections:
            if c.dst in comp and c.src in layer_of and c.src not in comp:
                layer = max(layer, layer_of[c.src] + 1)
        for u in comp:
            layer_of[u] = layer
    rows: Dict[int, int] = {}
    pos: Dict[str, Tuple[float, float]] = {}
    for comp in order:
        for u in comp:
            lay = layer_of[u]
            row = rows.get(lay, 0)
            rows[lay] = row + 1
            pos[u] = (lay * dx, row * dy)
    if pos:
        cx = sum(p[0] for p in pos.values()) / len(pos)
        cy = sum(p[1] for p in pos.values()) / len(pos)
        pos = {u: (x - cx, y - cy) for u, (x, y) in pos.items()}
    return pos


# --- results ---------------------------------------------------------------

@dataclass
class UnitResult:
    unit_id: str
    profile: dict
    converged: bool
    aborted: bool
    R: float
    D: float
    solves: int
    message: str
    si: object = None       # the resolved SolverInput this profile came from


@dataclass
class StreamState:
    conn_id: str
    src: str
    port: str
    dst: str
    stage: int
    flow: float
    comp: np.ndarray
    q: float
    torn: bool = False


@dataclass
class FlowsheetResult:
    units: Dict[str, UnitResult] = field(default_factory=dict)
    streams: Dict[str, StreamState] = field(default_factory=dict)
    tear_ids: List[str] = field(default_factory=list)
    tear_residual: float = 0.0
    tear_converged: bool = True
    tear_passes: int = 0
    tol_used: float = 0.0
    converged: bool = False
    aborted: bool = False
    message: str = ""
    feed_totals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    products: List[dict] = field(default_factory=list)
    closure: float = 0.0


# --- the cold start --------------------------------------------------------

def _sharp_split(w_in: np.ndarray, D_total: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fill the distillate from the lightest component down.

    The component order IS the volatility order everywhere in this program (lk
    and hk are indices into it), so "lightest down" needs no K-values -- which
    is the point: the seed must cost no column solves.
    """
    d = np.zeros_like(w_in)
    left = max(0.0, float(D_total))
    for i in range(len(w_in)):
        take = min(float(w_in[i]), left)
        d[i] = take
        left -= take
        if left <= 0.0:
            break
    return d, w_in - d


def _estimate_D(unit: Unit, w_in: np.ndarray) -> float:
    """Distillate rate for the seed: a rate spec if one pins it, else the
    light-key cut (the same heuristic operating_specs uses for its D0)."""
    F = float(w_in.sum())
    W = unit.draw_total()
    for sp in unit.specs:
        if sp.kind == SpecKind.DISTILLATE_RATE:
            return float(sp.value)
        if sp.kind == SpecKind.BOTTOMS_RATE:
            return F - W - float(sp.value)
        if sp.kind == SpecKind.DF_RATIO:
            return float(sp.value) * F
        if sp.kind == SpecKind.BF_RATIO:
            return F - W - float(sp.value) * F
    return float(w_in[: unit.lk + 1].sum())


def _split_unit(unit: Unit, w_in: np.ndarray) -> Dict[str, np.ndarray]:
    """The seed's model of one column: a sharp component splitter."""
    F = float(w_in.sum())
    ports = unit.ports()
    out = {k: np.zeros_like(w_in) for k in ports}
    if F <= 0.0:
        return out
    z = w_in / F
    taken = 0.0
    for key, port in ports.items():
        if port.kind == "draw":
            rate = next((liq + vap for k, _, liq, vap in unit.draws if k == key), 0.0)
        elif port.kind == "section":
            rate = next((s.product_flow for s in unit.sections if s.id == key), 0.0)
        else:
            continue
        rate = min(rate, max(0.0, F - taken))
        out[key] = rate * z
        taken += rate
    rest = w_in - sum(out[k] for k in out)
    rest = np.clip(rest, 0.0, None)
    D = min(max(_estimate_D(unit, w_in), 0.0), float(rest.sum()))
    out["D"], out["B"] = _sharp_split(rest, D)
    return out


def _linear_seed(fs: Flowsheet, passes: int = 300, tol: float = 1e-12) -> Dict[str, StreamState]:
    """Pre-solve the flowsheet as a network of sharp splitters, before any MESH
    solve exists to be wrong about.

    This is the general form of the lesson in side_sections: an unseeded first
    pass carries the DRAW without the RETURN, so the main column is solved short
    by the whole recycle and dies with "bottoms rate B=-5 must be positive" -- an
    error that blames the user's distillate rate for a missing internal stream.
    Here every unit's first real solve already sees a mass-balanced inlet
    including its recycle, so B = F_in - D - W is positive on pass 1.

    # ponytail: successive substitution on a piecewise-linear splitter model
    # rather than assembling and solving the linear recycle system. The map is
    # piecewise (a component split saturates), it contracts by the recycle
    # fraction, and 300 passes of a few numpy ops is microseconds. Build the
    # matrix if a seed ever measurably fails to settle.
    """
    C = fs.n_comps
    streams: Dict[str, StreamState] = {}
    for c in fs._sorted():
        src = fs.units[c.src]
        streams[c.id] = StreamState(
            conn_id=c.id, src=c.src, port=c.port, dst=c.dst, stage=c.stage,
            flow=0.0, comp=np.full(C, 1.0 / C),
            q=c.q if c.q is not None else natural_q(src, c.port))
    order = [u for comp in sccs(fs) for u in comp]
    for _ in range(passes):
        moved = 0.0
        for uid in order:
            unit = fs.units[uid]
            w_in = np.zeros(C)
            for stage, flow, z, _q in unit.feeds:
                w_in += float(flow) * np.asarray(z, float)
            for c in fs.in_edges(uid):
                st = streams[c.id]
                w_in += st.flow * st.comp
            split = _split_unit(unit, w_in)
            for c in fs.out_edges(uid):
                w = split[c.port] * c.split_fraction
                flow = float(w.sum())
                st = streams[c.id]
                moved = max(moved, abs(flow - st.flow))
                st.flow = flow
                if flow > 0.0:
                    st.comp = w / flow
        if moved < tol:
            break
    return streams


# --- the driver ------------------------------------------------------------

def _inlets(fs: Flowsheet, unit: Unit,
            streams: Dict[str, StreamState]) -> List[Tuple[int, float, np.ndarray, float]]:
    """Every feed the unit sees: its own external ones plus the connections."""
    out = [(int(s), float(f), np.asarray(z, float), float(q))
           for s, f, z, q in unit.feeds]
    for c in fs.in_edges(unit.id):
        st = streams[c.id]
        if st.flow > 0.0:
            out.append((c.stage, st.flow, st.comp.copy(), st.q))
    return out


def _tear_vec(st: StreamState, F_ref: float) -> np.ndarray:
    """[component flows / F_ref, q].

    Scaling by the external feed is not cosmetic: mixing kmol/h flows (order
    100) with a thermal quality (order 1) in one norm makes q invisible to both
    the Aitken ratio and the `moved < tol` test.
    """
    return np.concatenate([st.flow * st.comp / F_ref, [st.q]])


def _project_tear(x: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Aitken's jump, put back on the manifold: flows non-negative, q in [0, 1].
    Deliberately NOT renormalized to sum 1 — these are flows, not a composition."""
    y = np.asarray(x, float).copy()
    y[:-1] = np.clip(y[:-1], 0.0, None)
    y[-1] = min(1.0, max(0.0, float(y[-1])))
    return new if y[:-1].sum() <= 0.0 else y


def _apply_vec(st: StreamState, v: np.ndarray, F_ref: float) -> None:
    w = np.clip(v[:-1], 0.0, None) * F_ref
    flow = float(w.sum())
    st.flow = flow
    if flow > 0.0:
        st.comp = w / flow
    st.q = min(1.0, max(0.0, float(v[-1])))


def _solve_unit(fs: Flowsheet, unit: Unit, streams, warm, op, knobs,
                cancel) -> UnitResult:
    """One column, at the inlet it currently has. Mirrors the pipeline
    main_window._gather_rigorous_inputs runs today, per unit, per pass."""
    from core.column_solvers import solve_bubble_point, solve_inside_out

    if unit.antoine is None or unit.pressure is None:
        raise FlowsheetError(
            f"'{unit.id}' has no pressure/vapour-pressure data — it was built "
            "for topology only and cannot be solved.")
    inlets = _inlets(fs, unit, streams)
    F_in = sum(f[1] for f in inlets)
    if F_in <= 0.0:
        raise FlowsheetError(
            f"'{unit.id}' has no inlet flow — check its feed, or the connection "
            "that is supposed to feed it")
    z_in = sum(f[1] * f[2] for f in inlets) / F_in
    W = unit.draw_total()

    def build_si(R, D):
        si_feeds = list(inlets) + [
            (s.return_stage, s.return_flow, s.return_comp, s.return_q)
            for s in unit.sections if s.return_comp is not None]
        si_draws = [(stage, liq, vap) for _, stage, liq, vap in unit.draws]
        si_draws += [(s.draw_stage, *s.draw_rates()) for s in unit.sections]
        return build_solver_input(
            n_stages=unit.n_stages, comps=fs.comps, feeds=si_feeds,
            draws=si_draws, duties=unit.duties, pumparounds=unit.pumparounds,
            R=R, D=D, pressure=unit.pressure, antoine=unit.antoine,
            gamma_fn=fs.gamma_fn, phi_fn=fs.phi_fn,
            condenser=unit.condenser, subcooling=unit.subcooling)

    method = unit.method or fs.default_method
    is_io = "Inside-Out" in method
    solver = solve_inside_out if is_io else solve_bubble_point
    ku = dict(knobs or {})
    ku["efficiency"] = unit.efficiency
    if is_io and unit.flows_hook is not None:
        ku["flows_hook"] = unit.flows_hook
    if not is_io and any(s.kind in (SpecKind.CONDENSER_DUTY, SpecKind.REBOILER_DUTY)
                         for s in unit.specs):
        raise FlowsheetError(
            f"'{unit.id}' has a duty spec but runs the Bubble-Point solver, "
            "which is constant-molar-overflow and reports no duties. Switch "
            "this column to Inside-Out, or drop the duty spec.")
    if unit.sections:
        solver = make_side_solver(solver, unit.sections,
                                  lambda si: build_si(si.R, si.D))

    fixed_R = 0.0 if unit.condenser == "none" else None
    n = [0]

    def trial(R, D):
        n[0] += 1
        prof = solver(build_si(R, D), cancel=cancel,
                      x0=warm.get("x"), T0=warm.get("T"), **ku)
        if prof.get("message") == "Aborted.":
            raise _Aborted()
        if float(prof.get("residual", np.inf)) < 1e-2:
            warm["x"], warm["T"] = prof["x"], prof["T"]
        return prof

    from core.operating_specs import resolve_operating_point
    R0, D0 = op.get(unit.id, (2.0, None))
    ops = [s for s in unit.specs if s.kind in OPERATING_KINDS]
    try:
        R, D = resolve_operating_point(
            ops, F_in, z_in, solve_fn=trial, lk=unit.lk, hk=unit.hk,
            R0=R0, D0=D0, side_draw_total=W, fixed_R=fixed_R)
    except _Aborted:
        raise
    except ValueError as exc:
        # Unwrapped, this reads as a whole-flowsheet failure and names no column.
        raise FlowsheetError(
            f"'{unit.id}' could not meet its operating specs at an inlet of "
            f"{F_in:.4g} (feed + recycle): {exc}") from exc
    op[unit.id] = (float(R), float(D))

    si = build_si(float(R), float(D))
    prof = solver(si, cancel=cancel,
                  x0=warm.get("x"), T0=warm.get("T"), **ku)
    n[0] += 1
    aborted = prof.get("message") == "Aborted."
    if not aborted and float(prof.get("residual", np.inf)) < 1e-2:
        warm["x"], warm["T"] = prof["x"], prof["T"]
    return UnitResult(
        unit_id=unit.id, profile=prof,
        converged=bool(prof.get("converged", False)), aborted=aborted,
        R=float(R), D=float(D), solves=n[0],
        message=str(prof.get("message", "")), si=si)


class _Aborted(Exception):
    """Cancellation, raised out of a trial solve so the resolver's least-squares
    cannot keep root-finding on the half-solved profiles an abort returns."""


def solve_flowsheet(fs: Flowsheet, *, tol: float = 1e-5, max_passes: int = 40,
                    knobs: Optional[dict] = None, report: Optional[Callable] = None,
                    cancel: Optional[Callable] = None) -> FlowsheetResult:
    """Solve every column, converging the recycle tears.

    `tol` is auto-relaxed to 1e-4 for a component containing an implicit spec
    (a purity, a recovery, a duty). Those are hit by a least-squares that stops
    on a 1e-4 fractional residual, so the outer map carries ~1e-4 of resolver
    noise and a tighter outer tolerance just chases it to `max_passes`. The
    value actually used is reported as `tol_used`, so it is a stated choice
    rather than a hidden fudge.
    """
    problems = validate(fs)
    if problems:
        raise FlowsheetError("This flowsheet cannot be solved as posed:\n  - "
                             + "\n  - ".join(problems))

    C = fs.n_comps
    feed_totals = np.zeros(C)
    for u in fs.units.values():
        for _s, flow, z, _q in u.feeds:
            feed_totals += float(flow) * np.asarray(z, float)
    F_ref = float(feed_totals.sum()) or 1.0

    torn = set(tear_set(fs))
    streams = _linear_seed(fs)
    for cid in torn:
        streams[cid].torn = True

    res = FlowsheetResult(streams=streams, tear_ids=sorted(torn),
                          feed_totals=feed_totals)
    warm: Dict[str, dict] = {u: {} for u in fs.units}
    op: Dict[str, Tuple[float, float]] = {}

    total = max(1, len(fs.units) * NOMINAL_PASSES)
    done = [0]

    def tick(residual: float) -> None:
        done[0] += 1
        if report is not None:
            report(min(done[0], total), total, float(residual))

    def push(uid: str, prof: dict) -> Dict[str, np.ndarray]:
        """Update this unit's outgoing streams; return the torn ones' new values
        instead of applying them, so the pass can Aitken the whole step at once."""
        pending = {}
        for c in fs.out_edges(uid):
            flow, comp = port_stream(fs.units[uid], c.port, prof)
            st = streams[c.id]
            new = StreamState(c.id, c.src, c.port, c.dst, c.stage,
                              flow * c.split_fraction, comp.copy(),
                              c.q if c.q is not None else natural_q(fs.units[uid], c.port),
                              st.torn)
            if c.id in torn:
                pending[c.id] = _tear_vec(new, F_ref)
            else:
                st.flow, st.comp, st.q = new.flow, new.comp, new.q
        return pending

    moved_overall = 0.0
    passes_overall = 0
    tol_overall = tol
    try:
        for comp in sccs(fs):
            members = set(comp)
            comp_torn = [c.id for c in fs._sorted()
                         if c.id in torn and c.src in members]
            order = _unit_order(fs, comp, torn)

            implicit = any(
                s.kind in (SpecKind.LK_RECOVERY, SpecKind.HK_RECOVERY,
                           SpecKind.DIST_PURITY, SpecKind.BOTTOMS_PURITY,
                           SpecKind.CONDENSER_DUTY, SpecKind.REBOILER_DUTY)
                for u in comp for s in fs.units[u].specs)
            tol_c = max(tol, 1e-4) if (implicit and comp_torn) else tol
            tol_overall = max(tol_overall, tol_c)

            if not comp_torn:                      # acyclic: one pass, no tear
                for uid in order:
                    if cancel is not None and cancel():
                        raise _Aborted()
                    ur = _solve_unit(fs, fs.units[uid], streams, warm[uid], op,
                                     knobs, cancel)
                    res.units[uid] = ur
                    if ur.aborted:
                        raise _Aborted()
                    push(uid, ur.profile)
                    tick(float(ur.profile.get("residual", 0.0)))
                passes_overall = max(passes_overall, 1)
                continue

            prev_step: Dict[str, np.ndarray] = {}
            moved = np.inf
            for p in range(max_passes):
                if cancel is not None and cancel():
                    raise _Aborted()
                pending: Dict[str, np.ndarray] = {}
                for uid in order:
                    ur = _solve_unit(fs, fs.units[uid], streams, warm[uid], op,
                                     knobs, cancel)
                    res.units[uid] = ur
                    if ur.aborted:
                        raise _Aborted()
                    pending.update(push(uid, ur.profile))
                    tick(float(ur.profile.get("residual", 0.0)))

                moved = 0.0
                for cid, new_v in pending.items():
                    st = streams[cid]
                    cur = _tear_vec(st, F_ref)
                    step = new_v - cur
                    # `moved` stays the RAW fixed-point step, never the jump, so
                    # the convergence test means what it says.
                    moved = max(moved, float(np.max(np.abs(step))))
                    _apply_vec(st, aitken_step(new_v, step, prev_step.get(cid),
                                               project=_project_tear), F_ref)
                    prev_step[cid] = step
                passes_overall = max(passes_overall, p + 1)
                if moved < tol_c:
                    break
            moved_overall = max(moved_overall, moved)
            if moved >= tol_c:
                res.tear_converged = False
    except _Aborted:
        res.aborted = True
        res.converged = False
        res.message = "Aborted."
        res.tear_residual = moved_overall
        res.tear_passes = passes_overall
        res.tol_used = tol_overall
        return res

    res.tear_residual = moved_overall
    res.tear_passes = passes_overall
    res.tol_used = tol_overall
    _net_flowsheet_report(fs, res, streams)

    bad = sorted(u for u, r in res.units.items() if not r.converged)
    res.converged = not bad and res.tear_converged
    if res.converged:
        res.message = "Converged."
    else:
        parts = []
        if bad:
            worst = ", ".join(
                f"{u} (residual {res.units[u].profile.get('residual', float('nan')):.2e})"
                for u in bad)
            parts.append(f"{'columns' if len(bad) > 1 else 'column'} {worst} "
                         "did not converge")
        if not res.tear_converged:
            parts.append(
                f"the recycle tear is still moving {moved_overall:.2e} after "
                f"{passes_overall} passes (torn: {', '.join(res.tear_ids)})")
        elif torn:
            parts.append(f"the recycle tear converged ({moved_overall:.2e})")
        res.message = "Flowsheet NOT converged: " + "; ".join(parts) + "."
        # Say it where a single-column display already looks, too.
        if not res.tear_converged:
            for r in res.units.values():
                r.profile["converged"] = False
                r.profile["message"] = (
                    f"{r.profile.get('message', 'Solved')}  [flowsheet recycle "
                    f"NOT converged: {moved_overall:.2e} after "
                    f"{passes_overall} passes]")
    return res


def _net_flowsheet_report(fs: Flowsheet, res: FlowsheetResult,
                          streams: Dict[str, StreamState]) -> None:
    """External feeds in, external products out — an inter-unit stream is
    neither. A port that sends 0.98 of itself onward still yields a 2% product:
    that is the purge."""
    products: List[dict] = []
    out = np.zeros(fs.n_comps)
    for uid in sorted(fs.units):
        ur = res.units.get(uid)
        if ur is None:
            continue
        unit = fs.units[uid]
        for key, port in sorted(unit.ports().items()):
            flow, comp = port_stream(unit, key, ur.profile)
            sent = sum(c.split_fraction for c in fs.out_edges(uid) if c.port == key)
            free = max(0.0, 1.0 - sent)
            if flow * free <= 1e-9:
                continue
            products.append({"unit": uid, "port": key, "label": port.label,
                             "flow": flow * free, "comp": comp,
                             "purge": sent > 0.0})
            out += flow * free * comp
    res.products = products
    res.closure = float(np.max(np.abs(res.feed_totals - out))) if fs.n_comps else 0.0


# --- self-check ------------------------------------------------------------

_ANTOINE = np.array([(6.90565, 1211.033, 220.79),      # benzene
                     (6.95464, 1344.8, 219.48),        # toluene
                     (6.99052, 1453.43, 215.31)])      # xylene
_COMPS = ["benzene", "toluene", "xylene"]
_Z = [0.5, 0.3, 0.2]
_F = 100.0


def _col(uid, *, feeds=(), draws=(), R=3.0, D=50.0, n=16, lk=0):
    return Unit(id=uid, n_stages=n, pressure=760.0, antoine=_ANTOINE, lk=lk,
                feeds=list(feeds), draws=list(draws),
                specs=[Spec(SpecKind.REFLUX_RATIO, R, "condenser"),
                       Spec(SpecKind.DISTILLATE_RATE, D, "column")])


def _series():
    """C1 splits benzene off; its bottoms feeds C2, which splits toluene/xylene."""
    return Flowsheet(
        units={"C1": _col("C1", feeds=[(8, _F, _Z, 1.0)], D=50.0),
               "C2": _col("C2", D=30.0, lk=1)},
        connections=[Connection("C1.B->C2@8", "C1", "B", "C2", 8)],
        comps=_COMPS, default_method="Bubble-Point")


def _recycle(split=0.90, D2=30.0):
    """As above, plus C2's distillate mostly recycled to C1; the rest purges."""
    fs = _series()
    fs.units["C2"] = _col("C2", D=D2, lk=1)
    fs.connections.append(
        Connection("C2.D->C1@8", "C2", "D", "C1", 8, split_fraction=split))
    return fs


def _demo():
    knobs = dict(max_iter=300, tol=1e-8)

    # --- acyclic: solved in dependency order, nothing torn --------------------
    fs = _series()
    assert [c for c in sccs(fs)] == [["C1"], ["C2"]], sccs(fs)
    assert tear_set(fs) == [], tear_set(fs)
    ser = solve_flowsheet(fs, knobs=knobs)
    assert ser.converged, ser.message
    assert ser.tear_ids == [] and ser.tear_passes == 1, ser.tear_ids
    assert np.allclose(ser.feed_totals, _F * np.array(_Z)), ser.feed_totals
    assert ser.closure < 1e-3, ser.closure
    # the internal C1->C2 stream is neither an external feed nor a product
    assert {(p["unit"], p["port"]) for p in ser.products} == {
        ("C1", "D"), ("C2", "D"), ("C2", "B")}, ser.products
    # and it carries what C1's reboiler made
    mid = ser.streams["C1.B->C2@8"]
    assert abs(mid.flow - ser.units["C1"].profile["B"]) < 1e-9, mid.flow
    assert mid.q == 1.0 and not mid.torn

    # --- recycle: one back edge torn, deterministically -----------------------
    fs = _recycle()
    assert sccs(fs) == [["C1", "C2"]], sccs(fs)
    assert tear_set(fs) == ["C2.D->C1@8"], tear_set(fs)
    rec = solve_flowsheet(fs, tol=1e-5, max_passes=40, knobs=knobs)
    assert rec.converged, rec.message
    assert rec.tear_converged and rec.tear_residual < 1e-5, rec.tear_residual
    assert rec.streams["C2.D->C1@8"].torn
    # the recycle is not external feed, and the purge IS an external product
    assert np.allclose(rec.feed_totals, _F * np.array(_Z)), rec.feed_totals
    assert rec.closure < 1e-3, rec.closure
    purge = next(p for p in rec.products if (p["unit"], p["port"]) == ("C2", "D"))
    assert purge["purge"] and abs(purge["flow"] - 0.10 * 30.0) < 1e-6, purge["flow"]
    # the recycle really loads C1 -- more traffic than the once-through case
    assert (rec.units["C1"].profile["liquid_flow"].max()
            > ser.units["C1"].profile["liquid_flow"].max())

    # --- an implicit spec inside a recycle: a root-find nested in a tear ------
    # The riskiest path in this module. Each pass re-resolves (R, D) against a
    # moving inlet, so it only settles because the resolver is warm-started and
    # the outer tolerance is relaxed above the resolver's own 1e-4 gate.
    imp = _recycle()
    imp.units["C1"].specs = [Spec(SpecKind.REFLUX_RATIO, 3.0, "condenser"),
                             Spec(SpecKind.DIST_PURITY, 0.99, "column", component=0)]
    ir = solve_flowsheet(imp, tol=1e-5, max_passes=40, knobs=knobs)
    assert ir.converged, ir.message
    assert ir.tol_used == 1e-4, ir.tol_used          # relaxed, and says so
    assert ir.units["C1"].profile["xD"][0] >= 0.99 - 1e-3, ir.units["C1"].profile["xD"]
    # warm starts pay off: the resolve costs many solves once, then the tear is cheap
    assert ir.tear_passes <= 6, ir.tear_passes
    assert ir.closure < 1e-2, ir.closure             # looser tol, looser closure

    # --- honesty: a starved tear must NOT claim convergence -------------------
    starved = solve_flowsheet(_recycle(), tol=1e-14, max_passes=2, knobs=knobs)
    assert all(u.converged for u in starved.units.values()), starved.message
    assert not starved.tear_converged and not starved.converged, starved.message
    assert "tear" in starved.message and "NOT converged" in starved.message
    # and the unit-level display says so too, not just the flowsheet summary
    assert all("NOT converged" in u.profile["message"]
               for u in starved.units.values()), starved.units["C1"].profile

    # --- a failed unit and a failed tear are different sentences --------------
    stalled = solve_flowsheet(_recycle(), tol=1e-5, max_passes=40,
                              knobs=dict(max_iter=2, tol=1e-8))
    assert not stalled.converged
    assert "did not converge" in stalled.message, stalled.message

    # --- cancellation is aborted, not converged ------------------------------
    ab = solve_flowsheet(_recycle(), knobs=knobs, cancel=lambda: True)
    assert ab.aborted and not ab.converged and ab.message == "Aborted."

    # --- progress is monotonic and bounded -----------------------------------
    ticks = []
    solve_flowsheet(_recycle(), knobs=knobs,
                    report=lambda d, t, r: ticks.append((d, t)))
    assert ticks and all(a[0] <= b[0] for a, b in zip(ticks, ticks[1:])), ticks[:9]
    assert all(d <= t for d, t in ticks), ticks[:9]

    # --- an unfed recycle loop is refused, not silently zero -----------------
    orphan = _recycle()
    orphan.units["C1"] = _col("C1", D=50.0)          # no external feed anywhere
    for bad, why in ((orphan, "no external feed"),
                     (_bad_stage(), "interior tray"),
                     (_bad_split(), "of one stream"),
                     (_self_loop(), "cannot feed itself")):
        try:
            solve_flowsheet(bad, knobs=knobs)
        except FlowsheetError as exc:
            assert why in str(exc), (why, str(exc))
        else:
            raise AssertionError(f"expected a FlowsheetError mentioning {why!r}")

    # --- cold start: a recycle larger than the external feed still solves -----
    # The general form of side_sections' "bottoms rate B=-5 must be positive".
    big = solve_flowsheet(_recycle(split=0.98, D2=140.0), knobs=knobs,
                          max_passes=60)
    assert big.converged, big.message
    assert big.streams["C2.D->C1@8"].flow > _F, big.streams["C2.D->C1@8"].flow

    # --- layout is deterministic and centered --------------------------------
    pos = auto_layout(_series())
    assert pos["C1"][0] < pos["C2"][0], pos
    assert abs(pos["C1"][0] + pos["C2"][0]) < 1e-9, pos      # centered on x
    assert auto_layout(_series()) == pos                      # deterministic

    print("flowsheet self-check OK")


def _bad_stage():
    fs = _series()
    fs.connections = [Connection("bad", "C1", "B", "C2", 1)]     # condenser stage
    return fs


def _bad_split():
    fs = _series()
    fs.connections.append(Connection("x", "C1", "B", "C2", 9, split_fraction=0.5))
    return fs


def _self_loop():
    fs = _series()
    fs.connections.append(Connection("loop", "C1", "D", "C1", 5))
    return fs


if __name__ == "__main__":
    _demo()
