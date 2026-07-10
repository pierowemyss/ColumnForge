#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical rigorous-rating solver input: a column as open per-stage arrays.

One Case projects to this (ADR-0001). Everything that varies down the column is a
length-N array, so multiple feeds, side-draws and inter-stage duties are just
non-zero entries rather than special cases (ADR-0003: "per-stage source term").
The arrays are flat numpy — the shape a future C/FORTRAN kernel consumes across
FFI. `gamma_fn` is the one non-marshallable escape hatch; stage 2 replaces it
with a model id + params so the whole input crosses the boundary.

Stages are numbered 1..N top->bottom (1 = condenser stage, N = reboiler stage);
arrays are 0-based, so stage s lives at index s-1.

This module defines the *shape* and the projection. The rigorous solvers in
core.column_solvers consume a SolverInput directly (their legacy scalar
signature survives as a thin shim that builds one).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SolverInput:
    """A rating Solver's whole problem, as open per-stage arrays + operating point."""

    n_stages: int
    comps: List[str]            # length C; ordering for every composition axis
    feed: np.ndarray            # (N, C) molar component feed at each stage
    liquid_draw: np.ndarray     # (N,) side-liquid draw  [mol/time]
    vapor_draw: np.ndarray      # (N,) side-vapour draw  [mol/time]
    duty: np.ndarray            # (N,) inter-stage heat  [energy/time]; CMO ignores it
    pressure: np.ndarray        # (N,) stage pressure
    antoine: np.ndarray         # (C, k) psat coefficients aligned to comps
    R: float                    # reflux ratio  (resolved operating point)
    D: float                    # distillate molar rate
    gamma_fn: Optional[Callable] = None   # (x, T) -> activity coeffs; None => ideal
    phi_fn: Optional[Callable] = None     # (y, T, P) -> vapour fugacity coeffs;
                                          # None => ideal gas (see srk_phi_fn)
    q: Optional[np.ndarray] = None        # (N,) feed thermal quality per stage;
                                          # 1 = sat. liquid (default), 0 = sat. vapour
    condenser: str = "total"    # "total" | "partial" | "none" — top boundary:
                                # total => liquid distillate at y[top]; partial =>
                                # top stage is the equilibrium drum, vapour
                                # distillate; none => no reflux (R must be 0)
    subcooling: float = 0.0     # reflux/distillate ΔT below the bubble point at
                                # the condenser (total condenser only); 0 = sat.
                                # liquid. Consumed by the energy-balance hook.
    # ponytail: gamma_fn is a closure now; stage 2 swaps it for (model_id, params)
    # so the entire input marshals to C/FORTRAN. The arrays above already do.

    def __post_init__(self):
        if self.q is None:
            self.q = np.ones(self.n_stages)
        if self.condenser == "none" and abs(self.R) > 1e-12:
            raise ValueError("a column without a condenser has no reflux; R must be 0")

    @property
    def n_comps(self) -> int:
        return len(self.comps)

    @property
    def total_feed(self) -> float:
        """Total molar feed summed over stages and components."""
        return float(self.feed.sum())

    def feed_stages(self) -> List[int]:
        """1-based stages carrying any feed (legacy single-feed = one entry)."""
        return [j + 1 for j in range(self.n_stages) if self.feed[j].sum() > 0.0]


def _check_stage(stage: int, n_stages: int) -> None:
    if not (1 <= stage <= n_stages):
        raise ValueError(f"stage {stage} out of range 1..{n_stages}")


def build_solver_input(
    *,
    n_stages: int,
    comps: Sequence[str],
    feeds: Sequence[Tuple[int, float, Sequence[float]]],
    R: float,
    D: float,
    pressure,
    antoine,
    draws: Sequence[Tuple[int, float, float]] = (),
    duties: Sequence[Tuple[int, float]] = (),
    gamma_fn: Optional[Callable] = None,
    phi_fn: Optional[Callable] = None,
    condenser: str = "total",
    subcooling: float = 0.0,
) -> SolverInput:
    """Scatter feeds / draws / duties onto the per-stage arrays.

    feeds:    (stage, total_flow, composition[C]) or (stage, flow, comp, q)
              -- composition sums to ~1; q is the feed thermal quality
              (1 = saturated liquid, the default; 0 = saturated vapour)
    draws:    (stage, liquid_rate, vapor_rate)
    duties:   (stage, heat)
    pressure: scalar (uniform) or length-N sequence.

    A single-element `feeds` reproduces the legacy single-feed column; more than
    one element is multi-feed, with no other change. Two feeds on one stage get
    a flow-weighted q.
    """
    N = int(n_stages)
    C = len(comps)
    feed = np.zeros((N, C))
    liquid_draw = np.zeros(N)
    vapor_draw = np.zeros(N)
    duty = np.zeros(N)
    qF = np.zeros(N)               # flow-weighted q accumulator per stage

    for f in feeds:
        stage, flow, comp = f[0], f[1], f[2]
        fq = float(f[3]) if len(f) > 3 else 1.0
        _check_stage(stage, N)
        z = np.asarray(comp, float)
        if z.shape != (C,):
            raise ValueError(
                f"feed at stage {stage}: composition length {z.shape} != ({C},)")
        if abs(z.sum() - 1.0) > 1e-3:
            raise ValueError(
                f"feed at stage {stage}: composition sums to {z.sum():.4f}, not 1")
        feed[stage - 1] += float(flow) * z
        qF[stage - 1] += fq * float(flow)

    for stage, liq, vap in draws:
        _check_stage(stage, N)
        liquid_draw[stage - 1] += float(liq)
        vapor_draw[stage - 1] += float(vap)

    for stage, q in duties:
        _check_stage(stage, N)
        duty[stage - 1] += float(q)

    P = np.asarray(pressure, float)
    if P.ndim == 0:
        P = np.full(N, float(P))
    elif P.shape != (N,):
        raise ValueError(f"pressure must be scalar or length {N}, got {P.shape}")

    Ftot = feed.sum(axis=1)
    q = np.where(Ftot > 0.0, np.divide(qF, Ftot, out=np.ones(N), where=Ftot > 0.0), 1.0)

    return SolverInput(
        n_stages=N, comps=list(comps), feed=feed,
        liquid_draw=liquid_draw, vapor_draw=vapor_draw, duty=duty,
        pressure=P, antoine=np.asarray(antoine, float),
        R=float(R), D=float(D), gamma_fn=gamma_fn, phi_fn=phi_fn, q=q,
        condenser=str(condenser).lower(), subcooling=float(subcooling),
    )


def _demo() -> None:
    comps = ["benzene", "toluene", "xylene"]
    antoine = np.array([(6.90565, 1211.033, 220.79),
                        (6.95464, 1344.8, 219.48),
                        (6.99052, 1453.43, 215.31)])
    z = [0.5, 0.3, 0.2]

    # --- single feed reproduces the legacy single-feed column ---
    si = build_solver_input(
        n_stages=20, comps=comps,
        feeds=[(10, 100.0, z)], R=3.0, D=40.0, pressure=760.0, antoine=antoine)
    assert si.feed_stages() == [10], si.feed_stages()
    assert abs(si.total_feed - 100.0) < 1e-9, si.total_feed
    assert np.allclose(si.feed[9], 100.0 * np.array(z)), si.feed[9]
    assert np.count_nonzero(si.feed.sum(axis=1)) == 1   # only one feed stage
    assert si.pressure.shape == (20,) and np.all(si.pressure == 760.0)

    # --- two feeds: just two non-zero rows, totals add, nothing else special ---
    si2 = build_solver_input(
        n_stages=20, comps=comps,
        feeds=[(8, 60.0, z), (12, 40.0, [0.1, 0.2, 0.7])],
        R=3.0, D=40.0, pressure=760.0, antoine=antoine)
    assert si2.feed_stages() == [8, 12], si2.feed_stages()
    assert abs(si2.total_feed - 100.0) < 1e-9, si2.total_feed
    assert np.allclose(si2.feed[7], 60.0 * np.array(z))
    assert np.allclose(si2.feed[11], 40.0 * np.array([0.1, 0.2, 0.7]))

    # --- draws and inter-stage duty scatter to their stages (reserved seam) ---
    si3 = build_solver_input(
        n_stages=20, comps=comps, feeds=[(10, 100.0, z)],
        draws=[(5, 7.0, 0.0)], duties=[(15, -1.0e5)],
        R=3.0, D=40.0, pressure=760.0, antoine=antoine)
    assert si3.liquid_draw[4] == 7.0 and si3.liquid_draw.sum() == 7.0
    assert si3.duty[14] == -1.0e5 and np.count_nonzero(si3.duty) == 1

    # --- feed quality: default 1, 4-tuple feeds set it, co-fed stages blend ---
    assert np.all(si2.q == 1.0)
    si_q = build_solver_input(
        n_stages=20, comps=comps,
        feeds=[(10, 60.0, z, 1.0), (10, 40.0, z, 0.5)],
        R=3.0, D=40.0, pressure=760.0, antoine=antoine)
    assert abs(si_q.q[9] - 0.8) < 1e-12, si_q.q[9]     # flow-weighted
    assert si_q.q[0] == 1.0                             # feedless stages default

    # --- condenser type is carried; "none" demands R = 0 ---
    assert si.condenser == "total"
    try:
        build_solver_input(n_stages=20, comps=comps, feeds=[(10, 100.0, z)],
                           R=3.0, D=40.0, pressure=760.0, antoine=antoine,
                           condenser="none")
    except ValueError:
        pass
    else:
        raise AssertionError("condenser='none' with R>0 should raise")

    # --- per-stage pressure profile passes; wrong length is rejected ---
    prof = np.linspace(760.0, 780.0, 20)
    assert np.allclose(
        build_solver_input(n_stages=20, comps=comps, feeds=[(10, 100.0, z)],
                           R=3.0, D=40.0, pressure=prof, antoine=antoine).pressure,
        prof)
    for bad in (
        lambda: build_solver_input(n_stages=20, comps=comps, feeds=[(10, 100.0, z)],
                                   R=3.0, D=40.0, pressure=[1, 2, 3], antoine=antoine),
        lambda: build_solver_input(n_stages=20, comps=comps, feeds=[(99, 100.0, z)],
                                   R=3.0, D=40.0, pressure=760.0, antoine=antoine),
        lambda: build_solver_input(n_stages=20, comps=comps, feeds=[(10, 100.0, [0.5, 0.3])],
                                   R=3.0, D=40.0, pressure=760.0, antoine=antoine),
        lambda: build_solver_input(n_stages=20, comps=comps, feeds=[(10, 100.0, [0.6, 0.3, 0.2])],
                                   R=3.0, D=40.0, pressure=760.0, antoine=antoine),  # sums to 1.1
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    print("solver_input self-check OK")


if __name__ == "__main__":
    _demo()
