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

This module defines the *shape* and the projection only. The rigorous solvers
still take their legacy scalar args until they are migrated to consume SolverInput
(build order step 4); nothing here changes solver behaviour yet.
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
    # ponytail: gamma_fn is a closure now; stage 2 swaps it for (model_id, params)
    # so the entire input marshals to C/FORTRAN. The arrays above already do.

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
) -> SolverInput:
    """Scatter feeds / draws / duties onto the per-stage arrays.

    feeds:    (stage, total_flow, composition[C])  -- composition sums to ~1
    draws:    (stage, liquid_rate, vapor_rate)
    duties:   (stage, heat)
    pressure: scalar (uniform) or length-N sequence.

    A single-element `feeds` reproduces the legacy single-feed column; more than
    one element is multi-feed, with no other change.
    """
    N = int(n_stages)
    C = len(comps)
    feed = np.zeros((N, C))
    liquid_draw = np.zeros(N)
    vapor_draw = np.zeros(N)
    duty = np.zeros(N)

    for stage, flow, comp in feeds:
        _check_stage(stage, N)
        z = np.asarray(comp, float)
        if z.shape != (C,):
            raise ValueError(
                f"feed at stage {stage}: composition length {z.shape} != ({C},)")
        if abs(z.sum() - 1.0) > 1e-3:
            raise ValueError(
                f"feed at stage {stage}: composition sums to {z.sum():.4f}, not 1")
        feed[stage - 1] += float(flow) * z

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

    return SolverInput(
        n_stages=N, comps=list(comps), feed=feed,
        liquid_draw=liquid_draw, vapor_draw=vapor_draw, duty=duty,
        pressure=P, antoine=np.asarray(antoine, float),
        R=float(R), D=float(D), gamma_fn=gamma_fn,
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
