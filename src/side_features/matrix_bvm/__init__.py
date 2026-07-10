"""Matrix BVM — universal feasibility solver + MESH-initialization framework.

A self-contained side module (see MatBVM_blueprint.md). It builds the full
Naphtali-Sandholm residual system R(U)=0 in component-flow variables
{l_ij, v_ij, T_i} (2C+1 per stage), a structured initial guess U0, and offers
a damped-Newton / continuation solve on top. Thermodynamics is consumed from
FreeColumn's `core` layer through the ThermoProvider adapter only — this module
never reimplements VLE/enthalpy.

Package layout mirrors the goal's ten units:
    problem        topology + specs + square DOF ledger
    thermo_adapter ThermoProvider interface + FreeColumn wrapper
    residual       R(U) assembly (pure array kernels)
    jacobian       block-tridiagonal A_i, B_i, C_i (analytic / complex-step)
    linsolve       block-Thomas tridiagonal sweep
    initializer    U0: FUG shortcut -> CMO flows -> bubble-T -> component flows
    newton         damped Newton with Armijo backtracking + bounds
    continuation   homotopy (ideal->real) + pseudo-transient fallback
    diagnostics    feasibility classification (returns class + stages)
    api            assess_feasibility / initialize / converge

The kernels are pure functions over NumPy arrays with explicit shapes so they
port to C later; no Python objects live in the hot Newton loop.
"""

# Bootstrap: this module lives under src/side_features but imports FreeColumn's
# thermo as `core.*` (the repo's absolute cross-package convention, resolved by
# launch.py adding src/python to the path). Ensure that path is present so the
# package is importable on its own for tests/self-checks.
import os as _os
import sys as _sys

_SRC_PY = _os.path.normpath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "python"))
if _SRC_PY not in _sys.path:
    _sys.path.insert(0, _SRC_PY)

__all__ = [
    "problem", "thermo_adapter", "residual", "jacobian", "linsolve",
    "initializer", "newton", "continuation", "diagnostics", "api",
]
