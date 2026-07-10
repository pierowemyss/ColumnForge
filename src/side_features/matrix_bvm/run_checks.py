"""Run every module's self-check (`_demo`) in dependency order.

    python run_checks.py

Sets up the import path (the package dir + src/python for `core`) and runs the
per-kernel self-checks. The cross-cutting Section-15 validation lives in
tests/test_validation.py.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.normpath(os.path.join(_HERE, "..", "..", "python"))):
    if p not in sys.path:
        sys.path.insert(0, p)

_MODULES = [
    "thermo_adapter", "problem", "residual", "jacobian", "linsolve",
    "initializer", "newton", "continuation", "diagnostics", "api",
]


def main():
    import importlib
    for name in _MODULES:
        mod = importlib.import_module(name)
        mod._demo()
    print(f"\nAll {len(_MODULES)} matrix_bvm self-checks passed.")


if __name__ == "__main__":
    main()
