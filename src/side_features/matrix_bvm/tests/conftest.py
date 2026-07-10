"""Put the matrix_bvm package dir and src/python (for `core`) on the path."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.normpath(os.path.join(_HERE, ".."))
_SRC_PY = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "python"))
for p in (_PKG, _SRC_PY):
    if p not in sys.path:
        sys.path.insert(0, p)
