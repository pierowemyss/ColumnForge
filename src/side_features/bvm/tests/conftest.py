"""Put `src` (for side_features.bvm) and `src/python` (for core/gui) on the path."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))          # src
_SRC_PY = os.path.join(_SRC, "python")                                  # src/python
for p in (_SRC, _SRC_PY):
    if p not in sys.path:
        sys.path.insert(0, p)
