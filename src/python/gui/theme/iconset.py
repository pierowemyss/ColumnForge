"""Monochrome icon set for the toolbar/buttons.

Lucide-style single-stroke 24x24 glyphs. The source SVGs (stroke
`currentColor`) are written to gui/theme/icons/ on first use and a dark-theme
tinted copy to icons/tinted/ — file-based so a C++ build recolours the same way
(no runtime SVG generation beyond the one textual colour swap the loader does).

ponytail: the glyph paths live here as strings and are materialised to .svg on
demand; swap to hand-vendored Lucide/Tabler files by dropping them in icons/
and deleting the matching _SVG entry.
"""
import os

from PySide6.QtGui import QIcon

from . import palette

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
_TINT_DIR = os.path.join(_DIR, "tinted")

# name -> inner SVG (24x24, stroke=currentColor set by the wrapper). `fill`
# lets the play triangle be solid.
_SVG = {
    "new": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'
           '<path d="M14 3v5h5"/>',
    "open": '<path d="M4 5h5l2 2h9v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1z"/>',
    "save": '<path d="M5 3h11l3 3v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/>'
            '<path d="M7 3v5h8"/><rect x="8" y="13" width="8" height="6"/>',
    "run": '<path d="M6 4l14 8-14 8z" fill="currentColor" stroke="none"/>',
    "abort": '<rect x="6" y="6" width="12" height="12" rx="1"/>',
    "export": '<path d="M12 15V4"/><path d="M7 9l5-5 5 5"/>'
              '<path d="M5 17v2a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2"/>',
    "add": '<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>',
    "delete": '<path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M6 7l1 13h10l1-13"/>',
    "search": '<circle cx="11" cy="11" r="6"/><line x1="20" y1="20" x2="16" y2="16"/>',
    "settings": '<line x1="4" y1="8" x2="20" y2="8"/><circle cx="9" cy="8" r="2"/>'
                '<line x1="4" y1="16" x2="20" y2="16"/><circle cx="15" cy="16" r="2"/>',
    "plot": '<path d="M4 4v16h16"/><path d="M7 14l4-5 3 3 4-6"/>',
    "table": '<rect x="4" y="4" width="16" height="16" rx="1"/>'
             '<line x1="4" y1="10" x2="20" y2="10"/><line x1="10" y1="4" x2="10" y2="20"/>',
}

_CACHE = {}


def _wrap(inner):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
        'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f'{inner}</svg>'
    )


def _materialise(name):
    """Write source + tinted SVG for `name`, return the tinted path."""
    os.makedirs(_TINT_DIR, exist_ok=True)
    src = os.path.join(_DIR, f"{name}.svg")
    body = _wrap(_SVG[name])
    if not os.path.exists(src):
        with open(src, "w", encoding="utf-8") as f:
            f.write(body)
    tint = os.path.join(_TINT_DIR, f"{name}.svg")
    with open(tint, "w", encoding="utf-8") as f:
        f.write(body.replace("currentColor", palette.TEXT))
    return tint


def icon(name):
    """QIcon for a named glyph, tinted to the current text colour."""
    if name not in _SVG:
        raise KeyError(f"unknown icon {name!r}; have {sorted(_SVG)}")
    if name not in _CACHE:
        _CACHE[name] = QIcon(_materialise(name))
    return _CACHE[name]


def _demo():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    for n in _SVG:
        ic = icon(n)
        assert not ic.isNull(), n
        assert os.path.exists(os.path.join(_TINT_DIR, f"{n}.svg"))
    print(f"iconset OK: {len(_SVG)} glyphs")


if __name__ == "__main__":
    _demo()
