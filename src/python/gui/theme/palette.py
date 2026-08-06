"""The single source of theme colours and metrics (portable seam).

Every colour the app draws — QSS tokens, QPainter canvas colours, matplotlib
rcParams — resolves from here. In a C++ Qt port this file becomes a header of
the same constants; the .qss is substituted with these values by the same step
`load_theme()` does in Python.

One dark palette today; a light palette is a second constants block wired into
TOKENS later (explicit non-goal now — the QSS is already structured for it).
"""

# --- surface hierarchy (darkest -> lightest) --------------------------------
TOOLBAR = "#111111"      # toolbar / deepest chrome
RAISED = "#1a1a1a"       # side nav, unselected tabs
WINDOW = "#2d2d2d"       # main window / panels / group boxes
FIELD = "#3a3a3a"        # input backgrounds (edits, combos, spin boxes)
HOVER = "#3d3d3d"        # hover fill on menus / rows
DIVIDER = "#333333"      # thin separators
BORDER = "#444444"       # panel / group-box / control borders

# --- text -------------------------------------------------------------------
TEXT = "#cccccc"
TEXT_MUTED = "#888888"
TEXT_BRIGHT = "#ffffff"
TEXT_HINT = "#999999"
TEXT_ON_LIGHT = "#212529"   # dark ink for text sitting on a light accent fill

# --- accent + semantics (match the DoF status light) ------------------------
ACCENT = "#0078d4"
OK = "#2f9e44"
WARN = "#fb8500"
ERROR = "#e03131"

# --- metrics ----------------------------------------------------------------
RADIUS = 4
CONTROL_HEIGHT = 26
FONT_SIZE = 13
HEADER_FONT_SIZE = 16
MARGIN = 10
SPACING = 8


class canvas:
    """QPainter colours for the column-diagram canvas (was hard-coded light)."""
    BG = WINDOW
    SHELL_STROKE = "#adb5bd"
    SHELL_FILL = FIELD
    TRAY = TEXT_MUTED
    COND_FILL = "#ffec99"     # pastel accents read on dark with dark ink
    REBO_FILL = "#ffc9c9"
    MODULE_FILL = "#a5d8ff"
    CHIP_BG = FIELD
    CHIP_TEXT = TEXT
    CHIP_TEXT_SELECTED = TEXT_ON_LIGHT
    LABEL = TEXT_MUTED
    FEED = "#4dabf7"          # lighter than the old #1971c2 for dark contrast
    PRODUCT = "#ff6b6b"
    INTERNAL = "#51cf66"
    # --- flowsheet canvas ---------------------------------------------------
    # A recycle needs its own HUE, not just a dash: dashed INTERNAL green reads
    # as "selected" against everything else on this palette.
    RECYCLE = "#e599f7"
    EDGE_INVALID = "#ff922b"  # a connection that no longer validates
    NODE_ACTIVE = "#4dabf7"   # border of the column being edited
    NODE_LABEL = TEXT_BRIGHT  # a column's name, always legible
    STAGE_LABEL = TEXT_MUTED  # stage numbers, only drawn when zoomed in


# Token table consumed by app.qss (@name@ -> value). Only colours/metrics —
# the .qss carries all the structure.
def tokens():
    t = {
        "toolbar": TOOLBAR, "raised": RAISED, "window": WINDOW, "field": FIELD,
        "hover": HOVER, "divider": DIVIDER, "border": BORDER,
        "text": TEXT, "text_muted": TEXT_MUTED, "text_bright": TEXT_BRIGHT,
        "text_hint": TEXT_HINT, "text_on_light": TEXT_ON_LIGHT,
        "accent": ACCENT, "ok": OK, "warn": WARN, "error": ERROR,
        "radius": f"{RADIUS}px", "control_h": f"{CONTROL_HEIGHT}px",
        "font_size": f"{FONT_SIZE}px", "header_font": f"{HEADER_FONT_SIZE}px",
        "margin": f"{MARGIN}px", "spacing": f"{SPACING}px",
    }
    return t
