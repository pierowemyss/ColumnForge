"""Theme loader: substitute palette tokens into app.qss.

`@name@` placeholders are replaced textually with palette.tokens() values, so
the .qss stays valid QSS a C++ build can consume after the identical
substitution step (no Python-only theming machinery).
"""
import os

from . import palette

_QSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.qss")


def set_state(widget, state):
    """Set a widget's dynamic `state` property (neutral/ok/warn/error) and
    re-polish so the central QSS restyles it. The one path for status colours."""
    widget.setProperty("state", state)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def load_theme():
    """The full app stylesheet with tokens substituted — apply app-wide."""
    with open(_QSS_PATH, "r", encoding="utf-8") as f:
        qss = f.read()
    for name, value in palette.tokens().items():
        qss = qss.replace(f"@{name}@", value)
    return qss


def _demo():
    css = load_theme()
    # every token substituted — no stray @name@ left, and a known colour landed.
    assert "@" not in css, "unsubstituted token in app.qss"
    assert palette.ACCENT in css and palette.WINDOW in css
    # portability check: the substituted string is plain QSS (re-usable raw).
    assert "QGroupBox" in css and "subTabBar" in css
    print(f"theme OK: {len(css)} chars, {len(palette.tokens())} tokens")


if __name__ == "__main__":
    _demo()
