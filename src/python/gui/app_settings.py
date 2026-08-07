"""Application preferences — the ones that belong to the install, not the case.

A `.colx` describes a column; it must not also decide which parts of the UI you
can see. So this lives in QSettings, next to window geometry and recent files,
and a file written with a beta feature on still opens for someone who has it
off.

Kept in its own module rather than on WindowState so a test can flip a value
without a Case, and so `core` never sees it at all.
"""

import os

from PySide6.QtCore import QSettings

from core.units import DUTY, FLOW, TEMPERATURE, DisplayUnits

ORG = "ColumnForge"
APP = "ColumnForge"

#: Beta features are built and tested but not settled enough to be the default
#: experience. Today: the multi-column flowsheet editor. The solver underneath
#: it (`core/flowsheet.py`) is always active — a single column is a one-unit
#: flowsheet — so turning this off changes what is on screen, never what is
#: computed.
_BETA_KEY = "features/beta"


#: Route the activity-coefficient models through the compiled Fortran kernel
#: (`src/native/nifco2.f90`, built by `make -C src/native`). Same equations as
#: the NumPy paths and bit-identical results, so this changes speed, not
#: answers — and it stays off unless the library was built *and* asked for.
_NIFCO_KEY = "features/nifco"


def _settings() -> QSettings:
    return QSettings(ORG, APP)


def beta_enabled() -> bool:
    return _settings().value(_BETA_KEY, False, type=bool)


def set_beta_enabled(on: bool) -> None:
    s = _settings()
    s.setValue(_BETA_KEY, bool(on))
    s.sync()


def nifco_enabled() -> bool:
    return _settings().value(_NIFCO_KEY, False, type=bool)


def set_nifco_enabled(on: bool) -> None:
    s = _settings()
    s.setValue(_NIFCO_KEY, bool(on))
    s.sync()


def apply_nifco() -> bool:
    """Push the stored preference into `core.thermodynamics`; returns what
    actually took effect (False if no library is installed). Called at startup
    and whenever Preferences is accepted — the setting lives here, the switch
    lives in `core`, and this is the one place the two are joined."""
    from core.thermodynamics import set_native
    return set_native(nifco_enabled())


#: Display units a *new* case starts with. A `.colx` carries its own choice and
#: overrides these on load — this only seeds File -> New. Pressure is left out
#: on purpose: `DisplayUnits.pressure` reaches no widget today, and an enterable
#: value nothing consumes is the thing this project does not ship.
_UNIT_CHOICES = {"temperature": TEMPERATURE, "flow": FLOW, "duty": DUTY}


def default_units() -> DisplayUnits:
    s = _settings()
    fallback = DisplayUnits()
    picked = {}
    for name, choices in _UNIT_CHOICES.items():
        val = s.value(f"units/{name}", None, type=str)
        picked[name] = val if val in choices else getattr(fallback, name)
    return DisplayUnits(**picked)


def set_default_units(units: DisplayUnits) -> None:
    s = _settings()
    for name in _UNIT_CHOICES:
        s.setValue(f"units/{name}", getattr(units, name))
    s.sync()


_LOG_KEY = "log/level"


def log_level() -> str:
    val = _settings().value(_LOG_KEY, "INFO", type=str)
    return val if val in ("INFO", "DEBUG") else "INFO"


def set_log_level(level: str) -> None:
    s = _settings()
    s.setValue(_LOG_KEY, level)
    s.sync()


def log_dir() -> str:
    """~/.columnforge — the rotating log lives here, created on demand."""
    path = os.path.join(os.path.expanduser("~"), ".columnforge")
    os.makedirs(path, exist_ok=True)
    return path


def _demo():
    """Round-trip through a throwaway QSettings scope."""
    global ORG
    org, ORG = ORG, "ColumnForgeTest"
    try:
        _settings().clear()
        assert default_units() == DisplayUnits()          # untouched -> defaults
        set_default_units(DisplayUnits(temperature="K", flow="kg/h", duty="MW"))
        u = default_units()
        assert (u.temperature, u.flow, u.duty) == ("K", "kg/h", "MW")
        assert u.pressure == DisplayUnits().pressure      # not a stored dimension

        _settings().setValue("units/temperature", "furlongs")
        assert default_units().temperature == "degC"      # garbage -> fallback

        assert log_level() == "INFO"
        set_log_level("DEBUG")
        assert log_level() == "DEBUG"

        assert nifco_enabled() is False                   # never on by default
        set_nifco_enabled(True)
        assert nifco_enabled() is True
        from core.nifco import available
        assert apply_nifco() is available()                # off if never built
        set_nifco_enabled(False)
        assert apply_nifco() is False
        _settings().clear()
        print("app_settings self-check OK")
    finally:
        ORG = org


if __name__ == "__main__":
    _demo()
