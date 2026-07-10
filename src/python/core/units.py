"""Display-only unit conversions (roadmap Month 6).

Internals stay in solver units — the bundled Antoine/PLXANT fit's temperature is
degC, the canonical column pressure is bar, molar flows are kmol/h, and duties
are kJ/h (kmol/h x J/mol). These helpers convert *at the display/export edge*
only; nothing here feeds a solver.

Each dimension has a `from_internal` map: label -> callable(value) that maps the
internal unit to that label's unit. `DisplayUnits` bundles one choice per
dimension (defaults = the internal units, so an untouched config is a no-op).
"""
from __future__ import annotations

from dataclasses import dataclass

# --- scalar conversions from the internal unit --------------------------------

TEMPERATURE = {                       # internal: degC
    "degC": lambda c: c,
    "K": lambda c: c + 273.15,
    "degF": lambda c: c * 9.0 / 5.0 + 32.0,
}

PRESSURE = {                          # internal: bar
    "bar": lambda b: b,
    "atm": lambda b: b / 1.01325,
    "kPa": lambda b: b * 100.0,
    "mmHg": lambda b: b * 750.061683,
}

DUTY = {                              # internal: kJ/h
    "kW": lambda kjh: kjh / 3600.0,
    "MW": lambda kjh: kjh / 3.6e6,
    "kJ/h": lambda kjh: kjh,
}

# Flow needs a molar mass to reach a mass basis, so it is a two-arg family.
FLOW = {                              # internal: kmol/h
    "kmol/h": lambda kmolh, mw=None: kmolh,
    "kg/h": lambda kmolh, mw: kmolh * mw,       # mw = mixture kg/kmol
}


@dataclass
class DisplayUnits:
    """One display-unit choice per dimension. Defaults are the internal units,
    so `DisplayUnits()` leaves every value unchanged."""
    temperature: str = "degC"
    pressure: str = "bar"
    flow: str = "kmol/h"
    duty: str = "kW"                  # kW is the long-standing display default

    def T(self, degC):
        return TEMPERATURE[self.temperature](degC)

    def P(self, bar):
        return PRESSURE[self.pressure](bar)

    def Q(self, kjh):
        return DUTY[self.duty](kjh)

    def F(self, kmolh, mw=None):
        fn = FLOW[self.flow]
        return fn(kmolh, mw) if self.flow == "kg/h" else fn(kmolh)

    # labels for headers/axes
    def t_label(self): return self.temperature
    def p_label(self): return self.pressure
    def q_label(self): return self.duty
    def f_label(self): return self.flow


def _demo():
    u = DisplayUnits()                # all internal -> identity
    assert u.T(25.0) == 25.0 and u.P(1.0) == 1.0
    assert abs(u.Q(3600.0) - 1.0) < 1e-12                 # kJ/h -> kW default
    assert u.F(10.0) == 10.0

    k = DisplayUnits(temperature="K", pressure="atm", flow="kg/h", duty="MW")
    assert abs(k.T(0.0) - 273.15) < 1e-9
    assert abs(k.T(100.0) - 373.15) < 1e-9
    assert abs(DisplayUnits(temperature="degF").T(100.0) - 212.0) < 1e-9
    assert abs(k.P(1.01325) - 1.0) < 1e-9                 # bar -> atm
    assert abs(DisplayUnits(pressure="kPa").P(1.0) - 100.0) < 1e-9
    assert abs(DisplayUnits(pressure="mmHg").P(1.01325) - 760.0) < 0.1
    assert abs(k.Q(3.6e6) - 1.0) < 1e-9                   # kJ/h -> MW
    assert abs(k.F(2.0, mw=46.07) - 92.14) < 1e-6         # kmol/h -> kg/h (ethanol)
    # round-trip sanity: converting then dividing by the factor recovers input
    assert abs(DisplayUnits(temperature="degF").T(37.0) - 98.6) < 1e-9
    print("units self-check OK")


if __name__ == "__main__":
    _demo()
