"""Month-5 energy-balance features land their checks in the module _demo()s
(subcooling + duty splits in column_solvers, duty-spec inversion in
operating_specs, enthalpy feed-q in window_state). CI runs pytest, not the
`__main__` self-checks, so this just invokes them so a regression fails here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.column_solvers as cs
import core.operating_specs as ops
import core.enthalpy as enth
import gui.state.window_state as ws


def test_column_solvers_demo():
    cs._demo()          # energy balance, subcooling, CMO flows


def test_operating_specs_demo():
    ops._demo()         # algebraic + implicit + duty-spec inversion


def test_enthalpy_demo():
    enth._demo()        # Watson latent + hL/hV closures


def test_window_state_demo():
    ws._demo()          # enthalpy-based feed quality (subcooled/superheated)
