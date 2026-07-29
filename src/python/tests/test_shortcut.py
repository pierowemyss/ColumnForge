"""FUG shortcut design — the checks live in core.shortcut._demo (Fenske hand
calc, Underwood root, Gilliland limits, full-design consistency); this runs it
under pytest and pins one concrete Fenske value."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.shortcut as sc


def test_shortcut_demo():
    sc._demo()


def test_fenske_hand_value():
    # alpha=2.4, 95/5 keys both ends -> Nmin = ln(361)/ln(2.4) ~ 6.73 stages
    Nmin = sc.fenske_min_stages(np.array([2.4, 1.0]),
                                np.array([0.95, 0.05]),
                                np.array([0.05, 0.95]), 0, 1)
    assert abs(Nmin - 6.727) < 1e-2, Nmin
