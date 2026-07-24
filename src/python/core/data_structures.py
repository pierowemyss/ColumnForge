#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Structures Module
Provides utility classes for ColumnForge

Author: Piero Wemyss
Created: Based on freeRCM data_structures.py
"""

from enum import Enum


class SolverMode(Enum):
    """Canonical solver-mode enum, shared by the GUI and the core solvers.

    String values so SolverMode("bvm") round-trips with the string form used
    in the persisted Case (gui/state/persistence.py) and selectColumnSolver.
    """
    BVM = "bvm"
    HYSIM = "hysim"   # Inside-Out; name kept because "hysim" is in saved .colx files


class dict2struct(dict):
    """
    Dictionary that allows attribute access to keys
    Similar to MATLAB structs
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
