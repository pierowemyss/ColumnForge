"""The BVM module's reactive profile plot must show the reaction products.

`driver._restore_reactive` sets design["comps"] to the *transformed* component
list, which by construction drops the reaction's reference component — normally
the product. Plotting that list left MTBE off a reactive MTBE column's profile
entirely. The plot has to read design["physical"].
"""
import os
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "side_features", "bvm", "tests"))

# R = 3.0, not the 2.0 these used to run at: under the honest junction test this
# column is feasible only in a band (measured 2.5 <= R <= ~4.5, see
# BVM_REACTIVE_XFAIL in src/side_features/bvm/tests/test_validation.py). Nothing
# here is about where that band lies -- these are plotting tests and they need a
# design that exists.


@pytest.fixture(scope="module")
def mtbe_design():
    import test_validation as tv
    from side_features.bvm import driver
    from side_features.bvm.thermo_adapter import ColumnForgeThermo

    prob = tv.mtbe_problem()
    d = driver.size_column(prob, ColumnForgeThermo(tv.MTBE_ANTOINE), R=3.0)
    assert d["feasible"], [f.cls for f in d["findings"]]
    return d


def _plot(design):
    """Drive the widget's plot method on a bare figure (no full GUI needed)."""
    from matplotlib.figure import Figure
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from gui.modules.bvm_module import BVMModuleWidget

    inst = BVMModuleWidget.__new__(BVMModuleWidget)
    inst.figure = Figure()
    inst.canvas = type("_C", (), {"draw": lambda self: None})()
    inst._plot_full(design)
    return inst.figure


def test_the_reaction_product_is_absent_from_the_transformed_coordinates(mtbe_design):
    """The premise: this is why plotting design['comps'] was wrong."""
    assert "MTBE" not in mtbe_design["comps"]
    assert "MTBE" in mtbe_design["physical"]["comps"]


def test_reactive_profile_plots_the_product(mtbe_design):
    fig = _plot(mtbe_design)
    ax1 = fig.axes[0]
    labels = [l.get_label() for l in ax1.get_lines()
              if not l.get_label().startswith("_")]
    assert "MTBE" in labels, labels
    assert ax1.get_ylabel() == "Liquid x (physical)"


def test_reactive_profile_plots_the_extent(mtbe_design):
    """Extent belongs on the plot: it is the only thing that says a reaction
    happened at all."""
    fig = _plot(mtbe_design)
    assert len(fig.axes) == 3, "no twin axis for the extent profile"
    ext = [l for l in fig.axes[2].get_lines()]
    assert ext and np.asarray(ext[0].get_ydata())[-1] > 0.1


def test_product_actually_builds_up_down_the_column(mtbe_design):
    phys = mtbe_design["physical"]
    x = np.asarray(phys["x"])[:, phys["comps"].index("MTBE")]
    assert x[0] < 0.01 and x[-1] > 0.5, (x[0], x[-1])


def test_trace_components_are_not_drawn(mtbe_design):
    """The transform floors a stoichiometric coordinate at 1e-4; that floor must
    not become a flat noise line on the physical plot."""
    from gui.modules.bvm_module import BVMModuleWidget

    phys = dict(mtbe_design["physical"])
    x = np.asarray(phys["x"]).copy()
    x[:, phys["comps"].index("n-butane")] = 1e-4        # force one to the floor
    phys["x"] = x
    design = dict(mtbe_design, physical=phys)

    labels = [l.get_label() for l in _plot(design).axes[0].get_lines()
              if not l.get_label().startswith("_")]
    assert "n-butane" not in labels, labels
    assert "MTBE" in labels
    assert BVMModuleWidget._REACTIVE_TRACE > 1e-4      # above reactive._TRACE
