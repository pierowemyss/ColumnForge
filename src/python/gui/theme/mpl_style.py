"""Matplotlib rcParams from the palette — the one deliberately non-portable
piece of the theme (a C++ port swaps the charting layer, but these colour
constants carry over from palette.py).

apply() sets global rcParams once at startup; every figure created afterwards
(results tab, ternary view, module plots) picks up the dark styling.
"""
from . import palette


def apply():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": palette.WINDOW,
        "axes.facecolor": palette.RAISED,
        "axes.edgecolor": palette.BORDER,
        "axes.labelcolor": palette.TEXT,
        "axes.titlecolor": palette.TEXT,
        "text.color": palette.TEXT,
        "xtick.color": palette.TEXT_MUTED,
        "ytick.color": palette.TEXT_MUTED,
        "grid.color": palette.DIVIDER,
        "grid.alpha": 0.4,
        "legend.facecolor": palette.FIELD,
        "legend.edgecolor": palette.BORDER,
        "savefig.facecolor": palette.WINDOW,
        "axes.prop_cycle": mpl.cycler(color=[
            palette.canvas.FEED, palette.canvas.PRODUCT, palette.canvas.INTERNAL,
            palette.WARN, "#cc5de8", "#ffd43b", palette.ACCENT, "#ff8787",
        ]),
    })


def _demo():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import to_rgba
    apply()
    # rcParams keeps the raw value; normalise both through to_rgba to compare.
    assert to_rgba(matplotlib.rcParams["figure.facecolor"]) == to_rgba(palette.WINDOW)
    assert to_rgba(matplotlib.rcParams["axes.facecolor"]) == to_rgba(palette.RAISED)
    print("mpl_style OK")


if __name__ == "__main__":
    _demo()
