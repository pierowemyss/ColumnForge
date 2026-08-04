"""Public API over the rectification-body core -- what the GUI calls.

    analyze(prob, provider, r, EF)          -> one operating point, with bodies
    reflux_band(prob, provider, EF)         -> (r_min, r_max) at that entrainer flow
    operating_region(prob, provider, grid)  -> the feasible (E/F, r) region

`analyze` carries everything the diagram needs: per section, its pinch points
with stability and eigenvectors, its rectification bodies, and which body pair is
active at each junction.

RBM reports where a design is FEASIBLE and what it costs. It does not report a
stage count -- a rectification body approximates the set a profile can reach, not
the profile itself, so there is no stage to count along it. Take the operating
point RBM finds and size the column there with `side_features.bvm`.
"""

from .driver import (
    PI_EF_MIN,
    PI_R_MIN,
    analyze,
    operating_region,
    r_max,
    r_min,
    reflux_band,
)

__all__ = [
    "analyze",
    "operating_region",
    "reflux_band",
    "r_min",
    "r_max",
    "PI_EF_MIN",
    "PI_R_MIN",
]
