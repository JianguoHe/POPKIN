"""Microlensing observables for isolated compact objects.

Planned responsibilities include Einstein radii, event time-scales,
photometric magnification, astrometric shifts, and survey selection functions
for black holes, neutron stars, white dwarfs, and other compact remnants.
"""

import numpy as np


def estimate_bh_lens_fraction_by_timescale(t_e):
    """Estimate the black-hole lens fraction from microlensing event timescale.

    Args:
        t_e: Event timescale in days.

    Returns:
        Estimated black-hole lens fraction for each input timescale.
    """
    t_e = np.asarray(t_e, dtype=np.float64)

    conditions = [
        t_e < 30,
        (t_e >= 30) & (t_e < 68.36949708649836),
        (t_e >= 68.36949708649836) & (t_e < 146.26405672325487),
        (t_e >= 146.26405672325487) & (t_e <= 320),
        t_e > 320,
    ]

    slope1 = (0.082625 - 0.000908) / (68.36949708649836 - 30)
    slope2 = (0.311469 - 0.082625) / (146.26405672325487 - 68.36949708649836)
    slope3 = (0.494762 - 0.311469) / (320 - 146.26405672325487)

    functions = [
        lambda x: 0.0,
        lambda x: 0.000908 + (x - 30) * slope1,
        lambda x: 0.082625 + (x - 68.36949708649836) * slope2,
        lambda x: 0.311469 + (x - 146.26405672325487) * slope3,
        lambda x: 0.494762,
    ]

    return np.piecewise(t_e, conditions, functions)


__all__ = ["estimate_bh_lens_fraction_by_timescale"]
