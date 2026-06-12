"""Basic physical relations for stellar remnants."""

import numpy as np

from popkin.config.controls_default import M_ch


def estimate_white_dwarf_radius(m):
    """Estimate the white-dwarf radius with a simple mass-radius relation."""
    return 0.0115 * np.sqrt((M_ch / m) ** (2 / 3) - (m / M_ch) ** (2 / 3))


__all__ = ["estimate_white_dwarf_radius"]
