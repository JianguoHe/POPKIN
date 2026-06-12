"""Experimental Andromeda (M31) galaxy helpers.

This module is not yet a full galaxy model for POPKIN population synthesis.
It currently provides simple M31 star-formation and metallicity-history helper
fits, but it is not a drop-in replacement for ``MilkyWay``.
"""

from __future__ import annotations

import warnings

import numpy as np


class Andromeda:
    """Experimental Andromeda galaxy model.

    Status:
        Experimental and incomplete. The model is not integrated with the
        population-synthesis drivers and does not yet implement the same
        sampling interface as :class:`popkin.galaxies.MilkyWay`.

    Attributes:
        dist: Distance to M31.
    """

    status = "experimental"

    def __init__(
            self,
            dist: float = 0.02,
    ):
        """Initialize the M31 galaxy model.

        Args:
            dist: Distance to M31.
        """
        self.dist = dist
        warnings.warn(
            "Andromeda is experimental and incomplete. It currently provides "
            "helper fits only and is not a drop-in replacement for MilkyWay.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _init_metallicity_grid(self) -> np.ndarray:
        """Initialize the M31 metallicity grid.

        Raises:
            NotImplementedError: Always, because the M31 metallicity grid has
                not been finalized.
        """
        raise NotImplementedError(
            "The M31 metallicity grid is not implemented yet. "
            "Use MilkyWay for production population-synthesis runs."
        )

    def star_formation_rate(self, time: float) -> float:
        """Estimate the M31 star-formation rate at a given time.

        Args:
            time: Lookback time [Myr].

        Returns:
            Star-formation rate [systems / Myr].
        """
        times = np.array(
            [0, 6.1e3, 7.7e3, 9e3, 1e4, 1.08e4, 1.15e4, 1.2e4, 1.24e4, 1.27e4, 1.3e4, 1.321e4, 1.337e4, 1.35e4, 1.36e4,
             1.4e4])
        sfr_values = np.array([18, 3.9, 0.36, 4.8, 5.7, 1.65, 18.6, 4.8, 9, 5.4, 4.8, 2.94, 2.16, 0.81, 0.39]) * 1e6
        if time < times[0] or time >= times[-1]:
            raise ValueError(
                f"M31 star-formation fit is defined for {times[0]} <= time < {times[-1]} Myr; got {time}"
            )
        index = np.searchsorted(times, time, side='right') - 1
        sfr = sfr_values[index]
        return sfr  # unit: /Myr

    def history_time_from_metallicity(self, z: float) -> float:
        """Estimate the M31 historical time corresponding to a metallicity."""
        if 0.00022233 <= z <= 0.00813205:
            return 252.853507 * z - 0.05621717
        elif 0.00813205 < z <= 0.01999862:
            return 674.1628264 * z - 3.4823253
        elif 0.01999862 < z <= 0.0223795:
            return 2100.0799346 * z - 31.9986978
        raise ValueError(
            "M31 metallicity-history fit is defined for "
            f"0.00022233 <= z <= 0.0223795; got {z}"
        )
