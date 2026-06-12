"""Reusable binary-orbit physical relations."""

import numpy as np

from popkin.constants import G, M_sun, R_sun, day_per_year, period_to_sep, sec_per_year


def roche_lobe_radius_ratio(q):
    """Calculate the Roche-lobe radius ratio (R_rl / a).

    Args:
        q: Mass ratio.

    Returns:
        Roche-lobe radius normalized by orbital separation.
    """
    p = q ** (1 / 3)
    return 0.49 * p * p / (0.6 * p * p + np.log(1 + p))


def calculate_chirp_mass(mass1, mass2):
    """Calculate the chirp mass of a binary system."""
    return (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2


def calculate_binary_separation(period, mass1, mass2):
    """Calculate binary orbital separation from orbital period and masses."""
    period = period / day_per_year
    return period_to_sep * (period ** 2 * (mass1 + mass2)) ** (1 / 3)


def calculate_orbital_angular_momentum(m1, m2, tb, sep, ecc):
    """Calculate binary orbital angular momentum."""
    oorb = 2 * np.pi / (tb * 24 * 3600)
    jorb = m1 * m2 / (m1 + m2) * np.sqrt(1 - ecc ** 2) * sep * sep * oorb
    return jorb * M_sun * R_sun ** 2


def calculate_period_separation_constants():
    """Calculate period-separation conversion constants."""
    period_to_sep_constant = (
        G / 4 / np.pi ** 2 * M_sun * sec_per_year ** 2
    ) ** (1 / 3) / R_sun
    sep_to_period_constant = (
        4 * np.pi ** 2 / G * R_sun ** 3 / M_sun
    ) ** (1 / 2) / sec_per_year
    return period_to_sep_constant, sep_to_period_constant


__all__ = [
    "roche_lobe_radius_ratio",
    "calculate_chirp_mass",
    "calculate_binary_separation",
    "calculate_orbital_angular_momentum",
    "calculate_period_separation_constants",
]
