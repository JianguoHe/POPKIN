"""Observable calculations for X-ray binaries and related accreting systems.

These helpers provide basic luminosity and candidate-selection utilities. They
are not a full X-ray-binary population model; future prescriptions can add
duty cycles, spectral states, beaming, and survey-specific selection effects.
"""

import numpy as np

from popkin.constants import c_light
from popkin.observables.survey_selection import luminosity_to_flux


def eddington_luminosity(mass):
    """Calculate the Eddington luminosity.

    Args:
        mass: Compact-object mass [M_sun].

    Returns:
        Eddington luminosity [erg/s].
    """
    mass = np.asarray(mass, dtype=float)
    luminosity = 1.26e38 * mass
    return luminosity.item() if luminosity.shape == () else luminosity


def accretion_luminosity(m_dot, efficiency=0.1):
    """Convert accretion rate to bolometric accretion luminosity.

    Args:
        m_dot: Accretion rate [g/s].
        efficiency: Radiative efficiency.

    Returns:
        Luminosity [erg/s].
    """
    m_dot = np.asarray(m_dot, dtype=float)
    luminosity = efficiency * m_dot * c_light ** 2
    return luminosity.item() if luminosity.shape == () else luminosity


def capped_accretion_luminosity(mass, m_dot, efficiency=0.1, eddington_factor=1.0):
    """Calculate accretion luminosity capped at a multiple of Eddington."""
    luminosity = np.asarray(accretion_luminosity(m_dot, efficiency=efficiency), dtype=float)
    cap = eddington_factor * np.asarray(eddington_luminosity(mass), dtype=float)
    result = np.minimum(luminosity, cap)
    return result.item() if result.shape == () else result


def xray_flux(luminosity, distance_kpc):
    """Convert X-ray luminosity to observed flux."""
    return luminosity_to_flux(luminosity, distance_kpc)


def select_xray_binary_candidates(data, compact_types=("NS", "BH")):
    """Select rows containing one compact accretor and one non-compact companion.

    Args:
        data: Structured array containing ``type1`` and ``type2`` fields.
        compact_types: Stellar type labels treated as compact accretors.

    Returns:
        Boolean mask with candidate X-ray binary rows.
    """
    type1 = data["type1"]
    type2 = data["type2"]
    compact1 = np.isin(type1, compact_types)
    compact2 = np.isin(type2, compact_types)
    return compact1 ^ compact2


__all__ = [
    "eddington_luminosity",
    "accretion_luminosity",
    "capped_accretion_luminosity",
    "xray_flux",
    "select_xray_binary_candidates",
]
