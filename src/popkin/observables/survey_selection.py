"""Shared survey-selection utilities for observable catalogs.

The functions here are deliberately simple building blocks. More detailed
survey models can layer sky masks, cadence, extinction, and completeness
prescriptions on top of them.
"""

import numpy as np

from popkin.constants import kpc


def luminosity_to_flux(luminosity, distance_kpc):
    """Convert isotropic luminosity to flux.

    Args:
        luminosity: Source luminosity [erg/s].
        distance_kpc: Source distance [kpc].

    Returns:
        Flux [erg/s/cm^2].
    """
    luminosity_arr, distance_arr = np.broadcast_arrays(
        np.asarray(luminosity, dtype=float),
        np.asarray(distance_kpc, dtype=float),
    )
    flux = luminosity_arr / (4 * np.pi * (distance_arr * kpc) ** 2)
    return flux.item() if flux.shape == () else flux


def flux_to_luminosity(flux, distance_kpc):
    """Convert flux to isotropic luminosity.

    Args:
        flux: Source flux [erg/s/cm^2].
        distance_kpc: Source distance [kpc].

    Returns:
        Luminosity [erg/s].
    """
    flux_arr, distance_arr = np.broadcast_arrays(
        np.asarray(flux, dtype=float),
        np.asarray(distance_kpc, dtype=float),
    )
    luminosity = flux_arr * 4 * np.pi * (distance_arr * kpc) ** 2
    return luminosity.item() if luminosity.shape == () else luminosity


def flux_limited_mask(luminosity, distance_kpc, flux_limit):
    """Return a detection mask for a simple flux-limited survey."""
    return luminosity_to_flux(luminosity, distance_kpc) >= flux_limit


def threshold_mask(values, threshold, mode="above"):
    """Return a mask for a one-dimensional selection threshold.

    Args:
        values: Quantity to threshold.
        threshold: Threshold value.
        mode: ``"above"`` for values >= threshold or ``"below"`` for values <= threshold.
    """
    values = np.asarray(values)
    if mode == "above":
        return values >= threshold
    if mode == "below":
        return values <= threshold

    raise ValueError("Unsupported threshold mode. Expected one of: 'above', 'below'.")


__all__ = [
    "luminosity_to_flux",
    "flux_to_luminosity",
    "flux_limited_mask",
    "threshold_mask",
]
