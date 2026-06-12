"""Reusable physical relations used across POPKIN."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .accretion import (
        m_dot_bh_hot_accretion,
        m_dot_bondi_hoyle,
        m_dot_edd,
    )
    from .binary_orbits import (
        calculate_binary_separation,
        calculate_chirp_mass,
        calculate_orbital_angular_momentum,
        calculate_period_separation_constants,
        roche_lobe_radius_ratio,
    )
    from .stellar_remnants import estimate_white_dwarf_radius

__all__ = [
    "m_dot_edd",
    "m_dot_bondi_hoyle",
    "m_dot_bh_hot_accretion",
    "roche_lobe_radius_ratio",
    "calculate_chirp_mass",
    "calculate_binary_separation",
    "calculate_orbital_angular_momentum",
    "calculate_period_separation_constants",
    "estimate_white_dwarf_radius",
]


def __getattr__(name):
    if name in {"m_dot_edd", "m_dot_bondi_hoyle", "m_dot_bh_hot_accretion"}:
        from .accretion import m_dot_bh_hot_accretion, m_dot_bondi_hoyle, m_dot_edd

        exports = {
            "m_dot_edd": m_dot_edd,
            "m_dot_bondi_hoyle": m_dot_bondi_hoyle,
            "m_dot_bh_hot_accretion": m_dot_bh_hot_accretion,
        }
        return exports[name]

    if name in {
        "roche_lobe_radius_ratio",
        "calculate_chirp_mass",
        "calculate_binary_separation",
        "calculate_orbital_angular_momentum",
        "calculate_period_separation_constants",
    }:
        from .binary_orbits import (
            calculate_binary_separation,
            calculate_chirp_mass,
            calculate_orbital_angular_momentum,
            calculate_period_separation_constants,
            roche_lobe_radius_ratio,
        )

        exports = {
            "roche_lobe_radius_ratio": roche_lobe_radius_ratio,
            "calculate_chirp_mass": calculate_chirp_mass,
            "calculate_binary_separation": calculate_binary_separation,
            "calculate_orbital_angular_momentum": calculate_orbital_angular_momentum,
            "calculate_period_separation_constants": calculate_period_separation_constants,
        }
        return exports[name]

    if name == "estimate_white_dwarf_radius":
        from .stellar_remnants import estimate_white_dwarf_radius

        return estimate_white_dwarf_radius

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
