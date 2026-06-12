"""Kinematic evolution and astrometric utilities."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .astrometry import (
        convert_equatorial_pm_to_galactic,
        propagate_equatorial_pm_errors_to_galactic,
    )
    from .orbit import OrbitIntegrator

__all__ = [
    "OrbitIntegrator",
    "convert_equatorial_pm_to_galactic",
    "propagate_equatorial_pm_errors_to_galactic",
]


def __getattr__(name):
    if name == "OrbitIntegrator":
        from .orbit import OrbitIntegrator

        return OrbitIntegrator

    if name in {
        "convert_equatorial_pm_to_galactic",
        "propagate_equatorial_pm_errors_to_galactic",
    }:
        from .astrometry import (
            convert_equatorial_pm_to_galactic,
            propagate_equatorial_pm_errors_to_galactic,
        )

        exports = {
            "convert_equatorial_pm_to_galactic": convert_equatorial_pm_to_galactic,
            "propagate_equatorial_pm_errors_to_galactic": propagate_equatorial_pm_errors_to_galactic,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
