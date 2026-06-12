"""Galaxy models.

``MilkyWay`` is the production galaxy model used by the synthesis drivers.
``Andromeda`` is currently experimental and exposes helper fits only.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .M31.galaxy import Andromeda
    from .MW.galaxy import MilkyWay

__all__ = [
    "MilkyWay",
    "Andromeda",
]


def __getattr__(name):
    if name == "MilkyWay":
        from .MW.galaxy import MilkyWay

        return MilkyWay

    if name == "Andromeda":
        from .M31.galaxy import Andromeda

        return Andromeda

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
