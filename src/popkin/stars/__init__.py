"""Stellar evolution models."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .binary_star import BinaryStar
    from .single_star import SingleStar

__all__ = [
    "SingleStar",
    "BinaryStar",
]


def __getattr__(name):
    if name == "SingleStar":
        from .single_star import SingleStar

        return SingleStar

    if name == "BinaryStar":
        from .binary_star import BinaryStar

        return BinaryStar

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
