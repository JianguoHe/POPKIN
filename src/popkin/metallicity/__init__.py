"""Metallicity-dependent stellar-evolution coefficients."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .zcnst import zcnsts_set

__all__ = [
    "zcnsts_set",
]


def __getattr__(name):
    if name == "zcnsts_set":
        from .zcnst import zcnsts_set

        return zcnsts_set

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
