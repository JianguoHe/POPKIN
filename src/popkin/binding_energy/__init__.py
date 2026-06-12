"""Binding-energy parameter models."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .wjl2016 import z00001, z0001, z002
    from .xl2010 import lambda_XL2010

__all__ = [
    "z002",
    "z0001",
    "z00001",
    "lambda_XL2010",
]


def __getattr__(name):
    if name in {"z002", "z0001", "z00001"}:
        from .wjl2016 import z00001, z0001, z002

        exports = {
            "z002": z002,
            "z0001": z0001,
            "z00001": z00001,
        }
        return exports[name]

    if name == "lambda_XL2010":
        from .xl2010 import lambda_XL2010

        return lambda_XL2010

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
