"""POPKIN package public entry points."""

__version__ = "1.0.0"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import paths

__all__ = ["paths"]


def __getattr__(name):
    if name == "paths":
        from .config import paths

        return paths

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
