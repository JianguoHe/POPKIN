"""Program drivers for stellar and population synthesis runs."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bse import bse as run_bse
    from .popbin import popbin as run_popbin
    from .popsin import popsin as run_popsin
    from .sse import sse as run_sse

__all__ = [
    "run_sse",
    "run_bse",
    "run_popsin",
    "run_popbin",
]


def __getattr__(name):
    if name == "run_sse":
        from .sse import sse

        return sse

    if name == "run_bse":
        from .bse import bse

        return bse

    if name == "run_popsin":
        from .popsin import popsin

        return popsin

    if name == "run_popbin":
        from .popbin import popbin

        return popbin

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
