"""Global path management."""
from pathlib import Path


class Paths:
    """Global project paths."""

    # Project root directory (POPKIN).
    ROOT = Path(__file__).resolve().parent.parent.parent.parent

    # Source directory.
    SRC = ROOT / 'src'

    # Package directory.
    PKG = SRC / "popkin"

    # Configuration directory.
    config = PKG / "config"

    # Test directory.
    tests = ROOT / "tests"

    # User workspace, set at runtime.
    workspaces: Path | None = None

    # Log directory, set at runtime.
    logs: Path | None = None

    # Data directory, set at runtime.
    data: Path | None = None


# Shared instance for imports.
paths = Paths()

# print(paths.ROOT)
# print(paths.SRC)
# print(paths.PKG)
# print(paths.config)
# print(paths.tests)
