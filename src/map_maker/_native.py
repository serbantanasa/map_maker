"""Locate explicitly built native libraries without mutating the checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path


class NativeLibraryNotBuiltError(ImportError):
    """Raised when a required native library has not been built."""


def library_filename(name: str) -> str:
    if sys.platform.startswith("win"):
        return f"{name}.dll"
    if sys.platform == "darwin":
        return f"lib{name}.dylib"
    return f"lib{name}.so"


def workspace_root(start: Path | None = None) -> Path | None:
    configured = os.environ.get("MAP_MAKER_WORKSPACE_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if (root / "Cargo.toml").exists():
            return root

    current = (start or Path(__file__)).resolve()
    for parent in (current, *current.parents):
        manifest = parent / "Cargo.toml"
        if manifest.exists() and (parent / "pyproject.toml").exists():
            return parent
    return None


def native_library_path(name: str, *, profile: str | None = None) -> Path:
    filename = library_filename(name)
    profile = profile or os.environ.get("MAP_MAKER_NATIVE_PROFILE", "release")

    candidates: list[Path] = []
    configured = os.environ.get("MAP_MAKER_NATIVE_LIB_DIR")
    if configured:
        candidates.append(Path(configured).expanduser().resolve() / filename)

    candidates.append(Path(__file__).resolve().parent / "_native_libs" / filename)

    root = workspace_root()
    if root is not None:
        candidates.append(root / "target" / profile / filename)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n  - ".join(str(path) for path in candidates)
    raise NativeLibraryNotBuiltError(
        f"Native library {filename!r} is not available. Run "
        "`map-maker-build-native` from the repository root, or set "
        "MAP_MAKER_NATIVE_LIB_DIR to a directory containing the built libraries."
        f"\nSearched:\n  - {searched}"
    )


__all__ = [
    "NativeLibraryNotBuiltError",
    "library_filename",
    "native_library_path",
    "workspace_root",
]
