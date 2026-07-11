"""Locate explicitly built native libraries without mutating the checkout."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Iterable

NATIVE_ABI_VERSION = 2
SIMULATION_NATIVE_LIBRARIES = (
    "elevation_native",
    "erosion_native",
    "geology_native",
    "tectonics_native",
    "topology_native",
    "world_age_native",
)


class NativeLibraryNotBuiltError(ImportError):
    """Raised when a required native library has not been built."""


class NativeLibraryAbiError(ImportError):
    """Raised when a native library does not implement the expected ABI."""


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as library:
        for chunk in iter(lambda: library.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _native_library_info_cached(
    name: str, path_text: str, size: int, modified_ns: int
) -> dict[str, Any]:
    del size, modified_ns
    from cffi import FFI

    path = Path(path_text)
    symbol = f"{name}_abi_version"
    ffi = FFI()
    ffi.cdef(f"uint32_t {symbol}(void);")
    try:
        library = ffi.dlopen(str(path))
        actual_abi = int(getattr(library, symbol)())
    except (AttributeError, OSError) as exc:
        raise NativeLibraryAbiError(
            f"Native library {path} does not expose required ABI symbol {symbol!r}"
        ) from exc
    if actual_abi != NATIVE_ABI_VERSION:
        raise NativeLibraryAbiError(
            f"Native library {path} uses ABI {actual_abi}; expected {NATIVE_ABI_VERSION}"
        )
    return {
        "abi_version": actual_abi,
        "sha256": _file_sha256(path),
        "path": str(path),
    }


def native_library_info(name: str) -> dict[str, Any]:
    path = native_library_path(name)
    stat = path.stat()
    return _native_library_info_cached(name, str(path), stat.st_size, stat.st_mtime_ns)


def simulation_native_fingerprints(
    names: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    selected = SIMULATION_NATIVE_LIBRARIES if names is None else tuple(names)
    for name in sorted(selected):
        info = native_library_info(name)
        fingerprints[name] = {
            "abi_version": info["abi_version"],
            "sha256": info["sha256"],
        }
    return fingerprints


__all__ = [
    "NATIVE_ABI_VERSION",
    "NativeLibraryAbiError",
    "NativeLibraryNotBuiltError",
    "library_filename",
    "native_library_info",
    "native_library_path",
    "simulation_native_fingerprints",
    "workspace_root",
]
