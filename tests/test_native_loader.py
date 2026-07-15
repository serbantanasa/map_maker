from __future__ import annotations

from pathlib import Path

import pytest

from map_maker._native import (
    NATIVE_ABI_VERSION,
    NativeLibraryNotBuiltError,
    library_filename,
    native_library_info,
    native_library_path,
    workspace_root,
)


def test_workspace_root_finds_checkout() -> None:
    root = workspace_root(Path(__file__))
    assert root is not None
    assert (root / "Cargo.toml").is_file()
    assert (root / "pyproject.toml").is_file()


def test_native_library_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    library = tmp_path / library_filename("example_native")
    library.touch()
    monkeypatch.setenv("MAP_MAKER_NATIVE_LIB_DIR", str(tmp_path))

    assert native_library_path("example_native") == library


def test_missing_native_library_has_build_instructions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.setenv("MAP_MAKER_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.delenv("MAP_MAKER_NATIVE_LIB_DIR", raising=False)

    with pytest.raises(NativeLibraryNotBuiltError, match="map-maker-build-native"):
        native_library_path("missing_native")


def test_built_library_exposes_expected_abi_and_fingerprint() -> None:
    info = native_library_info("tectonics_native")

    assert info["abi_version"] == NATIVE_ABI_VERSION
    assert len(info["sha256"]) == 64
    assert Path(info["path"]).is_file()

    refinement = native_library_info("refinement_native")
    assert refinement["abi_version"] == 3
