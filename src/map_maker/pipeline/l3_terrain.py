"""Generate resumable, seamless, parent-conditioned L3 base terrain."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from PIL import Image, ImageDraw, ImageFont
import yaml  # type: ignore[import-untyped]
import zarr  # type: ignore[import-untyped]

from .._native import native_library_info
from ._l3_terrain_native import run_l3_terrain_chunk
from .regional_handoff import _file_checksum, _replace_directory, _tree_checksum

TERRAIN_FORMAT_VERSION = 2
TERRAIN_MODEL_VERSION = "l3_conditioned_terrain_v5"
EARTH_OROGENIC_REFERENCE_M = 1_500.0
RAW_CHUNK_ARRAY_PATHS = (
    "geometry/cell_id",
    "geometry/parent_l2_cell_id",
    "geometry/face",
    "geometry/row",
    "geometry/column",
    "geometry/xyz",
    "geometry/area_km2",
    "geometry/inside_catchment_core",
    "geometry/inside_process_halo",
    "geometry/outside_process_domain",
    "terrain/elevation_m",
    "terrain/offset_from_l2_m",
    "terrain/unresolved_relief_m",
    "terrain/raw_elevation_m",
    "terrain/raw_offset_from_l2_m",
)
CONDITIONED_CHUNK_ARRAY_PATHS = (
    "terrain/elevation_m",
    "terrain/offset_from_l2_m",
)


@dataclass(frozen=True)
class L3TerrainConfig:
    target_dir: Path
    output_dir: Path
    requested_cell_size_m: float = 200.0
    refinement_factor: int = 22
    chunk_parent_count: int = 64
    terrain_seed: int = 4_203_001
    relief_realization_fraction: float = 0.42
    base_wavelength_m: float = 16_000.0
    octave_count: int = 5
    persistence: float = 0.52
    domain_warp_fraction: float = 0.22
    orogenic_ridge_fraction: float = 0.32
    orogenic_reference_m: float = EARTH_OROGENIC_REFERENCE_M
    maximum_parent_mean_error_m: float = 15.0
    maximum_parent_mean_error_relief_fraction: float = 0.05
    maximum_parent_area_relative_error: float = 1e-9
    maximum_parent_boundary_residual_p95_ratio: float = 1.35
    maximum_chunk_boundary_residual_p95_ratio: float = 1.50
    maximum_cell_size_relative_error: float = 0.12
    minimum_terrain_offset_std_m: float = 2.0
    maximum_tile_bubble_correlation_p50: float = 0.35
    maximum_tile_bubble_correlation_p95: float = 0.80
    conditioning_maximum_iterations: int = 24
    conditioning_damping: float = 0.72
    maximum_center_correction_relief_fraction: float = 0.75
    maximum_base_cell_count: int = 3_000_000
    maximum_peak_memory_gb: float = 24.0
    maximum_storage_gb: float = 2.0
    source_config: Path | None = None

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        output_dir: Path | None = None,
    ) -> "L3TerrainConfig":
        source = Path(path).expanduser().resolve()
        data = yaml.safe_load(source.read_text(encoding="utf8"))
        if not isinstance(data, Mapping):
            raise TypeError("L3 terrain config must contain a mapping")
        grid = data.get("grid", {})
        terrain = data.get("terrain", {})
        limits = data.get("limits", {})
        if not isinstance(grid, Mapping) or not isinstance(terrain, Mapping):
            raise TypeError("L3 grid and terrain controls must be mappings")
        if not isinstance(limits, Mapping):
            raise TypeError("L3 terrain limits must be a mapping")
        raw_target = data.get("output_dir")
        raw_output = data.get("terrain_output_dir")
        if not raw_target or not raw_output:
            raise ValueError("L3 terrain config requires output_dir and terrain_output_dir")
        config = cls(
            target_dir=(source.parent / str(raw_target)).resolve(),
            output_dir=(
                output_dir.expanduser().resolve()
                if output_dir is not None
                else (source.parent / str(raw_output)).resolve()
            ),
            requested_cell_size_m=float(grid.get("base_cell_size_m", 200.0)),
            refinement_factor=int(grid.get("l3_refinement_factor", 22)),
            chunk_parent_count=int(terrain.get("chunk_parent_count", 64)),
            terrain_seed=int(terrain.get("seed", 4_203_001)),
            relief_realization_fraction=float(terrain.get("relief_realization_fraction", 0.42)),
            base_wavelength_m=float(terrain.get("base_wavelength_m", 16_000.0)),
            octave_count=int(terrain.get("octave_count", 5)),
            persistence=float(terrain.get("persistence", 0.52)),
            domain_warp_fraction=float(terrain.get("domain_warp_fraction", 0.22)),
            orogenic_ridge_fraction=float(terrain.get("orogenic_ridge_fraction", 0.32)),
            orogenic_reference_m=float(
                terrain.get("orogenic_reference_m", EARTH_OROGENIC_REFERENCE_M)
            ),
            maximum_parent_mean_error_m=float(terrain.get("maximum_parent_mean_error_m", 15.0)),
            maximum_parent_mean_error_relief_fraction=float(
                terrain.get("maximum_parent_mean_error_relief_fraction", 0.05)
            ),
            maximum_parent_area_relative_error=float(
                terrain.get("maximum_parent_area_relative_error", 1e-9)
            ),
            maximum_parent_boundary_residual_p95_ratio=float(
                terrain.get("maximum_parent_boundary_residual_p95_ratio", 1.35)
            ),
            maximum_chunk_boundary_residual_p95_ratio=float(
                terrain.get("maximum_chunk_boundary_residual_p95_ratio", 1.50)
            ),
            maximum_cell_size_relative_error=float(
                terrain.get("maximum_cell_size_relative_error", 0.12)
            ),
            minimum_terrain_offset_std_m=float(terrain.get("minimum_terrain_offset_std_m", 2.0)),
            maximum_tile_bubble_correlation_p50=float(
                terrain.get("maximum_tile_bubble_correlation_p50", 0.35)
            ),
            maximum_tile_bubble_correlation_p95=float(
                terrain.get("maximum_tile_bubble_correlation_p95", 0.80)
            ),
            conditioning_maximum_iterations=int(terrain.get("conditioning_maximum_iterations", 24)),
            conditioning_damping=float(terrain.get("conditioning_damping", 0.72)),
            maximum_center_correction_relief_fraction=float(
                terrain.get("maximum_center_correction_relief_fraction", 0.75)
            ),
            maximum_base_cell_count=int(limits.get("maximum_base_cell_count", 3_000_000)),
            maximum_peak_memory_gb=float(limits.get("maximum_peak_memory_gb", 24.0)),
            maximum_storage_gb=float(limits.get("maximum_terrain_storage_gb", 2.0)),
            source_config=source,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 100.0 <= self.requested_cell_size_m <= 250.0:
            raise ValueError("grid.base_cell_size_m must be in [100, 250]")
        if not 2 <= self.refinement_factor <= 64:
            raise ValueError("grid.l3_refinement_factor must be in [2, 64]")
        if not 1 <= self.chunk_parent_count <= 1_024:
            raise ValueError("terrain.chunk_parent_count must be in [1, 1024]")
        if not 0 <= self.terrain_seed <= np.iinfo(np.uint64).max:
            raise ValueError("terrain.seed must fit uint64")
        for name, value in (
            ("relief_realization_fraction", self.relief_realization_fraction),
            ("domain_warp_fraction", self.domain_warp_fraction),
            ("orogenic_ridge_fraction", self.orogenic_ridge_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"terrain.{name} must be finite and in [0, 1]")
        if not math.isfinite(self.base_wavelength_m) or self.base_wavelength_m <= 0.0:
            raise ValueError("terrain.base_wavelength_m must be finite and positive")
        if not 2 <= self.octave_count <= 8:
            raise ValueError("terrain.octave_count must be in [2, 8]")
        if not 1 <= self.conditioning_maximum_iterations <= 200:
            raise ValueError("terrain.conditioning_maximum_iterations must be in [1, 200]")
        if not math.isfinite(self.persistence) or not 0.0 < self.persistence < 1.0:
            raise ValueError("terrain.persistence must be finite and in (0, 1)")
        if self.conditioning_damping > 1.0:
            raise ValueError("terrain.conditioning_damping must be at most 1")
        if self.maximum_tile_bubble_correlation_p50 > 1.0:
            raise ValueError("terrain.maximum_tile_bubble_correlation_p50 must be at most 1")
        if self.maximum_tile_bubble_correlation_p95 > 1.0:
            raise ValueError("terrain.maximum_tile_bubble_correlation_p95 must be at most 1")
        if not math.isfinite(self.orogenic_reference_m) or self.orogenic_reference_m <= 0.0:
            raise ValueError("terrain.orogenic_reference_m must be finite and positive")
        for name, value in (
            ("maximum_parent_mean_error_m", self.maximum_parent_mean_error_m),
            (
                "maximum_parent_mean_error_relief_fraction",
                self.maximum_parent_mean_error_relief_fraction,
            ),
            ("maximum_parent_area_relative_error", self.maximum_parent_area_relative_error),
            (
                "maximum_parent_boundary_residual_p95_ratio",
                self.maximum_parent_boundary_residual_p95_ratio,
            ),
            (
                "maximum_chunk_boundary_residual_p95_ratio",
                self.maximum_chunk_boundary_residual_p95_ratio,
            ),
            ("maximum_cell_size_relative_error", self.maximum_cell_size_relative_error),
            ("minimum_terrain_offset_std_m", self.minimum_terrain_offset_std_m),
            (
                "maximum_tile_bubble_correlation_p50",
                self.maximum_tile_bubble_correlation_p50,
            ),
            (
                "maximum_tile_bubble_correlation_p95",
                self.maximum_tile_bubble_correlation_p95,
            ),
            ("conditioning_damping", self.conditioning_damping),
            (
                "maximum_center_correction_relief_fraction",
                self.maximum_center_correction_relief_fraction,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"terrain.{name} must be finite and positive")
        if self.maximum_base_cell_count <= 0:
            raise ValueError("limits.maximum_base_cell_count must be positive")
        if not 1.0 <= self.maximum_peak_memory_gb <= 28.0:
            raise ValueError("limits.maximum_peak_memory_gb must be in [1, 28]")
        if not 0.1 <= self.maximum_storage_gb <= 32.0:
            raise ValueError("limits.maximum_terrain_storage_gb must be in [0.1, 32]")


@dataclass(frozen=True)
class L3TerrainResult:
    output_dir: Path
    manifest_path: Path
    validation_path: Path
    zarr_path: Path
    preview_path: Path
    target_id: str
    parent_count: int
    cell_count: int
    actual_cell_size_m: float
    chunk_count: int
    resumed_chunk_count: int


@dataclass(frozen=True)
class _TerrainSources:
    target_id: str
    target_manifest_path: Path
    handoff_dir: Path
    handoff_manifest_path: Path
    parent_resolution: int
    planet_radius_m: float
    context_ids: np.ndarray
    context_elevation_m: np.ndarray
    context_relief_m: np.ndarray
    context_area_km2: np.ndarray
    context_rock_strength: np.ndarray
    context_orogenic_strength: np.ndarray
    context_ridge_direction_xyz: np.ndarray
    domain_ids: np.ndarray
    domain_handoff_rows: np.ndarray
    domain_inside_core: np.ndarray
    domain_inside_process_halo: np.ndarray
    domain_outside_process: np.ndarray
    domain_lake_fraction: np.ndarray
    domain_wetland_fraction: np.ndarray
    domain_ocean_fraction: np.ndarray


def _column_numpy(table: pa.Table, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    return np.ascontiguousarray(
        table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
    )


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_direction(direction: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(direction))
    result = direction / norm if norm > 1e-12 else fallback / np.linalg.norm(fallback)
    dominant = int(np.argmax(np.abs(result)))
    if result[dominant] < 0.0:
        result = -result
    return result


def _ridge_directions(root: Any) -> np.ndarray:
    parent_ids = np.asarray(root["parent/cell_id"][:], dtype=np.int32)
    xyz = np.asarray(root["parent_priors/geometry/GeometryXYZ"][:], dtype=np.float64)
    neighbors = np.asarray(root["parent_priors/geometry/NeighborsD4"][:], dtype=np.int32)
    orogenic = np.asarray(root["parent_priors/elevation/OrogenicElevationM"][:], dtype=np.float64)
    plate_velocity = np.asarray(
        root["parent_priors/tectonics/PlateField"][:, 4:7], dtype=np.float64
    )
    row_by_id = {int(cell_id): row for row, cell_id in enumerate(parent_ids)}
    output = np.empty_like(xyz, dtype=np.float32)
    for row, center in enumerate(xyz):
        gradient = np.zeros(3, dtype=np.float64)
        for neighbor_id in neighbors[row]:
            neighbor_row = row_by_id.get(int(neighbor_id))
            if neighbor_row is None:
                continue
            tangent = xyz[neighbor_row] - center * float(np.dot(xyz[neighbor_row], center))
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > 1e-12:
                gradient += tangent / tangent_norm * (orogenic[neighbor_row] - orogenic[row])
        ridge = np.cross(center, gradient)
        fallback = plate_velocity[row] - center * float(np.dot(plate_velocity[row], center))
        if float(np.linalg.norm(fallback)) <= 1e-12:
            axis = np.asarray([0.0, 0.0, 1.0])
            if abs(float(np.dot(center, axis))) > 0.9:
                axis = np.asarray([0.0, 1.0, 0.0])
            fallback = np.cross(center, axis)
        output[row] = _canonical_direction(ridge, fallback).astype(np.float32)
    return output


def _load_sources(config: L3TerrainConfig) -> _TerrainSources:
    target_manifest_path = config.target_dir / "manifest.json"
    target_validation_path = config.target_dir / "validation.json"
    target_table_path = config.target_dir / "tables/target_l2_cells.parquet"
    for required in (target_manifest_path, target_validation_path, target_table_path):
        if not required.exists():
            raise FileNotFoundError(required)
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf8"))
    target_validation = json.loads(target_validation_path.read_text(encoding="utf8"))
    if not target_manifest.get("validation_passed") or not target_validation.get("passed"):
        raise RuntimeError("L3 terrain source target has not passed validation")
    if int(target_manifest.get("format_version", 0)) < 2:
        raise RuntimeError("L3 terrain requires a continuous-window target package (format 2)")
    handoff_dir = Path(target_manifest["source"]["handoff_dir"]).expanduser().resolve()
    handoff_manifest_path = handoff_dir / "manifest.json"
    handoff_zarr_path = handoff_dir / "region.zarr"
    for required in (handoff_manifest_path, handoff_zarr_path):
        if not required.exists():
            raise FileNotFoundError(required)
    root = zarr.open_group(str(handoff_zarr_path), mode="r")
    target = pq.read_table(target_table_path).combine_chunks()
    target_ids = _column_numpy(target, "fine_cell_id", np.dtype(np.int32))
    handoff_rows = _column_numpy(target, "handoff_child_row", np.dtype(np.int32))
    inside_core = _column_numpy(target, "inside_target_core", np.dtype(bool))
    inside_window = _column_numpy(target, "inside_terrain_window", np.dtype(bool))
    inside_process_halo = _column_numpy(target, "inside_process_halo", np.dtype(bool))
    outside_process = _column_numpy(target, "outside_process_domain", np.dtype(bool))
    order = np.argsort(target_ids, kind="stable")
    target_ids = target_ids[order]
    handoff_rows = handoff_rows[order]
    inside_core = inside_core[order]
    inside_window = inside_window[order]
    inside_process_halo = inside_process_halo[order]
    outside_process = outside_process[order]
    if len(np.unique(target_ids)) != len(target_ids):
        raise RuntimeError("L3 target contains duplicate L2 cell IDs")

    context_elevation = np.asarray(
        root["l2/geometry/terrain_elevation_m"].oindex[handoff_rows], dtype=np.float32
    )
    context_relief = np.asarray(
        root["l2/geometry/parent_relief_m"].oindex[handoff_rows], dtype=np.float32
    )
    context_area = np.asarray(root["l2/geometry/area_km2"].oindex[handoff_rows], dtype=np.float64)
    context_l0_ids = np.asarray(
        root["l2/geometry/parent_cell_id"].oindex[handoff_rows], dtype=np.int32
    )
    parent_ids = np.asarray(root["parent/cell_id"][:], dtype=np.int32)
    parent_order = np.argsort(parent_ids)
    sorted_parent_ids = parent_ids[parent_order]
    parent_positions = np.searchsorted(sorted_parent_ids, context_l0_ids)
    if np.any(parent_positions >= len(sorted_parent_ids)) or np.any(
        sorted_parent_ids[parent_positions] != context_l0_ids
    ):
        raise RuntimeError("L3 target references an L0 parent outside the handoff")
    parent_rows = parent_order[parent_positions]
    rock_strength = np.asarray(root["parent_priors/geology/RockStrength"][:], dtype=np.float32)[
        parent_rows
    ]
    orogenic_elevation = np.asarray(
        root["parent_priors/elevation/OrogenicElevationM"][:], dtype=np.float32
    )[parent_rows]
    orogenic_strength = np.clip(orogenic_elevation / config.orogenic_reference_m, 0.0, 1.0).astype(
        np.float32
    )
    ridge_direction = _ridge_directions(root)[parent_rows]
    parent_area_km2 = np.asarray(root["parent/area_km2"][:], dtype=np.float64)
    parent_area_steradians = np.asarray(
        root["parent_priors/geometry/CellArea"][:], dtype=np.float64
    )
    radius_m = (
        math.sqrt(float(np.sum(parent_area_km2)) / float(np.sum(parent_area_steradians))) * 1_000.0
    )
    domain_ids = target_ids[inside_window]
    domain_handoff_rows = handoff_rows[inside_window]
    domain_inside_core = inside_core[inside_window]
    domain_inside_process_halo = inside_process_halo[inside_window]
    domain_outside_process = outside_process[inside_window]
    role_count = (
        domain_inside_core.astype(np.int8)
        + domain_inside_process_halo.astype(np.int8)
        + domain_outside_process.astype(np.int8)
    )
    if len(domain_ids) == 0 or not np.any(domain_inside_core):
        raise RuntimeError("L3 target contains no terrain window or catchment core")
    if np.any(role_count != 1):
        raise RuntimeError("L3 target domain masks must be mutually exclusive and exhaustive")
    parent_resolution = int(root.attrs["child_face_resolution"])
    face_size = parent_resolution * parent_resolution
    domain_face = domain_ids.astype(np.int64) // face_size
    domain_within_face = domain_ids.astype(np.int64) % face_size
    domain_row = domain_within_face // parent_resolution
    domain_column = domain_within_face % parent_resolution
    if (
        len(np.unique(domain_face)) != 1
        or np.any((domain_row == 0) | (domain_row + 1 == parent_resolution))
        or np.any((domain_column == 0) | (domain_column + 1 == parent_resolution))
    ):
        raise NotImplementedError(
            "L3 V0 bilinear terrain conditioning requires a core contained inside one "
            "cubed-sphere face"
        )
    expected_domain_count = int(
        (np.max(domain_row) - np.min(domain_row) + 1)
        * (np.max(domain_column) - np.min(domain_column) + 1)
    )
    if len(domain_ids) != expected_domain_count:
        raise RuntimeError("L3 terrain window is not a complete rectangle")
    context_neighbors = _neighbor_rows(domain_ids, target_ids, parent_resolution)
    if np.any(context_neighbors < 0):
        raise RuntimeError("L3 terrain window lacks its complete L2 source-context ring")
    lake_fraction = _column_numpy(target, "lake_fraction", np.dtype(np.float32))[order]
    wetland_fraction = _column_numpy(target, "wetland_fraction", np.dtype(np.float32))[order]
    ocean_fraction = _column_numpy(target, "ocean_fraction", np.dtype(np.float32))[order]
    return _TerrainSources(
        target_id=str(target_manifest["target_id"]),
        target_manifest_path=target_manifest_path,
        handoff_dir=handoff_dir,
        handoff_manifest_path=handoff_manifest_path,
        parent_resolution=parent_resolution,
        planet_radius_m=radius_m,
        context_ids=target_ids,
        context_elevation_m=np.ascontiguousarray(context_elevation),
        context_relief_m=np.ascontiguousarray(context_relief),
        context_area_km2=np.ascontiguousarray(context_area),
        context_rock_strength=np.ascontiguousarray(rock_strength),
        context_orogenic_strength=np.ascontiguousarray(orogenic_strength),
        context_ridge_direction_xyz=np.ascontiguousarray(ridge_direction),
        domain_ids=np.ascontiguousarray(domain_ids),
        domain_handoff_rows=np.ascontiguousarray(domain_handoff_rows),
        domain_inside_core=np.ascontiguousarray(domain_inside_core),
        domain_inside_process_halo=np.ascontiguousarray(domain_inside_process_halo),
        domain_outside_process=np.ascontiguousarray(domain_outside_process),
        domain_lake_fraction=np.ascontiguousarray(lake_fraction[inside_window]),
        domain_wetland_fraction=np.ascontiguousarray(wetland_fraction[inside_window]),
        domain_ocean_fraction=np.ascontiguousarray(ocean_fraction[inside_window]),
    )


def _run_fingerprint(
    config: L3TerrainConfig, sources: _TerrainSources
) -> tuple[str, dict[str, Any]]:
    native = native_library_info("l3_terrain_native")
    components = {
        "model_version": TERRAIN_MODEL_VERSION,
        "target_manifest_sha256": _file_checksum(sources.target_manifest_path),
        "handoff_manifest_sha256": _file_checksum(sources.handoff_manifest_path),
        "config_sha256": _file_checksum(config.source_config) if config.source_config else None,
        "native_abi_version": native["abi_version"],
        "native_sha256": native["sha256"],
        "orchestrator_sha256": _file_checksum(Path(__file__)),
        "binding_sha256": _file_checksum(Path(__file__).with_name("_l3_terrain_native.py")),
    }
    return _canonical_hash(components), components


def _controls(config: L3TerrainConfig, sources: _TerrainSources) -> dict[str, int | float]:
    return {
        "parent_resolution": sources.parent_resolution,
        "factor": config.refinement_factor,
        "planet_radius_m": sources.planet_radius_m,
        "terrain_seed": config.terrain_seed,
        "relief_realization_fraction": config.relief_realization_fraction,
        "base_wavelength_m": config.base_wavelength_m,
        "octave_count": config.octave_count,
        "persistence": config.persistence,
        "domain_warp_fraction": config.domain_warp_fraction,
        "orogenic_ridge_fraction": config.orogenic_ridge_fraction,
    }


def _zarr_dataset(
    group: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[Any],
    chunks: tuple[int, ...],
    **attrs: Any,
) -> Any:
    dataset = group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        overwrite=True,
    )
    dataset.attrs.update(attrs)
    return dataset


def _fsync_paths(paths: list[Path]) -> None:
    files = [path for path in paths if path.is_file()]
    for path in files:
        with path.open("rb") as source:
            os.fsync(source.fileno())
    if os.name != "posix":
        return
    directories = sorted(
        {path.parent for path in files}, key=lambda path: len(path.parts), reverse=True
    )
    for directory in directories:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_json_durable(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf8")
    with temporary.open("wb") as destination:
        destination.write(encoded)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    _fsync_paths([path])


def _zarr_chunk_path(zarr_path: Path, root: Any, array_path: str, chunk_index: int) -> Path:
    array = root[array_path]
    chunk_key = ".".join((str(chunk_index), *("0" for _ in range(array.ndim - 1))))
    return zarr_path / array_path / chunk_key


def _sync_zarr_chunk(
    zarr_path: Path,
    root: Any,
    array_paths: tuple[str, ...],
    chunk_index: int,
) -> None:
    paths = [_zarr_chunk_path(zarr_path, root, name, chunk_index) for name in array_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Zarr chunk write did not materialize: {', '.join(missing)}")
    _fsync_paths(paths)


def _sync_zarr_array(zarr_path: Path, array_path: str) -> None:
    directory = zarr_path / array_path
    paths = [
        path for path in directory.iterdir() if path.is_file() and not path.name.startswith(".")
    ]
    if not paths:
        raise RuntimeError(f"Zarr array {array_path} has no materialized chunks")
    _fsync_paths(paths)


def _initialize_partial(
    partial: Path,
    config: L3TerrainConfig,
    sources: _TerrainSources,
    run_fingerprint: str,
) -> Any:
    partial.mkdir(parents=True)
    factor = config.refinement_factor
    children_per_parent = factor * factor
    cell_count = len(sources.domain_ids) * children_per_parent
    chunk_rows = config.chunk_parent_count * children_per_parent
    chunk_count = math.ceil(len(sources.domain_ids) / config.chunk_parent_count)
    root = zarr.open_group(str(partial / "terrain.zarr"), mode="w")
    root.attrs.update(
        {
            "format_version": TERRAIN_FORMAT_VERSION,
            "model_version": TERRAIN_MODEL_VERSION,
            "status": "partial",
            "target_id": sources.target_id,
            "run_fingerprint": run_fingerprint,
            "parent_level": "L2",
            "child_level": "L3",
            "parent_face_resolution": sources.parent_resolution,
            "child_face_resolution": sources.parent_resolution * factor,
            "refinement_factor": factor,
            "children_per_parent": children_per_parent,
            "parent_count": len(sources.domain_ids),
            "catchment_core_parent_count": int(np.count_nonzero(sources.domain_inside_core)),
            "process_halo_parent_count": int(np.count_nonzero(sources.domain_inside_process_halo)),
            "outside_process_parent_count": int(np.count_nonzero(sources.domain_outside_process)),
            "cell_count": cell_count,
            "parent_major_storage": True,
            "chunk_parent_count": config.chunk_parent_count,
            "chunk_rows": chunk_rows,
        }
    )
    geometry = root.require_group("geometry")
    _zarr_dataset(geometry, "cell_id", shape=(cell_count,), dtype=np.uint64, chunks=(chunk_rows,))
    _zarr_dataset(
        geometry,
        "parent_l2_cell_id",
        shape=(cell_count,),
        dtype=np.int32,
        chunks=(chunk_rows,),
    )
    _zarr_dataset(geometry, "face", shape=(cell_count,), dtype=np.uint8, chunks=(chunk_rows,))
    _zarr_dataset(geometry, "row", shape=(cell_count,), dtype=np.int32, chunks=(chunk_rows,))
    _zarr_dataset(geometry, "column", shape=(cell_count,), dtype=np.int32, chunks=(chunk_rows,))
    _zarr_dataset(
        geometry,
        "xyz",
        shape=(cell_count, 3),
        dtype=np.float32,
        chunks=(chunk_rows, 3),
        basis="global_unit_sphere_xyz",
    )
    _zarr_dataset(
        geometry,
        "area_km2",
        shape=(cell_count,),
        dtype=np.float64,
        chunks=(chunk_rows,),
        units="km2",
    )
    for name, semantics in (
        ("inside_catchment_core", "hydrological acceptance and reported basin interior"),
        ("inside_process_halo", "context cells available to regional process solves"),
        ("outside_process_domain", "continuous terrain context excluded from process gates"),
    ):
        _zarr_dataset(
            geometry,
            name,
            shape=(cell_count,),
            dtype=bool,
            chunks=(chunk_rows,),
            semantics=semantics,
        )
    terrain = root.require_group("terrain")
    _zarr_dataset(
        terrain,
        "elevation_m",
        shape=(cell_count,),
        dtype=np.float32,
        chunks=(chunk_rows,),
        units="m",
        semantics="L2-mean-conditioned base terrain",
    )
    _zarr_dataset(
        terrain,
        "offset_from_l2_m",
        shape=(cell_count,),
        dtype=np.float32,
        chunks=(chunk_rows,),
        units="m",
    )
    _zarr_dataset(
        terrain,
        "unresolved_relief_m",
        shape=(cell_count,),
        dtype=np.float32,
        chunks=(chunk_rows,),
        units="m",
        semantics="sub-200m relief prior; not elevation added to the cell mean",
    )
    _zarr_dataset(
        terrain,
        "raw_elevation_m",
        shape=(cell_count,),
        dtype=np.float32,
        chunks=(chunk_rows,),
        units="m",
        semantics="deterministic pre-conditioning terrain retained for resumable replay",
    )
    _zarr_dataset(
        terrain,
        "raw_offset_from_l2_m",
        shape=(cell_count,),
        dtype=np.float32,
        chunks=(chunk_rows,),
        units="m",
    )
    conditioning = root.require_group("conditioning")
    context_chunks = (min(len(sources.context_ids), max(1, config.chunk_parent_count * 4)),)
    context_id_dataset = _zarr_dataset(
        conditioning,
        "context_l2_cell_id",
        shape=(len(sources.context_ids),),
        dtype=np.int32,
        chunks=context_chunks,
    )
    context_id_dataset[:] = sources.context_ids
    _zarr_dataset(
        conditioning,
        "center_correction_m",
        shape=(len(sources.context_ids),),
        dtype=np.float64,
        chunks=context_chunks,
        units="m",
        semantics="continuous center field; source context outside the terrain window is fixed to zero",
    )
    _zarr_dataset(
        conditioning,
        "raw_parent_mean_error_m",
        shape=(len(sources.domain_ids),),
        dtype=np.float64,
        chunks=(min(len(sources.domain_ids), config.chunk_parent_count),),
        units="m",
    )
    progress = root.require_group("progress")
    _zarr_dataset(
        progress,
        "chunk_complete",
        shape=(chunk_count,),
        dtype=bool,
        chunks=(min(chunk_count, 1_024),),
    )
    _zarr_dataset(
        progress,
        "chunk_conditioned",
        shape=(chunk_count,),
        dtype=bool,
        chunks=(min(chunk_count, 1_024),),
    )
    root.attrs["center_corrections_solved"] = False
    state = {
        "format_version": TERRAIN_FORMAT_VERSION,
        "model_version": TERRAIN_MODEL_VERSION,
        "run_fingerprint": run_fingerprint,
        "parent_count": len(sources.domain_ids),
        "cell_count": cell_count,
        "chunk_count": chunk_count,
    }
    _write_json_durable(partial / "run_state.json", state)
    (partial / "chunk_stats").mkdir()
    return root


def _open_partial(
    partial: Path,
    config: L3TerrainConfig,
    sources: _TerrainSources,
    run_fingerprint: str,
) -> tuple[Any, bool]:
    state_path = partial / "run_state.json"
    if partial.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf8"))
        except (FileNotFoundError, json.JSONDecodeError):
            state = {}
        expected_cells = len(sources.domain_ids) * config.refinement_factor**2
        valid = (
            state.get("run_fingerprint") == run_fingerprint
            and state.get("parent_count") == len(sources.domain_ids)
            and state.get("cell_count") == expected_cells
        )
        if valid:
            return zarr.open_group(str(partial / "terrain.zarr"), mode="r+"), True
        shutil.rmtree(partial)
    return _initialize_partial(partial, config, sources, run_fingerprint), False


def _write_chunk(
    root: Any,
    start: int,
    end: int,
    outputs: Mapping[str, np.ndarray],
    *,
    inside_core: np.ndarray,
    inside_process_halo: np.ndarray,
    outside_process: np.ndarray,
) -> None:
    root["geometry/cell_id"][start:end] = outputs["cell_id"]
    root["geometry/parent_l2_cell_id"][start:end] = outputs["parent_l2_cell_id"]
    root["geometry/face"][start:end] = outputs["face"]
    root["geometry/row"][start:end] = outputs["row"]
    root["geometry/column"][start:end] = outputs["column"]
    root["geometry/xyz"][start:end] = outputs["xyz"]
    root["geometry/area_km2"][start:end] = outputs["area_km2"]
    root["geometry/inside_catchment_core"][start:end] = inside_core
    root["geometry/inside_process_halo"][start:end] = inside_process_halo
    root["geometry/outside_process_domain"][start:end] = outside_process
    root["terrain/elevation_m"][start:end] = outputs["elevation_m"]
    root["terrain/offset_from_l2_m"][start:end] = outputs["offset_from_l2_m"]
    root["terrain/unresolved_relief_m"][start:end] = outputs["unresolved_relief_m"]
    root["terrain/raw_elevation_m"][start:end] = outputs["elevation_m"]
    root["terrain/raw_offset_from_l2_m"][start:end] = outputs["offset_from_l2_m"]


def _neighbor_rows(cell_ids: np.ndarray, available_ids: np.ndarray, resolution: int) -> np.ndarray:
    """Return same-face N/S/W/E/NW/NE/SW/SE rows for the bounded V0 target."""

    ids = np.asarray(cell_ids, dtype=np.int64)
    face_size = resolution * resolution
    within_face = ids % face_size
    row = within_face // resolution
    column = within_face % resolution
    candidates = np.stack(
        (
            np.where(row > 0, ids - resolution, -1),
            np.where(row + 1 < resolution, ids + resolution, -1),
            np.where(column > 0, ids - 1, -1),
            np.where(column + 1 < resolution, ids + 1, -1),
            np.where((row > 0) & (column > 0), ids - resolution - 1, -1),
            np.where((row > 0) & (column + 1 < resolution), ids - resolution + 1, -1),
            np.where((row + 1 < resolution) & (column > 0), ids + resolution - 1, -1),
            np.where(
                (row + 1 < resolution) & (column + 1 < resolution),
                ids + resolution + 1,
                -1,
            ),
        ),
        axis=1,
    )
    positions = np.searchsorted(available_ids, candidates)
    valid = (candidates >= 0) & (positions < len(available_ids))
    clipped = np.minimum(positions, max(0, len(available_ids) - 1))
    valid &= available_ids[clipped] == candidates
    return np.where(valid, positions, -1).astype(np.int32)


def _conditioning_basis(factor: int) -> tuple[np.ndarray, np.ndarray]:
    unit = (np.arange(factor, dtype=np.float64) + 0.5) / factor - 0.5
    local_y, local_x = np.meshgrid(unit, unit, indexing="ij")
    north = np.maximum(-local_y, 0.0)
    south = np.maximum(local_y, 0.0)
    west = np.maximum(-local_x, 0.0)
    east = np.maximum(local_x, 0.0)
    center_y = 1.0 - np.abs(local_y)
    center_x = 1.0 - np.abs(local_x)
    neighbor_basis = np.stack(
        (
            north * center_x,
            south * center_x,
            west * center_y,
            east * center_y,
            north * west,
            north * east,
            south * west,
            south * east,
        ),
        axis=0,
    ).reshape(8, -1)
    return (center_y * center_x).reshape(-1), neighbor_basis


def _conditioning_weights(child_area: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    center_basis, neighbor_basis = _conditioning_basis(factor)
    total_area = np.sum(child_area, axis=1)
    neighbor_weights = np.stack(
        [np.sum(child_area * component, axis=1) / total_area for component in neighbor_basis],
        axis=1,
    )
    center_weight = np.sum(child_area * center_basis, axis=1) / total_area
    return center_weight, neighbor_weights


def _restriction(
    area: np.ndarray,
    elevation: np.ndarray,
    parent_count: int,
    children_per_parent: int,
) -> tuple[np.ndarray, np.ndarray]:
    grouped_area = area.reshape(parent_count, children_per_parent)
    grouped_elevation = elevation.reshape(parent_count, children_per_parent)
    restricted_area = np.sum(grouped_area, axis=1)
    restricted_elevation = np.sum(grouped_area * grouped_elevation, axis=1) / restricted_area
    return restricted_area, restricted_elevation


def _solve_center_corrections(
    root: Any,
    sources: _TerrainSources,
    config: L3TerrainConfig,
) -> tuple[np.ndarray, dict[str, int | float]]:
    factor = config.refinement_factor
    children_per_parent = factor * factor
    parent_count = len(sources.domain_ids)
    area = np.asarray(root["geometry/area_km2"][:], dtype=np.float64)
    raw_elevation = np.asarray(root["terrain/raw_elevation_m"][:], dtype=np.float64)
    grouped_area = area.reshape(parent_count, children_per_parent)
    _, raw_parent_mean = _restriction(area, raw_elevation, parent_count, children_per_parent)
    domain_context_rows = np.searchsorted(sources.context_ids, sources.domain_ids)
    source_elevation = sources.context_elevation_m[domain_context_rows].astype(np.float64)
    source_relief = sources.context_relief_m[domain_context_rows].astype(np.float64)
    raw_error = raw_parent_mean - source_elevation
    center_weight, neighbor_weight = _conditioning_weights(grouped_area, factor)
    neighbors = _neighbor_rows(sources.domain_ids, sources.domain_ids, sources.parent_resolution)
    correction = np.zeros(parent_count, dtype=np.float64)
    final_error = raw_error.copy()
    iterations = 0
    converged = False
    for iteration in range(config.conditioning_maximum_iterations):
        for parent in range(parent_count):
            adjacent = 0.0
            for slot, neighbor in enumerate(neighbors[parent]):
                if neighbor >= 0:
                    adjacent += neighbor_weight[parent, slot] * correction[neighbor]
            target = (-raw_error[parent] - adjacent) / center_weight[parent]
            correction[parent] = (1.0 - config.conditioning_damping) * correction[
                parent
            ] + config.conditioning_damping * target
        final_error = raw_error + center_weight * correction
        for slot in range(neighbor_weight.shape[1]):
            present = neighbors[:, slot] >= 0
            final_error[present] += (
                neighbor_weight[present, slot] * correction[neighbors[present, slot]]
            )
        iterations = iteration + 1
        if (
            float(np.max(np.abs(final_error))) <= config.maximum_parent_mean_error_m
            and float(np.max(np.abs(final_error) / np.maximum(source_relief, 50.0)))
            <= config.maximum_parent_mean_error_relief_fraction
        ):
            converged = True
            break
    context_correction = np.zeros(len(sources.context_ids), dtype=np.float64)
    context_correction[domain_context_rows] = correction
    correction_relief_fraction = np.abs(correction) / np.maximum(source_relief, 50.0)
    metadata: dict[str, int | float] = {
        "conditioning_iteration_count": iterations,
        "conditioning_converged": int(converged),
        "conditioning_maximum_iterations": config.conditioning_maximum_iterations,
        "raw_parent_mean_error_max_m": float(np.max(np.abs(raw_error))),
        "conditioned_predicted_parent_mean_error_max_m": float(np.max(np.abs(final_error))),
        "conditioned_predicted_parent_mean_error_relief_fraction_max": float(
            np.max(np.abs(final_error) / np.maximum(source_relief, 50.0))
        ),
        "center_correction_p05_m": _percentile(correction, 5.0),
        "center_correction_p50_m": _percentile(correction, 50.0),
        "center_correction_p95_m": _percentile(correction, 95.0),
        "center_correction_max_abs_m": float(np.max(np.abs(correction))),
        "center_correction_relief_fraction_max": float(np.max(correction_relief_fraction)),
        "center_correction_bounded_valid": int(
            float(np.max(correction_relief_fraction))
            <= config.maximum_center_correction_relief_fraction
        ),
    }
    root["conditioning/raw_parent_mean_error_m"][:] = raw_error
    return context_correction, metadata


def _interpolated_center_correction(
    parent_ids: np.ndarray,
    context_ids: np.ndarray,
    context_correction_m: np.ndarray,
    parent_resolution: int,
    factor: int,
) -> np.ndarray:
    parent_rows = np.searchsorted(context_ids, parent_ids)
    if np.any(parent_rows >= len(context_ids)) or np.any(context_ids[parent_rows] != parent_ids):
        raise RuntimeError("conditioning references an L2 parent outside its context")
    neighbor_rows = _neighbor_rows(parent_ids, context_ids, parent_resolution)
    center = context_correction_m[parent_rows]
    adjacent = np.repeat(center[:, None], 8, axis=1)
    present = neighbor_rows >= 0
    adjacent[present] = context_correction_m[neighbor_rows[present]]
    center_basis, neighbor_basis = _conditioning_basis(factor)
    correction = center[:, None] * center_basis[None, :]
    for slot in range(neighbor_basis.shape[0]):
        correction += adjacent[:, slot, None] * neighbor_basis[slot][None, :]
    return correction.reshape(-1)


def _condition_chunk(
    root: Any,
    sources: _TerrainSources,
    config: L3TerrainConfig,
    parent_start: int,
    parent_end: int,
) -> None:
    children_per_parent = config.refinement_factor**2
    row_start = parent_start * children_per_parent
    row_end = parent_end * children_per_parent
    context_correction = np.asarray(root["conditioning/center_correction_m"][:], dtype=np.float64)
    correction = _interpolated_center_correction(
        sources.domain_ids[parent_start:parent_end],
        sources.context_ids,
        context_correction,
        sources.parent_resolution,
        config.refinement_factor,
    )
    raw_elevation = np.asarray(root["terrain/raw_elevation_m"][row_start:row_end], dtype=np.float64)
    raw_offset = np.asarray(
        root["terrain/raw_offset_from_l2_m"][row_start:row_end], dtype=np.float64
    )
    root["terrain/elevation_m"][row_start:row_end] = (raw_elevation + correction).astype(np.float32)
    root["terrain/offset_from_l2_m"][row_start:row_end] = (raw_offset + correction).astype(
        np.float32
    )


def _percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.percentile(values, quantile)) if values.size else 0.0


def _observed_peak_rss_bytes() -> int:
    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum_rss if sys.platform == "darwin" else maximum_rss * 1_024


def _seam_metrics(
    root: Any,
    sources: _TerrainSources,
    config: L3TerrainConfig,
) -> dict[str, int | float]:
    cell_ids = np.asarray(root["geometry/cell_id"][:], dtype=np.uint64)
    rows = np.asarray(root["geometry/row"][:], dtype=np.int32)
    columns = np.asarray(root["geometry/column"][:], dtype=np.int32)
    parent_ids = np.asarray(root["geometry/parent_l2_cell_id"][:], dtype=np.int32)
    elevation = np.asarray(root["terrain/elevation_m"][:], dtype=np.float64)
    fine_resolution = sources.parent_resolution * config.refinement_factor
    local_rows = np.arange(len(cell_ids), dtype=np.int64)
    east_mask = columns + 1 < fine_resolution
    south_mask = rows + 1 < fine_resolution
    source_rows = np.concatenate((local_rows[east_mask], local_rows[south_mask]))
    neighbor_ids = np.concatenate(
        (
            cell_ids[east_mask] + np.uint64(1),
            cell_ids[south_mask] + np.uint64(fine_resolution),
        )
    )
    order = np.argsort(cell_ids)
    sorted_ids = cell_ids[order]
    positions = np.searchsorted(sorted_ids, neighbor_ids)
    valid = positions < len(sorted_ids)
    valid[valid] &= sorted_ids[positions[valid]] == neighbor_ids[valid]
    source_rows = source_rows[valid]
    target_rows = order[positions[valid]]
    if source_rows.size == 0:
        raise RuntimeError("L3 terrain has no measurable adjacent cell edges")
    boundary = parent_ids[source_rows] != parent_ids[target_rows]
    interior_jump = np.abs(elevation[target_rows[~boundary]] - elevation[source_rows[~boundary]])
    boundary_delta = elevation[target_rows[boundary]] - elevation[source_rows[boundary]]
    context_rows = np.searchsorted(sources.context_ids, parent_ids[source_rows[boundary]])
    target_context_rows = np.searchsorted(sources.context_ids, parent_ids[target_rows[boundary]])
    expected_delta = (
        sources.context_elevation_m[target_context_rows] - sources.context_elevation_m[context_rows]
    ) / config.refinement_factor
    boundary_residual = np.abs(boundary_delta - expected_delta)
    children_per_parent = config.refinement_factor**2
    source_chunk = source_rows[boundary] // children_per_parent // config.chunk_parent_count
    target_chunk = target_rows[boundary] // children_per_parent // config.chunk_parent_count
    crosses_chunk = source_chunk != target_chunk
    chunk_residual = boundary_residual[crosses_chunk]
    interior_p95 = _percentile(interior_jump, 95.0)
    parent_residual_p95 = _percentile(boundary_residual, 95.0)
    chunk_residual_p95 = _percentile(chunk_residual, 95.0)
    return {
        "measured_edge_count": int(len(source_rows)),
        "interior_edge_count": int(np.count_nonzero(~boundary)),
        "l2_parent_boundary_edge_count": int(np.count_nonzero(boundary)),
        "chunk_boundary_edge_count": int(np.count_nonzero(crosses_chunk)),
        "interior_jump_p50_m": _percentile(interior_jump, 50.0),
        "interior_jump_p95_m": interior_p95,
        "l2_parent_boundary_jump_p50_m": _percentile(np.abs(boundary_delta), 50.0),
        "l2_parent_boundary_jump_p95_m": _percentile(np.abs(boundary_delta), 95.0),
        "l2_parent_boundary_residual_p95_m": parent_residual_p95,
        "l2_parent_boundary_residual_p95_ratio": parent_residual_p95 / max(interior_p95, 1e-9),
        "chunk_boundary_residual_p95_m": chunk_residual_p95,
        "chunk_boundary_residual_p95_ratio": chunk_residual_p95 / max(interior_p95, 1e-9),
    }


def _tile_motif_metrics(offsets: np.ndarray, factor: int) -> dict[str, float | int]:
    """Detect the repeated sine-bubble pattern formerly used for exact restriction."""

    patterns = np.asarray(offsets, dtype=np.float64).reshape(-1, factor * factor)
    stride = max(1, math.ceil(len(patterns) / 2_048))
    sample = patterns[::stride]
    local_y, local_x = np.mgrid[:factor, :factor]
    design = np.stack(
        (
            np.ones(factor * factor),
            local_x.reshape(-1),
            local_y.reshape(-1),
        ),
        axis=1,
    )
    coefficients = sample @ np.linalg.pinv(design).T
    residual = sample - coefficients @ design.T
    unit = (np.arange(factor, dtype=np.float64) + 0.5) / factor
    bubble = np.outer(np.sin(np.pi * unit), np.sin(np.pi * unit)).reshape(-1)
    bubble -= np.mean(bubble)
    bubble /= max(float(np.linalg.norm(bubble)), 1e-12)
    correlations = np.abs(residual @ bubble) / np.maximum(np.linalg.norm(residual, axis=1), 1e-12)
    return {
        "tile_motif_sample_parent_count": len(sample),
        "tile_bubble_absolute_correlation_p50": _percentile(correlations, 50.0),
        "tile_bubble_absolute_correlation_p95": _percentile(correlations, 95.0),
        "tile_bubble_absolute_correlation_max": float(np.max(correlations, initial=0.0)),
    }


def _chunk_replay_valid(
    root: Any,
    sources: _TerrainSources,
    config: L3TerrainConfig,
) -> bool:
    if len(sources.domain_ids) < 2:
        replay_ids = sources.domain_ids
        parent_start = 0
    else:
        boundary = min(config.chunk_parent_count, len(sources.domain_ids) - 1)
        parent_start = max(0, boundary - 1)
        replay_ids = sources.domain_ids[parent_start : parent_start + 2]
    replay, _ = run_l3_terrain_chunk(
        controls=_controls(config, sources),
        context_parent_ids=sources.context_ids,
        context_elevation_m=sources.context_elevation_m,
        context_relief_m=sources.context_relief_m,
        context_rock_strength=sources.context_rock_strength,
        context_orogenic_strength=sources.context_orogenic_strength,
        context_ridge_direction_xyz=sources.context_ridge_direction_xyz,
        chunk_parent_ids=np.ascontiguousarray(replay_ids),
    )
    context_correction = np.asarray(root["conditioning/center_correction_m"][:], dtype=np.float64)
    correction = _interpolated_center_correction(
        replay_ids,
        sources.context_ids,
        context_correction,
        sources.parent_resolution,
        config.refinement_factor,
    )
    replay["elevation_m"] = (replay["elevation_m"].astype(np.float64) + correction).astype(
        np.float32
    )
    child_count = config.refinement_factor**2
    start = parent_start * child_count
    end = start + len(replay_ids) * child_count
    comparisons = (
        ("geometry/cell_id", "cell_id"),
        ("geometry/area_km2", "area_km2"),
        ("terrain/elevation_m", "elevation_m"),
        ("terrain/unresolved_relief_m", "unresolved_relief_m"),
    )
    return all(
        np.array_equal(np.asarray(root[path][start:end]), replay[name])
        for path, name in comparisons
    )


def _validate_terrain(
    root: Any,
    sources: _TerrainSources,
    config: L3TerrainConfig,
    chunk_stats: list[dict[str, int | float]],
    conditioning_metrics: Mapping[str, int | float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    factor = config.refinement_factor
    children_per_parent = factor * factor
    parent_count = len(sources.domain_ids)
    cell_count = parent_count * children_per_parent
    cell_ids = np.asarray(root["geometry/cell_id"][:], dtype=np.uint64)
    child_parent_ids = np.asarray(root["geometry/parent_l2_cell_id"][:], dtype=np.int32)
    child_area = np.asarray(root["geometry/area_km2"][:], dtype=np.float64)
    elevation = np.asarray(root["terrain/elevation_m"][:], dtype=np.float32)
    offsets = np.asarray(root["terrain/offset_from_l2_m"][:], dtype=np.float32)
    unresolved = np.asarray(root["terrain/unresolved_relief_m"][:], dtype=np.float32)
    inside_core = np.asarray(root["geometry/inside_catchment_core"][:], dtype=bool)
    inside_process_halo = np.asarray(root["geometry/inside_process_halo"][:], dtype=bool)
    outside_process = np.asarray(root["geometry/outside_process_domain"][:], dtype=bool)
    if any(
        len(array) != cell_count for array in (cell_ids, child_parent_ids, child_area, elevation)
    ):
        raise RuntimeError("L3 terrain arrays do not match the declared cell count")
    domain_context_rows = np.searchsorted(sources.context_ids, sources.domain_ids)
    source_area = sources.context_area_km2[domain_context_rows]
    source_elevation = sources.context_elevation_m[domain_context_rows]
    source_relief = sources.context_relief_m[domain_context_rows]
    grouped_parent = child_parent_ids.reshape(parent_count, children_per_parent)
    grouped_area = child_area.reshape(parent_count, children_per_parent)
    grouped_elevation = elevation.reshape(parent_count, children_per_parent)
    restricted_area = np.sum(grouped_area, axis=1)
    restricted_elevation = np.sum(grouped_area * grouped_elevation, axis=1) / restricted_area
    area_error = np.abs(restricted_area - source_area) / source_area
    elevation_error = restricted_elevation - source_elevation
    elevation_error_relief_fraction = np.abs(elevation_error) / np.maximum(source_relief, 50.0)
    parent_grouping_valid = bool(np.all(grouped_parent == sources.domain_ids[:, None]))
    role_count = (
        inside_core.astype(np.int8)
        + inside_process_halo.astype(np.int8)
        + outside_process.astype(np.int8)
    )
    domain_roles_valid = bool(
        np.all(role_count == 1)
        and np.all(
            inside_core.reshape(parent_count, children_per_parent)
            == sources.domain_inside_core[:, None]
        )
        and np.all(
            inside_process_halo.reshape(parent_count, children_per_parent)
            == sources.domain_inside_process_halo[:, None]
        )
        and np.all(
            outside_process.reshape(parent_count, children_per_parent)
            == sources.domain_outside_process[:, None]
        )
    )
    fine_rows = np.asarray(root["geometry/row"][:], dtype=np.int32)
    fine_columns = np.asarray(root["geometry/column"][:], dtype=np.int32)
    dense_window_cells = int(
        (np.max(fine_rows) - np.min(fine_rows) + 1)
        * (np.max(fine_columns) - np.min(fine_columns) + 1)
    )
    continuous_window_valid = dense_window_cells == cell_count
    actual_cell_size_m = math.sqrt(float(np.sum(child_area)) / cell_count) * 1_000.0
    cell_size_relative_error = abs(actual_cell_size_m - config.requested_cell_size_m) / float(
        config.requested_cell_size_m
    )
    seam = _seam_metrics(root, sources, config)
    tile_motif = _tile_motif_metrics(offsets, factor)
    missing_context = int(
        sum(int(stats["missing_context_neighbor_count"]) for stats in chunk_stats)
    )
    unique_ids = len(np.unique(cell_ids)) == cell_count
    all_finite = bool(
        np.all(np.isfinite(child_area))
        and np.all(np.isfinite(elevation))
        and np.all(np.isfinite(offsets))
        and np.all(np.isfinite(unresolved))
    )
    validation_array_bytes = sum(
        array.nbytes
        for array in (
            cell_ids,
            child_parent_ids,
            child_area,
            elevation,
            offsets,
            unresolved,
            inside_core,
            inside_process_halo,
            outside_process,
        )
    )
    estimated_peak_memory_bytes = int(
        validation_array_bytes
        + cell_count * 64
        + config.chunk_parent_count * children_per_parent * 96
    )
    maximum_peak_memory_bytes = int(config.maximum_peak_memory_gb * 1024**3)
    observed_peak_rss_bytes = _observed_peak_rss_bytes()
    measured_or_estimated_peak_bytes = max(estimated_peak_memory_bytes, observed_peak_rss_bytes)
    validation: dict[str, Any] = {
        "format_version": TERRAIN_FORMAT_VERSION,
        "model_version": TERRAIN_MODEL_VERSION,
        "target_id": sources.target_id,
        "source_target_valid": 1,
        "parent_count": parent_count,
        "cell_count": cell_count,
        "maximum_base_cell_count": config.maximum_base_cell_count,
        "cell_count_valid": int(cell_count <= config.maximum_base_cell_count),
        "refinement_factor": factor,
        "parent_face_resolution": sources.parent_resolution,
        "child_face_resolution": sources.parent_resolution * factor,
        "requested_cell_size_m": config.requested_cell_size_m,
        "actual_area_equivalent_cell_size_m": actual_cell_size_m,
        "cell_size_relative_error": cell_size_relative_error,
        "cell_size_valid": int(cell_size_relative_error <= config.maximum_cell_size_relative_error),
        "parent_grouping_valid": int(parent_grouping_valid),
        "terrain_window_continuous_valid": int(continuous_window_valid),
        "terrain_domain_roles_valid": int(domain_roles_valid),
        "hydrological_acceptance_scope": "catchment_core_only",
        "catchment_core_parent_count": int(np.count_nonzero(sources.domain_inside_core)),
        "catchment_core_cell_count": int(np.count_nonzero(inside_core)),
        "process_halo_parent_count": int(np.count_nonzero(sources.domain_inside_process_halo)),
        "process_halo_cell_count": int(np.count_nonzero(inside_process_halo)),
        "outside_process_parent_count": int(np.count_nonzero(sources.domain_outside_process)),
        "outside_process_cell_count": int(np.count_nonzero(outside_process)),
        "unique_stable_ids_valid": int(unique_ids),
        "uint64_ids_required": int(int(np.max(cell_ids)) > np.iinfo(np.uint32).max),
        "all_finite_valid": int(all_finite),
        "estimated_peak_memory_bytes": estimated_peak_memory_bytes,
        "observed_process_peak_rss_bytes": observed_peak_rss_bytes,
        "memory_gate_peak_bytes": measured_or_estimated_peak_bytes,
        "maximum_peak_memory_bytes": maximum_peak_memory_bytes,
        "memory_budget_valid": int(measured_or_estimated_peak_bytes <= maximum_peak_memory_bytes),
        "missing_context_neighbor_count": missing_context,
        "context_boundary_valid": int(missing_context == 0),
        "maximum_parent_area_relative_error": float(np.max(area_error)),
        "maximum_allowed_parent_area_relative_error": config.maximum_parent_area_relative_error,
        "parent_area_conservation_valid": int(
            float(np.max(area_error)) <= config.maximum_parent_area_relative_error
        ),
        "maximum_parent_mean_elevation_error_m": float(np.max(np.abs(elevation_error))),
        "maximum_allowed_parent_mean_elevation_error_m": config.maximum_parent_mean_error_m,
        "maximum_parent_mean_elevation_error_relief_fraction": float(
            np.max(elevation_error_relief_fraction)
        ),
        "maximum_allowed_parent_mean_elevation_error_relief_fraction": (
            config.maximum_parent_mean_error_relief_fraction
        ),
        "parent_mean_conditioning_valid": int(
            float(np.max(np.abs(elevation_error))) <= config.maximum_parent_mean_error_m
            and float(np.max(elevation_error_relief_fraction))
            <= config.maximum_parent_mean_error_relief_fraction
        ),
        "terrain_elevation_min_m": float(np.min(elevation)),
        "terrain_elevation_p05_m": _percentile(elevation, 5.0),
        "terrain_elevation_p50_m": _percentile(elevation, 50.0),
        "terrain_elevation_p95_m": _percentile(elevation, 95.0),
        "terrain_elevation_max_m": float(np.max(elevation)),
        "terrain_offset_std_m": float(np.std(offsets)),
        "terrain_offset_p05_m": _percentile(offsets, 5.0),
        "terrain_offset_p95_m": _percentile(offsets, 95.0),
        "terrain_variation_valid": int(
            float(np.std(offsets)) >= config.minimum_terrain_offset_std_m
        ),
        "unresolved_relief_p50_m": _percentile(unresolved, 50.0),
        "unresolved_relief_p95_m": _percentile(unresolved, 95.0),
        "chunk_replay_identical": int(_chunk_replay_valid(root, sources, config)),
        **conditioning_metrics,
        **seam,
        **tile_motif,
        "l2_parent_boundary_continuity_valid": int(
            seam["l2_parent_boundary_residual_p95_ratio"]
            <= config.maximum_parent_boundary_residual_p95_ratio
        ),
        "chunk_boundary_continuity_valid": int(
            seam["chunk_boundary_residual_p95_ratio"]
            <= config.maximum_chunk_boundary_residual_p95_ratio
        ),
        "maximum_parent_boundary_residual_p95_ratio": (
            config.maximum_parent_boundary_residual_p95_ratio
        ),
        "maximum_chunk_boundary_residual_p95_ratio": (
            config.maximum_chunk_boundary_residual_p95_ratio
        ),
        "maximum_tile_bubble_correlation_p50": config.maximum_tile_bubble_correlation_p50,
        "maximum_tile_bubble_correlation_p95": config.maximum_tile_bubble_correlation_p95,
        "parent_tile_motif_valid": int(
            tile_motif["tile_bubble_absolute_correlation_p50"]
            <= config.maximum_tile_bubble_correlation_p50
            and tile_motif["tile_bubble_absolute_correlation_p95"]
            <= config.maximum_tile_bubble_correlation_p95
        ),
    }
    required = (
        "source_target_valid",
        "cell_count_valid",
        "cell_size_valid",
        "parent_grouping_valid",
        "terrain_window_continuous_valid",
        "terrain_domain_roles_valid",
        "unique_stable_ids_valid",
        "uint64_ids_required",
        "all_finite_valid",
        "memory_budget_valid",
        "context_boundary_valid",
        "parent_area_conservation_valid",
        "parent_mean_conditioning_valid",
        "terrain_variation_valid",
        "chunk_replay_identical",
        "l2_parent_boundary_continuity_valid",
        "chunk_boundary_continuity_valid",
        "parent_tile_motif_valid",
        "center_correction_bounded_valid",
        "conditioning_converged",
    )
    validation["passed"] = bool(all(validation[name] == 1 for name in required))
    parent_arrays = {
        "restricted_area_km2": restricted_area,
        "area_relative_error": area_error,
        "restricted_elevation_m": restricted_elevation,
        "elevation_error_m": elevation_error,
    }
    return validation, parent_arrays


def _terrain_colors(elevation: np.ndarray) -> np.ndarray:
    stops = np.asarray([-600.0, -1.0, 0.0, 120.0, 350.0, 700.0, 1_200.0, 2_000.0])
    colors = np.asarray(
        [
            [38, 84, 112],
            [73, 121, 139],
            [91, 137, 91],
            [116, 153, 91],
            [149, 157, 101],
            [151, 133, 101],
            [133, 116, 103],
            [222, 220, 211],
        ],
        dtype=np.float32,
    )
    output = np.empty((*elevation.shape, 3), dtype=np.float32)
    clipped = np.clip(elevation, stops[0], stops[-1])
    for channel in range(3):
        output[..., channel] = np.interp(clipped, stops, colors[:, channel])
    return output


def _diagnostic_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _inner_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, mode="constant", constant_values=False)
    interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return mask & ~interior


def _nice_scale_km(map_width_pixels: int, cell_size_m: float) -> float:
    maximum_km = max(map_width_pixels * 0.28 * cell_size_m / 1_000.0, 0.001)
    exponent = math.floor(math.log10(maximum_km))
    for multiplier in (5.0, 2.0, 1.0):
        candidate = multiplier * 10.0**exponent
        if candidate <= maximum_km:
            return candidate
    return 10.0 ** (exponent - 1)


def _render_terrain(
    root: Any,
    path: Path,
    actual_cell_size_m: float,
    *,
    show_domain: bool = False,
) -> None:
    rows = np.asarray(root["geometry/row"][:], dtype=np.int32)
    columns = np.asarray(root["geometry/column"][:], dtype=np.int32)
    elevation = np.asarray(root["terrain/elevation_m"][:], dtype=np.float32)
    inside_core = np.asarray(root["geometry/inside_catchment_core"][:], dtype=bool)
    inside_halo = np.asarray(root["geometry/inside_process_halo"][:], dtype=bool)
    outside_process = np.asarray(root["geometry/outside_process_domain"][:], dtype=bool)
    min_row, max_row = int(np.min(rows)), int(np.max(rows))
    min_col, max_col = int(np.min(columns)), int(np.max(columns))
    dense = np.full((max_row - min_row + 1, max_col - min_col + 1), np.nan, dtype=np.float32)
    dense[rows - min_row, columns - min_col] = elevation
    valid = np.isfinite(dense)
    if not np.all(valid):
        raise RuntimeError("L3 terrain diagnostic refuses a window with internal no-data holes")
    dense_core = np.zeros_like(valid)
    dense_halo = np.zeros_like(valid)
    dense_outside = np.zeros_like(valid)
    dense_core[rows - min_row, columns - min_col] = inside_core
    dense_halo[rows - min_row, columns - min_col] = inside_halo
    dense_outside[rows - min_row, columns - min_col] = outside_process
    padded = np.pad(dense, 1, mode="constant", constant_values=np.nan)
    center = padded[1:-1, 1:-1]
    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]
    north = np.where(np.isfinite(north), north, center)
    south = np.where(np.isfinite(south), south, center)
    west = np.where(np.isfinite(west), west, center)
    east = np.where(np.isfinite(east), east, center)
    dzdx = (east - west) / (2.0 * actual_cell_size_m)
    dzdy = (south - north) / (2.0 * actual_cell_size_m)
    normal_x = -dzdx
    normal_y = -dzdy
    normal_z = np.ones_like(dense)
    norm = np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
    illumination = (normal_x * -0.45 + normal_y * -0.55 + normal_z * 0.70) / norm
    shade = np.clip(0.58 + 0.48 * illumination, 0.42, 1.08)
    colors = _terrain_colors(dense) * shade[..., None]
    if show_domain:
        colors[dense_outside] = (
            colors[dense_outside] * 0.76 + np.asarray([214.0, 216.0, 210.0]) * 0.24
        )
        process_boundary = _inner_boundary(dense_core | dense_halo)
        core_boundary = _inner_boundary(dense_core)
        colors[process_boundary] = np.asarray([180.0, 177.0, 162.0])
        colors[core_boundary] = np.asarray([32.0, 35.0, 31.0])
    map_image = Image.fromarray(np.clip(colors, 0, 255).astype(np.uint8), mode="RGB")

    title_height = 50
    footer_height = 74
    legend_width = 270
    canvas = Image.new(
        "RGB",
        (map_image.width + legend_width, map_image.height + title_height + footer_height),
        (240, 241, 237),
    )
    canvas.paste(map_image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    title_font = _diagnostic_font(22)
    label_font = _diagnostic_font(17)
    small_font = _diagnostic_font(15)
    title = "L3 conditioned terrain and process domain" if show_domain else "L3 conditioned terrain"
    draw.text((18, 13), title, fill=(25, 30, 27), font=title_font)
    draw.line((map_image.width, 0, map_image.width, canvas.height), fill=(178, 181, 174), width=1)

    legend_x = map_image.width + 24
    draw.text((legend_x, 20), "Elevation", fill=(25, 30, 27), font=title_font)
    gradient_top = 68
    gradient_height = min(320, max(160, map_image.height // 4))
    gradient_values = np.linspace(2_000.0, -600.0, gradient_height, dtype=np.float32)[:, None]
    gradient_rgb = _terrain_colors(gradient_values)
    gradient = Image.fromarray(np.clip(gradient_rgb, 0, 255).astype(np.uint8), mode="RGB")
    gradient = gradient.resize((34, gradient_height))
    canvas.paste(gradient, (legend_x, gradient_top))
    draw.rectangle(
        (legend_x, gradient_top, legend_x + 34, gradient_top + gradient_height),
        outline=(70, 74, 70),
        width=1,
    )
    for value in (2_000, 1_000, 500, 0, -600):
        y = gradient_top + round((2_000 - value) / 2_600 * gradient_height)
        draw.line((legend_x + 34, y, legend_x + 42, y), fill=(50, 54, 51), width=1)
        draw.text((legend_x + 48, y - 8), f"{value:,} m", fill=(35, 39, 36), font=small_font)

    if show_domain:
        role_y = gradient_top + gradient_height + 38
        draw.text((legend_x, role_y), "Domain", fill=(25, 30, 27), font=label_font)
        role_y += 34
        draw.line((legend_x, role_y, legend_x + 36, role_y), fill=(32, 35, 31), width=4)
        draw.text(
            (legend_x + 48, role_y - 9),
            "Inherited core (L0)",
            fill=(35, 39, 36),
            font=small_font,
        )
        role_y += 34
        draw.line((legend_x, role_y, legend_x + 36, role_y), fill=(180, 177, 162), width=4)
        draw.text(
            (legend_x + 48, role_y - 9),
            "Process halo limit",
            fill=(35, 39, 36),
            font=small_font,
        )
        role_y += 34
        draw.rectangle((legend_x, role_y - 9, legend_x + 36, role_y + 9), fill=(196, 198, 193))
        draw.text(
            (legend_x + 48, role_y - 9),
            "Outside process",
            fill=(35, 39, 36),
            font=small_font,
        )

    scale_km = _nice_scale_km(map_image.width, actual_cell_size_m)
    scale_pixels = max(1, round(scale_km * 1_000.0 / actual_cell_size_m))
    scale_x = 24
    scale_y = title_height + map_image.height + 30
    segment_count = 4
    for segment in range(segment_count):
        left = scale_x + round(scale_pixels * segment / segment_count)
        right = scale_x + round(scale_pixels * (segment + 1) / segment_count)
        fill = (32, 35, 31) if segment % 2 == 0 else (240, 241, 237)
        draw.rectangle((left, scale_y, right, scale_y + 12), fill=fill, outline=(32, 35, 31))
    draw.text((scale_x, scale_y + 18), "0", fill=(25, 30, 27), font=small_font)
    draw.text(
        (scale_x + scale_pixels, scale_y + 18),
        f"{scale_km:g} km",
        fill=(25, 30, 27),
        font=small_font,
        anchor="ra",
    )
    draw.text(
        (scale_x + scale_pixels + 20, scale_y - 2),
        "approximate scale",
        fill=(70, 74, 70),
        font=small_font,
    )
    canvas.save(path, optimize=True)


def _write_parent_table(
    path: Path,
    sources: _TerrainSources,
    config: L3TerrainConfig,
    parent_arrays: Mapping[str, np.ndarray],
) -> None:
    context_rows = np.searchsorted(sources.context_ids, sources.domain_ids)
    child_count = config.refinement_factor**2
    table = pa.table(
        {
            "l2_cell_id": pa.array(sources.domain_ids, type=pa.int32()),
            "handoff_child_row": pa.array(sources.domain_handoff_rows, type=pa.int32()),
            "l3_row_offset": pa.array(
                np.arange(len(sources.domain_ids), dtype=np.int64) * child_count,
                type=pa.int64(),
            ),
            "l3_child_count": pa.array(
                np.full(len(sources.domain_ids), child_count, dtype=np.int32), type=pa.int32()
            ),
            "inside_catchment_core": pa.array(sources.domain_inside_core, type=pa.bool_()),
            "inside_process_halo": pa.array(sources.domain_inside_process_halo, type=pa.bool_()),
            "outside_process_domain": pa.array(sources.domain_outside_process, type=pa.bool_()),
            "source_area_km2": pa.array(sources.context_area_km2[context_rows], type=pa.float64()),
            "restricted_area_km2": pa.array(
                parent_arrays["restricted_area_km2"], type=pa.float64()
            ),
            "area_relative_error": pa.array(
                parent_arrays["area_relative_error"], type=pa.float64()
            ),
            "source_elevation_m": pa.array(
                sources.context_elevation_m[context_rows], type=pa.float32()
            ),
            "raw_parent_mean_error_m": pa.array(
                parent_arrays["raw_parent_mean_error_m"], type=pa.float64()
            ),
            "center_correction_m": pa.array(
                parent_arrays["center_correction_m"], type=pa.float64()
            ),
            "restricted_elevation_m": pa.array(
                parent_arrays["restricted_elevation_m"].astype(np.float32), type=pa.float32()
            ),
            "elevation_error_m": pa.array(
                parent_arrays["elevation_error_m"].astype(np.float32), type=pa.float32()
            ),
            "source_relief_m": pa.array(sources.context_relief_m[context_rows], type=pa.float32()),
            "rock_strength": pa.array(
                sources.context_rock_strength[context_rows], type=pa.float32()
            ),
            "orogenic_strength": pa.array(
                sources.context_orogenic_strength[context_rows], type=pa.float32()
            ),
            "lake_fraction_prior": pa.array(sources.domain_lake_fraction, type=pa.float32()),
            "wetland_fraction_prior": pa.array(sources.domain_wetland_fraction, type=pa.float32()),
            "ocean_fraction_prior": pa.array(sources.domain_ocean_fraction, type=pa.float32()),
        }
    )
    pq.write_table(table, path, compression="zstd")


def _verify_published_outputs(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RuntimeError("published L3 terrain manifest has no output checksums")
    for name, raw_record in outputs.items():
        if not isinstance(raw_record, Mapping) or not isinstance(raw_record.get("path"), str):
            raise RuntimeError(f"published L3 terrain manifest has an invalid {name} record")
        path = output_dir / str(raw_record["path"])
        if "sha256_tree" in raw_record:
            if not path.is_dir() or _tree_checksum(path) != raw_record["sha256_tree"]:
                raise RuntimeError(f"published L3 terrain integrity check failed for {name}")
        elif "sha256" in raw_record:
            if not path.is_file() or _file_checksum(path) != raw_record["sha256"]:
                raise RuntimeError(f"published L3 terrain integrity check failed for {name}")
        else:
            raise RuntimeError(f"published L3 terrain output {name} has no checksum")


def _existing_result(
    config: L3TerrainConfig,
    sources: _TerrainSources,
    run_fingerprint: str,
) -> L3TerrainResult | None:
    manifest_path = config.output_dir / "manifest.json"
    validation_path = config.output_dir / "validation.json"
    if not manifest_path.exists() or not validation_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        validation = json.loads(validation_path.read_text(encoding="utf8"))
    except json.JSONDecodeError:
        return None
    if manifest.get("run_fingerprint") != run_fingerprint or not validation.get("passed"):
        return None
    _verify_published_outputs(config.output_dir, manifest)
    return L3TerrainResult(
        output_dir=config.output_dir,
        manifest_path=manifest_path,
        validation_path=validation_path,
        zarr_path=config.output_dir / "terrain.zarr",
        preview_path=config.output_dir / manifest["outputs"]["terrain_preview"]["path"],
        target_id=sources.target_id,
        parent_count=int(validation["parent_count"]),
        cell_count=int(validation["cell_count"]),
        actual_cell_size_m=float(validation["actual_area_equivalent_cell_size_m"]),
        chunk_count=int(manifest["chunking"]["chunk_count"]),
        resumed_chunk_count=int(manifest["chunking"].get("resumed_chunk_count", 0)),
    )


def generate_l3_terrain(config: L3TerrainConfig) -> L3TerrainResult:
    """Generate or resume one deterministic L3 base-terrain artifact."""

    config.validate()
    sources = _load_sources(config)
    run_fingerprint, fingerprint_components = _run_fingerprint(config, sources)
    existing = _existing_result(config, sources, run_fingerprint)
    if existing is not None:
        return existing
    expected_cells = len(sources.domain_ids) * config.refinement_factor**2
    if expected_cells > config.maximum_base_cell_count:
        raise RuntimeError(
            f"L3 terrain requires {expected_cells} cells; maximum is "
            f"{config.maximum_base_cell_count}"
        )
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = config.output_dir.with_name(f".{config.output_dir.name}.partial")
    root, resumed = _open_partial(partial, config, sources, run_fingerprint)
    zarr_path = partial / "terrain.zarr"
    complete = root["progress/chunk_complete"]
    resumed_chunk_count = (
        int(np.count_nonzero(np.asarray(complete[:], dtype=bool))) if resumed else 0
    )
    children_per_parent = config.refinement_factor**2
    chunk_count = len(complete)
    started = time.perf_counter()
    controls = _controls(config, sources)
    for chunk_index in range(chunk_count):
        if bool(complete[chunk_index]):
            continue
        parent_start = chunk_index * config.chunk_parent_count
        parent_end = min(parent_start + config.chunk_parent_count, len(sources.domain_ids))
        chunk_parent_ids = np.ascontiguousarray(sources.domain_ids[parent_start:parent_end])
        outputs, stats = run_l3_terrain_chunk(
            controls=controls,
            context_parent_ids=sources.context_ids,
            context_elevation_m=sources.context_elevation_m,
            context_relief_m=sources.context_relief_m,
            context_rock_strength=sources.context_rock_strength,
            context_orogenic_strength=sources.context_orogenic_strength,
            context_ridge_direction_xyz=sources.context_ridge_direction_xyz,
            chunk_parent_ids=chunk_parent_ids,
        )
        if int(stats["missing_context_neighbor_count"]) != 0:
            raise RuntimeError(
                f"L3 chunk {chunk_index} lacks "
                f"{stats['missing_context_neighbor_count']} required context neighbors"
            )
        row_start = parent_start * children_per_parent
        row_end = parent_end * children_per_parent
        _write_chunk(
            root,
            row_start,
            row_end,
            outputs,
            inside_core=np.repeat(
                sources.domain_inside_core[parent_start:parent_end], children_per_parent
            ),
            inside_process_halo=np.repeat(
                sources.domain_inside_process_halo[parent_start:parent_end], children_per_parent
            ),
            outside_process=np.repeat(
                sources.domain_outside_process[parent_start:parent_end], children_per_parent
            ),
        )
        _sync_zarr_chunk(zarr_path, root, RAW_CHUNK_ARRAY_PATHS, chunk_index)
        stats.update(
            {
                "chunk_index": chunk_index,
                "parent_start": parent_start,
                "parent_end": parent_end,
                "row_start": row_start,
                "row_end": row_end,
            }
        )
        _write_json_durable(partial / "chunk_stats" / f"{chunk_index:05d}.json", stats)
        complete[chunk_index] = True
        _sync_zarr_chunk(
            zarr_path,
            root,
            ("progress/chunk_complete",),
            chunk_index // int(complete.chunks[0]),
        )
    if not bool(np.all(np.asarray(complete[:], dtype=bool))):
        raise RuntimeError("L3 terrain generation ended with incomplete chunks")
    conditioning_path = partial / "conditioning.json"
    if not bool(root.attrs.get("center_corrections_solved", False)):
        context_correction, conditioning_metrics = _solve_center_corrections(root, sources, config)
        root["conditioning/center_correction_m"][:] = context_correction
        _sync_zarr_array(zarr_path, "conditioning/raw_parent_mean_error_m")
        _sync_zarr_array(zarr_path, "conditioning/center_correction_m")
        _write_json_durable(conditioning_path, conditioning_metrics)
        root.attrs["center_corrections_solved"] = True
        _fsync_paths([zarr_path / ".zattrs"])
    else:
        conditioning_metrics = json.loads(conditioning_path.read_text(encoding="utf8"))
    conditioned = root["progress/chunk_conditioned"]
    for chunk_index in range(chunk_count):
        if bool(conditioned[chunk_index]):
            continue
        parent_start = chunk_index * config.chunk_parent_count
        parent_end = min(parent_start + config.chunk_parent_count, len(sources.domain_ids))
        _condition_chunk(root, sources, config, parent_start, parent_end)
        _sync_zarr_chunk(zarr_path, root, CONDITIONED_CHUNK_ARRAY_PATHS, chunk_index)
        conditioned[chunk_index] = True
        _sync_zarr_chunk(
            zarr_path,
            root,
            ("progress/chunk_conditioned",),
            chunk_index // int(conditioned.chunks[0]),
        )
    if not bool(np.all(np.asarray(conditioned[:], dtype=bool))):
        raise RuntimeError("L3 terrain conditioning ended with incomplete chunks")
    chunk_stats = [
        json.loads((partial / "chunk_stats" / f"{index:05d}.json").read_text(encoding="utf8"))
        for index in range(chunk_count)
    ]
    validation, parent_arrays = _validate_terrain(
        root, sources, config, chunk_stats, conditioning_metrics
    )
    domain_context_rows = np.searchsorted(sources.context_ids, sources.domain_ids)
    parent_arrays["raw_parent_mean_error_m"] = np.asarray(
        root["conditioning/raw_parent_mean_error_m"][:], dtype=np.float64
    )
    parent_arrays["center_correction_m"] = np.asarray(
        root["conditioning/center_correction_m"][:], dtype=np.float64
    )[domain_context_rows]
    tables_dir = partial / "tables"
    tables_dir.mkdir(exist_ok=True)
    parent_table_path = tables_dir / "l2_parent_conditioning.parquet"
    _write_parent_table(parent_table_path, sources, config, parent_arrays)
    chunk_table_path = tables_dir / "chunks.parquet"
    pq.write_table(pa.Table.from_pylist(chunk_stats), chunk_table_path, compression="zstd")
    preview_path = partial / "terrain.png"
    _render_terrain(
        root,
        preview_path,
        float(validation["actual_area_equivalent_cell_size_m"]),
    )
    domain_preview_path = partial / "terrain_domain.png"
    _render_terrain(
        root,
        domain_preview_path,
        float(validation["actual_area_equivalent_cell_size_m"]),
        show_domain=True,
    )
    root.attrs["status"] = "complete"
    root.attrs["actual_area_equivalent_cell_size_m"] = validation[
        "actual_area_equivalent_cell_size_m"
    ]
    zarr.consolidate_metadata(str(partial / "terrain.zarr"))
    storage_bytes = sum(path.stat().st_size for path in partial.rglob("*") if path.is_file())
    validation["storage_bytes"] = storage_bytes
    validation["maximum_storage_bytes"] = int(config.maximum_storage_gb * 1024**3)
    validation["storage_budget_valid"] = int(
        storage_bytes <= int(config.maximum_storage_gb * 1024**3)
    )
    validation["passed"] = bool(validation["passed"] and validation["storage_budget_valid"] == 1)
    validation_path = partial / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf8")
    if not validation["passed"]:
        raise RuntimeError(f"L3 terrain failed validation: {validation}")
    elapsed_seconds = time.perf_counter() - started
    manifest = {
        "format_version": TERRAIN_FORMAT_VERSION,
        "model_version": TERRAIN_MODEL_VERSION,
        "status": "complete",
        "target_id": sources.target_id,
        "run_fingerprint": run_fingerprint,
        "hierarchy": {
            "topology": "continuous rectangular parent-major window on one cubed-sphere face",
            "parent_level": "L2",
            "child_level": "L3",
            "parent_face_resolution": sources.parent_resolution,
            "child_face_resolution": sources.parent_resolution * config.refinement_factor,
            "refinement_factor": config.refinement_factor,
            "children_per_parent": children_per_parent,
            "parent_count": len(sources.domain_ids),
            "catchment_core_parent_count": int(np.count_nonzero(sources.domain_inside_core)),
            "process_halo_parent_count": int(np.count_nonzero(sources.domain_inside_process_halo)),
            "outside_process_parent_count": int(np.count_nonzero(sources.domain_outside_process)),
            "cell_count": expected_cells,
            "requested_cell_size_m": config.requested_cell_size_m,
            "actual_area_equivalent_cell_size_m": validation["actual_area_equivalent_cell_size_m"],
            "cell_id_dtype": "uint64",
            "domain_masks": [
                "geometry/inside_catchment_core",
                "geometry/inside_process_halo",
                "geometry/outside_process_domain",
            ],
            "hydrological_acceptance_scope": "catchment_core_only",
        },
        "chunking": {
            "parent_aligned": True,
            "chunk_parent_count": config.chunk_parent_count,
            "chunk_rows": config.chunk_parent_count * children_per_parent,
            "chunk_count": chunk_count,
            "resumed": resumed,
            "resumed_chunk_count": resumed_chunk_count,
            "elapsed_seconds_this_run": elapsed_seconds,
        },
        "terrain_model": {
            "seed": config.terrain_seed,
            "relief_realization_fraction": config.relief_realization_fraction,
            "base_wavelength_m": config.base_wavelength_m,
            "effective_longest_residual_wavelength_m": (config.base_wavelength_m / 4.0),
            "octave_count": config.octave_count,
            "persistence": config.persistence,
            "domain_warp_fraction": config.domain_warp_fraction,
            "orogenic_ridge_fraction": config.orogenic_ridge_fraction,
            "conditioning": [
                "L2 terrain mean",
                "L2 context values",
                "unresolved relief",
                "L0 rock strength prior",
                "L0 orogenic amplitude and tangent orientation prior",
            ],
            "mean_conditioning": (
                "globally continuous center-correction field constrained within "
                "absolute and relief-relative L2 tolerances"
            ),
            "conditioning_iteration_count": conditioning_metrics["conditioning_iteration_count"],
            "conditioning_converged": bool(conditioning_metrics["conditioning_converged"]),
            "remaining_relief_semantics": "sub-200m prior, not added wholesale to elevation",
        },
        "source": {
            "target_dir": str(config.target_dir),
            "handoff_dir": str(sources.handoff_dir),
            "target_manifest_sha256": fingerprint_components["target_manifest_sha256"],
            "handoff_manifest_sha256": fingerprint_components["handoff_manifest_sha256"],
            "config": str(config.source_config) if config.source_config else None,
            "config_sha256": fingerprint_components["config_sha256"],
            "native_abi_version": fingerprint_components["native_abi_version"],
            "native_sha256": fingerprint_components["native_sha256"],
            "orchestrator_sha256": fingerprint_components["orchestrator_sha256"],
            "binding_sha256": fingerprint_components["binding_sha256"],
        },
        "outputs": {
            "terrain_zarr": {
                "path": "terrain.zarr",
                "sha256_tree": _tree_checksum(partial / "terrain.zarr"),
            },
            "parent_conditioning": {
                "path": "tables/l2_parent_conditioning.parquet",
                "rows": len(sources.domain_ids),
                "sha256": _file_checksum(parent_table_path),
            },
            "chunks": {
                "path": "tables/chunks.parquet",
                "rows": chunk_count,
                "sha256": _file_checksum(chunk_table_path),
            },
            "conditioning": {
                "path": "conditioning.json",
                "sha256": _file_checksum(conditioning_path),
            },
            "terrain_preview": {
                "path": "terrain.png",
                "sha256": _file_checksum(preview_path),
                "projection": "native cubed-sphere face raster diagnostic",
                "legend": "elevation metres",
                "scale": "approximate kilometre scale from area-equivalent cell width",
            },
            "terrain_domain_preview": {
                "path": "terrain_domain.png",
                "sha256": _file_checksum(domain_preview_path),
                "projection": "native cubed-sphere face raster diagnostic",
                "legend": "elevation metres and domain-role boundaries",
                "scale": "approximate kilometre scale from area-equivalent cell width",
            },
            "validation": {
                "path": "validation.json",
                "sha256": _file_checksum(validation_path),
            },
        },
        "validation_passed": True,
    }
    manifest_path = partial / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf8")
    _replace_directory(partial, config.output_dir)
    return L3TerrainResult(
        output_dir=config.output_dir,
        manifest_path=config.output_dir / "manifest.json",
        validation_path=config.output_dir / "validation.json",
        zarr_path=config.output_dir / "terrain.zarr",
        preview_path=config.output_dir / "terrain.png",
        target_id=sources.target_id,
        parent_count=len(sources.domain_ids),
        cell_count=expected_cells,
        actual_cell_size_m=float(validation["actual_area_equivalent_cell_size_m"]),
        chunk_count=chunk_count,
        resumed_chunk_count=resumed_chunk_count,
    )


__all__ = [
    "L3TerrainConfig",
    "L3TerrainResult",
    "TERRAIN_FORMAT_VERSION",
    "TERRAIN_MODEL_VERSION",
    "generate_l3_terrain",
]
