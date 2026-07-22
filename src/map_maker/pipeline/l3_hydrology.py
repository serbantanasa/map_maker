"""Depression-aware monthly hydrology for one continuous L3 terrain window."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from PIL import Image, ImageDraw
import yaml  # type: ignore[import-untyped]
import zarr  # type: ignore[import-untyped]

from .._native import native_library_info
from ._hydrology_native import run_regional_hydrology
from .l3_terrain import (
    TERRAIN_DIAGNOSTIC_VERTICAL_EXAGGERATION,
    _diagnostic_font,
    _file_checksum,
    _fsync_paths,
    _inner_boundary,
    _nice_scale_km,
    _sync_zarr_array,
    _terrain_colors,
    _tree_checksum,
    _write_json_durable,
    _zarr_dataset,
)
from .regional_handoff import _replace_directory

HYDROLOGY_FORMAT_VERSION = 1
HYDROLOGY_MODEL_VERSION = "l3_depression_hydrology_v4"
MONTHS = 12
SECONDS_PER_YEAR = 365.2425 * 86_400.0
SECONDS_PER_MONTH = SECONDS_PER_YEAR / MONTHS

TERMINAL_NONE = 0
TERMINAL_PHYSICAL_OCEAN = 1
TERMINAL_PROCESS_BOUNDARY = 2
TERMINAL_REGISTERED_OUTLET = 3


@dataclass(frozen=True)
class L3HydrologyConfig:
    target_dir: Path
    terrain_dir: Path
    output_dir: Path
    chunk_rows: int = 262_144
    orographic_runoff_weight: float = 0.28
    orographic_precipitation_weight: float = 0.20
    highland_melt_weight: float = 0.22
    minimum_depression_depth_m: float = 2.0
    wetland_mean_depth_m: float = 2.5
    endorheic_aridity_threshold: float = 0.35
    maximum_fill_time_years: float = 10_000.0
    lake_seepage_mm_year: float = 30.0
    subgrid_relief_scale: float = 0.35
    subgrid_connected_basin_fraction: float = 1.0
    breach_score_threshold: float = 0.62
    maximum_breach_incision_m: float = 80.0
    maximum_cumulative_breach_incision_m: float = 160.0
    breach_length_cells: int = 12
    maximum_breach_rounds: int = 8
    river_discharge_threshold_m3s: float = 2.0
    river_contributing_area_threshold_km2: float = 20.0
    river_minimum_discharge_m3s: float = 0.15
    maximum_forcing_conservation_relative_error: float = 1e-6
    maximum_runoff_conservation_relative_error: float = 1e-6
    minimum_inherited_core_retained_fraction: float = 0.80
    minimum_routed_core_inherited_fraction: float = 0.80
    minimum_routed_to_inherited_area_ratio: float = 0.80
    maximum_routed_to_inherited_area_ratio: float = 1.20
    maximum_routed_core_process_boundary_contact_fraction: float = 0.001
    maximum_outlet_hydrograph_relative_error: float = 0.40
    maximum_core_open_water_fraction: float = 0.10
    maximum_peak_memory_gb: float = 24.0
    maximum_storage_gb: float = 4.0
    source_config: Path | None = None

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        output_dir: Path | None = None,
    ) -> "L3HydrologyConfig":
        source = Path(path).expanduser().resolve()
        data = yaml.safe_load(source.read_text(encoding="utf8"))
        if not isinstance(data, Mapping):
            raise TypeError("L3 hydrology config must contain a mapping")
        raw_target = data.get("output_dir")
        raw_terrain = data.get("terrain_output_dir")
        raw_output = data.get("hydrology_output_dir")
        if not raw_target or not raw_terrain or not raw_output:
            raise ValueError(
                "L3 hydrology requires output_dir, terrain_output_dir, and hydrology_output_dir"
            )
        controls = data.get("hydrology", {})
        limits = data.get("limits", {})
        if not isinstance(controls, Mapping) or not isinstance(limits, Mapping):
            raise TypeError("L3 hydrology and limits controls must be mappings")
        known = {
            "chunk_rows",
            "orographic_runoff_weight",
            "orographic_precipitation_weight",
            "highland_melt_weight",
            "minimum_depression_depth_m",
            "wetland_mean_depth_m",
            "endorheic_aridity_threshold",
            "maximum_fill_time_years",
            "lake_seepage_mm_year",
            "subgrid_relief_scale",
            "subgrid_connected_basin_fraction",
            "breach_score_threshold",
            "maximum_breach_incision_m",
            "maximum_cumulative_breach_incision_m",
            "breach_length_cells",
            "maximum_breach_rounds",
            "river_discharge_threshold_m3s",
            "river_contributing_area_threshold_km2",
            "river_minimum_discharge_m3s",
            "maximum_forcing_conservation_relative_error",
            "maximum_runoff_conservation_relative_error",
            "minimum_inherited_core_retained_fraction",
            "minimum_routed_core_inherited_fraction",
            "minimum_routed_to_inherited_area_ratio",
            "maximum_routed_to_inherited_area_ratio",
            "maximum_routed_core_process_boundary_contact_fraction",
            "maximum_outlet_hydrograph_relative_error",
            "maximum_core_open_water_fraction",
        }
        unknown = set(controls) - known
        if unknown:
            raise ValueError(f"Unknown L3 hydrology controls: {', '.join(sorted(unknown))}")

        integer_names = {"chunk_rows", "breach_length_cells", "maximum_breach_rounds"}
        values: dict[str, int | float] = {}
        for name in known:
            if name in controls:
                values[name] = (
                    int(controls[name]) if name in integer_names else float(controls[name])
                )
        config = cls(
            target_dir=(source.parent / str(raw_target)).resolve(),
            terrain_dir=(source.parent / str(raw_terrain)).resolve(),
            output_dir=(
                output_dir.expanduser().resolve()
                if output_dir is not None
                else (source.parent / str(raw_output)).resolve()
            ),
            maximum_peak_memory_gb=float(limits.get("maximum_peak_memory_gb", 24.0)),
            maximum_storage_gb=float(limits.get("maximum_hydrology_storage_gb", 4.0)),
            source_config=source,
            **values,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 16_384 <= self.chunk_rows <= 1_048_576:
            raise ValueError("hydrology.chunk_rows must be in [16384, 1048576]")
        positive = (
            "minimum_depression_depth_m",
            "wetland_mean_depth_m",
            "maximum_fill_time_years",
            "subgrid_relief_scale",
            "maximum_breach_incision_m",
            "maximum_cumulative_breach_incision_m",
            "river_discharge_threshold_m3s",
            "river_contributing_area_threshold_km2",
            "river_minimum_discharge_m3s",
            "maximum_forcing_conservation_relative_error",
            "maximum_runoff_conservation_relative_error",
            "maximum_outlet_hydrograph_relative_error",
            "maximum_peak_memory_gb",
            "maximum_storage_gb",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"hydrology.{name} must be finite and positive")
        fractions = (
            "orographic_runoff_weight",
            "orographic_precipitation_weight",
            "highland_melt_weight",
            "subgrid_connected_basin_fraction",
            "breach_score_threshold",
            "minimum_inherited_core_retained_fraction",
            "minimum_routed_core_inherited_fraction",
            "maximum_routed_core_process_boundary_contact_fraction",
            "maximum_core_open_water_fraction",
        )
        for name in fractions:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"hydrology.{name} must be finite and in [0, 1]")
        if not math.isfinite(self.lake_seepage_mm_year) or self.lake_seepage_mm_year < 0.0:
            raise ValueError("hydrology.lake_seepage_mm_year must be finite and nonnegative")
        if self.breach_length_cells < 1 or self.breach_length_cells > 128:
            raise ValueError("hydrology.breach_length_cells must be in [1, 128]")
        if self.maximum_breach_rounds < 1 or self.maximum_breach_rounds > 32:
            raise ValueError("hydrology.maximum_breach_rounds must be in [1, 32]")
        if self.river_minimum_discharge_m3s > self.river_discharge_threshold_m3s:
            raise ValueError("river minimum discharge cannot exceed its reporting threshold")
        for name in (
            "minimum_routed_to_inherited_area_ratio",
            "maximum_routed_to_inherited_area_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"hydrology.{name} must be finite and positive")
        if (
            self.minimum_routed_to_inherited_area_ratio
            > self.maximum_routed_to_inherited_area_ratio
        ):
            raise ValueError("routed-to-inherited area ratio bounds are inverted")
        if not 1.0 <= self.maximum_peak_memory_gb <= 28.0:
            raise ValueError("limits.maximum_peak_memory_gb must be in [1, 28]")
        if not 0.25 <= self.maximum_storage_gb <= 16.0:
            raise ValueError("limits.maximum_hydrology_storage_gb must be in [0.25, 16]")


@dataclass(frozen=True)
class L3HydrologyResult:
    output_dir: Path
    manifest_path: Path
    validation_path: Path
    zarr_path: Path
    preview_path: Path
    target_id: str
    cell_count: int
    process_cell_count: int
    river_reach_count: int
    lake_count: int
    validation_passed: bool


@dataclass(frozen=True)
class _HydrologySources:
    target_id: str
    target_manifest: dict[str, Any]
    terrain_manifest: dict[str, Any]
    handoff_dir: Path
    handoff_manifest: dict[str, Any]
    terrain_zarr_path: Path
    handoff_zarr_path: Path
    target_l2: pa.Table
    drainage: pa.Table
    inherited_reaches: pa.Table
    outlet_l0_cell_id: int
    outlet_receiver_l0_cell_id: int
    parent_face_resolution: int
    child_face_resolution: int
    planet_radius_m: float
    actual_cell_size_m: float
    cell_id: np.ndarray
    parent_l2_cell_id: np.ndarray
    l0_parent_cell_id: np.ndarray
    row: np.ndarray
    column: np.ndarray
    xyz: np.ndarray
    area_km2: np.ndarray
    elevation_m: np.ndarray
    unresolved_relief_m: np.ndarray
    inside_core: np.ndarray
    inside_process_halo: np.ndarray
    outside_process: np.ndarray
    l2_ocean_fraction: np.ndarray
    l2_lake_fraction: np.ndarray
    l2_wetland_fraction: np.ndarray
    source_parent_ids: np.ndarray
    source_parent_rows_by_l2: np.ndarray
    source_parent_area_km2: np.ndarray
    monthly_precipitation_mm: np.ndarray
    monthly_runoff_mm: np.ndarray
    monthly_snowmelt_mm: np.ndarray
    monthly_glacier_melt_mm: np.ndarray
    monthly_evaporation_mm: np.ndarray
    monthly_inherited_discharge_m3s: np.ndarray
    annual_aridity: np.ndarray
    rock_strength: np.ndarray
    sediment_accommodation: np.ndarray


def _column_numpy(table: pa.Table, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    column = table[name].combine_chunks()
    if pa.types.is_fixed_size_list(column.type):
        values = column.values.to_numpy(zero_copy_only=False).reshape(
            table.num_rows, column.type.list_size
        )
    else:
        values = column.to_numpy(zero_copy_only=False)
    return np.ascontiguousarray(values, dtype=dtype)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _load_sources(config: L3HydrologyConfig) -> _HydrologySources:
    target_manifest_path = config.target_dir / "manifest.json"
    target_validation_path = config.target_dir / "validation.json"
    terrain_manifest_path = config.terrain_dir / "manifest.json"
    terrain_validation_path = config.terrain_dir / "validation.json"
    terrain_zarr_path = config.terrain_dir / "terrain.zarr"
    target_l2_path = config.target_dir / "tables/target_l2_cells.parquet"
    drainage_path = config.target_dir / "tables/target_drainage_graph.parquet"
    required = (
        target_manifest_path,
        target_validation_path,
        terrain_manifest_path,
        terrain_validation_path,
        terrain_zarr_path,
        target_l2_path,
        drainage_path,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    target_manifest = _load_json(target_manifest_path)
    target_validation = _load_json(target_validation_path)
    terrain_manifest = _load_json(terrain_manifest_path)
    terrain_validation = _load_json(terrain_validation_path)
    if not target_manifest.get("validation_passed") or not target_validation.get("passed"):
        raise RuntimeError("L3 hydrology target has not passed validation")
    if not terrain_manifest.get("validation_passed") or not terrain_validation.get("passed"):
        raise RuntimeError("L3 hydrology terrain has not passed validation")
    if target_manifest.get("target_id") != terrain_manifest.get("target_id"):
        raise RuntimeError("L3 terrain and target IDs do not match")

    handoff_dir = Path(target_manifest["source"]["handoff_dir"]).expanduser().resolve()
    handoff_manifest_path = handoff_dir / "manifest.json"
    handoff_zarr_path = handoff_dir / "region.zarr"
    inherited_path = handoff_dir / "tables/refined_river_reaches.parquet"
    for path in (handoff_manifest_path, handoff_zarr_path, inherited_path):
        if not path.exists():
            raise FileNotFoundError(path)
    handoff_manifest = _load_json(handoff_manifest_path)
    target_l2 = pq.read_table(target_l2_path).combine_chunks()
    drainage = pq.read_table(drainage_path).combine_chunks()
    inherited_reaches = pq.read_table(inherited_path).combine_chunks()
    terrain = zarr.open_group(str(terrain_zarr_path), mode="r")
    handoff = zarr.open_group(str(handoff_zarr_path), mode="r")

    cell_id = np.asarray(terrain["geometry/cell_id"][:], dtype=np.uint64)
    parent_l2 = np.asarray(terrain["geometry/parent_l2_cell_id"][:], dtype=np.int32)
    row = np.asarray(terrain["geometry/row"][:], dtype=np.int32)
    column = np.asarray(terrain["geometry/column"][:], dtype=np.int32)
    xyz = np.asarray(terrain["geometry/xyz"][:], dtype=np.float32)
    area_km2 = np.asarray(terrain["geometry/area_km2"][:], dtype=np.float64)
    elevation = np.asarray(terrain["terrain/elevation_m"][:], dtype=np.float32)
    unresolved = np.asarray(terrain["terrain/unresolved_relief_m"][:], dtype=np.float32)
    inside_core = np.asarray(terrain["geometry/inside_catchment_core"][:], dtype=bool)
    inside_halo = np.asarray(terrain["geometry/inside_process_halo"][:], dtype=bool)
    outside = np.asarray(terrain["geometry/outside_process_domain"][:], dtype=bool)
    cell_count = len(cell_id)
    if any(
        len(values) != cell_count
        for values in (
            parent_l2,
            row,
            column,
            xyz,
            area_km2,
            elevation,
            unresolved,
            inside_core,
            inside_halo,
            outside,
        )
    ):
        raise RuntimeError("L3 terrain arrays have inconsistent lengths")
    if np.any(inside_core.astype(np.int8) + inside_halo + outside != 1):
        raise RuntimeError("L3 hydrology requires exhaustive terrain domain roles")

    target_ids = _column_numpy(target_l2, "fine_cell_id", np.dtype(np.int32))
    target_order = np.argsort(target_ids, kind="stable")
    target_ids = target_ids[target_order]
    terrain_parent_ids = np.unique(parent_l2)
    positions = np.searchsorted(target_ids, terrain_parent_ids)
    if np.any(positions >= len(target_ids)) or np.any(target_ids[positions] != terrain_parent_ids):
        raise RuntimeError("L3 terrain parent is missing from target L2 source table")
    target_rows = target_order[positions]
    domain_parent_ids = _column_numpy(target_l2, "parent_cell_id", np.dtype(np.int32))[target_rows]
    l2_ocean = _column_numpy(target_l2, "ocean_fraction", np.dtype(np.float32))[target_rows]
    l2_lake = _column_numpy(target_l2, "lake_fraction", np.dtype(np.float32))[target_rows]
    l2_wetland = _column_numpy(target_l2, "wetland_fraction", np.dtype(np.float32))[target_rows]

    expected_children = int(terrain.attrs["children_per_parent"])
    grouped_parent = parent_l2.reshape(-1, expected_children)
    if grouped_parent.shape[0] != len(terrain_parent_ids) or np.any(
        grouped_parent != terrain_parent_ids[:, None]
    ):
        raise RuntimeError("L3 hydrology requires parent-major terrain storage")
    l0_parent = np.repeat(domain_parent_ids, expected_children)

    source_parent_ids = np.asarray(handoff["parent/cell_id"][:], dtype=np.int32)
    source_order = np.argsort(source_parent_ids)
    sorted_source_ids = source_parent_ids[source_order]
    source_positions = np.searchsorted(sorted_source_ids, domain_parent_ids)
    if np.any(source_positions >= len(sorted_source_ids)) or np.any(
        sorted_source_ids[source_positions] != domain_parent_ids
    ):
        raise RuntimeError("L3 target references an L0 parent outside the handoff")
    source_rows = source_order[source_positions]

    outlet_l0 = int(target_manifest["selection"]["outlet_parent_cell_id"])
    drainage_ids = _column_numpy(drainage, "cell_id", np.dtype(np.int32))
    outlet_rows = np.flatnonzero(drainage_ids == outlet_l0)
    if len(outlet_rows) != 1:
        raise RuntimeError("registered L3 outlet is not unique in target drainage")
    outlet_receiver = int(
        _column_numpy(drainage, "receiver_id", np.dtype(np.int32))[outlet_rows[0]]
    )
    parent_area = np.asarray(handoff["parent/area_km2"][:], dtype=np.float64)
    planet_radius_m = (
        math.sqrt(
            float(np.sum(parent_area))
            / float(
                np.sum(np.asarray(handoff["parent_priors/geometry/CellArea"][:], dtype=np.float64))
            )
        )
        * 1_000.0
    )

    def prior(path: str, dtype: np.dtype[Any]) -> np.ndarray:
        return np.asarray(handoff[path][:], dtype=dtype)

    return _HydrologySources(
        target_id=str(target_manifest["target_id"]),
        target_manifest=target_manifest,
        terrain_manifest=terrain_manifest,
        handoff_dir=handoff_dir,
        handoff_manifest=handoff_manifest,
        terrain_zarr_path=terrain_zarr_path,
        handoff_zarr_path=handoff_zarr_path,
        target_l2=target_l2,
        drainage=drainage,
        inherited_reaches=inherited_reaches,
        outlet_l0_cell_id=outlet_l0,
        outlet_receiver_l0_cell_id=outlet_receiver,
        parent_face_resolution=int(terrain.attrs["parent_face_resolution"]),
        child_face_resolution=int(terrain.attrs["child_face_resolution"]),
        planet_radius_m=planet_radius_m,
        actual_cell_size_m=float(
            terrain_manifest["hierarchy"]["actual_area_equivalent_cell_size_m"]
        ),
        cell_id=np.ascontiguousarray(cell_id),
        parent_l2_cell_id=np.ascontiguousarray(parent_l2),
        l0_parent_cell_id=np.ascontiguousarray(l0_parent),
        row=np.ascontiguousarray(row),
        column=np.ascontiguousarray(column),
        xyz=np.ascontiguousarray(xyz),
        area_km2=np.ascontiguousarray(area_km2),
        elevation_m=np.ascontiguousarray(elevation),
        unresolved_relief_m=np.ascontiguousarray(unresolved),
        inside_core=np.ascontiguousarray(inside_core),
        inside_process_halo=np.ascontiguousarray(inside_halo),
        outside_process=np.ascontiguousarray(outside),
        l2_ocean_fraction=np.ascontiguousarray(l2_ocean),
        l2_lake_fraction=np.ascontiguousarray(l2_lake),
        l2_wetland_fraction=np.ascontiguousarray(l2_wetland),
        source_parent_ids=np.ascontiguousarray(source_parent_ids),
        source_parent_rows_by_l2=np.ascontiguousarray(source_rows),
        source_parent_area_km2=np.ascontiguousarray(parent_area),
        monthly_precipitation_mm=prior(
            "parent_priors/climate/MonthlyPrecipitationMm", np.dtype(np.float32)
        ),
        monthly_runoff_mm=prior(
            "parent_priors/cryosphere/MonthlyRunoffPotentialMm", np.dtype(np.float32)
        ),
        monthly_snowmelt_mm=prior(
            "parent_priors/cryosphere/MonthlySnowmeltMm", np.dtype(np.float32)
        ),
        monthly_glacier_melt_mm=prior(
            "parent_priors/cryosphere/MonthlyGlacierMeltMm", np.dtype(np.float32)
        ),
        monthly_evaporation_mm=prior(
            "parent_priors/climate/MonthlyEvaporationMm", np.dtype(np.float32)
        ),
        monthly_inherited_discharge_m3s=prior(
            "parent_priors/hydrology/MonthlyDischargeM3s", np.dtype(np.float32)
        ),
        annual_aridity=prior("parent_priors/climate/AnnualAridityIndex", np.dtype(np.float32)),
        rock_strength=prior("parent_priors/geology/RockStrength", np.dtype(np.float32)),
        sediment_accommodation=prior(
            "parent_priors/geology/SedimentAccommodation", np.dtype(np.float32)
        ),
    )


def _fingerprint(
    config: L3HydrologyConfig, sources: _HydrologySources
) -> tuple[str, dict[str, Any]]:
    native = native_library_info("hydrology_native")
    components = {
        "format_version": HYDROLOGY_FORMAT_VERSION,
        "model_version": HYDROLOGY_MODEL_VERSION,
        "target_manifest_sha256": _file_checksum(config.target_dir / "manifest.json"),
        "terrain_manifest_sha256": _file_checksum(config.terrain_dir / "manifest.json"),
        "terrain_zarr_sha256": sources.terrain_manifest["outputs"]["terrain_zarr"]["sha256_tree"],
        "handoff_manifest_sha256": _file_checksum(sources.handoff_dir / "manifest.json"),
        "config_sha256": (
            _file_checksum(config.source_config) if config.source_config is not None else None
        ),
        "controls": asdict(config)
        | {
            "target_dir": str(config.target_dir),
            "terrain_dir": str(config.terrain_dir),
            "output_dir": str(config.output_dir),
            "source_config": str(config.source_config) if config.source_config else None,
        },
        "native_abi_version": native["abi_version"],
        "native_sha256": native["sha256"],
        "orchestrator_sha256": _file_checksum(Path(__file__)),
    }
    return _canonical_hash(components), components


def _spatial_layout(sources: _HydrologySources) -> tuple[np.ndarray, np.ndarray, int, int]:
    order = np.lexsort((sources.column, sources.row)).astype(np.int32, copy=False)
    inverse = np.empty(len(order), dtype=np.int32)
    inverse[order] = np.arange(len(order), dtype=np.int32)
    height = int(np.max(sources.row) - np.min(sources.row) + 1)
    width = int(np.max(sources.column) - np.min(sources.column) + 1)
    if height * width != len(order):
        raise RuntimeError("L3 hydrology requires a complete rectangular terrain window")
    sorted_rows = sources.row[order] - int(np.min(sources.row))
    sorted_columns = sources.column[order] - int(np.min(sources.column))
    expected_rows, expected_columns = np.indices((height, width), dtype=np.int32)
    if not np.array_equal(sorted_rows, expected_rows.reshape(-1)) or not np.array_equal(
        sorted_columns, expected_columns.reshape(-1)
    ):
        raise RuntimeError("L3 terrain coordinates do not form a dense row-major window")
    return np.ascontiguousarray(order), inverse, height, width


def _d8_neighbors(height: int, width: int) -> np.ndarray:
    grid = np.arange(height * width, dtype=np.int32).reshape(height, width)
    neighbors = np.repeat(grid[:, :, None], 8, axis=2)
    offsets = (
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    )
    for index, (row_offset, column_offset) in enumerate(offsets):
        destination_rows = slice(max(0, -row_offset), min(height, height - row_offset))
        destination_columns = slice(max(0, -column_offset), min(width, width - column_offset))
        source_rows = slice(max(0, row_offset), min(height, height + row_offset))
        source_columns = slice(max(0, column_offset), min(width, width + column_offset))
        neighbors[destination_rows, destination_columns, index] = grid[source_rows, source_columns]
    return np.ascontiguousarray(neighbors.reshape(-1, 8))


def _realize_parent_fraction(
    parent_fraction: np.ndarray,
    area_km2: np.ndarray,
    elevation_m: np.ndarray,
    children_per_parent: int,
) -> np.ndarray:
    parent_fraction = np.asarray(parent_fraction, dtype=np.float64)
    parent_count = len(parent_fraction)
    if len(area_km2) != parent_count * children_per_parent:
        raise ValueError("fraction realization does not match parent-major geometry")
    output = np.zeros(len(area_km2), dtype=np.float32)
    for parent in range(parent_count):
        start = parent * children_per_parent
        end = start + children_per_parent
        target_area = float(parent_fraction[parent]) * float(np.sum(area_km2[start:end]))
        ranked = np.lexsort(
            (
                np.arange(children_per_parent, dtype=np.int32),
                elevation_m[start:end],
            )
        )
        for local in ranked:
            if target_area <= 1e-12:
                break
            cell = start + int(local)
            covered = min(target_area, float(area_km2[cell]))
            output[cell] = np.float32(covered / area_km2[cell])
            target_area -= covered
        if target_area > max(1e-9, float(np.sum(area_km2[start:end])) * 1e-8):
            raise RuntimeError("failed to realize an inherited L2 surface fraction")
    return output


def _l2_parent_lookup(sources: _HydrologySources) -> dict[int, int]:
    target_ids = _column_numpy(sources.target_l2, "fine_cell_id", np.dtype(np.int32))
    target_parent = _column_numpy(sources.target_l2, "parent_cell_id", np.dtype(np.int32))
    return dict(zip(map(int, target_ids), map(int, target_parent), strict=True))


def _registered_outlet(
    sources: _HydrologySources,
    spatial_order: np.ndarray,
    height: int,
    width: int,
) -> dict[str, int | float]:
    basin_id = int(sources.target_manifest["selection"]["basin_id"])
    l2_parent = _l2_parent_lookup(sources)
    candidates: list[tuple[float, int, int, int]] = []
    for reach in sources.inherited_reaches.filter(
        pc.equal(sources.inherited_reaches["basin_id"], pa.scalar(basin_id, pa.int32()))
    ).to_pylist():
        path = [int(value) for value in reach["fine_cell_path"]]
        for upstream, downstream in zip(path, path[1:], strict=False):
            if (
                l2_parent.get(upstream) == sources.outlet_l0_cell_id
                and l2_parent.get(downstream) == sources.outlet_receiver_l0_cell_id
            ):
                candidates.append(
                    (float(reach["discharge_mean"]), int(reach["reach_id"]), upstream, downstream)
                )
    if not candidates:
        raise RuntimeError("inherited reach graph does not cross the registered L3 outlet")
    discharge, reach_id, upstream_l2, downstream_l2 = max(candidates)
    parent_resolution = sources.parent_face_resolution
    face_size = parent_resolution * parent_resolution

    def decode(cell_id: int) -> tuple[int, int, int]:
        face = cell_id // face_size
        within = cell_id % face_size
        return face, within // parent_resolution, within % parent_resolution

    upstream_face, upstream_row, upstream_column = decode(upstream_l2)
    downstream_face, downstream_row, downstream_column = decode(downstream_l2)
    delta_row = downstream_row - upstream_row
    delta_column = downstream_column - upstream_column
    if upstream_face != downstream_face or max(abs(delta_row), abs(delta_column)) != 1:
        raise RuntimeError("registered inherited outlet is not an adjacent same-face L2 crossing")

    factor = sources.child_face_resolution // sources.parent_face_resolution
    minimum_row = int(np.min(sources.row))
    minimum_column = int(np.min(sources.column))
    upstream_rows = np.flatnonzero(sources.parent_l2_cell_id == upstream_l2)
    pairs: list[tuple[float, int, int]] = []
    for upstream_parent_row in upstream_rows:
        row = int(sources.row[upstream_parent_row])
        column = int(sources.column[upstream_parent_row])
        local_row = row - upstream_row * factor
        local_column = column - upstream_column * factor
        if delta_row < 0 and local_row != 0:
            continue
        if delta_row > 0 and local_row != factor - 1:
            continue
        if delta_column < 0 and local_column != 0:
            continue
        if delta_column > 0 and local_column != factor - 1:
            continue
        downstream_row_fine = row + delta_row
        downstream_column_fine = column + delta_column
        spatial_index = (downstream_row_fine - minimum_row) * width + (
            downstream_column_fine - minimum_column
        )
        if not 0 <= spatial_index < height * width:
            continue
        downstream_parent_row = int(spatial_order[spatial_index])
        if sources.parent_l2_cell_id[downstream_parent_row] != downstream_l2:
            continue
        saddle = max(
            float(sources.elevation_m[upstream_parent_row]),
            float(sources.elevation_m[downstream_parent_row]),
        )
        pairs.append((saddle, int(upstream_parent_row), downstream_parent_row))
    if not pairs:
        raise RuntimeError("registered L2 outlet crossing has no adjacent L3 child pair")
    saddle, upstream_cell, downstream_cell = min(pairs)
    return {
        "inherited_reach_id": reach_id,
        "inherited_discharge_m3s": discharge,
        "upstream_l2_cell_id": upstream_l2,
        "downstream_l2_cell_id": downstream_l2,
        "upstream_cell_row": upstream_cell,
        "downstream_cell_row": downstream_cell,
        "upstream_cell_id": int(sources.cell_id[upstream_cell]),
        "downstream_cell_id": int(sources.cell_id[downstream_cell]),
        "physical_saddle_elevation_m": saddle,
    }


def _topographic_weights(
    elevation_m: np.ndarray,
    area_km2: np.ndarray,
    eligible: np.ndarray,
    l0_parent_id: np.ndarray,
    coefficient: float,
) -> np.ndarray:
    weights = np.zeros(len(elevation_m), dtype=np.float64)
    for parent_id in np.unique(l0_parent_id[eligible]):
        rows = np.flatnonzero(eligible & (l0_parent_id == parent_id))
        if not len(rows):
            continue
        area = area_km2[rows]
        mean = float(np.average(elevation_m[rows], weights=area))
        variance = float(np.average((elevation_m[rows] - mean) ** 2, weights=area))
        scale = max(math.sqrt(max(variance, 0.0)), 100.0)
        standardized = np.clip((elevation_m[rows] - mean) / scale, -2.5, 2.5)
        local = np.exp(coefficient * standardized)
        normalization = float(np.sum(local * area) / np.sum(area))
        weights[rows] = local / max(normalization, 1e-12)
    return weights


def _downscale_monthly(
    parent_monthly: np.ndarray,
    source_rows_by_cell: np.ndarray,
    weights: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    if parent_monthly.ndim != 2 or parent_monthly.shape[1] != MONTHS:
        raise ValueError("inherited monthly forcing must have shape (parent, 12)")
    output = np.zeros((MONTHS, len(source_rows_by_cell)), dtype=np.float32)
    rows = np.flatnonzero(eligible)
    source_rows = source_rows_by_cell[rows]
    output[:, rows] = (
        parent_monthly[source_rows].T * np.asarray(weights[rows], dtype=np.float32)[None, :]
    )
    return output


def _forcing_error(
    downscaled: np.ndarray,
    source: np.ndarray,
    source_rows_by_cell: np.ndarray,
    l0_parent_id: np.ndarray,
    area_km2: np.ndarray,
    eligible: np.ndarray,
    inside_core: np.ndarray,
) -> float:
    maximum = 0.0
    for parent_id in np.unique(l0_parent_id[inside_core]):
        rows = np.flatnonzero(eligible & inside_core & (l0_parent_id == parent_id))
        if not len(rows):
            continue
        source_row = int(source_rows_by_cell[rows[0]])
        expected = np.asarray(source[source_row], dtype=np.float64) * float(np.sum(area_km2[rows]))
        actual = np.sum(
            np.asarray(downscaled[:, rows], dtype=np.float64) * area_km2[rows][None, :], axis=1
        )
        relative = np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-9)
        maximum = max(maximum, float(np.max(relative)))
    return maximum


def _initialize_partial(
    partial: Path,
    config: L3HydrologyConfig,
    sources: _HydrologySources,
    run_fingerprint: str,
) -> Any:
    partial.mkdir(parents=True)
    zarr_path = partial / "hydrology.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    cell_count = len(sources.cell_id)
    chunks = (min(config.chunk_rows, cell_count),)
    monthly_chunks = (1, min(config.chunk_rows, cell_count))
    root.attrs.update(
        {
            "format_version": HYDROLOGY_FORMAT_VERSION,
            "model_version": HYDROLOGY_MODEL_VERSION,
            "status": "partial",
            "target_id": sources.target_id,
            "run_fingerprint": run_fingerprint,
            "cell_count": cell_count,
            "parent_major_storage": True,
            "source_terrain_dir": str(config.terrain_dir),
            "hydrological_acceptance_scope": "fine routed catchment core only",
        }
    )
    geometry = root.require_group("geometry")
    l0_parent = _zarr_dataset(
        geometry,
        "l0_parent_cell_id",
        shape=(cell_count,),
        dtype=np.int32,
        chunks=chunks,
        semantics="stable inherited forcing parent",
    )
    l0_parent[:] = sources.l0_parent_cell_id
    for name, values, semantics in (
        (
            "inside_inherited_target_core",
            sources.inside_core,
            "L0-derived target envelope retained for provenance and forcing conservation",
        ),
        (
            "inside_process_halo",
            sources.inside_process_halo,
            "boundary context available to regional routing",
        ),
        (
            "outside_process_domain",
            sources.outside_process,
            "continuous terrain excluded from regional routing",
        ),
    ):
        dataset = _zarr_dataset(
            geometry,
            name,
            shape=(cell_count,),
            dtype=bool,
            chunks=chunks,
            semantics=semantics,
        )
        dataset[:] = values
    _zarr_dataset(
        geometry,
        "inside_routed_catchment_core",
        shape=(cell_count,),
        dtype=bool,
        chunks=chunks,
        semantics="fine D8 upstream closure of the registered outlet; hydrological acceptance core",
    )
    surface = root.require_group("surface")
    for name, semantics in (
        ("physical_ocean_fraction", "L2 area-conservative ocean realization on L3 terrain"),
        ("inherited_lake_fraction_prior", "L2 lake occupancy prior, not routed L3 water"),
        ("inherited_wetland_fraction_prior", "L2 wetland occupancy prior, not routed L3 water"),
        ("lake_fraction", "depression and water-balance-derived L3 open water"),
        ("wetland_fraction", "depression and water-balance-derived L3 wetland support"),
    ):
        _zarr_dataset(
            surface,
            name,
            shape=(cell_count,),
            dtype=np.float32,
            chunks=chunks,
            semantics=semantics,
        )
    _zarr_dataset(
        surface,
        "water_body_class",
        shape=(cell_count,),
        dtype=np.uint8,
        chunks=chunks,
    )

    forcing = root.require_group("forcing")
    for name, semantics in (
        ("monthly_precipitation_mm", "conservative L0 prior downscaled by L3 orography"),
        ("monthly_runoff_mm", "conservative L0 cryosphere runoff downscaled by L3 orography"),
        ("monthly_snowmelt_mm", "conservative L0 snowmelt attribution"),
        ("monthly_glacier_melt_mm", "conservative L0 glacier-melt attribution"),
        ("monthly_evaporation_mm", "inherited L0 evaporation forcing"),
    ):
        _zarr_dataset(
            forcing,
            name,
            shape=(MONTHS, cell_count),
            dtype=np.float32,
            chunks=monthly_chunks,
            units="mm/month",
            semantics=semantics,
        )
    for name, semantics in (
        ("annual_aridity_index", "inherited L0 climate prior"),
        ("rock_strength", "inherited L0 geology prior"),
        ("sediment_accommodation", "inherited L0 geology prior"),
    ):
        _zarr_dataset(
            forcing,
            name,
            shape=(cell_count,),
            dtype=np.float32,
            chunks=chunks,
            semantics=semantics,
        )

    routing = root.require_group("routing")
    one_dimensional = {
        "terminal_kind": np.uint8,
        "depression_id": np.int32,
        "lake_id": np.int32,
        "depression_fill_depth_m": np.float32,
        "hydrologic_elevation_m": np.float32,
        "prospective_breach_incision_m": np.float32,
        "flow_receiver_cell_id": np.int64,
        "flow_slope": np.float32,
        "contributing_area_km2": np.float64,
        "mean_discharge_m3s": np.float32,
        "mean_flow_velocity_mps": np.float32,
        "stream_power_w": np.float32,
        "basin_id": np.int32,
        "flow_sink_type": np.uint8,
        "river_corridor_fraction": np.float32,
        "reported_reach_support": bool,
        "waterbody_flow_connector": bool,
        "channel_fraction": np.float32,
        "floodplain_fraction": np.float32,
    }
    for name, dtype in one_dimensional.items():
        _zarr_dataset(routing, name, shape=(cell_count,), dtype=dtype, chunks=chunks)
    _zarr_dataset(
        routing,
        "flow_direction_xyz",
        shape=(cell_count, 3),
        dtype=np.float32,
        chunks=(chunks[0], 3),
    )
    _zarr_dataset(
        routing,
        "monthly_discharge_m3s",
        shape=(MONTHS, cell_count),
        dtype=np.float32,
        chunks=monthly_chunks,
    )

    progress = root.require_group("progress")
    _zarr_dataset(progress, "forcing_complete", shape=(1,), dtype=bool, chunks=(1,))
    _zarr_dataset(progress, "routing_complete", shape=(1,), dtype=bool, chunks=(1,))
    _write_json_durable(
        partial / "run_state.json",
        {
            "format_version": HYDROLOGY_FORMAT_VERSION,
            "model_version": HYDROLOGY_MODEL_VERSION,
            "run_fingerprint": run_fingerprint,
            "cell_count": cell_count,
        },
    )
    return root


def _open_partial(
    partial: Path,
    config: L3HydrologyConfig,
    sources: _HydrologySources,
    run_fingerprint: str,
) -> tuple[Any, bool]:
    if partial.exists():
        try:
            state = _load_json(partial / "run_state.json")
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            state = {}
        if state.get("run_fingerprint") == run_fingerprint and state.get("cell_count") == len(
            sources.cell_id
        ):
            return zarr.open_group(str(partial / "hydrology.zarr"), mode="r+"), True
        shutil.rmtree(partial)
    return _initialize_partial(partial, config, sources, run_fingerprint), False


def _write_forcing(
    root: Any,
    partial: Path,
    config: L3HydrologyConfig,
    sources: _HydrologySources,
    outlet: Mapping[str, int | float],
) -> dict[str, Any]:
    zarr_path = partial / "hydrology.zarr"
    children_per_parent = int(sources.terrain_manifest["hierarchy"]["children_per_parent"])
    physical_ocean = _realize_parent_fraction(
        sources.l2_ocean_fraction,
        sources.area_km2,
        sources.elevation_m,
        children_per_parent,
    )
    inherited_lake = _realize_parent_fraction(
        sources.l2_lake_fraction,
        sources.area_km2,
        sources.elevation_m,
        children_per_parent,
    )
    inherited_wetland = _realize_parent_fraction(
        sources.l2_wetland_fraction,
        sources.area_km2,
        sources.elevation_m,
        children_per_parent,
    )
    terminal_kind = np.full(len(sources.cell_id), TERMINAL_NONE, dtype=np.uint8)
    terminal_kind[physical_ocean >= 0.5] = TERMINAL_PHYSICAL_OCEAN
    terminal_kind[sources.outside_process] = TERMINAL_PROCESS_BOUNDARY
    terminal_kind[int(outlet["downstream_cell_row"])] = TERMINAL_REGISTERED_OUTLET
    if (
        not sources.inside_core[int(outlet["upstream_cell_row"])]
        or sources.inside_core[int(outlet["downstream_cell_row"])]
    ):
        raise RuntimeError("registered L3 outlet does not cross from core into context")
    active = sources.inside_core | sources.inside_process_halo
    eligible = active & (terminal_kind == TERMINAL_NONE)
    source_rows_by_cell = np.repeat(sources.source_parent_rows_by_l2, children_per_parent)
    if len(source_rows_by_cell) != len(sources.cell_id):
        raise RuntimeError("L0 forcing rows do not align with parent-major L3 terrain")

    runoff_weights = _topographic_weights(
        sources.elevation_m,
        sources.area_km2,
        eligible,
        sources.l0_parent_cell_id,
        config.orographic_runoff_weight,
    )
    precipitation_weights = _topographic_weights(
        sources.elevation_m,
        sources.area_km2,
        eligible,
        sources.l0_parent_cell_id,
        config.orographic_precipitation_weight,
    )
    melt_weights = _topographic_weights(
        sources.elevation_m,
        sources.area_km2,
        eligible,
        sources.l0_parent_cell_id,
        config.highland_melt_weight,
    )
    field_specs = (
        (
            "monthly_precipitation_mm",
            sources.monthly_precipitation_mm,
            precipitation_weights,
        ),
        ("monthly_runoff_mm", sources.monthly_runoff_mm, runoff_weights),
        ("monthly_snowmelt_mm", sources.monthly_snowmelt_mm, melt_weights),
        ("monthly_glacier_melt_mm", sources.monthly_glacier_melt_mm, melt_weights),
    )
    errors: dict[str, float] = {}
    for name, source, weights in field_specs:
        values = _downscale_monthly(source, source_rows_by_cell, weights, eligible)
        root[f"forcing/{name}"][:] = values
        _sync_zarr_array(zarr_path, f"forcing/{name}")
        errors[name] = _forcing_error(
            values,
            source,
            source_rows_by_cell,
            sources.l0_parent_cell_id,
            sources.area_km2,
            eligible,
            sources.inside_core,
        )
        del values

    evaporation = np.zeros((MONTHS, len(sources.cell_id)), dtype=np.float32)
    eligible_rows = np.flatnonzero(active)
    evaporation[:, eligible_rows] = sources.monthly_evaporation_mm[
        source_rows_by_cell[eligible_rows]
    ].T
    root["forcing/monthly_evaporation_mm"][:] = evaporation
    del evaporation
    for name, values in (
        ("annual_aridity_index", sources.annual_aridity[source_rows_by_cell]),
        ("rock_strength", sources.rock_strength[source_rows_by_cell]),
        (
            "sediment_accommodation",
            sources.sediment_accommodation[source_rows_by_cell],
        ),
    ):
        root[f"forcing/{name}"][:] = np.asarray(values, dtype=np.float32)

    root["surface/physical_ocean_fraction"][:] = physical_ocean
    root["surface/inherited_lake_fraction_prior"][:] = inherited_lake
    root["surface/inherited_wetland_fraction_prior"][:] = inherited_wetland
    root["routing/terminal_kind"][:] = terminal_kind
    for path in (
        "geometry/l0_parent_cell_id",
        "geometry/inside_inherited_target_core",
        "geometry/inside_process_halo",
        "geometry/outside_process_domain",
        "forcing/monthly_evaporation_mm",
        "forcing/annual_aridity_index",
        "forcing/rock_strength",
        "forcing/sediment_accommodation",
        "surface/physical_ocean_fraction",
        "surface/inherited_lake_fraction_prior",
        "surface/inherited_wetland_fraction_prior",
        "routing/terminal_kind",
    ):
        _sync_zarr_array(zarr_path, path)
    root["progress/forcing_complete"][0] = True
    _sync_zarr_array(zarr_path, "progress/forcing_complete")
    return {
        "maximum_core_forcing_conservation_relative_error": max(errors.values()),
        "forcing_conservation_relative_errors": errors,
        "physical_ocean_area_km2": float(np.sum(physical_ocean * sources.area_km2)),
        "eligible_process_cell_count": int(np.count_nonzero(eligible)),
        "terminal_counts": {
            str(kind): int(np.count_nonzero(terminal_kind == kind)) for kind in range(4)
        },
    }


def _forcing_metrics(
    root: Any,
    sources: _HydrologySources,
) -> dict[str, Any]:
    children_per_parent = int(sources.terrain_manifest["hierarchy"]["children_per_parent"])
    source_rows_by_cell = np.repeat(sources.source_parent_rows_by_l2, children_per_parent)
    terminal = np.asarray(root["routing/terminal_kind"][:], dtype=np.uint8)
    eligible = (sources.inside_core | sources.inside_process_halo) & (terminal == TERMINAL_NONE)
    errors = {}
    for name, source in (
        ("monthly_precipitation_mm", sources.monthly_precipitation_mm),
        ("monthly_runoff_mm", sources.monthly_runoff_mm),
        ("monthly_snowmelt_mm", sources.monthly_snowmelt_mm),
        ("monthly_glacier_melt_mm", sources.monthly_glacier_melt_mm),
    ):
        errors[name] = _forcing_error(
            np.asarray(root[f"forcing/{name}"][:], dtype=np.float32),
            source,
            source_rows_by_cell,
            sources.l0_parent_cell_id,
            sources.area_km2,
            eligible,
            sources.inside_core,
        )
    ocean = np.asarray(root["surface/physical_ocean_fraction"][:], dtype=np.float32)
    return {
        "maximum_core_forcing_conservation_relative_error": max(errors.values()),
        "forcing_conservation_relative_errors": errors,
        "physical_ocean_area_km2": float(np.sum(ocean * sources.area_km2)),
        "eligible_process_cell_count": int(np.count_nonzero(eligible)),
        "terminal_counts": {
            str(kind): int(np.count_nonzero(terminal == kind)) for kind in range(4)
        },
    }


def _native_controls(
    config: L3HydrologyConfig, sources: _HydrologySources
) -> dict[str, int | float]:
    return {
        "planet_radius_m": sources.planet_radius_m,
        "minimum_depression_depth_m": config.minimum_depression_depth_m,
        "wetland_mean_depth_m": config.wetland_mean_depth_m,
        "endorheic_aridity_threshold": config.endorheic_aridity_threshold,
        "maximum_fill_time_years": config.maximum_fill_time_years,
        "lake_seepage_mm_year": config.lake_seepage_mm_year,
        "subgrid_relief_scale": config.subgrid_relief_scale,
        "subgrid_connected_basin_fraction": config.subgrid_connected_basin_fraction,
        "breach_score_threshold": config.breach_score_threshold,
        "maximum_breach_incision_m": config.maximum_breach_incision_m,
        "breach_length_cells": config.breach_length_cells,
        "river_discharge_threshold_m3s": config.river_discharge_threshold_m3s,
        "river_contributing_area_threshold_km2": config.river_contributing_area_threshold_km2,
        "river_minimum_discharge_m3s": config.river_minimum_discharge_m3s,
    }


def _to_parent_major(values: np.ndarray, spatial_order: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        output = np.empty_like(values)
        output[spatial_order] = values
        return output
    if values.ndim == 2 and values.shape[0] == MONTHS:
        output = np.empty_like(values)
        output[:, spatial_order] = values
        return output
    if values.ndim == 2 and values.shape[1] == 3:
        output = np.empty_like(values)
        output[spatial_order] = values
        return output
    raise ValueError(f"unsupported regional output shape {values.shape}")


def _replace_index_column(
    table: pa.Table,
    source_name: str,
    target_name: str,
    cell_id_spatial: np.ndarray,
) -> pa.Table:
    local = _column_numpy(table, source_name, np.dtype(np.int32))
    stable = np.full(len(local), -1, dtype=np.int64)
    valid = local >= 0
    stable[valid] = cell_id_spatial[local[valid]].astype(np.int64)
    index = table.schema.get_field_index(source_name)
    return table.set_column(index, target_name, pa.array(stable, type=pa.int64()))


def _stable_lake_table(table: pa.Table, cell_id_spatial: np.ndarray) -> pa.Table:
    result = table
    for source, target in (
        ("sink_cell", "sink_cell_id"),
        ("outlet_cell", "outlet_cell_id"),
        ("outlet_receiver", "outlet_receiver_cell_id"),
    ):
        result = _replace_index_column(result, source, target, cell_id_spatial)
    return result


def _stable_breach_table(table: pa.Table, cell_id_spatial: np.ndarray) -> pa.Table:
    result = table
    for source, target in (
        ("outlet_cell", "outlet_cell_id"),
        ("downstream_cell", "downstream_cell_id"),
    ):
        result = _replace_index_column(result, source, target, cell_id_spatial)
    return result


def _stable_reach_table(
    table: pa.Table,
    cell_id_spatial: np.ndarray,
    xyz_spatial: np.ndarray,
    area_spatial: np.ndarray,
    core_spatial: np.ndarray,
) -> pa.Table:
    paths = [np.asarray(path, dtype=np.int32) for path in table["cell_path"].to_pylist()]
    stable_paths = [cell_id_spatial[path].astype(np.int64).tolist() for path in paths]
    polyline_type = pa.list_(pa.list_(pa.float32(), 3))
    polylines = [xyz_spatial[path].astype(np.float32).tolist() for path in paths]
    core_fraction = np.asarray(
        [
            (
                float(np.sum(area_spatial[path] * core_spatial[path]) / np.sum(area_spatial[path]))
                if len(path)
                else 0.0
            )
            for path in paths
        ],
        dtype=np.float32,
    )
    from_local = _column_numpy(table, "from_node", np.dtype(np.int32))
    to_local = _column_numpy(table, "to_node", np.dtype(np.int32))
    from_stable = cell_id_spatial[from_local].astype(np.int64)
    to_stable = cell_id_spatial[to_local].astype(np.int64)
    path_index = table.schema.get_field_index("cell_path")
    from_index = table.schema.get_field_index("from_node")
    to_index = table.schema.get_field_index("to_node")
    result = table.set_column(
        path_index,
        "cell_path",
        pa.array(stable_paths, type=pa.list_(pa.int64())),
    )
    result = result.set_column(from_index, "from_cell_id", pa.array(from_stable, type=pa.int64()))
    result = result.set_column(to_index, "to_cell_id", pa.array(to_stable, type=pa.int64()))
    incision_index = result.schema.get_field_index("incision_m")
    result = result.set_column(
        incision_index,
        "prospective_incision_m",
        result["incision_m"],
    )
    return (
        result.append_column(
            "polyline_on_cubed_sphere",
            pa.array(polylines, type=polyline_type),
        )
        .append_column("inside_core_fraction", pa.array(core_fraction, type=pa.float32()))
        .append_column("intersects_core", pa.array(core_fraction > 0.0, type=pa.bool_()))
    )


def _reach_support_masks(table: pa.Table, cell_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize vector support for diagnostics without turning it into water area."""

    network = np.zeros(cell_count, dtype=bool)
    connector = np.zeros(cell_count, dtype=bool)
    reach_kind = table["reach_kind"].to_pylist()
    for row, raw_path in enumerate(table["cell_path"].to_pylist()):
        path = np.asarray(raw_path, dtype=np.int32)
        path = path[(path >= 0) & (path < cell_count)]
        network[path] = True
        if reach_kind[row] == "connector":
            connector[path] = True
    return network, connector


def _inherited_reach_subset(sources: _HydrologySources) -> pa.Table:
    basin_id = int(sources.target_manifest["selection"]["basin_id"])
    basin = sources.inherited_reaches.filter(
        pc.equal(sources.inherited_reaches["basin_id"], pa.scalar(basin_id, pa.int32()))
    )
    domain_ids = set(map(int, np.unique(sources.parent_l2_cell_id)))
    keep = np.asarray(
        [
            any(int(cell) in domain_ids for cell in path)
            for path in basin["fine_cell_path"].to_pylist()
        ],
        dtype=bool,
    )
    return basin.filter(pa.array(keep, type=pa.bool_()))


def _inherited_alignment(
    inherited: pa.Table,
    river_parent_ids: set[int],
    domain_parent_ids: set[int],
) -> pa.Table:
    records = []
    for row in inherited.to_pylist():
        path = [int(cell) for cell in row["fine_cell_path"] if int(cell) in domain_parent_ids]
        supported = sum(cell in river_parent_ids for cell in path)
        records.append(
            {
                "inherited_reach_id": int(row["reach_id"]),
                "reach_kind": str(row["reach_kind"]),
                "domain_l2_cell_count": len(path),
                "supported_l2_cell_count": supported,
                "supported_l2_fraction": supported / max(len(path), 1),
                "inherited_discharge_m3s": float(row["discharge_mean"]),
                "inherited_strahler_order": int(row["strahler_order"]),
            }
        )
    return pa.Table.from_pylist(
        records,
        schema=pa.schema(
            [
                ("inherited_reach_id", pa.int32()),
                ("reach_kind", pa.string()),
                ("domain_l2_cell_count", pa.int32()),
                ("supported_l2_cell_count", pa.int32()),
                ("supported_l2_fraction", pa.float32()),
                ("inherited_discharge_m3s", pa.float32()),
                ("inherited_strahler_order", pa.int32()),
            ]
        ),
    )


def _registered_basin(
    outputs: Mapping[str, np.ndarray],
    downstream_spatial: int,
    terminal_spatial: np.ndarray,
) -> tuple[int, int, np.ndarray]:
    """Locate the generated basin that enters the registered outlet terminal."""

    receiver = np.asarray(outputs["FlowReceiverID"], dtype=np.int32)
    discharge = np.asarray(outputs["MeanDischargeM3s"], dtype=np.float32)
    basin = np.asarray(outputs["BasinID"], dtype=np.int32)
    entries = np.flatnonzero(receiver == downstream_spatial)
    if not len(entries):
        return -1, -1, np.zeros(len(receiver), dtype=bool)
    entry = int(entries[np.argmax(discharge[entries])])
    basin_id = int(basin[entry])
    if basin_id < 0:
        return entry, basin_id, np.zeros(len(receiver), dtype=bool)
    mask = (terminal_spatial == TERMINAL_NONE) & (basin == basin_id)
    return entry, basin_id, mask


def _run_routing(
    root: Any,
    partial: Path,
    config: L3HydrologyConfig,
    sources: _HydrologySources,
    outlet: Mapping[str, int | float],
    spatial_order: np.ndarray,
    height: int,
    width: int,
) -> tuple[dict[str, Any], pa.Table, pa.Table, pa.Table, pa.Table, pa.Table]:
    zarr_path = partial / "hydrology.zarr"
    neighbors = _d8_neighbors(height, width)
    terminal_parent = np.asarray(root["routing/terminal_kind"][:], dtype=np.uint8)
    terminal_spatial = np.ascontiguousarray((terminal_parent[spatial_order] > 0).astype(np.uint8))
    radius_km = sources.planet_radius_m / 1_000.0
    area_spatial = np.ascontiguousarray(sources.area_km2[spatial_order])
    elevation_spatial = np.ascontiguousarray(sources.elevation_m[spatial_order])
    xyz_spatial = np.ascontiguousarray(sources.xyz[spatial_order])
    inherited_core_spatial = np.ascontiguousarray(sources.inside_core[spatial_order])
    l2_spatial = np.ascontiguousarray(sources.parent_l2_cell_id[spatial_order])
    inverse_spatial = np.empty(len(spatial_order), dtype=np.int32)
    inverse_spatial[spatial_order] = np.arange(len(spatial_order), dtype=np.int32)
    downstream_spatial = int(inverse_spatial[int(outlet["downstream_cell_row"])])
    forcing_paths = {
        "runoff": "forcing/monthly_runoff_mm",
        "evaporation": "forcing/monthly_evaporation_mm",
        "aridity": "forcing/annual_aridity_index",
        "rock_strength": "forcing/rock_strength",
        "accommodation": "forcing/sediment_accommodation",
    }
    runoff = np.ascontiguousarray(
        np.asarray(root[forcing_paths["runoff"]][:], dtype=np.float32)[:, spatial_order]
    )
    evaporation = np.ascontiguousarray(
        np.asarray(root[forcing_paths["evaporation"]][:], dtype=np.float32)[:, spatial_order]
    )
    aridity = np.ascontiguousarray(
        np.asarray(root[forcing_paths["aridity"]][:], dtype=np.float32)[spatial_order]
    )
    rock = np.ascontiguousarray(
        np.asarray(root[forcing_paths["rock_strength"]][:], dtype=np.float32)[spatial_order]
    )
    accommodation = np.ascontiguousarray(
        np.asarray(root[forcing_paths["accommodation"]][:], dtype=np.float32)[spatial_order]
    )
    relief = np.ascontiguousarray(sources.unresolved_relief_m[spatial_order])
    area_steradians = np.ascontiguousarray(area_spatial / (radius_km * radius_km))
    routing_elevation = elevation_spatial
    breach_tables: list[pa.Table] = []
    round_metadata: list[dict[str, Any]] = []
    cumulative_breach_sediment_km3 = 0.0
    for routing_round in range(1, config.maximum_breach_rounds + 1):
        outputs, lakes, breaches, reaches, native_metadata = run_regional_hydrology(
            controls=_native_controls(config, sources),
            areas_steradians=area_steradians,
            neighbors=neighbors,
            xyz=xyz_spatial,
            elevation=np.ascontiguousarray(routing_elevation),
            relief=relief,
            rock_strength=rock,
            accommodation=accommodation,
            terminal=terminal_spatial,
            runoff=runoff,
            evaporation=evaporation,
            aridity=aridity,
        )
        if breaches.num_rows:
            breach_tables.append(
                breaches.append_column(
                    "routing_round",
                    pa.array(
                        np.full(breaches.num_rows, routing_round, dtype=np.int16),
                        type=pa.int16(),
                    ),
                )
            )
        cumulative_breach_sediment_km3 += float(native_metadata["breach_sediment_pulse_km3"])
        _, round_basin_id, round_core = _registered_basin(
            outputs, downstream_spatial, terminal_spatial
        )
        lake_area_km2 = float(
            np.sum(outputs["LakeFraction"][round_core] * area_spatial[round_core])
        )
        cumulative_incision = np.maximum(elevation_spatial - outputs["HydrologicElevationM"], 0.0)
        round_metadata.append(
            {
                "routing_round": routing_round,
                "depression_count": int(native_metadata["depression_count"]),
                "breach_count": int(native_metadata["breach_count"]),
                "lake_count": int(native_metadata["lake_count"]),
                "core_lake_area_km2": lake_area_km2,
                "registered_outlet_basin_id": round_basin_id,
                "routed_core_area_km2": float(np.sum(area_spatial[round_core])),
                "maximum_cumulative_prospective_incision_m": float(
                    np.max(cumulative_incision, initial=0.0)
                ),
            }
        )
        next_elevation = outputs["HydrologicElevationM"]
        maximum_change = float(np.max(np.abs(next_elevation - routing_elevation), initial=0.0))
        routing_elevation = next_elevation
        if int(native_metadata["breach_count"]) == 0 or maximum_change < 1e-3:
            break

    outputs["BreachIncisionM"] = np.maximum(elevation_spatial - routing_elevation, 0.0).astype(
        np.float32
    )
    outlet_entry, registered_basin_id, routed_core_spatial = _registered_basin(
        outputs, downstream_spatial, terminal_spatial
    )
    if breach_tables:
        breaches = pa.concat_tables(breach_tables, promote_options="default")
        breach_id_index = breaches.schema.get_field_index("breach_id")
        breaches = breaches.set_column(
            breach_id_index,
            "breach_event_id",
            pa.array(np.arange(breaches.num_rows, dtype=np.int32), type=pa.int32()),
        )

    cell_id_spatial = sources.cell_id[spatial_order]
    receiver_local = outputs["FlowReceiverID"]
    receiver_cell_id_spatial = np.full(len(receiver_local), -1, dtype=np.int64)
    valid_receiver = receiver_local >= 0
    receiver_cell_id_spatial[valid_receiver] = cell_id_spatial[
        receiver_local[valid_receiver]
    ].astype(np.int64)
    channel_width_m = np.where(
        outputs["RiverCorridor"] > 0.0,
        4.2 * np.maximum(outputs["MeanDischargeM3s"], 0.01) ** 0.45,
        0.0,
    )
    channel_fraction = np.clip(
        channel_width_m / max(sources.actual_cell_size_m, 1.0), 0.0, 1.0
    ).astype(np.float32)
    reported_reach_support, waterbody_flow_connector = _reach_support_masks(
        reaches, len(cell_id_spatial)
    )
    write_values = {
        "geometry/inside_routed_catchment_core": routed_core_spatial,
        "surface/lake_fraction": outputs["LakeFraction"],
        "surface/wetland_fraction": outputs["WetlandFraction"],
        "surface/water_body_class": outputs["WaterBodyClass"],
        "routing/depression_id": outputs["DepressionID"],
        "routing/lake_id": outputs["LakeID"],
        "routing/depression_fill_depth_m": outputs["DepressionFillDepthM"],
        "routing/hydrologic_elevation_m": outputs["HydrologicElevationM"],
        "routing/prospective_breach_incision_m": outputs["BreachIncisionM"],
        "routing/flow_receiver_cell_id": receiver_cell_id_spatial,
        "routing/flow_direction_xyz": outputs["FlowDirectionXYZ"],
        "routing/flow_slope": outputs["FlowSlope"],
        "routing/contributing_area_km2": outputs["ContributingAreaKm2"],
        "routing/monthly_discharge_m3s": outputs["MonthlyDischargeM3s"],
        "routing/mean_discharge_m3s": outputs["MeanDischargeM3s"],
        "routing/mean_flow_velocity_mps": outputs["MeanFlowVelocityMps"],
        "routing/stream_power_w": outputs["StreamPowerW"],
        "routing/basin_id": outputs["BasinID"],
        "routing/flow_sink_type": outputs["FlowSinkType"],
        "routing/river_corridor_fraction": outputs["RiverCorridor"],
        "routing/reported_reach_support": reported_reach_support,
        "routing/waterbody_flow_connector": waterbody_flow_connector,
        "routing/channel_fraction": channel_fraction,
        "routing/floodplain_fraction": outputs["FloodplainPotential"],
    }
    for path, values in write_values.items():
        root[path][:] = _to_parent_major(np.asarray(values), spatial_order)
        _sync_zarr_array(zarr_path, path)

    stable_lakes = _stable_lake_table(lakes, cell_id_spatial)
    stable_breaches = _stable_breach_table(breaches, cell_id_spatial)
    stable_reaches = _stable_reach_table(
        reaches, cell_id_spatial, xyz_spatial, area_spatial, routed_core_spatial
    )
    inherited = _inherited_reach_subset(sources)
    river_cells = outputs["RiverCorridor"] > 0.0
    alignment = _inherited_alignment(
        inherited,
        set(map(int, np.unique(l2_spatial[river_cells]))),
        set(map(int, np.unique(l2_spatial))),
    )
    native_metadata["neighbor_count"] = 8
    native_metadata["routing_round_count"] = len(round_metadata)
    native_metadata["routing_rounds"] = round_metadata
    native_metadata["cumulative_breach_event_count"] = sum(
        int(value["breach_count"]) for value in round_metadata
    )
    native_metadata["cumulative_breach_sediment_pulse_km3"] = cumulative_breach_sediment_km3
    native_metadata["maximum_cumulative_prospective_incision_m"] = float(
        np.max(outputs["BreachIncisionM"], initial=0.0)
    )
    native_metadata["routed_process_cell_count"] = int(
        np.count_nonzero(terminal_spatial == TERMINAL_NONE)
    )
    native_metadata["registered_outlet_entry_spatial_index"] = outlet_entry
    native_metadata["registered_outlet_basin_id"] = registered_basin_id
    native_metadata["routed_core_cell_count"] = int(np.count_nonzero(routed_core_spatial))
    native_metadata["inherited_core_overlap_cell_count"] = int(
        np.count_nonzero(routed_core_spatial & inherited_core_spatial)
    )
    tables_dir = partial / "tables"
    tables_dir.mkdir(exist_ok=True)
    table_values = {
        "lakes": stable_lakes,
        "breaches": stable_breaches,
        "river_reaches": stable_reaches,
        "inherited_river_reaches": inherited,
        "inherited_reach_alignment": alignment,
    }
    table_paths = []
    for name, table in table_values.items():
        path = tables_dir / f"{name}.parquet"
        pq.write_table(table, path, compression="zstd", use_dictionary=True)
        table_paths.append(path)
    _write_json_durable(partial / "routing_metadata.json", native_metadata)
    _fsync_paths(table_paths)
    root["progress/routing_complete"][0] = True
    _sync_zarr_array(zarr_path, "progress/routing_complete")
    return native_metadata, stable_lakes, stable_breaches, stable_reaches, inherited, alignment


def _stable_receivers_to_local(
    receiver_cell_id: np.ndarray,
    sources: _HydrologySources,
    height: int,
    width: int,
) -> np.ndarray:
    receiver_cell_id = np.asarray(receiver_cell_id, dtype=np.int64)
    local = np.full(len(receiver_cell_id), -1, dtype=np.int32)
    valid = receiver_cell_id >= 0
    if not np.any(valid):
        return local
    face_size = sources.child_face_resolution * sources.child_face_resolution
    values = receiver_cell_id[valid]
    faces = values // face_size
    within = values % face_size
    rows = within // sources.child_face_resolution
    columns = within % sources.child_face_resolution
    source_face = int(sources.cell_id[0]) // face_size
    minimum_row = int(np.min(sources.row))
    minimum_column = int(np.min(sources.column))
    in_window = (
        (faces == source_face)
        & (rows >= minimum_row)
        & (rows < minimum_row + height)
        & (columns >= minimum_column)
        & (columns < minimum_column + width)
    )
    positions = np.flatnonzero(valid)
    local[positions[in_window]] = (
        (rows[in_window] - minimum_row) * width + (columns[in_window] - minimum_column)
    ).astype(np.int32)
    return local


def _reach_length_km(path: list[int], sources: _HydrologySources) -> float:
    if len(path) < 2:
        return 0.0
    values = np.asarray(path, dtype=np.int64)
    face_size = sources.child_face_resolution * sources.child_face_resolution
    within = values % face_size
    rows = within // sources.child_face_resolution
    columns = within % sources.child_face_resolution
    diagonal = (np.diff(rows) != 0) & (np.diff(columns) != 0)
    return float(
        np.sum(np.where(diagonal, math.sqrt(2.0), 1.0)) * sources.actual_cell_size_m / 1_000.0
    )


def _longest_reach_chain_km(reaches: pa.Table, own_length: np.ndarray) -> float:
    if reaches.num_rows == 0:
        return 0.0
    reach_ids = _column_numpy(reaches, "reach_id", np.dtype(np.int32))
    downstream = _column_numpy(reaches, "downstream_reach_id", np.dtype(np.int32))
    row_by_id = {int(reach_id): row for row, reach_id in enumerate(reach_ids)}
    memo: dict[int, float] = {}

    def length_from(row: int, visiting: set[int]) -> float:
        if row in memo:
            return memo[row]
        if row in visiting:
            raise RuntimeError("published regional reach graph contains a cycle")
        target_id = int(downstream[row])
        target_row = row_by_id.get(target_id)
        remainder = length_from(target_row, visiting | {row}) if target_row is not None else 0.0
        memo[row] = float(own_length[row]) + remainder
        return memo[row]

    return max(length_from(row, set()) for row in range(len(reach_ids)))


def _adjacent_d8(mask: np.ndarray) -> np.ndarray:
    """Return cells touching the mask in any of eight raster directions."""

    source = np.asarray(mask, dtype=bool)
    height, width = source.shape
    adjacent = np.zeros_like(source)
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue
            destination_rows = slice(max(0, -row_offset), min(height, height - row_offset))
            destination_columns = slice(max(0, -column_offset), min(width, width - column_offset))
            source_rows = slice(max(0, row_offset), min(height, height + row_offset))
            source_columns = slice(max(0, column_offset), min(width, width + column_offset))
            adjacent[destination_rows, destination_columns] |= source[source_rows, source_columns]
    return adjacent


def _discharge_continuity_metrics(
    receiver: np.ndarray,
    core: np.ndarray,
    mean_discharge: np.ndarray,
    lake_fraction: np.ndarray,
    wetland_fraction: np.ndarray,
) -> dict[str, int | float]:
    """Audit material downstream flow losses and whether surface water explains them."""

    receiver = np.asarray(receiver, dtype=np.int32)
    core = np.asarray(core, dtype=bool)
    discharge = np.asarray(mean_discharge, dtype=np.float64)
    valid_receiver = receiver >= 0
    downstream_in_core = np.zeros(len(core), dtype=bool)
    downstream_in_core[valid_receiver] = core[receiver[valid_receiver]]
    upstream = np.flatnonzero(core & valid_receiver & downstream_in_core)
    downstream = receiver[upstream]
    loss = discharge[upstream] - discharge[downstream]
    material = loss > np.maximum(0.01 * discharge[upstream], 0.05)
    water = (
        (lake_fraction[upstream] > 1e-6)
        | (lake_fraction[downstream] > 1e-6)
        | (wetland_fraction[upstream] > 1e-6)
        | (wetland_fraction[downstream] > 1e-6)
    )
    unexplained = material & ~water
    relative_loss = loss / np.maximum(discharge[upstream], 0.01)
    return {
        "internal_routed_edge_count": int(len(upstream)),
        "material_mean_discharge_loss_edge_count": int(np.count_nonzero(material)),
        "waterbody_loss_edge_count": int(np.count_nonzero(material & water)),
        "unexplained_mean_discharge_loss_edge_count": int(np.count_nonzero(unexplained)),
        "maximum_material_mean_discharge_relative_loss": float(
            np.max(relative_loss[material], initial=0.0)
        ),
        "maximum_unexplained_mean_discharge_relative_loss": float(
            np.max(relative_loss[unexplained], initial=0.0)
        ),
    }


def _validate_hydrology(
    root: Any,
    config: L3HydrologyConfig,
    sources: _HydrologySources,
    outlet: Mapping[str, int | float],
    spatial_order: np.ndarray,
    height: int,
    width: int,
    forcing_metrics: Mapping[str, Any],
    native_metadata: Mapping[str, Any],
    lakes: pa.Table,
    reaches: pa.Table,
    alignment: pa.Table,
) -> dict[str, Any]:
    inverse = np.empty(len(spatial_order), dtype=np.int32)
    inverse[spatial_order] = np.arange(len(spatial_order), dtype=np.int32)
    inherited_core = sources.inside_core[spatial_order]
    core = np.asarray(root["geometry/inside_routed_catchment_core"][:], dtype=bool)[spatial_order]
    area = sources.area_km2[spatial_order]
    terminal = np.asarray(root["routing/terminal_kind"][:], dtype=np.uint8)[spatial_order]
    basin = np.asarray(root["routing/basin_id"][:], dtype=np.int32)[spatial_order]
    receiver_stable = np.asarray(root["routing/flow_receiver_cell_id"][:], dtype=np.int64)[
        spatial_order
    ]
    receiver = _stable_receivers_to_local(receiver_stable, sources, height, width)
    discharge = np.asarray(root["routing/monthly_discharge_m3s"][:], dtype=np.float32)[
        :, spatial_order
    ]
    mean_discharge = np.asarray(root["routing/mean_discharge_m3s"][:], dtype=np.float32)[
        spatial_order
    ]
    local_runoff = np.asarray(root["forcing/monthly_runoff_mm"][:], dtype=np.float64)[
        :, spatial_order
    ]
    lake_fraction = np.asarray(root["surface/lake_fraction"][:], dtype=np.float32)[spatial_order]
    wetland_fraction = np.asarray(root["surface/wetland_fraction"][:], dtype=np.float32)[
        spatial_order
    ]
    ocean_fraction = np.asarray(root["surface/physical_ocean_fraction"][:], dtype=np.float32)[
        spatial_order
    ]
    reported_support = np.asarray(root["routing/reported_reach_support"][:], dtype=bool)[
        spatial_order
    ]
    connector_support = np.asarray(root["routing/waterbody_flow_connector"][:], dtype=bool)[
        spatial_order
    ]
    downstream_spatial = int(inverse[int(outlet["downstream_cell_row"])])
    outlet_entries = np.flatnonzero(receiver == downstream_spatial)
    outlet_entry = (
        int(outlet_entries[np.argmax(mean_discharge[outlet_entries])])
        if len(outlet_entries)
        else -1
    )
    registered_basin = int(basin[outlet_entry]) if outlet_entry >= 0 else -1
    expected_core = (
        (terminal == TERMINAL_NONE) & (basin == registered_basin)
        if registered_basin >= 0
        else np.zeros(len(core), dtype=bool)
    )
    core_replay_valid = bool(np.array_equal(core, expected_core))
    local_runoff_volume = np.sum(local_runoff, axis=0) * area
    core_area = float(np.sum(area[core]))
    inherited_area = float(np.sum(area[inherited_core]))
    overlap = core & inherited_core
    overlap_area = float(np.sum(area[overlap]))
    inherited_retained_fraction = overlap_area / max(inherited_area, 1e-12)
    routed_inherited_fraction = overlap_area / max(core_area, 1e-12)
    routed_to_inherited_ratio = core_area / max(inherited_area, 1e-12)
    union_area = float(np.sum(area[core | inherited_core]))
    jaccard = overlap_area / max(union_area, 1e-12)
    inherited_runoff = float(np.sum(local_runoff_volume[inherited_core]))
    inherited_runoff_retained_fraction = float(
        np.sum(local_runoff_volume[overlap]) / max(inherited_runoff, 1e-12)
    )
    outside = sources.outside_process[spatial_order].reshape(height, width)
    boundary_contact = core & _adjacent_d8(outside).reshape(-1)
    boundary_contact_area = float(np.sum(area[boundary_contact]))
    boundary_contact_fraction = boundary_contact_area / max(core_area, 1e-12)
    continuity = _discharge_continuity_metrics(
        receiver,
        core,
        mean_discharge,
        lake_fraction,
        wetland_fraction,
    )

    source_outlet_rows = np.flatnonzero(sources.source_parent_ids == sources.outlet_l0_cell_id)
    if len(source_outlet_rows) != 1:
        raise RuntimeError("inherited outlet hydrograph source is not unique")
    inherited_hydrograph = np.asarray(
        sources.monthly_inherited_discharge_m3s[source_outlet_rows[0]], dtype=np.float64
    )
    generated_hydrograph = (
        np.asarray(discharge[:, outlet_entry], dtype=np.float64)
        if outlet_entry >= 0
        else np.zeros(MONTHS, dtype=np.float64)
    )
    outlet_hydrograph_error = float(
        np.sum(np.abs(generated_hydrograph - inherited_hydrograph))
        / max(np.sum(np.abs(inherited_hydrograph)), 1e-9)
    )

    core_land_area = float(np.sum(area[core] * (1.0 - ocean_fraction[core])))
    core_lake_area = float(np.sum(area[core] * lake_fraction[core]))
    core_wetland_area = float(np.sum(area[core] * wetland_fraction[core]))
    core_open_water_fraction = core_lake_area / max(core_land_area, 1e-12)
    routed_lake_ids = np.asarray(root["routing/lake_id"][:], dtype=np.int32)[spatial_order]
    core_lake_ids = routed_lake_ids[core]
    core_lake_count = int(len(np.unique(core_lake_ids[core_lake_ids >= 0])))
    core_lake_cells = core & (routed_lake_ids >= 0)
    core_lake_area_by_id = np.bincount(
        routed_lake_ids[core_lake_cells],
        weights=area[core_lake_cells] * lake_fraction[core_lake_cells],
    )

    reach_core_fraction = (
        _column_numpy(reaches, "inside_core_fraction", np.dtype(np.float32))
        if reaches.num_rows
        else np.empty(0, dtype=np.float32)
    )
    core_reaches = reaches.filter(pa.array(reach_core_fraction > 0.0, type=pa.bool_()))
    own_lengths = np.asarray(
        [_reach_length_km(path, sources) for path in reaches["cell_path"].to_pylist()],
        dtype=np.float64,
    )
    network_length = float(np.sum(own_lengths * reach_core_fraction))
    drainage_density = network_length / max(core_area, 1e-12)
    longest_chain = _longest_reach_chain_km(reaches, own_lengths)
    maximum_strahler = (
        int(np.max(_column_numpy(core_reaches, "strahler_order", np.dtype(np.int32))))
        if core_reaches.num_rows
        else 0
    )
    maximum_core_discharge = float(np.max(mean_discharge[core], initial=0.0))

    outlet_alignment_rows = np.flatnonzero(
        _column_numpy(alignment, "inherited_reach_id", np.dtype(np.int32))
        == int(outlet["inherited_reach_id"])
    )
    outlet_alignment = (
        float(
            _column_numpy(alignment, "supported_l2_fraction", np.dtype(np.float32))[
                outlet_alignment_rows[0]
            ]
        )
        if len(outlet_alignment_rows)
        else 0.0
    )
    alignment_values = (
        _column_numpy(alignment, "supported_l2_fraction", np.dtype(np.float32))
        if alignment.num_rows
        else np.empty(0, dtype=np.float32)
    )

    process_domain = sources.inside_core | sources.inside_process_halo
    process_cell_count = int(np.count_nonzero(process_domain))
    terrain_context_cell_count = int(np.count_nonzero(sources.outside_process))
    process_domain_partition_valid = bool(
        np.array_equal(process_domain, ~sources.outside_process)
        and not np.any(sources.inside_core & sources.inside_process_halo)
    )
    validation: dict[str, Any] = {
        "format_version": HYDROLOGY_FORMAT_VERSION,
        "model_version": HYDROLOGY_MODEL_VERSION,
        "cell_count": len(sources.cell_id),
        "process_domain_cell_count": process_cell_count,
        "process_domain_fraction_of_stored_window": process_cell_count / len(sources.cell_id),
        "terrain_context_cell_count": terrain_context_cell_count,
        "process_domain_partition_valid": int(process_domain_partition_valid),
        "core_cell_count": int(np.count_nonzero(core)),
        "inherited_target_core_cell_count": int(np.count_nonzero(inherited_core)),
        "process_halo_cell_count": int(np.count_nonzero(sources.inside_process_halo)),
        "neighbor_count": 8,
        "native_topology_valid": int(native_metadata["topology_valid"]),
        "native_runoff_conservation_relative_error": float(
            native_metadata["conservation_relative_error"]
        ),
        "maximum_cumulative_prospective_breach_incision_m": float(
            native_metadata["maximum_cumulative_prospective_incision_m"]
        ),
        "depression_count": int(native_metadata["depression_count"]),
        "endorheic_depression_count": int(native_metadata["endorheic_count"]),
        "closed_sink_count": int(native_metadata["closed_sink_count"]),
        **forcing_metrics,
        "registered_outlet_connected": int(outlet_entry >= 0 and registered_basin >= 0),
        "registered_outlet_basin_id": registered_basin,
        "registered_outlet_entry_cell_id": (
            int(sources.cell_id[spatial_order[outlet_entry]]) if outlet_entry >= 0 else -1
        ),
        "routed_core_replay_matches_graph": int(core_replay_valid),
        "inherited_target_core_area_km2": inherited_area,
        "routed_core_area_km2": core_area,
        "routed_core_inherited_overlap_area_km2": overlap_area,
        "routed_core_added_halo_area_km2": float(np.sum(area[core & ~inherited_core])),
        "inherited_core_excluded_area_km2": float(np.sum(area[inherited_core & ~core])),
        "inherited_core_retained_fraction": inherited_retained_fraction,
        "inherited_core_runoff_retained_fraction": inherited_runoff_retained_fraction,
        "routed_core_inherited_fraction": routed_inherited_fraction,
        "routed_to_inherited_area_ratio": routed_to_inherited_ratio,
        "routed_core_inherited_jaccard": jaccard,
        "routed_core_process_boundary_contact_cell_count": int(np.count_nonzero(boundary_contact)),
        "routed_core_process_boundary_contact_area_km2": boundary_contact_area,
        "routed_core_process_boundary_contact_fraction": boundary_contact_fraction,
        **continuity,
        "outlet_hydrograph_relative_error": outlet_hydrograph_error,
        "generated_outlet_monthly_discharge_m3s": generated_hydrograph.tolist(),
        "inherited_outlet_monthly_discharge_m3s": inherited_hydrograph.tolist(),
        "generated_outlet_mean_discharge_m3s": float(np.mean(generated_hydrograph)),
        "inherited_outlet_mean_discharge_m3s": float(np.mean(inherited_hydrograph)),
        "core_land_area_km2": core_land_area,
        "core_lake_area_km2": core_lake_area,
        "core_wetland_area_km2": core_wetland_area,
        "core_open_water_fraction": core_open_water_fraction,
        "core_lake_count": core_lake_count,
        "core_lake_area_max_km2": float(np.max(core_lake_area_by_id, initial=0.0)),
        "core_lake_count_at_least_50_km2": int(np.count_nonzero(core_lake_area_by_id >= 50.0)),
        "lake_catalog_count": lakes.num_rows,
        "river_reach_count": reaches.num_rows,
        "core_river_reach_count": core_reaches.num_rows,
        "core_reported_reach_support_cell_count": int(np.count_nonzero(reported_support & core)),
        "core_waterbody_flow_connector_cell_count": int(np.count_nonzero(connector_support & core)),
        "maximum_core_discharge_m3s": maximum_core_discharge,
        "maximum_core_strahler_order": maximum_strahler,
        "core_reported_network_length_km": network_length,
        "core_reported_drainage_density_km_per_km2": drainage_density,
        "longest_reported_reach_chain_km": longest_chain,
        "inherited_reach_count": alignment.num_rows,
        "inherited_reach_support_p50": (
            float(np.percentile(alignment_values, 50.0)) if len(alignment_values) else 0.0
        ),
        "inherited_outlet_reach_support_fraction": outlet_alignment,
        "maximum_allowed_forcing_conservation_relative_error": (
            config.maximum_forcing_conservation_relative_error
        ),
        "maximum_allowed_runoff_conservation_relative_error": (
            config.maximum_runoff_conservation_relative_error
        ),
        "minimum_inherited_core_retained_fraction": (
            config.minimum_inherited_core_retained_fraction
        ),
        "minimum_routed_core_inherited_fraction": (config.minimum_routed_core_inherited_fraction),
        "minimum_routed_to_inherited_area_ratio": (config.minimum_routed_to_inherited_area_ratio),
        "maximum_routed_to_inherited_area_ratio": (config.maximum_routed_to_inherited_area_ratio),
        "maximum_routed_core_process_boundary_contact_fraction": (
            config.maximum_routed_core_process_boundary_contact_fraction
        ),
        "maximum_allowed_outlet_hydrograph_relative_error": (
            config.maximum_outlet_hydrograph_relative_error
        ),
        "maximum_allowed_core_open_water_fraction": config.maximum_core_open_water_fraction,
        "maximum_allowed_cumulative_prospective_breach_incision_m": (
            config.maximum_cumulative_breach_incision_m
        ),
    }
    gates = {
        "native_topology_valid": validation["native_topology_valid"] == 1,
        "forcing_conservation_valid": (
            validation["maximum_core_forcing_conservation_relative_error"]
            <= config.maximum_forcing_conservation_relative_error
        ),
        "runoff_conservation_valid": (
            validation["native_runoff_conservation_relative_error"]
            <= config.maximum_runoff_conservation_relative_error
        ),
        "process_domain_partition_valid": process_domain_partition_valid,
        "registered_outlet_connected": validation["registered_outlet_connected"] == 1,
        "routed_core_replay_valid": core_replay_valid,
        "inherited_core_retention_valid": (
            inherited_retained_fraction >= config.minimum_inherited_core_retained_fraction
        ),
        "routed_core_overlap_valid": (
            routed_inherited_fraction >= config.minimum_routed_core_inherited_fraction
        ),
        "routed_core_area_ratio_valid": (
            config.minimum_routed_to_inherited_area_ratio
            <= routed_to_inherited_ratio
            <= config.maximum_routed_to_inherited_area_ratio
        ),
        "routed_core_process_boundary_valid": (
            boundary_contact_fraction
            <= config.maximum_routed_core_process_boundary_contact_fraction
        ),
        "river_flow_continuity_valid": (
            continuity["unexplained_mean_discharge_loss_edge_count"] == 0
        ),
        "outlet_hydrograph_valid": (
            outlet_hydrograph_error <= config.maximum_outlet_hydrograph_relative_error
        ),
        "core_open_water_valid": (
            core_open_water_fraction <= config.maximum_core_open_water_fraction
        ),
        "cumulative_prospective_breach_incision_valid": (
            float(native_metadata["maximum_cumulative_prospective_incision_m"])
            <= config.maximum_cumulative_breach_incision_m
        ),
        "river_network_present": core_reaches.num_rows > 0 and maximum_strahler >= 2,
        "inherited_outlet_supported": outlet_alignment >= 0.50,
    }
    validation.update({name: int(value) for name, value in gates.items()})
    validation["passed"] = bool(all(gates.values()))
    return validation


def _hillshaded_terrain(elevation: np.ndarray, cell_size_m: float) -> np.ndarray:
    padded = np.pad(elevation, 1, mode="edge")
    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]
    dzdx = (east - west) / (2.0 * cell_size_m) * TERRAIN_DIAGNOSTIC_VERTICAL_EXAGGERATION
    dzdy = (south - north) / (2.0 * cell_size_m) * TERRAIN_DIAGNOSTIC_VERTICAL_EXAGGERATION
    normal_x = -dzdx
    normal_y = -dzdy
    normal_z = np.ones_like(elevation)
    norm = np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
    illumination = (normal_x * -0.45 + normal_y * -0.55 + normal_z * 0.70) / norm
    shade = np.clip(0.58 + 0.48 * illumination, 0.42, 1.08)
    # Elevation below the datum is not water unless the physical ocean mask says
    # so. Regional closed basins must therefore retain a land hypsometric color.
    land_color_elevation = np.maximum(elevation, 0.0)
    return np.clip(_terrain_colors(land_color_elevation) * shade[..., None], 0.0, 255.0)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    height, width = result.shape
    source = np.asarray(mask, dtype=bool)
    for row_offset in range(-radius, radius + 1):
        for column_offset in range(-radius, radius + 1):
            if row_offset * row_offset + column_offset * column_offset > radius * radius:
                continue
            destination_rows = slice(max(0, -row_offset), min(height, height - row_offset))
            destination_columns = slice(max(0, -column_offset), min(width, width - column_offset))
            source_rows = slice(max(0, row_offset), min(height, height + row_offset))
            source_columns = slice(max(0, column_offset), min(width, width + column_offset))
            result[destination_rows, destination_columns] |= source[source_rows, source_columns]
    return result


def _draw_scale(
    draw: ImageDraw.ImageDraw,
    map_width: int,
    map_height: int,
    title_height: int,
    cell_size_m: float,
) -> None:
    font = _diagnostic_font(15)
    scale_km = _nice_scale_km(map_width, cell_size_m)
    scale_pixels = max(1, round(scale_km * 1_000.0 / cell_size_m))
    scale_x = 24
    scale_y = title_height + map_height + 30
    for segment in range(4):
        left = scale_x + round(scale_pixels * segment / 4)
        right = scale_x + round(scale_pixels * (segment + 1) / 4)
        fill = (31, 35, 33) if segment % 2 == 0 else (240, 241, 237)
        draw.rectangle((left, scale_y, right, scale_y + 12), fill=fill, outline=(31, 35, 33))
    draw.text((scale_x, scale_y + 18), "0", fill=(25, 30, 27), font=font)
    draw.text(
        (scale_x + scale_pixels, scale_y + 18),
        f"{scale_km:g} km",
        fill=(25, 30, 27),
        font=font,
        anchor="ra",
    )
    draw.text(
        (scale_x + scale_pixels + 20, scale_y - 2),
        "approximate scale",
        fill=(70, 74, 70),
        font=font,
    )


def _flow_arrow_cells(
    river: np.ndarray,
    discharge: np.ndarray,
    receiver: np.ndarray,
    *,
    minimum_discharge_m3s: float = 100.0,
    spacing_pixels: int = 90,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose at most one high-discharge downstream arrow per map bin."""

    height, width = river.shape
    source = np.flatnonzero(river.reshape(-1) & (discharge.reshape(-1) >= minimum_discharge_m3s))
    if not len(source):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    target = receiver[source]
    valid = (target >= 0) & (target < height * width) & (target != source)
    source = source[valid]
    target = target[valid]
    if not len(source):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    rows, columns = np.divmod(source, width)
    inset = 8
    interior = (
        (rows >= inset) & (rows < height - inset) & (columns >= inset) & (columns < width - inset)
    )
    source = source[interior]
    target = target[interior]
    rows = rows[interior]
    columns = columns[interior]
    if not len(source):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    bin_columns = math.ceil(width / spacing_pixels)
    bin_id = (rows // spacing_pixels) * bin_columns + columns // spacing_pixels
    order = np.lexsort((source, -discharge.reshape(-1)[source], bin_id))
    sorted_bins = bin_id[order]
    keep = np.concatenate(([True], sorted_bins[1:] != sorted_bins[:-1]))
    chosen = order[keep]
    return source[chosen].astype(np.int32), target[chosen].astype(np.int32)


def _draw_flow_arrows(
    image: Image.Image,
    river: np.ndarray,
    discharge: np.ndarray,
    receiver: np.ndarray,
) -> int:
    source, target = _flow_arrow_cells(river, discharge, receiver)
    draw = ImageDraw.Draw(image)
    width = river.shape[1]
    for upstream, downstream in zip(source, target, strict=True):
        y, x = divmod(int(upstream), width)
        downstream_y, downstream_x = divmod(int(downstream), width)
        dx = float(downstream_x - x)
        dy = float(downstream_y - y)
        length = math.hypot(dx, dy)
        if length == 0.0:
            continue
        dx /= length
        dy /= length
        normal_x, normal_y = -dy, dx
        tip = (x + 6.0 * dx, y + 6.0 * dy)
        base_x = x - 4.0 * dx
        base_y = y - 4.0 * dy
        wing_a = (base_x + 3.5 * normal_x, base_y + 3.5 * normal_y)
        wing_b = (base_x - 3.5 * normal_x, base_y - 3.5 * normal_y)
        draw.line((wing_a, tip, wing_b), fill=(242, 210, 90), width=2)
    return len(source)


def _render_hydrology(
    root: Any,
    sources: _HydrologySources,
    spatial_order: np.ndarray,
    height: int,
    width: int,
    outlet: Mapping[str, int | float],
    validation: Mapping[str, Any],
    path: Path,
) -> None:
    elevation = sources.elevation_m[spatial_order].reshape(height, width)
    colors = _hillshaded_terrain(elevation, sources.actual_cell_size_m)
    ocean = np.asarray(root["surface/physical_ocean_fraction"][:], dtype=np.float32)[
        spatial_order
    ].reshape(height, width)
    lake = np.asarray(root["surface/lake_fraction"][:], dtype=np.float32)[spatial_order].reshape(
        height, width
    )
    wetland = np.asarray(root["surface/wetland_fraction"][:], dtype=np.float32)[
        spatial_order
    ].reshape(height, width)
    process = (sources.inside_core | sources.inside_process_halo)[spatial_order].reshape(
        height, width
    )
    ocean_color = np.asarray((30.0, 93.0, 132.0))
    lake_color = np.asarray((45.0, 132.0, 168.0))
    wetland_color = np.asarray((63.0, 126.0, 91.0))
    colors = colors * (1.0 - ocean[..., None]) + ocean_color * ocean[..., None]
    colors = colors * (1.0 - lake[..., None]) + lake_color * lake[..., None]
    wetland_blend = np.clip(wetland * 0.70, 0.0, 0.70)
    colors = colors * (1.0 - wetland_blend[..., None]) + wetland_color * wetland_blend[..., None]
    terrain_context = ~process
    colors[terrain_context] = (
        colors[terrain_context] * 0.32 + np.asarray((226.0, 227.0, 222.0)) * 0.68
    )
    process_boundary = _inner_boundary(process)
    colors[process_boundary] = np.asarray((156.0, 100.0, 54.0))

    q = np.asarray(root["routing/mean_discharge_m3s"][:], dtype=np.float32)[spatial_order].reshape(
        height, width
    )
    river = (
        np.asarray(root["routing/reported_reach_support"][:], dtype=bool)[spatial_order].reshape(
            height, width
        )
        & process
    )
    connector = np.asarray(root["routing/waterbody_flow_connector"][:], dtype=bool)[
        spatial_order
    ].reshape(height, width)
    lake_id = np.asarray(root["routing/lake_id"][:], dtype=np.int32)
    valid_lake = lake_id >= 0
    lake_area_by_id = np.bincount(
        lake_id[valid_lake],
        weights=sources.area_km2[valid_lake]
        * np.asarray(root["surface/lake_fraction"][:], dtype=np.float32)[valid_lake],
    )
    large_lake_ids = np.flatnonzero(lake_area_by_id >= 50.0)
    large_lake = np.isin(lake_id[spatial_order], large_lake_ids).reshape(height, width)
    # Connectors remain in the canonical graph. Keep them visible through
    # small ponds so trunks do not stutter, but do not paint a river across a
    # physically broad lake or the ocean surface.
    connector_hidden = connector & ((ocean > 0.05) | (large_lake & (lake > 0.05)))
    river &= ~connector_hidden
    river_classes = (
        (river, 0, (43.0, 126.0, 179.0)),
        (river & (q >= 10.0), 1, (27.0, 103.0, 163.0)),
        (river & (q >= 100.0), 2, (17.0, 75.0, 137.0)),
        (river & (q >= 750.0), 3, (11.0, 57.0, 115.0)),
    )
    for mask, radius, color in river_classes:
        support = _dilate(mask, radius) if radius else mask
        colors[support] = np.asarray(color)

    map_image = Image.fromarray(np.clip(colors, 0, 255).astype(np.uint8), mode="RGB")
    receiver_stable = np.asarray(root["routing/flow_receiver_cell_id"][:], dtype=np.int64)[
        spatial_order
    ]
    receiver = _stable_receivers_to_local(receiver_stable, sources, height, width)
    arrow_count = _draw_flow_arrows(map_image, river, q, receiver)
    title_height = 52
    footer_height = 76
    legend_width = 330
    canvas = Image.new(
        "RGB",
        (width + legend_width, height + title_height + footer_height),
        (240, 241, 237),
    )
    canvas.paste(map_image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    title_font = _diagnostic_font(22)
    label_font = _diagnostic_font(17)
    small_font = _diagnostic_font(15)
    draw.text((18, 13), "L3 catchment hydrology", fill=(25, 30, 27), font=title_font)
    draw.line((width, 0, width, canvas.height), fill=(178, 181, 174), width=1)
    legend_x = width + 24
    draw.text((legend_x, 20), "Hydrology coverage", fill=(25, 30, 27), font=title_font)
    y = 70
    draw.rectangle(
        (legend_x, y, legend_x + 34, y + 18),
        fill=(220, 222, 217),
        outline=(50, 55, 52),
    )
    draw.text(
        (legend_x + 48, y - 1),
        "terrain-only context",
        fill=(35, 39, 36),
        font=small_font,
    )
    y += 34
    draw.line((legend_x, y + 8, legend_x + 38, y + 8), fill=(156, 100, 54), width=3)
    draw.text(
        (legend_x + 50, y),
        "hydrology process boundary",
        fill=(35, 39, 36),
        font=small_font,
    )
    y += 34
    draw.text(
        (legend_x, y),
        f"{validation['process_domain_fraction_of_stored_window']:.1%} of stored cells routed",
        fill=(50, 54, 51),
        font=small_font,
    )
    y += 44
    draw.text((legend_x, y), "Surface water", fill=(25, 30, 27), font=label_font)
    y += 36
    physical_ocean_present = validation["physical_ocean_area_km2"] > 0.0
    entries = (
        (
            (30, 93, 132),
            "Physical ocean" if physical_ocean_present else "Physical ocean (none in window)",
        ),
        ((45, 132, 168), "L3 lake"),
        ((63, 126, 91), "L3 wetland"),
    )
    for color, label in entries:
        draw.rectangle((legend_x, y, legend_x + 34, y + 18), fill=color, outline=(50, 55, 52))
        draw.text((legend_x + 48, y - 1), label, fill=(35, 39, 36), font=small_font)
        y += 34
    y += 10
    draw.text((legend_x, y), "River discharge", fill=(25, 30, 27), font=label_font)
    y += 36
    for color, line_width, label in (
        ((43, 126, 179), 1, "minor reach"),
        ((27, 103, 163), 3, ">= 10 m3/s"),
        ((17, 75, 137), 5, ">= 100 m3/s"),
        ((11, 57, 115), 7, ">= 750 m3/s"),
    ):
        draw.line((legend_x, y + 8, legend_x + 38, y + 8), fill=color, width=line_width)
        draw.text((legend_x + 50, y), label, fill=(35, 39, 36), font=small_font)
        y += 34
    draw.line(
        ((legend_x + 21, y + 2), (legend_x + 34, y + 8), (legend_x + 21, y + 14)),
        fill=(201, 157, 24),
        width=2,
    )
    draw.text(
        (legend_x + 50, y),
        f"downstream direction ({arrow_count} arrows)",
        fill=(35, 39, 36),
        font=small_font,
    )
    y += 34
    y += 12
    draw.text((legend_x, y), "Summary", fill=(25, 30, 27), font=label_font)
    y += 34
    summary_lines = (
        f"{validation['core_lake_count']:,} core lakes",
        f"{validation['core_river_reach_count']:,} core reaches",
        f"Q outlet {validation['generated_outlet_mean_discharge_m3s']:.0f} m3/s",
        f"longest chain {validation['longest_reported_reach_chain_km']:.0f} km",
        (
            f"{validation['closed_sink_count']} closed / "
            f"{validation['endorheic_depression_count']} endorheic sinks"
        ),
        "large-lake connectors hidden",
        (
            f"{validation['process_domain_cell_count']:,} process cells / "
            f"{validation['cell_count']:,} stored"
        ),
    )
    for line in summary_lines:
        draw.text((legend_x, y), line, fill=(50, 54, 51), font=small_font)
        y += 25
    downstream_parent_row = int(outlet["downstream_cell_row"])
    downstream_spatial = int(np.flatnonzero(spatial_order == downstream_parent_row)[0])
    outlet_y, outlet_x = divmod(downstream_spatial, width)
    draw.ellipse(
        (
            outlet_x - 5,
            title_height + outlet_y - 5,
            outlet_x + 5,
            title_height + outlet_y + 5,
        ),
        fill=(196, 55, 46),
        outline=(255, 255, 255),
        width=2,
    )
    draw.ellipse((legend_x, y + 4, legend_x + 12, y + 16), fill=(196, 55, 46))
    outlet_label = (
        "regional outlet handoff"
        if physical_ocean_present
        else "inland regional handoff (not coast)"
    )
    draw.text((legend_x + 22, y), outlet_label, fill=(50, 54, 51), font=small_font)
    _draw_scale(draw, width, height, title_height, sources.actual_cell_size_m)
    canvas.save(path, optimize=True)


def _basin_colors(values: np.ndarray) -> np.ndarray:
    identifiers = np.asarray(values, dtype=np.int64)
    hashed = (identifiers.astype(np.uint64) * np.uint64(11400714819323198485)) & np.uint64(
        0xFFFFFFFFFFFFFFFF
    )
    red = 72 + ((hashed >> np.uint64(8)) & np.uint64(127)).astype(np.uint8)
    green = 78 + ((hashed >> np.uint64(24)) & np.uint64(121)).astype(np.uint8)
    blue = 74 + ((hashed >> np.uint64(40)) & np.uint64(127)).astype(np.uint8)
    return np.stack((red, green, blue), axis=-1)


def _render_basins(
    root: Any,
    sources: _HydrologySources,
    spatial_order: np.ndarray,
    height: int,
    width: int,
    validation: Mapping[str, Any],
    path: Path,
) -> None:
    basin = np.asarray(root["routing/basin_id"][:], dtype=np.int32)[spatial_order].reshape(
        height, width
    )
    inherited_core = sources.inside_core[spatial_order].reshape(height, width)
    core = np.asarray(root["geometry/inside_routed_catchment_core"][:], dtype=bool)[
        spatial_order
    ].reshape(height, width)
    halo = sources.inside_process_halo[spatial_order].reshape(height, width)
    process = inherited_core | halo
    colors = np.full((height, width, 3), (218, 220, 215), dtype=np.uint8)
    valid = process & (basin >= 0)
    colors[valid] = _basin_colors(basin[valid])
    registered = int(validation["registered_outlet_basin_id"])
    colors[halo] = np.clip(colors[halo].astype(np.float32) * 0.78 + 48.0, 0, 255).astype(np.uint8)
    colors[core & (basin == registered)] = (91, 157, 107)
    q = np.asarray(root["routing/mean_discharge_m3s"][:], dtype=np.float32)[spatial_order].reshape(
        height, width
    )
    river = (
        np.asarray(root["routing/reported_reach_support"][:], dtype=bool)[spatial_order].reshape(
            height, width
        )
        & process
    )
    colors[river] = (30, 102, 166)
    colors[_dilate(river & (q >= 100.0), 1)] = (14, 68, 128)
    colors[_inner_boundary(inherited_core)] = (181, 101, 57)
    colors[_inner_boundary(core)] = (30, 33, 31)

    map_image = Image.fromarray(colors, mode="RGB")
    title_height = 52
    footer_height = 76
    legend_width = 310
    canvas = Image.new(
        "RGB",
        (width + legend_width, height + title_height + footer_height),
        (240, 241, 237),
    )
    canvas.paste(map_image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    title_font = _diagnostic_font(22)
    label_font = _diagnostic_font(17)
    small_font = _diagnostic_font(15)
    draw.text((18, 13), "L3 drainage basins", fill=(25, 30, 27), font=title_font)
    draw.line((width, 0, width, canvas.height), fill=(178, 181, 174), width=1)
    legend_x = width + 24
    draw.text((legend_x, 20), "Basin roles", fill=(25, 30, 27), font=title_font)
    y = 74
    for color, label in (
        ((91, 157, 107), "registered outlet basin"),
        ((151, 115, 177), "other explicit basin"),
        ((218, 220, 215), "outside process domain"),
    ):
        draw.rectangle((legend_x, y, legend_x + 34, y + 18), fill=color, outline=(55, 59, 56))
        draw.text((legend_x + 48, y - 1), label, fill=(35, 39, 36), font=small_font)
        y += 36
    draw.line((legend_x, y + 8, legend_x + 38, y + 8), fill=(30, 102, 166), width=3)
    draw.text((legend_x + 50, y), "reported river network", fill=(35, 39, 36), font=small_font)
    y += 44
    draw.line((legend_x, y + 8, legend_x + 38, y + 8), fill=(30, 33, 31), width=3)
    draw.text((legend_x + 50, y), "routed acceptance core", fill=(35, 39, 36), font=small_font)
    y += 34
    draw.line((legend_x, y + 8, legend_x + 38, y + 8), fill=(181, 101, 57), width=3)
    draw.text((legend_x + 50, y), "inherited target envelope", fill=(35, 39, 36), font=small_font)
    y += 58
    draw.text((legend_x, y), "Boundary refinement", fill=(25, 30, 27), font=label_font)
    y += 34
    for line in (
        f"{validation['inherited_core_retained_fraction'] * 100:.1f}% inherited retained",
        f"{validation['routed_core_inherited_fraction'] * 100:.1f}% routed overlap",
        f"area ratio {validation['routed_to_inherited_area_ratio']:.3f}",
        (
            "halo-edge contact "
            f"{validation['routed_core_process_boundary_contact_fraction'] * 100:.3f}%"
        ),
    ):
        draw.text((legend_x, y), line, fill=(50, 54, 51), font=small_font)
        y += 26
    _draw_scale(draw, width, height, title_height, sources.actual_cell_size_m)
    canvas.save(path, optimize=True)


def _observed_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1_024


def _load_routing_products(
    partial: Path,
) -> tuple[dict[str, Any], pa.Table, pa.Table, pa.Table, pa.Table, pa.Table]:
    return (
        _load_json(partial / "routing_metadata.json"),
        pq.read_table(partial / "tables/lakes.parquet").combine_chunks(),
        pq.read_table(partial / "tables/breaches.parquet").combine_chunks(),
        pq.read_table(partial / "tables/river_reaches.parquet").combine_chunks(),
        pq.read_table(partial / "tables/inherited_river_reaches.parquet").combine_chunks(),
        pq.read_table(partial / "tables/inherited_reach_alignment.parquet").combine_chunks(),
    )


def _verify_manifest_outputs(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, Mapping):
        raise RuntimeError("L3 hydrology manifest outputs are missing")
    for name, raw in outputs.items():
        if not isinstance(raw, Mapping) or "path" not in raw:
            raise RuntimeError(f"L3 hydrology output record {name} is malformed")
        path = output_dir / str(raw["path"])
        if not path.exists():
            raise RuntimeError(f"L3 hydrology cache output is missing: {path}")
        expected_tree = raw.get("sha256_tree")
        expected_file = raw.get("sha256")
        if expected_tree is not None and _tree_checksum(path) != expected_tree:
            raise RuntimeError(f"L3 hydrology cache tree checksum mismatch: {path}")
        if expected_file is not None and _file_checksum(path) != expected_file:
            raise RuntimeError(f"L3 hydrology cache file checksum mismatch: {path}")


def _existing_result(
    config: L3HydrologyConfig,
    run_fingerprint: str,
) -> L3HydrologyResult | None:
    manifest_path = config.output_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _load_json(manifest_path)
    if manifest.get("run_fingerprint") != run_fingerprint:
        return None
    if not manifest.get("validation_passed"):
        return None
    _verify_manifest_outputs(config.output_dir, manifest)
    summary = manifest["summary"]
    return L3HydrologyResult(
        output_dir=config.output_dir,
        manifest_path=manifest_path,
        validation_path=config.output_dir / "validation.json",
        zarr_path=config.output_dir / "hydrology.zarr",
        preview_path=config.output_dir / "hydrology.png",
        target_id=str(manifest["target_id"]),
        cell_count=int(summary["cell_count"]),
        process_cell_count=int(summary["process_cell_count"]),
        river_reach_count=int(summary["river_reach_count"]),
        lake_count=int(summary["lake_count"]),
        validation_passed=True,
    )


def generate_l3_hydrology(config: L3HydrologyConfig) -> L3HydrologyResult:
    """Generate conservative forcing, regional routing, lakes, and river vectors."""

    started = time.perf_counter()
    sources = _load_sources(config)
    run_fingerprint, fingerprint_components = _fingerprint(config, sources)
    existing = _existing_result(config, run_fingerprint)
    if existing is not None:
        return existing

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = config.output_dir.with_name(f".{config.output_dir.name}.partial")
    root, resumed = _open_partial(partial, config, sources, run_fingerprint)
    spatial_order, _, height, width = _spatial_layout(sources)
    outlet = _registered_outlet(sources, spatial_order, height, width)

    forcing_resumed = bool(np.asarray(root["progress/forcing_complete"][:], dtype=bool)[0])
    if forcing_resumed:
        forcing_metrics = _forcing_metrics(root, sources)
    else:
        forcing_metrics = _write_forcing(root, partial, config, sources, outlet)

    routing_resumed = bool(np.asarray(root["progress/routing_complete"][:], dtype=bool)[0])
    if routing_resumed:
        native_metadata, lakes, breaches, reaches, inherited, alignment = _load_routing_products(
            partial
        )
    else:
        native_metadata, lakes, breaches, reaches, inherited, alignment = _run_routing(
            root,
            partial,
            config,
            sources,
            outlet,
            spatial_order,
            height,
            width,
        )

    validation = _validate_hydrology(
        root,
        config,
        sources,
        outlet,
        spatial_order,
        height,
        width,
        forcing_metrics,
        native_metadata,
        lakes,
        reaches,
        alignment,
    )
    preview_path = partial / "hydrology.png"
    basin_path = partial / "drainage_basins.png"
    _render_hydrology(
        root,
        sources,
        spatial_order,
        height,
        width,
        outlet,
        validation,
        preview_path,
    )
    _render_basins(
        root,
        sources,
        spatial_order,
        height,
        width,
        validation,
        basin_path,
    )

    zarr_path = partial / "hydrology.zarr"
    root.attrs.update(
        {
            "status": "complete" if validation["passed"] else "validation_failed",
            "registered_outlet_cell_id": int(outlet["downstream_cell_id"]),
            "registered_outlet_basin_id": int(validation["registered_outlet_basin_id"]),
        }
    )
    zarr.consolidate_metadata(str(zarr_path))
    _fsync_paths([zarr_path / ".zattrs", zarr_path / ".zmetadata"])
    observed_peak = _observed_peak_rss_bytes()
    estimated_peak = int(len(sources.cell_id) * 320 + len(sources.cell_id) * MONTHS * 16)
    peak_memory = max(observed_peak, estimated_peak)
    storage_bytes = sum(path.stat().st_size for path in partial.rglob("*") if path.is_file())
    validation.update(
        {
            "observed_peak_rss_bytes": observed_peak,
            "estimated_peak_working_set_bytes": estimated_peak,
            "peak_memory_gate_bytes": peak_memory,
            "maximum_peak_memory_bytes": int(config.maximum_peak_memory_gb * 1024**3),
            "peak_memory_budget_valid": int(
                peak_memory <= int(config.maximum_peak_memory_gb * 1024**3)
            ),
            "storage_bytes_before_manifest": storage_bytes,
            "maximum_storage_bytes": int(config.maximum_storage_gb * 1024**3),
            "storage_budget_valid": int(storage_bytes <= int(config.maximum_storage_gb * 1024**3)),
        }
    )
    validation["passed"] = bool(
        validation["passed"]
        and validation["peak_memory_budget_valid"] == 1
        and validation["storage_budget_valid"] == 1
    )
    validation_path = partial / "validation.json"
    _write_json_durable(validation_path, validation)

    table_outputs = {}
    for path in sorted((partial / "tables").glob("*.parquet")):
        table_outputs[path.stem] = {
            "path": f"tables/{path.name}",
            "rows": pq.read_metadata(path).num_rows,
            "sha256": _file_checksum(path),
        }
    elapsed = time.perf_counter() - started
    manifest = {
        "format_version": HYDROLOGY_FORMAT_VERSION,
        "model_version": HYDROLOGY_MODEL_VERSION,
        "status": "complete" if validation["passed"] else "validation_failed",
        "target_id": sources.target_id,
        "run_fingerprint": run_fingerprint,
        "summary": {
            "cell_count": len(sources.cell_id),
            "process_cell_count": validation["process_domain_cell_count"],
            "terrain_context_cell_count": validation["terrain_context_cell_count"],
            "core_cell_count": validation["core_cell_count"],
            "river_reach_count": reaches.num_rows,
            "core_river_reach_count": validation["core_river_reach_count"],
            "lake_count": validation["core_lake_count"],
            "generated_outlet_mean_discharge_m3s": validation[
                "generated_outlet_mean_discharge_m3s"
            ],
            "longest_reported_reach_chain_km": validation["longest_reported_reach_chain_km"],
        },
        "model": {
            "routing": "D8 priority flood with explicit depressions, fill/spill, water balance, and bounded prospective breach controls",
            "forcing": "L0 monthly priors redistributed by L3 orography with exact represented-parent water-volume conservation",
            "river_identity": "generated L3 reach graph plus unmodified inherited L2 trunk constraints",
            "regional_outlet": "registered L2-to-L3 handoff terminal; not assumed to be an ocean mouth",
            "terrain_change": "none; hydrologic elevation and prospective breach incision are separate from base terrain",
            "water_occupancy": "physical ocean realized from L2 fractions; lakes and wetlands recomputed from L3 depressions",
            "hydrological_acceptance_scope": "fine routed catchment core only",
            "storage_scope": "complete terrain rectangle; hydrology is solved only on catchment core plus process halo",
            "neighbor_count": 8,
        },
        "registered_outlet": dict(outlet),
        "resume": {
            "resumed_partial": resumed,
            "forcing_resumed": forcing_resumed,
            "routing_resumed": routing_resumed,
            "elapsed_seconds_this_run": elapsed,
        },
        "source": {
            "target_dir": str(config.target_dir),
            "terrain_dir": str(config.terrain_dir),
            "handoff_dir": str(sources.handoff_dir),
            **fingerprint_components,
        },
        "outputs": {
            "hydrology_zarr": {
                "path": "hydrology.zarr",
                "sha256_tree": _tree_checksum(zarr_path),
            },
            **table_outputs,
            "routing_metadata": {
                "path": "routing_metadata.json",
                "sha256": _file_checksum(partial / "routing_metadata.json"),
            },
            "hydrology_preview": {
                "path": "hydrology.png",
                "sha256": _file_checksum(preview_path),
                "projection": "native cubed-sphere face raster diagnostic",
                "legend": "physical water and symbolized discharge hierarchy",
                "scale": "approximate kilometre scale from L3 area-equivalent cell width",
            },
            "drainage_basin_preview": {
                "path": "drainage_basins.png",
                "sha256": _file_checksum(basin_path),
                "projection": "native cubed-sphere face raster diagnostic",
                "legend": "categorical routed basins, core, and rivers",
                "scale": "approximate kilometre scale from L3 area-equivalent cell width",
            },
            "validation": {
                "path": "validation.json",
                "sha256": _file_checksum(validation_path),
            },
        },
        "validation_passed": bool(validation["passed"]),
    }
    manifest_path = partial / "manifest.json"
    _write_json_durable(manifest_path, manifest)
    if not validation["passed"]:
        raise RuntimeError(
            f"L3 hydrology failed validation; diagnostics retained in {partial}: {validation}"
        )
    _replace_directory(partial, config.output_dir)
    return L3HydrologyResult(
        output_dir=config.output_dir,
        manifest_path=config.output_dir / "manifest.json",
        validation_path=config.output_dir / "validation.json",
        zarr_path=config.output_dir / "hydrology.zarr",
        preview_path=config.output_dir / "hydrology.png",
        target_id=sources.target_id,
        cell_count=len(sources.cell_id),
        process_cell_count=int(validation["process_domain_cell_count"]),
        river_reach_count=reaches.num_rows,
        lake_count=int(validation["core_lake_count"]),
        validation_passed=True,
    )


__all__ = [
    "HYDROLOGY_FORMAT_VERSION",
    "HYDROLOGY_MODEL_VERSION",
    "L3HydrologyConfig",
    "L3HydrologyResult",
    "generate_l3_hydrology",
]
