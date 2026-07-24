"""Chunked L3 surface materials, initial soils, and monthly soil water."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
import yaml  # type: ignore[import-untyped]
import zarr  # type: ignore[import-untyped]

from .._native import native_library_info
from ._surface_materials_native import run_surface_materials
from .l3_channel_geometry import _verify_manifest_outputs
from .l3_hydrology import (
    _draw_scale,
    _hillshaded_terrain,
    _observed_peak_rss_bytes,
    _rectangular_mask_slices,
)
from .l3_terrain import (
    _diagnostic_font,
    _file_checksum,
    _fsync_paths,
    _sync_zarr_array,
    _sync_zarr_chunk,
    _tree_checksum,
    _write_json_durable,
    _zarr_chunk_path,
    _zarr_dataset,
)
from .regional_handoff import _replace_directory
from .stages.surface_materials import (
    MATERIAL_COLORS,
    MATERIAL_OUTPUTS,
    MONTHLY_OUTPUTS,
    SOIL_OUTPUTS,
    SurfaceMaterialsConfig,
)

SURFACE_MATERIALS_FORMAT_VERSION = 1
SURFACE_MATERIALS_MODEL_VERSION = "l3_surface_materials_v2"
MONTHS = 12

NATIVE_OUTPUT_NAMES = {
    "bedrock_out": MATERIAL_OUTPUTS[0],
    "residual_out": MATERIAL_OUTPUTS[1],
    "colluvium_out": MATERIAL_OUTPUTS[2],
    "alluvium_out": MATERIAL_OUTPUTS[3],
    "lacustrine_out": MATERIAL_OUTPUTS[4],
    "glacial_out": MATERIAL_OUTPUTS[5],
    "volcaniclastic_out": MATERIAL_OUTPUTS[6],
    "dominant_material_out": "DominantSurfaceMaterialCode",
    "soil_bearing_out": SOIL_OUTPUTS[0],
    "regolith_depth_out": SOIL_OUTPUTS[1],
    "soil_depth_out": SOIL_OUTPUTS[2],
    "sand_out": SOIL_OUTPUTS[3],
    "silt_out": SOIL_OUTPUTS[4],
    "clay_out": SOIL_OUTPUTS[5],
    "coarse_fragments_out": SOIL_OUTPUTS[6],
    "bulk_density_out": SOIL_OUTPUTS[7],
    "organic_carbon_out": SOIL_OUTPUTS[8],
    "soil_ph_out": SOIL_OUTPUTS[9],
    "carbonate_out": SOIL_OUTPUTS[10],
    "salinity_out": SOIL_OUTPUTS[11],
    "drainage_out": SOIL_OUTPUTS[12],
    "available_water_capacity_out": SOIL_OUTPUTS[13],
    "nutrient_potential_out": SOIL_OUTPUTS[14],
    "fertility_potential_out": SOIL_OUTPUTS[15],
    "erodibility_out": SOIL_OUTPUTS[16],
    "reset_age_out": SOIL_OUTPUTS[17],
    "hydric_fraction_out": SOIL_OUTPUTS[18],
    "soil_confidence_out": SOIL_OUTPUTS[19],
    "annual_storage_change_out": SOIL_OUTPUTS[20],
    "monthly_liquid_input_out": MONTHLY_OUTPUTS[0],
    "monthly_soil_water_out": MONTHLY_OUTPUTS[1],
    "monthly_saturation_out": MONTHLY_OUTPUTS[2],
    "monthly_evapotranspiration_out": MONTHLY_OUTPUTS[3],
    "monthly_runoff_out": MONTHLY_OUTPUTS[4],
    "monthly_deep_drainage_out": MONTHLY_OUTPUTS[5],
}

MATERIAL_PATHS = {name: f"materials/{name}" for name in MATERIAL_OUTPUTS}
MATERIAL_PATHS["DominantSurfaceMaterialCode"] = "materials/DominantSurfaceMaterialCode"
SOIL_PATHS = {name: f"soil/{name}" for name in SOIL_OUTPUTS}
MONTHLY_PATHS = {name: f"monthly/{name}" for name in MONTHLY_OUTPUTS}
DRIVER_PATHS = {
    "LocalReliefM": "drivers/LocalReliefM",
    "LocalTerrainSlope": "drivers/LocalTerrainSlope",
    "TemperatureAdjustmentC": "drivers/TemperatureAdjustmentC",
    "AlluvialLegacyFraction": "drivers/AlluvialLegacyFraction",
}
BOUNDED_SOIL_OUTPUTS = (
    "SoilBearingFraction",
    "SandFraction",
    "SiltFraction",
    "ClayFraction",
    "CoarseFragmentFraction",
    "SoilCarbonateFraction",
    "SoilSalinityIndex",
    "SoilDrainageIndex",
    "SoilNutrientPotential",
    "SoilFertilityPotential",
    "SoilErodibility",
    "HydricSoilFraction",
    "SoilConfidence",
)
NONNEGATIVE_SOIL_OUTPUTS = tuple(
    name for name in SOIL_OUTPUTS if name != "AnnualSoilWaterStorageChangeMm"
)
OUTPUT_PATHS = {
    **MATERIAL_PATHS,
    **SOIL_PATHS,
    **MONTHLY_PATHS,
    **DRIVER_PATHS,
}

REQUIRED_PARENT_PRIORS = (
    "parent_priors/geology/GeologicalProvinceClass",
    "parent_priors/geology/CrustAgeGa",
    "parent_priors/geology/RockStrength",
    "parent_priors/geology/SedimentAccommodation",
    "parent_priors/geology/ProvinceConfidence",
    "parent_priors/elevation/ElevationConfidence",
    "parent_priors/climate/ClimateOrographyM",
    "parent_priors/climate/MonthlySurfaceTemperatureC",
    "parent_priors/climate/MonthlyPrecipitationMm",
    "parent_priors/climate/MonthlyEvaporationMm",
    "parent_priors/climate/MonthlySnowfallMm",
    "parent_priors/cryosphere/GlacierIceFraction",
    *(f"parent_priors/surface_materials/{name}" for name in MATERIAL_OUTPUTS),
    "parent_priors/surface_materials/SoilDepthM",
    "parent_priors/surface_materials/SoilPH",
)


@dataclass(frozen=True)
class L3SurfaceMaterialsConfig:
    terrain_dir: Path
    hydrology_dir: Path
    channel_geometry_dir: Path
    output_dir: Path
    chunk_rows: int = 131_072
    local_relief_radius_cells: int = 12
    terrain_slope_smoothing_radius_cells: int = 12
    temperature_lapse_rate_c_per_m: float = 0.0065
    maximum_temperature_adjustment_c: float = 12.0
    spinup_years: int = 24
    maximum_regolith_depth_m: float = 20.0
    maximum_soil_depth_m: float = 3.0
    maximum_alluvial_fraction: float = 0.65
    maximum_lacustrine_fraction: float = 0.85
    maximum_glacial_fraction: float = 0.80
    alluvial_legacy_decay_m: float = 2_500.0
    alluvial_legacy_slope_scale: float = 0.012
    alluvial_legacy_background: float = 0.20
    alluvial_legacy_proximity_boost: float = 2.50
    weathering_temperature_scale_c: float = 22.0
    weathering_precipitation_scale_mm: float = 1_600.0
    soil_evaporation_factor: float = 1.0
    monthly_deep_drainage_fraction: float = 0.06
    maximum_component_balance_error: float = 1e-5
    maximum_water_balance_relative_error: float = 1e-5
    maximum_peak_memory_gb: float = 24.0
    maximum_storage_gb: float = 4.0
    source_config: Path | None = None

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        output_dir: Path | None = None,
    ) -> "L3SurfaceMaterialsConfig":
        source = Path(path).expanduser().resolve()
        data = yaml.safe_load(source.read_text(encoding="utf8"))
        if not isinstance(data, Mapping):
            raise TypeError("L3 surface-material config must contain a mapping")
        raw_terrain = data.get("terrain_output_dir")
        raw_hydrology = data.get("hydrology_output_dir")
        raw_channels = data.get("channel_geometry_output_dir")
        raw_output = data.get("surface_materials_output_dir")
        if not raw_terrain or not raw_hydrology or not raw_channels or not raw_output:
            raise ValueError(
                "L3 surface materials require terrain_output_dir, hydrology_output_dir, "
                "channel_geometry_output_dir, and surface_materials_output_dir"
            )
        controls = data.get("l3_surface_materials", {})
        limits = data.get("limits", {})
        if not isinstance(controls, Mapping) or not isinstance(limits, Mapping):
            raise TypeError("L3 l3_surface_materials and limits controls must be mappings")
        known = {
            "chunk_rows",
            "local_relief_radius_cells",
            "terrain_slope_smoothing_radius_cells",
            "temperature_lapse_rate_c_per_m",
            "maximum_temperature_adjustment_c",
            "spinup_years",
            "maximum_regolith_depth_m",
            "maximum_soil_depth_m",
            "maximum_alluvial_fraction",
            "maximum_lacustrine_fraction",
            "maximum_glacial_fraction",
            "alluvial_legacy_decay_m",
            "alluvial_legacy_slope_scale",
            "alluvial_legacy_background",
            "alluvial_legacy_proximity_boost",
            "weathering_temperature_scale_c",
            "weathering_precipitation_scale_mm",
            "soil_evaporation_factor",
            "monthly_deep_drainage_fraction",
            "maximum_component_balance_error",
            "maximum_water_balance_relative_error",
        }
        unknown = set(controls) - known
        if unknown:
            raise ValueError(f"Unknown L3 surface-material controls: {', '.join(sorted(unknown))}")
        integer_names = {
            "chunk_rows",
            "local_relief_radius_cells",
            "terrain_slope_smoothing_radius_cells",
            "spinup_years",
        }
        values: dict[str, int | float] = {}
        for name in known:
            if name in controls:
                values[name] = (
                    int(controls[name]) if name in integer_names else float(controls[name])
                )
        config = cls(
            terrain_dir=(source.parent / str(raw_terrain)).resolve(),
            hydrology_dir=(source.parent / str(raw_hydrology)).resolve(),
            channel_geometry_dir=(source.parent / str(raw_channels)).resolve(),
            output_dir=(
                output_dir.expanduser().resolve()
                if output_dir is not None
                else (source.parent / str(raw_output)).resolve()
            ),
            maximum_peak_memory_gb=float(limits.get("maximum_peak_memory_gb", 24.0)),
            maximum_storage_gb=float(limits.get("maximum_surface_materials_storage_gb", 4.0)),
            source_config=source,
            **values,
        )
        config.validate()
        return config

    def validate(self) -> None:
        _require_disjoint_output(
            self.output_dir,
            (
                self.terrain_dir,
                self.hydrology_dir,
                self.channel_geometry_dir,
            ),
        )
        if not 16_384 <= self.chunk_rows <= 524_288:
            raise ValueError("l3_surface_materials.chunk_rows must be in [16384, 524288]")
        if not 1 <= self.local_relief_radius_cells <= 64:
            raise ValueError("l3_surface_materials.local_relief_radius_cells must be in [1, 64]")
        if not 1 <= self.terrain_slope_smoothing_radius_cells <= 64:
            raise ValueError(
                "l3_surface_materials.terrain_slope_smoothing_radius_cells must be in [1, 64]"
            )
        if (
            not math.isfinite(self.temperature_lapse_rate_c_per_m)
            or not 0.0 <= self.temperature_lapse_rate_c_per_m <= 0.02
        ):
            raise ValueError(
                "l3_surface_materials.temperature_lapse_rate_c_per_m must be in [0, 0.02]"
            )
        if (
            not math.isfinite(self.maximum_temperature_adjustment_c)
            or not 0.0 < self.maximum_temperature_adjustment_c <= 40.0
        ):
            raise ValueError(
                "l3_surface_materials.maximum_temperature_adjustment_c must be in (0, 40]"
            )
        SurfaceMaterialsConfig.from_mapping(
            {
                "spinup_years": self.spinup_years,
                "maximum_regolith_depth_m": self.maximum_regolith_depth_m,
                "maximum_soil_depth_m": self.maximum_soil_depth_m,
                "maximum_alluvial_fraction": self.maximum_alluvial_fraction,
                "maximum_lacustrine_fraction": self.maximum_lacustrine_fraction,
                "maximum_glacial_fraction": self.maximum_glacial_fraction,
                "weathering_temperature_scale_c": self.weathering_temperature_scale_c,
                "weathering_precipitation_scale_mm": (self.weathering_precipitation_scale_mm),
                "soil_evaporation_factor": self.soil_evaporation_factor,
                "monthly_deep_drainage_fraction": self.monthly_deep_drainage_fraction,
                "maximum_component_balance_error": self.maximum_component_balance_error,
                "maximum_water_balance_relative_error": (self.maximum_water_balance_relative_error),
            }
        )
        if not 100.0 <= self.alluvial_legacy_decay_m <= 20_000.0:
            raise ValueError("l3_surface_materials.alluvial_legacy_decay_m must be in [100, 20000]")
        if not 1e-4 <= self.alluvial_legacy_slope_scale <= 0.2:
            raise ValueError(
                "l3_surface_materials.alluvial_legacy_slope_scale must be in [0.0001, 0.2]"
            )
        if not 0.0 <= self.alluvial_legacy_background <= 1.0:
            raise ValueError("l3_surface_materials.alluvial_legacy_background must be in [0, 1]")
        if not 0.0 <= self.alluvial_legacy_proximity_boost <= 10.0:
            raise ValueError(
                "l3_surface_materials.alluvial_legacy_proximity_boost must be in [0, 10]"
            )
        if not 1.0 <= self.maximum_peak_memory_gb <= 28.0:
            raise ValueError("limits.maximum_peak_memory_gb must be in [1, 28]")
        if not 0.5 <= self.maximum_storage_gb <= 12.0:
            raise ValueError("limits.maximum_surface_materials_storage_gb must be in [0.5, 12]")


@dataclass(frozen=True)
class L3SurfaceMaterialsResult:
    output_dir: Path
    manifest_path: Path
    validation_path: Path
    zarr_path: Path
    preview_path: Path
    target_id: str
    display_cell_count: int
    chunk_count: int
    validation_passed: bool


@dataclass(frozen=True)
class _MaterialSources:
    target_id: str
    terrain_manifest: dict[str, Any]
    hydrology_manifest: dict[str, Any]
    channel_manifest: dict[str, Any]
    handoff_manifest: dict[str, Any]
    handoff_dir: Path
    terrain: Any
    hydrology: Any
    channels: Any
    parent_ids: np.ndarray
    parent_priors: dict[str, np.ndarray]
    cell_id: np.ndarray
    face: np.ndarray
    row: np.ndarray
    column: np.ndarray
    l0_parent_id: np.ndarray
    area_km2: np.ndarray
    elevation_m: np.ndarray
    local_relief_m: np.ndarray
    local_terrain_slope: np.ndarray
    inside_display: np.ndarray
    inside_core: np.ndarray
    spatial_order: np.ndarray
    height: int
    width: int
    parent_face_resolution: int
    child_face_resolution: int
    actual_cell_size_m: float


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def _require_disjoint_output(output_dir: Path, source_dirs: tuple[Path, ...]) -> None:
    output = output_dir.resolve()
    partial = output.with_name(f".{output.name}.partial")
    for source_dir in source_dirs:
        source = source_dir.resolve()
        for candidate in (output, partial):
            if (
                candidate == source
                or candidate.is_relative_to(source)
                or source.is_relative_to(candidate)
            ):
                raise ValueError(
                    f"L3 surface-material output {candidate} overlaps source artifact {source}"
                )


def _verify_handoff_outputs(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RuntimeError(f"{output_dir} manifest has no output records")
    zarr_record = outputs.get("zarr")
    if not isinstance(zarr_record, Mapping) or "path" not in zarr_record:
        raise RuntimeError(f"Malformed Zarr output record in {output_dir}")
    zarr_path = output_dir / str(zarr_record["path"])
    if not zarr_path.exists():
        raise RuntimeError(f"Cached output is missing: {zarr_path}")
    if "sha256" in zarr_record and _tree_checksum(zarr_path) != zarr_record["sha256"]:
        raise RuntimeError(f"Cached tree checksum mismatch: {zarr_path}")
    tables = outputs.get("tables")
    if not isinstance(tables, Mapping):
        raise RuntimeError(f"Malformed output record tables in {output_dir}")
    for name, expected_checksum in tables.items():
        path = output_dir / "tables" / str(name)
        if not path.exists():
            raise RuntimeError(f"Cached output is missing: {path}")
        if _file_checksum(path) != expected_checksum:
            raise RuntimeError(f"Cached file checksum mismatch: {path}")
    for name in ("preview", "validation"):
        record = outputs.get(name)
        if not isinstance(record, Mapping) or "path" not in record:
            raise RuntimeError(f"Malformed output record {name} in {output_dir}")
        path = output_dir / str(record["path"])
        if not path.exists():
            raise RuntimeError(f"Cached output is missing: {path}")
        if "sha256" in record and _file_checksum(path) != record["sha256"]:
            raise RuntimeError(f"Cached file checksum mismatch: {path}")


def _require_source_manifest_checksum(
    manifest: Mapping[str, Any],
    key: str,
    expected_checksum: str,
    label: str,
) -> None:
    source = manifest.get("source")
    actual = source.get(key) if isinstance(source, Mapping) else None
    if actual != expected_checksum:
        raise RuntimeError(
            f"{label} source lineage mismatch for {key}: "
            f"expected {expected_checksum}, found {actual}"
        )


def _spatial_layout(
    row: np.ndarray,
    column: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    order = np.lexsort((column, row)).astype(np.int32, copy=False)
    height = int(np.max(row) - np.min(row) + 1)
    width = int(np.max(column) - np.min(column) + 1)
    if height * width != len(order):
        raise RuntimeError("L3 surface materials require a dense rectangular terrain window")
    expected_row, expected_column = np.indices((height, width), dtype=np.int32)
    if not np.array_equal(
        row[order] - int(np.min(row)), expected_row.reshape(-1)
    ) or not np.array_equal(column[order] - int(np.min(column)), expected_column.reshape(-1)):
        raise RuntimeError("L3 surface-material coordinates are not row-major dense")
    return np.ascontiguousarray(order), height, width


def _local_relief(
    elevation_m: np.ndarray,
    spatial_order: np.ndarray,
    height: int,
    width: int,
    radius: int,
) -> np.ndarray:
    elevation = elevation_m[spatial_order].reshape(height, width)
    size = radius * 2 + 1
    relief = maximum_filter(elevation, size=size, mode="nearest") - minimum_filter(
        elevation,
        size=size,
        mode="nearest",
    )
    parent_major = np.empty(len(elevation_m), dtype=np.float32)
    parent_major[spatial_order] = np.asarray(relief, dtype=np.float32).reshape(-1)
    return parent_major


def _local_terrain_slope(
    elevation_m: np.ndarray,
    spatial_order: np.ndarray,
    height: int,
    width: int,
    cell_size_m: float,
    radius: int,
) -> np.ndarray:
    if not math.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("local terrain slope requires a positive finite cell size")
    if radius < 1:
        raise ValueError("local terrain slope requires a positive smoothing radius")
    elevation = elevation_m[spatial_order].reshape(height, width).astype(np.float64)
    smoothed = gaussian_filter(
        elevation,
        sigma=max(radius / 2.0, 0.5),
        mode="nearest",
        truncate=2.0,
    )
    north_south, east_west = np.gradient(smoothed, cell_size_m)
    slope = np.hypot(east_west, north_south)
    parent_major = np.empty(len(elevation_m), dtype=np.float32)
    parent_major[spatial_order] = np.asarray(slope, dtype=np.float32).reshape(-1)
    return parent_major


def _load_sources(config: L3SurfaceMaterialsConfig) -> _MaterialSources:
    source_dirs = (
        config.terrain_dir,
        config.hydrology_dir,
        config.channel_geometry_dir,
    )
    manifests = []
    for directory in source_dirs:
        manifest_path = directory / "manifest.json"
        validation_path = directory / "validation.json"
        if not manifest_path.exists() or not validation_path.exists():
            raise FileNotFoundError(
                manifest_path if not manifest_path.exists() else validation_path
            )
        manifest = _load_json(manifest_path)
        validation = _load_json(validation_path)
        if not manifest.get("validation_passed") or not validation.get("passed"):
            raise RuntimeError(f"L3 surface materials require accepted source: {directory}")
        _verify_manifest_outputs(directory, manifest)
        manifests.append(manifest)
    terrain_manifest, hydrology_manifest, channel_manifest = manifests
    target_ids = {
        str(terrain_manifest.get("target_id")),
        str(hydrology_manifest.get("target_id")),
        str(channel_manifest.get("target_id")),
    }
    if len(target_ids) != 1:
        raise RuntimeError("L3 surface-material source target IDs differ")
    terrain_manifest_sha256 = _file_checksum(config.terrain_dir / "manifest.json")
    hydrology_manifest_sha256 = _file_checksum(config.hydrology_dir / "manifest.json")
    _require_source_manifest_checksum(
        hydrology_manifest,
        "terrain_manifest_sha256",
        terrain_manifest_sha256,
        "hydrology",
    )
    _require_source_manifest_checksum(
        channel_manifest,
        "terrain_manifest_sha256",
        terrain_manifest_sha256,
        "channel geometry",
    )
    _require_source_manifest_checksum(
        channel_manifest,
        "hydrology_manifest_sha256",
        hydrology_manifest_sha256,
        "channel geometry",
    )
    handoff_dir = Path(str(terrain_manifest["source"]["handoff_dir"])).resolve()
    _require_disjoint_output(
        config.output_dir,
        (
            config.terrain_dir,
            config.hydrology_dir,
            config.channel_geometry_dir,
            handoff_dir,
        ),
    )
    handoff_manifest_path = handoff_dir / "manifest.json"
    handoff_validation_path = handoff_dir / "validation.json"
    handoff_zarr_path = handoff_dir / "region.zarr"
    for path in (handoff_manifest_path, handoff_validation_path, handoff_zarr_path):
        if not path.exists():
            raise FileNotFoundError(path)
    handoff_manifest = _load_json(handoff_manifest_path)
    if not handoff_manifest.get("validation_passed") or not _load_json(handoff_validation_path).get(
        "passed"
    ):
        raise RuntimeError("L3 surface materials require an accepted regional handoff")
    _verify_handoff_outputs(handoff_dir, handoff_manifest)
    handoff_manifest_sha256 = _file_checksum(handoff_manifest_path)
    _require_source_manifest_checksum(
        terrain_manifest,
        "handoff_manifest_sha256",
        handoff_manifest_sha256,
        "terrain",
    )
    _require_source_manifest_checksum(
        hydrology_manifest,
        "handoff_manifest_sha256",
        handoff_manifest_sha256,
        "hydrology",
    )

    terrain = zarr.open_group(str(config.terrain_dir / "terrain.zarr"), mode="r")
    hydrology = zarr.open_group(str(config.hydrology_dir / "hydrology.zarr"), mode="r")
    channels = zarr.open_group(
        str(config.channel_geometry_dir / "channel_geometry.zarr"),
        mode="r",
    )
    handoff = zarr.open_group(str(handoff_zarr_path), mode="r")
    parent_ids = np.asarray(handoff["parent/cell_id"][:], dtype=np.int32)
    parent_order = np.argsort(parent_ids)
    parent_ids = np.ascontiguousarray(parent_ids[parent_order])
    parent_priors: dict[str, np.ndarray] = {}
    for path in REQUIRED_PARENT_PRIORS:
        if path not in handoff:
            raise KeyError(f"Regional handoff lacks required prior {path}")
        parent_priors[path] = np.ascontiguousarray(np.asarray(handoff[path][:])[parent_order])

    cell_id = np.asarray(terrain["geometry/cell_id"][:], dtype=np.uint64)
    face = np.asarray(terrain["geometry/face"][:], dtype=np.uint8)
    row = np.asarray(terrain["geometry/row"][:], dtype=np.int32)
    column = np.asarray(terrain["geometry/column"][:], dtype=np.int32)
    area_km2 = np.asarray(terrain["geometry/area_km2"][:], dtype=np.float64)
    elevation_m = np.asarray(terrain["terrain/elevation_m"][:], dtype=np.float32)
    l0_parent_id = np.asarray(hydrology["geometry/l0_parent_cell_id"][:], dtype=np.int32)
    inside_display = np.asarray(hydrology["geometry/inside_display_window"][:], dtype=bool)
    inside_core = np.asarray(hydrology["geometry/inside_routed_catchment_core"][:], dtype=bool)
    arrays = (
        face,
        row,
        column,
        area_km2,
        elevation_m,
        l0_parent_id,
        inside_display,
        inside_core,
    )
    if any(len(values) != len(cell_id) for values in arrays):
        raise RuntimeError("L3 surface-material source arrays have inconsistent lengths")
    order, height, width = _spatial_layout(row, column)
    relief = _local_relief(
        elevation_m,
        order,
        height,
        width,
        config.local_relief_radius_cells,
    )
    resolution = handoff_manifest["resolution"]
    parent_resolution = int(resolution["parent_face_resolution"])
    child_resolution = int(terrain_manifest["hierarchy"]["child_face_resolution"])
    if child_resolution % parent_resolution:
        raise RuntimeError("L3 and parent face resolutions are not integer-related")
    actual_cell_size_m = float(terrain_manifest["hierarchy"]["actual_area_equivalent_cell_size_m"])
    local_terrain_slope = _local_terrain_slope(
        elevation_m,
        order,
        height,
        width,
        actual_cell_size_m,
        config.terrain_slope_smoothing_radius_cells,
    )
    return _MaterialSources(
        target_id=target_ids.pop(),
        terrain_manifest=terrain_manifest,
        hydrology_manifest=hydrology_manifest,
        channel_manifest=channel_manifest,
        handoff_manifest=handoff_manifest,
        handoff_dir=handoff_dir,
        terrain=terrain,
        hydrology=hydrology,
        channels=channels,
        parent_ids=parent_ids,
        parent_priors=parent_priors,
        cell_id=np.ascontiguousarray(cell_id),
        face=np.ascontiguousarray(face),
        row=np.ascontiguousarray(row),
        column=np.ascontiguousarray(column),
        l0_parent_id=np.ascontiguousarray(l0_parent_id),
        area_km2=np.ascontiguousarray(area_km2),
        elevation_m=np.ascontiguousarray(elevation_m),
        local_relief_m=np.ascontiguousarray(relief),
        local_terrain_slope=np.ascontiguousarray(local_terrain_slope),
        inside_display=np.ascontiguousarray(inside_display),
        inside_core=np.ascontiguousarray(inside_core),
        spatial_order=order,
        height=height,
        width=width,
        parent_face_resolution=parent_resolution,
        child_face_resolution=child_resolution,
        actual_cell_size_m=actual_cell_size_m,
    )


def _fingerprint(
    config: L3SurfaceMaterialsConfig,
    sources: _MaterialSources,
) -> tuple[str, dict[str, Any]]:
    native = native_library_info("surface_materials_native")
    components = {
        "format_version": SURFACE_MATERIALS_FORMAT_VERSION,
        "model_version": SURFACE_MATERIALS_MODEL_VERSION,
        "terrain_manifest_sha256": _file_checksum(config.terrain_dir / "manifest.json"),
        "hydrology_manifest_sha256": _file_checksum(config.hydrology_dir / "manifest.json"),
        "channel_manifest_sha256": _file_checksum(config.channel_geometry_dir / "manifest.json"),
        "handoff_manifest_sha256": _file_checksum(sources.handoff_dir / "manifest.json"),
        "terrain_zarr_sha256": sources.terrain_manifest["outputs"]["terrain_zarr"]["sha256_tree"],
        "hydrology_zarr_sha256": sources.hydrology_manifest["outputs"]["hydrology_zarr"][
            "sha256_tree"
        ],
        "channel_zarr_sha256": sources.channel_manifest["outputs"]["channel_geometry_zarr"][
            "sha256_tree"
        ],
        "handoff_zarr_sha256": sources.handoff_manifest["outputs"]["zarr"]["sha256"],
        "controls": asdict(config)
        | {
            "terrain_dir": str(config.terrain_dir),
            "hydrology_dir": str(config.hydrology_dir),
            "channel_geometry_dir": str(config.channel_geometry_dir),
            "output_dir": str(config.output_dir),
            "source_config": str(config.source_config) if config.source_config else None,
        },
        "native_abi_version": native["abi_version"],
        "native_sha256": native["sha256"],
        "binding_sha256": _file_checksum(Path(__file__).with_name("_surface_materials_native.py")),
        "orchestrator_sha256": _file_checksum(Path(__file__)),
    }
    run_fingerprint = _canonical_hash(components)
    provenance = {
        **components,
        "generation_config_sha256": (
            _file_checksum(config.source_config) if config.source_config else None
        ),
    }
    return run_fingerprint, provenance


def _parent_rows(parent_ids: np.ndarray, requested_ids: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(parent_ids, requested_ids)
    valid = positions < len(parent_ids)
    if np.any(valid):
        valid[valid] &= parent_ids[positions[valid]] == requested_ids[valid]
    if not np.all(valid):
        raise RuntimeError("L3 cell references a parent absent from the regional handoff")
    return np.ascontiguousarray(positions, dtype=np.int32)


def _interpolation_stencil(
    sources: _MaterialSources,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor = sources.child_face_resolution // sources.parent_face_resolution
    row_position = (sources.row[start:end].astype(np.float64) + 0.5) / factor - 0.5
    column_position = (sources.column[start:end].astype(np.float64) + 0.5) / factor - 0.5
    row0 = np.floor(row_position).astype(np.int32)
    column0 = np.floor(column_position).astype(np.int32)
    row_fraction = row_position - row0
    column_fraction = column_position - column0
    resolution = sources.parent_face_resolution
    face = sources.face[start:end].astype(np.int64)
    row_candidates = np.stack((row0, row0, row0 + 1, row0 + 1), axis=1)
    column_candidates = np.stack(
        (column0, column0 + 1, column0, column0 + 1),
        axis=1,
    )
    candidates = (
        face[:, None] * resolution * resolution + row_candidates * resolution + column_candidates
    )
    weights = np.stack(
        (
            (1.0 - row_fraction) * (1.0 - column_fraction),
            (1.0 - row_fraction) * column_fraction,
            row_fraction * (1.0 - column_fraction),
            row_fraction * column_fraction,
        ),
        axis=1,
    )
    positions = np.searchsorted(sources.parent_ids, candidates)
    clipped = np.minimum(positions, len(sources.parent_ids) - 1)
    valid = (
        (row_candidates >= 0)
        & (row_candidates < resolution)
        & (column_candidates >= 0)
        & (column_candidates < resolution)
        & (positions < len(sources.parent_ids))
        & (sources.parent_ids[clipped] == candidates)
    )
    exact_rows = _parent_rows(sources.parent_ids, sources.l0_parent_id[start:end])
    positions = np.where(valid, positions, exact_rows[:, None])
    weights = np.where(valid, weights, 0.0)
    weight_sum = np.sum(weights, axis=1)
    missing = weight_sum <= 0.0
    if np.any(missing):
        positions[missing, 0] = exact_rows[missing]
        weights[missing, 0] = 1.0
        weight_sum[missing] = 1.0
    weights /= weight_sum[:, None]
    return (
        np.ascontiguousarray(positions, dtype=np.int32),
        np.ascontiguousarray(weights, dtype=np.float32),
        exact_rows,
    )


def _interpolate_prior(
    values: np.ndarray,
    rows: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    gathered = values[rows]
    result = np.einsum("ni,ni...->n...", weights, gathered, optimize=True)
    return np.ascontiguousarray(result)


def _prior(
    sources: _MaterialSources,
    name: str,
    rows: np.ndarray,
    weights: np.ndarray,
    *,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> np.ndarray:
    return np.ascontiguousarray(
        _interpolate_prior(sources.parent_priors[f"parent_priors/{name}"], rows, weights),
        dtype=dtype,
    )


def _alluvial_legacy_fraction(
    alluvium_prior: np.ndarray,
    distance_to_channel_m: np.ndarray,
    valley_fraction: np.ndarray,
    terrain_slope: np.ndarray,
    config: L3SurfaceMaterialsConfig,
) -> np.ndarray:
    proximity = np.maximum(
        np.exp(-np.maximum(distance_to_channel_m, 0.0) / config.alluvial_legacy_decay_m),
        np.clip(valley_fraction, 0.0, 1.0),
    )
    low_slope = np.exp(-np.maximum(terrain_slope, 0.0) / config.alluvial_legacy_slope_scale)
    legacy = (
        np.clip(alluvium_prior, 0.0, 1.0)
        * (config.alluvial_legacy_background + config.alluvial_legacy_proximity_boost * proximity)
        * (0.35 + 0.65 * low_slope)
    )
    return np.ascontiguousarray(
        np.clip(legacy, 0.0, config.maximum_alluvial_fraction),
        dtype=np.float32,
    )


def _allocate_outputs(cell_count: int) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for native_name, artifact_name in NATIVE_OUTPUT_NAMES.items():
        shape = (MONTHS, cell_count) if artifact_name in MONTHLY_OUTPUTS else (cell_count,)
        dtype = np.uint8 if artifact_name == "DominantSurfaceMaterialCode" else np.float32
        output[native_name] = np.zeros(shape, dtype=dtype)
    return output


def _chunk_inputs(
    sources: _MaterialSources,
    config: L3SurfaceMaterialsConfig,
    start: int,
    end: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    rows, weights, exact_rows = _interpolation_stencil(sources, start, end)
    hydrology = sources.hydrology
    channels = sources.channels
    monthly_precipitation = np.ascontiguousarray(
        hydrology["forcing/monthly_precipitation_mm"][:, start:end],
        dtype=np.float32,
    )
    monthly_evaporation = _prior(
        sources,
        "climate/MonthlyEvaporationMm",
        rows,
        weights,
    ).T
    parent_precipitation = _prior(
        sources,
        "climate/MonthlyPrecipitationMm",
        rows,
        weights,
    )
    parent_snowfall = _prior(
        sources,
        "climate/MonthlySnowfallMm",
        rows,
        weights,
    )
    snowfall_fraction = np.divide(
        parent_snowfall,
        np.maximum(parent_precipitation, 1e-6),
        out=np.zeros_like(parent_snowfall),
        where=parent_precipitation > 0.0,
    )
    monthly_snowfall = np.ascontiguousarray(
        monthly_precipitation * np.clip(snowfall_fraction.T, 0.0, 1.0),
        dtype=np.float32,
    )
    base_monthly_temperature = _prior(
        sources,
        "climate/MonthlySurfaceTemperatureC",
        rows,
        weights,
    )
    climate_orography = _prior(
        sources,
        "climate/ClimateOrographyM",
        rows,
        weights,
    )
    temperature_adjustment = np.clip(
        -config.temperature_lapse_rate_c_per_m
        * (sources.elevation_m[start:end] - climate_orography),
        -config.maximum_temperature_adjustment_c,
        config.maximum_temperature_adjustment_c,
    ).astype(np.float32)
    monthly_temperature = np.ascontiguousarray(
        (base_monthly_temperature + temperature_adjustment[:, None]).T,
        dtype=np.float32,
    )
    lake = np.ascontiguousarray(hydrology["surface/lake_fraction"][start:end], dtype=np.float32)
    wetland = np.ascontiguousarray(
        hydrology["surface/wetland_fraction"][start:end], dtype=np.float32
    )
    floodplain = np.ascontiguousarray(
        channels["support/floodplain_fraction"][start:end], dtype=np.float32
    )
    alluvial_legacy = _alluvial_legacy_fraction(
        _prior(
            sources,
            "surface_materials/AlluviumFraction",
            rows,
            weights,
        ),
        np.asarray(
            channels["support/distance_to_channel_m"][start:end],
            dtype=np.float32,
        ),
        np.asarray(
            channels["support/valley_fraction"][start:end],
            dtype=np.float32,
        ),
        np.asarray(sources.local_terrain_slope[start:end], dtype=np.float32),
        config,
    )
    hydroperiod = np.ascontiguousarray(
        np.clip(lake + 0.65 * wetland + 0.15 * floodplain, 0.0, 1.0),
        dtype=np.float32,
    )
    province_class_prior = sources.parent_priors["parent_priors/geology/GeologicalProvinceClass"]
    province_class = np.ascontiguousarray(province_class_prior[exact_rows], dtype=np.uint8)
    inputs = {
        "areas": np.ascontiguousarray(sources.area_km2[start:end], dtype=np.float64),
        "ocean": np.ascontiguousarray(
            hydrology["surface/physical_ocean_fraction"][start:end],
            dtype=np.float32,
        ),
        "province_class": province_class,
        "crust_age": _prior(sources, "geology/CrustAgeGa", rows, weights),
        "rock_strength": _prior(sources, "geology/RockStrength", rows, weights),
        "accommodation": _prior(sources, "geology/SedimentAccommodation", rows, weights),
        "province_confidence": _prior(sources, "geology/ProvinceConfidence", rows, weights),
        "elevation_confidence": _prior(sources, "elevation/ElevationConfidence", rows, weights),
        "relief": np.ascontiguousarray(sources.local_relief_m[start:end], dtype=np.float32),
        "terrain_slope": np.ascontiguousarray(
            sources.local_terrain_slope[start:end], dtype=np.float32
        ),
        "river_corridor": np.ascontiguousarray(
            channels["support/channel_fraction"][start:end], dtype=np.float32
        ),
        "floodplain": floodplain,
        "lake_fraction": lake,
        "wetland_fraction": wetland,
        "depression_fill_depth": np.ascontiguousarray(
            hydrology["routing/depression_fill_depth_m"][start:end],
            dtype=np.float32,
        ),
        "refined_mask": np.ones(end - start, dtype=np.float32),
        "refined_lake_fraction": lake.copy(),
        "refined_wetland_fraction": wetland.copy(),
        "refined_hydroperiod": hydroperiod,
        "refined_salinity": np.zeros(end - start, dtype=np.float32),
        "recent_erosion_depth": np.zeros(end - start, dtype=np.float32),
        # The native model treats two metres as a saturated geologically recent
        # deposition signal. Here it carries a localized inherited alluvial
        # history until a dedicated L3 sediment-transport pass exists.
        "recent_deposition_depth": np.ascontiguousarray(
            2.0 * alluvial_legacy,
            dtype=np.float32,
        ),
        "glacier_fraction": _prior(sources, "cryosphere/GlacierIceFraction", rows, weights),
        "annual_temperature": np.ascontiguousarray(
            np.mean(monthly_temperature, axis=0), dtype=np.float32
        ),
        "annual_precipitation": np.ascontiguousarray(
            np.sum(monthly_precipitation, axis=0), dtype=np.float32
        ),
        "monthly_temperature": monthly_temperature,
        "monthly_precipitation": monthly_precipitation,
        "monthly_evaporation": np.ascontiguousarray(monthly_evaporation, dtype=np.float32),
        "monthly_snowfall": monthly_snowfall,
        "monthly_snowmelt": np.ascontiguousarray(
            hydrology["forcing/monthly_snowmelt_mm"][:, start:end],
            dtype=np.float32,
        ),
        "monthly_glacier_melt": np.ascontiguousarray(
            hydrology["forcing/monthly_glacier_melt_mm"][:, start:end],
            dtype=np.float32,
        ),
    }
    drivers = {
        "LocalReliefM": inputs["relief"],
        "LocalTerrainSlope": inputs["terrain_slope"],
        "TemperatureAdjustmentC": temperature_adjustment,
        "AlluvialLegacyFraction": alluvial_legacy,
    }
    return inputs, drivers, exact_rows


def _kernel_controls(config: L3SurfaceMaterialsConfig) -> dict[str, int | float]:
    return {
        "spinup_years": config.spinup_years,
        "maximum_regolith_depth_m": config.maximum_regolith_depth_m,
        "maximum_soil_depth_m": config.maximum_soil_depth_m,
        "maximum_alluvial_fraction": config.maximum_alluvial_fraction,
        "maximum_lacustrine_fraction": config.maximum_lacustrine_fraction,
        "maximum_glacial_fraction": config.maximum_glacial_fraction,
        "weathering_temperature_scale_c": config.weathering_temperature_scale_c,
        "weathering_precipitation_scale_mm": (config.weathering_precipitation_scale_mm),
        "soil_evaporation_factor": config.soil_evaporation_factor,
        "monthly_deep_drainage_fraction": config.monthly_deep_drainage_fraction,
    }


def _initialize_partial(
    partial: Path,
    config: L3SurfaceMaterialsConfig,
    sources: _MaterialSources,
    run_fingerprint: str,
) -> Any:
    partial.mkdir(parents=True)
    zarr_path = partial / "surface_materials.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    cell_count = len(sources.cell_id)
    chunk_count = math.ceil(cell_count / config.chunk_rows)
    chunks = (min(config.chunk_rows, cell_count),)
    monthly_chunks = (min(config.chunk_rows, cell_count), MONTHS)
    root.attrs.update(
        {
            "format_version": SURFACE_MATERIALS_FORMAT_VERSION,
            "model_version": SURFACE_MATERIALS_MODEL_VERSION,
            "status": "partial",
            "target_id": sources.target_id,
            "run_fingerprint": run_fingerprint,
            "cell_count": cell_count,
            "chunk_count": chunk_count,
            "chunk_rows": config.chunk_rows,
            "parent_major_storage": True,
            "monthly_layout": "cell, month",
            "actual_cell_size_m": sources.actual_cell_size_m,
            "source_prior_semantics": (
                "continuous parent geology and climate are bilinearly conditioned; "
                "L0 soil outputs are comparison priors, not copied categories"
            ),
        }
    )
    geometry = root.require_group("geometry")
    geometry_specs = (
        (
            "cell_id",
            np.uint64,
            sources.cell_id,
            "stable global L3 cell ID",
        ),
        (
            "l0_parent_cell_id",
            np.int32,
            sources.l0_parent_id,
            "stable inherited L0 prior parent",
        ),
        (
            "inside_display_window",
            bool,
            sources.inside_display,
            "visible complete-process region",
        ),
        (
            "inside_routed_catchment_core",
            bool,
            sources.inside_core,
            "fine routed basin used for acceptance",
        ),
    )
    for name, dtype, values, semantics in geometry_specs:
        dataset = _zarr_dataset(
            geometry,
            name,
            shape=(cell_count,),
            dtype=dtype,
            chunks=chunks,
            semantics=semantics,
        )
        dataset[:] = values
        _sync_zarr_array(zarr_path, f"geometry/{name}")
    materials = root.require_group("materials")
    soil = root.require_group("soil")
    monthly = root.require_group("monthly")
    drivers = root.require_group("drivers")
    for name, path in MATERIAL_PATHS.items():
        group = materials
        dtype = np.uint8 if name == "DominantSurfaceMaterialCode" else np.float32
        _zarr_dataset(
            group,
            path.split("/", 1)[1],
            shape=(cell_count,),
            dtype=dtype,
            chunks=chunks,
            semantics=(
                "dominant derived label"
                if dtype == np.uint8
                else "mutually exclusive underlying surface-material area fraction"
            ),
        )
    for name, path in SOIL_PATHS.items():
        _zarr_dataset(
            soil,
            path.split("/", 1)[1],
            shape=(cell_count,),
            dtype=np.float32,
            chunks=chunks,
            semantics="L3 initial-soil physical or chemical property",
        )
    for name, path in MONTHLY_PATHS.items():
        _zarr_dataset(
            monthly,
            path.split("/", 1)[1],
            shape=(cell_count, MONTHS),
            dtype=np.float32,
            chunks=monthly_chunks,
            semantics="cell-first monthly soil-water state or flux",
        )
    for name, path in DRIVER_PATHS.items():
        _zarr_dataset(
            drivers,
            path.split("/", 1)[1],
            shape=(cell_count,),
            dtype=np.float32,
            chunks=chunks,
            semantics="persisted L3 conditioning driver",
        )
    progress = root.require_group("progress")
    _zarr_dataset(
        progress,
        "chunk_complete",
        shape=(chunk_count,),
        dtype=bool,
        chunks=(min(chunk_count, 1_024),),
    )
    _write_json_durable(
        partial / "run_state.json",
        {
            "format_version": SURFACE_MATERIALS_FORMAT_VERSION,
            "model_version": SURFACE_MATERIALS_MODEL_VERSION,
            "run_fingerprint": run_fingerprint,
            "cell_count": cell_count,
            "chunk_count": chunk_count,
        },
    )
    (partial / "chunk_stats").mkdir()
    return root


def _open_partial(
    partial: Path,
    config: L3SurfaceMaterialsConfig,
    sources: _MaterialSources,
    run_fingerprint: str,
) -> tuple[Any, bool]:
    state_path = partial / "run_state.json"
    if partial.exists():
        try:
            state = _load_json(state_path)
        except (FileNotFoundError, json.JSONDecodeError):
            state = {}
        expected_chunks = math.ceil(len(sources.cell_id) / config.chunk_rows)
        valid = (
            state.get("run_fingerprint") == run_fingerprint
            and state.get("cell_count") == len(sources.cell_id)
            and state.get("chunk_count") == expected_chunks
        )
        if valid:
            return (
                zarr.open_group(str(partial / "surface_materials.zarr"), mode="r+"),
                True,
            )
        shutil.rmtree(partial)
    return _initialize_partial(partial, config, sources, run_fingerprint), False


def _write_chunk(
    root: Any,
    start: int,
    end: int,
    outputs: Mapping[str, np.ndarray],
    drivers: Mapping[str, np.ndarray],
) -> None:
    for native_name, artifact_name in NATIVE_OUTPUT_NAMES.items():
        path = OUTPUT_PATHS[artifact_name]
        values = outputs[native_name]
        if artifact_name in MONTHLY_OUTPUTS:
            root[path][start:end, :] = values.T
        else:
            root[path][start:end] = values
    for name, values in drivers.items():
        root[DRIVER_PATHS[name]][start:end] = values


def _output_chunk_checksums(
    zarr_path: Path,
    root: Any,
    output_paths: tuple[str, ...],
    chunk_index: int,
) -> dict[str, str]:
    return {
        path: _file_checksum(_zarr_chunk_path(zarr_path, root, path, chunk_index))
        for path in output_paths
    }


def _completed_chunk_is_valid(
    root: Any,
    partial: Path,
    chunk_index: int,
    start: int,
    end: int,
    output_paths: tuple[str, ...],
) -> bool:
    try:
        stats = _load_json(partial / "chunk_stats" / f"{chunk_index:06d}.json")
        checksums = stats.get("output_chunk_sha256")
        if (
            stats.get("chunk_index") != chunk_index
            or stats.get("start") != start
            or stats.get("end") != end
            or not isinstance(checksums, Mapping)
            or set(checksums) != set(output_paths)
        ):
            return False
        zarr_path = partial / "surface_materials.zarr"
        for path in output_paths:
            chunk_path = _zarr_chunk_path(zarr_path, root, path, chunk_index)
            if not chunk_path.is_file() or _file_checksum(chunk_path) != checksums[path]:
                return False
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
        return False
    return True


def _generate_chunks(
    root: Any,
    partial: Path,
    config: L3SurfaceMaterialsConfig,
    sources: _MaterialSources,
) -> tuple[int, int]:
    zarr_path = partial / "surface_materials.zarr"
    completion = np.asarray(root["progress/chunk_complete"][:], dtype=bool)
    output_paths = tuple(OUTPUT_PATHS.values())
    invalid_completed: list[int] = []
    for chunk_index, start in enumerate(range(0, len(sources.cell_id), config.chunk_rows)):
        end = min(start + config.chunk_rows, len(sources.cell_id))
        if completion[chunk_index] and not _completed_chunk_is_valid(
            root,
            partial,
            chunk_index,
            start,
            end,
            output_paths,
        ):
            completion[chunk_index] = False
            invalid_completed.append(chunk_index)
    if invalid_completed:
        root["progress/chunk_complete"][:] = completion
        _sync_zarr_array(zarr_path, "progress/chunk_complete")
    resumed_count = int(np.count_nonzero(completion))
    for chunk_index, start in enumerate(range(0, len(sources.cell_id), config.chunk_rows)):
        if completion[chunk_index]:
            continue
        end = min(start + config.chunk_rows, len(sources.cell_id))
        inputs, drivers, _ = _chunk_inputs(sources, config, start, end)
        outputs = _allocate_outputs(end - start)
        stats = run_surface_materials(
            **_kernel_controls(config),
            **inputs,
            **outputs,
        )
        _write_chunk(root, start, end, outputs, drivers)
        _sync_zarr_chunk(zarr_path, root, output_paths, chunk_index)
        output_chunk_sha256 = _output_chunk_checksums(
            zarr_path,
            root,
            output_paths,
            chunk_index,
        )
        _write_json_durable(
            partial / "chunk_stats" / f"{chunk_index:06d}.json",
            {
                "chunk_index": chunk_index,
                "start": start,
                "end": end,
                "output_chunk_sha256": output_chunk_sha256,
                **stats,
            },
        )
        root["progress/chunk_complete"][chunk_index] = True
        _sync_zarr_array(zarr_path, "progress/chunk_complete")
    return resumed_count, len(completion)


def _represented_parent_metrics(
    root: Any,
    sources: _MaterialSources,
    config: L3SurfaceMaterialsConfig,
) -> dict[str, float]:
    parent_rows = _parent_rows(sources.parent_ids, sources.l0_parent_id)
    parent_count = len(sources.parent_ids)
    represented_area = np.zeros(parent_count, dtype=np.float64)
    material_area = np.zeros((parent_count, len(MATERIAL_OUTPUTS)), dtype=np.float64)
    soil_depth_area = np.zeros(parent_count, dtype=np.float64)
    soil_ph_area = np.zeros(parent_count, dtype=np.float64)
    for start in range(0, len(sources.cell_id), config.chunk_rows):
        end = min(start + config.chunk_rows, len(sources.cell_id))
        ocean = np.asarray(
            sources.hydrology["surface/physical_ocean_fraction"][start:end],
            dtype=np.float32,
        )
        land = ocean < 0.5
        rows = parent_rows[start:end][land]
        area = sources.area_km2[start:end][land]
        np.add.at(represented_area, rows, area)
        for index, name in enumerate(MATERIAL_OUTPUTS):
            values = np.asarray(root[MATERIAL_PATHS[name]][start:end], dtype=np.float64)
            np.add.at(material_area[:, index], rows, values[land] * area)
        soil_depth = np.asarray(root[SOIL_PATHS["SoilDepthM"]][start:end], dtype=np.float64)
        soil_ph = np.asarray(root[SOIL_PATHS["SoilPH"]][start:end], dtype=np.float64)
        np.add.at(soil_depth_area, rows, soil_depth[land] * area)
        np.add.at(soil_ph_area, rows, soil_ph[land] * area)
    represented = represented_area > 0.0
    material_mean = np.divide(
        material_area,
        represented_area[:, None],
        out=np.zeros_like(material_area),
        where=represented_area[:, None] > 0.0,
    )
    material_prior = np.stack(
        [
            sources.parent_priors[f"parent_priors/surface_materials/{name}"]
            for name in MATERIAL_OUTPUTS
        ],
        axis=1,
    )
    material_l1 = np.sum(
        np.abs(material_mean[represented] - material_prior[represented]),
        axis=1,
    )
    soil_depth_mean = np.divide(
        soil_depth_area,
        represented_area,
        out=np.zeros_like(soil_depth_area),
        where=represented_area > 0.0,
    )
    soil_ph_mean = np.divide(
        soil_ph_area,
        represented_area,
        out=np.zeros_like(soil_ph_area),
        where=represented_area > 0.0,
    )
    soil_depth_prior = sources.parent_priors["parent_priors/surface_materials/SoilDepthM"]
    soil_ph_prior = sources.parent_priors["parent_priors/surface_materials/SoilPH"]
    return {
        "represented_l0_parent_count": int(np.count_nonzero(represented)),
        "parent_material_fraction_l1_difference_p50": float(np.percentile(material_l1, 50.0)),
        "parent_material_fraction_l1_difference_p95": float(np.percentile(material_l1, 95.0)),
        "parent_soil_depth_absolute_difference_m_p95": float(
            np.percentile(
                np.abs(soil_depth_mean[represented] - soil_depth_prior[represented]),
                95.0,
            )
        ),
        "parent_soil_ph_absolute_difference_p95": float(
            np.percentile(
                np.abs(soil_ph_mean[represented] - soil_ph_prior[represented]),
                95.0,
            )
        ),
    }


def _validate(
    root: Any,
    config: L3SurfaceMaterialsConfig,
    sources: _MaterialSources,
) -> dict[str, Any]:
    maximum_material_error = 0.0
    maximum_texture_error = 0.0
    maximum_water_error = 0.0
    maximum_soil_depth_excess = 0.0
    maximum_soil_support_excess = 0.0
    bounded = True
    finite = True
    monthly_nonnegative = True
    static_nonnegative = True
    physical_bounds_valid = True
    ocean_outputs_zero = True
    dominant_material_valid = True
    local_terrain_drivers_valid = True
    maximum_local_terrain_slope = 0.0
    land_area = 0.0
    sums = {
        "soil_bearing": 0.0,
        "hydric": 0.0,
        "regolith_depth": 0.0,
        "soil_depth": 0.0,
        "organic_carbon": 0.0,
        "soil_ph": 0.0,
        "salinity": 0.0,
        "fertility": 0.0,
    }
    material_area = np.zeros(len(MATERIAL_OUTPUTS), dtype=np.float64)
    for start in range(0, len(sources.cell_id), config.chunk_rows):
        end = min(start + config.chunk_rows, len(sources.cell_id))
        area = sources.area_km2[start:end]
        ocean = np.asarray(
            sources.hydrology["surface/physical_ocean_fraction"][start:end],
            dtype=np.float64,
        )
        lake = np.asarray(
            sources.hydrology["surface/lake_fraction"][start:end],
            dtype=np.float64,
        )
        land = ocean < 0.5
        area_land = area[land]
        land_area += float(np.sum(area_land))
        materials = np.stack(
            [
                np.asarray(root[MATERIAL_PATHS[name]][start:end], dtype=np.float64)
                for name in MATERIAL_OUTPUTS
            ],
            axis=1,
        )
        textures = np.stack(
            [
                np.asarray(root[SOIL_PATHS[name]][start:end], dtype=np.float64)
                for name in ("SandFraction", "SiltFraction", "ClayFraction")
            ],
            axis=1,
        )
        maximum_material_error = max(
            maximum_material_error,
            float(np.max(np.abs(np.sum(materials[land], axis=1) - 1.0), initial=0.0)),
        )
        maximum_texture_error = max(
            maximum_texture_error,
            float(np.max(np.abs(np.sum(textures[land], axis=1) - 1.0), initial=0.0)),
        )
        dominant = np.asarray(
            root[MATERIAL_PATHS["DominantSurfaceMaterialCode"]][start:end],
            dtype=np.uint8,
        )
        land_dominant = dominant[land]
        dominant_material_valid &= bool(
            np.all((land_dominant >= 1) & (land_dominant <= len(MATERIAL_OUTPUTS)))
            and np.all(dominant[~land] == 0)
        )
        if len(land_dominant):
            selected = materials[land, :][
                np.arange(len(land_dominant)),
                land_dominant.astype(np.int64) - 1,
            ]
            dominant_material_valid &= bool(
                np.all(selected >= np.max(materials[land, :], axis=1) - 1e-6)
            )
        soil_values = {
            name: np.asarray(root[SOIL_PATHS[name]][start:end], dtype=np.float64)
            for name in SOIL_OUTPUTS
        }
        finite &= bool(
            np.all(np.isfinite(materials))
            and all(np.all(np.isfinite(values)) for values in soil_values.values())
        )
        for values in (
            materials,
            *(soil_values[name][:, None] for name in BOUNDED_SOIL_OUTPUTS),
        ):
            bounded &= bool(np.all((values >= 0.0) & (values <= 1.0)))
        static_nonnegative &= all(
            np.all(soil_values[name] >= 0.0) for name in NONNEGATIVE_SOIL_OUTPUTS
        )
        regolith_depth = soil_values["RegolithDepthM"]
        soil_depth = soil_values["SoilDepthM"]
        soil_ph = soil_values["SoilPH"]
        physical_bounds_valid &= bool(
            np.all(regolith_depth <= config.maximum_regolith_depth_m + 1e-6)
            and np.all(soil_depth <= config.maximum_soil_depth_m + 1e-6)
            and np.all((soil_ph >= 0.0) & (soil_ph <= 14.0))
        )
        if np.any(~land):
            ocean_outputs_zero &= bool(
                np.all(np.abs(materials[~land]) <= 1e-7)
                and all(np.all(np.abs(values[~land]) <= 1e-7) for values in soil_values.values())
            )
        maximum_soil_depth_excess = max(
            maximum_soil_depth_excess,
            float(np.max(soil_depth - regolith_depth, initial=0.0)),
        )
        soil_bearing = soil_values["SoilBearingFraction"]
        glacier = _prior(
            sources,
            "cryosphere/GlacierIceFraction",
            *_interpolation_stencil(sources, start, end)[:2],
        )
        maximum_soil_support_excess = max(
            maximum_soil_support_excess,
            float(
                np.max(
                    soil_bearing - np.maximum(1.0 - lake - glacier, 0.0),
                    initial=0.0,
                )
            ),
        )
        monthly_values = {
            name: np.asarray(root[MONTHLY_PATHS[name]][start:end], dtype=np.float64)
            for name in MONTHLY_OUTPUTS
        }
        monthly_input = monthly_values["MonthlySoilLiquidInputMm"]
        finite &= all(np.all(np.isfinite(values)) for values in monthly_values.values())
        monthly_nonnegative &= all(np.all(values >= 0.0) for values in monthly_values.values())
        if np.any(~land):
            ocean_outputs_zero &= all(
                np.all(np.abs(values[~land]) <= 1e-7) for values in monthly_values.values()
            )
        saturation = monthly_values["MonthlySoilSaturationFraction"]
        bounded &= bool(np.all((saturation >= 0.0) & (saturation <= 1.0)))
        alluvial_legacy = np.asarray(
            root[DRIVER_PATHS["AlluvialLegacyFraction"]][start:end],
            dtype=np.float64,
        )
        local_relief = np.asarray(
            root[DRIVER_PATHS["LocalReliefM"]][start:end],
            dtype=np.float64,
        )
        local_slope = np.asarray(
            root[DRIVER_PATHS["LocalTerrainSlope"]][start:end],
            dtype=np.float64,
        )
        temperature_adjustment = np.asarray(
            root[DRIVER_PATHS["TemperatureAdjustmentC"]][start:end],
            dtype=np.float64,
        )
        finite &= bool(
            np.all(np.isfinite(alluvial_legacy))
            and np.all(np.isfinite(local_relief))
            and np.all(np.isfinite(local_slope))
            and np.all(np.isfinite(temperature_adjustment))
        )
        local_terrain_drivers_valid &= bool(
            np.all(local_relief >= 0.0)
            and np.all(local_slope >= 0.0)
            and np.all(
                np.abs(temperature_adjustment) <= config.maximum_temperature_adjustment_c + 1e-6
            )
        )
        maximum_local_terrain_slope = max(
            maximum_local_terrain_slope,
            float(np.max(local_slope, initial=0.0)),
        )
        bounded &= bool(
            np.all((alluvial_legacy >= 0.0) & (alluvial_legacy <= config.maximum_alluvial_fraction))
        )
        monthly_losses = sum(
            monthly_values[name]
            for name in (
                "MonthlyActualEvapotranspirationMm",
                "MonthlySoilRunoffMm",
                "MonthlyDeepDrainageMm",
            )
        )
        storage_change = np.asarray(
            root[SOIL_PATHS["AnnualSoilWaterStorageChangeMm"]][start:end],
            dtype=np.float64,
        )
        residual = np.sum(monthly_input - monthly_losses, axis=1) - storage_change
        water_reference = float(np.sum(np.sum(monthly_input[land], axis=1) * area_land))
        water_error = float(np.sum(np.abs(residual[land]) * area_land)) / max(
            water_reference, 1e-12
        )
        maximum_water_error = max(maximum_water_error, water_error)
        for index, name in enumerate(MATERIAL_OUTPUTS):
            material_area[index] += float(np.sum(materials[land, index] * area_land))
        value_map = {
            "soil_bearing": soil_bearing,
            "hydric": soil_values["HydricSoilFraction"],
            "regolith_depth": regolith_depth,
            "soil_depth": soil_depth,
            "organic_carbon": soil_values["PotentialSoilOrganicCarbonKgM2"],
            "soil_ph": soil_ph,
            "salinity": soil_values["SoilSalinityIndex"],
            "fertility": soil_values["SoilFertilityPotential"],
        }
        for name, values in value_map.items():
            sums[name] += float(np.sum(values[land] * area_land))
    prior_metrics = _represented_parent_metrics(root, sources, config)
    metrics = {
        "model": SURFACE_MATERIALS_MODEL_VERSION,
        "target_id": sources.target_id,
        "cell_count": len(sources.cell_id),
        "display_cell_count": int(np.count_nonzero(sources.inside_display)),
        "core_cell_count": int(np.count_nonzero(sources.inside_core)),
        "land_area_km2": land_area,
        "material_balance_max_error": maximum_material_error,
        "texture_balance_max_error": maximum_texture_error,
        "maximum_chunk_water_balance_relative_error": maximum_water_error,
        "maximum_soil_depth_excess_m": maximum_soil_depth_excess,
        "maximum_soil_support_excess": maximum_soil_support_excess,
        "maximum_local_terrain_slope": maximum_local_terrain_slope,
        "soil_bearing_land_area_fraction": sums["soil_bearing"] / land_area,
        "hydric_soil_land_area_fraction": sums["hydric"] / land_area,
        "land_mean_regolith_depth_m": sums["regolith_depth"] / land_area,
        "land_mean_soil_depth_m": sums["soil_depth"] / land_area,
        "land_mean_organic_carbon_kg_m2": sums["organic_carbon"] / land_area,
        "land_mean_soil_ph": sums["soil_ph"] / land_area,
        "land_mean_soil_salinity_index": sums["salinity"] / land_area,
        "land_mean_soil_fertility_potential": sums["fertility"] / land_area,
        "land_mean_material_fractions": {
            name: float(material_area[index] / land_area)
            for index, name in enumerate(MATERIAL_OUTPUTS)
        },
        **prior_metrics,
    }
    gates = {
        "outputs_finite": finite,
        "bounded_fraction_outputs": bounded,
        "nonnegative_static_outputs": static_nonnegative,
        "physical_soil_bounds_valid": physical_bounds_valid,
        "ocean_outputs_zero": ocean_outputs_zero,
        "material_balance_valid": (
            maximum_material_error <= config.maximum_component_balance_error
        ),
        "dominant_material_valid": dominant_material_valid,
        "local_terrain_drivers_valid": local_terrain_drivers_valid,
        "texture_balance_valid": (maximum_texture_error <= config.maximum_component_balance_error),
        "monthly_water_balance_valid": (
            maximum_water_error <= config.maximum_water_balance_relative_error
        ),
        "monthly_outputs_nonnegative": monthly_nonnegative,
        "soil_depth_within_regolith": maximum_soil_depth_excess <= 1e-6,
        "soil_support_within_non_open_land": maximum_soil_support_excess <= 1e-6,
        "parent_material_prior_divergence_bounded": (
            prior_metrics["parent_material_fraction_l1_difference_p95"] <= 1.0
        ),
        "parent_soil_depth_prior_divergence_bounded": (
            prior_metrics["parent_soil_depth_absolute_difference_m_p95"] <= 2.5
        ),
        "parent_soil_ph_prior_divergence_bounded": (
            prior_metrics["parent_soil_ph_absolute_difference_p95"] <= 2.0
        ),
    }
    return {**metrics, "gates": gates, "passed": bool(all(gates.values()))}


def _material_colors(materials: np.ndarray) -> np.ndarray:
    colors = np.clip(materials @ MATERIAL_COLORS, 0.0, 255.0)
    return np.asarray(colors, dtype=np.uint8)


def _render(
    root: Any,
    sources: _MaterialSources,
    validation: Mapping[str, Any],
    path: Path,
) -> None:
    order = sources.spatial_order
    elevation = sources.elevation_m[order].reshape(sources.height, sources.width)
    terrain = _hillshaded_terrain(elevation, sources.actual_cell_size_m)
    materials = np.stack(
        [
            np.asarray(root[MATERIAL_PATHS[name]][:], dtype=np.float32)[order]
            for name in MATERIAL_OUTPUTS
        ],
        axis=1,
    ).reshape(sources.height, sources.width, len(MATERIAL_OUTPUTS))
    material_rgb = _material_colors(materials)
    colors = terrain * 0.42 + material_rgb.astype(np.float32) * 0.58
    hydric = np.asarray(root[SOIL_PATHS["HydricSoilFraction"]][:], dtype=np.float32)[order].reshape(
        sources.height, sources.width
    )
    hydric_alpha = np.clip(hydric * 0.55, 0.0, 0.55)
    colors = (
        colors * (1.0 - hydric_alpha[..., None])
        + np.asarray((55.0, 133.0, 130.0)) * hydric_alpha[..., None]
    )
    lake = np.asarray(sources.hydrology["surface/lake_fraction"][:], dtype=np.float32)[
        order
    ].reshape(sources.height, sources.width)
    wetland = np.asarray(sources.hydrology["surface/wetland_fraction"][:], dtype=np.float32)[
        order
    ].reshape(sources.height, sources.width)
    wetland_alpha = np.clip(wetland * 0.55, 0.0, 0.55)
    colors = (
        colors * (1.0 - wetland_alpha[..., None])
        + np.asarray((59.0, 124.0, 86.0)) * wetland_alpha[..., None]
    )
    colors = colors * (1.0 - lake[..., None]) + np.asarray((45.0, 132.0, 168.0)) * lake[..., None]
    channel = np.asarray(sources.channels["support/centerline_seed"][:], dtype=bool)[order].reshape(
        sources.height, sources.width
    )
    colors[channel] = np.asarray((24.0, 92.0, 159.0))
    display = sources.inside_display[order].reshape(sources.height, sources.width)
    row_slice, column_slice = _rectangular_mask_slices(display)
    colors = colors[row_slice, column_slice]
    map_image = Image.fromarray(np.clip(colors, 0, 255).astype(np.uint8), mode="RGB")
    title_height = 52
    footer_height = 76
    legend_width = 380
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
    draw.text(
        (18, 13),
        "L3 surface materials and initial soils",
        fill=(25, 30, 27),
        font=title_font,
    )
    draw.line(
        (map_image.width, 0, map_image.width, canvas.height),
        fill=(178, 181, 174),
        width=1,
    )
    x = map_image.width + 24
    y = 20
    draw.text((x, y), "Surface materials", fill=(25, 30, 27), font=title_font)
    y += 55
    labels = (
        "exposed bedrock",
        "residual regolith",
        "colluvium",
        "alluvium",
        "lacustrine sediment",
        "glacial deposit",
        "volcaniclastic",
    )
    for color, label in zip(MATERIAL_COLORS.astype(np.uint8), labels, strict=True):
        draw.rectangle(
            (x, y, x + 34, y + 18),
            fill=tuple(map(int, color)),
            outline=(55, 59, 56),
        )
        draw.text((x + 48, y - 1), label, fill=(35, 39, 36), font=small_font)
        y += 32
    for color, label in (
        ((55, 133, 130), "hydric-soil influence"),
        ((59, 124, 86), "wetland"),
        ((45, 132, 168), "lake"),
        ((24, 92, 159), "physical channel vector"),
    ):
        draw.rectangle((x, y, x + 34, y + 18), fill=color, outline=(55, 59, 56))
        draw.text((x + 48, y - 1), label, fill=(35, 39, 36), font=small_font)
        y += 32
    y += 14
    draw.text((x, y), "Summary", fill=(25, 30, 27), font=label_font)
    y += 34
    for line in (
        f"{validation['soil_bearing_land_area_fraction'] * 100:.1f}% soil-bearing",
        f"{validation['hydric_soil_land_area_fraction'] * 100:.1f}% hydric support",
        f"mean soil depth {validation['land_mean_soil_depth_m']:.2f} m",
        f"mean pH {validation['land_mean_soil_ph']:.2f}",
        f"mean fertility {validation['land_mean_soil_fertility_potential']:.2f}",
        (
            "parent material delta p95 "
            f"{validation['parent_material_fraction_l1_difference_p95']:.2f}"
        ),
        "initial mineral soil; no taxonomy",
        "no vegetation feedback yet",
    ):
        draw.text((x, y), line, fill=(50, 54, 51), font=small_font)
        y += 25
    _draw_scale(
        draw,
        map_image.width,
        map_image.height,
        title_height,
        sources.actual_cell_size_m,
    )
    canvas.save(path, optimize=True)


def _existing_result(
    config: L3SurfaceMaterialsConfig,
    run_fingerprint: str,
) -> L3SurfaceMaterialsResult | None:
    manifest_path = config.output_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _load_json(manifest_path)
    if manifest.get("run_fingerprint") != run_fingerprint or not manifest.get("validation_passed"):
        return None
    _verify_manifest_outputs(config.output_dir, manifest)
    summary = manifest["summary"]
    return L3SurfaceMaterialsResult(
        output_dir=config.output_dir,
        manifest_path=manifest_path,
        validation_path=config.output_dir / "validation.json",
        zarr_path=config.output_dir / "surface_materials.zarr",
        preview_path=config.output_dir / "surface_materials.png",
        target_id=str(manifest["target_id"]),
        display_cell_count=int(summary["display_cell_count"]),
        chunk_count=int(summary["chunk_count"]),
        validation_passed=True,
    )


def generate_l3_surface_materials(
    config: L3SurfaceMaterialsConfig,
) -> L3SurfaceMaterialsResult:
    """Realize L3 surface materials and initial soils in resumable native chunks."""

    started = time.perf_counter()
    sources = _load_sources(config)
    run_fingerprint, fingerprint_components = _fingerprint(config, sources)
    existing = _existing_result(config, run_fingerprint)
    if existing is not None:
        return existing
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = config.output_dir.with_name(f".{config.output_dir.name}.partial")
    root, resumed = _open_partial(partial, config, sources, run_fingerprint)
    resumed_chunks, chunk_count = _generate_chunks(root, partial, config, sources)
    validation = _validate(root, config, sources)
    preview_path = partial / "surface_materials.png"
    _render(root, sources, validation, preview_path)
    zarr_path = partial / "surface_materials.zarr"
    root.attrs["status"] = "validating"
    zarr.consolidate_metadata(str(zarr_path))
    _fsync_paths([zarr_path / ".zattrs", zarr_path / ".zmetadata"])
    observed_peak = _observed_peak_rss_bytes()
    estimated_peak = int(config.chunk_rows * 900 + len(sources.cell_id) * 80)
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
    validation["gates"]["peak_memory_budget_valid"] = bool(validation["peak_memory_budget_valid"])
    validation["gates"]["storage_budget_valid"] = bool(validation["storage_budget_valid"])
    validation["passed"] = bool(all(validation["gates"].values()))
    root.attrs["status"] = "complete" if validation["passed"] else "validation_failed"
    zarr.consolidate_metadata(str(zarr_path))
    _fsync_paths([zarr_path / ".zattrs", zarr_path / ".zmetadata"])
    validation_path = partial / "validation.json"
    _write_json_durable(validation_path, validation)
    elapsed = time.perf_counter() - started
    manifest = {
        "format_version": SURFACE_MATERIALS_FORMAT_VERSION,
        "model_version": SURFACE_MATERIALS_MODEL_VERSION,
        "status": "complete" if validation["passed"] else "validation_failed",
        "target_id": sources.target_id,
        "run_fingerprint": run_fingerprint,
        "summary": {
            "cell_count": len(sources.cell_id),
            "display_cell_count": validation["display_cell_count"],
            "core_cell_count": validation["core_cell_count"],
            "chunk_count": chunk_count,
            "soil_bearing_land_area_fraction": validation["soil_bearing_land_area_fraction"],
            "hydric_soil_land_area_fraction": validation["hydric_soil_land_area_fraction"],
            "land_mean_soil_depth_m": validation["land_mean_soil_depth_m"],
            "land_mean_soil_ph": validation["land_mean_soil_ph"],
            "land_mean_soil_fertility_potential": validation["land_mean_soil_fertility_potential"],
        },
        "model": {
            "materials_and_soils": (
                "existing Rust property-first model replayed on L3 terrain, water, "
                "channel, and floodplain state"
            ),
            "parent_priors": (
                "continuous geology and climate priors are bilinearly interpolated; "
                "categorical province identity is inherited; L0 soil outputs remain "
                "comparison priors"
            ),
            "alluvial_history": (
                "inherited coarse alluvium is a soft depositional-history prior localized "
                "by L3 channel distance, valley support, and slope; it does not widen active "
                "river water"
            ),
            "temperature": (
                "monthly parent climate plus bounded L3 elevation lapse-rate adjustment"
            ),
            "soil_water": (f"{config.spinup_years}-year periodic monthly bucket partition in Rust"),
            "groundwater": "not modeled",
            "vegetation_feedback": "not applied",
            "soil_taxonomy": "not emitted",
        },
        "resume": {
            "resumed_partial": resumed,
            "resumed_chunk_count": resumed_chunks,
            "elapsed_seconds_this_run": elapsed,
        },
        "source": {
            "terrain_dir": str(config.terrain_dir),
            "hydrology_dir": str(config.hydrology_dir),
            "channel_geometry_dir": str(config.channel_geometry_dir),
            "handoff_dir": str(sources.handoff_dir),
            **fingerprint_components,
        },
        "outputs": {
            "surface_materials_zarr": {
                "path": "surface_materials.zarr",
                "sha256_tree": _tree_checksum(zarr_path),
            },
            "surface_materials_preview": {
                "path": "surface_materials.png",
                "sha256": _file_checksum(preview_path),
                "projection": "native cubed-sphere face raster diagnostic",
                "legend": "surface-material mixtures, hydric influence, and water",
                "scale": "labelled kilometre scale from L3 area-equivalent cell width",
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
            f"L3 surface materials failed validation; diagnostics retained in {partial}: "
            f"{validation}"
        )
    _replace_directory(partial, config.output_dir)
    return L3SurfaceMaterialsResult(
        output_dir=config.output_dir,
        manifest_path=config.output_dir / "manifest.json",
        validation_path=config.output_dir / "validation.json",
        zarr_path=config.output_dir / "surface_materials.zarr",
        preview_path=config.output_dir / "surface_materials.png",
        target_id=sources.target_id,
        display_cell_count=int(validation["display_cell_count"]),
        chunk_count=chunk_count,
        validation_passed=True,
    )


__all__ = [
    "L3SurfaceMaterialsConfig",
    "L3SurfaceMaterialsResult",
    "SURFACE_MATERIALS_FORMAT_VERSION",
    "SURFACE_MATERIALS_MODEL_VERSION",
    "generate_l3_surface_materials",
]
