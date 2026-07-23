"""Chunked L3 biosphere, functional vegetation, and derived biome replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, cast

import numpy as np
from PIL import Image, ImageDraw
import yaml  # type: ignore[import-untyped]
import zarr  # type: ignore[import-untyped]

from .._native import native_library_info
from ._biosphere_envelope_native import run_biosphere_envelope
from ._derived_biomes_native import BIOME_COUNT, run_derived_biomes
from ._functional_vegetation_native import (
    FUNCTIONAL_TYPE_COUNT,
    NONVEGETATED_TYPE_COUNT,
    RESOURCE_POTENTIAL_COUNT,
    run_functional_vegetation,
)
from ._potential_biosphere_native import run_potential_biosphere
from .l3_channel_geometry import _verify_manifest_outputs
from .l3_hydrology import (
    _draw_scale,
    _hillshaded_terrain,
    _observed_peak_rss_bytes,
    _rectangular_mask_slices,
)
from .l3_surface_materials import (
    _require_source_manifest_checksum,
    _verify_handoff_outputs,
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
from .stages.atmosphere import AtmosphereConfig
from .stages.biosphere_envelope import (
    ANNUAL_ENVELOPE_OUTPUTS,
    MONTHLY_ENVELOPE_OUTPUTS,
    BiosphereEnvelopeConfig,
)
from .stages.derived_biomes import BIOMES, LANDSCAPES, DerivedBiomeConfig
from .stages.functional_vegetation import (
    FUNCTIONAL_TYPES,
    NONVEGETATED_TYPES,
    RESOURCE_POTENTIALS,
    FunctionalVegetationConfig,
)
from .stages.planet import PlanetConfig
from .stages.potential_biosphere import (
    MONTHLY_POTENTIAL_BIOSPHERE_OUTPUTS,
    SCALAR_POTENTIAL_BIOSPHERE_OUTPUTS,
    PotentialBiosphereConfig,
)

ECOLOGY_FORMAT_VERSION = 1
ECOLOGY_MODEL_VERSION = "l3_ecology_v0"
MONTHS = 12

ENVELOPE_NATIVE_OUTPUTS = {
    "monthly_par_out": "MonthlySurfacePARMJm2",
    "monthly_liquid_opportunity_out": "MonthlyLiquidWaterOpportunity",
    "monthly_thermal_opportunity_out": "MonthlyThermalOpportunity",
    "monthly_primary_energy_out": "MonthlyTerrestrialPrimaryEnergyPotentialMJm2",
    "annual_par_out": "AnnualSurfacePARMJm2",
    "annual_primary_energy_out": "AnnualTerrestrialPrimaryEnergyPotentialMJm2",
    "carbon_substrate_relative_out": "CarbonSubstrateRelativeToReference",
    "aerobic_oxygen_relative_out": "AerobicOxygenRelativeToReference",
    "terrestrial_surface_support_out": "TerrestrialSurfaceSupportFraction",
    "nutrient_support_out": "NutrientSupportIndex",
    "environmental_stress_out": "EnvironmentalStressIndex",
    "confidence_out": "BiosphereEnvelopeConfidence",
}

POTENTIAL_NATIVE_OUTPUTS = {
    "monthly_npp_out": "MonthlyPotentialNPPKgCM2",
    "annual_npp_out": "AnnualPotentialNPPKgCM2",
    "vegetation_cover_out": "PotentialVegetationCoverFraction",
    "standing_biomass_out": "PotentialStandingBiomassKgCM2",
    "growing_season_out": "GrowingSeasonFraction",
    "productivity_seasonality_out": "ProductivitySeasonalityIndex",
    "drought_pressure_out": "DroughtAdaptationPressure",
    "cold_pressure_out": "ColdAdaptationPressure",
    "heat_pressure_out": "HeatAdaptationPressure",
    "waterlogging_pressure_out": "WaterloggingAdaptationPressure",
    "salinity_pressure_out": "SalinityAdaptationPressure",
    "woody_trait_out": "PotentialWoodyAllocationTrait",
    "resource_conservative_trait_out": "PotentialResourceConservativeTrait",
    "rooting_depth_out": "PotentialRootingDepthM",
    "canopy_height_out": "PotentialCanopyHeightM",
    "leaf_area_index_out": "PotentialLeafAreaIndex",
    "fuel_continuity_out": "PotentialFuelContinuityIndex",
    "confidence_out": "PotentialBiosphereConfidence",
}

FUNCTIONAL_NATIVE_OUTPUTS = {
    "functional_type_fractions_out": "FunctionalTypeFractions",
    "nonvegetated_fractions_out": "NonVegetatedFractions",
    "resource_potentials_out": "FunctionalResourcePotentials",
    "confidence_out": "FunctionalVegetationConfidence",
    "dominant_cover_code_out": "DominantFunctionalCoverCode",
}

BIOME_NATIVE_OUTPUTS = {
    "biome_fractions_out": "BiomeFractions",
    "classification_confidence_out": "BiomeClassificationConfidence",
    "dominance_margin_out": "BiomeDominanceMargin",
    "transition_index_out": "BiomeTransitionIndex",
    "primary_biome_code_out": "DominantBiomeCode",
    "secondary_biome_code_out": "SecondaryBiomeCode",
    "dominant_landscape_code_out": "DominantLandscapeCode",
}

DRIVER_PATHS = {
    "MonthlyInsolationWm2": "drivers/MonthlyInsolationWm2",
    "MonthlyTemperatureC": "drivers/MonthlyTemperatureC",
    "AnnualMeanTemperatureC": "drivers/AnnualMeanTemperatureC",
    "AnnualPrecipitationMm": "drivers/AnnualPrecipitationMm",
    "GlacierIceFraction": "drivers/GlacierIceFraction",
    "CO2PartialPressurePa": "drivers/CO2PartialPressurePa",
    "OxygenPartialPressureKPa": "drivers/OxygenPartialPressureKPa",
}
GEOMETRY_PATHS = {
    "cell_id": "geometry/cell_id",
    "l0_parent_cell_id": "geometry/l0_parent_cell_id",
    "inside_display_window": "geometry/inside_display_window",
    "inside_routed_catchment_core": "geometry/inside_routed_catchment_core",
}

ENVELOPE_PATHS = {
    name: (
        f"envelope/monthly/{name}"
        if name in MONTHLY_ENVELOPE_OUTPUTS
        else f"envelope/annual/{name}"
    )
    for name in (*MONTHLY_ENVELOPE_OUTPUTS, *ANNUAL_ENVELOPE_OUTPUTS)
}
POTENTIAL_PATHS = {
    name: (
        f"potential/monthly/{name}"
        if name in MONTHLY_POTENTIAL_BIOSPHERE_OUTPUTS
        else f"potential/traits/{name}"
    )
    for name in (*MONTHLY_POTENTIAL_BIOSPHERE_OUTPUTS, *SCALAR_POTENTIAL_BIOSPHERE_OUTPUTS)
}
FUNCTIONAL_PATHS = {
    "FunctionalTypeFractions": "functional/FunctionalTypeFractions",
    "NonVegetatedFractions": "functional/NonVegetatedFractions",
    "FunctionalResourcePotentials": "functional/FunctionalResourcePotentials",
    "FunctionalVegetationConfidence": "functional/FunctionalVegetationConfidence",
    "DominantFunctionalCoverCode": "functional/DominantFunctionalCoverCode",
}
BIOME_PATHS = {
    "BiomeFractions": "biomes/BiomeFractions",
    "BiomeClassificationConfidence": "biomes/BiomeClassificationConfidence",
    "BiomeDominanceMargin": "biomes/BiomeDominanceMargin",
    "BiomeTransitionIndex": "biomes/BiomeTransitionIndex",
    "DominantBiomeCode": "biomes/DominantBiomeCode",
    "SecondaryBiomeCode": "biomes/SecondaryBiomeCode",
    "DominantLandscapeCode": "biomes/DominantLandscapeCode",
}
OUTPUT_PATHS = {
    **DRIVER_PATHS,
    **ENVELOPE_PATHS,
    **POTENTIAL_PATHS,
    **FUNCTIONAL_PATHS,
    **BIOME_PATHS,
}
CHECKSUM_CHUNK_PATHS = (*GEOMETRY_PATHS.values(), *OUTPUT_PATHS.values())

REQUIRED_PARENT_PRIORS = (
    "parent_priors/planet/MonthlyInsolationWm2",
    "parent_priors/climate/MonthlySurfaceTemperatureC",
    "parent_priors/climate/MonthlyPrecipitationMm",
    "parent_priors/cryosphere/GlacierIceFraction",
    "parent_priors/potential_biosphere/AnnualPotentialNPPKgCM2",
    "parent_priors/functional_vegetation/FunctionalTypeFractions",
    "parent_priors/derived_biomes/BiomeFractions",
)


@dataclass(frozen=True)
class L3EcologyConfig:
    terrain_dir: Path
    hydrology_dir: Path
    channel_geometry_dir: Path
    surface_materials_dir: Path
    output_dir: Path
    chunk_rows: int = 65_536
    minimum_parent_coverage_fraction: float = 0.75
    maximum_parent_npp_relative_difference_p95: float = 2.0
    maximum_parent_functional_l1_difference_p95: float = 1.50
    maximum_parent_biome_l1_difference_p95: float = 1.60
    maximum_parent_boundary_p95_ratio: float = 3.25
    maximum_parent_boundary_absolute_difference_p95: float = 0.075
    maximum_repeated_parent_motif_correlation_p95: float = 0.85
    minimum_wet_ecology_response: float = 0.01
    minimum_cold_highland_response: float = 0.01
    minimum_valley_productivity_response: float = 0.0
    minimum_valley_resource_response: float = 0.0
    maximum_peak_memory_gb: float = 24.0
    maximum_storage_gb: float = 6.0
    source_config: Path | None = None

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        output_dir: Path | None = None,
    ) -> "L3EcologyConfig":
        source = Path(path).expanduser().resolve()
        data = yaml.safe_load(source.read_text(encoding="utf8"))
        if not isinstance(data, Mapping):
            raise TypeError("L3 ecology config must contain a mapping")
        path_keys = {
            "terrain_dir": "terrain_output_dir",
            "hydrology_dir": "hydrology_output_dir",
            "channel_geometry_dir": "channel_geometry_output_dir",
            "surface_materials_dir": "surface_materials_output_dir",
            "output_dir": "ecology_output_dir",
        }
        missing = [raw for raw in path_keys.values() if not data.get(raw)]
        if missing:
            raise ValueError(f"L3 ecology config lacks paths: {', '.join(sorted(missing))}")
        controls = data.get("l3_ecology", {})
        limits = data.get("limits", {})
        if not isinstance(controls, Mapping) or not isinstance(limits, Mapping):
            raise TypeError("L3 l3_ecology and limits controls must be mappings")
        known = {
            "chunk_rows",
            "minimum_parent_coverage_fraction",
            "maximum_parent_npp_relative_difference_p95",
            "maximum_parent_functional_l1_difference_p95",
            "maximum_parent_biome_l1_difference_p95",
            "maximum_parent_boundary_p95_ratio",
            "maximum_parent_boundary_absolute_difference_p95",
            "maximum_repeated_parent_motif_correlation_p95",
            "minimum_wet_ecology_response",
            "minimum_cold_highland_response",
            "minimum_valley_productivity_response",
            "minimum_valley_resource_response",
        }
        unknown = set(controls) - known
        if unknown:
            raise ValueError(f"Unknown L3 ecology controls: {', '.join(sorted(unknown))}")
        values: dict[str, int | float] = {}
        for name in known:
            if name in controls:
                values[name] = (
                    int(controls[name]) if name == "chunk_rows" else float(controls[name])
                )
        resolved = {
            name: (source.parent / str(data[raw])).resolve()
            for name, raw in path_keys.items()
            if name != "output_dir"
        }
        config = cls(
            **resolved,
            output_dir=(
                output_dir.expanduser().resolve()
                if output_dir is not None
                else (source.parent / str(data[path_keys["output_dir"]])).resolve()
            ),
            maximum_peak_memory_gb=float(limits.get("maximum_peak_memory_gb", 24.0)),
            maximum_storage_gb=float(limits.get("maximum_ecology_storage_gb", 6.0)),
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
                self.surface_materials_dir,
            ),
        )
        if not 16_384 <= self.chunk_rows <= 262_144:
            raise ValueError("l3_ecology.chunk_rows must be in [16384, 262144]")
        unit_interval = (
            "minimum_parent_coverage_fraction",
            "maximum_repeated_parent_motif_correlation_p95",
        )
        for name in unit_interval:
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"l3_ecology.{name} must be in [0, 1]")
        positive = (
            "maximum_parent_npp_relative_difference_p95",
            "maximum_parent_functional_l1_difference_p95",
            "maximum_parent_biome_l1_difference_p95",
            "maximum_parent_boundary_p95_ratio",
            "maximum_parent_boundary_absolute_difference_p95",
        )
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"l3_ecology.{name} must be finite and positive")
        for name in (
            "minimum_wet_ecology_response",
            "minimum_cold_highland_response",
            "minimum_valley_productivity_response",
            "minimum_valley_resource_response",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"l3_ecology.{name} must be finite and in [-1, 1]")
        if not 1.0 <= self.maximum_peak_memory_gb <= 28.0:
            raise ValueError("limits.maximum_peak_memory_gb must be in [1, 28]")
        if not 1.0 <= self.maximum_storage_gb <= 16.0:
            raise ValueError("limits.maximum_ecology_storage_gb must be in [1, 16]")


@dataclass(frozen=True)
class L3EcologyResult:
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
class _EcologySources:
    target_id: str
    terrain_manifest: dict[str, Any]
    hydrology_manifest: dict[str, Any]
    channel_manifest: dict[str, Any]
    surface_manifest: dict[str, Any]
    handoff_manifest: dict[str, Any]
    handoff_dir: Path
    world_config_path: Path
    terrain: Any
    hydrology: Any
    channels: Any
    surface: Any
    parent_ids: np.ndarray
    parent_area_km2: np.ndarray
    parent_priors: dict[str, np.ndarray]
    cell_id: np.ndarray
    face: np.ndarray
    row: np.ndarray
    column: np.ndarray
    l0_parent_id: np.ndarray
    area_km2: np.ndarray
    elevation_m: np.ndarray
    local_relief_m: np.ndarray
    inside_display: np.ndarray
    inside_core: np.ndarray
    spatial_order: np.ndarray
    height: int
    width: int
    parent_face_resolution: int
    child_face_resolution: int
    actual_cell_size_m: float
    planet_config: PlanetConfig
    atmosphere_config: AtmosphereConfig
    envelope_config: BiosphereEnvelopeConfig
    potential_config: PotentialBiosphereConfig
    functional_config: FunctionalVegetationConfig
    biome_config: DerivedBiomeConfig


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
                raise ValueError(f"L3 ecology output {candidate} overlaps source {source}")


def _spatial_layout(row: np.ndarray, column: np.ndarray) -> tuple[np.ndarray, int, int]:
    order = np.lexsort((column, row)).astype(np.int32, copy=False)
    height = int(np.max(row) - np.min(row) + 1)
    width = int(np.max(column) - np.min(column) + 1)
    if height * width != len(order):
        raise RuntimeError("L3 ecology requires a dense rectangular terrain window")
    expected_row, expected_column = np.indices((height, width), dtype=np.int32)
    if not np.array_equal(
        row[order] - int(np.min(row)), expected_row.reshape(-1)
    ) or not np.array_equal(column[order] - int(np.min(column)), expected_column.reshape(-1)):
        raise RuntimeError("L3 ecology coordinates are not row-major dense")
    return np.ascontiguousarray(order), height, width


def _world_controls(
    world_config_path: Path,
) -> tuple[
    PlanetConfig,
    AtmosphereConfig,
    BiosphereEnvelopeConfig,
    PotentialBiosphereConfig,
    FunctionalVegetationConfig,
    DerivedBiomeConfig,
]:
    data = yaml.safe_load(world_config_path.read_text(encoding="utf8"))
    if not isinstance(data, Mapping):
        raise TypeError("Source world config must contain a mapping")
    overrides = data.get("stage_overrides", {})
    if not isinstance(overrides, Mapping):
        raise TypeError("Source world stage_overrides must be a mapping")

    def controls(name: str) -> Mapping[str, object]:
        raw = overrides.get(name, {})
        if not isinstance(raw, Mapping):
            raise TypeError(f"Source world {name} controls must be a mapping")
        return cast(Mapping[str, object], raw)

    return (
        PlanetConfig.from_mapping(controls("planet")),
        AtmosphereConfig.from_mapping(controls("atmosphere")),
        BiosphereEnvelopeConfig.from_mapping(controls("biosphere_envelope")),
        PotentialBiosphereConfig.from_mapping(controls("potential_biosphere")),
        FunctionalVegetationConfig.from_mapping(controls("functional_vegetation")),
        DerivedBiomeConfig.from_mapping(controls("derived_biomes")),
    )


def _load_sources(config: L3EcologyConfig) -> _EcologySources:
    source_dirs = (
        config.terrain_dir,
        config.hydrology_dir,
        config.channel_geometry_dir,
        config.surface_materials_dir,
    )
    _require_disjoint_output(config.output_dir, source_dirs)
    manifests: list[dict[str, Any]] = []
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
            raise RuntimeError(f"L3 ecology requires accepted source: {directory}")
        _verify_manifest_outputs(directory, manifest)
        manifests.append(manifest)
    terrain_manifest, hydrology_manifest, channel_manifest, surface_manifest = manifests
    target_ids = {str(manifest.get("target_id")) for manifest in manifests}
    if len(target_ids) != 1:
        raise RuntimeError("L3 ecology source target IDs differ")

    terrain_checksum = _file_checksum(config.terrain_dir / "manifest.json")
    hydrology_checksum = _file_checksum(config.hydrology_dir / "manifest.json")
    channel_checksum = _file_checksum(config.channel_geometry_dir / "manifest.json")
    _require_source_manifest_checksum(
        hydrology_manifest,
        "terrain_manifest_sha256",
        terrain_checksum,
        "hydrology",
    )
    _require_source_manifest_checksum(
        channel_manifest,
        "terrain_manifest_sha256",
        terrain_checksum,
        "channel geometry",
    )
    _require_source_manifest_checksum(
        channel_manifest,
        "hydrology_manifest_sha256",
        hydrology_checksum,
        "channel geometry",
    )
    _require_source_manifest_checksum(
        surface_manifest,
        "terrain_manifest_sha256",
        terrain_checksum,
        "surface materials",
    )
    _require_source_manifest_checksum(
        surface_manifest,
        "hydrology_manifest_sha256",
        hydrology_checksum,
        "surface materials",
    )
    _require_source_manifest_checksum(
        surface_manifest,
        "channel_manifest_sha256",
        channel_checksum,
        "surface materials",
    )

    handoff_dir = Path(str(terrain_manifest["source"]["handoff_dir"])).resolve()
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
        raise RuntimeError("L3 ecology requires an accepted regional handoff")
    _verify_handoff_outputs(handoff_dir, handoff_manifest)
    handoff_checksum = _file_checksum(handoff_manifest_path)
    for manifest, label in (
        (terrain_manifest, "terrain"),
        (hydrology_manifest, "hydrology"),
        (surface_manifest, "surface materials"),
    ):
        _require_source_manifest_checksum(
            manifest,
            "handoff_manifest_sha256",
            handoff_checksum,
            label,
        )

    source = handoff_manifest.get("source")
    if not isinstance(source, Mapping) or not source.get("world_config"):
        raise RuntimeError("Regional handoff does not identify its source world config")
    world_config_path = Path(str(source["world_config"])).resolve()
    if not world_config_path.exists():
        raise FileNotFoundError(world_config_path)
    expected_world_checksum = source.get("world_config_sha256")
    if expected_world_checksum and _file_checksum(world_config_path) != expected_world_checksum:
        raise RuntimeError("Source world config checksum differs from the regional handoff")
    (
        planet_config,
        atmosphere_config,
        envelope_config,
        potential_config,
        functional_config,
        biome_config,
    ) = _world_controls(world_config_path)

    terrain = zarr.open_group(str(config.terrain_dir / "terrain.zarr"), mode="r")
    hydrology = zarr.open_group(str(config.hydrology_dir / "hydrology.zarr"), mode="r")
    channels = zarr.open_group(
        str(config.channel_geometry_dir / "channel_geometry.zarr"),
        mode="r",
    )
    surface_group = zarr.open_group(
        str(config.surface_materials_dir / "surface_materials.zarr"),
        mode="r",
    )
    handoff = zarr.open_group(str(handoff_zarr_path), mode="r")
    parent_ids_raw = np.asarray(handoff["parent/cell_id"][:], dtype=np.int32)
    parent_order = np.argsort(parent_ids_raw)
    parent_ids = np.ascontiguousarray(parent_ids_raw[parent_order])
    parent_area = np.ascontiguousarray(
        np.asarray(handoff["parent/area_km2"][:], dtype=np.float64)[parent_order]
    )
    parent_priors: dict[str, np.ndarray] = {}
    for path in REQUIRED_PARENT_PRIORS:
        if path not in handoff:
            raise KeyError(f"Regional handoff lacks required ecology prior {path}")
        parent_priors[path] = np.ascontiguousarray(np.asarray(handoff[path][:])[parent_order])

    cell_id = np.asarray(terrain["geometry/cell_id"][:], dtype=np.uint64)
    surface_cell_id = np.asarray(surface_group["geometry/cell_id"][:], dtype=np.uint64)
    if not np.array_equal(cell_id, surface_cell_id):
        raise RuntimeError("L3 surface-material and terrain cell IDs differ")
    face = np.asarray(terrain["geometry/face"][:], dtype=np.uint8)
    row = np.asarray(terrain["geometry/row"][:], dtype=np.int32)
    column = np.asarray(terrain["geometry/column"][:], dtype=np.int32)
    area_km2 = np.asarray(terrain["geometry/area_km2"][:], dtype=np.float64)
    elevation_m = np.asarray(terrain["terrain/elevation_m"][:], dtype=np.float32)
    l0_parent_id = np.asarray(
        surface_group["geometry/l0_parent_cell_id"][:],
        dtype=np.int32,
    )
    local_relief = np.asarray(
        surface_group["drivers/LocalReliefM"][:],
        dtype=np.float32,
    )
    inside_display = np.asarray(
        surface_group["geometry/inside_display_window"][:],
        dtype=bool,
    )
    inside_core = np.asarray(
        surface_group["geometry/inside_routed_catchment_core"][:],
        dtype=bool,
    )
    arrays = (
        face,
        row,
        column,
        area_km2,
        elevation_m,
        l0_parent_id,
        local_relief,
        inside_display,
        inside_core,
    )
    if any(len(values) != len(cell_id) for values in arrays):
        raise RuntimeError("L3 ecology source arrays have inconsistent lengths")
    order, height, width = _spatial_layout(row, column)
    resolution = handoff_manifest["resolution"]
    parent_resolution = int(resolution["parent_face_resolution"])
    child_resolution = int(terrain_manifest["hierarchy"]["child_face_resolution"])
    if child_resolution % parent_resolution:
        raise RuntimeError("L3 and parent face resolutions are not integer-related")
    actual_cell_size_m = float(terrain_manifest["hierarchy"]["actual_area_equivalent_cell_size_m"])
    return _EcologySources(
        target_id=target_ids.pop(),
        terrain_manifest=terrain_manifest,
        hydrology_manifest=hydrology_manifest,
        channel_manifest=channel_manifest,
        surface_manifest=surface_manifest,
        handoff_manifest=handoff_manifest,
        handoff_dir=handoff_dir,
        world_config_path=world_config_path,
        terrain=terrain,
        hydrology=hydrology,
        channels=channels,
        surface=surface_group,
        parent_ids=parent_ids,
        parent_area_km2=parent_area,
        parent_priors=parent_priors,
        cell_id=np.ascontiguousarray(cell_id),
        face=np.ascontiguousarray(face),
        row=np.ascontiguousarray(row),
        column=np.ascontiguousarray(column),
        l0_parent_id=np.ascontiguousarray(l0_parent_id),
        area_km2=np.ascontiguousarray(area_km2),
        elevation_m=np.ascontiguousarray(elevation_m),
        local_relief_m=np.ascontiguousarray(local_relief),
        inside_display=np.ascontiguousarray(inside_display),
        inside_core=np.ascontiguousarray(inside_core),
        spatial_order=order,
        height=height,
        width=width,
        parent_face_resolution=parent_resolution,
        child_face_resolution=child_resolution,
        actual_cell_size_m=actual_cell_size_m,
        planet_config=planet_config,
        atmosphere_config=atmosphere_config,
        envelope_config=envelope_config,
        potential_config=potential_config,
        functional_config=functional_config,
        biome_config=biome_config,
    )


def _fingerprint(
    config: L3EcologyConfig,
    sources: _EcologySources,
) -> tuple[str, dict[str, Any]]:
    native_names = (
        "biosphere_envelope_native",
        "potential_biosphere_native",
        "functional_vegetation_native",
        "derived_biomes_native",
    )
    native = {name: native_library_info(name) for name in native_names}
    bindings = {
        name: _file_checksum(Path(__file__).with_name(filename))
        for name, filename in (
            ("biosphere_envelope", "_biosphere_envelope_native.py"),
            ("potential_biosphere", "_potential_biosphere_native.py"),
            ("functional_vegetation", "_functional_vegetation_native.py"),
            ("derived_biomes", "_derived_biomes_native.py"),
        )
    }
    components = {
        "format_version": ECOLOGY_FORMAT_VERSION,
        "model_version": ECOLOGY_MODEL_VERSION,
        "terrain_manifest_sha256": _file_checksum(config.terrain_dir / "manifest.json"),
        "hydrology_manifest_sha256": _file_checksum(config.hydrology_dir / "manifest.json"),
        "channel_manifest_sha256": _file_checksum(config.channel_geometry_dir / "manifest.json"),
        "surface_materials_manifest_sha256": _file_checksum(
            config.surface_materials_dir / "manifest.json"
        ),
        "handoff_manifest_sha256": _file_checksum(sources.handoff_dir / "manifest.json"),
        "world_config_sha256": _file_checksum(sources.world_config_path),
        "terrain_zarr_sha256": sources.terrain_manifest["outputs"]["terrain_zarr"]["sha256_tree"],
        "hydrology_zarr_sha256": sources.hydrology_manifest["outputs"]["hydrology_zarr"][
            "sha256_tree"
        ],
        "channel_zarr_sha256": sources.channel_manifest["outputs"]["channel_geometry_zarr"][
            "sha256_tree"
        ],
        "surface_materials_zarr_sha256": sources.surface_manifest["outputs"][
            "surface_materials_zarr"
        ]["sha256_tree"],
        "handoff_zarr_sha256": sources.handoff_manifest["outputs"]["zarr"]["sha256"],
        "controls": {
            "regional": {
                **asdict(config),
                "terrain_dir": str(config.terrain_dir),
                "hydrology_dir": str(config.hydrology_dir),
                "channel_geometry_dir": str(config.channel_geometry_dir),
                "surface_materials_dir": str(config.surface_materials_dir),
                "output_dir": str(config.output_dir),
                "source_config": str(config.source_config) if config.source_config else None,
            },
            "planet": asdict(sources.planet_config),
            "atmosphere": asdict(sources.atmosphere_config),
            "biosphere_envelope": asdict(sources.envelope_config),
            "potential_biosphere": asdict(sources.potential_config),
            "functional_vegetation": asdict(sources.functional_config),
            "derived_biomes": asdict(sources.biome_config),
        },
        "native": {
            name: {
                "abi_version": info["abi_version"],
                "sha256": info["sha256"],
            }
            for name, info in native.items()
        },
        "bindings": bindings,
        "orchestrator_sha256": _file_checksum(Path(__file__)),
    }
    fingerprint = _canonical_hash(components)
    return fingerprint, {
        **components,
        "generation_config_sha256": (
            _file_checksum(config.source_config) if config.source_config else None
        ),
    }


def _parent_rows(parent_ids: np.ndarray, requested_ids: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(parent_ids, requested_ids)
    valid = positions < len(parent_ids)
    if np.any(valid):
        valid[valid] &= parent_ids[positions[valid]] == requested_ids[valid]
    if not np.all(valid):
        raise RuntimeError("L3 ecology cell references a parent absent from the handoff")
    return np.ascontiguousarray(positions, dtype=np.int32)


def _interpolation_stencil(
    sources: _EcologySources,
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
    sources: _EcologySources,
    name: str,
    rows: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return np.ascontiguousarray(
        _interpolate_prior(sources.parent_priors[f"parent_priors/{name}"], rows, weights),
        dtype=np.float32,
    )


def _atmospheric_state(
    sources: _EcologySources,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    planet = sources.planet_config
    atmosphere = sources.atmosphere_config
    gravity_m_s2 = planet.surface_gravity_g * 9.80665
    molar_mass_kg_mol = atmosphere.mean_molar_mass_g_mol / 1_000.0
    scale_height_m = (
        8.314462618 * atmosphere.reference_temperature_k / (molar_mass_kg_mol * gravity_m_s2)
    )
    height = np.maximum(sources.elevation_m[start:end].astype(np.float64), 0.0)
    pressure_kpa = atmosphere.mean_surface_pressure_kpa * np.exp(-height / scale_height_m)
    co2 = pressure_kpa * 1_000.0 * atmosphere.carbon_dioxide_ppm * 1e-6
    oxygen = pressure_kpa * atmosphere.oxygen_dry_fraction
    return (
        np.ascontiguousarray(co2, dtype=np.float32),
        np.ascontiguousarray(oxygen, dtype=np.float32),
    )


def _chunk_drivers(
    sources: _EcologySources,
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    rows, weights, _ = _interpolation_stencil(sources, start, end)
    monthly_insolation = _prior(
        sources,
        "planet/MonthlyInsolationWm2",
        rows,
        weights,
    ).T
    base_temperature = _prior(
        sources,
        "climate/MonthlySurfaceTemperatureC",
        rows,
        weights,
    )
    temperature_adjustment = np.asarray(
        sources.surface["drivers/TemperatureAdjustmentC"][start:end],
        dtype=np.float32,
    )
    monthly_temperature = np.ascontiguousarray(
        (base_temperature + temperature_adjustment[:, None]).T,
        dtype=np.float32,
    )
    annual_temperature = np.ascontiguousarray(
        np.mean(monthly_temperature, axis=0),
        dtype=np.float32,
    )
    monthly_precipitation = np.ascontiguousarray(
        _prior(
            sources,
            "climate/MonthlyPrecipitationMm",
            rows,
            weights,
        ).T,
        dtype=np.float32,
    )
    annual_precipitation = np.ascontiguousarray(
        np.sum(monthly_precipitation, axis=0),
        dtype=np.float32,
    )
    glacier = _prior(
        sources,
        "cryosphere/GlacierIceFraction",
        rows,
        weights,
    )
    co2, oxygen = _atmospheric_state(sources, start, end)
    return {
        "MonthlyInsolationWm2": np.ascontiguousarray(monthly_insolation),
        "MonthlyTemperatureC": monthly_temperature,
        "AnnualMeanTemperatureC": annual_temperature,
        "AnnualPrecipitationMm": annual_precipitation,
        "GlacierIceFraction": np.ascontiguousarray(np.clip(glacier, 0.0, 1.0)),
        "CO2PartialPressurePa": co2,
        "OxygenPartialPressureKPa": oxygen,
    }


def _allocate_outputs(
    cell_count: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    envelope: dict[str, np.ndarray] = {}
    for native_name, artifact_name in ENVELOPE_NATIVE_OUTPUTS.items():
        shape = (MONTHS, cell_count) if artifact_name in MONTHLY_ENVELOPE_OUTPUTS else (cell_count,)
        envelope[native_name] = np.zeros(shape, dtype=np.float32)
    potential: dict[str, np.ndarray] = {}
    for native_name, artifact_name in POTENTIAL_NATIVE_OUTPUTS.items():
        shape = (
            (MONTHS, cell_count)
            if artifact_name in MONTHLY_POTENTIAL_BIOSPHERE_OUTPUTS
            else (cell_count,)
        )
        potential[native_name] = np.zeros(shape, dtype=np.float32)
    functional = {
        "functional_type_fractions_out": np.zeros(
            (FUNCTIONAL_TYPE_COUNT, cell_count), dtype=np.float32
        ),
        "nonvegetated_fractions_out": np.zeros(
            (NONVEGETATED_TYPE_COUNT, cell_count), dtype=np.float32
        ),
        "resource_potentials_out": np.zeros(
            (RESOURCE_POTENTIAL_COUNT, cell_count), dtype=np.float32
        ),
        "confidence_out": np.zeros(cell_count, dtype=np.float32),
        "dominant_cover_code_out": np.zeros(cell_count, dtype=np.uint8),
    }
    biome = {
        "biome_fractions_out": np.zeros((BIOME_COUNT, cell_count), dtype=np.float32),
        "classification_confidence_out": np.zeros(cell_count, dtype=np.float32),
        "dominance_margin_out": np.zeros(cell_count, dtype=np.float32),
        "transition_index_out": np.zeros(cell_count, dtype=np.float32),
        "primary_biome_code_out": np.zeros(cell_count, dtype=np.uint8),
        "secondary_biome_code_out": np.zeros(cell_count, dtype=np.uint8),
        "dominant_landscape_code_out": np.zeros(cell_count, dtype=np.uint8),
    }
    return envelope, potential, functional, biome


def _run_chunk(
    sources: _EcologySources,
    start: int,
    end: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float],
]:
    drivers = _chunk_drivers(sources, start, end)
    envelope, potential, functional, biome = _allocate_outputs(end - start)
    surface = sources.surface
    hydrology = sources.hydrology
    areas = np.ascontiguousarray(sources.area_km2[start:end], dtype=np.float64)
    ocean = np.ascontiguousarray(
        hydrology["surface/physical_ocean_fraction"][start:end],
        dtype=np.float32,
    )
    monthly_liquid = np.ascontiguousarray(
        np.asarray(
            surface["monthly/MonthlySoilLiquidInputMm"][start:end, :],
            dtype=np.float32,
        ).T
    )
    monthly_saturation = np.ascontiguousarray(
        np.asarray(
            surface["monthly/MonthlySoilSaturationFraction"][start:end, :],
            dtype=np.float32,
        ).T
    )
    soil_bearing = np.ascontiguousarray(
        surface["soil/SoilBearingFraction"][start:end],
        dtype=np.float32,
    )
    nutrient = np.ascontiguousarray(
        surface["soil/SoilNutrientPotential"][start:end],
        dtype=np.float32,
    )
    fertility = np.ascontiguousarray(
        surface["soil/SoilFertilityPotential"][start:end],
        dtype=np.float32,
    )
    salinity = np.ascontiguousarray(
        surface["soil/SoilSalinityIndex"][start:end],
        dtype=np.float32,
    )
    soil_confidence = np.ascontiguousarray(
        surface["soil/SoilConfidence"][start:end],
        dtype=np.float32,
    )
    envelope_config = sources.envelope_config
    atmosphere = sources.atmosphere_config
    reference_co2_pa = (
        atmosphere.mean_surface_pressure_kpa * 1_000.0 * atmosphere.reference_co2_ppm * 1e-6
    )
    profile_is_reference = atmosphere.validation_profile == "earthlike"
    envelope_stats = run_biosphere_envelope(
        seconds_per_month=sources.planet_config.orbital_period_days * 86_400.0 / MONTHS,
        par_fraction=envelope_config.par_fraction,
        shortwave_transmission=envelope_config.shortwave_transmission,
        thermal_minimum_c=envelope_config.thermal_minimum_c,
        thermal_optimum_low_c=envelope_config.thermal_optimum_low_c,
        thermal_optimum_high_c=envelope_config.thermal_optimum_high_c,
        thermal_maximum_c=envelope_config.thermal_maximum_c,
        water_input_half_saturation_mm=envelope_config.water_input_half_saturation_mm,
        nutrient_half_saturation_index=envelope_config.nutrient_half_saturation_index,
        co2_half_saturation_pa=envelope_config.co2_half_saturation_pa,
        reference_co2_partial_pressure_pa=reference_co2_pa,
        reference_oxygen_partial_pressure_kpa=(
            envelope_config.reference_oxygen_partial_pressure_kpa
        ),
        photosynthetic_conversion_efficiency=(envelope_config.photosynthetic_conversion_efficiency),
        minimum_productive_energy_mj_m2_year=(envelope_config.minimum_productive_energy_mj_m2_year),
        confidence_multiplier=(
            1.0
            if profile_is_reference
            else envelope_config.nonreference_profile_confidence_multiplier
        ),
        areas=areas,
        ocean=ocean,
        monthly_insolation=drivers["MonthlyInsolationWm2"],
        monthly_temperature=drivers["MonthlyTemperatureC"],
        monthly_liquid_input=monthly_liquid,
        monthly_soil_saturation=monthly_saturation,
        soil_bearing=soil_bearing,
        nutrient_potential=nutrient,
        fertility_potential=fertility,
        salinity=salinity,
        soil_confidence=soil_confidence,
        co2_partial_pressure=drivers["CO2PartialPressurePa"],
        oxygen_partial_pressure=drivers["OxygenPartialPressureKPa"],
        **envelope,
    )

    potential_config = sources.potential_config
    soil_depth = np.ascontiguousarray(
        surface["soil/SoilDepthM"][start:end],
        dtype=np.float32,
    )
    regolith_depth = np.ascontiguousarray(
        surface["soil/RegolithDepthM"][start:end],
        dtype=np.float32,
    )
    hydric = np.ascontiguousarray(
        surface["soil/HydricSoilFraction"][start:end],
        dtype=np.float32,
    )
    potential_stats = run_potential_biosphere(
        energy_per_kg_carbon_mj=potential_config.energy_per_kg_carbon_mj,
        cover_half_saturation_npp_kg_c_m2_year=(
            potential_config.cover_half_saturation_npp_kg_c_m2_year
        ),
        active_month_thermal_threshold=potential_config.active_month_thermal_threshold,
        active_month_water_threshold=potential_config.active_month_water_threshold,
        cold_pressure_reference_c=potential_config.cold_pressure_reference_c,
        cold_pressure_release_c=potential_config.cold_pressure_release_c,
        heat_pressure_onset_c=potential_config.heat_pressure_onset_c,
        heat_pressure_reference_c=potential_config.heat_pressure_reference_c,
        minimum_biomass_residence_years=potential_config.minimum_biomass_residence_years,
        maximum_biomass_residence_years=potential_config.maximum_biomass_residence_years,
        biomass_residence_baseline_fraction=(potential_config.biomass_residence_baseline_fraction),
        woody_biomass_residence_weight=(potential_config.woody_biomass_residence_weight),
        resource_conservative_biomass_residence_weight=(
            potential_config.resource_conservative_biomass_residence_weight
        ),
        low_productivity_biomass_residence_weight=(
            potential_config.low_productivity_biomass_residence_weight
        ),
        maximum_rooting_depth_m=potential_config.maximum_rooting_depth_m,
        maximum_canopy_height_m=potential_config.maximum_canopy_height_m,
        maximum_leaf_area_index=potential_config.maximum_leaf_area_index,
        maximum_standing_biomass_kg_c_m2=(potential_config.maximum_standing_biomass_kg_c_m2),
        areas=areas,
        ocean=ocean,
        monthly_primary_energy=envelope["monthly_primary_energy_out"],
        monthly_thermal_opportunity=envelope["monthly_thermal_opportunity_out"],
        monthly_water_opportunity=envelope["monthly_liquid_opportunity_out"],
        monthly_temperature=drivers["MonthlyTemperatureC"],
        monthly_soil_saturation=monthly_saturation,
        surface_support=envelope["terrestrial_surface_support_out"],
        nutrient_support=envelope["nutrient_support_out"],
        environmental_stress=envelope["environmental_stress_out"],
        soil_depth=soil_depth,
        regolith_depth=regolith_depth,
        salinity=salinity,
        hydric_fraction=hydric,
        soil_confidence=soil_confidence,
        envelope_confidence=envelope["confidence_out"],
        **potential,
    )

    functional_config = sources.functional_config
    lake = np.ascontiguousarray(
        hydrology["surface/lake_fraction"][start:end],
        dtype=np.float32,
    )
    wetland = np.ascontiguousarray(
        hydrology["surface/wetland_fraction"][start:end],
        dtype=np.float32,
    )
    drainage = np.ascontiguousarray(
        surface["soil/SoilDrainageIndex"][start:end],
        dtype=np.float32,
    )
    functional_stats = run_functional_vegetation(
        warm_transition_midpoint_c=functional_config.warm_transition_midpoint_c,
        warm_transition_width_c=functional_config.warm_transition_width_c,
        npp_response_half_saturation_kg_c_m2_year=(
            functional_config.npp_response_half_saturation_kg_c_m2_year
        ),
        biomass_response_half_saturation_kg_c_m2=(
            functional_config.biomass_response_half_saturation_kg_c_m2
        ),
        terrain_relief_half_saturation_m=(functional_config.terrain_relief_half_saturation_m),
        crop_soil_depth_half_saturation_m=(functional_config.crop_soil_depth_half_saturation_m),
        strategy_confidence_multiplier=functional_config.strategy_confidence_multiplier,
        areas=areas,
        ocean=ocean,
        vegetation_cover=potential["vegetation_cover_out"],
        annual_npp=potential["annual_npp_out"],
        standing_biomass=potential["standing_biomass_out"],
        growing_season=potential["growing_season_out"],
        productivity_seasonality=potential["productivity_seasonality_out"],
        drought_pressure=potential["drought_pressure_out"],
        cold_pressure=potential["cold_pressure_out"],
        heat_pressure=potential["heat_pressure_out"],
        waterlogging_pressure=potential["waterlogging_pressure_out"],
        salinity_pressure=potential["salinity_pressure_out"],
        woody_trait=potential["woody_trait_out"],
        resource_conservative_trait=potential["resource_conservative_trait_out"],
        fuel_continuity=potential["fuel_continuity_out"],
        biosphere_confidence=potential["confidence_out"],
        annual_temperature=drivers["AnnualMeanTemperatureC"],
        soil_fertility=fertility,
        soil_depth=soil_depth,
        soil_bearing=soil_bearing,
        soil_drainage=drainage,
        glacier_fraction=drivers["GlacierIceFraction"],
        lake_fraction=lake,
        wetland_fraction=wetland,
        terrain_relief=np.ascontiguousarray(
            sources.local_relief_m[start:end],
            dtype=np.float32,
        ),
        **functional,
    )

    biome_config = sources.biome_config
    biome_stats = run_derived_biomes(
        highland_elevation_start_m=biome_config.highland_elevation_start_m,
        highland_elevation_full_m=biome_config.highland_elevation_full_m,
        highland_relief_start_m=biome_config.highland_relief_start_m,
        highland_relief_full_m=biome_config.highland_relief_full_m,
        minimum_classifiable_ground_fraction=(biome_config.minimum_classifiable_ground_fraction),
        ambiguity_margin_threshold=biome_config.ambiguity_margin_threshold,
        transition_confidence_weight=biome_config.transition_confidence_weight,
        areas=areas,
        ocean=ocean,
        annual_temperature=drivers["AnnualMeanTemperatureC"],
        annual_precipitation=drivers["AnnualPrecipitationMm"],
        growing_season=potential["growing_season_out"],
        seasonality=potential["productivity_seasonality_out"],
        drought=potential["drought_pressure_out"],
        waterlogging=potential["waterlogging_pressure_out"],
        biosphere_confidence=potential["confidence_out"],
        functional_confidence=functional["confidence_out"],
        wetland_fraction=wetland,
        elevation=np.ascontiguousarray(
            sources.elevation_m[start:end],
            dtype=np.float32,
        ),
        relief=np.ascontiguousarray(
            sources.local_relief_m[start:end],
            dtype=np.float32,
        ),
        functional_type_fractions=functional["functional_type_fractions_out"],
        nonvegetated_fractions=functional["nonvegetated_fractions_out"],
        resource_potentials=functional["resource_potentials_out"],
        **biome,
    )
    stats = {
        **{f"envelope_{name}": value for name, value in envelope_stats.items()},
        **{f"potential_{name}": value for name, value in potential_stats.items()},
        **{f"functional_{name}": value for name, value in functional_stats.items()},
        **{f"biome_{name}": value for name, value in biome_stats.items()},
    }
    return drivers, envelope, potential, functional, biome, stats


def _dataset_spec(name: str) -> tuple[tuple[int, ...], np.dtype[Any], str]:
    if name in DRIVER_PATHS:
        if name in {"MonthlyInsolationWm2", "MonthlyTemperatureC"}:
            return (MONTHS,), np.dtype(np.float32), "persisted L3 monthly ecology driver"
        return (), np.dtype(np.float32), "persisted L3 ecology driver"
    if name in MONTHLY_ENVELOPE_OUTPUTS or name in MONTHLY_POTENTIAL_BIOSPHERE_OUTPUTS:
        return (MONTHS,), np.dtype(np.float32), "cell-first monthly ecological state"
    if name == "FunctionalTypeFractions":
        return (
            (FUNCTIONAL_TYPE_COUNT,),
            np.dtype(np.float32),
            "physical full-cell functional vegetation fractions",
        )
    if name == "NonVegetatedFractions":
        return (
            (NONVEGETATED_TYPE_COUNT,),
            np.dtype(np.float32),
            "physical full-cell nonvegetated fractions",
        )
    if name == "FunctionalResourcePotentials":
        return (
            (RESOURCE_POTENTIAL_COUNT,),
            np.dtype(np.float32),
            "bounded physical suitability, not actual land use",
        )
    if name == "BiomeFractions":
        return (
            (BIOME_COUNT,),
            np.dtype(np.float32),
            "derived full-cell ecological-ground biome fractions",
        )
    code_names = {
        "DominantFunctionalCoverCode",
        "DominantBiomeCode",
        "SecondaryBiomeCode",
        "DominantLandscapeCode",
    }
    if name in code_names:
        return (), np.dtype(np.uint8), "derived reproducible query code"
    return (), np.dtype(np.float32), "L3 ecology state"


def _initialize_partial(
    partial: Path,
    config: L3EcologyConfig,
    sources: _EcologySources,
    run_fingerprint: str,
) -> Any:
    partial.mkdir(parents=True)
    zarr_path = partial / "ecology.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    cell_count = len(sources.cell_id)
    chunk_count = math.ceil(cell_count / config.chunk_rows)
    chunks = (min(config.chunk_rows, cell_count),)
    root.attrs.update(
        {
            "format_version": ECOLOGY_FORMAT_VERSION,
            "model_version": ECOLOGY_MODEL_VERSION,
            "status": "partial",
            "target_id": sources.target_id,
            "run_fingerprint": run_fingerprint,
            "cell_count": cell_count,
            "chunk_count": chunk_count,
            "chunk_rows": config.chunk_rows,
            "storage_layout": "cell-first; month and class axes trail",
            "source_prior_semantics": (
                "coarse ecology is comparison and training context; L3 ecology is "
                "recomputed from L3 soil water and terrain"
            ),
        }
    )
    geometry = root.require_group("geometry")
    for name, dtype, values, semantics in (
        ("cell_id", np.uint64, sources.cell_id, "stable global L3 cell ID"),
        (
            "l0_parent_cell_id",
            np.int32,
            sources.l0_parent_id,
            "stable inherited L0 comparison parent",
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
            "fine routed basin used for hydrological acceptance",
        ),
    ):
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
    for artifact_name, path in OUTPUT_PATHS.items():
        trailing, dtype, semantics = _dataset_spec(artifact_name)
        group_path, dataset_name = path.rsplit("/", 1)
        group = root.require_group(group_path)
        shape = (cell_count, *trailing)
        dataset_chunks = (min(config.chunk_rows, cell_count), *trailing)
        _zarr_dataset(
            group,
            dataset_name,
            shape=shape,
            dtype=dtype,
            chunks=dataset_chunks,
            semantics=semantics,
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
            "format_version": ECOLOGY_FORMAT_VERSION,
            "model_version": ECOLOGY_MODEL_VERSION,
            "run_fingerprint": run_fingerprint,
            "cell_count": cell_count,
            "chunk_count": chunk_count,
        },
    )
    (partial / "chunk_stats").mkdir()
    return root


def _open_partial(
    partial: Path,
    config: L3EcologyConfig,
    sources: _EcologySources,
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
            return zarr.open_group(str(partial / "ecology.zarr"), mode="r+"), True
        shutil.rmtree(partial)
    return _initialize_partial(partial, config, sources, run_fingerprint), False


def _write_chunk(
    root: Any,
    start: int,
    end: int,
    drivers: Mapping[str, np.ndarray],
    envelope: Mapping[str, np.ndarray],
    potential: Mapping[str, np.ndarray],
    functional: Mapping[str, np.ndarray],
    biome: Mapping[str, np.ndarray],
) -> None:
    for name, values in drivers.items():
        root[DRIVER_PATHS[name]][start:end] = values.T if values.ndim == 2 else values
    for native_name, artifact_name in ENVELOPE_NATIVE_OUTPUTS.items():
        values = envelope[native_name]
        root[ENVELOPE_PATHS[artifact_name]][start:end] = values.T if values.ndim == 2 else values
    for native_name, artifact_name in POTENTIAL_NATIVE_OUTPUTS.items():
        values = potential[native_name]
        root[POTENTIAL_PATHS[artifact_name]][start:end] = values.T if values.ndim == 2 else values
    for native_name, artifact_name in FUNCTIONAL_NATIVE_OUTPUTS.items():
        values = functional[native_name]
        root[FUNCTIONAL_PATHS[artifact_name]][start:end] = values.T if values.ndim == 2 else values
    for native_name, artifact_name in BIOME_NATIVE_OUTPUTS.items():
        values = biome[native_name]
        root[BIOME_PATHS[artifact_name]][start:end] = values.T if values.ndim == 2 else values


def _geometry_chunk(
    sources: _EcologySources,
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    return {
        "cell_id": sources.cell_id[start:end],
        "l0_parent_cell_id": sources.l0_parent_id[start:end],
        "inside_display_window": sources.inside_display[start:end],
        "inside_routed_catchment_core": sources.inside_core[start:end],
    }


def _write_geometry_chunk(
    root: Any,
    sources: _EcologySources,
    start: int,
    end: int,
) -> None:
    for name, values in _geometry_chunk(sources, start, end).items():
        root[GEOMETRY_PATHS[name]][start:end] = values


def _output_chunk_checksums(
    zarr_path: Path,
    root: Any,
    chunk_index: int,
) -> dict[str, str]:
    return {
        path: _file_checksum(_zarr_chunk_path(zarr_path, root, path, chunk_index))
        for path in CHECKSUM_CHUNK_PATHS
    }


def _completed_chunk_is_valid(
    root: Any,
    partial: Path,
    chunk_index: int,
    start: int,
    end: int,
) -> bool:
    try:
        stats = _load_json(partial / "chunk_stats" / f"{chunk_index:06d}.json")
        checksums = stats.get("artifact_chunk_sha256")
        paths = CHECKSUM_CHUNK_PATHS
        if (
            stats.get("chunk_index") != chunk_index
            or stats.get("start") != start
            or stats.get("end") != end
            or not isinstance(checksums, Mapping)
            or set(checksums) != set(paths)
        ):
            return False
        zarr_path = partial / "ecology.zarr"
        for path in paths:
            chunk_path = _zarr_chunk_path(zarr_path, root, path, chunk_index)
            if not chunk_path.is_file() or _file_checksum(chunk_path) != checksums[path]:
                return False
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
        return False
    return True


def _generate_chunks(
    root: Any,
    partial: Path,
    config: L3EcologyConfig,
    sources: _EcologySources,
) -> tuple[int, int]:
    zarr_path = partial / "ecology.zarr"
    completion = np.asarray(root["progress/chunk_complete"][:], dtype=bool)
    invalid_completed: list[int] = []
    for chunk_index, start in enumerate(range(0, len(sources.cell_id), config.chunk_rows)):
        end = min(start + config.chunk_rows, len(sources.cell_id))
        if completion[chunk_index] and not _completed_chunk_is_valid(
            root,
            partial,
            chunk_index,
            start,
            end,
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
        drivers, envelope, potential, functional, biome, stats = _run_chunk(
            sources,
            start,
            end,
        )
        _write_geometry_chunk(root, sources, start, end)
        _write_chunk(
            root,
            start,
            end,
            drivers,
            envelope,
            potential,
            functional,
            biome,
        )
        _sync_zarr_chunk(zarr_path, root, CHECKSUM_CHUNK_PATHS, chunk_index)
        checksums = _output_chunk_checksums(zarr_path, root, chunk_index)
        _write_json_durable(
            partial / "chunk_stats" / f"{chunk_index:06d}.json",
            {
                "chunk_index": chunk_index,
                "start": start,
                "end": end,
                "artifact_chunk_sha256": checksums,
                **stats,
            },
        )
        root["progress/chunk_complete"][chunk_index] = True
        _sync_zarr_array(zarr_path, "progress/chunk_complete")
    return resumed_count, len(completion)


def _percentile(values: list[float] | np.ndarray, percentile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, percentile)) if array.size else 0.0


def _weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    selected_weights = np.asarray(weights[mask], dtype=np.float64)
    denominator = float(np.sum(selected_weights, dtype=np.float64))
    if denominator <= 0.0:
        return float("nan")
    return float(
        np.sum(np.asarray(values[mask], dtype=np.float64) * selected_weights, dtype=np.float64)
        / denominator
    )


def _represented_parent_metrics(
    root: Any,
    sources: _EcologySources,
    config: L3EcologyConfig,
) -> dict[str, float | int]:
    parent_rows = _parent_rows(sources.parent_ids, sources.l0_parent_id)
    parent_count = len(sources.parent_ids)
    represented_area = np.zeros(parent_count, dtype=np.float64)
    npp_area = np.zeros(parent_count, dtype=np.float64)
    functional_area = np.zeros((parent_count, FUNCTIONAL_TYPE_COUNT), dtype=np.float64)
    biome_area = np.zeros((parent_count, BIOME_COUNT), dtype=np.float64)
    for start in range(0, len(sources.cell_id), config.chunk_rows):
        end = min(start + config.chunk_rows, len(sources.cell_id))
        ocean = np.asarray(
            sources.hydrology["surface/physical_ocean_fraction"][start:end],
            dtype=np.float32,
        )
        represented = ocean < 0.5
        rows = parent_rows[start:end][represented]
        area = sources.area_km2[start:end][represented]
        np.add.at(represented_area, rows, area)
        npp = np.asarray(
            root[POTENTIAL_PATHS["AnnualPotentialNPPKgCM2"]][start:end],
            dtype=np.float64,
        )
        np.add.at(npp_area, rows, npp[represented] * area)
        functional = np.asarray(
            root[FUNCTIONAL_PATHS["FunctionalTypeFractions"]][start:end, :],
            dtype=np.float64,
        )
        biomes = np.asarray(
            root[BIOME_PATHS["BiomeFractions"]][start:end, :],
            dtype=np.float64,
        )
        for index in range(FUNCTIONAL_TYPE_COUNT):
            np.add.at(
                functional_area[:, index],
                rows,
                functional[represented, index] * area,
            )
        for index in range(BIOME_COUNT):
            np.add.at(
                biome_area[:, index],
                rows,
                biomes[represented, index] * area,
            )
    coverage = np.divide(
        represented_area,
        sources.parent_area_km2,
        out=np.zeros_like(represented_area),
        where=sources.parent_area_km2 > 0.0,
    )
    represented = coverage >= config.minimum_parent_coverage_fraction
    npp_mean = np.divide(
        npp_area,
        represented_area,
        out=np.zeros_like(npp_area),
        where=represented_area > 0.0,
    )
    functional_mean = np.divide(
        functional_area,
        represented_area[:, None],
        out=np.zeros_like(functional_area),
        where=represented_area[:, None] > 0.0,
    )
    biome_mean = np.divide(
        biome_area,
        represented_area[:, None],
        out=np.zeros_like(biome_area),
        where=represented_area[:, None] > 0.0,
    )
    npp_prior = np.asarray(
        sources.parent_priors["parent_priors/potential_biosphere/AnnualPotentialNPPKgCM2"],
        dtype=np.float64,
    )
    functional_prior = np.asarray(
        sources.parent_priors["parent_priors/functional_vegetation/FunctionalTypeFractions"],
        dtype=np.float64,
    )
    biome_prior = np.asarray(
        sources.parent_priors["parent_priors/derived_biomes/BiomeFractions"],
        dtype=np.float64,
    )
    if functional_prior.shape != functional_mean.shape:
        raise RuntimeError("Inherited functional prior axis differs from the native kernel")
    if biome_prior.shape != biome_mean.shape:
        raise RuntimeError("Inherited biome prior axis differs from the native kernel")
    selected_count = int(np.count_nonzero(represented))
    if selected_count:
        npp_relative = np.abs(npp_mean[represented] - npp_prior[represented]) / np.maximum(
            npp_prior[represented],
            0.02,
        )
        functional_l1 = np.sum(
            np.abs(functional_mean[represented] - functional_prior[represented]),
            axis=1,
        )
        biome_l1 = np.sum(
            np.abs(biome_mean[represented] - biome_prior[represented]),
            axis=1,
        )
    else:
        npp_relative = np.asarray([np.inf])
        functional_l1 = np.asarray([np.inf])
        biome_l1 = np.asarray([np.inf])
    return {
        "represented_parent_count": selected_count,
        "represented_parent_coverage_fraction_minimum": (
            float(np.min(coverage[represented])) if selected_count else 0.0
        ),
        "parent_npp_relative_difference_p50": _percentile(npp_relative, 50.0),
        "parent_npp_relative_difference_p95": _percentile(npp_relative, 95.0),
        "parent_functional_fraction_l1_difference_p50": _percentile(functional_l1, 50.0),
        "parent_functional_fraction_l1_difference_p95": _percentile(functional_l1, 95.0),
        "parent_biome_fraction_l1_difference_p50": _percentile(biome_l1, 50.0),
        "parent_biome_fraction_l1_difference_p95": _percentile(biome_l1, 95.0),
    }


def _boundary_metric(
    values: np.ndarray,
    parent: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    differences: list[np.ndarray] = []
    boundary_flags: list[np.ndarray] = []
    for first, second in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        valid = mask[first] & mask[second]
        if not np.any(valid):
            continue
        differences.append(np.abs(values[first][valid] - values[second][valid]))
        boundary_flags.append(parent[first][valid] != parent[second][valid])
    if not differences:
        return {
            "boundary_pair_count": 0,
            "interior_pair_count": 0,
            "boundary_difference_p95": 0.0,
            "interior_difference_p95": 0.0,
            "boundary_to_interior_p95_ratio": 0.0,
        }
    difference = np.concatenate(differences)
    boundary = np.concatenate(boundary_flags)
    boundary_values = difference[boundary]
    interior_values = difference[~boundary]
    boundary_p95 = _percentile(boundary_values, 95.0)
    interior_p95 = _percentile(interior_values, 95.0)
    ratio = boundary_p95 / max(interior_p95, 1e-6)
    return {
        "boundary_pair_count": int(boundary_values.size),
        "interior_pair_count": int(interior_values.size),
        "boundary_difference_p95": boundary_p95,
        "interior_difference_p95": interior_p95,
        "boundary_to_interior_p95_ratio": ratio,
    }


def _parent_motif_correlation(
    values: np.ndarray,
    sources: _EcologySources,
    mask: np.ndarray,
    *,
    bins_per_axis: int = 16,
) -> dict[str, float | int]:
    factor = sources.child_face_resolution // sources.parent_face_resolution
    templates: list[np.ndarray] = []
    for parent_id in np.unique(sources.l0_parent_id[mask]):
        selected = mask & (sources.l0_parent_id == parent_id)
        if np.count_nonzero(selected) < bins_per_axis * bins_per_axis * 4:
            continue
        local_row = (sources.row[selected] % factor).astype(np.float64) / max(factor - 1, 1)
        local_column = (sources.column[selected] % factor).astype(np.float64) / max(
            factor - 1,
            1,
        )
        selected_values = np.asarray(values[selected], dtype=np.float64)
        design = np.column_stack((np.ones(len(selected_values)), local_row, local_column))
        coefficients, *_ = np.linalg.lstsq(design, selected_values, rcond=None)
        residual = selected_values - design @ coefficients
        row_bin = np.minimum(
            (local_row * bins_per_axis).astype(np.int32),
            bins_per_axis - 1,
        )
        column_bin = np.minimum(
            (local_column * bins_per_axis).astype(np.int32),
            bins_per_axis - 1,
        )
        bin_id = row_bin * bins_per_axis + column_bin
        counts = np.bincount(bin_id, minlength=bins_per_axis**2)
        sums = np.bincount(bin_id, weights=residual, minlength=bins_per_axis**2)
        template = np.full(bins_per_axis**2, np.nan, dtype=np.float64)
        valid = counts >= 4
        template[valid] = sums[valid] / counts[valid]
        if np.count_nonzero(valid) >= bins_per_axis**2 // 3:
            templates.append(template)
    correlations: list[float] = []
    for first_index, first in enumerate(templates):
        for second in templates[first_index + 1 :]:
            common = np.isfinite(first) & np.isfinite(second)
            if np.count_nonzero(common) < bins_per_axis**2 // 4:
                continue
            first_values = first[common]
            second_values = second[common]
            if np.std(first_values) <= 1e-8 or np.std(second_values) <= 1e-8:
                continue
            correlation = float(np.corrcoef(first_values, second_values)[0, 1])
            if math.isfinite(correlation):
                correlations.append(max(correlation, 0.0))
    return {
        "motif_template_count": len(templates),
        "motif_pair_count": len(correlations),
        "repeated_parent_motif_correlation_p50": _percentile(correlations, 50.0),
        "repeated_parent_motif_correlation_p95": _percentile(correlations, 95.0),
        "repeated_parent_motif_correlation_maximum": (max(correlations) if correlations else 0.0),
    }


def _relational_metrics(
    root: Any,
    sources: _EcologySources,
) -> dict[str, float | int]:
    ocean = np.asarray(
        sources.hydrology["surface/physical_ocean_fraction"][:],
        dtype=np.float32,
    )
    lake = np.asarray(
        sources.hydrology["surface/lake_fraction"][:],
        dtype=np.float32,
    )
    wetland_surface = np.asarray(
        sources.hydrology["surface/wetland_fraction"][:],
        dtype=np.float32,
    )
    soil_bearing = np.asarray(
        sources.surface["soil/SoilBearingFraction"][:],
        dtype=np.float32,
    )
    hydric = np.asarray(
        sources.surface["soil/HydricSoilFraction"][:],
        dtype=np.float32,
    )
    soil_depth = np.asarray(
        sources.surface["soil/SoilDepthM"][:],
        dtype=np.float32,
    )
    fertility = np.asarray(
        sources.surface["soil/SoilFertilityPotential"][:],
        dtype=np.float32,
    )
    saturation = np.mean(
        np.asarray(
            sources.surface["monthly/MonthlySoilSaturationFraction"][:],
            dtype=np.float32,
        ),
        axis=1,
    )
    annual_npp = np.asarray(
        root[POTENTIAL_PATHS["AnnualPotentialNPPKgCM2"]][:],
        dtype=np.float32,
    )
    cold_pressure = np.asarray(
        root[POTENTIAL_PATHS["ColdAdaptationPressure"]][:],
        dtype=np.float32,
    )
    functional = root[FUNCTIONAL_PATHS["FunctionalTypeFractions"]]
    hydrophytic = np.asarray(functional[:, 6], dtype=np.float32)
    resources = np.asarray(
        root[FUNCTIONAL_PATHS["FunctionalResourcePotentials"]][:],
        dtype=np.float32,
    )
    biomes = root[BIOME_PATHS["BiomeFractions"]]
    wetland_biome = np.asarray(biomes[:, 12], dtype=np.float32)
    cold_biomes = np.sum(
        np.asarray(biomes[:, 8:12], dtype=np.float32),
        axis=1,
    )
    area = sources.area_km2
    analysis = sources.inside_display & (ocean < 0.5) & (lake < 0.5) & (soil_bearing > 0.05)

    wetness = np.clip(0.55 * saturation + 0.45 * hydric, 0.0, 1.0)
    wetness_values = wetness[analysis]
    if wetness_values.size:
        dry_threshold, wet_threshold = np.percentile(wetness_values, (25.0, 75.0))
    else:
        dry_threshold = wet_threshold = 0.0
    dry = analysis & (wetness <= dry_threshold)
    wet = analysis & (wetness >= wet_threshold)
    hydrophytic_dry = _weighted_mean(hydrophytic, area, dry)
    hydrophytic_wet = _weighted_mean(hydrophytic, area, wet)
    wetland_dry = _weighted_mean(wetland_biome, area, dry)
    wetland_wet = _weighted_mean(wetland_biome, area, wet)
    wet_response = np.nanmean(
        [
            hydrophytic_wet - hydrophytic_dry,
            wetland_wet - wetland_dry,
        ]
    )

    highland_elevation = np.clip(
        (sources.elevation_m - sources.biome_config.highland_elevation_start_m)
        / (
            sources.biome_config.highland_elevation_full_m
            - sources.biome_config.highland_elevation_start_m
        ),
        0.0,
        1.0,
    )
    highland_relief = np.clip(
        (sources.local_relief_m - sources.biome_config.highland_relief_start_m)
        / (
            sources.biome_config.highland_relief_full_m
            - sources.biome_config.highland_relief_start_m
        ),
        0.0,
        1.0,
    )
    cold_highland_score = np.clip(
        cold_pressure + 0.45 * highland_elevation + 0.30 * highland_relief,
        0.0,
        1.75,
    )
    cold_values = cold_highland_score[analysis]
    if cold_values.size:
        warm_threshold, cold_threshold = np.percentile(cold_values, (25.0, 75.0))
    else:
        warm_threshold = cold_threshold = 0.0
    warm_low = analysis & (cold_highland_score <= warm_threshold)
    cold_high = analysis & (cold_highland_score >= cold_threshold)
    cold_biome_low = _weighted_mean(cold_biomes, area, warm_low)
    cold_biome_high = _weighted_mean(cold_biomes, area, cold_high)
    cold_response = cold_biome_high - cold_biome_low

    valley = np.asarray(
        sources.channels["support/valley_fraction"][:],
        dtype=np.float32,
    )
    annual_temperature = np.asarray(
        root[DRIVER_PATHS["AnnualMeanTemperatureC"]][:],
        dtype=np.float32,
    )
    annual_precipitation = np.asarray(
        root[DRIVER_PATHS["AnnualPrecipitationMm"]][:],
        dtype=np.float32,
    )
    resource_mean = np.mean(resources[:, 1:5], axis=1)
    npp_deltas: list[float] = []
    resource_deltas: list[float] = []
    for parent_id in np.unique(sources.l0_parent_id[analysis]):
        parent = analysis & (sources.l0_parent_id == parent_id)
        if np.count_nonzero(parent) < 1_000:
            continue
        parent_depth_median = float(np.median(soil_depth[parent]))
        parent_fertility_median = float(np.median(fertility[parent]))
        valley_mask = (
            parent
            & (valley >= 0.45)
            & (soil_depth >= parent_depth_median)
            & (fertility >= parent_fertility_median)
            & (wetland_surface < 0.75)
        )
        if np.count_nonzero(valley_mask) < 100:
            continue
        temperature_target = _weighted_mean(annual_temperature, area, valley_mask)
        precipitation_target = _weighted_mean(annual_precipitation, area, valley_mask)
        slope_mask = (
            parent
            & (valley <= 0.15)
            & (wetland_surface < 0.20)
            & (np.abs(annual_temperature - temperature_target) <= 3.0)
            & (
                np.abs(annual_precipitation - precipitation_target)
                <= max(0.30 * precipitation_target, 100.0)
            )
        )
        if np.count_nonzero(slope_mask) < 100:
            continue
        npp_deltas.append(
            _weighted_mean(annual_npp, area, valley_mask)
            - _weighted_mean(annual_npp, area, slope_mask)
        )
        resource_deltas.append(
            _weighted_mean(resource_mean, area, valley_mask)
            - _weighted_mean(resource_mean, area, slope_mask)
        )
    return {
        "wet_comparison_cell_count": int(np.count_nonzero(wet)),
        "dry_comparison_cell_count": int(np.count_nonzero(dry)),
        "wet_hydrophytic_fraction_mean": hydrophytic_wet,
        "dry_hydrophytic_fraction_mean": hydrophytic_dry,
        "wet_wetland_biome_fraction_mean": wetland_wet,
        "dry_wetland_biome_fraction_mean": wetland_dry,
        "wet_ecology_response": float(wet_response),
        "cold_high_comparison_cell_count": int(np.count_nonzero(cold_high)),
        "warm_low_comparison_cell_count": int(np.count_nonzero(warm_low)),
        "cold_high_cold_biome_fraction_mean": cold_biome_high,
        "warm_low_cold_biome_fraction_mean": cold_biome_low,
        "cold_highland_response": float(cold_response),
        "valley_parent_comparison_count": len(npp_deltas),
        "valley_productivity_response_median_kg_c_m2": _percentile(
            npp_deltas,
            50.0,
        ),
        "valley_resource_response_median": _percentile(resource_deltas, 50.0),
    }


def _validate(
    root: Any,
    config: L3EcologyConfig,
    sources: _EcologySources,
) -> dict[str, Any]:
    finite = True
    bounded = True
    nonnegative = True
    terrestrial_ocean_zero = True
    functional_codes_valid = True
    biome_codes_valid = True
    strict_biome_code_mismatch_count = 0
    quantized_biome_tie_count = 0
    geometry_mismatch_count = 0
    maximum_envelope_aggregation_error = 0.0
    maximum_npp_aggregation_error = 0.0
    maximum_energy_conversion_error = 0.0
    maximum_functional_partition_error = 0.0
    maximum_biome_partition_error = 0.0
    maximum_rooting_excess_m = 0.0
    maximum_cover_excess = 0.0
    land_area = 0.0
    sums = {
        "annual_npp": 0.0,
        "vegetation_cover": 0.0,
        "standing_biomass": 0.0,
        "growing_season": 0.0,
        "transition": 0.0,
        "classification_confidence": 0.0,
    }
    functional_area = np.zeros(FUNCTIONAL_TYPE_COUNT, dtype=np.float64)
    nonvegetated_area = np.zeros(NONVEGETATED_TYPE_COUNT, dtype=np.float64)
    resource_area = np.zeros(RESOURCE_POTENTIAL_COUNT, dtype=np.float64)
    biome_area = np.zeros(BIOME_COUNT, dtype=np.float64)
    normalized_potential = (
        "PotentialVegetationCoverFraction",
        "GrowingSeasonFraction",
        "ProductivitySeasonalityIndex",
        "DroughtAdaptationPressure",
        "ColdAdaptationPressure",
        "HeatAdaptationPressure",
        "WaterloggingAdaptationPressure",
        "SalinityAdaptationPressure",
        "PotentialWoodyAllocationTrait",
        "PotentialResourceConservativeTrait",
        "PotentialFuelContinuityIndex",
        "PotentialBiosphereConfidence",
    )
    for start in range(0, len(sources.cell_id), config.chunk_rows):
        end = min(start + config.chunk_rows, len(sources.cell_id))
        for name, expected in _geometry_chunk(sources, start, end).items():
            try:
                persisted = np.asarray(root[GEOMETRY_PATHS[name]][start:end])
            except (OSError, RuntimeError, TypeError, ValueError):
                geometry_mismatch_count += end - start
                continue
            if persisted.shape != expected.shape:
                geometry_mismatch_count += end - start
            else:
                geometry_mismatch_count += int(np.count_nonzero(persisted != expected))
        area = sources.area_km2[start:end]
        ocean = np.asarray(
            sources.hydrology["surface/physical_ocean_fraction"][start:end],
            dtype=np.float64,
        )
        land = ocean < 0.5
        land_area_chunk = area[land]
        land_area += float(np.sum(land_area_chunk, dtype=np.float64))

        monthly_par = np.asarray(
            root[ENVELOPE_PATHS["MonthlySurfacePARMJm2"]][start:end, :],
            dtype=np.float64,
        )
        monthly_water = np.asarray(
            root[ENVELOPE_PATHS["MonthlyLiquidWaterOpportunity"]][start:end, :],
            dtype=np.float64,
        )
        monthly_thermal = np.asarray(
            root[ENVELOPE_PATHS["MonthlyThermalOpportunity"]][start:end, :],
            dtype=np.float64,
        )
        monthly_primary = np.asarray(
            root[ENVELOPE_PATHS["MonthlyTerrestrialPrimaryEnergyPotentialMJm2"]][start:end, :],
            dtype=np.float64,
        )
        annual_par = np.asarray(
            root[ENVELOPE_PATHS["AnnualSurfacePARMJm2"]][start:end],
            dtype=np.float64,
        )
        annual_primary = np.asarray(
            root[ENVELOPE_PATHS["AnnualTerrestrialPrimaryEnergyPotentialMJm2"]][start:end],
            dtype=np.float64,
        )
        envelope_scalars = {
            name: np.asarray(root[ENVELOPE_PATHS[name]][start:end], dtype=np.float64)
            for name in ANNUAL_ENVELOPE_OUTPUTS
        }
        finite &= bool(
            all(
                np.all(np.isfinite(values))
                for values in (
                    monthly_par,
                    monthly_water,
                    monthly_thermal,
                    monthly_primary,
                    *envelope_scalars.values(),
                )
            )
        )
        nonnegative &= bool(
            all(
                np.all(values >= 0.0)
                for values in (
                    monthly_par,
                    monthly_water,
                    monthly_thermal,
                    monthly_primary,
                    annual_par,
                    annual_primary,
                    envelope_scalars["CarbonSubstrateRelativeToReference"],
                    envelope_scalars["AerobicOxygenRelativeToReference"],
                )
            )
        )
        bounded &= bool(
            np.all((monthly_water >= 0.0) & (monthly_water <= 1.0))
            and np.all((monthly_thermal >= 0.0) & (monthly_thermal <= 1.0))
            and all(
                np.all((envelope_scalars[name] >= 0.0) & (envelope_scalars[name] <= 1.0))
                for name in (
                    "TerrestrialSurfaceSupportFraction",
                    "NutrientSupportIndex",
                    "EnvironmentalStressIndex",
                    "BiosphereEnvelopeConfidence",
                )
            )
        )
        envelope_scale = max(float(np.max(annual_par, initial=0.0)), 1e-12)
        maximum_envelope_aggregation_error = max(
            maximum_envelope_aggregation_error,
            float(np.max(np.abs(np.sum(monthly_par, axis=1) - annual_par), initial=0.0))
            / envelope_scale,
            float(
                np.max(
                    np.abs(np.sum(monthly_primary, axis=1) - annual_primary),
                    initial=0.0,
                )
            )
            / max(float(np.max(annual_primary, initial=0.0)), 1e-12),
        )
        bounded &= bool(
            np.all(
                monthly_primary
                <= monthly_par * sources.envelope_config.photosynthetic_conversion_efficiency + 1e-5
            )
        )

        monthly_npp = np.asarray(
            root[POTENTIAL_PATHS["MonthlyPotentialNPPKgCM2"]][start:end, :],
            dtype=np.float64,
        )
        potential_scalars = {
            name: np.asarray(root[POTENTIAL_PATHS[name]][start:end], dtype=np.float64)
            for name in SCALAR_POTENTIAL_BIOSPHERE_OUTPUTS
        }
        annual_npp = potential_scalars["AnnualPotentialNPPKgCM2"]
        finite &= bool(
            np.all(np.isfinite(monthly_npp))
            and all(np.all(np.isfinite(values)) for values in potential_scalars.values())
        )
        nonnegative &= bool(
            np.all(monthly_npp >= 0.0)
            and all(np.all(values >= 0.0) for values in potential_scalars.values())
        )
        bounded &= all(
            np.all((potential_scalars[name] >= 0.0) & (potential_scalars[name] <= 1.0))
            for name in normalized_potential
        )
        maximum_npp_aggregation_error = max(
            maximum_npp_aggregation_error,
            float(np.max(np.abs(np.sum(monthly_npp, axis=1) - annual_npp), initial=0.0))
            / max(float(np.max(annual_npp, initial=0.0)), 1e-12),
        )
        expected_npp = monthly_primary / sources.potential_config.energy_per_kg_carbon_mj
        maximum_energy_conversion_error = max(
            maximum_energy_conversion_error,
            float(np.max(np.abs(monthly_npp - expected_npp), initial=0.0))
            / max(float(np.max(expected_npp, initial=0.0)), 1e-12),
        )
        regolith = np.asarray(
            sources.surface["soil/RegolithDepthM"][start:end],
            dtype=np.float64,
        )
        roots = potential_scalars["PotentialRootingDepthM"]
        maximum_rooting_excess_m = max(
            maximum_rooting_excess_m,
            float(
                np.max(
                    roots
                    - np.minimum(
                        regolith,
                        sources.potential_config.maximum_rooting_depth_m,
                    ),
                    initial=0.0,
                )
            ),
        )
        physical_bounds = {
            "PotentialCanopyHeightM": sources.potential_config.maximum_canopy_height_m,
            "PotentialLeafAreaIndex": sources.potential_config.maximum_leaf_area_index,
            "PotentialStandingBiomassKgCM2": (
                sources.potential_config.maximum_standing_biomass_kg_c_m2
            ),
        }
        bounded &= all(
            np.all(values <= physical_bounds[name] + 1e-5)
            for name, values in potential_scalars.items()
            if name in physical_bounds
        )

        functional = np.asarray(
            root[FUNCTIONAL_PATHS["FunctionalTypeFractions"]][start:end, :],
            dtype=np.float64,
        )
        nonvegetated = np.asarray(
            root[FUNCTIONAL_PATHS["NonVegetatedFractions"]][start:end, :],
            dtype=np.float64,
        )
        resources = np.asarray(
            root[FUNCTIONAL_PATHS["FunctionalResourcePotentials"]][start:end, :],
            dtype=np.float64,
        )
        functional_confidence = np.asarray(
            root[FUNCTIONAL_PATHS["FunctionalVegetationConfidence"]][start:end],
            dtype=np.float64,
        )
        functional_code = np.asarray(
            root[FUNCTIONAL_PATHS["DominantFunctionalCoverCode"]][start:end],
            dtype=np.uint8,
        )
        finite &= bool(
            all(
                np.all(np.isfinite(values))
                for values in (
                    functional,
                    nonvegetated,
                    resources,
                    functional_confidence,
                )
            )
        )
        bounded &= bool(
            all(
                np.all((values >= 0.0) & (values <= 1.0))
                for values in (
                    functional,
                    nonvegetated,
                    resources,
                    functional_confidence,
                )
            )
        )
        functional_partition = np.sum(functional, axis=1) + np.sum(
            nonvegetated,
            axis=1,
        )
        maximum_functional_partition_error = max(
            maximum_functional_partition_error,
            float(np.max(np.abs(functional_partition[land] - 1.0), initial=0.0)),
        )
        vegetation_fraction = np.sum(functional, axis=1)
        maximum_cover_excess = max(
            maximum_cover_excess,
            float(
                np.max(
                    vegetation_fraction[land]
                    - potential_scalars["PotentialVegetationCoverFraction"][land],
                    initial=0.0,
                )
            ),
        )
        expected_functional_code = np.where(
            vegetation_fraction > np.max(nonvegetated, axis=1),
            np.argmax(functional, axis=1) + 1,
            np.argmax(nonvegetated, axis=1) + 9,
        ).astype(np.uint8)
        expected_functional_code[~land] = 0
        functional_codes_valid &= bool(np.array_equal(functional_code, expected_functional_code))

        biome = np.asarray(
            root[BIOME_PATHS["BiomeFractions"]][start:end, :],
            dtype=np.float64,
        )
        biome_confidence = np.asarray(
            root[BIOME_PATHS["BiomeClassificationConfidence"]][start:end],
            dtype=np.float64,
        )
        biome_margin = np.asarray(
            root[BIOME_PATHS["BiomeDominanceMargin"]][start:end],
            dtype=np.float64,
        )
        biome_transition = np.asarray(
            root[BIOME_PATHS["BiomeTransitionIndex"]][start:end],
            dtype=np.float64,
        )
        dominant = np.asarray(
            root[BIOME_PATHS["DominantBiomeCode"]][start:end],
            dtype=np.uint8,
        )
        secondary = np.asarray(
            root[BIOME_PATHS["SecondaryBiomeCode"]][start:end],
            dtype=np.uint8,
        )
        landscape = np.asarray(
            root[BIOME_PATHS["DominantLandscapeCode"]][start:end],
            dtype=np.uint8,
        )
        finite &= bool(
            all(
                np.all(np.isfinite(values))
                for values in (biome, biome_confidence, biome_margin, biome_transition)
            )
        )
        bounded &= bool(
            all(
                np.all((values >= 0.0) & (values <= 1.0))
                for values in (biome, biome_confidence, biome_margin, biome_transition)
            )
        )
        ice = nonvegetated[:, 2]
        water = nonvegetated[:, 3]
        ground = np.maximum(1.0 - ice - water, 0.0)
        biome_partition = np.sum(biome, axis=1) + ice + water
        maximum_biome_partition_error = max(
            maximum_biome_partition_error,
            float(np.max(np.abs(biome_partition[land] - 1.0), initial=0.0)),
            float(np.max(np.abs(np.sum(biome, axis=1)[land] - ground[land]), initial=0.0)),
        )
        classifiable = land & (ground >= sources.biome_config.minimum_classifiable_ground_fraction)
        strict_dominant = np.argmax(biome, axis=1).astype(np.uint8) + 1
        secondary_source = biome.copy()
        secondary_source[
            np.arange(len(secondary_source)),
            strict_dominant.astype(np.int64) - 1,
        ] = -1.0
        strict_secondary = np.argmax(secondary_source, axis=1).astype(np.uint8) + 1
        strict_dominant[~classifiable] = 0
        strict_secondary[~classifiable] = 0
        strict_mismatch = (
            (dominant != strict_dominant) | (secondary != strict_secondary)
        ) & classifiable
        strict_biome_code_mismatch_count += int(np.count_nonzero(strict_mismatch))

        dominant_range_valid = bool(
            np.all((dominant[classifiable] >= 1) & (dominant[classifiable] <= BIOME_COUNT))
            and np.all(dominant[~classifiable] == 0)
        )
        secondary_range_valid = bool(
            np.all((secondary[classifiable] >= 1) & (secondary[classifiable] <= BIOME_COUNT))
            and np.all(secondary[~classifiable] == 0)
        )
        dominant_selection_valid = True
        secondary_selection_valid = True
        if np.any(classifiable) and dominant_range_valid and secondary_range_valid:
            classifiable_rows = np.flatnonzero(classifiable)
            selected_dominant = biome[
                classifiable_rows,
                dominant[classifiable_rows].astype(np.int64) - 1,
            ]
            secondary_candidates = biome[classifiable_rows].copy()
            secondary_candidates[
                np.arange(len(classifiable_rows)),
                dominant[classifiable_rows].astype(np.int64) - 1,
            ] = -1.0
            selected_secondary = biome[
                classifiable_rows,
                secondary[classifiable_rows].astype(np.int64) - 1,
            ]
            dominant_maximum = np.max(biome[classifiable_rows], axis=1)
            secondary_maximum = np.max(secondary_candidates, axis=1)
            dominant_selection_valid = bool(np.all(selected_dominant >= dominant_maximum - 1e-6))
            secondary_selection_valid = bool(np.all(selected_secondary >= secondary_maximum - 1e-6))
            quantized_biome_tie_count += int(
                np.count_nonzero(
                    strict_mismatch[classifiable_rows]
                    & (selected_dominant >= dominant_maximum - 1e-6)
                    & (selected_secondary >= secondary_maximum - 1e-6)
                )
            )
        elif np.any(classifiable):
            dominant_selection_valid = False
            secondary_selection_valid = False
        expected_landscape = np.where(
            ground > np.maximum(ice, water),
            dominant,
            np.where(water >= ice, 14, 15),
        ).astype(np.uint8)
        expected_landscape[~land] = 0
        biome_codes_valid &= bool(
            dominant_range_valid
            and secondary_range_valid
            and dominant_selection_valid
            and secondary_selection_valid
            and np.array_equal(landscape, expected_landscape)
            and np.all(dominant[classifiable] != secondary[classifiable])
        )

        if np.any(~land):
            terrestrial_ocean_zero &= bool(
                np.all(monthly_water[~land] == 0.0)
                and np.all(monthly_primary[~land] == 0.0)
                and all(
                    np.all(envelope_scalars[name][~land] == 0.0)
                    for name in (
                        "AnnualTerrestrialPrimaryEnergyPotentialMJm2",
                        "TerrestrialSurfaceSupportFraction",
                        "NutrientSupportIndex",
                        "EnvironmentalStressIndex",
                        "BiosphereEnvelopeConfidence",
                    )
                )
                and np.all(monthly_npp[~land] == 0.0)
                and all(np.all(values[~land] == 0.0) for values in potential_scalars.values())
                and np.all(functional[~land] == 0.0)
                and np.all(nonvegetated[~land] == 0.0)
                and np.all(resources[~land] == 0.0)
                and np.all(functional_confidence[~land] == 0.0)
                and np.all(functional_code[~land] == 0)
                and np.all(biome[~land] == 0.0)
                and np.all(biome_confidence[~land] == 0.0)
                and np.all(biome_margin[~land] == 0.0)
                and np.all(biome_transition[~land] == 0.0)
                and np.all(dominant[~land] == 0)
                and np.all(secondary[~land] == 0)
                and np.all(landscape[~land] == 0)
            )

        for index in range(FUNCTIONAL_TYPE_COUNT):
            functional_area[index] += float(
                np.sum(functional[land, index] * land_area_chunk, dtype=np.float64)
            )
        for index in range(NONVEGETATED_TYPE_COUNT):
            nonvegetated_area[index] += float(
                np.sum(nonvegetated[land, index] * land_area_chunk, dtype=np.float64)
            )
        for index in range(RESOURCE_POTENTIAL_COUNT):
            resource_area[index] += float(
                np.sum(resources[land, index] * land_area_chunk, dtype=np.float64)
            )
        for index in range(BIOME_COUNT):
            biome_area[index] += float(
                np.sum(biome[land, index] * land_area_chunk, dtype=np.float64)
            )
        values = {
            "annual_npp": annual_npp,
            "vegetation_cover": potential_scalars["PotentialVegetationCoverFraction"],
            "standing_biomass": potential_scalars["PotentialStandingBiomassKgCM2"],
            "growing_season": potential_scalars["GrowingSeasonFraction"],
            "transition": biome_transition,
            "classification_confidence": biome_confidence,
        }
        for name, value in values.items():
            sums[name] += float(np.sum(value[land] * land_area_chunk, dtype=np.float64))

    parent_metrics = _represented_parent_metrics(root, sources, config)
    annual_npp_all = np.asarray(
        root[POTENTIAL_PATHS["AnnualPotentialNPPKgCM2"]][:],
        dtype=np.float32,
    )
    transition_all = np.asarray(
        root[BIOME_PATHS["BiomeTransitionIndex"]][:],
        dtype=np.float32,
    )
    order = sources.spatial_order
    ocean_all = np.asarray(
        sources.hydrology["surface/physical_ocean_fraction"][:],
        dtype=np.float32,
    )
    adjacency_mask = (sources.inside_display[order] & (ocean_all[order] < 0.5)).reshape(
        sources.height, sources.width
    )
    parent_grid = sources.l0_parent_id[order].reshape(sources.height, sources.width)
    npp_boundary = _boundary_metric(
        annual_npp_all[order].reshape(sources.height, sources.width),
        parent_grid,
        adjacency_mask,
    )
    transition_boundary = _boundary_metric(
        transition_all[order].reshape(sources.height, sources.width),
        parent_grid,
        adjacency_mask,
    )
    maximum_boundary_ratio = max(
        float(npp_boundary["boundary_to_interior_p95_ratio"]),
        float(transition_boundary["boundary_to_interior_p95_ratio"]),
    )
    maximum_boundary_absolute_difference = max(
        float(npp_boundary["boundary_difference_p95"]),
        float(transition_boundary["boundary_difference_p95"]),
    )
    motif = _parent_motif_correlation(
        annual_npp_all,
        sources,
        sources.inside_display & (ocean_all < 0.5),
    )
    relational = _relational_metrics(root, sources)
    metrics: dict[str, Any] = {
        "model": ECOLOGY_MODEL_VERSION,
        "target_id": sources.target_id,
        "cell_count": len(sources.cell_id),
        "display_cell_count": int(np.count_nonzero(sources.inside_display)),
        "core_cell_count": int(np.count_nonzero(sources.inside_core)),
        "geometry_mismatch_count": geometry_mismatch_count,
        "land_area_km2": land_area,
        "maximum_envelope_annual_aggregation_relative_error": (maximum_envelope_aggregation_error),
        "maximum_npp_annual_aggregation_relative_error": maximum_npp_aggregation_error,
        "maximum_energy_conversion_relative_error": maximum_energy_conversion_error,
        "maximum_functional_partition_absolute_error": (maximum_functional_partition_error),
        "maximum_biome_partition_absolute_error": maximum_biome_partition_error,
        "maximum_rooting_depth_excess_m": maximum_rooting_excess_m,
        "maximum_functional_cover_excess": maximum_cover_excess,
        "strict_biome_code_mismatch_count": strict_biome_code_mismatch_count,
        "quantized_biome_tie_count": quantized_biome_tie_count,
        "land_mean_annual_npp_kg_c_m2": sums["annual_npp"] / max(land_area, 1e-30),
        "land_mean_potential_vegetation_cover_fraction": (
            sums["vegetation_cover"] / max(land_area, 1e-30)
        ),
        "land_mean_standing_biomass_kg_c_m2": (sums["standing_biomass"] / max(land_area, 1e-30)),
        "land_mean_growing_season_fraction": (sums["growing_season"] / max(land_area, 1e-30)),
        "land_mean_biome_transition_index": sums["transition"] / max(land_area, 1e-30),
        "land_mean_biome_classification_confidence": (
            sums["classification_confidence"] / max(land_area, 1e-30)
        ),
        "land_mean_functional_type_fractions": {
            str(item["class_id"]): float(functional_area[cast(int, item["index"])] / land_area)
            for item in FUNCTIONAL_TYPES
        },
        "land_mean_nonvegetated_fractions": {
            str(item["class_id"]): float(nonvegetated_area[cast(int, item["index"])] / land_area)
            for item in NONVEGETATED_TYPES
        },
        "land_mean_resource_potentials": {
            str(item["potential_id"]): float(resource_area[cast(int, item["index"])] / land_area)
            for item in RESOURCE_POTENTIALS
        },
        "land_mean_biome_fractions": {
            str(item["class_id"]): float(biome_area[cast(int, item["index"])] / land_area)
            for item in BIOMES
        },
        **parent_metrics,
        "npp_parent_boundary": npp_boundary,
        "transition_parent_boundary": transition_boundary,
        "maximum_parent_boundary_p95_ratio": maximum_boundary_ratio,
        "maximum_parent_boundary_absolute_difference_p95": (maximum_boundary_absolute_difference),
        **motif,
        **relational,
    }
    gates = {
        "geometry_matches_sources": geometry_mismatch_count == 0,
        "outputs_finite": finite,
        "outputs_nonnegative": nonnegative,
        "bounded_fraction_outputs": bounded,
        "terrestrial_outputs_zero_over_ocean": terrestrial_ocean_zero,
        "envelope_annual_aggregation_valid": (
            maximum_envelope_aggregation_error
            <= sources.envelope_config.maximum_annual_aggregation_relative_error
        ),
        "npp_annual_aggregation_valid": (
            maximum_npp_aggregation_error
            <= sources.potential_config.maximum_annual_aggregation_relative_error
        ),
        "energy_conversion_valid": (
            maximum_energy_conversion_error
            <= sources.potential_config.maximum_energy_conversion_relative_error
        ),
        "functional_partition_valid": (
            maximum_functional_partition_error
            <= sources.functional_config.maximum_partition_absolute_error
        ),
        "functional_cover_within_potential": maximum_cover_excess <= 1e-6,
        "functional_codes_reconstructed": functional_codes_valid,
        "biome_partition_valid": (
            maximum_biome_partition_error <= sources.biome_config.maximum_partition_absolute_error
        ),
        "biome_codes_reconstructed": biome_codes_valid,
        "rooting_depth_within_regolith": maximum_rooting_excess_m <= 1e-6,
        "represented_parent_set_nonempty": int(parent_metrics["represented_parent_count"]) > 0,
        "parent_npp_divergence_bounded": (
            float(parent_metrics["parent_npp_relative_difference_p95"])
            <= config.maximum_parent_npp_relative_difference_p95
        ),
        "parent_functional_divergence_bounded": (
            float(parent_metrics["parent_functional_fraction_l1_difference_p95"])
            <= config.maximum_parent_functional_l1_difference_p95
        ),
        "parent_biome_divergence_bounded": (
            float(parent_metrics["parent_biome_fraction_l1_difference_p95"])
            <= config.maximum_parent_biome_l1_difference_p95
        ),
        "parent_boundary_relative_discontinuity_bounded": (
            maximum_boundary_ratio <= config.maximum_parent_boundary_p95_ratio
        ),
        "parent_boundary_absolute_discontinuity_bounded": (
            maximum_boundary_absolute_difference
            <= config.maximum_parent_boundary_absolute_difference_p95
        ),
        "repeated_parent_motif_bounded": (
            float(motif["repeated_parent_motif_correlation_p95"])
            <= config.maximum_repeated_parent_motif_correlation_p95
        ),
        "wet_soils_favor_wet_ecology": (
            math.isfinite(float(relational["wet_ecology_response"]))
            and float(relational["wet_ecology_response"]) >= config.minimum_wet_ecology_response
        ),
        "cold_highlands_favor_cold_biomes": (
            math.isfinite(float(relational["cold_highland_response"]))
            and float(relational["cold_highland_response"]) >= config.minimum_cold_highland_response
        ),
        "valley_comparison_available": int(relational["valley_parent_comparison_count"]) > 0,
        "fertile_valleys_improve_productivity": (
            float(relational["valley_productivity_response_median_kg_c_m2"])
            >= config.minimum_valley_productivity_response
        ),
        "fertile_valleys_improve_resource_potential": (
            float(relational["valley_resource_response_median"])
            >= config.minimum_valley_resource_response
        ),
    }
    return {**metrics, "gates": gates, "passed": bool(all(gates.values()))}


def _catalogs() -> dict[str, Any]:
    return {
        "functional_types": [
            {
                "axis_index": int(item["index"]),
                "code": int(item["code"]),
                "class_id": str(item["class_id"]),
                "label": str(item["label"]),
                "group": str(item["group"]),
                "color": list(cast(tuple[int, int, int], item["color"])),
            }
            for item in FUNCTIONAL_TYPES
        ],
        "nonvegetated_types": [
            {
                "axis_index": int(item["index"]),
                "code": int(item["code"]),
                "class_id": str(item["class_id"]),
                "label": str(item["label"]),
                "group": str(item["group"]),
                "color": list(cast(tuple[int, int, int], item["color"])),
            }
            for item in NONVEGETATED_TYPES
        ],
        "resource_potentials": [
            {
                "axis_index": int(item["index"]),
                "potential_id": str(item["potential_id"]),
                "label": str(item["label"]),
            }
            for item in RESOURCE_POTENTIALS
        ],
        "biomes": [
            {
                "axis_index": int(item["index"]),
                "code": int(item["code"]),
                "class_id": str(item["class_id"]),
                "label": str(item["label"]),
                "group": str(item["group"]),
                "color": list(cast(tuple[int, int, int], item["color"])),
            }
            for item in BIOMES
        ],
        "landscapes": [
            {
                "code": int(item["code"]),
                "class_id": str(item["class_id"]),
                "label": str(item["label"]),
                "group": str(item["group"]),
                "color": list(cast(tuple[int, int, int], item["color"])),
            }
            for item in LANDSCAPES
        ],
        "semantics": {
            "fractions": "physical fraction of full L3 cell area",
            "resource_potentials": "bounded suitability, not actual land use",
            "dominant_codes": "derived query helpers reproducible from canonical fractions",
        },
    }


def _render(
    root: Any,
    sources: _EcologySources,
    validation: Mapping[str, Any],
    path: Path,
) -> None:
    order = sources.spatial_order
    elevation = sources.elevation_m[order].reshape(sources.height, sources.width)
    terrain = _hillshaded_terrain(elevation, sources.actual_cell_size_m)
    fractions = np.asarray(
        root[BIOME_PATHS["BiomeFractions"]][:],
        dtype=np.float32,
    )[order].reshape(sources.height, sources.width, BIOME_COUNT)
    biome_colors = np.asarray(
        [cast(tuple[int, int, int], item["color"]) for item in BIOMES],
        dtype=np.float32,
    )
    ground = np.sum(fractions, axis=2)
    conditional = np.divide(
        fractions,
        ground[..., None],
        out=np.zeros_like(fractions),
        where=ground[..., None] > 1e-6,
    )
    biome_rgb = np.clip(conditional @ biome_colors, 0.0, 255.0)
    colors = terrain * 0.30 + biome_rgb * 0.70
    nonvegetated = np.asarray(
        root[FUNCTIONAL_PATHS["NonVegetatedFractions"]][:],
        dtype=np.float32,
    )[order].reshape(sources.height, sources.width, NONVEGETATED_TYPE_COUNT)
    ice = np.clip(nonvegetated[..., 2], 0.0, 1.0)
    lake = np.asarray(
        sources.hydrology["surface/lake_fraction"][:],
        dtype=np.float32,
    )[order].reshape(sources.height, sources.width)
    wetland = np.asarray(
        sources.hydrology["surface/wetland_fraction"][:],
        dtype=np.float32,
    )[order].reshape(sources.height, sources.width)
    wetland_alpha = np.clip(wetland * 0.35, 0.0, 0.35)
    colors = (
        colors * (1.0 - wetland_alpha[..., None])
        + np.asarray((44.0, 126.0, 120.0)) * wetland_alpha[..., None]
    )
    colors = colors * (1.0 - lake[..., None]) + np.asarray((48.0, 112.0, 153.0)) * lake[..., None]
    colors = colors * (1.0 - ice[..., None]) + np.asarray((230.0, 239.0, 241.0)) * ice[..., None]
    channel = np.asarray(
        sources.channels["support/centerline_seed"][:],
        dtype=bool,
    )[order].reshape(sources.height, sources.width)
    colors[channel & (lake < 0.5)] = np.asarray((24.0, 92.0, 159.0))
    display = sources.inside_display[order].reshape(sources.height, sources.width)
    row_slice, column_slice = _rectangular_mask_slices(display)
    colors = colors[row_slice, column_slice]
    map_image = Image.fromarray(np.clip(colors, 0.0, 255.0).astype(np.uint8), mode="RGB")

    title_height = 54
    footer_height = 76
    legend_width = 470
    canvas = Image.new(
        "RGB",
        (map_image.width + legend_width, map_image.height + title_height + footer_height),
        (240, 241, 237),
    )
    canvas.paste(map_image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    title_font = _diagnostic_font(22)
    label_font = _diagnostic_font(17)
    small_font = _diagnostic_font(14)
    draw.text(
        (18, 13),
        "L3 potential ecology and derived biome mixtures",
        fill=(25, 30, 27),
        font=title_font,
    )
    draw.line(
        (map_image.width, 0, map_image.width, canvas.height),
        fill=(178, 181, 174),
        width=1,
    )
    x = map_image.width + 22
    y = 18
    draw.text((x, y), "Biome mixture", fill=(25, 30, 27), font=title_font)
    y += 48
    for item in BIOMES:
        color = cast(tuple[int, int, int], item["color"])
        draw.rectangle(
            (x, y, x + 30, y + 16),
            fill=color,
            outline=(55, 59, 56),
        )
        fraction = float(validation["land_mean_biome_fractions"][str(item["class_id"])])
        draw.text(
            (x + 42, y - 1),
            f"{item['label']}  {fraction * 100:.1f}%",
            fill=(35, 39, 36),
            font=small_font,
        )
        y += 25
    y += 8
    for color, label in (
        ((48, 112, 153), "inland open water"),
        ((230, 239, 241), "persistent ice"),
        ((24, 92, 159), "physical channel vector"),
    ):
        draw.rectangle((x, y, x + 30, y + 16), fill=color, outline=(55, 59, 56))
        draw.text((x + 42, y - 1), label, fill=(35, 39, 36), font=small_font)
        y += 25
    y += 12
    draw.text((x, y), "Regional summary", fill=(25, 30, 27), font=label_font)
    y += 30
    summary_lines = (
        (f"mean NPP {validation['land_mean_annual_npp_kg_c_m2']:.2f} kg C/m2/yr"),
        (
            "potential cover "
            f"{validation['land_mean_potential_vegetation_cover_fraction'] * 100:.1f}%"
        ),
        (f"standing biomass {validation['land_mean_standing_biomass_kg_c_m2']:.1f} kg C/m2"),
        (f"wet ecology response {validation['wet_ecology_response']:+.3f}"),
        (f"cold/highland response {validation['cold_highland_response']:+.3f}"),
        (f"parent biome delta p95 {validation['parent_biome_fraction_l1_difference_p95']:.2f}"),
        "potential equilibrium, no succession",
        "no vegetation-to-water feedback in V0",
    )
    for line in summary_lines:
        draw.text((x, y), line, fill=(50, 54, 51), font=small_font)
        y += 23
    _draw_scale(
        draw,
        map_image.width,
        map_image.height,
        title_height,
        sources.actual_cell_size_m,
    )
    canvas.save(path, optimize=True)


def _existing_result(
    config: L3EcologyConfig,
    run_fingerprint: str,
) -> L3EcologyResult | None:
    manifest_path = config.output_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _load_json(manifest_path)
    if manifest.get("run_fingerprint") != run_fingerprint or not manifest.get("validation_passed"):
        return None
    _verify_manifest_outputs(config.output_dir, manifest)
    summary = manifest["summary"]
    return L3EcologyResult(
        output_dir=config.output_dir,
        manifest_path=manifest_path,
        validation_path=config.output_dir / "validation.json",
        zarr_path=config.output_dir / "ecology.zarr",
        preview_path=config.output_dir / "ecology.png",
        target_id=str(manifest["target_id"]),
        display_cell_count=int(summary["display_cell_count"]),
        chunk_count=int(summary["chunk_count"]),
        validation_passed=True,
    )


def generate_l3_ecology(config: L3EcologyConfig) -> L3EcologyResult:
    """Replay the calibrated ecology chain over accepted L3 physical state."""

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
    preview_path = partial / "ecology.png"
    _render(root, sources, validation, preview_path)
    catalogs_path = partial / "catalogs.json"
    _write_json_durable(catalogs_path, _catalogs())
    zarr_path = partial / "ecology.zarr"
    root.attrs["status"] = "validating"
    zarr.consolidate_metadata(str(zarr_path))
    _fsync_paths([zarr_path / ".zattrs", zarr_path / ".zmetadata"])
    observed_peak = _observed_peak_rss_bytes()
    estimated_peak = int(config.chunk_rows * 2_400 + len(sources.cell_id) * 190)
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
        "format_version": ECOLOGY_FORMAT_VERSION,
        "model_version": ECOLOGY_MODEL_VERSION,
        "status": "complete" if validation["passed"] else "validation_failed",
        "target_id": sources.target_id,
        "run_fingerprint": run_fingerprint,
        "summary": {
            "cell_count": len(sources.cell_id),
            "display_cell_count": validation["display_cell_count"],
            "core_cell_count": validation["core_cell_count"],
            "chunk_count": chunk_count,
            "land_mean_annual_npp_kg_c_m2": (validation["land_mean_annual_npp_kg_c_m2"]),
            "land_mean_potential_vegetation_cover_fraction": (
                validation["land_mean_potential_vegetation_cover_fraction"]
            ),
            "land_mean_standing_biomass_kg_c_m2": (
                validation["land_mean_standing_biomass_kg_c_m2"]
            ),
            "land_mean_biome_fractions": validation["land_mean_biome_fractions"],
        },
        "model": {
            "causal_order": (
                "L3 resource envelope -> potential biosphere -> functional vegetation "
                "partition -> familiar biome mixture"
            ),
            "climate": (
                "source monthly insolation and parent climate with the exact persisted "
                "L3 surface-material temperature adjustment; no regional atmosphere solve"
            ),
            "soil_water": (
                "accepted L3 monthly liquid input and saturation from surface materials"
            ),
            "atmosphere": (
                "source dry composition with hydrostatic CO2 and oxygen support at L3 elevation"
            ),
            "coarse_ecology": (
                "comparison and supervised-learning context, never interpolated as final labels"
            ),
            "fractions": "canonical physical mixtures; query codes are derived",
            "vegetation_feedback": "not applied",
            "succession_and_disturbance": "not simulated",
            "actual_land_use": "not simulated; resource outputs are potentials",
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
            "surface_materials_dir": str(config.surface_materials_dir),
            "handoff_dir": str(sources.handoff_dir),
            "world_config": str(sources.world_config_path),
            **fingerprint_components,
        },
        "outputs": {
            "ecology_zarr": {
                "path": "ecology.zarr",
                "sha256_tree": _tree_checksum(zarr_path),
            },
            "ecology_preview": {
                "path": "ecology.png",
                "sha256": _file_checksum(preview_path),
                "projection": "native cubed-sphere face raster diagnostic",
                "legend": "fractional familiar-biome palette, water, ice, and channels",
                "scale": "labelled kilometre scale from L3 area-equivalent cell width",
            },
            "catalogs": {
                "path": "catalogs.json",
                "sha256": _file_checksum(catalogs_path),
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
            f"L3 ecology failed validation; diagnostics retained in {partial}: {validation}"
        )
    _replace_directory(partial, config.output_dir)
    return L3EcologyResult(
        output_dir=config.output_dir,
        manifest_path=config.output_dir / "manifest.json",
        validation_path=config.output_dir / "validation.json",
        zarr_path=config.output_dir / "ecology.zarr",
        preview_path=config.output_dir / "ecology.png",
        target_id=sources.target_id,
        display_cell_count=int(validation["display_cell_count"]),
        chunk_count=chunk_count,
        validation_passed=True,
    )


__all__ = [
    "ECOLOGY_FORMAT_VERSION",
    "ECOLOGY_MODEL_VERSION",
    "L3EcologyConfig",
    "L3EcologyResult",
    "generate_l3_ecology",
]
