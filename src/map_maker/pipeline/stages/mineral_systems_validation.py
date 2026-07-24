"""Directional and catalog validation for causal mineral systems V0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage
from .mineral_systems import SYSTEM_NAMES

if TYPE_CHECKING:
    from ..execution import PipelineContext


@dataclass(frozen=True)
class MineralSystemsValidationConfig:
    minimum_family_peak_potential: float = 0.25
    minimum_directional_enrichment_ratio: float = 1.10
    minimum_directional_enrichment_difference: float = 0.02
    minimum_broad_cratonic_enrichment_ratio: float = 1.02
    minimum_broad_cratonic_enrichment_difference: float = 0.008
    minimum_candidates_per_family: int = 1
    driver_high_quantile: float = 0.80
    driver_low_quantile: float = 0.20
    maximum_candidate_local_maximum_error: float = 1e-7

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "MineralSystemsValidationConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown mineral validation controls: {', '.join(sorted(unknown))}")
        values: dict[str, int | float] = {}
        for name, field in cls.__dataclass_fields__.items():
            raw = mapping.get(name, field.default)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{name} must be numeric")
            values[name] = int(raw) if name == "minimum_candidates_per_family" else float(raw)
        config = cls(**values)  # type: ignore[arg-type]
        for name in (
            "minimum_family_peak_potential",
            "driver_high_quantile",
            "driver_low_quantile",
        ):
            value = getattr(config, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if config.driver_low_quantile >= config.driver_high_quantile:
            raise ValueError("driver_low_quantile must be below driver_high_quantile")
        if config.minimum_directional_enrichment_ratio < 1.0:
            raise ValueError("minimum_directional_enrichment_ratio must be at least 1")
        if config.minimum_broad_cratonic_enrichment_ratio < 1.0:
            raise ValueError("minimum_broad_cratonic_enrichment_ratio must be at least 1")
        for name in (
            "minimum_directional_enrichment_difference",
            "minimum_broad_cratonic_enrichment_difference",
        ):
            if not 0.0 <= getattr(config, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 1 <= config.minimum_candidates_per_family <= 10_000:
            raise ValueError("minimum_candidates_per_family must be in [1, 10000]")
        if not 0.0 <= config.maximum_candidate_local_maximum_error <= 1e-3:
            raise ValueError("maximum_candidate_local_maximum_error must be in [0, 1e-3]")
        return config


def _artifact_array(result: StageResult, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _artifact_table(result: StageResult, name: str) -> pa.Table:
    record = result.artifact_records.get(name)
    if record is None or record.value is None or not isinstance(record.value, pa.Table):
        raise KeyError(f"Missing dependency table '{name}'")
    return cast(pa.Table, record.value)


def _artifact_mapping(result: StageResult, name: str) -> Mapping[str, object]:
    record = result.artifact_records.get(name)
    if record is None or record.value is None or not isinstance(record.value, Mapping):
        raise KeyError(f"Missing dependency metadata '{name}'")
    return cast(Mapping[str, object], record.value)


def _ramp(value: np.ndarray, lower: float, upper: float) -> np.ndarray:
    bounded = np.clip((np.asarray(value, dtype=np.float64) - lower) / (upper - lower), 0.0, 1.0)
    return bounded * bounded * (3.0 - 2.0 * bounded)


def _class_is(province_class: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
    return np.isin(province_class, classes).astype(np.float64)


def _drivers(deps: Mapping[str, StageResult]) -> tuple[np.ndarray, np.ndarray]:
    geology = deps["geology"]
    tectonics = deps["tectonics"]
    world_age = deps["world_age"]
    elevation = deps["elevation"]
    sea_level = deps["sea_level"]
    climate = deps["climate"]
    hydrology = deps["hydrology"]
    materials = deps["surface_materials"]
    biosphere = deps["potential_biosphere"]
    ocean = _artifact_array(sea_level, "SurfaceOceanMask") >= 0.5
    land = ~ocean
    province_class = _artifact_array(geology, "GeologicalProvinceClass")
    crust_age = _artifact_array(geology, "CrustAgeGa")
    convergence = _artifact_array(tectonics, "BoundaryConvergence")
    divergence = _artifact_array(tectonics, "BoundaryDivergence")
    shear = _artifact_array(tectonics, "BoundaryShear")
    subduction = _artifact_array(tectonics, "BoundarySubduction")
    hotspot = _artifact_array(tectonics, "HotspotMap")
    compression = _artifact_array(world_age, "TectonicCompression")
    extension = _artifact_array(world_age, "TectonicExtension")
    surface_elevation = _artifact_array(sea_level, "SurfaceElevationM")
    relief = _artifact_array(elevation, "TerrainReliefM")
    high_elevation = _ramp(surface_elevation, 300.0, 2_800.0)
    high_relief = _ramp(relief, 250.0, 1_800.0)
    accommodation = _artifact_array(geology, "SedimentAccommodation")
    subsidence = _artifact_array(world_age, "SubsidenceRate")
    stiffness = _artifact_array(world_age, "LithosphereStiffness")
    temperature = _artifact_array(climate, "AnnualMeanTemperatureC")
    precipitation = _artifact_array(climate, "AnnualPrecipitationMm")
    aridity = _artifact_array(climate, "AnnualAridityIndex")
    river = _artifact_array(hydrology, "RiverCorridor")
    floodplain = _artifact_array(hydrology, "FloodplainPotential")
    alluvium = _artifact_array(materials, "AlluviumFraction")
    residual = _artifact_array(materials, "ResidualRegolithFraction")
    bedrock = _artifact_array(materials, "BedrockSurfaceFraction")
    salinity = _artifact_array(materials, "SoilSalinityIndex")
    lake = _artifact_array(materials, "EffectiveLakeFraction")
    wetland = _artifact_array(materials, "EffectiveWetlandFraction")
    hydric = _artifact_array(materials, "HydricSoilFraction")
    npp = _artifact_array(biosphere, "AnnualPotentialNPPKgCM2")
    biomass = _artifact_array(biosphere, "PotentialStandingBiomassKgCM2")

    arc = np.maximum.reduce(
        [
            _class_is(province_class, (6, 10)),
            _ramp(subduction, 0.01, 0.20),
            _ramp(convergence * hotspot, 0.025, 0.30),
            0.65 * np.clip(compression, 0.0, 1.0),
        ]
    )
    orogen = np.maximum.reduce(
        [
            _class_is(province_class, (4,)),
            np.clip(convergence, 0.0, 1.0),
            np.clip(shear, 0.0, 1.0),
            np.clip(compression, 0.0, 1.0),
            0.55 * high_elevation * high_relief,
        ]
    )
    mafic = np.maximum.reduce(
        [
            _class_is(province_class, (9, 10, 11)),
            _ramp(hotspot, 0.08, 0.35),
            _ramp(divergence, 0.02, 0.55),
            _ramp(extension, 0.05, 0.70),
            0.55 * _ramp(subduction, 0.01, 0.20),
        ]
    )
    basin = np.maximum.reduce(
        [
            _class_is(province_class, (3, 7, 8)),
            np.clip(accommodation, 0.0, 1.0),
            np.clip(subsidence, 0.0, 1.0),
            0.55 * np.clip(extension, 0.0, 1.0),
        ]
    )
    craton = np.maximum(
        _class_is(province_class, (1,)),
        np.maximum(
            _class_is(province_class, (2,))
            * (0.35 + 0.65 * _ramp(crust_age, 1.4, 3.2))
            * np.clip(stiffness, 0.0, 1.0),
            _ramp(crust_age, 1.4, 3.2) * np.clip(stiffness, 0.0, 1.0),
        ),
    )
    wet_climate = _ramp(precipitation, 250.0, 1_800.0) * (1.0 - 0.72 * np.clip(aridity, 0.0, 1.0))
    warm = _ramp(temperature, -2.0, 12.0) * (1.0 - _ramp(temperature, 30.0, 48.0))
    weathering = np.maximum(
        _ramp(warm * wet_climate, 0.10, 0.58),
        0.65 * _ramp(residual, 0.15, 0.75),
    )
    placer = np.maximum.reduce([river, floodplain, alluvium, high_relief * river])
    dry = 0.72 * _ramp(aridity, 0.42, 0.92) + 0.28 * (1.0 - _ramp(precipitation, 120.0, 700.0))
    evaporite = np.maximum.reduce([dry * basin, dry * np.maximum(lake, 0.05), salinity * basin])
    productive = 0.55 * _ramp(npp, 0.03, 0.75) + 0.45 * _ramp(biomass, 0.5, 12.0)
    coal = np.maximum.reduce([wetland, hydric, productive * accommodation])
    drivers = np.stack(
        [
            land * arc,
            land * orogen,
            land * mafic,
            ocean * mafic,
            land * basin,
            land * craton * (0.35 + 0.65 * np.clip(bedrock, 0.0, 1.0)),
            land * weathering,
            land * placer,
            land * evaporite,
            land * coal,
        ]
    )
    domains = np.stack(
        [
            land,
            land,
            land,
            ocean,
            land,
            land,
            land,
            land,
            land,
            land,
        ]
    )
    return drivers, domains


def _weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    selected_weights = weights[mask]
    if selected_weights.size == 0 or float(np.sum(selected_weights)) <= 0.0:
        return 0.0
    return float(np.sum(values[mask] * selected_weights) / np.sum(selected_weights))


@stage(
    "mineral_systems_validation",
    inputs=(
        "mineral_systems",
        "geology",
        "tectonics",
        "world_age",
        "elevation",
        "sea_level",
        "climate",
        "hydrology",
        "surface_materials",
        "potential_biosphere",
    ),
    outputs=(
        "MineralSystemsValidationCatalog",
        "MineralSystemsValidationMetadata",
    ),
    version="v5",
)
def mineral_systems_validation_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = MineralSystemsValidationConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("mineral validation requires topology: cubed_sphere")
    mineral = deps["mineral_systems"]
    mineral_metadata = _artifact_mapping(mineral, "MineralSystemsMetadata")
    if mineral_metadata.get("causal_mineral_systems_ready_for_validation") != 1:
        raise RuntimeError("mineral systems are not ready for validation")

    potential = np.asarray(_artifact_array(mineral, "MineralSystemPotential"), dtype=np.float64)
    drivers, domains = _drivers(deps)
    areas = np.asarray(context.topology.cell_areas, dtype=np.float64)
    deposit_catalog = _artifact_table(mineral, "MajorDepositCandidateCatalog")
    system_catalog = _artifact_table(mineral, "MineralSystemCatalog")
    candidate_codes = np.asarray(deposit_catalog["system_code"], dtype=np.int64)
    rows: list[dict[str, object]] = []
    hard_failures: list[str] = []
    for system_index, system_name in enumerate(SYSTEM_NAMES):
        driver = np.asarray(drivers[system_index], dtype=np.float64)
        domain = np.asarray(domains[system_index], dtype=bool)
        domain_values = driver[domain]
        if domain_values.size == 0:
            low_threshold = high_threshold = 0.0
        else:
            low_threshold = float(np.quantile(domain_values, config.driver_low_quantile))
            positive = domain_values[domain_values > low_threshold + 1e-12]
            high_threshold = float(
                np.quantile(
                    positive if positive.size else domain_values,
                    config.driver_high_quantile,
                )
            )
        high = domain & (driver >= high_threshold) & (driver > 0.0)
        low = domain & (driver <= low_threshold + 1e-12)
        high_mean = _weighted_mean(potential[system_index], areas, high)
        low_mean = _weighted_mean(potential[system_index], areas, low)
        difference = high_mean - low_mean
        ratio = (high_mean + 1e-9) / (low_mean + 1e-9)
        peak = float(np.max(potential[system_index]))
        candidate_count = int(np.count_nonzero(candidate_codes == system_index + 1))
        system_count = int(
            np.count_nonzero(
                np.asarray(system_catalog["system_code"], dtype=np.int64) == system_index + 1
            )
        )
        required_ratio = (
            config.minimum_broad_cratonic_enrichment_ratio
            if system_index == 5
            else config.minimum_directional_enrichment_ratio
        )
        required_difference = (
            config.minimum_broad_cratonic_enrichment_difference
            if system_index == 5
            else config.minimum_directional_enrichment_difference
        )
        passed = (
            peak >= config.minimum_family_peak_potential
            and candidate_count >= config.minimum_candidates_per_family
            and system_count > 0
            and difference >= required_difference
            and ratio >= required_ratio
            and np.count_nonzero(high) > 0
            and np.count_nonzero(low) > 0
        )
        if not passed:
            hard_failures.append(f"{system_name}_directional_or_noncollapse")
        rows.append(
            {
                "system_code": system_index + 1,
                "system_name": system_name,
                "peak_potential": peak,
                "driver_low_threshold": low_threshold,
                "driver_high_threshold": high_threshold,
                "low_driver_area_weighted_mean_potential": low_mean,
                "high_driver_area_weighted_mean_potential": high_mean,
                "directional_enrichment_difference": difference,
                "directional_enrichment_ratio": ratio,
                "required_directional_enrichment_difference": required_difference,
                "required_directional_enrichment_ratio": required_ratio,
                "high_driver_cell_count": int(np.count_nonzero(high)),
                "low_driver_cell_count": int(np.count_nonzero(low)),
                "mineral_system_count": system_count,
                "deposit_candidate_count": candidate_count,
                "passed": passed,
            }
        )

    total = context.topology.cell_count
    flat_potential = potential.reshape(len(SYSTEM_NAMES), total)
    neighbors = context.topology.neighbor_indices.reshape(total, 4)
    candidate_hosts = np.asarray(deposit_catalog["host_cell_id"], dtype=np.int64)
    candidate_systems = candidate_codes - 1
    candidate_values = np.asarray(deposit_catalog["system_potential"], dtype=np.float64)
    expected_values = flat_potential[candidate_systems, candidate_hosts]
    candidate_value_error = float(np.max(np.abs(candidate_values - expected_values), initial=0.0))
    neighbor_maximum = np.max(
        flat_potential[candidate_systems[:, None], neighbors[candidate_hosts]],
        axis=1,
    )
    local_maximum_error = float(np.max(neighbor_maximum - expected_values, initial=0.0))
    expected_ids = (candidate_systems + 1) * 1_000_000_000 + candidate_hosts
    candidate_ids = np.asarray(deposit_catalog["deposit_candidate_id"], dtype=np.int64)
    stable_id_valid = bool(np.array_equal(candidate_ids, expected_ids))
    if candidate_value_error > 1e-7:
        hard_failures.append("candidate_potential_mismatch")
    if local_maximum_error > config.maximum_candidate_local_maximum_error:
        hard_failures.append("candidate_not_local_maximum")
    if not stable_id_valid:
        hard_failures.append("candidate_stable_id_mismatch")

    validation_catalog = pa.Table.from_pylist(rows)
    metadata: dict[str, object] = {
        **asdict(config),
        "model": "causal_mineral_systems_validation_v0",
        "validated_mineral_model": mineral_metadata["model"],
        "family_count": len(SYSTEM_NAMES),
        "passing_family_count": int(sum(validation_catalog["passed"].to_pylist())),
        "candidate_potential_maximum_absolute_error": candidate_value_error,
        "candidate_local_maximum_error": local_maximum_error,
        "candidate_stable_ids_valid": int(stable_id_valid),
        "directional_driver_semantics": "independent_upstream_setting_contrast",
        "earth_abundance_quota_applied": 0,
        "hard_failures": sorted(set(hard_failures)),
        "hard_gate_pass": int(not hard_failures),
        "ready_for_l3_mineral_realization": int(not hard_failures),
    }
    context.logger.log_event(
        {
            "type": "mineral_systems_validation_summary",
            "stage": "mineral_systems_validation",
            **metadata,
        }
    )
    return {
        "MineralSystemsValidationCatalog": validation_catalog,
        "MineralSystemsValidationMetadata": metadata,
    }


__all__ = [
    "MineralSystemsValidationConfig",
    "mineral_systems_validation_stage",
]
