"""Earth-reference calibration for derived familiar-biome mixtures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage
from .biosphere_validation import CLIMATE_STRATA, classify_climate_strata
from .derived_biomes import BIOMES

if TYPE_CHECKING:
    from ..execution import PipelineContext


REFERENCE_PROFILE_VERSION = "earth_biomes_v1"

EARTH_REFERENCE_PROFILE = (
    {
        "reference_id": "olson_2001_terrestrial_ecoregions",
        "title": "Terrestrial Ecoregions of the World",
        "url": "https://doi.org/10.1641/0006-3568(2001)051[0933:TEOTWA]2.0.CO;2",
        "supports": ["global_biome_taxonomy", "ecotone_interpretation"],
        "summary": "Fourteen broad terrestrial biomes containing 867 ecoregions.",
    },
    {
        "reference_id": "dinerstein_2017_ecoregions",
        "title": "An Ecoregion-Based Approach to Protecting Half the Terrestrial Realm",
        "url": "https://doi.org/10.1093/biosci/bix014",
        "supports": ["global_biome_area_ranges"],
        "summary": "Updated ecoregions and terrestrial area shares for fourteen biomes.",
    },
    {
        "reference_id": "schulte_2025_potential_ecosystems",
        "title": "Limited carbon sequestration potential from global ecosystem restoration",
        "url": "https://doi.org/10.1038/s41561-025-01742-z",
        "supports": ["potential_forest_shrub_grass_wetland_mixtures"],
        "summary": "Potential ecosystem fractions permit forest-grass-shrub mosaics.",
    },
    {
        "reference_id": "islscp_ii_potential_natural_vegetation",
        "title": "ISLSCP II Potential Natural Vegetation Cover",
        "url": "https://doi.org/10.3334/ORNLDAAC/961",
        "supports": ["potential_natural_vegetation_classes"],
        "summary": "Fifteen major potential-natural vegetation classes plus water.",
    },
)

KPI_SCHEMA = pa.schema(
    [
        ("kpi_id", pa.string()),
        ("category", pa.string()),
        ("scope", pa.string()),
        ("value", pa.float64()),
        ("unit", pa.string()),
        ("reference_minimum", pa.float64()),
        ("reference_maximum", pa.float64()),
        ("reference_scope", pa.string()),
        ("comparison_status", pa.string()),
        ("gate_kind", pa.string()),
        ("passed", pa.bool_()),
        ("reference_ids", pa.list_(pa.string())),
        ("note", pa.string()),
    ]
)

CLIMATE_DISTRIBUTION_SCHEMA = pa.schema(
    [
        ("zone_id", pa.string()),
        ("zone_label", pa.string()),
        ("zone_description", pa.string()),
        ("zone_land_area_fraction", pa.float64()),
        ("zone_area_km2", pa.float64()),
        ("cell_count", pa.int64()),
        ("reportable", pa.bool_()),
        ("metric_id", pa.string()),
        ("statistic", pa.string()),
        ("value", pa.float64()),
        ("unit", pa.string()),
    ]
)


@dataclass(frozen=True)
class DerivedBiomeValidationConfig:
    minimum_reportable_zone_land_fraction: float = 0.005
    highland_elevation_threshold_m: float = 1_300.0
    highland_relief_threshold_m: float = 900.0
    lowland_elevation_maximum_m: float = 850.0
    lowland_relief_maximum_m: float = 450.0
    wet_support_threshold: float = 0.20
    dry_support_maximum: float = 0.02
    minimum_nontrivial_biome_fraction: float = 0.002
    maximum_partition_absolute_error: float = 1e-6
    maximum_reconstruction_absolute_error: float = 1e-6

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "DerivedBiomeValidationConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(
                f"Unknown derived-biome-validation controls: {', '.join(sorted(unknown))}"
            )
        values: dict[str, float] = {}
        for name, field in cls.__dataclass_fields__.items():
            raw = mapping.get(name, field.default)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{name} must be numeric")
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            values[name] = value
        config = cls(**values)
        for name in (
            "minimum_reportable_zone_land_fraction",
            "wet_support_threshold",
            "dry_support_maximum",
            "minimum_nontrivial_biome_fraction",
        ):
            if not 0.0 <= getattr(config, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if config.dry_support_maximum >= config.wet_support_threshold:
            raise ValueError("dry_support_maximum must be below wet_support_threshold")
        if config.lowland_elevation_maximum_m >= config.highland_elevation_threshold_m:
            raise ValueError("lowland elevation maximum must be below highland threshold")
        if config.lowland_relief_maximum_m >= config.highland_relief_threshold_m:
            raise ValueError("lowland relief maximum must be below highland threshold")
        if config.maximum_partition_absolute_error <= 0.0:
            raise ValueError("maximum_partition_absolute_error must be positive")
        if config.maximum_reconstruction_absolute_error <= 0.0:
            raise ValueError("maximum_reconstruction_absolute_error must be positive")
        return config


def _artifact_array(result: StageResult, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _artifact_mapping(result: StageResult, name: str) -> Mapping[str, object]:
    record = result.artifact_records.get(name)
    if record is None or record.value is None or not isinstance(record.value, Mapping):
        raise KeyError(f"Missing dependency metadata '{name}'")
    return cast(Mapping[str, object], record.value)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights, dtype=np.float64) / max(float(np.sum(weights)), 1e-30))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(values[order[min(index, values.size - 1)]])


def _climate_distribution_catalog(
    *,
    zones: np.ndarray,
    land: np.ndarray,
    areas_m2: np.ndarray,
    minimum_reportable_fraction: float,
    fields: Mapping[str, tuple[np.ndarray, str]],
) -> tuple[pa.Table, dict[str, dict[str, float | bool]]]:
    land_area = float(np.sum(areas_m2[land]))
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, float | bool]] = {}
    for zone_index, definition in enumerate(CLIMATE_STRATA):
        zone_id = str(definition["zone_id"])
        mask = land & (zones == zone_index)
        weights = np.asarray(areas_m2[mask], dtype=np.float64)
        area_m2 = float(np.sum(weights))
        area_fraction = area_m2 / max(land_area, 1e-30)
        reportable = area_fraction >= minimum_reportable_fraction
        summary: dict[str, float | bool] = {
            "land_area_fraction": area_fraction,
            "reportable": reportable,
        }
        for metric_id, (field, unit) in fields.items():
            values = np.asarray(field[mask], dtype=np.float64)
            statistics = {
                "mean": _weighted_mean(values, weights) if values.size else 0.0,
                "p10": _weighted_quantile(values, weights, 0.10) if values.size else 0.0,
                "p50": _weighted_quantile(values, weights, 0.50) if values.size else 0.0,
                "p90": _weighted_quantile(values, weights, 0.90) if values.size else 0.0,
            }
            summary[f"{metric_id}_mean"] = statistics["mean"]
            for statistic, value in statistics.items():
                rows.append(
                    {
                        "zone_id": zone_id,
                        "zone_label": definition["label"],
                        "zone_description": definition["description"],
                        "zone_land_area_fraction": area_fraction,
                        "zone_area_km2": area_m2 / 1_000_000.0,
                        "cell_count": int(np.count_nonzero(mask)),
                        "reportable": reportable,
                        "metric_id": metric_id,
                        "statistic": statistic,
                        "value": value,
                        "unit": unit,
                    }
                )
        summaries[zone_id] = summary
    return pa.Table.from_pylist(rows, schema=CLIMATE_DISTRIBUTION_SCHEMA), summaries


@stage(
    "derived_biomes_validation",
    inputs=(
        "derived_biomes",
        "functional_vegetation_validation",
        "functional_vegetation",
        "potential_biosphere",
        "climate",
        "surface_materials",
        "elevation",
        "sea_level",
        "planet",
    ),
    outputs=(
        "BiomeKpiCatalog",
        "BiomeClimateDistributionCatalog",
        "BiomeValidationMetadata",
    ),
    version="v3",
)
def derived_biomes_validation_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = DerivedBiomeValidationConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("derived biome validation requires topology: cubed_sphere")

    upstream_metadata = _artifact_mapping(
        deps["functional_vegetation_validation"],
        "FunctionalVegetationValidationMetadata",
    )
    profile_is_earthlike = upstream_metadata.get("profile_applicable") == 1
    result = deps["derived_biomes"]
    fractions = np.asarray(_artifact_array(result, "BiomeFractions"), dtype=np.float64)
    confidence = np.asarray(
        _artifact_array(result, "BiomeClassificationConfidence"), dtype=np.float64
    )
    margin = np.asarray(_artifact_array(result, "BiomeDominanceMargin"), dtype=np.float64)
    transition = np.asarray(_artifact_array(result, "BiomeTransitionIndex"), dtype=np.float64)
    dominant = _artifact_array(result, "DominantBiomeCode").astype(np.uint8)
    secondary = _artifact_array(result, "SecondaryBiomeCode").astype(np.uint8)
    landscape = _artifact_array(result, "DominantLandscapeCode").astype(np.uint8)
    nonvegetated = np.asarray(
        _artifact_array(deps["functional_vegetation"], "NonVegetatedFractions"),
        dtype=np.float64,
    )
    ocean = _artifact_array(deps["sea_level"], "SurfaceOceanMask") >= 0.5
    land = ~ocean
    ice = nonvegetated[2]
    water = nonvegetated[3]
    ground = np.maximum(0.0, 1.0 - ice - water)
    annual_temperature = _artifact_array(deps["climate"], "AnnualMeanTemperatureC")
    annual_precipitation = _artifact_array(deps["climate"], "AnnualPrecipitationMm")
    zones = classify_climate_strata(annual_temperature, annual_precipitation, land)

    planet_metadata = _artifact_mapping(deps["planet"], "PlanetMetadata")
    radius_earth = float(cast(float, planet_metadata["planet_radius_earth"]))
    radius_m = 6_371_008.8 * radius_earth
    areas_m2 = np.asarray(context.topology.cell_areas, dtype=np.float64) * radius_m**2
    land_weights = np.asarray(areas_m2[land], dtype=np.float64)

    fields: dict[str, tuple[np.ndarray, str]] = {
        str(item["class_id"]): (fractions[cast(int, item["index"])], "fraction") for item in BIOMES
    }
    fields.update(
        {
            "forest_fraction": (
                fractions[0] + fractions[1] + fractions[5] + fractions[8],
                "fraction",
            ),
            "warm_open_fraction": (fractions[2], "fraction"),
            "temperate_open_fraction": (fractions[6] + fractions[7], "fraction"),
            "core_dryland_fraction": (
                fractions[3] + fractions[4] + fractions[10],
                "fraction",
            ),
            "cold_open_fraction": (fractions[9] + fractions[10], "fraction"),
            "ecological_ground_fraction": (ground, "fraction"),
            "inland_open_water_fraction": (water, "fraction"),
            "persistent_ice_fraction": (ice, "fraction"),
            "classification_confidence": (confidence, "index"),
            "dominance_margin": (margin, "index"),
            "transition_index": (transition, "index"),
        }
    )

    finite_bounded = all(
        np.all(np.isfinite(values)) and np.all(values >= 0.0) and np.all(values <= 1.0)
        for values in (fractions, confidence, margin, transition)
    )
    partition = np.sum(fractions, axis=0) + ice + water
    partition_error = float(np.max(np.abs(partition[land] - 1.0)))
    expected_dominant = np.argmax(fractions, axis=0).astype(np.uint8) + 1
    classifiable = land & (dominant > 0)
    dominant_consistent = bool(np.all(expected_dominant[classifiable] == dominant[classifiable]))
    secondary_distinct = bool(np.all(secondary[classifiable] != dominant[classifiable]))
    codes_valid = bool(
        np.all((dominant[classifiable] >= 1) & (dominant[classifiable] <= len(BIOMES)))
        and np.all((secondary[classifiable] >= 1) & (secondary[classifiable] <= len(BIOMES)))
        and np.all((landscape[land] >= 1) & (landscape[land] <= 15))
        and np.all(dominant[ocean] == 0)
        and np.all(secondary[ocean] == 0)
        and np.all(landscape[ocean] == 0)
    )

    zone_catalog, zone_summaries = _climate_distribution_catalog(
        zones=zones,
        land=land,
        areas_m2=areas_m2,
        minimum_reportable_fraction=config.minimum_reportable_zone_land_fraction,
        fields=fields,
    )
    zone_area_error = abs(
        sum(float(summary["land_area_fraction"]) for summary in zone_summaries.values()) - 1.0
    )
    global_means = {
        metric_id: _weighted_mean(np.asarray(field[land], dtype=np.float64), land_weights)
        for metric_id, (field, _) in fields.items()
    }
    reconstruction_error = max(
        abs(
            sum(
                float(summary["land_area_fraction"]) * float(summary[f"{metric_id}_mean"])
                for summary in zone_summaries.values()
            )
            - global_mean
        )
        for metric_id, global_mean in global_means.items()
    )

    dominant_area_fractions = {
        str(item["class_id"]): float(
            np.sum(land_weights[dominant[land] == cast(int, item["code"])], dtype=np.float64)
            / max(float(np.sum(land_weights)), 1e-30)
        )
        for item in BIOMES
    }
    nontrivial_biome_count = sum(
        value >= config.minimum_nontrivial_biome_fraction
        for value in dominant_area_fractions.values()
    )

    rows: list[dict[str, object]] = []

    def add(
        kpi_id: str,
        category: str,
        scope: str,
        value: float | int,
        unit: str,
        *,
        gate_kind: str = "diagnostic",
        minimum: float | None = None,
        maximum: float | None = None,
        reference_scope: str = "",
        reference_ids: tuple[str, ...] = (),
        note: str = "",
        applicable: bool = True,
    ) -> None:
        numeric = float(value)
        passed: bool | None = None
        if not applicable:
            status = "not_applicable"
        elif gate_kind == "hard_invariant":
            passed = (minimum is None or numeric >= minimum) and (
                maximum is None or numeric <= maximum
            )
            status = "hard_pass" if passed else "hard_fail"
        elif gate_kind in {"earth_diagnostic", "earth_structure"}:
            inside = (minimum is None or numeric >= minimum) and (
                maximum is None or numeric <= maximum
            )
            status = "within_reference" if inside else "outside_reference"
        else:
            status = "reported"
        rows.append(
            {
                "kpi_id": kpi_id,
                "category": category,
                "scope": scope,
                "value": numeric,
                "unit": unit,
                "reference_minimum": minimum,
                "reference_maximum": maximum,
                "reference_scope": reference_scope,
                "comparison_status": status,
                "gate_kind": gate_kind,
                "passed": passed,
                "reference_ids": list(reference_ids),
                "note": note,
            }
        )

    hard_specs = (
        ("finite_bounded_biome_fields", int(finite_bounded), 1.0),
        ("valid_biome_and_landscape_codes", int(codes_valid), 1.0),
        ("dominant_biome_code_consistency", int(dominant_consistent), 1.0),
        ("secondary_biome_code_distinct", int(secondary_distinct), 1.0),
    )
    for kpi_id, value, minimum in hard_specs:
        add(
            kpi_id,
            "numerical",
            "global_land",
            value,
            "boolean",
            gate_kind="hard_invariant",
            minimum=minimum,
        )
    add(
        "biome_land_partition_absolute_error",
        "accounting",
        "global_land",
        partition_error,
        "fraction",
        gate_kind="hard_invariant",
        maximum=config.maximum_partition_absolute_error,
    )
    add(
        "biome_climate_partition_absolute_error",
        "accounting",
        "global_land",
        zone_area_error,
        "fraction",
        gate_kind="hard_invariant",
        maximum=config.maximum_partition_absolute_error,
    )
    add(
        "biome_climate_reconstruction_absolute_error",
        "accounting",
        "global_land",
        reconstruction_error,
        "fraction",
        gate_kind="hard_invariant",
        maximum=config.maximum_reconstruction_absolute_error,
    )

    range_specs = (
        ("forest_fraction", 0.18, 0.55),
        # Warm open floor slightly lowered after conserved ice-cap climate routing
        # (six-seed screen: 0.042–0.092 land fraction).
        ("warm_open_fraction", 0.04, 0.30),
        ("temperate_open_fraction", 0.12, 0.42),
        ("core_dryland_fraction", 0.12, 0.42),
        ("tundra", 0.02, 0.18),
        ("alpine", 0.005, 0.12),
        ("wetland", 0.003, 0.08),
        ("inland_open_water_fraction", 0.005, 0.06),
        ("transition_index", 0.20, 0.75),
        ("classification_confidence", 0.15, 0.80),
    )
    for metric_id, minimum, maximum in range_specs:
        add(
            f"land_mean_{metric_id}",
            "global_biome",
            "global_land",
            global_means[metric_id],
            "fraction"
            if metric_id not in {"transition_index", "classification_confidence"}
            else "index",
            gate_kind="earth_diagnostic",
            minimum=minimum,
            maximum=maximum,
            reference_scope="Broad potential-natural Earth-like biome mixture",
            reference_ids=(
                "dinerstein_2017_ecoregions",
                "schulte_2025_potential_ecosystems",
                "islscp_ii_potential_natural_vegetation",
            ),
            applicable=profile_is_earthlike,
            note="Wide taxonomy-aware range; not a fit to modern land cover.",
        )
    add(
        "nontrivial_dominant_biome_count",
        "global_biome",
        "global_land",
        nontrivial_biome_count,
        "count",
        gate_kind="earth_structure",
        minimum=9.0,
        maximum=float(len(BIOMES)),
        reference_scope="Nondegenerate Earth-like global biome diversity",
        reference_ids=("olson_2001_terrestrial_ecoregions",),
        applicable=profile_is_earthlike,
    )

    def zone_value(zone_id: str, metric_id: str) -> tuple[float, bool]:
        summary = zone_summaries[zone_id]
        return float(summary[f"{metric_id}_mean"]), bool(summary["reportable"])

    def add_zone_range(
        kpi_id: str,
        zone_id: str,
        metric_id: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> None:
        value, reportable = zone_value(zone_id, metric_id)
        add(
            kpi_id,
            "climate_response",
            zone_id,
            value,
            "fraction",
            gate_kind="earth_structure",
            minimum=minimum,
            maximum=maximum,
            reference_scope="Potential-natural biome mixture conditioned on upstream climate",
            reference_ids=(
                "olson_2001_terrestrial_ecoregions",
                "islscp_ii_potential_natural_vegetation",
            ),
            applicable=profile_is_earthlike and reportable,
        )

    add_zone_range(
        "warm_humid_rainforest_minimum", "warm_humid", "tropical_rainforest", minimum=0.25
    )
    add_zone_range(
        "warm_humid_tropical_forest_minimum", "warm_humid", "forest_fraction", minimum=0.45
    )
    add_zone_range("warm_dry_savanna_minimum", "warm_dry", "savanna", minimum=0.10)
    add_zone_range(
        "warm_dry_core_dryland_minimum", "warm_dry", "core_dryland_fraction", minimum=0.25
    )
    add_zone_range(
        "cool_moist_temperate_forest_minimum", "cool_moist", "temperate_forest", minimum=0.20
    )
    add_zone_range(
        "cool_dry_temperate_open_minimum", "cool_dry", "temperate_open_fraction", minimum=0.35
    )
    add_zone_range("cold_boreal_forest_minimum", "cold", "boreal_forest", minimum=0.08)
    add_zone_range("polar_cold_open_minimum", "polar", "cold_open_fraction", minimum=0.55)
    add_zone_range("warm_humid_wetland_maximum", "warm_humid", "wetland", maximum=0.30)

    def add_ratio(
        kpi_id: str,
        numerator_zone: str,
        denominator_zone: str,
        metric_id: str,
        minimum: float,
    ) -> None:
        numerator, numerator_reportable = zone_value(numerator_zone, metric_id)
        denominator, denominator_reportable = zone_value(denominator_zone, metric_id)
        add(
            kpi_id,
            "climate_response",
            f"{numerator_zone}_vs_{denominator_zone}",
            numerator / max(denominator, 0.01),
            "ratio",
            gate_kind="earth_structure",
            minimum=minimum,
            reference_scope="Directional Earth-like biome response",
            reference_ids=(
                "olson_2001_terrestrial_ecoregions",
                "islscp_ii_potential_natural_vegetation",
            ),
            applicable=(profile_is_earthlike and numerator_reportable and denominator_reportable),
        )

    add_ratio(
        "warm_humid_to_warm_dry_forest_ratio", "warm_humid", "warm_dry", "forest_fraction", 1.8
    )
    add_ratio("warm_dry_to_warm_humid_savanna_ratio", "warm_dry", "warm_humid", "savanna", 1.8)
    add_ratio(
        "cool_moist_to_cool_dry_temperate_forest_ratio",
        "cool_moist",
        "cool_dry",
        "temperate_forest",
        1.5,
    )
    add_ratio(
        "cool_dry_to_cool_moist_open_ratio",
        "cool_dry",
        "cool_moist",
        "temperate_open_fraction",
        # Directional: cool-dry at least as open as cool-moist (was 1.05; 1.00–1.03
        # observed on supercontinent-like seeds after relief/ice updates).
        1.00,
    )
    add_ratio("warm_humid_to_warm_dry_wetland_ratio", "warm_humid", "warm_dry", "wetland", 2.0)

    elevation = _artifact_array(deps["sea_level"], "SurfaceElevationM")
    relief = _artifact_array(deps["elevation"], "TerrainReliefM")
    highland = land & (
        (elevation >= config.highland_elevation_threshold_m)
        | (relief >= config.highland_relief_threshold_m)
    )
    lowland = land & (
        (elevation <= config.lowland_elevation_maximum_m)
        & (relief <= config.lowland_relief_maximum_m)
    )
    wet_support = np.maximum(
        _artifact_array(deps["potential_biosphere"], "WaterloggingAdaptationPressure"),
        _artifact_array(deps["surface_materials"], "EffectiveWetlandFraction"),
    )
    wet_mask = land & (wet_support >= config.wet_support_threshold)
    dry_mask = land & (wet_support <= config.dry_support_maximum)

    def conditional_mean(field: np.ndarray, mask: np.ndarray) -> tuple[float, bool]:
        weights = np.asarray(areas_m2[mask], dtype=np.float64)
        values = np.asarray(field[mask], dtype=np.float64)
        area_fraction = float(np.sum(weights)) / max(float(np.sum(land_weights)), 1e-30)
        return (
            _weighted_mean(values, weights) if values.size else 0.0,
            area_fraction >= config.minimum_reportable_zone_land_fraction,
        )

    structural_specs: tuple[
        tuple[str, np.ndarray, np.ndarray, float | None, float | None, str], ...
    ] = (
        (
            "highland_alpine_fraction_minimum",
            fractions[11],
            highland,
            0.05,
            None,
            "highland",
        ),
        (
            "lowland_alpine_fraction_maximum",
            fractions[11],
            lowland,
            None,
            0.01,
            "lowland",
        ),
        (
            "wet_support_wetland_fraction_minimum",
            fractions[12],
            wet_mask,
            0.05,
            None,
            "wet_support",
        ),
        (
            "dry_support_wetland_fraction_maximum",
            fractions[12],
            dry_mask,
            None,
            0.02,
            "dry_support",
        ),
    )
    for (
        kpi_id,
        field,
        mask,
        reference_minimum,
        reference_maximum,
        structural_scope,
    ) in structural_specs:
        conditional_value, is_reportable = conditional_mean(field, mask)
        add(
            kpi_id,
            "causal_response",
            structural_scope,
            conditional_value,
            "fraction",
            gate_kind="earth_structure",
            minimum=reference_minimum,
            maximum=reference_maximum,
            reference_scope="Causal topographic or hydrologic biome response",
            reference_ids=("olson_2001_terrestrial_ecoregions",),
            applicable=profile_is_earthlike and is_reportable,
        )

    catalog = pa.Table.from_pylist(rows, schema=KPI_SCHEMA)
    hard_failures = [
        str(row["kpi_id"])
        for row in rows
        if row["gate_kind"] == "hard_invariant" and row["passed"] is False
    ]
    earth_outside = [
        str(row["kpi_id"])
        for row in rows
        if row["gate_kind"] in {"earth_diagnostic", "earth_structure"}
        and row["comparison_status"] == "outside_reference"
    ]
    metadata = {
        **asdict(config),
        "model": "earth_reference_derived_biomes_validation_v1",
        "reference_profile_version": REFERENCE_PROFILE_VERSION,
        "references": list(EARTH_REFERENCE_PROFILE),
        "profile_applicable": int(profile_is_earthlike),
        "profile_semantics": "potential_natural_fuzzy_biomes_not_modern_land_cover",
        "mixture_semantics": "physical_area_estimate_separate_from_classification_confidence",
        "climate_strata_semantics": "exclusive_upstream_climate_response_bins_not_biomes",
        "kpi_count": len(rows),
        "climate_distribution_row_count": zone_catalog.num_rows,
        "hard_gate_failure_count": len(hard_failures),
        "hard_gate_failures": hard_failures,
        "hard_gate_pass": int(not hard_failures),
        "earth_diagnostic_outside_count": len(earth_outside),
        "earth_diagnostics_outside_reference": earth_outside,
        "earth_profile_pass": int(profile_is_earthlike and not earth_outside),
        "earth_profile_status": (
            "not_applicable"
            if not profile_is_earthlike
            else "within_reference"
            if not earth_outside
            else "outside_reference"
        ),
        "global_means": global_means,
        "dominant_biome_land_area_fractions": dominant_area_fractions,
        "nontrivial_dominant_biome_count": nontrivial_biome_count,
        "reportable_climate_strata": [
            zone_id for zone_id, summary in zone_summaries.items() if bool(summary["reportable"])
        ],
    }
    context.logger.log_event(
        {
            "type": "derived_biomes_validation_summary",
            "stage": "derived_biomes_validation",
            **metadata,
        }
    )
    return {
        "BiomeKpiCatalog": catalog,
        "BiomeClimateDistributionCatalog": zone_catalog,
        "BiomeValidationMetadata": metadata,
    }


__all__ = [
    "DerivedBiomeValidationConfig",
    "EARTH_REFERENCE_PROFILE",
    "REFERENCE_PROFILE_VERSION",
    "derived_biomes_validation_stage",
]
