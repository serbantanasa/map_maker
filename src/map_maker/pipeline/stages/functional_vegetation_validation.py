"""Earth-reference calibration for functional vegetation and resource potentials."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage
from .biosphere_validation import CLIMATE_STRATA, classify_climate_strata

if TYPE_CHECKING:
    from ..execution import PipelineContext


REFERENCE_PROFILE_VERSION = "earth_functional_vegetation_v1"

EARTH_REFERENCE_PROFILE = (
    {
        "reference_id": "modis_vcf_c61",
        "title": "MODIS Vegetation Continuous Fields Collection 6.1",
        "url": "https://modis-land.gsfc.nasa.gov/vcc.html",
        "supports": ["fractional_vegetation_structure"],
        "summary": "Global subpixel tree, non-tree vegetation, and nonvegetated cover.",
    },
    {
        "reference_id": "hengl_2018_potential_natural_vegetation",
        "title": "Global mapping of potential natural vegetation",
        "url": "https://doi.org/10.1111/geb.12759",
        "supports": ["potential_natural_vegetation_structure"],
        "summary": "Climate-conditioned potential vegetation independent of modern land use.",
    },
    {
        "reference_id": "schulte_2025_potential_ecosystems",
        "title": "Limited carbon sequestration potential from global ecosystem restoration",
        "url": "https://doi.org/10.1038/s41561-025-01742-z",
        "supports": ["woody_herbaceous_shrub_wetland_ranges"],
        "summary": "Broad global potential forest, shrubland, grassland, and wetland extents.",
    },
    {
        "reference_id": "glwd_v2",
        "title": "Global Lakes and Wetlands Database version 2",
        "url": "https://doi.org/10.5194/essd-17-2277-2025",
        "supports": ["hydrophytic_and_inland_water_ranges"],
        "summary": "Fractional global inland-water and wetland maximum extents.",
    },
    {
        "reference_id": "gfed4_burned_area",
        "title": "Analysis of burned area using GFED4",
        "url": "https://doi.org/10.1002/jgrg.20042",
        "supports": ["fire_climate_structure"],
        "summary": "Savannas dominate observed global burned area.",
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
class FunctionalVegetationValidationConfig:
    minimum_reportable_zone_land_fraction: float = 0.005
    maximum_partition_absolute_error: float = 1e-6
    maximum_reconstruction_absolute_error: float = 1e-6

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "FunctionalVegetationValidationConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(
                "Unknown functional-vegetation-validation controls: " + ", ".join(sorted(unknown))
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
        if not 0.0 <= config.minimum_reportable_zone_land_fraction <= 1.0:
            raise ValueError("minimum_reportable_zone_land_fraction must be in [0, 1]")
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
    "functional_vegetation_validation",
    inputs=("functional_vegetation", "biosphere_validation", "climate", "world_age", "planet"),
    outputs=(
        "FunctionalVegetationKpiCatalog",
        "FunctionalVegetationClimateDistributionCatalog",
        "FunctionalVegetationValidationMetadata",
    ),
    version="v1",
)
def functional_vegetation_validation_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = FunctionalVegetationValidationConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError(
            "functional vegetation validation requires topology: cubed_sphere"
        )

    functional_result = deps["functional_vegetation"]
    biosphere_metadata = _artifact_mapping(
        deps["biosphere_validation"], "BiosphereValidationMetadata"
    )
    profile_is_earthlike = biosphere_metadata.get("profile_applicable") == 1
    functional = np.asarray(
        _artifact_array(functional_result, "FunctionalTypeFractions"), dtype=np.float64
    )
    nonvegetated = np.asarray(
        _artifact_array(functional_result, "NonVegetatedFractions"), dtype=np.float64
    )
    resources = np.asarray(
        _artifact_array(functional_result, "FunctionalResourcePotentials"), dtype=np.float64
    )
    ocean = _artifact_array(deps["world_age"], "BaseOceanMask") >= 0.5
    land = ~ocean
    annual_temperature = _artifact_array(deps["climate"], "AnnualMeanTemperatureC")
    annual_precipitation = _artifact_array(deps["climate"], "AnnualPrecipitationMm")
    zones = classify_climate_strata(annual_temperature, annual_precipitation, land)

    planet_metadata = _artifact_mapping(deps["planet"], "PlanetMetadata")
    radius_earth = float(cast(float, planet_metadata["planet_radius_earth"]))
    radius_m = 6_371_008.8 * radius_earth
    areas_m2 = np.asarray(context.topology.cell_areas, dtype=np.float64) * radius_m**2
    land_weights = np.asarray(areas_m2[land], dtype=np.float64)

    fields = {
        "functional_vegetated_fraction": (np.sum(functional, axis=0), "fraction"),
        "functional_woody_fraction": (np.sum(functional[0:3], axis=0), "fraction"),
        "functional_xeric_shrub_fraction": (functional[3], "fraction"),
        "functional_herbaceous_fraction": (np.sum(functional[4:6], axis=0), "fraction"),
        "functional_hydrophytic_fraction": (functional[6], "fraction"),
        "functional_low_stature_fraction": (functional[7], "fraction"),
        "bare_ground_fraction": (nonvegetated[0], "fraction"),
        "saline_barren_fraction": (nonvegetated[1], "fraction"),
        "persistent_ice_fraction": (nonvegetated[2], "fraction"),
        "inland_open_water_fraction": (nonvegetated[3], "fraction"),
        "unsupported_surface_fraction": (nonvegetated[4], "fraction"),
        "resource_fire_tendency": (resources[0], "index"),
        "resource_grazing": (resources[1], "index"),
        "resource_forest": (resources[2], "index"),
        "resource_pasture": (resources[3], "index"),
        "resource_crop": (resources[4], "index"),
    }
    fields["functional_xeric_low_stature_fraction"] = (
        fields["functional_xeric_shrub_fraction"][0] + fields["functional_low_stature_fraction"][0],
        "fraction",
    )
    fields["nonvegetated_ground_fraction"] = (
        nonvegetated[0] + nonvegetated[1] + nonvegetated[4],
        "fraction",
    )

    finite_bounded = all(
        np.all(np.isfinite(values)) and np.all(values >= 0.0) and np.all(values <= 1.0)
        for values in (functional, nonvegetated, resources)
    )
    partition = np.sum(functional, axis=0) + np.sum(nonvegetated, axis=0)
    partition_error = float(np.max(np.abs(partition[land] - 1.0)))

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

    add(
        "finite_bounded_functional_fields",
        "numerical",
        "global_land",
        int(finite_bounded),
        "boolean",
        gate_kind="hard_invariant",
        minimum=1.0,
    )
    add(
        "functional_land_partition_absolute_error",
        "accounting",
        "global_land",
        partition_error,
        "fraction",
        gate_kind="hard_invariant",
        maximum=config.maximum_partition_absolute_error,
    )
    add(
        "functional_climate_partition_absolute_error",
        "accounting",
        "global_land",
        zone_area_error,
        "fraction",
        gate_kind="hard_invariant",
        maximum=config.maximum_partition_absolute_error,
    )
    add(
        "functional_climate_reconstruction_absolute_error",
        "accounting",
        "global_land",
        reconstruction_error,
        "fraction",
        gate_kind="hard_invariant",
        maximum=config.maximum_reconstruction_absolute_error,
    )

    range_specs = (
        ("functional_vegetated_fraction", 0.35, 0.75),
        ("functional_woody_fraction", 0.13, 0.40),
        ("functional_herbaceous_fraction", 0.12, 0.40),
        ("functional_xeric_low_stature_fraction", 0.05, 0.30),
        ("functional_hydrophytic_fraction", 0.005, 0.10),
        ("nonvegetated_ground_fraction", 0.20, 0.60),
        ("inland_open_water_fraction", 0.005, 0.06),
    )
    for metric_id, minimum, maximum in range_specs:
        add(
            f"land_mean_{metric_id}",
            "global_cover",
            "global_land",
            global_means[metric_id],
            "fraction",
            gate_kind="earth_diagnostic",
            minimum=minimum,
            maximum=maximum,
            reference_scope="Broad potential-natural fractional cover on Earth-like land",
            reference_ids=(
                "modis_vcf_c61",
                "hengl_2018_potential_natural_vegetation",
                "schulte_2025_potential_ecosystems",
                "glwd_v2",
            ),
            applicable=profile_is_earthlike,
            note="Wide structural range; not a fit to modern land use.",
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
            reference_scope="Potential-natural cover conditioned on upstream climate",
            reference_ids=(
                "modis_vcf_c61",
                "hengl_2018_potential_natural_vegetation",
            ),
            applicable=profile_is_earthlike and reportable,
        )

    add_zone_range(
        "cool_moist_woody_cover_minimum",
        "cool_moist",
        "functional_woody_fraction",
        minimum=0.15,
    )
    add_zone_range(
        "warm_humid_woody_cover_minimum",
        "warm_humid",
        "functional_woody_fraction",
        minimum=0.30,
    )
    add_zone_range(
        "warm_humid_hydrophytic_cover_maximum",
        "warm_humid",
        "functional_hydrophytic_fraction",
        maximum=0.20,
    )
    add_zone_range(
        "warm_dry_xeric_low_stature_minimum",
        "warm_dry",
        "functional_xeric_low_stature_fraction",
        minimum=0.08,
    )
    add_zone_range(
        "polar_woody_cover_maximum",
        "polar",
        "functional_woody_fraction",
        maximum=0.05,
    )
    add_zone_range(
        "warm_humid_vegetated_cover_minimum",
        "warm_humid",
        "functional_vegetated_fraction",
        minimum=0.65,
    )
    add_zone_range(
        "warm_dry_vegetated_cover_maximum",
        "warm_dry",
        "functional_vegetated_fraction",
        maximum=0.65,
    )

    def add_ratio(
        kpi_id: str,
        numerator_zone: str,
        denominator_zone: str,
        metric_id: str,
        minimum: float,
        *,
        reference_ids: tuple[str, ...],
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
            reference_scope="Directional Earth-like functional response",
            reference_ids=reference_ids,
            applicable=(profile_is_earthlike and numerator_reportable and denominator_reportable),
        )

    vegetation_references = (
        "modis_vcf_c61",
        "hengl_2018_potential_natural_vegetation",
    )
    add_ratio(
        "warm_humid_to_warm_dry_woody_ratio",
        "warm_humid",
        "warm_dry",
        "functional_woody_fraction",
        1.8,
        reference_ids=vegetation_references,
    )
    add_ratio(
        "cool_moist_to_cool_dry_woody_ratio",
        "cool_moist",
        "cool_dry",
        "functional_woody_fraction",
        1.8,
        reference_ids=vegetation_references,
    )
    add_ratio(
        "warm_dry_to_warm_humid_xeric_low_stature_ratio",
        "warm_dry",
        "warm_humid",
        "functional_xeric_low_stature_fraction",
        1.5,
        reference_ids=vegetation_references,
    )
    add_ratio(
        "warm_humid_to_warm_dry_hydrophytic_ratio",
        "warm_humid",
        "warm_dry",
        "functional_hydrophytic_fraction",
        2.0,
        reference_ids=("glwd_v2",),
    )
    add_ratio(
        "warm_seasonal_to_warm_humid_fire_ratio",
        "warm_seasonal",
        "warm_humid",
        "resource_fire_tendency",
        1.2,
        reference_ids=("gfed4_burned_area",),
    )
    add_ratio(
        "warm_humid_to_warm_dry_forest_resource_ratio",
        "warm_humid",
        "warm_dry",
        "resource_forest",
        1.5,
        reference_ids=vegetation_references,
    )
    add_ratio(
        "warm_dry_to_warm_humid_grazing_ratio",
        "warm_dry",
        "warm_humid",
        "resource_grazing",
        1.2,
        reference_ids=vegetation_references,
    )
    add_ratio(
        "warm_seasonal_to_polar_crop_resource_ratio",
        "warm_seasonal",
        "polar",
        "resource_crop",
        1.5,
        reference_ids=vegetation_references,
    )

    resource_p90_minimums = {
        "fire_tendency": 0.12,
        "grazing": 0.25,
        "forest": 0.15,
        "pasture": 0.12,
        "crop": 0.30,
    }
    for index, potential_id in enumerate(resource_p90_minimums):
        values = np.asarray(resources[index][land], dtype=np.float64)
        p90 = _weighted_quantile(values, land_weights, 0.90)
        add(
            f"land_{potential_id}_resource_p90",
            "resource_distribution",
            "global_land",
            p90,
            "index",
            gate_kind="earth_structure",
            minimum=resource_p90_minimums[potential_id],
            maximum=0.95,
            reference_scope="Nondegenerate physical suitability, not realized land use",
            reference_ids=vegetation_references,
            applicable=profile_is_earthlike,
            note="Amplitude gate is deliberately broad; directional climate gates carry meaning.",
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
        "model": "earth_reference_functional_vegetation_validation_v1",
        "reference_profile_version": REFERENCE_PROFILE_VERSION,
        "references": list(EARTH_REFERENCE_PROFILE),
        "profile_applicable": int(profile_is_earthlike),
        "profile_semantics": "potential_natural_functional_cover_not_modern_land_use",
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
        "reportable_climate_strata": [
            zone_id for zone_id, summary in zone_summaries.items() if bool(summary["reportable"])
        ],
        "resource_calibration_semantics": (
            "broad amplitude and directional physical suitability; no modern land-use fit"
        ),
    }
    context.logger.log_event(
        {
            "type": "functional_vegetation_validation_summary",
            "stage": "functional_vegetation_validation",
            **metadata,
        }
    )
    return {
        "FunctionalVegetationKpiCatalog": catalog,
        "FunctionalVegetationClimateDistributionCatalog": zone_catalog,
        "FunctionalVegetationValidationMetadata": metadata,
    }


__all__ = [
    "EARTH_REFERENCE_PROFILE",
    "FunctionalVegetationValidationConfig",
    "REFERENCE_PROFILE_VERSION",
    "functional_vegetation_validation_stage",
]
