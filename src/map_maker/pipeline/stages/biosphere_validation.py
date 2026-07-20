"""Earth-reference diagnostics for the trait-first potential biosphere."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage

if TYPE_CHECKING:
    from ..execution import PipelineContext


EARTH_RADIUS_M = 6_371_008.8
REFERENCE_PROFILE_VERSION = "earth_biosphere_v1"
EARTHLIKE_LAND_FRACTION_RANGE = (0.18, 0.36)

EARTH_REFERENCE_PROFILE = (
    {
        "reference_id": "nasa_earth_surface",
        "title": "NASA facts about Earth",
        "url": "https://science.nasa.gov/earth/facts/",
        "supports": ["global_land_surface_fraction"],
        "summary": "Earth's global ocean covers about 71% of the planetary surface.",
    },
    {
        "reference_id": "tian_2014_global_npp",
        "title": "Contemporary global terrestrial net primary production",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4234638/",
        "supports": ["global_potential_npp_pg_c_year", "land_mean_potential_npp_kg_c_m2_year"],
        "summary": "Contemporary terrestrial NPP was estimated at 52.8-56.4 Pg C/year.",
    },
    {
        "reference_id": "erb_2018_potential_biomass",
        "title": "Potential and actual global vegetation biomass",
        "url": "https://doi.org/10.1038/nature25138",
        "supports": [
            "global_potential_biomass_pg_c",
            "land_mean_potential_biomass_kg_c_m2",
        ],
        "summary": "Potential vegetation stores 771-1,107 Pg C under current climate.",
    },
    {
        "reference_id": "ornl_npp_multibiome",
        "title": "ORNL DAAC multi-biome NPP observations",
        "url": "https://doi.org/10.3334/ORNLDAAC/1352",
        "supports": ["climate_stratum_productivity_distributions"],
        "summary": "Field NPP and biomass observations span tundra through tropical forests.",
    },
    {
        "reference_id": "modis_vcf_c61",
        "title": "MODIS Vegetation Continuous Fields Collection 6.1",
        "url": "https://modis-land.gsfc.nasa.gov/ValStatus.php?ProductID=MOD44",
        "supports": ["potential_vegetation_cover_distributions"],
        "summary": "Subpixel tree, non-tree vegetation, and non-vegetated cover at 250 m.",
    },
    {
        "reference_id": "try_trait_database",
        "title": "TRY plant trait database",
        "url": "https://www.try-db.org/",
        "supports": ["trait_distributions"],
        "summary": "Plant trait observations for later climate-conditioned calibration.",
    },
)

# These are deliberately climate-response evaluation strata, not biome labels or a
# claim to implement Koppen classification. They partition every finite land cell
# using only upstream climate state, so biosphere output cannot move the goalposts.
CLIMATE_STRATA = (
    {
        "zone_id": "polar",
        "label": "Polar",
        "description": "annual mean temperature below -5 C",
    },
    {
        "zone_id": "cold",
        "label": "Cold",
        "description": "annual mean temperature from -5 C to below 5 C",
    },
    {
        "zone_id": "cool_dry",
        "label": "Cool dry",
        "description": "5-18 C and annual precipitation below 500 mm",
    },
    {
        "zone_id": "cool_moist",
        "label": "Cool moist",
        "description": "5-18 C and annual precipitation at least 500 mm",
    },
    {
        "zone_id": "warm_dry",
        "label": "Warm dry",
        "description": "at least 18 C and annual precipitation below 500 mm",
    },
    {
        "zone_id": "warm_seasonal",
        "label": "Warm seasonal",
        "description": "at least 18 C and annual precipitation from 500 to below 1,500 mm",
    },
    {
        "zone_id": "warm_humid",
        "label": "Warm humid",
        "description": "at least 18 C and annual precipitation at least 1,500 mm",
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
class BiosphereValidationConfig:
    minimum_reportable_zone_land_fraction: float = 0.005
    maximum_partition_relative_error: float = 1e-12
    maximum_reconstruction_relative_error: float = 1e-8

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "BiosphereValidationConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown biosphere-validation controls: {', '.join(sorted(unknown))}")
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
        if config.maximum_partition_relative_error <= 0.0:
            raise ValueError("maximum_partition_relative_error must be positive")
        if config.maximum_reconstruction_relative_error <= 0.0:
            raise ValueError("maximum_reconstruction_relative_error must be positive")
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


def _numeric(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def classify_climate_strata(
    annual_temperature_c: np.ndarray,
    annual_precipitation_mm: np.ndarray,
    land: np.ndarray,
) -> np.ndarray:
    """Return stable upstream-climate stratum IDs; ocean cells remain -1."""

    temperature = np.asarray(annual_temperature_c, dtype=np.float64)
    precipitation = np.asarray(annual_precipitation_mm, dtype=np.float64)
    land_mask = np.asarray(land, dtype=bool)
    if temperature.shape != precipitation.shape or temperature.shape != land_mask.shape:
        raise ValueError("temperature, precipitation, and land masks must share a shape")
    if np.any(~np.isfinite(temperature[land_mask])) or np.any(
        ~np.isfinite(precipitation[land_mask])
    ):
        raise ValueError("land climate inputs must be finite")

    zones = np.full(temperature.shape, -1, dtype=np.int8)
    zones[land_mask & (temperature < -5.0)] = 0
    zones[land_mask & (temperature >= -5.0) & (temperature < 5.0)] = 1
    cool = land_mask & (temperature >= 5.0) & (temperature < 18.0)
    zones[cool & (precipitation < 500.0)] = 2
    zones[cool & (precipitation >= 500.0)] = 3
    warm = land_mask & (temperature >= 18.0)
    zones[warm & (precipitation < 500.0)] = 4
    zones[warm & (precipitation >= 500.0) & (precipitation < 1_500.0)] = 5
    zones[warm & (precipitation >= 1_500.0)] = 6
    return zones


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(np.sum(weights))
    return float(np.sum(np.asarray(values, dtype=np.float64) * weights) / max(weight_sum, 1e-30))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    samples = np.asarray(values, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if samples.size == 0 or float(np.sum(sample_weights)) <= 0.0:
        return 0.0
    order = np.argsort(samples, kind="stable")
    sorted_values = samples[order]
    sorted_weights = sample_weights[order]
    centers = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    target = quantile * float(np.sum(sorted_weights))
    return float(
        np.interp(target, centers, sorted_values, left=sorted_values[0], right=sorted_values[-1])
    )


def _climate_distribution_catalog(
    *,
    zones: np.ndarray,
    land: np.ndarray,
    areas_m2: np.ndarray,
    minimum_reportable_fraction: float,
    fields: Mapping[str, tuple[np.ndarray, str]],
) -> tuple[pa.Table, dict[str, dict[str, float]]]:
    land_area = float(np.sum(areas_m2[land]))
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, float]] = {}
    for zone_index, definition in enumerate(CLIMATE_STRATA):
        mask = land & (zones == zone_index)
        weights = areas_m2[mask]
        area_m2 = float(np.sum(weights))
        area_fraction = area_m2 / max(land_area, 1e-30)
        reportable = area_fraction >= minimum_reportable_fraction
        zone_id = str(definition["zone_id"])
        summaries[zone_id] = {"land_area_fraction": area_fraction}
        for metric_id, (field, unit) in fields.items():
            values = np.asarray(field, dtype=np.float64)[mask]
            statistics = {
                "mean": _weighted_mean(values, weights) if values.size else 0.0,
                "p10": _weighted_quantile(values, weights, 0.10),
                "p50": _weighted_quantile(values, weights, 0.50),
                "p90": _weighted_quantile(values, weights, 0.90),
            }
            summaries[zone_id][f"{metric_id}_mean"] = statistics["mean"]
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
    return pa.Table.from_pylist(rows, schema=CLIMATE_DISTRIBUTION_SCHEMA), summaries


@stage(
    "biosphere_validation",
    inputs=("potential_biosphere", "climate", "sea_level", "planet"),
    outputs=(
        "BiosphereKpiCatalog",
        "BiosphereClimateDistributionCatalog",
        "BiosphereValidationMetadata",
    ),
    version="v5",
)
def biosphere_validation_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = BiosphereValidationConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("biosphere validation requires topology: cubed_sphere")

    potential = deps["potential_biosphere"]
    climate = deps["climate"]
    sea_level = deps["sea_level"]
    planet = deps["planet"]
    potential_metadata = _artifact_mapping(potential, "PotentialBiosphereMetadata")
    planet_metadata = _artifact_mapping(planet, "PlanetMetadata")
    profile_is_earthlike = potential_metadata.get("validation_profile") == "earthlike"

    ocean = _artifact_array(sea_level, "SurfaceOceanMask") >= 0.5
    land = ~ocean
    annual_temperature = _artifact_array(climate, "AnnualMeanTemperatureC")
    annual_precipitation = _artifact_array(climate, "AnnualPrecipitationMm")
    fields = {
        "annual_potential_npp": (
            _artifact_array(potential, "AnnualPotentialNPPKgCM2"),
            "kg C/m2/year",
        ),
        "potential_standing_biomass": (
            _artifact_array(potential, "PotentialStandingBiomassKgCM2"),
            "kg C/m2",
        ),
        "potential_vegetation_cover": (
            _artifact_array(potential, "PotentialVegetationCoverFraction"),
            "fraction",
        ),
        "growing_season": (
            _artifact_array(potential, "GrowingSeasonFraction"),
            "fraction",
        ),
        "woody_allocation": (
            _artifact_array(potential, "PotentialWoodyAllocationTrait"),
            "fraction",
        ),
        "canopy_height": (
            _artifact_array(potential, "PotentialCanopyHeightM"),
            "m",
        ),
        "leaf_area_index": (
            _artifact_array(potential, "PotentialLeafAreaIndex"),
            "m2/m2",
        ),
        "rooting_depth": (
            _artifact_array(potential, "PotentialRootingDepthM"),
            "m",
        ),
        "biosphere_confidence": (
            _artifact_array(potential, "PotentialBiosphereConfidence"),
            "fraction",
        ),
        "annual_mean_temperature": (annual_temperature, "C"),
        "annual_precipitation": (annual_precipitation, "mm/year"),
    }
    finite_nonnegative = all(
        np.all(np.isfinite(field[land])) and np.all(field[land] >= 0.0)
        for name, (field, _) in fields.items()
        if name not in {"annual_mean_temperature"}
    ) and np.all(np.isfinite(annual_temperature[land]))

    radius_m = EARTH_RADIUS_M * _numeric(
        planet_metadata["planet_radius_earth"], name="planet_radius_earth"
    )
    areas_m2 = np.asarray(context.topology.cell_areas, dtype=np.float64) * radius_m**2
    total_area_m2 = float(np.sum(areas_m2))
    land_area_m2 = float(np.sum(areas_m2[land]))
    land_fraction = land_area_m2 / max(total_area_m2, 1e-30)
    npp = np.asarray(fields["annual_potential_npp"][0], dtype=np.float64)
    biomass = np.asarray(fields["potential_standing_biomass"][0], dtype=np.float64)
    cover = np.asarray(fields["potential_vegetation_cover"][0], dtype=np.float64)
    growing_season = np.asarray(fields["growing_season"][0], dtype=np.float64)
    woody = np.asarray(fields["woody_allocation"][0], dtype=np.float64)
    canopy = np.asarray(fields["canopy_height"][0], dtype=np.float64)
    leaf_area = np.asarray(fields["leaf_area_index"][0], dtype=np.float64)
    rooting_depth = np.asarray(fields["rooting_depth"][0], dtype=np.float64)
    confidence = np.asarray(fields["biosphere_confidence"][0], dtype=np.float64)

    global_npp_kg_c_year = float(np.sum(npp[land] * areas_m2[land]))
    global_biomass_kg_c = float(np.sum(biomass[land] * areas_m2[land]))
    land_mean_npp = global_npp_kg_c_year / max(land_area_m2, 1e-30)
    land_mean_biomass = global_biomass_kg_c / max(land_area_m2, 1e-30)
    zones = classify_climate_strata(annual_temperature, annual_precipitation, land)
    zone_catalog, zone_summaries = _climate_distribution_catalog(
        zones=zones,
        land=land,
        areas_m2=areas_m2,
        minimum_reportable_fraction=config.minimum_reportable_zone_land_fraction,
        fields=fields,
    )

    zone_area_sum = sum(summary["land_area_fraction"] for summary in zone_summaries.values())
    partition_error = abs(zone_area_sum - 1.0)
    reconstructed_npp = sum(
        summary["land_area_fraction"] * summary["annual_potential_npp_mean"]
        for summary in zone_summaries.values()
    )
    reconstructed_biomass = sum(
        summary["land_area_fraction"] * summary["potential_standing_biomass_mean"]
        for summary in zone_summaries.values()
    )
    npp_reconstruction_error = abs(reconstructed_npp - land_mean_npp) / max(land_mean_npp, 1e-30)
    biomass_reconstruction_error = abs(reconstructed_biomass - land_mean_biomass) / max(
        land_mean_biomass, 1e-30
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
        elif gate_kind in {"earth_diagnostic", "earth_structure"} and (
            minimum is not None or maximum is not None
        ):
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
        "finite_nonnegative_biosphere_fields",
        "numerical",
        "global_land",
        int(finite_nonnegative),
        "boolean",
        gate_kind="hard_invariant",
        minimum=1.0,
    )
    add(
        "climate_stratum_partition_relative_error",
        "accounting",
        "global_land",
        partition_error,
        "relative",
        gate_kind="hard_invariant",
        maximum=config.maximum_partition_relative_error,
    )
    add(
        "zone_npp_reconstruction_relative_error",
        "accounting",
        "global_land",
        npp_reconstruction_error,
        "relative",
        gate_kind="hard_invariant",
        maximum=config.maximum_reconstruction_relative_error,
    )
    add(
        "zone_biomass_reconstruction_relative_error",
        "accounting",
        "global_land",
        biomass_reconstruction_error,
        "relative",
        gate_kind="hard_invariant",
        maximum=config.maximum_reconstruction_relative_error,
    )

    add(
        "global_land_surface_fraction",
        "surface",
        "global",
        land_fraction,
        "fraction",
        minimum=EARTHLIKE_LAND_FRACTION_RANGE[0],
        maximum=EARTHLIKE_LAND_FRACTION_RANGE[1],
        reference_scope="Generated Earthlike worlds; observed Earth baseline around 29% land",
        reference_ids=("nasa_earth_surface",),
        note="Narrow Earth baseline; other named world profiles may differ freely.",
        gate_kind="earth_diagnostic",
        applicable=profile_is_earthlike,
    )
    add(
        "global_potential_npp_pg_c_year",
        "productivity",
        "global_land",
        global_npp_kg_c_year / 1e12,
        "Pg C/year",
        minimum=50.0,
        maximum=75.0,
        reference_scope="Earth-size potential natural terrestrial vegetation",
        reference_ids=("tian_2014_global_npp", "ornl_npp_multibiome"),
        note="Broadens contemporary observed NPP to allow potential vegetation without land use.",
        gate_kind="earth_diagnostic",
        applicable=profile_is_earthlike,
    )
    add(
        "land_mean_potential_npp_kg_c_m2_year",
        "productivity",
        "global_land",
        land_mean_npp,
        "kg C/m2/year",
        minimum=0.28,
        maximum=0.55,
        reference_scope="Earthlike generated land area under potential natural vegetation",
        reference_ids=("tian_2014_global_npp", "ornl_npp_multibiome"),
        note="Normalized range includes the approved 18-36% generated land-area envelope.",
        gate_kind="earth_diagnostic",
        applicable=profile_is_earthlike,
    )
    add(
        "global_potential_biomass_pg_c",
        "biomass",
        "global_land",
        global_biomass_kg_c / 1e12,
        "Pg C",
        minimum=771.0,
        maximum=1_107.0,
        reference_scope="Potential vegetation under current Earth climate without land use",
        reference_ids=("erb_2018_potential_biomass",),
        gate_kind="earth_diagnostic",
        applicable=profile_is_earthlike,
    )
    add(
        "land_mean_potential_biomass_kg_c_m2",
        "biomass",
        "global_land",
        land_mean_biomass,
        "kg C/m2",
        minimum=4.2,
        maximum=8.1,
        reference_scope="Potential biomass normalized by generated Earthlike land area",
        reference_ids=("erb_2018_potential_biomass",),
        note="Normalized range includes the approved 18-36% generated land-area envelope.",
        gate_kind="earth_diagnostic",
        applicable=profile_is_earthlike,
    )

    global_diagnostics = (
        ("land_mean_potential_vegetation_cover_fraction", cover, "fraction"),
        ("potentially_vegetated_land_fraction", (cover >= 0.10).astype(np.float64), "fraction"),
        ("land_mean_growing_season_fraction", growing_season, "fraction"),
        ("land_mean_woody_allocation_trait", woody, "fraction"),
        ("land_mean_canopy_height_m", canopy, "m"),
        ("land_mean_leaf_area_index", leaf_area, "m2/m2"),
        ("land_mean_rooting_depth_m", rooting_depth, "m"),
        ("land_mean_biosphere_confidence", confidence, "fraction"),
    )
    for kpi_id, field, unit in global_diagnostics:
        add(
            kpi_id,
            "distribution",
            "global_land",
            _weighted_mean(field[land], areas_m2[land]),
            unit,
            reference_ids=("modis_vcf_c61", "try_trait_database"),
            note="Reported for distributional calibration; no defensible scalar Earth gate yet.",
        )
    add(
        "reportable_climate_stratum_count",
        "coverage",
        "global_land",
        sum(
            summary["land_area_fraction"] >= config.minimum_reportable_zone_land_fraction
            for summary in zone_summaries.values()
        ),
        "count",
    )

    def add_relationship(
        kpi_id: str,
        numerator_zone: str,
        denominator_zone: str,
        metric: str,
        minimum: float,
    ) -> None:
        numerator = zone_summaries[numerator_zone]
        denominator = zone_summaries[denominator_zone]
        reportable = all(
            summary["land_area_fraction"] >= config.minimum_reportable_zone_land_fraction
            for summary in (numerator, denominator)
        )
        ratio = numerator[f"{metric}_mean"] / max(denominator[f"{metric}_mean"], 0.01)
        add(
            kpi_id,
            "climate_response",
            f"{numerator_zone}_vs_{denominator_zone}",
            ratio,
            "ratio",
            gate_kind="earth_structure",
            minimum=minimum,
            reference_scope="directional Earth climate-productivity relationship",
            reference_ids=("ornl_npp_multibiome", "modis_vcf_c61"),
            applicable=profile_is_earthlike and reportable,
            note="Skipped when either climate stratum covers too little generated land.",
        )

    add_relationship(
        "warm_humid_to_warm_dry_npp_ratio",
        "warm_humid",
        "warm_dry",
        "annual_potential_npp",
        2.0,
    )
    add_relationship(
        "cool_moist_to_cool_dry_npp_ratio",
        "cool_moist",
        "cool_dry",
        "annual_potential_npp",
        1.5,
    )
    add_relationship(
        "warm_humid_to_polar_npp_ratio",
        "warm_humid",
        "polar",
        "annual_potential_npp",
        4.0,
    )
    add_relationship(
        "warm_humid_to_warm_dry_cover_ratio",
        "warm_humid",
        "warm_dry",
        "potential_vegetation_cover",
        1.5,
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
        "model": "earth_reference_potential_biosphere_validation_v1",
        "reference_profile_version": REFERENCE_PROFILE_VERSION,
        "references": list(EARTH_REFERENCE_PROFILE),
        "profile_applicable": int(profile_is_earthlike),
        "profile_semantics": "potential_natural_photosynthetic_terrestrial_biosphere",
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
        "planet_radius_m": radius_m,
        "global_surface_area_km2": total_area_m2 / 1_000_000.0,
        "global_land_area_km2": land_area_m2 / 1_000_000.0,
        "global_land_surface_fraction": land_fraction,
        "global_potential_npp_pg_c_year": global_npp_kg_c_year / 1e12,
        "global_potential_biomass_pg_c": global_biomass_kg_c / 1e12,
        "land_mean_potential_npp_kg_c_m2_year": land_mean_npp,
        "land_mean_potential_biomass_kg_c_m2": land_mean_biomass,
        "reportable_climate_strata": [
            zone_id
            for zone_id, summary in zone_summaries.items()
            if summary["land_area_fraction"] >= config.minimum_reportable_zone_land_fraction
        ],
        "comparison_rule": (
            "hard invariants gate accounting; Earth totals and climate-response relationships "
            "diagnose calibration and never clamp non-Earth scenarios"
        ),
    }
    context.logger.log_event(
        {"type": "biosphere_validation_summary", "stage": "biosphere_validation", **metadata}
    )
    return {
        "BiosphereKpiCatalog": catalog,
        "BiosphereClimateDistributionCatalog": zone_catalog,
        "BiosphereValidationMetadata": metadata,
    }


__all__ = [
    "BiosphereValidationConfig",
    "CLIMATE_STRATA",
    "EARTHLIKE_LAND_FRACTION_RANGE",
    "EARTH_REFERENCE_PROFILE",
    "REFERENCE_PROFILE_VERSION",
    "biosphere_validation_stage",
    "classify_climate_strata",
]
