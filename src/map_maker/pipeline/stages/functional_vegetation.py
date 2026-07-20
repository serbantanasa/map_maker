"""Conservative functional-vegetation mixtures derived from potential traits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from PIL import Image

from .._functional_vegetation_native import (
    FUNCTIONAL_TYPE_COUNT,
    NONVEGETATED_TYPE_COUNT,
    RESOURCE_POTENTIAL_COUNT,
    run_functional_vegetation,
)
from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult
from .climate import _cube_net_rgb

if TYPE_CHECKING:
    from ..execution import PipelineContext


FUNCTIONAL_TYPES = (
    {
        "index": 0,
        "code": 1,
        "class_id": "cold_woody",
        "label": "Cold-adapted woody",
        "group": "woody",
        "color": (48, 91, 73),
    },
    {
        "index": 1,
        "code": 2,
        "class_id": "warm_evergreen_woody",
        "label": "Warm evergreen woody",
        "group": "woody",
        "color": (35, 122, 75),
    },
    {
        "index": 2,
        "code": 3,
        "class_id": "seasonal_woody",
        "label": "Seasonal woody",
        "group": "woody",
        "color": (105, 132, 68),
    },
    {
        "index": 3,
        "code": 4,
        "class_id": "xeric_shrub",
        "label": "Xeric shrub",
        "group": "shrub",
        "color": (158, 132, 77),
    },
    {
        "index": 4,
        "code": 5,
        "class_id": "cool_season_herbaceous",
        "label": "Cool-season herbaceous",
        "group": "herbaceous",
        "color": (116, 157, 86),
    },
    {
        "index": 5,
        "code": 6,
        "class_id": "warm_season_herbaceous",
        "label": "Warm-season herbaceous",
        "group": "herbaceous",
        "color": (187, 166, 73),
    },
    {
        "index": 6,
        "code": 7,
        "class_id": "hydrophytic",
        "label": "Hydrophytic",
        "group": "wetland",
        "color": (47, 132, 130),
    },
    {
        "index": 7,
        "code": 8,
        "class_id": "low_stature_conservative",
        "label": "Low-stature resource-conservative",
        "group": "low_stature",
        "color": (126, 137, 112),
    },
)

NONVEGETATED_TYPES = (
    {
        "index": 0,
        "code": 9,
        "class_id": "bare_ground",
        "label": "Bare ground",
        "group": "nonvegetated",
        "color": (151, 128, 97),
    },
    {
        "index": 1,
        "code": 10,
        "class_id": "saline_barren",
        "label": "Saline barren",
        "group": "nonvegetated",
        "color": (203, 190, 164),
    },
    {
        "index": 2,
        "code": 11,
        "class_id": "persistent_ice",
        "label": "Glacier or persistent ice",
        "group": "nonvegetated",
        "color": (224, 236, 238),
    },
    {
        "index": 3,
        "code": 12,
        "class_id": "inland_open_water",
        "label": "Inland open water",
        "group": "nonvegetated",
        "color": (59, 116, 159),
    },
    {
        "index": 4,
        "code": 13,
        "class_id": "unsupported_surface",
        "label": "Unsupported or soil-free surface",
        "group": "nonvegetated",
        "color": (111, 108, 103),
    },
)

RESOURCE_POTENTIALS = (
    {"index": 0, "potential_id": "fire_tendency", "label": "Fire tendency"},
    {"index": 1, "potential_id": "grazing", "label": "Grazing potential"},
    {
        "index": 2,
        "potential_id": "forest_resource",
        "label": "Forest resource potential",
    },
    {"index": 3, "potential_id": "pasture", "label": "Pasture potential"},
    {"index": 4, "potential_id": "crop", "label": "Crop potential"},
)

CATALOG_SCHEMA = pa.schema(
    [
        ("category", pa.string()),
        ("axis_index", pa.int16()),
        ("dominant_code", pa.int16()),
        ("class_id", pa.string()),
        ("label", pa.string()),
        ("group", pa.string()),
        ("semantics", pa.string()),
    ]
)


@dataclass(frozen=True)
class FunctionalVegetationConfig:
    warm_transition_midpoint_c: float = 18.0
    warm_transition_width_c: float = 12.0
    npp_response_half_saturation_kg_c_m2_year: float = 0.25
    biomass_response_half_saturation_kg_c_m2: float = 5.0
    terrain_relief_half_saturation_m: float = 500.0
    crop_soil_depth_half_saturation_m: float = 0.5
    strategy_confidence_multiplier: float = 0.75
    maximum_partition_absolute_error: float = 1e-6

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "FunctionalVegetationConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(
                f"Unknown functional-vegetation controls: {', '.join(sorted(unknown))}"
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
        positive = (
            "warm_transition_width_c",
            "npp_response_half_saturation_kg_c_m2_year",
            "biomass_response_half_saturation_kg_c_m2",
            "terrain_relief_half_saturation_m",
            "crop_soil_depth_half_saturation_m",
            "maximum_partition_absolute_error",
        )
        for name in positive:
            if getattr(config, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= config.strategy_confidence_multiplier <= 1.0:
            raise ValueError("strategy_confidence_multiplier must be in [0, 1]")
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


def _result_array(result: StageResult, name: str) -> np.ndarray | None:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        return None
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _catalog() -> pa.Table:
    rows: list[dict[str, object]] = [
        {
            "category": "ocean",
            "axis_index": -1,
            "dominant_code": 0,
            "class_id": "ocean",
            "label": "Ocean",
            "group": "outside_terrestrial_partition",
            "semantics": "reserved dominant-cover code; no terrestrial fraction",
        }
    ]
    rows.extend(
        {
            "category": "functional_type",
            "axis_index": item["index"],
            "dominant_code": item["code"],
            "class_id": item["class_id"],
            "label": item["label"],
            "group": item["group"],
            "semantics": "physical fraction of full cell area",
        }
        for item in FUNCTIONAL_TYPES
    )
    rows.extend(
        {
            "category": "nonvegetated",
            "axis_index": item["index"],
            "dominant_code": item["code"],
            "class_id": item["class_id"],
            "label": item["label"],
            "group": item["group"],
            "semantics": "physical fraction of full cell area",
        }
        for item in NONVEGETATED_TYPES
    )
    rows.extend(
        {
            "category": "resource_potential",
            "axis_index": item["index"],
            "dominant_code": -1,
            "class_id": item["potential_id"],
            "label": item["label"],
            "group": "bounded_potential",
            "semantics": "dimensionless physical suitability; not actual land use",
        }
        for item in RESOURCE_POTENTIALS
    )
    return pa.Table.from_pylist(rows, schema=CATALOG_SCHEMA)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(values[order[min(index, values.size - 1)]])


def _resource_diagnostics(
    resources: np.ndarray, areas: np.ndarray, land: np.ndarray
) -> dict[str, float]:
    weights = np.asarray(areas[land], dtype=np.float64)
    total_area = float(np.sum(weights, dtype=np.float64))
    diagnostics: dict[str, float] = {}
    for item in RESOURCE_POTENTIALS:
        potential_id = str(item["potential_id"])
        values = np.asarray(resources[cast(int, item["index"])][land], dtype=np.float64)
        prefix = f"land_{potential_id}"
        diagnostics[f"{prefix}_area_weighted_mean"] = float(
            np.sum(values * weights, dtype=np.float64) / total_area
        )
        diagnostics[f"{prefix}_area_weighted_p90"] = _weighted_quantile(values, weights, 0.90)
        diagnostics[f"{prefix}_maximum"] = float(np.max(values))
        diagnostics[f"{prefix}_area_fraction_at_least_0_5"] = float(
            np.sum(weights[values >= 0.5], dtype=np.float64) / total_area
        )
    return diagnostics


def _visualizer(
    result: StageResult, request: VisualizationRequest
) -> list[VisualizationResult] | None:
    functional = _result_array(result, "FunctionalTypeFractions")
    nonvegetated = _result_array(result, "NonVegetatedFractions")
    resources = _result_array(result, "FunctionalResourcePotentials")
    dominant = _result_array(result, "DominantFunctionalCoverCode")
    if any(value is None for value in (functional, nonvegetated, resources, dominant)):
        return None
    assert functional is not None
    assert nonvegetated is not None
    assert resources is not None
    assert dominant is not None

    colors = np.zeros((14, 3), dtype=np.uint8)
    colors[0] = (27, 61, 91)
    for item in (*FUNCTIONAL_TYPES, *NONVEGETATED_TYPES):
        colors[cast(int, item["code"])] = cast(tuple[int, int, int], item["color"])
    dominant_rgb = colors[np.asarray(dominant, dtype=np.uint8)]
    woody = np.sum(functional[0:3], axis=0)
    herbaceous = np.sum(functional[4:6], axis=0)
    hydrophytic = functional[6]
    mixture_rgb = (
        np.stack((herbaceous + 0.35 * functional[3], woody, hydrophytic), axis=-1).clip(0.0, 1.0)
        * 255.0
    ).astype(np.uint8)
    resource_rgb = (
        np.stack((resources[0], resources[4], resources[1]), axis=-1).clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)

    outputs = []
    for filename, image, artifact, metadata in (
        (
            "dominant_functional_cover.png",
            dominant_rgb,
            "DominantFunctionalCoverCode",
            {"palette": "FunctionalVegetationCatalog"},
        ),
        (
            "functional_mixture.png",
            mixture_rgb,
            "FunctionalTypeFractions",
            {"red": "herbaceous", "green": "woody", "blue": "hydrophytic"},
        ),
        (
            "functional_resource_potentials.png",
            resource_rgb,
            "FunctionalResourcePotentials",
            {"red": "fire", "green": "crop", "blue": "grazing"},
        ),
    ):
        output = request.output_dir / filename
        Image.fromarray(_cube_net_rgb(image), mode="RGB").save(output)
        outputs.append(
            VisualizationResult(
                output,
                artifact,
                {"model": "functional_vegetation_mixture_v4", **metadata},
            )
        )
    return outputs


@stage(
    "functional_vegetation",
    inputs=(
        "biosphere_validation",
        "potential_biosphere",
        "surface_materials",
        "climate",
        "cryosphere",
        "elevation",
        "world_age",
    ),
    outputs=(
        "FunctionalTypeFractions",
        "NonVegetatedFractions",
        "FunctionalResourcePotentials",
        "FunctionalVegetationConfidence",
        "DominantFunctionalCoverCode",
        "FunctionalVegetationCatalog",
        "FunctionalVegetationMetadata",
    ),
    version="v4",
    native_libraries=("functional_vegetation_native",),
    visualizer=_visualizer,
)
def functional_vegetation_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = FunctionalVegetationConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("functional vegetation requires topology: cubed_sphere")
    validation_metadata = _artifact_mapping(
        deps["biosphere_validation"], "BiosphereValidationMetadata"
    )
    if validation_metadata.get("hard_gate_pass") != 1:
        raise RuntimeError("functional vegetation requires passing biosphere hard gates")

    shape = context.topology.face_shape
    output_shapes = {
        "FunctionalTypeFractions": (FUNCTIONAL_TYPE_COUNT, *shape),
        "NonVegetatedFractions": (NONVEGETATED_TYPE_COUNT, *shape),
        "FunctionalResourcePotentials": (RESOURCE_POTENTIAL_COUNT, *shape),
        "FunctionalVegetationConfidence": shape,
        "DominantFunctionalCoverCode": shape,
    }
    handles = {
        name: context.arena.allocate_array(
            f"functional_vegetation_{name.lower()}",
            output_shape,
            np.dtype(np.uint8 if name == "DominantFunctionalCoverCode" else np.float32),
        )
        for name, output_shape in output_shapes.items()
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}
    potential = deps["potential_biosphere"]
    materials = deps["surface_materials"]
    with context.timed("functional_vegetation_mixture_kernel"):
        native_metadata = run_functional_vegetation(
            warm_transition_midpoint_c=config.warm_transition_midpoint_c,
            warm_transition_width_c=config.warm_transition_width_c,
            npp_response_half_saturation_kg_c_m2_year=(
                config.npp_response_half_saturation_kg_c_m2_year
            ),
            biomass_response_half_saturation_kg_c_m2=(
                config.biomass_response_half_saturation_kg_c_m2
            ),
            terrain_relief_half_saturation_m=config.terrain_relief_half_saturation_m,
            crop_soil_depth_half_saturation_m=config.crop_soil_depth_half_saturation_m,
            strategy_confidence_multiplier=config.strategy_confidence_multiplier,
            areas=np.ascontiguousarray(context.topology.cell_areas, dtype=np.float64),
            ocean=np.ascontiguousarray(
                _artifact_array(deps["world_age"], "BaseOceanMask"), dtype=np.float32
            ),
            vegetation_cover=np.ascontiguousarray(
                _artifact_array(potential, "PotentialVegetationCoverFraction"), dtype=np.float32
            ),
            annual_npp=np.ascontiguousarray(
                _artifact_array(potential, "AnnualPotentialNPPKgCM2"), dtype=np.float32
            ),
            standing_biomass=np.ascontiguousarray(
                _artifact_array(potential, "PotentialStandingBiomassKgCM2"), dtype=np.float32
            ),
            growing_season=np.ascontiguousarray(
                _artifact_array(potential, "GrowingSeasonFraction"), dtype=np.float32
            ),
            productivity_seasonality=np.ascontiguousarray(
                _artifact_array(potential, "ProductivitySeasonalityIndex"), dtype=np.float32
            ),
            drought_pressure=np.ascontiguousarray(
                _artifact_array(potential, "DroughtAdaptationPressure"), dtype=np.float32
            ),
            cold_pressure=np.ascontiguousarray(
                _artifact_array(potential, "ColdAdaptationPressure"), dtype=np.float32
            ),
            heat_pressure=np.ascontiguousarray(
                _artifact_array(potential, "HeatAdaptationPressure"), dtype=np.float32
            ),
            waterlogging_pressure=np.ascontiguousarray(
                _artifact_array(potential, "WaterloggingAdaptationPressure"), dtype=np.float32
            ),
            salinity_pressure=np.ascontiguousarray(
                _artifact_array(potential, "SalinityAdaptationPressure"), dtype=np.float32
            ),
            woody_trait=np.ascontiguousarray(
                _artifact_array(potential, "PotentialWoodyAllocationTrait"), dtype=np.float32
            ),
            resource_conservative_trait=np.ascontiguousarray(
                _artifact_array(potential, "PotentialResourceConservativeTrait"),
                dtype=np.float32,
            ),
            fuel_continuity=np.ascontiguousarray(
                _artifact_array(potential, "PotentialFuelContinuityIndex"), dtype=np.float32
            ),
            biosphere_confidence=np.ascontiguousarray(
                _artifact_array(potential, "PotentialBiosphereConfidence"), dtype=np.float32
            ),
            annual_temperature=np.ascontiguousarray(
                _artifact_array(deps["climate"], "AnnualMeanTemperatureC"), dtype=np.float32
            ),
            soil_fertility=np.ascontiguousarray(
                _artifact_array(materials, "SoilFertilityPotential"), dtype=np.float32
            ),
            soil_depth=np.ascontiguousarray(
                _artifact_array(materials, "SoilDepthM"), dtype=np.float32
            ),
            soil_bearing=np.ascontiguousarray(
                _artifact_array(materials, "SoilBearingFraction"), dtype=np.float32
            ),
            soil_drainage=np.ascontiguousarray(
                _artifact_array(materials, "SoilDrainageIndex"), dtype=np.float32
            ),
            glacier_fraction=np.ascontiguousarray(
                _artifact_array(deps["cryosphere"], "GlacierIceFraction"), dtype=np.float32
            ),
            lake_fraction=np.ascontiguousarray(
                _artifact_array(materials, "EffectiveLakeFraction"), dtype=np.float32
            ),
            wetland_fraction=np.ascontiguousarray(
                _artifact_array(materials, "EffectiveWetlandFraction"), dtype=np.float32
            ),
            terrain_relief=np.ascontiguousarray(
                _artifact_array(deps["elevation"], "TerrainReliefM"), dtype=np.float32
            ),
            functional_type_fractions_out=views["FunctionalTypeFractions"],
            nonvegetated_fractions_out=views["NonVegetatedFractions"],
            resource_potentials_out=views["FunctionalResourcePotentials"],
            confidence_out=views["FunctionalVegetationConfidence"],
            dominant_cover_code_out=views["DominantFunctionalCoverCode"],
        )

    functional = np.asarray(views["FunctionalTypeFractions"], dtype=np.float64)
    nonvegetated = np.asarray(views["NonVegetatedFractions"], dtype=np.float64)
    resources = np.asarray(views["FunctionalResourcePotentials"], dtype=np.float64)
    confidence = np.asarray(views["FunctionalVegetationConfidence"], dtype=np.float64)
    dominant = np.asarray(views["DominantFunctionalCoverCode"], dtype=np.uint8)
    ocean = _artifact_array(deps["world_age"], "BaseOceanMask") >= 0.5
    land = ~ocean
    bounded_outputs = (functional, nonvegetated, resources, confidence)
    if any(np.any(~np.isfinite(values)) for values in bounded_outputs):
        raise RuntimeError("functional vegetation produced non-finite output")
    if any(np.any(values < 0.0) or np.any(values > 1.0) for values in bounded_outputs):
        raise RuntimeError("functional vegetation produced values outside [0, 1]")
    if any(np.any(values[..., ocean] != 0.0) for values in (functional, nonvegetated, resources)):
        raise RuntimeError("terrestrial functional vegetation is nonzero over ocean")
    if np.any(confidence[ocean] != 0.0) or np.any(dominant[ocean] != 0):
        raise RuntimeError("ocean confidence or dominant code is nonzero")

    partition = np.sum(functional, axis=0) + np.sum(nonvegetated, axis=0)
    partition_error = float(np.max(np.abs(partition[land] - 1.0)))
    if partition_error > config.maximum_partition_absolute_error:
        raise RuntimeError("functional vegetation land-cover partition does not close")
    upstream_cover = _artifact_array(potential, "PotentialVegetationCoverFraction")
    if np.any(np.sum(functional, axis=0)[land] > upstream_cover[land] + 1e-6):
        raise RuntimeError("functional vegetation exceeds upstream potential cover")
    if np.any((dominant[land] < 1) | (dominant[land] > 13)):
        raise RuntimeError("functional vegetation emitted an unknown dominant code")

    vegetated_fraction = np.sum(functional, axis=0)
    dominant_strategy = np.argmax(functional, axis=0).astype(np.uint8) + 1
    dominant_nonvegetated = np.argmax(nonvegetated, axis=0).astype(np.uint8) + 9
    maximum_nonvegetated = np.max(nonvegetated, axis=0)
    expected_code = np.where(
        vegetated_fraction > maximum_nonvegetated,
        dominant_strategy,
        dominant_nonvegetated,
    ).astype(np.uint8)
    expected_code[ocean] = 0
    dominant_mismatch_count = int(np.count_nonzero(expected_code != dominant))
    if dominant_mismatch_count:
        raise RuntimeError("dominant functional-cover code disagrees with physical fractions")

    for handle in handles.values():
        handle.seal()
    metadata = {
        **native_metadata,
        **asdict(config),
        "model": "functional_vegetation_mixture_v4",
        "topology": "cubed_sphere",
        "fraction_semantics": "physical_fraction_of_full_coarse_cell_area",
        "functional_type_axis": [item["class_id"] for item in FUNCTIONAL_TYPES],
        "nonvegetated_axis": [item["class_id"] for item in NONVEGETATED_TYPES],
        "resource_potential_axis": [item["potential_id"] for item in RESOURCE_POTENTIALS],
        **_resource_diagnostics(resources, context.topology.cell_areas, land),
        "maximum_partition_absolute_error": partition_error,
        "dominant_cover_mismatch_count": dominant_mismatch_count,
        "upstream_earth_profile_status": validation_metadata["earth_profile_status"],
        "hard_gate_pass": 1,
        "functional_vegetation_ready_for_derived_biomes": 1,
        "actual_land_use_implemented": 0,
        "biome_labels_implemented": 0,
        "vegetation_feedback_implemented": 0,
        "strategy_semantics": "continuous_producer_community_functional_mixture",
        "resource_semantics": "physical_suitability_not_actual_land_use",
        "dominant_code_semantics": (
            "aggregate_vegetation_vs_each_nonvegetated_class_then_dominant_strategy"
        ),
    }
    context.logger.log_event(
        {"type": "functional_vegetation_summary", "stage": "functional_vegetation", **metadata}
    )
    return {
        **handles,
        "FunctionalVegetationCatalog": _catalog(),
        "FunctionalVegetationMetadata": metadata,
    }


__all__ = [
    "FUNCTIONAL_TYPES",
    "NONVEGETATED_TYPES",
    "RESOURCE_POTENTIALS",
    "FunctionalVegetationConfig",
    "functional_vegetation_stage",
]
