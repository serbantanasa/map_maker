"""Derived familiar-biome mixtures over causal functional vegetation state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from PIL import Image

from .._derived_biomes_native import BIOME_COUNT, run_derived_biomes
from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult
from .climate import _cube_net_rgb, _palette
from .sea_level import _equirectangular_rgb

if TYPE_CHECKING:
    from ..execution import PipelineContext


BIOMES = (
    {
        "index": 0,
        "code": 1,
        "class_id": "tropical_rainforest",
        "label": "Tropical rainforest",
        "group": "forest",
        "color": (26, 91, 55),
    },
    {
        "index": 1,
        "code": 2,
        "class_id": "tropical_seasonal_forest",
        "label": "Tropical seasonal forest",
        "group": "forest",
        "color": (75, 119, 60),
    },
    {
        "index": 2,
        "code": 3,
        "class_id": "savanna",
        "label": "Savanna",
        "group": "warm_open",
        "color": (183, 157, 69),
    },
    {
        "index": 3,
        "code": 4,
        "class_id": "hot_desert",
        "label": "Hot desert",
        "group": "dryland",
        "color": (215, 177, 103),
    },
    {
        "index": 4,
        "code": 5,
        "class_id": "xeric_shrubland",
        "label": "Xeric shrubland",
        "group": "dryland",
        "color": (151, 126, 78),
    },
    {
        "index": 5,
        "code": 6,
        "class_id": "temperate_forest",
        "label": "Temperate forest",
        "group": "forest",
        "color": (66, 112, 75),
    },
    {
        "index": 6,
        "code": 7,
        "class_id": "temperate_grassland",
        "label": "Temperate grassland",
        "group": "cool_open",
        "color": (137, 162, 84),
    },
    {
        "index": 7,
        "code": 8,
        "class_id": "steppe",
        "label": "Steppe",
        "group": "dryland",
        "color": (178, 151, 87),
    },
    {
        "index": 8,
        "code": 9,
        "class_id": "boreal_forest",
        "label": "Boreal forest",
        "group": "forest",
        "color": (58, 99, 87),
    },
    {
        "index": 9,
        "code": 10,
        "class_id": "tundra",
        "label": "Tundra",
        "group": "cold_open",
        "color": (145, 156, 139),
    },
    {
        "index": 10,
        "code": 11,
        "class_id": "cold_desert",
        "label": "Cold desert",
        "group": "dryland",
        "color": (177, 166, 145),
    },
    {
        "index": 11,
        "code": 12,
        "class_id": "alpine",
        "label": "Alpine",
        "group": "highland",
        "color": (126, 133, 138),
    },
    {
        "index": 12,
        "code": 13,
        "class_id": "wetland",
        "label": "Wetland",
        "group": "wetland",
        "color": (44, 126, 120),
    },
)

LANDSCAPES = (
    {
        "code": 0,
        "class_id": "ocean",
        "label": "Ocean",
        "group": "outside_terrestrial_partition",
        "color": (35, 78, 109),
    },
    {
        "code": 14,
        "class_id": "inland_open_water",
        "label": "Inland open water",
        "group": "aquatic",
        "color": (48, 112, 153),
    },
    {
        "code": 15,
        "class_id": "persistent_ice",
        "label": "Persistent ice",
        "group": "cryosphere",
        "color": (230, 239, 241),
    },
)

CATALOG_SCHEMA = pa.schema(
    [
        ("category", pa.string()),
        ("axis_index", pa.int16()),
        ("code", pa.int16()),
        ("class_id", pa.string()),
        ("label", pa.string()),
        ("group", pa.string()),
        ("red", pa.uint8()),
        ("green", pa.uint8()),
        ("blue", pa.uint8()),
        ("semantics", pa.string()),
    ]
)


@dataclass(frozen=True)
class DerivedBiomeConfig:
    highland_elevation_start_m: float = 1_000.0
    highland_elevation_full_m: float = 3_000.0
    highland_relief_start_m: float = 250.0
    highland_relief_full_m: float = 800.0
    minimum_classifiable_ground_fraction: float = 0.05
    ambiguity_margin_threshold: float = 0.12
    transition_confidence_weight: float = 0.45
    maximum_partition_absolute_error: float = 1e-6

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "DerivedBiomeConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown derived-biome controls: {', '.join(sorted(unknown))}")
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
        if config.highland_elevation_full_m <= config.highland_elevation_start_m:
            raise ValueError("highland_elevation_full_m must exceed its start")
        if config.highland_relief_full_m <= config.highland_relief_start_m:
            raise ValueError("highland_relief_full_m must exceed its start")
        for name in (
            "minimum_classifiable_ground_fraction",
            "ambiguity_margin_threshold",
            "transition_confidence_weight",
        ):
            if not 0.0 <= getattr(config, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if config.maximum_partition_absolute_error <= 0.0:
            raise ValueError("maximum_partition_absolute_error must be positive")
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
    rows: list[dict[str, object]] = []
    for item in BIOMES:
        red, green, blue = cast(tuple[int, int, int], item["color"])
        rows.append(
            {
                "category": "biome",
                "axis_index": item["index"],
                "code": item["code"],
                "class_id": item["class_id"],
                "label": item["label"],
                "group": item["group"],
                "red": red,
                "green": green,
                "blue": blue,
                "semantics": "derived full-cell ecological-ground fraction",
            }
        )
    for item in LANDSCAPES:
        red, green, blue = cast(tuple[int, int, int], item["color"])
        rows.append(
            {
                "category": "landscape",
                "axis_index": -1,
                "code": item["code"],
                "class_id": item["class_id"],
                "label": item["label"],
                "group": item["group"],
                "red": red,
                "green": green,
                "blue": blue,
                "semantics": "reserved dominant-landscape code",
            }
        )
    return pa.Table.from_pylist(rows, schema=CATALOG_SCHEMA)


def _area_weighted_diagnostics(
    fractions: np.ndarray,
    dominant: np.ndarray,
    landscape: np.ndarray,
    areas: np.ndarray,
    land: np.ndarray,
) -> dict[str, object]:
    weights = np.asarray(areas[land], dtype=np.float64)
    land_area = float(np.sum(weights, dtype=np.float64))
    biome_means: dict[str, float] = {}
    dominant_area: dict[str, float] = {}
    for item in BIOMES:
        index = cast(int, item["index"])
        code = cast(int, item["code"])
        class_id = str(item["class_id"])
        biome_means[class_id] = float(
            np.sum(np.asarray(fractions[index][land], dtype=np.float64) * weights)
            / max(land_area, 1e-30)
        )
        dominant_area[class_id] = float(
            np.sum(weights[dominant[land] == code], dtype=np.float64) / max(land_area, 1e-30)
        )
    landscape_area = {
        str(item["class_id"]): float(
            np.sum(weights[landscape[land] == cast(int, item["code"])], dtype=np.float64)
            / max(land_area, 1e-30)
        )
        for item in LANDSCAPES[1:]
    }
    return {
        "land_mean_biome_fractions": biome_means,
        "dominant_biome_land_area_fractions": dominant_area,
        "dominant_abiotic_landscape_land_area_fractions": landscape_area,
    }


def _visualizer(
    result: StageResult, request: VisualizationRequest
) -> list[VisualizationResult] | None:
    fractions = _result_array(result, "BiomeFractions")
    landscape = _result_array(result, "DominantLandscapeCode")
    transition = _result_array(result, "BiomeTransitionIndex")
    if fractions is None or landscape is None or transition is None:
        return None

    colors = np.zeros((16, 3), dtype=np.uint8)
    for item in (*BIOMES, *LANDSCAPES):
        colors[cast(int, item["code"])] = cast(tuple[int, int, int], item["color"])
    dominant_rgb = dominant_landscape_rgb(landscape)

    float_colors = np.asarray(
        [cast(tuple[int, int, int], item["color"]) for item in BIOMES], dtype=np.float64
    )
    mixture_rgb = np.einsum(
        "ifrc,ij->frcj",
        np.asarray(fractions, dtype=np.float64),
        float_colors,
        optimize=True,
    )
    inland_water = np.asarray(landscape) == 14
    persistent_ice = np.asarray(landscape) == 15
    ocean = np.asarray(landscape) == 0
    mixture_rgb[inland_water] = colors[14]
    mixture_rgb[persistent_ice] = colors[15]
    mixture_rgb[ocean] = colors[0]
    mixture_rgb = np.clip(mixture_rgb, 0.0, 255.0).astype(np.uint8)
    transition_rgb = _palette(
        transition,
        (
            (0.0, (45, 62, 54)),
            (0.35, (116, 143, 91)),
            (0.65, (203, 172, 91)),
            (1.0, (172, 78, 68)),
        ),
    )

    outputs = []
    renderings = (
        (
            "dominant_biomes.png",
            _cube_net_rgb(dominant_rgb),
            "DominantLandscapeCode",
            {"palette": "BiomeCatalog", "projection": "cube_net"},
        ),
        (
            "dominant_biomes_global.png",
            _equirectangular_rgb(dominant_rgb),
            "DominantLandscapeCode",
            {"palette": "BiomeCatalog", "projection": "equirectangular"},
        ),
        (
            "biome_mixture.png",
            _cube_net_rgb(mixture_rgb),
            "BiomeFractions",
            {"blend": "area_weighted_biome_palette", "projection": "cube_net"},
        ),
        (
            "biome_mixture_global.png",
            _equirectangular_rgb(mixture_rgb),
            "BiomeFractions",
            {"blend": "area_weighted_biome_palette", "projection": "equirectangular"},
        ),
        (
            "biome_transitions.png",
            _cube_net_rgb(transition_rgb),
            "BiomeTransitionIndex",
            {"scale": [0.0, 1.0], "projection": "cube_net"},
        ),
        (
            "biome_transitions_global.png",
            _equirectangular_rgb(transition_rgb),
            "BiomeTransitionIndex",
            {"scale": [0.0, 1.0], "projection": "equirectangular"},
        ),
    )
    for filename, image, artifact, metadata in renderings:
        output = request.output_dir / filename
        Image.fromarray(image, mode="RGB").save(output)
        outputs.append(
            VisualizationResult(
                output,
                artifact,
                {"model": "causal_derived_biome_mixture_v1", **metadata},
            )
        )
    return outputs


def dominant_landscape_rgb(landscape: np.ndarray) -> np.ndarray:
    """Map persisted landscape codes to the canonical biome palette."""

    colors = np.zeros((16, 3), dtype=np.uint8)
    for item in (*BIOMES, *LANDSCAPES):
        colors[cast(int, item["code"])] = cast(tuple[int, int, int], item["color"])
    codes = np.asarray(landscape, dtype=np.uint8)
    if np.any(codes > 15):
        raise ValueError("landscape contains an unknown palette code")
    return colors[codes]


@stage(
    "derived_biomes",
    inputs=(
        "functional_vegetation_validation",
        "functional_vegetation",
        "potential_biosphere",
        "climate",
        "surface_materials",
        "elevation",
        "sea_level",
    ),
    outputs=(
        "BiomeFractions",
        "BiomeClassificationConfidence",
        "BiomeDominanceMargin",
        "BiomeTransitionIndex",
        "DominantBiomeCode",
        "SecondaryBiomeCode",
        "DominantLandscapeCode",
        "BiomeCatalog",
        "DerivedBiomeMetadata",
    ),
    version="v2",
    native_libraries=("derived_biomes_native",),
    visualizer=_visualizer,
)
def derived_biomes_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = DerivedBiomeConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("derived biomes require topology: cubed_sphere")
    validation_metadata = _artifact_mapping(
        deps["functional_vegetation_validation"],
        "FunctionalVegetationValidationMetadata",
    )
    if validation_metadata.get("hard_gate_pass") != 1:
        raise RuntimeError("derived biomes require passing functional-vegetation hard gates")

    shape = context.topology.face_shape
    output_shapes = {
        "BiomeFractions": (BIOME_COUNT, *shape),
        "BiomeClassificationConfidence": shape,
        "BiomeDominanceMargin": shape,
        "BiomeTransitionIndex": shape,
        "DominantBiomeCode": shape,
        "SecondaryBiomeCode": shape,
        "DominantLandscapeCode": shape,
    }
    code_outputs = {"DominantBiomeCode", "SecondaryBiomeCode", "DominantLandscapeCode"}
    handles = {
        name: context.arena.allocate_array(
            f"derived_biomes_{name.lower()}",
            output_shape,
            np.dtype(np.uint8 if name in code_outputs else np.float32),
        )
        for name, output_shape in output_shapes.items()
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}
    functional = deps["functional_vegetation"]
    potential = deps["potential_biosphere"]
    climate = deps["climate"]
    with context.timed("derived_biome_mixture_kernel"):
        native_metadata = run_derived_biomes(
            highland_elevation_start_m=config.highland_elevation_start_m,
            highland_elevation_full_m=config.highland_elevation_full_m,
            highland_relief_start_m=config.highland_relief_start_m,
            highland_relief_full_m=config.highland_relief_full_m,
            minimum_classifiable_ground_fraction=(config.minimum_classifiable_ground_fraction),
            ambiguity_margin_threshold=config.ambiguity_margin_threshold,
            transition_confidence_weight=config.transition_confidence_weight,
            areas=np.ascontiguousarray(context.topology.cell_areas, dtype=np.float64),
            ocean=np.ascontiguousarray(
                _artifact_array(deps["sea_level"], "SurfaceOceanMask"), dtype=np.float32
            ),
            annual_temperature=np.ascontiguousarray(
                _artifact_array(climate, "AnnualMeanTemperatureC"), dtype=np.float32
            ),
            annual_precipitation=np.ascontiguousarray(
                _artifact_array(climate, "AnnualPrecipitationMm"), dtype=np.float32
            ),
            growing_season=np.ascontiguousarray(
                _artifact_array(potential, "GrowingSeasonFraction"), dtype=np.float32
            ),
            seasonality=np.ascontiguousarray(
                _artifact_array(potential, "ProductivitySeasonalityIndex"), dtype=np.float32
            ),
            drought=np.ascontiguousarray(
                _artifact_array(potential, "DroughtAdaptationPressure"), dtype=np.float32
            ),
            waterlogging=np.ascontiguousarray(
                _artifact_array(potential, "WaterloggingAdaptationPressure"), dtype=np.float32
            ),
            biosphere_confidence=np.ascontiguousarray(
                _artifact_array(potential, "PotentialBiosphereConfidence"), dtype=np.float32
            ),
            functional_confidence=np.ascontiguousarray(
                _artifact_array(functional, "FunctionalVegetationConfidence"), dtype=np.float32
            ),
            wetland_fraction=np.ascontiguousarray(
                _artifact_array(deps["surface_materials"], "EffectiveWetlandFraction"),
                dtype=np.float32,
            ),
            elevation=np.ascontiguousarray(
                _artifact_array(deps["sea_level"], "SurfaceElevationM"), dtype=np.float32
            ),
            relief=np.ascontiguousarray(
                _artifact_array(deps["elevation"], "TerrainReliefM"), dtype=np.float32
            ),
            functional_type_fractions=np.ascontiguousarray(
                _artifact_array(functional, "FunctionalTypeFractions"), dtype=np.float32
            ),
            nonvegetated_fractions=np.ascontiguousarray(
                _artifact_array(functional, "NonVegetatedFractions"), dtype=np.float32
            ),
            resource_potentials=np.ascontiguousarray(
                _artifact_array(functional, "FunctionalResourcePotentials"), dtype=np.float32
            ),
            biome_fractions_out=views["BiomeFractions"],
            classification_confidence_out=views["BiomeClassificationConfidence"],
            dominance_margin_out=views["BiomeDominanceMargin"],
            transition_index_out=views["BiomeTransitionIndex"],
            primary_biome_code_out=views["DominantBiomeCode"],
            secondary_biome_code_out=views["SecondaryBiomeCode"],
            dominant_landscape_code_out=views["DominantLandscapeCode"],
        )

    fractions = np.asarray(views["BiomeFractions"], dtype=np.float64)
    confidence = np.asarray(views["BiomeClassificationConfidence"], dtype=np.float64)
    margin = np.asarray(views["BiomeDominanceMargin"], dtype=np.float64)
    transition = np.asarray(views["BiomeTransitionIndex"], dtype=np.float64)
    dominant = np.asarray(views["DominantBiomeCode"], dtype=np.uint8)
    secondary = np.asarray(views["SecondaryBiomeCode"], dtype=np.uint8)
    landscape = np.asarray(views["DominantLandscapeCode"], dtype=np.uint8)
    ocean = _artifact_array(deps["sea_level"], "SurfaceOceanMask") >= 0.5
    land = ~ocean
    nonvegetated = _artifact_array(functional, "NonVegetatedFractions")
    ice = np.asarray(nonvegetated[2], dtype=np.float64)
    water = np.asarray(nonvegetated[3], dtype=np.float64)
    ground = np.maximum(0.0, 1.0 - ice - water)

    bounded_outputs = (fractions, confidence, margin, transition)
    if any(np.any(~np.isfinite(values)) for values in bounded_outputs):
        raise RuntimeError("derived biomes produced non-finite output")
    if any(np.any(values < 0.0) or np.any(values > 1.0) for values in bounded_outputs):
        raise RuntimeError("derived biomes produced values outside [0, 1]")
    if any(np.any(values[..., ocean] != 0.0) for values in bounded_outputs):
        raise RuntimeError("derived-biome state is nonzero over ocean")
    if any(np.any(values[ocean] != 0) for values in (dominant, secondary, landscape)):
        raise RuntimeError("derived-biome codes are nonzero over ocean")

    partition = np.sum(fractions, axis=0) + ice + water
    partition_error = float(np.max(np.abs(partition[land] - 1.0)))
    if partition_error > config.maximum_partition_absolute_error:
        raise RuntimeError("derived-biome land partition does not close")
    if np.any(np.abs(np.sum(fractions, axis=0)[land] - ground[land]) > 1e-6):
        raise RuntimeError("biome fractions do not equal ecological ground support")
    classifiable = land & (ground >= config.minimum_classifiable_ground_fraction)
    if np.any((dominant[classifiable] < 1) | (dominant[classifiable] > BIOME_COUNT)):
        raise RuntimeError("derived biomes emitted an unknown dominant biome code")
    if np.any((secondary[classifiable] < 1) | (secondary[classifiable] > BIOME_COUNT)):
        raise RuntimeError("derived biomes emitted an unknown secondary biome code")
    if np.any(dominant[classifiable] == secondary[classifiable]):
        raise RuntimeError("dominant and secondary biome codes must differ")
    expected_dominant = np.argmax(fractions, axis=0).astype(np.uint8) + 1
    if np.any(expected_dominant[classifiable] != dominant[classifiable]):
        raise RuntimeError("dominant biome code disagrees with biome fractions")
    if np.any((landscape[land] < 1) | (landscape[land] > 15)):
        raise RuntimeError("derived biomes emitted an unknown landscape code")

    for handle in handles.values():
        handle.seal()
    metadata = {
        **native_metadata,
        **asdict(config),
        **_area_weighted_diagnostics(
            fractions,
            dominant,
            landscape,
            np.asarray(context.topology.cell_areas, dtype=np.float64),
            land,
        ),
        "model": "causal_derived_biome_mixture_v1",
        "topology": "cubed_sphere",
        "biome_axis": [item["class_id"] for item in BIOMES],
        "fraction_semantics": "derived_full_cell_ecological_ground_fraction",
        "transition_semantics": "normalized_entropy_of_conditional_biome_mixture",
        "confidence_semantics": "upstream_confidence_discounted_by_transition_and_ground_support",
        "maximum_partition_absolute_error": partition_error,
        "canonical_state": 0,
        "derived_interpretation": 1,
        "named_biomes_implemented": 1,
        "vegetation_feedback_implemented": 0,
        "earth_profile_status": validation_metadata["earth_profile_status"],
        "hard_gate_pass": 1,
    }
    context.logger.log_event(
        {"type": "derived_biome_summary", "stage": "derived_biomes", **metadata}
    )
    return {
        **handles,
        "BiomeCatalog": _catalog(),
        "DerivedBiomeMetadata": metadata,
    }


__all__ = [
    "BIOMES",
    "DerivedBiomeConfig",
    "derived_biomes_stage",
    "dominant_landscape_rgb",
]
