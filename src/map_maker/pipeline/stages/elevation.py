"""Causal pre-erosion elevation and orogenic morphology."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from PIL import Image

from .._elevation_native import run_cubed_sphere_elevation
from ..cubed_sphere import CubedSphereGrid
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult


@dataclass(frozen=True)
class ElevationConfig:
    collision_height_m: float = 5200.0
    arc_height_m: float = 2800.0
    ridge_height_m: float = 1800.0
    trench_depth_m: float = 3600.0
    rift_depth_m: float = 950.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "ElevationConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown elevation controls: {', '.join(sorted(unknown))}")
        values = {name: float(mapping[name]) for name in known if name in mapping}
        config = cls(**values)
        for name, value in asdict(config).items():
            if not np.isfinite(value) or value < 0.0 or value > 20_000.0:
                raise ValueError(f"{name} must be finite and in [0, 20000]")
        return config


def _artifact_array(result, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _result_array(result, name: str) -> np.ndarray | None:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        return None
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _hotspot_event_grid(result, shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    grid = np.zeros(shape, dtype=np.float32)
    record = result.artifact_records.get("HotspotEvents")
    if record is None or record.value is None:
        raise KeyError("Missing dependency artifact 'HotspotEvents'")
    table = record.value
    required = {"global_cell_id", "strength", "plume_factor"}
    if not required.issubset(set(getattr(table, "column_names", ()))):
        raise ValueError("HotspotEvents lacks canonical cubed-sphere event columns")
    global_ids = np.asarray(table["global_cell_id"].to_numpy(), dtype=np.int64)
    strengths = np.asarray(table["strength"].to_numpy(), dtype=np.float32)
    plume = np.asarray(table["plume_factor"].to_numpy(), dtype=np.float32)
    valid = (global_ids >= 0) & (global_ids < grid.size)
    values = strengths[valid] * (0.65 + 0.35 * plume[valid])
    np.maximum.at(grid.reshape(-1), global_ids[valid], values)
    return grid, int(np.count_nonzero(valid))


def _cube_net_rgb(faces: np.ndarray) -> np.ndarray:
    resolution = faces.shape[1]
    net = np.zeros((resolution * 3, resolution * 4, 3), dtype=np.uint8)
    placements = {3: (1, 0), 0: (1, 1), 2: (1, 2), 1: (1, 3), 4: (0, 1), 5: (2, 1)}
    for face, (net_row, net_col) in placements.items():
        row = net_row * resolution
        col = net_col * resolution
        net[row : row + resolution, col : col + resolution] = faces[face]
    return net


def _palette(
    values: np.ndarray, stops: tuple[tuple[float, tuple[int, int, int]], ...]
) -> np.ndarray:
    positions = np.array([position for position, _ in stops], dtype=np.float32)
    colors = np.array([color for _, color in stops], dtype=np.float32)
    channels = [np.interp(values, positions, colors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=-1).clip(0, 255).astype(np.uint8)


def _elevation_visualizer(
    result, request: VisualizationRequest
) -> list[VisualizationResult] | None:
    bedrock = _result_array(result, "BedrockElevationM")
    crustal = _result_array(result, "CrustalElevationM")
    orogenic = _result_array(result, "OrogenicElevationM")
    basin = _result_array(result, "BasinDepressionM")
    relief = _result_array(result, "TerrainReliefM")
    if (
        bedrock is None
        or crustal is None
        or orogenic is None
        or basin is None
        or relief is None
        or bedrock.ndim != 3
        or bedrock.shape[0] != 6
    ):
        return None

    ocean_rgb = _palette(
        bedrock,
        (
            (-7500.0, (5, 20, 54)),
            (-5000.0, (15, 61, 112)),
            (-3000.0, (37, 112, 158)),
            (-500.0, (100, 177, 196)),
            (1000.0, (155, 205, 208)),
        ),
    )
    land_rgb = _palette(
        bedrock,
        (
            (-2000.0, (45, 92, 64)),
            (-200.0, (54, 111, 73)),
            (500.0, (95, 137, 74)),
            (1500.0, (157, 143, 92)),
            (3000.0, (137, 109, 83)),
            (5000.0, (210, 205, 190)),
            (7500.0, (252, 252, 250)),
        ),
    )
    inherited_land = crustal > -1000.0
    elevation_rgb = np.where(inherited_land[..., None], land_rgb, ocean_rgb)
    elevation_path = request.output_dir / "bedrock_elevation.png"
    Image.fromarray(_cube_net_rgb(elevation_rgb), mode="RGB").save(elevation_path)

    morphology_rgb = np.zeros((*orogenic.shape, 3), dtype=np.uint8)
    morphology_rgb[..., 0] = np.clip(orogenic / 5000.0 * 255.0, 0.0, 255.0).astype(np.uint8)
    morphology_rgb[..., 1] = np.clip(relief / 1800.0 * 190.0, 0.0, 190.0).astype(np.uint8)
    morphology_rgb[..., 2] = np.clip(basin / 4000.0 * 255.0, 0.0, 255.0).astype(np.uint8)
    morphology_path = request.output_dir / "orogenic_morphology.png"
    Image.fromarray(_cube_net_rgb(morphology_rgb), mode="RGB").save(morphology_path)

    uplift_rgb = _palette(
        orogenic,
        (
            (0.0, (3, 12, 10)),
            (250.0, (32, 88, 62)),
            (1000.0, (181, 125, 55)),
            (3000.0, (224, 211, 174)),
            (5500.0, (255, 255, 252)),
        ),
    )
    uplift_path = request.output_dir / "orogenic_elevation.png"
    Image.fromarray(_cube_net_rgb(uplift_rgb), mode="RGB").save(uplift_path)

    depression_rgb = _palette(
        basin,
        (
            (0.0, (4, 9, 16)),
            (250.0, (17, 68, 91)),
            (1000.0, (31, 124, 158)),
            (2500.0, (112, 188, 203)),
            (5000.0, (218, 241, 239)),
        ),
    )
    depression_path = request.output_dir / "basin_depression.png"
    Image.fromarray(_cube_net_rgb(depression_rgb), mode="RGB").save(depression_path)

    return [
        VisualizationResult(elevation_path, "BedrockElevationM", {"scale": "fixed_meters"}),
        VisualizationResult(
            morphology_path,
            "OrogenicElevationM",
            {"red": "orogenic", "green": "relief", "blue": "basin"},
        ),
        VisualizationResult(uplift_path, "OrogenicElevationM", {"scale": "fixed_meters"}),
        VisualizationResult(depression_path, "BasinDepressionM", {"scale": "fixed_meters"}),
    ]


@stage(
    "elevation",
    inputs=("tectonics", "world_age", "geology"),
    outputs=(
        "CrustalElevationM",
        "OrogenicElevationM",
        "BasinDepressionM",
        "BedrockElevationM",
        "TerrainReliefM",
        "ElevationConfidence",
        "ElevationMetadata",
    ),
    version="v2",
    native_libraries=("elevation_native",),
    visualizer=_elevation_visualizer,
)
def elevation_stage(context, deps, config_mapping: Mapping[str, object]):
    config = ElevationConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("canonical elevation requires topology: cubed_sphere")

    shape = context.topology.face_shape
    handles = {
        "CrustalElevationM": context.arena.allocate_array(
            "elevation_crustal_m", shape, dtype=np.float32
        ),
        "OrogenicElevationM": context.arena.allocate_array(
            "elevation_orogenic_m", shape, dtype=np.float32
        ),
        "BasinDepressionM": context.arena.allocate_array(
            "elevation_basin_m", shape, dtype=np.float32
        ),
        "BedrockElevationM": context.arena.allocate_array(
            "elevation_bedrock_m", shape, dtype=np.float32
        ),
        "TerrainReliefM": context.arena.allocate_array(
            "elevation_relief_m", shape, dtype=np.float32
        ),
        "ElevationConfidence": context.arena.allocate_array(
            "elevation_confidence", shape, dtype=np.float32
        ),
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}
    tectonics = deps["tectonics"]
    world_age = deps["world_age"]
    geology = deps["geology"]
    seed = int(context.rng("elevation").integers(0, 2**63 - 1))
    hotspot_events, hotspot_event_count = _hotspot_event_grid(world_age, shape)

    with context.timed("elevation_kernel"):
        metadata = run_cubed_sphere_elevation(
            seed=seed,
            **asdict(config),
            areas=context.topology.cell_areas,
            neighbors=context.topology.neighbor_indices,
            plate_field=_artifact_array(tectonics, "PlateField"),
            crust_thickness=_artifact_array(world_age, "CrustThickness"),
            isostasy=_artifact_array(world_age, "IsostaticOffset"),
            uplift=_artifact_array(world_age, "UpliftRate"),
            subsidence=_artifact_array(world_age, "SubsidenceRate"),
            compression=_artifact_array(world_age, "TectonicCompression"),
            extension=_artifact_array(world_age, "TectonicExtension"),
            shear=_artifact_array(world_age, "ShearMagnitude"),
            stiffness=_artifact_array(world_age, "LithosphereStiffness"),
            proto_ocean=_artifact_array(world_age, "BaseOceanMask"),
            hotspot=hotspot_events,
            crust_age=_artifact_array(geology, "CrustAgeGa"),
            rock_strength=_artifact_array(geology, "RockStrength"),
            accommodation=_artifact_array(geology, "SedimentAccommodation"),
            province_confidence=_artifact_array(geology, "ProvinceConfidence"),
            boundary_regime=_artifact_array(geology, "BoundaryRegime"),
            boundary_confidence=_artifact_array(geology, "BoundaryConfidence"),
            crustal_out=views["CrustalElevationM"],
            orogenic_out=views["OrogenicElevationM"],
            basin_out=views["BasinDepressionM"],
            bedrock_out=views["BedrockElevationM"],
            relief_out=views["TerrainReliefM"],
            confidence_out=views["ElevationConfidence"],
        )

    for handle in handles.values():
        handle.seal()
    metadata.update(
        {
            **asdict(config),
            "rng_seed": seed,
            "hotspot_event_count": hotspot_event_count,
            "topology": "cubed_sphere",
            "datum": "provisional_pre_sea_level_zero",
            "model": "causal_pre_erosion_components_v2",
            "history_semantics": "initial_morphology_not_eroded_present_day",
        }
    )
    context.logger.log_event({"type": "elevation_summary", "stage": "elevation", **metadata})
    return {**handles, "ElevationMetadata": metadata}


__all__ = ["ElevationConfig", "elevation_stage"]
