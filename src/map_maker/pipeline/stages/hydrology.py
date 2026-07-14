"""Depression-aware lakes, breaches, drainage basins, and river reaches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from PIL import Image, ImageDraw

from .._hydrology_native import WATER_BODY_CLASSES, run_cubed_sphere_hydrology
from ..cubed_sphere import CubedSphereGrid
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult

EARTH_RADIUS_M = 6_371_000.0
EARTH_LIKE_LAKE_AREA_FRACTION_RANGE = (0.015, 0.040)


@dataclass(frozen=True)
class HydrologyConfig:
    minimum_depression_depth_m: float = 20.0
    wetland_mean_depth_m: float = 35.0
    endorheic_aridity_threshold: float = 0.35
    maximum_fill_time_years: float = 50_000.0
    lake_seepage_mm_year: float = 30.0
    subgrid_relief_scale: float = 1.0
    subgrid_connected_basin_fraction: float = 0.50
    breach_score_threshold: float = 0.58
    maximum_breach_incision_m: float = 800.0
    breach_length_cells: int = 4
    river_discharge_threshold_m3s: float = 300.0
    river_contributing_area_threshold_km2: float = 200_000.0
    river_minimum_discharge_m3s: float = 25.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "HydrologyConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown hydrology controls: {', '.join(sorted(unknown))}")
        values: dict[str, int | float] = {}
        for name in known:
            if name not in mapping:
                continue
            values[name] = (
                int(mapping[name]) if name == "breach_length_cells" else float(mapping[name])
            )
        config = cls(**values)
        bounds = {
            "minimum_depression_depth_m": (0.1, 2_000.0),
            "wetland_mean_depth_m": (0.1, 500.0),
            "endorheic_aridity_threshold": (0.05, 10.0),
            "maximum_fill_time_years": (1.0, 10_000_000.0),
            "lake_seepage_mm_year": (0.0, 5_000.0),
            "subgrid_relief_scale": (0.1, 8.0),
            "subgrid_connected_basin_fraction": (0.05, 1.0),
            "breach_score_threshold": (0.0, 1.0),
            "maximum_breach_incision_m": (1.0, 5_000.0),
            "breach_length_cells": (1, 64),
            "river_discharge_threshold_m3s": (0.01, 1_000_000.0),
            "river_contributing_area_threshold_km2": (1.0, 100_000_000.0),
            "river_minimum_discharge_m3s": (0.0, 100_000.0),
        }
        for name, (minimum, maximum) in bounds.items():
            value = getattr(config, name)
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
        if config.river_minimum_discharge_m3s > config.river_discharge_threshold_m3s:
            raise ValueError(
                "river_minimum_discharge_m3s must not exceed river_discharge_threshold_m3s"
            )
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


def _cube_net_rgb(faces: np.ndarray) -> np.ndarray:
    resolution = faces.shape[1]
    net = np.zeros((resolution * 3, resolution * 4, 3), dtype=np.uint8)
    placements = {3: (1, 0), 0: (1, 1), 2: (1, 2), 1: (1, 3), 4: (0, 1), 5: (2, 1)}
    for face, (net_row, net_col) in placements.items():
        row = net_row * resolution
        col = net_col * resolution
        net[row : row + resolution, col : col + resolution] = faces[face]
    return net


def _terrain_rgb(elevation: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    land_height = np.clip(elevation / 4_500.0, 0.0, 1.0)
    ocean_depth = np.clip(-elevation / 6_000.0, 0.0, 1.0)
    rgb = np.empty((*elevation.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.where(ocean, 18 + 18 * (1 - ocean_depth), 66 + 150 * land_height)
    rgb[..., 1] = np.where(ocean, 62 + 40 * (1 - ocean_depth), 112 + 90 * land_height)
    rgb[..., 2] = np.where(ocean, 105 + 55 * (1 - ocean_depth), 68 + 90 * land_height)
    return rgb.clip(0, 255).astype(np.uint8)


def _smooth_spherical_polyline(points: np.ndarray, passes: int = 2) -> np.ndarray:
    smoothed = np.asarray(points, dtype=np.float64)
    if len(smoothed) < 2:
        return smoothed.astype(np.float32)
    for _ in range(passes):
        first = 0.75 * smoothed[:-1] + 0.25 * smoothed[1:]
        second = 0.25 * smoothed[:-1] + 0.75 * smoothed[1:]
        combined = np.empty((2 * len(smoothed), 3), dtype=np.float64)
        combined[0] = smoothed[0]
        combined[-1] = smoothed[-1]
        combined[1:-1:2] = first
        combined[2:-1:2] = second
        combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-12)
        smoothed = combined
    return smoothed.astype(np.float32)


def _add_reach_geometry(reaches: pa.Table, xyz: np.ndarray) -> pa.Table:
    flat_xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    polylines = [
        _smooth_spherical_polyline(flat_xyz[np.asarray(path, dtype=np.intp)]).tolist()
        for path in reaches["cell_path"].to_pylist()
    ]
    geometry_type = pa.list_(pa.list_(pa.float32(), 3))
    return reaches.append_column(
        "polyline_on_cubed_sphere", pa.array(polylines, type=geometry_type)
    )


def _project_to_cube_net(
    points: np.ndarray, resolution: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    dominant = np.argmax(np.abs(points), axis=1)
    face = np.empty(len(points), dtype=np.intp)
    face[(dominant == 0) & (points[:, 0] >= 0)] = 0
    face[(dominant == 0) & (points[:, 0] < 0)] = 1
    face[(dominant == 1) & (points[:, 1] >= 0)] = 2
    face[(dominant == 1) & (points[:, 1] < 0)] = 3
    face[(dominant == 2) & (points[:, 2] >= 0)] = 4
    face[(dominant == 2) & (points[:, 2] < 0)] = 5
    normals = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    rights = np.array(
        [[0, 1, 0], [0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
        dtype=np.float64,
    )
    downs = np.array(
        [[0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1], [1, 0, 0], [-1, 0, 0]],
        dtype=np.float64,
    )
    denominator = np.sum(points * normals[face], axis=1)
    u = np.sum(points * rights[face], axis=1) / np.maximum(denominator, 1e-12)
    v = np.sum(points * downs[face], axis=1) / np.maximum(denominator, 1e-12)
    col = (np.arctan(u) / (0.5 * np.pi) + 0.5) * resolution
    row = (np.arctan(v) / (0.5 * np.pi) + 0.5) * resolution
    placements = np.array([[1, 1], [1, 3], [1, 2], [1, 0], [0, 1], [2, 1]])
    return face, placements[face, 0] * resolution + row, placements[face, 1] * resolution + col


def _draw_vector_rivers(base: np.ndarray, reaches: pa.Table, resolution: int) -> np.ndarray:
    supersample = 4
    image = Image.fromarray(_cube_net_rgb(base), mode="RGB").resize(
        (resolution * 4 * supersample, resolution * 3 * supersample),
        Image.Resampling.BICUBIC,
    )
    draw = ImageDraw.Draw(image)
    discharge = np.asarray(reaches["discharge_mean"].combine_chunks())
    maximum_log_q = max(float(np.percentile(np.log1p(discharge), 99.0)), 1e-6)
    geometries = reaches["polyline_on_cubed_sphere"]
    for reach_index in np.argsort(discharge):
        points = np.asarray(geometries[int(reach_index)].as_py(), dtype=np.float64)
        if len(points) < 2:
            continue
        face, row, col = _project_to_cube_net(points, resolution)
        normalized_q = min(float(np.log1p(discharge[reach_index]) / maximum_log_q), 1.0)
        width = max(2, round(supersample * (0.45 + 1.15 * normalized_q)))
        color = (
            round(37 - 8 * normalized_q),
            round(133 - 20 * normalized_q),
            round(205 + 30 * normalized_q),
        )
        segment_start = 0
        for point_index in range(1, len(points) + 1):
            if point_index == len(points) or face[point_index] != face[point_index - 1]:
                if point_index - segment_start >= 2:
                    line = [
                        (float(col[index] * supersample), float(row[index] * supersample))
                        for index in range(segment_start, point_index)
                    ]
                    draw.line(line, fill=color, width=width, joint="curve")
                segment_start = point_index
    return np.asarray(image.resize((resolution * 4, resolution * 3), Image.Resampling.LANCZOS))


def _hydrology_visualizer(
    result, request: VisualizationRequest
) -> list[VisualizationResult] | None:
    elevation = _result_array(result, "HydrologicElevationM")
    water = _result_array(result, "WaterBodyClass")
    lake_fraction = _result_array(result, "LakeFraction")
    wetland_fraction = _result_array(result, "WetlandFraction")
    corridor = _result_array(result, "RiverCorridor")
    floodplain = _result_array(result, "FloodplainPotential")
    basin = _result_array(result, "BasinID")
    discharge = _result_array(result, "MeanDischargeM3s")
    breach = _result_array(result, "BreachIncisionM")
    reach_record = result.artifact_records.get("RiverReachCatalog")
    if (
        any(
            value is None
            for value in (
                elevation,
                water,
                lake_fraction,
                wetland_fraction,
                corridor,
                floodplain,
                basin,
                discharge,
                breach,
            )
        )
        or reach_record is None
        or not isinstance(reach_record.value, pa.Table)
    ):
        return None
    assert elevation is not None
    assert water is not None
    assert lake_fraction is not None
    assert wetland_fraction is not None
    assert corridor is not None
    assert floodplain is not None
    assert basin is not None
    assert discharge is not None
    assert breach is not None

    ocean = basin < 0
    terrain = _terrain_rgb(elevation, ocean)
    water_palette = np.array(
        [
            [0, 0, 0],
            [60, 151, 137],
            [57, 122, 155],
            [49, 112, 171],
            [74, 148, 190],
            [196, 91, 55],
        ],
        dtype=np.uint8,
    )
    water_coverage = np.clip(lake_fraction + wetland_fraction, 0.0, 1.0)[..., None]
    water_color = water_palette[np.clip(water, 0, 5)]
    terrain = (
        terrain.astype(np.float32) * (1.0 - water_coverage)
        + water_color.astype(np.float32) * water_coverage
    ).astype(np.uint8)
    floodplain_alpha = (0.12 * np.clip(floodplain, 0.0, 1.0))[..., None]
    floodplain_color = np.array([88, 145, 103], dtype=np.float32)
    terrain = (
        (
            terrain.astype(np.float32) * (1.0 - floodplain_alpha)
            + floodplain_color * floodplain_alpha
        )
        .clip(0, 255)
        .astype(np.uint8)
    )
    hydro_path = request.output_dir / "lakes_and_rivers.png"
    Image.fromarray(
        _draw_vector_rivers(terrain, reach_record.value, elevation.shape[1]), mode="RGB"
    ).save(hydro_path)

    basin_ids = np.maximum(basin.astype(np.int64), 0)
    basin_rgb = np.empty((*basin.shape, 3), dtype=np.uint8)
    basin_rgb[..., 0] = ((basin_ids * 73 + 61) % 181 + 45).astype(np.uint8)
    basin_rgb[..., 1] = ((basin_ids * 109 + 37) % 171 + 50).astype(np.uint8)
    basin_rgb[..., 2] = ((basin_ids * 47 + 83) % 166 + 55).astype(np.uint8)
    basin_rgb[ocean] = [18, 48, 79]
    basin_path = request.output_dir / "drainage_basins.png"
    Image.fromarray(_cube_net_rgb(basin_rgb), mode="RGB").save(basin_path)

    log_discharge = np.log1p(np.maximum(discharge, 0.0))
    scale = max(float(np.percentile(log_discharge[~ocean], 99.5)), 1e-6)
    normalized_q = np.clip(log_discharge / scale, 0.0, 1.0)
    network_rgb = (
        np.stack(
            (
                30 + 80 * floodplain,
                45 + 155 * normalized_q,
                55 + 195 * np.maximum(normalized_q, corridor),
            ),
            axis=-1,
        )
        .clip(0, 255)
        .astype(np.uint8)
    )
    network_rgb[ocean] = [15, 38, 64]
    network_path = request.output_dir / "discharge_and_floodplains.png"
    Image.fromarray(_cube_net_rgb(network_rgb), mode="RGB").save(network_path)

    positive_breach = breach[breach > 0.0]
    breach_scale = (
        max(float(np.percentile(positive_breach, 95)), 1.0) if positive_breach.size else 1.0
    )
    breach_norm = np.clip(breach / breach_scale, 0.0, 1.0)
    breach_rgb = (
        np.stack(
            (35 + 220 * breach_norm, 35 + 115 * (1 - breach_norm), 45 + 65 * (1 - breach_norm)),
            axis=-1,
        )
        .clip(0, 255)
        .astype(np.uint8)
    )
    breach_rgb[ocean] = [15, 38, 64]
    breach_path = request.output_dir / "breach_incision.png"
    Image.fromarray(_cube_net_rgb(breach_rgb), mode="RGB").save(breach_path)
    return [
        VisualizationResult(hydro_path, "RiverCorridor", {"water_classes": WATER_BODY_CLASSES}),
        VisualizationResult(basin_path, "BasinID", {}),
        VisualizationResult(network_path, "MeanDischargeM3s", {"scale": "log1p"}),
        VisualizationResult(breach_path, "BreachIncisionM", {"scale": "p95_positive"}),
    ]


def _basin_catalog(
    basin_id: np.ndarray,
    sink_type: np.ndarray,
    receiver: np.ndarray,
    mean_discharge: np.ndarray,
    ocean: np.ndarray,
    areas_km2: np.ndarray,
) -> pa.Table:
    flat_basin = basin_id.reshape(-1)
    land = flat_basin >= 0
    count = int(np.max(flat_basin, initial=-1)) + 1
    cell_count = np.bincount(flat_basin[land], minlength=count)
    area = np.bincount(flat_basin[land], weights=areas_km2.reshape(-1)[land], minlength=count)
    flat_receiver = receiver.reshape(-1)
    flat_ocean = ocean.reshape(-1)
    terminals = np.flatnonzero(
        land
        & ((flat_receiver < 0) | ((flat_receiver >= 0) & flat_ocean[np.maximum(flat_receiver, 0)]))
    )
    outlet = np.full(count, -1, dtype=np.int32)
    for cell in terminals:
        outlet[flat_basin[cell]] = cell
    outlet_safe = np.maximum(outlet, 0)
    return pa.table(
        {
            "basin_id": pa.array(np.arange(count, dtype=np.int32), type=pa.int32()),
            "outlet_cell": pa.array(outlet, type=pa.int32()),
            "sink_type": pa.array(
                sink_type.reshape(-1)[outlet_safe].astype(np.uint8), type=pa.uint8()
            ),
            "cell_count": pa.array(cell_count.astype(np.int32), type=pa.int32()),
            "area_km2": pa.array(area, type=pa.float64()),
            "outlet_mean_discharge_m3s": pa.array(
                mean_discharge.reshape(-1)[outlet_safe], type=pa.float32()
            ),
        }
    )


def _drainage_graph(
    receiver: np.ndarray,
    basin_id: np.ndarray,
    sink_type: np.ndarray,
    depression_id: np.ndarray,
    lake_id: np.ndarray,
    lake_fraction: np.ndarray,
    wetland_fraction: np.ndarray,
    contributing_area: np.ndarray,
    mean_discharge: np.ndarray,
    ocean: np.ndarray,
) -> pa.Table:
    land_cells = np.flatnonzero(~ocean.reshape(-1)).astype(np.int32)
    return pa.table(
        {
            "cell_id": pa.array(land_cells, type=pa.int32()),
            "receiver_id": pa.array(receiver.reshape(-1)[land_cells], type=pa.int32()),
            "basin_id": pa.array(basin_id.reshape(-1)[land_cells], type=pa.int32()),
            "sink_type": pa.array(sink_type.reshape(-1)[land_cells], type=pa.uint8()),
            "depression_id": pa.array(depression_id.reshape(-1)[land_cells], type=pa.int32()),
            "lake_id": pa.array(lake_id.reshape(-1)[land_cells], type=pa.int32()),
            "lake_fraction": pa.array(lake_fraction.reshape(-1)[land_cells], type=pa.float32()),
            "wetland_fraction": pa.array(
                wetland_fraction.reshape(-1)[land_cells], type=pa.float32()
            ),
            "contributing_area_km2": pa.array(
                contributing_area.reshape(-1)[land_cells], type=pa.float64()
            ),
            "mean_discharge_m3s": pa.array(
                mean_discharge.reshape(-1)[land_cells], type=pa.float32()
            ),
        }
    )


def _waterbody_cell_catalog(
    depression_id: np.ndarray,
    lake_id: np.ndarray,
    water_class: np.ndarray,
    lake_fraction: np.ndarray,
    wetland_fraction: np.ndarray,
    areas_km2: np.ndarray,
) -> pa.Table:
    flat_lake_fraction = lake_fraction.reshape(-1)
    flat_wetland_fraction = wetland_fraction.reshape(-1)
    coverage = flat_lake_fraction + flat_wetland_fraction
    cells = np.flatnonzero(coverage > 0.0).astype(np.int32)
    classes = water_class.reshape(-1)[cells].astype(np.uint8)
    return pa.table(
        {
            "cell_id": pa.array(cells, type=pa.int32()),
            "waterbody_id": pa.array(depression_id.reshape(-1)[cells], type=pa.int32()),
            "depression_id": pa.array(depression_id.reshape(-1)[cells], type=pa.int32()),
            "lake_id": pa.array(lake_id.reshape(-1)[cells], type=pa.int32()),
            "class_code": pa.array(classes, type=pa.uint8()),
            "class_name": pa.array(
                [WATER_BODY_CLASSES.get(int(code), "unknown") for code in classes],
                type=pa.string(),
            ),
            "lake_fraction": pa.array(flat_lake_fraction[cells], type=pa.float32()),
            "wetland_fraction": pa.array(flat_wetland_fraction[cells], type=pa.float32()),
            "covered_area_km2": pa.array(
                areas_km2.reshape(-1)[cells] * coverage[cells], type=pa.float64()
            ),
        }
    )


ARRAY_DTYPES = {
    "DepressionID": np.int32,
    "LakeID": np.int32,
    "WaterBodyClass": np.uint8,
    "LakeFraction": np.float32,
    "WetlandFraction": np.float32,
    "DepressionFillDepthM": np.float32,
    "HydrologicElevationM": np.float32,
    "BreachIncisionM": np.float32,
    "FlowReceiverID": np.int32,
    "FlowDirectionXYZ": np.float32,
    "FlowSlope": np.float32,
    "ContributingAreaKm2": np.float64,
    "MonthlyDischargeM3s": np.float32,
    "MeanDischargeM3s": np.float32,
    "MeanFlowVelocityMps": np.float32,
    "StreamPowerW": np.float32,
    "BasinID": np.int32,
    "FlowSinkType": np.uint8,
    "RiverCorridor": np.float32,
    "FloodplainPotential": np.float32,
}


@stage(
    "hydrology",
    inputs=("climate", "elevation", "geology", "world_age", "planet"),
    outputs=(
        *ARRAY_DTYPES,
        "DepressionCatalog",
        "LakeCatalog",
        "WetlandCatalog",
        "BreachCatalog",
        "BasinCatalog",
        "DrainageGraph",
        "WaterBodyCellCatalog",
        "RiverReachCatalog",
        "HydrologyMetadata",
    ),
    version="v5",
    native_libraries=("hydrology_native",),
    visualizer=_hydrology_visualizer,
)
def hydrology_stage(context, deps, config_mapping: Mapping[str, object]):
    config = HydrologyConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("canonical hydrology requires topology: cubed_sphere")

    shape = context.topology.face_shape
    artifact_shapes = {
        **{name: shape for name in ARRAY_DTYPES},
        "FlowDirectionXYZ": (*shape, 3),
        "MonthlyDischargeM3s": (12, *shape),
    }
    handles = {
        name: context.arena.allocate_array(
            f"hydrology_{name.lower()}", artifact_shapes[name], dtype=dtype
        )
        for name, dtype in ARRAY_DTYPES.items()
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}
    planet_metadata = deps["planet"].artifact_records["PlanetMetadata"].value
    radius_m = float(planet_metadata["planet_radius_earth"]) * EARTH_RADIUS_M
    radius_km = radius_m / 1_000.0
    physical_areas_km2 = context.topology.cell_areas * radius_km * radius_km
    land_mask = _artifact_array(deps["world_age"], "BaseOceanMask") < 0.5
    median_land_cell_area = float(np.median(physical_areas_km2[land_mask]))
    controls = {
        **asdict(config),
        "planet_radius_m": radius_m,
        "river_contributing_area_threshold_km2": max(
            config.river_contributing_area_threshold_km2, 4.0 * median_land_cell_area
        ),
    }
    controls["breach_length_cells"] = int(config.breach_length_cells)
    ocean = np.ascontiguousarray(~land_mask, dtype=np.uint8)

    with context.timed("depression_aware_hydrology_kernel"):
        depression_catalog, breach_catalog, reach_catalog, metadata = run_cubed_sphere_hydrology(
            controls=controls,
            areas=context.topology.cell_areas,
            neighbors=context.topology.neighbor_indices,
            xyz=context.topology.xyz,
            elevation=_artifact_array(deps["elevation"], "BedrockElevationM"),
            relief=_artifact_array(deps["elevation"], "TerrainReliefM"),
            rock_strength=_artifact_array(deps["geology"], "RockStrength"),
            accommodation=_artifact_array(deps["geology"], "SedimentAccommodation"),
            ocean=ocean,
            runoff=_artifact_array(deps["climate"], "MonthlyRunoffPotentialMm"),
            evaporation=_artifact_array(deps["climate"], "MonthlyEvaporationMm"),
            aridity=_artifact_array(deps["climate"], "AnnualAridityIndex"),
            outputs=views,
        )
    reach_catalog = _add_reach_geometry(reach_catalog, context.topology.xyz)

    for handle in handles.values():
        handle.seal()
    if metadata["topology_valid"] != 1:
        raise RuntimeError("hydrology kernel returned an invalid drainage topology")
    if metadata["conservation_relative_error"] > 1e-6:
        raise RuntimeError(
            "hydrology runoff conservation error exceeds tolerance: "
            f"{metadata['conservation_relative_error']:.3e}"
        )

    registered_lake = pc.greater_equal(depression_catalog["lake_id"], 0)
    wetland_class = pc.equal(depression_catalog["class_code"], 1)
    lake_catalog = depression_catalog.filter(registered_lake)
    wetland_catalog = depression_catalog.filter(wetland_class)
    basin_catalog = _basin_catalog(
        views["BasinID"],
        views["FlowSinkType"],
        views["FlowReceiverID"],
        views["MeanDischargeM3s"],
        ocean.astype(bool),
        physical_areas_km2,
    )
    drainage_graph = _drainage_graph(
        views["FlowReceiverID"],
        views["BasinID"],
        views["FlowSinkType"],
        views["DepressionID"],
        views["LakeID"],
        views["LakeFraction"],
        views["WetlandFraction"],
        views["ContributingAreaKm2"],
        views["MeanDischargeM3s"],
        ocean.astype(bool),
    )
    waterbody_cell_catalog = _waterbody_cell_catalog(
        views["DepressionID"],
        views["LakeID"],
        views["WaterBodyClass"],
        views["LakeFraction"],
        views["WetlandFraction"],
        physical_areas_km2,
    )
    flat_sink = views["FlowSinkType"][land_mask]
    closed_land_fraction = float(np.mean(flat_sink != 1))
    waterbody_support_land_cell_fraction = float(
        np.mean((views["LakeFraction"] + views["WetlandFraction"])[land_mask] > 0.0)
    )
    lake_support_land_cell_fraction = float(np.mean(views["LakeFraction"][land_mask] > 0.0))
    wetland_support_land_cell_fraction = float(np.mean(views["WetlandFraction"][land_mask] > 0.0))
    land_area_km2 = float(np.sum(physical_areas_km2[land_mask]))
    lake_land_area_fraction = float(
        np.sum(physical_areas_km2[land_mask] * views["LakeFraction"][land_mask]) / land_area_km2
    )
    wetland_land_area_fraction = float(
        np.sum(physical_areas_km2[land_mask] * views["WetlandFraction"][land_mask]) / land_area_km2
    )
    validation_minimum, validation_maximum = EARTH_LIKE_LAKE_AREA_FRACTION_RANGE
    lake_area_km2 = float(pc.sum(lake_catalog["water_area_km2"]).as_py() or 0.0)
    mean_lake_depth_m = float(metadata["lake_volume_km3"] * 1_000.0 / max(lake_area_km2, 1e-12))
    metadata.update(
        {
            **asdict(config),
            "planet_radius_m": radius_m,
            "median_land_cell_area_km2": median_land_cell_area,
            "effective_river_contributing_area_threshold_km2": controls[
                "river_contributing_area_threshold_km2"
            ],
            "waterbody_count": lake_catalog.num_rows + wetland_catalog.num_rows,
            "open_lake_count": int(pc.sum(lake_catalog["open_outlet"]).as_py() or 0),
            "open_wetland_count": int(pc.sum(wetland_catalog["open_outlet"]).as_py() or 0),
            "mean_lake_depth_m": mean_lake_depth_m,
            "closed_drainage_land_fraction": closed_land_fraction,
            "lake_land_cell_fraction": lake_support_land_cell_fraction,
            "lake_support_land_cell_fraction": lake_support_land_cell_fraction,
            "wetland_support_land_cell_fraction": wetland_support_land_cell_fraction,
            "waterbody_support_land_cell_fraction": waterbody_support_land_cell_fraction,
            "lake_land_area_fraction": lake_land_area_fraction,
            "wetland_land_area_fraction": wetland_land_area_fraction,
            "inland_water_and_wetland_land_area_fraction": (
                lake_land_area_fraction + wetland_land_area_fraction
            ),
            "earth_like_lake_area_fraction_range": [
                validation_minimum,
                validation_maximum,
            ],
            "earth_like_lake_area_validation_pass": int(
                validation_minimum <= lake_land_area_fraction <= validation_maximum
            ),
            "water_body_classes": {str(code): name for code, name in WATER_BODY_CLASSES.items()},
            "topology": "cubed_sphere",
            "model": "fractional_priority_flood_fill_spill_breach_hydrology_v2",
            "runoff_semantics": "monthly_climate_runoff_potential_routed_conservatively",
            "lake_semantics": "subgrid_fractional_open_or_closed_water_balance_depressions",
            "river_semantics": "vector_reach_graph_with_support_rasters",
            "breach_semantics": "selective_sustained_overflow_outlet_incision",
        }
    )
    context.logger.log_event({"type": "hydrology_summary", "stage": "hydrology", **metadata})
    return {
        **handles,
        "DepressionCatalog": depression_catalog,
        "LakeCatalog": lake_catalog,
        "WetlandCatalog": wetland_catalog,
        "BreachCatalog": breach_catalog,
        "BasinCatalog": basin_catalog,
        "DrainageGraph": drainage_graph,
        "WaterBodyCellCatalog": waterbody_cell_catalog,
        "RiverReachCatalog": reach_catalog,
        "HydrologyMetadata": metadata,
    }


__all__ = ["HydrologyConfig", "hydrology_stage"]
