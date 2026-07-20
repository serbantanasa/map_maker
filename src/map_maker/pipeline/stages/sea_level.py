"""Connected global ocean, solved datum, and fractional coarse coastlines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional

import numpy as np
from PIL import Image

from .._sea_level_native import run_cubed_sphere_sea_level
from ..cubed_sphere import CubedSphereGrid
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult


@dataclass(frozen=True)
class SeaLevelConfig:
    target_ocean_area_fraction: float = 0.65
    shelf_depth_m: float = 200.0
    minimum_coastal_relief_m: float = 40.0
    coastal_relief_scale: float = 0.45
    maximum_fractional_area_error: float = 1e-6

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "SeaLevelConfig":
        if not mapping:
            return cls()
        return cls(
            target_ocean_area_fraction=float(
                mapping.get("target_ocean_area_fraction", cls.target_ocean_area_fraction)
            ),
            shelf_depth_m=float(mapping.get("shelf_depth_m", cls.shelf_depth_m)),
            minimum_coastal_relief_m=float(
                mapping.get("minimum_coastal_relief_m", cls.minimum_coastal_relief_m)
            ),
            coastal_relief_scale=float(
                mapping.get("coastal_relief_scale", cls.coastal_relief_scale)
            ),
            maximum_fractional_area_error=float(
                mapping.get("maximum_fractional_area_error", cls.maximum_fractional_area_error)
            ),
        )


def _artifact_array(result, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"missing dependency artifact {name!r}")
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


def _equirectangular_rgb(faces: np.ndarray) -> np.ndarray:
    resolution = faces.shape[1]
    height = resolution * 4
    width = resolution * 8
    longitude = (np.arange(width) + 0.5) / width * (2.0 * np.pi) - np.pi
    latitude = np.pi / 2.0 - (np.arange(height) + 0.5) / height * np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    directions = np.stack(
        (
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ),
        axis=-1,
    ).reshape(-1, 3)
    dominant_axis = np.argmax(np.abs(directions), axis=1)
    face = np.empty(dominant_axis.shape, dtype=np.intp)
    face[(dominant_axis == 0) & (directions[:, 0] >= 0.0)] = 0
    face[(dominant_axis == 0) & (directions[:, 0] < 0.0)] = 1
    face[(dominant_axis == 1) & (directions[:, 1] >= 0.0)] = 2
    face[(dominant_axis == 1) & (directions[:, 1] < 0.0)] = 3
    face[(dominant_axis == 2) & (directions[:, 2] >= 0.0)] = 4
    face[(dominant_axis == 2) & (directions[:, 2] < 0.0)] = 5
    normals = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    right = np.array(
        [[0, 1, 0], [0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
        dtype=np.float64,
    )
    down = np.array(
        [[0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1], [1, 0, 0], [-1, 0, 0]],
        dtype=np.float64,
    )
    denominator = np.einsum("ij,ij->i", directions, normals[face])
    u = np.einsum("ij,ij->i", directions, right[face]) / denominator
    v = np.einsum("ij,ij->i", directions, down[face]) / denominator
    column = np.clip(
        np.floor((np.arctan(u) / (np.pi / 2.0) + 0.5) * resolution),
        0,
        resolution - 1,
    ).astype(np.intp)
    row = np.clip(
        np.floor((np.arctan(v) / (np.pi / 2.0) + 0.5) * resolution),
        0,
        resolution - 1,
    ).astype(np.intp)
    return faces[face, row, column].reshape(height, width, 3)


def _hypsometric_rgb(
    surface_elevation: np.ndarray,
    ocean_fraction: np.ndarray,
    shelf_fraction: np.ndarray,
) -> np.ndarray:
    elevation = np.asarray(surface_elevation, dtype=np.float64)
    ocean = np.asarray(ocean_fraction, dtype=np.float64)[..., None]
    shelf = np.asarray(shelf_fraction, dtype=np.float64)[..., None]
    depth_scale = np.clip(-elevation / 6_000.0, 0.0, 1.0)[..., None]
    shallow = np.array([72.0, 151.0, 184.0])
    deep = np.array([17.0, 61.0, 105.0])
    water = shallow * (1.0 - depth_scale) + deep * depth_scale
    water = water * (1.0 - 0.18 * shelf) + np.array([104.0, 184.0, 194.0]) * 0.18 * shelf

    height_scale = np.clip(elevation / 4_500.0, 0.0, 1.0)[..., None]
    lowland = np.array([89.0, 142.0, 82.0])
    highland = np.array([169.0, 137.0, 91.0])
    land = lowland * (1.0 - height_scale) + highland * height_scale
    snow = np.clip((elevation - 2_400.0) / 1_800.0, 0.0, 1.0)[..., None]
    land = land * (1.0 - snow) + np.array([232.0, 235.0, 228.0]) * snow
    return np.clip(land * (1.0 - ocean) + water * ocean, 0.0, 255.0).astype(np.uint8)


def _sea_level_visualizer(
    result, request: VisualizationRequest
) -> Optional[list[VisualizationResult]]:
    try:
        surface = _artifact_array(result, "SurfaceElevationM")
        ocean_fraction = _artifact_array(result, "SurfaceOceanFraction")
        shelf_fraction = _artifact_array(result, "ContinentalShelfFraction")
        inland = _artifact_array(result, "InlandBelowSeaLevelMask") >= 0.5
    except KeyError:
        return None
    rgb = _hypsometric_rgb(surface, ocean_fraction, shelf_fraction)
    cube_path = request.output_dir / "surface_geography_cube.png"
    global_path = request.output_dir / "surface_geography_global.png"
    Image.fromarray(_cube_net_rgb(rgb), mode="RGB").save(cube_path)
    Image.fromarray(_equirectangular_rgb(rgb), mode="RGB").save(global_path)

    diagnostic = np.zeros((*surface.shape, 3), dtype=np.uint8)
    diagnostic[..., 2] = np.clip(ocean_fraction * 210.0, 0.0, 255.0).astype(np.uint8)
    diagnostic[..., 1] = np.clip(shelf_fraction * 255.0, 0.0, 255.0).astype(np.uint8)
    diagnostic[inland] = np.array([220, 70, 120], dtype=np.uint8)
    diagnostic_path = request.output_dir / "shelves_and_inland_basins.png"
    Image.fromarray(_cube_net_rgb(diagnostic), mode="RGB").save(diagnostic_path)
    return [
        VisualizationResult(cube_path, "SurfaceElevationM", {"projection": "cube_net"}),
        VisualizationResult(global_path, "SurfaceElevationM", {"projection": "equirectangular"}),
        VisualizationResult(
            diagnostic_path,
            "ContinentalShelfFraction",
            {"green": "shelf", "blue": "ocean", "magenta": "isolated_below_sea_level"},
        ),
    ]


@stage(
    "sea_level",
    inputs=("elevation",),
    outputs=(
        "SurfaceOceanMask",
        "SurfaceOceanFraction",
        "SurfaceElevationM",
        "OceanDepthM",
        "ContinentalShelfFraction",
        "CoastalCellMask",
        "InlandBelowSeaLevelMask",
        "SeaLevelMetadata",
    ),
    version="v2",
    native_libraries=("sea_level_native",),
    visualizer=_sea_level_visualizer,
)
def sea_level_stage(context, deps, config_mapping: Mapping[str, object]):
    config = SeaLevelConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("canonical sea level requires topology: cubed_sphere")
    shape = context.topology.face_shape
    names = (
        "SurfaceOceanMask",
        "SurfaceOceanFraction",
        "SurfaceElevationM",
        "OceanDepthM",
        "ContinentalShelfFraction",
        "CoastalCellMask",
        "InlandBelowSeaLevelMask",
    )
    handles = {
        name: context.arena.allocate_array(f"sea_level_{name.lower()}", shape, np.float32)
        for name in names
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}
    elevation = deps["elevation"]
    controls = asdict(config)
    maximum_error = controls.pop("maximum_fractional_area_error")
    with context.timed("connected_sea_level_kernel"):
        metadata = run_cubed_sphere_sea_level(
            **controls,
            areas=context.topology.cell_areas,
            neighbors=context.topology.neighbor_indices,
            elevation=_artifact_array(elevation, "BedrockElevationM"),
            relief=_artifact_array(elevation, "TerrainReliefM"),
            ocean_mask_out=views["SurfaceOceanMask"],
            ocean_fraction_out=views["SurfaceOceanFraction"],
            surface_elevation_out=views["SurfaceElevationM"],
            ocean_depth_out=views["OceanDepthM"],
            shelf_fraction_out=views["ContinentalShelfFraction"],
            coastal_mask_out=views["CoastalCellMask"],
            inland_below_sea_level_out=views["InlandBelowSeaLevelMask"],
        )
    fractional_error = abs(
        metadata["ocean_fractional_area_fraction"] - config.target_ocean_area_fraction
    )
    if fractional_error > maximum_error:
        raise RuntimeError(
            f"fractional ocean area misses target: {fractional_error:.3e} > {maximum_error:.3e}"
        )
    for handle in handles.values():
        handle.seal()
    metadata.update(
        {
            **asdict(config),
            "fractional_area_error": fractional_error,
            "topology": "cubed_sphere",
            "model": "connected_ocean_fractional_coast_v1",
            "datum_semantics": "surface_elevation_relative_to_solved_global_ocean_level",
            "crust_mask_semantics": "independent_input_not_surface_water",
        }
    )
    context.logger.log_event({"type": "sea_level_summary", "stage": "sea_level", **metadata})
    return {**handles, "SeaLevelMetadata": metadata}


__all__ = ["SeaLevelConfig", "sea_level_stage"]
