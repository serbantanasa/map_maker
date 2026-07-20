"""Versioned physical-atlas export over immutable canonical world artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray
import pyarrow as pa  # type: ignore[import-untyped]
from PIL import Image, ImageDraw, ImageFilter
from rasterio.crs import CRS  # type: ignore[import-untyped]
from rasterio.enums import ColorInterp, Resampling  # type: ignore[import-untyped]
from rasterio.features import rasterize  # type: ignore[import-untyped]
from rasterio.transform import Affine, from_bounds  # type: ignore[import-untyped]
from rasterio.warp import reproject, transform as transform_coordinates  # type: ignore[import-untyped]
from rasterio.warp import transform_bounds
import rasterio  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from .config import PipelineConfig
from .execution import ExecutionEngine
from .models import StageResult


ATLAS_STYLE_VERSION = "physical_atlas_v1"
SOURCE_CRS = CRS.from_epsg(4326)
EQUAL_EARTH_CRS = CRS.from_epsg(8857)
EARTH_RADIUS_M = 6_371_008.8

BIOME_COLORS: NDArray[np.float64] = np.asarray(
    [
        (29, 91, 55),
        (75, 119, 64),
        (176, 155, 76),
        (210, 180, 117),
        (152, 132, 88),
        (67, 111, 73),
        (132, 157, 87),
        (174, 151, 94),
        (56, 94, 79),
        (145, 154, 138),
        (174, 166, 147),
        (126, 128, 124),
        (48, 119, 103),
    ],
    dtype=np.float64,
)


def _numeric_value(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be numeric")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _integer_value(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class AtlasStyleConfig:
    width_px: int = 2_400
    border_fraction: float = 0.025
    central_meridian_deg: float | None = None
    terrain_vertical_exaggeration: float = 16.0
    land_hillshade_strength: float = 0.52
    ocean_hillshade_strength: float = 0.22
    coastline_width_px: int = 0
    draw_rivers: bool = True
    minimum_river_discharge_m3s: float = 4_000.0
    maximum_river_width_px: float = 3.5

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "AtlasStyleConfig":
        mapping = mapping or {}
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown atlas style controls: {', '.join(sorted(unknown))}")

        central_raw = mapping.get("central_meridian_deg", "auto")
        central = (
            None
            if central_raw is None or str(central_raw).lower() == "auto"
            else _numeric_value("central_meridian_deg", central_raw)
        )
        draw_rivers = mapping.get("draw_rivers", cls.draw_rivers)
        if not isinstance(draw_rivers, bool):
            raise ValueError("draw_rivers must be boolean")
        config = cls(
            width_px=_integer_value("width_px", mapping.get("width_px", cls.width_px)),
            border_fraction=_numeric_value(
                "border_fraction", mapping.get("border_fraction", cls.border_fraction)
            ),
            central_meridian_deg=central,
            terrain_vertical_exaggeration=_numeric_value(
                "terrain_vertical_exaggeration",
                mapping.get("terrain_vertical_exaggeration", cls.terrain_vertical_exaggeration),
            ),
            land_hillshade_strength=_numeric_value(
                "land_hillshade_strength",
                mapping.get("land_hillshade_strength", cls.land_hillshade_strength),
            ),
            ocean_hillshade_strength=_numeric_value(
                "ocean_hillshade_strength",
                mapping.get("ocean_hillshade_strength", cls.ocean_hillshade_strength),
            ),
            coastline_width_px=_integer_value(
                "coastline_width_px",
                mapping.get("coastline_width_px", cls.coastline_width_px),
            ),
            draw_rivers=draw_rivers,
            minimum_river_discharge_m3s=_numeric_value(
                "minimum_river_discharge_m3s",
                mapping.get("minimum_river_discharge_m3s", cls.minimum_river_discharge_m3s),
            ),
            maximum_river_width_px=_numeric_value(
                "maximum_river_width_px",
                mapping.get("maximum_river_width_px", cls.maximum_river_width_px),
            ),
        )
        if not 512 <= config.width_px <= 12_000:
            raise ValueError("width_px must be in [512, 12000]")
        if not 0.0 <= config.border_fraction <= 0.15:
            raise ValueError("border_fraction must be in [0, 0.15]")
        if config.central_meridian_deg is not None and not math.isfinite(
            config.central_meridian_deg
        ):
            raise ValueError("central_meridian_deg must be finite or auto")
        for name in (
            "terrain_vertical_exaggeration",
            "land_hillshade_strength",
            "ocean_hillshade_strength",
            "minimum_river_discharge_m3s",
            "maximum_river_width_px",
        ):
            value = float(getattr(config, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= config.coastline_width_px <= 8:
            raise ValueError("coastline_width_px must be in [0, 8]")
        return config


@dataclass(frozen=True)
class AtlasExportConfig:
    world_config: Path
    output_dir: Path
    style: AtlasStyleConfig

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        output_dir: Path | None = None,
        width_px: int | None = None,
        central_meridian_deg: float | None = None,
        draw_rivers: bool | None = None,
    ) -> "AtlasExportConfig":
        path = Path(path).expanduser().resolve()
        data = yaml.safe_load(path.read_text(encoding="utf8"))
        if not isinstance(data, Mapping):
            raise TypeError("Atlas config must contain a mapping")
        if int(data.get("format_version", 1)) != 1:
            raise ValueError("Unsupported atlas format_version")
        raw_world = data.get("world_config")
        if not raw_world:
            raise ValueError("Atlas config requires world_config")
        world_config = (path.parent / str(raw_world)).resolve()
        configured_output = (path.parent / str(data.get("output_dir", "../out/atlas"))).resolve()
        style_mapping = data.get("style", {})
        if not isinstance(style_mapping, Mapping):
            raise TypeError("Atlas style must contain a mapping")
        style = AtlasStyleConfig.from_mapping(cast(Mapping[str, object], style_mapping))
        if width_px is not None:
            style = replace(style, width_px=width_px)
        if central_meridian_deg is not None:
            style = replace(style, central_meridian_deg=central_meridian_deg)
        if draw_rivers is not None:
            style = replace(style, draw_rivers=draw_rivers)
        style = AtlasStyleConfig.from_mapping(asdict(style))
        return cls(
            world_config=world_config,
            output_dir=(output_dir.expanduser().resolve() if output_dir else configured_output),
            style=style,
        )


@dataclass(frozen=True)
class AtlasExportResult:
    png_path: Path
    geotiff_path: Path
    metadata_path: Path
    width_px: int
    height_px: int
    central_meridian_deg: float
    rendered_river_count: int


def _artifact_array(result: StageResult, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing atlas input {result.stage_name}.{name}")
    value = record.value
    return cast(np.ndarray, np.asarray(value.array() if hasattr(value, "array") else value))


def _artifact_table(result: StageResult, name: str) -> pa.Table:
    record = result.artifact_records.get(name)
    if record is None or not isinstance(record.value, pa.Table):
        raise KeyError(f"Missing atlas table {result.stage_name}.{name}")
    return record.value.combine_chunks()


def cubed_sphere_to_equirectangular(faces: np.ndarray) -> np.ndarray:
    """Nearest-sample a cubed-sphere field into a diagnostic longitude grid."""

    values = np.asarray(faces)
    if values.ndim < 3 or values.shape[0] != 6 or values.shape[1] != values.shape[2]:
        raise ValueError("cubed-sphere values must have shape (6, N, N, ...)")
    resolution = values.shape[1]
    height = resolution * 4
    width = resolution * 8
    longitude = (np.arange(width) + 0.5) / width * (2.0 * np.pi) - np.pi
    latitude = np.pi / 2.0 - (np.arange(height) + 0.5) / height * np.pi
    lon: NDArray[np.float64]
    lat: NDArray[np.float64]
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
    normals: NDArray[np.float64] = np.asarray(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    right: NDArray[np.float64] = np.asarray(
        [[0, 1, 0], [0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
        dtype=np.float64,
    )
    down: NDArray[np.float64] = np.asarray(
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
    trailing = values.shape[3:]
    return cast(np.ndarray, values[face, row, column].reshape(height, width, *trailing))


def choose_central_meridian(ocean_fraction: np.ndarray) -> float:
    """Choose a deterministic ocean-heavy seam and return its opposite meridian."""

    ocean = np.asarray(ocean_fraction, dtype=np.float64)
    if ocean.ndim != 2:
        raise ValueError("ocean_fraction must be two-dimensional")
    height, width = ocean.shape
    latitude = np.pi / 2.0 - (np.arange(height) + 0.5) / height * np.pi
    weights = np.maximum(np.cos(latitude), 0.02)[:, None]
    seam_cost = np.sum(np.clip(1.0 - ocean, 0.0, 1.0) * weights, axis=0)
    radius = max(1, width // 180)
    smoothed = sum(np.roll(seam_cost, shift) for shift in range(-radius, radius + 1))
    smoothed /= 2 * radius + 1
    seam_column = int(np.argmin(smoothed))
    seam_longitude = -180.0 + (seam_column + 0.5) * 360.0 / width
    return float(((seam_longitude + 360.0) % 360.0) - 180.0)


def _roll_to_central(values: np.ndarray, central_meridian_deg: float) -> np.ndarray:
    width = values.shape[1]
    seam_longitude = ((central_meridian_deg - 180.0 + 180.0) % 360.0) - 180.0
    seam_column = int(round((seam_longitude + 180.0) / 360.0 * width - 0.5)) % width
    return cast(np.ndarray, np.roll(values, -seam_column, axis=1))


def _smooth_global(values: np.ndarray, passes: int = 2) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for _ in range(passes):
        north = np.concatenate((result[:1], result[:-1]), axis=0)
        south = np.concatenate((result[1:], result[-1:]), axis=0)
        result = (
            result * 4.0 + north + south + np.roll(result, 1, axis=1) + np.roll(result, -1, axis=1)
        ) / 8.0
    return cast(np.ndarray, result)


def _multidirectional_hillshade(
    elevation_m: np.ndarray, *, vertical_exaggeration: float
) -> np.ndarray:
    elevation = _smooth_global(elevation_m, passes=2)
    height, width = elevation.shape
    latitude = np.pi / 2.0 - (np.arange(height) + 0.5) / height * np.pi
    dx = EARTH_RADIUS_M * (2.0 * np.pi / width) * np.maximum(np.cos(latitude), 0.04)
    dy = EARTH_RADIUS_M * np.pi / height
    dzdx = (np.roll(elevation, -1, axis=1) - np.roll(elevation, 1, axis=1)) / (2.0 * dx[:, None])
    north = np.concatenate((elevation[:1], elevation[:-1]), axis=0)
    south = np.concatenate((elevation[1:], elevation[-1:]), axis=0)
    dzdy = (north - south) / (2.0 * dy)
    nx = -dzdx * vertical_exaggeration
    ny = -dzdy * vertical_exaggeration
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= norm
    ny /= norm
    nz /= norm

    illumination = np.zeros_like(elevation)
    lights = ((315.0, 45.0, 0.55), (45.0, 50.0, 0.25), (270.0, 40.0, 0.20))
    flat_reference = 0.0
    for azimuth_deg, altitude_deg, weight in lights:
        azimuth = math.radians(azimuth_deg)
        altitude = math.radians(altitude_deg)
        lx = math.sin(azimuth) * math.cos(altitude)
        ly = math.cos(azimuth) * math.cos(altitude)
        lz = math.sin(altitude)
        illumination += weight * np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
        flat_reference += weight * lz
    return cast(np.ndarray, illumination - flat_reference)


def compose_physical_texture(
    layers: Mapping[str, np.ndarray], style: AtlasStyleConfig
) -> np.ndarray:
    """Compose an equirectangular atlas texture without modifying source state."""

    elevation = np.asarray(layers["elevation_m"], dtype=np.float64)
    ocean = np.clip(np.asarray(layers["ocean_fraction"], dtype=np.float64), 0.0, 1.0)
    depth = np.maximum(np.asarray(layers["ocean_depth_m"], dtype=np.float64), 0.0)
    shelf = np.clip(np.asarray(layers["shelf_fraction"], dtype=np.float64), 0.0, 1.0)
    biomes = np.maximum(np.asarray(layers["biome_fractions"], dtype=np.float64), 0.0)
    if biomes.shape != (*elevation.shape, len(BIOME_COLORS)):
        raise ValueError("biome_fractions shape does not match the atlas taxonomy")

    ecological_ground = np.sum(biomes, axis=-1)
    land_rgb = biomes @ BIOME_COLORS
    fallback = np.asarray((118.0, 137.0, 88.0))
    land_rgb = np.where(
        (ecological_ground > 1e-9)[..., None],
        land_rgb / np.maximum(ecological_ground[..., None], 1e-9),
        fallback,
    )

    altitude = np.clip(elevation / 4_500.0, 0.0, 1.0) ** 0.75
    highland_color = np.asarray((137.0, 124.0, 108.0))
    altitude_weight = (0.24 * altitude)[..., None]
    land_rgb = land_rgb * (1.0 - altitude_weight) + highland_color * altitude_weight

    terrain_relief = np.maximum(np.asarray(layers["terrain_relief_m"], dtype=np.float64), 0.0)
    ruggedness = np.clip(terrain_relief / 1_800.0, 0.0, 1.0) ** 0.7
    rugged_weight = (0.12 * ruggedness)[..., None]
    land_rgb = land_rgb * (1.0 - rugged_weight) + highland_color * rugged_weight

    bedrock = np.clip(np.asarray(layers["bedrock_fraction"], dtype=np.float64), 0.0, 1.0)
    rock_weight = (0.48 * bedrock)[..., None]
    rock_color = np.asarray((126.0, 117.0, 105.0))
    land_rgb = land_rgb * (1.0 - rock_weight) + rock_color * rock_weight

    salinity = np.clip(np.asarray(layers["soil_salinity"], dtype=np.float64), 0.0, 1.0)
    salt_weight = (0.22 * salinity)[..., None]
    salt_color = np.asarray((201.0, 190.0, 158.0))
    land_rgb = land_rgb * (1.0 - salt_weight) + salt_color * salt_weight

    wetland = np.clip(np.asarray(layers["wetland_fraction"], dtype=np.float64), 0.0, 1.0)
    wetland_weight = np.where(wetland > 0.002, np.clip(np.sqrt(wetland) * 0.9, 0.0, 0.55), 0.0)
    wetland_color = np.asarray((47.0, 112.0, 99.0))
    land_rgb = (
        land_rgb * (1.0 - wetland_weight[..., None]) + wetland_color * wetland_weight[..., None]
    )

    snow = np.clip(np.asarray(layers["snow_persistence"], dtype=np.float64), 0.0, 1.0)
    snow_weight = np.clip((snow - 0.08) * 0.55, 0.0, 0.42)
    snow_color = np.asarray((229.0, 233.0, 226.0))
    land_rgb = land_rgb * (1.0 - snow_weight[..., None]) + snow_color * snow_weight[..., None]
    glacier = np.clip(np.asarray(layers["glacier_fraction"], dtype=np.float64), 0.0, 1.0)
    glacier_weight = np.clip(glacier * 1.6, 0.0, 1.0)
    glacier_color = np.asarray((235.0, 244.0, 244.0))
    land_rgb = (
        land_rgb * (1.0 - glacier_weight[..., None]) + glacier_color * glacier_weight[..., None]
    )

    depth_scale = np.clip(np.log1p(depth) / np.log1p(6_500.0), 0.0, 1.0)
    ocean_stops: NDArray[np.float64] = np.asarray(
        ((105, 176, 190), (52, 125, 157), (29, 79, 116), (15, 45, 78)),
        dtype=np.float64,
    )
    ocean_positions = np.asarray((0.0, 0.35, 0.72, 1.0))
    ocean_rgb = np.stack(
        [np.interp(depth_scale, ocean_positions, ocean_stops[:, channel]) for channel in range(3)],
        axis=-1,
    )
    shelf_conditional = np.clip(shelf / np.maximum(ocean, 1e-6), 0.0, 1.0)
    shelf_weight = (0.34 * shelf_conditional)[..., None]
    shelf_color = np.asarray((104.0, 181.0, 189.0))
    ocean_rgb = ocean_rgb * (1.0 - shelf_weight) + shelf_color * shelf_weight

    rgb = land_rgb * (1.0 - ocean[..., None]) + ocean_rgb * ocean[..., None]
    lake = np.clip(np.asarray(layers["lake_fraction"], dtype=np.float64), 0.0, 1.0)
    lake_weight = np.where(lake > 0.003, np.clip(np.sqrt(lake) * 1.7, 0.0, 0.90), 0.0)
    lake_color = np.asarray((60.0, 137.0, 163.0))
    rgb = rgb * (1.0 - lake_weight[..., None]) + lake_color * lake_weight[..., None]

    relief = _multidirectional_hillshade(
        elevation,
        vertical_exaggeration=style.terrain_vertical_exaggeration,
    )
    shade_strength = (
        style.land_hillshade_strength * (1.0 - ocean) + style.ocean_hillshade_strength * ocean
    )
    shade = np.clip(1.0 + shade_strength * relief, 0.72, 1.16)
    return cast(np.ndarray, np.clip(rgb * shade[..., None], 0.0, 255.0).astype(np.uint8))


def _project_equal_earth(
    rgb: np.ndarray,
    ocean_fraction: np.ndarray,
    *,
    width_px: int,
    border_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Affine]:
    source_height, source_width = rgb.shape[:2]
    left, bottom, right, top = transform_bounds(
        SOURCE_CRS,
        EQUAL_EARTH_CRS,
        -180.0,
        -90.0,
        180.0,
        90.0,
        densify_pts=181,
    )
    x_padding = (right - left) * border_fraction
    y_padding = (top - bottom) * border_fraction
    left -= x_padding
    right += x_padding
    bottom -= y_padding
    top += y_padding
    height_px = max(256, round(width_px * (top - bottom) / (right - left)))
    source_transform = from_bounds(-180.0, -90.0, 180.0, 90.0, source_width, source_height)
    destination_transform = from_bounds(left, bottom, right, top, width_px, height_px)

    projected = np.zeros((3, height_px, width_px), dtype=np.uint8)
    for channel in range(3):
        reproject(
            source=rgb[..., channel],
            destination=projected[channel],
            src_transform=source_transform,
            src_crs=SOURCE_CRS,
            dst_transform=destination_transform,
            dst_crs=EQUAL_EARTH_CRS,
            resampling=Resampling.bilinear,
            dst_nodata=0,
        )
    edge_samples = max(721, width_px // 2)
    longitude_edge = np.linspace(-180.0, 180.0, edge_samples)
    latitude_edge = np.linspace(-90.0, 90.0, edge_samples)
    perimeter_lon = np.concatenate(
        (
            longitude_edge,
            np.full(edge_samples, 180.0),
            longitude_edge[::-1],
            np.full(edge_samples, -180.0),
        )
    )
    perimeter_lat = np.concatenate(
        (
            np.full(edge_samples, -90.0),
            latitude_edge,
            np.full(edge_samples, 90.0),
            latitude_edge[::-1],
        )
    )
    perimeter_x, perimeter_y = transform_coordinates(
        SOURCE_CRS,
        EQUAL_EARTH_CRS,
        perimeter_lon.tolist(),
        perimeter_lat.tolist(),
    )
    outline = {
        "type": "Polygon",
        "coordinates": [list(zip(perimeter_x, perimeter_y, strict=True))],
    }
    coverage = rasterize(
        ((outline, 255),),
        out_shape=(height_px, width_px),
        transform=destination_transform,
        fill=0,
        dtype=np.uint8,
    )
    projected_ocean = np.zeros((height_px, width_px), dtype=np.float32)
    reproject(
        source=np.asarray(ocean_fraction, dtype=np.float32),
        destination=projected_ocean,
        src_transform=source_transform,
        src_crs=SOURCE_CRS,
        dst_transform=destination_transform,
        dst_crs=EQUAL_EARTH_CRS,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )
    background = np.asarray((236.0, 237.0, 232.0))
    alpha = coverage.astype(np.float64) / 255.0
    canvas = np.moveaxis(projected, 0, -1).astype(np.float64) * alpha[..., None] + background * (
        1.0 - alpha[..., None]
    )
    return (
        np.clip(canvas, 0.0, 255.0).astype(np.uint8),
        coverage,
        projected_ocean,
        destination_transform,
    )


def _outline_coasts(
    image: Image.Image,
    projected_ocean: np.ndarray,
    coverage: np.ndarray,
    width_px: int,
) -> Image.Image:
    if width_px <= 0:
        return image
    valid = coverage > 0
    land = (projected_ocean < 0.5) & valid
    edge = valid & (
        (land != np.roll(land, 1, axis=0))
        | (land != np.roll(land, -1, axis=0))
        | (land != np.roll(land, 1, axis=1))
        | (land != np.roll(land, -1, axis=1))
    )
    edge_image = Image.fromarray(edge.astype(np.uint8) * 255, mode="L")
    if width_px > 1:
        edge_image = edge_image.filter(ImageFilter.MaxFilter(width_px * 2 - 1))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay.paste((36, 57, 58, 92), mask=edge_image)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _relative_longitude(longitude_deg: np.ndarray, central_meridian_deg: float) -> np.ndarray:
    return cast(np.ndarray, (longitude_deg - central_meridian_deg + 180.0) % 360.0 - 180.0)


def _smooth_polyline(
    points: Sequence[tuple[float, float]], *, passes: int = 2
) -> list[tuple[float, float]]:
    """Round coarse cell-center bends while preserving endpoints and order."""

    line: NDArray[np.float64] = np.asarray(points, dtype=np.float64)
    for _ in range(passes):
        if len(line) < 2:
            break
        first = 0.75 * line[:-1] + 0.25 * line[1:]
        second = 0.25 * line[:-1] + 0.75 * line[1:]
        smoothed: NDArray[np.float64] = np.empty((len(line) * 2, 2), dtype=np.float64)
        smoothed[0] = line[0]
        smoothed[-1] = line[-1]
        smoothed[1:-1:2] = first
        smoothed[2:-1:2] = second
        line = smoothed
    return [(float(x), float(y)) for x, y in line]


def _draw_rivers(
    image: Image.Image,
    reaches: pa.Table,
    transform: Affine,
    style: AtlasStyleConfig,
    central_meridian_deg: float,
) -> tuple[Image.Image, int]:
    rows = [
        row
        for row in reaches.to_pylist()
        if row["reach_kind"] == "channel"
        and float(row["discharge_mean"]) >= style.minimum_river_discharge_m3s
        and len(row["polyline_on_cubed_sphere"]) >= 2
    ]
    if not rows:
        return image, 0
    rows.sort(key=lambda row: float(row["discharge_mean"]))
    maximum_discharge = max(float(row["discharge_mean"]) for row in rows)
    minimum_log = math.log1p(style.minimum_river_discharge_m3s)
    maximum_log = max(math.log1p(maximum_discharge), minimum_log + 1e-9)
    inverse_transform = ~transform
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    rendered = 0
    for row in rows:
        xyz = np.asarray(row["polyline_on_cubed_sphere"], dtype=np.float64)
        longitude = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
        latitude = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))
        relative_longitude = _relative_longitude(longitude, central_meridian_deg)
        split_after: NDArray[np.intp] = (
            np.flatnonzero(np.abs(np.diff(relative_longitude)) > 180.0) + 1
        )
        segments = np.split(np.arange(len(xyz)), split_after)
        discharge = float(row["discharge_mean"])
        scale = (math.log1p(discharge) - minimum_log) / (maximum_log - minimum_log)
        line_width = max(1, round(1.0 + scale * (style.maximum_river_width_px - 1.0)))
        opacity = round(92 + 112 * scale)
        any_segment = False
        for segment in segments:
            if len(segment) < 2:
                continue
            xs, ys = transform_coordinates(
                SOURCE_CRS,
                EQUAL_EARTH_CRS,
                relative_longitude[segment].tolist(),
                latitude[segment].tolist(),
            )
            points = _smooth_polyline(
                [inverse_transform * (x, y) for x, y in zip(xs, ys, strict=True)]
            )
            draw.line(points, fill=(28, 92, 137, opacity), width=line_width, joint="curve")
            any_segment = True
        rendered += int(any_segment)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"), rendered


def _write_geotiff(
    path: Path,
    image: np.ndarray,
    coverage: np.ndarray,
    transform: Affine,
    central_meridian_deg: float,
) -> CRS:
    atlas_crs = CRS.from_string(
        f"+proj=eqearth +lon_0={central_meridian_deg:.10f} +datum=WGS84 +units=m +no_defs"
    )
    height, width = image.shape[:2]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=4,
        dtype=np.uint8,
        crs=atlas_crs,
        transform=transform,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dataset:
        dataset.write(np.moveaxis(image, -1, 0), indexes=(1, 2, 3))
        dataset.write(coverage, indexes=4)
        dataset.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )
    return atlas_crs


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_provenance(results: Mapping[str, StageResult]) -> dict[str, dict[str, str]]:
    inputs = {
        "sea_level": (
            "SurfaceElevationM",
            "SurfaceOceanFraction",
            "OceanDepthM",
            "ContinentalShelfFraction",
        ),
        "elevation": ("TerrainReliefM",),
        "cryosphere": ("MonthlySnowWaterEquivalentMm", "GlacierIceFraction"),
        "surface_materials": (
            "BedrockSurfaceFraction",
            "SoilSalinityIndex",
            "EffectiveLakeFraction",
            "EffectiveWetlandFraction",
        ),
        "derived_biomes": ("BiomeFractions",),
        "hydrology": ("RiverReachCatalog",),
    }
    return {
        stage_name: {
            name: results[stage_name].artifact_records[name].checksum for name in artifact_names
        }
        for stage_name, artifact_names in inputs.items()
    }


def _equirectangular_layers(results: Mapping[str, StageResult]) -> dict[str, np.ndarray]:
    sea_level = results["sea_level"]
    elevation = results["elevation"]
    cryosphere = results["cryosphere"]
    materials = results["surface_materials"]
    biomes = results["derived_biomes"]
    snow = np.moveaxis(
        _artifact_array(cryosphere, "MonthlySnowWaterEquivalentMm"),
        0,
        -1,
    )
    snow_persistence = np.mean(np.clip((snow - 20.0) / 180.0, 0.0, 1.0), axis=-1)
    biome_fractions = np.moveaxis(_artifact_array(biomes, "BiomeFractions"), 0, -1)
    source_layers = {
        "elevation_m": _artifact_array(sea_level, "SurfaceElevationM"),
        "terrain_relief_m": _artifact_array(elevation, "TerrainReliefM"),
        "ocean_fraction": _artifact_array(sea_level, "SurfaceOceanFraction"),
        "ocean_depth_m": _artifact_array(sea_level, "OceanDepthM"),
        "shelf_fraction": _artifact_array(sea_level, "ContinentalShelfFraction"),
        "biome_fractions": biome_fractions,
        "bedrock_fraction": _artifact_array(materials, "BedrockSurfaceFraction"),
        "soil_salinity": _artifact_array(materials, "SoilSalinityIndex"),
        "wetland_fraction": _artifact_array(materials, "EffectiveWetlandFraction"),
        "lake_fraction": _artifact_array(materials, "EffectiveLakeFraction"),
        "snow_persistence": snow_persistence,
        "glacier_fraction": _artifact_array(cryosphere, "GlacierIceFraction"),
    }
    return {name: cubed_sphere_to_equirectangular(values) for name, values in source_layers.items()}


def export_physical_atlas(config: AtlasExportConfig) -> AtlasExportResult:
    """Run through immutable biome state and export a projected physical atlas."""

    from .stages import ensure_builtin_stages

    ensure_builtin_stages()
    started = time.perf_counter()
    world = PipelineConfig.from_file(config.world_config)
    if world.topology.lower() != "cubed_sphere":
        raise ValueError("physical atlas export requires topology: cubed_sphere")
    engine = ExecutionEngine(world, generate_visuals=False)
    results = engine.run(["derived_biomes"])
    layers = _equirectangular_layers(results)
    central_meridian = (
        choose_central_meridian(layers["ocean_fraction"])
        if config.style.central_meridian_deg is None
        else ((config.style.central_meridian_deg + 180.0) % 360.0) - 180.0
    )
    rotated_layers = {
        name: _roll_to_central(values, central_meridian) for name, values in layers.items()
    }
    texture = compose_physical_texture(rotated_layers, config.style)
    image_array, coverage, projected_ocean, transform = _project_equal_earth(
        texture,
        rotated_layers["ocean_fraction"],
        width_px=config.style.width_px,
        border_fraction=config.style.border_fraction,
    )
    image = _outline_coasts(
        Image.fromarray(image_array, mode="RGB"),
        projected_ocean,
        coverage,
        config.style.coastline_width_px,
    )
    rendered_river_count = 0
    if config.style.draw_rivers:
        image, rendered_river_count = _draw_rivers(
            image,
            _artifact_table(results["hydrology"], "RiverReachCatalog"),
            transform,
            config.style,
            central_meridian,
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = config.output_dir / "physical_world_map.png"
    geotiff_path = config.output_dir / "physical_world_map.tif"
    metadata_path = config.output_dir / "physical_world_map.json"
    image.save(png_path, optimize=True)
    final_array = np.asarray(image, dtype=np.uint8)
    atlas_crs = _write_geotiff(
        geotiff_path,
        final_array,
        coverage,
        transform,
        central_meridian,
    )
    elapsed = time.perf_counter() - started
    metadata: dict[str, Any] = {
        "format_version": 1,
        "atlas_style_version": ATLAS_STYLE_VERSION,
        "status": "complete",
        "world_config": str(config.world_config),
        "run_id": world.run_id,
        "seed": world.rng_seed,
        "topology": world.topology,
        "source_resolution": world.resolution_set.native.to_dict(),
        "projection": "Equal Earth",
        "projection_crs": atlas_crs.to_string(),
        "projection_wkt": atlas_crs.to_wkt(),
        "central_meridian_deg": central_meridian,
        "width_px": image.width,
        "height_px": image.height,
        "style": asdict(config.style),
        "rendered_river_count": rendered_river_count,
        "elapsed_seconds": elapsed,
        "outputs": {
            "png": {"path": png_path.name, "sha256": _checksum(png_path)},
            "geotiff": {"path": geotiff_path.name, "sha256": _checksum(geotiff_path)},
        },
        "input_artifacts": _input_provenance(results),
        "semantics": {
            "truth_state_modified": False,
            "coast_generalization": (
                "fractional coast"
                if config.style.coastline_width_px == 0
                else f"fractional coast plus {config.style.coastline_width_px}-pixel outline"
            ),
            "lake_generalization": "subgrid fractional area receives bounded display emphasis",
            "river_generalization": "vector reaches filtered and width-scaled by mean discharge",
            "projection_source": "nearest cubed-sphere sampling followed by bilinear atlas warp",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf8")
    return AtlasExportResult(
        png_path=png_path,
        geotiff_path=geotiff_path,
        metadata_path=metadata_path,
        width_px=image.width,
        height_px=image.height,
        central_meridian_deg=central_meridian,
        rendered_river_count=rendered_river_count,
    )


__all__ = [
    "ATLAS_STYLE_VERSION",
    "AtlasExportConfig",
    "AtlasExportResult",
    "AtlasStyleConfig",
    "choose_central_meridian",
    "compose_physical_texture",
    "cubed_sphere_to_equirectangular",
    "export_physical_atlas",
]
