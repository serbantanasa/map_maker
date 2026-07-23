"""Physical centerlines and ecology support derived from accepted L3 hydrology."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from PIL import Image, ImageDraw
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString, Point
import yaml  # type: ignore[import-untyped]
import zarr  # type: ignore[import-untyped]

from .l3_hydrology import (
    _draw_scale,
    _hillshaded_terrain,
    _observed_peak_rss_bytes,
    _rectangular_mask_slices,
)
from .l3_terrain import (
    _diagnostic_font,
    _file_checksum,
    _fsync_paths,
    _sync_zarr_array,
    _tree_checksum,
    _write_json_durable,
    _zarr_dataset,
)
from .regional_handoff import _replace_directory

CHANNEL_GEOMETRY_FORMAT_VERSION = 1
CHANNEL_GEOMETRY_MODEL_VERSION = "l3_channel_geometry_v1"
MONTHS = 12


@dataclass(frozen=True)
class L3ChannelGeometryConfig:
    terrain_dir: Path
    hydrology_dir: Path
    output_dir: Path
    smoothing_iterations: int = 2
    maximum_centerline_offset_cells: float = 0.75
    minimum_centerline_length_ratio: float = 0.70
    maximum_centerline_length_ratio: float = 1.01
    active_month_discharge_m3s: float = 0.15
    reliable_flow_minimum_active_months: int = 6
    riparian_position_between_channel_and_floodplain: float = 0.35
    chunk_rows: int = 262_144
    maximum_peak_memory_gb: float = 24.0
    maximum_storage_gb: float = 2.0
    source_config: Path | None = None

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        output_dir: Path | None = None,
    ) -> "L3ChannelGeometryConfig":
        source = Path(path).expanduser().resolve()
        data = yaml.safe_load(source.read_text(encoding="utf8"))
        if not isinstance(data, Mapping):
            raise TypeError("L3 channel-geometry config must contain a mapping")
        raw_terrain = data.get("terrain_output_dir")
        raw_hydrology = data.get("hydrology_output_dir")
        raw_output = data.get("channel_geometry_output_dir")
        if not raw_terrain or not raw_hydrology or not raw_output:
            raise ValueError(
                "L3 channel geometry requires terrain_output_dir, hydrology_output_dir, "
                "and channel_geometry_output_dir"
            )
        controls = data.get("channel_geometry", {})
        limits = data.get("limits", {})
        if not isinstance(controls, Mapping) or not isinstance(limits, Mapping):
            raise TypeError("L3 channel_geometry and limits controls must be mappings")
        known = {
            "smoothing_iterations",
            "maximum_centerline_offset_cells",
            "minimum_centerline_length_ratio",
            "maximum_centerline_length_ratio",
            "active_month_discharge_m3s",
            "reliable_flow_minimum_active_months",
            "riparian_position_between_channel_and_floodplain",
            "chunk_rows",
        }
        unknown = set(controls) - known
        if unknown:
            raise ValueError(f"Unknown L3 channel-geometry controls: {', '.join(sorted(unknown))}")
        integer_names = {
            "smoothing_iterations",
            "reliable_flow_minimum_active_months",
            "chunk_rows",
        }
        values: dict[str, int | float] = {}
        for name in known:
            if name in controls:
                values[name] = (
                    int(controls[name]) if name in integer_names else float(controls[name])
                )
        config = cls(
            terrain_dir=(source.parent / str(raw_terrain)).resolve(),
            hydrology_dir=(source.parent / str(raw_hydrology)).resolve(),
            output_dir=(
                output_dir.expanduser().resolve()
                if output_dir is not None
                else (source.parent / str(raw_output)).resolve()
            ),
            maximum_peak_memory_gb=float(limits.get("maximum_peak_memory_gb", 24.0)),
            maximum_storage_gb=float(limits.get("maximum_channel_geometry_storage_gb", 2.0)),
            source_config=source,
            **values,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0 <= self.smoothing_iterations <= 4:
            raise ValueError("channel_geometry.smoothing_iterations must be in [0, 4]")
        if not 0.0 < self.maximum_centerline_offset_cells <= 2.0:
            raise ValueError("channel_geometry.maximum_centerline_offset_cells must be in (0, 2]")
        if not 0.0 < self.minimum_centerline_length_ratio <= 1.0:
            raise ValueError("channel_geometry.minimum_centerline_length_ratio must be in (0, 1]")
        if not 1.0 <= self.maximum_centerline_length_ratio <= 1.25:
            raise ValueError(
                "channel_geometry.maximum_centerline_length_ratio must be in [1, 1.25]"
            )
        if self.minimum_centerline_length_ratio > self.maximum_centerline_length_ratio:
            raise ValueError("channel-geometry length-ratio bounds are inverted")
        if (
            not math.isfinite(self.active_month_discharge_m3s)
            or self.active_month_discharge_m3s <= 0.0
        ):
            raise ValueError(
                "channel_geometry.active_month_discharge_m3s must be finite and positive"
            )
        if not 1 <= self.reliable_flow_minimum_active_months <= MONTHS:
            raise ValueError(
                "channel_geometry.reliable_flow_minimum_active_months must be in [1, 12]"
            )
        if not 0.0 <= self.riparian_position_between_channel_and_floodplain <= 1.0:
            raise ValueError(
                "channel_geometry.riparian_position_between_channel_and_floodplain "
                "must be in [0, 1]"
            )
        if not 16_384 <= self.chunk_rows <= 1_048_576:
            raise ValueError("channel_geometry.chunk_rows must be in [16384, 1048576]")
        if not 1.0 <= self.maximum_peak_memory_gb <= 28.0:
            raise ValueError("limits.maximum_peak_memory_gb must be in [1, 28]")
        if not 0.25 <= self.maximum_storage_gb <= 8.0:
            raise ValueError("limits.maximum_channel_geometry_storage_gb must be in [0.25, 8]")


@dataclass(frozen=True)
class L3ChannelGeometryResult:
    output_dir: Path
    manifest_path: Path
    validation_path: Path
    zarr_path: Path
    preview_path: Path
    target_id: str
    display_cell_count: int
    channel_reach_count: int
    reliable_flow_reach_count: int
    validation_passed: bool


@dataclass(frozen=True)
class _ChannelSources:
    target_id: str
    terrain_manifest: dict[str, Any]
    hydrology_manifest: dict[str, Any]
    reaches: pa.Table
    actual_cell_size_m: float
    cell_id: np.ndarray
    row: np.ndarray
    column: np.ndarray
    elevation_m: np.ndarray
    inside_display: np.ndarray
    inside_core: np.ndarray
    lake_fraction: np.ndarray
    wetland_fraction: np.ndarray
    inherited_floodplain_fraction: np.ndarray
    spatial_order: np.ndarray
    inverse_spatial_order: np.ndarray
    height: int
    width: int


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_manifest_outputs(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RuntimeError(f"{output_dir} manifest has no output records")
    for name, raw in outputs.items():
        if not isinstance(raw, Mapping) or "path" not in raw:
            raise RuntimeError(f"Malformed output record {name} in {output_dir}")
        path = output_dir / str(raw["path"])
        if not path.exists():
            raise RuntimeError(f"Cached output is missing: {path}")
        if "sha256" in raw and _file_checksum(path) != raw["sha256"]:
            raise RuntimeError(f"Cached file checksum mismatch: {path}")
        if "sha256_tree" in raw and _tree_checksum(path) != raw["sha256_tree"]:
            raise RuntimeError(f"Cached tree checksum mismatch: {path}")


def _spatial_layout(
    row: np.ndarray,
    column: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    order = np.lexsort((column, row)).astype(np.int32, copy=False)
    inverse = np.empty(len(order), dtype=np.int32)
    inverse[order] = np.arange(len(order), dtype=np.int32)
    height = int(np.max(row) - np.min(row) + 1)
    width = int(np.max(column) - np.min(column) + 1)
    if height * width != len(order):
        raise RuntimeError("L3 channel geometry requires a complete rectangular terrain window")
    sorted_row = row[order] - int(np.min(row))
    sorted_column = column[order] - int(np.min(column))
    expected_row, expected_column = np.indices((height, width), dtype=np.int32)
    if not np.array_equal(sorted_row, expected_row.reshape(-1)) or not np.array_equal(
        sorted_column, expected_column.reshape(-1)
    ):
        raise RuntimeError("L3 channel geometry source is not a dense row-major window")
    return np.ascontiguousarray(order), inverse, height, width


def _load_sources(config: L3ChannelGeometryConfig) -> _ChannelSources:
    terrain_manifest_path = config.terrain_dir / "manifest.json"
    terrain_validation_path = config.terrain_dir / "validation.json"
    terrain_zarr_path = config.terrain_dir / "terrain.zarr"
    hydrology_manifest_path = config.hydrology_dir / "manifest.json"
    hydrology_validation_path = config.hydrology_dir / "validation.json"
    hydrology_zarr_path = config.hydrology_dir / "hydrology.zarr"
    reach_path = config.hydrology_dir / "tables/river_reaches.parquet"
    required = (
        terrain_manifest_path,
        terrain_validation_path,
        terrain_zarr_path,
        hydrology_manifest_path,
        hydrology_validation_path,
        hydrology_zarr_path,
        reach_path,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    terrain_manifest = _load_json(terrain_manifest_path)
    hydrology_manifest = _load_json(hydrology_manifest_path)
    if (
        not terrain_manifest.get("validation_passed")
        or not hydrology_manifest.get("validation_passed")
        or not _load_json(terrain_validation_path).get("passed")
        or not _load_json(hydrology_validation_path).get("passed")
    ):
        raise RuntimeError("L3 channel geometry requires accepted terrain and hydrology")
    if terrain_manifest.get("target_id") != hydrology_manifest.get("target_id"):
        raise RuntimeError("L3 terrain and hydrology target IDs differ")
    _verify_manifest_outputs(config.terrain_dir, terrain_manifest)
    _verify_manifest_outputs(config.hydrology_dir, hydrology_manifest)

    terrain = zarr.open_group(str(terrain_zarr_path), mode="r")
    hydrology = zarr.open_group(str(hydrology_zarr_path), mode="r")
    cell_id = np.asarray(terrain["geometry/cell_id"][:], dtype=np.uint64)
    row = np.asarray(terrain["geometry/row"][:], dtype=np.int32)
    column = np.asarray(terrain["geometry/column"][:], dtype=np.int32)
    elevation = np.asarray(terrain["terrain/elevation_m"][:], dtype=np.float32)
    inside_display = np.asarray(hydrology["geometry/inside_display_window"][:], dtype=bool)
    inside_core = np.asarray(hydrology["geometry/inside_routed_catchment_core"][:], dtype=bool)
    lake = np.asarray(hydrology["surface/lake_fraction"][:], dtype=np.float32)
    wetland = np.asarray(hydrology["surface/wetland_fraction"][:], dtype=np.float32)
    floodplain = np.asarray(hydrology["routing/floodplain_fraction"][:], dtype=np.float32)
    arrays = (
        row,
        column,
        elevation,
        inside_display,
        inside_core,
        lake,
        wetland,
        floodplain,
    )
    if any(len(values) != len(cell_id) for values in arrays):
        raise RuntimeError("L3 source arrays have inconsistent cell counts")
    order, inverse, height, width = _spatial_layout(row, column)
    reaches = pq.read_table(reach_path).combine_chunks()
    if reaches.num_rows == 0:
        raise RuntimeError("L3 hydrology contains no reported reaches")
    required_columns = {
        "reach_id",
        "cell_path",
        "reach_kind",
        "discharge_mean",
        "discharge_seasonal",
        "channel_width_m",
        "floodplain_width_m",
        "valley_width_m",
        "polyline_on_cubed_sphere",
    }
    missing = required_columns - set(reaches.column_names)
    if missing:
        raise RuntimeError(f"L3 reach table lacks columns: {', '.join(sorted(missing))}")
    actual_cell_size_m = float(terrain_manifest["hierarchy"]["actual_area_equivalent_cell_size_m"])
    if not math.isfinite(actual_cell_size_m) or actual_cell_size_m <= 0.0:
        raise RuntimeError("L3 terrain has no valid physical cell size")
    return _ChannelSources(
        target_id=str(terrain_manifest["target_id"]),
        terrain_manifest=terrain_manifest,
        hydrology_manifest=hydrology_manifest,
        reaches=reaches,
        actual_cell_size_m=actual_cell_size_m,
        cell_id=np.ascontiguousarray(cell_id),
        row=np.ascontiguousarray(row),
        column=np.ascontiguousarray(column),
        elevation_m=np.ascontiguousarray(elevation),
        inside_display=np.ascontiguousarray(inside_display),
        inside_core=np.ascontiguousarray(inside_core),
        lake_fraction=np.ascontiguousarray(lake),
        wetland_fraction=np.ascontiguousarray(wetland),
        inherited_floodplain_fraction=np.ascontiguousarray(floodplain),
        spatial_order=order,
        inverse_spatial_order=inverse,
        height=height,
        width=width,
    )


def _fingerprint(
    config: L3ChannelGeometryConfig,
    sources: _ChannelSources,
) -> tuple[str, dict[str, Any]]:
    components = {
        "format_version": CHANNEL_GEOMETRY_FORMAT_VERSION,
        "model_version": CHANNEL_GEOMETRY_MODEL_VERSION,
        "terrain_manifest_sha256": _file_checksum(config.terrain_dir / "manifest.json"),
        "terrain_zarr_sha256": sources.terrain_manifest["outputs"]["terrain_zarr"]["sha256_tree"],
        "hydrology_manifest_sha256": _file_checksum(config.hydrology_dir / "manifest.json"),
        "hydrology_zarr_sha256": sources.hydrology_manifest["outputs"]["hydrology_zarr"][
            "sha256_tree"
        ],
        "river_reaches_sha256": _file_checksum(
            config.hydrology_dir / "tables/river_reaches.parquet"
        ),
        "config_sha256": (_file_checksum(config.source_config) if config.source_config else None),
        "controls": asdict(config)
        | {
            "terrain_dir": str(config.terrain_dir),
            "hydrology_dir": str(config.hydrology_dir),
            "output_dir": str(config.output_dir),
            "source_config": str(config.source_config) if config.source_config else None,
        },
        "orchestrator_sha256": _file_checksum(Path(__file__)),
        "scipy_version": package_version("scipy"),
        "rasterio_version": package_version("rasterio"),
        "shapely_version": package_version("shapely"),
    }
    return _canonical_hash(components), components


def _chaikin_polyline(points: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or len(result) == 0:
        raise ValueError("polyline points must be a non-empty matrix")
    if len(result) < 3 or iterations == 0:
        return result.copy()
    for _ in range(iterations):
        first = 0.75 * result[:-1] + 0.25 * result[1:]
        second = 0.25 * result[:-1] + 0.75 * result[1:]
        refined = np.empty((2 * len(result), result.shape[1]), dtype=np.float64)
        refined[0] = result[0]
        refined[-1] = result[-1]
        refined[1:-1:2] = first
        refined[2:-1:2] = second
        result = refined
    return result


def _polyline_distance(points: np.ndarray, scale: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    distance = np.zeros(len(points), dtype=np.float64)
    if len(points) > 1:
        distance[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1) * scale)
    return distance


def _maximum_polyline_offset(points: np.ndarray, reference: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if len(reference) == 1:
        return float(np.max(np.linalg.norm(points - reference[0], axis=1), initial=0.0))
    start = reference[:-1]
    segment = reference[1:] - start
    denominator = np.sum(segment * segment, axis=1)
    delta = points[:, None, :] - start[None, :, :]
    fraction = np.divide(
        np.sum(delta * segment[None, :, :], axis=2),
        denominator[None, :],
        out=np.zeros((len(points), len(segment)), dtype=np.float64),
        where=denominator[None, :] > 0.0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    projected = start[None, :, :] + fraction[..., None] * segment[None, :, :]
    distance = np.linalg.norm(points[:, None, :] - projected, axis=2)
    return float(np.max(np.min(distance, axis=1), initial=0.0))


def _mean_turn_angle_deg(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = (lengths[:-1] > 0.0) & (lengths[1:] > 0.0)
    if not np.any(valid):
        return 0.0
    cosine = np.sum(vectors[:-1] * vectors[1:], axis=1)
    cosine = cosine[valid] / (lengths[:-1][valid] * lengths[1:][valid])
    return float(np.mean(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))


def _normalize_xyz(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    norm = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("centerline spherical coordinate has zero length")
    return points / norm


def _stable_paths_to_parent_rows(
    paths: list[list[int]],
    cell_id: np.ndarray,
) -> list[np.ndarray]:
    sorted_rows = np.argsort(cell_id)
    sorted_ids = cell_id[sorted_rows]
    result: list[np.ndarray] = []
    for path in paths:
        values = np.asarray(path, dtype=np.uint64)
        positions = np.searchsorted(sorted_ids, values)
        valid = positions < len(sorted_ids)
        if np.any(valid):
            valid[valid] &= sorted_ids[positions[valid]] == values[valid]
        if not np.all(valid):
            raise RuntimeError("Reach path references a cell outside the L3 terrain window")
        result.append(np.ascontiguousarray(sorted_rows[positions], dtype=np.int32))
    return result


def _build_centerline_table(
    sources: _ChannelSources,
    config: L3ChannelGeometryConfig,
) -> tuple[pa.Table, dict[str, Any]]:
    table = sources.reaches
    reach_id = np.asarray(table["reach_id"], dtype=np.int32)
    if np.any(reach_id < 0) or len(np.unique(reach_id)) != len(reach_id):
        raise RuntimeError("L3 reach IDs must be unique and nonnegative")
    raw_paths = table["cell_path"].to_pylist()
    parent_paths = _stable_paths_to_parent_rows(raw_paths, sources.cell_id)
    raw_xyz_paths = table["polyline_on_cubed_sphere"].to_pylist()
    kind = table["reach_kind"].to_pylist()
    seasonal = np.asarray(
        table["discharge_seasonal"].combine_chunks().values, dtype=np.float32
    ).reshape(table.num_rows, MONTHS)
    active_month_count = np.count_nonzero(
        seasonal >= config.active_month_discharge_m3s,
        axis=1,
    ).astype(np.uint8)
    reliable = active_month_count >= config.reliable_flow_minimum_active_months

    minimum_row = int(np.min(sources.row))
    minimum_column = int(np.min(sources.column))
    smooth_xy_values: list[list[list[float]]] = []
    smooth_xyz_values: list[list[list[float]]] = []
    smooth_distance_values: list[list[float]] = []
    raw_length = np.zeros(table.num_rows, dtype=np.float32)
    smooth_length = np.zeros(table.num_rows, dtype=np.float32)
    maximum_offset = np.zeros(table.num_rows, dtype=np.float32)
    smoothing_applied = np.zeros(table.num_rows, dtype=np.uint8)
    simple = np.ones(table.num_rows, dtype=bool)
    raw_turn = np.zeros(table.num_rows, dtype=np.float32)
    smooth_turn = np.zeros(table.num_rows, dtype=np.float32)
    endpoint_drift = np.zeros(table.num_rows, dtype=np.float32)

    for index, (parent_path, raw_xyz, reach_kind) in enumerate(
        zip(parent_paths, raw_xyz_paths, kind, strict=True)
    ):
        raw_xy = np.stack(
            (
                sources.column[parent_path] - minimum_column + 0.5,
                sources.row[parent_path] - minimum_row + 0.5,
            ),
            axis=1,
        ).astype(np.float64)
        raw_xyz_array = np.asarray(raw_xyz, dtype=np.float64)
        if raw_xyz_array.shape != (len(raw_xy), 3):
            raise RuntimeError("Reach path and spherical polyline lengths differ")
        iterations = (
            config.smoothing_iterations if reach_kind == "channel" and len(raw_xy) >= 3 else 0
        )
        smooth_xy = _chaikin_polyline(raw_xy, iterations)
        smooth_xyz = _normalize_xyz(_chaikin_polyline(raw_xyz_array, iterations))
        smooth_xy[0] = raw_xy[0]
        smooth_xy[-1] = raw_xy[-1]
        smooth_xyz[0] = raw_xyz_array[0]
        smooth_xyz[-1] = raw_xyz_array[-1]
        distance = _polyline_distance(smooth_xy, sources.actual_cell_size_m)
        raw_distances = _polyline_distance(raw_xy, sources.actual_cell_size_m)
        raw_length[index] = raw_distances[-1]
        smooth_length[index] = distance[-1]
        maximum_offset[index] = (
            _maximum_polyline_offset(smooth_xy, raw_xy) * sources.actual_cell_size_m
        )
        smoothing_applied[index] = iterations
        raw_turn[index] = _mean_turn_angle_deg(raw_xy)
        smooth_turn[index] = _mean_turn_angle_deg(smooth_xy)
        endpoint_drift[index] = (
            max(
                float(np.linalg.norm(smooth_xy[0] - raw_xy[0])),
                float(np.linalg.norm(smooth_xy[-1] - raw_xy[-1])),
            )
            * sources.actual_cell_size_m
        )
        if len(smooth_xy) > 1:
            simple[index] = bool(LineString(smooth_xy).is_simple)
        smooth_xy_values.append(smooth_xy.astype(np.float32).tolist())
        smooth_xyz_values.append(smooth_xyz.astype(np.float32).tolist())
        smooth_distance_values.append(distance.astype(np.float32).tolist())

    channel = np.asarray([value == "channel" for value in kind], dtype=bool)
    positive = channel & (raw_length > 0.0)
    length_ratio = np.ones(table.num_rows, dtype=np.float32)
    length_ratio[positive] = smooth_length[positive] / raw_length[positive]
    table = (
        table.append_column(
            "smoothed_centerline_xy_cells",
            pa.array(
                smooth_xy_values,
                type=pa.list_(pa.list_(pa.float32(), 2)),
            ),
        )
        .append_column(
            "smoothed_centerline_on_cubed_sphere",
            pa.array(
                smooth_xyz_values,
                type=pa.list_(pa.list_(pa.float32(), 3)),
            ),
        )
        .append_column(
            "smoothed_centerline_distance_m",
            pa.array(smooth_distance_values, type=pa.list_(pa.float32())),
        )
        .append_column(
            "raw_centerline_length_m",
            pa.array(raw_length, type=pa.float32()),
        )
        .append_column(
            "smoothed_centerline_length_m",
            pa.array(smooth_length, type=pa.float32()),
        )
        .append_column(
            "centerline_length_ratio",
            pa.array(length_ratio, type=pa.float32()),
        )
        .append_column(
            "maximum_centerline_offset_m",
            pa.array(maximum_offset, type=pa.float32()),
        )
        .append_column(
            "centerline_smoothing_iterations",
            pa.array(smoothing_applied, type=pa.uint8()),
        )
        .append_column(
            "raw_mean_turn_angle_deg",
            pa.array(raw_turn, type=pa.float32()),
        )
        .append_column(
            "smoothed_mean_turn_angle_deg",
            pa.array(smooth_turn, type=pa.float32()),
        )
        .append_column(
            "active_month_count",
            pa.array(active_month_count, type=pa.uint8()),
        )
        .append_column(
            "flow_persistence_fraction",
            pa.array(active_month_count / MONTHS, type=pa.float32()),
        )
        .append_column(
            "reliable_flow",
            pa.array(reliable, type=pa.bool_()),
        )
        .append_column(
            "centerline_is_simple",
            pa.array(simple, type=pa.bool_()),
        )
    )
    channel_ratios = length_ratio[positive]
    channel_offsets = maximum_offset[channel]
    channel_raw_turn = raw_turn[channel & (raw_turn > 0.0)]
    channel_smooth_turn = smooth_turn[channel & (raw_turn > 0.0)]
    return table, {
        "source_reach_count": sources.reaches.num_rows,
        "channel_reach_count": int(np.count_nonzero(channel)),
        "connector_reach_count": int(np.count_nonzero(~channel)),
        "reliable_flow_reach_count": int(np.count_nonzero(channel & reliable)),
        "zero_flow_month_present_reach_count": int(
            np.count_nonzero(channel & (active_month_count < MONTHS))
        ),
        "maximum_endpoint_drift_m": float(np.max(endpoint_drift, initial=0.0)),
        "maximum_centerline_offset_m": float(np.max(channel_offsets, initial=0.0)),
        "minimum_centerline_length_ratio": float(np.min(channel_ratios, initial=1.0)),
        "maximum_centerline_length_ratio": float(np.max(channel_ratios, initial=1.0)),
        "non_simple_source_or_smoothed_centerline_count": int(np.count_nonzero(channel & ~simple)),
        "raw_mean_turn_angle_deg": (
            float(np.mean(channel_raw_turn)) if len(channel_raw_turn) else 0.0
        ),
        "smoothed_mean_turn_angle_deg": (
            float(np.mean(channel_smooth_turn)) if len(channel_smooth_turn) else 0.0
        ),
    }


def _line_geometry(points: list[list[float]]) -> LineString | Point:
    if len(points) == 1:
        return Point(points[0])
    return LineString(points)


def _rasterize_centerlines(
    table: pa.Table,
    height: int,
    width: int,
    *,
    reliable_only: bool,
) -> np.ndarray:
    kind = np.asarray(table["reach_kind"].to_pylist(), dtype=object)
    reliable = np.asarray(table["reliable_flow"], dtype=bool)
    discharge = np.asarray(table["discharge_mean"], dtype=np.float32)
    reach_id = np.asarray(table["reach_id"], dtype=np.int32)
    selected = kind == "channel"
    if reliable_only:
        selected &= reliable
    rows = np.flatnonzero(selected)
    if not len(rows):
        raise RuntimeError("No reaches satisfy the requested centerline raster subset")
    rows = rows[np.lexsort((reach_id[rows], discharge[rows]))]
    polylines = table["smoothed_centerline_xy_cells"].to_pylist()
    shapes = [(_line_geometry(polylines[int(row)]), int(reach_id[row]) + 1) for row in rows]
    return rasterize(
        shapes=shapes,
        out_shape=(height, width),
        fill=0,
        transform=Affine.identity(),
        all_touched=True,
        dtype=np.int32,
    )


def _distance_and_nearest(
    labels: np.ndarray,
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if labels.ndim != 2 or not np.any(labels > 0):
        raise ValueError("centerline labels must be a non-empty two-dimensional raster")
    distance, indices = distance_transform_edt(
        labels == 0,
        sampling=(cell_size_m, cell_size_m),
        return_indices=True,
    )
    nearest = labels[indices[0], indices[1]].astype(np.int32) - 1
    if np.any(nearest < 0):
        raise RuntimeError("distance transform failed to assign a nearest reach")
    return np.asarray(distance, dtype=np.float32), np.ascontiguousarray(nearest)


def _cross_section_fraction(
    distance_m: np.ndarray,
    width_m: np.ndarray,
    cell_size_m: float,
) -> np.ndarray:
    distance = np.asarray(distance_m, dtype=np.float64)
    half_width = 0.5 * np.maximum(np.asarray(width_m, dtype=np.float64), 0.0)
    half_cell = 0.5 * cell_size_m
    left = np.maximum(distance - half_cell, -half_width)
    right = np.minimum(distance + half_cell, half_width)
    return np.asarray(np.clip((right - left) / cell_size_m, 0.0, 1.0), dtype=np.float32)


def _reach_lookup(table: pa.Table, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    reach_id = np.asarray(table["reach_id"], dtype=np.int32)
    output = np.zeros(int(np.max(reach_id)) + 1, dtype=dtype)
    output[reach_id] = np.asarray(table[name], dtype=dtype)
    return output


def _support_fields(
    table: pa.Table,
    sources: _ChannelSources,
    config: L3ChannelGeometryConfig,
) -> dict[str, np.ndarray]:
    labels = _rasterize_centerlines(
        table,
        sources.height,
        sources.width,
        reliable_only=False,
    )
    reliable_labels = _rasterize_centerlines(
        table,
        sources.height,
        sources.width,
        reliable_only=True,
    )
    distance, nearest = _distance_and_nearest(labels, sources.actual_cell_size_m)
    reliable_distance, nearest_reliable = _distance_and_nearest(
        reliable_labels,
        sources.actual_cell_size_m,
    )
    channel_width = _reach_lookup(table, "channel_width_m", np.dtype(np.float32))
    floodplain_width = _reach_lookup(table, "floodplain_width_m", np.dtype(np.float32))
    valley_width = _reach_lookup(table, "valley_width_m", np.dtype(np.float32))
    persistence = _reach_lookup(table, "flow_persistence_fraction", np.dtype(np.float32))
    nearest_channel_width = channel_width[nearest]
    nearest_floodplain_width = np.maximum(
        floodplain_width[nearest],
        nearest_channel_width,
    )
    nearest_valley_width = np.maximum(
        valley_width[nearest],
        nearest_floodplain_width,
    )
    riparian_width = (
        nearest_channel_width
        + (nearest_floodplain_width - nearest_channel_width)
        * config.riparian_position_between_channel_and_floodplain
    )
    channel_fraction = _cross_section_fraction(
        distance,
        nearest_channel_width,
        sources.actual_cell_size_m,
    )
    riparian_fraction = _cross_section_fraction(
        distance,
        riparian_width,
        sources.actual_cell_size_m,
    )
    floodplain_fraction = _cross_section_fraction(
        distance,
        nearest_floodplain_width,
        sources.actual_cell_size_m,
    )
    inherited_floodplain = sources.inherited_floodplain_fraction[sources.spatial_order].reshape(
        sources.height, sources.width
    )
    floodplain_fraction = np.maximum(floodplain_fraction, inherited_floodplain)
    valley_fraction = np.maximum(
        _cross_section_fraction(
            distance,
            nearest_valley_width,
            sources.actual_cell_size_m,
        ),
        floodplain_fraction,
    )
    riparian_fraction = np.minimum(
        np.maximum(riparian_fraction, channel_fraction),
        floodplain_fraction,
    )
    return {
        "centerline_seed": labels > 0,
        "reliable_flow_centerline_seed": reliable_labels > 0,
        "nearest_channel_reach_id": nearest,
        "nearest_reliable_flow_reach_id": nearest_reliable,
        "distance_to_channel_m": distance,
        "distance_to_reliable_flow_channel_m": reliable_distance,
        "nearest_channel_flow_persistence_fraction": persistence[nearest],
        "channel_fraction": channel_fraction,
        "riparian_fraction": riparian_fraction,
        "floodplain_fraction": floodplain_fraction,
        "valley_fraction": valley_fraction,
    }


def _to_parent_major(values: np.ndarray, spatial_order: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    flattened = values.reshape(-1)
    output = np.empty_like(flattened)
    output[spatial_order] = flattened
    return output


def _write_support_zarr(
    partial: Path,
    config: L3ChannelGeometryConfig,
    sources: _ChannelSources,
    run_fingerprint: str,
    fields: Mapping[str, np.ndarray],
) -> Path:
    zarr_path = partial / "channel_geometry.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    cell_count = len(sources.cell_id)
    chunks = (min(config.chunk_rows, cell_count),)
    root.attrs.update(
        {
            "format_version": CHANNEL_GEOMETRY_FORMAT_VERSION,
            "model_version": CHANNEL_GEOMETRY_MODEL_VERSION,
            "status": "partial",
            "target_id": sources.target_id,
            "run_fingerprint": run_fingerprint,
            "cell_count": cell_count,
            "parent_major_storage": True,
            "actual_cell_size_m": sources.actual_cell_size_m,
            "support_semantics": (
                "fractional cross-sectional influence around smoothed vectors; "
                "not categorical whole-cell river occupancy"
            ),
        }
    )
    geometry = root.require_group("geometry")
    support = root.require_group("support")
    datasets = (
        (
            geometry,
            "inside_display_window",
            bool,
            sources.inside_display,
            "visible cells with complete terrain, hydrology, and channel support",
        ),
        (
            geometry,
            "inside_routed_catchment_core",
            bool,
            sources.inside_core,
            "fine routed basin used for hydrological and ecology acceptance",
        ),
    )
    for group, name, dtype, values, semantics in datasets:
        dataset = _zarr_dataset(
            group,
            name,
            shape=(cell_count,),
            dtype=dtype,
            chunks=chunks,
            semantics=semantics,
        )
        dataset[:] = values
        _sync_zarr_array(zarr_path, f"geometry/{name}")
    field_specs = {
        "centerline_seed": (
            bool,
            "cells touched by a smoothed physical channel vector",
        ),
        "reliable_flow_centerline_seed": (
            bool,
            "channel-vector cells meeting the declared active-month reliability threshold",
        ),
        "nearest_channel_reach_id": (
            np.int32,
            "stable ID of the nearest physical channel reach",
        ),
        "nearest_reliable_flow_reach_id": (
            np.int32,
            "stable ID of the nearest reach meeting the active-month reliability threshold",
        ),
        "distance_to_channel_m": (
            np.float32,
            "cell-center distance to the nearest smoothed physical channel support cell",
        ),
        "distance_to_reliable_flow_channel_m": (
            np.float32,
            "cell-center distance to the nearest reliably flowing channel support cell",
        ),
        "nearest_channel_flow_persistence_fraction": (
            np.float32,
            "fraction of months meeting the configured active-flow discharge threshold",
        ),
        "channel_fraction": (
            np.float32,
            "sub-cell channel cross-section support around the smoothed centerline",
        ),
        "riparian_fraction": (
            np.float32,
            "nested ecology-facing riparian support between channel and floodplain widths",
        ),
        "floodplain_fraction": (
            np.float32,
            "nested physical floodplain support including inherited routed potential",
        ),
        "valley_fraction": (
            np.float32,
            "nested valley-bottom support derived from physical reach width",
        ),
    }
    for name, (dtype, semantics) in field_specs.items():
        dataset = _zarr_dataset(
            support,
            name,
            shape=(cell_count,),
            dtype=dtype,
            chunks=chunks,
            semantics=semantics,
        )
        dataset[:] = _to_parent_major(fields[name], sources.spatial_order)
        _sync_zarr_array(zarr_path, f"support/{name}")
    root.attrs["status"] = "complete"
    zarr.consolidate_metadata(str(zarr_path))
    _fsync_paths([zarr_path / ".zattrs", zarr_path / ".zmetadata"])
    return zarr_path


def _validate(
    table: pa.Table,
    fields: Mapping[str, np.ndarray],
    sources: _ChannelSources,
    config: L3ChannelGeometryConfig,
    centerline_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    fractions = [
        np.asarray(fields[name], dtype=np.float32)
        for name in (
            "channel_fraction",
            "riparian_fraction",
            "floodplain_fraction",
            "valley_fraction",
        )
    ]
    finite = all(np.all(np.isfinite(values)) for values in fields.values())
    bounded = all(np.all((values >= 0.0) & (values <= 1.0)) for values in fractions)
    channel, riparian, floodplain, valley = fractions
    nesting_error = float(
        max(
            np.max(channel - riparian, initial=0.0),
            np.max(riparian - floodplain, initial=0.0),
            np.max(floodplain - valley, initial=0.0),
        )
    )
    display_spatial = sources.inside_display[sources.spatial_order].reshape(
        sources.height,
        sources.width,
    )
    core_spatial = sources.inside_core[sources.spatial_order].reshape(
        sources.height,
        sources.width,
    )
    distance = np.asarray(fields["distance_to_channel_m"], dtype=np.float32)
    reliable_distance = np.asarray(fields["distance_to_reliable_flow_channel_m"], dtype=np.float32)
    centerline_seed = np.asarray(fields["centerline_seed"], dtype=bool)
    reliable_seed = np.asarray(fields["reliable_flow_centerline_seed"], dtype=bool)
    reach_id = np.asarray(table["reach_id"], dtype=np.int32)
    source_reach_id = np.asarray(sources.reaches["reach_id"], dtype=np.int32)
    graph_identity_preserved = bool(
        table.num_rows == sources.reaches.num_rows
        and np.array_equal(reach_id, source_reach_id)
        and table["from_cell_id"].equals(sources.reaches["from_cell_id"])
        and table["to_cell_id"].equals(sources.reaches["to_cell_id"])
        and table["downstream_reach_id"].equals(sources.reaches["downstream_reach_id"])
    )
    maximum_allowed_offset_m = config.maximum_centerline_offset_cells * sources.actual_cell_size_m
    gates = {
        "source_graph_identity_preserved": graph_identity_preserved,
        "endpoint_anchor_valid": centerline_metrics["maximum_endpoint_drift_m"] <= 1e-4,
        "centerline_corridor_valid": (
            centerline_metrics["maximum_centerline_offset_m"] <= maximum_allowed_offset_m + 1e-4
        ),
        "centerline_length_ratio_valid": (
            centerline_metrics["minimum_centerline_length_ratio"]
            >= config.minimum_centerline_length_ratio
            and centerline_metrics["maximum_centerline_length_ratio"]
            <= config.maximum_centerline_length_ratio
        ),
        "centerline_simple_valid": (
            centerline_metrics["non_simple_source_or_smoothed_centerline_count"] == 0
        ),
        "reliable_flow_subset_nonempty": (centerline_metrics["reliable_flow_reach_count"] > 0),
        "support_finite": finite,
        "support_fraction_bounded": bounded,
        "support_nesting_valid": nesting_error <= 1e-6,
        "display_distance_complete": bool(
            np.all(np.isfinite(distance[display_spatial]))
            and np.all(np.isfinite(reliable_distance[display_spatial]))
        ),
        "centerline_distance_origin_valid": bool(
            np.max(np.abs(distance[centerline_seed]), initial=0.0) <= 1e-6
            and np.max(np.abs(reliable_distance[reliable_seed]), initial=0.0) <= 1e-6
        ),
    }
    validation = {
        "model": CHANNEL_GEOMETRY_MODEL_VERSION,
        "target_id": sources.target_id,
        "cell_count": len(sources.cell_id),
        "display_cell_count": int(np.count_nonzero(display_spatial)),
        "core_cell_count": int(np.count_nonzero(core_spatial)),
        "actual_cell_size_m": sources.actual_cell_size_m,
        "maximum_allowed_centerline_offset_m": maximum_allowed_offset_m,
        "maximum_support_nesting_error": nesting_error,
        "maximum_display_distance_to_channel_m": float(
            np.max(distance[display_spatial], initial=0.0)
        ),
        "maximum_core_distance_to_channel_m": float(np.max(distance[core_spatial], initial=0.0)),
        "display_channel_support_cell_count": int(
            np.count_nonzero((channel > 0.0) & display_spatial)
        ),
        "display_riparian_support_cell_count": int(
            np.count_nonzero((riparian > 0.0) & display_spatial)
        ),
        "display_floodplain_support_cell_count": int(
            np.count_nonzero((floodplain > 0.0) & display_spatial)
        ),
        "display_valley_support_cell_count": int(
            np.count_nonzero((valley > 0.0) & display_spatial)
        ),
        "flow_reliability_semantics": (
            f"at least {config.reliable_flow_minimum_active_months} of 12 months at or above "
            f"{config.active_month_discharge_m3s:g} m3/s; not a perennial classification "
            "because groundwater/baseflow is not modeled"
        ),
        **centerline_metrics,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    return validation


def _render(
    path: Path,
    table: pa.Table,
    fields: Mapping[str, np.ndarray],
    sources: _ChannelSources,
    validation: Mapping[str, Any],
) -> None:
    order = sources.spatial_order
    elevation = sources.elevation_m[order].reshape(sources.height, sources.width)
    colors = _hillshaded_terrain(elevation, sources.actual_cell_size_m)
    floodplain = np.asarray(fields["floodplain_fraction"], dtype=np.float32)
    riparian = np.asarray(fields["riparian_fraction"], dtype=np.float32)
    floodplain_alpha = np.clip(floodplain * 0.30, 0.0, 0.30)
    riparian_alpha = np.clip(riparian * 0.45, 0.0, 0.45)
    floodplain_color = np.asarray((117.0, 155.0, 99.0))
    riparian_color = np.asarray((55.0, 124.0, 78.0))
    colors = (
        colors * (1.0 - floodplain_alpha[..., None])
        + floodplain_color * floodplain_alpha[..., None]
    )
    colors = colors * (1.0 - riparian_alpha[..., None]) + riparian_color * riparian_alpha[..., None]
    lake = sources.lake_fraction[order].reshape(sources.height, sources.width)
    wetland = sources.wetland_fraction[order].reshape(sources.height, sources.width)
    lake_alpha = np.clip(lake, 0.0, 1.0)
    wetland_alpha = np.clip(wetland * 0.65, 0.0, 0.65)
    colors = (
        colors * (1.0 - wetland_alpha[..., None])
        + np.asarray((62.0, 127.0, 91.0)) * wetland_alpha[..., None]
    )
    colors = (
        colors * (1.0 - lake_alpha[..., None])
        + np.asarray((45.0, 132.0, 168.0)) * lake_alpha[..., None]
    )
    full_image = Image.fromarray(np.clip(colors, 0, 255).astype(np.uint8), mode="RGB")
    draw_full = ImageDraw.Draw(full_image)
    kind = np.asarray(table["reach_kind"].to_pylist(), dtype=object)
    reliable = np.asarray(table["reliable_flow"], dtype=bool)
    discharge = np.asarray(table["discharge_mean"], dtype=np.float32)
    reach_id = np.asarray(table["reach_id"], dtype=np.int32)
    rows = np.flatnonzero(kind == "channel")
    rows = rows[np.lexsort((reach_id[rows], discharge[rows]))]
    polylines = table["smoothed_centerline_xy_cells"].to_pylist()
    for row in rows:
        q = float(discharge[row])
        line_width = 1 + int(q >= 10.0) + int(q >= 100.0) + int(q >= 750.0)
        color = (22, 89, 157) if reliable[row] else (45, 139, 186)
        points = [(float(x), float(y)) for x, y in polylines[int(row)]]
        if len(points) == 1:
            x, y = points[0]
            draw_full.point((x, y), fill=color)
        else:
            draw_full.line(points, fill=color, width=line_width, joint="curve")

    display = sources.inside_display[order].reshape(sources.height, sources.width)
    row_slice, column_slice = _rectangular_mask_slices(display)
    map_image = full_image.crop(
        (column_slice.start, row_slice.start, column_slice.stop, row_slice.stop)
    )
    title_height = 52
    footer_height = 76
    legend_width = 350
    canvas = Image.new(
        "RGB",
        (map_image.width + legend_width, map_image.height + title_height + footer_height),
        (240, 241, 237),
    )
    canvas.paste(map_image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    title_font = _diagnostic_font(22)
    label_font = _diagnostic_font(17)
    small_font = _diagnostic_font(15)
    draw.text(
        (18, 13),
        "L3 physical river centerlines and ecology support",
        fill=(25, 30, 27),
        font=title_font,
    )
    draw.line(
        (map_image.width, 0, map_image.width, canvas.height),
        fill=(178, 181, 174),
        width=1,
    )
    x = map_image.width + 24
    y = 20
    draw.text((x, y), "Channel geometry", fill=(25, 30, 27), font=title_font)
    y += 55
    for color, label in (
        ((22, 89, 157), "reliably flowing channel"),
        ((45, 139, 186), "strongly seasonal channel"),
    ):
        draw.line((x, y + 8, x + 40, y + 8), fill=color, width=3)
        draw.text((x + 52, y), label, fill=(35, 39, 36), font=small_font)
        y += 34
    for color, label in (
        ((55, 124, 78), "riparian support"),
        ((117, 155, 99), "floodplain / valley support"),
        ((45, 132, 168), "lake"),
        ((62, 127, 91), "wetland"),
    ):
        draw.rectangle((x, y, x + 34, y + 18), fill=color, outline=(55, 59, 56))
        draw.text((x + 48, y - 1), label, fill=(35, 39, 36), font=small_font)
        y += 34
    y += 14
    draw.text((x, y), "Summary", fill=(25, 30, 27), font=label_font)
    y += 34
    for line in (
        f"{validation['channel_reach_count']:,} channel reaches",
        f"{validation['reliable_flow_reach_count']:,} reliable-flow reaches",
        (f"max centerline offset " f"{validation['maximum_centerline_offset_m']:.1f} m"),
        (
            f"turn angle {validation['raw_mean_turn_angle_deg']:.1f} -> "
            f"{validation['smoothed_mean_turn_angle_deg']:.1f} deg"
        ),
        f"{validation['display_cell_count']:,} fully supported display cells",
        "reliable flow is not perennial flow",
        "baseflow is not yet modeled",
    ):
        draw.text((x, y), line, fill=(50, 54, 51), font=small_font)
        y += 26
    _draw_scale(
        draw,
        map_image.width,
        map_image.height,
        title_height,
        sources.actual_cell_size_m,
    )
    canvas.save(path, optimize=True)


def _existing_result(
    config: L3ChannelGeometryConfig,
    run_fingerprint: str,
) -> L3ChannelGeometryResult | None:
    manifest_path = config.output_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _load_json(manifest_path)
    if manifest.get("run_fingerprint") != run_fingerprint or not manifest.get("validation_passed"):
        return None
    _verify_manifest_outputs(config.output_dir, manifest)
    summary = manifest["summary"]
    return L3ChannelGeometryResult(
        output_dir=config.output_dir,
        manifest_path=manifest_path,
        validation_path=config.output_dir / "validation.json",
        zarr_path=config.output_dir / "channel_geometry.zarr",
        preview_path=config.output_dir / "channel_geometry.png",
        target_id=str(manifest["target_id"]),
        display_cell_count=int(summary["display_cell_count"]),
        channel_reach_count=int(summary["channel_reach_count"]),
        reliable_flow_reach_count=int(summary["reliable_flow_reach_count"]),
        validation_passed=True,
    )


def _prepare_partial(
    partial: Path,
    run_fingerprint: str,
) -> tuple[bool, pa.Table | None, dict[str, Any] | None]:
    state_path = partial / "centerline_state.json"
    table_path = partial / "tables/river_centerlines.parquet"
    if state_path.exists() and table_path.exists():
        state = _load_json(state_path)
        if (
            state.get("run_fingerprint") == run_fingerprint
            and state.get("table_sha256") == _file_checksum(table_path)
            and isinstance(state.get("metrics"), Mapping)
        ):
            return True, pq.read_table(table_path).combine_chunks(), dict(state["metrics"])
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    (partial / "tables").mkdir()
    return False, None, None


def generate_l3_channel_geometry(
    config: L3ChannelGeometryConfig,
) -> L3ChannelGeometryResult:
    """Smooth accepted river vectors and derive ecology-facing support fields."""

    started = time.perf_counter()
    sources = _load_sources(config)
    run_fingerprint, fingerprint_components = _fingerprint(config, sources)
    existing = _existing_result(config, run_fingerprint)
    if existing is not None:
        return existing
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = config.output_dir.with_name(f".{config.output_dir.name}.partial")
    resumed, centerlines, centerline_metrics = _prepare_partial(
        partial,
        run_fingerprint,
    )
    centerline_path = partial / "tables/river_centerlines.parquet"
    if centerlines is None or centerline_metrics is None:
        centerlines, centerline_metrics = _build_centerline_table(sources, config)
        pq.write_table(
            centerlines,
            centerline_path,
            compression="zstd",
            use_dictionary=True,
        )
        _fsync_paths([centerline_path])
        _write_json_durable(
            partial / "centerline_state.json",
            {
                "run_fingerprint": run_fingerprint,
                "table_sha256": _file_checksum(centerline_path),
                "metrics": centerline_metrics,
            },
        )

    fields = _support_fields(centerlines, sources, config)
    zarr_path = _write_support_zarr(
        partial,
        config,
        sources,
        run_fingerprint,
        fields,
    )
    validation = _validate(
        centerlines,
        fields,
        sources,
        config,
        centerline_metrics,
    )
    preview_path = partial / "channel_geometry.png"
    _render(preview_path, centerlines, fields, sources, validation)
    observed_peak = _observed_peak_rss_bytes()
    estimated_peak = int(len(sources.cell_id) * 160)
    peak_memory = max(observed_peak, estimated_peak)
    storage_bytes = sum(path.stat().st_size for path in partial.rglob("*") if path.is_file())
    validation.update(
        {
            "observed_peak_rss_bytes": observed_peak,
            "estimated_peak_working_set_bytes": estimated_peak,
            "peak_memory_gate_bytes": peak_memory,
            "maximum_peak_memory_bytes": int(config.maximum_peak_memory_gb * 1024**3),
            "peak_memory_budget_valid": int(
                peak_memory <= int(config.maximum_peak_memory_gb * 1024**3)
            ),
            "storage_bytes_before_manifest": storage_bytes,
            "maximum_storage_bytes": int(config.maximum_storage_gb * 1024**3),
            "storage_budget_valid": int(storage_bytes <= int(config.maximum_storage_gb * 1024**3)),
        }
    )
    validation["gates"]["peak_memory_budget_valid"] = bool(validation["peak_memory_budget_valid"])
    validation["gates"]["storage_budget_valid"] = bool(validation["storage_budget_valid"])
    validation["passed"] = bool(all(validation["gates"].values()))
    validation_path = partial / "validation.json"
    _write_json_durable(validation_path, validation)

    elapsed = time.perf_counter() - started
    manifest = {
        "format_version": CHANNEL_GEOMETRY_FORMAT_VERSION,
        "model_version": CHANNEL_GEOMETRY_MODEL_VERSION,
        "status": "complete" if validation["passed"] else "validation_failed",
        "target_id": sources.target_id,
        "run_fingerprint": run_fingerprint,
        "summary": {
            "cell_count": len(sources.cell_id),
            "display_cell_count": validation["display_cell_count"],
            "channel_reach_count": validation["channel_reach_count"],
            "connector_reach_count": validation["connector_reach_count"],
            "reliable_flow_reach_count": validation["reliable_flow_reach_count"],
            "maximum_centerline_offset_m": validation["maximum_centerline_offset_m"],
        },
        "model": {
            "centerline": ("endpoint-anchored Chaikin smoothing inside the raw D8 path corridor"),
            "river_identity": "unchanged stable L3 hydrology reach graph and endpoints",
            "channel_occupancy": (
                "rivers remain vectors; raster fields are fractional cross-sectional support"
            ),
            "flow_reliability": validation["flow_reliability_semantics"],
            "terrain_change": "none",
            "hydrology_change": "none",
        },
        "resume": {
            "centerline_table_resumed": resumed,
            "elapsed_seconds_this_run": elapsed,
        },
        "source": {
            "terrain_dir": str(config.terrain_dir),
            "hydrology_dir": str(config.hydrology_dir),
            **fingerprint_components,
        },
        "outputs": {
            "channel_geometry_zarr": {
                "path": "channel_geometry.zarr",
                "sha256_tree": _tree_checksum(zarr_path),
            },
            "river_centerlines": {
                "path": "tables/river_centerlines.parquet",
                "rows": centerlines.num_rows,
                "sha256": _file_checksum(centerline_path),
            },
            "channel_geometry_preview": {
                "path": "channel_geometry.png",
                "sha256": _file_checksum(preview_path),
                "projection": "native cubed-sphere face raster diagnostic",
                "legend": "physical centerlines and fractional ecology support",
                "scale": "labelled kilometre scale from L3 area-equivalent cell width",
            },
            "validation": {
                "path": "validation.json",
                "sha256": _file_checksum(validation_path),
            },
        },
        "validation_passed": bool(validation["passed"]),
    }
    manifest_path = partial / "manifest.json"
    _write_json_durable(manifest_path, manifest)
    if not validation["passed"]:
        raise RuntimeError(
            f"L3 channel geometry failed validation; diagnostics retained in {partial}: "
            f"{validation}"
        )
    (partial / "centerline_state.json").unlink(missing_ok=True)
    _replace_directory(partial, config.output_dir)
    return L3ChannelGeometryResult(
        output_dir=config.output_dir,
        manifest_path=config.output_dir / "manifest.json",
        validation_path=config.output_dir / "validation.json",
        zarr_path=config.output_dir / "channel_geometry.zarr",
        preview_path=config.output_dir / "channel_geometry.png",
        target_id=sources.target_id,
        display_cell_count=int(validation["display_cell_count"]),
        channel_reach_count=int(validation["channel_reach_count"]),
        reliable_flow_reach_count=int(validation["reliable_flow_reach_count"]),
        validation_passed=True,
    )


__all__ = [
    "CHANNEL_GEOMETRY_FORMAT_VERSION",
    "CHANNEL_GEOMETRY_MODEL_VERSION",
    "L3ChannelGeometryConfig",
    "L3ChannelGeometryResult",
    "generate_l3_channel_geometry",
]
