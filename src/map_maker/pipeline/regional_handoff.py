"""Chunked L2 regional handoff export over canonical world artifacts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from PIL import Image, ImageDraw
import yaml  # type: ignore[import-untyped]
import zarr  # type: ignore[import-untyped]

from .._native import simulation_native_fingerprints
from .config import PipelineConfig
from .execution import ExecutionEngine
from .models import StageResult

HANDOFF_FORMAT_VERSION = 1
HANDOFF_MODEL_VERSION = "l2_regional_handoff_v1"
PARENT_PRIOR_STAGES = (
    "geometry",
    "planet",
    "tectonics",
    "world_age",
    "geology",
    "elevation",
    "sea_level",
    "climate",
    "cryosphere",
    "hydrology",
    "hydrology_pass2",
    "surface_water",
    "outlet_incision",
    "lake_hydrographs",
    "surface_water_final",
    "surface_materials",
    "biosphere_envelope",
    "potential_biosphere",
    "functional_vegetation",
    "derived_biomes",
)


def _numeric_value(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class RegionalHandoffConfig:
    world_config: Path
    output_dir: Path
    source_config: Path | None = None
    basin_id: int | None = None
    halo_parent_rings: int = 2
    refinement_factor: int = 16
    chunk_rows: int = 65_536

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        output_dir: Path | None = None,
        basin_id: int | None = None,
        halo_parent_rings: int | None = None,
        refinement_factor: int | None = None,
    ) -> "RegionalHandoffConfig":
        path = Path(path).expanduser().resolve()
        data = yaml.safe_load(path.read_text(encoding="utf8"))
        if not isinstance(data, Mapping):
            raise TypeError("Regional handoff config must contain a mapping")
        if int(data.get("format_version", 1)) != HANDOFF_FORMAT_VERSION:
            raise ValueError("Unsupported regional handoff format_version")
        raw_world = data.get("world_config")
        if not raw_world:
            raise ValueError("Regional handoff config requires world_config")
        raw_region = data.get("region", {})
        if not isinstance(raw_region, Mapping):
            raise TypeError("Regional handoff region controls must be a mapping")
        unknown = set(raw_region) - {
            "basin_id",
            "halo_parent_rings",
            "refinement_factor",
            "chunk_rows",
        }
        if unknown:
            raise ValueError(
                f"Unknown regional handoff controls: {', '.join(sorted(map(str, unknown)))}"
            )

        raw_basin = raw_region.get("basin_id", "auto")
        configured_basin = (
            None if raw_basin is None or str(raw_basin).lower() == "auto" else int(raw_basin)
        )
        configured_output = (
            path.parent / str(data.get("output_dir", "../out/regional-handoff"))
        ).resolve()
        config = cls(
            world_config=(path.parent / str(raw_world)).resolve(),
            output_dir=(output_dir.expanduser().resolve() if output_dir else configured_output),
            source_config=path,
            basin_id=configured_basin if basin_id is None else basin_id,
            halo_parent_rings=(
                _numeric_value(
                    "halo_parent_rings",
                    raw_region.get("halo_parent_rings", cls.halo_parent_rings),
                )
                if halo_parent_rings is None
                else halo_parent_rings
            ),
            refinement_factor=(
                _numeric_value(
                    "refinement_factor",
                    raw_region.get("refinement_factor", cls.refinement_factor),
                )
                if refinement_factor is None
                else refinement_factor
            ),
            chunk_rows=_numeric_value("chunk_rows", raw_region.get("chunk_rows", cls.chunk_rows)),
        )
        if config.basin_id is not None and config.basin_id < 0:
            raise ValueError("basin_id must be nonnegative or auto")
        if not 0 <= config.halo_parent_rings <= 8:
            raise ValueError("halo_parent_rings must be in [0, 8]")
        if (
            config.refinement_factor < 2
            or config.refinement_factor > 32
            or config.refinement_factor & (config.refinement_factor - 1)
        ):
            raise ValueError("refinement_factor must be a power of two in [2, 32]")
        if not 256 <= config.chunk_rows <= 1_048_576:
            raise ValueError("chunk_rows must be in [256, 1048576]")
        return config


@dataclass(frozen=True)
class RegionalHandoffResult:
    output_dir: Path
    manifest_path: Path
    validation_path: Path
    preview_path: Path
    zarr_path: Path
    basin_id: int
    parent_count: int
    core_parent_count: int
    child_count: int
    validation_passed: bool


def _artifact_array(result: StageResult, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing handoff input {result.stage_name}.{name}")
    value = record.value
    return cast(np.ndarray, np.asarray(value.array() if hasattr(value, "array") else value))


def _artifact_table(result: StageResult, name: str) -> pa.Table:
    record = result.artifact_records.get(name)
    if record is None or not isinstance(record.value, pa.Table):
        raise KeyError(f"Missing handoff table {result.stage_name}.{name}")
    return record.value.combine_chunks()


def _column_numpy(table: pa.Table, name: str, dtype: np.dtype | None = None) -> np.ndarray:
    column = table[name].combine_chunks()
    if pa.types.is_fixed_size_list(column.type):
        size = column.type.list_size
        values = column.values.to_numpy(zero_copy_only=False).reshape(table.num_rows, size)
    else:
        values = column.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=dtype)


def _grid_axis_start(shape: tuple[int, ...], face_resolution: int) -> int | None:
    target = (6, face_resolution, face_resolution)
    for start in range(max(0, len(shape) - 2)):
        if tuple(shape[start : start + 3]) == target:
            return start
    return None


def _restrict_grid_array(
    values: np.ndarray,
    parent_ids: np.ndarray,
    face_resolution: int,
) -> tuple[np.ndarray, int] | None:
    array = np.asarray(values)
    start = _grid_axis_start(array.shape, face_resolution)
    if start is None or array.dtype.hasobject:
        return None
    moved = np.moveaxis(array, (start, start + 1, start + 2), (0, 1, 2))
    flattened = moved.reshape(6 * face_resolution * face_resolution, *moved.shape[3:])
    return np.ascontiguousarray(flattened[parent_ids]), start


def _zarr_array(group, name: str, values: np.ndarray, chunk_rows: int, **attrs: Any) -> None:
    array = np.ascontiguousarray(values)
    chunks = (min(chunk_rows, max(1, array.shape[0])), *array.shape[1:])
    dataset = group.create_dataset(
        name,
        data=array,
        shape=array.shape,
        dtype=array.dtype,
        chunks=chunks,
        overwrite=True,
    )
    dataset.attrs.update(attrs)


def _halo_rings(
    parent_ids: np.ndarray,
    core_parent_ids: np.ndarray,
    neighbors: np.ndarray,
) -> np.ndarray:
    selected = np.zeros(len(neighbors), dtype=bool)
    selected[parent_ids] = True
    distance = np.full(len(neighbors), -1, dtype=np.int16)
    distance[core_parent_ids] = 0
    ready = deque(int(cell) for cell in core_parent_ids)
    while ready:
        cell = ready.popleft()
        next_distance = int(distance[cell]) + 1
        for neighbor in neighbors[cell]:
            neighbor = int(neighbor)
            if neighbor >= 0 and selected[neighbor] and distance[neighbor] < 0:
                distance[neighbor] = next_distance
                ready.append(neighbor)
    result = distance[parent_ids]
    if np.any(result < 0):
        raise RuntimeError("regional handoff contains a parent disconnected from its core")
    return result


def _realize_surface_fractions(
    parent_targets: Mapping[str, np.ndarray],
    child_parent_row: np.ndarray,
    child_area_km2: np.ndarray,
    child_terrain_m: np.ndarray,
    fine_cell_ids: np.ndarray,
    parent_count: int,
) -> dict[str, np.ndarray]:
    fields = tuple(parent_targets)
    targets = {name: np.asarray(value, dtype=np.float64) for name, value in parent_targets.items()}
    for name, values in targets.items():
        if values.shape != (parent_count,) or np.any(~np.isfinite(values)):
            raise RuntimeError(f"invalid parent surface target {name}")
        if np.any((values < -1e-7) | (values > 1.0 + 1e-7)):
            raise RuntimeError(f"parent surface target {name} is outside [0, 1]")
    total_target = np.sum(np.stack(list(targets.values()), axis=0), axis=0)
    if np.any(total_target > 1.0 + 1e-6):
        raise RuntimeError("parent ocean/lake/wetland targets exceed unit surface area")

    outputs = {name: np.zeros(len(child_parent_row), dtype=np.float32) for name in fields}
    order = np.argsort(child_parent_row, kind="stable")
    counts = np.bincount(child_parent_row, minlength=parent_count)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for parent_row in range(parent_count):
        rows = order[offsets[parent_row] : offsets[parent_row + 1]]
        ranked = rows[np.lexsort((fine_cell_ids[rows], child_terrain_m[rows]))]
        remaining_capacity = np.ones(len(ranked), dtype=np.float64)
        parent_area = float(np.sum(child_area_km2[ranked]))
        for name in fields:
            remaining_area = float(targets[name][parent_row]) * parent_area
            for position, child_row in enumerate(ranked):
                if remaining_area <= 1e-10:
                    break
                capacity_area = remaining_capacity[position] * child_area_km2[child_row]
                take_area = min(remaining_area, float(capacity_area))
                fraction = take_area / child_area_km2[child_row]
                outputs[name][child_row] = np.float32(fraction)
                remaining_capacity[position] -= fraction
                remaining_area -= take_area
            if remaining_area > max(1e-8, parent_area * 1e-8):
                raise RuntimeError(f"insufficient child capacity for parent surface {name}")
    return outputs


def _parent_weighted_fraction(
    values: np.ndarray,
    child_parent_row: np.ndarray,
    child_area_km2: np.ndarray,
    parent_area_km2: np.ndarray,
) -> np.ndarray:
    represented = np.bincount(
        child_parent_row,
        weights=np.asarray(values, dtype=np.float64) * child_area_km2,
        minlength=len(parent_area_km2),
    )
    return represented / parent_area_km2


def _parent_weighted_mean(
    values: np.ndarray,
    child_parent_row: np.ndarray,
    child_area_km2: np.ndarray,
    parent_area_km2: np.ndarray,
) -> np.ndarray:
    represented = np.bincount(
        child_parent_row,
        weights=np.asarray(values, dtype=np.float64) * child_area_km2,
        minlength=len(parent_area_km2),
    )
    return represented / parent_area_km2


def _filter_isin(table: pa.Table, column: str, values: np.ndarray) -> pa.Table:
    if table.num_rows == 0 or values.size == 0:
        return table.slice(0, 0)
    return table.filter(
        pc.is_in(table[column], value_set=pa.array(np.unique(values), type=table[column].type))
    )


def _write_parquet(path: Path, table: pa.Table) -> None:
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_checksum(child)))
    return digest.hexdigest()


def _relative_longitude(longitude: np.ndarray, center: float) -> np.ndarray:
    return (longitude - center + 180.0) % 360.0 - 180.0


def _render_preview(
    path: Path,
    xyz: np.ndarray,
    surface_elevation_m: np.ndarray,
    ocean_fraction: np.ndarray,
    lake_fraction: np.ndarray,
    wetland_fraction: np.ndarray,
    core_mask: np.ndarray,
    reaches: pa.Table,
    fine_cell_ids: np.ndarray,
) -> None:
    longitude = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    latitude = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))
    center = math.degrees(
        math.atan2(
            float(np.mean(np.sin(np.radians(longitude)))),
            float(np.mean(np.cos(np.radians(longitude)))),
        )
    )
    relative = _relative_longitude(longitude, center)
    latitude_scale = max(math.cos(math.radians(float(np.mean(latitude)))), 0.2)
    map_x = relative * latitude_scale
    padding_x = max(float(np.ptp(map_x)) * 0.04, 0.25)
    padding_y = max(float(np.ptp(latitude)) * 0.04, 0.25)
    left, right = float(np.min(map_x) - padding_x), float(np.max(map_x) + padding_x)
    bottom, top = float(np.min(latitude) - padding_y), float(np.max(latitude) + padding_y)
    width = 1_600
    height = int(np.clip(round(width * (top - bottom) / max(right - left, 1e-6)), 500, 1_100))
    x = np.clip(((map_x - left) / (right - left) * (width - 1)).round(), 0, width - 1).astype(
        np.int32
    )
    y = np.clip(((top - latitude) / (top - bottom) * (height - 1)).round(), 0, height - 1).astype(
        np.int32
    )

    colors = np.empty((len(x), 3), dtype=np.uint8)
    positive = surface_elevation_m[core_mask & (ocean_fraction < 0.5)]
    low, high = np.percentile(positive, [5.0, 95.0]) if positive.size else np.asarray((0.0, 1.0))
    normalized = np.clip((surface_elevation_m - low) / max(float(high - low), 1.0), 0.0, 1.0)
    low_color = np.asarray((83.0, 143.0, 91.0))
    high_color = np.asarray((187.0, 166.0, 119.0))
    colors[:] = np.clip(
        low_color * (1.0 - normalized[:, None]) + high_color * normalized[:, None], 0, 255
    ).astype(np.uint8)
    ocean = ocean_fraction >= 0.5
    colors[ocean] = (31, 88, 128)
    lake_weight = np.clip(lake_fraction, 0.0, 1.0)[:, None]
    colors = np.clip(
        colors.astype(np.float64) * (1.0 - lake_weight)
        + np.asarray((52.0, 135.0, 166.0)) * lake_weight,
        0,
        255,
    ).astype(np.uint8)
    wet_weight = np.clip(wetland_fraction * 0.7, 0.0, 0.7)[:, None]
    colors = np.clip(
        colors.astype(np.float64) * (1.0 - wet_weight)
        + np.asarray((55.0, 121.0, 93.0)) * wet_weight,
        0,
        255,
    ).astype(np.uint8)
    colors[~core_mask] = np.clip(
        colors[~core_mask].astype(np.float64) * 0.55 + np.asarray((215.0, 216.0, 208.0)) * 0.45,
        0,
        255,
    ).astype(np.uint8)

    canvas = np.full((height, width, 3), (239, 240, 235), dtype=np.uint8)
    for delta_y, delta_x in ((0, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, 0)):
        target_x = np.clip(x + delta_x, 0, width - 1)
        target_y = np.clip(y + delta_y, 0, height - 1)
        canvas[target_y, target_x] = colors
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    fine_order = np.argsort(fine_cell_ids)
    sorted_fine_ids = fine_cell_ids[fine_order]
    for row in reaches.to_pylist():
        cell_path = np.asarray(row["fine_cell_path"], dtype=np.int32)
        positions = np.searchsorted(sorted_fine_ids, cell_path)
        if np.any(positions >= len(sorted_fine_ids)) or np.any(
            sorted_fine_ids[positions] != cell_path
        ):
            continue
        child_rows = fine_order[positions]
        points = [(int(x[index]), int(y[index])) for index in child_rows]
        if len(points) >= 2:
            draw.line(
                points,
                fill=(17, 88, 145) if row["reach_kind"] == "channel" else (74, 116, 132),
                width=3 if row["reach_kind"] == "channel" else 2,
                joint="curve",
            )
    image.save(path, optimize=True)


def _source_provenance(
    results: Mapping[str, StageResult],
    packaged_fields: list[dict[str, Any]],
    direct_sources: set[tuple[str, str]],
) -> dict[str, dict[str, str]]:
    provenance: dict[str, dict[str, str]] = {}
    for field in packaged_fields:
        stage = str(field["stage"])
        artifact = str(field["artifact"])
        provenance.setdefault(stage, {})[artifact] = (
            results[stage].artifact_records[artifact].checksum
        )
    for stage, artifact in direct_sources:
        provenance.setdefault(stage, {})[artifact] = (
            results[stage].artifact_records[artifact].checksum
        )
    return {stage: dict(sorted(values.items())) for stage, values in sorted(provenance.items())}


def _replace_directory(staging: Path, output: Path) -> None:
    previous = output.with_name(f".{output.name}.previous")
    if previous.exists():
        shutil.rmtree(previous)
    if output.exists():
        output.rename(previous)
    try:
        staging.rename(output)
    except BaseException:
        if previous.exists() and not output.exists():
            previous.rename(output)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def export_regional_handoff(config: RegionalHandoffConfig) -> RegionalHandoffResult:
    """Build and validate one immutable selected-basin L2 handoff package."""

    from .stages import ensure_builtin_stages

    ensure_builtin_stages()
    world = PipelineConfig.from_file(config.world_config)
    if world.topology.lower() != "cubed_sphere":
        raise ValueError("regional handoff requires topology: cubed_sphere")
    parent_resolution = world.resolution_set.native.face_resolution
    if parent_resolution is None:
        raise ValueError("regional handoff requires a cubed-sphere face resolution")

    overrides = {name: dict(values) for name, values in world.stage_overrides.items()}
    refinement = dict(overrides.get("basin_refinement", {}))
    refinement.update(
        {
            "basin_id": config.basin_id,
            "halo_parent_rings": config.halo_parent_rings,
            "refinement_factor": config.refinement_factor,
        }
    )
    overrides["basin_refinement"] = refinement
    world.stage_overrides = overrides

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(
        tempfile.mkdtemp(prefix=f".{config.output_dir.name}-run-", dir=config.output_dir.parent)
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{config.output_dir.name}-package-", dir=config.output_dir.parent)
    )
    published = False
    try:
        world.output_dir = scratch / "out"
        world.log_dir = scratch / "logs"
        world.run_id = "l2-regional-handoff-build"
        engine = ExecutionEngine(world, generate_visuals=False)
        results = engine.run(["derived_biomes", "basin_refinement"])
        refinement_result = results["basin_refinement"]
        refinement_metadata = cast(
            dict[str, Any], refinement_result.artifact_records["BasinRefinementMetadata"].value
        )
        selected_basin_id = int(refinement_metadata["selected_basin_id"])
        parents = _artifact_table(refinement_result, "RefinedBasinParentCatalog")
        cells = _artifact_table(refinement_result, "RefinedBasinCellCatalog")
        refined_reaches = _artifact_table(refinement_result, "RefinedRiverReachCatalog")
        refined_reach_cells = _artifact_table(refinement_result, "RefinedReachCellCatalog")

        parent_ids = _column_numpy(parents, "parent_cell_id", np.dtype(np.int32))
        inside_core = _column_numpy(parents, "inside_selected_basin", np.dtype(bool))
        core_parent_ids = parent_ids[inside_core]
        parent_paths = refined_reaches["parent_cell_path"].to_pylist()
        reach_parent_ids = np.unique(
            np.fromiter(
                (int(cell) for path_values in parent_paths for cell in path_values),
                dtype=np.int32,
                count=sum(map(len, parent_paths)),
            )
        )
        inside_reach_path_support = np.isin(parent_ids, reach_parent_ids) & ~inside_core
        halo_seed_parent_ids = np.union1d(core_parent_ids, reach_parent_ids).astype(
            np.int32, copy=False
        )
        neighbors = np.asarray(engine.context.topology.neighbor_indices, dtype=np.int32).reshape(
            -1, 4
        )
        halo_ring = _halo_rings(parent_ids, halo_seed_parent_ids, neighbors)
        parent_area_km2 = _column_numpy(parents, "restricted_child_area_km2", np.dtype(np.float64))

        fine_cell_ids = _column_numpy(cells, "fine_cell_id", np.dtype(np.int32))
        child_parent_ids = _column_numpy(cells, "parent_cell_id", np.dtype(np.int32))
        child_parent_row = np.searchsorted(parent_ids, child_parent_ids).astype(np.int32)
        if np.any(child_parent_row >= len(parent_ids)) or np.any(
            parent_ids[child_parent_row] != child_parent_ids
        ):
            raise RuntimeError("refined children do not map to packaged parents")
        child_area_km2 = _column_numpy(cells, "area_km2", np.dtype(np.float64))
        child_terrain_m = _column_numpy(cells, "terrain_elevation_m", np.dtype(np.float32))
        child_xyz = _column_numpy(cells, "xyz", np.dtype(np.float32))
        child_core = inside_core[child_parent_row]

        ocean_target = _artifact_array(results["sea_level"], "SurfaceOceanFraction").reshape(-1)[
            parent_ids
        ]
        lake_target = _artifact_array(
            results["surface_materials"], "EffectiveLakeFraction"
        ).reshape(-1)[parent_ids]
        wetland_target = _artifact_array(
            results["surface_materials"], "EffectiveWetlandFraction"
        ).reshape(-1)[parent_ids]
        realized = _realize_surface_fractions(
            {
                "ocean_fraction": ocean_target,
                "lake_fraction": lake_target,
                "wetland_fraction": wetland_target,
            },
            child_parent_row,
            child_area_km2,
            child_terrain_m,
            fine_cell_ids,
            len(parent_ids),
        )
        parent_surface = _artifact_array(results["sea_level"], "SurfaceElevationM").reshape(-1)[
            parent_ids
        ]
        parent_refined_elevation = _column_numpy(
            parents, "parent_elevation_m", np.dtype(np.float32)
        )
        child_surface_m = (
            child_terrain_m + (parent_surface - parent_refined_elevation)[child_parent_row]
        )
        total_water = (
            realized["ocean_fraction"] + realized["lake_fraction"] + realized["wetland_fraction"]
        )

        zarr_path = staging / "region.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")
        root.attrs.update(
            {
                "format_version": HANDOFF_FORMAT_VERSION,
                "model_version": HANDOFF_MODEL_VERSION,
                "basin_id": selected_basin_id,
                "parent_level": "L0",
                "child_level": "L2",
                "parent_face_resolution": parent_resolution,
                "child_face_resolution": parent_resolution * config.refinement_factor,
                "refinement_factor": config.refinement_factor,
            }
        )
        parent_group = root.require_group("parent")
        _zarr_array(parent_group, "cell_id", parent_ids, config.chunk_rows)
        _zarr_array(parent_group, "inside_core", inside_core, config.chunk_rows)
        _zarr_array(
            parent_group,
            "inside_reach_path_support",
            inside_reach_path_support,
            config.chunk_rows,
        )
        _zarr_array(parent_group, "halo_ring", halo_ring, config.chunk_rows)
        _zarr_array(parent_group, "area_km2", parent_area_km2, config.chunk_rows, units="km2")

        geometry_group = root.require_group("l2/geometry")
        geometry_fields = {
            "cell_id": fine_cell_ids,
            "parent_cell_id": child_parent_ids,
            "parent_row": child_parent_row,
            "inside_core": child_core,
            "inside_reach_path_support": inside_reach_path_support[child_parent_row],
            "halo_ring": halo_ring[child_parent_row],
            "face": _column_numpy(cells, "face", np.dtype(np.int32)),
            "row": _column_numpy(cells, "row", np.dtype(np.int32)),
            "column": _column_numpy(cells, "col", np.dtype(np.int32)),
            "xyz": child_xyz,
            "area_km2": child_area_km2,
            "terrain_elevation_m": child_terrain_m,
            "terrain_offset_m": _column_numpy(cells, "terrain_offset_m", np.dtype(np.float32)),
            "parent_relief_m": _column_numpy(cells, "parent_relief_m", np.dtype(np.float32)),
            "channel_surface_prior_m": _column_numpy(
                cells, "channel_surface_prior_m", np.dtype(np.float32)
            ),
            "hydraulic_surface_controlled": _column_numpy(
                cells, "hydraulic_surface_controlled", np.dtype(bool)
            ),
            "process_excluded": _column_numpy(cells, "process_excluded", np.dtype(bool)),
        }
        for name, values in geometry_fields.items():
            units = "km2" if name == "area_km2" else "m" if name.endswith("_m") else None
            attrs = {"units": units} if units else {}
            _zarr_array(geometry_group, name, values, config.chunk_rows, **attrs)

        surface_group = root.require_group("l2/surface")
        surface_fields = {
            **realized,
            "land_fraction": 1.0 - realized["ocean_fraction"],
            "unoccupied_land_fraction": 1.0 - total_water,
            "surface_elevation_m": child_surface_m.astype(np.float32),
            "ocean_depth_m": (
                np.maximum(-child_surface_m, 0.0) * realized["ocean_fraction"]
            ).astype(np.float32),
        }
        for name, values in surface_fields.items():
            _zarr_array(
                surface_group,
                name,
                np.asarray(values),
                config.chunk_rows,
                semantics="conditional_child_realization",
            )

        priors_group = root.require_group("parent_priors")
        packaged_fields: list[dict[str, Any]] = []
        for stage_name in PARENT_PRIOR_STAGES:
            stage_result = results.get(stage_name)
            if stage_result is None:
                continue
            stage_group = priors_group.require_group(stage_name)
            for artifact_name, record in sorted(stage_result.artifact_records.items()):
                if record.value is None or isinstance(record.value, (dict, list, pa.Table)):
                    continue
                value = record.value
                array = np.asarray(value.array() if hasattr(value, "array") else value)
                restricted = _restrict_grid_array(array, parent_ids, parent_resolution)
                if restricted is None:
                    continue
                restricted_values, grid_axis_start = restricted
                _zarr_array(
                    stage_group,
                    artifact_name,
                    restricted_values,
                    config.chunk_rows,
                    source_checksum=record.checksum,
                    source_shape=list(array.shape),
                    grid_axis_start=grid_axis_start,
                    semantics="inherited_parent_prior",
                )
                packaged_fields.append(
                    {
                        "stage": stage_name,
                        "artifact": artifact_name,
                        "path": f"parent_priors/{stage_name}/{artifact_name}",
                        "shape": list(restricted_values.shape),
                        "dtype": str(restricted_values.dtype),
                        "source_checksum": record.checksum,
                    }
                )
        zarr.consolidate_metadata(str(zarr_path))

        tables_dir = staging / "tables"
        tables_dir.mkdir()
        hydrology = results["hydrology"]
        drainage_graph = _filter_isin(
            _artifact_table(hydrology, "DrainageGraph"), "cell_id", parent_ids
        )
        represented_basins = _column_numpy(drainage_graph, "basin_id", np.dtype(np.int32))
        basin_catalog = _filter_isin(
            _artifact_table(hydrology, "BasinCatalog"), "basin_id", represented_basins
        )
        waterbody_cells = _filter_isin(
            _artifact_table(hydrology, "WaterBodyCellCatalog"), "cell_id", parent_ids
        )
        depression_ids = (
            _column_numpy(waterbody_cells, "depression_id", np.dtype(np.int32))
            if waterbody_cells.num_rows
            else np.empty(0, dtype=np.int32)
        )
        river_reaches = _artifact_table(hydrology, "RiverReachCatalog")
        source_reaches = river_reaches.filter(
            pc.equal(
                river_reaches["basin_id"],
                pa.scalar(selected_basin_id, type=pa.int32()),
            )
        )
        table_values: dict[str, pa.Table] = {
            "parent_cells": parents,
            "basins": basin_catalog,
            "drainage_graph": drainage_graph,
            "waterbody_cells": waterbody_cells,
            "depression_catalog": _filter_isin(
                _artifact_table(hydrology, "DepressionCatalog"),
                "depression_id",
                depression_ids,
            ),
            "lake_catalog": _filter_isin(
                _artifact_table(hydrology, "LakeCatalog"), "depression_id", depression_ids
            ),
            "wetland_catalog": _filter_isin(
                _artifact_table(hydrology, "WetlandCatalog"), "depression_id", depression_ids
            ),
            "river_reaches": source_reaches,
            "refined_river_reaches": refined_reaches,
            "refined_reach_cells": refined_reach_cells,
        }
        table_sources = {
            "basins": ("hydrology", "BasinCatalog"),
            "drainage_graph": ("hydrology", "DrainageGraph"),
            "waterbody_cells": ("hydrology", "WaterBodyCellCatalog"),
            "depression_catalog": ("hydrology", "DepressionCatalog"),
            "lake_catalog": ("hydrology", "LakeCatalog"),
            "wetland_catalog": ("hydrology", "WetlandCatalog"),
            "river_reaches": ("hydrology", "RiverReachCatalog"),
            "parent_cells": ("basin_refinement", "RefinedBasinParentCatalog"),
            "refined_river_reaches": (
                "basin_refinement",
                "RefinedRiverReachCatalog",
            ),
            "refined_reach_cells": ("basin_refinement", "RefinedReachCellCatalog"),
        }
        for name, table in table_values.items():
            _write_parquet(tables_dir / f"{name}.parquet", table)

        preview_path = staging / "preview.png"
        _render_preview(
            preview_path,
            child_xyz,
            child_surface_m,
            realized["ocean_fraction"],
            realized["lake_fraction"],
            realized["wetland_fraction"],
            child_core,
            refined_reaches,
            fine_cell_ids,
        )

        represented_surface = {
            name: _parent_weighted_fraction(
                values,
                child_parent_row,
                child_area_km2,
                parent_area_km2,
            )
            for name, values in realized.items()
        }
        surface_errors = {
            "ocean_fraction": float(
                np.max(np.abs(represented_surface["ocean_fraction"] - ocean_target))
            ),
            "lake_fraction": float(
                np.max(np.abs(represented_surface["lake_fraction"] - lake_target))
            ),
            "wetland_fraction": float(
                np.max(np.abs(represented_surface["wetland_fraction"] - wetland_target))
            ),
        }
        represented_surface_elevation = _parent_weighted_mean(
            child_surface_m,
            child_parent_row,
            child_area_km2,
            parent_area_km2,
        )
        maximum_surface_elevation_error_m = float(
            np.max(np.abs(represented_surface_elevation - parent_surface))
        )
        fine_paths = refined_reaches["fine_cell_path"].to_pylist()
        referenced_fine_ids = np.fromiter(
            (int(cell) for path_values in fine_paths for cell in path_values),
            dtype=np.int32,
            count=sum(map(len, fine_paths)),
        )
        path_cells_packaged = bool(np.all(np.isin(referenced_fine_ids, fine_cell_ids)))
        validation = {
            "format_version": HANDOFF_FORMAT_VERSION,
            "model_version": HANDOFF_MODEL_VERSION,
            "basin_id": selected_basin_id,
            "parent_count": len(parent_ids),
            "core_parent_count": int(np.count_nonzero(inside_core)),
            "reach_path_support_parent_count": int(np.count_nonzero(inside_reach_path_support)),
            "halo_parent_count": int(np.count_nonzero(halo_ring > 0)),
            "context_parent_count": int(np.count_nonzero(~inside_core)),
            "child_count": len(fine_cell_ids),
            "parent_prior_field_count": len(packaged_fields),
            "maximum_parent_area_relative_error": float(
                refinement_metadata["maximum_parent_area_relative_error"]
            ),
            "maximum_parent_terrain_mean_error_m": float(
                refinement_metadata["maximum_parent_elevation_error_m"]
            ),
            "maximum_parent_surface_elevation_mean_error_m": (maximum_surface_elevation_error_m),
            "maximum_surface_fraction_error": max(surface_errors.values()),
            "surface_fraction_errors": surface_errors,
            "maximum_child_surface_occupancy": float(np.max(total_water, initial=0.0)),
            "minimum_child_surface_occupancy": float(np.min(total_water)),
            "unique_child_ids": int(len(np.unique(fine_cell_ids)) == len(fine_cell_ids)),
            "all_child_parents_packaged": int(
                np.all(np.isin(np.unique(child_parent_ids), parent_ids))
            ),
            "all_refined_path_cells_packaged": int(path_cells_packaged),
            "refined_path_graph_valid": int(refinement_metadata["directed_path_graph_valid"]),
        }
        validation["passed"] = bool(
            validation["maximum_parent_area_relative_error"] <= 1e-9
            and validation["maximum_parent_terrain_mean_error_m"] <= 1e-3
            and validation["maximum_parent_surface_elevation_mean_error_m"] <= 1e-3
            and validation["maximum_surface_fraction_error"] <= 1e-6
            and validation["maximum_child_surface_occupancy"] <= 1.0 + 1e-6
            and validation["minimum_child_surface_occupancy"] >= -1e-7
            and validation["unique_child_ids"] == 1
            and validation["all_child_parents_packaged"] == 1
            and validation["all_refined_path_cells_packaged"] == 1
            and validation["refined_path_graph_valid"] == 1
        )
        if not validation["passed"]:
            raise RuntimeError("regional handoff failed its conservation or topology contract")
        validation_path = staging / "validation.json"
        validation_path.write_text(
            json.dumps(validation, indent=2, sort_keys=True), encoding="utf8"
        )

        table_checksums = {
            path.name: _file_checksum(path) for path in sorted(tables_dir.glob("*.parquet"))
        }
        direct_sources = set(table_sources.values())
        direct_sources.update(
            {
                ("basin_refinement", "RefinedBasinCellCatalog"),
                ("basin_refinement", "BasinRefinementMetadata"),
                ("sea_level", "SurfaceElevationM"),
                ("sea_level", "SurfaceOceanFraction"),
                ("surface_materials", "EffectiveLakeFraction"),
                ("surface_materials", "EffectiveWetlandFraction"),
            }
        )
        source_artifacts = _source_provenance(results, packaged_fields, direct_sources)
        child_mean_area = float(np.mean(child_area_km2))
        manifest = {
            "format_version": HANDOFF_FORMAT_VERSION,
            "model_version": HANDOFF_MODEL_VERSION,
            "status": "complete",
            "region_id": f"basin-{selected_basin_id}-l2-f{config.refinement_factor}",
            "selection": {
                "kind": "complete_drainage_basin_with_parent_halo",
                "basin_id": selected_basin_id,
                "halo_parent_rings": config.halo_parent_rings,
                "parent_count": len(parent_ids),
                "core_parent_count": int(np.count_nonzero(inside_core)),
                "reach_path_support_parent_count": int(np.count_nonzero(inside_reach_path_support)),
                "halo_parent_count": int(np.count_nonzero(halo_ring > 0)),
                "context_parent_count": int(np.count_nonzero(~inside_core)),
                "maximum_halo_ring": int(np.max(halo_ring, initial=0)),
            },
            "resolution": {
                "parent_level": "L0",
                "parent_face_resolution": parent_resolution,
                "parent_mean_cell_area_km2": float(np.mean(parent_area_km2)),
                "child_level": "L2",
                "child_face_resolution_equivalent": parent_resolution * config.refinement_factor,
                "refinement_factor": config.refinement_factor,
                "child_mean_cell_area_km2": child_mean_area,
                "child_mean_cell_width_km": math.sqrt(child_mean_area),
                "sparse_child_count": len(fine_cell_ids),
            },
            "storage": {
                "arrays": "Zarr v2 row-chunked",
                "tables": "Parquet with Zstandard compression",
                "chunk_rows": config.chunk_rows,
            },
            "outputs": {
                "zarr": {"path": "region.zarr", "sha256": _tree_checksum(zarr_path)},
                "tables": table_checksums,
                "preview": {"path": "preview.png", "sha256": _file_checksum(preview_path)},
                "validation": {
                    "path": "validation.json",
                    "sha256": _file_checksum(validation_path),
                },
            },
            "source": {
                "handoff_config": (
                    str(config.source_config) if config.source_config is not None else None
                ),
                "handoff_config_sha256": (
                    _file_checksum(config.source_config)
                    if config.source_config is not None
                    else None
                ),
                "world_config": str(config.world_config),
                "world_config_sha256": _file_checksum(config.world_config),
                "world_run_id": PipelineConfig.from_file(config.world_config).run_id,
                "world_seed": world.rng_seed,
                "topology": world.topology,
                "artifacts": source_artifacts,
            },
            "software": {
                "package": "map-maker",
                "version": package_version("map-maker"),
                "native_libraries": simulation_native_fingerprints(),
            },
            "parent_prior_fields": packaged_fields,
            "tables": {
                name: {"path": f"tables/{name}.parquet", "rows": table.num_rows}
                for name, table in table_values.items()
            },
            "semantics": {
                "terrain": "deterministic parent-mean-conserving L2 conditional realization",
                "surface": "terrain-ranked parent-area-conserving conditional occupancy",
                "parent_priors": "inherited values; not recomputed or claimed as L2 downscaling",
                "rivers": "inherited graph/vector identities with no applied L2 incision",
                "resources": "absent until an upstream mineral and energy system exists",
            },
            "validation_passed": True,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf8")

        _replace_directory(staging, config.output_dir)
        published = True
        return RegionalHandoffResult(
            output_dir=config.output_dir,
            manifest_path=config.output_dir / "manifest.json",
            validation_path=config.output_dir / "validation.json",
            preview_path=config.output_dir / "preview.png",
            zarr_path=config.output_dir / "region.zarr",
            basin_id=selected_basin_id,
            parent_count=len(parent_ids),
            core_parent_count=int(np.count_nonzero(inside_core)),
            child_count=len(fine_cell_ids),
            validation_passed=True,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "HANDOFF_FORMAT_VERSION",
    "HANDOFF_MODEL_VERSION",
    "RegionalHandoffConfig",
    "RegionalHandoffResult",
    "export_regional_handoff",
]
