"""Select and package a bounded L3 vertical-slice target from an L2 handoff."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from PIL import Image, ImageDraw
import yaml  # type: ignore[import-untyped]
import zarr  # type: ignore[import-untyped]

from .regional_handoff import (
    _file_checksum,
    _relative_longitude,
    _render_preview,
    _replace_directory,
)

TARGET_FORMAT_VERSION = 2
TARGET_MODEL_VERSION = "l3_vertical_slice_target_v2"


@dataclass(frozen=True)
class L3TargetConfig:
    handoff_dir: Path
    output_dir: Path
    target_id: str
    outlet_parent_cell_id: int
    context_parent_rings: int = 2
    minimum_area_km2: float = 80_000.0
    maximum_area_km2: float = 120_000.0
    base_cell_size_m: float = 200.0
    adaptive_minimum_cell_size_m: float = 25.0
    adaptive_maximum_cell_size_m: float = 50.0
    terrain_window_halo_l2_cells: int = 4
    terrain_source_context_l2_cells: int = 1
    maximum_base_cell_count: int = 3_000_000
    maximum_peak_memory_gb: float = 24.0
    source_config: Path | None = None

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        output_dir: Path | None = None,
    ) -> "L3TargetConfig":
        source = Path(path).expanduser().resolve()
        data = yaml.safe_load(source.read_text(encoding="utf8"))
        if not isinstance(data, Mapping):
            raise TypeError("L3 target config must contain a mapping")
        if int(data.get("format_version", 1)) != TARGET_FORMAT_VERSION:
            raise ValueError("unsupported L3 target format_version")
        target = data.get("target", {})
        grid = data.get("grid", {})
        limits = data.get("limits", {})
        if not isinstance(target, Mapping) or not isinstance(grid, Mapping):
            raise TypeError("L3 target and grid controls must be mappings")
        if not isinstance(limits, Mapping):
            raise TypeError("L3 target limits must be a mapping")
        raw_handoff = data.get("handoff_dir")
        raw_output = data.get("output_dir")
        if not raw_handoff or not raw_output:
            raise ValueError("L3 target config requires handoff_dir and output_dir")
        config = cls(
            handoff_dir=(source.parent / str(raw_handoff)).resolve(),
            output_dir=(
                output_dir.expanduser().resolve()
                if output_dir is not None
                else (source.parent / str(raw_output)).resolve()
            ),
            target_id=str(target.get("id", "")).strip(),
            outlet_parent_cell_id=int(target.get("outlet_parent_cell_id", -1)),
            context_parent_rings=int(target.get("context_parent_rings", 2)),
            minimum_area_km2=float(target.get("minimum_area_km2", 80_000.0)),
            maximum_area_km2=float(target.get("maximum_area_km2", 120_000.0)),
            base_cell_size_m=float(grid.get("base_cell_size_m", 200.0)),
            adaptive_minimum_cell_size_m=float(grid.get("adaptive_minimum_cell_size_m", 25.0)),
            adaptive_maximum_cell_size_m=float(grid.get("adaptive_maximum_cell_size_m", 50.0)),
            terrain_window_halo_l2_cells=int(grid.get("terrain_window_halo_l2_cells", 4)),
            terrain_source_context_l2_cells=int(grid.get("terrain_source_context_l2_cells", 1)),
            maximum_base_cell_count=int(limits.get("maximum_base_cell_count", 3_000_000)),
            maximum_peak_memory_gb=float(limits.get("maximum_peak_memory_gb", 24.0)),
            source_config=source,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.target_id or any(character.isspace() for character in self.target_id):
            raise ValueError("target.id must be a nonempty identifier without whitespace")
        if self.outlet_parent_cell_id < 0:
            raise ValueError("target.outlet_parent_cell_id must be nonnegative")
        if not 0 <= self.context_parent_rings <= 4:
            raise ValueError("target.context_parent_rings must be in [0, 4]")
        if not 0.0 < self.minimum_area_km2 <= self.maximum_area_km2:
            raise ValueError("target area limits must be positive and ordered")
        if not 100.0 <= self.base_cell_size_m <= 250.0:
            raise ValueError("grid.base_cell_size_m must be in [100, 250]")
        if not (
            10.0
            <= self.adaptive_minimum_cell_size_m
            <= self.adaptive_maximum_cell_size_m
            < self.base_cell_size_m
        ):
            raise ValueError("adaptive grid sizes must be ordered below the base cell size")
        if not 1 <= self.terrain_window_halo_l2_cells <= 16:
            raise ValueError("grid.terrain_window_halo_l2_cells must be in [1, 16]")
        if not 1 <= self.terrain_source_context_l2_cells <= 4:
            raise ValueError("grid.terrain_source_context_l2_cells must be in [1, 4]")
        if self.maximum_base_cell_count <= 0:
            raise ValueError("limits.maximum_base_cell_count must be positive")
        if not 1.0 <= self.maximum_peak_memory_gb <= 28.0:
            raise ValueError("limits.maximum_peak_memory_gb must be in [1, 28]")


@dataclass(frozen=True)
class L3TargetResult:
    output_dir: Path
    manifest_path: Path
    validation_path: Path
    preview_path: Path
    target_id: str
    outlet_parent_cell_id: int
    core_parent_count: int
    context_parent_count: int
    core_area_km2: float
    estimated_base_cell_count: int


def _column_numpy(table: pa.Table, name: str, dtype: np.dtype) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype)


def _upstream_closure(
    outlet: int,
    cell_ids: np.ndarray,
    receiver_ids: np.ndarray,
) -> np.ndarray:
    upstream: dict[int, list[int]] = {int(cell): [] for cell in cell_ids}
    for cell, receiver in zip(cell_ids, receiver_ids, strict=True):
        if int(receiver) in upstream:
            upstream[int(receiver)].append(int(cell))
    selected = {int(outlet)}
    ready = [int(outlet)]
    while ready:
        cell = ready.pop()
        for source in upstream.get(cell, ()):
            if source not in selected:
                selected.add(source)
                ready.append(source)
    return np.asarray(sorted(selected), dtype=np.int32)


def _context_selection(
    core_ids: np.ndarray,
    parent_ids: np.ndarray,
    neighbors: np.ndarray,
    rings: int,
) -> tuple[np.ndarray, np.ndarray, set[int]]:
    available = set(int(value) for value in parent_ids)
    row_by_id = {int(value): row for row, value in enumerate(parent_ids)}
    distance = {int(value): 0 for value in core_ids}
    frontier = set(distance)
    missing: set[int] = set()
    for ring in range(1, rings + 1):
        next_frontier: set[int] = set()
        for cell in frontier:
            for neighbor in neighbors[row_by_id[cell]]:
                neighbor = int(neighbor)
                if neighbor < 0 or neighbor in distance:
                    continue
                if neighbor not in available:
                    missing.add(neighbor)
                    continue
                distance[neighbor] = ring
                next_frontier.add(neighbor)
        frontier = next_frontier
    selected = np.asarray(sorted(distance), dtype=np.int32)
    ring_by_id = np.asarray([distance[int(value)] for value in selected], dtype=np.int16)
    return selected, ring_by_id, missing


def _d8_context_rings(core: np.ndarray, maximum_ring: int) -> np.ndarray:
    """Label an eight-neighbor dilation without wrapping across array edges."""

    core = np.asarray(core, dtype=bool)
    rings = np.full(core.shape, -1, dtype=np.int16)
    rings[core] = 0
    reached = core.copy()
    for ring in range(1, maximum_ring + 1):
        padded = np.pad(reached, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(reached)
        for row_offset in range(3):
            for column_offset in range(3):
                expanded |= padded[
                    row_offset : row_offset + reached.shape[0],
                    column_offset : column_offset + reached.shape[1],
                ]
        newly_reached = expanded & ~reached
        rings[newly_reached] = ring
        reached |= newly_reached
    return rings


def _rectangular_l2_window(
    cell_ids: np.ndarray,
    core_mask: np.ndarray,
    resolution: int,
    halo_cells: int,
    source_context_cells: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    """Select a complete terrain rectangle and classify its process roles."""

    ids = np.asarray(cell_ids, dtype=np.int64)
    core = np.asarray(core_mask, dtype=bool)
    if ids.ndim != 1 or core.shape != ids.shape or not np.any(core):
        raise ValueError("L3 terrain window requires a nonempty one-dimensional core")
    face_size = resolution * resolution
    face = ids // face_size
    within_face = ids % face_size
    row = within_face // resolution
    column = within_face % resolution
    core_faces = np.unique(face[core])
    if len(core_faces) != 1:
        raise NotImplementedError("L3 V0 terrain window must fit within one cubed-sphere face")
    core_face = int(core_faces[0])
    terrain_min_row = int(np.min(row[core])) - halo_cells
    terrain_max_row = int(np.max(row[core])) + halo_cells
    terrain_min_column = int(np.min(column[core])) - halo_cells
    terrain_max_column = int(np.max(column[core])) + halo_cells
    source_min_row = terrain_min_row - source_context_cells
    source_max_row = terrain_max_row + source_context_cells
    source_min_column = terrain_min_column - source_context_cells
    source_max_column = terrain_max_column + source_context_cells
    if (
        min(source_min_row, source_min_column) < 0
        or max(source_max_row, source_max_column) >= resolution
    ):
        raise NotImplementedError("L3 V0 terrain window may not cross a cube-face edge")

    selected = (
        (face == core_face)
        & (row >= source_min_row)
        & (row <= source_max_row)
        & (column >= source_min_column)
        & (column <= source_max_column)
    )
    selected_ids = np.sort(ids[selected])
    expected_ids = (
        core_face * face_size
        + np.arange(source_min_row, source_max_row + 1, dtype=np.int64)[:, None] * resolution
        + np.arange(source_min_column, source_max_column + 1, dtype=np.int64)[None, :]
    ).reshape(-1)
    missing = np.setdiff1d(expected_ids, selected_ids, assume_unique=False)
    if len(missing):
        raise RuntimeError(
            "source handoff omits "
            f"{len(missing)} L2 cells required by the continuous L3 window; "
            "regenerate it with a wider L0 halo"
        )
    if len(selected_ids) != len(expected_ids) or len(np.unique(selected_ids)) != len(selected_ids):
        raise RuntimeError("source handoff does not provide unique rectangular L2 coverage")

    selected_row = row[selected]
    selected_column = column[selected]
    selected_core = core[selected]
    height = source_max_row - source_min_row + 1
    width = source_max_column - source_min_column + 1
    dense_core = np.zeros((height, width), dtype=bool)
    dense_core[
        selected_row[selected_core] - source_min_row,
        selected_column[selected_core] - source_min_column,
    ] = True
    dense_rings = _d8_context_rings(dense_core, halo_cells)
    selected_rings = dense_rings[
        selected_row - source_min_row,
        selected_column - source_min_column,
    ]
    inside_window = (
        (selected_row >= terrain_min_row)
        & (selected_row <= terrain_max_row)
        & (selected_column >= terrain_min_column)
        & (selected_column <= terrain_max_column)
    )
    inside_core = selected_core & inside_window
    inside_process_halo = inside_window & (selected_rings > 0)
    outside_process_domain = inside_window & ~inside_core & ~inside_process_halo
    source_context_only = ~inside_window
    role_count = (
        inside_core.astype(np.int8)
        + inside_process_halo.astype(np.int8)
        + outside_process_domain.astype(np.int8)
    )
    if np.any(role_count[inside_window] != 1) or np.any(role_count[source_context_only] != 0):
        raise RuntimeError("L3 terrain window roles are not mutually exclusive")

    masks = {
        "inside_terrain_window": inside_window,
        "inside_target_core": inside_core,
        "inside_process_halo": inside_process_halo,
        "outside_process_domain": outside_process_domain,
        "source_context_only": source_context_only,
        "process_halo_l2_ring": selected_rings,
    }
    metadata = {
        "face": core_face,
        "terrain_min_row": terrain_min_row,
        "terrain_max_row": terrain_max_row,
        "terrain_min_column": terrain_min_column,
        "terrain_max_column": terrain_max_column,
        "source_min_row": source_min_row,
        "source_max_row": source_max_row,
        "source_min_column": source_min_column,
        "source_max_column": source_max_column,
        "terrain_l2_cell_count": int(np.count_nonzero(inside_window)),
        "source_l2_cell_count": len(selected_ids),
    }
    return selected, masks, metadata


def _fixed_list(values: np.ndarray, size: int) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32()), size
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=np.float64), weights=weights))


def _overlay_parent_drainage(
    path: Path,
    child_xyz: np.ndarray,
    selected_parent_ids: np.ndarray,
    selected_parent_xyz: np.ndarray,
    core_parent_ids: np.ndarray,
    drainage_ids: np.ndarray,
    receiver_ids: np.ndarray,
    discharge_m3s: np.ndarray,
    outlet_parent_id: int,
) -> None:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    child_longitude = np.degrees(np.arctan2(child_xyz[:, 1], child_xyz[:, 0]))
    child_latitude = np.degrees(np.arcsin(np.clip(child_xyz[:, 2], -1.0, 1.0)))
    center = math.degrees(
        math.atan2(
            float(np.mean(np.sin(np.radians(child_longitude)))),
            float(np.mean(np.cos(np.radians(child_longitude)))),
        )
    )
    relative = _relative_longitude(child_longitude, center)
    latitude_scale = max(math.cos(math.radians(float(np.mean(child_latitude)))), 0.2)
    child_x = relative * latitude_scale
    padding_x = max(float(np.ptp(child_x)) * 0.04, 0.25)
    padding_y = max(float(np.ptp(child_latitude)) * 0.04, 0.25)
    left = float(np.min(child_x) - padding_x)
    right = float(np.max(child_x) + padding_x)
    bottom = float(np.min(child_latitude) - padding_y)
    top = float(np.max(child_latitude) + padding_y)

    parent_longitude = np.degrees(np.arctan2(selected_parent_xyz[:, 1], selected_parent_xyz[:, 0]))
    parent_latitude = np.degrees(np.arcsin(np.clip(selected_parent_xyz[:, 2], -1.0, 1.0)))
    parent_x = _relative_longitude(parent_longitude, center) * latitude_scale
    pixel_x = np.clip(
        ((parent_x - left) / (right - left) * (width - 1)).round(), 0, width - 1
    ).astype(np.int32)
    pixel_y = np.clip(
        ((top - parent_latitude) / (top - bottom) * (height - 1)).round(), 0, height - 1
    ).astype(np.int32)
    point_by_id = {
        int(cell): (int(pixel_x[row]), int(pixel_y[row]))
        for row, cell in enumerate(selected_parent_ids)
    }
    drainage_row = {int(cell): row for row, cell in enumerate(drainage_ids)}
    core_discharge = np.asarray(
        [discharge_m3s[drainage_row[int(cell)]] for cell in core_parent_ids],
        dtype=np.float64,
    )
    maximum_discharge = max(float(np.max(core_discharge, initial=0.0)), 1.0)
    draw = ImageDraw.Draw(image)
    for cell in core_parent_ids:
        row = drainage_row[int(cell)]
        receiver = int(receiver_ids[row])
        if receiver not in point_by_id:
            continue
        width_px = max(
            2,
            round(2.0 + 5.0 * math.sqrt(float(discharge_m3s[row]) / maximum_discharge)),
        )
        draw.line(
            (point_by_id[int(cell)], point_by_id[receiver]),
            fill=(18, 84, 132),
            width=width_px,
        )
    outlet = point_by_id[outlet_parent_id]
    radius = 8
    draw.ellipse(
        (outlet[0] - radius, outlet[1] - radius, outlet[0] + radius, outlet[1] + radius),
        fill=(15, 61, 96),
        outline=(242, 242, 235),
        width=2,
    )
    image.save(path, optimize=True)


def export_l3_target(config: L3TargetConfig) -> L3TargetResult:
    """Persist a validated target index over an immutable L2 handoff."""

    config.validate()
    handoff_manifest_path = config.handoff_dir / "manifest.json"
    handoff_validation_path = config.handoff_dir / "validation.json"
    handoff_zarr_path = config.handoff_dir / "region.zarr"
    for required in (handoff_manifest_path, handoff_validation_path, handoff_zarr_path):
        if not required.exists():
            raise FileNotFoundError(required)
    handoff_manifest = json.loads(handoff_manifest_path.read_text(encoding="utf8"))
    handoff_validation = json.loads(handoff_validation_path.read_text(encoding="utf8"))
    if not handoff_manifest.get("validation_passed") or not handoff_validation.get("passed"):
        raise RuntimeError("L3 target source handoff has not passed validation")

    root = zarr.open_group(str(handoff_zarr_path), mode="r")
    parent_ids = np.asarray(root["parent/cell_id"][:], dtype=np.int32)
    parent_area = np.asarray(root["parent/area_km2"][:], dtype=np.float64)
    parent_neighbors = np.asarray(root["parent_priors/geometry/NeighborsD4"][:], dtype=np.int32)
    parent_xyz = np.asarray(root["parent_priors/geometry/GeometryXYZ"][:], dtype=np.float64)
    drainage = pq.read_table(config.handoff_dir / "tables/drainage_graph.parquet")
    drainage_ids = _column_numpy(drainage, "cell_id", np.dtype(np.int32))
    receiver_ids = _column_numpy(drainage, "receiver_id", np.dtype(np.int32))
    basin_ids = _column_numpy(drainage, "basin_id", np.dtype(np.int32))
    drainage_row = {int(value): row for row, value in enumerate(drainage_ids)}
    outlet_row = drainage_row.get(config.outlet_parent_cell_id)
    if outlet_row is None:
        raise ValueError(
            f"outlet parent {config.outlet_parent_cell_id} is not in the handoff drainage graph"
        )
    outlet_basin_id = int(basin_ids[outlet_row])
    core_ids = _upstream_closure(config.outlet_parent_cell_id, drainage_ids, receiver_ids)
    if np.any(
        np.asarray([basin_ids[drainage_row[int(cell)]] for cell in core_ids]) != outlet_basin_id
    ):
        raise RuntimeError("selected upstream closure crosses a BasinID boundary")

    parent_row = {int(value): row for row, value in enumerate(parent_ids)}
    missing_core = sorted(set(map(int, core_ids)) - set(parent_row))
    if missing_core:
        raise RuntimeError(f"source handoff omits {len(missing_core)} target core parents")
    context_ids, context_rings, missing_context = _context_selection(
        core_ids,
        parent_ids,
        parent_neighbors,
        config.context_parent_rings,
    )
    core_set = set(map(int, core_ids))
    core_parent_rows = np.asarray([parent_row[int(value)] for value in core_ids])
    core_area_km2 = float(np.sum(parent_area[core_parent_rows]))

    core_receivers = np.asarray(
        [receiver_ids[drainage_row[int(cell)]] for cell in core_ids], dtype=np.int32
    )
    leaving = core_ids[~np.isin(core_receivers, core_ids)]
    single_outlet = bool(len(leaving) == 1 and int(leaving[0]) == config.outlet_parent_cell_id)

    child_ids = np.asarray(root["l2/geometry/cell_id"][:], dtype=np.int32)
    child_parent_ids = np.asarray(root["l2/geometry/parent_cell_id"][:], dtype=np.int32)
    child_core = np.isin(child_parent_ids, core_ids)
    child_resolution = int(root.attrs["child_face_resolution"])
    child_selection, child_roles, window = _rectangular_l2_window(
        child_ids,
        child_core,
        child_resolution,
        config.terrain_window_halo_l2_cells,
        config.terrain_source_context_l2_cells,
    )
    selected_child_rows = np.flatnonzero(child_selection).astype(np.int32)
    selected_child_ids = child_ids[child_selection]
    selected_child_parent = child_parent_ids[child_selection]
    child_inside_core = child_roles["inside_target_core"]
    source_parent_ids = np.unique(selected_child_parent)
    selected_ids = np.union1d(context_ids, source_parent_ids).astype(np.int32, copy=False)
    selected_parent_rows = np.asarray([parent_row[int(value)] for value in selected_ids])
    inside_core_parent = np.asarray([int(value) in core_set for value in selected_ids], dtype=bool)
    ring_lookup = dict(zip(map(int, context_ids), map(int, context_rings), strict=True))
    selected_parent_rings = np.asarray(
        [ring_lookup.get(int(value), -1) for value in selected_ids], dtype=np.int16
    )
    child_ring = np.asarray(
        [ring_lookup.get(int(value), -1) for value in selected_child_parent], dtype=np.int16
    )

    base_cell_area_km2 = (config.base_cell_size_m / 1_000.0) ** 2
    selected_child_area = np.asarray(root["l2/geometry/area_km2"][:], dtype=np.float64)[
        child_selection
    ]
    terrain_window_area_km2 = float(
        np.sum(selected_child_area[child_roles["inside_terrain_window"]])
    )
    estimated_core_base_cells = int(math.ceil(core_area_km2 / base_cell_area_km2))
    estimated_base_cells = int(math.ceil(terrain_window_area_km2 / base_cell_area_km2))
    area_valid = config.minimum_area_km2 <= core_area_km2 <= config.maximum_area_km2
    terrain_role_count = sum(
        np.asarray(child_roles[name], dtype=np.int8)
        for name in (
            "inside_target_core",
            "inside_process_halo",
            "outside_process_domain",
        )
    )
    terrain_window_mask = child_roles["inside_terrain_window"]
    terrain_roles_valid = bool(
        np.all(terrain_role_count[terrain_window_mask] == 1)
        and np.all(terrain_role_count[~terrain_window_mask] == 0)
    )
    validation = {
        "format_version": TARGET_FORMAT_VERSION,
        "model_version": TARGET_MODEL_VERSION,
        "target_id": config.target_id,
        "source_handoff_valid": 1,
        "source_l2_seam_valid": int(
            handoff_validation.get("terrain_parent_boundary_continuity_valid", 0)
        ),
        "outlet_parent_cell_id": config.outlet_parent_cell_id,
        "single_outlet_valid": int(single_outlet),
        "core_parent_count": len(core_ids),
        "context_parent_count": int(len(selected_ids) - len(core_ids)),
        "missing_core_parent_count": len(missing_core),
        "missing_context_parent_count": len(missing_context),
        "core_area_km2": core_area_km2,
        "minimum_area_km2": config.minimum_area_km2,
        "maximum_area_km2": config.maximum_area_km2,
        "area_valid": int(area_valid),
        "selected_l2_child_count": len(selected_child_rows),
        "core_l2_child_count": int(np.count_nonzero(child_inside_core)),
        "terrain_window_l2_cell_count": window["terrain_l2_cell_count"],
        "terrain_source_l2_cell_count": window["source_l2_cell_count"],
        "process_halo_l2_cell_count": int(np.count_nonzero(child_roles["inside_process_halo"])),
        "outside_process_domain_l2_cell_count": int(
            np.count_nonzero(child_roles["outside_process_domain"])
        ),
        "source_context_only_l2_cell_count": int(
            np.count_nonzero(child_roles["source_context_only"])
        ),
        "terrain_window_area_km2": terrain_window_area_km2,
        "terrain_window_halo_l2_cells": config.terrain_window_halo_l2_cells,
        "terrain_source_context_l2_cells": config.terrain_source_context_l2_cells,
        "terrain_window_continuous_valid": 1,
        "terrain_domain_roles_valid": int(terrain_roles_valid),
        "minimum_process_halo_ring_valid": int(
            config.terrain_window_halo_l2_cells >= 1 and np.any(child_roles["inside_process_halo"])
        ),
        "estimated_core_base_cell_count": estimated_core_base_cells,
        "estimated_base_cell_count": estimated_base_cells,
        "maximum_base_cell_count": config.maximum_base_cell_count,
        "base_cell_budget_valid": int(estimated_base_cells <= config.maximum_base_cell_count),
        "unique_selected_l2_child_ids": int(
            len(np.unique(selected_child_ids)) == len(selected_child_ids)
        ),
    }
    validation["passed"] = bool(
        validation["source_handoff_valid"] == 1
        and validation["source_l2_seam_valid"] == 1
        and validation["single_outlet_valid"] == 1
        and validation["missing_core_parent_count"] == 0
        and validation["missing_context_parent_count"] == 0
        and validation["area_valid"] == 1
        and validation["terrain_window_continuous_valid"] == 1
        and validation["terrain_domain_roles_valid"] == 1
        and validation["minimum_process_halo_ring_valid"] == 1
        and validation["base_cell_budget_valid"] == 1
        and validation["unique_selected_l2_child_ids"] == 1
    )
    if not validation["passed"]:
        raise RuntimeError(f"L3 target selection failed validation: {validation}")

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{config.output_dir.name}-target-", dir=config.output_dir.parent)
    )
    published = False
    try:
        tables_dir = staging / "tables"
        tables_dir.mkdir()
        parents = pq.read_table(config.handoff_dir / "tables/parent_cells.parquet")
        parent_table = parents.take(pa.array(selected_parent_rows, type=pa.int64()))
        parent_table = (
            parent_table.append_column(
                "inside_target_core", pa.array(inside_core_parent, type=pa.bool_())
            )
            .append_column("target_context_ring", pa.array(selected_parent_rings, type=pa.int16()))
            .append_column(
                "intersects_terrain_window",
                pa.array(
                    np.isin(
                        selected_ids,
                        np.unique(selected_child_parent[child_roles["inside_terrain_window"]]),
                    ),
                    type=pa.bool_(),
                ),
            )
            .append_column(
                "intersects_process_halo",
                pa.array(
                    np.isin(
                        selected_ids,
                        np.unique(selected_child_parent[child_roles["inside_process_halo"]]),
                    ),
                    type=pa.bool_(),
                ),
            )
        )
        pq.write_table(parent_table, tables_dir / "target_parent_cells.parquet", compression="zstd")

        child_xyz = np.asarray(root["l2/geometry/xyz"][:], dtype=np.float32)[child_selection]
        child_area = selected_child_area
        child_terrain = np.asarray(root["l2/geometry/terrain_elevation_m"][:], dtype=np.float32)[
            child_selection
        ]
        child_surface = np.asarray(root["l2/surface/surface_elevation_m"][:], dtype=np.float32)[
            child_selection
        ]
        ocean = np.asarray(root["l2/surface/ocean_fraction"][:], dtype=np.float32)[child_selection]
        lake = np.asarray(root["l2/surface/lake_fraction"][:], dtype=np.float32)[child_selection]
        wetland = np.asarray(root["l2/surface/wetland_fraction"][:], dtype=np.float32)[
            child_selection
        ]
        child_table = pa.table(
            {
                "handoff_child_row": pa.array(selected_child_rows, type=pa.int32()),
                "fine_cell_id": pa.array(selected_child_ids, type=pa.int32()),
                "parent_cell_id": pa.array(selected_child_parent, type=pa.int32()),
                "inside_target_core": pa.array(child_inside_core, type=pa.bool_()),
                "inside_terrain_window": pa.array(
                    child_roles["inside_terrain_window"], type=pa.bool_()
                ),
                "inside_process_halo": pa.array(
                    child_roles["inside_process_halo"], type=pa.bool_()
                ),
                "outside_process_domain": pa.array(
                    child_roles["outside_process_domain"], type=pa.bool_()
                ),
                "source_context_only": pa.array(
                    child_roles["source_context_only"], type=pa.bool_()
                ),
                "process_halo_l2_ring": pa.array(
                    child_roles["process_halo_l2_ring"], type=pa.int16()
                ),
                "target_context_ring": pa.array(child_ring, type=pa.int16()),
                "xyz": _fixed_list(child_xyz, 3),
                "area_km2": pa.array(child_area, type=pa.float64()),
                "terrain_elevation_m": pa.array(child_terrain, type=pa.float32()),
                "surface_elevation_m": pa.array(child_surface, type=pa.float32()),
                "ocean_fraction": pa.array(ocean, type=pa.float32()),
                "lake_fraction": pa.array(lake, type=pa.float32()),
                "wetland_fraction": pa.array(wetland, type=pa.float32()),
            }
        )
        pq.write_table(child_table, tables_dir / "target_l2_cells.parquet", compression="zstd")

        selected_drainage = drainage.filter(
            pc.is_in(drainage["cell_id"], value_set=pa.array(selected_ids, type=pa.int32()))
        )
        pq.write_table(
            selected_drainage,
            tables_dir / "target_drainage_graph.parquet",
            compression="zstd",
        )

        empty_reaches = pa.table(
            {
                "fine_cell_path": pa.array([], type=pa.list_(pa.int32())),
                "reach_kind": pa.array([], type=pa.string()),
            }
        )
        preview_path = staging / "preview.png"
        _render_preview(
            preview_path,
            child_xyz,
            child_surface,
            ocean,
            lake,
            wetland,
            child_inside_core,
            empty_reaches,
            selected_child_ids,
        )
        drainage_preview_path = staging / "coarse_drainage.png"
        shutil.copyfile(preview_path, drainage_preview_path)
        _overlay_parent_drainage(
            drainage_preview_path,
            child_xyz,
            selected_ids,
            parent_xyz[selected_parent_rows],
            core_ids,
            drainage_ids,
            receiver_ids,
            _column_numpy(drainage, "mean_discharge_m3s", np.dtype(np.float32)),
            config.outlet_parent_cell_id,
        )

        core_weights = parent_area[core_parent_rows]
        parent_elevation = np.asarray(
            root["parent_priors/sea_level/SurfaceElevationM"][:], dtype=np.float64
        )[core_parent_rows]
        parent_relief = np.asarray(
            root["parent_priors/elevation/TerrainReliefM"][:], dtype=np.float64
        )[core_parent_rows]
        annual_precipitation = np.asarray(
            root["parent_priors/climate/AnnualPrecipitationMm"][:], dtype=np.float64
        )[core_parent_rows]
        annual_temperature = np.asarray(
            root["parent_priors/climate/AnnualMeanTemperatureC"][:], dtype=np.float64
        )[core_parent_rows]
        biome_codes = np.asarray(
            root["parent_priors/derived_biomes/DominantBiomeCode"][:], dtype=np.int32
        )[core_parent_rows]
        center = np.average(parent_xyz[core_parent_rows], axis=0, weights=core_weights)
        center /= np.linalg.norm(center)
        center_latitude = math.degrees(math.asin(float(center[2])))
        center_longitude = math.degrees(math.atan2(float(center[1]), float(center[0])))
        outlet_discharge = float(drainage["mean_discharge_m3s"][outlet_row].as_py())
        outlet_contributing_area = float(drainage["contributing_area_km2"][outlet_row].as_py())

        validation_path = staging / "validation.json"
        validation_path.write_text(
            json.dumps(validation, indent=2, sort_keys=True), encoding="utf8"
        )
        table_outputs = {
            path.name: {
                "path": f"tables/{path.name}",
                "sha256": _file_checksum(path),
                "rows": pq.read_metadata(path).num_rows,
            }
            for path in sorted(tables_dir.glob("*.parquet"))
        }
        manifest = {
            "format_version": TARGET_FORMAT_VERSION,
            "model_version": TARGET_MODEL_VERSION,
            "status": "complete",
            "target_id": config.target_id,
            "selection": {
                "kind": "complete_L0_upstream_catchment_in_continuous_L2_window",
                "outlet_parent_cell_id": config.outlet_parent_cell_id,
                "basin_id": outlet_basin_id,
                "core_parent_ids": core_ids.tolist(),
                "core_parent_count": len(core_ids),
                "context_parent_count": int(len(selected_ids) - len(core_ids)),
                "context_parent_rings": config.context_parent_rings,
                "terrain_window": {
                    **window,
                    "halo_l2_cells": config.terrain_window_halo_l2_cells,
                    "source_context_l2_cells": config.terrain_source_context_l2_cells,
                    "process_connectivity": "D8",
                    "roles": [
                        "catchment_core",
                        "process_halo",
                        "outside",
                    ],
                },
                "core_area_km2": core_area_km2,
                "outlet_contributing_area_km2": outlet_contributing_area,
                "outlet_mean_discharge_m3s": outlet_discharge,
                "center_latitude_deg": center_latitude,
                "center_longitude_deg": center_longitude,
            },
            "physical_summary": {
                "elevation_p05_m": float(np.percentile(parent_elevation, 5.0)),
                "elevation_p50_m": float(np.percentile(parent_elevation, 50.0)),
                "elevation_p95_m": float(np.percentile(parent_elevation, 95.0)),
                "relief_p95_m": float(np.percentile(parent_relief, 95.0)),
                "annual_precipitation_area_mean_mm": _weighted_mean(
                    annual_precipitation, core_weights
                ),
                "annual_temperature_area_mean_c": _weighted_mean(annual_temperature, core_weights),
                "dominant_biome_codes": sorted(set(map(int, biome_codes))),
            },
            "l3_grid_contract": {
                "base_cell_size_m": config.base_cell_size_m,
                "estimated_base_cell_count": estimated_base_cells,
                "estimated_core_base_cell_count": estimated_core_base_cells,
                "terrain_window_area_km2": terrain_window_area_km2,
                "hydrological_acceptance_scope": "catchment_core_only",
                "terrain_domain": "continuous rectangular window without internal holes",
                "domain_masks": [
                    "inside_target_core",
                    "inside_process_halo",
                    "outside_process_domain",
                ],
                "adaptive_river_corridor_cell_size_m": [
                    config.adaptive_minimum_cell_size_m,
                    config.adaptive_maximum_cell_size_m,
                ],
                "terrain_storage": "chunked Zarr",
                "maximum_peak_memory_gb": config.maximum_peak_memory_gb,
                "river_identity": "canonical graph and vectors with fractional raster support",
            },
            "source": {
                "handoff_dir": str(config.handoff_dir),
                "handoff_manifest_sha256": _file_checksum(handoff_manifest_path),
                "handoff_region_id": handoff_manifest["region_id"],
                "target_config": str(config.source_config) if config.source_config else None,
                "target_config_sha256": (
                    _file_checksum(config.source_config) if config.source_config else None
                ),
            },
            "outputs": {
                "tables": table_outputs,
                "preview": {"path": "preview.png", "sha256": _file_checksum(preview_path)},
                "coarse_drainage_preview": {
                    "path": "coarse_drainage.png",
                    "sha256": _file_checksum(drainage_preview_path),
                    "semantics": "L0 routing diagnostic; not L3 river geometry",
                },
                "validation": {
                    "path": "validation.json",
                    "sha256": _file_checksum(validation_path),
                },
            },
            "validation_passed": True,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf8")
        _replace_directory(staging, config.output_dir)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return L3TargetResult(
        output_dir=config.output_dir,
        manifest_path=config.output_dir / "manifest.json",
        validation_path=config.output_dir / "validation.json",
        preview_path=config.output_dir / "preview.png",
        target_id=config.target_id,
        outlet_parent_cell_id=config.outlet_parent_cell_id,
        core_parent_count=len(core_ids),
        context_parent_count=int(len(selected_ids) - len(core_ids)),
        core_area_km2=core_area_km2,
        estimated_base_cell_count=estimated_base_cells,
    )


__all__ = [
    "L3TargetConfig",
    "L3TargetResult",
    "TARGET_FORMAT_VERSION",
    "TARGET_MODEL_VERSION",
    "export_l3_target",
]
