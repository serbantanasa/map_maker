"""Bounded sparse Hydrology Pass 2 for one eroded refined basin."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pyarrow as pa
from PIL import Image, ImageDraw

from .._hydrology_pass2_native import run_hydrology_pass2
from ..cubed_sphere import CubedSphereGrid
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult

EARTH_RADIUS_M = 6_371_000.0
NO_FIXED_RECEIVER = np.iinfo(np.int32).min
TERMINAL_OCEAN = -1
TERMINAL_HANDOFF = -2
ANCHOR_NORMAL = 0
ANCHOR_CHANNEL = 1
ANCHOR_EXCLUDED = 2
ANCHOR_OUTSIDE = 3


@dataclass(frozen=True)
class HydrologyPass2Config:
    minimum_depression_depth_m: float = 5.0
    maximum_receiver_change_fraction: float = 0.15
    maximum_receiver_change_cell_fraction: float = 0.15
    maximum_new_depression_area_fraction: float = 0.02

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "HydrologyPass2Config":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown Hydrology Pass 2 controls: {', '.join(sorted(unknown))}")
        values = {
            name: float(mapping.get(name, field.default))
            for name, field in cls.__dataclass_fields__.items()
        }
        if (
            not np.isfinite(values["minimum_depression_depth_m"])
            or values["minimum_depression_depth_m"] <= 0.0
        ):
            raise ValueError("minimum_depression_depth_m must be finite and positive")
        for name in (
            "maximum_receiver_change_fraction",
            "maximum_receiver_change_cell_fraction",
            "maximum_new_depression_area_fraction",
        ):
            if not np.isfinite(values[name]) or not 0.0 <= values[name] <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        return cls(**values)


def _artifact_table(result, name: str) -> pa.Table:
    record = result.artifact_records.get(name)
    if record is None or not isinstance(record.value, pa.Table):
        raise KeyError(f"Missing dependency table '{name}'")
    return record.value


def _column(table: pa.Table, name: str, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(
        table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
    )


def _fixed_list_column(table: pa.Table, name: str, width: int) -> np.ndarray:
    values = table[name].combine_chunks().values.to_numpy(zero_copy_only=False)
    return np.ascontiguousarray(values.reshape(table.num_rows, width), dtype=np.float32)


def _fixed_list(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32()), width
    )


def _lookup_rows(ids: np.ndarray, query: np.ndarray, *, label: str) -> np.ndarray:
    order = np.argsort(ids)
    sorted_ids = ids[order]
    positions = np.searchsorted(sorted_ids, query)
    if np.any(positions >= len(sorted_ids)) or np.any(sorted_ids[positions] != query):
        raise RuntimeError(f"{label} references an unknown identifier")
    return order[positions]


def _trunk_contract(
    profiles: pa.Table,
    reaches: pa.Table,
    cell_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixed_receivers = np.full(len(cell_ids), NO_FIXED_RECEIVER, dtype=np.int32)
    routing_bed = np.full(len(cell_ids), np.nan, dtype=np.float64)
    channel_cells = np.unique(np.asarray(profiles["fine_cell_id"], dtype=np.int32))
    channel_rows = _lookup_rows(cell_ids, channel_cells, label="channel bed")

    profile_cell_ids = np.asarray(profiles["fine_cell_id"], dtype=np.int32)
    profile_beds = np.asarray(profiles["bed_elevation_m"], dtype=np.float64)
    profile_channel_rows = _lookup_rows(
        channel_cells, profile_cell_ids, label="profile channel cell"
    )
    minimum_bed = np.full(len(channel_cells), np.inf, dtype=np.float64)
    maximum_bed = np.full(len(channel_cells), -np.inf, dtype=np.float64)
    np.minimum.at(minimum_bed, profile_channel_rows, profile_beds)
    np.maximum.at(maximum_bed, profile_channel_rows, profile_beds)
    if np.any(maximum_bed - minimum_bed > 1e-10):
        raise RuntimeError("shared channel cell has inconsistent persisted bed elevations")
    routing_bed[channel_rows] = minimum_bed

    edge_by_source: dict[int, int] = {}
    last_cell_by_reach: dict[int, int] = {}
    profile_reach_ids = np.asarray(profiles["reach_id"], dtype=np.int32)
    path_order = np.asarray(profiles["path_order"], dtype=np.int32)
    for reach_id in np.unique(profile_reach_ids):
        rows = np.flatnonzero(profile_reach_ids == reach_id)
        rows = rows[np.argsort(path_order[rows], kind="stable")]
        last_cell_by_reach[int(reach_id)] = int(profile_cell_ids[rows[-1]])
        for source_row, target_row in zip(rows[:-1], rows[1:], strict=True):
            if path_order[target_row] != path_order[source_row] + 1:
                continue
            source = int(profile_cell_ids[source_row])
            target = int(profile_cell_ids[target_row])
            previous = edge_by_source.setdefault(source, target)
            if previous != target:
                raise RuntimeError("physical channel cell has multiple downstream trunk receivers")

    reach_ids = np.asarray(reaches["reach_id"], dtype=np.int32)
    downstream_ids = np.asarray(reaches["downstream_reach_id"], dtype=np.int32)
    terminal_kinds = reaches["terminal_kind"].to_pylist()
    row_by_reach = {int(reach_id): row for row, reach_id in enumerate(reach_ids)}
    terminal_by_cell: dict[int, int] = defaultdict(lambda: TERMINAL_HANDOFF)
    for reach_id, last_cell in last_cell_by_reach.items():
        row = row_by_reach[reach_id]
        if downstream_ids[row] < 0 and terminal_kinds[row] == "ocean":
            terminal_by_cell[last_cell] = TERMINAL_OCEAN

    for cell_id, row in zip(channel_cells, channel_rows, strict=True):
        fixed_receivers[row] = edge_by_source.get(int(cell_id), terminal_by_cell[int(cell_id)])
    return channel_rows, fixed_receivers, routing_bed


def _cell_table(source: pa.Table, records: np.ndarray, routing_surface: np.ndarray) -> pa.Table:
    source_ids = np.asarray(source["fine_cell_id"], dtype=np.int32)
    if not np.array_equal(source_ids, records["fine_cell_id"]):
        raise RuntimeError("Hydrology Pass 2 changed cell order or identity")
    anchor_names = np.array(
        ["ordinary", "channel", "preserved_handoff", "outside_terminal"], dtype=object
    )[records["anchor_kind"]]
    terminal_names = np.array(["routed", "ocean", "preserved_handoff"], dtype=object)[
        records["terminal_kind"]
    ]
    table = source.append_column(
        "routing_surface_after_m", pa.array(routing_surface, type=pa.float64())
    )
    for name in (
        "baseline_receiver_id",
        "stabilized_receiver_id",
        "baseline_anchor_cell_id",
        "stabilized_anchor_cell_id",
        "baseline_depression_id",
        "stabilized_depression_id",
    ):
        table = table.append_column(name, pa.array(records[name], type=pa.int32()))
    table = table.append_column("routing_anchor_kind", pa.array(anchor_names, type=pa.string()))
    table = table.append_column("terminal_kind_pass2", pa.array(terminal_names, type=pa.string()))
    table = table.append_column(
        "receiver_changed", pa.array(records["receiver_changed"] != 0, type=pa.bool_())
    )
    table = table.append_column(
        "depression_changed", pa.array(records["depression_changed"] != 0, type=pa.bool_())
    )
    for name in (
        "baseline_hydrologic_elevation_m",
        "stabilized_hydrologic_elevation_m",
        "baseline_fill_depth_m",
        "stabilized_fill_depth_m",
        "stabilized_flow_slope",
        "contributing_area_km2",
    ):
        table = table.append_column(name, pa.array(records[name], type=pa.float64()))
    table = table.append_column(
        "stabilized_flow_direction_xyz", _fixed_list(records["flow_direction_xyz"], 3)
    )
    return table


def _depression_catalog(cells: pa.Table) -> tuple[pa.Table, dict[str, float | int]]:
    cell_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    receiver_ids = np.asarray(cells["stabilized_receiver_id"], dtype=np.int32)
    baseline_ids = np.asarray(cells["baseline_depression_id"], dtype=np.int32)
    stabilized_ids = np.asarray(cells["stabilized_depression_id"], dtype=np.int32)
    area = np.asarray(cells["area_km2"], dtype=np.float64)
    depth = np.asarray(cells["stabilized_fill_depth_m"], dtype=np.float64)
    level = np.asarray(cells["stabilized_hydrologic_elevation_m"], dtype=np.float64)
    receiver_depression = np.full(len(cells), -1, dtype=np.int32)
    routed = receiver_ids >= 0
    target_rows = np.full(len(cells), -1, dtype=np.int64)
    if np.any(routed):
        target_rows[routed] = _lookup_rows(
            cell_ids, receiver_ids[routed], label="depression spill receiver"
        )
        receiver_depression[routed] = stabilized_ids[target_rows[routed]]
    upstream_count = np.zeros(len(cells), dtype=np.int32)
    np.add.at(upstream_count, target_rows[routed], 1)
    ready = deque(np.flatnonzero(upstream_count == 0).tolist())
    downstream_rank = np.full(len(cells), -1, dtype=np.int64)
    processed = 0
    while ready:
        row = ready.popleft()
        downstream_rank[row] = processed
        processed += 1
        target = target_rows[row]
        if target >= 0:
            upstream_count[target] -= 1
            if upstream_count[target] == 0:
                ready.append(int(target))
    if processed != len(cells):
        raise RuntimeError("local depression candidates use a cyclic receiver graph")

    rows: list[dict[str, object]] = []
    new_area = float(np.sum(area[(stabilized_ids >= 0) & (baseline_ids < 0)]))
    for depression_id in np.unique(stabilized_ids[stabilized_ids >= 0]):
        mask = stabilized_ids == depression_id
        boundary = mask & (receiver_depression != depression_id)
        boundary_rows = np.flatnonzero(boundary)
        if len(boundary_rows) == 0:
            raise RuntimeError("local depression candidate has no spill receiver")
        if float(np.ptp(level[mask])) > 1e-6:
            raise RuntimeError("local depression candidate does not have a level spill surface")
        spill_row = max(
            boundary_rows,
            key=lambda row: (int(downstream_rank[row]), -int(cell_ids[row])),
        )
        overlaps = baseline_ids[mask]
        overlap_area = area[mask]
        valid_overlap = overlaps >= 0
        dominant_baseline = -1
        dominant_overlap_area = 0.0
        if np.any(valid_overlap):
            overlap_totals: dict[int, float] = defaultdict(float)
            for inherited_id, cell_area in zip(
                overlaps[valid_overlap], overlap_area[valid_overlap], strict=True
            ):
                overlap_totals[int(inherited_id)] += float(cell_area)
            dominant_baseline, dominant_overlap_area = max(
                overlap_totals.items(), key=lambda item: (item[1], -item[0])
            )
        candidate_area = float(np.sum(area[mask]))
        if dominant_baseline < 0:
            status = "new"
        elif np.all(baseline_ids[mask] == depression_id) and np.all(
            stabilized_ids[baseline_ids == depression_id] == depression_id
        ):
            status = "stable"
        else:
            status = "changed"
        rows.append(
            {
                "depression_id": int(depression_id),
                "spill_cell_id": int(cell_ids[spill_row]),
                "spill_receiver_id": int(receiver_ids[spill_row]),
                "cell_count": int(np.count_nonzero(mask)),
                "area_km2": candidate_area,
                "potential_fill_volume_km3": float(np.sum(area[mask] * depth[mask]) / 1_000.0),
                "maximum_fill_depth_m": float(np.max(depth[mask])),
                "spill_elevation_m": float(level[spill_row]),
                "dominant_baseline_depression_id": dominant_baseline,
                "baseline_overlap_fraction": dominant_overlap_area / max(candidate_area, 1e-12),
                "status": status,
            }
        )
    schema = pa.schema(
        [
            ("depression_id", pa.int32()),
            ("spill_cell_id", pa.int32()),
            ("spill_receiver_id", pa.int32()),
            ("cell_count", pa.int32()),
            ("area_km2", pa.float64()),
            ("potential_fill_volume_km3", pa.float64()),
            ("maximum_fill_depth_m", pa.float64()),
            ("spill_elevation_m", pa.float64()),
            ("dominant_baseline_depression_id", pa.int32()),
            ("baseline_overlap_fraction", pa.float64()),
            ("status", pa.string()),
        ]
    )
    catalog = pa.Table.from_pylist(rows, schema=schema)
    removed_area = float(np.sum(area[(baseline_ids >= 0) & (stabilized_ids < 0)]))
    return catalog, {
        "new_depression_area_km2": new_area,
        "removed_depression_area_km2": removed_area,
        "new_depression_count": sum(row["status"] == "new" for row in rows),
        "changed_depression_count": sum(row["status"] == "changed" for row in rows),
        "stable_depression_count": sum(row["status"] == "stable" for row in rows),
    }


def _correction_catalog(cells: pa.Table) -> pa.Table:
    changed = np.asarray(cells["receiver_changed"]) | np.asarray(cells["depression_changed"])
    source = cells.filter(pa.array(changed, type=pa.bool_()))
    receiver_changed = np.asarray(source["receiver_changed"])
    depression_changed = np.asarray(source["depression_changed"])
    reason = np.where(
        receiver_changed & depression_changed,
        "receiver_and_depression",
        np.where(receiver_changed, "receiver", "depression"),
    )
    return pa.table(
        {
            "fine_cell_id": source["fine_cell_id"],
            "parent_cell_id": source["parent_cell_id"],
            "area_km2": source["area_km2"],
            "baseline_receiver_id": source["baseline_receiver_id"],
            "stabilized_receiver_id": source["stabilized_receiver_id"],
            "baseline_depression_id": source["baseline_depression_id"],
            "stabilized_depression_id": source["stabilized_depression_id"],
            "correction_reason": pa.array(reason, type=pa.string()),
        }
    )


def _graph_audit(
    cells: pa.Table,
    fixed_receivers: np.ndarray,
    channel_rows: np.ndarray,
    *,
    radius_m: float,
) -> dict[str, float | int]:
    cell_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    receivers = np.asarray(cells["stabilized_receiver_id"], dtype=np.int32)
    baseline_receivers = np.asarray(cells["baseline_receiver_id"], dtype=np.int32)
    baseline_depressions = np.asarray(cells["baseline_depression_id"], dtype=np.int32)
    stabilized_depressions = np.asarray(cells["stabilized_depression_id"], dtype=np.int32)
    contributing = np.asarray(cells["contributing_area_km2"], dtype=np.float64)
    area = np.asarray(cells["area_km2"], dtype=np.float64)
    source_active = np.asarray(cells["source_active"])
    receiver_changed = (baseline_receivers != receivers) & source_active
    depression_changed = (baseline_depressions != stabilized_depressions) & source_active
    routed = receivers >= 0
    target_rows = np.full(len(cells), -1, dtype=np.int64)
    if np.any(routed):
        target_rows[routed] = _lookup_rows(cell_ids, receivers[routed], label="stabilized receiver")
    upstream_count = np.zeros(len(cells), dtype=np.int32)
    np.add.at(upstream_count, target_rows[routed], 1)
    ready = deque(np.flatnonzero(upstream_count == 0).tolist())
    processed = 0
    while ready:
        row = ready.popleft()
        processed += 1
        target = target_rows[row]
        if target >= 0:
            upstream_count[target] -= 1
            if upstream_count[target] == 0:
                ready.append(int(target))
    nondecreasing_error = float(
        np.max(
            np.maximum(contributing[routed] - contributing[target_rows[routed]], 0.0),
            initial=0.0,
        )
    )
    terminal_area = float(np.sum(contributing[~routed]))
    active_area = float(np.sum(area[source_active]))
    trunk_error = int(np.count_nonzero(receivers[channel_rows] != fixed_receivers[channel_rows]))
    xyz = _fixed_list_column(cells, "xyz", 3).astype(np.float64)
    direction = _fixed_list_column(cells, "stabilized_flow_direction_xyz", 3).astype(np.float64)
    direction_norm = np.linalg.norm(direction[routed], axis=1)
    tangent_error = np.abs(np.sum(direction[routed] * xyz[routed], axis=1))
    source_xyz = xyz[routed]
    target_xyz = xyz[target_rows[routed]]
    length_m = np.arccos(np.clip(np.sum(source_xyz * target_xyz, axis=1), -1.0, 1.0)) * radius_m
    hydrologic = np.asarray(cells["stabilized_hydrologic_elevation_m"], dtype=np.float64)
    expected_slope = np.maximum(
        (hydrologic[routed] - hydrologic[target_rows[routed]]) / length_m, 0.0
    )
    published_slope = np.asarray(cells["stabilized_flow_slope"], dtype=np.float64)[routed]
    excluded = np.asarray(cells["process_excluded"])
    process_exclusion_valid = bool(
        np.all(receivers[excluded] == TERMINAL_HANDOFF)
        and np.all(baseline_receivers[excluded] == TERMINAL_HANDOFF)
        and np.all(stabilized_depressions[excluded] < 0)
        and np.all(baseline_depressions[excluded] < 0)
    )
    valid_terminal = (receivers == TERMINAL_OCEAN) | (receivers == TERMINAL_HANDOFF)
    valid_baseline_terminal = (baseline_receivers == TERMINAL_OCEAN) | (
        baseline_receivers == TERMINAL_HANDOFF
    )
    invalid_terminal_count = int(
        np.count_nonzero((receivers < 0) & ~valid_terminal)
        + np.count_nonzero((baseline_receivers < 0) & ~valid_baseline_terminal)
    )
    terminal_direction_error = float(
        np.max(np.linalg.norm(direction[~routed], axis=1), initial=0.0)
    )
    terminal_slope_error = float(
        np.max(
            np.abs(np.asarray(cells["stabilized_flow_slope"], dtype=np.float64)[~routed]),
            initial=0.0,
        )
    )
    receiver_changed_area = float(np.sum(area[receiver_changed]))
    return {
        "independent_graph_valid": int(processed == len(cells)),
        "independent_processed_cell_count": processed,
        "independent_terminal_area_km2": terminal_area,
        "independent_active_area_km2": active_area,
        "independent_contributing_area_residual_km2": terminal_area - active_area,
        "maximum_downstream_contributing_area_error_km2": nondecreasing_error,
        "trunk_receiver_mismatch_count": trunk_error,
        "independent_process_exclusion_valid": int(process_exclusion_valid),
        "invalid_terminal_receiver_count": invalid_terminal_count,
        "independent_receiver_changed_cell_count": int(np.count_nonzero(receiver_changed)),
        "independent_receiver_changed_area_km2": receiver_changed_area,
        "independent_receiver_changed_area_fraction": receiver_changed_area
        / max(active_area, 1e-12),
        "independent_depression_changed_cell_count": int(np.count_nonzero(depression_changed)),
        "maximum_flow_direction_norm_error": float(
            np.max(np.abs(direction_norm - 1.0), initial=0.0)
        ),
        "maximum_flow_direction_tangent_error": float(np.max(tangent_error, initial=0.0)),
        "maximum_flow_slope_relation_error": float(
            np.max(np.abs(expected_slope - published_slope), initial=0.0)
        ),
        "maximum_terminal_flow_direction_error": terminal_direction_error,
        "maximum_terminal_flow_slope_error": terminal_slope_error,
    }


def _cube_net_visualizer(result, request: VisualizationRequest) -> VisualizationResult | None:
    cell_record = result.artifact_records.get("StabilizedBasinCellCatalog")
    metadata_record = result.artifact_records.get("HydrologyPass2Metadata")
    if (
        cell_record is None
        or metadata_record is None
        or not isinstance(cell_record.value, pa.Table)
    ):
        return None
    cells = cell_record.value
    metadata = metadata_record.value
    fine_resolution = int(metadata["fine_resolution"])
    display_resolution = min(fine_resolution, 768)
    placements = np.array([[1, 1], [1, 3], [1, 2], [1, 0], [0, 1], [2, 1]])
    face = np.asarray(cells["face"], dtype=np.int32)
    row = np.asarray(cells["row"], dtype=np.int32)
    col = np.asarray(cells["col"], dtype=np.int32)
    elevation = np.asarray(cells["terrain_elevation_after_m"], dtype=np.float64)
    low, high = np.percentile(elevation, [2.0, 98.0])
    normalized = np.clip((elevation - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    colors = np.stack(
        (48 + 150 * normalized, 73 + 138 * normalized, 49 + 105 * normalized), axis=1
    ).astype(np.uint8)
    image = np.full((display_resolution * 3, display_resolution * 4, 3), 18, dtype=np.uint8)
    display_row = np.minimum(row * display_resolution // fine_resolution, display_resolution - 1)
    display_col = np.minimum(col * display_resolution // fine_resolution, display_resolution - 1)
    net_row = placements[face, 0] * display_resolution + display_row
    net_col = placements[face, 1] * display_resolution + display_col
    image[net_row, net_col] = colors
    depression = np.asarray(cells["stabilized_depression_id"], dtype=np.int32) >= 0
    changed = np.asarray(cells["receiver_changed"])
    channel = np.asarray(cells["routing_anchor_kind"].to_pylist()) == "channel"
    depression_depth = np.asarray(cells["stabilized_fill_depth_m"], dtype=np.float64)
    if np.any(depression):
        depression_strength = np.clip(
            np.log1p(depression_depth[depression])
            / max(float(np.log1p(np.max(depression_depth[depression]))), 1e-12),
            0.10,
            0.42,
        )
        base = image[net_row[depression], net_col[depression]].astype(np.float64)
        target = np.array([47.0, 118.0, 145.0])
        image[net_row[depression], net_col[depression]] = (
            base * (1.0 - depression_strength[:, None]) + target * depression_strength[:, None]
        ).astype(np.uint8)
    image[net_row[changed], net_col[changed]] = (219, 146, 54)
    image[net_row[channel], net_col[channel]] = (20, 169, 226)
    rendered = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(rendered)
    correction_count = int(metadata["receiver_changed_cell_count"])
    draw.text((20, 20), f"Pass 2 receiver corrections: {correction_count:,}", fill=(235, 236, 231))
    output = request.output_dir / "stabilized_drainage.png"
    padding = max(12, display_resolution // 40)
    left = max(0, int(np.min(net_col)) - padding)
    upper = max(0, int(np.min(net_row)) - padding)
    right = min(rendered.width, int(np.max(net_col)) + padding + 1)
    lower = min(rendered.height, int(np.max(net_row)) + padding + 1)
    rendered = rendered.crop((left, upper, right, lower))
    scale = min(1600 / rendered.width, 1000 / rendered.height)
    if scale > 1.0:
        rendered = rendered.resize(
            (max(1, round(rendered.width * scale)), max(1, round(rendered.height * scale))),
            Image.Resampling.NEAREST,
        )
    rendered.save(output)
    return VisualizationResult(
        output,
        "StabilizedBasinCellCatalog",
        {
            "receiver_changed_area_fraction": metadata["receiver_changed_area_fraction"],
            "stabilized_depression_count": metadata["stabilized_depression_count"],
        },
    )


@stage(
    "hydrology_pass2",
    inputs=("basin_erosion", "basin_refinement", "planet"),
    outputs=(
        "StabilizedBasinCellCatalog",
        "StabilizedRiverReachCatalog",
        "LocalDepressionCandidateCatalog",
        "HydrologyCorrectionCatalog",
        "HydrologyPass2Metadata",
    ),
    version="v2",
    native_libraries=("hydrology_pass2_native",),
    visualizer=_cube_net_visualizer,
)
def hydrology_pass2_stage(context, deps, config_mapping: Mapping[str, object]):
    config = HydrologyPass2Config.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("Hydrology Pass 2 requires topology: cubed_sphere")
    erosion = deps["basin_erosion"]
    refinement = deps["basin_refinement"]
    erosion_metadata = erosion.artifact_records["BasinErosionMetadata"].value
    if erosion_metadata["source_to_sink_ready"] != 1:
        raise RuntimeError("Hydrology Pass 2 requires source-to-sink-ready erosion artifacts")
    cells = _artifact_table(erosion, "ErodedBasinCellCatalog")
    parents = _artifact_table(erosion, "BasinErosionParentCatalog")
    reaches = _artifact_table(erosion, "FluvialRiverReachCatalog")
    profiles = _artifact_table(erosion, "ChannelBedProfileCatalog")
    memberships = _artifact_table(refinement, "RefinedReachCellCatalog")
    cell_ids = _column(cells, "fine_cell_id", np.dtype(np.int32))
    parent_ids = _column(cells, "parent_cell_id", np.dtype(np.int32))
    channel_rows, fixed_receivers, routing_bed = _trunk_contract(profiles, reaches, cell_ids)
    parent_catalog_ids = _column(parents, "parent_cell_id", np.dtype(np.int32))
    parent_rows = _lookup_rows(parent_catalog_ids, parent_ids, label="child parent")
    inside = np.asarray(parents["inside_selected_basin"])[parent_rows]
    excluded = np.asarray(cells["process_excluded"])
    source_active = np.ascontiguousarray(inside & ~excluded, dtype=np.uint8)
    if np.any(excluded[channel_rows]):
        raise RuntimeError("physical channel support overlaps a process-excluded child")
    anchor_kinds = np.full(cells.num_rows, ANCHOR_NORMAL, dtype=np.uint8)
    anchor_kinds[excluded] = ANCHOR_EXCLUDED
    anchor_kinds[~inside] = ANCHOR_OUTSIDE
    anchor_kinds[channel_rows] = ANCHOR_CHANNEL
    routing_surface = _column(cells, "terrain_elevation_after_m", np.dtype(np.float64)).copy()
    routing_surface[channel_rows] = routing_bed[channel_rows]
    planet = deps["planet"].artifact_records["PlanetMetadata"].value
    radius_m = float(planet["planet_radius_earth"]) * EARTH_RADIUS_M
    controls = {
        "fine_resolution": int(erosion_metadata["fine_resolution"]),
        "minimum_depression_depth_m": config.minimum_depression_depth_m,
        "planet_radius_m": radius_m,
    }
    with context.timed("sparse_hydrology_pass2_kernel"):
        records, metadata = run_hydrology_pass2(
            controls=controls,
            cell_ids=cell_ids,
            terrain_before_m=_column(cells, "terrain_elevation_m", np.dtype(np.float64)),
            routing_surface_after_m=routing_surface,
            cell_areas_km2=_column(cells, "area_km2", np.dtype(np.float64)),
            cell_xyz=_fixed_list_column(cells, "xyz", 3),
            anchor_kinds=anchor_kinds,
            source_active=source_active,
            fixed_receiver_ids=fixed_receivers,
        )
    stabilized_cells = _cell_table(cells, records, routing_surface)
    stabilized_cells = stabilized_cells.append_column(
        "source_active", pa.array(source_active != 0, type=pa.bool_())
    )
    depression_catalog, depression_metadata = _depression_catalog(stabilized_cells)
    corrections = _correction_catalog(stabilized_cells)
    graph_metadata = _graph_audit(
        stabilized_cells, fixed_receivers, channel_rows, radius_m=radius_m
    )
    reach_status = np.where(
        np.asarray(reaches["reach_kind"].to_pylist()) == "channel",
        "physical_trunk_preserved",
        "nonphysical_handoff_preserved",
    )
    stabilized_reaches = reaches.append_column(
        "hydrology_pass2_status", pa.array(reach_status, type=pa.string())
    )
    stabilized_reaches = stabilized_reaches.append_column(
        "trunk_identity_preserved",
        pa.array(np.ones(reaches.num_rows, dtype=bool), type=pa.bool_()),
    )
    active_cell_count = int(np.count_nonzero(source_active))
    receiver_change_cell_fraction = graph_metadata["independent_receiver_changed_cell_count"] / max(
        active_cell_count, 1
    )
    active_area_km2 = float(graph_metadata["independent_active_area_km2"])
    new_depression_area_fraction = float(depression_metadata["new_depression_area_km2"]) / max(
        active_area_km2, 1e-12
    )
    corridor_ids = np.asarray(memberships["fine_cell_id"], dtype=np.int32)
    new_depression_ids = set(
        int(value)
        for value, status in zip(
            depression_catalog["depression_id"].to_pylist(),
            depression_catalog["status"].to_pylist(),
            strict=True,
        )
        if status == "new"
    )
    stabilized_depression_ids = np.asarray(
        stabilized_cells["stabilized_depression_id"], dtype=np.int32
    )
    corridor_rows = _lookup_rows(cell_ids, corridor_ids, label="refined corridor")
    new_corridor_depression_count = int(
        np.count_nonzero(
            np.isin(stabilized_depression_ids[corridor_rows], list(new_depression_ids))
        )
    )
    additional_erosion_correction_required = new_corridor_depression_count > 0
    metadata.update(
        {
            **asdict(config),
            **depression_metadata,
            **graph_metadata,
            "selected_basin_id": int(erosion_metadata["selected_basin_id"]),
            "fine_resolution": int(erosion_metadata["fine_resolution"]),
            "receiver_change_cell_fraction": receiver_change_cell_fraction,
            "new_depression_area_fraction": new_depression_area_fraction,
            "new_corridor_depression_membership_count": new_corridor_depression_count,
            "bounded_reroute_applied": int(
                graph_metadata["independent_receiver_changed_cell_count"] > 0
            ),
            "additional_erosion_correction_required": int(additional_erosion_correction_required),
            "routing_semantics": "volume_adjusted_means_with_subgrid_channel_bed_anchors",
            "depression_semantics": "topographic_storage_candidates_not_waterbody_labels",
            "trunk_semantics": "pass1_reach_and_connector_identities_preserved",
        }
    )
    area_tolerance = 1e-8 * max(active_area_km2, 1.0)
    if metadata["graph_valid"] != 1 or graph_metadata["independent_graph_valid"] != 1:
        raise RuntimeError("Hydrology Pass 2 produced a cyclic or uncovered receiver graph")
    if metadata["trunk_preserved_valid"] != 1 or graph_metadata["trunk_receiver_mismatch_count"]:
        raise RuntimeError("Hydrology Pass 2 changed the accepted physical trunk graph")
    if (
        metadata["process_exclusion_valid"] != 1
        or graph_metadata["independent_process_exclusion_valid"] != 1
    ):
        raise RuntimeError("Hydrology Pass 2 routed process through preserved depression support")
    if graph_metadata["invalid_terminal_receiver_count"]:
        raise RuntimeError("Hydrology Pass 2 emitted an invalid terminal receiver sentinel")
    if (
        abs(float(metadata["contributing_area_residual_km2"])) > area_tolerance
        or abs(float(graph_metadata["independent_contributing_area_residual_km2"])) > area_tolerance
        or float(graph_metadata["maximum_downstream_contributing_area_error_km2"]) > area_tolerance
    ):
        raise RuntimeError("Hydrology Pass 2 does not conserve contributing area")
    if (
        float(graph_metadata["maximum_flow_direction_norm_error"]) > 1e-5
        or float(graph_metadata["maximum_flow_direction_tangent_error"]) > 1e-5
        or float(graph_metadata["maximum_flow_slope_relation_error"]) > 1e-12
        or float(graph_metadata["maximum_terminal_flow_direction_error"]) > 1e-12
        or float(graph_metadata["maximum_terminal_flow_slope_error"]) > 1e-12
    ):
        raise RuntimeError("Hydrology Pass 2 flow vectors or slopes are inconsistent")
    count_fields_agree = (
        metadata["cell_count"] == cells.num_rows
        and metadata["active_cell_count"] == active_cell_count
        and metadata["receiver_changed_cell_count"]
        == graph_metadata["independent_receiver_changed_cell_count"]
        and metadata["depression_changed_cell_count"]
        == graph_metadata["independent_depression_changed_cell_count"]
    )
    area_fields_agree = (
        abs(float(metadata["active_area_km2"]) - active_area_km2) <= area_tolerance
        and abs(
            float(metadata["receiver_changed_area_km2"])
            - float(graph_metadata["independent_receiver_changed_area_km2"])
        )
        <= area_tolerance
    )
    if not count_fields_agree or not area_fields_agree:
        raise RuntimeError("Hydrology Pass 2 native and emitted correction budgets disagree")
    if (
        graph_metadata["independent_receiver_changed_area_fraction"]
        > config.maximum_receiver_change_fraction
    ):
        raise RuntimeError("Hydrology Pass 2 receiver-change area exceeds the stability bound")
    if receiver_change_cell_fraction > config.maximum_receiver_change_cell_fraction:
        raise RuntimeError("Hydrology Pass 2 receiver-change count exceeds the stability bound")
    if new_depression_area_fraction > config.maximum_new_depression_area_fraction:
        raise RuntimeError("Hydrology Pass 2 creates excessive new depression-candidate area")
    context.logger.log_event(
        {"type": "hydrology_pass2_summary", "stage": "hydrology_pass2", **metadata}
    )
    return {
        "StabilizedBasinCellCatalog": stabilized_cells,
        "StabilizedRiverReachCatalog": stabilized_reaches,
        "LocalDepressionCandidateCatalog": depression_catalog,
        "HydrologyCorrectionCatalog": corrections,
        "HydrologyPass2Metadata": metadata,
    }


__all__ = ["HydrologyPass2Config", "hydrology_pass2_stage"]
