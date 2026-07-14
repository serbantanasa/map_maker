"""Sparse hierarchical refinement of one complete coarse basin footprint."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import heapq
from typing import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from PIL import Image, ImageDraw

from .._refinement_native import run_basin_refinement
from ..cubed_sphere import CubedSphereGrid
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class BasinRefinementConfig:
    refinement_factor: int = 16
    basin_id: int | None = None
    terrain_noise_fraction: float = 0.45

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "BasinRefinementConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown basin refinement controls: {', '.join(sorted(unknown))}")
        factor = int(mapping.get("refinement_factor", cls.refinement_factor))
        basin_value = mapping.get("basin_id", cls.basin_id)
        basin_id = None if basin_value is None else int(basin_value)
        terrain_noise_fraction = float(
            mapping.get("terrain_noise_fraction", cls.terrain_noise_fraction)
        )
        if factor < 2 or factor > 32 or factor & (factor - 1):
            raise ValueError("refinement_factor must be a power of two in [2, 32]")
        if basin_id is not None and basin_id < 0:
            raise ValueError("basin_id must be non-negative when provided")
        if not np.isfinite(terrain_noise_fraction) or not 0.0 <= terrain_noise_fraction <= 1.0:
            raise ValueError("terrain_noise_fraction must be finite and in [0, 1]")
        return cls(
            refinement_factor=factor,
            basin_id=basin_id,
            terrain_noise_fraction=terrain_noise_fraction,
        )


def _artifact_array(result, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _artifact_table(result, name: str) -> pa.Table:
    record = result.artifact_records.get(name)
    if record is None or not isinstance(record.value, pa.Table):
        raise KeyError(f"Missing dependency table '{name}'")
    return record.value


def _column_numpy(table: pa.Table, name: str, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(
        table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
    )


def _select_basin(
    basin_catalog: pa.Table, reaches: pa.Table, requested_basin_id: int | None
) -> int:
    reach_basins = set(int(value) for value in reaches["basin_id"].to_pylist())
    catalog_ids = np.asarray(basin_catalog["basin_id"], dtype=np.int32)
    if requested_basin_id is not None:
        if requested_basin_id not in set(int(value) for value in catalog_ids):
            raise ValueError(f"basin_id {requested_basin_id} does not exist")
        if requested_basin_id not in reach_basins:
            raise ValueError(f"basin_id {requested_basin_id} has no registered river reaches")
        return requested_basin_id

    sink_type = np.asarray(basin_catalog["sink_type"], dtype=np.uint8)
    area = np.asarray(basin_catalog["area_km2"], dtype=np.float64)
    candidates = [
        index
        for index, basin_id in enumerate(catalog_ids)
        if int(basin_id) in reach_basins and int(sink_type[index]) == 1
    ]
    if not candidates:
        candidates = [
            index for index, basin_id in enumerate(catalog_ids) if int(basin_id) in reach_basins
        ]
    if not candidates:
        raise ValueError("hydrology produced no basin with registered river reaches")
    return int(max(candidates, key=lambda index: (float(area[index]), -int(catalog_ids[index]))))


def _downstream_first_reaches(reaches: pa.Table) -> pa.Table:
    reach_ids = np.asarray(reaches["reach_id"], dtype=np.int32)
    downstream_ids = np.asarray(reaches["downstream_reach_id"], dtype=np.int32)
    row_by_id = {int(reach_id): row for row, reach_id in enumerate(reach_ids)}
    upstream_rows: dict[int, list[int]] = defaultdict(list)
    ready: list[tuple[int, int]] = []
    for row, (reach_id, downstream_id) in enumerate(zip(reach_ids, downstream_ids, strict=True)):
        downstream_row = row_by_id.get(int(downstream_id))
        if downstream_row is None:
            heapq.heappush(ready, (int(reach_id), row))
        else:
            upstream_rows[downstream_row].append(row)

    order: list[int] = []
    while ready:
        _, row = heapq.heappop(ready)
        order.append(row)
        for upstream_row in upstream_rows.get(row, ()):
            heapq.heappush(ready, (int(reach_ids[upstream_row]), upstream_row))
    if len(order) != len(reach_ids):
        raise RuntimeError("inherited river reach graph contains a cycle")
    return reaches.take(pa.array(order, type=pa.int64()))


def _flatten_paths(paths: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(paths) + 1, dtype=np.int32)
    for index, path in enumerate(paths):
        offsets[index + 1] = offsets[index] + len(path)
    flattened = np.fromiter(
        (cell for path in paths for cell in path), dtype=np.int32, count=int(offsets[-1])
    )
    return offsets, flattened


def _fixed_list(values: np.ndarray, size: int) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32()), size
    )


def _cell_table(records: np.ndarray) -> pa.Table:
    return pa.table(
        {
            "fine_cell_id": pa.array(records["fine_cell_id"], type=pa.int32()),
            "parent_cell_id": pa.array(records["parent_cell_id"], type=pa.int32()),
            "face": pa.array(records["face"], type=pa.int32()),
            "row": pa.array(records["row"], type=pa.int32()),
            "col": pa.array(records["col"], type=pa.int32()),
            "xyz": _fixed_list(records["xyz"], 3),
            "area_km2": pa.array(records["area_km2"], type=pa.float64()),
            "terrain_elevation_m": pa.array(records["terrain_elevation_m"], type=pa.float32()),
            "terrain_offset_m": pa.array(records["terrain_offset_m"], type=pa.float32()),
            "parent_relief_m": pa.array(records["parent_relief_m"], type=pa.float32()),
        }
    )


def _parent_table(
    records: np.ndarray,
    parent_ids: np.ndarray,
    inside_selected_basin: np.ndarray,
    parent_elevation_m: np.ndarray,
    parent_relief_m: np.ndarray,
    parent_areas_km2: np.ndarray,
    factor: int,
) -> pa.Table:
    children_per_parent = factor * factor
    child_areas = records["area_km2"].reshape(len(parent_ids), children_per_parent)
    child_elevation = records["terrain_elevation_m"].reshape(len(parent_ids), children_per_parent)
    child_parent = records["parent_cell_id"].reshape(len(parent_ids), children_per_parent)
    if not np.all(child_parent == parent_ids[:, None]):
        raise RuntimeError("native refinement returned children outside their parent groups")
    restricted_area = np.sum(child_areas, axis=1)
    restricted_elevation = np.sum(child_areas * child_elevation, axis=1) / restricted_area
    return pa.table(
        {
            "parent_cell_id": pa.array(parent_ids, type=pa.int32()),
            "inside_selected_basin": pa.array(inside_selected_basin, type=pa.bool_()),
            "child_count": pa.array(
                np.full(len(parent_ids), children_per_parent, dtype=np.int32), type=pa.int32()
            ),
            "parent_area_km2": pa.array(parent_areas_km2, type=pa.float64()),
            "restricted_child_area_km2": pa.array(restricted_area, type=pa.float64()),
            "area_relative_error": pa.array(
                np.abs(restricted_area - parent_areas_km2) / parent_areas_km2,
                type=pa.float64(),
            ),
            "parent_elevation_m": pa.array(parent_elevation_m, type=pa.float32()),
            "restricted_child_elevation_m": pa.array(
                restricted_elevation.astype(np.float32), type=pa.float32()
            ),
            "elevation_error_m": pa.array(
                (restricted_elevation - parent_elevation_m).astype(np.float32),
                type=pa.float32(),
            ),
            "parent_relief_m": pa.array(parent_relief_m, type=pa.float32()),
        }
    )


def _lookup_rows(sorted_ids: np.ndarray, order: np.ndarray, query: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_ids, query)
    if np.any(positions >= len(sorted_ids)) or np.any(sorted_ids[positions] != query):
        raise RuntimeError("refined reach path references a cell outside the selected basin")
    return order[positions]


def _reach_table(
    source: pa.Table,
    native_reaches: np.ndarray,
    path_cells: np.ndarray,
    cells: np.ndarray,
    *,
    fine_resolution: int,
    factor: int,
    planet_radius_m: float,
) -> tuple[pa.Table, bool, float]:
    fine_ids = cells["fine_cell_id"]
    fine_order = np.argsort(fine_ids)
    sorted_ids = fine_ids[fine_order]
    fine_paths: list[list[int]] = []
    polylines: list[list[list[float]]] = []
    maximum_length_error = 0.0
    coarse_resolution = fine_resolution // factor
    fine_face_size = fine_resolution * fine_resolution
    coarse_face_size = coarse_resolution * coarse_resolution
    parent_paths = source["cell_path"].to_pylist()
    for reach_index, record in enumerate(native_reaches):
        start = int(record["path_offset"])
        end = start + int(record["path_count"])
        path = path_cells[start:end]
        rows = _lookup_rows(sorted_ids, fine_order, path)
        face = path // fine_face_size
        within_face = path % fine_face_size
        fine_row = within_face // fine_resolution
        fine_col = within_face % fine_resolution
        parent_path = (
            face * coarse_face_size + (fine_row // factor) * coarse_resolution + fine_col // factor
        )
        if _collapse_adjacent(parent_path) != parent_paths[reach_index]:
            raise RuntimeError("fine reach path does not preserve its inherited parent path")
        fine_paths.append(path.tolist())
        xyz = cells["xyz"][rows].astype(np.float64)
        polylines.append(xyz.tolist())
        edge_length_m = (
            np.arccos(np.clip(np.sum(xyz[:-1] * xyz[1:], axis=1), -1.0, 1.0)) * planet_radius_m
        )
        maximum_length_error = max(
            maximum_length_error,
            abs(float(np.sum(edge_length_m)) - float(record["path_length_m"])),
        )

    table = pa.table(
        {
            "reach_id": source["reach_id"],
            "parent_reach_id": source["reach_id"],
            "basin_id": source["basin_id"],
            "from_parent_cell": source["from_node"],
            "to_parent_cell": source["to_node"],
            "downstream_reach_id": source["downstream_reach_id"],
            "parent_cell_path": source["cell_path"],
            "fine_cell_path": pa.array(fine_paths, type=pa.list_(pa.int32())),
            "polyline_on_cubed_sphere": pa.array(
                polylines, type=pa.list_(pa.list_(pa.float32(), 3))
            ),
            "entry_fine_cell": pa.array(native_reaches["entry_fine_cell"], type=pa.int32()),
            "exit_fine_cell": pa.array(native_reaches["exit_fine_cell"], type=pa.int32()),
            "path_length_km": pa.array(
                native_reaches["path_length_m"] / 1_000.0, type=pa.float64()
            ),
            "discharge_mean": source["discharge_mean"],
            "discharge_seasonal": source["discharge_seasonal"],
            "velocity_mean": source["velocity_mean"],
            "slope": source["slope"],
            "stream_power": source["stream_power"],
            "strahler_order": source["strahler_order"],
            "channel_width_m": source["channel_width_m"],
            "channel_depth_m": source["channel_depth_m"],
            "valley_width_m": source["valley_width_m"],
            "floodplain_width_m": source["floodplain_width_m"],
            "incision_m": source["incision_m"],
            "sediment_load": source["sediment_load"],
            "meander_index": source["meander_index"],
            "braiding_index": source["braiding_index"],
            "morphology_class": source["morphology_class"],
            "bed_material": source["bed_material"],
        }
    )
    path_by_reach = dict(
        zip(
            np.asarray(table["reach_id"], dtype=np.int32),
            (set(path) for path in fine_paths),
            strict=True,
        )
    )
    junctions_valid = True
    downstream_joins: list[int] = []
    for downstream, exit_cell in zip(
        np.asarray(table["downstream_reach_id"], dtype=np.int32),
        np.asarray(table["exit_fine_cell"], dtype=np.int32),
        strict=True,
    ):
        if downstream >= 0 and int(downstream) in path_by_reach:
            merges_downstream = int(exit_cell) in path_by_reach[int(downstream)]
            junctions_valid &= merges_downstream
            downstream_joins.append(int(exit_cell) if merges_downstream else -1)
        else:
            downstream_joins.append(-1)
    table = table.append_column(
        "downstream_join_fine_cell", pa.array(downstream_joins, type=pa.int32())
    )
    return table, junctions_valid, maximum_length_error


def _collapse_adjacent(values: np.ndarray) -> list[int]:
    return [
        int(value) for index, value in enumerate(values) if index == 0 or value != values[index - 1]
    ]


def _directed_path_graph_metadata(reaches: pa.Table) -> dict[str, int]:
    edges: set[tuple[int, int]] = set()
    for path in reaches["fine_cell_path"].to_pylist():
        edges.update((int(source), int(target)) for source, target in zip(path[:-1], path[1:]))

    reverse_conflicts = {
        (min(source, target), max(source, target))
        for source, target in edges
        if (target, source) in edges
    }
    adjacency: dict[int, set[int]] = defaultdict(set)
    indegree: dict[int, int] = {}
    for source, target in edges:
        adjacency[source].add(target)
        indegree.setdefault(source, 0)
        indegree[target] = indegree.get(target, 0) + 1
    ready = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        source = ready.popleft()
        visited += 1
        for target in adjacency.get(source, ()):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    dag_valid = visited == len(indegree)
    return {
        "directed_edge_count": len(edges),
        "reverse_directed_edge_conflict_count": len(reverse_conflicts),
        "directed_path_dag_valid": int(dag_valid),
        "directed_path_graph_valid": int(dag_valid and not reverse_conflicts),
    }


def _classify_reach_terminals(
    reaches: pa.Table,
    basin_ids: np.ndarray,
    flow_receiver: np.ndarray,
    flow_sink_type: np.ndarray,
) -> tuple[pa.Table, dict[str, int]]:
    reach_ids = set(int(value) for value in reaches["reach_id"].to_pylist())
    kinds: list[str] = []
    terminal_cells: list[int] = []
    receiver_cells: list[int] = []
    sink_types: list[int] = []
    resolved: list[bool] = []
    counts: dict[str, int] = defaultdict(int)
    for downstream, to_node in zip(
        np.asarray(reaches["downstream_reach_id"], dtype=np.int32),
        np.asarray(reaches["to_parent_cell"], dtype=np.int32),
        strict=True,
    ):
        if int(downstream) >= 0:
            kind = (
                "downstream_reach"
                if int(downstream) in reach_ids
                else "unresolved_downstream_reference"
            )
            terminal_cell = -1
            receiver = -1
            sink_type = 0
            is_resolved = kind == "downstream_reach"
        else:
            terminal_cell = int(to_node)
            if not 0 <= terminal_cell < len(basin_ids):
                receiver = -1
                sink_type = 0
                kind = "unresolved_terminal"
            else:
                receiver = int(flow_receiver[terminal_cell])
                sink_type = int(flow_sink_type[terminal_cell])
                receiver_is_ocean = 0 <= receiver < len(basin_ids) and int(basin_ids[receiver]) < 0
                if int(basin_ids[terminal_cell]) < 0 or receiver_is_ocean:
                    kind = "ocean"
                elif receiver < 0 and sink_type in (2, 3, 4):
                    kind = "registered_sink"
                elif receiver >= 0:
                    kind = "unresolved_threshold_gap"
                else:
                    kind = "unresolved_terminal"
            is_resolved = kind in {"ocean", "registered_sink"}
        kinds.append(kind)
        terminal_cells.append(terminal_cell)
        receiver_cells.append(receiver)
        sink_types.append(sink_type)
        resolved.append(is_resolved)
        counts[kind] += 1

    reaches = reaches.append_column("terminal_kind", pa.array(kinds, type=pa.string()))
    reaches = reaches.append_column(
        "terminal_parent_cell", pa.array(terminal_cells, type=pa.int32())
    )
    reaches = reaches.append_column(
        "terminal_receiver_cell", pa.array(receiver_cells, type=pa.int32())
    )
    reaches = reaches.append_column("terminal_sink_type", pa.array(sink_types, type=pa.uint8()))
    reaches = reaches.append_column("terminal_resolved", pa.array(resolved, type=pa.bool_()))
    unresolved = sum(count for kind, count in counts.items() if kind.startswith("unresolved_"))
    metadata = {
        "terminal_reach_count": sum(
            count for kind, count in counts.items() if kind != "downstream_reach"
        ),
        "terminal_ocean_count": counts["ocean"],
        "terminal_registered_sink_count": counts["registered_sink"],
        "terminal_unresolved_threshold_gap_count": counts["unresolved_threshold_gap"],
        "terminal_unresolved_downstream_reference_count": counts["unresolved_downstream_reference"],
        "terminal_unresolved_other_count": counts["unresolved_terminal"],
        "source_to_sink_ready": int(unresolved == 0),
    }
    return reaches, metadata


def _corridor_area_metadata(
    reaches: pa.Table, metadata: Mapping[str, int | float]
) -> dict[str, float]:
    path_length_m = np.asarray(reaches["path_length_km"], dtype=np.float64) * 1_000.0
    result: dict[str, float] = {}
    for corridor in ("channel", "valley", "floodplain"):
        requested = float(
            np.sum(path_length_m * np.asarray(reaches[f"{corridor}_width_m"], dtype=np.float64))
            / 1_000_000.0
        )
        represented = float(metadata[f"represented_{corridor}_area_km2"])
        result[f"requested_{corridor}_area_km2"] = requested
        result[f"{corridor}_area_retention_fraction"] = represented / max(requested, 1e-12)
    return result


def _membership_table(records: np.ndarray) -> pa.Table:
    return pa.table(
        {
            "reach_id": pa.array(records["reach_id"], type=pa.int32()),
            "fine_cell_id": pa.array(records["fine_cell_id"], type=pa.int32()),
            "parent_cell_id": pa.array(records["parent_cell_id"], type=pa.int32()),
            "path_order": pa.array(records["path_order"], type=pa.int32()),
            "reach_length_m": pa.array(records["reach_length_m"], type=pa.float64()),
            "channel_fraction": pa.array(records["channel_fraction"], type=pa.float32()),
            "valley_fraction": pa.array(records["valley_fraction"], type=pa.float32()),
            "floodplain_fraction": pa.array(records["floodplain_fraction"], type=pa.float32()),
            "potential_incised_volume_m3": pa.array(
                records["potential_incised_volume_m3"], type=pa.float64()
            ),
        }
    )


def _cube_net_visualizer(result, request: VisualizationRequest) -> VisualizationResult | None:
    cell_record = result.artifact_records.get("RefinedBasinCellCatalog")
    reach_record = result.artifact_records.get("RefinedRiverReachCatalog")
    metadata_record = result.artifact_records.get("BasinRefinementMetadata")
    if (
        cell_record is None
        or reach_record is None
        or metadata_record is None
        or not isinstance(cell_record.value, pa.Table)
        or not isinstance(reach_record.value, pa.Table)
    ):
        return None
    cells = cell_record.value
    reaches = reach_record.value
    metadata = metadata_record.value
    fine_resolution = int(metadata["fine_resolution"])
    display_resolution = min(fine_resolution, 768)
    image = np.full((display_resolution * 3, display_resolution * 4, 3), 18, dtype=np.uint8)
    placements = np.array([[1, 1], [1, 3], [1, 2], [1, 0], [0, 1], [2, 1]])
    face = np.asarray(cells["face"], dtype=np.int32)
    row = np.asarray(cells["row"], dtype=np.int32)
    col = np.asarray(cells["col"], dtype=np.int32)
    elevation = np.asarray(cells["terrain_elevation_m"], dtype=np.float32)
    low, high = np.percentile(elevation, [2.0, 98.0])
    normalized = np.clip((elevation - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    colors = np.stack(
        (55 + 150 * normalized, 85 + 125 * normalized, 55 + 95 * normalized), axis=1
    ).astype(np.uint8)
    display_row = np.minimum(row * display_resolution // fine_resolution, display_resolution - 1)
    display_col = np.minimum(col * display_resolution // fine_resolution, display_resolution - 1)
    net_row = placements[face, 0] * display_resolution + display_row
    net_col = placements[face, 1] * display_resolution + display_col
    image[net_row, net_col] = colors

    rendered = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(rendered)
    fine_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    order = np.argsort(fine_ids)
    sorted_ids = fine_ids[order]
    for path in reaches["fine_cell_path"].to_pylist():
        path_array = np.asarray(path, dtype=np.int32)
        rows = _lookup_rows(sorted_ids, order, path_array)
        path_face = face[rows]
        path_row = placements[path_face, 0] * display_resolution + np.minimum(
            row[rows] * display_resolution // fine_resolution, display_resolution - 1
        )
        path_col = placements[path_face, 1] * display_resolution + np.minimum(
            col[rows] * display_resolution // fine_resolution, display_resolution - 1
        )
        segment_start = 0
        for index in range(1, len(rows) + 1):
            if index == len(rows) or path_face[index] != path_face[index - 1]:
                if index - segment_start >= 2:
                    draw.line(
                        [
                            (int(path_col[position]), int(path_row[position]))
                            for position in range(segment_start, index)
                        ],
                        fill=(29, 174, 235),
                        width=2,
                    )
                segment_start = index
    output = request.output_dir / "refined_basin.png"
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
        "RefinedBasinCellCatalog",
        {"basin_id": metadata["selected_basin_id"], "fine_resolution": fine_resolution},
    )


@stage(
    "basin_refinement",
    inputs=("hydrology", "elevation", "planet"),
    outputs=(
        "RefinedBasinCellCatalog",
        "RefinedBasinParentCatalog",
        "RefinedRiverReachCatalog",
        "RefinedReachCellCatalog",
        "BasinRefinementMetadata",
    ),
    version="v4",
    native_libraries=("refinement_native",),
    visualizer=_cube_net_visualizer,
)
def basin_refinement_stage(context, deps, config_mapping: Mapping[str, object]):
    config = BasinRefinementConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("basin refinement requires topology: cubed_sphere")
    coarse_resolution = context.topology.face_resolution
    fine_resolution = coarse_resolution * config.refinement_factor
    if 6 * fine_resolution * fine_resolution > np.iinfo(np.int32).max:
        raise ValueError("refinement_factor exceeds the global fine-cell ID capacity")

    hydrology = deps["hydrology"]
    basin_catalog = _artifact_table(hydrology, "BasinCatalog")
    source_reaches = _artifact_table(hydrology, "RiverReachCatalog")
    selected_basin_id = _select_basin(basin_catalog, source_reaches, config.basin_id)
    selected_reaches = _downstream_first_reaches(
        source_reaches.filter(pc.equal(source_reaches["basin_id"], selected_basin_id))
    )
    if selected_reaches.num_rows == 0:
        raise ValueError(f"selected basin {selected_basin_id} has no river reaches")

    paths = [[int(cell) for cell in path] for path in selected_reaches["cell_path"].to_pylist()]
    basin_ids = _artifact_array(hydrology, "BasinID").reshape(-1)
    flow_receiver = _artifact_array(hydrology, "FlowReceiverID").reshape(-1)
    flow_sink_type = _artifact_array(hydrology, "FlowSinkType").reshape(-1)
    basin_parent_ids = np.flatnonzero(basin_ids == selected_basin_id).astype(np.int32)
    inherited_path_ids = np.fromiter(
        (cell for path in paths for cell in path), dtype=np.int32, count=sum(map(len, paths))
    )
    parent_ids = np.ascontiguousarray(
        np.union1d(basin_parent_ids, inherited_path_ids), dtype=np.int32
    )
    inside_selected_basin = basin_ids[parent_ids] == selected_basin_id
    bedrock = _artifact_array(deps["elevation"], "BedrockElevationM").reshape(-1)
    breach = _artifact_array(hydrology, "BreachIncisionM").reshape(-1)
    relief = _artifact_array(deps["elevation"], "TerrainReliefM").reshape(-1)
    parent_elevation = np.ascontiguousarray(
        bedrock[parent_ids] - breach[parent_ids], dtype=np.float32
    )
    parent_relief = np.ascontiguousarray(relief[parent_ids], dtype=np.float32)
    parent_area_steradians = np.ascontiguousarray(
        context.topology.cell_areas.reshape(-1)[parent_ids], dtype=np.float64
    )

    reach_offsets, reach_parent_cells = _flatten_paths(paths)
    planet = deps["planet"].artifact_records["PlanetMetadata"].value
    radius_m = float(planet["planet_radius_earth"]) * EARTH_RADIUS_M
    rng = context.rng("basin_refinement")
    terrain_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    controls = {
        "coarse_resolution": coarse_resolution,
        "factor": config.refinement_factor,
        "planet_radius_m": radius_m,
        "terrain_seed": terrain_seed,
        "terrain_noise_fraction": config.terrain_noise_fraction,
    }
    with context.timed("sparse_basin_refinement_kernel"):
        cell_records, reach_records, path_cells, memberships, metadata = run_basin_refinement(
            controls=controls,
            parent_ids=parent_ids,
            parent_elevation_m=parent_elevation,
            parent_relief_m=parent_relief,
            parent_area_steradians=parent_area_steradians,
            reach_ids=_column_numpy(selected_reaches, "reach_id", np.dtype(np.int32)),
            reach_from_nodes=_column_numpy(selected_reaches, "from_node", np.dtype(np.int32)),
            reach_to_nodes=_column_numpy(selected_reaches, "to_node", np.dtype(np.int32)),
            reach_offsets=reach_offsets,
            reach_parent_cells=reach_parent_cells,
            channel_width_m=_column_numpy(
                selected_reaches, "channel_width_m", np.dtype(np.float32)
            ),
            valley_width_m=_column_numpy(selected_reaches, "valley_width_m", np.dtype(np.float32)),
            floodplain_width_m=_column_numpy(
                selected_reaches, "floodplain_width_m", np.dtype(np.float32)
            ),
            incision_m=_column_numpy(selected_reaches, "incision_m", np.dtype(np.float32)),
        )

    radius_km = radius_m / 1_000.0
    parent_areas_km2 = parent_area_steradians * radius_km * radius_km
    cells = _cell_table(cell_records)
    parents = _parent_table(
        cell_records,
        parent_ids,
        inside_selected_basin,
        parent_elevation,
        parent_relief,
        parent_areas_km2,
        config.refinement_factor,
    )
    reaches, junctions_valid, maximum_length_error = _reach_table(
        selected_reaches,
        reach_records,
        path_cells,
        cell_records,
        fine_resolution=fine_resolution,
        factor=config.refinement_factor,
        planet_radius_m=radius_m,
    )
    reaches, terminal_metadata = _classify_reach_terminals(
        reaches, basin_ids, flow_receiver, flow_sink_type
    )
    reach_cells = _membership_table(memberships)
    path_graph_metadata = _directed_path_graph_metadata(reaches)
    corridor_area_metadata = _corridor_area_metadata(reaches, metadata)
    metadata.update(
        {
            **asdict(config),
            **terminal_metadata,
            **path_graph_metadata,
            **corridor_area_metadata,
            "selected_basin_id": selected_basin_id,
            "selected_basin_parent_count": int(np.count_nonzero(inside_selected_basin)),
            "boundary_parent_count": int(np.count_nonzero(~inside_selected_basin)),
            "selected_basin_area_km2": float(np.sum(parent_areas_km2[inside_selected_basin])),
            "coarse_resolution": coarse_resolution,
            "terrain_seed": terrain_seed,
            "junction_merge_valid": int(junctions_valid),
            "maximum_reach_length_conservation_error_m": maximum_length_error,
            "inherited_discharge_relative_error": 0.0,
            "topology": "sparse_selected_basin_on_cubed_sphere",
            "terrain_semantics": "parent_mean_conserving_unresolved_relief_realization",
            "river_semantics": "subgrid_physical_width_vector_reaches",
            "corridor_area_semantics": (
                "represented_centerline_cell_support_with_requested_unclipped_rectangles"
            ),
            "incision_semantics": "potential_reach_volume_not_cell_wide_elevation_change",
        }
    )
    if metadata["path_topology_valid"] != 1:
        raise RuntimeError("native refinement published an invalid fine reach path")
    if metadata["maximum_parent_area_relative_error"] > 1e-9:
        raise RuntimeError("refined child areas do not conserve their parent areas")
    if metadata["maximum_parent_elevation_error_m"] > 1e-3:
        raise RuntimeError("refined terrain does not conserve parent mean elevation")
    if maximum_length_error > 1e-4:
        raise RuntimeError("refined path length does not match its spherical geometry")
    if not junctions_valid:
        raise RuntimeError("refined upstream reaches do not merge into their downstream paths")
    if metadata["directed_path_graph_valid"] != 1:
        raise RuntimeError(
            "refined reach union is not a directed acyclic drainage graph: "
            f"reverse_conflicts={metadata['reverse_directed_edge_conflict_count']}, "
            f"dag_valid={metadata['directed_path_dag_valid']}"
        )
    context.logger.log_event(
        {"type": "basin_refinement_summary", "stage": "basin_refinement", **metadata}
    )
    return {
        "RefinedBasinCellCatalog": cells,
        "RefinedBasinParentCatalog": parents,
        "RefinedRiverReachCatalog": reaches,
        "RefinedReachCellCatalog": reach_cells,
        "BasinRefinementMetadata": metadata,
    }


__all__ = ["BasinRefinementConfig", "basin_refinement_stage"]
