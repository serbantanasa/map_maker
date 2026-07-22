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
from .hydrology import WATER_DOMINATED_CELL_FRACTION

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class BasinRefinementConfig:
    refinement_factor: int = 16
    basin_id: int | None = None
    terrain_noise_fraction: float = 0.45
    halo_parent_rings: int = 0

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
        halo_parent_rings = int(mapping.get("halo_parent_rings", cls.halo_parent_rings))
        if factor < 2 or factor > 32 or factor & (factor - 1):
            raise ValueError("refinement_factor must be a power of two in [2, 32]")
        if basin_id is not None and basin_id < 0:
            raise ValueError("basin_id must be non-negative when provided")
        if not np.isfinite(terrain_noise_fraction) or not 0.0 <= terrain_noise_fraction <= 1.0:
            raise ValueError("terrain_noise_fraction must be finite and in [0, 1]")
        if not 0 <= halo_parent_rings <= 8:
            raise ValueError("halo_parent_rings must be in [0, 8]")
        return cls(
            refinement_factor=factor,
            basin_id=basin_id,
            terrain_noise_fraction=terrain_noise_fraction,
            halo_parent_rings=halo_parent_rings,
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


def _expand_parent_halo(
    parent_ids: np.ndarray,
    neighbors: np.ndarray,
    rings: int,
) -> np.ndarray:
    selected = np.zeros(len(neighbors), dtype=bool)
    selected[np.asarray(parent_ids, dtype=np.int32)] = True
    frontier = np.asarray(parent_ids, dtype=np.int32)
    for _ in range(rings):
        candidates = np.unique(neighbors[frontier].reshape(-1))
        candidates = candidates[candidates >= 0]
        candidates = candidates[~selected[candidates]]
        if candidates.size == 0:
            break
        selected[candidates] = True
        frontier = candidates.astype(np.int32, copy=False)
    return np.flatnonzero(selected).astype(np.int32)


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


def _parent_channel_surface_prior(
    hydrology,
    parent_ids: np.ndarray,
    parent_elevation_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prior = np.asarray(parent_elevation_m, dtype=np.float32).copy()
    fill_depth = _artifact_array(hydrology, "DepressionFillDepthM").reshape(-1)[parent_ids]
    filled = fill_depth > 0.0
    prior[filled] += fill_depth[filled]
    depression_ids = _artifact_array(hydrology, "DepressionID").reshape(-1)[parent_ids]
    catalog = _artifact_table(hydrology, "DepressionCatalog")
    catalog_ids = np.asarray(catalog["depression_id"], dtype=np.int32)
    catalog_surfaces = np.asarray(catalog["surface_elevation_m"], dtype=np.float32)
    registered = np.asarray(catalog["water_area_km2"], dtype=np.float64) > 0.0
    surface_by_id = {
        int(depression_id): float(surface)
        for depression_id, surface, keep in zip(
            catalog_ids, catalog_surfaces, registered, strict=True
        )
        if keep
    }
    registered_control = np.asarray(
        [int(depression_id) in surface_by_id for depression_id in depression_ids],
        dtype=bool,
    )
    registered_rows = np.flatnonzero(registered_control)
    try:
        prior[registered_rows] = np.asarray(
            [surface_by_id[int(depression_ids[row])] for row in registered_rows],
            dtype=np.float32,
        )
    except KeyError as error:
        raise RuntimeError(
            "fractional standing water has no registered hydraulic surface"
        ) from error
    if np.any(~np.isfinite(prior)):
        raise RuntimeError("channel surface prior contains a non-finite elevation")
    return prior, filled | registered_control


def _extend_hydraulic_surface_over_submerged_approaches(
    fine_cell_ids: np.ndarray,
    surface_prior_m: np.ndarray,
    hydraulic_surface_controlled: np.ndarray,
    reaches: pa.Table,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Carry a receiving water surface over unresolved submerged approach terrain."""

    prior = np.asarray(surface_prior_m, dtype=np.float32).copy()
    controlled = np.asarray(hydraulic_surface_controlled, dtype=bool).copy()
    initially_controlled = controlled.copy()
    fine_ids = np.asarray(fine_cell_ids, dtype=np.int32)
    order = np.argsort(fine_ids)
    sorted_ids = fine_ids[order]
    channel_paths = [
        np.asarray(path, dtype=np.int32)
        for path, kind in zip(
            reaches["fine_cell_path"].to_pylist(),
            reaches["reach_kind"].to_pylist(),
            strict=True,
        )
        if kind == "channel"
    ]

    # Revisit the paths because a downstream reach may expose a hydraulic
    # control at a shared junction after its upstream reach was inspected.
    for _ in range(len(channel_paths) + 1):
        changed = False
        for path in channel_paths:
            rows = _lookup_rows(sorted_ids, order, path)
            receiving_surface: float | None = None
            for row in rows[::-1]:
                if controlled[row]:
                    surface = float(prior[row])
                    receiving_surface = (
                        surface if receiving_surface is None else max(receiving_surface, surface)
                    )
                    continue
                if receiving_surface is None:
                    continue
                if float(prior[row]) <= receiving_surface:
                    prior[row] = receiving_surface
                    controlled[row] = True
                    changed = True
                    continue
                # This is the first emerged point upstream of the water body.
                receiving_surface = None
        if not changed:
            break
    else:
        raise RuntimeError("submerged channel-approach conditioning did not converge")

    approach_count = int(np.count_nonzero(controlled & ~initially_controlled))
    return prior, controlled, approach_count


def _parent_table(
    records: np.ndarray,
    parent_ids: np.ndarray,
    inside_selected_basin: np.ndarray,
    parent_elevation_m: np.ndarray,
    parent_relief_m: np.ndarray,
    parent_areas_km2: np.ndarray,
    parent_process_excluded: np.ndarray,
    parent_standing_water_fraction: np.ndarray,
    parent_channel_surface_prior_m: np.ndarray,
    parent_hydraulic_surface_controlled: np.ndarray,
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
            "process_excluded": pa.array(parent_process_excluded != 0, type=pa.bool_()),
            "standing_water_fraction": pa.array(parent_standing_water_fraction, type=pa.float32()),
            "channel_surface_prior_m": pa.array(parent_channel_surface_prior_m, type=pa.float32()),
            "hydraulic_surface_controlled": pa.array(
                parent_hydraulic_surface_controlled, type=pa.bool_()
            ),
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
            "reach_kind": source["reach_kind"],
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
    edge_multiplicity: dict[tuple[int, int], int] = defaultdict(int)
    physical_edge_multiplicity: dict[tuple[int, int], int] = defaultdict(int)
    reach_edge_count = 0
    for path, reach_kind in zip(
        reaches["fine_cell_path"].to_pylist(),
        reaches["reach_kind"].to_pylist(),
        strict=True,
    ):
        for source, target in zip(path[:-1], path[1:]):
            edge = (int(source), int(target))
            edges.add(edge)
            edge_multiplicity[edge] += 1
            reach_edge_count += 1
            if reach_kind == "channel":
                physical_edge_multiplicity[edge] += 1

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
        "reach_directed_edge_count": reach_edge_count,
        "shared_directed_edge_count": sum(count > 1 for count in edge_multiplicity.values()),
        "shared_physical_directed_edge_count": sum(
            count > 1 for count in physical_edge_multiplicity.values()
        ),
        "maximum_directed_edge_multiplicity": max(edge_multiplicity.values(), default=0),
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
    path_length_m = np.asarray(reaches["physical_channel_length_km"], dtype=np.float64) * 1_000.0
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
            "support_role": pa.array(
                np.where(records["reach_length_m"] > 0.0, "centerline", "lateral"),
                type=pa.string(),
            ),
            "channel_fraction": pa.array(records["channel_fraction"], type=pa.float32()),
            "valley_fraction": pa.array(records["valley_fraction"], type=pa.float32()),
            "floodplain_fraction": pa.array(records["floodplain_fraction"], type=pa.float32()),
            "potential_incised_volume_m3": pa.array(
                records["potential_incised_volume_m3"], type=pa.float64()
            ),
        }
    )


def _append_physical_reach_lengths(reaches: pa.Table, memberships: pa.Table) -> pa.Table:
    reach_ids = np.asarray(reaches["reach_id"], dtype=np.int32)
    row_by_reach = {int(reach_id): row for row, reach_id in enumerate(reach_ids)}
    physical_length_m = np.zeros(reaches.num_rows, dtype=np.float64)
    for reach_id, length_m in zip(
        np.asarray(memberships["reach_id"], dtype=np.int32),
        np.asarray(memberships["reach_length_m"], dtype=np.float64),
        strict=True,
    ):
        physical_length_m[row_by_reach[int(reach_id)]] += float(length_m)
    return reaches.append_column(
        "physical_channel_length_km", pa.array(physical_length_m / 1_000.0, type=pa.float64())
    )


def _corridor_capacity_metadata(memberships: pa.Table) -> dict[str, int | float]:
    fine_cell_ids = np.asarray(memberships["fine_cell_id"], dtype=np.int32)
    _, inverse = np.unique(fine_cell_ids, return_inverse=True)
    metadata: dict[str, int | float] = {
        "centerline_membership_count": int(
            pc.sum(pc.equal(memberships["support_role"], "centerline")).as_py() or 0
        ),
        "lateral_support_membership_count": int(
            pc.sum(pc.equal(memberships["support_role"], "lateral")).as_py() or 0
        ),
    }
    capacity_valid = True
    for corridor in ("channel", "valley", "floodplain"):
        fraction = np.asarray(memberships[f"{corridor}_fraction"], dtype=np.float64)
        summed = np.bincount(inverse, weights=fraction)
        maximum = float(np.max(summed, initial=0.0))
        metadata[f"maximum_cell_{corridor}_fraction_sum"] = maximum
        capacity_valid &= maximum <= 1.0 + 1e-6
    channel = np.asarray(memberships["channel_fraction"], dtype=np.float64)
    valley = np.asarray(memberships["valley_fraction"], dtype=np.float64)
    floodplain = np.asarray(memberships["floodplain_fraction"], dtype=np.float64)
    nested_valid = bool(
        np.all(channel <= floodplain + 1e-7) and np.all(floodplain <= valley + 1e-7)
    )
    metadata["nested_corridor_support_valid"] = int(nested_valid)
    metadata["corridor_cell_capacity_valid"] = int(capacity_valid)
    return metadata


def _cube_net_visualizer(result, request: VisualizationRequest) -> VisualizationResult | None:
    cell_record = result.artifact_records.get("RefinedBasinCellCatalog")
    reach_record = result.artifact_records.get("RefinedRiverReachCatalog")
    membership_record = result.artifact_records.get("RefinedReachCellCatalog")
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

    if membership_record is not None and isinstance(membership_record.value, pa.Table):
        memberships = membership_record.value
        fine_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
        order = np.argsort(fine_ids)
        sorted_ids = fine_ids[order]
        membership_ids = np.asarray(memberships["fine_cell_id"], dtype=np.int32)
        support_rows = _lookup_rows(sorted_ids, order, membership_ids)
        valley = np.asarray(memberships["valley_fraction"], dtype=np.float32)
        floodplain = np.asarray(memberships["floodplain_fraction"], dtype=np.float32)
        support_strength: dict[tuple[int, int], tuple[float, float]] = {}
        for row_index, valley_fraction, floodplain_fraction in zip(
            support_rows, valley, floodplain, strict=True
        ):
            support_row = int(
                placements[face[row_index], 0] * display_resolution
                + min(
                    row[row_index] * display_resolution // fine_resolution, display_resolution - 1
                )
            )
            support_col = int(
                placements[face[row_index], 1] * display_resolution
                + min(
                    col[row_index] * display_resolution // fine_resolution, display_resolution - 1
                )
            )
            previous = support_strength.get((support_row, support_col), (0.0, 0.0))
            support_strength[(support_row, support_col)] = (
                max(previous[0], float(valley_fraction)),
                max(previous[1], float(floodplain_fraction)),
            )
        for (support_row, support_col), (
            valley_fraction,
            floodplain_fraction,
        ) in support_strength.items():
            target = np.array(
                (63, 137, 91) if floodplain_fraction > 0.0 else (104, 132, 73),
                dtype=np.float32,
            )
            alpha = min(0.50, 0.16 + 0.34 * max(valley_fraction, floodplain_fraction))
            image[support_row, support_col] = (
                image[support_row, support_col].astype(np.float32) * (1.0 - alpha) + target * alpha
            ).astype(np.uint8)

    rendered = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(rendered)
    fine_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    order = np.argsort(fine_ids)
    sorted_ids = fine_ids[order]
    reach_kinds = reaches["reach_kind"].to_pylist()
    for path, reach_kind in zip(reaches["fine_cell_path"].to_pylist(), reach_kinds, strict=True):
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
                        fill=(29, 174, 235) if reach_kind == "channel" else (73, 122, 143),
                        width=2 if reach_kind == "channel" else 1,
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
    version="v12",
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
    selected_connectors = np.asarray(pc.equal(selected_reaches["reach_kind"], "connector"))
    for field in (
        "channel_width_m",
        "channel_depth_m",
        "valley_width_m",
        "floodplain_width_m",
        "velocity_mean",
        "stream_power",
        "incision_m",
    ):
        values = np.asarray(selected_reaches[field], dtype=np.float32)
        if np.any(values[selected_connectors] != 0.0):
            raise RuntimeError(f"hydrologic connectors must publish zero {field}")

    paths = [[int(cell) for cell in path] for path in selected_reaches["cell_path"].to_pylist()]
    basin_ids = _artifact_array(hydrology, "BasinID").reshape(-1)
    flow_receiver = _artifact_array(hydrology, "FlowReceiverID").reshape(-1)
    flow_sink_type = _artifact_array(hydrology, "FlowSinkType").reshape(-1)
    basin_parent_ids = np.flatnonzero(basin_ids == selected_basin_id).astype(np.int32)
    inherited_path_ids = np.fromiter(
        (cell for path in paths for cell in path), dtype=np.int32, count=sum(map(len, paths))
    )
    core_parent_ids = np.ascontiguousarray(
        np.union1d(basin_parent_ids, inherited_path_ids), dtype=np.int32
    )
    parent_ids = _expand_parent_halo(
        core_parent_ids,
        context.topology.neighbor_indices.reshape(-1, 4),
        config.halo_parent_rings,
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
    standing_water_fraction = _artifact_array(hydrology, "LakeFraction").reshape(
        -1
    ) + _artifact_array(hydrology, "WetlandFraction").reshape(-1)
    parent_process_excluded = np.ascontiguousarray(
        standing_water_fraction[parent_ids] >= WATER_DOMINATED_CELL_FRACTION,
        dtype=np.uint8,
    )
    parent_standing_water_fraction = np.ascontiguousarray(
        standing_water_fraction[parent_ids], dtype=np.float32
    )
    parent_channel_surface_prior, parent_hydraulic_surface_controlled = (
        _parent_channel_surface_prior(
            hydrology,
            parent_ids,
            parent_elevation,
        )
    )

    reach_offsets, reach_parent_cells = _flatten_paths(paths)
    river_corridor = _artifact_array(hydrology, "RiverCorridor").reshape(-1)
    reach_parent_channel_support = np.ascontiguousarray(
        river_corridor[reach_parent_cells] > 0.0, dtype=np.uint8
    )
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
            parent_process_excluded=parent_process_excluded,
            reach_ids=_column_numpy(selected_reaches, "reach_id", np.dtype(np.int32)),
            reach_from_nodes=_column_numpy(selected_reaches, "from_node", np.dtype(np.int32)),
            reach_to_nodes=_column_numpy(selected_reaches, "to_node", np.dtype(np.int32)),
            reach_offsets=reach_offsets,
            reach_parent_cells=reach_parent_cells,
            reach_parent_channel_support=reach_parent_channel_support,
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
    reaches, junctions_valid, maximum_length_error = _reach_table(
        selected_reaches,
        reach_records,
        path_cells,
        cell_records,
        fine_resolution=fine_resolution,
        factor=config.refinement_factor,
        planet_radius_m=radius_m,
    )
    children_per_parent = config.refinement_factor**2
    child_standing_water_fraction = np.repeat(parent_standing_water_fraction, children_per_parent)
    child_channel_surface_prior = np.asarray(cells["terrain_elevation_m"], dtype=np.float32).copy()
    controlled_children = np.repeat(parent_hydraulic_surface_controlled, children_per_parent)
    child_channel_surface_prior[controlled_children] = np.repeat(
        parent_channel_surface_prior, children_per_parent
    )[controlled_children]
    (
        child_channel_surface_prior,
        controlled_children,
        hydraulic_approach_child_count,
    ) = _extend_hydraulic_surface_over_submerged_approaches(
        np.asarray(cells["fine_cell_id"], dtype=np.int32),
        child_channel_surface_prior,
        controlled_children,
        reaches,
    )
    cells = cells.append_column(
        "standing_water_fraction",
        pa.array(child_standing_water_fraction, type=pa.float32()),
    )
    cells = cells.append_column(
        "channel_surface_prior_m",
        pa.array(child_channel_surface_prior, type=pa.float32()),
    )
    cells = cells.append_column(
        "hydraulic_surface_controlled",
        pa.array(controlled_children, type=pa.bool_()),
    )
    cells = cells.append_column(
        "process_excluded",
        pa.array(
            np.repeat(parent_process_excluded != 0, config.refinement_factor**2),
            type=pa.bool_(),
        ),
    )
    parents = _parent_table(
        cell_records,
        parent_ids,
        inside_selected_basin,
        parent_elevation,
        parent_relief,
        parent_areas_km2,
        parent_process_excluded,
        parent_standing_water_fraction,
        parent_channel_surface_prior,
        parent_hydraulic_surface_controlled,
        config.refinement_factor,
    )
    reaches, terminal_metadata = _classify_reach_terminals(
        reaches, basin_ids, flow_receiver, flow_sink_type
    )
    reach_cells = _membership_table(memberships)
    reaches = _append_physical_reach_lengths(reaches, reach_cells)
    process_excluded_parent_ids = parent_ids[parent_process_excluded != 0]
    process_exclusion_valid = not np.any(
        np.isin(
            np.asarray(reach_cells["parent_cell_id"], dtype=np.int32),
            process_excluded_parent_ids,
        )
    )
    path_graph_metadata = _directed_path_graph_metadata(reaches)
    corridor_area_metadata = _corridor_area_metadata(reaches, metadata)
    corridor_capacity_metadata = _corridor_capacity_metadata(reach_cells)
    metadata.update(
        {
            **asdict(config),
            **terminal_metadata,
            **path_graph_metadata,
            **corridor_area_metadata,
            **corridor_capacity_metadata,
            "selected_basin_id": selected_basin_id,
            "selected_basin_parent_count": int(np.count_nonzero(inside_selected_basin)),
            "boundary_parent_count": int(np.count_nonzero(~inside_selected_basin)),
            "halo_parent_count": int(len(parent_ids) - len(core_parent_ids)),
            "process_excluded_parent_count": int(np.count_nonzero(parent_process_excluded)),
            "fractional_water_channel_prior_parent_count": int(
                np.count_nonzero(parent_standing_water_fraction > 0.0)
            ),
            "hydraulic_surface_prior_parent_count": int(
                np.count_nonzero(parent_hydraulic_surface_controlled)
            ),
            "hydraulic_backwater_approach_child_count": hydraulic_approach_child_count,
            "process_exclusion_valid": int(process_exclusion_valid),
            "channel_reach_count": int(
                selected_reaches.num_rows - np.count_nonzero(selected_connectors)
            ),
            "connector_reach_count": int(np.count_nonzero(selected_connectors)),
            "selected_basin_area_km2": float(np.sum(parent_areas_km2[inside_selected_basin])),
            "total_physical_channel_length_km": float(
                pc.sum(reaches["physical_channel_length_km"]).as_py() or 0.0
            ),
            "coarse_resolution": coarse_resolution,
            "terrain_seed": terrain_seed,
            "junction_merge_valid": int(junctions_valid),
            "maximum_reach_length_conservation_error_m": maximum_length_error,
            "inherited_discharge_relative_error": 0.0,
            "topology": "sparse_selected_basin_on_cubed_sphere",
            "terrain_semantics": "parent_mean_conserving_unresolved_relief_realization",
            "channel_surface_prior_semantics": (
                "priority_flood_control_surface_over_unresolved_fill_with_registered_"
                "water_surface_override_and_submerged_backwater_approach_else_refined_terrain"
            ),
            "river_semantics": "physical_channel_reaches_with_zero_width_hydrologic_connectors",
            "corridor_area_semantics": (
                "conservative_nested_lateral_support_with_per_cell_capacity"
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
    if metadata["source_to_sink_ready"] != 1:
        raise RuntimeError("refined reach graph contains an unresolved source-to-sink terminal")
    if metadata["corridor_cell_capacity_valid"] != 1:
        raise RuntimeError("refined corridor support exceeds fine-cell physical capacity")
    if metadata["nested_corridor_support_valid"] != 1:
        raise RuntimeError("refined channel, floodplain, and valley support is not nested")
    if not process_exclusion_valid:
        raise RuntimeError("refined physical reach support enters a process-excluded parent")
    for corridor in ("channel", "valley", "floodplain"):
        if metadata[f"{corridor}_area_retention_fraction"] < 1.0 - 1e-6:
            raise RuntimeError(f"refined {corridor} support does not conserve requested area")
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
