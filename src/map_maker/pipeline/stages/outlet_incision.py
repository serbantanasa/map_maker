"""Bounded subgrid outlet incision and local hydrology rerouting."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pyarrow as pa
from PIL import Image

from .._hydrology_pass2_native import run_hydrology_pass2
from ..cubed_sphere import CubedSphereGrid
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult
from . import hydrology_pass2 as pass2


@dataclass(frozen=True)
class OutletIncisionConfig:
    minimum_outlet_width_m: float = 5.0
    outlet_width_coefficient: float = 8.0
    outlet_width_exponent: float = 0.5
    maximum_outlet_width_m: float = 500.0
    minimum_outlet_bed_slope: float = 1e-5
    maximum_outlet_path_cells: int = 64
    maximum_reroute_repair_rounds: int = 64
    maximum_incision_depth_m: float = 250.0
    maximum_cell_mean_lowering_m: float = 5.0
    minimum_depression_depth_m: float = 5.0
    maximum_corrected_area_fraction: float = 0.10
    maximum_receiver_change_area_fraction: float = 0.15
    maximum_receiver_change_cell_fraction: float = 0.15
    maximum_reroute_constraint_cell_fraction: float = 0.15
    maximum_volume_relative_error: float = 1e-10

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "OutletIncisionConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown outlet-incision controls: {', '.join(sorted(unknown))}")
        values = {
            name: (
                int(mapping.get(name, field.default))
                if name in {"maximum_outlet_path_cells", "maximum_reroute_repair_rounds"}
                else float(mapping.get(name, field.default))
            )
            for name, field in cls.__dataclass_fields__.items()
        }
        config = cls(**values)
        if not 1 <= config.maximum_outlet_path_cells <= 4096:
            raise ValueError("maximum_outlet_path_cells must be in [1, 4096]")
        if not 1 <= config.maximum_reroute_repair_rounds <= 4096:
            raise ValueError("maximum_reroute_repair_rounds must be in [1, 4096]")
        positive = (
            "minimum_outlet_width_m",
            "outlet_width_coefficient",
            "outlet_width_exponent",
            "maximum_outlet_width_m",
            "minimum_outlet_bed_slope",
            "maximum_incision_depth_m",
            "maximum_cell_mean_lowering_m",
            "minimum_depression_depth_m",
            "maximum_volume_relative_error",
        )
        for name in positive:
            value = getattr(config, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if config.minimum_outlet_width_m > config.maximum_outlet_width_m:
            raise ValueError("minimum_outlet_width_m must not exceed maximum_outlet_width_m")
        for name in (
            "maximum_corrected_area_fraction",
            "maximum_receiver_change_area_fraction",
            "maximum_receiver_change_cell_fraction",
            "maximum_reroute_constraint_cell_fraction",
        ):
            value = getattr(config, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        return config


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
    values = np.asarray(table[name].combine_chunks().values, dtype=np.float32)
    return np.ascontiguousarray(values.reshape(table.num_rows, width), dtype=np.float32)


def _lookup_rows(ids: np.ndarray, query: np.ndarray, *, label: str) -> np.ndarray:
    order = np.argsort(ids)
    sorted_ids = ids[order]
    positions = np.searchsorted(sorted_ids, query)
    if np.any(positions >= len(sorted_ids)) or np.any(sorted_ids[positions] != query):
        raise RuntimeError(f"{label} references an unknown identifier")
    return order[positions]


def _cyclic_rows(cell_ids: np.ndarray, receiver_ids: np.ndarray) -> np.ndarray:
    routed = receiver_ids >= 0
    target_rows = np.full(len(cell_ids), -1, dtype=np.int64)
    if np.any(routed):
        target_rows[routed] = _lookup_rows(cell_ids, receiver_ids[routed], label="cyclic receiver")
    upstream_count = np.zeros(len(cell_ids), dtype=np.int32)
    np.add.at(upstream_count, target_rows[routed], 1)
    ready = deque(np.flatnonzero(upstream_count == 0).tolist())
    removed = np.zeros(len(cell_ids), dtype=bool)
    while ready:
        row = ready.popleft()
        removed[row] = True
        target = target_rows[row]
        if target >= 0:
            upstream_count[target] -= 1
            if upstream_count[target] == 0:
                ready.append(int(target))
    return np.flatnonzero(~removed)


def _candidate_order(candidate_ids: np.ndarray, downstream_ids: np.ndarray) -> list[int]:
    row_by_id = {int(value): row for row, value in enumerate(candidate_ids)}
    downstream = np.full(len(candidate_ids), -1, dtype=np.int32)
    indegree = np.zeros(len(candidate_ids), dtype=np.int32)
    for row, downstream_id in enumerate(downstream_ids):
        if downstream_id < 0:
            continue
        target = row_by_id.get(int(downstream_id))
        if target is None or target == row:
            raise RuntimeError("outlet-incision candidate graph has an invalid edge")
        downstream[row] = target
        indegree[target] += 1
    ready = deque(np.flatnonzero(indegree == 0).tolist())
    order: list[int] = []
    while ready:
        row = ready.popleft()
        order.append(row)
        target = downstream[row]
        if target >= 0:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(int(target))
    if len(order) != len(candidate_ids):
        raise RuntimeError("outlet-incision candidate graph is cyclic")
    return order


def _edge_length_m(xyz: np.ndarray, source: int, target: int, radius_m: float) -> float:
    cosine = float(np.clip(np.dot(xyz[source], xyz[target]), -1.0, 1.0))
    return float(np.arccos(cosine) * radius_m)


def _trace_path(
    *,
    start_row: int,
    candidate_id: int,
    requested_bed_m: float,
    receiver_rows: np.ndarray,
    depression_ids: np.ndarray,
    anchor_kind: np.ndarray,
    routing_surface_m: np.ndarray,
    xyz: np.ndarray,
    radius_m: float,
    maximum_cells: int,
) -> tuple[list[int], list[float], int, str, int]:
    path = [start_row]
    distance = [0.0]
    current = start_row
    for _ in range(maximum_cells - 1):
        target = int(receiver_rows[current])
        if target < 0:
            return path, distance, -1, "terminal", -1
        edge_length = _edge_length_m(xyz, current, target, radius_m)
        target_kind = str(anchor_kind[target])
        target_depression = int(depression_ids[target])
        if target_kind != "ordinary":
            return path, distance, target, target_kind, target_depression
        if target_depression >= 0 and target_depression != candidate_id:
            return path, distance, target, "candidate", target_depression
        if routing_surface_m[target] <= requested_bed_m:
            return path, distance, target, "lower_terrain", target_depression
        path.append(target)
        distance.append(distance[-1] + edge_length)
        current = target
    target = int(receiver_rows[current])
    if target < 0:
        return path, distance, -1, "terminal", -1
    return path, distance, target, "path_limit", int(depression_ids[target])


def _plan_outlets(
    config: OutletIncisionConfig,
    cells: pa.Table,
    candidates: pa.Table,
    *,
    radius_m: float,
) -> tuple[pa.Table, pa.Table, np.ndarray, np.ndarray, dict[str, object]]:
    cell_ids = _column(cells, "fine_cell_id", np.dtype(np.int32))
    receiver_ids = _column(cells, "stabilized_receiver_id", np.dtype(np.int32))
    depression_ids = _column(cells, "stabilized_depression_id", np.dtype(np.int32))
    routing_surface = _column(cells, "routing_surface_after_m", np.dtype(np.float64))
    area_km2 = _column(cells, "area_km2", np.dtype(np.float64))
    xyz = _fixed_list_column(cells, "xyz", 3).astype(np.float64)
    anchor_kind = np.asarray(cells["routing_anchor_kind"].to_pylist(), dtype=object)
    routing_support_kind = anchor_kind.copy()
    if "outlet_fixed_receiver_id" in cells.column_names:
        prior_fixed_receivers = _column(cells, "outlet_fixed_receiver_id", np.dtype(np.int32))
        prior_depth = (
            _column(cells, "outlet_incision_depth_m", np.dtype(np.float64))
            if "outlet_incision_depth_m" in cells.column_names
            else np.zeros(cells.num_rows, dtype=np.float64)
        )
        prior_outlet_support = (
            (prior_fixed_receivers != pass2.NO_FIXED_RECEIVER)
            & (anchor_kind == "ordinary")
            & (prior_depth > 0.0)
        )
        routing_support_kind[prior_outlet_support] = "outlet_constraint"
    routed = receiver_ids >= 0
    receiver_rows = np.full(len(cells), -1, dtype=np.int64)
    if np.any(routed):
        receiver_rows[routed] = _lookup_rows(
            cell_ids, receiver_ids[routed], label="outlet receiver"
        )

    candidate_ids = _column(candidates, "depression_id", np.dtype(np.int32))
    downstream_ids = _column(candidates, "downstream_depression_id", np.dtype(np.int32))
    required = np.asarray(candidates["outlet_erosion_required"], dtype=bool)
    spill_rows = _lookup_rows(
        cell_ids,
        _column(candidates, "spill_cell_id", np.dtype(np.int32)),
        label="outlet spill cell",
    )
    spill_level = _column(candidates, "spill_elevation_m", np.dtype(np.float64))
    recommended = np.minimum(
        _column(candidates, "recommended_outlet_incision_m", np.dtype(np.float64)),
        config.maximum_incision_depth_m,
    )
    annual_overflow = _column(candidates, "annual_overflow_km3", np.dtype(np.float64))
    overflow_m3s = annual_overflow * 1e9 / (365.2422 * 86_400.0)
    width_m = np.clip(
        config.outlet_width_coefficient
        * np.power(np.maximum(overflow_m3s, 0.0), config.outlet_width_exponent),
        config.minimum_outlet_width_m,
        config.maximum_outlet_width_m,
    )
    order = _candidate_order(candidate_ids, downstream_ids)
    candidate_row_by_id = {int(value): row for row, value in enumerate(candidate_ids)}
    planned_level = spill_level.copy()
    requested_bed_by_cell = routing_surface.copy()
    width_by_cell = np.zeros(len(cells), dtype=np.float64)
    contributors: dict[int, set[int]] = {}
    requested_beds_by_candidate: dict[int, list[tuple[int, float]]] = {}
    plan_rows: list[dict[str, object]] = []

    for candidate_row in reversed(order):
        if not required[candidate_row]:
            continue
        candidate_id = int(candidate_ids[candidate_row])
        start_row = int(spill_rows[candidate_row])
        if anchor_kind[start_row] != "ordinary" or depression_ids[start_row] != candidate_id:
            raise RuntimeError("outlet incision starts outside active ordinary candidate support")
        requested_bed = spill_level[candidate_row] - recommended[candidate_row]
        path, distance, boundary_row, termination, boundary_depression = _trace_path(
            start_row=start_row,
            candidate_id=candidate_id,
            requested_bed_m=float(requested_bed),
            receiver_rows=receiver_rows,
            depression_ids=depression_ids,
            anchor_kind=routing_support_kind,
            routing_surface_m=routing_surface,
            xyz=xyz,
            radius_m=radius_m,
            maximum_cells=config.maximum_outlet_path_cells,
        )
        expected_downstream = int(downstream_ids[candidate_row])
        if termination == "candidate" and boundary_depression != expected_downstream:
            raise RuntimeError("outlet path reaches a different downstream candidate")
        if termination == "path_limit":
            applied_depth = 0.0
            start_bed = spill_level[candidate_row]
            status = "blocked_path_limit"
        else:
            distance_to_boundary = distance[-1]
            if boundary_row >= 0:
                distance_to_boundary += _edge_length_m(xyz, path[-1], boundary_row, radius_m)
                if termination == "candidate":
                    boundary_candidate_row = candidate_row_by_id[boundary_depression]
                    boundary_elevation = planned_level[boundary_candidate_row]
                elif termination == "preserved_handoff":
                    boundary_elevation = float(
                        np.asarray(cells["stabilized_hydrologic_elevation_m"], dtype=np.float64)[
                            boundary_row
                        ]
                    )
                else:
                    boundary_elevation = routing_surface[boundary_row]
                minimum_start_bed = (
                    boundary_elevation + config.minimum_outlet_bed_slope * distance_to_boundary
                )
                start_bed = max(float(requested_bed), float(minimum_start_bed))
            else:
                distance_to_boundary = distance[-1]
                start_bed = float(requested_bed)
            start_bed = min(start_bed, float(spill_level[candidate_row]))
            applied_depth = max(float(spill_level[candidate_row] - start_bed), 0.0)
            status = "applied" if applied_depth > 1e-9 else "blocked_downstream_grade"
        planned_level[candidate_row] = start_bed
        if applied_depth > 1e-9:
            candidate_requests: list[tuple[int, float]] = []
            for row, downstream_distance in zip(path, distance, strict=True):
                bed = start_bed - config.minimum_outlet_bed_slope * downstream_distance
                requested_bed_by_cell[row] = min(requested_bed_by_cell[row], bed)
                width_by_cell[row] = max(width_by_cell[row], width_m[candidate_row])
                contributors.setdefault(row, set()).add(candidate_id)
                candidate_requests.append((row, bed))
            requested_beds_by_candidate[candidate_id] = candidate_requests
        plan_rows.append(
            {
                "depression_id": candidate_id,
                "downstream_depression_id": expected_downstream,
                "spill_cell_id": int(cell_ids[start_row]),
                "termination_kind": termination,
                "termination_cell_id": int(cell_ids[boundary_row]) if boundary_row >= 0 else -1,
                "termination_depression_id": boundary_depression,
                "status": status,
                "path_cell_count": len(path),
                "path_distance_m": float(distance_to_boundary),
                "mean_overflow_discharge_m3s": float(overflow_m3s[candidate_row]),
                "outlet_width_m": float(width_m[candidate_row]),
                "old_spill_elevation_m": float(spill_level[candidate_row]),
                "requested_incision_m": float(recommended[candidate_row]),
                "planned_incision_m": applied_depth,
                "planned_spill_bed_m": float(start_bed),
            }
        )

    corrected_bed = np.minimum(routing_surface, requested_bed_by_cell)
    incision_depth = routing_surface - corrected_bed
    corrected = incision_depth > 1e-9
    for plan in plan_rows:
        if plan["status"] != "applied":
            continue
        requests = requested_beds_by_candidate[int(plan["depression_id"])]
        if not any(routing_surface[cell_row] - bed > 1e-9 for cell_row, bed in requests):
            plan["status"] = "already_satisfied"
    representative_length_m = np.sqrt(area_km2 * 1e6)
    eroded_volume_m3 = incision_depth * width_by_cell * representative_length_m
    maximum_volume = config.maximum_cell_mean_lowering_m * area_km2 * 1e6
    eroded_volume_m3 = np.minimum(eroded_volume_m3, maximum_volume)
    mean_lowering_m = eroded_volume_m3 / (area_km2 * 1e6)
    corrected_rows = np.flatnonzero(corrected)
    cell_catalog = pa.table(
        {
            "fine_cell_id": pa.array(cell_ids[corrected_rows], type=pa.int32()),
            "parent_cell_id": pa.array(
                _column(cells, "parent_cell_id", np.dtype(np.int32))[corrected_rows],
                type=pa.int32(),
            ),
            "area_km2": pa.array(area_km2[corrected_rows], type=pa.float64()),
            "old_routing_bed_m": pa.array(routing_surface[corrected_rows], type=pa.float64()),
            "corrected_routing_bed_m": pa.array(corrected_bed[corrected_rows], type=pa.float64()),
            "incision_depth_m": pa.array(incision_depth[corrected_rows], type=pa.float64()),
            "outlet_width_m": pa.array(width_by_cell[corrected_rows], type=pa.float64()),
            "representative_length_m": pa.array(
                representative_length_m[corrected_rows], type=pa.float64()
            ),
            "eroded_volume_m3": pa.array(eroded_volume_m3[corrected_rows], type=pa.float64()),
            "terrain_mean_lowering_m": pa.array(mean_lowering_m[corrected_rows], type=pa.float64()),
            "contributing_candidate_count": pa.array(
                [len(contributors[int(row)]) for row in corrected_rows], type=pa.int32()
            ),
            "minimum_contributing_depression_id": pa.array(
                [min(contributors[int(row)]) for row in corrected_rows], type=pa.int32()
            ),
        }
    )
    candidate_schema = pa.schema(
        [
            ("depression_id", pa.int32()),
            ("downstream_depression_id", pa.int32()),
            ("spill_cell_id", pa.int32()),
            ("termination_kind", pa.string()),
            ("termination_cell_id", pa.int32()),
            ("termination_depression_id", pa.int32()),
            ("status", pa.string()),
            ("path_cell_count", pa.int32()),
            ("path_distance_m", pa.float64()),
            ("mean_overflow_discharge_m3s", pa.float64()),
            ("outlet_width_m", pa.float64()),
            ("old_spill_elevation_m", pa.float64()),
            ("requested_incision_m", pa.float64()),
            ("planned_incision_m", pa.float64()),
            ("planned_spill_bed_m", pa.float64()),
        ]
    )
    candidate_catalog = pa.Table.from_pylist(plan_rows, schema=candidate_schema)
    active = np.asarray(cells["source_active"], dtype=bool)
    metadata = {
        "requested_candidate_count": int(np.count_nonzero(required)),
        "applied_candidate_count": sum(plan["status"] == "applied" for plan in plan_rows),
        "blocked_candidate_count": sum(plan["status"] != "applied" for plan in plan_rows),
        "path_limit_candidate_count": sum(
            plan["status"] == "blocked_path_limit" for plan in plan_rows
        ),
        "already_satisfied_candidate_count": sum(
            plan["status"] == "already_satisfied" for plan in plan_rows
        ),
        "corrected_cell_count": int(np.count_nonzero(corrected)),
        "corrected_area_km2": float(np.sum(area_km2[corrected])),
        "corrected_area_fraction": float(np.sum(area_km2[corrected]))
        / max(float(np.sum(area_km2[active])), 1e-12),
        "total_eroded_volume_m3": float(np.sum(eroded_volume_m3)),
        "maximum_incision_depth_m": float(np.max(incision_depth, initial=0.0)),
        "maximum_terrain_mean_lowering_m": float(np.max(mean_lowering_m, initial=0.0)),
        "maximum_path_cell_count": 0,
        "grade_blocked_spill_cell_ids": [
            int(plan["spill_cell_id"])
            for plan in plan_rows
            if plan["status"] == "blocked_downstream_grade"
        ],
        "outlet_feedback_suppressed_spill_cell_ids": [
            int(plan["spill_cell_id"])
            for plan in plan_rows
            if plan["status"] in {"blocked_downstream_grade", "already_satisfied"}
        ],
    }
    if plan_rows:
        metadata["maximum_path_cell_count"] = max(
            int(plan["path_cell_count"]) for plan in plan_rows
        )
        metadata["maximum_path_distance_m"] = max(
            float(plan["path_distance_m"]) for plan in plan_rows
        )
    else:
        metadata["maximum_path_distance_m"] = 0.0
    return candidate_catalog, cell_catalog, corrected_bed, mean_lowering_m, metadata


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Missing column '{name}'")
    return table.set_column(index, name, values)


def _corrected_cell_table(
    source: pa.Table,
    records: np.ndarray,
    corrected_routing_surface: np.ndarray,
    mean_lowering_m: np.ndarray,
    outlet_cells: pa.Table,
    outlet_fixed_receivers: np.ndarray,
) -> pa.Table:
    old_receiver = _column(source, "stabilized_receiver_id", np.dtype(np.int32))
    old_depression = _column(source, "stabilized_depression_id", np.dtype(np.int32))
    old_hydrologic = _column(source, "stabilized_hydrologic_elevation_m", np.dtype(np.float64))
    old_routing = _column(source, "routing_surface_after_m", np.dtype(np.float64))
    replaceable_fields = (
        "pre_outlet_receiver_id",
        "pre_outlet_depression_id",
        "pre_outlet_hydrologic_elevation_m",
        "pre_outlet_routing_surface_m",
        "outlet_terrain_mean_lowering_m",
        "outlet_incision_depth_m",
        "outlet_width_m",
        "outlet_contributing_candidate_count",
        "outlet_fixed_receiver_id",
    )
    existing_replaceable = [name for name in replaceable_fields if name in source.column_names]
    table = source.drop_columns(existing_replaceable) if existing_replaceable else source
    table = table.append_column("pre_outlet_receiver_id", pa.array(old_receiver, type=pa.int32()))
    table = table.append_column(
        "pre_outlet_depression_id", pa.array(old_depression, type=pa.int32())
    )
    table = table.append_column(
        "pre_outlet_hydrologic_elevation_m", pa.array(old_hydrologic, type=pa.float64())
    )
    table = table.append_column(
        "pre_outlet_routing_surface_m", pa.array(old_routing, type=pa.float64())
    )
    rerun_fields = (
        "routing_surface_after_m",
        "baseline_receiver_id",
        "stabilized_receiver_id",
        "baseline_anchor_cell_id",
        "stabilized_anchor_cell_id",
        "baseline_depression_id",
        "stabilized_depression_id",
        "routing_anchor_kind",
        "terminal_kind_pass2",
        "receiver_changed",
        "depression_changed",
        "baseline_hydrologic_elevation_m",
        "stabilized_hydrologic_elevation_m",
        "baseline_fill_depth_m",
        "stabilized_fill_depth_m",
        "stabilized_flow_slope",
        "contributing_area_km2",
        "stabilized_flow_direction_xyz",
    )
    table = table.drop_columns(rerun_fields)
    terrain_after = _column(source, "terrain_elevation_after_m", np.dtype(np.float64))
    table = _replace_column(
        table,
        "terrain_elevation_after_m",
        pa.array(terrain_after - mean_lowering_m, type=pa.float64()),
    )
    previous_mean_lowering = (
        _column(source, "outlet_terrain_mean_lowering_m", np.dtype(np.float64))
        if "outlet_terrain_mean_lowering_m" in source.column_names
        else np.zeros(source.num_rows, dtype=np.float64)
    )
    table = table.append_column(
        "outlet_terrain_mean_lowering_m",
        pa.array(previous_mean_lowering + mean_lowering_m, type=pa.float64()),
    )
    outlet_depth = np.zeros(source.num_rows, dtype=np.float64)
    outlet_width = np.zeros(source.num_rows, dtype=np.float64)
    outlet_contributors = np.zeros(source.num_rows, dtype=np.int32)
    if outlet_cells.num_rows:
        source_ids = _column(source, "fine_cell_id", np.dtype(np.int32))
        rows = _lookup_rows(
            source_ids,
            _column(outlet_cells, "fine_cell_id", np.dtype(np.int32)),
            label="outlet correction cell",
        )
        outlet_depth[rows] = _column(outlet_cells, "incision_depth_m", np.dtype(np.float64))
        outlet_width[rows] = _column(outlet_cells, "outlet_width_m", np.dtype(np.float64))
        outlet_contributors[rows] = _column(
            outlet_cells, "contributing_candidate_count", np.dtype(np.int32)
        )
    previous_depth = (
        _column(source, "outlet_incision_depth_m", np.dtype(np.float64))
        if "outlet_incision_depth_m" in source.column_names
        else np.zeros(source.num_rows, dtype=np.float64)
    )
    previous_width = (
        _column(source, "outlet_width_m", np.dtype(np.float64))
        if "outlet_width_m" in source.column_names
        else np.zeros(source.num_rows, dtype=np.float64)
    )
    previous_contributors = (
        _column(source, "outlet_contributing_candidate_count", np.dtype(np.int32))
        if "outlet_contributing_candidate_count" in source.column_names
        else np.zeros(source.num_rows, dtype=np.int32)
    )
    table = table.append_column(
        "outlet_incision_depth_m", pa.array(previous_depth + outlet_depth, type=pa.float64())
    )
    table = table.append_column(
        "outlet_width_m", pa.array(np.maximum(previous_width, outlet_width), type=pa.float64())
    )
    table = table.append_column(
        "outlet_contributing_candidate_count",
        pa.array(previous_contributors + outlet_contributors, type=pa.int32()),
    )
    table = table.append_column(
        "outlet_fixed_receiver_id", pa.array(outlet_fixed_receivers, type=pa.int32())
    )
    return pass2._cell_table(table, records, corrected_routing_surface)


def _parent_catalog(cells: pa.Table, outlet_cells: pa.Table) -> tuple[pa.Table, float]:
    parent_ids = np.unique(_column(cells, "parent_cell_id", np.dtype(np.int32)))
    volume = np.zeros(len(parent_ids), dtype=np.float64)
    corrected_count = np.zeros(len(parent_ids), dtype=np.int32)
    corrected_area = np.zeros(len(parent_ids), dtype=np.float64)
    if outlet_cells.num_rows:
        outlet_parent = _column(outlet_cells, "parent_cell_id", np.dtype(np.int32))
        parent_rows = _lookup_rows(parent_ids, outlet_parent, label="outlet correction parent")
        np.add.at(
            volume,
            parent_rows,
            _column(outlet_cells, "eroded_volume_m3", np.dtype(np.float64)),
        )
        np.add.at(corrected_count, parent_rows, 1)
        np.add.at(
            corrected_area,
            parent_rows,
            _column(outlet_cells, "area_km2", np.dtype(np.float64)),
        )
    table = pa.table(
        {
            "parent_cell_id": pa.array(parent_ids, type=pa.int32()),
            "outlet_eroded_volume_m3": pa.array(volume, type=pa.float64()),
            "corrected_child_count": pa.array(corrected_count, type=pa.int32()),
            "corrected_child_area_km2": pa.array(corrected_area, type=pa.float64()),
        }
    )
    return table, float(np.sum(volume))


def _cube_net_visualizer(result, request: VisualizationRequest) -> VisualizationResult | None:
    cell_record = result.artifact_records.get("OutletCorrectedBasinCellCatalog")
    metadata_record = result.artifact_records.get("OutletIncisionMetadata")
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
    display_row = np.minimum(row * display_resolution // fine_resolution, display_resolution - 1)
    display_col = np.minimum(col * display_resolution // fine_resolution, display_resolution - 1)
    net_row = placements[face, 0] * display_resolution + display_row
    net_col = placements[face, 1] * display_resolution + display_col
    image = np.full((display_resolution * 3, display_resolution * 4, 3), 17, dtype=np.uint8)
    active = np.asarray(cells["source_active"], dtype=bool)
    image[net_row[active], net_col[active]] = (48, 64, 52)
    incision = np.asarray(cells["outlet_incision_depth_m"], dtype=np.float64)
    corrected = incision > 0.0
    if np.any(corrected):
        strength = np.clip(
            np.log1p(incision[corrected])
            / max(float(np.log1p(np.max(incision[corrected]))), 1e-12),
            0.25,
            1.0,
        )
        base = image[net_row[corrected], net_col[corrected]].astype(np.float64)
        target = np.array([232.0, 146.0, 55.0])
        image[net_row[corrected], net_col[corrected]] = (
            base * (1.0 - strength[:, None]) + target * strength[:, None]
        ).astype(np.uint8)
    rendered = Image.fromarray(image, mode="RGB")
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
    output = request.output_dir / "outlet_incision.png"
    rendered.save(output)
    return VisualizationResult(
        output,
        "OutletCorrectedBasinCellCatalog",
        {
            "requested_candidate_count": metadata["requested_candidate_count"],
            "corrected_cell_count": metadata["corrected_cell_count"],
        },
    )


@stage(
    "outlet_incision",
    inputs=("surface_water", "hydrology_pass2", "planet"),
    outputs=(
        "OutletIncisionCandidateCatalog",
        "OutletIncisionCellCatalog",
        "OutletIncisionParentCatalog",
        "OutletCorrectedBasinCellCatalog",
        "PostIncisionDepressionCandidateCatalog",
        "StabilizedBasinCellCatalog",
        "StabilizedRiverReachCatalog",
        "LocalDepressionCandidateCatalog",
        "HydrologyCorrectionCatalog",
        "HydrologyPass2Metadata",
        "OutletIncisionMetadata",
    ),
    version="v5",
    native_libraries=("hydrology_pass2_native",),
    visualizer=_cube_net_visualizer,
)
def outlet_incision_stage(context, deps, config_mapping: Mapping[str, object]):
    config = OutletIncisionConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("outlet incision requires topology: cubed_sphere")
    pass2_result = deps["hydrology_pass2"]
    surface_result = deps["surface_water"]
    cells = _artifact_table(pass2_result, "StabilizedBasinCellCatalog")
    candidates = _artifact_table(surface_result, "SurfaceWaterCandidateCatalog")
    reaches = _artifact_table(pass2_result, "StabilizedRiverReachCatalog")
    pass2_metadata = pass2_result.artifact_records["HydrologyPass2Metadata"].value
    surface_metadata = surface_result.artifact_records["SurfaceWaterMetadata"].value
    required_count = int(np.count_nonzero(np.asarray(candidates["outlet_erosion_required"])))
    if required_count != int(surface_metadata["outlet_erosion_required_count"]):
        raise RuntimeError("outlet feedback count disagrees with surface-water metadata")
    planet = deps["planet"].artifact_records["PlanetMetadata"].value
    radius_m = float(planet["planet_radius_earth"]) * pass2.EARTH_RADIUS_M

    with context.timed("bounded_outlet_path_planning"):
        (
            candidate_catalog,
            cell_catalog,
            corrected_routing_surface,
            mean_lowering_m,
            path_metadata,
        ) = _plan_outlets(config, cells, candidates, radius_m=radius_m)

    prior_suppressed_spill_ids = {
        int(value)
        for value in pass2_metadata.get(
            "outlet_feedback_suppressed_spill_cell_ids",
            pass2_metadata.get("grade_blocked_spill_cell_ids", []),
        )
    }
    suppressed_spill_values = path_metadata["outlet_feedback_suppressed_spill_cell_ids"]
    if not isinstance(suppressed_spill_values, list):
        raise RuntimeError("outlet planner emitted invalid suppressed-spill metadata")
    current_suppressed_spill_ids = {int(value) for value in suppressed_spill_values}
    path_metadata["prior_suppressed_outlet_spill_cell_count"] = len(prior_suppressed_spill_ids)
    combined_suppressed_spill_ids = sorted(
        prior_suppressed_spill_ids | current_suppressed_spill_ids
    )
    path_metadata["outlet_feedback_suppressed_spill_cell_ids"] = combined_suppressed_spill_ids
    path_metadata["suppressed_outlet_spill_cell_count"] = len(combined_suppressed_spill_ids)

    cell_ids = _column(cells, "fine_cell_id", np.dtype(np.int32))
    old_receivers = _column(cells, "stabilized_receiver_id", np.dtype(np.int32))
    anchor_names = np.asarray(cells["routing_anchor_kind"].to_pylist(), dtype=object)
    anchor_code_by_name = {
        "ordinary": pass2.ANCHOR_NORMAL,
        "channel": pass2.ANCHOR_CHANNEL,
        "preserved_handoff": pass2.ANCHOR_EXCLUDED,
        "outside_terminal": pass2.ANCHOR_OUTSIDE,
    }
    try:
        anchor_kinds = np.array(
            [anchor_code_by_name[str(value)] for value in anchor_names], dtype=np.uint8
        )
    except KeyError as exc:
        raise RuntimeError("outlet incision received an unknown routing anchor kind") from exc
    channel_rows = np.flatnonzero(anchor_kinds == pass2.ANCHOR_CHANNEL)
    fixed_receivers = np.full(len(cells), pass2.NO_FIXED_RECEIVER, dtype=np.int32)
    fixed_receivers[channel_rows] = old_receivers[channel_rows]
    if "outlet_fixed_receiver_id" in cells.column_names:
        prior_fixed_receivers = _column(cells, "outlet_fixed_receiver_id", np.dtype(np.int32))
        prior_constraint_rows = np.flatnonzero(prior_fixed_receivers != pass2.NO_FIXED_RECEIVER)
        if np.any(anchor_kinds[prior_constraint_rows] != pass2.ANCHOR_NORMAL):
            raise RuntimeError("prior outlet constraints overlap a non-ordinary routing anchor")
        fixed_receivers[prior_constraint_rows] = prior_fixed_receivers[prior_constraint_rows]
    else:
        prior_constraint_rows = np.empty(0, dtype=np.int64)
    if cell_catalog.num_rows:
        new_constraint_rows = _lookup_rows(
            cell_ids,
            _column(cell_catalog, "fine_cell_id", np.dtype(np.int32)),
            label="outlet constraint cell",
        )
        if np.any(anchor_kinds[new_constraint_rows] != pass2.ANCHOR_NORMAL):
            raise RuntimeError("outlet constraints overlap a non-ordinary routing anchor")
        fixed_receivers[new_constraint_rows] = old_receivers[new_constraint_rows]
    else:
        new_constraint_rows = np.empty(0, dtype=np.int64)
    ordinary_fixed_receivers = np.full(len(cells), pass2.NO_FIXED_RECEIVER, dtype=np.int32)
    ordinary_constraint_rows = np.flatnonzero(
        (fixed_receivers != pass2.NO_FIXED_RECEIVER) & (anchor_kinds == pass2.ANCHOR_NORMAL)
    )
    ordinary_fixed_receivers[ordinary_constraint_rows] = fixed_receivers[ordinary_constraint_rows]
    source_active = np.ascontiguousarray(np.asarray(cells["source_active"]), dtype=np.uint8)
    old_routing_surface = _column(cells, "routing_surface_after_m", np.dtype(np.float64))
    controls = {
        "fine_resolution": int(pass2_metadata["fine_resolution"]),
        "minimum_depression_depth_m": config.minimum_depression_depth_m,
        "planet_radius_m": radius_m,
    }
    repair_round = 0
    repair_cell_count = 0
    active_count = int(np.count_nonzero(source_active))
    constrained_active_count = int(
        np.count_nonzero(
            (ordinary_fixed_receivers != pass2.NO_FIXED_RECEIVER) & (source_active != 0)
        )
    )
    if (
        constrained_active_count / max(active_count, 1)
        > config.maximum_reroute_constraint_cell_fraction
    ):
        raise RuntimeError("post-outlet reroute constraint scope exceeds its bound")
    with context.timed("post_outlet_hydrology_kernel"):
        while True:
            records, native_metadata = run_hydrology_pass2(
                controls=controls,
                cell_ids=cell_ids,
                terrain_before_m=old_routing_surface,
                routing_surface_after_m=corrected_routing_surface,
                cell_areas_km2=_column(cells, "area_km2", np.dtype(np.float64)),
                cell_xyz=_fixed_list_column(cells, "xyz", 3),
                anchor_kinds=anchor_kinds,
                source_active=source_active,
                fixed_receiver_ids=fixed_receivers,
            )
            if native_metadata["graph_valid"] == 1:
                break
            if repair_round >= config.maximum_reroute_repair_rounds:
                raise RuntimeError("post-outlet reroute exceeded its cycle-repair round bound")
            cyclic_rows = _cyclic_rows(
                cell_ids,
                np.ascontiguousarray(records["stabilized_receiver_id"], dtype=np.int32),
            )
            repair_rows = cyclic_rows[
                (anchor_kinds[cyclic_rows] == pass2.ANCHOR_NORMAL)
                & (fixed_receivers[cyclic_rows] == pass2.NO_FIXED_RECEIVER)
            ]
            if not len(repair_rows):
                raise RuntimeError("post-outlet reroute has an irreparable fixed-receiver cycle")
            fixed_receivers[repair_rows] = old_receivers[repair_rows]
            ordinary_fixed_receivers[repair_rows] = old_receivers[repair_rows]
            repair_cell_count += len(repair_rows)
            repair_round += 1
            constrained_active_count = int(
                np.count_nonzero(
                    (ordinary_fixed_receivers != pass2.NO_FIXED_RECEIVER) & (source_active != 0)
                )
            )
            if (
                constrained_active_count / max(active_count, 1)
                > config.maximum_reroute_constraint_cell_fraction
            ):
                raise RuntimeError("post-outlet reroute constraint scope exceeds its bound")
    if native_metadata["trunk_preserved_valid"] != 1:
        raise RuntimeError("post-outlet hydrology changed the accepted physical trunk")
    if native_metadata["process_exclusion_valid"] != 1:
        raise RuntimeError("post-outlet hydrology violated process-excluded support")
    constrained_rows = np.flatnonzero(fixed_receivers != pass2.NO_FIXED_RECEIVER)
    if np.any(
        records["stabilized_receiver_id"][constrained_rows] != fixed_receivers[constrained_rows]
    ):
        raise RuntimeError("post-outlet hydrology changed a fixed receiver")
    corrected_cells = _corrected_cell_table(
        cells,
        records,
        corrected_routing_surface,
        mean_lowering_m,
        cell_catalog,
        ordinary_fixed_receivers,
    )
    post_candidates, depression_metadata = pass2._depression_catalog(corrected_cells)
    hydrology_corrections = pass2._correction_catalog(corrected_cells)
    graph_metadata = pass2._graph_audit(
        corrected_cells, fixed_receivers, channel_rows, radius_m=radius_m
    )
    parent_catalog, parent_total_volume_m3 = _parent_catalog(cells, cell_catalog)
    cell_total_volume_m3 = float(
        np.sum(_column(cell_catalog, "eroded_volume_m3", np.dtype(np.float64)))
    )
    volume_relative_error = abs(parent_total_volume_m3 - cell_total_volume_m3) / max(
        cell_total_volume_m3, 1e-12
    )
    receiver_change_cell_fraction = int(
        graph_metadata["independent_receiver_changed_cell_count"]
    ) / max(active_count, 1)
    active_area_km2 = float(graph_metadata["independent_active_area_km2"])
    area_tolerance = 1e-8 * max(active_area_km2, 1.0)
    metadata = {
        **asdict(config),
        **path_metadata,
        **depression_metadata,
        **native_metadata,
        **graph_metadata,
        "selected_basin_id": int(pass2_metadata["selected_basin_id"]),
        "fine_resolution": int(pass2_metadata["fine_resolution"]),
        "pre_incision_candidate_count": candidates.num_rows,
        "post_incision_candidate_count": post_candidates.num_rows,
        "cell_total_eroded_volume_m3": cell_total_volume_m3,
        "parent_total_eroded_volume_m3": parent_total_volume_m3,
        "volume_reconstruction_relative_error": volume_relative_error,
        "receiver_change_cell_fraction": receiver_change_cell_fraction,
        "new_outlet_constraint_cell_count": int(len(new_constraint_rows)),
        "prior_outlet_constraint_cell_count": int(len(prior_constraint_rows)),
        "reroute_cycle_repair_round_count": repair_round,
        "new_reroute_repair_cell_count": repair_cell_count,
        "total_outlet_routing_constraint_cell_count": int(
            np.count_nonzero(ordinary_fixed_receivers != pass2.NO_FIXED_RECEIVER)
        ),
        "total_fixed_receiver_cell_count": int(len(constrained_rows)),
        "total_fixed_channel_cell_count": int(len(channel_rows)),
        "surface_water_stage_name": "surface_water_final",
        "routing_semantics": "bounded_subgrid_outlet_beds_with_constrained_receivers",
        "terrain_semantics": "channel_volume_divided_by_child_area",
    }
    if path_metadata["requested_candidate_count"] != candidate_catalog.num_rows:
        raise RuntimeError("outlet-incision candidate catalog does not cover every request")
    if path_metadata["path_limit_candidate_count"]:
        raise RuntimeError("outlet-incision path exceeds the configured cell bound")
    if path_metadata["corrected_area_fraction"] > config.maximum_corrected_area_fraction:
        raise RuntimeError("outlet-incision corrected area exceeds the configured bound")
    if volume_relative_error > config.maximum_volume_relative_error:
        raise RuntimeError("outlet-incision parent and cell volumes do not agree")
    if native_metadata["graph_valid"] != 1 or graph_metadata["independent_graph_valid"] != 1:
        raise RuntimeError("post-outlet hydrology produced a cyclic or uncovered graph")
    if (
        native_metadata["trunk_preserved_valid"] != 1
        or graph_metadata["trunk_receiver_mismatch_count"]
    ):
        raise RuntimeError("outlet incision changed the accepted physical trunk")
    if (
        native_metadata["process_exclusion_valid"] != 1
        or graph_metadata["independent_process_exclusion_valid"] != 1
    ):
        raise RuntimeError("outlet incision routed through process-excluded support")
    if graph_metadata["invalid_terminal_receiver_count"]:
        raise RuntimeError("outlet incision emitted an invalid terminal receiver")
    if (
        abs(float(native_metadata["contributing_area_residual_km2"])) > area_tolerance
        or abs(float(graph_metadata["independent_contributing_area_residual_km2"])) > area_tolerance
        or float(graph_metadata["maximum_downstream_contributing_area_error_km2"]) > area_tolerance
    ):
        raise RuntimeError("post-outlet hydrology does not conserve contributing area")
    if (
        graph_metadata["independent_receiver_changed_area_fraction"]
        > config.maximum_receiver_change_area_fraction
        or receiver_change_cell_fraction > config.maximum_receiver_change_cell_fraction
    ):
        raise RuntimeError("post-outlet receiver change exceeds the configured bound")
    if (
        float(graph_metadata["maximum_flow_direction_norm_error"]) > 1e-5
        or float(graph_metadata["maximum_flow_direction_tangent_error"]) > 1e-5
        or float(graph_metadata["maximum_flow_slope_relation_error"]) > 1e-12
    ):
        raise RuntimeError("post-outlet flow vectors or slopes are inconsistent")
    context.logger.log_event(
        {"type": "outlet_incision_summary", "stage": "outlet_incision", **metadata}
    )
    return {
        "OutletIncisionCandidateCatalog": candidate_catalog,
        "OutletIncisionCellCatalog": cell_catalog,
        "OutletIncisionParentCatalog": parent_catalog,
        "OutletCorrectedBasinCellCatalog": corrected_cells,
        "PostIncisionDepressionCandidateCatalog": post_candidates,
        "StabilizedBasinCellCatalog": corrected_cells,
        "StabilizedRiverReachCatalog": reaches,
        "LocalDepressionCandidateCatalog": post_candidates,
        "HydrologyCorrectionCatalog": hydrology_corrections,
        "HydrologyPass2Metadata": metadata,
        "OutletIncisionMetadata": metadata,
    }


__all__ = ["OutletIncisionConfig", "outlet_incision_stage"]
