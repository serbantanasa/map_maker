"""Prospective fluvial profiles and sediment routing for one refined basin."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pyarrow as pa
from PIL import Image, ImageDraw

from .._fluvial_native import run_fluvial_erosion
from ..cubed_sphere import CubedSphereGrid
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class BasinErosionConfig:
    minimum_bed_slope: float = 1e-5
    maximum_deposition_fraction: float = 0.35
    deposition_slope_scale: float = 0.001
    maximum_deposition_depth_m: float = 10.0
    bank_incision_fraction: float = 0.0
    hard_maximum_channel_incision_m: float = 2_000.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "BasinErosionConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown basin erosion controls: {', '.join(sorted(unknown))}")
        values = {
            name: float(mapping.get(name, field.default))
            for name, field in cls.__dataclass_fields__.items()
        }
        if not np.isfinite(values["minimum_bed_slope"]) or values["minimum_bed_slope"] < 0.0:
            raise ValueError("minimum_bed_slope must be finite and non-negative")
        maximum_deposition_fraction = values["maximum_deposition_fraction"]
        if not np.isfinite(maximum_deposition_fraction) or not (
            0.0 <= maximum_deposition_fraction <= 1.0
        ):
            raise ValueError("maximum_deposition_fraction must be finite and in [0, 1]")
        if (
            not np.isfinite(values["deposition_slope_scale"])
            or values["deposition_slope_scale"] <= 0.0
        ):
            raise ValueError("deposition_slope_scale must be finite and positive")
        if (
            not np.isfinite(values["maximum_deposition_depth_m"])
            or values["maximum_deposition_depth_m"] < 0.0
        ):
            raise ValueError("maximum_deposition_depth_m must be finite and non-negative")
        bank_incision_fraction = values["bank_incision_fraction"]
        if not np.isfinite(bank_incision_fraction) or bank_incision_fraction != 0.0:
            raise ValueError(
                "bank_incision_fraction must be 0 in the sparse coarse-scale stage; "
                "physical bank erosion belongs to regional refinement"
            )
        if (
            not np.isfinite(values["hard_maximum_channel_incision_m"])
            or values["hard_maximum_channel_incision_m"] <= 0.0
        ):
            raise ValueError("hard_maximum_channel_incision_m must be finite and positive")
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


def _lookup_rows(ids: np.ndarray, query: np.ndarray, *, label: str) -> np.ndarray:
    order = np.argsort(ids)
    sorted_ids = ids[order]
    positions = np.searchsorted(sorted_ids, query)
    if np.any(positions >= len(sorted_ids)) or np.any(sorted_ids[positions] != query):
        raise RuntimeError(f"{label} references an unknown identifier")
    return order[positions]


def _profile_table(records: np.ndarray) -> pa.Table:
    return pa.table(
        {
            "reach_id": pa.array(records["reach_id"], type=pa.int32()),
            "fine_cell_id": pa.array(records["fine_cell_id"], type=pa.int32()),
            "parent_cell_id": pa.array(records["parent_cell_id"], type=pa.int32()),
            "path_order": pa.array(records["path_order"], type=pa.int32()),
            "terrain_elevation_m": pa.array(records["terrain_elevation_m"], type=pa.float64()),
            "bed_elevation_m": pa.array(records["bed_elevation_m"], type=pa.float64()),
            "incision_depth_m": pa.array(records["incision_depth_m"], type=pa.float64()),
            "reach_length_m": pa.array(records["reach_length_m"], type=pa.float64()),
            "eroded_volume_m3": pa.array(records["eroded_volume_m3"], type=pa.float64()),
        }
    )


def _reach_table(source: pa.Table, records: np.ndarray) -> pa.Table:
    source_ids = np.asarray(source["reach_id"], dtype=np.int32)
    if not np.array_equal(source_ids, records["reach_id"]):
        raise RuntimeError("fluvial kernel changed reach order or identity")
    has_bed = records["has_physical_bed"] != 0
    table = source
    table = table.append_column("has_physical_bed", pa.array(has_bed, type=pa.bool_()))
    for name in (
        "entry_bed_elevation_m",
        "exit_bed_elevation_m",
        "minimum_realized_slope",
        "maximum_incision_depth_m",
    ):
        values = records[name]
        table = table.append_column(
            name,
            pa.array(values, mask=~has_bed | ~np.isfinite(values), type=pa.float64()),
        )
    for name in (
        "upstream_input_volume_m3",
        "local_erosion_volume_m3",
        "bank_eroded_volume_m3",
        "available_sediment_volume_m3",
        "floodplain_deposition_volume_m3",
        "downstream_transfer_volume_m3",
        "terminal_deposition_volume_m3",
        "exported_sediment_volume_m3",
    ):
        table = table.append_column(name, pa.array(records[name], type=pa.float64()))
    return table


def _cell_table(source: pa.Table, records: np.ndarray) -> tuple[pa.Table, dict[str, float]]:
    fine_ids = np.asarray(source["fine_cell_id"], dtype=np.int32)
    areas_m2 = np.asarray(source["area_km2"], dtype=np.float64) * 1_000_000.0
    terrain = np.asarray(source["terrain_elevation_m"], dtype=np.float32)
    prospective_excavation = np.zeros(source.num_rows, dtype=np.float64)
    prospective_deposition = np.zeros(source.num_rows, dtype=np.float64)
    maximum_incision = np.zeros(source.num_rows, dtype=np.float64)
    if len(records):
        rows = _lookup_rows(fine_ids, records["fine_cell_id"], label="fluvial cell budget")
        if not np.array_equal(
            np.asarray(source["parent_cell_id"], dtype=np.int32)[rows],
            records["parent_cell_id"],
        ):
            raise RuntimeError("fluvial cell budget changed parent identity")
        prospective_excavation[rows] = records["eroded_volume_m3"]
        prospective_deposition[rows] = records["deposited_volume_m3"]
        maximum_incision[rows] = records["maximum_incision_depth_m"]
    applied_erosion = np.zeros(source.num_rows, dtype=np.float64)
    applied_deposition = np.zeros(source.num_rows, dtype=np.float64)
    mean_delta = np.zeros(source.num_rows, dtype=np.float64)
    terrain_after = terrain.astype(np.float64)
    table = source.append_column(
        "prospective_channel_excavation_volume_m3",
        pa.array(prospective_excavation, type=pa.float64()),
    )
    table = table.append_column(
        "prospective_floodplain_deposition_volume_m3",
        pa.array(prospective_deposition, type=pa.float64()),
    )
    table = table.append_column(
        "prospective_maximum_channel_incision_m",
        pa.array(maximum_incision, type=pa.float64()),
    )
    table = table.append_column(
        "applied_terrain_erosion_volume_m3", pa.array(applied_erosion, type=pa.float64())
    )
    table = table.append_column(
        "applied_terrain_deposition_volume_m3",
        pa.array(applied_deposition, type=pa.float64()),
    )
    table = table.append_column("terrain_mean_delta_m", pa.array(mean_delta, type=pa.float64()))
    table = table.append_column(
        "terrain_elevation_after_m", pa.array(terrain_after, type=pa.float64())
    )
    represented_delta_volume = float(np.sum(mean_delta * areas_m2))
    expected_delta_volume = float(np.sum(applied_deposition) - np.sum(applied_erosion))
    return table, {
        "maximum_absolute_cell_mean_delta_m": float(np.max(np.abs(mean_delta), initial=0.0)),
        "cell_mean_volume_relation_residual_m3": represented_delta_volume - expected_delta_volume,
        "raster_terrain_feedback_applied": 0,
    }


def _parent_table(source: pa.Table, cells: pa.Table) -> tuple[pa.Table, dict[str, float]]:
    parent_ids = np.asarray(source["parent_cell_id"], dtype=np.int32)
    child_parent_ids = np.asarray(cells["parent_cell_id"], dtype=np.int32)
    parent_rows = _lookup_rows(parent_ids, child_parent_ids, label="eroded child parent")
    prospective_excavation = np.zeros(source.num_rows, dtype=np.float64)
    prospective_deposition = np.zeros(source.num_rows, dtype=np.float64)
    np.add.at(
        prospective_excavation,
        parent_rows,
        np.asarray(cells["prospective_channel_excavation_volume_m3"], dtype=np.float64),
    )
    np.add.at(
        prospective_deposition,
        parent_rows,
        np.asarray(cells["prospective_floodplain_deposition_volume_m3"], dtype=np.float64),
    )
    applied_erosion = np.zeros(source.num_rows, dtype=np.float64)
    applied_deposition = np.zeros(source.num_rows, dtype=np.float64)
    mean_delta = np.zeros(source.num_rows, dtype=np.float64)
    table = source.append_column(
        "prospective_child_channel_excavation_volume_m3",
        pa.array(prospective_excavation, type=pa.float64()),
    )
    table = table.append_column(
        "prospective_child_floodplain_deposition_volume_m3",
        pa.array(prospective_deposition, type=pa.float64()),
    )
    table = table.append_column(
        "applied_child_terrain_erosion_volume_m3",
        pa.array(applied_erosion, type=pa.float64()),
    )
    table = table.append_column(
        "applied_child_terrain_deposition_volume_m3",
        pa.array(applied_deposition, type=pa.float64()),
    )
    table = table.append_column(
        "restricted_terrain_mean_delta_m", pa.array(mean_delta, type=pa.float64())
    )
    prospective_child_net = float(
        np.sum(np.asarray(cells["prospective_floodplain_deposition_volume_m3"], dtype=np.float64))
        - np.sum(np.asarray(cells["prospective_channel_excavation_volume_m3"], dtype=np.float64))
    )
    return table, {
        "parent_budget_residual_m3": float(np.sum(prospective_deposition - prospective_excavation))
        - prospective_child_net,
        "maximum_absolute_parent_mean_delta_m": float(np.max(np.abs(mean_delta), initial=0.0)),
    }


def _maximum_junction_error(profiles: pa.Table) -> float:
    cell_ids = np.asarray(profiles["fine_cell_id"], dtype=np.int32)
    beds = np.asarray(profiles["bed_elevation_m"], dtype=np.float64)
    order = np.argsort(cell_ids, kind="stable")
    sorted_ids = cell_ids[order]
    sorted_beds = beds[order]
    maximum = 0.0
    start = 0
    while start < len(sorted_ids):
        end = start + 1
        while end < len(sorted_ids) and sorted_ids[end] == sorted_ids[start]:
            end += 1
        maximum = max(
            maximum, float(np.max(sorted_beds[start:end]) - np.min(sorted_beds[start:end]))
        )
        start = end
    return maximum


def _group_sums(
    target_ids: np.ndarray,
    source_ids: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    totals = np.zeros(len(target_ids), dtype=np.float64)
    if len(source_ids):
        rows = _lookup_rows(target_ids, source_ids, label=label)
        np.add.at(totals, rows, values)
    return totals


def _profile_audit(
    profiles: pa.Table,
    reaches: pa.Table,
    cells: pa.Table,
    *,
    radius_m: float,
) -> dict[str, float | int]:
    reach_ids = np.asarray(profiles["reach_id"], dtype=np.int32)
    cell_ids = np.asarray(profiles["fine_cell_id"], dtype=np.int32)
    path_order = np.asarray(profiles["path_order"], dtype=np.int32)
    terrain = np.asarray(profiles["terrain_elevation_m"], dtype=np.float64)
    bed = np.asarray(profiles["bed_elevation_m"], dtype=np.float64)
    depth = np.asarray(profiles["incision_depth_m"], dtype=np.float64)
    length_m = np.asarray(profiles["reach_length_m"], dtype=np.float64)
    volume_m3 = np.asarray(profiles["eroded_volume_m3"], dtype=np.float64)
    catalog_reach_ids = np.asarray(reaches["reach_id"], dtype=np.int32)
    reach_rows = _lookup_rows(catalog_reach_ids, reach_ids, label="bed profile reach")
    width_m = np.asarray(reaches["channel_width_m"], dtype=np.float64)[reach_rows]
    expected_volume = width_m * length_m * depth

    fine_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    profile_cell_rows = _lookup_rows(fine_ids, cell_ids, label="bed profile cell")
    xyz = _fixed_list_column(cells, "xyz", 3).astype(np.float64)
    profile_xyz = xyz[profile_cell_rows]
    sort_order = np.lexsort((path_order, reach_ids))
    sorted_reaches = reach_ids[sort_order]
    sorted_paths = path_order[sort_order]
    edge_mask = (sorted_reaches[1:] == sorted_reaches[:-1]) & (
        sorted_paths[1:] == sorted_paths[:-1] + 1
    )
    reach_edge_sources = sort_order[:-1][edge_mask]
    reach_edge_targets = sort_order[1:][edge_mask]
    edge_pairs = np.column_stack((cell_ids[reach_edge_sources], cell_ids[reach_edge_targets]))
    if len(edge_pairs):
        _, unique_edge_indices = np.unique(edge_pairs, axis=0, return_index=True)
        edge_sources = reach_edge_sources[unique_edge_indices]
        edge_targets = reach_edge_targets[unique_edge_indices]
    else:
        edge_sources = reach_edge_sources
        edge_targets = reach_edge_targets
    if len(edge_sources):
        source_xyz = profile_xyz[edge_sources]
        target_xyz = profile_xyz[edge_targets]
        dot = (
            source_xyz[:, 0] * target_xyz[:, 0]
            + source_xyz[:, 1] * target_xyz[:, 1]
            + source_xyz[:, 2] * target_xyz[:, 2]
        )
        edge_length_m = np.arccos(np.clip(dot, -1.0, 1.0)) * radius_m
        if np.any(~np.isfinite(edge_length_m)) or np.any(edge_length_m <= 0.0):
            raise RuntimeError("published bed profile contains an invalid physical edge")
        slopes = (bed[edge_sources] - bed[edge_targets]) / edge_length_m
        minimum_slope = float(np.min(slopes))
    else:
        minimum_slope = 0.0

    return {
        "emitted_physical_edge_count": int(len(edge_sources)),
        "emitted_reach_edge_count": int(len(reach_edge_sources)),
        "shared_directed_edge_occurrence_count": int(len(reach_edge_sources) - len(edge_sources)),
        "emitted_minimum_realized_slope": minimum_slope,
        "maximum_junction_bed_error_m": _maximum_junction_error(profiles),
        "maximum_bed_above_terrain_error_m": float(
            np.max(np.maximum(bed - terrain, 0.0), initial=0.0)
        ),
        "maximum_profile_depth_relation_error_m": float(
            np.max(np.abs(depth - (terrain - bed)), initial=0.0)
        ),
        "maximum_profile_volume_relation_residual_m3": float(
            np.max(np.abs(volume_m3 - expected_volume), initial=0.0)
        ),
    }


def _budget_audit(
    profiles: pa.Table,
    reaches: pa.Table,
    cells: pa.Table,
    parents: pa.Table,
) -> dict[str, float]:
    reach_ids = np.asarray(reaches["reach_id"], dtype=np.int32)
    profile_reach_ids = np.asarray(profiles["reach_id"], dtype=np.int32)
    profile_volume = np.asarray(profiles["eroded_volume_m3"], dtype=np.float64)
    local_erosion = np.asarray(reaches["local_erosion_volume_m3"], dtype=np.float64)
    bank_eroded = np.asarray(reaches["bank_eroded_volume_m3"], dtype=np.float64)
    profile_by_reach = _group_sums(
        reach_ids, profile_reach_ids, profile_volume, label="profile erosion reach"
    )

    fine_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    profile_cell_ids = np.asarray(profiles["fine_cell_id"], dtype=np.int32)
    profile_by_cell = _group_sums(
        fine_ids, profile_cell_ids, profile_volume, label="profile erosion cell"
    )
    cell_eroded = np.asarray(cells["prospective_channel_excavation_volume_m3"], dtype=np.float64)
    cell_deposited = np.asarray(
        cells["prospective_floodplain_deposition_volume_m3"], dtype=np.float64
    )
    reach_deposited = np.asarray(reaches["floodplain_deposition_volume_m3"], dtype=np.float64)

    upstream_input = np.asarray(reaches["upstream_input_volume_m3"], dtype=np.float64)
    available = np.asarray(reaches["available_sediment_volume_m3"], dtype=np.float64)
    transfer = np.asarray(reaches["downstream_transfer_volume_m3"], dtype=np.float64)
    terminal = np.asarray(reaches["terminal_deposition_volume_m3"], dtype=np.float64)
    exported = np.asarray(reaches["exported_sediment_volume_m3"], dtype=np.float64)
    downstream_ids = np.asarray(reaches["downstream_reach_id"], dtype=np.int32)
    expected_upstream = np.zeros(reaches.num_rows, dtype=np.float64)
    routed = downstream_ids >= 0
    if np.any(routed):
        downstream_rows = _lookup_rows(
            reach_ids, downstream_ids[routed], label="downstream sediment reach"
        )
        np.add.at(expected_upstream, downstream_rows, transfer[routed])

    parent_ids = np.asarray(parents["parent_cell_id"], dtype=np.int32)
    cell_parent_ids = np.asarray(cells["parent_cell_id"], dtype=np.int32)
    eroded_by_parent = _group_sums(
        parent_ids, cell_parent_ids, cell_eroded, label="cell erosion parent"
    )
    deposited_by_parent = _group_sums(
        parent_ids, cell_parent_ids, cell_deposited, label="cell deposition parent"
    )
    published_parent_eroded = np.asarray(
        parents["prospective_child_channel_excavation_volume_m3"], dtype=np.float64
    )
    published_parent_deposited = np.asarray(
        parents["prospective_child_floodplain_deposition_volume_m3"], dtype=np.float64
    )

    # These volumes audit the candidate vector profile; none is applied to raster terrain here.
    total_channel_eroded = float(np.sum(profile_volume))
    total_bank_eroded = float(np.sum(bank_eroded))
    total_process_eroded = total_channel_eroded + total_bank_eroded
    total_cell_eroded = float(np.sum(cell_eroded))
    total_cell_deposited = float(np.sum(cell_deposited))
    independent_sediment_residual = (
        total_process_eroded
        - total_cell_deposited
        - float(np.sum(terminal))
        - float(np.sum(exported))
    )
    return {
        "independent_total_eroded_volume_m3": total_process_eroded,
        "independent_total_channel_eroded_volume_m3": total_channel_eroded,
        "independent_total_bank_eroded_volume_m3": total_bank_eroded,
        "independent_total_floodplain_deposition_volume_m3": total_cell_deposited,
        "independent_total_terminal_deposition_volume_m3": float(np.sum(terminal)),
        "independent_total_exported_sediment_volume_m3": float(np.sum(exported)),
        "independent_sediment_conservation_residual_m3": independent_sediment_residual,
        "maximum_reach_local_erosion_residual_m3": float(
            np.max(np.abs(profile_by_reach + bank_eroded - local_erosion), initial=0.0)
        ),
        "maximum_reach_channel_erosion_residual_m3": float(
            np.max(np.abs(profile_by_reach - (local_erosion - bank_eroded)), initial=0.0)
        ),
        "maximum_cell_channel_erosion_residual_m3": float(
            np.max(np.maximum(profile_by_cell - cell_eroded, 0.0), initial=0.0)
        ),
        "cell_total_erosion_catalog_residual_m3": total_cell_eroded - total_process_eroded,
        "floodplain_deposition_catalog_residual_m3": total_cell_deposited
        - float(np.sum(reach_deposited)),
        "maximum_reach_available_residual_m3": float(
            np.max(np.abs(available - upstream_input - local_erosion), initial=0.0)
        ),
        "maximum_reach_output_balance_residual_m3": float(
            np.max(
                np.abs(available - reach_deposited - transfer - terminal - exported),
                initial=0.0,
            )
        ),
        "maximum_downstream_input_residual_m3": float(
            np.max(np.abs(expected_upstream - upstream_input), initial=0.0)
        ),
        "maximum_parent_erosion_residual_m3": float(
            np.max(np.abs(eroded_by_parent - published_parent_eroded), initial=0.0)
        ),
        "maximum_parent_deposition_residual_m3": float(
            np.max(np.abs(deposited_by_parent - published_parent_deposited), initial=0.0)
        ),
    }


def _vector_catalog(
    profiles: pa.Table,
    reaches: pa.Table,
    cells: pa.Table,
    *,
    radius_m: float,
) -> tuple[pa.Table, dict[str, float | int]]:
    """First-class river vectors: polyline vertices with width and bed z."""

    if profiles.num_rows == 0:
        empty = pa.table(
            {
                "reach_id": pa.array([], type=pa.int32()),
                "path_order": pa.array([], type=pa.int32()),
                "fine_cell_id": pa.array([], type=pa.int32()),
                "xyz": pa.array([], type=pa.list_(pa.float32(), 3)),
                "terrain_elevation_m": pa.array([], type=pa.float64()),
                "bed_elevation_m": pa.array([], type=pa.float64()),
                "channel_width_m": pa.array([], type=pa.float32()),
                "segment_length_m": pa.array([], type=pa.float64()),
                "bed_slope_to_next": pa.array([], type=pa.float64()),
            }
        )
        return empty, {
            "vector_vertex_count": 0,
            "vector_reach_count": 0,
            "vector_minimum_bed_slope": 0.0,
            "vector_uphill_segment_count": 0,
        }

    reach_ids = np.asarray(profiles["reach_id"], dtype=np.int32)
    path_order = np.asarray(profiles["path_order"], dtype=np.int32)
    fine_ids = np.asarray(profiles["fine_cell_id"], dtype=np.int32)
    terrain = np.asarray(profiles["terrain_elevation_m"], dtype=np.float64)
    bed = np.asarray(profiles["bed_elevation_m"], dtype=np.float64)
    catalog_reach_ids = np.asarray(reaches["reach_id"], dtype=np.int32)
    width_by_row = _lookup_rows(catalog_reach_ids, reach_ids, label="vector reach width")
    widths = np.asarray(reaches["channel_width_m"], dtype=np.float32)[width_by_row]

    cell_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    cell_rows = _lookup_rows(cell_ids, fine_ids, label="vector cell xyz")
    xyz = _fixed_list_column(cells, "xyz", 3).astype(np.float64)[cell_rows]

    order = np.lexsort((path_order, reach_ids))
    reach_ids = reach_ids[order]
    path_order = path_order[order]
    fine_ids = fine_ids[order]
    terrain = terrain[order]
    bed = bed[order]
    widths = widths[order]
    xyz = xyz[order]

    segment_length = np.full(len(reach_ids), np.nan, dtype=np.float64)
    bed_slope = np.full(len(reach_ids), np.nan, dtype=np.float64)
    consecutive = (reach_ids[1:] == reach_ids[:-1]) & (path_order[1:] == path_order[:-1] + 1)
    consecutive_rows = np.flatnonzero(consecutive)
    if len(consecutive_rows):
        source = xyz[consecutive_rows]
        target = xyz[consecutive_rows + 1]
        dot = np.sum(source * target, axis=1)
        length_m = np.arccos(np.clip(dot, -1.0, 1.0)) * radius_m
        slopes = (bed[consecutive_rows] - bed[consecutive_rows + 1]) / length_m
        segment_length[consecutive_rows] = length_m
        bed_slope[consecutive_rows] = slopes

    finite_slopes = bed_slope[np.isfinite(bed_slope)]
    uphill = int(np.sum(finite_slopes < -1e-12)) if len(finite_slopes) else 0
    minimum_slope = float(np.min(finite_slopes)) if len(finite_slopes) else 0.0
    segment_mask = ~np.isfinite(segment_length)
    slope_mask = ~np.isfinite(bed_slope)
    table = pa.table(
        {
            "reach_id": pa.array(reach_ids, type=pa.int32()),
            "path_order": pa.array(path_order, type=pa.int32()),
            "fine_cell_id": pa.array(fine_ids, type=pa.int32()),
            "xyz": pa.FixedSizeListArray.from_arrays(
                pa.array(xyz.reshape(-1).astype(np.float32), type=pa.float32()),
                3,
            ),
            "terrain_elevation_m": pa.array(terrain, type=pa.float64()),
            "bed_elevation_m": pa.array(bed, type=pa.float64()),
            "channel_width_m": pa.array(widths, type=pa.float32()),
            # Null (not NaN) for terminal vertices so Arrow equality is stable.
            "segment_length_m": pa.array(segment_length, mask=segment_mask, type=pa.float64()),
            "bed_slope_to_next": pa.array(bed_slope, mask=slope_mask, type=pa.float64()),
        }
    )
    return table, {
        "vector_vertex_count": int(table.num_rows),
        "vector_reach_count": int(len(np.unique(reach_ids))),
        "vector_minimum_bed_slope": minimum_slope,
        "vector_uphill_segment_count": uphill,
    }


def _local_dem_low_audit(
    profiles: pa.Table,
    cells: pa.Table,
    memberships: pa.Table,
) -> dict[str, float | int]:
    """Require channel bed samples sit at or below valley-support bank DEM cells.

    Uses lateral valley memberships (zero in-cell reach length) for the same
    reach and nearby path order — not arbitrary D4 neighbors, which may be ocean
    or out-of-corridor terrain and are not process banks.
    """

    if profiles.num_rows == 0 or cells.num_rows == 0 or memberships.num_rows == 0:
        return {
            "local_dem_sample_count": 0,
            "local_dem_unchecked_profile_count": int(profiles.num_rows),
            "local_dem_low_violation_count": 0,
            "local_dem_low_violation_fraction": 0.0,
            "maximum_local_dem_excess_m": 0.0,
            "local_dem_low_valid": 1,
        }

    fine_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    cell_faces = np.asarray(cells["face"], dtype=np.int32)
    cell_rows = np.asarray(cells["row"], dtype=np.int32)
    cell_cols = np.asarray(cells["col"], dtype=np.int32)
    surface_column = (
        "channel_surface_prior_m"
        if "channel_surface_prior_m" in cells.column_names
        else "terrain_elevation_after_m"
    )
    terrain_after = np.asarray(cells[surface_column], dtype=np.float64)
    terrain_by_id = {
        int(cell_id): float(elevation)
        for cell_id, elevation in zip(fine_ids, terrain_after, strict=True)
    }
    coordinate_by_id = {
        int(cell_id): (int(face), int(row), int(col))
        for cell_id, face, row, col in zip(fine_ids, cell_faces, cell_rows, cell_cols, strict=True)
    }

    membership_reach = np.asarray(memberships["reach_id"], dtype=np.int32)
    membership_cell = np.asarray(memberships["fine_cell_id"], dtype=np.int32)
    membership_path = np.asarray(memberships["path_order"], dtype=np.int32)
    membership_length = np.asarray(memberships["reach_length_m"], dtype=np.float64)
    membership_valley = np.asarray(memberships["valley_fraction"], dtype=np.float64)
    lateral_mask = (membership_length <= 0.0) & (membership_valley > 0.0)
    laterals_by_reach_path: dict[tuple[int, int], list[int]] = {}
    for reach_id, cell_id, path_order in zip(
        membership_reach[lateral_mask],
        membership_cell[lateral_mask],
        membership_path[lateral_mask],
        strict=True,
    ):
        laterals_by_reach_path.setdefault((int(reach_id), int(path_order)), []).append(int(cell_id))

    profile_reach = np.asarray(profiles["reach_id"], dtype=np.int32)
    profile_path = np.asarray(profiles["path_order"], dtype=np.int32)
    profile_cell = np.asarray(profiles["fine_cell_id"], dtype=np.int32)
    profile_bed = np.asarray(profiles["bed_elevation_m"], dtype=np.float64)

    samples = 0
    violations = 0
    maximum_excess = 0.0
    for reach_id, path_order, centerline_cell, bed in zip(
        profile_reach, profile_path, profile_cell, profile_bed, strict=True
    ):
        centerline_coordinate = coordinate_by_id.get(int(centerline_cell))
        if centerline_coordinate is None:
            continue
        face, row, col = centerline_coordinate
        bank_elevations: list[float] = []
        for candidate_path in range(int(path_order) - 1, int(path_order) + 2):
            for lateral_cell in laterals_by_reach_path.get((int(reach_id), candidate_path), ()):
                coordinate = coordinate_by_id.get(lateral_cell)
                if coordinate is None:
                    continue
                lateral_face, lateral_row, lateral_col = coordinate
                # Lateral memberships are corridor support, not necessarily banks.
                # Only a true D4 neighbor on this face is a defensible DEM-low sample;
                # face-edge samples are left unchecked rather than misclassified.
                if lateral_face != face or abs(lateral_row - row) + abs(lateral_col - col) != 1:
                    continue
                bank_elevations.append(terrain_by_id[lateral_cell])
        if not bank_elevations:
            continue
        samples += 1
        local_bank_min = min(bank_elevations)
        excess = max(0.0, float(bed) - local_bank_min)
        maximum_excess = max(maximum_excess, excess)
        if excess > 1e-3:
            violations += 1

    return {
        "local_dem_sample_count": samples,
        "local_dem_unchecked_profile_count": int(profiles.num_rows) - samples,
        "local_dem_low_violation_count": violations,
        "local_dem_low_violation_fraction": violations / max(samples, 1),
        "maximum_local_dem_excess_m": maximum_excess,
        "local_dem_low_valid": int(violations == 0),
    }


def _cube_net_visualizer(result, request: VisualizationRequest) -> VisualizationResult | None:
    cell_record = result.artifact_records.get("ErodedBasinCellCatalog")
    profile_record = result.artifact_records.get("ChannelBedProfileCatalog")
    reach_record = result.artifact_records.get("FluvialRiverReachCatalog")
    metadata_record = result.artifact_records.get("BasinErosionMetadata")
    if (
        cell_record is None
        or profile_record is None
        or reach_record is None
        or metadata_record is None
        or not isinstance(cell_record.value, pa.Table)
        or not isinstance(profile_record.value, pa.Table)
        or not isinstance(reach_record.value, pa.Table)
    ):
        return None
    cells = cell_record.value
    profiles = profile_record.value
    reaches = reach_record.value
    metadata = metadata_record.value
    fine_resolution = int(metadata["fine_resolution"])
    display_resolution = min(fine_resolution, 768)
    placements = np.array([[1, 1], [1, 3], [1, 2], [1, 0], [0, 1], [2, 1]])
    image = np.full((display_resolution * 3, display_resolution * 4, 3), 18, dtype=np.uint8)
    face = np.asarray(cells["face"], dtype=np.int32)
    row = np.asarray(cells["row"], dtype=np.int32)
    col = np.asarray(cells["col"], dtype=np.int32)
    elevation = np.asarray(cells["terrain_elevation_after_m"], dtype=np.float64)
    low, high = np.percentile(elevation, [2.0, 98.0])
    normalized = np.clip((elevation - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    colors = np.stack(
        (52 + 153 * normalized, 80 + 132 * normalized, 51 + 101 * normalized), axis=1
    ).astype(np.uint8)
    display_row = np.minimum(row * display_resolution // fine_resolution, display_resolution - 1)
    display_col = np.minimum(col * display_resolution // fine_resolution, display_resolution - 1)
    net_row = placements[face, 0] * display_resolution + display_row
    net_col = placements[face, 1] * display_resolution + display_col
    image[net_row, net_col] = colors

    incision = np.asarray(cells["prospective_maximum_channel_incision_m"], dtype=np.float64)
    deposition = np.asarray(cells["prospective_floodplain_deposition_volume_m3"], dtype=np.float64)
    positive_incision = incision[incision > 0.0]
    incision_scale = (
        float(np.percentile(positive_incision, 95.0)) if len(positive_incision) else 1.0
    )
    positive_deposition = deposition[deposition > 0.0]
    deposition_scale = (
        float(np.percentile(positive_deposition, 95.0)) if len(positive_deposition) else 1.0
    )
    for index in np.flatnonzero((incision > 0.0) | (deposition > 0.0)):
        target = np.array((118, 77, 50), dtype=np.float32)
        strength = min(1.0, float(incision[index]) / max(incision_scale, 1e-9))
        if deposition[index] > 0.0:
            target = np.array((155, 151, 80), dtype=np.float32)
            strength = max(
                strength,
                min(1.0, float(deposition[index]) / max(deposition_scale, 1e-9)),
            )
        alpha = 0.18 + 0.38 * strength
        image[net_row[index], net_col[index]] = (
            image[net_row[index], net_col[index]].astype(np.float32) * (1.0 - alpha)
            + target * alpha
        ).astype(np.uint8)

    rendered = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(rendered)
    fine_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    for path, reach_kind in zip(
        reaches["fine_cell_path"].to_pylist(), reaches["reach_kind"].to_pylist(), strict=True
    ):
        if reach_kind != "connector":
            continue
        rows = _lookup_rows(
            fine_ids, np.asarray(path, dtype=np.int32), label="connector path visual"
        )
        path_face = face[rows]
        path_row = placements[path_face, 0] * display_resolution + display_row[rows]
        path_col = placements[path_face, 1] * display_resolution + display_col[rows]
        segment_start = 0
        for index in range(1, len(rows) + 1):
            if index == len(rows) or path_face[index] != path_face[index - 1]:
                if index - segment_start >= 2:
                    draw.line(
                        [
                            (int(path_col[position]), int(path_row[position]))
                            for position in range(segment_start, index)
                        ],
                        fill=(69, 114, 132),
                        width=1,
                    )
                segment_start = index
    profile_reaches = np.asarray(profiles["reach_id"], dtype=np.int32)
    profile_cells = np.asarray(profiles["fine_cell_id"], dtype=np.int32)
    profile_orders = np.asarray(profiles["path_order"], dtype=np.int32)
    for reach_id in np.unique(profile_reaches):
        mask = profile_reaches == reach_id
        sequence = np.flatnonzero(mask)[np.argsort(profile_orders[mask])]
        rows = _lookup_rows(fine_ids, profile_cells[sequence], label="bed profile visual")
        path_face = face[rows]
        path_row = placements[path_face, 0] * display_resolution + display_row[rows]
        path_col = placements[path_face, 1] * display_resolution + display_col[rows]
        segment_start = 0
        for index in range(1, len(rows) + 1):
            split = index == len(rows) or (
                profile_orders[sequence[index]] != profile_orders[sequence[index - 1]] + 1
                if index < len(rows)
                else True
            )
            if split:
                if index - segment_start >= 2:
                    draw.line(
                        [
                            (int(path_col[position]), int(path_row[position]))
                            for position in range(segment_start, index)
                        ],
                        fill=(26, 165, 221),
                        width=2,
                    )
                segment_start = index
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
    output = request.output_dir / "eroded_basin.png"
    rendered.save(output)
    local_erosion = np.asarray(reaches["local_erosion_volume_m3"], dtype=np.float64)
    diagnostic_reach = int(
        np.asarray(reaches["reach_id"], dtype=np.int32)[np.argmax(local_erosion)]
    )
    diagnostic_mask = profile_reaches == diagnostic_reach
    diagnostic_rows = np.flatnonzero(diagnostic_mask)[np.argsort(profile_orders[diagnostic_mask])]
    if len(diagnostic_rows) >= 2:
        distances_km = (
            np.cumsum(np.asarray(profiles["reach_length_m"], dtype=np.float64)[diagnostic_rows])
            / 1_000.0
        )
        terrain_profile = np.asarray(profiles["terrain_elevation_m"], dtype=np.float64)[
            diagnostic_rows
        ]
        bed_profile = np.asarray(profiles["bed_elevation_m"], dtype=np.float64)[diagnostic_rows]
        chart = Image.new("RGB", (1400, 720), (20, 22, 21))
        chart_draw = ImageDraw.Draw(chart)
        left_margin, right_margin = 88, 42
        top_margin, bottom_margin = 58, 72
        chart_width = chart.width - left_margin - right_margin
        chart_height = chart.height - top_margin - bottom_margin
        elevation_low = float(min(np.min(bed_profile), np.min(terrain_profile)))
        elevation_high = float(max(np.max(bed_profile), np.max(terrain_profile)))
        elevation_span = max(elevation_high - elevation_low, 1.0)
        distance_span = max(float(distances_km[-1] - distances_km[0]), 1e-9)

        def chart_points(values: np.ndarray) -> list[tuple[int, int]]:
            return [
                (
                    round(left_margin + (distance - distances_km[0]) / distance_span * chart_width),
                    round(top_margin + (elevation_high - value) / elevation_span * chart_height),
                )
                for distance, value in zip(distances_km, values, strict=True)
            ]

        for fraction in np.linspace(0.0, 1.0, 6):
            y = round(top_margin + fraction * chart_height)
            chart_draw.line(
                [(left_margin, y), (left_margin + chart_width, y)],
                fill=(48, 52, 49),
                width=1,
            )
            value = elevation_high - fraction * elevation_span
            chart_draw.text((12, y - 7), f"{value:.0f} m", fill=(160, 165, 158))
        terrain_points = chart_points(terrain_profile)
        bed_points = chart_points(bed_profile)
        chart_draw.polygon(terrain_points + list(reversed(bed_points)), fill=(72, 55, 39))
        chart_draw.line(terrain_points, fill=(216, 178, 101), width=3)
        chart_draw.line(bed_points, fill=(34, 178, 229), width=3)
        chart_draw.text(
            (left_margin, 20),
            f"Reach {diagnostic_reach}: terrain prior and least-incision bed",
            fill=(225, 229, 222),
        )
        chart_draw.text(
            (left_margin, chart.height - 42),
            f"{distances_km[0]:.0f} km",
            fill=(160, 165, 158),
        )
        chart_draw.text(
            (left_margin + chart_width - 56, chart.height - 42),
            f"{distances_km[-1]:.0f} km",
            fill=(160, 165, 158),
        )
        chart.save(request.output_dir / "longitudinal_profile.png")
    return VisualizationResult(
        output,
        "ErodedBasinCellCatalog",
        {
            "basin_id": metadata["selected_basin_id"],
            "eroded_volume_km3": metadata["total_eroded_volume_km3"],
        },
    )


@stage(
    "basin_erosion",
    inputs=("basin_refinement", "planet"),
    outputs=(
        "ChannelBedProfileCatalog",
        "FluvialRiverReachCatalog",
        "FluvialRiverVectorCatalog",
        "ErodedBasinCellCatalog",
        "BasinErosionParentCatalog",
        "BasinErosionMetadata",
    ),
    version="v8",
    native_libraries=("fluvial_native",),
    visualizer=_cube_net_visualizer,
)
def basin_erosion_stage(context, deps, config_mapping: Mapping[str, object]):
    config = BasinErosionConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("basin erosion requires topology: cubed_sphere")
    refinement = deps["basin_refinement"]
    refinement_metadata = refinement.artifact_records["BasinRefinementMetadata"].value
    if refinement_metadata["source_to_sink_ready"] != 1:
        raise RuntimeError("basin erosion requires a source-to-sink-ready refined reach graph")
    for gate in (
        "directed_path_graph_valid",
        "corridor_cell_capacity_valid",
        "nested_corridor_support_valid",
        "process_exclusion_valid",
    ):
        if refinement_metadata[gate] != 1:
            raise RuntimeError(f"basin erosion requires refinement gate {gate}")
    cells = _artifact_table(refinement, "RefinedBasinCellCatalog")
    parents = _artifact_table(refinement, "RefinedBasinParentCatalog")
    reaches = _artifact_table(refinement, "RefinedRiverReachCatalog")
    memberships = _artifact_table(refinement, "RefinedReachCellCatalog")
    reach_kind_names = reaches["reach_kind"].to_pylist()
    reach_kinds = np.ascontiguousarray(
        [
            1 if value == "channel" else 2 if value == "connector" else 0
            for value in reach_kind_names
        ],
        dtype=np.uint8,
    )
    if np.any(reach_kinds == 0):
        raise RuntimeError("basin erosion received an unknown reach kind")
    terminal_kind_names = reaches["terminal_kind"].to_pylist()
    terminal_kinds = np.ascontiguousarray(
        [
            (
                0
                if value == "downstream_reach"
                else 1 if value == "ocean" else 2 if value == "registered_sink" else 255
            )
            for value in terminal_kind_names
        ],
        dtype=np.uint8,
    )
    if np.any(terminal_kinds == 255):
        raise RuntimeError("basin erosion received an unresolved reach terminal")
    reach_ids = _column(reaches, "reach_id", np.dtype(np.int32))
    membership_reach_ids = _column(memberships, "reach_id", np.dtype(np.int32))
    membership_parent_ids = _column(memberships, "parent_cell_id", np.dtype(np.int32))
    connector = reach_kinds == 2
    connector_ids = reach_ids[connector]
    if np.any(np.isin(membership_reach_ids, connector_ids)):
        raise RuntimeError("hydrologic connectors must not publish physical cell support")
    excluded_parent_ids = np.asarray(
        parents.filter(parents["process_excluded"])["parent_cell_id"], dtype=np.int32
    )
    if np.any(np.isin(membership_parent_ids, excluded_parent_ids)):
        raise RuntimeError("refined reach support enters a process-excluded parent")
    planet = deps["planet"].artifact_records["PlanetMetadata"].value
    radius_m = float(planet["planet_radius_earth"]) * EARTH_RADIUS_M
    controls = {
        "planet_radius_m": radius_m,
        "minimum_bed_slope": config.minimum_bed_slope,
        "maximum_deposition_fraction": config.maximum_deposition_fraction,
        "deposition_slope_scale": config.deposition_slope_scale,
        "maximum_deposition_depth_m": config.maximum_deposition_depth_m,
        # Coarse child cells are still about 4.5 km across at the canonical
        # factor. Physical bank and valley erosion belongs to regional terrain.
        "bank_incision_fraction": 0.0,
    }
    with context.timed("conservative_fluvial_erosion_kernel"):
        profile_records, reach_records, cell_records, metadata = run_fluvial_erosion(
            controls=controls,
            cell_ids=_column(cells, "fine_cell_id", np.dtype(np.int32)),
            cell_parent_ids=_column(cells, "parent_cell_id", np.dtype(np.int32)),
            cell_terrain_m=_column(cells, "channel_surface_prior_m", np.dtype(np.float32)),
            cell_areas_km2=_column(cells, "area_km2", np.dtype(np.float64)),
            cell_xyz=_fixed_list_column(cells, "xyz", 3),
            reach_ids=reach_ids,
            downstream_reach_ids=_column(reaches, "downstream_reach_id", np.dtype(np.int32)),
            reach_kinds=reach_kinds,
            terminal_kinds=terminal_kinds,
            channel_width_m=_column(reaches, "channel_width_m", np.dtype(np.float32)),
            reach_slope=_column(reaches, "slope", np.dtype(np.float32)),
            membership_reach_ids=membership_reach_ids,
            membership_cell_ids=_column(memberships, "fine_cell_id", np.dtype(np.int32)),
            membership_parent_ids=membership_parent_ids,
            membership_path_order=_column(memberships, "path_order", np.dtype(np.int32)),
            membership_reach_length_m=_column(memberships, "reach_length_m", np.dtype(np.float64)),
            membership_channel_fraction=_column(
                memberships, "channel_fraction", np.dtype(np.float32)
            ),
            membership_valley_fraction=_column(
                memberships, "valley_fraction", np.dtype(np.float32)
            ),
            membership_floodplain_fraction=_column(
                memberships, "floodplain_fraction", np.dtype(np.float32)
            ),
        )
    profiles = _profile_table(profile_records)
    eroded_reaches = _reach_table(reaches, reach_records)
    eroded_cells, cell_metadata = _cell_table(cells, cell_records)
    eroded_parents, parent_metadata = _parent_table(parents, eroded_cells)
    vectors, vector_audit = _vector_catalog(
        profiles, eroded_reaches, eroded_cells, radius_m=radius_m
    )
    profile_audit = _profile_audit(profiles, eroded_reaches, eroded_cells, radius_m=radius_m)
    budget_audit = _budget_audit(profiles, eroded_reaches, eroded_cells, eroded_parents)
    dem_low_audit = _local_dem_low_audit(profiles, eroded_cells, memberships)
    cell_process = (cell_records["eroded_volume_m3"] != 0.0) | (
        cell_records["deposited_volume_m3"] != 0.0
    )
    process_exclusion_valid = bool(
        not np.any(np.isin(profile_records["parent_cell_id"], excluded_parent_ids))
        and not np.any(np.isin(cell_records["parent_cell_id"][cell_process], excluded_parent_ids))
    )
    connector_process_valid = bool(
        np.all(reach_records["has_physical_bed"][connector] == 0)
        and np.all(reach_records["local_erosion_volume_m3"][connector] == 0.0)
        and np.all(reach_records["floodplain_deposition_volume_m3"][connector] == 0.0)
        and np.all(reach_records["bank_eroded_volume_m3"][connector] == 0.0)
        and not np.any(np.isin(profile_records["reach_id"], connector_ids))
    )
    potential_volume_m3 = float(
        np.sum(np.asarray(memberships["potential_incised_volume_m3"], dtype=np.float64))
    )
    total_eroded_m3 = float(budget_audit["independent_total_eroded_volume_m3"])
    total_bank_m3 = float(budget_audit["independent_total_bank_eroded_volume_m3"])
    total_channel_m3 = float(budget_audit["independent_total_channel_eroded_volume_m3"])
    bank_carve_valid = bool(total_bank_m3 == 0.0)
    incision_plausibility_valid = bool(
        float(metadata["maximum_incision_depth_m"]) <= config.hard_maximum_channel_incision_m
    )
    raster_feedback_valid = bool(
        cell_metadata["maximum_absolute_cell_mean_delta_m"] == 0.0
        and parent_metadata["maximum_absolute_parent_mean_delta_m"] == 0.0
        and np.array_equal(
            np.asarray(eroded_cells["terrain_elevation_after_m"], dtype=np.float64),
            np.asarray(eroded_cells["terrain_elevation_m"], dtype=np.float64),
        )
    )
    native_catalog_residuals = {
        "native_total_eroded_catalog_residual_m3": float(metadata["total_eroded_volume_m3"])
        - total_eroded_m3,
        "native_channel_eroded_catalog_residual_m3": float(
            metadata.get("total_channel_eroded_volume_m3", total_channel_m3)
        )
        - total_channel_m3,
        "native_bank_eroded_catalog_residual_m3": float(
            metadata.get("total_bank_eroded_volume_m3", total_bank_m3)
        )
        - total_bank_m3,
        "native_floodplain_deposition_catalog_residual_m3": float(
            metadata["total_floodplain_deposition_volume_m3"]
        )
        - float(budget_audit["independent_total_floodplain_deposition_volume_m3"]),
        "native_terminal_deposition_catalog_residual_m3": float(
            metadata["total_terminal_deposition_volume_m3"]
        )
        - float(budget_audit["independent_total_terminal_deposition_volume_m3"]),
        "native_export_catalog_residual_m3": float(metadata["total_exported_sediment_volume_m3"])
        - float(budget_audit["independent_total_exported_sediment_volume_m3"]),
    }
    metadata.update(
        {
            **asdict(config),
            **cell_metadata,
            **parent_metadata,
            **profile_audit,
            **budget_audit,
            **vector_audit,
            **dem_low_audit,
            **native_catalog_residuals,
            "selected_basin_id": int(refinement_metadata["selected_basin_id"]),
            "fine_resolution": int(refinement_metadata["fine_resolution"]),
            "source_to_sink_ready": 1,
            "process_exclusion_valid": int(process_exclusion_valid),
            "connector_process_valid": int(connector_process_valid),
            "bank_carve_valid": int(bank_carve_valid),
            "incision_plausibility_valid": int(incision_plausibility_valid),
            "raster_feedback_valid": int(raster_feedback_valid),
            "bank_carve_enabled": 0,
            "potential_incised_volume_km3": potential_volume_m3 / 1_000_000_000.0,
            "actual_to_potential_incision_ratio": total_eroded_m3 / max(potential_volume_m3, 1e-12),
            "prospective_total_excavation_volume_km3": total_eroded_m3 / 1_000_000_000.0,
            "prospective_channel_excavation_volume_km3": total_channel_m3 / 1_000_000_000.0,
            "applied_terrain_erosion_volume_km3": 0.0,
            "applied_terrain_deposition_volume_km3": 0.0,
            "total_eroded_volume_km3": total_eroded_m3 / 1_000_000_000.0,
            "total_channel_eroded_volume_km3": total_channel_m3 / 1_000_000_000.0,
            "total_bank_eroded_volume_km3": total_bank_m3 / 1_000_000_000.0,
            "total_floodplain_deposition_volume_km3": float(
                budget_audit["independent_total_floodplain_deposition_volume_m3"]
            )
            / 1_000_000_000.0,
            "total_terminal_deposition_volume_km3": float(
                budget_audit["independent_total_terminal_deposition_volume_m3"]
            )
            / 1_000_000_000.0,
            "total_exported_sediment_volume_km3": float(
                budget_audit["independent_total_exported_sediment_volume_m3"]
            )
            / 1_000_000_000.0,
            "profile_semantics": "prospective_least_incision_vector_constraint",
            "profile_surface_prior_semantics": (
                "priority_flood_control_surface_over_unresolved_fill_with_registered_"
                "water_surface_override_else_refined_terrain"
            ),
            "incision_semantics": (
                "prospective_channel_prism_without_bank_carve_or_raster_terrain_feedback"
            ),
            "vector_semantics": "polyline_vertices_with_channel_width_and_bed_elevation",
            "sediment_semantics": "prospective_solid_volume_budget_routed_source_to_sink",
            "inherited_sediment_load_semantics": "preserved_flux_diagnostic_not_mixed_with_history_volume",
            "regional_refinement_owns_physical_incision": 1,
        }
    )
    tolerance_m3 = 1e-8 * max(total_eroded_m3, 1.0)
    if (
        metadata["bed_profile_valid"] != 1
        or profile_audit["maximum_junction_bed_error_m"] > 1e-10
        or profile_audit["maximum_bed_above_terrain_error_m"] > 1e-10
        or profile_audit["maximum_profile_depth_relation_error_m"] > 1e-10
        or profile_audit["emitted_physical_edge_count"] != metadata["physical_edge_count"]
    ):
        raise RuntimeError(
            "fluvial kernel produced an invalid or junction-discontinuous bed profile"
        )
    if profile_audit["emitted_minimum_realized_slope"] < config.minimum_bed_slope - 1e-12:
        raise RuntimeError("fluvial bed profile violates the configured downstream grade")
    if vector_audit["vector_uphill_segment_count"] != 0:
        raise RuntimeError("fluvial river vectors contain uphill bed segments")
    if vector_audit["vector_minimum_bed_slope"] < config.minimum_bed_slope - 1e-12:
        raise RuntimeError("fluvial river vectors violate the configured downstream grade")
    if not bank_carve_valid:
        raise RuntimeError("sparse coarse-scale fluvial stage applied forbidden bank erosion")
    if not incision_plausibility_valid:
        raise RuntimeError(
            "fluvial incision exceeds the configured physical plausibility envelope: "
            f"channel={metadata['maximum_incision_depth_m']:.1f} m / "
            f"{config.hard_maximum_channel_incision_m:.1f} m"
        )
    if not raster_feedback_valid:
        raise RuntimeError("sparse coarse-scale fluvial stage changed raster terrain")
    if (
        metadata["sediment_conservation_valid"] != 1
        or abs(metadata["sediment_conservation_residual_m3"]) > tolerance_m3
    ):
        raise RuntimeError("fluvial sediment routing does not conserve eroded volume")
    if abs(cell_metadata["cell_mean_volume_relation_residual_m3"]) > tolerance_m3:
        raise RuntimeError("fluvial cell mean changes do not match physical process volume")
    if abs(parent_metadata["parent_budget_residual_m3"]) > tolerance_m3:
        raise RuntimeError("fluvial child process volumes do not restrict to parent budgets")
    cross_catalog_residuals = (
        profile_audit["maximum_profile_volume_relation_residual_m3"],
        budget_audit["independent_sediment_conservation_residual_m3"],
        budget_audit["maximum_reach_local_erosion_residual_m3"],
        budget_audit["maximum_reach_channel_erosion_residual_m3"],
        budget_audit["maximum_cell_channel_erosion_residual_m3"],
        budget_audit["cell_total_erosion_catalog_residual_m3"],
        budget_audit["floodplain_deposition_catalog_residual_m3"],
        budget_audit["maximum_reach_available_residual_m3"],
        budget_audit["maximum_reach_output_balance_residual_m3"],
        budget_audit["maximum_downstream_input_residual_m3"],
        budget_audit["maximum_parent_erosion_residual_m3"],
        budget_audit["maximum_parent_deposition_residual_m3"],
        *native_catalog_residuals.values(),
    )
    if any(abs(float(residual)) > tolerance_m3 for residual in cross_catalog_residuals):
        raise RuntimeError("published fluvial catalogs disagree on process volume or routing")
    if not process_exclusion_valid:
        raise RuntimeError("fluvial process volume enters a process-excluded parent")
    if not connector_process_valid:
        raise RuntimeError("hydrologic connector acquired physical erosion or deposition")
    context.logger.log_event(
        {"type": "basin_erosion_summary", "stage": "basin_erosion", **metadata}
    )
    return {
        "ChannelBedProfileCatalog": profiles,
        "FluvialRiverReachCatalog": eroded_reaches,
        "FluvialRiverVectorCatalog": vectors,
        "ErodedBasinCellCatalog": eroded_cells,
        "BasinErosionParentCatalog": eroded_parents,
        "BasinErosionMetadata": metadata,
    }


__all__ = ["BasinErosionConfig", "basin_erosion_stage"]
