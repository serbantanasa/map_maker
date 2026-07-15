"""Conservative fluvial incision and sediment routing for one refined basin."""

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
        table = table.append_column(
            name,
            pa.array(records[name], mask=~has_bed, type=pa.float64()),
        )
    for name in (
        "upstream_input_volume_m3",
        "local_erosion_volume_m3",
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
    eroded = np.zeros(source.num_rows, dtype=np.float64)
    deposited = np.zeros(source.num_rows, dtype=np.float64)
    maximum_incision = np.zeros(source.num_rows, dtype=np.float64)
    if len(records):
        rows = _lookup_rows(fine_ids, records["fine_cell_id"], label="fluvial cell budget")
        if not np.array_equal(
            np.asarray(source["parent_cell_id"], dtype=np.int32)[rows],
            records["parent_cell_id"],
        ):
            raise RuntimeError("fluvial cell budget changed parent identity")
        eroded[rows] = records["eroded_volume_m3"]
        deposited[rows] = records["deposited_volume_m3"]
        maximum_incision[rows] = records["maximum_incision_depth_m"]
    mean_delta = (deposited - eroded) / areas_m2
    terrain_after = terrain.astype(np.float64) + mean_delta
    table = source.append_column("subgrid_eroded_volume_m3", pa.array(eroded, type=pa.float64()))
    table = table.append_column(
        "floodplain_deposited_volume_m3", pa.array(deposited, type=pa.float64())
    )
    table = table.append_column(
        "maximum_channel_incision_m", pa.array(maximum_incision, type=pa.float64())
    )
    table = table.append_column("terrain_mean_delta_m", pa.array(mean_delta, type=pa.float64()))
    table = table.append_column(
        "terrain_elevation_after_m", pa.array(terrain_after, type=pa.float64())
    )
    represented_delta_volume = float(np.sum(mean_delta * areas_m2))
    expected_delta_volume = float(np.sum(deposited) - np.sum(eroded))
    return table, {
        "maximum_absolute_cell_mean_delta_m": float(np.max(np.abs(mean_delta), initial=0.0)),
        "cell_mean_volume_relation_residual_m3": represented_delta_volume - expected_delta_volume,
    }


def _parent_table(source: pa.Table, cells: pa.Table) -> tuple[pa.Table, dict[str, float]]:
    parent_ids = np.asarray(source["parent_cell_id"], dtype=np.int32)
    child_parent_ids = np.asarray(cells["parent_cell_id"], dtype=np.int32)
    parent_rows = _lookup_rows(parent_ids, child_parent_ids, label="eroded child parent")
    eroded = np.zeros(source.num_rows, dtype=np.float64)
    deposited = np.zeros(source.num_rows, dtype=np.float64)
    np.add.at(
        eroded,
        parent_rows,
        np.asarray(cells["subgrid_eroded_volume_m3"], dtype=np.float64),
    )
    np.add.at(
        deposited,
        parent_rows,
        np.asarray(cells["floodplain_deposited_volume_m3"], dtype=np.float64),
    )
    area_m2 = np.asarray(source["parent_area_km2"], dtype=np.float64) * 1_000_000.0
    mean_delta = (deposited - eroded) / area_m2
    table = source.append_column("child_eroded_volume_m3", pa.array(eroded, type=pa.float64()))
    table = table.append_column("child_deposited_volume_m3", pa.array(deposited, type=pa.float64()))
    table = table.append_column(
        "restricted_terrain_mean_delta_m", pa.array(mean_delta, type=pa.float64())
    )
    child_net = float(
        np.sum(np.asarray(cells["floodplain_deposited_volume_m3"], dtype=np.float64))
        - np.sum(np.asarray(cells["subgrid_eroded_volume_m3"], dtype=np.float64))
    )
    return table, {
        "parent_budget_residual_m3": float(np.sum(deposited - eroded)) - child_net,
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
    edge_sources = sort_order[:-1][edge_mask]
    edge_targets = sort_order[1:][edge_mask]
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
    profile_by_reach = _group_sums(
        reach_ids, profile_reach_ids, profile_volume, label="profile erosion reach"
    )

    fine_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    profile_cell_ids = np.asarray(profiles["fine_cell_id"], dtype=np.int32)
    profile_by_cell = _group_sums(
        fine_ids, profile_cell_ids, profile_volume, label="profile erosion cell"
    )
    cell_eroded = np.asarray(cells["subgrid_eroded_volume_m3"], dtype=np.float64)
    cell_deposited = np.asarray(cells["floodplain_deposited_volume_m3"], dtype=np.float64)
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
    published_parent_eroded = np.asarray(parents["child_eroded_volume_m3"], dtype=np.float64)
    published_parent_deposited = np.asarray(parents["child_deposited_volume_m3"], dtype=np.float64)

    total_profile_eroded = float(np.sum(profile_volume))
    total_cell_deposited = float(np.sum(cell_deposited))
    independent_sediment_residual = (
        total_profile_eroded
        - total_cell_deposited
        - float(np.sum(terminal))
        - float(np.sum(exported))
    )
    return {
        "independent_total_eroded_volume_m3": total_profile_eroded,
        "independent_total_floodplain_deposition_volume_m3": total_cell_deposited,
        "independent_total_terminal_deposition_volume_m3": float(np.sum(terminal)),
        "independent_total_exported_sediment_volume_m3": float(np.sum(exported)),
        "independent_sediment_conservation_residual_m3": independent_sediment_residual,
        "maximum_reach_local_erosion_residual_m3": float(
            np.max(np.abs(profile_by_reach - local_erosion), initial=0.0)
        ),
        "maximum_cell_erosion_residual_m3": float(
            np.max(np.abs(profile_by_cell - cell_eroded), initial=0.0)
        ),
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

    incision = np.asarray(cells["maximum_channel_incision_m"], dtype=np.float64)
    deposition = np.asarray(cells["floodplain_deposited_volume_m3"], dtype=np.float64)
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
        "ErodedBasinCellCatalog",
        "BasinErosionParentCatalog",
        "BasinErosionMetadata",
    ),
    version="v1",
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
    controls = {**asdict(config), "planet_radius_m": radius_m}
    with context.timed("conservative_fluvial_erosion_kernel"):
        profile_records, reach_records, cell_records, metadata = run_fluvial_erosion(
            controls=controls,
            cell_ids=_column(cells, "fine_cell_id", np.dtype(np.int32)),
            cell_parent_ids=_column(cells, "parent_cell_id", np.dtype(np.int32)),
            cell_terrain_m=_column(cells, "terrain_elevation_m", np.dtype(np.float32)),
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
    profile_audit = _profile_audit(profiles, eroded_reaches, eroded_cells, radius_m=radius_m)
    budget_audit = _budget_audit(profiles, eroded_reaches, eroded_cells, eroded_parents)
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
        and not np.any(np.isin(profile_records["reach_id"], connector_ids))
    )
    potential_volume_m3 = float(
        np.sum(np.asarray(memberships["potential_incised_volume_m3"], dtype=np.float64))
    )
    total_eroded_m3 = float(budget_audit["independent_total_eroded_volume_m3"])
    native_catalog_residuals = {
        "native_total_eroded_catalog_residual_m3": float(metadata["total_eroded_volume_m3"])
        - total_eroded_m3,
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
            **native_catalog_residuals,
            "selected_basin_id": int(refinement_metadata["selected_basin_id"]),
            "fine_resolution": int(refinement_metadata["fine_resolution"]),
            "source_to_sink_ready": 1,
            "process_exclusion_valid": int(process_exclusion_valid),
            "connector_process_valid": int(connector_process_valid),
            "potential_incised_volume_km3": potential_volume_m3 / 1_000_000_000.0,
            "actual_to_potential_incision_ratio": total_eroded_m3 / max(potential_volume_m3, 1e-12),
            "total_eroded_volume_km3": total_eroded_m3 / 1_000_000_000.0,
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
            "profile_semantics": "least_incision_downstream_monotone_bedrock_envelope",
            "incision_semantics": "subgrid_channel_volume_with_cell_mean_volume_feedback",
            "sediment_semantics": "newly_eroded_solid_volume_routed_source_to_sink",
            "inherited_sediment_load_semantics": "preserved_flux_diagnostic_not_mixed_with_history_volume",
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
        budget_audit["maximum_cell_erosion_residual_m3"],
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
        "ErodedBasinCellCatalog": eroded_cells,
        "BasinErosionParentCatalog": eroded_parents,
        "BasinErosionMetadata": metadata,
    }


__all__ = ["BasinErosionConfig", "basin_erosion_stage"]
