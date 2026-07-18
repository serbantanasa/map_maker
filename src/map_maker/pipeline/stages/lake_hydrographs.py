"""Conservatively couple final lake-network overflow into river hydrographs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, NamedTuple

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from ..models import StageResult
from ..registry import stage

if TYPE_CHECKING:
    from ..execution import PipelineContext

MONTHS = 12
SECONDS_PER_MONTH = 365.2425 * 86_400.0 / MONTHS


@dataclass(frozen=True)
class LakeHydrographConfig:
    maximum_balance_relative_error: float = 1e-9
    maximum_negative_discharge_m3s: float = 1e-6

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "LakeHydrographConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown lake-hydrograph controls: {', '.join(sorted(unknown))}")

        def control(name: str) -> float:
            raw = mapping.get(name, cls.__dataclass_fields__[name].default)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{name} must be numeric")
            return float(raw)

        config = cls(
            maximum_balance_relative_error=control("maximum_balance_relative_error"),
            maximum_negative_discharge_m3s=control("maximum_negative_discharge_m3s"),
        )
        if not 1e-14 <= config.maximum_balance_relative_error <= 0.01:
            raise ValueError("maximum_balance_relative_error must be in [1e-14, 0.01]")
        if not 0.0 <= config.maximum_negative_discharge_m3s <= 1.0:
            raise ValueError("maximum_negative_discharge_m3s must be in [0, 1]")
        return config


class _ReachTarget(NamedTuple):
    reach_id: int
    joins_at_entry: bool
    routing_mode: str
    routed_cell_id: int


def _artifact_table(result: StageResult, name: str) -> pa.Table:
    record = result.artifact_records.get(name)
    if record is None or not isinstance(record.value, pa.Table):
        raise KeyError(f"Missing dependency table '{name}'")
    return record.value


def _artifact_array(result: StageResult, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)  # type: ignore[no-any-return]


def _column(table: pa.Table, name: str, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(  # type: ignore[no-any-return]
        table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
    )


def _fixed_list(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().values, dtype=np.float64).reshape(  # type: ignore[no-any-return]
        table.num_rows, MONTHS
    )


def _fixed_list_array(values: np.ndarray) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(values, dtype=np.float64).reshape(-1), type=pa.float64()), MONTHS
    )


def _network_roots(
    candidate_ids: np.ndarray, downstream_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    row_by_id = {int(value): row for row, value in enumerate(candidate_ids)}
    if len(row_by_id) != len(candidate_ids):
        raise RuntimeError("surface-water candidates contain duplicate identifiers")
    roots: np.ndarray = np.full(len(candidate_ids), -1, dtype=np.int32)
    state: np.ndarray = np.zeros(len(candidate_ids), dtype=np.uint8)
    for start in range(len(candidate_ids)):
        if state[start] == 2:
            continue
        trail: list[int] = []
        position: dict[int, int] = {}
        row = start
        while state[row] != 2 and row not in position:
            position[row] = len(trail)
            trail.append(row)
            state[row] = 1
            downstream = int(downstream_ids[row])
            if downstream < 0:
                root = row
                break
            if downstream not in row_by_id:
                raise RuntimeError("surface-water candidate references an unknown downstream ID")
            row = row_by_id[downstream]
        else:
            if row in position:
                raise RuntimeError("surface-water candidate graph contains a cycle")
            root = int(roots[row])
        for trail_row in reversed(trail):
            roots[trail_row] = root
            state[trail_row] = 2
    terminal_rows: np.ndarray = np.flatnonzero(downstream_ids < 0).astype(np.int32)
    if np.any(roots < 0) or not np.all(np.isin(roots, terminal_rows)):
        raise RuntimeError("surface-water candidate graph has an unresolved terminal")
    return roots, terminal_rows


def _reach_path(
    start_reach_id: int,
    downstream_reach_ids: np.ndarray,
    reach_row_by_id: Mapping[int, int],
) -> list[int]:
    path: list[int] = []
    seen: set[int] = set()
    reach_id = start_reach_id
    while reach_id >= 0:
        if reach_id in seen:
            raise RuntimeError("river reach graph contains a cycle")
        if reach_id not in reach_row_by_id:
            raise RuntimeError("river reach references an unknown downstream reach")
        seen.add(reach_id)
        path.append(reach_id)
        reach_id = int(downstream_reach_ids[reach_row_by_id[reach_id]])
    return path


def _routing_maps(
    reaches: pa.Table,
) -> tuple[
    dict[int, list[tuple[int, int]]],
    dict[int, int],
    dict[int, list[tuple[int, int]]],
    dict[int, int],
]:
    reach_ids = _column(reaches, "reach_id", np.dtype(np.int32))
    reach_kinds = reaches["reach_kind"].to_pylist()
    entry_cells = _column(reaches, "entry_fine_cell", np.dtype(np.int32))
    from_parents = _column(reaches, "from_parent_cell", np.dtype(np.int32))
    fine_members: dict[int, list[tuple[int, int]]] = {}
    parent_members: dict[int, list[tuple[int, int]]] = {}
    fine_entries: dict[int, int] = {}
    parent_entries: dict[int, int] = {}
    for reach_id, reach_kind, entry_cell, from_parent, fine_path, parent_path in zip(
        reach_ids,
        reach_kinds,
        entry_cells,
        from_parents,
        reaches["fine_cell_path"].to_pylist(),
        reaches["parent_cell_path"].to_pylist(),
        strict=True,
    ):
        rid = int(reach_id)
        if reach_kind == "channel":
            fine_entries[int(entry_cell)] = rid
            for order, cell_id in enumerate(fine_path):
                fine_members.setdefault(int(cell_id), []).append((rid, order))
        parent_entries[int(from_parent)] = rid
        for order, parent_id in enumerate(parent_path):
            parent_members.setdefault(int(parent_id), []).append((rid, order))
    return fine_members, fine_entries, parent_members, parent_entries


def _choose_member(members: list[tuple[int, int]]) -> int:
    return max(members, key=lambda value: value[1])[0]


def _route_target(
    spill_receiver_id: int,
    *,
    cell_row_by_id: Mapping[int, int],
    receiver_ids: np.ndarray,
    parent_ids: np.ndarray,
    anchor_kinds: list[str],
    coarse_receiver_ids: np.ndarray,
    fine_members: Mapping[int, list[tuple[int, int]]],
    fine_entries: Mapping[int, int],
    parent_members: Mapping[int, list[tuple[int, int]]],
    parent_entries: Mapping[int, int],
) -> _ReachTarget:
    cell_id = spill_receiver_id
    seen: set[int] = set()
    while cell_id >= 0:
        if cell_id in seen:
            raise RuntimeError("lake outlet routing encountered a fine-cell cycle")
        seen.add(cell_id)
        if cell_id in fine_members:
            if cell_id in fine_entries:
                return _ReachTarget(fine_entries[cell_id], True, "fine_channel", cell_id)
            return _ReachTarget(
                _choose_member(fine_members[cell_id]), False, "fine_channel", cell_id
            )
        if cell_id not in cell_row_by_id:
            raise RuntimeError("lake outlet references a cell outside the refined domain")
        row = cell_row_by_id[cell_id]
        receiver = int(receiver_ids[row])
        if receiver >= 0:
            cell_id = receiver
            continue
        if anchor_kinds[row] == "outside_terminal":
            return _ReachTarget(-1, False, "outside_terminal", cell_id)
        if anchor_kinds[row] != "preserved_handoff":
            raise RuntimeError("lake outlet ended at an unresolved refined routing anchor")
        parent_id = int(parent_ids[row])
        coarse_seen: set[int] = set()
        while parent_id >= 0 and parent_id not in parent_members:
            if parent_id in coarse_seen or parent_id >= len(coarse_receiver_ids):
                raise RuntimeError("lake outlet routing encountered an invalid coarse path")
            coarse_seen.add(parent_id)
            parent_id = int(coarse_receiver_ids[parent_id])
        if parent_id < 0:
            return _ReachTarget(-1, False, "outside_terminal", cell_id)
        if parent_id in parent_entries:
            return _ReachTarget(parent_entries[parent_id], True, "preserved_handoff", cell_id)
        return _ReachTarget(
            _choose_member(parent_members[parent_id]),
            False,
            "preserved_handoff",
            cell_id,
        )
    return _ReachTarget(-1, False, "outside_terminal", spill_receiver_id)


def _find_effective_start(
    path_rows: list[int],
    month: int,
    delta_m3s: float,
    joins_at_entry: bool,
    base_entry: np.ndarray,
    base_exit: np.ndarray,
    entry_adjustment: np.ndarray,
    exit_adjustment: np.ndarray,
    tolerance_m3s: float,
) -> int:
    if delta_m3s >= 0.0:
        return 0
    required = -delta_m3s
    for start in range(len(path_rows)):
        exit_available = min(
            base_exit[row, month] + exit_adjustment[row, month] for row in path_rows[start:]
        )
        entry_start = start if joins_at_entry and start == 0 else start + 1
        entry_available = min(
            (
                base_entry[row, month] + entry_adjustment[row, month]
                for row in path_rows[entry_start:]
            ),
            default=np.inf,
        )
        if min(exit_available, entry_available) + tolerance_m3s >= required:
            return start
    raise RuntimeError("lake loss exceeds available downstream river discharge")


def _couple_hydrographs(
    config: LakeHydrographConfig,
    candidates: pa.Table,
    cells: pa.Table,
    reaches: pa.Table,
    coarse_receiver_ids: np.ndarray,
    coarse_monthly_discharge: np.ndarray,
) -> tuple[pa.Table, pa.Table, dict[str, object]]:
    candidate_ids = _column(candidates, "depression_id", np.dtype(np.int32))
    downstream_candidate_ids = _column(candidates, "downstream_depression_id", np.dtype(np.int32))
    roots, terminal_rows = _network_roots(candidate_ids, downstream_candidate_ids)
    direct_inflow = _fixed_list(candidates, "monthly_direct_inflow_km3")
    overflow = _fixed_list(candidates, "monthly_overflow_km3")
    network_direct = np.zeros_like(direct_inflow)
    np.add.at(network_direct, roots, direct_inflow)

    reach_ids = _column(reaches, "reach_id", np.dtype(np.int32))
    downstream_reach_ids = _column(reaches, "downstream_reach_id", np.dtype(np.int32))
    reach_row_by_id = {int(value): row for row, value in enumerate(reach_ids)}
    if len(reach_row_by_id) != len(reach_ids):
        raise RuntimeError("river reach catalog contains duplicate identifiers")
    base_entry = _fixed_list(reaches, "discharge_seasonal")
    to_parent_ids = _column(reaches, "to_parent_cell", np.dtype(np.int32))
    exit_sample_ids = to_parent_ids.copy()
    for row, (terminal_kind, parent_path) in enumerate(
        zip(
            reaches["terminal_kind"].to_pylist(),
            reaches["parent_cell_path"].to_pylist(),
            strict=True,
        )
    ):
        if terminal_kind == "ocean" and len(parent_path) >= 2:
            exit_sample_ids[row] = int(parent_path[-2])
    if np.any(exit_sample_ids < 0) or np.any(exit_sample_ids >= coarse_monthly_discharge.shape[1]):
        raise RuntimeError("river reach exit references discharge outside the coarse grid")
    base_exit = np.asarray(coarse_monthly_discharge[:, exit_sample_ids].T, dtype=np.float64)
    entry_adjustment = np.zeros_like(base_entry)
    exit_adjustment = np.zeros_like(base_exit)

    cell_ids = _column(cells, "fine_cell_id", np.dtype(np.int32))
    cell_row_by_id = {int(value): row for row, value in enumerate(cell_ids)}
    if len(cell_row_by_id) != len(cell_ids):
        raise RuntimeError("refined hydrology contains duplicate cell identifiers")
    receiver_ids = _column(cells, "stabilized_receiver_id", np.dtype(np.int32))
    parent_ids = _column(cells, "parent_cell_id", np.dtype(np.int32))
    anchor_kinds = [str(value) for value in cells["routing_anchor_kind"].to_pylist()]
    fine_members, fine_entries, parent_members, parent_entries = _routing_maps(reaches)
    spill_receiver_ids = _column(candidates, "spill_receiver_id", np.dtype(np.int32))

    rows: list[dict[str, object]] = []
    remapped_month_count = 0
    fine_target_count = 0
    handoff_target_count = 0
    outside_target_count = 0
    applied_adjustment_km3 = 0.0
    outside_adjustment_km3 = 0.0
    for terminal_row in terminal_rows:
        target = _route_target(
            int(spill_receiver_ids[terminal_row]),
            cell_row_by_id=cell_row_by_id,
            receiver_ids=receiver_ids,
            parent_ids=parent_ids,
            anchor_kinds=anchor_kinds,
            coarse_receiver_ids=coarse_receiver_ids,
            fine_members=fine_members,
            fine_entries=fine_entries,
            parent_members=parent_members,
            parent_entries=parent_entries,
        )
        if target.routing_mode == "fine_channel":
            fine_target_count += 1
        elif target.routing_mode == "preserved_handoff":
            handoff_target_count += 1
        else:
            outside_target_count += 1
        path_ids = (
            _reach_path(target.reach_id, downstream_reach_ids, reach_row_by_id)
            if target.reach_id >= 0
            else []
        )
        path_rows = [reach_row_by_id[reach_id] for reach_id in path_ids]
        for month in range(MONTHS):
            source_km3 = float(network_direct[terminal_row, month])
            overflow_km3 = float(overflow[terminal_row, month])
            delta_km3 = overflow_km3 - source_km3
            delta_m3s = delta_km3 * 1e9 / SECONDS_PER_MONTH
            effective_index = -1
            effective_reach_id = -1
            joins_at_effective_entry = False
            if path_rows:
                effective_index = _find_effective_start(
                    path_rows,
                    month,
                    delta_m3s,
                    target.joins_at_entry,
                    base_entry,
                    base_exit,
                    entry_adjustment,
                    exit_adjustment,
                    config.maximum_negative_discharge_m3s,
                )
                effective_reach_id = path_ids[effective_index]
                joins_at_effective_entry = target.joins_at_entry and effective_index == 0
                remapped_month_count += int(effective_index > 0)
                for row in path_rows[effective_index:]:
                    exit_adjustment[row, month] += delta_m3s
                entry_start = effective_index if joins_at_effective_entry else effective_index + 1
                for row in path_rows[entry_start:]:
                    entry_adjustment[row, month] += delta_m3s
                applied_adjustment_km3 += delta_km3
            else:
                outside_adjustment_km3 += delta_km3
            rows.append(
                {
                    "terminal_depression_id": int(candidate_ids[terminal_row]),
                    "month": month + 1,
                    "source_runoff_km3": source_km3,
                    "overflow_km3": overflow_km3,
                    "hydrograph_adjustment_km3": delta_km3,
                    "hydrograph_adjustment_m3s": delta_m3s,
                    "nominal_reach_id": target.reach_id,
                    "effective_reach_id": effective_reach_id,
                    "joins_at_effective_entry": joins_at_effective_entry,
                    "downstream_remapped": effective_index > 0,
                    "routing_mode": target.routing_mode,
                    "routed_cell_id": target.routed_cell_id,
                }
            )

    coupled_entry = base_entry + entry_adjustment
    coupled_exit = base_exit + exit_adjustment
    minimum_discharge = float(min(np.min(coupled_entry), np.min(coupled_exit)))
    if minimum_discharge < -config.maximum_negative_discharge_m3s:
        raise RuntimeError("lake-coupled hydrograph contains negative discharge")
    coupled_entry = np.maximum(coupled_entry, 0.0)
    coupled_exit = np.maximum(coupled_exit, 0.0)

    evaporation = np.asarray(candidates["annual_evaporation_km3"], dtype=np.float64)
    seepage = np.asarray(candidates["annual_seepage_km3"], dtype=np.float64)
    storage_change = np.asarray(candidates["annual_storage_change_km3"], dtype=np.float64)
    annual_source = float(np.sum(direct_inflow))
    annual_overflow = float(np.sum(overflow[terminal_rows]))
    annual_losses = float(np.sum(evaporation + seepage + storage_change))
    balance_residual = annual_source - annual_losses - annual_overflow
    balance_relative_error = abs(balance_residual) / max(annual_source, 1e-12)
    if balance_relative_error > config.maximum_balance_relative_error:
        raise RuntimeError("lake-network hydrograph coupling does not conserve water")

    coupled = reaches.append_column(
        "pre_lake_discharge_mean", reaches["discharge_mean"].combine_chunks()
    )
    coupled = coupled.append_column(
        "pre_lake_discharge_seasonal", reaches["discharge_seasonal"].combine_chunks()
    )
    coupled = coupled.set_column(
        coupled.schema.get_field_index("discharge_mean"),
        "discharge_mean",
        pa.array(np.mean(coupled_entry, axis=1), type=pa.float64()),
    )
    coupled = coupled.set_column(
        coupled.schema.get_field_index("discharge_seasonal"),
        "discharge_seasonal",
        _fixed_list_array(coupled_entry),
    )
    for name, values in (
        ("lake_entry_adjustment_seasonal", entry_adjustment),
        ("exit_discharge_seasonal", coupled_exit),
        ("pre_lake_exit_discharge_seasonal", base_exit),
        ("lake_exit_adjustment_seasonal", exit_adjustment),
    ):
        coupled = coupled.append_column(name, _fixed_list_array(values))
    coupled = coupled.append_column(
        "exit_discharge_mean",
        pa.array(np.mean(coupled_exit, axis=1), type=pa.float64()),
    )
    coupled = coupled.append_column(
        "lake_hydrograph_coupled",
        pa.array(np.ones(reaches.num_rows, dtype=bool), type=pa.bool_()),
    )
    adjustment_catalog = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("terminal_depression_id", pa.int32()),
                ("month", pa.int8()),
                ("source_runoff_km3", pa.float64()),
                ("overflow_km3", pa.float64()),
                ("hydrograph_adjustment_km3", pa.float64()),
                ("hydrograph_adjustment_m3s", pa.float64()),
                ("nominal_reach_id", pa.int32()),
                ("effective_reach_id", pa.int32()),
                ("joins_at_effective_entry", pa.bool_()),
                ("downstream_remapped", pa.bool_()),
                ("routing_mode", pa.string()),
                ("routed_cell_id", pa.int32()),
            ]
        ),
    )
    metadata: dict[str, object] = {
        **asdict(config),
        "model": "conservative_terminal_lake_delta_hydrographs_v1",
        "terminal_lake_network_count": len(terminal_rows),
        "fine_channel_target_count": fine_target_count,
        "preserved_handoff_target_count": handoff_target_count,
        "outside_terminal_target_count": outside_target_count,
        "monthly_adjustment_record_count": adjustment_catalog.num_rows,
        "downstream_remapped_month_count": remapped_month_count,
        "annual_network_source_runoff_km3": annual_source,
        "annual_terminal_overflow_km3": annual_overflow,
        "annual_network_loss_and_storage_change_km3": annual_losses,
        "network_balance_residual_km3": balance_residual,
        "network_balance_relative_error": balance_relative_error,
        "applied_reach_adjustment_km3": applied_adjustment_km3,
        "outside_terminal_adjustment_km3": outside_adjustment_km3,
        "minimum_coupled_discharge_m3s": minimum_discharge,
        "entry_semantics": "reach_start_after_final_lake_storage_and_loss",
        "exit_semantics": "reach_end_after_final_lake_storage_and_loss",
        "lake_reach_hydrograph_coupling_implemented": 1,
    }
    return coupled, adjustment_catalog, metadata


@stage(  # type: ignore[untyped-decorator]
    "lake_hydrographs",
    inputs=("surface_water_final", "outlet_incision", "hydrology"),
    outputs=(
        "LakeCoupledRiverReachCatalog",
        "LakeHydrographAdjustmentCatalog",
        "LakeHydrographMetadata",
    ),
    version="v1",
)
def lake_hydrograph_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = LakeHydrographConfig.from_mapping(config_mapping)
    candidates = _artifact_table(deps["surface_water_final"], "SurfaceWaterCandidateCatalog")
    cells = _artifact_table(deps["surface_water_final"], "FinalOutletCorrectedBasinCellCatalog")
    reaches = _artifact_table(deps["outlet_incision"], "StabilizedRiverReachCatalog")
    coarse_receiver_ids = np.asarray(
        _artifact_array(deps["hydrology"], "FlowReceiverID"), dtype=np.int32
    ).reshape(-1)
    coarse_monthly_discharge = np.asarray(
        _artifact_array(deps["hydrology"], "MonthlyDischargeM3s"), dtype=np.float64
    ).reshape(MONTHS, -1)
    with context.timed("lake_reach_hydrograph_coupling"):
        coupled, adjustments, metadata = _couple_hydrographs(
            config,
            candidates,
            cells,
            reaches,
            coarse_receiver_ids,
            coarse_monthly_discharge,
        )
    context.logger.log_event(
        {"type": "lake_hydrograph_summary", "stage": "lake_hydrographs", **metadata}
    )
    return {
        "LakeCoupledRiverReachCatalog": coupled,
        "LakeHydrographAdjustmentCatalog": adjustments,
        "LakeHydrographMetadata": metadata,
    }


__all__ = ["LakeHydrographConfig", "lake_hydrograph_stage"]
