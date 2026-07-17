"""Monthly fractional surface-water balance over refined local depressions."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pyarrow as pa
from PIL import Image

from .._surface_water_native import SURFACE_WATER_CLASSES, run_surface_water_balance
from ..cubed_sphere import CubedSphereGrid
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult

MONTHS = 12


@dataclass(frozen=True)
class SurfaceWaterConfig:
    minimum_solver_iterations: int = 8
    maximum_solver_iterations: int = 64
    transient_max_months: int = 3
    permanent_min_months: int = 12
    convergence_tolerance_fraction: float = 1e-6
    open_water_evaporation_factor: float = 1.15
    seepage_mm_year: float = 30.0
    subgrid_relief_scale: float = 1.0
    minimum_subgrid_relief_m: float = 10.0
    maximum_connected_inundation_fraction: float = 0.25
    minimum_wet_area_fraction: float = 0.01
    wetland_max_mean_depth_m: float = 3.0
    outlet_erosion_score_threshold: float = 0.30
    outlet_erosion_depth_scale_m: float = 200.0
    minimum_outlet_erosion_discharge_m3s: float = 0.10
    maximum_water_balance_relative_error: float = 1e-9
    maximum_area_reconstruction_relative_error: float = 1e-6

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "SurfaceWaterConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown surface-water controls: {', '.join(sorted(unknown))}")
        integer_names = {
            "minimum_solver_iterations",
            "maximum_solver_iterations",
            "transient_max_months",
            "permanent_min_months",
        }
        values = {
            name: (
                int(mapping.get(name, field.default))
                if name in integer_names
                else float(mapping.get(name, field.default))
            )
            for name, field in cls.__dataclass_fields__.items()
        }
        config = cls(**values)
        if not 1 <= config.minimum_solver_iterations <= config.maximum_solver_iterations <= 256:
            raise ValueError(
                "surface-water solver iterations must satisfy 1 <= minimum <= maximum <= 256"
            )
        if not 0 <= config.transient_max_months < config.permanent_min_months == MONTHS:
            raise ValueError(
                "surface-water month thresholds must satisfy 0 <= transient < permanent == 12"
            )
        bounds = {
            "convergence_tolerance_fraction": (1e-12, 0.1),
            "open_water_evaporation_factor": (0.0, 10.0),
            "seepage_mm_year": (0.0, 10_000.0),
            "subgrid_relief_scale": (0.01, 10.0),
            "minimum_subgrid_relief_m": (0.1, 1_000.0),
            "maximum_connected_inundation_fraction": (0.01, 1.0),
            "wetland_max_mean_depth_m": (0.1, 100.0),
            "outlet_erosion_score_threshold": (0.0, 1.0),
            "outlet_erosion_depth_scale_m": (1.0, 5_000.0),
            "minimum_outlet_erosion_discharge_m3s": (0.0, 100_000.0),
            "maximum_water_balance_relative_error": (1e-14, 0.01),
            "maximum_area_reconstruction_relative_error": (1e-12, 0.01),
        }
        for name, (minimum, maximum) in bounds.items():
            value = getattr(config, name)
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
        if (
            not np.isfinite(config.minimum_wet_area_fraction)
            or not 0.0 < config.minimum_wet_area_fraction <= 1.0
        ):
            raise ValueError("minimum_wet_area_fraction must be finite and in (0, 1]")
        return config


@dataclass(frozen=True)
class SurfaceWaterFinalConfig:
    maximum_outlet_incision_rounds: int = 8
    require_soil_readiness: bool = False

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "SurfaceWaterFinalConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown final surface-water controls: {', '.join(sorted(unknown))}")
        maximum_rounds = int(
            mapping.get(
                "maximum_outlet_incision_rounds",
                cls.__dataclass_fields__["maximum_outlet_incision_rounds"].default,
            )
        )
        raw_required = mapping.get(
            "require_soil_readiness",
            cls.__dataclass_fields__["require_soil_readiness"].default,
        )
        if not isinstance(raw_required, bool):
            raise ValueError("require_soil_readiness must be a boolean")
        if not 1 <= maximum_rounds <= 32:
            raise ValueError("maximum_outlet_incision_rounds must be in [1, 32]")
        return cls(maximum_rounds, raw_required)


def _artifact_table(result, name: str) -> pa.Table:
    record = result.artifact_records.get(name)
    if record is None or not isinstance(record.value, pa.Table):
        raise KeyError(f"Missing dependency table '{name}'")
    return record.value


def _artifact_array(result, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _column(table: pa.Table, name: str, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(
        table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
    )


def _lookup_rows(ids: np.ndarray, query: np.ndarray, *, label: str) -> np.ndarray:
    order = np.argsort(ids)
    sorted_ids = ids[order]
    positions = np.searchsorted(sorted_ids, query)
    if np.any(positions >= len(sorted_ids)) or np.any(sorted_ids[positions] != query):
        raise RuntimeError(f"{label} references an unknown identifier")
    return order[positions]


def _fixed_list(values: np.ndarray, value_type: pa.DataType) -> pa.FixedSizeListArray:
    values = np.asarray(values)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=value_type), values.shape[1]
    )


def _candidate_table(
    source: pa.Table,
    records: np.ndarray,
    final_class_codes: np.ndarray,
    erosion_score: np.ndarray,
    erosion_required: np.ndarray,
    recommended_incision_m: np.ndarray,
    classification_reason: np.ndarray,
) -> pa.Table:
    source_ids = np.asarray(source["depression_id"], dtype=np.int32)
    if not np.array_equal(source_ids, records["depression_id"]):
        raise RuntimeError("surface-water kernel changed candidate order or identity")
    pre_adjustment_classes = np.array(
        [SURFACE_WATER_CLASSES[int(code)] for code in records["class_code"]], dtype=object
    )
    classes = np.array(
        [SURFACE_WATER_CLASSES[int(code)] for code in final_class_codes], dtype=object
    )
    table = source
    integer_fields = (
        "downstream_depression_id",
        "catchment_cell_count",
        "wet_month_count",
        "solver_iterations",
    )
    for name in integer_fields:
        table = table.append_column(name, pa.array(records[name], type=pa.int32()))
    table = table.append_column(
        "pre_adjustment_class_code", pa.array(records["class_code"], type=pa.int32())
    )
    table = table.append_column(
        "pre_adjustment_surface_water_class",
        pa.array(pre_adjustment_classes, type=pa.string()),
    )
    table = table.append_column("class_code", pa.array(final_class_codes, type=pa.int32()))
    table = table.append_column("surface_water_class", pa.array(classes, type=pa.string()))
    table = table.append_column(
        "classification_reason", pa.array(classification_reason, type=pa.string())
    )
    table = table.append_column("converged", pa.array(records["converged"] != 0, type=pa.bool_()))
    table = table.append_column(
        "open_outlet", pa.array(records["open_outlet"] != 0, type=pa.bool_())
    )
    table = table.append_column(
        "outlet_erosion_required", pa.array(erosion_required, type=pa.bool_())
    )
    table = table.append_column("outlet_erosion_score", pa.array(erosion_score, type=pa.float64()))
    table = table.append_column(
        "recommended_outlet_incision_m",
        pa.array(recommended_incision_m, type=pa.float64()),
    )
    scalar_fields = (
        "catchment_area_km2",
        "potential_water_area_km2",
        "storage_capacity_km3",
        "annual_direct_inflow_km3",
        "annual_upstream_inflow_km3",
        "annual_total_inflow_km3",
        "annual_evaporation_km3",
        "annual_seepage_km3",
        "annual_overflow_km3",
        "annual_terminal_overflow_km3",
        "annual_storage_change_km3",
        "water_balance_residual_km3",
        "hydroperiod_fraction",
        "minimum_water_area_km2",
        "mean_water_area_km2",
        "maximum_water_area_km2",
        "mean_wetted_depth_m",
        "maximum_mean_depth_m",
        "salinity_index",
    )
    for name in scalar_fields:
        table = table.append_column(name, pa.array(records[name], type=pa.float64()))
    monthly_fields = (
        "monthly_direct_inflow_km3",
        "monthly_upstream_inflow_km3",
        "monthly_total_inflow_km3",
        "monthly_evaporation_km3",
        "monthly_seepage_km3",
        "monthly_overflow_km3",
        "monthly_storage_km3",
        "monthly_water_area_km2",
    )
    for name in monthly_fields:
        table = table.append_column(name, _fixed_list(records[name], pa.float64()))
    return table


def _cell_table(
    records: np.ndarray, candidate_ids: np.ndarray, final_class_codes: np.ndarray
) -> pa.Table:
    candidate_rows = _lookup_rows(
        candidate_ids, records["depression_id"], label="surface-water cell candidate"
    )
    cell_class_codes = final_class_codes[candidate_rows]
    classes = np.array(
        [SURFACE_WATER_CLASSES[int(code)] for code in cell_class_codes], dtype=object
    )
    return pa.table(
        {
            "fine_cell_id": pa.array(records["fine_cell_id"], type=pa.int32()),
            "depression_id": pa.array(records["depression_id"], type=pa.int32()),
            "pre_adjustment_class_code": pa.array(records["class_code"], type=pa.int32()),
            "class_code": pa.array(cell_class_codes, type=pa.int32()),
            "surface_water_class": pa.array(classes, type=pa.string()),
            "potential_inundation_fraction": pa.array(
                records["potential_inundation_fraction"], type=pa.float32()
            ),
            "minimum_inundation_fraction": pa.array(
                records["minimum_inundation_fraction"], type=pa.float32()
            ),
            "mean_inundation_fraction": pa.array(
                records["mean_inundation_fraction"], type=pa.float32()
            ),
            "maximum_inundation_fraction": pa.array(
                records["maximum_inundation_fraction"], type=pa.float32()
            ),
            "monthly_inundation_fraction": _fixed_list(
                records["monthly_inundation_fraction"], pa.float32()
            ),
        }
    )


def _monthly_table(records: np.ndarray) -> pa.Table:
    return pa.table(
        {
            "depression_id": pa.array(np.repeat(records["depression_id"], MONTHS), type=pa.int32()),
            "month": pa.array(
                np.tile(np.arange(1, MONTHS + 1, dtype=np.int8), len(records)), type=pa.int8()
            ),
            "direct_inflow_km3": pa.array(
                records["monthly_direct_inflow_km3"].reshape(-1), type=pa.float64()
            ),
            "upstream_inflow_km3": pa.array(
                records["monthly_upstream_inflow_km3"].reshape(-1), type=pa.float64()
            ),
            "total_inflow_km3": pa.array(
                records["monthly_total_inflow_km3"].reshape(-1), type=pa.float64()
            ),
            "evaporation_km3": pa.array(
                records["monthly_evaporation_km3"].reshape(-1), type=pa.float64()
            ),
            "seepage_km3": pa.array(records["monthly_seepage_km3"].reshape(-1), type=pa.float64()),
            "overflow_km3": pa.array(
                records["monthly_overflow_km3"].reshape(-1), type=pa.float64()
            ),
            "storage_km3": pa.array(records["monthly_storage_km3"].reshape(-1), type=pa.float64()),
            "water_area_km2": pa.array(
                records["monthly_water_area_km2"].reshape(-1), type=pa.float64()
            ),
        }
    )


def _outlet_erosion_feedback(
    config: SurfaceWaterConfig,
    cells: pa.Table,
    candidates: pa.Table,
    candidate_records: np.ndarray,
    cell_records: np.ndarray,
    rock_strength: np.ndarray,
    accommodation: np.ndarray,
    suppressed_outlet_spill_cell_ids: set[int] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
    np.ndarray,
]:
    cell_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    source_rows = _lookup_rows(
        cell_ids, cell_records["fine_cell_id"], label="surface-water feedback cell"
    )
    candidate_ids = np.asarray(candidates["depression_id"], dtype=np.int32)
    candidate_rows = _lookup_rows(
        candidate_ids,
        cell_records["depression_id"],
        label="surface-water feedback candidate",
    )
    covered_area = np.asarray(cells["area_km2"], dtype=np.float64)[source_rows] * cell_records[
        "potential_inundation_fraction"
    ].astype(np.float64)
    support_area = np.zeros(len(candidates), dtype=np.float64)
    rock_total = np.zeros(len(candidates), dtype=np.float64)
    accommodation_total = np.zeros(len(candidates), dtype=np.float64)
    np.add.at(support_area, candidate_rows, covered_area)
    np.add.at(rock_total, candidate_rows, covered_area * rock_strength[source_rows])
    np.add.at(
        accommodation_total,
        candidate_rows,
        covered_area * accommodation[source_rows],
    )
    mean_rock = rock_total / np.maximum(support_area, 1e-12)
    mean_accommodation = accommodation_total / np.maximum(support_area, 1e-12)
    maximum_head_m = np.asarray(candidates["maximum_fill_depth_m"], dtype=np.float64)
    annual_overflow_km3 = candidate_records["annual_overflow_km3"]
    mean_overflow_m3s = annual_overflow_km3 * 1e9 / (365.2422 * 86_400.0)
    depth_score = np.clip(maximum_head_m / config.outlet_erosion_depth_scale_m, 0.0, 1.0)
    discharge_score = np.clip(np.log1p(mean_overflow_m3s) / 6.0, 0.0, 1.0)
    erosion_score = (
        0.35 * depth_score
        + 0.30 * (1.0 - mean_rock)
        + 0.15 * (1.0 - mean_accommodation)
        + 0.20 * discharge_score
    )
    pre_adjustment_codes = candidate_records["class_code"]
    lake_like = (pre_adjustment_codes == 2) | (pre_adjustment_codes == 3)
    suppressed_spill_ids = suppressed_outlet_spill_cell_ids or set()
    if suppressed_spill_ids:
        bounded_outlet_resolved = lake_like & np.isin(
            np.asarray(candidates["spill_cell_id"], dtype=np.int32),
            np.fromiter(suppressed_spill_ids, dtype=np.int32),
        )
    else:
        bounded_outlet_resolved = np.zeros(len(candidates), dtype=bool)
    erosion_required = (
        lake_like
        & ~bounded_outlet_resolved
        & (mean_overflow_m3s > 0.0)
        & (mean_overflow_m3s >= config.minimum_outlet_erosion_discharge_m3s)
        & (erosion_score >= config.outlet_erosion_score_threshold)
    )
    final_class_codes = pre_adjustment_codes.copy()
    final_class_codes[erosion_required] = 1
    recommended_incision_m = np.where(
        erosion_required,
        maximum_head_m * np.clip(0.20 + erosion_score, 0.0, 0.75),
        0.0,
    )
    reasons = np.select(
        [
            erosion_required,
            bounded_outlet_resolved,
            pre_adjustment_codes == 0,
            pre_adjustment_codes == 1,
            pre_adjustment_codes == 2,
            pre_adjustment_codes == 3,
            pre_adjustment_codes == 4,
        ],
        [
            "outlet_erosion_feedback",
            "bounded_outlet_feedback_resolved",
            "no_material_water",
            "short_hydroperiod",
            "seasonal_hydroperiod",
            "year_round_inundation",
            "shallow_recurrent_inundation",
        ],
        default="unclassified",
    )
    mean_area = candidate_records["mean_water_area_km2"]
    feedback_metadata: dict[str, object] = {
        "outlet_erosion_required_count": int(np.count_nonzero(erosion_required)),
        "outlet_erosion_required_mean_water_area_km2": float(np.sum(mean_area[erosion_required])),
        "bounded_outlet_feedback_resolved_count": int(np.count_nonzero(bounded_outlet_resolved)),
        "bounded_outlet_feedback_resolved_mean_water_area_km2": float(
            np.sum(mean_area[bounded_outlet_resolved])
        ),
        "maximum_outlet_erosion_score": float(np.max(erosion_score)),
        "maximum_recommended_outlet_incision_m": float(np.max(recommended_incision_m)),
        "outlet_erosion_score_semantics": ("head_weak_rock_low_accommodation_sustained_overflow"),
        "pre_adjustment_water_area_semantics": (
            "monthly_fill_spill_state_before_required_outlet_incision"
        ),
        "classification_reasons": sorted(set(str(value) for value in reasons)),
    }
    return (
        final_class_codes,
        erosion_score,
        erosion_required,
        recommended_incision_m,
        feedback_metadata,
        reasons,
    )


def _catchment_audit(
    cells: pa.Table,
    candidates: pa.Table,
    monthly_runoff_mm: np.ndarray,
) -> dict[str, float | int]:
    cell_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    receiver_ids = np.asarray(cells["stabilized_receiver_id"], dtype=np.int32)
    depression_ids = np.asarray(cells["stabilized_depression_id"], dtype=np.int32)
    source_active = np.asarray(cells["source_active"])
    area = np.asarray(cells["area_km2"], dtype=np.float64)
    candidate_ids = np.asarray(candidates["depression_id"], dtype=np.int32)
    candidate_by_id = {int(value): index for index, value in enumerate(candidate_ids)}
    routed = receiver_ids >= 0
    target_rows = np.full(len(cells), -1, dtype=np.int64)
    if np.any(routed):
        target_rows[routed] = _lookup_rows(cell_ids, receiver_ids[routed], label="receiver")
    upstream_count = np.zeros(len(cells), dtype=np.int32)
    np.add.at(upstream_count, target_rows[routed], 1)
    ready = deque(np.flatnonzero(upstream_count == 0).tolist())
    order: list[int] = []
    while ready:
        row = ready.popleft()
        order.append(row)
        target = target_rows[row]
        if target >= 0:
            upstream_count[target] -= 1
            if upstream_count[target] == 0:
                ready.append(int(target))
    if len(order) != len(cells):
        raise RuntimeError("surface-water audit found a cyclic stabilized cell graph")
    owner = np.full(len(cells), -1, dtype=np.int32)
    for row in reversed(order):
        depression_id = int(depression_ids[row])
        if depression_id >= 0:
            owner[row] = candidate_by_id[depression_id]
        elif target_rows[row] >= 0:
            owner[row] = owner[target_rows[row]]

    direct = np.zeros((len(candidates), MONTHS), dtype=np.float64)
    owned = source_active & (owner >= 0)
    for month in range(MONTHS):
        np.add.at(
            direct[:, month],
            owner[owned],
            monthly_runoff_mm[month, owned] * area[owned] / 1_000_000.0,
        )
    catchment_area = np.zeros(len(candidates), dtype=np.float64)
    catchment_count = np.zeros(len(candidates), dtype=np.int32)
    np.add.at(catchment_area, owner[owned], area[owned])
    np.add.at(catchment_count, owner[owned], 1)

    spill_receivers = np.asarray(candidates["spill_receiver_id"], dtype=np.int32)
    downstream = np.full(len(candidates), -1, dtype=np.int32)
    routed_spills = spill_receivers >= 0
    if np.any(routed_spills):
        spill_rows = _lookup_rows(
            cell_ids, spill_receivers[routed_spills], label="candidate spill receiver"
        )
        downstream[routed_spills] = owner[spill_rows]
    published_downstream_ids = np.asarray(candidates["downstream_depression_id"], dtype=np.int32)
    expected_downstream_ids = np.where(
        downstream >= 0, candidate_ids[np.maximum(downstream, 0)], -1
    )
    published_direct = np.asarray(
        candidates["monthly_direct_inflow_km3"].combine_chunks().values
    ).reshape(len(candidates), MONTHS)
    published_overflow = np.asarray(
        candidates["monthly_overflow_km3"].combine_chunks().values
    ).reshape(len(candidates), MONTHS)
    published_upstream = np.asarray(
        candidates["monthly_upstream_inflow_km3"].combine_chunks().values
    ).reshape(len(candidates), MONTHS)
    expected_upstream = np.zeros((len(candidates), MONTHS), dtype=np.float64)
    for source_row, target_row in enumerate(downstream):
        if target_row >= 0:
            expected_upstream[target_row] += published_overflow[source_row]
    return {
        "independent_cell_graph_valid": 1,
        "independent_owned_source_cell_count": int(np.count_nonzero(owned)),
        "independent_owned_catchment_area_km2": float(np.sum(area[owned])),
        "maximum_direct_inflow_error_km3": float(np.max(np.abs(direct - published_direct))),
        "maximum_upstream_inflow_error_km3": float(
            np.max(np.abs(expected_upstream - published_upstream))
        ),
        "maximum_catchment_area_error_km2": float(
            np.max(
                np.abs(
                    catchment_area - np.asarray(candidates["catchment_area_km2"], dtype=np.float64)
                )
            )
        ),
        "maximum_catchment_cell_count_error": int(
            np.max(
                np.abs(
                    catchment_count - np.asarray(candidates["catchment_cell_count"], dtype=np.int32)
                )
            )
        ),
        "downstream_candidate_mismatch_count": int(
            np.count_nonzero(expected_downstream_ids != published_downstream_ids)
        ),
    }


def _area_audit(
    cells: pa.Table,
    candidates: pa.Table,
    water_cells: pa.Table,
) -> dict[str, float]:
    source_cell_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    water_cell_ids = np.asarray(water_cells["fine_cell_id"], dtype=np.int32)
    source_rows = _lookup_rows(source_cell_ids, water_cell_ids, label="surface-water cell")
    cell_area = np.asarray(cells["area_km2"], dtype=np.float64)[source_rows]
    candidate_ids = np.asarray(candidates["depression_id"], dtype=np.int32)
    water_depression_ids = np.asarray(water_cells["depression_id"], dtype=np.int32)
    candidate_rows = _lookup_rows(
        candidate_ids, water_depression_ids, label="surface-water candidate"
    )
    fractions = np.asarray(
        water_cells["monthly_inundation_fraction"].combine_chunks().values,
        dtype=np.float32,
    ).reshape(len(water_cells), MONTHS)
    reconstructed = np.zeros((len(candidates), MONTHS), dtype=np.float64)
    for month in range(MONTHS):
        np.add.at(reconstructed[:, month], candidate_rows, fractions[:, month] * cell_area)
    published = np.asarray(
        candidates["monthly_water_area_km2"].combine_chunks().values, dtype=np.float64
    ).reshape(len(candidates), MONTHS)
    absolute_error = np.abs(reconstructed - published)
    relative_error = absolute_error / np.maximum(published, 1.0)
    return {
        "maximum_area_reconstruction_error_km2": float(np.max(absolute_error)),
        "maximum_area_reconstruction_relative_error": float(np.max(relative_error)),
    }


def _cube_net_visualizer(result, request: VisualizationRequest) -> VisualizationResult | None:
    candidate_record = result.artifact_records.get("SurfaceWaterCandidateCatalog")
    cell_record = result.artifact_records.get("SeasonalSurfaceWaterCellCatalog")
    metadata_record = result.artifact_records.get("SurfaceWaterMetadata")
    if (
        candidate_record is None
        or cell_record is None
        or metadata_record is None
        or not isinstance(candidate_record.value, pa.Table)
        or not isinstance(cell_record.value, pa.Table)
    ):
        return None
    candidates = candidate_record.value
    water_cells = cell_record.value
    metadata = metadata_record.value
    fine_resolution = int(metadata["fine_resolution"])
    display_resolution = min(fine_resolution, 768)
    placements = np.array([[1, 1], [1, 3], [1, 2], [1, 0], [0, 1], [2, 1]])

    cell_ids = np.asarray(water_cells["fine_cell_id"], dtype=np.int64)
    face_size = fine_resolution * fine_resolution
    face = cell_ids // face_size
    within = cell_ids % face_size
    row = within // fine_resolution
    col = within % fine_resolution
    display_row = np.minimum(row * display_resolution // fine_resolution, display_resolution - 1)
    display_col = np.minimum(col * display_resolution // fine_resolution, display_resolution - 1)
    net_row = placements[face, 0] * display_resolution + display_row
    net_col = placements[face, 1] * display_resolution + display_col

    image = np.full((display_resolution * 3, display_resolution * 4, 3), 17, dtype=np.uint8)
    class_code = np.asarray(water_cells["class_code"], dtype=np.int32)
    mean_fraction = np.asarray(water_cells["mean_inundation_fraction"], dtype=np.float32)
    colors = np.array(
        [
            [111, 88, 56],
            [94, 159, 183],
            [31, 132, 196],
            [15, 83, 156],
            [62, 145, 91],
        ],
        dtype=np.float64,
    )[class_code]
    strength = np.where(class_code == 0, 0.30, 0.45 + 0.55 * mean_fraction)[:, None]
    background = np.array([53.0, 65.0, 48.0])
    blended = background * (1.0 - strength) + colors * strength
    image[net_row, net_col] = blended.astype(np.uint8)

    rendered = Image.fromarray(image, mode="RGB")
    padding = max(12, display_resolution // 40)
    rendered = rendered.crop(
        (
            max(0, int(np.min(net_col)) - padding),
            max(0, int(np.min(net_row)) - padding),
            min(rendered.width, int(np.max(net_col)) + padding + 1),
            min(rendered.height, int(np.max(net_row)) + padding + 1),
        )
    )
    scale = min(1600 / rendered.width, 1000 / rendered.height)
    if scale > 1.0:
        rendered = rendered.resize(
            (max(1, round(rendered.width * scale)), max(1, round(rendered.height * scale))),
            Image.Resampling.NEAREST,
        )
    output = request.output_dir / "seasonal_surface_water.png"
    rendered.save(output)
    return VisualizationResult(
        output,
        "SurfaceWaterCandidateCatalog",
        {
            "class_counts": {
                name: int(np.count_nonzero(np.asarray(candidates["surface_water_class"]) == name))
                for name in SURFACE_WATER_CLASSES.values()
            }
        },
    )


@stage(
    "surface_water",
    inputs=("hydrology_pass2", "climate", "geology", "basin_refinement"),
    outputs=(
        "SurfaceWaterCandidateCatalog",
        "SeasonalSurfaceWaterCellCatalog",
        "SurfaceWaterMonthlyStateCatalog",
        "SurfaceWaterMetadata",
    ),
    version="v3",
    native_libraries=("surface_water_native",),
    visualizer=_cube_net_visualizer,
)
def surface_water_stage(context, deps, config_mapping: Mapping[str, object]):
    config = SurfaceWaterConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("surface-water balance requires topology: cubed_sphere")
    cells = _artifact_table(deps["hydrology_pass2"], "StabilizedBasinCellCatalog")
    candidates = _artifact_table(deps["hydrology_pass2"], "LocalDepressionCandidateCatalog")
    parents = _artifact_table(deps["basin_refinement"], "RefinedBasinParentCatalog")
    refinement_metadata = deps["basin_refinement"].artifact_records["BasinRefinementMetadata"].value
    pass2_metadata = deps["hydrology_pass2"].artifact_records["HydrologyPass2Metadata"].value
    if candidates.num_rows == 0:
        raise RuntimeError("surface-water balance requires at least one local candidate")

    cell_ids = _column(cells, "fine_cell_id", np.dtype(np.int32))
    parent_ids = _column(cells, "parent_cell_id", np.dtype(np.int32))
    depression_ids = _column(cells, "stabilized_depression_id", np.dtype(np.int32))
    candidate_mask = depression_ids >= 0
    anchor_kind = np.asarray(cells["routing_anchor_kind"].to_pylist())
    source_active_bool = np.asarray(cells["source_active"])
    process_excluded = np.asarray(cells["process_excluded"])
    if (
        np.any(anchor_kind[candidate_mask] != "ordinary")
        or np.any(~source_active_bool[candidate_mask])
        or np.any(process_excluded[candidate_mask])
    ):
        raise RuntimeError("local surface-water candidates must use active ordinary child support")

    monthly_runoff_parent = np.asarray(
        _artifact_array(deps["climate"], "MonthlyRunoffPotentialMm"), dtype=np.float32
    ).reshape(MONTHS, -1)
    monthly_evaporation_parent = np.asarray(
        _artifact_array(deps["climate"], "MonthlyEvaporationMm"), dtype=np.float32
    ).reshape(MONTHS, -1)
    accommodation_parent = np.asarray(
        _artifact_array(deps["geology"], "SedimentAccommodation"), dtype=np.float32
    ).reshape(-1)
    rock_strength_parent = np.asarray(
        _artifact_array(deps["geology"], "RockStrength"), dtype=np.float32
    ).reshape(-1)
    if np.any(parent_ids < 0) or np.any(parent_ids >= monthly_runoff_parent.shape[1]):
        raise RuntimeError("refined child references climate outside the coarse cubed sphere")
    monthly_runoff = np.ascontiguousarray(monthly_runoff_parent[:, parent_ids], dtype=np.float32)
    monthly_evaporation = np.ascontiguousarray(
        monthly_evaporation_parent[:, parent_ids], dtype=np.float32
    )
    accommodation = np.ascontiguousarray(accommodation_parent[parent_ids], dtype=np.float32)
    rock_strength = np.ascontiguousarray(rock_strength_parent[parent_ids], dtype=np.float32)

    candidate_ids = _column(candidates, "depression_id", np.dtype(np.int32))
    controls = {
        key: value
        for key, value in asdict(config).items()
        if key
        not in {
            "maximum_water_balance_relative_error",
            "maximum_area_reconstruction_relative_error",
            "outlet_erosion_score_threshold",
            "outlet_erosion_depth_scale_m",
            "minimum_outlet_erosion_discharge_m3s",
        }
    }
    controls["refinement_factor"] = int(refinement_metadata["refinement_factor"])
    with context.timed("refined_monthly_surface_water_kernel"):
        candidate_records, cell_records, metadata = run_surface_water_balance(
            controls=controls,
            cell_ids=cell_ids,
            receiver_ids=_column(cells, "stabilized_receiver_id", np.dtype(np.int32)),
            depression_ids=depression_ids,
            source_active=np.ascontiguousarray(source_active_bool, dtype=np.uint8),
            area_km2=_column(cells, "area_km2", np.dtype(np.float64)),
            terrain_elevation_m=_column(cells, "routing_surface_after_m", np.dtype(np.float64)),
            hydrologic_elevation_m=_column(
                cells, "stabilized_hydrologic_elevation_m", np.dtype(np.float64)
            ),
            parent_relief_m=_column(cells, "parent_relief_m", np.dtype(np.float32)),
            monthly_runoff_mm=monthly_runoff,
            monthly_evaporation_mm=monthly_evaporation,
            sediment_accommodation=accommodation,
            candidate_ids=candidate_ids,
            spill_receiver_ids=_column(candidates, "spill_receiver_id", np.dtype(np.int32)),
        )
    (
        final_class_codes,
        erosion_score,
        erosion_required,
        recommended_incision_m,
        feedback_metadata,
        classification_reason,
    ) = _outlet_erosion_feedback(
        config,
        cells,
        candidates,
        candidate_records,
        cell_records,
        rock_strength,
        accommodation,
        {
            int(value)
            for value in pass2_metadata.get(
                "outlet_feedback_suppressed_spill_cell_ids",
                pass2_metadata.get("grade_blocked_spill_cell_ids", []),
            )
        },
    )
    candidate_table = _candidate_table(
        candidates,
        candidate_records,
        final_class_codes,
        erosion_score,
        erosion_required,
        recommended_incision_m,
        classification_reason,
    )
    cell_table = _cell_table(cell_records, candidate_ids, final_class_codes)
    monthly_table = _monthly_table(candidate_records)

    parent_catalog_ids = _column(parents, "parent_cell_id", np.dtype(np.int32))
    parent_rows = _lookup_rows(
        parent_catalog_ids, np.unique(parent_ids), label="represented parent"
    )
    represented_parent_ids = parent_catalog_ids[parent_rows]
    represented_parent_area = np.asarray(parents["restricted_child_area_km2"], dtype=np.float64)[
        parent_rows
    ]
    child_area = np.asarray(cells["area_km2"], dtype=np.float64)
    parent_inverse = _lookup_rows(
        represented_parent_ids, parent_ids, label="child represented parent"
    )
    reconstructed_parent_area = np.zeros(len(represented_parent_ids), dtype=np.float64)
    np.add.at(reconstructed_parent_area, parent_inverse, child_area)
    parent_area_relative_error = np.max(
        np.abs(reconstructed_parent_area - represented_parent_area)
        / np.maximum(represented_parent_area, 1e-12)
    )
    expected_runoff_volume = float(
        np.sum(
            monthly_runoff_parent[:, represented_parent_ids]
            * represented_parent_area[None, :]
            / 1_000_000.0
        )
    )
    inherited_runoff_volume = float(np.sum(monthly_runoff * child_area[None, :] / 1_000_000.0))
    runoff_inheritance_relative_error = abs(inherited_runoff_volume - expected_runoff_volume) / max(
        expected_runoff_volume, 1e-12
    )

    catchment_metadata = _catchment_audit(cells, candidate_table, monthly_runoff)
    area_metadata = _area_audit(cells, candidate_table, cell_table)
    independent_residual = float(
        np.sum(candidate_records["annual_direct_inflow_km3"])
        - np.sum(candidate_records["annual_evaporation_km3"])
        - np.sum(candidate_records["annual_seepage_km3"])
        - np.sum(candidate_records["annual_terminal_overflow_km3"])
        - np.sum(candidate_records["annual_storage_change_km3"])
    )
    independent_scale = max(float(np.sum(candidate_records["annual_direct_inflow_km3"])), 1e-12)
    independent_relative_error = abs(independent_residual) / independent_scale
    pre_adjustment_class_counts = {
        name: int(np.count_nonzero(candidate_records["class_code"] == code))
        for code, name in SURFACE_WATER_CLASSES.items()
    }
    published_class_counts = {
        name: int(np.count_nonzero(final_class_codes == code))
        for code, name in SURFACE_WATER_CLASSES.items()
    }
    mean_water_area = candidate_records["mean_water_area_km2"]
    published_class_areas = {
        name: float(np.sum(mean_water_area[final_class_codes == code]))
        for code, name in SURFACE_WATER_CLASSES.items()
    }
    native_count_fields = {
        "dry_depression": "dry_count",
        "transient_storage": "transient_count",
        "seasonal_lake": "seasonal_lake_count",
        "permanent_lake": "permanent_lake_count",
        "hydrologic_wetland": "hydrologic_wetland_count",
    }
    pre_adjustment_native_counts = {
        f"pre_adjustment_{field}": int(metadata[field]) for field in native_count_fields.values()
    }
    native_area_fields = {
        "dry_depression": "dry_mean_water_area_km2",
        "transient_storage": "transient_mean_water_area_km2",
        "seasonal_lake": "seasonal_lake_mean_water_area_km2",
        "permanent_lake": "permanent_lake_mean_water_area_km2",
        "hydrologic_wetland": "hydrologic_wetland_mean_water_area_km2",
    }
    pre_adjustment_native_areas = {
        f"pre_adjustment_{field}": float(metadata[field]) for field in native_area_fields.values()
    }
    metadata.update(
        {
            **asdict(config),
            **catchment_metadata,
            **area_metadata,
            **feedback_metadata,
            **pre_adjustment_native_counts,
            **pre_adjustment_native_areas,
            **{
                field: published_class_counts[class_name]
                for class_name, field in native_count_fields.items()
            },
            **{
                field: published_class_areas[class_name]
                for class_name, field in native_area_fields.items()
            },
            "selected_basin_id": int(pass2_metadata["selected_basin_id"]),
            "fine_resolution": int(pass2_metadata["fine_resolution"]),
            "refinement_factor": int(refinement_metadata["refinement_factor"]),
            "maximum_parent_area_relative_error": float(parent_area_relative_error),
            "represented_parent_runoff_volume_km3": expected_runoff_volume,
            "inherited_child_runoff_volume_km3": inherited_runoff_volume,
            "runoff_inheritance_relative_error": runoff_inheritance_relative_error,
            "independent_water_balance_residual_km3": independent_residual,
            "independent_water_balance_relative_error": independent_relative_error,
            "surface_water_classes": {
                str(code): name for code, name in SURFACE_WATER_CLASSES.items()
            },
            "published_class_counts": published_class_counts,
            "pre_adjustment_class_counts": pre_adjustment_class_counts,
            "published_class_mean_water_area_km2": published_class_areas,
            "accepted_standing_water_mean_area_km2": float(
                np.sum(mean_water_area[np.isin(final_class_codes, [2, 3, 4])])
            ),
            "surface_water_ready_for_soils": int(not np.any(erosion_required)),
            "climate_semantics": "coarse_parent_monthly_depth_inherited_by_area",
            "hypsometry_semantics": "fractional_uniform_subcell_relief_at_pass2_spill_level",
            "wetland_semantics": "hydrologic_candidate_pending_soil_and_vegetation_confirmation",
            "model": "refined_candidate_dag_monthly_fill_spill_balance_v1",
        }
    )
    if metadata["graph_valid"] != 1 or catchment_metadata["independent_cell_graph_valid"] != 1:
        raise RuntimeError("surface-water balance received a cyclic cell or candidate graph")
    if metadata["convergence_valid"] != 1 or not np.all(candidate_records["converged"] != 0):
        raise RuntimeError("surface-water monthly storage did not reach a periodic state")
    if metadata["fraction_valid"] != 1 or not (
        0.0
        <= metadata["minimum_inundation_fraction"]
        <= metadata["maximum_inundation_fraction"]
        <= 1.0
    ):
        raise RuntimeError("surface-water inundation fractions are invalid")
    if metadata["storage_valid"] != 1 or metadata["direct_catchment_valid"] != 1:
        raise RuntimeError("surface-water storage or direct catchment ownership is invalid")
    if any(
        pre_adjustment_class_counts[class_name] != int(metadata[f"pre_adjustment_{field}"])
        for class_name, field in native_count_fields.items()
    ):
        raise RuntimeError("native surface-water class counts disagree with emitted candidates")
    if any(
        published_class_counts[class_name] != int(metadata[field])
        for class_name, field in native_count_fields.items()
    ):
        raise RuntimeError("accepted surface-water class counts disagree with emitted candidates")
    if catchment_metadata["maximum_direct_inflow_error_km3"] > 1e-10:
        raise RuntimeError("surface-water direct catchment inflow disagrees with emitted records")
    if catchment_metadata["maximum_upstream_inflow_error_km3"] > 1e-10:
        raise RuntimeError(
            "surface-water upstream overflow transfer disagrees with emitted records"
        )
    if (
        catchment_metadata["maximum_catchment_area_error_km2"] > 1e-8
        or catchment_metadata["maximum_catchment_cell_count_error"] != 0
        or catchment_metadata["downstream_candidate_mismatch_count"] != 0
    ):
        raise RuntimeError("surface-water catchment or candidate topology audit failed")
    if parent_area_relative_error > 1e-9 or runoff_inheritance_relative_error > 1e-12:
        raise RuntimeError("surface-water climate inheritance does not conserve parent support")
    if (
        metadata["water_balance_relative_error"] > config.maximum_water_balance_relative_error
        or independent_relative_error > config.maximum_water_balance_relative_error
    ):
        raise RuntimeError("surface-water candidate network does not conserve water")
    if (
        area_metadata["maximum_area_reconstruction_relative_error"]
        > config.maximum_area_reconstruction_relative_error
    ):
        raise RuntimeError("surface-water cell fractions do not reconstruct candidate area")
    summary_stage = str(pass2_metadata.get("surface_water_stage_name", "surface_water"))
    context.logger.log_event({"type": "surface_water_summary", "stage": summary_stage, **metadata})
    return {
        "SurfaceWaterCandidateCatalog": candidate_table,
        "SeasonalSurfaceWaterCellCatalog": cell_table,
        "SurfaceWaterMonthlyStateCatalog": monthly_table,
        "SurfaceWaterMetadata": metadata,
    }


@stage(
    "surface_water_final",
    inputs=("outlet_incision", "climate", "geology", "basin_refinement", "planet"),
    outputs=(
        "SurfaceWaterCandidateCatalog",
        "SeasonalSurfaceWaterCellCatalog",
        "SurfaceWaterMonthlyStateCatalog",
        "SurfaceWaterMetadata",
        "OutletIncisionIterationCatalog",
        "FinalOutletIncisionCandidateCatalog",
        "FinalOutletIncisionCellCatalog",
        "FinalOutletCorrectedBasinCellCatalog",
        "FinalPostIncisionDepressionCandidateCatalog",
        "FinalOutletIncisionMetadata",
    ),
    version="v3",
    native_libraries=("surface_water_native",),
    visualizer=_cube_net_visualizer,
)
def surface_water_final_stage(context, deps, config_mapping: Mapping[str, object]):
    final_config = SurfaceWaterFinalConfig.from_mapping(config_mapping)
    surface_controls = dict(context.config.stage_config("surface_water"))
    outlet_controls = dict(context.config.stage_config("outlet_incision"))

    def memory_result(output: Mapping[str, object]):
        return SimpleNamespace(
            artifact_records={name: SimpleNamespace(value=value) for name, value in output.items()}
        )

    def with_iteration(table: pa.Table, iteration: int) -> pa.Table:
        return table.append_column(
            "outlet_incision_iteration",
            pa.array(np.full(table.num_rows, iteration, dtype=np.int32), type=pa.int32()),
        )

    current_hydrology = deps["outlet_incision"]
    output = dict(
        surface_water_stage(
            context,
            {
                "hydrology_pass2": current_hydrology,
                "climate": deps["climate"],
                "geology": deps["geology"],
                "basin_refinement": deps["basin_refinement"],
            },
            surface_controls,
        )
    )
    candidate_corrections = [
        with_iteration(
            _artifact_table(deps["outlet_incision"], "OutletIncisionCandidateCatalog"), 1
        )
    ]
    cell_corrections = [
        with_iteration(_artifact_table(deps["outlet_incision"], "OutletIncisionCellCatalog"), 1)
    ]
    iteration_rows: list[dict[str, object]] = []
    iteration = 1
    outlet_metadata = deps["outlet_incision"].artifact_records["OutletIncisionMetadata"].value

    def record_iteration(
        iteration_number: int,
        correction_metadata: Mapping[str, object],
        balance_metadata: Mapping[str, object],
    ) -> None:
        iteration_rows.append(
            {
                "iteration": iteration_number,
                "requested_candidate_count": int(correction_metadata["requested_candidate_count"]),
                "applied_candidate_count": int(correction_metadata["applied_candidate_count"]),
                "blocked_candidate_count": int(correction_metadata["blocked_candidate_count"]),
                "corrected_cell_count": int(correction_metadata["corrected_cell_count"]),
                "eroded_volume_m3": float(correction_metadata["total_eroded_volume_m3"]),
                "post_correction_candidate_count": int(balance_metadata["candidate_count"]),
                "residual_feedback_candidate_count": int(
                    balance_metadata["outlet_erosion_required_count"]
                ),
                "accepted_standing_water_mean_area_km2": float(
                    balance_metadata["accepted_standing_water_mean_area_km2"]
                ),
                "water_balance_relative_error": float(
                    balance_metadata["independent_water_balance_relative_error"]
                ),
            }
        )

    record_iteration(iteration, outlet_metadata, output["SurfaceWaterMetadata"])
    while (
        int(output["SurfaceWaterMetadata"]["outlet_erosion_required_count"]) > 0
        and iteration < final_config.maximum_outlet_incision_rounds
    ):
        from .outlet_incision import outlet_incision_stage

        iteration += 1
        correction_output = dict(
            outlet_incision_stage(
                context,
                {
                    "surface_water": memory_result(output),
                    "hydrology_pass2": current_hydrology,
                    "planet": deps["planet"],
                },
                outlet_controls,
            )
        )
        current_hydrology = memory_result(correction_output)
        outlet_metadata = correction_output["OutletIncisionMetadata"]
        candidate_corrections.append(
            with_iteration(correction_output["OutletIncisionCandidateCatalog"], iteration)
        )
        cell_corrections.append(
            with_iteration(correction_output["OutletIncisionCellCatalog"], iteration)
        )
        output = dict(
            surface_water_stage(
                context,
                {
                    "hydrology_pass2": current_hydrology,
                    "climate": deps["climate"],
                    "geology": deps["geology"],
                    "basin_refinement": deps["basin_refinement"],
                },
                surface_controls,
            )
        )
        record_iteration(iteration, outlet_metadata, output["SurfaceWaterMetadata"])

    metadata = dict(output["SurfaceWaterMetadata"])
    metadata.update(
        {
            "surface_water_phase": "post_outlet_incision",
            "outlet_incision_iteration_count": iteration,
            "maximum_outlet_incision_rounds": final_config.maximum_outlet_incision_rounds,
            "total_requested_outlet_corrections": int(
                sum(row["requested_candidate_count"] for row in iteration_rows)
            ),
            "total_applied_outlet_corrections": int(
                sum(row["applied_candidate_count"] for row in iteration_rows)
            ),
            "total_blocked_outlet_corrections": int(
                sum(row["blocked_candidate_count"] for row in iteration_rows)
            ),
            "total_outlet_eroded_volume_m3": float(
                sum(row["eroded_volume_m3"] for row in iteration_rows)
            ),
            "outlet_correction_converged": int(metadata["outlet_erosion_required_count"] == 0),
            "model": "bounded_iterative_outlet_candidate_monthly_balance_v1",
        }
    )
    output["SurfaceWaterMetadata"] = metadata
    output["OutletIncisionIterationCatalog"] = pa.Table.from_pylist(
        iteration_rows,
        schema=pa.schema(
            [
                ("iteration", pa.int32()),
                ("requested_candidate_count", pa.int32()),
                ("applied_candidate_count", pa.int32()),
                ("blocked_candidate_count", pa.int32()),
                ("corrected_cell_count", pa.int32()),
                ("eroded_volume_m3", pa.float64()),
                ("post_correction_candidate_count", pa.int32()),
                ("residual_feedback_candidate_count", pa.int32()),
                ("accepted_standing_water_mean_area_km2", pa.float64()),
                ("water_balance_relative_error", pa.float64()),
            ]
        ),
    )
    output["FinalOutletIncisionCandidateCatalog"] = pa.concat_tables(candidate_corrections)
    output["FinalOutletIncisionCellCatalog"] = pa.concat_tables(cell_corrections)
    output["FinalOutletCorrectedBasinCellCatalog"] = _artifact_table(
        current_hydrology, "OutletCorrectedBasinCellCatalog"
    )
    output["FinalPostIncisionDepressionCandidateCatalog"] = _artifact_table(
        current_hydrology, "PostIncisionDepressionCandidateCatalog"
    )
    output["FinalOutletIncisionMetadata"] = dict(outlet_metadata)
    if final_config.require_soil_readiness and not metadata["surface_water_ready_for_soils"]:
        raise RuntimeError(
            "bounded outlet-incision rounds ended with residual surface-water feedback"
        )
    context.logger.log_event(
        {"type": "surface_water_final_summary", "stage": "surface_water_final", **metadata}
    )
    return output


__all__ = [
    "SurfaceWaterConfig",
    "SurfaceWaterFinalConfig",
    "surface_water_stage",
    "surface_water_final_stage",
]
