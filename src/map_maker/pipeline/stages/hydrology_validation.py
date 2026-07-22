"""Executable hydrology invariants and Earth-reference diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage

if TYPE_CHECKING:
    from ..execution import PipelineContext


EARTH_RADIUS_KM = 6_371.0
MONTHS = 12
SECONDS_PER_YEAR = 365.2422 * 86_400.0
ALLOWED_REACH_TERMINALS = {
    "ocean",
    "registered_sink",
    "lake",
    "wetland",
    "endorheic_sink",
    "endorheic_lake",
}

EARTH_REFERENCE_PROFILE = (
    {
        "reference_id": "hydrolakes_2016",
        "title": "HydroLAKES global natural-lake inventory",
        "url": "https://doi.org/10.1038/ncomms13603",
        "supports": [
            "global_lake_land_area_fraction",
            "global_lake_volume_km3",
            "global_lake_area_weighted_mean_depth_m",
        ],
        "summary": (
            "Natural lakes at least 0.1 km2 cover about 1.8% of global land, store "
            "about 181,900 km3, and have an area-weighted mean depth near 64 m."
        ),
    },
    {
        "reference_id": "verpoorter_2014",
        "title": "High-resolution global lake inventory",
        "url": "https://doi.org/10.1002/2014GL060641",
        "supports": ["global_lake_land_area_fraction"],
        "summary": "Lakes at least 0.002 km2 cover about 3.7% of nonglaciated land.",
    },
    {
        "reference_id": "merit_plus_2024",
        "title": "MERIT-Plus endorheic drainage basins",
        "url": "https://doi.org/10.1038/s41597-023-02875-9",
        "supports": ["global_closed_drainage_land_fraction"],
        "summary": "Mapped endorheic basins comprise about 18.8% of global land.",
    },
    {
        "reference_id": "dai_trenberth_2002",
        "title": "Freshwater discharge from continents",
        "url": "https://doi.org/10.1175/1525-7541(2002)003%3C0660:EOFDFC%3E2.0.CO;2",
        "supports": ["global_runoff_depth_mm_year"],
        "summary": "Observed continental discharge is 37,288 +/- 662 km3/year.",
    },
    {
        "reference_id": "davidson_2018",
        "title": "Global extent and distribution of wetlands",
        "url": "https://doi.org/10.1071/MF17019",
        "supports": ["global_hydrologic_wetland_land_area_fraction"],
        "summary": "Wetland estimates exceed 12.1 million km2 but definitions vary strongly.",
    },
    {
        "reference_id": "gfplain_2019",
        "title": "GFPLAIN250m global floodplain dataset",
        "url": "https://doi.org/10.1038/sdata.2018.309",
        "supports": ["floodplain_inundation_implemented"],
        "summary": "A global geomorphic floodplain reference exists for later spatial validation.",
    },
    {
        "reference_id": "nsidc_sea_ice_index_v4",
        "title": "NSIDC Sea Ice Index, Version 4",
        "url": "https://nsidc.org/data/g02135/versions/4",
        "supports": [
            "global_minimum_sea_ice_ocean_area_fraction",
            "global_maximum_sea_ice_ocean_area_fraction",
            "global_mean_sea_ice_ocean_area_fraction",
        ],
        "summary": (
            "Monthly Arctic and Antarctic concentration and extent climatologies provide "
            "the structural seasonal envelope; model metrics are area-equivalent concentration."
        ),
    },
)


def _numeric(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _metadata_float(metadata: Mapping[str, object], name: str) -> float:
    return _numeric(metadata[name], name=name)


def _metadata_int(metadata: Mapping[str, object], name: str) -> int:
    value = metadata[name]
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class HydrologyValidationConfig:
    maximum_water_balance_relative_error: float = 1e-9
    maximum_downstream_discharge_drop_fraction: float = 1e-6
    major_river_discharge_threshold_m3s: float = 1_000.0
    snow_cover_threshold_mm: float = 10.0
    perennial_snow_threshold_mm: float = 10.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "HydrologyValidationConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown hydrology-validation controls: {', '.join(sorted(unknown))}")
        values: dict[str, float] = {
            name: _numeric(mapping.get(name, field.default), name=name)
            for name, field in cls.__dataclass_fields__.items()
        }
        config = cls(**values)
        nonnegative = (
            "maximum_water_balance_relative_error",
            "maximum_downstream_discharge_drop_fraction",
            "snow_cover_threshold_mm",
            "perennial_snow_threshold_mm",
        )
        for name in nonnegative:
            value = getattr(config, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not np.isfinite(config.major_river_discharge_threshold_m3s)
            or config.major_river_discharge_threshold_m3s <= 0.0
        ):
            raise ValueError("major_river_discharge_threshold_m3s must be finite and positive")
        return config


def _artifact_array(result: StageResult, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    value = record.value
    array: np.ndarray = np.asarray(value.array() if hasattr(value, "array") else value)
    return array


def _artifact_table(result: StageResult, name: str) -> pa.Table:
    record = result.artifact_records.get(name)
    if record is None or not isinstance(record.value, pa.Table):
        raise KeyError(f"Missing dependency table '{name}'")
    return record.value


def _artifact_metadata(result: StageResult, name: str) -> Mapping[str, object]:
    record = result.artifact_records.get(name)
    if record is None or not isinstance(record.value, Mapping):
        raise KeyError(f"Missing dependency metadata '{name}'")
    return cast(Mapping[str, object], record.value)


def _fixed_list(table: pa.Table, name: str, width: int) -> np.ndarray:
    values = np.asarray(table[name].combine_chunks().values, dtype=np.float64)
    result: np.ndarray = values.reshape(table.num_rows, width)
    return result


def _candidate_chain_metrics(candidates: pa.Table) -> tuple[int, int, int]:
    candidate_ids = np.asarray(candidates["depression_id"], dtype=np.int64)
    downstream_ids = np.asarray(candidates["downstream_depression_id"], dtype=np.int64)
    row_by_id = {int(value): row for row, value in enumerate(candidate_ids)}
    downstream_rows: np.ndarray = np.full(len(candidate_ids), -1, dtype=np.int64)
    unknown_count = 0
    for row, downstream_id in enumerate(downstream_ids):
        if downstream_id < 0:
            continue
        target = row_by_id.get(int(downstream_id))
        if target is None:
            unknown_count += 1
        else:
            downstream_rows[row] = target

    state: np.ndarray = np.zeros(len(candidate_ids), dtype=np.uint8)
    depth: np.ndarray = np.zeros(len(candidate_ids), dtype=np.int32)
    cycle_count = 0
    for start in range(len(candidate_ids)):
        if state[start] == 2:
            continue
        trail: list[int] = []
        position: dict[int, int] = {}
        row = start
        while row >= 0 and state[row] != 2 and row not in position:
            position[row] = len(trail)
            trail.append(row)
            state[row] = 1
            row = int(downstream_rows[row])
        if row >= 0 and row in position:
            cycle_count += 1
            base_depth = 0
        else:
            base_depth = int(depth[row]) if row >= 0 else 0
        for trail_row in reversed(trail):
            base_depth += 1
            depth[trail_row] = base_depth
            state[trail_row] = 2
    return int(np.max(depth, initial=0)), cycle_count, unknown_count


def _reach_metrics(
    reaches: pa.Table,
    *,
    major_discharge_threshold_m3s: float,
    maximum_drop_fraction: float,
) -> dict[str, float | int]:
    reach_ids = np.asarray(reaches["reach_id"], dtype=np.int64)
    downstream_ids = np.asarray(reaches["downstream_reach_id"], dtype=np.int64)
    discharge_mean = np.asarray(reaches["discharge_mean"], dtype=np.float64)
    discharge_monthly = _fixed_list(reaches, "discharge_seasonal", MONTHS)
    terminal_kind = reaches["terminal_kind"].to_pylist()
    terminal_resolved = np.asarray(reaches["terminal_resolved"], dtype=bool)
    trunk_preserved = np.asarray(reaches["trunk_identity_preserved"], dtype=bool)
    row_by_id = {int(value): row for row, value in enumerate(reach_ids)}
    graph_issue_count = int(len(row_by_id) != len(reach_ids))
    mean_drop_count = 0
    monthly_drop_count = 0
    maximum_mean_drop = 0.0
    maximum_monthly_drop = 0.0
    for source, downstream_id in enumerate(downstream_ids):
        if downstream_id < 0:
            continue
        target = row_by_id.get(int(downstream_id))
        if target is None:
            graph_issue_count += 1
            continue
        mean_drop = max(
            0.0,
            (discharge_mean[source] - discharge_mean[target]) / max(discharge_mean[source], 1e-12),
        )
        maximum_mean_drop = max(maximum_mean_drop, mean_drop)
        if mean_drop > maximum_drop_fraction:
            mean_drop_count += 1
        monthly_drop = np.maximum(
            0.0,
            (discharge_monthly[source] - discharge_monthly[target])
            / np.maximum(discharge_monthly[source], 1e-12),
        )
        maximum_monthly_drop = max(maximum_monthly_drop, float(np.max(monthly_drop)))
        monthly_drop_count += int(np.count_nonzero(monthly_drop > maximum_drop_fraction))

    allowed_terminal_count = 0
    dead_end_count = 0
    cycle_count = 0
    major_dead_end_count = 0
    for start in range(len(reach_ids)):
        seen: set[int] = set()
        row = start
        while downstream_ids[row] >= 0:
            if row in seen:
                cycle_count += 1
                dead_end_count += 1
                if discharge_mean[start] >= major_discharge_threshold_m3s:
                    major_dead_end_count += 1
                break
            seen.add(row)
            target = row_by_id.get(int(downstream_ids[row]))
            if target is None:
                dead_end_count += 1
                if discharge_mean[start] >= major_discharge_threshold_m3s:
                    major_dead_end_count += 1
                break
            row = target
        else:
            allowed = bool(terminal_resolved[row]) and terminal_kind[row] in ALLOWED_REACH_TERMINALS
            allowed_terminal_count += int(allowed)
            if not allowed:
                dead_end_count += 1
                if discharge_mean[start] >= major_discharge_threshold_m3s:
                    major_dead_end_count += 1

    terminal_rows = downstream_ids < 0
    outlet_discharge = discharge_mean[terminal_rows]
    return {
        "reach_count": len(reach_ids),
        "resolved_terminal_fraction": float(np.mean(terminal_resolved)) if len(reaches) else 1.0,
        "allowed_terminal_path_fraction": (
            allowed_terminal_count / len(reaches) if len(reaches) else 1.0
        ),
        "trunk_preserved_fraction": float(np.mean(trunk_preserved)) if len(reaches) else 1.0,
        "reach_graph_issue_count": graph_issue_count + cycle_count,
        "mean_downstream_discharge_regression_count": mean_drop_count,
        "monthly_downstream_discharge_regression_count": monthly_drop_count,
        "maximum_mean_downstream_discharge_drop_fraction": maximum_mean_drop,
        "maximum_monthly_downstream_discharge_drop_fraction": maximum_monthly_drop,
        "reach_dead_end_count": dead_end_count,
        "major_river_dead_end_count": major_dead_end_count,
        "maximum_terminal_discharge_m3s": float(np.max(outlet_discharge, initial=0.0)),
    }


def _global_river_length_metrics(reaches: pa.Table) -> dict[str, float | int]:
    reach_ids = np.asarray(reaches["reach_id"], dtype=np.int64)
    downstream_ids = np.asarray(reaches["downstream_reach_id"], dtype=np.int64)
    basin_ids = np.asarray(reaches["basin_id"], dtype=np.int64)
    reach_kinds = np.asarray(reaches["reach_kind"].to_pylist(), dtype=object)
    upstream_ids = reaches["upstream_reach_ids"].to_pylist()
    row_by_id = {int(reach_id): row for row, reach_id in enumerate(reach_ids)}
    graph_issue_count = int(len(row_by_id) != len(reach_ids))

    reach_lengths_km = np.zeros(len(reaches), dtype=np.float64)
    for row, polyline in enumerate(reaches["polyline_on_cubed_sphere"].to_pylist()):
        xyz = np.asarray(polyline, dtype=np.float64)
        if len(xyz) < 2:
            continue
        segment_angles = np.arccos(np.clip(np.sum(xyz[:-1] * xyz[1:], axis=1), -1.0, 1.0))
        reach_lengths_km[row] = float(np.sum(segment_angles) * EARTH_RADIUS_KM)

    memo: dict[int, tuple[float, float]] = {}
    active: set[int] = set()

    def path_to_terminal(row: int) -> tuple[float, float]:
        nonlocal graph_issue_count
        if row in memo:
            return memo[row]
        if row in active:
            graph_issue_count += 1
            return (0.0, 0.0)
        active.add(row)
        downstream_row = row_by_id.get(int(downstream_ids[row]))
        downstream_length = (0.0, 0.0)
        if downstream_ids[row] >= 0:
            if downstream_row is None:
                graph_issue_count += 1
            else:
                downstream_length = path_to_terminal(downstream_row)
        active.remove(row)
        local_channel_length = reach_lengths_km[row] if reach_kinds[row] == "channel" else 0.0
        result = (
            reach_lengths_km[row] + downstream_length[0],
            local_channel_length + downstream_length[1],
        )
        memo[row] = result
        return result

    source_rows = [row for row, upstream in enumerate(upstream_ids) if not upstream]
    basin_longest_path: dict[int, float] = {}
    basin_longest_channel: dict[int, float] = {}
    source_paths: list[tuple[float, float]] = []
    for row in source_rows:
        path_length, channel_length = path_to_terminal(row)
        source_paths.append((path_length, channel_length))
        basin_id = int(basin_ids[row])
        basin_longest_path[basin_id] = max(basin_longest_path.get(basin_id, 0.0), path_length)
        basin_longest_channel[basin_id] = max(
            basin_longest_channel.get(basin_id, 0.0), channel_length
        )

    longest_path, channel_on_longest_path = max(
        source_paths, key=lambda values: (values[0], values[1]), default=(0.0, 0.0)
    )
    longest_channel = max((channel for _, channel in source_paths), default=0.0)
    return {
        "global_reach_graph_issue_count": graph_issue_count,
        "global_source_reach_count": len(source_rows),
        "global_total_reach_length_km": float(np.sum(reach_lengths_km)),
        "global_longest_source_to_terminal_path_km": longest_path,
        "global_longest_source_to_terminal_channel_km": longest_channel,
        "global_longest_river_channel_fraction": channel_on_longest_path / max(longest_path, 1e-12),
        "global_basin_count_with_3000km_path": sum(
            length >= 3_000.0 for length in basin_longest_path.values()
        ),
        "global_basin_count_with_4000km_path": sum(
            length >= 4_000.0 for length in basin_longest_path.values()
        ),
        "global_basin_count_with_5000km_path": sum(
            length >= 5_000.0 for length in basin_longest_path.values()
        ),
        "global_basin_count_with_3000km_channel": sum(
            length >= 3_000.0 for length in basin_longest_channel.values()
        ),
        "global_basin_count_with_4000km_channel": sum(
            length >= 4_000.0 for length in basin_longest_channel.values()
        ),
        "global_basin_count_with_5000km_channel": sum(
            length >= 5_000.0 for length in basin_longest_channel.values()
        ),
    }


KPI_SCHEMA = pa.schema(
    [
        ("kpi_id", pa.string()),
        ("category", pa.string()),
        ("scope", pa.string()),
        ("value", pa.float64()),
        ("unit", pa.string()),
        ("reference_minimum", pa.float64()),
        ("reference_maximum", pa.float64()),
        ("reference_scope", pa.string()),
        ("comparison_status", pa.string()),
        ("gate_kind", pa.string()),
        ("passed", pa.bool_()),
        ("reference_ids", pa.list_(pa.string())),
        ("note", pa.string()),
    ]
)

REACH_LOSS_SCHEMA = pa.schema(
    [
        ("upstream_reach_id", pa.int32()),
        ("downstream_reach_id", pa.int32()),
        ("month", pa.int32()),
        ("upstream_discharge_m3s", pa.float64()),
        ("downstream_discharge_m3s", pa.float64()),
        ("discharge_drop_m3s", pa.float64()),
        ("discharge_drop_fraction", pa.float64()),
        ("accounted_by_registered_storage", pa.bool_()),
        ("depression_ids", pa.list_(pa.int32())),
        ("waterbody_ids", pa.list_(pa.int32())),
        ("surface_water_depression_ids", pa.list_(pa.int32())),
    ]
)


def _reach_loss_catalog(
    reaches: pa.Table,
    depression_catalog: pa.Table,
    depression_by_cell: np.ndarray,
    lake_adjustments: pa.Table,
    *,
    maximum_drop_fraction: float,
) -> tuple[pa.Table, int, int, float]:
    reach_ids = np.asarray(reaches["reach_id"], dtype=np.int64)
    downstream_ids = np.asarray(reaches["downstream_reach_id"], dtype=np.int64)
    monthly = _fixed_list(reaches, "discharge_seasonal", MONTHS)
    row_by_id = {int(value): row for row, value in enumerate(reach_ids)}
    depression_ids = np.asarray(depression_catalog["depression_id"], dtype=np.int64)
    lake_ids = np.asarray(depression_catalog["lake_id"], dtype=np.int64)
    registered_waterbody = {
        int(depression_id): int(lake_id)
        for depression_id, lake_id in zip(depression_ids, lake_ids, strict=True)
        if lake_id >= 0
    }
    parent_paths = reaches["parent_cell_path"].to_pylist()
    storage_by_reach_entry_month: dict[tuple[int, int], set[int]] = {}
    for adjustment in lake_adjustments.select(
        [
            "terminal_depression_id",
            "month",
            "hydrograph_adjustment_m3s",
            "effective_reach_id",
            "joins_at_effective_entry",
        ]
    ).to_pylist():
        if adjustment["hydrograph_adjustment_m3s"] >= 0.0:
            continue
        effective_id = int(adjustment["effective_reach_id"])
        if effective_id < 0 or effective_id not in row_by_id:
            continue
        affected_reaches: list[int] = []
        seen: set[int] = set()
        reach_id = effective_id
        while reach_id >= 0:
            if reach_id in seen or reach_id not in row_by_id:
                raise RuntimeError("lake adjustment references an invalid downstream reach path")
            seen.add(reach_id)
            affected_reaches.append(reach_id)
            reach_id = int(downstream_ids[row_by_id[reach_id]])
        entry_start = 0 if adjustment["joins_at_effective_entry"] else 1
        for affected_reach in affected_reaches[entry_start:]:
            key = (affected_reach, int(adjustment["month"]))
            storage_by_reach_entry_month.setdefault(key, set()).add(
                int(adjustment["terminal_depression_id"])
            )
    rows: list[dict[str, object]] = []
    accounted_count = 0
    unaccounted_count = 0
    maximum_unaccounted_drop = 0.0
    for source, downstream_id in enumerate(downstream_ids):
        if downstream_id < 0 or int(downstream_id) not in row_by_id:
            continue
        target = row_by_id[int(downstream_id)]
        support = np.asarray(parent_paths[source], dtype=np.int64)
        if len(parent_paths[target]):
            support = np.append(support, int(parent_paths[target][0]))
        if np.any(support < 0) or np.any(support >= len(depression_by_cell)):
            raise RuntimeError("reach continuity audit found invalid parent support")
        crossed_depressions = sorted(
            int(value) for value in np.unique(depression_by_cell[support]) if value >= 0
        )
        crossed_waterbodies = sorted(
            {
                registered_waterbody[depression_id]
                for depression_id in crossed_depressions
                if depression_id in registered_waterbody
            }
        )
        drops = np.maximum(monthly[source] - monthly[target], 0.0)
        drop_fractions = drops / np.maximum(monthly[source], 1e-12)
        for month in np.flatnonzero(drop_fractions > maximum_drop_fraction):
            month_number = int(month) + 1
            target_storage = storage_by_reach_entry_month.get(
                (int(downstream_id), month_number), set()
            )
            source_storage = storage_by_reach_entry_month.get(
                (int(reach_ids[source]), month_number), set()
            )
            local_depressions = sorted(target_storage - source_storage)
            accounted = bool(crossed_waterbodies or local_depressions)
            accounted_count += int(accounted)
            unaccounted_count += int(not accounted)
            if not accounted:
                maximum_unaccounted_drop = max(
                    maximum_unaccounted_drop, float(drop_fractions[month])
                )
            rows.append(
                {
                    "upstream_reach_id": int(reach_ids[source]),
                    "downstream_reach_id": int(downstream_id),
                    "month": int(month) + 1,
                    "upstream_discharge_m3s": float(monthly[source, month]),
                    "downstream_discharge_m3s": float(monthly[target, month]),
                    "discharge_drop_m3s": float(drops[month]),
                    "discharge_drop_fraction": float(drop_fractions[month]),
                    "accounted_by_registered_storage": accounted,
                    "depression_ids": crossed_depressions,
                    "waterbody_ids": crossed_waterbodies,
                    "surface_water_depression_ids": local_depressions,
                }
            )
    return (
        pa.Table.from_pylist(rows, schema=REACH_LOSS_SCHEMA),
        accounted_count,
        unaccounted_count,
        maximum_unaccounted_drop,
    )


@stage(
    "hydrology_validation",
    inputs=(
        "surface_water_final",
        "lake_hydrographs",
        "outlet_incision",
        "hydrology",
        "climate",
        "cryosphere",
        "basin_refinement",
        "sea_level",
        "planet",
    ),
    outputs=(
        "HydrologyKpiCatalog",
        "HydrologyReachLossCatalog",
        "HydrologyValidationMetadata",
    ),
    version="v14",
)
def hydrology_validation_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = HydrologyValidationConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("hydrology validation requires topology: cubed_sphere")

    surface = _artifact_table(deps["surface_water_final"], "SurfaceWaterCandidateCatalog")
    surface_metadata = _artifact_metadata(deps["surface_water_final"], "SurfaceWaterMetadata")
    reaches = _artifact_table(deps["lake_hydrographs"], "LakeCoupledRiverReachCatalog")
    lake_hydrograph_metadata = _artifact_metadata(
        deps["lake_hydrographs"], "LakeHydrographMetadata"
    )
    lake_adjustments = _artifact_table(deps["lake_hydrographs"], "LakeHydrographAdjustmentCatalog")
    cryosphere_metadata = _artifact_metadata(deps["cryosphere"], "CryosphereMetadata")
    hydrology_metadata = _artifact_metadata(deps["hydrology"], "HydrologyMetadata")
    global_reaches = _artifact_table(deps["hydrology"], "RiverReachCatalog")
    depression_catalog = _artifact_table(deps["hydrology"], "DepressionCatalog")
    parents = _artifact_table(deps["basin_refinement"], "RefinedBasinParentCatalog")
    planet_metadata = _artifact_metadata(deps["planet"], "PlanetMetadata")

    planet_radius_km = _metadata_float(planet_metadata, "planet_radius_earth") * EARTH_RADIUS_KM
    areas_km2 = np.asarray(context.topology.cell_areas, dtype=np.float64) * planet_radius_km**2
    land_mask = _artifact_array(deps["sea_level"], "SurfaceOceanMask") < 0.5
    land_area_km2 = float(np.sum(areas_km2[land_mask]))

    monthly_snowfall = np.asarray(
        _artifact_array(deps["cryosphere"], "MonthlySnowfallMm"), dtype=np.float64
    ).reshape(MONTHS, -1)
    monthly_snowmelt = np.asarray(
        _artifact_array(deps["cryosphere"], "MonthlySnowmeltMm"), dtype=np.float64
    ).reshape(MONTHS, -1)
    monthly_snowpack = np.asarray(
        _artifact_array(deps["cryosphere"], "MonthlySnowWaterEquivalentMm"), dtype=np.float64
    ).reshape(MONTHS, -1)
    monthly_precipitation = np.asarray(
        _artifact_array(deps["climate"], "MonthlyPrecipitationMm"), dtype=np.float64
    ).reshape(MONTHS, -1)
    monthly_runoff = np.asarray(
        _artifact_array(deps["cryosphere"], "MonthlyRunoffPotentialMm"), dtype=np.float64
    ).reshape(MONTHS, -1)
    monthly_glacier_melt = np.asarray(
        _artifact_array(deps["cryosphere"], "MonthlyGlacierMeltMm"), dtype=np.float64
    ).reshape(MONTHS, -1)
    flat_land = land_mask.reshape(-1)
    flat_area = areas_km2.reshape(-1)
    snow_affected = np.max(monthly_snowpack, axis=0) >= config.snow_cover_threshold_mm
    perennial_snow = np.min(monthly_snowpack, axis=0) >= config.perennial_snow_threshold_mm
    snow_affected_land_fraction = float(
        np.sum(flat_area[flat_land & snow_affected]) / max(land_area_km2, 1e-12)
    )
    perennial_snow_land_fraction = float(
        np.sum(flat_area[flat_land & perennial_snow]) / max(land_area_km2, 1e-12)
    )

    parent_ids = np.asarray(parents["parent_cell_id"], dtype=np.int64)
    parent_inside = np.asarray(parents["inside_selected_basin"], dtype=bool)
    parent_excluded = np.asarray(parents["process_excluded"], dtype=bool)
    selected = parent_inside & ~parent_excluded
    selected_ids = parent_ids[selected]
    selected_area = np.asarray(parents["restricted_child_area_km2"], dtype=np.float64)[selected]
    if np.any(selected_ids < 0) or np.any(selected_ids >= monthly_runoff.shape[1]):
        raise RuntimeError("hydrology validation found an invalid refined parent identifier")

    def selected_monthly_volume(field: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.sum(
            field[:, selected_ids] * selected_area[None, :] / 1_000_000.0, axis=1
        )
        return result

    selected_snowfall = selected_monthly_volume(monthly_snowfall)
    selected_snowmelt = selected_monthly_volume(monthly_snowmelt)
    selected_glacier_melt = selected_monthly_volume(monthly_glacier_melt)
    selected_precipitation = selected_monthly_volume(monthly_precipitation)
    selected_runoff = selected_monthly_volume(monthly_runoff)
    selected_rainfall = np.maximum(selected_precipitation - selected_snowfall, 0.0)
    selected_liquid_input = selected_rainfall + selected_snowmelt + selected_glacier_melt

    chain_length, candidate_cycle_count, unknown_candidate_count = _candidate_chain_metrics(surface)
    reach_metrics = _reach_metrics(
        reaches,
        major_discharge_threshold_m3s=config.major_river_discharge_threshold_m3s,
        maximum_drop_fraction=config.maximum_downstream_discharge_drop_fraction,
    )
    global_river_metrics = _global_river_length_metrics(global_reaches)
    (
        reach_loss_catalog,
        accounted_reach_loss_count,
        unaccounted_reach_loss_count,
        maximum_unaccounted_drop,
    ) = _reach_loss_catalog(
        reaches,
        depression_catalog,
        _artifact_array(deps["hydrology"], "DepressionID").reshape(-1),
        lake_adjustments,
        maximum_drop_fraction=config.maximum_downstream_discharge_drop_fraction,
    )
    downstream_ids = np.asarray(surface["downstream_depression_id"], dtype=np.int64)
    annual_overflow = np.asarray(surface["annual_overflow_km3"], dtype=np.float64)
    monthly_overflow = _fixed_list(surface, "monthly_overflow_km3", MONTHS)
    terminal_monthly_overflow = np.sum(monthly_overflow[downstream_ids < 0], axis=0)
    annual_direct_inflow = _metadata_float(surface_metadata, "annual_direct_inflow_km3")
    annual_terminal_overflow = _metadata_float(surface_metadata, "annual_terminal_overflow_km3")
    annual_evaporation = _metadata_float(surface_metadata, "annual_evaporation_km3")
    annual_seepage = _metadata_float(surface_metadata, "annual_seepage_km3")
    active_source_area = _metadata_float(surface_metadata, "active_source_area_km2")
    accepted_standing_water_area = _metadata_float(
        surface_metadata, "accepted_standing_water_mean_area_km2"
    )
    global_runoff_depth = (
        _metadata_float(hydrology_metadata, "annual_runoff_km3")
        * 1_000_000.0
        / max(land_area_km2, 1e-12)
    )

    rows: list[dict[str, object]] = []

    def add(
        kpi_id: str,
        category: str,
        scope: str,
        value: float | int,
        unit: str,
        *,
        gate_kind: str = "diagnostic",
        minimum: float | None = None,
        maximum: float | None = None,
        reference_scope: str = "",
        reference_ids: tuple[str, ...] = (),
        note: str = "",
    ) -> None:
        numeric = float(value)
        passed: bool | None = None
        if gate_kind == "hard_invariant":
            passed = (minimum is None or numeric >= minimum) and (
                maximum is None or numeric <= maximum
            )
            status = "hard_pass" if passed else "hard_fail"
        elif gate_kind == "earth_diagnostic" and (minimum is not None or maximum is not None):
            inside = (minimum is None or numeric >= minimum) and (
                maximum is None or numeric <= maximum
            )
            status = "within_reference" if inside else "outside_reference"
        elif gate_kind == "capability":
            status = "implemented" if numeric >= 1.0 else "not_implemented"
        else:
            status = "reported"
        rows.append(
            {
                "kpi_id": kpi_id,
                "category": category,
                "scope": scope,
                "value": numeric,
                "unit": unit,
                "reference_minimum": minimum,
                "reference_maximum": maximum,
                "reference_scope": reference_scope,
                "comparison_status": status,
                "gate_kind": gate_kind,
                "passed": passed,
                "reference_ids": list(reference_ids),
                "note": note,
            }
        )

    add(
        "candidate_graph_valid",
        "topology",
        "selected_basin",
        int(
            _metadata_int(surface_metadata, "graph_valid") == 1
            and candidate_cycle_count == 0
            and unknown_candidate_count == 0
        ),
        "boolean",
        gate_kind="hard_invariant",
        minimum=1.0,
        note="The final lake-candidate spill graph must be acyclic and closed over known IDs.",
    )
    add(
        "candidate_water_balance_relative_error",
        "conservation",
        "selected_basin",
        _metadata_float(surface_metadata, "independent_water_balance_relative_error"),
        "fraction",
        gate_kind="hard_invariant",
        maximum=config.maximum_water_balance_relative_error,
    )
    add(
        "lake_hydrograph_network_balance_relative_error",
        "conservation",
        "selected_basin",
        _metadata_float(lake_hydrograph_metadata, "network_balance_relative_error"),
        "fraction",
        gate_kind="hard_invariant",
        maximum=_metadata_float(lake_hydrograph_metadata, "maximum_balance_relative_error"),
    )
    add(
        "minimum_lake_coupled_discharge_m3s",
        "river_continuity",
        "selected_basin",
        _metadata_float(lake_hydrograph_metadata, "minimum_coupled_discharge_m3s"),
        "m3/s",
        gate_kind="hard_invariant",
        minimum=-_metadata_float(lake_hydrograph_metadata, "maximum_negative_discharge_m3s"),
    )
    add(
        "cryosphere_mass_balance_relative_error",
        "conservation",
        "global",
        _metadata_float(cryosphere_metadata, "mass_balance_relative_error"),
        "fraction",
        gate_kind="hard_invariant",
        maximum=_metadata_float(cryosphere_metadata, "maximum_mass_balance_relative_error"),
    )
    add(
        "resolved_reach_terminal_fraction",
        "river_continuity",
        "selected_basin",
        reach_metrics["resolved_terminal_fraction"],
        "fraction",
        gate_kind="hard_invariant",
        minimum=1.0,
    )
    add(
        "allowed_reach_terminal_path_fraction",
        "river_continuity",
        "selected_basin",
        reach_metrics["allowed_terminal_path_fraction"],
        "fraction",
        gate_kind="hard_invariant",
        minimum=1.0,
        note="Every reach path must end at ocean or an explicit registered inland sink.",
    )
    add(
        "trunk_preserved_fraction",
        "river_continuity",
        "selected_basin",
        reach_metrics["trunk_preserved_fraction"],
        "fraction",
        gate_kind="hard_invariant",
        minimum=1.0,
    )
    for kpi_id in (
        "reach_graph_issue_count",
        "reach_dead_end_count",
        "major_river_dead_end_count",
    ):
        add(
            kpi_id,
            "river_continuity",
            "selected_basin",
            reach_metrics[kpi_id],
            "count",
            gate_kind="hard_invariant",
            maximum=0.0,
        )
    add(
        "mean_downstream_discharge_regression_count",
        "river_continuity",
        "selected_basin",
        reach_metrics["mean_downstream_discharge_regression_count"],
        "count",
        note=(
            "Raw mean reach-entry regressions are diagnostic; the monthly storage-aware "
            "audit is the executable continuity gate."
        ),
    )
    add(
        "unaccounted_monthly_downstream_discharge_regression_count",
        "river_continuity",
        "selected_basin",
        unaccounted_reach_loss_count,
        "count",
        gate_kind="hard_invariant",
        maximum=0.0,
        note="Monthly discharge may decrease only across an identified registered storage node.",
    )
    add(
        "outlet_resolution_contract_satisfied",
        "topology",
        "selected_basin",
        int(
            surface_metadata.get(
                "outlet_resolution_contract_satisfied",
                _metadata_int(surface_metadata, "outlet_correction_converged"),
            )
        ),
        "boolean",
        gate_kind="hard_invariant",
        minimum=1.0,
        note="Residual coarse outlet feedback must converge or be explicitly handed to regional refinement.",
    )
    add(
        "outlet_correction_converged",
        "topology",
        "selected_basin",
        _metadata_int(surface_metadata, "outlet_correction_converged"),
        "boolean",
        note="Diagnostic distinguishes physical coarse convergence from a bounded scale handoff.",
    )
    add(
        "regional_refinement_deferred_outlet_candidate_count",
        "topology",
        "selected_basin",
        int(surface_metadata.get("regional_refinement_deferred_outlet_candidate_count", 0)),
        "count",
        note="Round-limited moving spill edges retained as standing water for regional realization.",
    )

    add(
        "global_lake_land_area_fraction",
        "surface_water",
        "global",
        _metadata_float(hydrology_metadata, "lake_land_area_fraction"),
        "fraction",
        gate_kind="earth_diagnostic",
        minimum=0.018,
        maximum=0.037,
        reference_scope="global Earth land; inventory threshold dependent",
        reference_ids=("hydrolakes_2016", "verpoorter_2014"),
    )
    add(
        "global_lake_volume_km3",
        "surface_water",
        "global",
        _metadata_float(hydrology_metadata, "lake_volume_km3"),
        "km3",
        gate_kind="earth_diagnostic",
        minimum=150_000.0,
        maximum=250_000.0,
        reference_scope="global Earth lakes at least 0.1 km2; broad Earth-like tolerance",
        reference_ids=("hydrolakes_2016",),
    )
    add(
        "global_lake_area_weighted_mean_depth_m",
        "surface_water",
        "global",
        _metadata_float(hydrology_metadata, "mean_lake_depth_m"),
        "m",
        gate_kind="earth_diagnostic",
        minimum=40.0,
        maximum=100.0,
        reference_scope="global Earth lakes at least 0.1 km2; broad Earth-like tolerance",
        reference_ids=("hydrolakes_2016",),
    )
    add(
        "selected_basin_standing_water_area_fraction",
        "surface_water",
        "selected_basin",
        accepted_standing_water_area / max(active_source_area, 1e-12),
        "fraction",
        gate_kind="diagnostic",
        minimum=0.018,
        maximum=0.037,
        reference_scope="global Earth range shown for context only",
        reference_ids=("hydrolakes_2016", "verpoorter_2014"),
        note="A single basin must eventually be compared with climate and glacial analogs.",
    )
    add(
        "global_closed_drainage_land_fraction",
        "drainage",
        "global",
        _metadata_float(hydrology_metadata, "closed_drainage_land_fraction"),
        "fraction",
        gate_kind="earth_diagnostic",
        minimum=0.11,
        maximum=0.19,
        reference_scope="global Earth land",
        reference_ids=("merit_plus_2024",),
    )
    add(
        "global_runoff_depth_mm_year",
        "runoff",
        "global",
        global_runoff_depth,
        "mm/year",
        gate_kind="earth_diagnostic",
        minimum=220.0,
        maximum=280.0,
        reference_scope="Earth continental discharge normalized by land area",
        reference_ids=("dai_trenberth_2002",),
        note="Model runoff includes closed drainage; the reference is ocean discharge.",
    )
    add(
        "global_hydrologic_wetland_land_area_fraction",
        "surface_water",
        "global",
        _metadata_float(hydrology_metadata, "wetland_land_area_fraction"),
        "fraction",
        gate_kind="diagnostic",
        reference_scope="not directly comparable to ecological wetland inventories",
        reference_ids=("davidson_2018",),
    )
    add(
        "candidate_overflow_fraction",
        "lake_throughflow",
        "selected_basin",
        float(np.mean(annual_overflow > 1e-12)),
        "fraction",
    )
    add(
        "candidate_to_candidate_spill_fraction",
        "lake_throughflow",
        "selected_basin",
        float(np.mean(downstream_ids >= 0)),
        "fraction",
    )
    add(
        "candidate_network_terminal_throughflow_fraction",
        "lake_throughflow",
        "selected_basin",
        annual_terminal_overflow / max(annual_direct_inflow, 1e-12),
        "fraction_of_direct_inflow",
    )
    add(
        "candidate_network_consumptive_loss_fraction",
        "lake_throughflow",
        "selected_basin",
        (annual_evaporation + annual_seepage) / max(annual_direct_inflow, 1e-12),
        "fraction_of_direct_inflow",
    )
    add(
        "maximum_candidate_spill_chain_length",
        "lake_throughflow",
        "selected_basin",
        chain_length,
        "candidate_count",
    )
    add(
        "maximum_terminal_river_discharge_m3s",
        "river_continuity",
        "selected_basin",
        reach_metrics["maximum_terminal_discharge_m3s"],
        "m3/s",
    )
    add(
        "global_longest_source_to_terminal_river_path_km",
        "river_morphology",
        "global",
        global_river_metrics["global_longest_source_to_terminal_path_km"],
        "km",
        gate_kind="earth_diagnostic",
        minimum=4_000.0,
        maximum=8_000.0,
        reference_scope="Earthlike globe; broad longest-river morphology envelope",
        note="Includes registered lake connectors so a river is not truncated at standing water.",
    )
    add(
        "global_longest_source_to_terminal_channel_km",
        "river_morphology",
        "global",
        global_river_metrics["global_longest_source_to_terminal_channel_km"],
        "km",
        note="Counts physical vector-channel reaches and excludes lake connectors.",
    )
    add(
        "global_longest_river_channel_fraction",
        "river_morphology",
        "global",
        global_river_metrics["global_longest_river_channel_fraction"],
        "fraction",
        note="Diagnostic for major trunks being dominated by lake connectors.",
    )
    for threshold_km in (3_000, 4_000, 5_000):
        add(
            f"global_basin_count_with_{threshold_km}km_river_path",
            "river_morphology",
            "global",
            global_river_metrics[f"global_basin_count_with_{threshold_km}km_path"],
            "basin_count",
            note="Counts each drainage basin once using its longest source-to-terminal path.",
        )
        add(
            f"global_basin_count_with_{threshold_km}km_channel",
            "river_morphology",
            "global",
            global_river_metrics[f"global_basin_count_with_{threshold_km}km_channel"],
            "basin_count",
            note="Same basin-level count after excluding registered lake connectors.",
        )
    add(
        "global_river_reach_graph_issue_count",
        "river_continuity",
        "global",
        global_river_metrics["global_reach_graph_issue_count"],
        "count",
        gate_kind="hard_invariant",
        maximum=0.0,
    )
    add(
        "monthly_downstream_discharge_regression_count",
        "river_continuity",
        "selected_basin",
        reach_metrics["monthly_downstream_discharge_regression_count"],
        "count",
        note="Raw count before attributing losses to registered lake storage.",
    )
    add(
        "registered_storage_discharge_regression_count",
        "river_continuity",
        "selected_basin",
        accounted_reach_loss_count,
        "count",
    )
    add(
        "maximum_unaccounted_monthly_discharge_drop_fraction",
        "river_continuity",
        "selected_basin",
        maximum_unaccounted_drop,
        "fraction",
    )
    add(
        "maximum_monthly_downstream_discharge_drop_fraction",
        "river_continuity",
        "selected_basin",
        reach_metrics["maximum_monthly_downstream_discharge_drop_fraction"],
        "fraction",
    )
    add(
        "terminal_overflow_peak_to_mean_ratio",
        "seasonality",
        "selected_basin",
        float(np.max(terminal_monthly_overflow) / max(np.mean(terminal_monthly_overflow), 1e-12)),
        "ratio",
    )
    add(
        "selected_basin_runoff_depth_mm_year",
        "runoff",
        "selected_basin",
        float(np.sum(selected_runoff) * 1_000_000.0 / max(np.sum(selected_area), 1e-12)),
        "mm/year",
        note="Requires a climate-matched Earth basin distribution before scoring.",
    )

    add(
        "global_snow_affected_land_area_fraction",
        "cryosphere",
        "global",
        snow_affected_land_fraction,
        "fraction",
        note="Area whose maximum monthly snow-water equivalent exceeds the configured threshold.",
    )
    add(
        "global_perennial_snow_land_area_fraction",
        "cryosphere",
        "global",
        perennial_snow_land_fraction,
        "fraction",
        note="Area whose minimum monthly snow-water equivalent exceeds the configured threshold.",
    )
    add(
        "global_glacierized_land_area_fraction",
        "cryosphere",
        "global",
        _metadata_float(cryosphere_metadata, "glacierized_land_area_fraction"),
        "fraction",
    )
    add(
        "global_glacier_ice_land_area_fraction",
        "cryosphere",
        "global",
        _metadata_float(cryosphere_metadata, "glacier_ice_land_area_fraction"),
        "fraction",
        note="Area-equivalent glacier cover derived from the coarse-cell ice reservoir.",
    )
    sea_ice_reference = ("nsidc_sea_ice_index_v4",)
    add(
        "global_minimum_sea_ice_ocean_area_fraction",
        "cryosphere",
        "global",
        _metadata_float(cryosphere_metadata, "minimum_sea_ice_ocean_area_fraction"),
        "fraction",
        gate_kind="earth_diagnostic",
        minimum=0.02,
        maximum=0.08,
        reference_scope="Earth ocean; area-equivalent concentration rather than extent",
        reference_ids=sea_ice_reference,
    )
    add(
        "global_maximum_sea_ice_ocean_area_fraction",
        "cryosphere",
        "global",
        _metadata_float(cryosphere_metadata, "maximum_sea_ice_ocean_area_fraction"),
        "fraction",
        gate_kind="earth_diagnostic",
        minimum=0.05,
        maximum=0.12,
        reference_scope="Earth ocean; area-equivalent concentration rather than extent",
        reference_ids=sea_ice_reference,
    )
    add(
        "global_mean_sea_ice_ocean_area_fraction",
        "cryosphere",
        "global",
        _metadata_float(cryosphere_metadata, "mean_sea_ice_ocean_area_fraction"),
        "fraction",
        gate_kind="earth_diagnostic",
        minimum=0.035,
        maximum=0.09,
        reference_scope="Earth ocean; broad combined-hemisphere seasonal envelope",
        reference_ids=sea_ice_reference,
    )
    add(
        "selected_basin_snowmelt_liquid_input_fraction",
        "cryosphere",
        "selected_basin",
        float(np.sum(selected_snowmelt) / max(np.sum(selected_liquid_input), 1e-12)),
        "fraction",
        note="Snowmelt source fraction before soils, groundwater, and routing.",
    )
    add(
        "selected_basin_glacier_melt_liquid_input_fraction",
        "cryosphere",
        "selected_basin",
        float(np.sum(selected_glacier_melt) / max(np.sum(selected_liquid_input), 1e-12)),
        "fraction",
        note="Glacier-ice melt source fraction before soils, groundwater, and routing.",
    )
    add(
        "selected_basin_peak_snowmelt_month",
        "cryosphere",
        "selected_basin",
        int(np.argmax(selected_snowmelt)) + 1,
        "month_index",
    )
    add(
        "selected_basin_peak_runoff_month",
        "seasonality",
        "selected_basin",
        int(np.argmax(selected_runoff)) + 1,
        "month_index",
    )
    add(
        "seasonal_snow_storage_implemented",
        "capability",
        "global",
        1,
        "boolean",
        gate_kind="capability",
        note="Monthly snowpack is spun to a periodic climatology and contributes melt to runoff.",
    )
    add(
        "lake_reach_hydrograph_coupling_implemented",
        "capability",
        "selected_basin",
        _metadata_float(lake_hydrograph_metadata, "lake_reach_hydrograph_coupling_implemented"),
        "boolean",
        gate_kind="capability",
        note="Final terminal lake-network overflow and losses alter reach entry and exit hydrographs.",
    )
    add(
        "glacier_mass_balance_implemented",
        "capability",
        "global",
        _metadata_float(cryosphere_metadata, "glacier_mass_balance_implemented"),
        "boolean",
        gate_kind="capability",
        note="Age-tracked firn/ice storage has monthly melt and parameterized downslope transfer.",
    )
    add(
        "seasonal_sea_ice_implemented",
        "capability",
        "global",
        _metadata_float(cryosphere_metadata, "sea_ice_implemented"),
        "boolean",
        gate_kind="capability",
        reference_ids=("nsidc_sea_ice_index_v4",),
        note="Monthly thermodynamic concentration and thickness persist through the annual cycle.",
    )
    for kpi_id, note, reference_ids in (
        (
            "floodplain_inundation_implemented",
            "Floodplain widths exist, but monthly inundation depth and duration do not.",
            ("gfplain_2019",),
        ),
        (
            "ecological_wetland_confirmation_implemented",
            "Hydrologic wetness is not yet coupled to soils, groundwater, or vegetation.",
            ("davidson_2018",),
        ),
    ):
        add(
            kpi_id,
            "capability",
            "global",
            0,
            "boolean",
            gate_kind="capability",
            reference_ids=reference_ids,
            note=note,
        )

    catalog = pa.Table.from_pylist(rows, schema=KPI_SCHEMA)
    hard_failures = [
        str(row["kpi_id"])
        for row in rows
        if row["gate_kind"] == "hard_invariant" and row["passed"] is False
    ]
    earth_outside = [
        str(row["kpi_id"])
        for row in rows
        if row["gate_kind"] == "earth_diagnostic"
        and row["comparison_status"] == "outside_reference"
    ]
    capability_gaps = [
        str(row["kpi_id"])
        for row in rows
        if row["gate_kind"] == "capability" and row["comparison_status"] == "not_implemented"
    ]
    metadata = {
        **asdict(config),
        "model": "scale_aware_hydrology_kpi_profile_v4",
        "reference_profile_version": "earth_hydrology_v1",
        "references": list(EARTH_REFERENCE_PROFILE),
        "kpi_count": len(rows),
        "hard_gate_count": sum(row["gate_kind"] == "hard_invariant" for row in rows),
        "hard_gate_failure_count": len(hard_failures),
        "hard_gate_pass": int(not hard_failures),
        "hard_gate_failures": hard_failures,
        "earth_diagnostic_outside_count": len(earth_outside),
        "earth_diagnostics_outside_reference": earth_outside,
        "capability_gap_count": len(capability_gaps),
        "capability_gaps": capability_gaps,
        "selected_basin_id": _metadata_int(surface_metadata, "selected_basin_id"),
        "selected_basin_parent_count": int(np.count_nonzero(selected)),
        "selected_basin_parent_area_km2": float(np.sum(selected_area)),
        "global_land_area_km2": land_area_km2,
        "selected_monthly_snowfall_km3": selected_snowfall.tolist(),
        "selected_monthly_snowmelt_km3": selected_snowmelt.tolist(),
        "selected_monthly_glacier_melt_km3": selected_glacier_melt.tolist(),
        "selected_monthly_runoff_km3": selected_runoff.tolist(),
        "terminal_monthly_overflow_km3": terminal_monthly_overflow.tolist(),
        "reach_loss_record_count": reach_loss_catalog.num_rows,
        "registered_storage_reach_loss_count": accounted_reach_loss_count,
        "unaccounted_reach_loss_count": unaccounted_reach_loss_count,
        "comparison_rule": (
            "hard invariants gate topology and conservation; Earth ranges are diagnostics until "
            "conditioned by climate, relief, geology, scale, and inventory threshold"
        ),
    }
    context.logger.log_event(
        {"type": "hydrology_validation_summary", "stage": "hydrology_validation", **metadata}
    )
    return {
        "HydrologyKpiCatalog": catalog,
        "HydrologyReachLossCatalog": reach_loss_catalog,
        "HydrologyValidationMetadata": metadata,
    }


__all__ = [
    "HydrologyValidationConfig",
    "hydrology_validation_stage",
]
