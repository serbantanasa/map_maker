"""Rust-backed refined seasonal surface-water balance bindings."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from cffi import FFI

from .._native import native_library_info, native_library_path

MONTHS = 12

SURFACE_WATER_CLASSES = {
    0: "dry_depression",
    1: "transient_storage",
    2: "seasonal_lake",
    3: "permanent_lake",
    4: "hydrologic_wetland",
}

CANDIDATE_DTYPE = np.dtype(
    [
        ("depression_id", "=i4"),
        ("downstream_depression_id", "=i4"),
        ("class_code", "=i4"),
        ("cell_count", "=i4"),
        ("catchment_cell_count", "=i4"),
        ("wet_month_count", "=i4"),
        ("solver_iterations", "=i4"),
        ("converged", "=i4"),
        ("open_outlet", "=i4"),
        ("catchment_area_km2", "=f8"),
        ("potential_water_area_km2", "=f8"),
        ("storage_capacity_km3", "=f8"),
        ("annual_direct_inflow_km3", "=f8"),
        ("annual_upstream_inflow_km3", "=f8"),
        ("annual_total_inflow_km3", "=f8"),
        ("annual_evaporation_km3", "=f8"),
        ("annual_seepage_km3", "=f8"),
        ("annual_overflow_km3", "=f8"),
        ("annual_terminal_overflow_km3", "=f8"),
        ("annual_storage_change_km3", "=f8"),
        ("water_balance_residual_km3", "=f8"),
        ("hydroperiod_fraction", "=f8"),
        ("minimum_water_area_km2", "=f8"),
        ("mean_water_area_km2", "=f8"),
        ("maximum_water_area_km2", "=f8"),
        ("mean_wetted_depth_m", "=f8"),
        ("maximum_mean_depth_m", "=f8"),
        ("salinity_index", "=f8"),
        ("monthly_direct_inflow_km3", "=f8", (MONTHS,)),
        ("monthly_upstream_inflow_km3", "=f8", (MONTHS,)),
        ("monthly_total_inflow_km3", "=f8", (MONTHS,)),
        ("monthly_evaporation_km3", "=f8", (MONTHS,)),
        ("monthly_seepage_km3", "=f8", (MONTHS,)),
        ("monthly_overflow_km3", "=f8", (MONTHS,)),
        ("monthly_storage_km3", "=f8", (MONTHS,)),
        ("monthly_water_area_km2", "=f8", (MONTHS,)),
    ],
    align=True,
)

CELL_DTYPE = np.dtype(
    [
        ("fine_cell_id", "=i4"),
        ("depression_id", "=i4"),
        ("class_code", "=i4"),
        ("potential_inundation_fraction", "=f4"),
        ("minimum_inundation_fraction", "=f4"),
        ("mean_inundation_fraction", "=f4"),
        ("maximum_inundation_fraction", "=f4"),
        ("monthly_inundation_fraction", "=f4", (MONTHS,)),
    ],
    align=True,
)

_CDEF = """
typedef struct {
    int32_t refinement_factor;
    int32_t minimum_solver_iterations;
    int32_t maximum_solver_iterations;
    int32_t transient_max_months;
    int32_t permanent_min_months;
    double convergence_tolerance_fraction;
    double open_water_evaporation_factor;
    double seepage_mm_year;
    double subgrid_relief_scale;
    double minimum_subgrid_relief_m;
    double maximum_connected_inundation_fraction;
    double minimum_wet_area_fraction;
    double wetland_max_mean_depth_m;
} SurfaceWaterConfig;

typedef struct {
    int32_t depression_id;
    int32_t downstream_depression_id;
    int32_t class_code;
    int32_t cell_count;
    int32_t catchment_cell_count;
    int32_t wet_month_count;
    int32_t solver_iterations;
    int32_t converged;
    int32_t open_outlet;
    double catchment_area_km2;
    double potential_water_area_km2;
    double storage_capacity_km3;
    double annual_direct_inflow_km3;
    double annual_upstream_inflow_km3;
    double annual_total_inflow_km3;
    double annual_evaporation_km3;
    double annual_seepage_km3;
    double annual_overflow_km3;
    double annual_terminal_overflow_km3;
    double annual_storage_change_km3;
    double water_balance_residual_km3;
    double hydroperiod_fraction;
    double minimum_water_area_km2;
    double mean_water_area_km2;
    double maximum_water_area_km2;
    double mean_wetted_depth_m;
    double maximum_mean_depth_m;
    double salinity_index;
    double monthly_direct_inflow_km3[12];
    double monthly_upstream_inflow_km3[12];
    double monthly_total_inflow_km3[12];
    double monthly_evaporation_km3[12];
    double monthly_seepage_km3[12];
    double monthly_overflow_km3[12];
    double monthly_storage_km3[12];
    double monthly_water_area_km2[12];
} SurfaceWaterCandidateRecord;
typedef struct { SurfaceWaterCandidateRecord* data; size_t len; } SurfaceWaterCandidateArray;

typedef struct {
    int32_t fine_cell_id;
    int32_t depression_id;
    int32_t class_code;
    float potential_inundation_fraction;
    float minimum_inundation_fraction;
    float mean_inundation_fraction;
    float maximum_inundation_fraction;
    float monthly_inundation_fraction[12];
} SurfaceWaterCellRecord;
typedef struct { SurfaceWaterCellRecord* data; size_t len; } SurfaceWaterCellArray;

typedef struct {
    int32_t cell_count;
    int32_t candidate_count;
    int32_t candidate_cell_count;
    int32_t owned_source_cell_count;
    int32_t dry_count;
    int32_t transient_count;
    int32_t seasonal_lake_count;
    int32_t permanent_lake_count;
    int32_t hydrologic_wetland_count;
    int32_t graph_valid;
    int32_t convergence_valid;
    int32_t fraction_valid;
    int32_t storage_valid;
    int32_t direct_catchment_valid;
    int32_t maximum_solver_iterations_used;
    double active_source_area_km2;
    double owned_catchment_area_km2;
    double potential_water_area_km2;
    double storage_capacity_km3;
    double annual_direct_inflow_km3;
    double annual_evaporation_km3;
    double annual_seepage_km3;
    double annual_terminal_overflow_km3;
    double annual_storage_change_km3;
    double water_balance_residual_km3;
    double water_balance_relative_error;
    double minimum_inundation_fraction;
    double maximum_inundation_fraction;
    double dry_mean_water_area_km2;
    double transient_mean_water_area_km2;
    double seasonal_lake_mean_water_area_km2;
    double permanent_lake_mean_water_area_km2;
    double hydrologic_wetland_mean_water_area_km2;
} SurfaceWaterStats;

int32_t surface_water_run(
    SurfaceWaterConfig config,
    size_t cell_count,
    size_t candidate_count,
    const int32_t* cell_ids,
    const int32_t* receiver_ids,
    const int32_t* depression_ids,
    const uint8_t* source_active,
    const double* area_km2,
    const double* terrain_elevation_m,
    const double* hydrologic_elevation_m,
    const float* parent_relief_m,
    const float* monthly_runoff_mm,
    const float* monthly_evaporation_mm,
    const float* sediment_accommodation,
    const int32_t* candidate_ids,
    const int32_t* spill_receiver_ids,
    SurfaceWaterCandidateArray* candidates_out,
    SurfaceWaterCellArray* cells_out,
    SurfaceWaterStats* stats_out
);
void surface_water_free_candidates(SurfaceWaterCandidateArray array);
void surface_water_free_cells(SurfaceWaterCellArray array);
size_t surface_water_native_struct_size(uint32_t kind);
"""

_ffi = FFI()
_ffi.cdef(_CDEF)


def _validate_layout(c_name: str, dtype: np.dtype) -> None:
    if not dtype.isnative or dtype.itemsize != _ffi.sizeof(c_name):
        raise ImportError(
            f"{c_name} native layout mismatch: numpy={dtype.itemsize}, "
            f"cffi={_ffi.sizeof(c_name)}, native_byte_order={dtype.isnative}"
        )
    for field in dtype.names or ():
        numpy_offset = int(dtype.fields[field][1])
        cffi_offset = int(_ffi.offsetof(c_name, field))
        if numpy_offset != cffi_offset:
            raise ImportError(
                f"{c_name}.{field} offset mismatch: numpy={numpy_offset}, cffi={cffi_offset}"
            )


_validate_layout("SurfaceWaterCandidateRecord", CANDIDATE_DTYPE)
_validate_layout("SurfaceWaterCellRecord", CELL_DTYPE)
native_library_info("surface_water_native")
_lib = _ffi.dlopen(str(native_library_path("surface_water_native")))
for _kind, _c_name in enumerate(
    (
        "SurfaceWaterConfig",
        "SurfaceWaterCandidateRecord",
        "SurfaceWaterCellRecord",
        "SurfaceWaterStats",
    )
):
    _native_size = int(_lib.surface_water_native_struct_size(_kind))
    _declared_size = int(_ffi.sizeof(_c_name))
    if _native_size != _declared_size:
        raise ImportError(
            f"{_c_name} Rust/CFFI size mismatch: native={_native_size}, " f"cffi={_declared_size}"
        )


def _input(values: np.ndarray, *, name: str, dtype: np.dtype, dimensions: int = 1) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != dtype or not array.flags.c_contiguous:
        array = np.ascontiguousarray(array, dtype=dtype)
    if array.ndim != dimensions:
        raise ValueError(f"{name} must be {dimensions}-dimensional, got {array.shape}")
    return array


def _records(record_array: Any, dtype: np.dtype, free) -> np.ndarray:
    length = int(record_array.len)
    if length == 0:
        free(record_array)
        return np.empty(0, dtype=dtype)
    buffer = _ffi.buffer(record_array.data, length * dtype.itemsize)
    records = np.frombuffer(buffer, dtype=dtype, count=length).copy()
    free(record_array)
    return records


def run_surface_water_balance(
    *,
    controls: Mapping[str, float | int],
    cell_ids: np.ndarray,
    receiver_ids: np.ndarray,
    depression_ids: np.ndarray,
    source_active: np.ndarray,
    area_km2: np.ndarray,
    terrain_elevation_m: np.ndarray,
    hydrologic_elevation_m: np.ndarray,
    parent_relief_m: np.ndarray,
    monthly_runoff_mm: np.ndarray,
    monthly_evaporation_mm: np.ndarray,
    sediment_accommodation: np.ndarray,
    candidate_ids: np.ndarray,
    spill_receiver_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    control_names = {
        "refinement_factor",
        "minimum_solver_iterations",
        "maximum_solver_iterations",
        "transient_max_months",
        "permanent_min_months",
        "convergence_tolerance_fraction",
        "open_water_evaporation_factor",
        "seepage_mm_year",
        "subgrid_relief_scale",
        "minimum_subgrid_relief_m",
        "maximum_connected_inundation_fraction",
        "minimum_wet_area_fraction",
        "wetland_max_mean_depth_m",
    }
    if set(controls) != control_names:
        raise ValueError(
            "Surface-water controls mismatch: "
            f"missing={sorted(control_names - set(controls))}, "
            f"extra={sorted(set(controls) - control_names)}"
        )
    inputs = {
        "cell_ids": _input(cell_ids, name="cell_ids", dtype=np.dtype(np.int32)),
        "receiver_ids": _input(receiver_ids, name="receiver_ids", dtype=np.dtype(np.int32)),
        "depression_ids": _input(depression_ids, name="depression_ids", dtype=np.dtype(np.int32)),
        "source_active": _input(source_active, name="source_active", dtype=np.dtype(np.uint8)),
        "area_km2": _input(area_km2, name="area_km2", dtype=np.dtype(np.float64)),
        "terrain_elevation_m": _input(
            terrain_elevation_m, name="terrain_elevation_m", dtype=np.dtype(np.float64)
        ),
        "hydrologic_elevation_m": _input(
            hydrologic_elevation_m,
            name="hydrologic_elevation_m",
            dtype=np.dtype(np.float64),
        ),
        "parent_relief_m": _input(
            parent_relief_m, name="parent_relief_m", dtype=np.dtype(np.float32)
        ),
        "sediment_accommodation": _input(
            sediment_accommodation,
            name="sediment_accommodation",
            dtype=np.dtype(np.float32),
        ),
    }
    cell_count = len(inputs["cell_ids"])
    if any(len(values) != cell_count for values in inputs.values()):
        raise ValueError("Surface-water cell inputs must have equal lengths")
    runoff = _input(
        monthly_runoff_mm,
        name="monthly_runoff_mm",
        dtype=np.dtype(np.float32),
        dimensions=2,
    )
    evaporation = _input(
        monthly_evaporation_mm,
        name="monthly_evaporation_mm",
        dtype=np.dtype(np.float32),
        dimensions=2,
    )
    if runoff.shape != (MONTHS, cell_count) or evaporation.shape != (MONTHS, cell_count):
        raise ValueError(
            "Surface-water monthly inputs must have shape "
            f"{(MONTHS, cell_count)}, got runoff={runoff.shape}, evaporation={evaporation.shape}"
        )
    candidates = _input(candidate_ids, name="candidate_ids", dtype=np.dtype(np.int32))
    spill_receivers = _input(
        spill_receiver_ids, name="spill_receiver_ids", dtype=np.dtype(np.int32)
    )
    if len(candidates) == 0 or len(candidates) != len(spill_receivers):
        raise ValueError("Surface-water candidate inputs must be non-empty and equal length")

    config = _ffi.new(
        "SurfaceWaterConfig*",
        {
            name: (
                int(controls[name])
                if name
                in {
                    "refinement_factor",
                    "minimum_solver_iterations",
                    "maximum_solver_iterations",
                    "transient_max_months",
                    "permanent_min_months",
                }
                else float(controls[name])
            )
            for name in control_names
        },
    )[0]
    candidates_out = _ffi.new("SurfaceWaterCandidateArray*")
    cells_out = _ffi.new("SurfaceWaterCellArray*")
    stats_out = _ffi.new("SurfaceWaterStats*")

    def pointer(array: np.ndarray, ctype: str):
        return _ffi.cast(
            f"const {ctype}*", _ffi.from_buffer(f"{ctype}[]", array, require_writable=False)
        )

    status = _lib.surface_water_run(
        config,
        cell_count,
        len(candidates),
        pointer(inputs["cell_ids"], "int32_t"),
        pointer(inputs["receiver_ids"], "int32_t"),
        pointer(inputs["depression_ids"], "int32_t"),
        pointer(inputs["source_active"], "uint8_t"),
        pointer(inputs["area_km2"], "double"),
        pointer(inputs["terrain_elevation_m"], "double"),
        pointer(inputs["hydrologic_elevation_m"], "double"),
        pointer(inputs["parent_relief_m"], "float"),
        pointer(runoff.reshape(-1), "float"),
        pointer(evaporation.reshape(-1), "float"),
        pointer(inputs["sediment_accommodation"], "float"),
        pointer(candidates, "int32_t"),
        pointer(spill_receivers, "int32_t"),
        candidates_out,
        cells_out,
        stats_out,
    )
    if status != 0:
        messages = {
            1: "invalid controls or array lengths",
            2: "invalid cell identity or physical input",
            3: "invalid candidate identity or hypsometry",
            4: "cyclic cell or candidate graph",
            5: "null input or output pointer",
        }
        raise RuntimeError(f"surface_water_run failed: {messages.get(status, f'status {status}')}")

    candidate_records = _records(
        candidates_out[0], CANDIDATE_DTYPE, _lib.surface_water_free_candidates
    )
    cell_records = _records(cells_out[0], CELL_DTYPE, _lib.surface_water_free_cells)
    stats = stats_out[0]
    integer_names = (
        "cell_count",
        "candidate_count",
        "candidate_cell_count",
        "owned_source_cell_count",
        "dry_count",
        "transient_count",
        "seasonal_lake_count",
        "permanent_lake_count",
        "hydrologic_wetland_count",
        "graph_valid",
        "convergence_valid",
        "fraction_valid",
        "storage_valid",
        "direct_catchment_valid",
        "maximum_solver_iterations_used",
    )
    float_names = (
        "active_source_area_km2",
        "owned_catchment_area_km2",
        "potential_water_area_km2",
        "storage_capacity_km3",
        "annual_direct_inflow_km3",
        "annual_evaporation_km3",
        "annual_seepage_km3",
        "annual_terminal_overflow_km3",
        "annual_storage_change_km3",
        "water_balance_residual_km3",
        "water_balance_relative_error",
        "minimum_inundation_fraction",
        "maximum_inundation_fraction",
        "dry_mean_water_area_km2",
        "transient_mean_water_area_km2",
        "seasonal_lake_mean_water_area_km2",
        "permanent_lake_mean_water_area_km2",
        "hydrologic_wetland_mean_water_area_km2",
    )
    metadata = {name: int(getattr(stats, name)) for name in integer_names}
    metadata.update({name: float(getattr(stats, name)) for name in float_names})
    return candidate_records, cell_records, metadata


__all__ = ["SURFACE_WATER_CLASSES", "run_surface_water_balance"]
