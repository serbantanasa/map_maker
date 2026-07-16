"""Rust-backed sparse Hydrology Pass 2 bindings."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from cffi import FFI

from .._native import native_library_info, native_library_path

STABILIZED_CELL_DTYPE = np.dtype(
    [
        ("fine_cell_id", "=i4"),
        ("baseline_receiver_id", "=i4"),
        ("stabilized_receiver_id", "=i4"),
        ("baseline_anchor_cell_id", "=i4"),
        ("stabilized_anchor_cell_id", "=i4"),
        ("baseline_depression_id", "=i4"),
        ("stabilized_depression_id", "=i4"),
        ("anchor_kind", "u1"),
        ("receiver_changed", "u1"),
        ("depression_changed", "u1"),
        ("terminal_kind", "u1"),
        ("baseline_hydrologic_elevation_m", "=f8"),
        ("stabilized_hydrologic_elevation_m", "=f8"),
        ("baseline_fill_depth_m", "=f8"),
        ("stabilized_fill_depth_m", "=f8"),
        ("stabilized_flow_slope", "=f8"),
        ("contributing_area_km2", "=f8"),
        ("flow_direction_xyz", "=f4", (3,)),
    ],
    align=True,
)

_CDEF = """
typedef struct {
    int32_t fine_resolution;
    double minimum_depression_depth_m;
    double planet_radius_m;
} HydrologyPass2Config;

typedef struct {
    int32_t fine_cell_id;
    int32_t baseline_receiver_id;
    int32_t stabilized_receiver_id;
    int32_t baseline_anchor_cell_id;
    int32_t stabilized_anchor_cell_id;
    int32_t baseline_depression_id;
    int32_t stabilized_depression_id;
    uint8_t anchor_kind;
    uint8_t receiver_changed;
    uint8_t depression_changed;
    uint8_t terminal_kind;
    double baseline_hydrologic_elevation_m;
    double stabilized_hydrologic_elevation_m;
    double baseline_fill_depth_m;
    double stabilized_fill_depth_m;
    double stabilized_flow_slope;
    double contributing_area_km2;
    float flow_direction_xyz[3];
} StabilizedCellRecord;

typedef struct { StabilizedCellRecord* data; size_t len; } StabilizedCellArray;

typedef struct {
    int32_t cell_count;
    int32_t active_cell_count;
    int32_t channel_cell_count;
    int32_t excluded_cell_count;
    int32_t outside_cell_count;
    int32_t physical_trunk_edge_count;
    int32_t baseline_uncovered_cell_count;
    int32_t stabilized_uncovered_cell_count;
    int32_t baseline_depression_count;
    int32_t stabilized_depression_count;
    int32_t receiver_changed_cell_count;
    int32_t depression_changed_cell_count;
    int32_t graph_valid;
    int32_t trunk_preserved_valid;
    int32_t process_exclusion_valid;
    double active_area_km2;
    double receiver_changed_area_km2;
    double receiver_changed_area_fraction;
    double terminal_accumulated_area_km2;
    double contributing_area_residual_km2;
    double baseline_depression_area_km2;
    double stabilized_depression_area_km2;
    double baseline_depression_volume_km3;
    double stabilized_depression_volume_km3;
    double maximum_baseline_fill_depth_m;
    double maximum_stabilized_fill_depth_m;
} HydrologyPass2Stats;

int32_t hydrology_pass2_run(
    HydrologyPass2Config config,
    size_t cell_count,
    const int32_t* cell_ids,
    const double* terrain_before_m,
    const double* routing_surface_after_m,
    const double* cell_areas_km2,
    const float* cell_xyz,
    const uint8_t* anchor_kinds,
    const uint8_t* source_active,
    const int32_t* fixed_receiver_ids,
    StabilizedCellArray* cells_out,
    HydrologyPass2Stats* stats_out
);

void hydrology_pass2_free_cells(StabilizedCellArray array);
"""

_ffi = FFI()
_ffi.cdef(_CDEF)


def _validate_record_layout(c_name: str, dtype: np.dtype) -> None:
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


_validate_record_layout("StabilizedCellRecord", STABILIZED_CELL_DTYPE)
native_library_info("hydrology_pass2_native")
_lib = _ffi.dlopen(str(native_library_path("hydrology_pass2_native")))


def _input(values: np.ndarray, *, name: str, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != dtype:
        array = array.astype(dtype)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    return array


def _records(record_array: Any) -> np.ndarray:
    length = int(record_array.len)
    if length == 0:
        _lib.hydrology_pass2_free_cells(record_array)
        return np.empty(0, dtype=STABILIZED_CELL_DTYPE)
    buffer = _ffi.buffer(record_array.data, length * STABILIZED_CELL_DTYPE.itemsize)
    records = np.frombuffer(buffer, dtype=STABILIZED_CELL_DTYPE, count=length).copy()
    _lib.hydrology_pass2_free_cells(record_array)
    return records


def run_hydrology_pass2(
    *,
    controls: Mapping[str, float | int],
    cell_ids: np.ndarray,
    terrain_before_m: np.ndarray,
    routing_surface_after_m: np.ndarray,
    cell_areas_km2: np.ndarray,
    cell_xyz: np.ndarray,
    anchor_kinds: np.ndarray,
    source_active: np.ndarray,
    fixed_receiver_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    expected_controls = {
        "fine_resolution",
        "minimum_depression_depth_m",
        "planet_radius_m",
    }
    if set(controls) != expected_controls:
        raise ValueError(
            "Hydrology Pass 2 controls mismatch: "
            f"missing={sorted(expected_controls - set(controls))}, "
            f"extra={sorted(set(controls) - expected_controls)}"
        )
    inputs = {
        "cell_ids": _input(cell_ids, name="cell_ids", dtype=np.dtype(np.int32)),
        "terrain_before_m": _input(
            terrain_before_m, name="terrain_before_m", dtype=np.dtype(np.float64)
        ),
        "routing_surface_after_m": _input(
            routing_surface_after_m,
            name="routing_surface_after_m",
            dtype=np.dtype(np.float64),
        ),
        "cell_areas_km2": _input(cell_areas_km2, name="cell_areas_km2", dtype=np.dtype(np.float64)),
        "anchor_kinds": _input(anchor_kinds, name="anchor_kinds", dtype=np.dtype(np.uint8)),
        "source_active": _input(source_active, name="source_active", dtype=np.dtype(np.uint8)),
        "fixed_receiver_ids": _input(
            fixed_receiver_ids,
            name="fixed_receiver_ids",
            dtype=np.dtype(np.int32),
        ),
    }
    cell_count = len(inputs["cell_ids"])
    if any(len(values) != cell_count for values in inputs.values()):
        raise ValueError("Hydrology Pass 2 cell inputs must have equal lengths")
    xyz = np.ascontiguousarray(cell_xyz, dtype=np.float32)
    if xyz.shape != (cell_count, 3):
        raise ValueError(f"cell_xyz expected shape {(cell_count, 3)}, got {xyz.shape}")

    config = _ffi.new(
        "HydrologyPass2Config*",
        {
            "fine_resolution": int(controls["fine_resolution"]),
            "minimum_depression_depth_m": float(controls["minimum_depression_depth_m"]),
            "planet_radius_m": float(controls["planet_radius_m"]),
        },
    )[0]
    cells_out = _ffi.new("StabilizedCellArray*")
    stats_out = _ffi.new("HydrologyPass2Stats*")

    def pointer(array: np.ndarray, ctype: str):
        return _ffi.cast(
            f"const {ctype}*", _ffi.from_buffer(f"{ctype}[]", array, require_writable=False)
        )

    status = _lib.hydrology_pass2_run(
        config,
        cell_count,
        pointer(inputs["cell_ids"], "int32_t"),
        pointer(inputs["terrain_before_m"], "double"),
        pointer(inputs["routing_surface_after_m"], "double"),
        pointer(inputs["cell_areas_km2"], "double"),
        pointer(xyz.reshape(-1), "float"),
        pointer(inputs["anchor_kinds"], "uint8_t"),
        pointer(inputs["source_active"], "uint8_t"),
        pointer(inputs["fixed_receiver_ids"], "int32_t"),
        cells_out,
        stats_out,
    )
    if status != 0:
        messages = {
            1: "invalid controls or array lengths",
            2: "invalid cell identity or physical input",
            3: "invalid anchor or fixed trunk receiver",
            4: "null input or output pointer",
        }
        raise RuntimeError(
            f"hydrology_pass2_run failed: {messages.get(status, f'status {status}')}"
        )

    records = _records(cells_out[0])
    stats = stats_out[0]
    metadata = {
        name: int(getattr(stats, name))
        for name in (
            "cell_count",
            "active_cell_count",
            "channel_cell_count",
            "excluded_cell_count",
            "outside_cell_count",
            "physical_trunk_edge_count",
            "baseline_uncovered_cell_count",
            "stabilized_uncovered_cell_count",
            "baseline_depression_count",
            "stabilized_depression_count",
            "receiver_changed_cell_count",
            "depression_changed_cell_count",
            "graph_valid",
            "trunk_preserved_valid",
            "process_exclusion_valid",
        )
    }
    metadata.update(
        {
            name: float(getattr(stats, name))
            for name in (
                "active_area_km2",
                "receiver_changed_area_km2",
                "receiver_changed_area_fraction",
                "terminal_accumulated_area_km2",
                "contributing_area_residual_km2",
                "baseline_depression_area_km2",
                "stabilized_depression_area_km2",
                "baseline_depression_volume_km3",
                "stabilized_depression_volume_km3",
                "maximum_baseline_fill_depth_m",
                "maximum_stabilized_fill_depth_m",
            )
        }
    )
    return records, metadata


__all__ = ["run_hydrology_pass2"]
