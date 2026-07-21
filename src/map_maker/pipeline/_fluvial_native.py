"""Rust-backed sparse-basin fluvial erosion bindings."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from cffi import FFI

from .._native import native_library_info, native_library_path

BED_PROFILE_DTYPE = np.dtype(
    [
        ("reach_id", "=i4"),
        ("fine_cell_id", "=i4"),
        ("parent_cell_id", "=i4"),
        ("path_order", "=i4"),
        ("terrain_elevation_m", "=f8"),
        ("bed_elevation_m", "=f8"),
        ("incision_depth_m", "=f8"),
        ("reach_length_m", "=f8"),
        ("eroded_volume_m3", "=f8"),
    ],
    align=True,
)

REACH_BUDGET_DTYPE = np.dtype(
    [
        ("reach_id", "=i4"),
        ("has_physical_bed", "u1"),
        ("entry_bed_elevation_m", "=f8"),
        ("exit_bed_elevation_m", "=f8"),
        ("minimum_realized_slope", "=f8"),
        ("maximum_incision_depth_m", "=f8"),
        ("upstream_input_volume_m3", "=f8"),
        ("local_erosion_volume_m3", "=f8"),
        ("bank_eroded_volume_m3", "=f8"),
        ("available_sediment_volume_m3", "=f8"),
        ("floodplain_deposition_volume_m3", "=f8"),
        ("downstream_transfer_volume_m3", "=f8"),
        ("terminal_deposition_volume_m3", "=f8"),
        ("exported_sediment_volume_m3", "=f8"),
    ],
    align=True,
)

CELL_BUDGET_DTYPE = np.dtype(
    [
        ("fine_cell_id", "=i4"),
        ("parent_cell_id", "=i4"),
        ("eroded_volume_m3", "=f8"),
        ("deposited_volume_m3", "=f8"),
        ("maximum_incision_depth_m", "=f8"),
    ],
    align=True,
)

_CDEF = """
typedef struct {
    double planet_radius_m;
    double minimum_bed_slope;
    double maximum_deposition_fraction;
    double deposition_slope_scale;
    double maximum_deposition_depth_m;
    double bank_incision_fraction;
} FluvialConfig;

typedef struct {
    int32_t reach_id;
    int32_t fine_cell_id;
    int32_t parent_cell_id;
    int32_t path_order;
    double terrain_elevation_m;
    double bed_elevation_m;
    double incision_depth_m;
    double reach_length_m;
    double eroded_volume_m3;
} BedProfileRecord;

typedef struct {
    int32_t reach_id;
    uint8_t has_physical_bed;
    double entry_bed_elevation_m;
    double exit_bed_elevation_m;
    double minimum_realized_slope;
    double maximum_incision_depth_m;
    double upstream_input_volume_m3;
    double local_erosion_volume_m3;
    double bank_eroded_volume_m3;
    double available_sediment_volume_m3;
    double floodplain_deposition_volume_m3;
    double downstream_transfer_volume_m3;
    double terminal_deposition_volume_m3;
    double exported_sediment_volume_m3;
} ReachBudgetRecord;

typedef struct {
    int32_t fine_cell_id;
    int32_t parent_cell_id;
    double eroded_volume_m3;
    double deposited_volume_m3;
    double maximum_incision_depth_m;
} CellBudgetRecord;

typedef struct { BedProfileRecord* data; size_t len; } BedProfileArray;
typedef struct { ReachBudgetRecord* data; size_t len; } ReachBudgetArray;
typedef struct { CellBudgetRecord* data; size_t len; } CellBudgetArray;

typedef struct {
    int32_t physical_node_count;
    int32_t physical_edge_count;
    int32_t physical_component_count;
    int32_t profile_record_count;
    int32_t reach_count;
    int32_t connector_reach_count;
    double maximum_incision_depth_m;
    double minimum_realized_slope;
    double total_eroded_volume_m3;
    double total_channel_eroded_volume_m3;
    double total_bank_eroded_volume_m3;
    double total_floodplain_deposition_volume_m3;
    double total_terminal_deposition_volume_m3;
    double total_exported_sediment_volume_m3;
    double sediment_conservation_residual_m3;
    double maximum_junction_bed_error_m;
    int32_t bed_profile_valid;
    int32_t sediment_conservation_valid;
} FluvialStats;

int32_t fluvial_run(
    FluvialConfig config,
    size_t cell_count,
    size_t reach_count,
    size_t membership_count,
    const int32_t* cell_ids,
    const int32_t* cell_parent_ids,
    const float* cell_terrain_m,
    const double* cell_areas_km2,
    const float* cell_xyz,
    const int32_t* reach_ids,
    const int32_t* downstream_reach_ids,
    const uint8_t* reach_kinds,
    const uint8_t* terminal_kinds,
    const float* channel_width_m,
    const float* reach_slope,
    const int32_t* membership_reach_ids,
    const int32_t* membership_cell_ids,
    const int32_t* membership_parent_ids,
    const int32_t* membership_path_order,
    const double* membership_reach_length_m,
    const float* membership_channel_fraction,
    const float* membership_valley_fraction,
    const float* membership_floodplain_fraction,
    BedProfileArray* profiles_out,
    ReachBudgetArray* reaches_out,
    CellBudgetArray* cells_out,
    FluvialStats* stats_out
);

void fluvial_free_profiles(BedProfileArray array);
void fluvial_free_reaches(ReachBudgetArray array);
void fluvial_free_cells(CellBudgetArray array);
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


for _c_name, _dtype in (
    ("BedProfileRecord", BED_PROFILE_DTYPE),
    ("ReachBudgetRecord", REACH_BUDGET_DTYPE),
    ("CellBudgetRecord", CELL_BUDGET_DTYPE),
):
    _validate_record_layout(_c_name, _dtype)

native_library_info("fluvial_native")
_lib = _ffi.dlopen(str(native_library_path("fluvial_native")))


def _input(values: np.ndarray, *, name: str, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != dtype:
        array = array.astype(dtype)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    return array


def _records(record_array: Any, *, dtype: np.dtype, free) -> np.ndarray:
    length = int(record_array.len)
    if length == 0:
        free(record_array)
        return np.empty(0, dtype=dtype)
    buffer = _ffi.buffer(record_array.data, length * dtype.itemsize)
    records = np.frombuffer(buffer, dtype=dtype, count=length).copy()
    free(record_array)
    return records


def run_fluvial_erosion(
    *,
    controls: Mapping[str, float],
    cell_ids: np.ndarray,
    cell_parent_ids: np.ndarray,
    cell_terrain_m: np.ndarray,
    cell_areas_km2: np.ndarray,
    cell_xyz: np.ndarray,
    reach_ids: np.ndarray,
    downstream_reach_ids: np.ndarray,
    reach_kinds: np.ndarray,
    terminal_kinds: np.ndarray,
    channel_width_m: np.ndarray,
    reach_slope: np.ndarray,
    membership_reach_ids: np.ndarray,
    membership_cell_ids: np.ndarray,
    membership_parent_ids: np.ndarray,
    membership_path_order: np.ndarray,
    membership_reach_length_m: np.ndarray,
    membership_channel_fraction: np.ndarray,
    membership_valley_fraction: np.ndarray,
    membership_floodplain_fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    expected_controls = {
        "planet_radius_m",
        "minimum_bed_slope",
        "maximum_deposition_fraction",
        "deposition_slope_scale",
        "maximum_deposition_depth_m",
        "bank_incision_fraction",
    }
    if set(controls) != expected_controls:
        raise ValueError(
            "fluvial controls mismatch: "
            f"missing={sorted(expected_controls - set(controls))}, "
            f"extra={sorted(set(controls) - expected_controls)}"
        )
    cell_inputs = {
        "cell_ids": _input(cell_ids, name="cell_ids", dtype=np.dtype(np.int32)),
        "cell_parent_ids": _input(
            cell_parent_ids, name="cell_parent_ids", dtype=np.dtype(np.int32)
        ),
        "cell_terrain_m": _input(cell_terrain_m, name="cell_terrain_m", dtype=np.dtype(np.float32)),
        "cell_areas_km2": _input(cell_areas_km2, name="cell_areas_km2", dtype=np.dtype(np.float64)),
    }
    cell_count = len(cell_inputs["cell_ids"])
    if any(len(values) != cell_count for values in cell_inputs.values()):
        raise ValueError("fluvial cell inputs must have equal lengths")
    xyz = np.ascontiguousarray(cell_xyz, dtype=np.float32)
    if xyz.shape != (cell_count, 3):
        raise ValueError(f"cell_xyz expected shape {(cell_count, 3)}, got {xyz.shape}")
    xyz_flat = xyz.reshape(-1)

    reach_inputs = {
        "reach_ids": _input(reach_ids, name="reach_ids", dtype=np.dtype(np.int32)),
        "downstream_reach_ids": _input(
            downstream_reach_ids, name="downstream_reach_ids", dtype=np.dtype(np.int32)
        ),
        "reach_kinds": _input(reach_kinds, name="reach_kinds", dtype=np.dtype(np.uint8)),
        "terminal_kinds": _input(terminal_kinds, name="terminal_kinds", dtype=np.dtype(np.uint8)),
        "channel_width_m": _input(
            channel_width_m, name="channel_width_m", dtype=np.dtype(np.float32)
        ),
        "reach_slope": _input(reach_slope, name="reach_slope", dtype=np.dtype(np.float32)),
    }
    reach_count = len(reach_inputs["reach_ids"])
    if any(len(values) != reach_count for values in reach_inputs.values()):
        raise ValueError("fluvial reach inputs must have equal lengths")

    membership_inputs = {
        "membership_reach_ids": _input(
            membership_reach_ids, name="membership_reach_ids", dtype=np.dtype(np.int32)
        ),
        "membership_cell_ids": _input(
            membership_cell_ids, name="membership_cell_ids", dtype=np.dtype(np.int32)
        ),
        "membership_parent_ids": _input(
            membership_parent_ids, name="membership_parent_ids", dtype=np.dtype(np.int32)
        ),
        "membership_path_order": _input(
            membership_path_order, name="membership_path_order", dtype=np.dtype(np.int32)
        ),
        "membership_reach_length_m": _input(
            membership_reach_length_m,
            name="membership_reach_length_m",
            dtype=np.dtype(np.float64),
        ),
        "membership_channel_fraction": _input(
            membership_channel_fraction,
            name="membership_channel_fraction",
            dtype=np.dtype(np.float32),
        ),
        "membership_valley_fraction": _input(
            membership_valley_fraction,
            name="membership_valley_fraction",
            dtype=np.dtype(np.float32),
        ),
        "membership_floodplain_fraction": _input(
            membership_floodplain_fraction,
            name="membership_floodplain_fraction",
            dtype=np.dtype(np.float32),
        ),
    }
    membership_count = len(membership_inputs["membership_reach_ids"])
    if any(len(values) != membership_count for values in membership_inputs.values()):
        raise ValueError("fluvial membership inputs must have equal lengths")

    config = _ffi.new(
        "FluvialConfig*",
        {
            name: float(controls[name])
            for name in (
                "planet_radius_m",
                "minimum_bed_slope",
                "maximum_deposition_fraction",
                "deposition_slope_scale",
                "maximum_deposition_depth_m",
                "bank_incision_fraction",
            )
        },
    )[0]
    profiles_out = _ffi.new("BedProfileArray*")
    reaches_out = _ffi.new("ReachBudgetArray*")
    cells_out = _ffi.new("CellBudgetArray*")
    stats_out = _ffi.new("FluvialStats*")

    def pointer(array: np.ndarray, ctype: str):
        return _ffi.cast(
            f"const {ctype}*", _ffi.from_buffer(f"{ctype}[]", array, require_writable=False)
        )

    status = _lib.fluvial_run(
        config,
        cell_count,
        reach_count,
        membership_count,
        pointer(cell_inputs["cell_ids"], "int32_t"),
        pointer(cell_inputs["cell_parent_ids"], "int32_t"),
        pointer(cell_inputs["cell_terrain_m"], "float"),
        pointer(cell_inputs["cell_areas_km2"], "double"),
        pointer(xyz_flat, "float"),
        pointer(reach_inputs["reach_ids"], "int32_t"),
        pointer(reach_inputs["downstream_reach_ids"], "int32_t"),
        pointer(reach_inputs["reach_kinds"], "uint8_t"),
        pointer(reach_inputs["terminal_kinds"], "uint8_t"),
        pointer(reach_inputs["channel_width_m"], "float"),
        pointer(reach_inputs["reach_slope"], "float"),
        pointer(membership_inputs["membership_reach_ids"], "int32_t"),
        pointer(membership_inputs["membership_cell_ids"], "int32_t"),
        pointer(membership_inputs["membership_parent_ids"], "int32_t"),
        pointer(membership_inputs["membership_path_order"], "int32_t"),
        pointer(membership_inputs["membership_reach_length_m"], "double"),
        pointer(membership_inputs["membership_channel_fraction"], "float"),
        pointer(membership_inputs["membership_valley_fraction"], "float"),
        pointer(membership_inputs["membership_floodplain_fraction"], "float"),
        profiles_out,
        reaches_out,
        cells_out,
        stats_out,
    )
    if status != 0:
        messages = {
            1: "invalid controls or array lengths",
            2: "null input or output pointer",
            3: "unknown or inconsistent cell/reach reference",
            4: "invalid channel/connector physical support",
            5: "cyclic or reversed physical/reach graph",
            6: "non-finite or invalid physical input",
        }
        raise RuntimeError(f"fluvial_run failed: {messages.get(status, f'status {status}')}")

    profiles = _records(profiles_out[0], dtype=BED_PROFILE_DTYPE, free=_lib.fluvial_free_profiles)
    reaches = _records(reaches_out[0], dtype=REACH_BUDGET_DTYPE, free=_lib.fluvial_free_reaches)
    cells = _records(cells_out[0], dtype=CELL_BUDGET_DTYPE, free=_lib.fluvial_free_cells)
    stats = stats_out[0]
    metadata = {
        "physical_node_count": int(stats.physical_node_count),
        "physical_edge_count": int(stats.physical_edge_count),
        "physical_component_count": int(stats.physical_component_count),
        "profile_record_count": int(stats.profile_record_count),
        "reach_count": int(stats.reach_count),
        "connector_reach_count": int(stats.connector_reach_count),
        "maximum_incision_depth_m": float(stats.maximum_incision_depth_m),
        "minimum_realized_slope": float(stats.minimum_realized_slope),
        "total_eroded_volume_m3": float(stats.total_eroded_volume_m3),
        "total_channel_eroded_volume_m3": float(stats.total_channel_eroded_volume_m3),
        "total_bank_eroded_volume_m3": float(stats.total_bank_eroded_volume_m3),
        "total_floodplain_deposition_volume_m3": float(stats.total_floodplain_deposition_volume_m3),
        "total_terminal_deposition_volume_m3": float(stats.total_terminal_deposition_volume_m3),
        "total_exported_sediment_volume_m3": float(stats.total_exported_sediment_volume_m3),
        "sediment_conservation_residual_m3": float(stats.sediment_conservation_residual_m3),
        "maximum_junction_bed_error_m": float(stats.maximum_junction_bed_error_m),
        "bed_profile_valid": int(stats.bed_profile_valid),
        "sediment_conservation_valid": int(stats.sediment_conservation_valid),
    }
    return profiles, reaches, cells, metadata


__all__ = ["run_fluvial_erosion"]
