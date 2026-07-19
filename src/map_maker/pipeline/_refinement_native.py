"""Rust-backed sparse selected-basin refinement bindings."""

from __future__ import annotations

from typing import Any, Mapping

from cffi import FFI
import numpy as np

from .._native import NativeLibraryAbiError, native_library_info, native_library_path

REFINEMENT_CONTROL_NAMES = {
    "coarse_resolution",
    "factor",
    "planet_radius_m",
    "terrain_seed",
    "terrain_noise_fraction",
}

REFINED_CELL_DTYPE = np.dtype(
    [
        ("fine_cell_id", "<i4"),
        ("parent_cell_id", "<i4"),
        ("face", "<i4"),
        ("row", "<i4"),
        ("col", "<i4"),
        ("xyz", "<f4", (3,)),
        ("area_km2", "<f8"),
        ("terrain_elevation_m", "<f4"),
        ("terrain_offset_m", "<f4"),
        ("parent_relief_m", "<f4"),
    ],
    align=True,
)

REFINED_REACH_DTYPE = np.dtype(
    [
        ("reach_id", "<i4"),
        ("path_offset", "<i4"),
        ("path_count", "<i4"),
        ("entry_fine_cell", "<i4"),
        ("exit_fine_cell", "<i4"),
        ("path_length_m", "<f8"),
    ],
    align=True,
)

REACH_CELL_DTYPE = np.dtype(
    [
        ("reach_id", "<i4"),
        ("fine_cell_id", "<i4"),
        ("parent_cell_id", "<i4"),
        ("path_order", "<i4"),
        ("reach_length_m", "<f8"),
        ("channel_fraction", "<f4"),
        ("valley_fraction", "<f4"),
        ("floodplain_fraction", "<f4"),
        ("potential_incised_volume_m3", "<f8"),
    ],
    align=True,
)

_CDEF = """
typedef struct {
    int32_t coarse_resolution;
    int32_t factor;
    double planet_radius_m;
    uint64_t terrain_seed;
    float terrain_noise_fraction;
} RefinementConfig;

typedef struct {
    int32_t fine_cell_id;
    int32_t parent_cell_id;
    int32_t face;
    int32_t row;
    int32_t col;
    float xyz[3];
    double area_km2;
    float terrain_elevation_m;
    float terrain_offset_m;
    float parent_relief_m;
} RefinedCellRecord;

typedef struct {
    int32_t reach_id;
    int32_t path_offset;
    int32_t path_count;
    int32_t entry_fine_cell;
    int32_t exit_fine_cell;
    double path_length_m;
} RefinedReachRecord;

typedef struct {
    int32_t reach_id;
    int32_t fine_cell_id;
    int32_t parent_cell_id;
    int32_t path_order;
    double reach_length_m;
    float channel_fraction;
    float valley_fraction;
    float floodplain_fraction;
    double potential_incised_volume_m3;
} ReachCellRecord;

typedef struct { RefinedCellRecord* data; size_t len; } RefinedCellArray;
typedef struct { RefinedReachRecord* data; size_t len; } RefinedReachArray;
typedef struct { ReachCellRecord* data; size_t len; } ReachCellArray;
typedef struct { int32_t* data; size_t len; } Int32Array;

typedef struct {
    int32_t parent_count;
    int32_t child_count;
    int32_t reach_count;
    int32_t reach_cell_count;
    int32_t fine_resolution;
    int32_t path_topology_valid;
    double selected_area_km2;
    double maximum_parent_area_relative_error;
    double maximum_parent_elevation_error_m;
    double total_reach_length_km;
    double represented_channel_area_km2;
    double represented_valley_area_km2;
    double represented_floodplain_area_km2;
    double total_potential_incised_volume_km3;
} RefinementStats;

uint32_t refinement_native_abi_version(void);
int32_t refinement_run_basin(
    const RefinementConfig* config,
    int32_t parent_count,
    const int32_t* parent_ids,
    const float* parent_elevation_m,
    const float* parent_relief_m,
    const double* parent_area_steradians,
    const uint8_t* parent_process_excluded,
    int32_t reach_count,
    const int32_t* reach_ids,
    const int32_t* reach_from_nodes,
    const int32_t* reach_to_nodes,
    const int32_t* reach_offsets,
    int32_t reach_parent_cell_count,
    const int32_t* reach_parent_cells,
    const uint8_t* reach_parent_channel_support,
    const float* channel_width_m,
    const float* valley_width_m,
    const float* floodplain_width_m,
    const float* incision_m,
    RefinedCellArray* cells_out,
    RefinedReachArray* reaches_out,
    Int32Array* path_cells_out,
    ReachCellArray* memberships_out,
    RefinementStats* stats_out
);
void refinement_free_cells(RefinedCellArray array);
void refinement_free_reaches(RefinedReachArray array);
void refinement_free_i32(Int32Array array);
void refinement_free_memberships(ReachCellArray array);
"""

_ffi = FFI()
_ffi.cdef(_CDEF)
native_library_info("refinement_native")
_lib = _ffi.dlopen(str(native_library_path("refinement_native")))
try:
    _abi_version = int(_lib.refinement_native_abi_version())
except AttributeError as exc:
    raise NativeLibraryAbiError("refinement_native lacks its required API") from exc
if _abi_version != 3:
    raise NativeLibraryAbiError(f"refinement_native uses ABI {_abi_version}; expected ABI 3")

for c_name, dtype in (
    ("RefinedCellRecord", REFINED_CELL_DTYPE),
    ("RefinedReachRecord", REFINED_REACH_DTYPE),
    ("ReachCellRecord", REACH_CELL_DTYPE),
):
    if _ffi.sizeof(c_name) != dtype.itemsize:
        raise NativeLibraryAbiError(
            f"{c_name} layout mismatch: C has {_ffi.sizeof(c_name)} bytes, "
            f"NumPy has {dtype.itemsize}"
        )


def _input(value: np.ndarray, *, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {array.dtype}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError(f"{name} must be contiguous and aligned")
    return array


def _copy_records(array: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    length = int(array.len)
    if length == 0:
        return np.empty(0, dtype=dtype)
    return np.frombuffer(
        _ffi.buffer(array.data, length * dtype.itemsize), dtype=dtype, count=length
    ).copy()


def _copy_i32(array: Any) -> np.ndarray:
    length = int(array.len)
    if length == 0:
        return np.empty(0, dtype=np.int32)
    return np.frombuffer(
        _ffi.buffer(array.data, length * np.dtype(np.int32).itemsize),
        dtype=np.int32,
        count=length,
    ).copy()


def run_basin_refinement(
    *,
    controls: Mapping[str, int | float],
    parent_ids: np.ndarray,
    parent_elevation_m: np.ndarray,
    parent_relief_m: np.ndarray,
    parent_area_steradians: np.ndarray,
    parent_process_excluded: np.ndarray,
    reach_ids: np.ndarray,
    reach_from_nodes: np.ndarray,
    reach_to_nodes: np.ndarray,
    reach_offsets: np.ndarray,
    reach_parent_cells: np.ndarray,
    reach_parent_channel_support: np.ndarray,
    channel_width_m: np.ndarray,
    valley_width_m: np.ndarray,
    floodplain_width_m: np.ndarray,
    incision_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    if set(controls) != REFINEMENT_CONTROL_NAMES:
        missing = sorted(REFINEMENT_CONTROL_NAMES - set(controls))
        extra = sorted(set(controls) - REFINEMENT_CONTROL_NAMES)
        raise ValueError(f"refinement controls mismatch; missing={missing}, extra={extra}")
    if any(not np.isfinite(value) for value in controls.values()):
        raise ValueError("refinement controls must be finite")

    arrays = {
        "parent_ids": _input(parent_ids, name="parent_ids", dtype=np.dtype(np.int32)),
        "parent_elevation_m": _input(
            parent_elevation_m, name="parent_elevation_m", dtype=np.dtype(np.float32)
        ),
        "parent_relief_m": _input(
            parent_relief_m, name="parent_relief_m", dtype=np.dtype(np.float32)
        ),
        "parent_area_steradians": _input(
            parent_area_steradians,
            name="parent_area_steradians",
            dtype=np.dtype(np.float64),
        ),
        "parent_process_excluded": _input(
            parent_process_excluded,
            name="parent_process_excluded",
            dtype=np.dtype(np.uint8),
        ),
        "reach_ids": _input(reach_ids, name="reach_ids", dtype=np.dtype(np.int32)),
        "reach_from_nodes": _input(
            reach_from_nodes, name="reach_from_nodes", dtype=np.dtype(np.int32)
        ),
        "reach_to_nodes": _input(reach_to_nodes, name="reach_to_nodes", dtype=np.dtype(np.int32)),
        "reach_offsets": _input(reach_offsets, name="reach_offsets", dtype=np.dtype(np.int32)),
        "reach_parent_cells": _input(
            reach_parent_cells, name="reach_parent_cells", dtype=np.dtype(np.int32)
        ),
        "reach_parent_channel_support": _input(
            reach_parent_channel_support,
            name="reach_parent_channel_support",
            dtype=np.dtype(np.uint8),
        ),
        "channel_width_m": _input(
            channel_width_m, name="channel_width_m", dtype=np.dtype(np.float32)
        ),
        "valley_width_m": _input(valley_width_m, name="valley_width_m", dtype=np.dtype(np.float32)),
        "floodplain_width_m": _input(
            floodplain_width_m, name="floodplain_width_m", dtype=np.dtype(np.float32)
        ),
        "incision_m": _input(incision_m, name="incision_m", dtype=np.dtype(np.float32)),
    }
    parent_count = len(arrays["parent_ids"])
    reach_count = len(arrays["reach_ids"])
    if parent_count == 0 or reach_count == 0:
        raise ValueError("selected basin must contain parent cells and river reaches")
    for name in (
        "parent_elevation_m",
        "parent_relief_m",
        "parent_area_steradians",
        "parent_process_excluded",
    ):
        if len(arrays[name]) != parent_count:
            raise ValueError(f"{name} must contain parent_count values")
    for name in (
        "reach_from_nodes",
        "reach_to_nodes",
        "channel_width_m",
        "valley_width_m",
        "floodplain_width_m",
        "incision_m",
    ):
        if len(arrays[name]) != reach_count:
            raise ValueError(f"{name} must contain reach_count values")
    if len(arrays["reach_offsets"]) != reach_count + 1:
        raise ValueError("reach_offsets must contain reach_count + 1 values")
    if len(arrays["reach_parent_channel_support"]) != len(arrays["reach_parent_cells"]):
        raise ValueError(
            "reach_parent_channel_support must contain one value per reach parent cell"
        )

    config = _ffi.new("RefinementConfig*")
    for name, value in controls.items():
        setattr(config[0], name, value)
    cells_out = _ffi.new("RefinedCellArray*")
    reaches_out = _ffi.new("RefinedReachArray*")
    paths_out = _ffi.new("Int32Array*")
    memberships_out = _ffi.new("ReachCellArray*")
    stats_out = _ffi.new("RefinementStats*")

    def pointer(name: str, c_type: str) -> Any:
        array = arrays[name]
        return _ffi.cast(
            f"const {c_type}*",
            _ffi.from_buffer(f"{c_type}[]", array, require_writable=False),
        )

    status = _lib.refinement_run_basin(
        config,
        parent_count,
        pointer("parent_ids", "int32_t"),
        pointer("parent_elevation_m", "float"),
        pointer("parent_relief_m", "float"),
        pointer("parent_area_steradians", "double"),
        pointer("parent_process_excluded", "uint8_t"),
        reach_count,
        pointer("reach_ids", "int32_t"),
        pointer("reach_from_nodes", "int32_t"),
        pointer("reach_to_nodes", "int32_t"),
        pointer("reach_offsets", "int32_t"),
        len(arrays["reach_parent_cells"]),
        pointer("reach_parent_cells", "int32_t"),
        pointer("reach_parent_channel_support", "uint8_t"),
        pointer("channel_width_m", "float"),
        pointer("valley_width_m", "float"),
        pointer("floodplain_width_m", "float"),
        pointer("incision_m", "float"),
        cells_out,
        reaches_out,
        paths_out,
        memberships_out,
        stats_out,
    )
    if status != 0:
        messages = {
            1: "null, empty, or length-mismatched argument",
            2: "invalid refinement controls",
            3: "invalid or duplicate parent cells",
            4: "invalid inherited reach path or physical attributes",
            5: "fine routing could not satisfy the inherited topology",
            6: "fine corridor support could not be allocated conservatively",
            50: "fine routing could not choose a parent-cell anchor",
            51: "fine routing exhausted an inherited parent boundary",
            52: "fine routing received no downstream target",
            53: "fine routing referenced a missing generated cell",
            54: "fine routing could not connect an anchor to its downstream target",
            55: "fine routing could not reconstruct its selected path",
            56: "fine routing could not find a reach entry anchor",
            57: "fine routing could not find its generated downstream reach",
            58: "fine routing could not find a terminal reach anchor",
            59: "fine routing generated a path shorter than one edge",
            60: "fine routing generated a non-adjacent path step",
            61: "fine routing generated a reverse-edge conflict",
        }
        raise RuntimeError(
            f"refinement_run_basin failed: {messages.get(status, f'status {status}')}"
        )

    try:
        cells = _copy_records(cells_out[0], dtype=REFINED_CELL_DTYPE)
        reaches = _copy_records(reaches_out[0], dtype=REFINED_REACH_DTYPE)
        path_cells = _copy_i32(paths_out[0])
        memberships = _copy_records(memberships_out[0], dtype=REACH_CELL_DTYPE)
    finally:
        _lib.refinement_free_cells(cells_out[0])
        _lib.refinement_free_reaches(reaches_out[0])
        _lib.refinement_free_i32(paths_out[0])
        _lib.refinement_free_memberships(memberships_out[0])
    stats = stats_out[0]
    metadata: dict[str, int | float] = {
        name: int(getattr(stats, name))
        for name in (
            "parent_count",
            "child_count",
            "reach_count",
            "reach_cell_count",
            "fine_resolution",
            "path_topology_valid",
        )
    }
    metadata.update(
        {
            name: float(getattr(stats, name))
            for name in (
                "selected_area_km2",
                "maximum_parent_area_relative_error",
                "maximum_parent_elevation_error_m",
                "total_reach_length_km",
                "represented_channel_area_km2",
                "represented_valley_area_km2",
                "represented_floodplain_area_km2",
                "total_potential_incised_volume_km3",
            )
        }
    )
    return cells, reaches, path_cells, memberships, metadata


__all__ = [
    "REFINED_CELL_DTYPE",
    "REFINED_REACH_DTYPE",
    "REACH_CELL_DTYPE",
    "run_basin_refinement",
]
