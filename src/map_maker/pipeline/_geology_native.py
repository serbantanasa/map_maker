"""Rust-backed bindings for geological province initialization."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
from cffi import FFI

from .._native import NativeLibraryAbiError, native_library_info, native_library_path

CUBED_SPHERE_GEOLOGY_ABI_VERSION = 1

PROVINCE_CLASSES = {
    1: "shield",
    2: "stable_platform",
    3: "sedimentary_basin",
    4: "orogen",
    5: "continental_rift",
    6: "continental_arc",
    7: "shelf_or_passive_margin",
    8: "abyssal_basin",
    9: "oceanic_ridge",
    10: "intra_oceanic_arc",
    11: "volcanic_province",
}

BOUNDARY_REGIMES = {
    1: "inactive_boundary",
    2: "continental_collision",
    3: "subduction_margin",
    4: "intra_oceanic_subduction",
    5: "continental_rift",
    6: "spreading_ridge",
    7: "transform",
}

PROVINCE_RECORD_DTYPE = np.dtype(
    [
        ("province_id", "<i4"),
        ("class_code", "<i4"),
        ("parent_plate_id", "<i4"),
        ("cell_count", "<i4"),
        ("area_steradians", "<f8"),
        ("mean_crust_age_ga", "<f4"),
        ("mean_rock_strength", "<f4"),
        ("mean_accommodation", "<f4"),
        ("mean_confidence", "<f4"),
    ],
    align=True,
)

BOUNDARY_RECORD_DTYPE = np.dtype(
    [
        ("segment_id", "<i4"),
        ("regime_code", "<i4"),
        ("plate_a", "<i4"),
        ("plate_b", "<i4"),
        ("edge_count", "<i4"),
        ("angular_length", "<f8"),
        ("mean_compression", "<f4"),
        ("mean_extension", "<f4"),
        ("mean_shear", "<f4"),
        ("mean_subduction", "<f4"),
        ("mean_confidence", "<f4"),
    ],
    align=True,
)

_CDEF = """
typedef struct {
    int32_t province_id;
    int32_t class_code;
    int32_t parent_plate_id;
    int32_t cell_count;
    double area_steradians;
    float mean_crust_age_ga;
    float mean_rock_strength;
    float mean_accommodation;
    float mean_confidence;
} ProvinceRecord;
typedef struct { ProvinceRecord* data; size_t len; } ProvinceRecordArray;

typedef struct {
    int32_t segment_id;
    int32_t regime_code;
    int32_t plate_a;
    int32_t plate_b;
    int32_t edge_count;
    double angular_length;
    float mean_compression;
    float mean_extension;
    float mean_shear;
    float mean_subduction;
    float mean_confidence;
} BoundarySegmentRecord;
typedef struct { BoundarySegmentRecord* data; size_t len; } BoundarySegmentRecordArray;

typedef struct {
    int32_t province_count;
    int32_t boundary_segment_count;
    int32_t active_boundary_segment_count;
    int32_t mixed_plate_province_count;
    double continental_area;
    double oceanic_area;
    float mean_crust_age_ga;
    float mean_confidence;
} GeologyStats;

uint32_t cubed_sphere_geology_abi_version(void);
int32_t geology_run_cubed_sphere(
    int32_t cell_count,
    float world_age_ga,
    int32_t plate_components,
    const double* area,
    const int32_t* neighbors,
    const float* plate_field,
    const float* subduction,
    const float* isostasy,
    const float* uplift,
    const float* subsidence,
    const float* compression,
    const float* extension,
    const float* shear,
    const float* margin,
    const float* stiffness,
    const float* proto_ocean,
    int32_t* province_id_out,
    uint8_t* province_class_out,
    float* crust_age_out,
    float* rock_strength_out,
    float* accommodation_out,
    float* province_confidence_out,
    int32_t* boundary_segment_id_out,
    uint8_t* boundary_regime_out,
    float* boundary_confidence_out,
    ProvinceRecordArray* provinces_out,
    BoundarySegmentRecordArray* segments_out,
    GeologyStats* stats_out
);
void geology_free_provinces(ProvinceRecordArray array);
void geology_free_boundary_segments(BoundarySegmentRecordArray array);
"""


def _read_float32(array: np.ndarray, *, name: str) -> np.ndarray:
    value = np.array(array, copy=False)
    if value.dtype != np.float32:
        raise ValueError(f"{name} must be float32, got {value.dtype}")
    if not value.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be contiguous")
    if not value.flags["ALIGNED"]:
        raise ValueError(f"{name} must be aligned")
    return value


def _write_array(array: np.ndarray, *, name: str, dtype: np.dtype) -> np.ndarray:
    value = np.array(array, copy=False)
    if value.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {value.dtype}")
    if not value.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be contiguous")
    if not value.flags["ALIGNED"]:
        raise ValueError(f"{name} must be aligned")
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _require_disjoint(buffers: dict[str, np.ndarray]) -> None:
    entries = list(buffers.items())
    for index, (first_name, first) in enumerate(entries):
        for second_name, second in entries[index + 1 :]:
            if np.shares_memory(first, second):
                raise ValueError(f"{first_name} and {second_name} buffers must not overlap")


_ffi = FFI()
_ffi.cdef(_CDEF)
native_library_info("geology_native")
_lib = _ffi.dlopen(str(native_library_path("geology_native")))
try:
    _cubed_sphere_abi = int(_lib.cubed_sphere_geology_abi_version())
except AttributeError as exc:
    raise NativeLibraryAbiError(
        "geology_native lacks the cubed-sphere API; rebuild native libraries"
    ) from exc
if _cubed_sphere_abi != CUBED_SPHERE_GEOLOGY_ABI_VERSION:
    raise NativeLibraryAbiError(
        "geology_native cubed-sphere ABI "
        f"{_cubed_sphere_abi} does not match expected {CUBED_SPHERE_GEOLOGY_ABI_VERSION}"
    )


def _copy_records(record_array: Any, *, dtype: np.dtype, free) -> np.ndarray:
    length = int(record_array.len)
    if length <= 0 or int(_ffi.cast("uintptr_t", record_array.data)) == 0:
        free(record_array)
        return np.empty(0, dtype=dtype)
    try:
        buffer = _ffi.buffer(record_array.data, length * dtype.itemsize)
        return np.frombuffer(buffer, dtype=dtype, count=length).copy()
    finally:
        free(record_array)


def _province_table(records: np.ndarray) -> pa.Table:
    class_codes = records["class_code"]
    return pa.table(
        {
            "province_id": pa.array(records["province_id"], type=pa.int32()),
            "class_code": pa.array(class_codes, type=pa.int32()),
            "class_name": pa.array(
                [PROVINCE_CLASSES.get(int(code), "unknown") for code in class_codes],
                type=pa.string(),
            ),
            "parent_plate_id": pa.array(records["parent_plate_id"], type=pa.int32()),
            "cell_count": pa.array(records["cell_count"], type=pa.int32()),
            "area_steradians": pa.array(records["area_steradians"], type=pa.float64()),
            "mean_crust_age_ga": pa.array(records["mean_crust_age_ga"], type=pa.float32()),
            "mean_rock_strength": pa.array(records["mean_rock_strength"], type=pa.float32()),
            "mean_accommodation": pa.array(records["mean_accommodation"], type=pa.float32()),
            "mean_confidence": pa.array(records["mean_confidence"], type=pa.float32()),
        }
    )


def _boundary_table(records: np.ndarray) -> pa.Table:
    regime_codes = records["regime_code"]
    return pa.table(
        {
            "segment_id": pa.array(records["segment_id"], type=pa.int32()),
            "regime_code": pa.array(regime_codes, type=pa.int32()),
            "regime_name": pa.array(
                [BOUNDARY_REGIMES.get(int(code), "unknown") for code in regime_codes],
                type=pa.string(),
            ),
            "plate_a": pa.array(records["plate_a"], type=pa.int32()),
            "plate_b": pa.array(records["plate_b"], type=pa.int32()),
            "edge_count": pa.array(records["edge_count"], type=pa.int32()),
            "angular_length": pa.array(records["angular_length"], type=pa.float64()),
            "mean_compression": pa.array(records["mean_compression"], type=pa.float32()),
            "mean_extension": pa.array(records["mean_extension"], type=pa.float32()),
            "mean_shear": pa.array(records["mean_shear"], type=pa.float32()),
            "mean_subduction": pa.array(records["mean_subduction"], type=pa.float32()),
            "mean_confidence": pa.array(records["mean_confidence"], type=pa.float32()),
        }
    )


def run_cubed_sphere_geology(
    *,
    world_age_ga: float,
    areas: np.ndarray,
    neighbors: np.ndarray,
    plate_field: np.ndarray,
    subduction: np.ndarray,
    isostasy: np.ndarray,
    uplift: np.ndarray,
    subsidence: np.ndarray,
    compression: np.ndarray,
    extension: np.ndarray,
    shear: np.ndarray,
    margin: np.ndarray,
    stiffness: np.ndarray,
    proto_ocean: np.ndarray,
    province_id_out: np.ndarray,
    province_class_out: np.ndarray,
    crust_age_out: np.ndarray,
    rock_strength_out: np.ndarray,
    accommodation_out: np.ndarray,
    province_confidence_out: np.ndarray,
    boundary_segment_id_out: np.ndarray,
    boundary_regime_out: np.ndarray,
    boundary_confidence_out: np.ndarray,
) -> tuple[pa.Table, pa.Table, dict[str, Any]]:
    area_array = np.require(areas, dtype=np.float64, requirements=["C", "A"])
    neighbor_array = np.require(neighbors, dtype=np.int32, requirements=["C", "A"])
    if (
        area_array.ndim != 3
        or area_array.shape[0] != 6
        or area_array.shape[1] != area_array.shape[2]
    ):
        raise ValueError("areas must have shape (6, n, n)")
    shape = area_array.shape
    if neighbor_array.shape != (*shape, 4):
        raise ValueError(f"neighbors must have shape {(*shape, 4)}")
    plate_array = _read_float32(plate_field, name="plate_field")
    if plate_array.ndim != 4 or plate_array.shape[:3] != shape or plate_array.shape[3] < 7:
        raise ValueError(f"plate_field must have shape {(*shape, 7)} or more components")

    inputs = {
        name: _read_float32(array, name=name)
        for name, array in {
            "subduction": subduction,
            "isostasy": isostasy,
            "uplift": uplift,
            "subsidence": subsidence,
            "compression": compression,
            "extension": extension,
            "shear": shear,
            "margin": margin,
            "stiffness": stiffness,
            "proto_ocean": proto_ocean,
        }.items()
    }
    outputs = {
        "province_id_out": _write_array(
            province_id_out, name="province_id_out", dtype=np.dtype(np.int32)
        ),
        "province_class_out": _write_array(
            province_class_out, name="province_class_out", dtype=np.dtype(np.uint8)
        ),
        "crust_age_out": _write_array(
            crust_age_out, name="crust_age_out", dtype=np.dtype(np.float32)
        ),
        "rock_strength_out": _write_array(
            rock_strength_out, name="rock_strength_out", dtype=np.dtype(np.float32)
        ),
        "accommodation_out": _write_array(
            accommodation_out, name="accommodation_out", dtype=np.dtype(np.float32)
        ),
        "province_confidence_out": _write_array(
            province_confidence_out,
            name="province_confidence_out",
            dtype=np.dtype(np.float32),
        ),
        "boundary_segment_id_out": _write_array(
            boundary_segment_id_out,
            name="boundary_segment_id_out",
            dtype=np.dtype(np.int32),
        ),
        "boundary_regime_out": _write_array(
            boundary_regime_out, name="boundary_regime_out", dtype=np.dtype(np.uint8)
        ),
        "boundary_confidence_out": _write_array(
            boundary_confidence_out,
            name="boundary_confidence_out",
            dtype=np.dtype(np.float32),
        ),
    }
    for name, array in inputs.items():
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    edge_shape = (*shape, 4)
    for name, array in outputs.items():
        expected_shape = edge_shape if name.startswith("boundary_") else shape
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    _require_disjoint(
        {
            "areas": area_array,
            "neighbors": neighbor_array,
            "plate_field": plate_array,
            **inputs,
            **outputs,
        }
    )

    provinces_ptr = _ffi.new("ProvinceRecordArray*")
    segments_ptr = _ffi.new("BoundarySegmentRecordArray*")
    stats_ptr = _ffi.new("GeologyStats*")
    result = _lib.geology_run_cubed_sphere(
        int(np.prod(shape, dtype=np.int64)),
        float(world_age_ga),
        int(plate_array.shape[3]),
        _ffi.cast("const double*", _ffi.from_buffer("double[]", area_array)),
        _ffi.cast("const int32_t*", _ffi.from_buffer("int32_t[]", neighbor_array)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", plate_array, require_writable=False)),
        *[
            _ffi.cast("const float*", _ffi.from_buffer("float[]", array, require_writable=False))
            for array in inputs.values()
        ],
        _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", outputs["province_id_out"])),
        _ffi.cast("uint8_t*", _ffi.from_buffer("uint8_t[]", outputs["province_class_out"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", outputs["crust_age_out"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", outputs["rock_strength_out"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", outputs["accommodation_out"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", outputs["province_confidence_out"])),
        _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", outputs["boundary_segment_id_out"])),
        _ffi.cast("uint8_t*", _ffi.from_buffer("uint8_t[]", outputs["boundary_regime_out"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", outputs["boundary_confidence_out"])),
        provinces_ptr,
        segments_ptr,
        stats_ptr,
    )
    if result != 0:
        raise RuntimeError(f"geology_run_cubed_sphere failed with code {result}")

    province_records = _copy_records(
        provinces_ptr[0], dtype=PROVINCE_RECORD_DTYPE, free=_lib.geology_free_provinces
    )
    segment_records = _copy_records(
        segments_ptr[0], dtype=BOUNDARY_RECORD_DTYPE, free=_lib.geology_free_boundary_segments
    )
    stats = stats_ptr[0]
    metadata = {
        "province_count": int(stats.province_count),
        "boundary_segment_count": int(stats.boundary_segment_count),
        "active_boundary_segment_count": int(stats.active_boundary_segment_count),
        "mixed_plate_province_count": int(stats.mixed_plate_province_count),
        "continental_area_steradians": float(stats.continental_area),
        "oceanic_area_steradians": float(stats.oceanic_area),
        "mean_crust_age_ga": float(stats.mean_crust_age_ga),
        "mean_confidence": float(stats.mean_confidence),
        "province_model": "current_process_evidence_v1",
        "history_semantics": "initialization_not_simulated_deep_time",
        "boundary_length_semantics": "edge_center_spacing_from_area_approximation",
        "boundary_storage": "reciprocal_directed_d4_edge_slots",
    }
    return _province_table(province_records), _boundary_table(segment_records), metadata


__all__ = [
    "BOUNDARY_REGIMES",
    "CUBED_SPHERE_GEOLOGY_ABI_VERSION",
    "PROVINCE_CLASSES",
    "run_cubed_sphere_geology",
]
