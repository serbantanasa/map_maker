"""CFFI binding for the Rust cubed-sphere geometry kernel."""

from __future__ import annotations

import numpy as np
from cffi import FFI

from .._native import NativeLibraryAbiError, native_library_info, native_library_path

FACE_COUNT = 6
D4_NEIGHBORS = 4
HIERARCHY_ABI_VERSION = 1

_CDEF = """
int32_t cubed_sphere_generate(
    int32_t face_resolution,
    float* xyz,
    double* longitude,
    double* latitude,
    double* area,
    int32_t* neighbors
);
int32_t cubed_sphere_parent_map(
    int32_t fine_resolution,
    int32_t factor,
    int32_t* parent
);
int32_t cubed_sphere_children_map(
    int32_t coarse_resolution,
    int32_t factor,
    int32_t* children
);
int32_t cubed_sphere_restrict_extensive_f64(
    int32_t fine_resolution,
    int32_t factor,
    const double* fine_values,
    double* coarse_values
);
int32_t cubed_sphere_restrict_intensive_f64(
    int32_t fine_resolution,
    int32_t factor,
    const double* fine_values,
    const double* fine_areas,
    double* coarse_values
);
int32_t cubed_sphere_prolongate_constant_f64(
    int32_t coarse_resolution,
    int32_t factor,
    const double* coarse_values,
    double* fine_values
);
int32_t cubed_sphere_fill_d4_halo_f32(
    int32_t face_resolution,
    const float* values,
    float* halo
);
int32_t cubed_sphere_fill_d4_halo_f64(
    int32_t face_resolution,
    const double* values,
    double* halo
);
uint32_t cubed_sphere_hierarchy_abi_version(void);
"""

_ffi = FFI()
_ffi.cdef(_CDEF)
native_library_info("topology_native")
_lib = _ffi.dlopen(str(native_library_path("topology_native")))
try:
    _hierarchy_abi_version = int(_lib.cubed_sphere_hierarchy_abi_version())
except AttributeError as exc:
    raise NativeLibraryAbiError(
        "topology_native lacks the cubed-sphere hierarchy API; rebuild native libraries"
    ) from exc
if _hierarchy_abi_version != HIERARCHY_ABI_VERSION:
    raise NativeLibraryAbiError(
        "topology_native cubed-sphere hierarchy ABI "
        f"{_hierarchy_abi_version} does not match expected {HIERARCHY_ABI_VERSION}"
    )


def _cell_count(face_resolution: int) -> int:
    if face_resolution <= 0:
        raise ValueError("face_resolution must be positive")
    cells = FACE_COUNT * face_resolution * face_resolution
    if cells > np.iinfo(np.int32).max:
        raise ValueError("face_resolution exceeds int32 global-index capacity")
    return cells


def _validate_factor(face_resolution: int, factor: int, *, require_divisible: bool) -> None:
    if factor <= 1:
        raise ValueError("factor must be greater than one")
    if require_divisible and face_resolution % factor != 0:
        raise ValueError("factor must divide face_resolution exactly")


def _face_field(values: np.ndarray, dtype: np.dtype, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[0] != FACE_COUNT or array.shape[1] != array.shape[2]:
        raise ValueError(f"{name} must have shape (6, n, n)")
    _cell_count(array.shape[1])
    return np.ascontiguousarray(array, dtype=dtype)


def _raise_for_status(operation: str, status: int) -> None:
    if status == 0:
        return
    if status == 3:
        raise ValueError(f"{operation} requires finite values and positive finite cell areas")
    raise RuntimeError(f"{operation} kernel failed with status {status}")


def generate_cubed_sphere(
    face_resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _cell_count(face_resolution)
    xyz = np.empty((FACE_COUNT, face_resolution, face_resolution, 3), dtype=np.float32)
    longitude = np.empty((FACE_COUNT, face_resolution, face_resolution), dtype=np.float64)
    latitude = np.empty_like(longitude)
    area = np.empty_like(longitude)
    neighbors = np.empty(
        (FACE_COUNT, face_resolution, face_resolution, D4_NEIGHBORS), dtype=np.int32
    )
    status = _lib.cubed_sphere_generate(
        int(face_resolution),
        _ffi.cast("float*", _ffi.from_buffer("float[]", xyz)),
        _ffi.cast("double*", _ffi.from_buffer("double[]", longitude)),
        _ffi.cast("double*", _ffi.from_buffer("double[]", latitude)),
        _ffi.cast("double*", _ffi.from_buffer("double[]", area)),
        _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", neighbors)),
    )
    _raise_for_status("cubed-sphere generation", status)
    return xyz, longitude, latitude, area, neighbors


def parent_map(fine_resolution: int, factor: int) -> np.ndarray:
    _cell_count(fine_resolution)
    _validate_factor(fine_resolution, factor, require_divisible=True)
    parents = np.empty((FACE_COUNT, fine_resolution, fine_resolution), dtype=np.int32)
    status = _lib.cubed_sphere_parent_map(
        fine_resolution,
        factor,
        _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", parents)),
    )
    _raise_for_status("cubed-sphere parent mapping", status)
    return parents


def children_map(coarse_resolution: int, factor: int) -> np.ndarray:
    _cell_count(coarse_resolution)
    _validate_factor(coarse_resolution, factor, require_divisible=False)
    fine_resolution = coarse_resolution * factor
    _cell_count(fine_resolution)
    children = np.empty(
        (FACE_COUNT, coarse_resolution, coarse_resolution, factor * factor), dtype=np.int32
    )
    status = _lib.cubed_sphere_children_map(
        coarse_resolution,
        factor,
        _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", children)),
    )
    _raise_for_status("cubed-sphere children mapping", status)
    return children


def restrict_extensive(values: np.ndarray, factor: int) -> np.ndarray:
    fine = _face_field(values, np.dtype(np.float64), name="values")
    fine_resolution = fine.shape[1]
    _validate_factor(fine_resolution, factor, require_divisible=True)
    coarse_resolution = fine_resolution // factor
    coarse = np.empty((FACE_COUNT, coarse_resolution, coarse_resolution), dtype=np.float64)
    status = _lib.cubed_sphere_restrict_extensive_f64(
        fine_resolution,
        factor,
        _ffi.cast("const double*", _ffi.from_buffer("double[]", fine)),
        _ffi.cast("double*", _ffi.from_buffer("double[]", coarse)),
    )
    _raise_for_status("cubed-sphere extensive restriction", status)
    return coarse


def restrict_intensive(values: np.ndarray, areas: np.ndarray, factor: int) -> np.ndarray:
    fine = _face_field(values, np.dtype(np.float64), name="values")
    fine_areas = _face_field(areas, np.dtype(np.float64), name="areas")
    if fine.shape != fine_areas.shape:
        raise ValueError("values and areas must have the same shape")
    fine_resolution = fine.shape[1]
    _validate_factor(fine_resolution, factor, require_divisible=True)
    coarse_resolution = fine_resolution // factor
    coarse = np.empty((FACE_COUNT, coarse_resolution, coarse_resolution), dtype=np.float64)
    status = _lib.cubed_sphere_restrict_intensive_f64(
        fine_resolution,
        factor,
        _ffi.cast("const double*", _ffi.from_buffer("double[]", fine)),
        _ffi.cast("const double*", _ffi.from_buffer("double[]", fine_areas)),
        _ffi.cast("double*", _ffi.from_buffer("double[]", coarse)),
    )
    _raise_for_status("cubed-sphere intensive restriction", status)
    return coarse


def prolongate_constant(values: np.ndarray, factor: int) -> np.ndarray:
    coarse = _face_field(values, np.dtype(np.float64), name="values")
    coarse_resolution = coarse.shape[1]
    _validate_factor(coarse_resolution, factor, require_divisible=False)
    fine_resolution = coarse_resolution * factor
    _cell_count(fine_resolution)
    fine = np.empty((FACE_COUNT, fine_resolution, fine_resolution), dtype=np.float64)
    status = _lib.cubed_sphere_prolongate_constant_f64(
        coarse_resolution,
        factor,
        _ffi.cast("const double*", _ffi.from_buffer("double[]", coarse)),
        _ffi.cast("double*", _ffi.from_buffer("double[]", fine)),
    )
    _raise_for_status("cubed-sphere constant prolongation", status)
    return fine


def fill_d4_halo(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("values must use float32 or float64 for D4 halo exchange")
    field = _face_field(array, array.dtype, name="values")
    resolution = field.shape[1]
    halo = np.empty((FACE_COUNT, resolution + 2, resolution + 2), dtype=field.dtype)
    if field.dtype == np.dtype(np.float32):
        status = _lib.cubed_sphere_fill_d4_halo_f32(
            resolution,
            _ffi.cast("const float*", _ffi.from_buffer("float[]", field)),
            _ffi.cast("float*", _ffi.from_buffer("float[]", halo)),
        )
    else:
        status = _lib.cubed_sphere_fill_d4_halo_f64(
            resolution,
            _ffi.cast("const double*", _ffi.from_buffer("double[]", field)),
            _ffi.cast("double*", _ffi.from_buffer("double[]", halo)),
        )
    _raise_for_status("cubed-sphere D4 halo", status)
    return halo


__all__ = [
    "D4_NEIGHBORS",
    "FACE_COUNT",
    "HIERARCHY_ABI_VERSION",
    "children_map",
    "fill_d4_halo",
    "generate_cubed_sphere",
    "parent_map",
    "prolongate_constant",
    "restrict_extensive",
    "restrict_intensive",
]
