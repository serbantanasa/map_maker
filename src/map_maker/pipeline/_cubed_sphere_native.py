"""CFFI binding for the Rust cubed-sphere geometry kernel."""

from __future__ import annotations

import numpy as np
from cffi import FFI

from .._native import native_library_info, native_library_path

FACE_COUNT = 6
D4_NEIGHBORS = 4

_CDEF = """
int32_t cubed_sphere_generate(
    int32_t face_resolution,
    float* xyz,
    double* longitude,
    double* latitude,
    double* area,
    int32_t* neighbors
);
"""

_ffi = FFI()
_ffi.cdef(_CDEF)
native_library_info("topology_native")
_lib = _ffi.dlopen(str(native_library_path("topology_native")))


def generate_cubed_sphere(
    face_resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if face_resolution <= 0:
        raise ValueError("face_resolution must be positive")
    cells = FACE_COUNT * face_resolution * face_resolution
    if cells > np.iinfo(np.int32).max:
        raise ValueError("face_resolution exceeds int32 global-index capacity")
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
    if status != 0:
        raise RuntimeError(f"cubed-sphere kernel failed with status {status}")
    return xyz, longitude, latitude, area, neighbors


__all__ = ["D4_NEIGHBORS", "FACE_COUNT", "generate_cubed_sphere"]
