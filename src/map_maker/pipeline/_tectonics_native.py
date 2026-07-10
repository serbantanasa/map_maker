"""Rust-backed native bindings for the tectonics stage."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from cffi import FFI

from .._native import NativeLibraryAbiError, native_library_info, native_library_path

PLATE_FIELD_COMPONENTS = 6
SPHERICAL_PLATE_FIELD_COMPONENTS = 7
CUBED_SPHERE_TECTONICS_ABI_VERSION = 1

_CDEF = """
typedef struct {
    int32_t plate_count;
    double continental_fraction;
    double velocity_mean;
    double velocity_std;
    double hotspot_mean;
    double boundary_metric_mean;
    double convergence_sum;
    double divergence_sum;
    double shear_sum;
    double subduction_mean;
    int32_t hotspot_count;
} TectonicsStats;

void tectonics_run(
    int32_t height,
    int32_t width,
    uint64_t seed,
    int32_t num_plates,
    float continental_fraction_target,
    float velocity_scale,
    float drift_bias,
    float hotspot_density,
    float subduction_bias,
    int32_t lloyd_iterations,
    int32_t time_steps,
    float time_step,
    int32_t wrap_x,
    int32_t wrap_y,
    float* plate_field,
    float* convergence_field,
    float* divergence_field,
    float* shear_field,
    float* subduction_field,
    float* hotspot_field,
    TectonicsStats* out_stats
);
uint32_t cubed_sphere_tectonics_abi_version(void);
int32_t tectonics_run_cubed_sphere(
    int32_t cell_count,
    uint64_t seed,
    int32_t num_plates,
    float continental_fraction_target,
    float velocity_scale,
    float drift_bias,
    float hotspot_density,
    float subduction_bias,
    int32_t lloyd_iterations,
    const float* xyz,
    const double* area,
    const int32_t* neighbors,
    float* plate_field,
    float* convergence_field,
    float* divergence_field,
    float* shear_field,
    float* subduction_field,
    float* hotspot_field,
    TectonicsStats* out_stats
);
"""


_ffi = FFI()
_ffi.cdef(_CDEF)
native_library_info("tectonics_native")
_lib = _ffi.dlopen(str(native_library_path("tectonics_native")))
try:
    _cubed_sphere_abi = int(_lib.cubed_sphere_tectonics_abi_version())
except AttributeError as exc:
    raise NativeLibraryAbiError(
        "tectonics_native lacks the cubed-sphere tectonics API; rebuild native libraries"
    ) from exc
if _cubed_sphere_abi != CUBED_SPHERE_TECTONICS_ABI_VERSION:
    raise NativeLibraryAbiError(
        "tectonics_native cubed-sphere ABI "
        f"{_cubed_sphere_abi} does not match expected {CUBED_SPHERE_TECTONICS_ABI_VERSION}"
    )


def _as_write_array(array: np.ndarray, *, name: str) -> np.ndarray:
    if array.dtype != np.float32:
        raise ValueError(f"{name} must be float32, got {array.dtype}")
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be contiguous")
    if not array.flags["ALIGNED"]:
        raise ValueError(f"{name} must be aligned")
    if not array.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return array


def _require_disjoint(buffers: dict[str, np.ndarray]) -> None:
    items = list(buffers.items())
    for index, (first_name, first) in enumerate(items):
        for second_name, second in items[index + 1 :]:
            if np.shares_memory(first, second):
                raise ValueError(f"{first_name} and {second_name} buffers must not overlap")


def run_tectonics_kernels(
    height: int,
    width: int,
    seed: int,
    num_plates: int,
    continental_fraction: float,
    velocity_scale: float,
    drift_bias: float,
    hotspot_density: float,
    subduction_bias: float,
    lloyd_iterations: int,
    time_steps: int,
    time_step: float,
    wrap_x: bool,
    wrap_y: bool,
    plate_field: np.ndarray,
    convergence_field: np.ndarray,
    divergence_field: np.ndarray,
    shear_field: np.ndarray,
    subduction_field: np.ndarray,
    hotspot_field: np.ndarray,
) -> Dict[str, Any]:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    plate_arr = _as_write_array(plate_field, name="plate_field")
    conv_arr = _as_write_array(convergence_field, name="convergence_field")
    div_arr = _as_write_array(divergence_field, name="divergence_field")
    shear_arr = _as_write_array(shear_field, name="shear_field")
    subduction_arr = _as_write_array(subduction_field, name="subduction_field")
    hotspot_arr = _as_write_array(hotspot_field, name="hotspot_field")
    expected_grid_shape = (height, width)
    if plate_arr.shape != (*expected_grid_shape, PLATE_FIELD_COMPONENTS):
        raise ValueError(
            f"plate_field must have shape {(*expected_grid_shape, PLATE_FIELD_COMPONENTS)}"
        )
    for name, array in {
        "convergence_field": conv_arr,
        "divergence_field": div_arr,
        "shear_field": shear_arr,
        "subduction_field": subduction_arr,
        "hotspot_field": hotspot_arr,
    }.items():
        if array.shape != expected_grid_shape:
            raise ValueError(f"{name} must have shape {expected_grid_shape}")
    _require_disjoint(
        {
            "plate_field": plate_arr,
            "convergence_field": conv_arr,
            "divergence_field": div_arr,
            "shear_field": shear_arr,
            "subduction_field": subduction_arr,
            "hotspot_field": hotspot_arr,
        }
    )

    stats_ptr = _ffi.new("TectonicsStats*")

    _lib.tectonics_run(
        int(height),
        int(width),
        int(seed),
        int(num_plates),
        float(continental_fraction),
        float(velocity_scale),
        float(drift_bias),
        float(hotspot_density),
        float(subduction_bias),
        int(lloyd_iterations),
        int(time_steps),
        float(time_step),
        1 if wrap_x else 0,
        1 if wrap_y else 0,
        _ffi.cast("float*", _ffi.from_buffer("float[]", plate_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", conv_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", div_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", shear_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", subduction_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", hotspot_arr)),
        stats_ptr,
    )

    stats = stats_ptr[0]
    return {
        "plate_count": int(stats.plate_count),
        "continental_fraction": float(stats.continental_fraction),
        "velocity_mean": float(stats.velocity_mean),
        "velocity_std": float(stats.velocity_std),
        "hotspot_mean": float(stats.hotspot_mean),
        "boundary_metric_mean": float(stats.boundary_metric_mean),
        "convergence_sum": float(stats.convergence_sum),
        "divergence_sum": float(stats.divergence_sum),
        "shear_sum": float(stats.shear_sum),
        "subduction_mean": float(stats.subduction_mean),
        "hotspot_count": int(stats.hotspot_count),
    }


def run_cubed_sphere_tectonics(
    *,
    xyz: np.ndarray,
    areas: np.ndarray,
    neighbors: np.ndarray,
    seed: int,
    num_plates: int,
    continental_fraction: float,
    velocity_scale: float,
    drift_bias: float,
    hotspot_density: float,
    subduction_bias: float,
    lloyd_iterations: int,
    plate_field: np.ndarray,
    convergence_field: np.ndarray,
    divergence_field: np.ndarray,
    shear_field: np.ndarray,
    subduction_field: np.ndarray,
    hotspot_field: np.ndarray,
) -> Dict[str, Any]:
    xyz_array = np.require(xyz, dtype=np.float32, requirements=["C", "A"])
    area_array = np.require(areas, dtype=np.float64, requirements=["C", "A"])
    neighbor_array = np.require(neighbors, dtype=np.int32, requirements=["C", "A"])
    if xyz_array.ndim != 4 or xyz_array.shape[0] != 6 or xyz_array.shape[-1] != 3:
        raise ValueError("xyz must have shape (6, n, n, 3)")
    face_shape = xyz_array.shape[:-1]
    if area_array.shape != face_shape:
        raise ValueError(f"areas must have shape {face_shape}")
    if neighbor_array.shape != (*face_shape, 4):
        raise ValueError(f"neighbors must have shape {(*face_shape, 4)}")
    cell_count = int(np.prod(face_shape, dtype=np.int64))
    if not 2 <= num_plates <= min(cell_count // 4, 4096):
        raise ValueError("num_plates must be between 2 and min(cell_count // 4, 4096)")
    if lloyd_iterations < 0:
        raise ValueError("lloyd_iterations must be non-negative")
    if not 0.0 <= continental_fraction <= 1.0:
        raise ValueError("continental_fraction must be in [0, 1]")
    if plate_field.shape != (*face_shape, SPHERICAL_PLATE_FIELD_COMPONENTS):
        raise ValueError(
            "plate_field must have shape " f"{(*face_shape, SPHERICAL_PLATE_FIELD_COMPONENTS)}"
        )
    scalar_outputs = {
        "convergence_field": convergence_field,
        "divergence_field": divergence_field,
        "shear_field": shear_field,
        "subduction_field": subduction_field,
        "hotspot_field": hotspot_field,
    }
    plate_array = _as_write_array(plate_field, name="plate_field")
    output_arrays = {
        name: _as_write_array(array, name=name) for name, array in scalar_outputs.items()
    }
    for name, array in output_arrays.items():
        if array.shape != face_shape:
            raise ValueError(f"{name} must have shape {face_shape}")
    buffers = {
        "xyz": xyz_array,
        "areas": area_array,
        "neighbors": neighbor_array,
        "plate_field": plate_array,
        **output_arrays,
    }
    _require_disjoint(buffers)

    stats_ptr = _ffi.new("TectonicsStats*")
    status = _lib.tectonics_run_cubed_sphere(
        cell_count,
        int(seed),
        int(num_plates),
        float(continental_fraction),
        float(velocity_scale),
        float(drift_bias),
        float(hotspot_density),
        float(subduction_bias),
        int(lloyd_iterations),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", xyz_array)),
        _ffi.cast("const double*", _ffi.from_buffer("double[]", area_array)),
        _ffi.cast("const int32_t*", _ffi.from_buffer("int32_t[]", neighbor_array)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", plate_array)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays["convergence_field"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays["divergence_field"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays["shear_field"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays["subduction_field"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays["hotspot_field"])),
        stats_ptr,
    )
    if status != 0:
        raise RuntimeError(f"cubed-sphere tectonics kernel failed with status {status}")
    stats = stats_ptr[0]
    return {
        "plate_count": int(stats.plate_count),
        "continental_fraction": float(stats.continental_fraction),
        "velocity_mean": float(stats.velocity_mean),
        "velocity_std": float(stats.velocity_std),
        "hotspot_mean": float(stats.hotspot_mean),
        "boundary_metric_mean": float(stats.boundary_metric_mean),
        "convergence_sum": float(stats.convergence_sum),
        "divergence_sum": float(stats.divergence_sum),
        "shear_sum": float(stats.shear_sum),
        "subduction_mean": float(stats.subduction_mean),
        "hotspot_count": int(stats.hotspot_count),
    }


__all__ = [
    "CUBED_SPHERE_TECTONICS_ABI_VERSION",
    "PLATE_FIELD_COMPONENTS",
    "SPHERICAL_PLATE_FIELD_COMPONENTS",
    "run_cubed_sphere_tectonics",
    "run_tectonics_kernels",
]
