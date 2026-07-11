"""Rust-backed bindings for canonical cubed-sphere elevation synthesis."""

from __future__ import annotations

from typing import Any

import numpy as np
from cffi import FFI

from .._native import NativeLibraryAbiError, native_library_info, native_library_path

CUBED_SPHERE_ELEVATION_ABI_VERSION = 1

_CDEF = """
typedef struct {
    float elevation_min_m;
    float elevation_mean_m;
    float elevation_max_m;
    float continental_mean_m;
    float oceanic_mean_m;
    float maximum_orogenic_m;
    float maximum_basin_m;
    double high_mountain_area_fraction;
    double deep_ocean_area_fraction;
    double active_relief_area_fraction;
} ElevationStats;

uint32_t cubed_sphere_elevation_abi_version(void);
int32_t elevation_run_cubed_sphere(
    int32_t cell_count,
    uint64_t seed,
    float collision_height_m,
    float arc_height_m,
    float ridge_height_m,
    float trench_depth_m,
    float rift_depth_m,
    int32_t plate_components,
    const double* area,
    const int32_t* neighbors,
    const float* plate_field,
    const float* crust_thickness,
    const float* isostasy,
    const float* uplift,
    const float* subsidence,
    const float* compression,
    const float* extension,
    const float* shear,
    const float* stiffness,
    const float* proto_ocean,
    const float* hotspot,
    const float* crust_age,
    const float* rock_strength,
    const float* accommodation,
    const float* province_confidence,
    const uint8_t* boundary_regime,
    const float* boundary_confidence,
    float* crustal_out,
    float* orogenic_out,
    float* basin_out,
    float* bedrock_out,
    float* relief_out,
    float* confidence_out,
    ElevationStats* stats_out
);
"""


def _read_array(array: np.ndarray, *, name: str, dtype: np.dtype) -> np.ndarray:
    value = np.array(array, copy=False)
    if value.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {value.dtype}")
    if not value.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be contiguous")
    if not value.flags["ALIGNED"]:
        raise ValueError(f"{name} must be aligned")
    return value


def _write_array(array: np.ndarray, *, name: str) -> np.ndarray:
    value = _read_array(array, name=name, dtype=np.dtype(np.float32))
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _require_disjoint(inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]) -> None:
    output_items = list(outputs.items())
    for index, (first_name, first) in enumerate(output_items):
        for second_name, second in output_items[index + 1 :]:
            if np.shares_memory(first, second):
                raise ValueError(f"{first_name} and {second_name} buffers must not overlap")
        for input_name, input_array in inputs.items():
            if np.shares_memory(first, input_array):
                raise ValueError(f"{first_name} and {input_name} buffers must not overlap")


_ffi = FFI()
_ffi.cdef(_CDEF)
native_library_info("elevation_native")
_lib = _ffi.dlopen(str(native_library_path("elevation_native")))
try:
    _cubed_sphere_abi = int(_lib.cubed_sphere_elevation_abi_version())
except AttributeError as exc:
    raise NativeLibraryAbiError(
        "elevation_native lacks the cubed-sphere API; rebuild native libraries"
    ) from exc
if _cubed_sphere_abi != CUBED_SPHERE_ELEVATION_ABI_VERSION:
    raise NativeLibraryAbiError(
        "elevation_native cubed-sphere ABI "
        f"{_cubed_sphere_abi} does not match expected {CUBED_SPHERE_ELEVATION_ABI_VERSION}"
    )


def run_cubed_sphere_elevation(
    *,
    seed: int,
    collision_height_m: float,
    arc_height_m: float,
    ridge_height_m: float,
    trench_depth_m: float,
    rift_depth_m: float,
    areas: np.ndarray,
    neighbors: np.ndarray,
    plate_field: np.ndarray,
    crust_thickness: np.ndarray,
    isostasy: np.ndarray,
    uplift: np.ndarray,
    subsidence: np.ndarray,
    compression: np.ndarray,
    extension: np.ndarray,
    shear: np.ndarray,
    stiffness: np.ndarray,
    proto_ocean: np.ndarray,
    hotspot: np.ndarray,
    crust_age: np.ndarray,
    rock_strength: np.ndarray,
    accommodation: np.ndarray,
    province_confidence: np.ndarray,
    boundary_regime: np.ndarray,
    boundary_confidence: np.ndarray,
    crustal_out: np.ndarray,
    orogenic_out: np.ndarray,
    basin_out: np.ndarray,
    bedrock_out: np.ndarray,
    relief_out: np.ndarray,
    confidence_out: np.ndarray,
) -> dict[str, Any]:
    area_array = np.require(areas, dtype=np.float64, requirements=["C", "A"])
    neighbor_array = np.require(neighbors, dtype=np.int32, requirements=["C", "A"])
    if (
        area_array.ndim != 3
        or area_array.shape[0] != 6
        or area_array.shape[1] != area_array.shape[2]
    ):
        raise ValueError("areas must have shape (6, n, n)")
    shape = area_array.shape
    edge_shape = (*shape, 4)
    if neighbor_array.shape != edge_shape:
        raise ValueError(f"neighbors must have shape {edge_shape}")

    plate_array = _read_array(plate_field, name="plate_field", dtype=np.dtype(np.float32))
    if plate_array.ndim != 4 or plate_array.shape[:3] != shape or plate_array.shape[3] < 4:
        raise ValueError(f"plate_field must have shape {(*shape, 4)} or more components")

    scalar_inputs = {
        name: _read_array(array, name=name, dtype=np.dtype(np.float32))
        for name, array in {
            "crust_thickness": crust_thickness,
            "isostasy": isostasy,
            "uplift": uplift,
            "subsidence": subsidence,
            "compression": compression,
            "extension": extension,
            "shear": shear,
            "stiffness": stiffness,
            "proto_ocean": proto_ocean,
            "hotspot": hotspot,
            "crust_age": crust_age,
            "rock_strength": rock_strength,
            "accommodation": accommodation,
            "province_confidence": province_confidence,
        }.items()
    }
    for name, array in scalar_inputs.items():
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    regime_array = _read_array(boundary_regime, name="boundary_regime", dtype=np.dtype(np.uint8))
    boundary_confidence_array = _read_array(
        boundary_confidence, name="boundary_confidence", dtype=np.dtype(np.float32)
    )
    if regime_array.shape != edge_shape:
        raise ValueError(f"boundary_regime must have shape {edge_shape}")
    if boundary_confidence_array.shape != edge_shape:
        raise ValueError(f"boundary_confidence must have shape {edge_shape}")

    output_arrays = {
        name: _write_array(array, name=name)
        for name, array in {
            "crustal_out": crustal_out,
            "orogenic_out": orogenic_out,
            "basin_out": basin_out,
            "bedrock_out": bedrock_out,
            "relief_out": relief_out,
            "confidence_out": confidence_out,
        }.items()
    }
    for name, array in output_arrays.items():
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

    all_inputs = {
        "areas": area_array,
        "neighbors": neighbor_array,
        "plate_field": plate_array,
        "boundary_regime": regime_array,
        "boundary_confidence": boundary_confidence_array,
        **scalar_inputs,
    }
    _require_disjoint(all_inputs, output_arrays)

    stats = _ffi.new("ElevationStats*")
    status = int(
        _lib.elevation_run_cubed_sphere(
            int(np.prod(shape, dtype=np.int64)),
            int(seed) & ((1 << 64) - 1),
            float(collision_height_m),
            float(arc_height_m),
            float(ridge_height_m),
            float(trench_depth_m),
            float(rift_depth_m),
            int(plate_array.shape[-1]),
            _ffi.cast("double*", _ffi.from_buffer("double[]", area_array)),
            _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", neighbor_array)),
            _ffi.cast("float*", _ffi.from_buffer("float[]", plate_array)),
            *(
                _ffi.cast("float*", _ffi.from_buffer("float[]", scalar_inputs[name]))
                for name in (
                    "crust_thickness",
                    "isostasy",
                    "uplift",
                    "subsidence",
                    "compression",
                    "extension",
                    "shear",
                    "stiffness",
                    "proto_ocean",
                    "hotspot",
                    "crust_age",
                    "rock_strength",
                    "accommodation",
                    "province_confidence",
                )
            ),
            _ffi.cast("uint8_t*", _ffi.from_buffer("uint8_t[]", regime_array)),
            _ffi.cast("float*", _ffi.from_buffer("float[]", boundary_confidence_array)),
            *(
                _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays[name]))
                for name in (
                    "crustal_out",
                    "orogenic_out",
                    "basin_out",
                    "bedrock_out",
                    "relief_out",
                    "confidence_out",
                )
            ),
            stats,
        )
    )
    if status != 0:
        messages = {
            1: "invalid dimensions or null buffers",
            2: "invalid morphology parameters",
            3: "non-finite numeric inputs",
            4: "invalid cubed-sphere topology",
        }
        raise ValueError(
            f"cubed-sphere elevation kernel failed: {messages.get(status, f'status {status}')}"
        )

    return {
        "elevation_min_m": float(stats.elevation_min_m),
        "elevation_mean_m": float(stats.elevation_mean_m),
        "elevation_max_m": float(stats.elevation_max_m),
        "continental_mean_m": float(stats.continental_mean_m),
        "oceanic_mean_m": float(stats.oceanic_mean_m),
        "maximum_orogenic_m": float(stats.maximum_orogenic_m),
        "maximum_basin_m": float(stats.maximum_basin_m),
        "high_mountain_area_fraction": float(stats.high_mountain_area_fraction),
        "deep_ocean_area_fraction": float(stats.deep_ocean_area_fraction),
        "active_relief_area_fraction": float(stats.active_relief_area_fraction),
    }


__all__ = ["CUBED_SPHERE_ELEVATION_ABI_VERSION", "run_cubed_sphere_elevation"]
