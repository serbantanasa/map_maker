"""Rust-backed connected sea-level and surface-geography bindings."""

from __future__ import annotations

from typing import Any

import numpy as np
from cffi import FFI

from .._native import NativeLibraryAbiError, native_library_info, native_library_path

CUBED_SPHERE_SEA_LEVEL_ABI_VERSION = 2

_CDEF = """
typedef struct {
    float sea_level_m;
    double target_ocean_area_fraction;
    double ocean_mask_area_fraction;
    double ocean_fractional_area_fraction;
    double emerged_land_area_fraction;
    double coastal_cell_area_fraction;
    double continental_shelf_area_fraction;
    double inland_below_sea_level_area_fraction;
    double largest_inland_basin_area_fraction;
    double largest_land_component_area_fraction;
    double largest_land_component_share;
    double largest_land_component_coastline_complexity;
    float maximum_ocean_depth_m;
    float maximum_land_elevation_m;
    int32_t below_level_component_count;
    int32_t land_component_count;
    int32_t significant_land_component_count;
    uint64_t coastline_edge_count;
} SeaLevelStats;

uint32_t cubed_sphere_sea_level_abi_version(void);
int32_t sea_level_run_cubed_sphere(
    int32_t cell_count,
    double target_ocean_area_fraction,
    float shelf_depth_m,
    float minimum_coastal_relief_m,
    float coastal_relief_scale,
    const double* area,
    const int32_t* neighbors,
    const float* elevation,
    const float* relief,
    float* ocean_mask_out,
    float* ocean_fraction_out,
    float* surface_elevation_out,
    float* ocean_depth_out,
    float* shelf_fraction_out,
    float* coastal_mask_out,
    float* inland_below_sea_level_out,
    SeaLevelStats* stats_out
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
native_library_info("sea_level_native")
_lib = _ffi.dlopen(str(native_library_path("sea_level_native")))
try:
    _cubed_sphere_abi = int(_lib.cubed_sphere_sea_level_abi_version())
except AttributeError as exc:
    raise NativeLibraryAbiError(
        "sea_level_native lacks the cubed-sphere API; rebuild native libraries"
    ) from exc
if _cubed_sphere_abi != CUBED_SPHERE_SEA_LEVEL_ABI_VERSION:
    raise NativeLibraryAbiError(
        "sea_level_native cubed-sphere ABI "
        f"{_cubed_sphere_abi} does not match expected {CUBED_SPHERE_SEA_LEVEL_ABI_VERSION}"
    )


def run_cubed_sphere_sea_level(
    *,
    target_ocean_area_fraction: float,
    shelf_depth_m: float,
    minimum_coastal_relief_m: float,
    coastal_relief_scale: float,
    areas: np.ndarray,
    neighbors: np.ndarray,
    elevation: np.ndarray,
    relief: np.ndarray,
    ocean_mask_out: np.ndarray,
    ocean_fraction_out: np.ndarray,
    surface_elevation_out: np.ndarray,
    ocean_depth_out: np.ndarray,
    shelf_fraction_out: np.ndarray,
    coastal_mask_out: np.ndarray,
    inland_below_sea_level_out: np.ndarray,
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
    if neighbor_array.shape != (*shape, 4):
        raise ValueError(f"neighbors must have shape {(*shape, 4)}")

    inputs = {
        "areas": area_array,
        "neighbors": neighbor_array,
        "elevation": _read_array(elevation, name="elevation", dtype=np.dtype(np.float32)),
        "relief": _read_array(relief, name="relief", dtype=np.dtype(np.float32)),
    }
    for name in ("elevation", "relief"):
        if inputs[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    outputs = {
        name: _write_array(array, name=name)
        for name, array in {
            "ocean_mask_out": ocean_mask_out,
            "ocean_fraction_out": ocean_fraction_out,
            "surface_elevation_out": surface_elevation_out,
            "ocean_depth_out": ocean_depth_out,
            "shelf_fraction_out": shelf_fraction_out,
            "coastal_mask_out": coastal_mask_out,
            "inland_below_sea_level_out": inland_below_sea_level_out,
        }.items()
    }
    for name, array in outputs.items():
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    _require_disjoint(inputs, outputs)

    stats = _ffi.new("SeaLevelStats*")
    status = int(
        _lib.sea_level_run_cubed_sphere(
            int(np.prod(shape, dtype=np.int64)),
            float(target_ocean_area_fraction),
            float(shelf_depth_m),
            float(minimum_coastal_relief_m),
            float(coastal_relief_scale),
            _ffi.cast("double*", _ffi.from_buffer("double[]", area_array)),
            _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", neighbor_array)),
            _ffi.cast("float*", _ffi.from_buffer("float[]", inputs["elevation"])),
            _ffi.cast("float*", _ffi.from_buffer("float[]", inputs["relief"])),
            *(
                _ffi.cast("float*", _ffi.from_buffer("float[]", outputs[name]))
                for name in (
                    "ocean_mask_out",
                    "ocean_fraction_out",
                    "surface_elevation_out",
                    "ocean_depth_out",
                    "shelf_fraction_out",
                    "coastal_mask_out",
                    "inland_below_sea_level_out",
                )
            ),
            stats,
        )
    )
    if status != 0:
        messages = {
            1: "invalid dimensions or null buffers",
            2: "invalid sea-level parameters",
            3: "non-finite numeric inputs",
            4: "invalid cubed-sphere topology",
        }
        raise ValueError(
            f"cubed-sphere sea-level kernel failed: {messages.get(status, f'status {status}')}"
        )

    return {
        "sea_level_m": float(stats.sea_level_m),
        "target_ocean_area_fraction": float(stats.target_ocean_area_fraction),
        "ocean_mask_area_fraction": float(stats.ocean_mask_area_fraction),
        "ocean_fractional_area_fraction": float(stats.ocean_fractional_area_fraction),
        "emerged_land_area_fraction": float(stats.emerged_land_area_fraction),
        "coastal_cell_area_fraction": float(stats.coastal_cell_area_fraction),
        "continental_shelf_area_fraction": float(stats.continental_shelf_area_fraction),
        "inland_below_sea_level_area_fraction": float(stats.inland_below_sea_level_area_fraction),
        "largest_inland_basin_area_fraction": float(stats.largest_inland_basin_area_fraction),
        "largest_land_component_area_fraction": float(stats.largest_land_component_area_fraction),
        "largest_land_component_share": float(stats.largest_land_component_share),
        "largest_land_component_coastline_complexity": float(
            stats.largest_land_component_coastline_complexity
        ),
        "maximum_ocean_depth_m": float(stats.maximum_ocean_depth_m),
        "maximum_land_elevation_m": float(stats.maximum_land_elevation_m),
        "below_level_component_count": int(stats.below_level_component_count),
        "land_component_count": int(stats.land_component_count),
        "significant_land_component_count": int(stats.significant_land_component_count),
        "coastline_edge_count": int(stats.coastline_edge_count),
    }


__all__ = ["CUBED_SPHERE_SEA_LEVEL_ABI_VERSION", "run_cubed_sphere_sea_level"]
