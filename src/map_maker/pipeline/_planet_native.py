"""Rust-backed bindings for canonical cubed-sphere planetary forcing."""

from __future__ import annotations

from typing import Any

import numpy as np
from cffi import FFI

from .._native import NativeLibraryAbiError, native_library_info, native_library_path

CUBED_SPHERE_PLANET_ABI_VERSION = 1

_CDEF = """
typedef struct {
    float global_mean_insolation_w_m2;
    float equatorial_mean_insolation_w_m2;
    float polar_mean_insolation_w_m2;
    float mean_seasonality_w_m2;
    float maximum_monthly_insolation_w_m2;
    float minimum_orbital_distance_au;
    float maximum_orbital_distance_au;
    float tide_strength_index;
    float obliquity_stability_index;
} PlanetStats;

uint32_t cubed_sphere_planet_abi_version(void);
int32_t planet_run_cubed_sphere(
    int32_t cell_count,
    double star_luminosity_solar,
    double semi_major_axis_au,
    double eccentricity,
    double obliquity_radians,
    double rotation_period_hours,
    double orbital_period_days,
    double perihelion_day,
    double northern_vernal_equinox_day,
    double moon_mass_lunar,
    double moon_distance_km,
    const double* area,
    const double* latitude,
    float* monthly_insolation,
    float* monthly_daylight,
    float* annual_mean,
    float* seasonality,
    float* polar_extreme_fraction,
    float* orbital_distance,
    float* solar_declination,
    PlanetStats* stats_out
);
"""


def _read_float64(array: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype != np.float64:
        raise ValueError(f"{name} must be float64, got {value.dtype}")
    if not value.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be contiguous")
    if not value.flags["ALIGNED"]:
        raise ValueError(f"{name} must be aligned")
    return value


def _write_float32(array: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype != np.float32:
        raise ValueError(f"{name} must be float32, got {value.dtype}")
    if not value.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be contiguous")
    if not value.flags["ALIGNED"]:
        raise ValueError(f"{name} must be aligned")
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
native_library_info("planet_native")
_lib = _ffi.dlopen(str(native_library_path("planet_native")))
try:
    _cubed_sphere_abi = int(_lib.cubed_sphere_planet_abi_version())
except AttributeError as exc:
    raise NativeLibraryAbiError(
        "planet_native lacks the cubed-sphere API; rebuild native libraries"
    ) from exc
if _cubed_sphere_abi != CUBED_SPHERE_PLANET_ABI_VERSION:
    raise NativeLibraryAbiError(
        "planet_native cubed-sphere ABI "
        f"{_cubed_sphere_abi} does not match expected {CUBED_SPHERE_PLANET_ABI_VERSION}"
    )


def run_cubed_sphere_planet(
    *,
    star_luminosity_solar: float,
    semi_major_axis_au: float,
    eccentricity: float,
    obliquity_radians: float,
    rotation_period_hours: float,
    orbital_period_days: float,
    perihelion_day: float,
    northern_vernal_equinox_day: float,
    moon_mass_lunar: float,
    moon_distance_km: float,
    areas: np.ndarray,
    latitudes: np.ndarray,
    monthly_insolation_out: np.ndarray,
    monthly_daylight_out: np.ndarray,
    annual_mean_out: np.ndarray,
    seasonality_out: np.ndarray,
    polar_extreme_fraction_out: np.ndarray,
    orbital_distance_out: np.ndarray,
    solar_declination_out: np.ndarray,
) -> dict[str, Any]:
    area_array = _read_float64(areas, name="areas")
    latitude_array = _read_float64(latitudes, name="latitudes")
    if (
        area_array.ndim != 3
        or area_array.shape[0] != 6
        or area_array.shape[1] != area_array.shape[2]
    ):
        raise ValueError("areas must have shape (6, n, n)")
    shape = area_array.shape
    if latitude_array.shape != shape:
        raise ValueError(f"latitudes must have shape {shape}")

    output_arrays = {
        name: _write_float32(array, name=name)
        for name, array in {
            "monthly_insolation_out": monthly_insolation_out,
            "monthly_daylight_out": monthly_daylight_out,
            "annual_mean_out": annual_mean_out,
            "seasonality_out": seasonality_out,
            "polar_extreme_fraction_out": polar_extreme_fraction_out,
            "orbital_distance_out": orbital_distance_out,
            "solar_declination_out": solar_declination_out,
        }.items()
    }
    expected_shapes = {
        "monthly_insolation_out": (12, *shape),
        "monthly_daylight_out": (12, *shape),
        "annual_mean_out": shape,
        "seasonality_out": shape,
        "polar_extreme_fraction_out": shape,
        "orbital_distance_out": (12,),
        "solar_declination_out": (12,),
    }
    for name, expected_shape in expected_shapes.items():
        if output_arrays[name].shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    _require_disjoint({"areas": area_array, "latitudes": latitude_array}, output_arrays)

    stats = _ffi.new("PlanetStats*")
    status = int(
        _lib.planet_run_cubed_sphere(
            int(np.prod(shape, dtype=np.int64)),
            float(star_luminosity_solar),
            float(semi_major_axis_au),
            float(eccentricity),
            float(obliquity_radians),
            float(rotation_period_hours),
            float(orbital_period_days),
            float(perihelion_day),
            float(northern_vernal_equinox_day),
            float(moon_mass_lunar),
            float(moon_distance_km),
            _ffi.cast("double*", _ffi.from_buffer("double[]", area_array)),
            _ffi.cast("double*", _ffi.from_buffer("double[]", latitude_array)),
            *(
                _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays[name]))
                for name in expected_shapes
            ),
            stats,
        )
    )
    if status != 0:
        messages = {
            1: "invalid dimensions or null buffers",
            2: "invalid orbital parameters",
            3: "non-finite geometry inputs",
        }
        raise ValueError(
            f"cubed-sphere planet kernel failed: {messages.get(status, f'status {status}')}"
        )

    return {
        "global_mean_insolation_w_m2": float(stats.global_mean_insolation_w_m2),
        "equatorial_mean_insolation_w_m2": float(stats.equatorial_mean_insolation_w_m2),
        "polar_mean_insolation_w_m2": float(stats.polar_mean_insolation_w_m2),
        "mean_seasonality_w_m2": float(stats.mean_seasonality_w_m2),
        "maximum_monthly_insolation_w_m2": float(stats.maximum_monthly_insolation_w_m2),
        "minimum_orbital_distance_au": float(stats.minimum_orbital_distance_au),
        "maximum_orbital_distance_au": float(stats.maximum_orbital_distance_au),
        "tide_strength_index": float(stats.tide_strength_index),
        "obliquity_stability_index": float(stats.obliquity_stability_index),
    }


__all__ = ["CUBED_SPHERE_PLANET_ABI_VERSION", "run_cubed_sphere_planet"]
