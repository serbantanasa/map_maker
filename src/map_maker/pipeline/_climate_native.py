"""Rust-backed bindings for canonical cubed-sphere seasonal climate."""

from __future__ import annotations

from typing import Any

import numpy as np
from cffi import FFI

from .._native import NativeLibraryAbiError, native_library_info, native_library_path

CUBED_SPHERE_CLIMATE_ABI_VERSION = 2

_CDEF = """
typedef struct {
    float global_mean_temperature_c;
    float land_mean_temperature_c;
    float ocean_mean_temperature_c;
    float minimum_monthly_temperature_c;
    float maximum_monthly_temperature_c;
    float global_mean_annual_precipitation_mm;
    float land_mean_annual_precipitation_mm;
    float dry_land_area_fraction;
    float wet_land_area_fraction;
    float persistent_snow_land_area_fraction;
    float maximum_wind_speed_m_s;
} ClimateStats;

uint32_t cubed_sphere_climate_abi_version(void);
int32_t climate_run_cubed_sphere(
    int32_t cell_count,
    int32_t spinup_years,
    int32_t moisture_spinup_years,
    int32_t moisture_steps_per_month,
    int32_t synoptic_mixing_passes,
    double greenhouse_offset_c,
    double land_albedo,
    double ocean_albedo,
    double olr_intercept_w_m2,
    double olr_slope_w_m2_c,
    double heat_transport_w_m2,
    double land_thermal_response,
    double ocean_thermal_response,
    double atmospheric_exchange,
    double lapse_rate_c_per_km,
    double wind_scale,
    double moisture_advection_fraction,
    double moisture_diffusion_fraction,
    double orographic_factor,
    double rain_shadow_factor,
    double runoff_base_fraction,
    const double* area,
    const int32_t* neighbors,
    const float* xyz,
    const double* latitude,
    const float* elevation,
    const float* relief,
    const float* ocean,
    const float* insolation,
    const float* declination,
    float* climate_orography,
    float* temperature,
    float* wind_xyz,
    float* wind_speed,
    float* precipitation,
    float* humidity,
    float* snowfall,
    float* snowmelt,
    float* snowpack,
    float* evaporation,
    float* runoff,
    float* annual_temperature,
    float* annual_precipitation,
    float* aridity,
    ClimateStats* stats_out
);
"""


def _read_array(array: np.ndarray, *, name: str, dtype: np.dtype) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {value.dtype}")
    if not value.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be contiguous")
    if not value.flags["ALIGNED"]:
        raise ValueError(f"{name} must be aligned")
    return value


def _write_float32(array: np.ndarray, *, name: str) -> np.ndarray:
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
native_library_info("climate_native")
_lib = _ffi.dlopen(str(native_library_path("climate_native")))
try:
    _cubed_sphere_abi = int(_lib.cubed_sphere_climate_abi_version())
except AttributeError as exc:
    raise NativeLibraryAbiError(
        "climate_native lacks the cubed-sphere API; rebuild native libraries"
    ) from exc
if _cubed_sphere_abi != CUBED_SPHERE_CLIMATE_ABI_VERSION:
    raise NativeLibraryAbiError(
        "climate_native cubed-sphere ABI "
        f"{_cubed_sphere_abi} does not match expected {CUBED_SPHERE_CLIMATE_ABI_VERSION}"
    )


def run_cubed_sphere_climate(
    *,
    spinup_years: int,
    moisture_spinup_years: int,
    moisture_steps_per_month: int,
    synoptic_mixing_passes: int,
    greenhouse_offset_c: float,
    land_albedo: float,
    ocean_albedo: float,
    olr_intercept_w_m2: float,
    olr_slope_w_m2_c: float,
    heat_transport_w_m2: float,
    land_thermal_response: float,
    ocean_thermal_response: float,
    atmospheric_exchange: float,
    lapse_rate_c_per_km: float,
    wind_scale: float,
    moisture_advection_fraction: float,
    moisture_diffusion_fraction: float,
    orographic_factor: float,
    rain_shadow_factor: float,
    runoff_base_fraction: float,
    areas: np.ndarray,
    neighbors: np.ndarray,
    xyz: np.ndarray,
    latitudes: np.ndarray,
    elevation: np.ndarray,
    relief: np.ndarray,
    ocean: np.ndarray,
    insolation: np.ndarray,
    declination: np.ndarray,
    climate_orography_out: np.ndarray,
    temperature_out: np.ndarray,
    wind_xyz_out: np.ndarray,
    wind_speed_out: np.ndarray,
    precipitation_out: np.ndarray,
    humidity_out: np.ndarray,
    snowfall_out: np.ndarray,
    snowmelt_out: np.ndarray,
    snowpack_out: np.ndarray,
    evaporation_out: np.ndarray,
    runoff_out: np.ndarray,
    annual_temperature_out: np.ndarray,
    annual_precipitation_out: np.ndarray,
    aridity_out: np.ndarray,
) -> dict[str, Any]:
    area_array = _read_array(areas, name="areas", dtype=np.dtype(np.float64))
    if (
        area_array.ndim != 3
        or area_array.shape[0] != 6
        or area_array.shape[1] != area_array.shape[2]
    ):
        raise ValueError("areas must have shape (6, n, n)")
    shape = area_array.shape
    monthly_shape = (12, *shape)
    input_arrays = {
        "areas": area_array,
        "neighbors": _read_array(neighbors, name="neighbors", dtype=np.dtype(np.int32)),
        "xyz": _read_array(xyz, name="xyz", dtype=np.dtype(np.float32)),
        "latitudes": _read_array(latitudes, name="latitudes", dtype=np.dtype(np.float64)),
        "elevation": _read_array(elevation, name="elevation", dtype=np.dtype(np.float32)),
        "relief": _read_array(relief, name="relief", dtype=np.dtype(np.float32)),
        "ocean": _read_array(ocean, name="ocean", dtype=np.dtype(np.float32)),
        "insolation": _read_array(insolation, name="insolation", dtype=np.dtype(np.float32)),
        "declination": _read_array(declination, name="declination", dtype=np.dtype(np.float32)),
    }
    expected_input_shapes = {
        "areas": shape,
        "neighbors": (*shape, 4),
        "xyz": (*shape, 3),
        "latitudes": shape,
        "elevation": shape,
        "relief": shape,
        "ocean": shape,
        "insolation": monthly_shape,
        "declination": (12,),
    }
    for name, expected_shape in expected_input_shapes.items():
        if input_arrays[name].shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")

    output_arrays = {
        name: _write_float32(array, name=name)
        for name, array in {
            "climate_orography_out": climate_orography_out,
            "temperature_out": temperature_out,
            "wind_xyz_out": wind_xyz_out,
            "wind_speed_out": wind_speed_out,
            "precipitation_out": precipitation_out,
            "humidity_out": humidity_out,
            "snowfall_out": snowfall_out,
            "snowmelt_out": snowmelt_out,
            "snowpack_out": snowpack_out,
            "evaporation_out": evaporation_out,
            "runoff_out": runoff_out,
            "annual_temperature_out": annual_temperature_out,
            "annual_precipitation_out": annual_precipitation_out,
            "aridity_out": aridity_out,
        }.items()
    }
    expected_output_shapes = {
        "climate_orography_out": shape,
        "temperature_out": monthly_shape,
        "wind_xyz_out": (*monthly_shape, 3),
        "wind_speed_out": monthly_shape,
        "precipitation_out": monthly_shape,
        "humidity_out": monthly_shape,
        "snowfall_out": monthly_shape,
        "snowmelt_out": monthly_shape,
        "snowpack_out": monthly_shape,
        "evaporation_out": monthly_shape,
        "runoff_out": monthly_shape,
        "annual_temperature_out": shape,
        "annual_precipitation_out": shape,
        "aridity_out": shape,
    }
    for name, expected_shape in expected_output_shapes.items():
        if output_arrays[name].shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    _require_disjoint(input_arrays, output_arrays)

    stats = _ffi.new("ClimateStats*")
    status = int(
        _lib.climate_run_cubed_sphere(
            int(np.prod(shape, dtype=np.int64)),
            int(spinup_years),
            int(moisture_spinup_years),
            int(moisture_steps_per_month),
            int(synoptic_mixing_passes),
            float(greenhouse_offset_c),
            float(land_albedo),
            float(ocean_albedo),
            float(olr_intercept_w_m2),
            float(olr_slope_w_m2_c),
            float(heat_transport_w_m2),
            float(land_thermal_response),
            float(ocean_thermal_response),
            float(atmospheric_exchange),
            float(lapse_rate_c_per_km),
            float(wind_scale),
            float(moisture_advection_fraction),
            float(moisture_diffusion_fraction),
            float(orographic_factor),
            float(rain_shadow_factor),
            float(runoff_base_fraction),
            _ffi.cast("double*", _ffi.from_buffer("double[]", input_arrays["areas"])),
            _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", input_arrays["neighbors"])),
            _ffi.cast("float*", _ffi.from_buffer("float[]", input_arrays["xyz"])),
            _ffi.cast("double*", _ffi.from_buffer("double[]", input_arrays["latitudes"])),
            *(
                _ffi.cast("float*", _ffi.from_buffer("float[]", input_arrays[name]))
                for name in ("elevation", "relief", "ocean", "insolation", "declination")
            ),
            *(
                _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays[name]))
                for name in expected_output_shapes
            ),
            stats,
        )
    )
    if status != 0:
        messages = {
            1: "invalid dimensions or null buffers",
            2: "invalid climate controls",
            3: "non-finite climate inputs",
            4: "invalid cubed-sphere topology",
        }
        raise ValueError(
            f"cubed-sphere climate kernel failed: {messages.get(status, f'status {status}')}"
        )

    return {
        "global_mean_temperature_c": float(stats.global_mean_temperature_c),
        "land_mean_temperature_c": float(stats.land_mean_temperature_c),
        "ocean_mean_temperature_c": float(stats.ocean_mean_temperature_c),
        "minimum_monthly_temperature_c": float(stats.minimum_monthly_temperature_c),
        "maximum_monthly_temperature_c": float(stats.maximum_monthly_temperature_c),
        "global_mean_annual_precipitation_mm": float(stats.global_mean_annual_precipitation_mm),
        "land_mean_annual_precipitation_mm": float(stats.land_mean_annual_precipitation_mm),
        "dry_land_area_fraction": float(stats.dry_land_area_fraction),
        "wet_land_area_fraction": float(stats.wet_land_area_fraction),
        "persistent_snow_land_area_fraction": float(stats.persistent_snow_land_area_fraction),
        "maximum_wind_speed_m_s": float(stats.maximum_wind_speed_m_s),
    }


__all__ = ["CUBED_SPHERE_CLIMATE_ABI_VERSION", "run_cubed_sphere_climate"]
