"""Rust-backed bindings for seasonal snow, firn, and glacier storage."""

from __future__ import annotations

from typing import Any

import numpy as np
from cffi import FFI  # type: ignore[import-untyped]

from .._native import native_library_info, native_library_path

_CDEF = """
typedef struct {
    float seasonal_snow_land_area_fraction;
    float perennial_snow_land_area_fraction;
    float glacierized_land_area_fraction;
    float glacier_ice_land_area_fraction;
    float maximum_glacier_ice_water_equivalent_mm;
    float land_mean_annual_glacier_melt_mm;
    float land_mean_annual_glacier_calving_mm;
    float minimum_sea_ice_ocean_area_fraction;
    float maximum_sea_ice_ocean_area_fraction;
    float mean_sea_ice_ocean_area_fraction;
    float perennial_sea_ice_ocean_area_fraction;
    float seasonal_sea_ice_ocean_area_fraction;
    float maximum_sea_ice_thickness_m;
} CryosphereStats;

uint32_t cryosphere_native_abi_version(void);
int32_t cryosphere_run(
    int32_t cell_count,
    int32_t spinup_years,
    double lapse_rate_c_per_km,
    double relief_elevation_multiplier,
    double maximum_highland_fraction,
    double snow_degree_day_melt_mm_c_month,
    double glacier_degree_day_melt_mm_c_month,
    double firn_conversion_fraction_year,
    double snow_sublimation_fraction_month,
    double glacier_sublimation_fraction_month,
    double glacier_flow_activation_mm,
    double glacier_flow_fraction_year,
    double glacier_reference_thickness_mm,
    double sea_ice_freezing_temperature_c,
    double sea_ice_melt_temperature_c,
    double sea_ice_freeze_rate_mm_c_month,
    double sea_ice_melt_rate_mm_c_month,
    double sea_ice_reference_thickness_mm,
    double sea_ice_maximum_thickness_mm,
    double runoff_base_fraction,
    const double* areas,
    const int32_t* neighbors,
    const float* ocean,
    const float* elevation,
    const float* relief,
    const float* temperature,
    const float* precipitation,
    const float* evaporation,
    float* snowfall,
    float* snowmelt,
    float* snowpack,
    float* firn_to_ice,
    float* glacier_melt,
    float* glacier_ice,
    float* sea_ice_fraction,
    float* sea_ice_thickness_m,
    float* runoff,
    float* annual_mass_balance,
    float* annual_flow_export,
    float* annual_flow_import,
    float* annual_calving,
    float* annual_sublimation,
    float* glacier_fraction,
    CryosphereStats* stats_out
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
native_library_info("cryosphere_native")
_lib = _ffi.dlopen(str(native_library_path("cryosphere_native")))


def run_cryosphere(
    *,
    spinup_years: int,
    lapse_rate_c_per_km: float,
    relief_elevation_multiplier: float,
    maximum_highland_fraction: float,
    snow_degree_day_melt_mm_c_month: float,
    glacier_degree_day_melt_mm_c_month: float,
    firn_conversion_fraction_year: float,
    snow_sublimation_fraction_month: float,
    glacier_sublimation_fraction_month: float,
    glacier_flow_activation_mm: float,
    glacier_flow_fraction_year: float,
    glacier_reference_thickness_mm: float,
    sea_ice_freezing_temperature_c: float,
    sea_ice_melt_temperature_c: float,
    sea_ice_freeze_rate_mm_c_month: float,
    sea_ice_melt_rate_mm_c_month: float,
    sea_ice_reference_thickness_mm: float,
    sea_ice_maximum_thickness_mm: float,
    runoff_base_fraction: float,
    areas: np.ndarray,
    neighbors: np.ndarray,
    ocean: np.ndarray,
    elevation: np.ndarray,
    relief: np.ndarray,
    temperature: np.ndarray,
    precipitation: np.ndarray,
    evaporation: np.ndarray,
    snowfall_out: np.ndarray,
    snowmelt_out: np.ndarray,
    snowpack_out: np.ndarray,
    firn_to_ice_out: np.ndarray,
    glacier_melt_out: np.ndarray,
    glacier_ice_out: np.ndarray,
    sea_ice_fraction_out: np.ndarray,
    sea_ice_thickness_m_out: np.ndarray,
    runoff_out: np.ndarray,
    annual_mass_balance_out: np.ndarray,
    annual_flow_export_out: np.ndarray,
    annual_flow_import_out: np.ndarray,
    annual_calving_out: np.ndarray,
    annual_sublimation_out: np.ndarray,
    glacier_fraction_out: np.ndarray,
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
    inputs = {
        "areas": area_array,
        "neighbors": _read_array(neighbors, name="neighbors", dtype=np.dtype(np.int32)),
        "ocean": _read_array(ocean, name="ocean", dtype=np.dtype(np.float32)),
        "elevation": _read_array(elevation, name="elevation", dtype=np.dtype(np.float32)),
        "relief": _read_array(relief, name="relief", dtype=np.dtype(np.float32)),
        "temperature": _read_array(temperature, name="temperature", dtype=np.dtype(np.float32)),
        "precipitation": _read_array(
            precipitation, name="precipitation", dtype=np.dtype(np.float32)
        ),
        "evaporation": _read_array(evaporation, name="evaporation", dtype=np.dtype(np.float32)),
    }
    expected_inputs = {
        "areas": shape,
        "neighbors": (*shape, 4),
        "ocean": shape,
        "elevation": shape,
        "relief": shape,
        "temperature": monthly_shape,
        "precipitation": monthly_shape,
        "evaporation": monthly_shape,
    }
    for name, expected in expected_inputs.items():
        if inputs[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    outputs = {
        name: _write_float32(value, name=name)
        for name, value in {
            "snowfall_out": snowfall_out,
            "snowmelt_out": snowmelt_out,
            "snowpack_out": snowpack_out,
            "firn_to_ice_out": firn_to_ice_out,
            "glacier_melt_out": glacier_melt_out,
            "glacier_ice_out": glacier_ice_out,
            "sea_ice_fraction_out": sea_ice_fraction_out,
            "sea_ice_thickness_m_out": sea_ice_thickness_m_out,
            "runoff_out": runoff_out,
            "annual_mass_balance_out": annual_mass_balance_out,
            "annual_flow_export_out": annual_flow_export_out,
            "annual_flow_import_out": annual_flow_import_out,
            "annual_calving_out": annual_calving_out,
            "annual_sublimation_out": annual_sublimation_out,
            "glacier_fraction_out": glacier_fraction_out,
        }.items()
    }
    monthly_outputs = {
        "snowfall_out",
        "snowmelt_out",
        "snowpack_out",
        "firn_to_ice_out",
        "glacier_melt_out",
        "glacier_ice_out",
        "sea_ice_fraction_out",
        "sea_ice_thickness_m_out",
        "runoff_out",
    }
    for name, value in outputs.items():
        expected = monthly_shape if name in monthly_outputs else shape
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    _require_disjoint(inputs, outputs)

    stats = _ffi.new("CryosphereStats*")
    status = int(
        _lib.cryosphere_run(
            int(np.prod(shape, dtype=np.int64)),
            int(spinup_years),
            float(lapse_rate_c_per_km),
            float(relief_elevation_multiplier),
            float(maximum_highland_fraction),
            float(snow_degree_day_melt_mm_c_month),
            float(glacier_degree_day_melt_mm_c_month),
            float(firn_conversion_fraction_year),
            float(snow_sublimation_fraction_month),
            float(glacier_sublimation_fraction_month),
            float(glacier_flow_activation_mm),
            float(glacier_flow_fraction_year),
            float(glacier_reference_thickness_mm),
            float(sea_ice_freezing_temperature_c),
            float(sea_ice_melt_temperature_c),
            float(sea_ice_freeze_rate_mm_c_month),
            float(sea_ice_melt_rate_mm_c_month),
            float(sea_ice_reference_thickness_mm),
            float(sea_ice_maximum_thickness_mm),
            float(runoff_base_fraction),
            _ffi.cast("double*", _ffi.from_buffer("double[]", inputs["areas"])),
            _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", inputs["neighbors"])),
            *(
                _ffi.cast("float*", _ffi.from_buffer("float[]", inputs[name]))
                for name in (
                    "ocean",
                    "elevation",
                    "relief",
                    "temperature",
                    "precipitation",
                    "evaporation",
                )
            ),
            *(
                _ffi.cast("float*", _ffi.from_buffer("float[]", outputs[name]))
                for name in (
                    "snowfall_out",
                    "snowmelt_out",
                    "snowpack_out",
                    "firn_to_ice_out",
                    "glacier_melt_out",
                    "glacier_ice_out",
                    "sea_ice_fraction_out",
                    "sea_ice_thickness_m_out",
                    "runoff_out",
                    "annual_mass_balance_out",
                    "annual_flow_export_out",
                    "annual_flow_import_out",
                    "annual_calving_out",
                    "annual_sublimation_out",
                    "glacier_fraction_out",
                )
            ),
            stats,
        )
    )
    if status != 0:
        messages = {
            1: "invalid dimensions or null buffers",
            2: "invalid cryosphere controls",
            3: "non-finite cryosphere inputs",
            4: "invalid cubed-sphere topology",
        }
        raise ValueError(f"cryosphere kernel failed: {messages.get(status, f'status {status}')}")
    return {
        "seasonal_snow_land_area_fraction": float(stats.seasonal_snow_land_area_fraction),
        "perennial_snow_land_area_fraction": float(stats.perennial_snow_land_area_fraction),
        "glacierized_land_area_fraction": float(stats.glacierized_land_area_fraction),
        "glacier_ice_land_area_fraction": float(stats.glacier_ice_land_area_fraction),
        "maximum_glacier_ice_water_equivalent_mm": float(
            stats.maximum_glacier_ice_water_equivalent_mm
        ),
        "land_mean_annual_glacier_melt_mm": float(stats.land_mean_annual_glacier_melt_mm),
        "land_mean_annual_glacier_calving_mm": float(stats.land_mean_annual_glacier_calving_mm),
        "minimum_sea_ice_ocean_area_fraction": float(
            stats.minimum_sea_ice_ocean_area_fraction
        ),
        "maximum_sea_ice_ocean_area_fraction": float(
            stats.maximum_sea_ice_ocean_area_fraction
        ),
        "mean_sea_ice_ocean_area_fraction": float(stats.mean_sea_ice_ocean_area_fraction),
        "perennial_sea_ice_ocean_area_fraction": float(
            stats.perennial_sea_ice_ocean_area_fraction
        ),
        "seasonal_sea_ice_ocean_area_fraction": float(
            stats.seasonal_sea_ice_ocean_area_fraction
        ),
        "maximum_sea_ice_thickness_m": float(stats.maximum_sea_ice_thickness_m),
    }


__all__ = ["run_cryosphere"]
