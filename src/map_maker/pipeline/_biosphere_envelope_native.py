"""Rust-backed bindings for the terrestrial biosphere resource envelope."""

from __future__ import annotations

from typing import Any

import numpy as np
from cffi import FFI  # type: ignore[import-untyped]

from .._native import native_library_info, native_library_path

_CDEF = """
typedef struct {
    float land_area_fraction;
    float land_mean_annual_par_mj_m2;
    float land_mean_annual_primary_energy_mj_m2;
    float land_mean_thermal_opportunity;
    float land_mean_liquid_water_opportunity;
    float land_mean_carbon_substrate_relative;
    float land_mean_aerobic_oxygen_relative;
    float potentially_productive_land_area_fraction;
} BiosphereEnvelopeStats;

uint32_t biosphere_envelope_native_abi_version(void);
int32_t biosphere_envelope_run(
    int32_t cell_count,
    double seconds_per_month,
    double par_fraction,
    double shortwave_transmission,
    double thermal_minimum_c,
    double thermal_optimum_low_c,
    double thermal_optimum_high_c,
    double thermal_maximum_c,
    double water_input_half_saturation_mm,
    double nutrient_half_saturation_index,
    double co2_half_saturation_pa,
    double reference_co2_partial_pressure_pa,
    double reference_oxygen_partial_pressure_kpa,
    double photosynthetic_conversion_efficiency,
    double minimum_productive_energy_mj_m2_year,
    double confidence_multiplier,
    const double* areas,
    const float* ocean,
    const float* monthly_insolation,
    const float* monthly_temperature,
    const float* monthly_liquid_input,
    const float* monthly_soil_saturation,
    const float* soil_bearing,
    const float* nutrient_potential,
    const float* fertility_potential,
    const float* salinity,
    const float* soil_confidence,
    const float* co2_partial_pressure,
    const float* oxygen_partial_pressure,
    float* monthly_par,
    float* monthly_liquid_opportunity,
    float* monthly_thermal_opportunity,
    float* monthly_primary_energy,
    float* annual_par,
    float* annual_primary_energy,
    float* carbon_substrate_relative,
    float* aerobic_oxygen_relative,
    float* terrestrial_surface_support,
    float* nutrient_support,
    float* environmental_stress,
    float* confidence,
    BiosphereEnvelopeStats* stats_out
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
native_library_info("biosphere_envelope_native")
_lib = _ffi.dlopen(str(native_library_path("biosphere_envelope_native")))


def run_biosphere_envelope(  # noqa: PLR0913
    *,
    seconds_per_month: float,
    par_fraction: float,
    shortwave_transmission: float,
    thermal_minimum_c: float,
    thermal_optimum_low_c: float,
    thermal_optimum_high_c: float,
    thermal_maximum_c: float,
    water_input_half_saturation_mm: float,
    nutrient_half_saturation_index: float,
    co2_half_saturation_pa: float,
    reference_co2_partial_pressure_pa: float,
    reference_oxygen_partial_pressure_kpa: float,
    photosynthetic_conversion_efficiency: float,
    minimum_productive_energy_mj_m2_year: float,
    confidence_multiplier: float,
    areas: np.ndarray,
    ocean: np.ndarray,
    monthly_insolation: np.ndarray,
    monthly_temperature: np.ndarray,
    monthly_liquid_input: np.ndarray,
    monthly_soil_saturation: np.ndarray,
    soil_bearing: np.ndarray,
    nutrient_potential: np.ndarray,
    fertility_potential: np.ndarray,
    salinity: np.ndarray,
    soil_confidence: np.ndarray,
    co2_partial_pressure: np.ndarray,
    oxygen_partial_pressure: np.ndarray,
    **output_arrays: np.ndarray,
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
    float_inputs_raw = {
        "ocean": ocean,
        "monthly_insolation": monthly_insolation,
        "monthly_temperature": monthly_temperature,
        "monthly_liquid_input": monthly_liquid_input,
        "monthly_soil_saturation": monthly_soil_saturation,
        "soil_bearing": soil_bearing,
        "nutrient_potential": nutrient_potential,
        "fertility_potential": fertility_potential,
        "salinity": salinity,
        "soil_confidence": soil_confidence,
        "co2_partial_pressure": co2_partial_pressure,
        "oxygen_partial_pressure": oxygen_partial_pressure,
    }
    inputs = {
        "areas": area_array,
        **{
            name: _read_array(value, name=name, dtype=np.dtype(np.float32))
            for name, value in float_inputs_raw.items()
        },
    }
    monthly_inputs = {
        "monthly_insolation",
        "monthly_temperature",
        "monthly_liquid_input",
        "monthly_soil_saturation",
    }
    for name, value in inputs.items():
        expected = monthly_shape if name in monthly_inputs else shape
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")

    monthly_outputs = {
        "monthly_par_out",
        "monthly_liquid_opportunity_out",
        "monthly_thermal_opportunity_out",
        "monthly_primary_energy_out",
    }
    expected_outputs = monthly_outputs | {
        "annual_par_out",
        "annual_primary_energy_out",
        "carbon_substrate_relative_out",
        "aerobic_oxygen_relative_out",
        "terrestrial_surface_support_out",
        "nutrient_support_out",
        "environmental_stress_out",
        "confidence_out",
    }
    unknown = set(output_arrays) - expected_outputs
    missing = expected_outputs - set(output_arrays)
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown outputs: {', '.join(sorted(unknown))}")
        if missing:
            details.append(f"missing outputs: {', '.join(sorted(missing))}")
        raise ValueError("; ".join(details))
    outputs: dict[str, np.ndarray] = {}
    for name, value in output_arrays.items():
        output = _write_array(value, name=name)
        expected = monthly_shape if name in monthly_outputs else shape
        if output.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        outputs[name] = output
    _require_disjoint(inputs, outputs)

    input_ptr = {
        name: _ffi.cast("const float*", _ffi.from_buffer("float[]", value))
        for name, value in inputs.items()
        if value.dtype == np.float32
    }
    output_ptr = {
        name: _ffi.cast("float*", _ffi.from_buffer("float[]", value))
        for name, value in outputs.items()
    }
    stats = _ffi.new("BiosphereEnvelopeStats*")
    status = int(
        _lib.biosphere_envelope_run(
            int(np.prod(shape)),
            seconds_per_month,
            par_fraction,
            shortwave_transmission,
            thermal_minimum_c,
            thermal_optimum_low_c,
            thermal_optimum_high_c,
            thermal_maximum_c,
            water_input_half_saturation_mm,
            nutrient_half_saturation_index,
            co2_half_saturation_pa,
            reference_co2_partial_pressure_pa,
            reference_oxygen_partial_pressure_kpa,
            photosynthetic_conversion_efficiency,
            minimum_productive_energy_mj_m2_year,
            confidence_multiplier,
            _ffi.cast("const double*", _ffi.from_buffer("double[]", inputs["areas"])),
            input_ptr["ocean"],
            input_ptr["monthly_insolation"],
            input_ptr["monthly_temperature"],
            input_ptr["monthly_liquid_input"],
            input_ptr["monthly_soil_saturation"],
            input_ptr["soil_bearing"],
            input_ptr["nutrient_potential"],
            input_ptr["fertility_potential"],
            input_ptr["salinity"],
            input_ptr["soil_confidence"],
            input_ptr["co2_partial_pressure"],
            input_ptr["oxygen_partial_pressure"],
            output_ptr["monthly_par_out"],
            output_ptr["monthly_liquid_opportunity_out"],
            output_ptr["monthly_thermal_opportunity_out"],
            output_ptr["monthly_primary_energy_out"],
            output_ptr["annual_par_out"],
            output_ptr["annual_primary_energy_out"],
            output_ptr["carbon_substrate_relative_out"],
            output_ptr["aerobic_oxygen_relative_out"],
            output_ptr["terrestrial_surface_support_out"],
            output_ptr["nutrient_support_out"],
            output_ptr["environmental_stress_out"],
            output_ptr["confidence_out"],
            stats,
        )
    )
    if status != 0:
        messages = {
            1: "invalid dimensions or null buffers",
            2: "invalid biosphere-envelope controls",
            3: "non-finite or physically invalid biosphere-envelope inputs",
        }
        raise ValueError(
            f"biosphere-envelope kernel failed: {messages.get(status, f'status {status}')}"
        )
    return {
        name: float(getattr(stats, name))
        for name in (
            "land_area_fraction",
            "land_mean_annual_par_mj_m2",
            "land_mean_annual_primary_energy_mj_m2",
            "land_mean_thermal_opportunity",
            "land_mean_liquid_water_opportunity",
            "land_mean_carbon_substrate_relative",
            "land_mean_aerobic_oxygen_relative",
            "potentially_productive_land_area_fraction",
        )
    }


__all__ = ["run_biosphere_envelope"]
