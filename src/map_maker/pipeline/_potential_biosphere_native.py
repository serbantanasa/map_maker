"""Rust-backed bindings for continuous potential-biosphere traits."""

from __future__ import annotations

from typing import Any

import numpy as np
from cffi import FFI  # type: ignore[import-untyped]

from .._native import native_library_info, native_library_path

_CDEF = """
typedef struct {
    float land_mean_annual_npp_kg_c_m2;
    float land_mean_vegetation_cover_fraction;
    float land_mean_standing_biomass_kg_c_m2;
    float potentially_vegetated_land_area_fraction;
    float land_mean_growing_season_fraction;
    float land_mean_woody_allocation_trait;
    float maximum_rooting_depth_m;
    float maximum_canopy_height_m;
} PotentialBiosphereStats;

uint32_t potential_biosphere_native_abi_version(void);
int32_t potential_biosphere_run(
    int32_t cell_count,
    double energy_per_kg_carbon_mj,
    double cover_half_saturation_npp_kg_c_m2_year,
    double active_month_thermal_threshold,
    double active_month_water_threshold,
    double cold_pressure_reference_c,
    double cold_pressure_release_c,
    double heat_pressure_onset_c,
    double heat_pressure_reference_c,
    double minimum_biomass_residence_years,
    double maximum_biomass_residence_years,
    double biomass_residence_baseline_fraction,
    double woody_biomass_residence_weight,
    double resource_conservative_biomass_residence_weight,
    double low_productivity_biomass_residence_weight,
    double maximum_rooting_depth_m,
    double maximum_canopy_height_m,
    double maximum_leaf_area_index,
    double maximum_standing_biomass_kg_c_m2,
    const double* areas,
    const float* ocean,
    const float* monthly_primary_energy,
    const float* monthly_thermal_opportunity,
    const float* monthly_water_opportunity,
    const float* monthly_temperature,
    const float* monthly_soil_saturation,
    const float* surface_support,
    const float* nutrient_support,
    const float* environmental_stress,
    const float* soil_depth,
    const float* regolith_depth,
    const float* salinity,
    const float* hydric_fraction,
    const float* soil_confidence,
    const float* envelope_confidence,
    float* monthly_npp,
    float* annual_npp,
    float* vegetation_cover,
    float* standing_biomass,
    float* growing_season,
    float* productivity_seasonality,
    float* drought_pressure,
    float* cold_pressure,
    float* heat_pressure,
    float* waterlogging_pressure,
    float* salinity_pressure,
    float* woody_trait,
    float* resource_conservative_trait,
    float* rooting_depth,
    float* canopy_height,
    float* leaf_area_index,
    float* fuel_continuity,
    float* confidence,
    PotentialBiosphereStats* stats_out
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
native_library_info("potential_biosphere_native")
_lib = _ffi.dlopen(str(native_library_path("potential_biosphere_native")))


def run_potential_biosphere(  # noqa: PLR0913
    *,
    energy_per_kg_carbon_mj: float,
    cover_half_saturation_npp_kg_c_m2_year: float,
    active_month_thermal_threshold: float,
    active_month_water_threshold: float,
    cold_pressure_reference_c: float,
    cold_pressure_release_c: float,
    heat_pressure_onset_c: float,
    heat_pressure_reference_c: float,
    minimum_biomass_residence_years: float,
    maximum_biomass_residence_years: float,
    biomass_residence_baseline_fraction: float,
    woody_biomass_residence_weight: float,
    resource_conservative_biomass_residence_weight: float,
    low_productivity_biomass_residence_weight: float,
    maximum_rooting_depth_m: float,
    maximum_canopy_height_m: float,
    maximum_leaf_area_index: float,
    maximum_standing_biomass_kg_c_m2: float,
    areas: np.ndarray,
    ocean: np.ndarray,
    monthly_primary_energy: np.ndarray,
    monthly_thermal_opportunity: np.ndarray,
    monthly_water_opportunity: np.ndarray,
    monthly_temperature: np.ndarray,
    monthly_soil_saturation: np.ndarray,
    surface_support: np.ndarray,
    nutrient_support: np.ndarray,
    environmental_stress: np.ndarray,
    soil_depth: np.ndarray,
    regolith_depth: np.ndarray,
    salinity: np.ndarray,
    hydric_fraction: np.ndarray,
    soil_confidence: np.ndarray,
    envelope_confidence: np.ndarray,
    **output_arrays: np.ndarray,
) -> dict[str, Any]:
    area_array = _read_array(areas, name="areas", dtype=np.dtype(np.float64))
    if area_array.ndim < 1 or area_array.size == 0:
        raise ValueError("areas must contain at least one spatial cell")
    shape = area_array.shape
    monthly_shape = (12, *shape)
    float_inputs_raw = {
        "ocean": ocean,
        "monthly_primary_energy": monthly_primary_energy,
        "monthly_thermal_opportunity": monthly_thermal_opportunity,
        "monthly_water_opportunity": monthly_water_opportunity,
        "monthly_temperature": monthly_temperature,
        "monthly_soil_saturation": monthly_soil_saturation,
        "surface_support": surface_support,
        "nutrient_support": nutrient_support,
        "environmental_stress": environmental_stress,
        "soil_depth": soil_depth,
        "regolith_depth": regolith_depth,
        "salinity": salinity,
        "hydric_fraction": hydric_fraction,
        "soil_confidence": soil_confidence,
        "envelope_confidence": envelope_confidence,
    }
    inputs = {
        "areas": area_array,
        **{
            name: _read_array(value, name=name, dtype=np.dtype(np.float32))
            for name, value in float_inputs_raw.items()
        },
    }
    monthly_inputs = {
        "monthly_primary_energy",
        "monthly_thermal_opportunity",
        "monthly_water_opportunity",
        "monthly_temperature",
        "monthly_soil_saturation",
    }
    for name, value in inputs.items():
        expected = monthly_shape if name in monthly_inputs else shape
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")

    expected_outputs = {
        "monthly_npp_out",
        "annual_npp_out",
        "vegetation_cover_out",
        "standing_biomass_out",
        "growing_season_out",
        "productivity_seasonality_out",
        "drought_pressure_out",
        "cold_pressure_out",
        "heat_pressure_out",
        "waterlogging_pressure_out",
        "salinity_pressure_out",
        "woody_trait_out",
        "resource_conservative_trait_out",
        "rooting_depth_out",
        "canopy_height_out",
        "leaf_area_index_out",
        "fuel_continuity_out",
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
        expected = monthly_shape if name == "monthly_npp_out" else shape
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
    stats = _ffi.new("PotentialBiosphereStats*")
    status = int(
        _lib.potential_biosphere_run(
            int(np.prod(shape)),
            energy_per_kg_carbon_mj,
            cover_half_saturation_npp_kg_c_m2_year,
            active_month_thermal_threshold,
            active_month_water_threshold,
            cold_pressure_reference_c,
            cold_pressure_release_c,
            heat_pressure_onset_c,
            heat_pressure_reference_c,
            minimum_biomass_residence_years,
            maximum_biomass_residence_years,
            biomass_residence_baseline_fraction,
            woody_biomass_residence_weight,
            resource_conservative_biomass_residence_weight,
            low_productivity_biomass_residence_weight,
            maximum_rooting_depth_m,
            maximum_canopy_height_m,
            maximum_leaf_area_index,
            maximum_standing_biomass_kg_c_m2,
            _ffi.cast("const double*", _ffi.from_buffer("double[]", inputs["areas"])),
            input_ptr["ocean"],
            input_ptr["monthly_primary_energy"],
            input_ptr["monthly_thermal_opportunity"],
            input_ptr["monthly_water_opportunity"],
            input_ptr["monthly_temperature"],
            input_ptr["monthly_soil_saturation"],
            input_ptr["surface_support"],
            input_ptr["nutrient_support"],
            input_ptr["environmental_stress"],
            input_ptr["soil_depth"],
            input_ptr["regolith_depth"],
            input_ptr["salinity"],
            input_ptr["hydric_fraction"],
            input_ptr["soil_confidence"],
            input_ptr["envelope_confidence"],
            output_ptr["monthly_npp_out"],
            output_ptr["annual_npp_out"],
            output_ptr["vegetation_cover_out"],
            output_ptr["standing_biomass_out"],
            output_ptr["growing_season_out"],
            output_ptr["productivity_seasonality_out"],
            output_ptr["drought_pressure_out"],
            output_ptr["cold_pressure_out"],
            output_ptr["heat_pressure_out"],
            output_ptr["waterlogging_pressure_out"],
            output_ptr["salinity_pressure_out"],
            output_ptr["woody_trait_out"],
            output_ptr["resource_conservative_trait_out"],
            output_ptr["rooting_depth_out"],
            output_ptr["canopy_height_out"],
            output_ptr["leaf_area_index_out"],
            output_ptr["fuel_continuity_out"],
            output_ptr["confidence_out"],
            stats,
        )
    )
    if status != 0:
        messages = {
            1: "invalid dimensions or null buffers",
            2: "invalid potential-biosphere controls",
            3: "non-finite or physically invalid potential-biosphere inputs",
        }
        raise ValueError(
            f"potential-biosphere kernel failed: {messages.get(status, f'status {status}')}"
        )
    return {
        name: float(getattr(stats, name))
        for name in (
            "land_mean_annual_npp_kg_c_m2",
            "land_mean_vegetation_cover_fraction",
            "land_mean_standing_biomass_kg_c_m2",
            "potentially_vegetated_land_area_fraction",
            "land_mean_growing_season_fraction",
            "land_mean_woody_allocation_trait",
            "maximum_rooting_depth_m",
            "maximum_canopy_height_m",
        )
    }


__all__ = ["run_potential_biosphere"]
