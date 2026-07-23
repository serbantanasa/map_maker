"""Rust-backed bindings for conservative functional-vegetation mixtures."""

from __future__ import annotations

from typing import Any

import numpy as np
from cffi import FFI  # type: ignore[import-untyped]

from .._native import native_library_info, native_library_path

FUNCTIONAL_TYPE_COUNT = 8
NONVEGETATED_TYPE_COUNT = 5
RESOURCE_POTENTIAL_COUNT = 5

_CDEF = """
typedef struct {
    float land_mean_functional_vegetated_fraction;
    float vegetated_woody_fraction;
    float vegetated_herbaceous_fraction;
    float vegetated_hydrophytic_fraction;
    float land_mean_bare_ground_fraction;
    float land_mean_saline_barren_fraction;
    float land_mean_ice_fraction;
    float land_mean_inland_water_fraction;
    float land_mean_unsupported_surface_fraction;
    float land_mean_fire_tendency;
    float land_mean_crop_potential;
    float maximum_partition_absolute_error;
} FunctionalVegetationStats;

uint32_t functional_vegetation_native_abi_version(void);
int32_t functional_vegetation_run(
    int32_t cell_count,
    double warm_transition_midpoint_c,
    double warm_transition_width_c,
    double npp_response_half_saturation_kg_c_m2_year,
    double biomass_response_half_saturation_kg_c_m2,
    double terrain_relief_half_saturation_m,
    double crop_soil_depth_half_saturation_m,
    double strategy_confidence_multiplier,
    const double* areas,
    const float* ocean,
    const float* vegetation_cover,
    const float* annual_npp,
    const float* standing_biomass,
    const float* growing_season,
    const float* productivity_seasonality,
    const float* drought_pressure,
    const float* cold_pressure,
    const float* heat_pressure,
    const float* waterlogging_pressure,
    const float* salinity_pressure,
    const float* woody_trait,
    const float* resource_conservative_trait,
    const float* fuel_continuity,
    const float* biosphere_confidence,
    const float* annual_temperature,
    const float* soil_fertility,
    const float* soil_depth,
    const float* soil_bearing,
    const float* soil_drainage,
    const float* glacier_fraction,
    const float* lake_fraction,
    const float* wetland_fraction,
    const float* terrain_relief,
    float* functional_type_fractions,
    float* nonvegetated_fractions,
    float* resource_potentials,
    float* confidence,
    uint8_t* dominant_cover_code,
    FunctionalVegetationStats* stats_out
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


def _write_array(array: np.ndarray, *, name: str, dtype: np.dtype) -> np.ndarray:
    value = _read_array(array, name=name, dtype=dtype)
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
native_library_info("functional_vegetation_native")
_lib = _ffi.dlopen(str(native_library_path("functional_vegetation_native")))


def run_functional_vegetation(  # noqa: PLR0913
    *,
    warm_transition_midpoint_c: float,
    warm_transition_width_c: float,
    npp_response_half_saturation_kg_c_m2_year: float,
    biomass_response_half_saturation_kg_c_m2: float,
    terrain_relief_half_saturation_m: float,
    crop_soil_depth_half_saturation_m: float,
    strategy_confidence_multiplier: float,
    areas: np.ndarray,
    ocean: np.ndarray,
    vegetation_cover: np.ndarray,
    annual_npp: np.ndarray,
    standing_biomass: np.ndarray,
    growing_season: np.ndarray,
    productivity_seasonality: np.ndarray,
    drought_pressure: np.ndarray,
    cold_pressure: np.ndarray,
    heat_pressure: np.ndarray,
    waterlogging_pressure: np.ndarray,
    salinity_pressure: np.ndarray,
    woody_trait: np.ndarray,
    resource_conservative_trait: np.ndarray,
    fuel_continuity: np.ndarray,
    biosphere_confidence: np.ndarray,
    annual_temperature: np.ndarray,
    soil_fertility: np.ndarray,
    soil_depth: np.ndarray,
    soil_bearing: np.ndarray,
    soil_drainage: np.ndarray,
    glacier_fraction: np.ndarray,
    lake_fraction: np.ndarray,
    wetland_fraction: np.ndarray,
    terrain_relief: np.ndarray,
    **output_arrays: np.ndarray,
) -> dict[str, Any]:
    area_array = _read_array(areas, name="areas", dtype=np.dtype(np.float64))
    if area_array.ndim < 1 or area_array.size == 0:
        raise ValueError("areas must contain at least one spatial cell")
    shape = area_array.shape
    float_inputs_raw = {
        "ocean": ocean,
        "vegetation_cover": vegetation_cover,
        "annual_npp": annual_npp,
        "standing_biomass": standing_biomass,
        "growing_season": growing_season,
        "productivity_seasonality": productivity_seasonality,
        "drought_pressure": drought_pressure,
        "cold_pressure": cold_pressure,
        "heat_pressure": heat_pressure,
        "waterlogging_pressure": waterlogging_pressure,
        "salinity_pressure": salinity_pressure,
        "woody_trait": woody_trait,
        "resource_conservative_trait": resource_conservative_trait,
        "fuel_continuity": fuel_continuity,
        "biosphere_confidence": biosphere_confidence,
        "annual_temperature": annual_temperature,
        "soil_fertility": soil_fertility,
        "soil_depth": soil_depth,
        "soil_bearing": soil_bearing,
        "soil_drainage": soil_drainage,
        "glacier_fraction": glacier_fraction,
        "lake_fraction": lake_fraction,
        "wetland_fraction": wetland_fraction,
        "terrain_relief": terrain_relief,
    }
    inputs = {
        "areas": area_array,
        **{
            name: _read_array(value, name=name, dtype=np.dtype(np.float32))
            for name, value in float_inputs_raw.items()
        },
    }
    for name, value in inputs.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

    expected_shapes = {
        "functional_type_fractions_out": (FUNCTIONAL_TYPE_COUNT, *shape),
        "nonvegetated_fractions_out": (NONVEGETATED_TYPE_COUNT, *shape),
        "resource_potentials_out": (RESOURCE_POTENTIAL_COUNT, *shape),
        "confidence_out": shape,
        "dominant_cover_code_out": shape,
    }
    unknown = set(output_arrays) - set(expected_shapes)
    missing = set(expected_shapes) - set(output_arrays)
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown outputs: {', '.join(sorted(unknown))}")
        if missing:
            details.append(f"missing outputs: {', '.join(sorted(missing))}")
        raise ValueError("; ".join(details))
    outputs: dict[str, np.ndarray] = {}
    for name, expected_shape in expected_shapes.items():
        dtype = np.dtype(np.uint8) if name == "dominant_cover_code_out" else np.dtype(np.float32)
        output = _write_array(output_arrays[name], name=name, dtype=dtype)
        if output.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        outputs[name] = output
    _require_disjoint(inputs, outputs)

    input_ptr = {
        name: _ffi.cast("const float*", _ffi.from_buffer("float[]", value))
        for name, value in inputs.items()
        if value.dtype == np.float32
    }
    stats = _ffi.new("FunctionalVegetationStats*")
    status = int(
        _lib.functional_vegetation_run(
            int(np.prod(shape)),
            warm_transition_midpoint_c,
            warm_transition_width_c,
            npp_response_half_saturation_kg_c_m2_year,
            biomass_response_half_saturation_kg_c_m2,
            terrain_relief_half_saturation_m,
            crop_soil_depth_half_saturation_m,
            strategy_confidence_multiplier,
            _ffi.cast("const double*", _ffi.from_buffer("double[]", inputs["areas"])),
            input_ptr["ocean"],
            input_ptr["vegetation_cover"],
            input_ptr["annual_npp"],
            input_ptr["standing_biomass"],
            input_ptr["growing_season"],
            input_ptr["productivity_seasonality"],
            input_ptr["drought_pressure"],
            input_ptr["cold_pressure"],
            input_ptr["heat_pressure"],
            input_ptr["waterlogging_pressure"],
            input_ptr["salinity_pressure"],
            input_ptr["woody_trait"],
            input_ptr["resource_conservative_trait"],
            input_ptr["fuel_continuity"],
            input_ptr["biosphere_confidence"],
            input_ptr["annual_temperature"],
            input_ptr["soil_fertility"],
            input_ptr["soil_depth"],
            input_ptr["soil_bearing"],
            input_ptr["soil_drainage"],
            input_ptr["glacier_fraction"],
            input_ptr["lake_fraction"],
            input_ptr["wetland_fraction"],
            input_ptr["terrain_relief"],
            _ffi.cast(
                "float*",
                _ffi.from_buffer("float[]", outputs["functional_type_fractions_out"]),
            ),
            _ffi.cast("float*", _ffi.from_buffer("float[]", outputs["nonvegetated_fractions_out"])),
            _ffi.cast("float*", _ffi.from_buffer("float[]", outputs["resource_potentials_out"])),
            _ffi.cast("float*", _ffi.from_buffer("float[]", outputs["confidence_out"])),
            _ffi.cast(
                "uint8_t*", _ffi.from_buffer("uint8_t[]", outputs["dominant_cover_code_out"])
            ),
            stats,
        )
    )
    if status != 0:
        messages = {
            1: "invalid dimensions or null buffers",
            2: "invalid functional-vegetation controls",
            3: "non-finite or physically invalid functional-vegetation inputs",
        }
        raise ValueError(
            f"functional-vegetation kernel failed: {messages.get(status, f'status {status}')}"
        )
    return {
        name: float(getattr(stats, name))
        for name in (
            "land_mean_functional_vegetated_fraction",
            "vegetated_woody_fraction",
            "vegetated_herbaceous_fraction",
            "vegetated_hydrophytic_fraction",
            "land_mean_bare_ground_fraction",
            "land_mean_saline_barren_fraction",
            "land_mean_ice_fraction",
            "land_mean_inland_water_fraction",
            "land_mean_unsupported_surface_fraction",
            "land_mean_fire_tendency",
            "land_mean_crop_potential",
            "maximum_partition_absolute_error",
        )
    }


__all__ = [
    "FUNCTIONAL_TYPE_COUNT",
    "NONVEGETATED_TYPE_COUNT",
    "RESOURCE_POTENTIAL_COUNT",
    "run_functional_vegetation",
]
