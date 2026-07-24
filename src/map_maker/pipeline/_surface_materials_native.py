"""Rust-backed bindings for L2 surface materials and initial soils."""

from __future__ import annotations

from typing import Any

import numpy as np
from cffi import FFI  # type: ignore[import-untyped]

from .._native import native_library_info, native_library_path

_CDEF = """
typedef struct {
    float soil_bearing_land_area_fraction;
    float bedrock_land_area_fraction;
    float residual_land_area_fraction;
    float colluvium_land_area_fraction;
    float alluvium_land_area_fraction;
    float lacustrine_land_area_fraction;
    float glacial_land_area_fraction;
    float volcaniclastic_land_area_fraction;
    float hydric_soil_land_area_fraction;
    float land_mean_regolith_depth_m;
    float land_mean_soil_depth_m;
    float land_mean_organic_carbon_kg_m2;
    float material_balance_max_error;
    float texture_balance_max_error;
    double water_balance_relative_error;
} SurfaceMaterialsStats;

uint32_t surface_materials_native_abi_version(void);
int32_t surface_materials_run(
    int32_t cell_count,
    int32_t spinup_years,
    double maximum_regolith_depth_m,
    double maximum_soil_depth_m,
    double maximum_alluvial_fraction,
    double maximum_lacustrine_fraction,
    double maximum_glacial_fraction,
    double weathering_temperature_scale_c,
    double weathering_precipitation_scale_mm,
    double soil_evaporation_factor,
    double monthly_deep_drainage_fraction,
    const double* areas,
    const float* ocean,
    const uint8_t* province_class,
    const float* crust_age,
    const float* rock_strength,
    const float* accommodation,
    const float* province_confidence,
    const float* elevation_confidence,
    const float* relief,
    const float* terrain_slope,
    const float* river_corridor,
    const float* floodplain,
    const float* lake_fraction,
    const float* wetland_fraction,
    const float* depression_fill_depth,
    const float* refined_mask,
    const float* refined_lake_fraction,
    const float* refined_wetland_fraction,
    const float* refined_hydroperiod,
    const float* refined_salinity,
    const float* recent_erosion_depth,
    const float* recent_deposition_depth,
    const float* glacier_fraction,
    const float* annual_temperature,
    const float* annual_precipitation,
    const float* monthly_temperature,
    const float* monthly_precipitation,
    const float* monthly_evaporation,
    const float* monthly_snowfall,
    const float* monthly_snowmelt,
    const float* monthly_glacier_melt,
    float* bedrock,
    float* residual,
    float* colluvium,
    float* alluvium,
    float* lacustrine,
    float* glacial,
    float* volcaniclastic,
    uint8_t* dominant_material,
    float* soil_bearing,
    float* regolith_depth,
    float* soil_depth,
    float* sand,
    float* silt,
    float* clay,
    float* coarse_fragments,
    float* bulk_density,
    float* organic_carbon,
    float* soil_ph,
    float* carbonate,
    float* salinity,
    float* drainage,
    float* available_water_capacity,
    float* nutrient_potential,
    float* fertility_potential,
    float* erodibility,
    float* reset_age,
    float* hydric_fraction,
    float* soil_confidence,
    float* monthly_liquid_input,
    float* monthly_soil_water,
    float* monthly_saturation,
    float* monthly_evapotranspiration,
    float* monthly_runoff,
    float* monthly_deep_drainage,
    float* annual_storage_change,
    SurfaceMaterialsStats* stats_out
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
    return value  # type: ignore[no-any-return]


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
native_library_info("surface_materials_native")
_lib = _ffi.dlopen(str(native_library_path("surface_materials_native")))


def run_surface_materials(  # noqa: PLR0913
    *,
    spinup_years: int,
    maximum_regolith_depth_m: float,
    maximum_soil_depth_m: float,
    maximum_alluvial_fraction: float,
    maximum_lacustrine_fraction: float,
    maximum_glacial_fraction: float,
    weathering_temperature_scale_c: float,
    weathering_precipitation_scale_mm: float,
    soil_evaporation_factor: float,
    monthly_deep_drainage_fraction: float,
    areas: np.ndarray,
    ocean: np.ndarray,
    province_class: np.ndarray,
    crust_age: np.ndarray,
    rock_strength: np.ndarray,
    accommodation: np.ndarray,
    province_confidence: np.ndarray,
    elevation_confidence: np.ndarray,
    relief: np.ndarray,
    terrain_slope: np.ndarray,
    river_corridor: np.ndarray,
    floodplain: np.ndarray,
    lake_fraction: np.ndarray,
    wetland_fraction: np.ndarray,
    depression_fill_depth: np.ndarray,
    refined_mask: np.ndarray,
    refined_lake_fraction: np.ndarray,
    refined_wetland_fraction: np.ndarray,
    refined_hydroperiod: np.ndarray,
    refined_salinity: np.ndarray,
    recent_erosion_depth: np.ndarray,
    recent_deposition_depth: np.ndarray,
    glacier_fraction: np.ndarray,
    annual_temperature: np.ndarray,
    annual_precipitation: np.ndarray,
    monthly_temperature: np.ndarray,
    monthly_precipitation: np.ndarray,
    monthly_evaporation: np.ndarray,
    monthly_snowfall: np.ndarray,
    monthly_snowmelt: np.ndarray,
    monthly_glacier_melt: np.ndarray,
    **output_arrays: np.ndarray,
) -> dict[str, Any]:
    area_array = _read_array(areas, name="areas", dtype=np.dtype(np.float64))
    if area_array.ndim < 1 or area_array.size == 0:
        raise ValueError("areas must be a non-empty spatial array")
    if area_array.size > np.iinfo(np.int32).max:
        raise ValueError("surface-material batch exceeds the native int32 cell limit")
    shape = area_array.shape
    monthly_shape = (12, *shape)
    float_inputs = {
        "ocean": ocean,
        "crust_age": crust_age,
        "rock_strength": rock_strength,
        "accommodation": accommodation,
        "province_confidence": province_confidence,
        "elevation_confidence": elevation_confidence,
        "relief": relief,
        "terrain_slope": terrain_slope,
        "river_corridor": river_corridor,
        "floodplain": floodplain,
        "lake_fraction": lake_fraction,
        "wetland_fraction": wetland_fraction,
        "depression_fill_depth": depression_fill_depth,
        "refined_mask": refined_mask,
        "refined_lake_fraction": refined_lake_fraction,
        "refined_wetland_fraction": refined_wetland_fraction,
        "refined_hydroperiod": refined_hydroperiod,
        "refined_salinity": refined_salinity,
        "recent_erosion_depth": recent_erosion_depth,
        "recent_deposition_depth": recent_deposition_depth,
        "glacier_fraction": glacier_fraction,
        "annual_temperature": annual_temperature,
        "annual_precipitation": annual_precipitation,
        "monthly_temperature": monthly_temperature,
        "monthly_precipitation": monthly_precipitation,
        "monthly_evaporation": monthly_evaporation,
        "monthly_snowfall": monthly_snowfall,
        "monthly_snowmelt": monthly_snowmelt,
        "monthly_glacier_melt": monthly_glacier_melt,
    }
    inputs = {
        "areas": area_array,
        "province_class": _read_array(
            province_class, name="province_class", dtype=np.dtype(np.uint8)
        ),
        **{
            name: _read_array(value, name=name, dtype=np.dtype(np.float32))
            for name, value in float_inputs.items()
        },
    }
    monthly_inputs = {
        "monthly_temperature",
        "monthly_precipitation",
        "monthly_evaporation",
        "monthly_snowfall",
        "monthly_snowmelt",
        "monthly_glacier_melt",
    }
    for name, value in inputs.items():
        expected = monthly_shape if name in monthly_inputs else shape
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")

    expected_outputs = {
        "bedrock_out",
        "residual_out",
        "colluvium_out",
        "alluvium_out",
        "lacustrine_out",
        "glacial_out",
        "volcaniclastic_out",
        "dominant_material_out",
        "soil_bearing_out",
        "regolith_depth_out",
        "soil_depth_out",
        "sand_out",
        "silt_out",
        "clay_out",
        "coarse_fragments_out",
        "bulk_density_out",
        "organic_carbon_out",
        "soil_ph_out",
        "carbonate_out",
        "salinity_out",
        "drainage_out",
        "available_water_capacity_out",
        "nutrient_potential_out",
        "fertility_potential_out",
        "erodibility_out",
        "reset_age_out",
        "hydric_fraction_out",
        "soil_confidence_out",
        "monthly_liquid_input_out",
        "monthly_soil_water_out",
        "monthly_saturation_out",
        "monthly_evapotranspiration_out",
        "monthly_runoff_out",
        "monthly_deep_drainage_out",
        "annual_storage_change_out",
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
    monthly_outputs = {
        "monthly_liquid_input_out",
        "monthly_soil_water_out",
        "monthly_saturation_out",
        "monthly_evapotranspiration_out",
        "monthly_runoff_out",
        "monthly_deep_drainage_out",
    }
    outputs: dict[str, np.ndarray] = {}
    for name, value in output_arrays.items():
        dtype = np.dtype(np.uint8) if name == "dominant_material_out" else np.dtype(np.float32)
        output = _write_array(value, name=name, dtype=dtype)
        expected = monthly_shape if name in monthly_outputs else shape
        if output.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        outputs[name] = output
    _require_disjoint(inputs, outputs)

    float_input_ptr = {
        name: _ffi.cast("const float*", _ffi.from_buffer("float[]", value))
        for name, value in inputs.items()
        if value.dtype == np.float32
    }
    float_output_ptr = {
        name: _ffi.cast("float*", _ffi.from_buffer("float[]", value))
        for name, value in outputs.items()
        if value.dtype == np.float32
    }
    stats = _ffi.new("SurfaceMaterialsStats*")
    status = int(
        _lib.surface_materials_run(
            int(np.prod(shape)),
            spinup_years,
            maximum_regolith_depth_m,
            maximum_soil_depth_m,
            maximum_alluvial_fraction,
            maximum_lacustrine_fraction,
            maximum_glacial_fraction,
            weathering_temperature_scale_c,
            weathering_precipitation_scale_mm,
            soil_evaporation_factor,
            monthly_deep_drainage_fraction,
            _ffi.cast("const double*", _ffi.from_buffer("double[]", inputs["areas"])),
            float_input_ptr["ocean"],
            _ffi.cast("const uint8_t*", _ffi.from_buffer("uint8_t[]", inputs["province_class"])),
            float_input_ptr["crust_age"],
            float_input_ptr["rock_strength"],
            float_input_ptr["accommodation"],
            float_input_ptr["province_confidence"],
            float_input_ptr["elevation_confidence"],
            float_input_ptr["relief"],
            float_input_ptr["terrain_slope"],
            float_input_ptr["river_corridor"],
            float_input_ptr["floodplain"],
            float_input_ptr["lake_fraction"],
            float_input_ptr["wetland_fraction"],
            float_input_ptr["depression_fill_depth"],
            float_input_ptr["refined_mask"],
            float_input_ptr["refined_lake_fraction"],
            float_input_ptr["refined_wetland_fraction"],
            float_input_ptr["refined_hydroperiod"],
            float_input_ptr["refined_salinity"],
            float_input_ptr["recent_erosion_depth"],
            float_input_ptr["recent_deposition_depth"],
            float_input_ptr["glacier_fraction"],
            float_input_ptr["annual_temperature"],
            float_input_ptr["annual_precipitation"],
            float_input_ptr["monthly_temperature"],
            float_input_ptr["monthly_precipitation"],
            float_input_ptr["monthly_evaporation"],
            float_input_ptr["monthly_snowfall"],
            float_input_ptr["monthly_snowmelt"],
            float_input_ptr["monthly_glacier_melt"],
            float_output_ptr["bedrock_out"],
            float_output_ptr["residual_out"],
            float_output_ptr["colluvium_out"],
            float_output_ptr["alluvium_out"],
            float_output_ptr["lacustrine_out"],
            float_output_ptr["glacial_out"],
            float_output_ptr["volcaniclastic_out"],
            _ffi.cast(
                "uint8_t*",
                _ffi.from_buffer("uint8_t[]", outputs["dominant_material_out"]),
            ),
            float_output_ptr["soil_bearing_out"],
            float_output_ptr["regolith_depth_out"],
            float_output_ptr["soil_depth_out"],
            float_output_ptr["sand_out"],
            float_output_ptr["silt_out"],
            float_output_ptr["clay_out"],
            float_output_ptr["coarse_fragments_out"],
            float_output_ptr["bulk_density_out"],
            float_output_ptr["organic_carbon_out"],
            float_output_ptr["soil_ph_out"],
            float_output_ptr["carbonate_out"],
            float_output_ptr["salinity_out"],
            float_output_ptr["drainage_out"],
            float_output_ptr["available_water_capacity_out"],
            float_output_ptr["nutrient_potential_out"],
            float_output_ptr["fertility_potential_out"],
            float_output_ptr["erodibility_out"],
            float_output_ptr["reset_age_out"],
            float_output_ptr["hydric_fraction_out"],
            float_output_ptr["soil_confidence_out"],
            float_output_ptr["monthly_liquid_input_out"],
            float_output_ptr["monthly_soil_water_out"],
            float_output_ptr["monthly_saturation_out"],
            float_output_ptr["monthly_evapotranspiration_out"],
            float_output_ptr["monthly_runoff_out"],
            float_output_ptr["monthly_deep_drainage_out"],
            float_output_ptr["annual_storage_change_out"],
            stats,
        )
    )
    if status != 0:
        messages = {
            1: "invalid dimensions or null buffers",
            2: "invalid surface-material controls",
            3: "non-finite surface-material inputs",
        }
        raise ValueError(
            f"surface-material kernel failed: {messages.get(status, f'status {status}') }"
        )
    return {
        name: float(getattr(stats, name))
        for name in (
            "soil_bearing_land_area_fraction",
            "bedrock_land_area_fraction",
            "residual_land_area_fraction",
            "colluvium_land_area_fraction",
            "alluvium_land_area_fraction",
            "lacustrine_land_area_fraction",
            "glacial_land_area_fraction",
            "volcaniclastic_land_area_fraction",
            "hydric_soil_land_area_fraction",
            "land_mean_regolith_depth_m",
            "land_mean_soil_depth_m",
            "land_mean_organic_carbon_kg_m2",
            "material_balance_max_error",
            "texture_balance_max_error",
            "water_balance_relative_error",
        )
    }


__all__ = ["run_surface_materials"]
