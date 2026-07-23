"""Rust-backed bindings for derived biome mixtures and landscape labels."""

from __future__ import annotations

from typing import Any

import numpy as np
from cffi import FFI  # type: ignore[import-untyped]

from .._native import native_library_info, native_library_path

BIOME_COUNT = 13
FUNCTIONAL_TYPE_COUNT = 8
NONVEGETATED_TYPE_COUNT = 5
RESOURCE_POTENTIAL_COUNT = 5

_CDEF = """
typedef struct {
    float land_mean_classification_confidence;
    float land_mean_dominance_margin;
    float land_mean_transition_index;
    float ambiguous_land_area_fraction;
    float classifiable_land_area_fraction;
    float maximum_partition_absolute_error;
} DerivedBiomeStats;

uint32_t derived_biomes_native_abi_version(void);
int32_t derived_biomes_run(
    int32_t cell_count,
    double highland_elevation_start_m,
    double highland_elevation_full_m,
    double highland_relief_start_m,
    double highland_relief_full_m,
    double minimum_classifiable_ground_fraction,
    double ambiguity_margin_threshold,
    double transition_confidence_weight,
    const double* areas,
    const float* ocean,
    const float* annual_temperature,
    const float* annual_precipitation,
    const float* growing_season,
    const float* seasonality,
    const float* drought,
    const float* waterlogging,
    const float* biosphere_confidence,
    const float* functional_confidence,
    const float* wetland_fraction,
    const float* elevation,
    const float* relief,
    const float* functional_type_fractions,
    const float* nonvegetated_fractions,
    const float* resource_potentials,
    float* biome_fractions,
    float* classification_confidence,
    float* dominance_margin,
    float* transition_index,
    uint8_t* primary_biome_code,
    uint8_t* secondary_biome_code,
    uint8_t* dominant_landscape_code,
    DerivedBiomeStats* stats_out
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
native_library_info("derived_biomes_native")
_lib = _ffi.dlopen(str(native_library_path("derived_biomes_native")))


def run_derived_biomes(  # noqa: PLR0913
    *,
    highland_elevation_start_m: float,
    highland_elevation_full_m: float,
    highland_relief_start_m: float,
    highland_relief_full_m: float,
    minimum_classifiable_ground_fraction: float,
    ambiguity_margin_threshold: float,
    transition_confidence_weight: float,
    areas: np.ndarray,
    ocean: np.ndarray,
    annual_temperature: np.ndarray,
    annual_precipitation: np.ndarray,
    growing_season: np.ndarray,
    seasonality: np.ndarray,
    drought: np.ndarray,
    waterlogging: np.ndarray,
    biosphere_confidence: np.ndarray,
    functional_confidence: np.ndarray,
    wetland_fraction: np.ndarray,
    elevation: np.ndarray,
    relief: np.ndarray,
    functional_type_fractions: np.ndarray,
    nonvegetated_fractions: np.ndarray,
    resource_potentials: np.ndarray,
    **output_arrays: np.ndarray,
) -> dict[str, Any]:
    area_array = _read_array(areas, name="areas", dtype=np.dtype(np.float64))
    if area_array.ndim < 1 or area_array.size == 0:
        raise ValueError("areas must contain at least one spatial cell")
    shape = area_array.shape
    scalar_inputs_raw = {
        "ocean": ocean,
        "annual_temperature": annual_temperature,
        "annual_precipitation": annual_precipitation,
        "growing_season": growing_season,
        "seasonality": seasonality,
        "drought": drought,
        "waterlogging": waterlogging,
        "biosphere_confidence": biosphere_confidence,
        "functional_confidence": functional_confidence,
        "wetland_fraction": wetland_fraction,
        "elevation": elevation,
        "relief": relief,
    }
    stacked_inputs_raw = {
        "functional_type_fractions": (
            functional_type_fractions,
            (FUNCTIONAL_TYPE_COUNT, *shape),
        ),
        "nonvegetated_fractions": (
            nonvegetated_fractions,
            (NONVEGETATED_TYPE_COUNT, *shape),
        ),
        "resource_potentials": (
            resource_potentials,
            (RESOURCE_POTENTIAL_COUNT, *shape),
        ),
    }
    inputs = {
        "areas": area_array,
        **{
            name: _read_array(value, name=name, dtype=np.dtype(np.float32))
            for name, value in scalar_inputs_raw.items()
        },
        **{
            name: _read_array(value, name=name, dtype=np.dtype(np.float32))
            for name, (value, _) in stacked_inputs_raw.items()
        },
    }
    for name in scalar_inputs_raw:
        if inputs[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for name, (_, expected_shape) in stacked_inputs_raw.items():
        if inputs[name].shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")

    expected_shapes = {
        "biome_fractions_out": (BIOME_COUNT, *shape),
        "classification_confidence_out": shape,
        "dominance_margin_out": shape,
        "transition_index_out": shape,
        "primary_biome_code_out": shape,
        "secondary_biome_code_out": shape,
        "dominant_landscape_code_out": shape,
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
    code_outputs = {
        "primary_biome_code_out",
        "secondary_biome_code_out",
        "dominant_landscape_code_out",
    }
    outputs: dict[str, np.ndarray] = {}
    for name, expected_shape in expected_shapes.items():
        dtype = np.dtype(np.uint8) if name in code_outputs else np.dtype(np.float32)
        output = _write_array(output_arrays[name], name=name, dtype=dtype)
        if output.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        outputs[name] = output
    _require_disjoint(inputs, outputs)

    float_ptr = {
        name: _ffi.cast("const float*", _ffi.from_buffer("float[]", value))
        for name, value in inputs.items()
        if value.dtype == np.float32
    }
    stats = _ffi.new("DerivedBiomeStats*")
    status = int(
        _lib.derived_biomes_run(
            int(np.prod(shape)),
            highland_elevation_start_m,
            highland_elevation_full_m,
            highland_relief_start_m,
            highland_relief_full_m,
            minimum_classifiable_ground_fraction,
            ambiguity_margin_threshold,
            transition_confidence_weight,
            _ffi.cast("const double*", _ffi.from_buffer("double[]", inputs["areas"])),
            float_ptr["ocean"],
            float_ptr["annual_temperature"],
            float_ptr["annual_precipitation"],
            float_ptr["growing_season"],
            float_ptr["seasonality"],
            float_ptr["drought"],
            float_ptr["waterlogging"],
            float_ptr["biosphere_confidence"],
            float_ptr["functional_confidence"],
            float_ptr["wetland_fraction"],
            float_ptr["elevation"],
            float_ptr["relief"],
            float_ptr["functional_type_fractions"],
            float_ptr["nonvegetated_fractions"],
            float_ptr["resource_potentials"],
            _ffi.cast("float*", _ffi.from_buffer("float[]", outputs["biome_fractions_out"])),
            _ffi.cast(
                "float*",
                _ffi.from_buffer("float[]", outputs["classification_confidence_out"]),
            ),
            _ffi.cast("float*", _ffi.from_buffer("float[]", outputs["dominance_margin_out"])),
            _ffi.cast("float*", _ffi.from_buffer("float[]", outputs["transition_index_out"])),
            _ffi.cast("uint8_t*", _ffi.from_buffer("uint8_t[]", outputs["primary_biome_code_out"])),
            _ffi.cast(
                "uint8_t*", _ffi.from_buffer("uint8_t[]", outputs["secondary_biome_code_out"])
            ),
            _ffi.cast(
                "uint8_t*",
                _ffi.from_buffer("uint8_t[]", outputs["dominant_landscape_code_out"]),
            ),
            stats,
        )
    )
    if status != 0:
        messages = {
            1: "invalid dimensions or null buffers",
            2: "invalid derived-biome controls",
            3: "non-finite or physically invalid derived-biome inputs",
        }
        raise ValueError(f"derived-biome kernel failed: {messages.get(status, f'status {status}')}")
    return {
        name: float(getattr(stats, name))
        for name in (
            "land_mean_classification_confidence",
            "land_mean_dominance_margin",
            "land_mean_transition_index",
            "ambiguous_land_area_fraction",
            "classifiable_land_area_fraction",
            "maximum_partition_absolute_error",
        )
    }


__all__ = ["BIOME_COUNT", "run_derived_biomes"]
