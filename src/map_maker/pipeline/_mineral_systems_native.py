"""Rust-backed causal mineral-system kernel."""

from __future__ import annotations

import numpy as np
from cffi import FFI  # type: ignore[import-untyped]

from .._native import native_library_info, native_library_path

SYSTEM_COUNT = 10
COMMODITY_COUNT = 15

_CDEF = """
uint32_t mineral_systems_native_abi_version(void);
int32_t mineral_systems_run(
    int32_t cell_count,
    int32_t face_resolution,
    uint64_t seed,
    double minimum_dominant_potential,
    const float* xyz,
    const float* ocean,
    const float* shelf,
    const float* relief,
    const float* elevation,
    const float* terrain_slope,
    const uint8_t* province_class,
    const float* crust_age,
    const float* rock_strength,
    const float* accommodation,
    const float* province_confidence,
    const float* elevation_confidence,
    const float* convergence,
    const float* divergence,
    const float* shear,
    const float* subduction,
    const float* hotspot,
    const float* uplift,
    const float* subsidence,
    const float* compression,
    const float* extension,
    const float* stiffness,
    const float* temperature,
    const float* precipitation,
    const float* aridity,
    const double* contributing_area,
    const float* stream_power,
    const float* river,
    const float* floodplain,
    const float* lake,
    const float* wetland,
    const float* bedrock,
    const float* residual_regolith,
    const float* alluvium,
    const float* lacustrine,
    const float* volcaniclastic,
    const float* soil_depth,
    const float* salinity,
    const float* drainage,
    const float* hydric_soil,
    const float* soil_confidence,
    const float* annual_npp,
    const float* standing_biomass,
    const float* vegetation_cover,
    const float* biosphere_confidence,
    float* source_out,
    float* process_out,
    float* transport_out,
    float* trap_out,
    float* timing_out,
    float* preservation_out,
    float* unresolved_out,
    float* potential_out,
    float* confidence_out,
    float* commodity_out,
    uint8_t* dominant_system_out
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
native_library_info("mineral_systems_native")
_lib = _ffi.dlopen(str(native_library_path("mineral_systems_native")))


def run_mineral_systems(  # noqa: PLR0913
    *,
    face_resolution: int,
    seed: int,
    minimum_dominant_potential: float,
    xyz: np.ndarray,
    ocean: np.ndarray,
    shelf: np.ndarray,
    relief: np.ndarray,
    elevation: np.ndarray,
    terrain_slope: np.ndarray,
    province_class: np.ndarray,
    crust_age: np.ndarray,
    rock_strength: np.ndarray,
    accommodation: np.ndarray,
    province_confidence: np.ndarray,
    elevation_confidence: np.ndarray,
    convergence: np.ndarray,
    divergence: np.ndarray,
    shear: np.ndarray,
    subduction: np.ndarray,
    hotspot: np.ndarray,
    uplift: np.ndarray,
    subsidence: np.ndarray,
    compression: np.ndarray,
    extension: np.ndarray,
    stiffness: np.ndarray,
    temperature: np.ndarray,
    precipitation: np.ndarray,
    aridity: np.ndarray,
    contributing_area: np.ndarray,
    stream_power: np.ndarray,
    river: np.ndarray,
    floodplain: np.ndarray,
    lake: np.ndarray,
    wetland: np.ndarray,
    bedrock: np.ndarray,
    residual_regolith: np.ndarray,
    alluvium: np.ndarray,
    lacustrine: np.ndarray,
    volcaniclastic: np.ndarray,
    soil_depth: np.ndarray,
    salinity: np.ndarray,
    drainage: np.ndarray,
    hydric_soil: np.ndarray,
    soil_confidence: np.ndarray,
    annual_npp: np.ndarray,
    standing_biomass: np.ndarray,
    vegetation_cover: np.ndarray,
    biosphere_confidence: np.ndarray,
    **output_arrays: np.ndarray,
) -> None:
    if (
        isinstance(face_resolution, bool)
        or not isinstance(face_resolution, int)
        or face_resolution <= 0
    ):
        raise ValueError("face_resolution must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    scalar_float_raw = {
        "ocean": ocean,
        "shelf": shelf,
        "relief": relief,
        "elevation": elevation,
        "terrain_slope": terrain_slope,
        "crust_age": crust_age,
        "rock_strength": rock_strength,
        "accommodation": accommodation,
        "province_confidence": province_confidence,
        "elevation_confidence": elevation_confidence,
        "convergence": convergence,
        "divergence": divergence,
        "shear": shear,
        "subduction": subduction,
        "hotspot": hotspot,
        "uplift": uplift,
        "subsidence": subsidence,
        "compression": compression,
        "extension": extension,
        "stiffness": stiffness,
        "temperature": temperature,
        "precipitation": precipitation,
        "aridity": aridity,
        "stream_power": stream_power,
        "river": river,
        "floodplain": floodplain,
        "lake": lake,
        "wetland": wetland,
        "bedrock": bedrock,
        "residual_regolith": residual_regolith,
        "alluvium": alluvium,
        "lacustrine": lacustrine,
        "volcaniclastic": volcaniclastic,
        "soil_depth": soil_depth,
        "salinity": salinity,
        "drainage": drainage,
        "hydric_soil": hydric_soil,
        "soil_confidence": soil_confidence,
        "annual_npp": annual_npp,
        "standing_biomass": standing_biomass,
        "vegetation_cover": vegetation_cover,
        "biosphere_confidence": biosphere_confidence,
    }
    first = np.asarray(ocean)
    if first.ndim < 1 or first.size == 0:
        raise ValueError("ocean must contain at least one spatial cell")
    shape = first.shape
    inputs = {
        name: _read_array(value, name=name, dtype=np.dtype(np.float32))
        for name, value in scalar_float_raw.items()
    }
    inputs["province_class"] = _read_array(
        province_class, name="province_class", dtype=np.dtype(np.uint8)
    )
    inputs["contributing_area"] = _read_array(
        contributing_area, name="contributing_area", dtype=np.dtype(np.float64)
    )
    xyz_array = _read_array(xyz, name="xyz", dtype=np.dtype(np.float32))
    if xyz_array.shape != (*shape, 3):
        raise ValueError(f"xyz must have shape {(*shape, 3)}, got {xyz_array.shape}")
    inputs["xyz"] = xyz_array
    for name, value in inputs.items():
        if name == "xyz":
            continue
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")

    expected_shapes = {
        "source_out": (SYSTEM_COUNT, *shape),
        "process_out": (SYSTEM_COUNT, *shape),
        "transport_out": (SYSTEM_COUNT, *shape),
        "trap_out": (SYSTEM_COUNT, *shape),
        "timing_out": (SYSTEM_COUNT, *shape),
        "preservation_out": (SYSTEM_COUNT, *shape),
        "unresolved_out": (SYSTEM_COUNT, *shape),
        "potential_out": (SYSTEM_COUNT, *shape),
        "confidence_out": (SYSTEM_COUNT, *shape),
        "commodity_out": (COMMODITY_COUNT, *shape),
        "dominant_system_out": shape,
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
        dtype = np.dtype(np.uint8) if name == "dominant_system_out" else np.dtype(np.float32)
        output = _write_array(output_arrays[name], name=name, dtype=dtype)
        if output.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {output.shape}")
        outputs[name] = output
    _require_disjoint(inputs, outputs)

    float_ptr = {
        name: _ffi.cast("const float*", _ffi.from_buffer("float[]", value))
        for name, value in inputs.items()
        if value.dtype == np.float32
    }
    output_float_ptr = {
        name: _ffi.cast("float*", _ffi.from_buffer("float[]", value))
        for name, value in outputs.items()
        if value.dtype == np.float32
    }
    status = int(
        _lib.mineral_systems_run(
            int(np.prod(shape)),
            face_resolution,
            seed,
            minimum_dominant_potential,
            float_ptr["xyz"],
            *(
                float_ptr[name]
                for name in (
                    "ocean",
                    "shelf",
                    "relief",
                    "elevation",
                    "terrain_slope",
                )
            ),
            _ffi.cast(
                "const uint8_t*",
                _ffi.from_buffer("uint8_t[]", inputs["province_class"]),
            ),
            *(
                float_ptr[name]
                for name in (
                    "crust_age",
                    "rock_strength",
                    "accommodation",
                    "province_confidence",
                    "elevation_confidence",
                    "convergence",
                    "divergence",
                    "shear",
                    "subduction",
                    "hotspot",
                    "uplift",
                    "subsidence",
                    "compression",
                    "extension",
                    "stiffness",
                    "temperature",
                    "precipitation",
                    "aridity",
                )
            ),
            _ffi.cast(
                "const double*",
                _ffi.from_buffer("double[]", inputs["contributing_area"]),
            ),
            *(
                float_ptr[name]
                for name in (
                    "stream_power",
                    "river",
                    "floodplain",
                    "lake",
                    "wetland",
                    "bedrock",
                    "residual_regolith",
                    "alluvium",
                    "lacustrine",
                    "volcaniclastic",
                    "soil_depth",
                    "salinity",
                    "drainage",
                    "hydric_soil",
                    "soil_confidence",
                    "annual_npp",
                    "standing_biomass",
                    "vegetation_cover",
                    "biosphere_confidence",
                )
            ),
            *(
                output_float_ptr[name]
                for name in (
                    "source_out",
                    "process_out",
                    "transport_out",
                    "trap_out",
                    "timing_out",
                    "preservation_out",
                    "unresolved_out",
                    "potential_out",
                    "confidence_out",
                    "commodity_out",
                )
            ),
            _ffi.cast(
                "uint8_t*",
                _ffi.from_buffer("uint8_t[]", outputs["dominant_system_out"]),
            ),
        )
    )
    if status != 0:
        messages = {
            1: "invalid dimensions or null buffers",
            2: "invalid mineral-system controls",
            3: "non-finite or physically invalid mineral-system inputs",
        }
        raise ValueError(
            f"mineral-system kernel failed: {messages.get(status, f'status {status}')}"
        )


__all__ = ["COMMODITY_COUNT", "SYSTEM_COUNT", "run_mineral_systems"]
