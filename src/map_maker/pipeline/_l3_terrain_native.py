"""Rust-backed chunked L3 terrain bindings."""

from __future__ import annotations

from typing import Any, Mapping

from cffi import FFI
import numpy as np

from .._native import NativeLibraryAbiError, native_library_info, native_library_path

L3_TERRAIN_CONTROL_NAMES = {
    "parent_resolution",
    "factor",
    "planet_radius_m",
    "terrain_seed",
    "relief_realization_fraction",
    "base_wavelength_m",
    "octave_count",
    "persistence",
    "domain_warp_fraction",
    "orogenic_ridge_fraction",
}

_CDEF = """
typedef struct {
    int32_t parent_resolution;
    int32_t factor;
    double planet_radius_m;
    uint64_t terrain_seed;
    float relief_realization_fraction;
    float base_wavelength_m;
    int32_t octave_count;
    float persistence;
    float domain_warp_fraction;
    float orogenic_ridge_fraction;
} L3TerrainConfig;

typedef struct {
    int32_t parent_count;
    int64_t cell_count;
    int32_t fine_resolution;
    int64_t context_neighbor_count;
    int64_t missing_context_neighbor_count;
    double selected_area_km2;
    double maximum_parent_area_relative_error;
    double maximum_parent_elevation_error_m;
    float minimum_elevation_m;
    float maximum_elevation_m;
} L3TerrainStats;

uint32_t l3_terrain_native_abi_version(void);
int32_t l3_terrain_generate_chunk(
    const L3TerrainConfig* config,
    int32_t context_count,
    const int32_t* context_parent_ids,
    const float* context_elevation_m,
    const float* context_relief_m,
    const float* context_rock_strength,
    const float* context_orogenic_strength,
    const float* context_ridge_direction_xyz,
    int32_t chunk_parent_count,
    const int32_t* chunk_parent_ids,
    uint64_t* cell_id_out,
    int32_t* parent_id_out,
    uint8_t* face_out,
    int32_t* row_out,
    int32_t* col_out,
    float* xyz_out,
    double* area_km2_out,
    float* elevation_m_out,
    float* offset_m_out,
    float* unresolved_relief_m_out,
    L3TerrainStats* stats_out
);
"""

_ffi = FFI()
_ffi.cdef(_CDEF)
native_library_info("l3_terrain_native")
_lib = _ffi.dlopen(str(native_library_path("l3_terrain_native")))
try:
    _abi_version = int(_lib.l3_terrain_native_abi_version())
except AttributeError as exc:
    raise NativeLibraryAbiError("l3_terrain_native lacks its required API") from exc
if _abi_version != 1:
    raise NativeLibraryAbiError(f"l3_terrain_native uses ABI {_abi_version}; expected ABI 1")


def _input(
    value: np.ndarray, *, name: str, dtype: np.dtype[Any], dimensions: int = 1
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {array.dtype}")
    if array.ndim != dimensions:
        raise ValueError(f"{name} must be {dimensions}-dimensional")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError(f"{name} must be contiguous and aligned")
    return array


def run_l3_terrain_chunk(
    *,
    controls: Mapping[str, int | float],
    context_parent_ids: np.ndarray,
    context_elevation_m: np.ndarray,
    context_relief_m: np.ndarray,
    context_rock_strength: np.ndarray,
    context_orogenic_strength: np.ndarray,
    context_ridge_direction_xyz: np.ndarray,
    chunk_parent_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    """Generate one parent-aligned chunk without retaining native memory."""

    if set(controls) != L3_TERRAIN_CONTROL_NAMES:
        missing = sorted(L3_TERRAIN_CONTROL_NAMES - set(controls))
        extra = sorted(set(controls) - L3_TERRAIN_CONTROL_NAMES)
        raise ValueError(f"L3 terrain controls mismatch; missing={missing}, extra={extra}")
    if any(not np.isfinite(value) for value in controls.values()):
        raise ValueError("L3 terrain controls must be finite")
    arrays = {
        "context_parent_ids": _input(
            context_parent_ids, name="context_parent_ids", dtype=np.dtype(np.int32)
        ),
        "context_elevation_m": _input(
            context_elevation_m, name="context_elevation_m", dtype=np.dtype(np.float32)
        ),
        "context_relief_m": _input(
            context_relief_m, name="context_relief_m", dtype=np.dtype(np.float32)
        ),
        "context_rock_strength": _input(
            context_rock_strength, name="context_rock_strength", dtype=np.dtype(np.float32)
        ),
        "context_orogenic_strength": _input(
            context_orogenic_strength,
            name="context_orogenic_strength",
            dtype=np.dtype(np.float32),
        ),
        "context_ridge_direction_xyz": _input(
            context_ridge_direction_xyz,
            name="context_ridge_direction_xyz",
            dtype=np.dtype(np.float32),
            dimensions=2,
        ),
        "chunk_parent_ids": _input(
            chunk_parent_ids, name="chunk_parent_ids", dtype=np.dtype(np.int32)
        ),
    }
    context_count = len(arrays["context_parent_ids"])
    if context_count == 0 or len(arrays["chunk_parent_ids"]) == 0:
        raise ValueError("L3 terrain chunks require context and output parents")
    for name in (
        "context_elevation_m",
        "context_relief_m",
        "context_rock_strength",
        "context_orogenic_strength",
    ):
        if len(arrays[name]) != context_count:
            raise ValueError(f"{name} must contain one value per context parent")
    if arrays["context_ridge_direction_xyz"].shape != (context_count, 3):
        raise ValueError("context_ridge_direction_xyz must have shape (context_count, 3)")

    factor = int(controls["factor"])
    output_count = len(arrays["chunk_parent_ids"]) * factor * factor
    outputs = {
        "cell_id": np.empty(output_count, dtype=np.uint64),
        "parent_l2_cell_id": np.empty(output_count, dtype=np.int32),
        "face": np.empty(output_count, dtype=np.uint8),
        "row": np.empty(output_count, dtype=np.int32),
        "column": np.empty(output_count, dtype=np.int32),
        "xyz": np.empty((output_count, 3), dtype=np.float32),
        "area_km2": np.empty(output_count, dtype=np.float64),
        "elevation_m": np.empty(output_count, dtype=np.float32),
        "offset_from_l2_m": np.empty(output_count, dtype=np.float32),
        "unresolved_relief_m": np.empty(output_count, dtype=np.float32),
    }
    config = _ffi.new("L3TerrainConfig*")
    for name, value in controls.items():
        setattr(config[0], name, value)
    stats = _ffi.new("L3TerrainStats*")

    def pointer(array: np.ndarray, c_type: str, *, writable: bool = False) -> Any:
        qualifier = "" if writable else "const "
        return _ffi.cast(
            f"{qualifier}{c_type}*",
            _ffi.from_buffer(f"{c_type}[]", array, require_writable=writable),
        )

    status = _lib.l3_terrain_generate_chunk(
        config,
        context_count,
        pointer(arrays["context_parent_ids"], "int32_t"),
        pointer(arrays["context_elevation_m"], "float"),
        pointer(arrays["context_relief_m"], "float"),
        pointer(arrays["context_rock_strength"], "float"),
        pointer(arrays["context_orogenic_strength"], "float"),
        pointer(arrays["context_ridge_direction_xyz"], "float"),
        len(arrays["chunk_parent_ids"]),
        pointer(arrays["chunk_parent_ids"], "int32_t"),
        pointer(outputs["cell_id"], "uint64_t", writable=True),
        pointer(outputs["parent_l2_cell_id"], "int32_t", writable=True),
        pointer(outputs["face"], "uint8_t", writable=True),
        pointer(outputs["row"], "int32_t", writable=True),
        pointer(outputs["column"], "int32_t", writable=True),
        pointer(outputs["xyz"], "float", writable=True),
        pointer(outputs["area_km2"], "double", writable=True),
        pointer(outputs["elevation_m"], "float", writable=True),
        pointer(outputs["offset_from_l2_m"], "float", writable=True),
        pointer(outputs["unresolved_relief_m"], "float", writable=True),
        stats,
    )
    if status != 0:
        messages = {
            1: "null, empty, or length-mismatched argument",
            2: "invalid terrain controls or output dimensions",
            3: "invalid or duplicate context parent",
            4: "chunk parent is missing, duplicate, or topologically invalid",
        }
        raise RuntimeError(
            f"l3_terrain_generate_chunk failed: {messages.get(status, f'status {status}') }"
        )
    native_stats = stats[0]
    metadata: dict[str, int | float] = {
        "parent_count": int(native_stats.parent_count),
        "cell_count": int(native_stats.cell_count),
        "fine_resolution": int(native_stats.fine_resolution),
        "context_neighbor_count": int(native_stats.context_neighbor_count),
        "missing_context_neighbor_count": int(native_stats.missing_context_neighbor_count),
        "selected_area_km2": float(native_stats.selected_area_km2),
        "maximum_parent_area_relative_error": float(
            native_stats.maximum_parent_area_relative_error
        ),
        "maximum_parent_elevation_error_m": float(native_stats.maximum_parent_elevation_error_m),
        "minimum_elevation_m": float(native_stats.minimum_elevation_m),
        "maximum_elevation_m": float(native_stats.maximum_elevation_m),
    }
    return outputs, metadata


__all__ = ["L3_TERRAIN_CONTROL_NAMES", "run_l3_terrain_chunk"]
