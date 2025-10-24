"""Rust-backed native bindings for the world age stage."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pyarrow as pa
from cffi import FFI

HOTSPOT_EVENT_DTYPE = np.dtype(
    [
        ("row", "<i4"),
        ("col", "<i4"),
        ("strength", "<f4"),
        ("plume_factor", "<f4"),
    ]
)

_CDEF = """
typedef struct {
    int32_t row;
    int32_t col;
    float strength;
    float plume_factor;
} HotspotEvent;

typedef struct {
    HotspotEvent* data;
    size_t len;
} HotspotEventArray;

typedef struct {
    float convective_vigor;
    float mean_crust_thickness;
    float std_crust_thickness;
    float mean_isostatic_offset;
    int32_t hotspot_count;
    float uplift_mean;
    float subsidence_mean;
    float thermal_decay_factor;
    float water_fraction;
    float uplift_sigma_gt1;
    float uplift_sigma_gt2;
    float uplift_sigma_gt3;
    float subsidence_sigma_gt1;
    float subsidence_sigma_gt2;
    float subsidence_sigma_gt3;
    float hotspot_density;
} WorldAgeStats;

int32_t world_age_run(
    int32_t height,
    int32_t width,
    uint64_t seed,
    float world_age,
    float thermal_decay_half_life,
    float hotspot_scale,
    float isostasy_factor,
    float radiogenic_heat_scale,
    int32_t plate_components,
    const float* plate_field,
    const float* convergence_field,
    const float* divergence_field,
    const float* subduction_field,
    const float* shear_field,
    const float* hotspot_field,
    float* crust_thickness_out,
    float* isostasy_out,
    float* uplift_out,
    float* subsidence_out,
    float* compression_out,
    float* extension_out,
    float* shear_out,
    float* coastal_exposure_out,
    float* lithosphere_stiffness_out,
    float* base_ocean_mask_out,
    HotspotEventArray* events_out,
    WorldAgeStats* stats_out
);

void world_age_free_events(HotspotEventArray array);
"""


def _library_name() -> str:
    if sys.platform.startswith("win"):
        return "world_age_native.dll"
    if sys.platform == "darwin":
        return "libworld_age_native.dylib"
    return "libworld_age_native.so"


def _crate_dir() -> Path:
    return Path(__file__).resolve().parent / "native" / "world_age"


def _library_path() -> Path:
    return _crate_dir() / "target" / "release" / _library_name()


def _needs_rebuild(library_path: Path, crate_dir: Path) -> bool:
    if not library_path.exists():
        return True
    lib_mtime = library_path.stat().st_mtime
    candidates = [crate_dir / "Cargo.toml"]
    candidates.extend(crate_dir.rglob("*.rs"))
    return any(path.stat().st_mtime > lib_mtime for path in candidates if path.exists())


def _build_library() -> Path:
    crate = _crate_dir()
    library = _library_path()
    if _needs_rebuild(library, crate):
        env = {**os.environ}
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=str(crate),
            check=True,
            env=env,
        )
    return library


def _as_read_array(array: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.array(array, copy=False)
    if arr.dtype != np.float32:
        raise ValueError(f"{name} must be float32, got {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be contiguous")
    return arr


def _as_write_array(array: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.array(array, copy=False)
    if arr.dtype != np.float32:
        raise ValueError(f"{name} must be float32, got {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be contiguous")
    if not arr.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return arr


_ffi = FFI()
_ffi.cdef(_CDEF)
_lib = _ffi.dlopen(str(_build_library()))


def _events_to_arrow(event_struct: Any, *, free_after: bool = True) -> pa.Table:
    length = int(event_struct.len)
    if length <= 0 or int(_ffi.cast("uintptr_t", event_struct.data)) == 0:
        if free_after:
            _lib.world_age_free_events(event_struct)
        return pa.table(
            {
                "row": pa.array([], type=pa.int32()),
                "col": pa.array([], type=pa.int32()),
                "strength": pa.array([], type=pa.float32()),
                "plume_factor": pa.array([], type=pa.float32()),
            }
        )

    buffer = _ffi.buffer(event_struct.data, length * HOTSPOT_EVENT_DTYPE.itemsize)
    events_np = np.frombuffer(buffer, dtype=HOTSPOT_EVENT_DTYPE, count=length).copy()
    if free_after:
        _lib.world_age_free_events(event_struct)
    return pa.table(
        {
            "row": pa.array(events_np["row"], type=pa.int32()),
            "col": pa.array(events_np["col"], type=pa.int32()),
            "strength": pa.array(events_np["strength"], type=pa.float32()),
            "plume_factor": pa.array(events_np["plume_factor"], type=pa.float32()),
        }
    )


def run_world_age_kernels(
    *,
    height: int,
    width: int,
    seed: int,
    world_age: float,
    thermal_decay_half_life: float,
    hotspot_scale: float,
    isostasy_factor: float,
    radiogenic_heat_scale: float,
    plate_field: np.ndarray,
    convergence_field: np.ndarray,
    divergence_field: np.ndarray,
    subduction_field: np.ndarray,
    shear_field: np.ndarray,
    hotspot_field: np.ndarray,
    crust_thickness_out: np.ndarray,
    isostatic_offset_out: np.ndarray,
    uplift_out: np.ndarray,
    subsidence_out: np.ndarray,
    compression_out: np.ndarray,
    extension_out: np.ndarray,
    shear_out: np.ndarray,
    coastal_exposure_out: np.ndarray,
    lithosphere_stiffness_out: np.ndarray,
    base_ocean_mask_out: np.ndarray,
) -> Tuple[pa.Table, Dict[str, Any]]:
    plate_arr = _as_read_array(plate_field, name="plate_field")
    if plate_arr.ndim != 3:
        raise ValueError(f"plate_field must be 3D (H, W, C); got shape {plate_arr.shape}")
    expected_grid = (int(height), int(width))
    if tuple(int(dim) for dim in plate_arr.shape[:2]) != expected_grid:
        raise ValueError(f"plate_field expected leading shape {expected_grid}, got {plate_arr.shape}")
    convergence_arr = _as_read_array(convergence_field, name="convergence_field")
    divergence_arr = _as_read_array(divergence_field, name="divergence_field")
    subduction_arr = _as_read_array(subduction_field, name="subduction_field")
    shear_arr = _as_read_array(shear_field, name="shear_field")
    hotspot_arr = _as_read_array(hotspot_field, name="hotspot_field")
    for arr, name in (
        (convergence_arr, "convergence_field"),
        (divergence_arr, "divergence_field"),
        (subduction_arr, "subduction_field"),
        (shear_arr, "shear_field"),
        (hotspot_arr, "hotspot_field"),
    ):
        if arr.shape != expected_grid:
            raise ValueError(f"{name} expected shape {expected_grid}, got {arr.shape}")

    crust_arr = _as_write_array(crust_thickness_out, name="crust_thickness_out")
    isostasy_arr = _as_write_array(isostatic_offset_out, name="isostatic_offset_out")
    uplift_arr = _as_write_array(uplift_out, name="uplift_out")
    subsidence_arr = _as_write_array(subsidence_out, name="subsidence_out")
    compression_arr = _as_write_array(compression_out, name="compression_out")
    extension_arr = _as_write_array(extension_out, name="extension_out")
    shear_out_arr = _as_write_array(shear_out, name="shear_out")
    coastal_exposure_arr = _as_write_array(coastal_exposure_out, name="coastal_exposure_out")
    lithosphere_arr = _as_write_array(lithosphere_stiffness_out, name="lithosphere_stiffness_out")
    base_ocean_mask_arr = _as_write_array(base_ocean_mask_out, name="base_ocean_mask_out")

    for arr, name in (
        (crust_arr, "crust_thickness_out"),
        (isostasy_arr, "isostatic_offset_out"),
        (uplift_arr, "uplift_out"),
        (subsidence_arr, "subsidence_out"),
        (compression_arr, "compression_out"),
        (extension_arr, "extension_out"),
        (shear_out_arr, "shear_out"),
        (coastal_exposure_arr, "coastal_exposure_out"),
        (lithosphere_arr, "lithosphere_stiffness_out"),
        (base_ocean_mask_arr, "base_ocean_mask_out"),
    ):
        if arr.shape != expected_grid:
            raise ValueError(f"{name} expected shape {expected_grid}, got {arr.shape}")

    events_ptr = _ffi.new("HotspotEventArray*")
    stats_ptr = _ffi.new("WorldAgeStats*")

    result = _lib.world_age_run(
        int(height),
        int(width),
        int(seed),
        float(world_age),
        float(thermal_decay_half_life),
        float(hotspot_scale),
        float(isostasy_factor),
        float(radiogenic_heat_scale),
        int(plate_arr.shape[2] if plate_arr.ndim == 3 else 1),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", plate_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", convergence_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", divergence_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", subduction_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", shear_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", hotspot_arr, require_writable=False)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", crust_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", isostasy_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", uplift_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", subsidence_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", compression_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", extension_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", shear_out_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", coastal_exposure_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", lithosphere_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", base_ocean_mask_arr)),
        events_ptr,
        stats_ptr,
    )
    if result != 0:
        raise RuntimeError(f"world_age_run failed with code {result}")

    events_table = _events_to_arrow(events_ptr[0])
    stats = stats_ptr[0]
    metadata = {
        "convective_vigor": float(stats.convective_vigor),
        "mean_crust_thickness": float(stats.mean_crust_thickness),
        "std_crust_thickness": float(stats.std_crust_thickness),
        "mean_isostatic_offset": float(stats.mean_isostatic_offset),
        "hotspot_count": int(stats.hotspot_count),
        "uplift_mean": float(stats.uplift_mean),
        "subsidence_mean": float(stats.subsidence_mean),
        "thermal_decay_factor": float(stats.thermal_decay_factor),
        "water_fraction": float(stats.water_fraction),
        "uplift_sigma_gt1": float(stats.uplift_sigma_gt1),
        "uplift_sigma_gt2": float(stats.uplift_sigma_gt2),
        "uplift_sigma_gt3": float(stats.uplift_sigma_gt3),
        "subsidence_sigma_gt1": float(stats.subsidence_sigma_gt1),
        "subsidence_sigma_gt2": float(stats.subsidence_sigma_gt2),
        "subsidence_sigma_gt3": float(stats.subsidence_sigma_gt3),
        "hotspot_density": float(stats.hotspot_density),
    }
    return events_table, metadata


__all__ = ["HOTSPOT_EVENT_DTYPE", "run_world_age_kernels"]
