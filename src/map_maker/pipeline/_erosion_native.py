"""Rust-backed native bindings for the erosion stage."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pyarrow as pa
from cffi import FFI

_CDEF = """
typedef struct {
    int32_t step;
    float mean_elevation;
    float mass_removed;
    float mass_deposited;
} IterDiagnostic;

typedef struct {
    IterDiagnostic* data;
    size_t len;
} IterDiagnosticArray;

typedef struct {
    float total_mass_removed;
    float total_mass_deposited;
    float sediment_mass;
    float final_mean_elevation;
    float final_min_elevation;
    float final_max_elevation;
    float mass_residual;
    int32_t steps_run;
} ErosionStats;

int32_t erosion_run(
    int32_t height,
    int32_t width,
    int32_t steps,
    float dt,
    float stream_power_k,
    float sediment_capacity,
    float coastal_wave_energy,
    int32_t plate_components,
    const float* plate_field,
    const float* crust_thickness,
    const float* isostatic_offset,
    const float* uplift_rate,
    const float* subsidence_rate,
    const float* compression,
    const float* extension,
    const float* shear,
    const float* coastal_exposure,
    const float* lithosphere_stiffness,
    const float* base_ocean_mask,
    const float* hotspot_influence,
    float* elevation_out,
    float* sediment_out,
    float* incision_out,
    IterDiagnosticArray* diagnostics_out,
    ErosionStats* stats_out
);

void erosion_free_diagnostics(IterDiagnosticArray array);
"""


def _library_name() -> str:
    if sys.platform.startswith("win"):
        return "erosion_native.dll"
    if sys.platform == "darwin":
        return "liberosion_native.dylib"
    return "liberosion_native.so"


def _crate_dir() -> Path:
    return Path(__file__).resolve().parent / "native" / "erosion"


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


def _require_float32(array: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.array(array, copy=False)
    if arr.dtype != np.float32:
        raise ValueError(f"{name} must be float32, got {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be contiguous")
    return arr


def _require_write_array(array: np.ndarray, *, name: str) -> np.ndarray:
    arr = _require_float32(array, name=name)
    if not arr.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return arr


_ffi = FFI()
_ffi.cdef(_CDEF)
_lib = _ffi.dlopen(str(_build_library()))


def _diagnostics_to_arrow(diag_struct: Any, *, free_after: bool = True) -> pa.Table:
    length = int(diag_struct.len)
    if length <= 0 or int(_ffi.cast("uintptr_t", diag_struct.data)) == 0:
        if free_after:
            _lib.erosion_free_diagnostics(diag_struct)
        return pa.table(
            {
                "step": pa.array([], type=pa.int32()),
                "mean_elevation": pa.array([], type=pa.float32()),
                "mass_removed": pa.array([], type=pa.float32()),
                "mass_deposited": pa.array([], type=pa.float32()),
            }
        )
    buffer = _ffi.buffer(diag_struct.data, length * _ffi.sizeof("IterDiagnostic"))
    dtype = np.dtype([
        ("step", np.int32),
        ("mean_elevation", np.float32),
        ("mass_removed", np.float32),
        ("mass_deposited", np.float32),
    ])
    diagnostics = np.frombuffer(buffer, dtype=dtype, count=length).copy()
    if free_after:
        _lib.erosion_free_diagnostics(diag_struct)
    return pa.table(
        {
            "step": pa.array(diagnostics["step"], type=pa.int32()),
            "mean_elevation": pa.array(diagnostics["mean_elevation"], type=pa.float32()),
            "mass_removed": pa.array(diagnostics["mass_removed"], type=pa.float32()),
            "mass_deposited": pa.array(diagnostics["mass_deposited"], type=pa.float32()),
        }
    )


def run_erosion_kernels(
    *,
    height: int,
    width: int,
    steps: int,
    dt: float,
    stream_power_k: float,
    sediment_capacity: float,
    coastal_wave_energy: float,
    plate_field: np.ndarray,
    crust_thickness: np.ndarray,
    isostatic_offset: np.ndarray,
    uplift_rate: np.ndarray,
    subsidence_rate: np.ndarray,
    compression: np.ndarray,
    extension: np.ndarray,
    shear: np.ndarray,
    coastal_exposure: np.ndarray,
    lithosphere_stiffness: np.ndarray,
    base_ocean_mask: np.ndarray,
    hotspot_influence: np.ndarray,
    elevation_out: np.ndarray,
    sediment_out: np.ndarray,
    incision_out: np.ndarray,
) -> Tuple[pa.Table, Dict[str, Any]]:
    plate_arr = _require_float32(plate_field, name="plate_field")
    if plate_arr.ndim != 3:
        raise ValueError(f"plate_field must be 3D (H, W, C); got {plate_arr.shape}")
    expected_grid = (int(height), int(width))
    if tuple(int(dim) for dim in plate_arr.shape[:2]) != expected_grid:
        raise ValueError(f"plate_field expected leading shape {expected_grid}, got {plate_arr.shape}")

    def _require_grid(arr: np.ndarray, name: str) -> np.ndarray:
        arr32 = _require_float32(arr, name=name)
        if arr32.shape != expected_grid:
            raise ValueError(f"{name} expected shape {expected_grid}, got {arr32.shape}")
        return arr32

    crust_arr = _require_grid(crust_thickness, "crust_thickness")
    isostasy_arr = _require_grid(isostatic_offset, "isostatic_offset")
    uplift_arr = _require_grid(uplift_rate, "uplift_rate")
    subsidence_arr = _require_grid(subsidence_rate, "subsidence_rate")
    compression_arr = _require_grid(compression, "compression")
    extension_arr = _require_grid(extension, "extension")
    shear_arr = _require_grid(shear, "shear")
    coastal_arr = _require_grid(coastal_exposure, "coastal_exposure")
    lithosphere_arr = _require_grid(lithosphere_stiffness, "lithosphere_stiffness")
    ocean_arr = _require_grid(base_ocean_mask, "base_ocean_mask")
    hotspot_arr = _require_grid(hotspot_influence, "hotspot_influence")

    elevation_arr = _require_write_array(elevation_out, name="elevation_out")
    sediment_arr = _require_write_array(sediment_out, name="sediment_out")
    incision_arr = _require_write_array(incision_out, name="incision_out")

    diagnostics_ptr = _ffi.new("IterDiagnosticArray*")
    stats_ptr = _ffi.new("ErosionStats*")

    result = _lib.erosion_run(
        int(height),
        int(width),
        int(steps),
        float(dt),
        float(stream_power_k),
        float(sediment_capacity),
        float(coastal_wave_energy),
        int(plate_arr.shape[2]),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", plate_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", crust_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", isostasy_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", uplift_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", subsidence_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", compression_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", extension_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", shear_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", coastal_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", lithosphere_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", ocean_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", hotspot_arr, require_writable=False)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", elevation_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", sediment_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", incision_arr)),
        diagnostics_ptr,
        stats_ptr,
    )
    if result != 0:
        raise RuntimeError(f"erosion_run failed with code {result}")

    diagnostics_table = _diagnostics_to_arrow(diagnostics_ptr[0])
    stats = stats_ptr[0]
    metadata = {
        "total_mass_removed": float(stats.total_mass_removed),
        "total_mass_deposited": float(stats.total_mass_deposited),
        "sediment_mass": float(stats.sediment_mass),
        "mass_residual": float(stats.mass_residual),
        "final_mean_elevation": float(stats.final_mean_elevation),
        "final_min_elevation": float(stats.final_min_elevation),
        "final_max_elevation": float(stats.final_max_elevation),
        "steps_run": int(stats.steps_run),
    }
    return diagnostics_table, metadata


__all__ = ["run_erosion_kernels"]
