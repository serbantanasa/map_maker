"""Rust-backed native bindings for the tectonics stage."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from cffi import FFI

PLATE_FIELD_COMPONENTS = 6

_CDEF = """
typedef struct {
    int32_t plate_count;
    double continental_fraction;
    double velocity_mean;
    double velocity_std;
    double hotspot_mean;
    double boundary_metric_mean;
    double convergence_sum;
    double divergence_sum;
    double shear_sum;
    double subduction_mean;
    int32_t hotspot_count;
} TectonicsStats;

void tectonics_run(
    int32_t height,
    int32_t width,
    uint64_t seed,
    int32_t num_plates,
    float continental_fraction_target,
    float velocity_scale,
    float drift_bias,
    float hotspot_density,
    float subduction_bias,
    int32_t lloyd_iterations,
    int32_t time_steps,
    float time_step,
    int32_t wrap_x,
    int32_t wrap_y,
    float* plate_field,
    float* convergence_field,
    float* divergence_field,
    float* shear_field,
    float* subduction_field,
    float* hotspot_field,
    TectonicsStats* out_stats
);
"""


def _library_name() -> str:
    if sys.platform.startswith("win"):
        return "tectonics_native.dll"
    if sys.platform == "darwin":
        return "libtectonics_native.dylib"
    return "libtectonics_native.so"


def _crate_dir() -> Path:
    return Path(__file__).resolve().parent / "native" / "tectonics"


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
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=str(crate),
            check=True,
            env={**os.environ},
        )
    return library


_ffi = FFI()
_ffi.cdef(_CDEF)
_lib = _ffi.dlopen(str(_build_library()))

def _as_write_array(array: np.ndarray, *, name: str) -> np.ndarray:
    if array.dtype != np.float32:
        raise ValueError(f"{name} must be float32, got {array.dtype}")
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be contiguous")
    if not array.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return array


def run_tectonics_kernels(
    height: int,
    width: int,
    seed: int,
    num_plates: int,
    continental_fraction: float,
    velocity_scale: float,
    drift_bias: float,
    hotspot_density: float,
    subduction_bias: float,
    lloyd_iterations: int,
    time_steps: int,
    time_step: float,
    wrap_x: bool,
    wrap_y: bool,
    plate_field: np.ndarray,
    convergence_field: np.ndarray,
    divergence_field: np.ndarray,
    shear_field: np.ndarray,
    subduction_field: np.ndarray,
    hotspot_field: np.ndarray,
) -> Dict[str, Any]:
    plate_arr = _as_write_array(plate_field, name="plate_field")
    conv_arr = _as_write_array(convergence_field, name="convergence_field")
    div_arr = _as_write_array(divergence_field, name="divergence_field")
    shear_arr = _as_write_array(shear_field, name="shear_field")
    subduction_arr = _as_write_array(subduction_field, name="subduction_field")
    hotspot_arr = _as_write_array(hotspot_field, name="hotspot_field")

    stats_ptr = _ffi.new("TectonicsStats*")

    _lib.tectonics_run(
        int(height),
        int(width),
        int(seed),
        int(num_plates),
        float(continental_fraction),
        float(velocity_scale),
        float(drift_bias),
        float(hotspot_density),
        float(subduction_bias),
        int(lloyd_iterations),
        int(time_steps),
        float(time_step),
        1 if wrap_x else 0,
        1 if wrap_y else 0,
        _ffi.cast("float*", _ffi.from_buffer("float[]", plate_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", conv_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", div_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", shear_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", subduction_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", hotspot_arr)),
        stats_ptr,
    )

    stats = stats_ptr[0]
    return {
        "plate_count": int(stats.plate_count),
        "continental_fraction": float(stats.continental_fraction),
        "velocity_mean": float(stats.velocity_mean),
        "velocity_std": float(stats.velocity_std),
        "hotspot_mean": float(stats.hotspot_mean),
        "boundary_metric_mean": float(stats.boundary_metric_mean),
        "convergence_sum": float(stats.convergence_sum),
        "divergence_sum": float(stats.divergence_sum),
        "shear_sum": float(stats.shear_sum),
        "subduction_mean": float(stats.subduction_mean),
        "hotspot_count": int(stats.hotspot_count),
    }


__all__ = ["PLATE_FIELD_COMPONENTS", "run_tectonics_kernels"]
